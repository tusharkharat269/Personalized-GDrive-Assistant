from datetime import datetime, timezone

import base64
import hashlib
import secrets

import httpx
from fastapi import APIRouter, Depends, Query, Request, Response
from google_auth_oauthlib.flow import Flow
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import getSettings
from app.core.database import getDb
from app.core.deps import getCurrentUser
from app.core.exceptions import AppException, UnauthorizedException
from app.core.logging import logger
from app.core.security import (
    createAccessToken,
    createRefreshToken,
    decodeToken,
    decryptValue,
    encryptValue,
)
from app.models.drive_token import GoogleDriveToken
from app.models.user import User
from app.schemas.auth import AuthUrlResponse, PermissionsResponse, ReauthRequest, RefreshTokenRequest, TokenResponse

router = APIRouter(prefix="/auth", tags=["Authentication"])
settings = getSettings()

_PKCE_COOKIE_NAME = "gDrivePkce"

_BASE_SCOPES = [
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "openid",
]

_SCOPE_SETS = {
    "readonly": _BASE_SCOPES + [
        "https://www.googleapis.com/auth/drive.readonly",
    ],
    "readwrite": _BASE_SCOPES + [
        "https://www.googleapis.com/auth/drive.readonly",
        "https://www.googleapis.com/auth/drive.file",
    ],
}

_WRITE_SCOPE = "https://www.googleapis.com/auth/drive.file"


def _base64UrlNoPad(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _generatePkcePair() -> tuple[str, str]:
    # RFC 7636: verifier length 43-128, allowed chars are URL-safe.
    codeVerifier = secrets.token_urlsafe(64)[:128]
    codeChallenge = _base64UrlNoPad(hashlib.sha256(codeVerifier.encode("ascii")).digest())
    return codeVerifier, codeChallenge


def _buildFlow(*, state: str | None = None, codeVerifier: str | None = None) -> Flow:
    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": settings.GOOGLE_CLIENT_ID,
                "client_secret": settings.GOOGLE_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [settings.GOOGLE_REDIRECT_URI],
            }
        },
        scopes=settings.GOOGLE_SCOPES,
        state=state,
        code_verifier=codeVerifier,
    )
    flow.redirect_uri = settings.GOOGLE_REDIRECT_URI
    return flow


@router.get("/google/login", response_model=AuthUrlResponse)
async def googleLogin(response: Response):
    codeVerifier, codeChallenge = _generatePkcePair()
    flow = _buildFlow(codeVerifier=codeVerifier)
    authUrl, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
        code_challenge=codeChallenge,
        code_challenge_method="S256",
    )

    # Store state+verifier in an HttpOnly cookie (encrypted with Fernet)
    payload = encryptValue(f"{state}:{codeVerifier}")
    response.set_cookie(
        key=_PKCE_COOKIE_NAME,
        value=payload,
        httponly=True,
        secure=not settings.DEBUG,
        samesite="lax",
        max_age=10 * 60,
    )
    return AuthUrlResponse(authUrl=authUrl)


@router.get("/google/callback", response_model=TokenResponse)
async def googleCallback(
    request: Request,
    response: Response,
    code: str = Query(...),
    state: str = Query(...),
    db: AsyncSession = Depends(getDb),
):
    pkceCookie = request.cookies.get(_PKCE_COOKIE_NAME)
    if not pkceCookie:
        raise AppException(400, "OAUTH_FAILED", "Missing OAuth verifier cookie. Retry login.")

    try:
        raw = decryptValue(pkceCookie)
        storedState, codeVerifier = raw.split(":", 1)
    except Exception as e:
        logger.error("google_oauth_pkce_cookie_invalid", error=str(e))
        raise AppException(400, "OAUTH_FAILED", "Invalid OAuth verifier cookie. Retry login.")

    if storedState != state or not codeVerifier:
        raise AppException(400, "OAUTH_FAILED", "OAuth state mismatch. Retry login.")

    flow = _buildFlow(state=state, codeVerifier=codeVerifier)
    try:
        flow.fetch_token(code=code)
    except Exception as e:
        logger.error("google_oauth_token_exchange_failed", error=str(e))
        raise AppException(400, "OAUTH_FAILED", "Failed to exchange authorization code")

    # One-time cookie
    response.delete_cookie(_PKCE_COOKIE_NAME)
    credentials = flow.credentials

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {credentials.token}"},
        )
        if resp.status_code != 200:
            raise AppException(400, "OAUTH_FAILED", "Failed to fetch user info from Google")
        userInfo = resp.json()

    email = userInfo.get("email")
    if not email:
        raise AppException(400, "OAUTH_FAILED", "Email not provided by Google")

    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()

    if not user:
        user = User(
            email=email,
            name=userInfo.get("name"),
            picture=userInfo.get("picture"),
        )
        db.add(user)
        await db.flush()
        logger.info("user_created", userId=str(user.id), email=email)

    result = await db.execute(
        select(GoogleDriveToken).where(GoogleDriveToken.userId == user.id)
    )
    existingToken = result.scalar_one_or_none()

    tokenData = {
        "accessToken": encryptValue(credentials.token),
        "refreshToken": encryptValue(credentials.refresh_token or ""),
        "tokenExpiry": credentials.expiry.replace(tzinfo=timezone.utc) if credentials.expiry else datetime.now(timezone.utc),
        "scopes": " ".join(credentials.scopes or []),
    }

    if existingToken:
        for k, v in tokenData.items():
            setattr(existingToken, k, v)
    else:
        driveToken = GoogleDriveToken(userId=user.id, **tokenData)
        db.add(driveToken)

    await db.flush()
    logger.info("google_drive_connected", userId=str(user.id))

    accessToken = createAccessToken({"sub": str(user.id), "email": email})
    refreshToken = createRefreshToken({"sub": str(user.id)})
    return TokenResponse(accessToken=accessToken, refreshToken=refreshToken)


@router.post("/refresh", response_model=TokenResponse)
async def refreshAccessToken(
    body: RefreshTokenRequest,
    db: AsyncSession = Depends(getDb),
):
    payload = decodeToken(body.refreshToken)
    if not payload or payload.get("type") != "refresh":
        raise UnauthorizedException("Invalid refresh token")

    userId = payload.get("sub")
    result = await db.execute(select(User).where(User.id == userId))
    user = result.scalar_one_or_none()
    if not user:
        raise UnauthorizedException("User not found")

    accessToken = createAccessToken({"sub": str(user.id), "email": user.email})
    refreshToken = createRefreshToken({"sub": str(user.id)})
    return TokenResponse(accessToken=accessToken, refreshToken=refreshToken)


@router.get("/permissions", response_model=PermissionsResponse)
async def getPermissions(
    user: User = Depends(getCurrentUser),
    db: AsyncSession = Depends(getDb),
):
    result = await db.execute(
        select(GoogleDriveToken).where(GoogleDriveToken.userId == user.id)
    )
    token = result.scalar_one_or_none()
    granted = (token.scopes or "").split() if token else []
    hasWrite = _WRITE_SCOPE in granted
    return PermissionsResponse(
        accessLevel="readwrite" if hasWrite else "readonly",
        scopes=granted,
    )


@router.post("/google/reauth", response_model=AuthUrlResponse)
async def googleReauth(
    body: ReauthRequest,
    response: Response,
    _user: User = Depends(getCurrentUser),
):
    scopeSet = _SCOPE_SETS.get(body.accessLevel)
    if not scopeSet:
        raise AppException(400, "INVALID_ACCESS_LEVEL", "accessLevel must be 'readonly' or 'readwrite'")

    codeVerifier, codeChallenge = _generatePkcePair()
    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": settings.GOOGLE_CLIENT_ID,
                "client_secret": settings.GOOGLE_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [settings.GOOGLE_REDIRECT_URI],
            }
        },
        scopes=scopeSet,
        code_verifier=codeVerifier,
    )
    flow.redirect_uri = settings.GOOGLE_REDIRECT_URI
    authUrl, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
        code_challenge=codeChallenge,
        code_challenge_method="S256",
    )

    payload = encryptValue(f"{state}:{codeVerifier}")
    response.set_cookie(
        key=_PKCE_COOKIE_NAME,
        value=payload,
        httponly=True,
        secure=not settings.DEBUG,
        samesite="lax",
        max_age=10 * 60,
    )
    return AuthUrlResponse(authUrl=authUrl)


@router.post("/logout")
async def logout(user: User = Depends(getCurrentUser)):
    logger.info("user_logged_out", userId=str(user.id))
    return {"message": "Logged out successfully"}
