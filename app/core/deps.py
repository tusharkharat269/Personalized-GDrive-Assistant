import uuid

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import getDb
from app.core.exceptions import UnauthorizedException
from app.core.security import decodeToken
from app.models.user import User

_bearer = HTTPBearer()


async def getCurrentUser(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
    db: AsyncSession = Depends(getDb),
) -> User:
    payload = decodeToken(credentials.credentials)
    if not payload or payload.get("type") != "access":
        raise UnauthorizedException("Invalid or expired token")

    userId = payload.get("sub")
    if not userId:
        raise UnauthorizedException("Invalid token payload")

    result = await db.execute(select(User).where(User.id == uuid.UUID(userId)))
    user = result.scalar_one_or_none()
    if not user or not user.isActive:
        raise UnauthorizedException("User not found or deactivated")
    return user
