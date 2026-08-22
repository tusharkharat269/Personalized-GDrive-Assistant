from pydantic import BaseModel, Field


class AuthUrlResponse(BaseModel):
    authUrl: str


class TokenResponse(BaseModel):
    accessToken: str
    refreshToken: str
    tokenType: str = "bearer"


class RefreshTokenRequest(BaseModel):
    refreshToken: str


class ReauthRequest(BaseModel):
    accessLevel: str = Field(description="'readonly' or 'readwrite'")


class PermissionsResponse(BaseModel):
    accessLevel: str
    scopes: list[str]
