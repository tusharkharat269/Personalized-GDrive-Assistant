from datetime import datetime, timedelta, timezone
from typing import Any

from cryptography.fernet import Fernet
from jose import JWTError, jwt

from app.core.config import getSettings

settings = getSettings()
_fernet = Fernet(settings.ENCRYPTION_KEY.encode())


def createAccessToken(data: dict[str, Any], expiresDelta: timedelta | None = None) -> str:
    toEncode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expiresDelta or timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    toEncode.update({"exp": expire, "type": "access"})
    return jwt.encode(toEncode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def createRefreshToken(data: dict[str, Any]) -> str:
    toEncode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS)
    toEncode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(toEncode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decodeToken(token: str) -> dict[str, Any] | None:
    try:
        return jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    except JWTError:
        return None


def encryptValue(value: str) -> str:
    return _fernet.encrypt(value.encode()).decode()


def decryptValue(encrypted: str) -> str:
    return _fernet.decrypt(encrypted.encode()).decode()
