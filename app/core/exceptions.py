from fastapi import HTTPException, status


class AppException(HTTPException):
    def __init__(self, statusCode: int, errorCode: str, detail: str):
        super().__init__(status_code=statusCode, detail={"errorCode": errorCode, "message": detail})


class UnauthorizedException(AppException):
    def __init__(self, detail: str = "Authentication required"):
        super().__init__(status.HTTP_401_UNAUTHORIZED, "AUTH_REQUIRED", detail)


class ForbiddenException(AppException):
    def __init__(self, detail: str = "Insufficient permissions"):
        super().__init__(status.HTTP_403_FORBIDDEN, "FORBIDDEN", detail)


class NotFoundException(AppException):
    def __init__(self, resource: str = "Resource"):
        super().__init__(status.HTTP_404_NOT_FOUND, "NOT_FOUND", f"{resource} not found")


class GoogleDriveException(AppException):
    def __init__(self, detail: str = "Google Drive operation failed"):
        super().__init__(status.HTTP_502_BAD_GATEWAY, "DRIVE_ERROR", detail)


class TokenExpiredException(AppException):
    def __init__(self):
        super().__init__(status.HTTP_401_UNAUTHORIZED, "TOKEN_EXPIRED", "Google Drive token expired, re-authenticate")
