from fastapi import Request
from fastapi.responses import JSONResponse

from app.core.exceptions import AppException
from app.core.logging import logger


async def appExceptionHandler(request: Request, exc: AppException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=exc.detail)


async def unhandledExceptionHandler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("unhandled_exception", path=request.url.path, error=str(exc), exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"errorCode": "INTERNAL_ERROR", "message": "An unexpected error occurred"},
    )
