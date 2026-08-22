import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging import logger


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, callNext: RequestResponseEndpoint) -> Response:
        requestId = str(uuid.uuid4())[:8]
        start = time.perf_counter()

        response = await callNext(request)

        duration = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "http_request",
            requestId=requestId,
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            durationMs=duration,
        )
        response.headers["X-Request-ID"] = requestId
        return response
