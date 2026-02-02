import time
import uuid
from collections import defaultdict, deque
from collections.abc import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.status import HTTP_413_REQUEST_ENTITY_TOO_LARGE, HTTP_429_TOO_MANY_REQUESTS

from app.core.config import get_settings
from app.core.logging import set_request_id


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        request.state.request_id = request_id
        set_request_id(request_id)
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        return response


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        settings = get_settings()
        limit_bytes = settings.request_size_limit_mb * 1024 * 1024
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > limit_bytes:
            return Response(status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE)
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: Callable):
        super().__init__(app)
        self.requests: dict[str, deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        settings = get_settings()
        window = 60
        max_requests = settings.rate_limit_per_minute
        now = time.time()
        client_ip = request.client.host if request.client else "unknown"
        queue = self.requests[client_ip]
        while queue and queue[0] <= now - window:
            queue.popleft()
        if len(queue) >= max_requests:
            return Response(status_code=HTTP_429_TOO_MANY_REQUESTS)
        queue.append(now)
        return await call_next(request)
