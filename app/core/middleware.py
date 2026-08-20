import time
import uuid
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_429_TOO_MANY_REQUESTS,
)
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.config import get_settings
from app.core.logging import set_request_id

HTTP_413_CONTENT_TOO_LARGE = 413


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        request.state.request_id = request_id
        set_request_id(request_id)
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        return response


class RequestSizeLimitMiddleware:
    """Reject oversized declared and streamed bodies without buffering them."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        settings = get_settings()
        limit_bytes = settings.request_size_limit_mb * 1024 * 1024
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        raw_content_length = headers.get(b"content-length")
        if raw_content_length:
            try:
                declared_size = int(raw_content_length)
            except (TypeError, ValueError):
                response = JSONResponse(
                    status_code=HTTP_400_BAD_REQUEST,
                    content={"code": "invalid_content_length", "message": "Invalid Content-Length"},
                )
                await response(scope, receive, send)
                return
            if declared_size < 0:
                response = JSONResponse(
                    status_code=HTTP_400_BAD_REQUEST,
                    content={"code": "invalid_content_length", "message": "Invalid Content-Length"},
                )
                await response(scope, receive, send)
                return
            if declared_size > limit_bytes:
                response = JSONResponse(
                    status_code=HTTP_413_CONTENT_TOO_LARGE,
                    content={"code": "request_too_large", "message": "Request body is too large"},
                )
                await response(scope, receive, send)
                return

        received_bytes = 0
        body_too_large = False
        rejection_sent = False

        async def limited_receive() -> Message:
            nonlocal body_too_large, received_bytes
            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > limit_bytes:
                    body_too_large = True
                    return {"type": "http.request", "body": b"", "more_body": False}
            return message

        async def limited_send(message: Message) -> None:
            nonlocal rejection_sent
            if not body_too_large:
                await send(message)
                return
            if message["type"] == "http.response.start" and not rejection_sent:
                rejection_sent = True
                response = JSONResponse(
                    status_code=HTTP_413_CONTENT_TOO_LARGE,
                    content={"code": "request_too_large", "message": "Request body is too large"},
                )
                await response(scope, receive, send)

        await self.app(scope, limited_receive, limited_send)


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: Callable):
        super().__init__(app)
        self.requests: dict[str, deque[float]] = defaultdict(deque)
        self.next_cleanup = 0.0

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        settings = get_settings()
        window = 60
        max_requests = settings.rate_limit_per_minute
        now = time.monotonic()
        client_ip = request.client.host if request.client else "unknown"
        if now >= self.next_cleanup:
            for tracked_ip, tracked_queue in list(self.requests.items()):
                while tracked_queue and tracked_queue[0] <= now - window:
                    tracked_queue.popleft()
                if not tracked_queue:
                    self.requests.pop(tracked_ip, None)
            self.next_cleanup = now + window
        queue = self.requests[client_ip]
        while queue and queue[0] <= now - window:
            queue.popleft()
        if len(queue) >= max_requests:
            return JSONResponse(
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                headers={"Retry-After": str(window)},
                content={"code": "rate_limited", "message": "Rate limit exceeded"},
            )
        queue.append(now)
        return await call_next(request)
