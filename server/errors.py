"""统一错误契约。

改造前出错是 ``st.code(traceback.format_exc())`` 直接糊到页面上（``app.py:149``），
既泄露内部路径又对调用方毫无契约可言。这里把所有异常收敛成一种响应形状：

    {"error": {"code": "not_found", "message": "...", "request_id": "..."}}

``request_id`` 同时写进日志，用户报错时凭这一个 id 就能捞到服务端上下文，
不必让他复述堆栈。
"""

import logging
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger("movie_agent.api")

REQUEST_ID_HEADER = "X-Request-ID"


class APIError(Exception):
    """业务异常基类。``code`` 是给调用方分支用的稳定字符串。"""

    status_code = status.HTTP_400_BAD_REQUEST
    code = "bad_request"
    message = "Bad request"

    def __init__(self, message: str | None = None, **extra: Any) -> None:
        super().__init__(message or self.message)
        self.message = message or self.message
        self.extra = extra


class NotFound(APIError):
    status_code = status.HTTP_404_NOT_FOUND
    code = "not_found"
    message = "Resource not found"


class Unauthorized(APIError):
    status_code = status.HTTP_401_UNAUTHORIZED
    code = "unauthorized"
    message = "Authentication required"


class Forbidden(APIError):
    status_code = status.HTTP_403_FORBIDDEN
    code = "forbidden"
    message = "Operation not permitted"


class Conflict(APIError):
    status_code = status.HTTP_409_CONFLICT
    code = "conflict"
    message = "Conflicting state"


class ServiceUnavailable(APIError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "service_unavailable"
    message = "Dependency unavailable"


def _request_id(request: Request) -> str:
    existing = getattr(request.state, "request_id", None)
    if existing:
        return str(existing)
    generated = uuid.uuid4().hex
    request.state.request_id = generated
    return generated


def _envelope(request: Request, code: str, message: str, **extra: Any) -> JSONResponse:
    rid = _request_id(request)
    body: dict[str, Any] = {"error": {"code": code, "message": message, "request_id": rid}}
    if extra:
        body["error"]["details"] = extra
    return JSONResponse(status_code=_STATUS_BY_CODE.get(code, 400), content=body, headers={REQUEST_ID_HEADER: rid})


_STATUS_BY_CODE = {
    "bad_request": 400,
    "unauthorized": 401,
    "forbidden": 403,
    "not_found": 404,
    "conflict": 409,
    "validation_error": 422,
    "service_unavailable": 503,
    "internal_error": 500,
}


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(APIError)
    async def _handle_api_error(request: Request, exc: APIError) -> JSONResponse:
        rid = _request_id(request)
        logger.info("api_error code=%s rid=%s path=%s", exc.code, rid, request.url.path)
        return _envelope(request, exc.code, exc.message, **exc.extra)

    @app.exception_handler(HTTPException)
    async def _handle_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
        code = {
            401: "unauthorized",
            403: "forbidden",
            404: "not_found",
            409: "conflict",
        }.get(exc.status_code, "bad_request" if exc.status_code < 500 else "internal_error")
        return _envelope(request, code, str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def _handle_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        # 只回字段与原因，不回收到的值 —— 校验失败的载荷里可能就有密码
        fields = [
            {"loc": ".".join(str(p) for p in err["loc"][1:]), "reason": err["msg"]}
            for err in exc.errors()
        ]
        return _envelope(request, "validation_error", "Request validation failed", fields=fields)

    @app.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        rid = _request_id(request)
        # 堆栈只进日志。给调用方的永远是一句话 + request_id。
        logger.exception("unhandled_error rid=%s path=%s", rid, request.url.path)
        return _envelope(request, "internal_error", "Internal server error")
