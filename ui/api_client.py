"""Streamlit 用的 HTTP 客户端。

这个模块的存在本身就是本次改造的重点：**UI 进程不再 import ``movie_agent``**。

改造前 ``app.py`` 直接 ``asyncio.run(create_movie_agent())``，于是撞上
Streamlit 的事件循环亲和性问题 —— 每次 rerun 都新建再销毁一个 loop，跨请求
复用的连接池活不下来，只能退回进程内状态。现在 agent 跑在 uvicorn 里，UI 这边
全部是**同步 httpx**，一个 ``asyncio.run()`` 都不需要。

SSE 解析故意写得很朴素（``iter_lines`` + ``data:`` 前缀），不引第三方客户端库：
服务端用的事件格式就这么点，多一个依赖不如多看懂五行代码。
"""

import json
import os
from collections.abc import Iterator
from typing import Any

import httpx

DEFAULT_TIMEOUT = httpx.Timeout(10.0, read=300.0)
"""读超时给 300 秒。一轮带工具调用的对话几十秒是常态，默认 5 秒会在模型还在
想的时候把连接掐了，表现为"回答到一半就没了"。"""


def api_base() -> str:
    return os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


class ApiError(RuntimeError):
    """服务端返回了错误信封。``code`` 用于让 UI 区分「该重新登录」和「真出错了」。"""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(f"[{status}] {code}: {message}")
        self.status = status
        self.code = code
        self.message = message


def _unwrap(response: httpx.Response) -> Any:
    """统一拆错误信封。

    服务端的错误格式是 ``{"error": {"code", "message", "request_id"}}``，这里把它
    转成 ``ApiError``，UI 就永远不需要碰 ``response.status_code``。
    """
    if response.is_success:
        return None if response.status_code == 204 else response.json()

    code, message = "http_error", response.text[:300]
    try:
        payload = response.json().get("error") or {}
        code = payload.get("code", code)
        message = payload.get("message", message)
    except (ValueError, AttributeError):
        pass
    raise ApiError(response.status_code, code, message)


class ApiClient:
    """一个 token 一个客户端。``token=None`` 时只能打不需要认证的接口。"""

    def __init__(self, token: str | None = None) -> None:
        self._token = token

    # -- 底层 ---------------------------------------------------------------

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = dict(extra or {})
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        with httpx.Client(base_url=api_base(), timeout=DEFAULT_TIMEOUT) as client:
            response = client.request(method, path, headers=self._headers(), **kwargs)
        return _unwrap(response)

    # -- auth ---------------------------------------------------------------

    def register(self, email: str, password: str) -> dict:
        return self._request("POST", "/v1/auth/register", json={"email": email, "password": password})

    def login(self, email: str, password: str) -> dict:
        return self._request("POST", "/v1/auth/login", json={"email": email, "password": password})

    def logout(self, refresh_token: str) -> None:
        self._request("POST", "/v1/auth/logout", json={"refresh_token": refresh_token})

    # -- 会话 ---------------------------------------------------------------

    def list_conversations(self, limit: int = 20) -> list[dict]:
        return self._request("GET", "/v1/conversations", params={"limit": limit})

    def create_conversation(self, title: str | None = None) -> dict:
        return self._request("POST", "/v1/conversations", json={"title": title})

    def delete_conversation(self, conversation_id: str) -> None:
        self._request("DELETE", f"/v1/conversations/{conversation_id}")

    def list_messages(self, conversation_id: str, limit: int = 50) -> list[dict]:
        return self._request(
            "GET", f"/v1/conversations/{conversation_id}/messages", params={"limit": limit}
        )

    # -- 自助 ---------------------------------------------------------------

    def me(self) -> dict:
        return self._request("GET", "/v1/users/me")

    def profile(self) -> dict:
        return self._request("GET", "/v1/users/me/profile")

    def clear_profile(self) -> None:
        self._request("DELETE", "/v1/users/me/profile")

    def usage(self, days: int = 30) -> dict:
        return self._request("GET", "/v1/users/me/usage", params={"days": days})

    def capabilities(self) -> dict:
        return self._request("GET", "/v1/capabilities")

    # -- 流式发消息 ---------------------------------------------------------

    def send_message(
        self,
        conversation_id: str,
        content: str,
        idempotency_key: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """发消息，逐个 yield SSE 事件（``{"event": ..., "data": {...}}``）。

        ``idempotency_key`` 让超时重试不会重复烧 token —— 一轮对话几千 token，
        重复扣费是真金白银。
        """
        headers = self._headers({"Accept": "text/event-stream"})
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key

        with httpx.Client(base_url=api_base(), timeout=DEFAULT_TIMEOUT) as client:
            with client.stream(
                "POST",
                f"/v1/conversations/{conversation_id}/messages",
                json={"content": content},
                headers=headers,
            ) as response:
                if not response.is_success:
                    # 流还没开始就被拒（401 / 404 / 校验失败），此时 body 是普通
                    # JSON 错误信封，必须先 read() 才能读到。
                    response.read()
                    _unwrap(response)

                event_name = "message"
                for line in response.iter_lines():
                    if not line:
                        # 空行 = 一个事件结束。sse-starlette 每条只发一个 data，
                        # 所以这里不需要拼多行。
                        event_name = "message"
                        continue
                    if line.startswith("event:"):
                        event_name = line[6:].strip()
                    elif line.startswith("data:"):
                        raw = line[5:].strip()
                        if not raw:
                            continue
                        try:
                            payload = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        yield {"event": event_name, "data": payload}


def readyz() -> tuple[bool, str]:
    """探服务端是否可用，给 UI 一个「后端没起」的明确提示而不是一堆连接异常。"""
    try:
        with httpx.Client(base_url=api_base(), timeout=httpx.Timeout(5.0)) as client:
            response = client.get("/readyz")
        if response.is_success:
            return True, "ok"
        return False, _describe_not_ready(response)
    except httpx.HTTPError as exc:
        return False, f"cannot reach {api_base()} ({type(exc).__name__})"


def _describe_not_ready(response: httpx.Response) -> str:
    try:
        return (response.json().get("error") or {}).get("message", response.text[:200])
    except (ValueError, AttributeError):
        return response.text[:200]
