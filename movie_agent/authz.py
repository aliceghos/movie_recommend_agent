"""工具级授权与沙箱隔离。

改造前 filesystem MCP server 以 ``data/watchlist/`` 为根单进程启动，所有用户
共用一个目录 —— A 能 ``read_text_file("b_的清单.md")`` 读走 B 的观影清单。

这里**不改 server 的启动方式**（按用户 spawn 进程会让进程数随用户数线性爆炸），
而是在工具调用的入口做两件事：

1. **角色授权**（双层）
   - ``awrap_model_call``：按角色过滤 ``request.tools``，模型压根看不见没权限的
     工具。省 token，也省掉"模型调了又被拒"的无效往返。
   - ``awrap_tool_call``：强制校验。第一层是给模型的提示，这一层才是防线 ——
     模型可以硬编造一个没在列表里的工具名，历史 checkpoint 里也可能残留旧工具
     调用。只靠过滤不算授权。

2. **路径重写**：把工具参数里的相对路径钉到 ``data/watchlist/{user_id}/``
   之下，``resolve()`` 之后校验前缀，拦 ``../`` 与绝对路径逃逸。

越权返回 ``ToolMessage`` 而不抛异常：模型看到"不允许"可以换个说法继续，抛异常
会让整轮对话直接失败。
"""

import sys
from pathlib import Path
from typing import Any, Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from movie_agent.context import ROLE_READONLY

# 写入类文件工具。只读角色拿不到这些。
WRITE_TOOLS = frozenset({"write_file", "edit_file", "move_file", "create_directory"})

# filesystem MCP server 的全部工具，及各自需要重写的路径参数名。
#
# 参数名是实跑 `@modelcontextprotocol/server-filesystem` 的 tool schema 抄下来
# 的，不是猜的：多数是 `path`，但 read_multiple_files 用 `paths` 数组、move_file
# 用 `source` + `destination`。漏掉任何一个都等于开了个绕过沙箱的后门。
FS_PATH_PARAMS: dict[str, tuple[str, ...]] = {
    "read_file": ("path",),
    "read_text_file": ("path",),
    "read_media_file": ("path",),
    "read_multiple_files": ("paths",),
    "write_file": ("path",),
    "edit_file": ("path",),
    "create_directory": ("path",),
    "list_directory": ("path",),
    "list_directory_with_sizes": ("path",),
    "directory_tree": ("path",),
    "move_file": ("source", "destination"),
    "search_files": ("path",),
    "get_file_info": ("path",),
    "list_allowed_directories": (),
}


class PathEscape(ValueError):
    """路径越出了该用户的沙箱。"""


def user_sandbox(root: Path, user_id: str) -> Path:
    """该用户的沙箱目录，不存在则建。"""
    path = (root / user_id).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def confine_path(raw: Any, sandbox: Path) -> str:
    """把一个路径参数钉进沙箱，越界就抛。

    绝对路径先按原样规范化：落在沙箱内就放行（模型常把上一轮返回的绝对
    路径回显进来）。落在外面则只网开一面：``/watchlist.md`` 这种只有单层文件名
    的，当成模型把沙箱根当成了文件系统根的笔误，重映射进沙箱；其余一律拒绝。

    不把 ``/etc/passwd`` 之类“去掉根再拼”成 ``<sandbox>/etc/passwd``：那虽然也关
    在沙箱里，但把一次明确的越权尝试静默改写成了一个莫名其妙的合法路径，
    既丢了审计信号，也让模型得不到“这条路不通”的反馈。

    **无论走哪条分支，最后都过同一个前缀校验**。这里不能提前 return：
    ``Path.relative_to`` 是纯字面比较，``<sandbox>/../../evil`` 对着 sandbox 算
    相对路径是「成功」的，只有 resolve 之后再比前缀才拦得住。

    ``strict=False`` 让不存在的路径也能规范化（write_file 要新建文件），同时会
    展开软链接 —— 沙箱里放一个指向 ``/etc`` 的软链接是绕过前缀校验的经典手法。
    """
    text = str(raw).strip()
    if not text:
        raise PathEscape("empty path")

    candidate = Path(text)
    if candidate.is_absolute():
        resolved = candidate.resolve(strict=False)
        if not _inside(resolved, sandbox):
            # 只救 "/单个文件名" 这一种形式
            parts = candidate.parts[1:]
            if len(parts) != 1:
                raise PathEscape(f"absolute path outside the sandbox: {text}")
            resolved = (sandbox / parts[0]).resolve(strict=False)
    else:
        resolved = (sandbox / candidate).resolve(strict=False)

    if not _inside(resolved, sandbox):
        raise PathEscape(f"path escapes the sandbox: {text}")
    return str(resolved)


def _inside(path: Path, sandbox: Path) -> bool:
    return path == sandbox or sandbox in path.parents


def allowed_tools(role: str, tool_names: list[str]) -> set[str]:
    """角色白名单。默认全放，只读角色去掉写入类。"""
    if role == ROLE_READONLY:
        return {n for n in tool_names if n not in WRITE_TOOLS}
    return set(tool_names)


class ToolAuthorizationMiddleware(AgentMiddleware):
    """按角色授权工具，并把文件路径限制在每用户沙箱内。"""

    def __init__(self, watchlist_root: Path) -> None:
        super().__init__()
        self._root = Path(watchlist_root)

    # -- 第一层：不让模型看见没权限的工具 --------------------------------

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[Any]],
    ) -> Any:
        role = getattr(request.runtime.context, "role", None)
        if role == ROLE_READONLY:
            kept = [t for t in request.tools if t.name not in WRITE_TOOLS]
            request = request.override(tools=kept)
        return await handler(request)

    # -- 第二层：真正的强制点 --------------------------------------------

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        context = request.runtime.context
        user_id = getattr(context, "user_id", None)
        role = getattr(context, "role", None)
        name = request.tool_call["name"]
        call_id = request.tool_call["id"]

        if not user_id:
            # context 没装上就说明调用方绕过了 server 的装配路径。这时候
            # 拒绝执行，而不是拿一个默认身份去跑。
            return self._deny(call_id, name, "missing user context")

        if role == ROLE_READONLY and name in WRITE_TOOLS:
            return self._deny(
                call_id, name,
                "your role is read-only and cannot modify files; "
                "you can still read and list the watchlist",
            )

        path_params = FS_PATH_PARAMS.get(name)
        if not path_params:
            # 非文件工具（TMDB / RAG / skills）无路径可谈，直接放行
            return await handler(request)

        try:
            args = self._rewrite_paths(dict(request.tool_call["args"]), path_params, user_id)
        except PathEscape as exc:
            print(f"[authz] blocked {name} for user {user_id}: {exc}", file=sys.stderr)
            return self._deny(
                call_id, name,
                "the requested path is outside your watchlist directory; "
                "use a plain relative filename such as 'watchlist.md'",
            )

        new_call = {**request.tool_call, "args": args}
        return await handler(request.override(tool_call=new_call))

    # -- 内部 -------------------------------------------------------------

    def _rewrite_paths(
        self, args: dict[str, Any], path_params: tuple[str, ...], user_id: str
    ) -> dict[str, Any]:
        sandbox = user_sandbox(self._root, user_id)
        for param in path_params:
            if param not in args or args[param] is None:
                continue
            value = args[param]
            if isinstance(value, list):
                args[param] = [confine_path(v, sandbox) for v in value]
            else:
                args[param] = confine_path(value, sandbox)
        return args

    @staticmethod
    def _deny(call_id: str, tool_name: str, reason: str) -> ToolMessage:
        """拒绝也是一条正常的工具返回。

        不抛异常：模型收到这条消息后能改用别的方式达成目标，抛异常则整轮对话
        直接失败，用户只会看到一个 500。
        """
        return ToolMessage(
            content=f"Tool '{tool_name}' was not permitted: {reason}",
            tool_call_id=call_id,
            name=tool_name,
            status="error",
        )
