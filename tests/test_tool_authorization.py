"""工具级授权与沙箱隔离。

这个文件测的是 ``ToolAuthorizationMiddleware``，不经过 HTTP。它是双层防御里的
第二层 —— 也就是真正的强制点，所以这里的每个用例都对应一条"如果这层漏了会
发生什么"。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import ToolMessage

from movie_agent.authz import (
    FS_PATH_PARAMS,
    WRITE_TOOLS,
    PathEscape,
    ToolAuthorizationMiddleware,
    confine_path,
    user_sandbox,
)
from movie_agent.context import ROLE_READONLY, ROLE_USER, MovieAgentContext


# ---------------------------------------------------------------------------
# 替身：只需要 request.runtime.context 和 request.tool_call
# ---------------------------------------------------------------------------

@dataclass
class FakeRuntime:
    context: Any


@dataclass
class FakeToolRequest:
    tool_call: dict[str, Any]
    runtime: FakeRuntime

    def override(self, **changes: Any) -> "FakeToolRequest":
        """模拟框架的 ``ToolCallRequest.override``。

        框架里直接给字段赋值已经 deprecated，必须走 override 返回新对象。替身
        照抄这个语义，否则测试会掩盖真实用法上的错误。
        """
        return FakeToolRequest(
            tool_call=changes.get("tool_call", self.tool_call),
            runtime=changes.get("runtime", self.runtime),
        )


@dataclass
class FakeTool:
    name: str


@dataclass
class FakeModelRequest:
    tools: list[FakeTool]
    runtime: FakeRuntime

    def override(self, **changes: Any) -> "FakeModelRequest":
        return FakeModelRequest(
            tools=changes.get("tools", self.tools),
            runtime=changes.get("runtime", self.runtime),
        )


def make_request(tool: str, args: dict[str, Any], *, user_id="alice", role=ROLE_USER):
    return FakeToolRequest(
        tool_call={"name": tool, "args": args, "id": "call-1"},
        runtime=FakeRuntime(context=MovieAgentContext(user_id=user_id, role=role)),
    )


async def run(middleware, request):
    """跑一遍中间件，返回 ``(结果, 真正传给工具的参数)``。"""
    seen: dict[str, Any] = {}

    async def handler(req):
        seen.update(req.tool_call["args"])
        return ToolMessage(content="ok", tool_call_id=req.tool_call["id"], name=req.tool_call["name"])

    result = await middleware.awrap_tool_call(request, handler)
    return result, seen


@pytest.fixture
def root(tmp_path) -> Path:
    return tmp_path / "watchlist"


@pytest.fixture
def middleware(root) -> ToolAuthorizationMiddleware:
    return ToolAuthorizationMiddleware(root)


def denied(result) -> bool:
    return isinstance(result, ToolMessage) and result.status == "error" and "not permitted" in result.content


# ---------------------------------------------------------------------------
# 角色授权
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tool", sorted(WRITE_TOOLS))
async def test_readonly_role_cannot_call_write_tools(middleware, tool):
    args = {p: "watchlist.md" for p in FS_PATH_PARAMS[tool]}
    result, seen = await run(middleware, make_request(tool, args, role=ROLE_READONLY))

    assert denied(result), f"{tool} should have been refused"
    assert not seen, "the tool must never have been reached"


async def test_readonly_role_can_still_read(middleware):
    result, seen = await run(
        middleware, make_request("read_text_file", {"path": "watchlist.md"}, role=ROLE_READONLY)
    )
    assert not denied(result)
    assert seen["path"].endswith("/watchlist.md")


async def test_denial_is_a_tool_message_not_an_exception(middleware):
    """越权返回 ToolMessage 而不是抛异常。

    抛异常会让整轮对话失败，用户只看到一个 500；返回消息则让模型知道"这条路
    不通"，可以换个说法继续。
    """
    result, _ = await run(
        middleware, make_request("write_file", {"path": "x.md"}, role=ROLE_READONLY)
    )
    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call-1"
    assert result.status == "error"


async def test_missing_user_context_is_refused(middleware):
    """context 没装上说明调用方绕过了 server 的装配路径。

    这时候拒绝执行，而不是拿一个默认身份去跑 —— 那正是改造前 ``user_id()``
    回落到 ``"default"`` 的老毛病。
    """
    request = FakeToolRequest(
        tool_call={"name": "read_text_file", "args": {"path": "a.md"}, "id": "call-1"},
        runtime=FakeRuntime(context=None),
    )
    result, seen = await run(middleware, request)
    assert denied(result)
    assert not seen


async def test_model_never_sees_write_tools_when_readonly(middleware):
    """第一层：过滤 ``request.tools``，省 token 也省掉无效往返。"""
    tools = [FakeTool("read_text_file"), FakeTool("write_file"), FakeTool("search_movies")]
    captured: list[str] = []

    async def handler(req):
        captured.extend(t.name for t in req.tools)
        return "response"

    request = FakeModelRequest(
        tools=tools,
        runtime=FakeRuntime(context=MovieAgentContext(user_id="alice", role=ROLE_READONLY)),
    )
    await middleware.awrap_model_call(request, handler)

    assert "write_file" not in captured
    assert {"read_text_file", "search_movies"} <= set(captured)


async def test_normal_role_sees_everything(middleware):
    tools = [FakeTool("read_text_file"), FakeTool("write_file")]
    captured: list[str] = []

    async def handler(req):
        captured.extend(t.name for t in req.tools)
        return "response"

    request = FakeModelRequest(
        tools=tools,
        runtime=FakeRuntime(context=MovieAgentContext(user_id="alice", role=ROLE_USER)),
    )
    await middleware.awrap_model_call(request, handler)
    assert captured == ["read_text_file", "write_file"]


# ---------------------------------------------------------------------------
# 沙箱隔离
# ---------------------------------------------------------------------------

async def test_relative_paths_land_in_the_callers_own_directory(middleware, root):
    result, seen = await run(middleware, make_request("write_file", {"path": "watchlist.md"}))
    assert not denied(result)
    assert seen["path"] == str((root / "alice" / "watchlist.md").resolve())


async def test_two_users_get_two_directories(middleware, root):
    _, alice_args = await run(middleware, make_request("write_file", {"path": "w.md"}, user_id="alice"))
    _, bob_args = await run(middleware, make_request("write_file", {"path": "w.md"}, user_id="bob"))

    assert alice_args["path"] != bob_args["path"]
    assert "/alice/" in alice_args["path"]
    assert "/bob/" in bob_args["path"]


async def test_reaching_into_another_users_directory_is_blocked(middleware, root):
    """改造前的真实漏洞：共用一个 ``data/watchlist/``，A 直接读得到 B 的清单。"""
    user_sandbox(root, "bob")  # 让目标真的存在，别让用例靠"文件不存在"通过
    result, seen = await run(
        middleware, make_request("read_text_file", {"path": "../bob/watchlist.md"})
    )
    assert denied(result)
    assert not seen


@pytest.mark.parametrize(
    "path",
    [
        "../../.env",
        "../../../etc/passwd",
        "..",
        "../",
        "sub/../../../outside.md",
        "/etc/passwd",
        "/etc/../etc/passwd",
        "/",
        "",
        "   ",
    ],
)
async def test_path_escapes_are_blocked(middleware, path):
    result, seen = await run(middleware, make_request("read_text_file", {"path": path}))
    assert denied(result), f"{path!r} escaped"
    assert not seen


async def test_symlink_escape_is_blocked(middleware, root):
    """软链接是绕过前缀校验的经典手法。

    ``resolve()`` 会展开它，所以校验必须发生在 resolve **之后** —— 先比字符串
    前缀再 resolve 就会被这一手打穿。
    """
    sandbox = user_sandbox(root, "alice")
    (sandbox / "escape_link").symlink_to("/etc")

    result, seen = await run(
        middleware, make_request("read_text_file", {"path": "escape_link/passwd"})
    )
    assert denied(result)
    assert not seen


async def test_symlink_into_another_sandbox_is_blocked(middleware, root):
    alice_dir = user_sandbox(root, "alice")
    bob_dir = user_sandbox(root, "bob")
    (bob_dir / "secret.md").write_text("bob's private list")
    (alice_dir / "peek").symlink_to(bob_dir)

    result, seen = await run(middleware, make_request("read_text_file", {"path": "peek/secret.md"}))
    assert denied(result)
    assert not seen


async def test_every_path_param_is_rewritten_not_just_the_first(middleware, root):
    """``move_file`` 有两个路径参数。只重写一个就等于留了个后门。"""
    result, seen = await run(
        middleware, make_request("move_file", {"source": "a.md", "destination": "b.md"})
    )
    assert not denied(result)
    sandbox = str((root / "alice").resolve())
    assert seen["source"].startswith(sandbox)
    assert seen["destination"].startswith(sandbox)


async def test_move_file_is_blocked_if_either_side_escapes(middleware, root):
    result, seen = await run(
        middleware, make_request("move_file", {"source": "a.md", "destination": "../../stolen.md"})
    )
    assert denied(result)
    assert not seen


async def test_path_arrays_are_rewritten_elementwise(middleware, root):
    """``read_multiple_files`` 收的是 ``paths`` 数组，不是 ``path``。"""
    result, seen = await run(
        middleware, make_request("read_multiple_files", {"paths": ["a.md", "b.md"]})
    )
    assert not denied(result)
    sandbox = str((root / "alice").resolve())
    assert all(p.startswith(sandbox) for p in seen["paths"])


async def test_one_bad_element_blocks_the_whole_array(middleware):
    result, seen = await run(
        middleware, make_request("read_multiple_files", {"paths": ["a.md", "../../.env"]})
    )
    assert denied(result)
    assert not seen


async def test_non_file_tools_pass_through_untouched(middleware):
    """TMDB / RAG / skills 没有路径可谈，不该被沙箱逻辑碰。"""
    result, seen = await run(middleware, make_request("search_movies", {"query": "Inception"}))
    assert not denied(result)
    assert seen == {"query": "Inception"}


async def test_every_filesystem_tool_has_a_path_mapping():
    """映射表的完整性。

    漏掉一个工具就等于开一个绕过沙箱的后门：``FS_PATH_PARAMS.get(name)`` 返回
    ``None`` 时会当成"非文件工具"直接放行。参数名是实跑 server schema 抄的。
    """
    assert FS_PATH_PARAMS["read_multiple_files"] == ("paths",)
    assert FS_PATH_PARAMS["move_file"] == ("source", "destination")
    assert FS_PATH_PARAMS["list_allowed_directories"] == ()
    # 除了 list_allowed_directories，每个工具都必须至少有一个路径参数
    for name, params in FS_PATH_PARAMS.items():
        if name != "list_allowed_directories":
            assert params, f"{name} has no path parameter mapped"


# ---------------------------------------------------------------------------
# confine_path 直接测
# ---------------------------------------------------------------------------

def test_absolute_path_inside_the_sandbox_is_accepted(tmp_path):
    """模型经常把上一轮返回的绝对路径回显进来，这必须能用。"""
    sandbox = user_sandbox(tmp_path, "alice")
    target = sandbox / "watchlist.md"
    assert confine_path(str(target), sandbox) == str(target)


def test_single_segment_absolute_path_is_remapped(tmp_path):
    """``/watchlist.md`` 当成"把沙箱根误当文件系统根"的笔误处理。"""
    sandbox = user_sandbox(tmp_path, "alice")
    assert confine_path("/watchlist.md", sandbox) == str(sandbox / "watchlist.md")


def test_deep_absolute_path_outside_is_rejected_not_remapped(tmp_path):
    """``/etc/passwd`` 不会被"去根重拼"成 ``<sandbox>/etc/passwd``。

    那样虽然也关在沙箱里，但把一次明确的越权尝试静默改写成了一个莫名其妙的
    合法路径：审计信号丢了，模型也得不到"这条路不通"的反馈。
    """
    sandbox = user_sandbox(tmp_path, "alice")
    with pytest.raises(PathEscape):
        confine_path("/etc/passwd", sandbox)


def test_relative_to_alone_would_not_have_caught_this(tmp_path):
    """回归用例，钉住一个真被写出来过的 bug。

    第一版在绝对路径分支里用 ``candidate.relative_to(sandbox)`` 成功就提前
    return。但 ``relative_to`` 是**纯字面比较** —— ``<sandbox>/../../evil`` 对着
    sandbox 算相对路径是"成功"的，于是它绕过了前缀校验。修法是让所有分支合流，
    最后统一过一次 resolve 后的前缀校验。
    """
    sandbox = user_sandbox(tmp_path, "alice")
    sneaky = f"{sandbox}/../../evil.md"

    # 字面比较认为它在沙箱里
    assert Path(sneaky).is_relative_to(sandbox)
    # 但 resolve 之后并不是，所以必须被拦
    with pytest.raises(PathEscape):
        confine_path(sneaky, sandbox)
