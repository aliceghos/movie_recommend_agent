"""工具调用的记账：``ToolDebugHandler`` 的配对与落库前的摘要处理。

这两处都不影响对话能不能跑通，所以出了错没人会发现 —— 表现只是
``tool_invocations`` 里多了几行 output 为空的记录，看着像工具本来就没返回东西。
但这张表是排查「回复为什么慢」「工具到底调没调成」的唯一依据，记错了比不记更糟。

这里的用例来自一次真实的端到端跑：一轮对话产生 6 条 ``tool_invocations``，其中
3 条 ``duration_ms`` 为空、``result_preview`` 是字符串 ``"None"``。
"""

from movie_agent.callbacks import ToolDebugHandler
from server.routers.conversations import _preview


def start(handler: ToolDebugHandler, name: str, run_id: str, args: str = "{}") -> None:
    handler.on_tool_start({"name": name}, args, run_id=run_id)


# ---------------------------------------------------------------------------
# 并行批次的配对
# ---------------------------------------------------------------------------

def test_parallel_batch_pairs_each_end_with_its_own_start():
    """模型是**成批**发工具调用的，三个 start 之后才来第一个 end 是常态。

    按到达顺序配对（``self.calls[-1]``）在这里必然错：批次里先开始的调用永远拿不到
    自己的 output，最后开始的那个则被别人的 output 反复覆盖，还配上一个从无关起点
    算出来的耗时。
    """
    handler = ToolDebugHandler()

    start(handler, "get_genres", "r1")
    start(handler, "search_movies", "r2")
    start(handler, "discover_movies", "r3")

    # 结束顺序和开始顺序不同 —— 谁先返回取决于网络，不取决于调用顺序
    handler.on_tool_end("genre list", run_id="r1")
    handler.on_tool_end("movie rows", run_id="r3")
    handler.on_tool_end("search hits", run_id="r2")

    recorded = {c["name"]: c for c in handler.calls}
    assert len(handler.calls) == 3
    assert recorded["get_genres"]["output"] == "genre list"
    assert recorded["search_movies"]["output"] == "search hits"
    assert recorded["discover_movies"]["output"] == "movie rows"

    for call in handler.calls:
        assert call["duration_ms"] is not None, f"{call['name']} lost its timing"


def test_an_error_in_a_batch_only_marks_its_own_call():
    handler = ToolDebugHandler()
    start(handler, "search_movies", "r1")
    start(handler, "write_file", "r2")

    handler.on_tool_error(RuntimeError("permission denied"), run_id="r2")
    handler.on_tool_end("3 results", run_id="r1")

    recorded = {c["name"]: c for c in handler.calls}
    assert recorded["write_file"]["error"] == "permission denied"
    assert recorded["write_file"]["output"] is None
    assert recorded["search_movies"]["error"] is None
    assert recorded["search_movies"]["output"] == "3 results"


def test_sequential_calls_still_work_without_a_run_id():
    """没有 ``run_id`` 时退回「最后一个」。

    顺序调用下这个退路是对的，而顺序调用正是缺 run_id 所暗示的情形。
    """
    handler = ToolDebugHandler()
    handler.on_tool_start({"name": "load_skill"}, "{}")
    handler.on_tool_end("skill text")

    assert handler.calls[0]["output"] == "skill text"


def test_tool_message_content_is_unwrapped():
    class FakeToolMessage:
        content = "the actual text"

    handler = ToolDebugHandler()
    start(handler, "search_movies", "r1")
    handler.on_tool_end(FakeToolMessage(), run_id="r1")

    assert handler.calls[0]["output"] == "the actual text"


def test_in_flight_entries_do_not_accumulate():
    """一轮结束后不该留下悬挂的 run_id —— agent 是进程级单例，泄漏会一直涨。"""
    handler = ToolDebugHandler()
    for i in range(5):
        start(handler, "search_movies", f"r{i}")
    for i in range(5):
        handler.on_tool_end("ok", run_id=f"r{i}")

    assert handler._in_flight == {}


# ---------------------------------------------------------------------------
# 落库前的摘要
# ---------------------------------------------------------------------------

def test_missing_output_is_stored_as_null_not_the_string_none():
    """``output`` 键存在但为 None 时不能写成字面量 ``"None"``。

    实测踩过：``str(call.get("output", ""))`` 对 ``{"output": None}`` 返回 ``"None"``，
    于是表里出现四个字符的假输出，排查时看着像工具真返回了这么个东西。
    """
    assert _preview({"name": "get_genres", "output": None}) is None
    assert _preview({"name": "get_genres"}) is None
    assert _preview({"name": "get_genres", "output": ""}) is None


def test_error_is_recorded_instead_of_being_dropped():
    call = {"name": "write_file", "output": None, "error": "permission denied"}
    assert _preview(call) == "ERROR: permission denied"


def test_long_output_is_truncated():
    preview = _preview({"name": "search_movies", "output": "x" * 5000})
    assert preview is not None
    assert len(preview) == 500
