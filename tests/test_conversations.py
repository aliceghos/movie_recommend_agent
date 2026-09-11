"""对话接口：SSE 契约、落库、幂等键、同会话串行化。

LLM 用替身。这里要验的是**接口与并发的正确性**，不是模型质量：真模型会让每个
用例慢几十秒，而且回答不确定，反而测不出"消息有没有丢"这类问题。
"""

import asyncio
import json
import time
from typing import Any, AsyncIterator

import pytest
from sqlalchemy import func, select

from server.db.models import IdempotencyKey, Message, ToolInvocation, UsageRecord

FAKE_REPLY = "You might like Chungking Express."


def fake_chat_stream(*, delay: float = 0.0, tool_calls: list[dict] | None = None, record: list | None = None):
    """造一个 ``chat_stream`` 替身。

    ``record`` 用来记进出时刻 —— 会话锁的测试靠"两段执行区间不重叠"来断言，
    这是唯一能从外部观察到串行化的方式。
    """

    async def _stream(runtime, user_message, *, thread_id, context) -> AsyncIterator[dict[str, Any]]:
        started = time.perf_counter()
        for chunk in ("You might like ", "Chungking Express."):
            yield {"type": "token", "content": chunk}
            if delay:
                await asyncio.sleep(delay)
        yield {
            "type": "done",
            "response": FAKE_REPLY,
            "token_usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            "tool_calls": tool_calls or [],
        }
        if record is not None:
            record.append((started, time.perf_counter()))

    return _stream


@pytest.fixture
def patch_stream(monkeypatch):
    """替换路由模块里的 ``chat_stream``。

    路由是 ``from movie_agent.agent import chat_stream`` 导进来的，所以必须打
    ``server.routers.conversations`` 这个名字，打原模块没用。
    """

    def _patch(stream):
        monkeypatch.setattr("server.routers.conversations.chat_stream", stream)

    return _patch


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    name = "message"
    for line in text.splitlines():
        if line.startswith("event:"):
            name = line[6:].strip()
        elif line.startswith("data:"):
            payload = line[5:].strip()
            if payload:
                events.append((name, json.loads(payload)))
    return events


async def send(client, account, conversation_id: str, content: str = "recommend a movie", key: str | None = None):
    headers = dict(account.headers)
    if key:
        headers["Idempotency-Key"] = key
    response = await client.post(
        f"/v1/conversations/{conversation_id}/messages",
        json={"content": content},
        headers=headers,
    )
    return response, parse_sse(response.text)


@pytest.fixture
async def conversation(client, alice) -> dict:
    created = await client.post("/v1/conversations", json={}, headers=alice.headers)
    assert created.status_code == 201
    return created.json()


# ---------------------------------------------------------------------------
# SSE 契约
# ---------------------------------------------------------------------------

async def test_stream_emits_tokens_then_done(client, alice, conversation, patch_stream):
    patch_stream(fake_chat_stream())
    response, events = await send(client, alice, conversation["id"])

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    kinds = [name for name, _ in events]
    assert kinds[:2] == ["token", "token"]
    assert kinds[-1] == "done"

    done = events[-1][1]
    assert done["response"] == FAKE_REPLY
    assert done["token_usage"]["total_tokens"] == 120
    assert done["message_id"]
    assert done["replayed"] is False


async def test_tool_events_are_forwarded(client, alice, conversation, patch_stream):
    async def with_tools(runtime, user_message, *, thread_id, context):
        yield {"type": "tool_start", "name": "search_movies", "input": '{"query": "noir"}'}
        yield {"type": "tool_end", "name": "search_movies", "output": "3 results"}
        yield {"type": "token", "content": FAKE_REPLY}
        yield {
            "type": "done",
            "response": FAKE_REPLY,
            "token_usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "tool_calls": [
                {"name": "search_movies", "input": '{"query": "noir"}', "output": "3 results", "duration_ms": 42}
            ],
        }

    patch_stream(with_tools)
    _, events = await send(client, alice, conversation["id"])

    kinds = [name for name, _ in events]
    assert "tool_start" in kinds and "tool_end" in kinds


async def test_both_messages_and_usage_are_persisted(client, alice, conversation, patch_stream, db):
    patch_stream(
        fake_chat_stream(
            tool_calls=[
                {"name": "search_movies", "input": '{"q": "noir"}', "output": "ok", "duration_ms": 42}
            ]
        )
    )
    await send(client, alice, conversation["id"], content="recommend a noir")

    messages = (
        await db.execute(select(Message).order_by(Message.created_at))
    ).scalars().all()
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[0].content == "recommend a noir"
    assert messages[1].content == FAKE_REPLY

    invocations = (await db.execute(select(ToolInvocation))).scalars().all()
    assert len(invocations) == 1
    assert invocations[0].tool_name == "search_movies"
    assert invocations[0].duration_ms == 42
    assert invocations[0].arguments == {"raw": '{"q": "noir"}'}

    usage = (await db.execute(select(UsageRecord))).scalars().all()
    assert len(usage) == 1
    assert usage[0].total_tokens == 120
    assert usage[0].model, "model must be recorded or the bill cannot be split per model"


async def test_history_is_readable_afterwards(client, alice, conversation, patch_stream):
    patch_stream(fake_chat_stream())
    await send(client, alice, conversation["id"], content="hello there")

    history = (
        await client.get(f"/v1/conversations/{conversation['id']}/messages", headers=alice.headers)
    ).json()
    assert [m["content"] for m in history] == ["hello there", FAKE_REPLY]


async def test_first_message_becomes_the_title(client, alice, conversation, patch_stream):
    patch_stream(fake_chat_stream())
    await send(client, alice, conversation["id"], content="something about film noir")

    listed = (await client.get("/v1/conversations", headers=alice.headers)).json()
    assert listed[0]["title"] == "something about film noir"


async def test_a_failure_inside_the_stream_becomes_an_error_event(
    client, alice, conversation, patch_stream, db
):
    """流内失败不能靠状态码 —— 响应头早就发走了。

    而且必须回滚：用户消息在流开始前就入库了，如果不回滚就会留下一条"用户问了
    但没人答"的孤儿记录。
    """

    async def explodes(runtime, user_message, *, thread_id, context):
        yield {"type": "token", "content": "starting..."}
        raise RuntimeError("the model gateway hung up")

    patch_stream(explodes)
    response, events = await send(client, alice, conversation["id"])

    assert response.status_code == 200  # 头已经发出去了，改不了
    assert events[-1][0] == "error"
    assert events[-1][1]["code"] == "internal_error"

    count = await db.scalar(select(func.count(Message.id)))
    assert count == 0, "a failed turn must not leave a half-written exchange behind"


# ---------------------------------------------------------------------------
# 幂等
# ---------------------------------------------------------------------------

async def test_replaying_an_idempotency_key_returns_the_same_message(
    client, alice, conversation, patch_stream, db
):
    """一轮对话烧几千 token，前端超时重试不能重复扣费。"""
    patch_stream(fake_chat_stream())

    _, first = await send(client, alice, conversation["id"], key="retry-me")
    _, second = await send(client, alice, conversation["id"], key="retry-me")

    first_done = first[-1][1]
    second_done = second[-1][1]

    assert second_done["message_id"] == first_done["message_id"]
    assert second_done["replayed"] is True
    assert second[-1][0] == "done"

    # 只烧了一次 token，也只留了一轮消息
    assert await db.scalar(select(func.count(UsageRecord.id))) == 1
    assert await db.scalar(select(func.count(Message.id))) == 2


async def test_replay_does_not_call_the_model_again(client, alice, conversation, patch_stream):
    calls: list[str] = []

    async def counting(runtime, user_message, *, thread_id, context):
        calls.append(user_message)
        yield {
            "type": "done",
            "response": FAKE_REPLY,
            "token_usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "tool_calls": [],
        }

    patch_stream(counting)
    await send(client, alice, conversation["id"], key="once")
    await send(client, alice, conversation["id"], key="once")

    assert len(calls) == 1


async def test_idempotency_keys_are_scoped_per_user(client, alice, bob, patch_stream, db):
    """同一个 key 字符串在两个用户之间不能互相命中。

    唯一约束落在 ``(user_id, key)``：否则 A 用一个好猜的 key，就能拿到 B 的回复。
    """
    patch_stream(fake_chat_stream())
    alice_conv = (await client.post("/v1/conversations", json={}, headers=alice.headers)).json()
    bob_conv = (await client.post("/v1/conversations", json={}, headers=bob.headers)).json()

    _, alice_events = await send(client, alice, alice_conv["id"], key="shared-key")
    _, bob_events = await send(client, bob, bob_conv["id"], key="shared-key")

    assert alice_events[-1][1]["message_id"] != bob_events[-1][1]["message_id"]
    assert bob_events[-1][1]["replayed"] is False
    assert await db.scalar(select(func.count(IdempotencyKey.id))) == 2


async def test_no_key_means_no_deduplication(client, alice, conversation, patch_stream, db):
    """不带 key 就是两轮独立的对话，不该被当成重放。"""
    patch_stream(fake_chat_stream())
    await send(client, alice, conversation["id"])
    await send(client, alice, conversation["id"])

    assert await db.scalar(select(func.count(Message.id))) == 4
    assert await db.scalar(select(func.count(UsageRecord.id))) == 2


# ---------------------------------------------------------------------------
# 同会话串行化
# ---------------------------------------------------------------------------

async def test_concurrent_messages_to_one_conversation_are_serialized(
    client, alice, conversation, patch_stream, db
):
    """用户手快连发两条时，两个请求会并发写同一个 thread 的 checkpoint 互相覆盖。

    这种 bug 只有压测才碰得到，但一旦发生就是消息凭空消失，从日志上完全看不
    出来。防线是发消息前取一个事务级 advisory lock。

    断言分两层：
      - 两段执行区间**不重叠** —— 锁真的生效了
      - 四条消息一条不少 —— 串行的结果是正确的
    """
    intervals: list[tuple[float, float]] = []
    patch_stream(fake_chat_stream(delay=0.05, record=intervals))

    first, second = await asyncio.gather(
        send(client, alice, conversation["id"], content="first"),
        send(client, alice, conversation["id"], content="second"),
    )
    assert first[0].status_code == second[0].status_code == 200
    assert first[1][-1][0] == second[1][-1][0] == "done"

    assert len(intervals) == 2
    earlier, later = sorted(intervals)
    assert earlier[1] <= later[0], f"the two turns overlapped: {intervals}"

    rows = (await db.execute(select(Message).order_by(Message.created_at))).scalars().all()
    assert len(rows) == 4, "no message may be lost"
    assert sorted(m.content for m in rows if m.role == "user") == ["first", "second"]


async def test_concurrent_messages_to_different_conversations_are_not_blocked(
    client, alice, patch_stream, db
):
    """锁的粒度是会话，不是用户。

    锁到用户就等于同一个人不能开两个标签页 —— 那不是并发保护，那是把并发关掉。
    """
    first_conv = (await client.post("/v1/conversations", json={}, headers=alice.headers)).json()
    second_conv = (await client.post("/v1/conversations", json={}, headers=alice.headers)).json()

    intervals: list[tuple[float, float]] = []
    patch_stream(fake_chat_stream(delay=0.05, record=intervals))

    await asyncio.gather(
        send(client, alice, first_conv["id"], content="a"),
        send(client, alice, second_conv["id"], content="b"),
    )

    earlier, later = sorted(intervals)
    assert earlier[1] > later[0], "different conversations should have run concurrently"
    assert await db.scalar(select(func.count(Message.id))) == 4


# ---------------------------------------------------------------------------
# 会话生命周期
# ---------------------------------------------------------------------------

async def test_deleting_a_conversation_cascades_to_its_messages(
    client, alice, conversation, patch_stream, db
):
    patch_stream(fake_chat_stream())
    await send(client, alice, conversation["id"])
    assert await db.scalar(select(func.count(Message.id))) == 2

    deleted = await client.delete(f"/v1/conversations/{conversation['id']}", headers=alice.headers)
    assert deleted.status_code == 204

    assert await db.scalar(select(func.count(Message.id))) == 0
    # 用量记录**留着**：会话删了不等于账不用算
    assert await db.scalar(select(func.count(UsageRecord.id))) == 1


async def test_empty_message_is_rejected(client, alice, conversation):
    response = await client.post(
        f"/v1/conversations/{conversation['id']}/messages",
        json={"content": ""},
        headers=alice.headers,
    )
    assert response.status_code == 422
