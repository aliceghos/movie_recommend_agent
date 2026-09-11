"""对话路由。

发消息这一条是整个服务里唯一复杂的接口，它要同时满足四件事：

1. **流式**：SSE，事件类型沿用改造前 ``chat_stream`` 的契约
   （``token`` / ``tool_start`` / ``tool_end`` / ``done``），新增 ``error``。
2. **同会话串行**：Postgres advisory lock。
3. **幂等**：``Idempotency-Key`` 请求头。
4. **落库**：用户消息先入库；assistant 消息 + 工具轨迹 + 用量在 ``done`` 时
   同一个事务提交。

**SSE 里的错误不能靠 HTTP 状态码**：响应头在第一个 token 吐出去的时候就已经
发走了，之后再出错也改不成 500。所以流内的失败只能作为一个 ``error`` 事件送出，
客户端必须处理它 —— 这是流式接口和普通接口的根本差异。
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, AsyncIterator

from fastapi import APIRouter, Header, Query, status
from sqlalchemy import select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from movie_agent.agent import chat_stream
from movie_agent.context import MovieAgentContext
from movie_agent.llm import model_name
from server.db.models import Conversation, IdempotencyKey, Message, ToolInvocation, UsageRecord
from server.db.session import get_sessionmaker
from server.deps import AgentRuntime, CurrentUser, DbDep, OwnedConversation
from server.schemas import (
    ConversationCreateIn,
    ConversationOut,
    MessageIn,
    MessageOut,
)

router = APIRouter(prefix="/v1/conversations", tags=["conversations"])

MAX_PAGE_SIZE = 100
_TOOL_PREVIEW_CHARS = 500


@router.get("", response_model=list[ConversationOut])
async def list_conversations(
    user: CurrentUser,
    db: DbDep,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = 20,
    before: Annotated[datetime | None, Query()] = None,
) -> list[Conversation]:
    """我的会话，按最近活跃倒序。``before`` 是游标（上一页最后一条的 updated_at）。"""
    stmt = select(Conversation).where(Conversation.user_id == user.id)
    if before is not None:
        stmt = stmt.where(Conversation.updated_at < before)
    stmt = stmt.order_by(Conversation.updated_at.desc()).limit(limit)
    return list((await db.execute(stmt)).scalars())


@router.post("", response_model=ConversationOut, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    payload: ConversationCreateIn,
    user: CurrentUser,
    db: DbDep,
) -> Conversation:
    """建会话。``thread_id`` 由服务端生成，客户端不能指定。

    让客户端自带 thread_id 就等于让它去猜别人的 —— checkpoint 只认这个字符串。
    """
    conversation = Conversation(
        user_id=user.id,
        thread_id=uuid.uuid4().hex,
        title=payload.title,
    )
    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)
    return conversation


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(conversation: OwnedConversation, db: DbDep) -> None:
    """删会话。

    只删业务数据。对应的 checkpoint 会留在框架表里成为孤儿 —— 那是刻意的取舍：
    去动框架的内部表是把业务逻辑长在别人的 schema 上，版本一升就碎。
    checkpoint 该由定期清理任务按时间回收（本次不做）。
    """
    await db.delete(conversation)
    await db.commit()


@router.get("/{conversation_id}/messages", response_model=list[MessageOut])
async def list_messages(
    conversation: OwnedConversation,
    db: DbDep,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = 50,
    after: Annotated[datetime | None, Query()] = None,
) -> list[Message]:
    """会话历史。读 ``messages`` 表，不读 checkpoint。"""
    stmt = select(Message).where(Message.conversation_id == conversation.id)
    if after is not None:
        stmt = stmt.where(Message.created_at > after)
    stmt = stmt.order_by(Message.created_at).limit(limit)
    return list((await db.execute(stmt)).scalars())


def _sse(event: str, data: dict[str, Any]) -> dict[str, str]:
    return {"event": event, "data": json.dumps(data, ensure_ascii=False)}


async def _replayed_message(
    db: AsyncSession, user_id: uuid.UUID, key: str
) -> Message | None:
    """这个幂等键是否已经有结果了。"""
    stmt = select(IdempotencyKey).where(
        IdempotencyKey.user_id == user_id, IdempotencyKey.key == key
    )
    record = (await db.execute(stmt)).scalar_one_or_none()
    if record is None or record.response_message_id is None:
        return None
    return await db.get(Message, record.response_message_id)


@router.post("/{conversation_id}/messages")
async def send_message(
    payload: MessageIn,
    conversation: OwnedConversation,
    user: CurrentUser,
    db: DbDep,
    runtime: AgentRuntime,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> EventSourceResponse:
    """发消息，SSE 流式返回。

    注意这里**先把所有权校验和幂等检查做完，才开始流**：一旦
    ``EventSourceResponse`` 返回，状态码就定死是 200 了，那之后没法再拒绝请求。
    """
    if idempotency_key:
        replayed = await _replayed_message(db, user.id, idempotency_key)
        if replayed is not None:
            # 重放：直接返回首次的结果，不再烧一次 token
            async def replay() -> AsyncIterator[dict[str, str]]:
                yield _sse("done", {
                    "message_id": str(replayed.id),
                    "response": replayed.content,
                    "token_usage": None,
                    "replayed": True,
                })

            return EventSourceResponse(replay())

    conversation_id = conversation.id
    thread_id = conversation.thread_id
    context = MovieAgentContext(user_id=str(user.id), role=user.role)
    user_content = payload.content

    async def stream() -> AsyncIterator[dict[str, str]]:
        # 流式生成器跑在响应阶段，路由的请求级 session 此时可能已经关了，
        # 所以这里自己开一个。
        sessionmaker = get_sessionmaker()
        async with sessionmaker() as session:
            try:
                # 同会话串行化。事务级 advisory lock，事务结束自动释放，
                # 不会因为客户端断连而漏掉解锁。
                #
                # 用户手快连发两条时，两个请求会并发写同一个 thread 的
                # checkpoint 互相覆盖 —— 这种 bug 只有压测才碰得到，但一旦
                # 发生就是消息凭空消失，从日志上完全看不出来。
                await session.execute(
                    text("SELECT pg_advisory_xact_lock(hashtextextended(:tid, 0))"),
                    {"tid": thread_id},
                )

                if idempotency_key:
                    # 锁内再查一次：两个并发的同 key 请求都可能在锁外查到「没有」
                    replayed = await _replayed_message(session, user.id, idempotency_key)
                    if replayed is not None:
                        yield _sse("done", {
                            "message_id": str(replayed.id),
                            "response": replayed.content,
                            "token_usage": None,
                            "replayed": True,
                        })
                        await session.commit()
                        return

                    session.add(IdempotencyKey(key=idempotency_key, user_id=user.id))
                    try:
                        await session.flush()
                    except IntegrityError:
                        await session.rollback()
                        yield _sse("error", {
                            "code": "conflict",
                            "message": "a request with this Idempotency-Key is already in flight",
                        })
                        return

                session.add(Message(
                    conversation_id=conversation_id, role="user", content=user_content
                ))
                await session.flush()

                full_response = ""
                token_usage: dict[str, Any] | None = None
                tool_calls: list[dict[str, Any]] = []

                async for event in chat_stream(
                    runtime, user_content, thread_id=thread_id, context=context
                ):
                    kind = event["type"]
                    if kind == "token":
                        yield _sse("token", {"content": event["content"]})
                    elif kind == "tool_start":
                        yield _sse("tool_start", {
                            "name": event["name"], "input": str(event.get("input", ""))
                        })
                    elif kind == "tool_end":
                        yield _sse("tool_end", {
                            "name": event["name"], "output": str(event.get("output", ""))
                        })
                    elif kind == "done":
                        full_response = event.get("response") or ""
                        token_usage = event.get("token_usage")
                        tool_calls = event.get("tool_calls") or []

                assistant = Message(
                    conversation_id=conversation_id, role="assistant", content=full_response
                )
                session.add(assistant)
                await session.flush()

                for call in tool_calls:
                    session.add(ToolInvocation(
                        message_id=assistant.id,
                        tool_name=str(call.get("name", ""))[:128],
                        arguments=_as_jsonb(call.get("input")),
                        result_preview=_preview(call),
                        duration_ms=call.get("duration_ms"),
                    ))

                if token_usage:
                    session.add(UsageRecord(
                        user_id=user.id,
                        conversation_id=conversation_id,
                        message_id=assistant.id,
                        # TokenTracker 只报数量不报模型，模型名从配置取。
                        # 留空的话账单没法按模型拆分，换模型时也回溯不了。
                        model=str(token_usage.get("model") or model_name())[:128] or None,
                        prompt_tokens=int(token_usage.get("prompt_tokens") or 0),
                        completion_tokens=int(token_usage.get("completion_tokens") or 0),
                        total_tokens=int(token_usage.get("total_tokens") or 0),
                    ))

                if idempotency_key:
                    await session.execute(
                        update(IdempotencyKey)
                        .where(
                            IdempotencyKey.user_id == user.id,
                            IdempotencyKey.key == idempotency_key,
                        )
                        .values(response_message_id=assistant.id)
                    )

                conv = await session.get(Conversation, conversation_id)
                if conv is not None:
                    conv.updated_at = datetime.now(timezone.utc)
                    if not conv.title:
                        # 首条消息当标题，省掉一次 LLM 调用
                        conv.title = user_content[:60]

                # 用户消息、assistant 消息、工具轨迹、用量、幂等键一次提交。
                # 分开提交会留下"扣了 token 但没有消息"之类的半截状态。
                await session.commit()

                yield _sse("done", {
                    "message_id": str(assistant.id),
                    "response": full_response,
                    "token_usage": token_usage,
                    "replayed": False,
                })

            except Exception as exc:  # noqa: BLE001 - 流内异常改不了状态码，只能当事件送
                await session.rollback()
                yield _sse("error", {
                    "code": "internal_error",
                    "message": f"{type(exc).__name__}: {exc}",
                })

    return EventSourceResponse(stream())


def _as_jsonb(raw: Any) -> dict | None:
    """工具入参存 JSONB。不是 dict 的包一层，保持列类型统一。"""
    if isinstance(raw, dict):
        return raw
    if raw in (None, ""):
        return None
    return {"raw": str(raw)}


def _preview(call: dict[str, Any]) -> str | None:
    """工具结果摘要。没有结果就存 NULL，报错的存错误本身。

    直接 ``str(call.get("output", ""))`` 会在 output 存在但为 None 时写下字面量
    ``"None"`` —— 排查时看着像工具真的返回了这四个字符。报错也要留痕：工具失败
    本身就是要归档的事实，丢掉它就只剩一条不知为何没有输出的记录。
    """
    if call.get("error"):
        return f"ERROR: {call['error']}"[:_TOOL_PREVIEW_CHARS]
    output = call.get("output")
    if output is None:
        return None
    return str(output)[:_TOOL_PREVIEW_CHARS] or None
