"""
长期记忆的领域层 —— 对 LangGraph ``BaseStore`` 接口编程，不关心后端实现。

两层记忆的分工：

- **短期**：``AsyncPostgresSaver`` checkpointer，按 ``thread_id`` 存本次会话的
  消息，由 LangGraph 自己管（见 ``server/main.py`` 的 lifespan）。
- **长期**：本模块的两个命名空间，跨会话存活：

      ("users", <uid>, "profile")   key="main"  结构化画像
      ("users", <uid>, "episodes")  key=<uuid>  情节记忆，text 字段被向量索引

后端是 ``AsyncPostgresStore``，由 server 装配后注入。之前这里有个手写的
``PersistentStore``（InMemoryStore + JSON 落盘），是"还没有数据库"时期的权宜
之计，已随本次改造删除：全量 dump 在多 worker 下会互相覆盖，而官方实现把落盘、
向量索引、命名空间三件事都覆盖了。历史数据用
``scripts/migrate_store_to_pg.py`` 一次性搬迁。

**uid 必填**是这里最重要的约定，见 ``_require_uid``。
"""

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langgraph.store.base import IndexConfig

# 改造前的 JSON 落盘位置，只有迁移脚本还会读它
LEGACY_STORE_PATH = Path(__file__).resolve().parent.parent / "data" / "memory" / "store.json"

EMBEDDING_DIMS = 384

PROFILE_KEY = "main"

EMPTY_PROFILE: dict[str, Any] = {
    "liked_genres": [],
    "disliked_genres": [],
    "liked_tones": [],
    "disliked_tones": [],
    "liked_movies": [],
    "disliked_movies": [],
    "last_updated": "",
}

PREFERENCE_KEYS = tuple(k for k in EMPTY_PROFILE if k != "last_updated")


def _require_uid(uid: str | None) -> str:
    """uid 没有默认值，传空就炸。

    改造前这里会回落到环境变量或字符串 ``"default"``。那个后门在单用户 demo 里
    无害，多用户下是最危险的一类 bug：忘传 uid 的代码路径不会报错，而是静默把
    A 的画像写进公共命名空间，再当成 B 的记忆读出来 —— 越权发生了但没有任何
    痕迹。宁可让调用方立刻崩，也不能让它悄悄串号。
    """
    if not uid or not str(uid).strip():
        raise ValueError("uid is required for memory namespaces; refusing to fall back to a shared default")
    return str(uid).strip()


def profile_ns(uid: str) -> tuple[str, ...]:
    return ("users", _require_uid(uid), "profile")


def episodes_ns(uid: str) -> tuple[str, ...]:
    return ("users", _require_uid(uid), "episodes")


def build_index_config() -> IndexConfig | None:
    """情节记忆的语义检索配置。嵌入模型不可用时降级为纯 key-value。

    与 RAG 复用同一个 384 维嵌入模型，省一份模型驻留内存。
    """
    try:
        from movie_agent.rag import _get_embeddings

        return IndexConfig(dims=EMBEDDING_DIMS, embed=_get_embeddings(), fields=["text"])
    except Exception as exc:  # noqa: BLE001 - 没有嵌入模型也要能存画像
        print(f"[store] semantic memory disabled: {type(exc).__name__}: {exc}")
        return None


# ---------------------------------------------------------------------------
# 画像读写
# ---------------------------------------------------------------------------

async def aload_profile(store: Any, uid: str) -> dict[str, Any]:
    """读画像，缺字段用默认值补齐。"""
    item = await store.aget(profile_ns(uid), PROFILE_KEY)
    if item is None:
        return dict(EMPTY_PROFILE)
    return {**EMPTY_PROFILE, **item.value}


async def asave_profile(store: Any, profile: dict[str, Any], uid: str) -> None:
    profile["last_updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    await store.aput(profile_ns(uid), PROFILE_KEY, profile)


async def aclear_profile(store: Any, uid: str) -> None:
    """清空长期画像，供 ``DELETE /v1/users/me/profile`` 用。"""
    await store.adelete(profile_ns(uid), PROFILE_KEY)


def has_preferences(profile: dict[str, Any]) -> bool:
    return any(profile.get(k) for k in PREFERENCE_KEYS)


# ---------------------------------------------------------------------------
# 情节记忆
# ---------------------------------------------------------------------------

async def add_episode(store: Any, text: str, uid: str) -> None:
    """追加一条情节记忆。``text`` 字段会被向量索引，可语义召回。"""
    await store.aput(
        episodes_ns(uid),
        uuid.uuid4().hex,
        {"text": text, "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")},
    )


async def recall_episodes(store: Any, query: str, uid: str, limit: int = 2) -> list[str]:
    """按语义相关度召回历史情节。检索不可用时回落到最近几条。"""
    ns = episodes_ns(uid)
    try:
        items = await store.asearch(ns, query=query, limit=limit)
    except Exception:  # noqa: BLE001 - 未配索引或嵌入失败时按时间兜底
        items = await store.asearch(ns, limit=limit)
    ordered = sorted(items, key=lambda i: i.value.get("ts", ""), reverse=True)
    return [i.value.get("text", "") for i in ordered if i.value.get("text")]
