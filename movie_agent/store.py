"""
长期记忆的存储层 —— LangGraph ``BaseStore`` 接口 + JSON 落盘。

两层记忆的分工：

- **短期**：``InMemorySaver`` checkpointer，按 ``thread_id`` 存本次会话的消息，
  由 LangGraph 自己管，进程退出即消失（见 ``agent.py``）。
- **长期**：本模块，跨会话存活。两个命名空间：

      ("users", <uid>, "profile")   key="main"  结构化画像
      ("users", <uid>, "episodes")  key=<uuid>  情节记忆，text 字段被向量索引

为什么自己实现持久化：langgraph 只捆绑 ``store/base`` 与 ``store/memory``，
没有任何落盘后端（``langgraph.store.sqlite`` 不存在，已实测）。这里继承
``InMemoryStore`` 只补一层写后落盘 —— ``batch``/``abatch`` 是它唯一的抽象方法，
覆写这两个就能拦住全部读写路径，语义检索等能力原样继承。

刻意不用连接型后端：Streamlit 每轮 rerun 都新建一个 ``asyncio.run`` 事件循环，
持有连接的 Store 会跨循环失效。纯内存 + 落盘没有这个问题。
"""

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from langgraph.store.base import IndexConfig, Item, Op, PutOp, Result
from langgraph.store.memory import InMemoryStore

_MEMORY_DIR = Path(__file__).resolve().parent.parent / "data" / "memory"
_STORE_PATH = _MEMORY_DIR / "store.json"
_LEGACY_PROFILE_PATH = _MEMORY_DIR / "user_profile.json"

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

_store: "PersistentStore | None" = None


def user_id() -> str:
    return os.getenv("MOVIE_AGENT_USER_ID", "").strip() or "default"


def profile_ns(uid: str | None = None) -> tuple[str, ...]:
    return ("users", uid or user_id(), "profile")


def episodes_ns(uid: str | None = None) -> tuple[str, ...]:
    return ("users", uid or user_id(), "episodes")


class PersistentStore(InMemoryStore):
    """``InMemoryStore`` + 写后落盘到 JSON。"""

    def __init__(self, path: Path = _STORE_PATH, *, index: IndexConfig | None = None) -> None:
        super().__init__(index=index)
        self._path = path
        self._load()

    # -- 落盘 -------------------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[store] ignoring unreadable {self._path.name}: {exc}")
            return

        for entry in raw.get("items", []):
            try:
                namespace = tuple(entry["namespace"])
                key = entry["key"]
                value = entry["value"]
            except (KeyError, TypeError) as exc:
                print(f"[store] skipping malformed entry: {exc}")
                continue

            # 走父类的 put 路径，顺便把向量算回来 —— 嵌入是可从 value
            # 重算的派生数据，写进 JSON 会让文件膨胀几十倍，所以不落盘。
            super().batch([PutOp(namespace=namespace, key=key, value=value)])

            # put 会把时间戳刷成当下，这里把原始时间戳贴回去
            stored = self._data[namespace].get(key)
            if stored is not None and entry.get("created_at") and entry.get("updated_at"):
                self._data[namespace][key] = Item(
                    value=stored.value,
                    key=key,
                    namespace=namespace,
                    created_at=entry["created_at"],
                    updated_at=entry["updated_at"],
                )

    def _dump(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "items": [
                item.dict()
                for ns in sorted(self._data, key=lambda n: "/".join(n))
                for item in self._data[ns].values()
            ]
        }
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self._path)  # 原子替换，避免写一半被读到

    @staticmethod
    def _is_write(ops: list[Op]) -> bool:
        return any(isinstance(op, PutOp) for op in ops)

    # -- BaseStore 接口 ---------------------------------------------------

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        ops = list(ops)
        results = super().batch(ops)
        if self._is_write(ops):
            self._dump()
        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        ops = list(ops)
        results = await super().abatch(ops)
        if self._is_write(ops):
            self._dump()
        return results


def _build_index_config() -> IndexConfig | None:
    """情节记忆的语义检索配置。嵌入模型不可用时降级为纯 key-value。"""
    try:
        from movie_agent.rag import _get_embeddings

        return IndexConfig(dims=384, embed=_get_embeddings(), fields=["text"])
    except Exception as exc:  # noqa: BLE001 - 没有嵌入模型也要能存画像
        print(f"[store] semantic memory disabled: {type(exc).__name__}: {exc}")
        return None


def get_store() -> PersistentStore:
    """进程级 Store 单例。"""
    global _store
    if _store is None:
        _store = PersistentStore(index=_build_index_config())
        migrate_legacy_profile(_store)
    return _store


# ---------------------------------------------------------------------------
# 画像读写
# ---------------------------------------------------------------------------

def load_profile(store: PersistentStore | None = None, uid: str | None = None) -> dict[str, Any]:
    """读画像，缺字段用默认值补齐。"""
    store = store or get_store()
    item = store.get(profile_ns(uid), PROFILE_KEY)
    if item is None:
        return dict(EMPTY_PROFILE)
    return {**EMPTY_PROFILE, **item.value}


async def aload_profile(store: Any, uid: str | None = None) -> dict[str, Any]:
    item = await store.aget(profile_ns(uid), PROFILE_KEY)
    if item is None:
        return dict(EMPTY_PROFILE)
    return {**EMPTY_PROFILE, **item.value}


async def asave_profile(store: Any, profile: dict[str, Any], uid: str | None = None) -> None:
    profile["last_updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    await store.aput(profile_ns(uid), PROFILE_KEY, profile)


def has_preferences(profile: dict[str, Any]) -> bool:
    return any(profile.get(k) for k in PREFERENCE_KEYS)


# ---------------------------------------------------------------------------
# 情节记忆
# ---------------------------------------------------------------------------

async def add_episode(store: Any, text: str, uid: str | None = None) -> None:
    """追加一条情节记忆。``text`` 字段会被向量索引，可语义召回。"""
    await store.aput(
        episodes_ns(uid),
        uuid.uuid4().hex,
        {"text": text, "ts": datetime.now(timezone.utc).isoformat(timespec="seconds")},
    )


async def recall_episodes(store: Any, query: str, limit: int = 2, uid: str | None = None) -> list[str]:
    """按语义相关度召回历史情节。检索不可用时回落到最近几条。"""
    ns = episodes_ns(uid)
    try:
        items = await store.asearch(ns, query=query, limit=limit)
    except Exception:  # noqa: BLE001 - 未配索引或嵌入失败时按时间兜底
        items = await store.asearch(ns, limit=limit)
    ordered = sorted(items, key=lambda i: i.value.get("ts", ""), reverse=True)
    return [i.value.get("text", "") for i in ordered if i.value.get("text")]


# ---------------------------------------------------------------------------
# 旧格式迁移
# ---------------------------------------------------------------------------

def migrate_legacy_profile(store: PersistentStore) -> None:
    """把改造前 ``data/memory/user_profile.json`` 的画像导入 Store。

    只在 Store 里还没有画像时执行一次，之后旧文件就只是遗留物。
    """
    if not _LEGACY_PROFILE_PATH.exists():
        return
    if store.get(profile_ns(), PROFILE_KEY) is not None:
        return

    try:
        legacy = json.loads(_LEGACY_PROFILE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[store] legacy profile not migrated: {exc}")
        return

    profile = {**EMPTY_PROFILE, **{k: v for k, v in legacy.items() if k in EMPTY_PROFILE}}
    if not has_preferences(profile):
        return

    store.put(profile_ns(), PROFILE_KEY, profile)
    print(f"[store] migrated legacy profile from {_LEGACY_PROFILE_PATH.name}")
