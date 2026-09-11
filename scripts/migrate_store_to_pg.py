"""把改造前的 JSON 长期记忆搬进 Postgres。一次性脚本，跑完即可删。

改造前 ``PersistentStore`` 把所有记忆 dump 成一个 ``data/memory/store.json``，
命名空间里的 uid 恒为 ``"default"``（单用户 demo 的产物）。现在 uid 是真实用户
的 UUID，所以搬迁的实质是**换命名空间**：

    ("users", "default", "profile")   ->  ("users", <uuid>, "profile")
    ("users", "default", "episodes")  ->  ("users", <uuid>, "episodes")

用法::

    # 先注册目标账号（脚本刻意不建用户，见下）
    curl -X POST localhost:8000/v1/auth/register -d '{"email":"...","password":"..."}'

    python -m scripts.migrate_store_to_pg --email you@example.com --dry-run
    python -m scripts.migrate_store_to_pg --email you@example.com

**脚本不创建用户**：建账号要过密码策略和唯一约束，那是 ``/v1/auth/register``
的职责。让一个迁移脚本也能凭空造账号，等于多开一条绕过认证的旁路。

情节记忆保留原 key 与原 ``ts``，不走 ``add_episode()`` —— 后者会生成新 uuid 和
当前时间戳，把"去年看的片"记成今天的事，重复跑还会翻倍。原 key 幂等：同一条搬
两次是覆盖，不是新增。
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.store.postgres.aio import AsyncPostgresStore  # noqa: E402
from sqlalchemy import select  # noqa: E402

from movie_agent.store import (  # noqa: E402
    EMPTY_PROFILE,
    LEGACY_STORE_PATH,
    PROFILE_KEY,
    build_index_config,
    episodes_ns,
    profile_ns,
)
from server.config import get_settings  # noqa: E402
from server.db.models import User  # noqa: E402
from server.db.session import dispose_engine, get_sessionmaker, init_engine  # noqa: E402


def read_legacy(path: Path, legacy_uid: str) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """从 JSON 里挑出指定 legacy uid 的画像与情节，其余命名空间忽略。"""
    if not path.exists():
        raise SystemExit(f"legacy store not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    profile: dict[str, Any] | None = None
    episodes: list[dict[str, Any]] = []

    for item in payload.get("items", []):
        ns = tuple(item.get("namespace", ()))
        if len(ns) != 3 or ns[0] != "users" or ns[1] != legacy_uid:
            continue
        if ns[2] == "profile":
            profile = item.get("value") or {}
        elif ns[2] == "episodes":
            episodes.append({"key": item["key"], "value": item.get("value") or {}})

    episodes.sort(key=lambda e: e["value"].get("ts", ""))
    return profile, episodes


async def resolve_user_id(email: str) -> str:
    settings = get_settings()
    init_engine(settings)
    try:
        async with get_sessionmaker()() as session:
            row = await session.execute(select(User).where(User.email == email.lower().strip()))
            user = row.scalar_one_or_none()
            if user is None:
                raise SystemExit(f"no such user: {email} — register the account first")
            return str(user.id)
    finally:
        await dispose_engine()


async def migrate(email: str, source: Path, legacy_uid: str, dry_run: bool) -> None:
    profile, episodes = read_legacy(source, legacy_uid)
    if profile is None and not episodes:
        raise SystemExit(f"nothing to migrate under ('users', {legacy_uid!r}, ...)")

    uid = await resolve_user_id(email)
    print(f"target user: {email} -> {uid}")
    print(f"profile: {'yes' if profile else 'none'}   episodes: {len(episodes)}")

    if dry_run:
        if profile:
            kept = {k: v for k, v in profile.items() if k in EMPTY_PROFILE and v}
            print(f"  [dry-run] profile fields: {kept}")
        for ep in episodes:
            print(f"  [dry-run] {ep['value'].get('ts', '?')}  {ep['value'].get('text', '')[:80]}")
        print("dry run only, nothing written")
        return

    settings = get_settings()
    index = build_index_config()
    async with AsyncPostgresStore.from_conn_string(settings.database_url, index=index) as store:
        await store.setup()

        if profile:
            # 原样写入，包括 last_updated —— 画像的"上次更新"是它自己的事实,
            # 搬迁不该把它刷成今天，所以这里不走 asave_profile()。
            await store.aput(profile_ns(uid), PROFILE_KEY, {**EMPTY_PROFILE, **profile})
            print("profile migrated")

        for ep in episodes:
            await store.aput(episodes_ns(uid), ep["key"], ep["value"])
        print(f"{len(episodes)} episodes migrated")

    print(f"\ndone. verify via GET /v1/users/me/profile, then remove {source}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--email", required=True, help="目标用户邮箱，必须已注册")
    parser.add_argument("--source", type=Path, default=LEGACY_STORE_PATH, help="旧 store.json 路径")
    parser.add_argument("--legacy-user", default="default", help="旧命名空间里的 uid")
    parser.add_argument("--dry-run", action="store_true", help="只打印将要搬的内容")
    args = parser.parse_args()

    asyncio.run(migrate(args.email, args.source, args.legacy_user, args.dry_run))


if __name__ == "__main__":
    main()
