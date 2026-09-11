"""Alembic 环境。

连接串不写在 ``alembic.ini`` 里，而是从 ``server.config`` 读 —— 迁移和服务
必须永远指向同一个库，两处各写一份连接串迟早会漂移。

``target_metadata`` 只挂业务表的 ``Base``。LangGraph 的 checkpoint / store 表
由框架 ``setup()`` 建，不在 metadata 里 —— 但 autogenerate 会反射真实库，
默认会把它们当成"多余的表"生成 ``drop_table``。``include_object`` 就是为了
拦这一下：漏了这个过滤器，一次 autogenerate 就能把所有会话记忆删干净。
"""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from server.config import get_settings
from server.db.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

config.set_main_option("sqlalchemy.url", get_settings().sqlalchemy_url)

target_metadata = Base.metadata

# LangGraph 自己建的表，Alembic 一律不碰。名字取自 AsyncPostgresSaver.setup()
# 与 AsyncPostgresStore.setup() 实际建出来的表。
FRAMEWORK_TABLES = {
    "checkpoints",
    "checkpoint_blobs",
    "checkpoint_writes",
    "checkpoint_migrations",
    "store",
    "store_migrations",
    "store_vectors",
    "vector_migrations",
}


def include_object(object_, name, type_, reflected, compare_to):
    if type_ == "table" and name in FRAMEWORK_TABLES:
        return False
    return True


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        include_object=include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        include_object=include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
