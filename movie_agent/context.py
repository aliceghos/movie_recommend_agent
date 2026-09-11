"""每请求的 Agent 上下文。

用框架原生的 ``context_schema`` 而不是往 ``config["configurable"]`` 里塞：
``configurable`` 会被 checkpointer 连同 ``thread_id`` 一起序列化进 checkpoint，
把 user_id / role 写进去等于把授权信息写进了可恢复的历史 —— 改了角色之后旧
checkpoint 里还是旧角色。``context`` 是每次调用现传的运行时参数，不落盘。

单独一个模块是为了断开循环 import：``agent`` 与 ``middleware`` 都要用它，而
``agent`` 又要 import ``middleware``。
"""

from dataclasses import dataclass

ROLE_USER = "user"
ROLE_READONLY = "readonly"


@dataclass
class MovieAgentContext:
    """每次 agent 调用现传的身份。

    ``user_id`` 只从校验过的 JWT 来，绝不从请求体或 query 取 —— 否则客户端
    改一个字段就能读别人的记忆。
    """

    user_id: str
    role: str = ROLE_USER

    def __post_init__(self) -> None:
        if not self.user_id or not str(self.user_id).strip():
            raise ValueError("MovieAgentContext.user_id is required")
        self.user_id = str(self.user_id).strip()
