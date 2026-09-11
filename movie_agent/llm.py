"""
LLM 工厂。

项目里有两类 LLM 用途，参数取向相反，因此分开构造：

- **对话模型**：面向用户，需要流式、带温度。每个 Agent 实例一个。
- **工具模型**：面向内部流程（RAG 精排、摘要压缩、工具筛选、偏好抽取），
  要求确定性输出、不流式、不思考。全进程复用一个即可。

两者共用同一套 ``OPENAI_*`` 环境变量，所以换任何 OpenAI 兼容网关都是改配置。
放在独立模块里是为了让 rag / middleware / agent 都能引用而不构成循环导入。
"""

import os
from typing import Any

from langchain_openai import ChatOpenAI

# 默认指向 DeepSeek，但配置项是通用的 —— 换成任何 OpenAI 兼容的网关/厂商
# 只需改 .env 里的 OPENAI_BASE_URL 与 OPENAI_MODEL。
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-flash"

# 内部流程不需要思维链：既白花延迟，也让部分思考模型拒绝强制 tool_choice
# （DeepSeek 会报 400 ``Thinking mode does not support this tool_choice``，而
# ``with_structured_output(method="function_calling")`` 正是强制指定工具的）。
# ``reasoning_effort`` 是 OpenAI 标准参数；若某个网关不认它，把下面的环境变量
# 设为空字符串即不下发。
UTILITY_REASONING_EFFORT_ENV = "OPENAI_UTILITY_REASONING_EFFORT"
DEFAULT_UTILITY_REASONING_EFFORT = "none"

_utility_llm: ChatOpenAI | None = None


class _CompatChatOpenAI(ChatOpenAI):
    """把结构化输出的默认实现从 json_schema 换成 function_calling。

    ``ChatOpenAI.with_structured_output`` 默认 ``method="json_schema"``，会发出
    ``response_format={"type": "json_schema", ...}``。这是 OpenAI 的私有扩展，
    DeepSeek 直接返回 400 ``This response_format type is unavailable now``　——
    ``LLMToolSelectorMiddleware`` 内部就走这条路，于是每一轮对话都在第一次模型
    调用时崩掉（已在浏览器里实测复现）。

    改用 function_calling 后结构化输出借道 tool calling 完成，而 tool calling 是
    OpenAI 兼容网关的通用能力（本 Agent 的全部工具调用都依赖它），所以这条路径
    对任何兼容端点都成立。显式传 ``method`` 的调用方不受影响。
    """

    def with_structured_output(  # type: ignore[override]
        self,
        schema: Any = None,
        *,
        method: str = "function_calling",
        **kwargs: Any,
    ) -> Any:
        return super().with_structured_output(schema, method=method, **kwargs)


def _require_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY must be set.")
    return api_key


def model_name() -> str:
    return os.getenv("OPENAI_MODEL") or DEFAULT_MODEL


def base_url() -> str:
    return os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL


def get_chat_llm() -> ChatOpenAI:
    """面向用户的对话模型。"""
    return ChatOpenAI(
        model=model_name(),
        base_url=base_url(),
        api_key=_require_api_key(),
        streaming=True,  # required for on_llm_new_token / on_chat_model_stream
        temperature=0.7,
        # DeepSeek 等兼容网关的流式响应默认不带 usage，显式开启以便统计 token
        stream_usage=True,
    )


def get_utility_llm() -> ChatOpenAI:
    """内部流程用的确定性模型（进程级单例）。"""
    global _utility_llm
    if _utility_llm is None:
        effort = os.getenv(
            UTILITY_REASONING_EFFORT_ENV, DEFAULT_UTILITY_REASONING_EFFORT
        ).strip()
        _utility_llm = _CompatChatOpenAI(
            model=model_name(),
            base_url=base_url(),
            api_key=_require_api_key(),
            streaming=False,
            temperature=0,
            timeout=30,
            max_retries=1,
            reasoning_effort=effort or None,
        )
    return _utility_llm
