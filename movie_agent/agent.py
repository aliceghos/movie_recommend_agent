"""
基于 LlamaIndex ReActAgent + MiniMax LLM 的电影推荐 Agent。
对外暴露 chat() 函数供 app.py 调用。
"""

import json
import os
import re
from typing import Any

from pydantic import BaseModel, ValidationError

from llama_index.core.agent.react.output_parser import ReActOutputParser
from llama_index.core.agent.react.types import BaseReasoningStep
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.openai_like import OpenAILike

from movie_agent.tools import TOOLS


class MiniMaxToolCall(BaseModel):
    name: str
    parameters: dict[str, Any]


def _extract_first_tool_call(output: str) -> tuple[str, str] | None:
    """
    从 MiniMax 输出中提取第一个工具调用，返回 (tool_name, raw_params)。
    支持三种已知变体，多个工具调用只取第一个。

    变体一（XML invoke）：
        minimax:tool_call <invoke name="search_movies">
        <tool_caller_parameters>{"query": "..."}</tool_caller_parameters></invoke>

    变体二（XML action）：
        minimax:tool_call <action>search_movies</action>
        <action_input>{"query": "..."}</action_input></action>

    变体三（JSON 风格，格式畸形）：
        minimax:tool_call {"search_movies", "query": "..."}
        或
        minimax:tool_call {"name": "search_movies", "query": "..."}
    """
    # 变体一
    m = re.search(
        r'minimax:tool_call\s*<invoke\s+name="([^"]+)">\s*'
        r"<tool_caller_parameters>(.*?)</tool_caller_parameters>\s*</invoke>",
        output,
        re.DOTALL,
    )
    if m:
        return m.group(1), m.group(2).strip()

    # 变体二
    m = re.search(
        r"minimax:tool_call\s*<action>(.*?)</action>\s*<action_input>(.*?)</action>",
        output,
        re.DOTALL,
    )
    if m:
        return m.group(1).strip(), m.group(2).strip()

    # 变体三：minimax:tool_call {"tool_name", "key": "value", ...}
    # 第一个字符串字面量视为工具名，其余键值对视为参数
    m = re.search(r'minimax:tool_call\s*\{(.*?)\}', output, re.DOTALL)
    if m:
        inner = m.group(1).strip()
        # 提取工具名（第一个被引号包裹的字符串）
        name_match = re.match(r'"([^"]+)"', inner)
        if name_match:
            tool_name = name_match.group(1)
            # 把剩余部分拼成合法 JSON 对象来提取参数
            rest = inner[name_match.end():].lstrip(", ")
            raw_params = "{" + rest + "}" if rest else "{}"
            return tool_name, raw_params

    return None


class MiniMaxOutputParser(ReActOutputParser):
    """处理 MiniMax 非标准工具调用格式，转换为标准 ReAct 文本格式后交给父类解析。"""

    def parse(self, output: str, is_streaming: bool = False) -> BaseReasoningStep:
        # 提取 <think>...</think> 中的思考内容
        think_match = re.search(r"<think>(.*?)</think>", output, re.DOTALL)
        thought = think_match.group(1).strip() if think_match else ""

        result = _extract_first_tool_call(output)
        if result:
            tool_name, raw_params = result
            try:
                params = json.loads(raw_params)
                params = {k: v for k, v in params.items() if v is not None}
                tool_call = MiniMaxToolCall(name=tool_name, parameters=params)
            except (json.JSONDecodeError, ValidationError):
                tool_call = MiniMaxToolCall(name=tool_name, parameters={})

            # 重建为标准 ReAct 格式后交给父类处理
            reconstructed = (
                f"Thought: {thought or 'Using tool.'}\n"
                f"Action: {tool_call.name}\n"
                f"Action Input: {json.dumps(tool_call.parameters, ensure_ascii=False)}"
            )
            return super().parse(reconstructed, is_streaming=is_streaming)

        # 无工具调用，直接交给父类处理（Answer: 或纯文本）
        return super().parse(output, is_streaming=is_streaming)


SYSTEM_PROMPT = """\
You are a movie recommendation assistant. Use the TMDB tools to answer user requests.

STRICT OUTPUT FORMAT — follow this exactly every single step:

Thought: <your reasoning>
Action: <tool_name>
Action Input: <json object with parameters>

After receiving tool results (Observation), continue with the same format.
When ready to give the final answer:

Thought: I now have enough information.
Answer: <your final response>

RULES:
- Output exactly ONE Action per response, never multiple.
- Do NOT use <think>, </think>, or any XML tags.
- Do NOT use minimax:tool_call syntax.
- Tool names: search_movies, get_movie_details, get_recommendations, discover_movies, get_popular_movies, get_genres.
- Always include TMDB URL for each movie: https://www.themoviedb.org/movie/{movie_id}
"""


def create_agent() -> tuple[ReActAgent, ChatMemoryBuffer]:
    """初始化 ReActAgent，使用 MiniMax LLM 和 TMDB 工具集，返回 (agent, memory) 元组。"""
    api_key = os.getenv("MINIMAX_API_KEY", "")
    group_id = os.getenv("MINIMAX_GROUP_ID", "")
    if not api_key or not group_id:
        raise ValueError("MINIMAX_API_KEY and MINIMAX_GROUP_ID must be set.")

    llm = OpenAILike(
        model="MiniMax-M2.7",
        api_base="https://api.minimaxi.com/v1",
        api_key=api_key,
        is_chat_model=True,
        is_function_calling_model=False,
        default_headers={"GroupId": group_id},
    )

    memory = ChatMemoryBuffer.from_defaults(token_limit=4096)

    agent = ReActAgent(
        tools=TOOLS,
        llm=llm,
        system_prompt=SYSTEM_PROMPT,
        output_parser=MiniMaxOutputParser(),
        max_iterations=8,
        verbose=True,
    )
    return agent, memory


async def chat(agent: ReActAgent, memory: ChatMemoryBuffer, user_message: str) -> str:
    """向 Agent 发送消息并返回文本回复，memory 跨轮次保持对话历史。"""
    handler = agent.run(user_msg=user_message, memory=memory)
    result = await handler
    return str(result.response.content)
