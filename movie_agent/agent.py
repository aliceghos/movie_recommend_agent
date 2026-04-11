"""
基于 LangChain ReAct Agent + MiniMax LLM 的电影推荐 Agent。
对外暴露 chat() 函数供 app.py 调用。
"""

import asyncio
import os
from typing import Any

import httpx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from movie_agent.memory import (
    extract_and_save_preferences,
    load_profile,
    maybe_compress_history,
)
from movie_agent.tools import TOOLS

SYSTEM_PROMPT = """\
You are a movie recommendation assistant. You have access to two types of tools:

1. TMDB tools — for real-time movie data:
   search_movies, get_movie_details, get_recommendations, discover_movies,
   get_popular_movies, get_genres

2. Local knowledge base tool — for editorial content:
   search_local_knowledge — searches curated movie reviews, genre articles, and film art books.
   Use this when the user asks for critical opinions, review excerpts, background
   on film genres (cyberpunk, film noir), or film art theory and filmmaking concepts.
   Covered movies: Inception, Harry Potter and the Prisoner of Azkaban, Jane Eyre,
   Spirited Away, Wuthering Heights.
   Also contains film art book content for questions about cinematography,
   directing techniques, or film theory.

When you need to call a tool, reason step by step. After receiving tool results,
continue reasoning until you have enough information to give a final answer.

Always include the TMDB URL for each movie: https://www.themoviedb.org/movie/{movie_id}
"""


def _build_system_prompt(profile: dict[str, Any]) -> SystemMessage:
    """组合基础 prompt 与用户画像，生成每轮对话的 SystemMessage。"""
    parts = [SYSTEM_PROMPT]

    has_prefs = any(
        profile.get(k)
        for k in (
            "liked_genres", "disliked_genres",
            "liked_tones", "disliked_tones",
            "liked_movies", "disliked_movies",
        )
    )

    if has_prefs or profile.get("conversation_summary"):
        parts.append("\n--- User Profile (from long-term memory) ---")
        if profile.get("liked_genres"):
            parts.append(f"Liked genres: {', '.join(profile['liked_genres'])}")
        if profile.get("disliked_genres"):
            parts.append(f"Disliked genres: {', '.join(profile['disliked_genres'])}")
        if profile.get("liked_tones"):
            parts.append(f"Liked tones: {', '.join(profile['liked_tones'])}")
        if profile.get("disliked_tones"):
            parts.append(f"Disliked tones: {', '.join(profile['disliked_tones'])}")
        if profile.get("liked_movies"):
            parts.append(f"Movies the user has enjoyed: {', '.join(profile['liked_movies'])}")
        if profile.get("disliked_movies"):
            parts.append(f"Movies the user disliked: {', '.join(profile['disliked_movies'])}")
        if profile.get("conversation_summary"):
            parts.append(f"\nPrevious conversation summary:\n{profile['conversation_summary']}")
        parts.append("\nUse this profile to personalize your recommendations.")

    return SystemMessage(content="\n".join(parts))


def create_agent() -> dict[str, Any]:
    """初始化 LangChain ReAct Agent，使用 MiniMax LLM 和 TMDB 工具集。

    Returns:
        包含 'agent'、'history'、'llm'、'profile' 的字典，供 chat() 跨轮次使用。
    """
    api_key = os.getenv("MINIMAX_API_KEY", "")
    group_id = os.getenv("MINIMAX_GROUP_ID", "")
    if not api_key or not group_id:
        raise ValueError("MINIMAX_API_KEY and MINIMAX_GROUP_ID must be set.")

    llm = ChatOpenAI(
        model="MiniMax-M2.7",
        base_url="https://api.minimaxi.com/v1",
        api_key=api_key,
        http_client=httpx.Client(headers={"GroupId": group_id}),
        http_async_client=httpx.AsyncClient(headers={"GroupId": group_id}),
    )

    agent = create_react_agent(
        model=llm,
        tools=TOOLS,
    )

    return {"agent": agent, "history": [], "llm": llm, "profile": load_profile()}


async def chat(agent_state: dict[str, Any], user_message: str) -> str:
    """向 Agent 发送消息并返回文本回复，自动处理历史压缩和偏好提取。"""
    profile = agent_state["profile"]
    llm = agent_state["llm"]

    agent_state["history"].append(HumanMessage(content=user_message))

    # 历史超过阈值时压缩旧消息
    agent_state["history"] = await maybe_compress_history(
        agent_state["history"], llm, profile
    )

    # 每轮注入最新用户画像
    messages_to_send = [_build_system_prompt(profile)] + agent_state["history"]

    result = await agent_state["agent"].ainvoke({"messages": messages_to_send})

    messages: list[BaseMessage] = result["messages"]
    response_msg = next(
        (m for m in reversed(messages) if isinstance(m, AIMessage)),
        None,
    )
    response_text = response_msg.content if response_msg else "Sorry, I could not generate a response."

    agent_state["history"].append(AIMessage(content=response_text))

    # 异步提取偏好，不阻塞回复返回
    asyncio.create_task(
        extract_and_save_preferences(user_message, response_text, llm, profile)
    )

    return response_text
