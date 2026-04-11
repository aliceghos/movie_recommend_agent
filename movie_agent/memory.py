"""
长期记忆模块：负责用户偏好的持久化、对话历史摘要压缩、以及 LLM 驱动的偏好提取。
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

_MEMORY_DIR = Path(__file__).parent.parent / "data" / "memory"
_PROFILE_PATH = _MEMORY_DIR / "user_profile.json"

COMPRESSION_THRESHOLD = 20  # 10 轮对话 × 2 条消息
KEEP_RECENT = 8             # 保留最近 4 轮原始消息

_EMPTY_PROFILE: dict[str, Any] = {
    "liked_genres": [],
    "disliked_genres": [],
    "liked_tones": [],
    "disliked_tones": [],
    "liked_movies": [],
    "disliked_movies": [],
    "conversation_summary": "",
    "last_updated": "",
}


def load_profile() -> dict[str, Any]:
    """从磁盘加载用户画像。文件不存在或损坏时返回空默认值。"""
    if not _PROFILE_PATH.exists():
        return dict(_EMPTY_PROFILE)
    try:
        with _PROFILE_PATH.open(encoding="utf-8") as f:
            data = json.load(f)
        return {**_EMPTY_PROFILE, **data}
    except (json.JSONDecodeError, OSError):
        return dict(_EMPTY_PROFILE)


def save_profile(profile: dict[str, Any]) -> None:
    """将用户画像原子性地写入磁盘（先写临时文件再重命名）。"""
    _MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    profile["last_updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    tmp_path = _PROFILE_PATH.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
    tmp_path.replace(_PROFILE_PATH)


async def maybe_compress_history(
    history: list[BaseMessage],
    llm: Any,
    profile: dict[str, Any],
) -> list[BaseMessage]:
    """
    当历史消息数量超过阈值时，将旧消息压缩成一条 SystemMessage 摘要。
    返回（可能已压缩的）消息列表。
    """
    if len(history) <= COMPRESSION_THRESHOLD:
        return history

    old_messages = history[:-KEEP_RECENT]
    recent_messages = history[-KEEP_RECENT:]

    transcript_lines = []
    for m in old_messages:
        if isinstance(m, SystemMessage):
            transcript_lines.append(f"[Previous Summary]: {m.content}")
        elif isinstance(m, HumanMessage):
            transcript_lines.append(f"User: {m.content}")
        elif isinstance(m, AIMessage):
            transcript_lines.append(f"Assistant: {m.content}")
    transcript = "\n".join(transcript_lines)

    existing_summary = profile.get("conversation_summary", "")
    summary_prompt = (
        "You are summarizing a movie recommendation conversation to compress it.\n\n"
        + (f"Previous summary: {existing_summary}\n\n" if existing_summary else "")
        + f"New conversation to add to the summary:\n{transcript}\n\n"
        "Write a concise summary (3-5 sentences) capturing: the user's expressed genre/tone "
        "preferences, specific movies they liked or disliked, and any notable requests. "
        "Be factual and terse. Output only the summary text."
    )

    result = await llm.ainvoke([HumanMessage(content=summary_prompt)])
    new_summary = result.content.strip()

    profile["conversation_summary"] = new_summary
    save_profile(profile)

    summary_msg = SystemMessage(content=f"[Conversation summary so far]\n{new_summary}")
    return [summary_msg] + list(recent_messages)


async def extract_and_save_preferences(
    user_message: str,
    ai_response: str,
    llm: Any,
    profile: dict[str, Any],
) -> None:
    """
    使用 LLM 从单轮对话中提取偏好信号，去重后追加到画像并保存。
    所有异常静默处理，保证不影响主对话流程。
    """
    extraction_prompt = (
        "Analyze this movie conversation exchange and extract any preference signals.\n\n"
        f"User: {user_message}\n"
        f"Assistant: {ai_response}\n\n"
        "Current profile:\n"
        f"- Liked genres: {profile['liked_genres']}\n"
        f"- Disliked genres: {profile['disliked_genres']}\n"
        f"- Liked tones: {profile['liked_tones']}\n"
        f"- Disliked tones: {profile['disliked_tones']}\n"
        f"- Liked movies: {profile['liked_movies']}\n"
        f"- Disliked movies: {profile['disliked_movies']}\n\n"
        "Return ONLY a JSON object with these exact keys. Only include items that are NEW "
        "(not already in the profile). If nothing new, return all empty lists.\n\n"
        '{"liked_genres": [], "disliked_genres": [], "liked_tones": [], '
        '"disliked_tones": [], "liked_movies": [], "disliked_movies": []}'
    )

    try:
        result = await llm.ainvoke([HumanMessage(content=extraction_prompt)])
        raw = result.content.strip()
        # 去掉 LLM 可能包裹的 markdown 代码块
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]
        extracted: dict = json.loads(raw.strip())

        changed = False
        for key in (
            "liked_genres", "disliked_genres",
            "liked_tones", "disliked_tones",
            "liked_movies", "disliked_movies",
        ):
            new_items = extracted.get(key, [])
            if isinstance(new_items, list):
                for item in new_items:
                    item_clean = str(item).strip()
                    if item_clean and item_clean not in profile[key]:
                        profile[key].append(item_clean)
                        changed = True

        if changed:
            save_profile(profile)

    except Exception:
        pass
