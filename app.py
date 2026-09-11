"""
电影推荐 Agent 的 Streamlit 聊天界面。

启动方式：
    ./scripts/dev.sh            # 一并起 MCP Server（推荐）
    streamlit run app.py        # 只起界面，TMDB 走本地降级工具

脚本执行顺序是刻意的：先建 Agent，再渲染对话，**最后**渲染侧边栏 ——
侧边栏要显示的 token 累计与用户画像都是本轮对话产生的，写在前面会永远慢一轮。
"""

import asyncio
import os
import traceback

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Movie Recommendation Agent",
    page_icon="🎬",
    layout="centered",
)

# --- 累计 token 统计 ---
for _key in ("total_tokens", "total_prompt_tokens", "total_completion_tokens", "llm_call_count"):
    st.session_state.setdefault(_key, 0)

# --- 检查必要的 API Key ---
api_key = os.getenv("OPENAI_API_KEY", "").strip()
tmdb_key = os.getenv("TMDB_API_KEY", "").strip()

if not api_key or not tmdb_key:
    st.title("🎬 Movie Recommendation Agent")
    st.error("**Setup required** — one or more API keys are missing.")
    if not api_key:
        st.markdown(
            "- **OPENAI_API_KEY** — any OpenAI-compatible key; the default endpoint is "
            "[platform.deepseek.com](https://platform.deepseek.com/)"
        )
    if not tmdb_key:
        st.markdown(
            "- **TMDB_API_KEY** — get yours at "
            "[themoviedb.org/settings/api](https://www.themoviedb.org/settings/api)"
        )
    st.markdown("Add both keys to your `.env` file, then restart the app.")
    st.stop()

# --- 初始化 Agent（每个 session 只创建一次）---
# create_movie_agent 是 async 的：它要 await MCP Server 的工具清单。
if st.session_state.get("agent_state") is None:
    try:
        from movie_agent.agent import create_movie_agent

        with st.spinner("Connecting to MCP servers, loading index and skills..."):
            st.session_state.agent_state = asyncio.run(create_movie_agent())
    except Exception as exc:
        st.title("🎬 Movie Recommendation Agent")
        st.error(f"Failed to initialize agent: {exc}")
        st.code(traceback.format_exc())
        st.stop()

agent_state = st.session_state.agent_state
st.session_state.setdefault("messages", [])

# --- 页面标题 ---
st.title("🎬 Movie Recommendation Agent")

# --- 显示历史消息 ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 聊天输入框 ---
user_input = st.chat_input("Ask me about movies...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()

        tool_expander = st.expander("🔍 Show agent reasoning", expanded=False)
        tool_placeholder = tool_expander.empty()

        token_placeholder = st.empty()

        async def _consume_stream() -> str:
            """Consume ``chat_stream`` events and update UI placeholders."""
            from movie_agent.agent import chat_stream

            displayed_text = ""
            tool_lines: list[str] = []
            final_response = ""

            try:
                async for event in chat_stream(agent_state, user_input):
                    etype = event["type"]

                    if etype == "token":
                        displayed_text += event["content"]
                        response_placeholder.markdown(displayed_text + "▌")

                    elif etype == "tool_start":
                        name = event["name"]
                        inp = str(event.get("input", ""))[:200]
                        tool_lines.append(f"🔧 **{name}**\n> {inp}")
                        tool_placeholder.markdown("\n\n".join(tool_lines))

                    elif etype == "tool_end":
                        if tool_lines:
                            tool_lines[-1] = tool_lines[-1].replace("🔧", "✅")
                            tool_placeholder.markdown("\n\n".join(tool_lines))

                    elif etype == "done":
                        final_response = event.get("response", displayed_text)
                        # Final render without the blinking cursor
                        response_placeholder.markdown(final_response)

                        usage = event.get("token_usage", {})
                        if usage:
                            prompt_t = usage.get("prompt_tokens", 0)
                            comp_t = usage.get("completion_tokens", 0)
                            total_t = usage.get("total_tokens", 0)

                            st.session_state.total_prompt_tokens += prompt_t
                            st.session_state.total_completion_tokens += comp_t
                            st.session_state.total_tokens += total_t
                            st.session_state.llm_call_count += 1

                            token_placeholder.caption(
                                f"📊 This message: {prompt_t} prompt + {comp_t} "
                                f"completion = **{total_t} tokens**"
                            )

                        if not tool_lines:
                            tool_expander.empty()

            except Exception:
                tb = traceback.format_exc()
                print(f"[Agent Error]\n{tb}")
                error_msg = "Sorry, something went wrong."
                response_placeholder.markdown(error_msg)
                with tool_expander:
                    st.code(tb)
                final_response = error_msg

            return final_response

        response = asyncio.run(_consume_stream())

    st.session_state.messages.append({"role": "assistant", "content": response})

# ---------------------------------------------------------------------------
# 侧边栏 —— 放在最后渲染，读到的是本轮对话之后的状态
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🎬 Movie Agent")
    st.markdown(
        "Conversational movie recommendations, built on LangChain + LangGraph with "
        "MCP tools, hybrid RAG, agent skills, and two-tier memory."
    )
    st.divider()

    st.markdown("**Try asking:**")
    st.markdown("- Recommend me a sci-fi movie from the 90s")
    st.markdown("- I loved Inception, what should I watch next?")
    st.markdown("- 聊聊黑色电影的视觉风格")
    st.markdown("- Add Blade Runner to my watchlist")
    st.divider()

    # -- 能力装配状态 --
    st.markdown("**🔌 Capabilities**")
    warnings = agent_state.get("warnings", [])
    if warnings:
        st.warning("MCP degraded:\n" + "\n".join(f"- {w}" for w in warnings))
    else:
        st.caption("MCP: tmdb + watchlist servers connected")
    st.caption(f"Tools available: {len(agent_state.get('tool_names', []))}")
    skills = agent_state.get("skills", [])
    st.caption("Skills: " + (", ".join(skills) if skills else "none loaded"))
    st.divider()

    # -- Token 用量 --
    st.markdown("**📊 Token Usage**")
    if st.session_state.llm_call_count > 0:
        st.markdown(
            f"LLM calls: **{st.session_state.llm_call_count}**\n\n"
            f"Prompt tokens: **{st.session_state.total_prompt_tokens}**\n\n"
            f"Completion tokens: **{st.session_state.total_completion_tokens}**\n\n"
            f"Total tokens: **{st.session_state.total_tokens}**"
        )
    else:
        st.caption("No usage recorded yet.")
    st.divider()

    # -- 长期画像（来自 Store，跨会话存活）--
    st.markdown("**🧠 Long-term Profile**")
    try:
        from movie_agent.agent import get_profile
        from movie_agent.store import PREFERENCE_KEYS

        profile = get_profile(agent_state)
        labels = {
            "liked_genres": "Genres you like",
            "disliked_genres": "Genres you avoid",
            "liked_tones": "Tones you enjoy",
            "disliked_tones": "Tones you avoid",
            "liked_movies": "Enjoyed",
            "disliked_movies": "Disliked",
        }
        if not any(profile.get(k) for k in PREFERENCE_KEYS):
            st.caption("No preferences recorded yet.")
        else:
            for key, label in labels.items():
                if profile.get(key):
                    st.markdown(f"{label}: {', '.join(profile[key])}")
            if profile.get("last_updated"):
                st.caption(f"Updated: {profile['last_updated'][:10]}")
    except Exception as exc:
        st.caption(f"Profile unavailable: {exc}")

    st.divider()
    if st.button("🗑️ Clear conversation"):
        # 只清短期记忆：换一个 thread_id 就等于开新会话，
        # 长期画像留在 Store 里，这正是两层记忆的分工。
        st.session_state.messages = []
        st.session_state.agent_state = None
        for key in ("total_tokens", "total_prompt_tokens", "total_completion_tokens", "llm_call_count"):
            st.session_state[key] = 0
        st.rerun()
