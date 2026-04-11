"""
电影推荐 Agent 的 Streamlit 聊天界面（LangChain 版本）。

启动方式：
    streamlit run app.py
"""

import asyncio
import os
import traceback

import streamlit as st
from dotenv import load_dotenv

from movie_agent.memory import load_profile

load_dotenv()

st.set_page_config(
    page_title="Movie Recommendation Agent",
    page_icon="🎬",
    layout="centered",
)

# --- 侧边栏 ---
with st.sidebar:
    st.title("🎬 Movie Agent")
    st.markdown(
        "A conversational movie recommendation assistant powered by "
        "[Minimax](https://www.minimaxi.com/) and [TMDB](https://www.themoviedb.org/)."
    )
    st.divider()
    st.markdown("**Try asking:**")
    st.markdown("- Recommend me a sci-fi movie from the 90s")
    st.markdown("- I loved Inception, what should I watch next?")
    st.markdown("- Show me popular movies right now")
    st.markdown("- I'm in the mood for a funny romantic comedy")
    st.divider()
    st.markdown("**Your Profile**")
    _profile = load_profile()
    _pref_keys = ("liked_genres", "disliked_genres", "liked_tones", "disliked_tones", "liked_movies", "disliked_movies")
    if not any(_profile.get(k) for k in _pref_keys):
        st.caption("No preferences recorded yet.")
    else:
        if _profile.get("liked_genres"):
            st.markdown(f"Genres you like: {', '.join(_profile['liked_genres'])}")
        if _profile.get("disliked_genres"):
            st.markdown(f"Genres you avoid: {', '.join(_profile['disliked_genres'])}")
        if _profile.get("liked_tones"):
            st.markdown(f"Tones you enjoy: {', '.join(_profile['liked_tones'])}")
        if _profile.get("disliked_tones"):
            st.markdown(f"Tones you avoid: {', '.join(_profile['disliked_tones'])}")
        if _profile.get("liked_movies"):
            st.markdown(f"Enjoyed: {', '.join(_profile['liked_movies'])}")
        if _profile.get("disliked_movies"):
            st.markdown(f"Disliked: {', '.join(_profile['disliked_movies'])}")
        if _profile.get("last_updated"):
            st.caption(f"Updated: {_profile['last_updated'][:10]}")
    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.session_state.agent_state = None
        st.rerun()

# --- 检查必要的 API Key ---
api_key = os.getenv("MINIMAX_API_KEY", "").strip()
group_id = os.getenv("MINIMAX_GROUP_ID", "").strip()
tmdb_key = os.getenv("TMDB_API_KEY", "").strip()

if not api_key or not tmdb_key:
    st.title("🎬 Movie Recommendation Agent")
    st.error("**Setup required** — one or more API keys are missing.")
    if not api_key:
        st.markdown(
            "- **MINIMAX_API_KEY** — get yours at [minimaxi.com](https://www.minimaxi.com/)"
        )
    if not tmdb_key:
        st.markdown(
            "- **TMDB_API_KEY** — get yours at [themoviedb.org/settings/api](https://www.themoviedb.org/settings/api)"
        )
    st.markdown("Add both keys to your `.env` file, then restart the app.")
    st.stop()

# --- 初始化 Agent（每个 session 只创建一次）---
if "agent_state" not in st.session_state or st.session_state.agent_state is None:
    try:
        from movie_agent.agent import create_agent
        st.session_state.agent_state = create_agent()
    except Exception as e:
        st.error(f"Failed to initialize agent: {e}")
        st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

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
        with st.spinner("Thinking..."):
            try:
                from movie_agent.agent import chat
                response = asyncio.run(chat(st.session_state.agent_state, user_input))
            except Exception as e:
                tb = traceback.format_exc()
                print(f"[Agent Error]\n{tb}")
                response = f"Sorry, something went wrong: {e}"
                with st.expander("Error details (debug)"):
                    st.code(tb)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
