"""
电影推荐 Agent 的 Streamlit 聊天界面。

启动方式：
    streamlit run app.py
"""

import asyncio
import os

import streamlit as st
from dotenv import load_dotenv

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
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.session_state.agent = None
        st.session_state.memory = None
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
            "- **API_KEY** — get yours at [console.anthropic.com](https://console.anthropic.com)"
        )
    if not tmdb_key:
        st.markdown(
            "- **TMDB_API_KEY** — get yours at [themoviedb.org/settings/api](https://www.themoviedb.org/settings/api)"
        )
    st.markdown("Add both keys to your `.env` file, then restart the app.")
    st.stop()

# --- 初始化 Agent（每个 session 只创建一次）---
if "agent" not in st.session_state or st.session_state.agent is None:
    try:
        from movie_agent.agent import create_agent
        st.session_state.agent, st.session_state.memory = create_agent()
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
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 获取 Agent 回复
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                from movie_agent.agent import chat
                response = asyncio.run(chat(st.session_state.agent, st.session_state.memory, user_input))
            except Exception as e:
                response = f"Sorry, something went wrong: {e}"
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
