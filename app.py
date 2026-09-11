"""电影推荐 Agent 的 Streamlit 界面 —— 纯 HTTP 客户端。

启动方式：
    ./scripts/dev.sh                 # MCP Server + uvicorn + Streamlit 一起起
    streamlit run app.py             # 只起界面，后端需另外起

**这个文件不再 import ``movie_agent``**，也没有一个 ``asyncio.run()``。

改造前它自己装配 agent，于是被 Streamlit 的事件循环亲和性钉死：每次 rerun 都
新建再销毁一个 loop，数据库连接池活不下来，只能退回进程内状态（也就意味着单
用户、重启失忆）。agent 移进 uvicorn 之后这个约束自然消失，UI 只剩三件事：
拿 token、列会话、消费 SSE。

渲染顺序仍然是「对话在前、侧边栏在后」：侧边栏要显示的画像和用量都是本轮对话
产生的，写在前面会永远慢一轮。
"""

import uuid

import streamlit as st
from dotenv import load_dotenv

from ui.api_client import ApiClient, ApiError, api_base, readyz

load_dotenv()

st.set_page_config(page_title="Movie Recommendation Agent", page_icon="🎬", layout="centered")

for _key, _default in (
    ("access_token", None),
    ("refresh_token", None),
    ("user", None),
    ("conversation", None),
    ("messages", []),
    ("last_usage", None),
):
    st.session_state.setdefault(_key, _default)


def client() -> ApiClient:
    return ApiClient(st.session_state.access_token)


def sign_out() -> None:
    """登出。撤销 refresh token 后清空本地状态。

    撤销失败也照样清本地 —— 用户点了登出就必须登出，服务端连不上不是留在登录
    态的理由。
    """
    if st.session_state.refresh_token:
        try:
            client().logout(st.session_state.refresh_token)
        except (ApiError, OSError):
            pass
    for key in ("access_token", "refresh_token", "user", "conversation", "messages", "last_usage"):
        st.session_state[key] = None
    st.session_state.messages = []


# ---------------------------------------------------------------------------
# 后端可用性
# ---------------------------------------------------------------------------

ready, detail = readyz()
if not ready:
    st.title("🎬 Movie Recommendation Agent")
    st.error(f"**Backend not ready** — {detail}")
    st.markdown(
        f"界面只是客户端，agent 跑在 `{api_base()}`。用 `./scripts/dev.sh` 起全栈，"
        "或单独起 `uvicorn server.main:app`。"
    )
    if st.button("Retry"):
        st.rerun()
    st.stop()


# ---------------------------------------------------------------------------
# 登录 / 注册
# ---------------------------------------------------------------------------

if not st.session_state.access_token:
    st.title("🎬 Movie Recommendation Agent")
    st.caption("Sign in — 会话历史与长期画像都按账号隔离。")

    login_tab, register_tab = st.tabs(["Sign in", "Register"])

    with login_tab:
        with st.form("login"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            if st.form_submit_button("Sign in"):
                try:
                    tokens = ApiClient().login(email, password)
                    st.session_state.access_token = tokens["access_token"]
                    st.session_state.refresh_token = tokens["refresh_token"]
                    st.session_state.user = client().me()
                    st.rerun()
                except ApiError as exc:
                    st.error(exc.message)

    with register_tab:
        with st.form("register"):
            new_email = st.text_input("Email", key="reg_email")
            new_password = st.text_input(
                "Password", type="password", key="reg_password", help="至少 8 位"
            )
            if st.form_submit_button("Create account"):
                try:
                    ApiClient().register(new_email, new_password)
                    st.success("Account created — switch to the Sign in tab.")
                except ApiError as exc:
                    st.error(exc.message)

    st.stop()


api = client()

# ---------------------------------------------------------------------------
# 会话选择
# ---------------------------------------------------------------------------

try:
    conversations = api.list_conversations()
except ApiError as exc:
    if exc.status == 401:
        # access token 15 分钟就过期，这里不静默续期：让用户重新登录一次，
        # 比在 UI 里维护一套 refresh 竞态更可靠。
        sign_out()
        st.warning("Session expired — please sign in again.")
        st.rerun()
    st.error(exc.message)
    st.stop()

if st.session_state.conversation is None:
    if conversations:
        st.session_state.conversation = conversations[0]
    else:
        st.session_state.conversation = api.create_conversation()
        conversations = [st.session_state.conversation]

conversation = st.session_state.conversation

st.title("🎬 Movie Recommendation Agent")

# 切换会话后要把历史从服务端重新拉一遍（``messages`` 表，不是 checkpoint）
if st.session_state.get("loaded_conversation") != conversation["id"]:
    try:
        history = api.list_messages(conversation["id"])
    except ApiError as exc:
        st.error(exc.message)
        history = []
    st.session_state.messages = [
        {"role": m["role"], "content": m["content"]} for m in history
    ]
    st.session_state.loaded_conversation = conversation["id"]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------------------------------------------------------------------
# 发消息
# ---------------------------------------------------------------------------

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

        displayed = ""
        tool_lines: list[str] = []
        final_response = ""

        try:
            events = api.send_message(
                conversation["id"],
                user_input,
                # 每次发送一个新 key。Streamlit 在网络抖动时会重跑脚本，
                # 没有这个 key 的话重跑就是重新烧一遍 token。
                idempotency_key=uuid.uuid4().hex,
            )
            for event in events:
                kind, data = event["event"], event["data"]

                if kind == "token":
                    displayed += data.get("content", "")
                    response_placeholder.markdown(displayed + "▌")

                elif kind == "tool_start":
                    tool_lines.append(f"🔧 **{data['name']}**\n> {str(data.get('input', ''))[:200]}")
                    tool_placeholder.markdown("\n\n".join(tool_lines))

                elif kind == "tool_end":
                    if tool_lines:
                        tool_lines[-1] = tool_lines[-1].replace("🔧", "✅")
                        tool_placeholder.markdown("\n\n".join(tool_lines))

                elif kind == "done":
                    final_response = data.get("response") or displayed
                    response_placeholder.markdown(final_response)
                    usage = data.get("token_usage")
                    if usage:
                        st.session_state.last_usage = usage
                        token_placeholder.caption(
                            f"📊 This message: {usage.get('prompt_tokens', 0)} prompt + "
                            f"{usage.get('completion_tokens', 0)} completion = "
                            f"**{usage.get('total_tokens', 0)} tokens**"
                        )
                    if not tool_lines:
                        tool_expander.empty()

                elif kind == "error":
                    # 流内错误：状态码早就发走了，只能作为事件到达。
                    final_response = displayed or "Sorry, something went wrong."
                    response_placeholder.markdown(final_response)
                    st.error(f"{data.get('code')}: {data.get('message')}")

        except ApiError as exc:
            final_response = "Sorry, the request was rejected."
            response_placeholder.markdown(final_response)
            st.error(exc.message)
        except OSError as exc:
            final_response = "Sorry, the connection dropped."
            response_placeholder.markdown(final_response)
            st.error(f"{type(exc).__name__}: {exc}")

    st.session_state.messages.append({"role": "assistant", "content": final_response})


# ---------------------------------------------------------------------------
# 侧边栏 —— 最后渲染，读到的是本轮对话之后的状态
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🎬 Movie Agent")
    user = st.session_state.user or {}
    st.caption(f"Signed in as **{user.get('email', '?')}** ({user.get('role', 'user')})")
    if st.button("Sign out"):
        sign_out()
        st.rerun()
    st.divider()

    # -- 会话 --
    st.markdown("**💬 Conversations**")
    if st.button("➕ New conversation"):
        try:
            st.session_state.conversation = api.create_conversation()
            st.session_state.messages = []
            st.session_state.loaded_conversation = None
            st.rerun()
        except ApiError as exc:
            st.error(exc.message)

    labels = {c["id"]: (c["title"] or "Untitled")[:40] for c in conversations}
    if labels:
        ids = list(labels)
        current = conversation["id"] if conversation["id"] in labels else ids[0]
        picked = st.radio(
            "Pick one",
            ids,
            index=ids.index(current),
            format_func=lambda cid: labels[cid],
            label_visibility="collapsed",
        )
        if picked != conversation["id"]:
            st.session_state.conversation = next(c for c in conversations if c["id"] == picked)
            st.rerun()

        if st.button("🗑️ Delete this conversation"):
            try:
                api.delete_conversation(conversation["id"])
                st.session_state.conversation = None
                st.session_state.messages = []
                st.session_state.loaded_conversation = None
                st.rerun()
            except ApiError as exc:
                st.error(exc.message)
    st.divider()

    st.markdown("**Try asking:**")
    st.markdown("- Recommend me a sci-fi movie from the 90s")
    st.markdown("- I loved Inception, what should I watch next?")
    st.markdown("- 聊聊黑色电影的视觉风格")
    st.markdown("- Add Blade Runner to my watchlist")
    st.divider()

    # -- 能力装配状态（服务端自述，不是 UI 猜的）--
    st.markdown("**🔌 Capabilities**")
    try:
        caps = api.capabilities()
        if caps["warnings"]:
            st.warning("MCP degraded:\n" + "\n".join(f"- {w}" for w in caps["warnings"]))
        else:
            st.caption("MCP: tmdb + watchlist servers connected")
        st.caption(f"Tools available: {caps['tool_count']}")
        st.caption("Skills: " + (", ".join(caps["skills"]) or "none loaded"))
        st.caption(f"Model: {caps['model']}")
    except ApiError as exc:
        st.caption(f"Capabilities unavailable: {exc.message}")
    st.divider()

    # -- Token 用量（服务端 usage_records 的真实汇总）--
    st.markdown("**📊 Token Usage**")
    try:
        usage = api.usage()
        if usage["request_count"]:
            st.markdown(
                f"Requests (30d): **{usage['request_count']}**\n\n"
                f"Prompt tokens: **{usage['prompt_tokens']}**\n\n"
                f"Completion tokens: **{usage['completion_tokens']}**\n\n"
                f"Total tokens: **{usage['total_tokens']}**"
            )
        else:
            st.caption("No usage recorded yet.")
    except ApiError as exc:
        st.caption(f"Usage unavailable: {exc.message}")
    st.divider()

    # -- 长期画像（跨会话存活）--
    st.markdown("**🧠 Long-term Profile**")
    labels_map = {
        "liked_genres": "Genres you like",
        "disliked_genres": "Genres you avoid",
        "liked_tones": "Tones you enjoy",
        "disliked_tones": "Tones you avoid",
        "liked_movies": "Enjoyed",
        "disliked_movies": "Disliked",
    }
    try:
        profile = api.profile()
        if not any(profile.get(k) for k in labels_map):
            st.caption("No preferences recorded yet.")
        else:
            for key, label in labels_map.items():
                if profile.get(key):
                    st.markdown(f"{label}: {', '.join(profile[key])}")
            if profile.get("last_updated"):
                st.caption(f"Updated: {profile['last_updated'][:10]}")

        if st.button("🧹 Clear my profile"):
            # 只清画像。会话历史与情节记忆不受影响 —— 这正是两层记忆的分工。
            api.clear_profile()
            st.rerun()
    except ApiError as exc:
        st.caption(f"Profile unavailable: {exc.message}")
