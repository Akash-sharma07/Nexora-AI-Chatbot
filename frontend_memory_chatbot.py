"""Streamlit frontend with per-user authentication and isolated chat history."""
from __future__ import annotations

import queue
import uuid
from html import escape
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.types import Command

from backend_memory_chatbot import (
    chatbot,
    get_thread_history,
    async_submit_task,
    ingest_pdf,
    thread_document_metadata,
    run_async,
    get_title,
    save_title,
    get_saved_title,
    assert_thread_owner,
    get_memories,
    forget_memories,
)
from Auth_db import register_user, authenticate_user, create_conversation

st.set_page_config(
    page_title="Nexora AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css() -> None:
    """Load the local Nexora theme without changing backend behavior."""
    base_dir = Path(__file__).parent
    # Prefer the production filename; fallback supports local files named style(1).css.
    css_path = base_dir / "style.css"
    if not css_path.exists():
        css_path = base_dir / "style(1).css"
    if css_path.exists():
        st.markdown(
            f"<style>{css_path.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )
    else:
        st.warning("Nexora theme stylesheet not found. Place style.css beside this frontend file.")


load_css()


# ---------------- Authentication ----------------
def render_auth_screen() -> None:
    st.markdown(
        """
        <div class="auth-brand">
            <div class="nexora-logo">✦</div>
            <div>
                <div class="auth-brand-title">Nexora AI</div>
                <div class="auth-brand-subtitle">Your personal AI assistant</div>
            </div>
        </div>
        <div class="auth-intro">
            <h1>Welcome back</h1>
            <p>Sign in to continue your conversations and access your saved memories.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    mode = st.radio("Choose an option", ["Log in", "Create account"], horizontal=True)

    if mode == "Log in":
        with st.form("login_form"):
            email = st.text_input("Email", autocomplete="email")
            password = st.text_input("Password", type="password", autocomplete="current-password")
            submitted = st.form_submit_button("Log in", type="primary", use_container_width=True)
        if submitted:
            user = authenticate_user(email, password)
            if user:
                st.session_state["auth_user"] = user
                st.rerun()
            st.error("Email or password is incorrect.")
    else:
        with st.form("signup_form"):
            name = st.text_input("Full name", max_chars=80)
            email = st.text_input("Email", autocomplete="email")
            password = st.text_input("Password (at least 10 characters)", type="password", autocomplete="new-password")
            confirm = st.text_input("Confirm password", type="password", autocomplete="new-password")
            submitted = st.form_submit_button("Create account", type="primary", use_container_width=True)
        if submitted:
            if password != confirm:
                st.error("Passwords do not match.")
            else:
                ok, message, _ = register_user(email, name, password)
                if ok:
                    st.success(message + " Please log in.")
                else:
                    st.error(message)


if not st.session_state.get("auth_user"):
    render_auth_screen()
    st.stop()

user = st.session_state["auth_user"]
user_id = user["user_id"]  # Comes from server-side session after password verification.


# ---------------- Conversation helpers ----------------
def generate_thread_id() -> str:
    return str(uuid.uuid4())


def convert_messages(messages):
    visible = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            visible.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage) and msg.content:
            visible.append({"role": "assistant", "content": msg.content})
    return visible


def load_conversation(owner_id: str, thread_id: str):
    assert_thread_owner(owner_id, thread_id)

    async def _get():
        state = await chatbot.aget_state(
            config={"configurable": {"thread_id": thread_id, "user_id": owner_id}}
        )
        return state.values.get("messages", [])

    return run_async(_get())


def create_new_chat(owner_id: str) -> str:
    thread_id = generate_thread_id()
    create_conversation(owner_id, thread_id, "New Chat")
    return thread_id


def reset_chat() -> None:
    thread_id = create_new_chat(user_id)
    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []
    st.session_state["pending_interrupt"] = None
    st.session_state["thread_history"] = get_thread_history(user_id)


def stream_graph_response(input_value, config):
    # Validate ownership before every graph run, including resuming an interrupt.
    cfg = (config or {}).get("configurable", {})
    assert_thread_owner(cfg.get("user_id"), cfg.get("thread_id"))

    status = {"box": None}

    def ai_only_stream():
        event_queue: queue.Queue = queue.Queue()

        async def run_stream():
            try:
                async for chunks, metadata in chatbot.astream(
                    input_value,
                    config=config,
                    stream_mode="messages",
                ):
                    event_queue.put((chunks, metadata))
            except Exception as exc:
                event_queue.put(("error", exc))
            finally:
                event_queue.put(None)

        async_submit_task(run_stream())
        while True:
            item = event_queue.get()
            if item is None:
                break
            chunks, metadata = item
            if chunks == "error":
                raise metadata
            if isinstance(chunks, ToolMessage):
                tool_name = getattr(chunks, "name", "Tool")
                if status["box"] is None:
                    status["box"] = st.status(f"Using {tool_name}...", expanded=True)
                else:
                    status["box"].update(label=f"Using {tool_name}...", expanded=True, state="running")
            if (
                metadata.get("langgraph_node") == "chat_node"
                and isinstance(chunks, AIMessage) 
                and chunks.content
            ):
                yield chunks.content

    ai_messages = st.write_stream(ai_only_stream())
    if status["box"] is not None:
        status["box"].update(label="Tool finished", expanded=False, state="complete")

    state = run_async(chatbot.aget_state(config))
    interrupt_payload = None
    for task in state.tasks:
        if task.interrupts:
            interrupt_payload = task.interrupts[0].value
            break
    return ai_messages, interrupt_payload


# ---------------- Per-user session initialization ----------------
# A Streamlit browser session must never reuse another account's chat state.
if st.session_state.get("_active_user_id") != user_id:
    for key in (
        "thread_id", "message_history", "thread_history", "ingested_doc",
        "thread_title", "pending_interrupt",
    ):
        st.session_state.pop(key, None)
    st.session_state["_active_user_id"] = user_id
    st.session_state["thread_history"] = get_thread_history(user_id)
    if st.session_state["thread_history"]:
        initial_thread = st.session_state["thread_history"][0]
        st.session_state["thread_id"] = initial_thread
        st.session_state["message_history"] = convert_messages(load_conversation(user_id, initial_thread))
    else:
        st.session_state["thread_id"] = create_new_chat(user_id)
        st.session_state["thread_history"] = get_thread_history(user_id)
        st.session_state["message_history"] = []
    st.session_state["ingested_doc"] = {}
    st.session_state["thread_title"] = {}
    st.session_state["pending_interrupt"] = None

for key, default in (
    ("message_history", []),
    ("thread_history", get_thread_history(user_id)),
    ("ingested_doc", {}),
    ("thread_title", {}),
    ("pending_interrupt", None),
):
    if key not in st.session_state:
        st.session_state[key] = default

# Re-query the database; never trust a browser-side list as an ownership check.
st.session_state["thread_history"] = get_thread_history(user_id)
current_thread = st.session_state["thread_id"]
assert_thread_owner(user_id, current_thread)


# ---------------- Nexora AI sidebar and main header ----------------
st.sidebar.markdown(
    """
    <div class="sidebar-brand">
        <div class="sidebar-brand-icon">✦</div>
        <div class="sidebar-brand-copy">
            <div class="sidebar-brand-title">Nexora AI</div>
            <div class="sidebar-brand-subtitle">Your personal AI assistant</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.sidebar.button("＋  New Chat", key="new_chat_button", use_container_width=True):
    reset_chat()
    st.rerun()

search_query = st.sidebar.text_input(
    "Search conversations",
    placeholder="⌕  Search conversations...",
    key="conversation_search",
    label_visibility="collapsed",
)
st.sidebar.markdown('<div class="sidebar-section-heading">CONVERSATIONS</div>', unsafe_allow_html=True)

visible_threads = []
for thread_id in st.session_state["thread_history"]:
    # Conversation names are still retrieved from the owner-filtered database.
    thread_title = (
        st.session_state["thread_title"].get(thread_id)
        or get_saved_title(user_id, thread_id)
        or "New Chat"
    )
    if not search_query or search_query.casefold() in thread_title.casefold():
        visible_threads.append((thread_id, thread_title))

if visible_threads:
    for thread_id, thread_title in visible_threads:
        is_current = thread_id == current_thread
        button_label = ("▸  " if is_current else "   ") + thread_title
        if st.sidebar.button(
            button_label,
            key=f"thread_{thread_id}",
            use_container_width=True,
            help=thread_title,
        ):
            assert_thread_owner(user_id, thread_id)
            st.session_state["thread_id"] = thread_id
            st.session_state["message_history"] = convert_messages(load_conversation(user_id, thread_id))
            st.session_state["pending_interrupt"] = None
            st.rerun()
else:
    st.sidebar.caption("No matching conversations.")

st.sidebar.divider()
st.sidebar.markdown('<div class="sidebar-section-heading">TOOLS &amp; DOCUMENTS</div>', unsafe_allow_html=True)
st.sidebar.markdown(
    '<div class="sidebar-nav-item"><span class="sidebar-nav-icon">▤</span>'
    '<span><b>PDF analysis</b><small>Upload PDFs from the chat bar</small></span></div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    '<div class="sidebar-nav-item"><span class="sidebar-nav-icon">⌘</span>'
    '<span><b>MCP tools</b><small>Connected tools run through your workflow</small></span></div>',
    unsafe_allow_html=True,
)

with st.sidebar.expander("✧  Long-term memory", expanded=False):
    memories = get_memories(user_id)
    if memories:
        for memory in memories:
            st.caption(f"• {memory}")
        if st.button("Forget all saved memories", key="forget_all_memories", use_container_width=True):
            forget_memories(user_id, "all")
            st.success("Your saved memories were deleted.")
            st.rerun()
    else:
        st.caption("No saved memories yet. In chat, type: Remember: I prefer concise answers.")

# Documents remain isolated by both user ID and thread ID.
doc_metadata = thread_document_metadata(user_id, current_thread)
if doc_metadata and doc_metadata.get("files"):
    st.sidebar.markdown(
        f'<div class="sidebar-document-status">● {doc_metadata.get("total_files", 0)} PDF(s) indexed in this chat</div>',
        unsafe_allow_html=True,
    )
    for file in doc_metadata["files"]:
        st.sidebar.caption(f"▤  {file['filename']} · {file['chunks']} chunks")
else:
    st.sidebar.markdown('<div class="sidebar-muted-note">No PDF indexed for this chat yet</div>', unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-account-divider"></div>', unsafe_allow_html=True)
safe_display_name = escape(str(user.get("name") or "User"))
safe_avatar_initial = escape((str(user.get("name") or "U")[:1]).upper())
st.sidebar.markdown(
    f"""
    <div class="sidebar-account-card">
        <div class="account-avatar">{safe_avatar_initial}</div>
        <div class="account-copy">
            <div class="account-name">{safe_display_name}</div>
            <div class="account-subtitle">Personal workspace</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
if st.sidebar.button("Log out", key="logout_button", use_container_width=True):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# Empty-state landing panel: show only before the first message in a chat.
if not st.session_state["message_history"] and not st.session_state["pending_interrupt"]:
    st.markdown(
        """
        <section class="nexora-hero">
            <div class="hero-orbit"><div class="hero-orbit-core">✦</div></div>
            <h1>What can I help you with today?</h1>
            <p>Chat naturally, analyze your documents, use powerful tools,<br class="desktop-break">
            and get personalized assistance.</p>
        </section>
        <div class="feature-grid">
            <div class="feature-card feature-blue">
                <div class="feature-icon">☏</div>
                <div><h3>Chat &amp; Learn</h3><p>Ask questions and get clear answers</p></div><span class="feature-arrow">›</span>
            </div>
            <div class="feature-card feature-pink">
                <div class="feature-icon">▤</div>
                <div><h3>Analyze PDFs</h3><p>Upload documents and get insights</p></div><span class="feature-arrow">›</span>
            </div>
            <div class="feature-card feature-cyan">
                <div class="feature-icon">⚒</div>
                <div><h3>Use MCP Tools</h3><p>Access connected tools and data</p></div><span class="feature-arrow">›</span>
            </div>
            <div class="feature-card feature-violet">
                <div class="feature-icon">▤</div>
                <div><h3>Remember Context</h3><p>Personalized help across chats</p></div><span class="feature-arrow">›</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------- Render existing messages ----------------
for message in st.session_state["message_history"]:
    avatar = "✨" if message["role"] == "assistant" else "👤"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

config = {
    "configurable": {
        "thread_id": current_thread,
        "user_id": user_id,
    },
    "metadata": {"thread_id": current_thread, "user_id": user_id},
    "run_name": "chat turn",
}


# ---------------- Tool approval flow ----------------
if st.session_state["pending_interrupt"]:
    data = st.session_state["pending_interrupt"]
    # A pending approval is valid only for the current user's current thread.
    pending_config = data["config"]
    assert_thread_owner(
        pending_config["configurable"].get("user_id"),
        pending_config["configurable"].get("thread_id"),
    )
    if pending_config["configurable"].get("thread_id") != current_thread:
        st.session_state["pending_interrupt"] = None
        st.rerun()

    with st.chat_message("assistant", avatar="✨"):
        st.warning("Approval required before running this tool call:")
        for tc in data["payload"]["tool_calls"]:
            st.code(f"{tc['name']}({tc['args']})")
        col1, col2 = st.columns(2)
        approve_clicked = col1.button("Approve", key="approve_tool")
        reject_clicked = col2.button("Reject", key="reject_tool")

    if approve_clicked or reject_clicked:
        resume_config = data["config"]
        with st.chat_message("assistant", avatar="✨"):
            ai_messages, new_interrupt = stream_graph_response(
                Command(resume={"approved": approve_clicked}),
                config=resume_config,
            )
        if ai_messages:
            st.session_state["message_history"].append({"role": "assistant", "content": ai_messages})
        st.session_state["pending_interrupt"] = (
            {"payload": new_interrupt, "config": resume_config} if new_interrupt else None
        )
        st.rerun()
else:
    prompt = st.chat_input(
        "Ask Nexora anything...",
        accept_file="multiple",
        max_upload_size=50,
        file_type=["pdf"],
    )

    if prompt:
        user_input = (prompt.text or "").strip()
        uploaded_files = prompt.files or []
        thread_doc = st.session_state["ingested_doc"].setdefault(current_thread, {})
        existing_names = {f["filename"] for f in doc_metadata.get("files", [])}

        if uploaded_files:
            for uploaded_pdf in uploaded_files:
                if uploaded_pdf.name in existing_names or uploaded_pdf.name in thread_doc:
                    st.info(f"`{uploaded_pdf.name}` is already processed for this chat.")
                    continue
                with st.status(f"Indexing {uploaded_pdf.name}...", expanded=True) as status_box:
                    summary = ingest_pdf(
                        file_bytes=uploaded_pdf.getvalue(),
                        user_id=user_id,
                        thread_id=current_thread,
                        filename=uploaded_pdf.name,
                    )
                    thread_doc[uploaded_pdf.name] = summary
                    status_box.update(label=f"{uploaded_pdf.name} indexed", state="complete", expanded=False)

            if not user_input:
                filenames = ", ".join(file.name for file in uploaded_files)
                user_input = (
                    f"I uploaded these documents: {filenames}. "
                    "Please give me a brief summary of each document."
                )

        if user_input:
            if not get_saved_title(user_id, current_thread) or get_saved_title(user_id, current_thread) == "New Chat":
                title = get_title(user_input)
                st.session_state["thread_title"][current_thread] = title
                save_title(user_id, current_thread, title)

            st.session_state["message_history"].append({"role": "user", "content": user_input})
            with st.chat_message("user", avatar="👤"):
                st.markdown(user_input)

            with st.chat_message("assistant", avatar="✨"):
                ai_text, interrupt_payload = stream_graph_response(
                    input_value={"messages": [HumanMessage(content=user_input)]},
                    config=config,
                )

            if interrupt_payload:
                st.session_state["pending_interrupt"] = {
                    "payload": interrupt_payload,
                    "config": config,
                }
                st.rerun()
            else:
                st.session_state["message_history"].append({
                    "role": "assistant",
                    "content": ai_text or "No response was generated.",
                })
                st.session_state["thread_history"] = get_thread_history(user_id)
