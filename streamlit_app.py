"""
StemAgent — Streamlit Frontend
Connects to a FastAPI + LangChain backend hosted on Hugging Face Spaces.
"""

import os
import uuid
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BACKEND_URL = "https://Codemaster67-GoolgeLangchainAgent.hf.space"

AVAILABLE_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
]

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="StemAgent",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>S</text></svg>",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hide Streamlit defaults */
    #MainMenu, footer {visibility: hidden;}
    .stDeployButton {display: none;}
    header[data-testid="stHeader"] {
        background: transparent !important;
        backdrop-filter: none !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #111118;
        border-right: 1px solid rgba(124, 58, 237, 0.12);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #E4E4E7;
        letter-spacing: -0.02em;
    }

    /* Chat messages */
    .stChatMessage {
        border-radius: 12px !important;
        padding: 1rem 1.25rem !important;
        margin-bottom: 0.5rem !important;
        border: 1px solid rgba(255,255,255,0.04) !important;
    }
    [data-testid="stChatMessageContent"] p {
        font-size: 0.925rem;
        line-height: 1.65;
        color: #d4d4d8;
    }

    /* Chat input */
    .stChatInput > div {
        border-radius: 14px !important;
        border: 1px solid rgba(124, 58, 237, 0.25) !important;
        background: #16161e !important;
        transition: border-color 0.2s ease;
    }
    .stChatInput > div:focus-within {
        border-color: rgba(124, 58, 237, 0.6) !important;
        box-shadow: 0 0 0 2px rgba(124, 58, 237, 0.1) !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 500;
        letter-spacing: 0.01em;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(124, 58, 237, 0.2);
    }

    /* Status pill */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 500;
        letter-spacing: 0.02em;
    }
    .status-connected {
        background: rgba(34, 197, 94, 0.12);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.2);
    }

    /* Divider */
    .sidebar-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(124,58,237,0.2), transparent);
        margin: 1.25rem 0;
    }

    /* ---- Setup page ---- */
    .setup-container {
        max-width: 460px;
        margin: 6vh auto 0 auto;
        padding: 2.5rem;
        border-radius: 20px;
        background: #16161e;
        border: 1px solid rgba(124, 58, 237, 0.15);
        box-shadow: 0 8px 40px rgba(0,0,0,0.4);
    }
    .setup-title {
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.04em;
        color: #fafafa;
        text-align: center;
        margin: 0 0 4px 0;
    }
    .setup-sub {
        font-size: 0.9rem;
        color: #71717a;
        text-align: center;
        margin: 0 0 2rem 0;
        line-height: 1.5;
    }
    .setup-label {
        font-size: 0.8rem;
        font-weight: 500;
        color: #a1a1aa;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }
    .setup-hint {
        font-size: 0.78rem;
        color: #52525b;
        margin-top: 4px;
    }

    /* Sidebar header */
    .sidebar-title {
        font-size: 1.25rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        color: #fafafa;
        margin: 0;
    }
    .sidebar-sub {
        font-size: 0.78rem;
        color: #71717a;
        margin-top: 2px;
    }
    .model-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 14px;
        border-radius: 8px;
        font-size: 0.78rem;
        font-weight: 500;
        color: #c4b5fd;
        background: rgba(124, 58, 237, 0.1);
        border: 1px solid rgba(124, 58, 237, 0.2);
    }

    /* Empty state */
    .empty-state {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 4rem 2rem;
        text-align: center;
    }
    .empty-state h2 {
        font-size: 1.35rem;
        font-weight: 600;
        color: #e4e4e7;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    .empty-state p {
        font-size: 0.875rem;
        color: #71717a;
        max-width: 360px;
        line-height: 1.5;
    }

    /* File upload */
    [data-testid="stFileUploader"] { border-radius: 10px; }
    [data-testid="stFileUploader"] > div { border-radius: 10px !important; }

    /* Spinner */
    .stSpinner > div { border-top-color: #7C3AED !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_default_api_key() -> str:
    """Return the pre-configured API key, or empty string."""
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass
    return os.environ.get("GOOGLE_API_KEY", "")


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent_ready" not in st.session_state:
    st.session_state.agent_ready = False
if "file_data" not in st.session_state:
    st.session_state.file_data = None
if "active_api_key" not in st.session_state:
    st.session_state.active_api_key = _get_default_api_key()
if "active_model" not in st.session_state:
    st.session_state.active_model = ""


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 1 — SETUP                                                       ║
# ╚═════════════════════════════════════════════════════════════════════════╝
def render_setup_page():
    """Startup page: model selection + optional API key override."""

    st.markdown('<div class="setup-container">', unsafe_allow_html=True)

    st.markdown('<p class="setup-title">StemAgent</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="setup-sub">Configure your research assistant, then start chatting.</p>',
        unsafe_allow_html=True,
    )

    # ---- Model ----
    st.markdown('<p class="setup-label">Model</p>', unsafe_allow_html=True)
    model = st.selectbox(
        "Model",
        AVAILABLE_MODELS,
        index=0,
        label_visibility="collapsed",
        key="setup_model",
    )

    st.markdown("", unsafe_allow_html=True)  # spacer

    # ---- API Key ----
    has_default_key = bool(st.session_state.active_api_key)

    if has_default_key:
        st.markdown('<p class="setup-label">API Key</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="setup-hint">A default key is configured. '
            'Override it below if needed, or leave blank to use the default.</p>',
            unsafe_allow_html=True,
        )
        override_key = st.text_input(
            "Override API Key",
            type="password",
            placeholder="Paste a different key (optional)",
            label_visibility="collapsed",
            key="setup_override_key",
        )
    else:
        st.markdown('<p class="setup-label">Google API Key</p>', unsafe_allow_html=True)
        override_key = st.text_input(
            "API Key",
            type="password",
            placeholder="Enter your Gemini API key",
            label_visibility="collapsed",
            key="setup_override_key",
        )

    st.markdown("", unsafe_allow_html=True)  # spacer

    # ---- Launch button ----
    api_key_to_use = override_key.strip() if override_key.strip() else st.session_state.active_api_key
    can_launch = bool(api_key_to_use)

    if st.button("Start Session", use_container_width=True, disabled=not can_launch, type="primary"):
        with st.spinner("Initializing agent..."):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/initialize",
                    data={"api_key": api_key_to_use, "model_name": model},
                    timeout=60,
                )
                if resp.status_code == 200:
                    st.session_state.active_api_key = api_key_to_use
                    st.session_state.active_model = model
                    st.session_state.agent_ready = True
                    st.rerun()
                else:
                    detail = resp.json().get("detail", "Unknown error")
                    st.error(f"Initialization failed: {detail}")
            except requests.exceptions.ConnectionError:
                st.error("Could not reach the backend. Is it running?")
            except Exception as exc:
                st.error(f"Error: {exc}")

    if not can_launch:
        st.caption("Enter an API key to continue.")

    st.markdown('</div>', unsafe_allow_html=True)


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 2 — CHAT                                                        ║
# ╚═════════════════════════════════════════════════════════════════════════╝
def render_chat_page():
    """Main chat UI with a minimal sidebar."""

    # ---- Sidebar ----
    with st.sidebar:
        st.markdown('<p class="sidebar-title">StemAgent</p>', unsafe_allow_html=True)
        st.markdown('<p class="sidebar-sub">Research assistant</p>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # Active model display
        st.markdown(
            f'<span class="model-badge">{st.session_state.active_model}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<span class="status-pill status-connected">Connected</span>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # File upload
        st.markdown("### Attach File")
        uploaded = st.file_uploader(
            "Upload a document or image",
            label_visibility="collapsed",
            type=["pdf", "png", "jpg", "jpeg", "csv", "txt", "md"],
        )
        if uploaded is not None:
            st.session_state.file_data = {
                "name": uploaded.name,
                "bytes": uploaded.getvalue(),
                "mime": uploaded.type or "application/octet-stream",
            }
            st.caption(f"Attached: {uploaded.name}")
        elif st.session_state.file_data:
            st.caption(f"Attached: {st.session_state.file_data['name']}")

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # Session controls
        st.markdown("### Session")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.messages = []
                try:
                    requests.delete(
                        f"{BACKEND_URL}/session/{st.session_state.session_id}",
                        timeout=10,
                    )
                except Exception:
                    pass
                st.session_state.session_id = str(uuid.uuid4())
                st.rerun()
        with col2:
            if st.button("New Session", use_container_width=True):
                st.session_state.messages = []
                st.session_state.file_data = None
                try:
                    requests.delete(
                        f"{BACKEND_URL}/session/{st.session_state.session_id}",
                        timeout=10,
                    )
                except Exception:
                    pass
                st.session_state.session_id = str(uuid.uuid4())
                st.rerun()

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

        # Reconfigure — go back to setup
        if st.button("Reconfigure", use_container_width=True):
            st.session_state.agent_ready = False
            st.session_state.active_model = ""
            st.session_state.messages = []
            st.session_state.file_data = None
            try:
                requests.delete(
                    f"{BACKEND_URL}/session/{st.session_state.session_id}",
                    timeout=10,
                )
            except Exception:
                pass
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()

    # ---- Chat area ----
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if not st.session_state.messages:
        st.markdown(
            """
            <div class="empty-state">
                <h2>What would you like to research?</h2>
                <p>Ask any research question. Attach files from the sidebar for document-aware queries.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Chat input
    if prompt := st.chat_input("Ask a research question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        form_data = {
            "message": prompt,
            "session_id": st.session_state.session_id,
        }
        files = None
        if st.session_state.file_data is not None:
            fd = st.session_state.file_data
            files = {"file": (fd["name"], fd["bytes"], fd["mime"])}

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/chat",
                        data=form_data,
                        files=files,
                        timeout=300,
                    )
                    if resp.status_code == 200:
                        body = resp.json()
                        answer = body.get("response") or body.get("error", "No response received.")
                    else:
                        answer = f"Backend returned status {resp.status_code}."
                except requests.exceptions.ConnectionError:
                    answer = "Could not reach the backend. Please check if it is running."
                except requests.exceptions.Timeout:
                    answer = "The request timed out. The query may be too complex — try again."
                except Exception as exc:
                    answer = f"An unexpected error occurred: {exc}"

            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.file_data = None


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  ROUTER                                                                ║
# ╚═════════════════════════════════════════════════════════════════════════╝
if st.session_state.agent_ready:
    render_chat_page()
else:
    render_setup_page()
