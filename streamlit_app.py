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
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — minimal, clean, premium dark UI
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    /* ---- Global ---- */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ---- Hide default Streamlit clutter ---- */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}

    /* ---- Sidebar ---- */
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

    /* ---- Chat messages ---- */
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

    /* ---- Chat input ---- */
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

    /* ---- Buttons ---- */
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

    /* ---- Status pill ---- */
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
    .status-disconnected {
        background: rgba(239, 68, 68, 0.12);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.2);
    }

    /* ---- Divider ---- */
    .sidebar-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(124,58,237,0.2), transparent);
        margin: 1.25rem 0;
    }

    /* ---- Hero header ---- */
    .hero-title {
        font-size: 1.75rem;
        font-weight: 600;
        letter-spacing: -0.03em;
        color: #fafafa;
        margin: 0;
        padding: 0;
    }
    .hero-sub {
        font-size: 0.85rem;
        color: #71717a;
        margin-top: 2px;
        font-weight: 400;
    }

    /* ---- Empty state ---- */
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

    /* ---- File upload area ---- */
    [data-testid="stFileUploader"] {
        border-radius: 10px;
    }
    [data-testid="stFileUploader"] > div {
        border-radius: 10px !important;
    }

    /* ---- Spinner ---- */
    .stSpinner > div {
        border-top-color: #7C3AED !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Resolve default API key from environment / Streamlit secrets
# ---------------------------------------------------------------------------
def _get_default_api_key() -> str:
    """Return the pre-configured API key, or empty string if none."""
    # Streamlit Cloud stores secrets in st.secrets
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass
    # Fallback to OS environment variable
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
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "active_api_key" not in st.session_state:
    st.session_state.active_api_key = _get_default_api_key()
if "init_model" not in st.session_state:
    st.session_state.init_model = ""


# ---------------------------------------------------------------------------
# Auto-initialize on first load if a default key exists
# ---------------------------------------------------------------------------
def _auto_initialize(api_key: str, model: str):
    """Call the backend /initialize endpoint and update session state."""
    try:
        resp = requests.post(
            f"{BACKEND_URL}/initialize",
            data={"api_key": api_key, "model_name": model},
            timeout=60,
        )
        if resp.status_code == 200:
            st.session_state.agent_ready = True
            st.session_state.init_model = model
        else:
            st.session_state.agent_ready = False
    except Exception:
        st.session_state.agent_ready = False


if (
    not st.session_state.agent_ready
    and st.session_state.active_api_key
    and not st.session_state.init_model
):
    _auto_initialize(st.session_state.active_api_key, AVAILABLE_MODELS[0])

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<p class="hero-title">StemAgent</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Research assistant powered by LangChain</p>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # -- Model selector --
    st.markdown("### Configuration")

    current_model_index = (
        AVAILABLE_MODELS.index(st.session_state.init_model)
        if st.session_state.init_model in AVAILABLE_MODELS
        else 0
    )
    model_name = st.selectbox(
        "Model",
        AVAILABLE_MODELS,
        index=current_model_index,
        help="Select the Gemini model variant to use.",
    )

    # Re-initialize if user switches model while already connected
    if (
        st.session_state.agent_ready
        and st.session_state.init_model
        and model_name != st.session_state.init_model
    ):
        with st.spinner("Switching model..."):
            _auto_initialize(st.session_state.active_api_key, model_name)
            if st.session_state.agent_ready:
                st.session_state.init_model = model_name

    # -- Change API Key (collapsible) --
    with st.expander("Change API Key"):
        new_key = st.text_input(
            "New Google API Key",
            type="password",
            placeholder="Paste a different key here",
            label_visibility="collapsed",
        )
        apply_btn = st.button(
            "Apply Key",
            use_container_width=True,
            disabled=not new_key,
        )
        if apply_btn and new_key:
            with st.spinner("Re-initializing with new key..."):
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/initialize",
                        data={"api_key": new_key, "model_name": model_name},
                        timeout=60,
                    )
                    if resp.status_code == 200:
                        st.session_state.active_api_key = new_key
                        st.session_state.agent_ready = True
                        st.session_state.init_model = model_name
                        st.success("Key updated. Agent is ready.")
                    else:
                        detail = resp.json().get("detail", "Unknown error")
                        st.error(f"Invalid key: {detail}")
                except requests.exceptions.ConnectionError:
                    st.error("Could not reach the backend. Is it running?")
                except Exception as exc:
                    st.error(f"Error: {exc}")

    # Show a hint if no key at all
    if not st.session_state.active_api_key and not st.session_state.agent_ready:
        st.caption("No default key found. Expand 'Change API Key' above to set one.")

    # -- Status --
    if st.session_state.agent_ready:
        st.markdown(
            '<span class="status-pill status-connected">Connected</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-pill status-disconnected">Not initialized</span>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # -- File upload --
    st.markdown("### Attach File")
    uploaded = st.file_uploader(
        "Upload a document or image",
        label_visibility="collapsed",
        type=["pdf", "png", "jpg", "jpeg", "csv", "txt", "md"],
    )
    if uploaded is not None:
        st.session_state.uploaded_file = uploaded
        st.caption(f"Attached: {uploaded.name}")

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # -- Session controls --
    st.markdown("### Session")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            # Tell backend to drop the thread
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
            st.session_state.agent_ready = False
            st.session_state.uploaded_file = None
            try:
                requests.delete(
                    f"{BACKEND_URL}/session/{st.session_state.session_id}",
                    timeout=10,
                )
            except Exception:
                pass
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Empty-state prompt
if not st.session_state.messages:
    if st.session_state.agent_ready:
        empty_msg = "Ask any research question. Attach files from the sidebar for document-aware queries."
    else:
        empty_msg = "Set your API key in the sidebar to get started, then ask any research question."
    st.markdown(
        f"""
        <div class="empty-state">
            <h2>What would you like to research?</h2>
            <p>{empty_msg}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Chat input
if prompt := st.chat_input("Ask a research question...", disabled=not st.session_state.agent_ready):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build request payload
    form_data = {
        "message": prompt,
        "session_id": st.session_state.session_id,
    }
    files = None
    if st.session_state.uploaded_file is not None:
        uf = st.session_state.uploaded_file
        files = {"file": (uf.name, uf.getvalue(), uf.type)}

    # Call the backend
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

    # Clear file after sending so it's not re-sent on every message
    st.session_state.uploaded_file = None
