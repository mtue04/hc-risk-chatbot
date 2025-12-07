import os
from typing import Dict
from uuid import uuid4

import requests
import streamlit as st


st.set_page_config(
    page_title="HomeCredit Risk Analytics",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

CHATBOT_URL = os.getenv("CHATBOT_API_URL", "http://localhost:8500/chat")
MODEL_URL = os.getenv("MODEL_API_URL", "http://localhost:8000/health")


# Minimal Design System
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
    :root {
        --gray-50: #fafafa;
        --gray-100: #f5f5f5;
        --gray-200: #e5e5e5;
        --gray-300: #d4d4d4;
        --gray-400: #a3a3a3;
        --gray-500: #737373;
        --gray-600: #525252;
        --gray-700: #404040;
        --gray-800: #262626;
        --gray-900: #171717;
        --blue-600: #2563eb;
        --green-600: #16a34a;
        --amber-600: #d97706;
        --red-600: #dc2626;
    }

    * {
        font-family: 'Inter', -apple-system, sans-serif;
    }

    .main {
        background: white;
    }

    .block-container {
        padding: 3rem 4rem;
        max-width: 1200px;
    }

    /* Typography */
    h1, h2, h3 {
        font-weight: 300;
        color: var(--gray-900);
        letter-spacing: -0.02em;
    }

    h1 {
        font-size: 2rem !important;
        margin-bottom: 0.5rem !important;
    }

    h2 {
        font-size: 1.25rem !important;
        margin: 2.5rem 0 1rem 0 !important;
        font-weight: 400 !important;
    }

    h3 {
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--gray-500);
        margin-bottom: 1rem !important;
    }

    p, div, span, label {
        color: var(--gray-700);
        font-size: 0.9375rem;
        line-height: 1.6;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--gray-50);
        border-right: 1px solid var(--gray-200);
        padding: 2rem 1.25rem !important;
    }

    [data-testid="stSidebar"] h3 {
        font-size: 0.6875rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--gray-500);
        margin-bottom: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--gray-200);
    }

    /* Status Badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 0.75rem;
        background: white;
        border: 1px solid var(--gray-200);
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.6875rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--gray-700);
    }

    .status-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: var(--green-600);
    }

    .status-dot.error {
        background: var(--red-600);
    }

    /* Tool List */
    .tool-item {
        padding: 0.625rem 0;
        border-bottom: 1px solid var(--gray-100);
        font-size: 0.8125rem;
        color: var(--gray-600);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .tool-item:last-child {
        border-bottom: none;
    }

    .tool-item i {
        color: var(--gray-400);
        font-size: 0.75rem;
        width: 14px;
    }

    /* Buttons */
    .stButton > button {
        background: white;
        color: var(--gray-700);
        border: 1px solid var(--gray-200);
        border-radius: 6px;
        font-size: 0.8125rem;
        font-weight: 500;
        padding: 0.625rem 1rem;
        transition: all 0.15s ease;
    }

    .stButton > button:hover {
        background: var(--gray-50);
        border-color: var(--gray-300);
    }

    /* Quick Actions */
    .quick-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.75rem;
        margin: 1.5rem 0 2.5rem 0;
    }

    .quick-card {
        background: white;
        border: 1px solid var(--gray-200);
        border-radius: 8px;
        padding: 1.25rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.15s ease;
    }

    .quick-card:hover {
        border-color: var(--gray-300);
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    .quick-card i {
        font-size: 1.25rem;
        color: var(--gray-400);
        margin-bottom: 0.5rem;
        display: block;
    }

    .quick-card-label {
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--gray-600);
    }

    /* Risk Gauge */
    .risk-container {
        background: white;
        border: 1px solid var(--gray-200);
        border-radius: 8px;
        padding: 2rem;
        text-align: center;
    }

    .risk-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 3rem;
        font-weight: 400;
        color: var(--gray-900);
        line-height: 1;
    }

    .risk-label {
        font-size: 0.6875rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--gray-500);
        margin-top: 0.5rem;
    }

    .risk-bar {
        width: 100%;
        height: 4px;
        background: var(--gray-100);
        margin: 1.5rem 0;
        border-radius: 2px;
        overflow: hidden;
    }

    .risk-fill {
        height: 100%;
        transition: width 0.5s ease;
        border-radius: 2px;
    }

    .risk-low { background: var(--green-600); }
    .risk-medium { background: var(--amber-600); }
    .risk-high { background: var(--red-600); }

    /* Metrics */
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.5rem !important;
        font-weight: 400 !important;
        color: var(--gray-900) !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.6875rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
        color: var(--gray-500) !important;
    }

    /* Chat */
    .stChatMessage {
        background: transparent;
        border: none;
        padding: 1.25rem 0;
        border-bottom: 1px solid var(--gray-100);
    }

    .stChatMessage:last-child {
        border-bottom: none;
    }

    /* Input */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8125rem;
        border: 1px solid var(--gray-200);
        border-radius: 6px;
        background: white;
        color: var(--gray-900);
    }

    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: var(--gray-400);
        box-shadow: 0 0 0 3px rgba(0,0,0,0.02);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
        border-bottom: 1px solid var(--gray-200);
    }

    .stTabs [data-baseweb="tab"] {
        font-size: 0.8125rem;
        font-weight: 500;
        color: var(--gray-500);
        background: transparent;
        border: none;
        padding: 0.75rem 0;
        border-bottom: 2px solid transparent;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: var(--gray-900);
        border-bottom-color: var(--gray-900);
    }

    /* Divider */
    hr {
        border: none;
        border-top: 1px solid var(--gray-200);
        margin: 2rem 0;
    }

    /* Alerts */
    .stAlert {
        border-radius: 8px;
        border: 1px solid var(--gray-200);
    }

    /* JSON */
    .stJson {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
    }

    /* Spinner */
    .stSpinner > div {
        border-color: var(--gray-900) transparent transparent transparent;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _check_model_health() -> Dict[str, str]:
    try:
        response = requests.get(MODEL_URL, timeout=2)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return {"status": "unavailable"}


def initialize_session_state():
    """Initialize session state variables."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "applicant_id" not in st.session_state:
        st.session_state.applicant_id = None
    if "last_risk_score" not in st.session_state:
        st.session_state.last_risk_score = None


def display_message(role: str, content: str):
    """Display a chat message."""
    with st.chat_message(role):
        st.markdown(content)


def render_risk_gauge(risk_prob: float):
    """Render a minimal risk gauge."""
    risk_pct = risk_prob * 100

    if risk_prob < 0.3:
        risk_label = "Low Risk"
        risk_class = "risk-low"
    elif risk_prob < 0.6:
        risk_label = "Medium Risk"
        risk_class = "risk-medium"
    else:
        risk_label = "High Risk"
        risk_class = "risk-high"

    st.markdown(f"""
    <div class="risk-container">
        <div class="risk-value">{risk_pct:.1f}%</div>
        <div class="risk-label">{risk_label}</div>
        <div class="risk-bar">
            <div class="risk-fill {risk_class}" style="width: {risk_pct}%;"></div>
        </div>
        <div style="font-size: 0.75rem; color: var(--gray-500);">Default Probability</div>
    </div>
    """, unsafe_allow_html=True)


def main():
    # Simple header
    st.markdown("# HomeCredit Risk Analytics")
    st.markdown("AI-Powered Credit Intelligence")

    st.markdown("<hr style='margin: 1.5rem 0;'>", unsafe_allow_html=True)

    initialize_session_state()

    # Sidebar
    with st.sidebar:
        # Status
        health = _check_model_health()
        status_ok = health.get("status") == "ok"
        status_class = "" if status_ok else "error"

        st.markdown(f"""
        <div class="status-badge">
            <span class="status-dot {status_class}"></span>
            <span>{health.get('status', 'unknown').upper()}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Session
        st.markdown("### Session")
        st.text_input("ID", value=st.session_state.session_id[:8] + "...", disabled=True, label_visibility="collapsed")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("New", use_container_width=True):
                st.session_state.session_id = str(uuid4())
                st.session_state.messages = []
                st.session_state.last_risk_score = None
                st.rerun()
        with col2:
            msg_count = len(st.session_state.messages) // 2
            st.metric("Turns", msg_count, label_visibility="collapsed")

        st.markdown("<hr>", unsafe_allow_html=True)

        # Applicant
        st.markdown("### Applicant")
        applicant_input = st.number_input(
            "ID",
            min_value=100000,
            max_value=999999,
            value=st.session_state.applicant_id or 100001,
            step=1,
            label_visibility="collapsed"
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Set", use_container_width=True):
                st.session_state.applicant_id = applicant_input
        with col2:
            if st.button("Clear", use_container_width=True):
                st.session_state.applicant_id = None

        st.markdown("<hr>", unsafe_allow_html=True)

        # Tools
        st.markdown("### Tools")
        tools = [
            ("fa-circle", "Risk Prediction"),
            ("fa-circle", "Applicant Data"),
            ("fa-circle", "Feature Plot"),
            ("fa-circle", "Compare"),
            ("fa-circle", "Explain Factors"),
            ("fa-circle", "Bureau History"),
            ("fa-circle", "Portfolio Stats"),
            ("fa-circle", "What-If Analysis"),
        ]

        for icon, name in tools:
            st.markdown(f'<div class="tool-item"><i class="fas {icon}"></i><span>{name}</span></div>', unsafe_allow_html=True)

    # Main Content
    st.markdown("## Quick Actions")

    quick_action_data = [
        ("fa-circle-dot", "Risk", f"What is the risk score for applicant {st.session_state.applicant_id or 100001}?"),
        ("fa-user", "Profile", f"Show me the profile for applicant {st.session_state.applicant_id or 100001}"),
        ("fa-star", "Factors", f"What are the top risk factors for applicant {st.session_state.applicant_id or 100001}?"),
        ("fa-layer-group", "Compare", f"Compare applicant {st.session_state.applicant_id or 100001} with similar applicants"),
        ("fa-wand-magic-sparkles", "What-If", f"What if applicant {st.session_state.applicant_id or 100001} had income of 300000?"),
        ("fa-clock", "History", f"Show credit bureau history for applicant {st.session_state.applicant_id or 100001}"),
        ("fa-chart-simple", "Portfolio", "What is the overall portfolio risk distribution?"),
        ("fa-table", "Features", f"Compare income, credit, and employment for applicant {st.session_state.applicant_id or 100001}"),
    ]

    cols = st.columns(4)
    for idx, (icon, label, query) in enumerate(quick_action_data):
        with cols[idx % 4]:
            st.markdown(f"<div style='text-align: center; margin-bottom: 0.5rem;'><i class='fas {icon}' style='font-size: 1.25rem; color: var(--gray-400);'></i></div>", unsafe_allow_html=True)
            if st.button(label, use_container_width=True, key=f"qa_{idx}"):
                # Add user message and set pending flag
                st.session_state.messages.append({"role": "user", "content": query})
                st.session_state.pending_query = query
                st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

    # Risk Dashboard
    if st.session_state.last_risk_score is not None:
        st.markdown("## Risk Assessment")

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            render_risk_gauge(st.session_state.last_risk_score)

        with col2:
            st.metric("Applicant", st.session_state.applicant_id or "N/A")

        with col3:
            risk_pct = st.session_state.last_risk_score * 100
            threshold = 50.0
            delta_pct = risk_pct - threshold
            st.metric(
                "vs Threshold",
                f"{abs(delta_pct):.1f}%",
                delta=f"{delta_pct:+.1f}%",
                delta_color="inverse"
            )

        st.markdown("<hr>", unsafe_allow_html=True)

    # Conversation
    st.markdown("## Conversation")

    for msg in st.session_state.messages:
        display_message(msg["role"], msg["content"])

    # Handle pending query from quick actions
    if hasattr(st.session_state, 'pending_query') and st.session_state.pending_query:
        prompt = st.session_state.pending_query
        st.session_state.pending_query = None  # Clear the flag

        request_data = {
            "question": prompt,
            "session_id": st.session_state.session_id,
        }

        if st.session_state.applicant_id:
            request_data["applicant_id"] = st.session_state.applicant_id

        with st.spinner("Analyzing..."):
            try:
                response = requests.post(CHATBOT_URL, json=request_data, timeout=60)
                response.raise_for_status()
                data = response.json()

                answer = data.get("answer", "I couldn't generate a response.")
                st.session_state.messages.append({"role": "assistant", "content": answer})

                if data.get("risk_probability") is not None:
                    st.session_state.last_risk_score = data["risk_probability"]

                st.rerun()

            except requests.RequestException as exc:
                st.error(f"Request failed: {exc}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Error: {exc}"
                })
                st.rerun()

    if prompt := st.chat_input("Ask about credit risk, applicants, or portfolio analytics..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_message("user", prompt)

        request_data = {
            "question": prompt,
            "session_id": st.session_state.session_id,
        }

        if st.session_state.applicant_id:
            request_data["applicant_id"] = st.session_state.applicant_id

        with st.spinner("Analyzing..."):
            try:
                response = requests.post(CHATBOT_URL, json=request_data, timeout=60)
                response.raise_for_status()
                data = response.json()

                answer = data.get("answer", "I couldn't generate a response.")
                st.session_state.messages.append({"role": "assistant", "content": answer})

                if data.get("risk_probability") is not None:
                    st.session_state.last_risk_score = data["risk_probability"]

                # Rerun to show the new message in the conversation
                st.rerun()

            except requests.RequestException as exc:
                st.error(f"Request failed: {exc}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Error: {exc}"
                })


if __name__ == "__main__":
    main()
