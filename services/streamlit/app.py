import json
import os
from typing import Dict
from uuid import uuid4

import requests
import streamlit as st


st.set_page_config(page_title="HC Risk Chatbot", layout="wide", page_icon="💬")

CHATBOT_URL = os.getenv("CHATBOT_API_URL", "http://localhost:8500/chat")
MODEL_URL = os.getenv("MODEL_API_URL", "http://localhost:8000/health")


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


def display_message(role: str, content: str):
    """Display a chat message."""
    with st.chat_message(role):
        st.markdown(content)


def main():
    st.title("🏦 HomeCredit Risk Chatbot")
    st.caption("AI-powered conversational credit risk analytics powered by LangGraph + Gemini")

    # Initialize session state
    initialize_session_state()

    # Sidebar
    with st.sidebar:
        st.subheader("🔧 Configuration")
        
        # Health check
        health = _check_model_health()
        status_color = "🟢" if health.get("status") == "ok" else "🔴"
        st.write(f"{status_color} Model API: {health.get('status', 'unknown')}")
        
        # Session info
        st.divider()
        st.subheader("💬 Session")
        st.text_input("Session ID", value=st.session_state.session_id, disabled=True)
        
        if st.button("🔄 New Session"):
            st.session_state.session_id = str(uuid4())
            st.session_state.messages = []
            st.rerun()
        
        # Applicant ID
        st.divider()
        st.subheader("👤 Applicant Context")
        applicant_input = st.number_input(
            "Applicant ID (SK_ID_CURR)",
            min_value=100000,
            max_value=999999,
            value=st.session_state.applicant_id or 100001,
            step=1,
            help="Set the applicant ID to focus the conversation on a specific applicant",
        )
        
        if st.button("Set Applicant"):
            st.session_state.applicant_id = applicant_input
            st.success(f"✅ Focused on applicant {applicant_input}")
        
        if st.button("Clear Applicant"):
            st.session_state.applicant_id = None
            st.info(" Cleared applicant context")

        # Examples
        st.divider()
        st.subheader("💡 Example Queries")
        st.markdown("""
        - "What is the risk score for applicant 100001?"
        - "Show me the top factors for applicant 100002"
        - "Compare income and credit amount for applicant 100003"
        - "What's the average default rate in the portfolio?"
        - "Explain why applicant 100005 is high risk"
        """)

    # Display conversation history
    for msg in st.session_state.messages:
        display_message(msg["role"], msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask about credit risk, applicants, or portfolio analytics..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_message("user", prompt)

        # Prepare request
        request_data = {
            "question": prompt,
            "session_id": st.session_state.session_id,
        }
        
        if st.session_state.applicant_id:
            request_data["applicant_id"] = st.session_state.applicant_id

        # Call chatbot API
        with st.spinner("🤔 Thinking..."):
            try:
                response = requests.post(
                    CHATBOT_URL,
                    json=request_data,
                    timeout=60,
                )
                response.raise_for_status()
                data = response.json()

                # Extract response
                answer = data.get("answer", "I couldn't generate a response.")
                
                # Add assistant message to history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                display_message("assistant", answer)

                # Display additional information
                col1, col2 = st.columns(2)
                
                with col1:
                    # Risk probability
                    if data.get("risk_probability") is not None:
                        risk_prob = data["risk_probability"]
                        risk_pct = risk_prob * 100
                        
                        st.metric(
                            "Default Probability",
                            f"{risk_pct:.2f}%",
                            delta=None,
                            delta_color="inverse",
                        )
                        
                        # Risk indicator
                        if risk_prob > 0.5:
                            st.error("⚠️ High Risk")
                        else:
                            st.success("✅ Low Risk")

                with col2:
                    # Applicant ID
                    if data.get("applicant_id"):
                        st.info(f"👤 Applicant: {data['applicant_id']}")

                # Tool outputs (collapsed)
                if data.get("tool_outputs"):
                    with st.expander("🔧 Tool Calls"):
                        st.json(data["tool_outputs"])

                # Conversation history (collapsed)
                if data.get("conversation_history"):
                    with st.expander("💬 Recent History"):
                        for hist_msg in data["conversation_history"][-6:]:
                            role_icon = "👤" if hist_msg["role"] == "human" else "🤖"
                            st.text(f"{role_icon} {hist_msg['role'].upper()}: {hist_msg['content'][:100]}...")

            except requests.RequestException as exc:
                st.error(f"❌ Chatbot request failed: {exc}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Error: {exc}"
                })


if __name__ == "__main__":
    main()

