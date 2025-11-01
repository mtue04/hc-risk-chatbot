import json
import os
from typing import Dict

import requests
import streamlit as st


st.set_page_config(page_title="HC Risk Chatbot", layout="wide")

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


def main():
    st.title("HC Risk Chatbot")
    st.caption("Conversational credit risk analytics (prototype scaffold).")

    health = _check_model_health()
    st.sidebar.subheader("Service Health")
    st.sidebar.write(f"Model API: {health.get('status', 'unknown')}")

    query = st.text_area("Ask a question about portfolio risk or a specific applicant:", height=150)
    feature_input = st.text_area(
        "Optional feature overrides (JSON map)",
        help="Provide feature_name:value pairs for model inference. Example: {\"loan_amount\": 5000}",
    )

    if st.button("Send"):
        if not query.strip():
            st.warning("Please enter a question first.")
            st.stop()

        features: Dict[str, float] = {}
        if feature_input.strip():
            try:
                overrides = json.loads(feature_input)
                if not isinstance(overrides, dict):
                    raise ValueError("Expected a JSON object.")
                features = st.session_state.get("feature_overrides_cache") or {}
                features.update(overrides)
                st.session_state["feature_overrides_cache"] = features
            except Exception as exc:
                st.error(f"Could not parse features: {exc}")
                st.stop()

        try:
            response = requests.post(
                CHATBOT_URL,
                json={"question": query, "inferred_features": features},
                timeout=30,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            st.error(f"Chatbot request failed: {exc}")
            st.stop()

        payload = response.json()
        st.subheader("Chatbot Response")
        st.write(payload.get("answer"))

        if payload.get("risk_probability") is not None:
            st.metric("Default Probability", f"{payload['risk_probability']:.2%}")

        if payload.get("explanation"):
            st.write("Explanation")
            st.json(payload["explanation"])

        if payload.get("notes"):
            st.info(payload["notes"])


if __name__ == "__main__":
    main()
