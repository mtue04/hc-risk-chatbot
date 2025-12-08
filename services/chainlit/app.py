import ast
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from uuid import uuid4

import chainlit as cl
import httpx
import plotly.graph_objects as go

# docker compose build chainlit && docker compose up -d chainlit

CHATBOT_API_URL = os.getenv("CHATBOT_API_URL", "http://chatbot:8500")

EXAMPLE_QUESTIONS = [
    "What is the risk score for applicant 100001?",
    "Explain the top risk factors for applicant 100002.",
    "Compare income and credit between applicant 100002 and 100003.",
    "Show applicant 100001's income vs population average.",
    "Plot EXT_SOURCE_2 and EXT_SOURCE_3 distribution for applicant 100004.",
    "Display credit-to-income comparison chart for applicant 100003.",
]


def _maybe_parse_json(text: str) -> Optional[Any]:
    """Attempt to parse a string into JSON/dict."""
    if not text:
        return None
    stripped = text.strip()
    if not stripped:
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(stripped)
            if isinstance(parsed, (dict, list)):
                return parsed
        except Exception:
            continue
    return None


def _build_feature_comparison_chart(payload: Dict[str, Any]) -> Optional[cl.Plotly]:
    features = payload.get("features")
    if not isinstance(features, dict):
        return None

    categories: List[str] = []
    applicant_vals: List[float] = []
    population_vals: List[float] = []

    for feature_name, stats in features.items():
        if not isinstance(stats, dict):
            continue
        applicant_value = stats.get("applicant_value")
        population_mean = stats.get("population_mean")
        if applicant_value is None and population_mean is None:
            continue

        categories.append(feature_name)
        applicant_vals.append(applicant_value)
        population_vals.append(population_mean)

    if not categories:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Applicant",
            x=categories,
            y=applicant_vals,
            marker_color="#E31E24",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Population Avg",
            x=categories,
            y=population_vals,
            marker_color="#4A90E2",
        )
    )
    fig.update_layout(
        title="Applicant vs Population Benchmark",
        barmode="group",
        template="plotly_dark",
        margin=dict(t=60, b=60, l=40, r=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    display_name = f"feature_comparison_{uuid4().hex[:6]}"
    return cl.Plotly(name=display_name, figure=fig)


def _chart_elements_from_tool_messages(tool_messages: Optional[List[Dict[str, Any]]]) -> List[cl.Plotly]:
    if not tool_messages:
        return []

    elements: List[cl.Plotly] = []
    for tool_msg in tool_messages:
        payload = tool_msg.get("parsed_output")
        if not isinstance(payload, dict):
            raw_text = tool_msg.get("raw_text")
            if isinstance(raw_text, str):
                parsed = _maybe_parse_json(raw_text)
                if isinstance(parsed, dict):
                    payload = parsed
        if not isinstance(payload, dict):
            continue

        plot_type = payload.get("plot_type")
        if plot_type == "feature_comparison":
            chart = _build_feature_comparison_chart(payload)
            if chart:
                elements.append(chart)

    return elements


def _format_history(messages: List[Dict[str, Any]]) -> str:
    # if not messages:
    #     return "No previous conversation turns recorded for this session."

    lines: List[str] = []
    for entry in messages:
        role = entry.get("role", "ai")
        role_label = "User" if role == "human" else "Assistant"
        content = entry.get("content") or ""
        lines.append(f"- **{role_label}:** {content}")
    return "\n".join(lines)


async def refresh_history_in_sidebar(session_id: str) -> None:
    history = await fetch_history(session_id)
    history_text = _format_history(history or [])
    history_msg: Optional[cl.Message] = cl.user_session.get("sidebar_history_msg")

    if history_msg:
        await history_msg.remove()

    history_msg = await cl.Message(
        author="History",
        content=history_text,
    ).send()
    cl.user_session.set("sidebar_history_msg", history_msg)


async def call_chatbot(question: str, session_id: str) -> Dict[str, Any]:
    payload = {"question": question, "session_id": session_id}
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(f"{CHATBOT_API_URL}/chat", json=payload)
        response.raise_for_status()
        return response.json()


async def call_chatbot_multimodal(
    question: str, session_id: str, attachment: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Send multimodal payload (text + file) to the chatbot API using multipart/form-data.
    """
    data = {"question": question or "", "session_id": session_id}
    field = "audio" if attachment["kind"] == "audio" else "image"
    files = {
        field: (attachment["name"], attachment["data"], attachment["mime"])
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{CHATBOT_API_URL}/chat/multimodal",
            data=data,
            files=files,
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        return response.json()


async def fetch_history(session_id: str) -> Optional[List[Dict[str, Any]]]:
    if not session_id:
        return None
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            f"{CHATBOT_API_URL}/chat/{session_id}/history",
            headers={"Accept": "application/json"},
        )
    if response.status_code != 200:
        return None
    data = response.json()
    return data.get("messages")


def _read_element_bytes(element) -> Optional[bytes]:
    if getattr(element, "path", None):
        try:
            return Path(element.path).read_bytes()
        except OSError:
            return None
    content = getattr(element, "content", None)
    if isinstance(content, bytes):
        return content
    return None


async def extract_attachment(message: cl.Message) -> Optional[Dict[str, Any]]:
    if not message.elements:
        return None

    for element in message.elements:
        data = _read_element_bytes(element)
        if not data:
            continue

        mime = getattr(element, "mime", None) or "application/octet-stream"
        name = getattr(element, "name", None) or f"attachment_{uuid4().hex}"
        kind = "audio" if mime.startswith("audio/") else "image"
        return {"name": name, "data": data, "mime": mime, "kind": kind}

    return None


async def process_question(
    question: str,
    session_id: str,
    attachment: Optional[Dict[str, Any]] = None,
) -> None:
    try:
        if attachment:
            api_response = await call_chatbot_multimodal(
                question or "(See attachment)", session_id, attachment
            )
        else:
            api_response = await call_chatbot(question or "", session_id)
        answer = api_response.get("answer", "No response from chatbot.")
        chart_elements = _chart_elements_from_tool_messages(
            api_response.get("tool_messages")
        )
    except Exception as exc:
        answer = f":warning: Unable to reach chatbot API ({exc})."
        chart_elements = []

    await cl.Message(
        author="Assistant",
        content=answer,
        elements=chart_elements or None,
    ).send()


@cl.on_chat_start
async def start_chat():
    session_id = str(uuid4())
    cl.user_session.set("session_id", session_id)
    # await refresh_history_in_sidebar(session_id)
    examples_list = "\n".join(f"- {question}" for question in EXAMPLE_QUESTIONS)
    example_actions = [
        cl.Action(
            name="example_prompt",
            label=question,
            payload={"prompt": question},
        )
        for question in EXAMPLE_QUESTIONS
    ]

    await cl.Message(
        content=(
            "Welcome to the Home Credit Risk assistant\n"
            "You can ask about applicant risk scores, feature importance, or "
            "portfolio insights.\n\n"
        ),
        actions=example_actions,
    ).send()


@cl.on_message
async def handle_message(message: cl.Message):
    session_id = cl.user_session.get("session_id") or str(uuid4())
    cl.user_session.set("session_id", session_id)

    attachment = await extract_attachment(message)
    await process_question(message.content or "", session_id, attachment)


@cl.action_callback("example_prompt")
async def handle_example_prompt(action: cl.Action):
    session_id = cl.user_session.get("session_id") or str(uuid4())
    cl.user_session.set("session_id", session_id)
    payload = action.payload or {}
    prompt = payload.get("prompt") or action.label or ""
    if not prompt:
        return

    await cl.Message(author="You", content=prompt or "(No text provided)").send()
    await process_question(prompt, session_id)
