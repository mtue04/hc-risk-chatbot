from __future__ import annotations

import os
from typing import Dict, Optional

import httpx
import structlog
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


logger = structlog.get_logger()

app = FastAPI(title="HC Risk Chatbot", version="0.1.0")

MODEL_API_URL = os.getenv("FASTAPI_ENDPOINT", "http://model-serving:8000")
MODEL_PREDICT_ROUTE = f"{MODEL_API_URL}/predict"
MODEL_EXPLAIN_ROUTE = f"{MODEL_API_URL}/explain"


class ChatRequest(BaseModel):
    question: str = Field(..., description="Natural language user question.")
    inferred_features: Dict[str, float] = Field(
        default_factory=dict,
        description="Optional feature overrides supplied by the UI or prior tool calls.",
    )
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    risk_probability: Optional[float] = None
    explanation: Optional[Dict[str, float]] = None
    notes: Optional[str] = None


@app.get("/health")
def health_check():
    return {"status": "ok"}


async def _call_model(features: Dict[str, float]) -> Dict[str, float]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            prediction = await client.post(MODEL_PREDICT_ROUTE, json={"features": features})
            prediction.raise_for_status()
            explain = await client.post(MODEL_EXPLAIN_ROUTE, json={"features": features})
            explain.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("model_call_failed", error=str(exc))
            raise HTTPException(status_code=502, detail="Model service unavailable")

    return {
        "probability": prediction.json().get("probability"),
        "explanation": explain.json(),
    }


def _compose_answer(question: str, probability: Optional[float], explanation: Optional[Dict]) -> str:
    if probability is None:
        return (
            "I could not reach the credit risk model right now, but I recorded your question: "
            f"'{question}'. Please retry after the services are healthy."
        )

    base = (
        f"The current estimated default probability is {probability:.2%}. "
        "This score is derived from the engineered features stored in the feature "
        "repository and scored by the credit risk model."
    )

    if explanation and isinstance(explanation, dict) and explanation.get("contributions"):
        top_features = list(explanation["contributions"].items())[:3]
        extra = ", ".join(f"{name} ({value:+.2f})" for name, value in top_features)
        base += f" Key contributors: {extra}."

    return base


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    logger.info("chat_request", session=request.session_id, question=request.question)
    try:
        model_payload = await _call_model(request.inferred_features)
        probability = model_payload.get("probability")
        explanation = model_payload.get("explanation")
    except HTTPException:
        probability = None
        explanation = None

    answer = _compose_answer(request.question, probability, explanation)
    notes = (
        "LLM integration pending. The current response is stitched together without Gemini outputs."
        if os.getenv("GEMINI_API_KEY", "changeme") == "changeme"
        else None
    )

    return ChatResponse(
        answer=answer,
        risk_probability=probability,
        explanation=explanation,
        notes=notes,
    )
