from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
from uuid import uuid4

import structlog
from fastapi import FastAPI, HTTPException
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from .graph import get_chatbot_graph

logger = structlog.get_logger()

app = FastAPI(
    title="HC Risk Chatbot",
    version="0.2.0",
    description="LangGraph-powered conversational credit risk analysis",
)

# In-memory conversation storage (replace with Redis/DB in production)
_conversations: Dict[str, List[Dict[str, Any]]] = {}


class ChatRequest(BaseModel):
    question: str = Field(..., description="Natural language user question.")
    session_id: Optional[str] = Field(
        None,
        description="Session ID for conversation continuity. Auto-generated if not provided.",
    )
    applicant_id: Optional[int] = Field(
        None,
        description="Applicant ID (SK_ID_CURR) to focus the conversation on.",
    )


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    applicant_id: Optional[int] = None
    risk_probability: Optional[float] = None
    tool_outputs: Optional[List[Dict[str, Any]]] = None
    conversation_history: Optional[List[Dict[str, str]]] = None


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "version": "0.2.0",
        "langgraph_enabled": True,
        "gemini_configured": os.getenv("GEMINI_API_KEY", "changeme") != "changeme",
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Chat endpoint with LangGraph integration.

    Supports conversational analysis using Gemini and tool calling for:
    - Risk predictions
    - Applicant data queries
    - Feature visualizations
    """
    # Generate or use existing session ID
    session_id = request.session_id or str(uuid4())

    logger.info(
        "chat_request",
        session=session_id,
        question=request.question,
        has_applicant_id=request.applicant_id is not None,
    )

    # Get conversation history
    conversation_history = _conversations.get(session_id, [])

    # Create human message
    human_message = HumanMessage(content=request.question)

    # Build initial state
    initial_state = {
        "messages": conversation_history + [human_message],
        "session_id": session_id,
        "applicant_id": request.applicant_id,
        "risk_score": None,
        "last_tool_output": None,
    }

    try:
        # Get chatbot graph
        graph = get_chatbot_graph()

        # Invoke the graph
        result = graph.invoke(initial_state)

        # Extract messages
        messages = result["messages"]
        
        # Update conversation history
        _conversations[session_id] = messages

        # Get the last AI message
        ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]
        if not ai_messages:
            raise HTTPException(
                status_code=500,
                detail="No response generated from chatbot"
            )

        last_ai_message = ai_messages[-1]
        answer = last_ai_message.content

        # Extract tool outputs if any
        tool_outputs = []
        for msg in reversed(messages):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_outputs.extend(msg.tool_calls)

        # Try to extract risk score from tool outputs or state
        risk_probability = result.get("risk_score")

        # Format conversation history for response
        formatted_history = [
            {"role": "human" if isinstance(msg, HumanMessage) else "ai", "content": msg.content}
            for msg in messages
        ]

        logger.info(
            "chat_response_generated",
            session=session_id,
            num_messages=len(messages),
            has_tool_calls=len(tool_outputs) > 0,
        )

        return ChatResponse(
            answer=answer,
            session_id=session_id,
            applicant_id=result.get("applicant_id"),
            risk_probability=risk_probability,
            tool_outputs=tool_outputs if tool_outputs else None,
            conversation_history=formatted_history[-6:],  # Last 3 turns
        )

    except Exception as exc:
        logger.error("chat_processing_error", error=str(exc), session=session_id)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat request: {str(exc)}"
        )


@app.delete("/chat/{session_id}")
def clear_conversation(session_id: str):
    """Clear conversation history for a session."""
    if session_id in _conversations:
        del _conversations[session_id]
        return {"status": "cleared", "session_id": session_id}
    return {"status": "not_found", "session_id": session_id}


@app.get("/chat/{session_id}/history")
def get_conversation_history(session_id: str):
    """Get conversation history for a session."""
    if session_id not in _conversations:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = _conversations[session_id]
    formatted = [
        {
            "role": "human" if isinstance(msg, HumanMessage) else "ai",
            "content": msg.content,
            "timestamp": getattr(msg, "timestamp", None),
        }
        for msg in messages
    ]

    return {"session_id": session_id, "messages": formatted}