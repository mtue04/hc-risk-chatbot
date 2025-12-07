from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List, Optional
from uuid import uuid4

import httpx
import structlog
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

# Import graph
from .graph import get_chatbot_graph

logger = structlog.get_logger()

# Configuration
USE_ENHANCED_GRAPH = os.getenv("USE_ENHANCED_GRAPH", "true").lower() == "true"
ENABLE_CHECKPOINTING = os.getenv("ENABLE_CHECKPOINTING", "true").lower() == "true"

app = FastAPI(
    title="HC Risk Chatbot",
    version="0.3.0",  # Bumped version for enhanced features
    description="LangGraph-powered conversational credit risk analysis with enhanced tools and persistence",
)

# In-memory conversation storage (used only when checkpointing is disabled)
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
    langsmith_key = os.getenv("LANGSMITH_API_KEY", "")
    return {
        "status": "ok",
        "version": "0.3.0",
        "langgraph_enabled": True,
        "enhanced_features": USE_ENHANCED_GRAPH,
        "checkpointing_enabled": ENABLE_CHECKPOINTING,
        "langsmith_tracing": langsmith_key != "" and langsmith_key != "changeme",
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
        # Initialize enhanced state fields if needed
        if USE_ENHANCED_GRAPH:
            initial_state["conversation_summary"] = None
            initial_state["mentioned_applicants"] = []
            initial_state["key_insights"] = []
            initial_state["turn_count"] = 0

        # Get chatbot graph
        graph = get_chatbot_graph()

        # Invoke the graph
        if ENABLE_CHECKPOINTING:
            # With checkpointing, we need to provide a config with thread_id
            config = {"configurable": {"thread_id": session_id}}
            result = graph.invoke(initial_state, config=config)
        else:
            result = graph.invoke(initial_state)

        # Extract messages
        messages = result["messages"]

        # Update conversation history (only if not using checkpointing)
        if not ENABLE_CHECKPOINTING:
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


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint with real-time updates.

    Returns Server-Sent Events (SSE) stream with:
    - Agent thinking events
    - Tool execution events
    - Token-by-token response
    """
    # Generate or use existing session ID
    session_id = request.session_id or str(uuid4())

    logger.info(
        "chat_stream_request",
        session=session_id,
        question=request.question,
    )

    async def generate():
        try:
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

            # Initialize enhanced state fields if needed
            if USE_ENHANCED_GRAPH:
                initial_state["conversation_summary"] = None
                initial_state["mentioned_applicants"] = []
                initial_state["key_insights"] = []
                initial_state["turn_count"] = 0

            # Get chatbot graph
            graph = get_chatbot_graph()

            # Configure for streaming
            config = {"configurable": {"thread_id": session_id}} if ENABLE_CHECKPOINTING else {}

            # Stream events from graph
            async for event in graph.astream(initial_state, config=config):
                # Agent node - send thinking status
                if "agent" in event:
                    yield f"data: {json.dumps({'type': 'agent', 'status': 'thinking'})}\n\n"

                # Tools node - send tool execution status
                elif "tools" in event:
                    tool_name = "unknown"
                    if event["tools"].get("messages"):
                        for msg in event["tools"]["messages"]:
                            if hasattr(msg, "name"):
                                tool_name = msg.name
                                break

                    yield f"data: {json.dumps({'type': 'tool', 'name': tool_name, 'status': 'executing'})}\n\n"

                # Extract insights node
                elif "extract_insights" in event:
                    yield f"data: {json.dumps({'type': 'insight', 'status': 'extracting'})}\n\n"

                # Summarization node
                elif "summarize" in event:
                    yield f"data: {json.dumps({'type': 'summarize', 'status': 'compressing_history'})}\n\n"

            # Get final result
            if ENABLE_CHECKPOINTING:
                config = {"configurable": {"thread_id": session_id}}
                result = graph.invoke(initial_state, config=config)
            else:
                result = graph.invoke(initial_state)

            # Extract final message
            messages = result["messages"]
            ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]

            if ai_messages:
                final_answer = ai_messages[-1].content

                # Stream final answer token by token
                words = final_answer.split()
                for i, word in enumerate(words):
                    token = word + (" " if i < len(words) - 1 else "")
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

                # Send completion event
                yield f"data: {json.dumps({'type': 'done', 'session_id': session_id})}\n\n"

            # Update conversation history
            if not ENABLE_CHECKPOINTING:
                _conversations[session_id] = messages

            logger.info("chat_stream_completed", session=session_id)

        except Exception as exc:
            logger.error("chat_stream_error", error=str(exc), session=session_id)
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


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


# ============================================================================
# Batch Processing Endpoints
# ============================================================================

class BatchAnalysisRequest(BaseModel):
    applicant_ids: List[int] = Field(..., description="List of applicant IDs to analyze", min_items=1, max_items=100)
    include_comparisons: bool = Field(default=False, description="Include pairwise comparisons")
    risk_threshold: float = Field(default=0.5, description="Risk threshold for classification")


class ApplicantAnalysis(BaseModel):
    applicant_id: int
    risk_probability: float
    risk_category: str
    top_risk_factor: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class BatchAnalysisResponse(BaseModel):
    total_analyzed: int
    successful: int
    failed: int
    analyses: List[ApplicantAnalysis]
    summary: Dict[str, Any]


@app.post("/analyze/batch", response_model=BatchAnalysisResponse)
async def analyze_batch(request: BatchAnalysisRequest):
    """
    Batch analyze multiple applicants in parallel.

    Useful for portfolio analysis, risk screening, or comparing multiple applicants.
    Maximum 100 applicants per request.
    """
    logger.info("batch_analysis_request", count=len(request.applicant_ids))

    # Model API URL
    MODEL_API_URL = os.getenv("MODEL_API_URL", "http://model_serving:8000")

    async def analyze_single(applicant_id: int) -> ApplicantAnalysis:
        """Analyze a single applicant"""
        try:
            # Call model API
            async with httpx.AsyncClient(timeout=10.0) as client:
                pred_response = await client.get(f"{MODEL_API_URL}/predict/applicant/{applicant_id}")
                pred_response.raise_for_status()
                prediction = pred_response.json()

            probability = prediction.get("probability", 0.0)
            category = "High Risk" if probability > request.risk_threshold else "Low Risk"

            # Get top risk factor from SHAP
            top_factor = None
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    explain_response = await client.get(f"{MODEL_API_URL}/explain/applicant/{applicant_id}")
                    explain_response.raise_for_status()
                    explanation = explain_response.json()

                contributions = explanation.get("contributions", {})
                if contributions:
                    top_feature, top_value = max(contributions.items(), key=lambda x: abs(x[1]))
                    top_factor = {"feature": top_feature, "impact": top_value}

            except Exception:
                pass  # SHAP explanation is optional

            return ApplicantAnalysis(
                applicant_id=applicant_id,
                risk_probability=probability,
                risk_category=category,
                top_risk_factor=top_factor,
            )

        except Exception as exc:
            logger.error("batch_analysis_error", applicant_id=applicant_id, error=str(exc))
            return ApplicantAnalysis(
                applicant_id=applicant_id,
                risk_probability=0.0,
                risk_category="Error",
                error=str(exc),
            )

    # Analyze all applicants in parallel
    analyses = await asyncio.gather(*[
        analyze_single(app_id) for app_id in request.applicant_ids
    ])

    # Compute summary statistics
    successful = [a for a in analyses if a.error is None]
    failed = [a for a in analyses if a.error is not None]

    probabilities = [a.risk_probability for a in successful]
    high_risk_count = sum(1 for a in successful if a.risk_category == "High Risk")

    summary = {
        "avg_risk": sum(probabilities) / len(probabilities) if probabilities else 0.0,
        "max_risk": max(probabilities) if probabilities else 0.0,
        "min_risk": min(probabilities) if probabilities else 0.0,
        "high_risk_count": high_risk_count,
        "high_risk_percentage": (high_risk_count / len(successful) * 100) if successful else 0.0,
        "risk_threshold": request.risk_threshold,
    }

    # Add top 10 riskiest if requested
    if successful:
        top_10_risky = sorted(successful, key=lambda x: -x.risk_probability)[:10]
        summary["top_10_riskiest"] = [
            {"applicant_id": a.applicant_id, "probability": a.risk_probability}
            for a in top_10_risky
        ]

    logger.info(
        "batch_analysis_completed",
        total=len(analyses),
        successful=len(successful),
        failed=len(failed),
    )

    return BatchAnalysisResponse(
        total_analyzed=len(analyses),
        successful=len(successful),
        failed=len(failed),
        analyses=analyses,
        summary=summary,
    )


@app.post("/analyze/portfolio")
async def analyze_portfolio(request: BatchAnalysisRequest):
    """
    Analyze portfolio and return aggregated statistics only (no individual results).

    Faster than /analyze/batch when you only need summary statistics.
    """
    logger.info("portfolio_analysis_request", count=len(request.applicant_ids))

    MODEL_API_URL = os.getenv("MODEL_API_URL", "http://model_serving:8000")

    async def get_probability(applicant_id: int) -> Optional[float]:
        """Get just the probability for an applicant"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{MODEL_API_URL}/predict/applicant/{applicant_id}")
                response.raise_for_status()
                return response.json().get("probability", 0.0)
        except Exception as exc:
            logger.error("portfolio_prediction_error", applicant_id=applicant_id, error=str(exc))
            return None

    # Get all probabilities in parallel
    probabilities = await asyncio.gather(*[
        get_probability(app_id) for app_id in request.applicant_ids
    ])

    # Filter out failures
    valid_probs = [p for p in probabilities if p is not None]

    if not valid_probs:
        raise HTTPException(status_code=500, detail="Failed to analyze any applicants")

    # Compute statistics
    high_risk_count = sum(1 for p in valid_probs if p > request.risk_threshold)

    # Risk distribution buckets
    buckets = {
        "very_low": sum(1 for p in valid_probs if p < 0.2),
        "low": sum(1 for p in valid_probs if 0.2 <= p < 0.4),
        "medium": sum(1 for p in valid_probs if 0.4 <= p < 0.6),
        "high": sum(1 for p in valid_probs if 0.6 <= p < 0.8),
        "very_high": sum(1 for p in valid_probs if p >= 0.8),
    }

    logger.info("portfolio_analysis_completed", total=len(valid_probs))

    return {
        "total_applicants": len(request.applicant_ids),
        "analyzed": len(valid_probs),
        "failed": len(request.applicant_ids) - len(valid_probs),
        "statistics": {
            "avg_risk": sum(valid_probs) / len(valid_probs),
            "max_risk": max(valid_probs),
            "min_risk": min(valid_probs),
            "median_risk": sorted(valid_probs)[len(valid_probs) // 2],
            "std_dev": (sum((p - sum(valid_probs)/len(valid_probs))**2 for p in valid_probs) / len(valid_probs)) ** 0.5,
        },
        "classification": {
            "high_risk_count": high_risk_count,
            "low_risk_count": len(valid_probs) - high_risk_count,
            "high_risk_percentage": (high_risk_count / len(valid_probs) * 100),
            "threshold": request.risk_threshold,
        },
        "distribution": buckets,
    }