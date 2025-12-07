from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
from uuid import uuid4

import structlog
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from .graph import get_chatbot_graph
from .analysis_graph import get_analysis_graph
from .analysis_state import AnalysisState

try:
    from .multimodal import get_multimodal_processor
    MULTIMODAL_AVAILABLE = True
except ImportError:
    MULTIMODAL_AVAILABLE = False

logger = structlog.get_logger()

app = FastAPI(
    title="HC Risk Chatbot",
    version="0.3.0",
    description="LangGraph-powered conversational credit risk analysis with multi-step iterative analysis",
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
        "version": "0.3.0",
        "langgraph_enabled": True,
        "analysis_workflow_enabled": True,
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


@app.post("/chat/multimodal")
async def chat_multimodal(
    question: str = Form(None),
    audio: UploadFile = File(None),
    image: UploadFile = File(None),
    session_id: str = Form(None),
    applicant_id: int = Form(None),
):
    """
    Multimodal chat endpoint supporting voice and image input.

    Accepts:
    - Text question (form field)
    - Audio file for voice transcription (WAV, MP3, OGG, WEBM)
    - Image file for document extraction (PNG, JPEG)
    - Session ID for conversation continuity
    - Applicant ID for focused analysis

    Returns standard chat response with processed multimodal context.
    """
    if not MULTIMODAL_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Multimodal processing not available. Install required dependencies."
        )

    processor = get_multimodal_processor()
    final_question = question or ""
    extracted_context = None

    # Process audio if provided
    if audio:
        try:
            audio_bytes = await audio.read()
            transcription_result = await processor.transcribe_audio(audio_bytes)

            if "error" in transcription_result:
                logger.warning(f"Audio transcription failed: {transcription_result['error']}")
            else:
                transcribed_text = transcription_result.get("text", "")
                if transcribed_text:
                    final_question = transcribed_text
                    logger.info(f"Audio transcribed: {len(transcribed_text)} chars")
        except Exception as e:
            logger.error(f"Audio processing error: {e}")

    # Process document/image if provided
    if image:
        try:
            file_bytes = await image.read()
            filename = image.filename or "document"

            extraction_result = await processor.extract_text_from_file(file_bytes, filename)

            if "error" in extraction_result:
                logger.warning(f"Document extraction failed: {extraction_result['error']}")
            else:
                extracted_context = extraction_result
                logger.info(f"Document processed: {extraction_result.get('file_type', 'unknown')}")
        except Exception as e:
            logger.error(f"Document processing error: {e}")

    # Build question with context if document was processed
    if extracted_context:
        # Handle different return formats from multimodal processor
        if "text" in extracted_context and extracted_context["text"]:
            context_str = extracted_context["text"][:2000]
        elif "data" in extracted_context:
            context_str = str(extracted_context["data"])
        elif "raw_text" in extracted_context and extracted_context["raw_text"]:
            context_str = extracted_context["raw_text"][:2000]
        elif "raw_response" in extracted_context:
            context_str = extracted_context["raw_response"][:2000]
        else:
            context_str = str(extracted_context)

        if final_question:
            final_question = f"{final_question}\n\n[Document content: {context_str}]"
        else:
            final_question = f"Analyze this document content and respond to any questions about it:\n\n{context_str}"

        logger.info(f"Document context added: {len(context_str)} chars")

    if not final_question:
        raise HTTPException(
            status_code=400,
            detail="No question provided. Send text, audio, or image input."
        )

    # Forward to regular chat endpoint
    chat_request = ChatRequest(
        question=final_question,
        session_id=session_id,
        applicant_id=applicant_id,
    )

    response = await chat(chat_request)

    # Add multimodal metadata to response
    return {
        **response.model_dump(),
        "multimodal_info": {
            "audio_processed": audio is not None,
            "image_processed": image is not None,
            "extracted_context": extracted_context,
        }
    }


# ============================================================================
# Multi-Step Analysis Endpoints
# ============================================================================


class AnalysisRequest(BaseModel):
    """Request to start a new multi-step analysis."""

    user_request: str = Field(
        ...,
        description="Natural language description of the analysis to perform",
        examples=["Analyze monthly revenue trends", "Compare user segments by age"],
    )
    thread_id: Optional[str] = Field(
        None,
        description="Thread ID for resuming an existing analysis. Auto-generated if not provided.",
    )


class AnalysisResponse(BaseModel):
    """Response from analysis workflow."""

    thread_id: str
    status: str
    plan: Optional[Dict[str, Any]] = None
    step_results: Optional[List[Dict[str, Any]]] = None
    final_summary: Optional[str] = None
    current_step: Optional[int] = None
    message: Optional[str] = None


class PlanApprovalRequest(BaseModel):
    """Request to approve or edit an analysis plan."""

    approved: bool = Field(..., description="Whether to approve the plan")
    edits: Optional[str] = Field(
        None,
        description="Requested modifications to the plan (if not approved)",
    )


@app.post("/analysis/start", response_model=AnalysisResponse)
async def start_analysis(request: AnalysisRequest) -> AnalysisResponse:
    """
    Start a new multi-step analysis workflow.

    This endpoint:
    1. Reads the database schema
    2. Generates an analysis plan using LLM
    3. Returns the plan for user review

    The workflow will pause at the human review step, waiting for approval.
    """
    thread_id = request.thread_id or str(uuid4())

    logger.info(
        "analysis_start_requested",
        thread_id=thread_id,
        request=request.user_request,
    )

    # Initialize analysis state
    initial_state: AnalysisState = {
        "messages": [],
        "user_request": request.user_request,
        "schema_info": None,
        "plan": None,
        "current_step": 0,
        "step_results": [],
        "final_summary": None,
        "workflow_status": "planning",
    }

    try:
        # Get analysis graph
        graph = get_analysis_graph()

        # Configure with thread ID for checkpointing
        config = {"configurable": {"thread_id": thread_id}}

        # Invoke the graph - it will pause at human_review
        result = graph.invoke(initial_state, config)

        logger.info(
            "analysis_plan_generated",
            thread_id=thread_id,
            status=result.get("workflow_status"),
        )

        return AnalysisResponse(
            thread_id=thread_id,
            status=result.get("workflow_status", "planning"),
            plan=result.get("plan"),
            message="Analysis plan generated. Please review and approve to proceed.",
        )

    except Exception as exc:
        logger.error("analysis_start_error", error=str(exc), thread_id=thread_id)
        raise HTTPException(
            status_code=500,
            detail=f"Error starting analysis: {str(exc)}",
        )


@app.post("/analysis/{thread_id}/approve", response_model=AnalysisResponse)
async def approve_analysis_plan(
    thread_id: str,
    approval: PlanApprovalRequest,
) -> AnalysisResponse:
    """
    Approve or request edits to the analysis plan.

    If approved, the workflow will continue with execution.
    If edits are requested, the workflow will regenerate the plan.
    """
    logger.info(
        "plan_approval_received",
        thread_id=thread_id,
        approved=approval.approved,
        has_edits=approval.edits is not None,
    )

    try:
        # Get the current state from the graph
        graph = get_analysis_graph()
        config = {"configurable": {"thread_id": thread_id}}

        # Get current state
        current_state = graph.get_state(config)

        if not current_state or not current_state.values:
            raise HTTPException(
                status_code=404,
                detail=f"Analysis thread {thread_id} not found",
            )

        state_values = current_state.values

        # Update the plan with approval/edits
        plan = state_values.get("plan", {})
        if not plan:
            raise HTTPException(
                status_code=400,
                detail="No plan found to approve",
            )

        plan["approved"] = approval.approved
        plan["user_edits"] = approval.edits

        # Update state - the graph will use the modified plan
        graph.update_state(config, {"plan": plan})

        # Resume the workflow from the interrupt
        result = graph.invoke(None, config)  # Resume with existing state

        logger.info(
            "analysis_resumed",
            thread_id=thread_id,
            status=result.get("workflow_status"),
        )

        return AnalysisResponse(
            thread_id=thread_id,
            status=result.get("workflow_status", "executing"),
            plan=result.get("plan"),
            step_results=result.get("step_results"),
            final_summary=result.get("final_summary"),
            current_step=result.get("current_step"),
            message="Analysis workflow resumed.",
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("plan_approval_error", error=str(exc), thread_id=thread_id)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing plan approval: {str(exc)}",
        )


@app.get("/analysis/{thread_id}/status", response_model=AnalysisResponse)
async def get_analysis_status(thread_id: str) -> AnalysisResponse:
    """
    Get the current status of an analysis workflow.

    Returns the current state including:
    - Workflow status
    - Analysis plan
    - Completed step results
    - Final summary (if completed)
    """
    try:
        graph = get_analysis_graph()
        config = {"configurable": {"thread_id": thread_id}}

        # Get current state
        state = graph.get_state(config)

        if not state or not state.values:
            raise HTTPException(
                status_code=404,
                detail=f"Analysis thread {thread_id} not found",
            )

        values = state.values

        return AnalysisResponse(
            thread_id=thread_id,
            status=values.get("workflow_status", "unknown"),
            plan=values.get("plan"),
            step_results=values.get("step_results"),
            final_summary=values.get("final_summary"),
            current_step=values.get("current_step"),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("status_check_error", error=str(exc), thread_id=thread_id)
        raise HTTPException(
            status_code=500,
            detail=f"Error checking analysis status: {str(exc)}",
        )
