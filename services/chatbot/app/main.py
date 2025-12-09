from __future__ import annotations

import base64
import os
from typing import Any, Dict, List, Optional
from uuid import uuid4

import structlog
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import FileResponse
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from .graph import get_chatbot_graph
from .analysis_graph import get_analysis_graph
from .analysis_state import AnalysisState
from .analysis_nodes import encode_image_to_base64

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

# In-memory chart storage (separate from conversation to avoid token bloat)
# Format: {thread_id: {step_number: {"path": str, "base64": str}}}
_analysis_charts: Dict[str, Dict[int, Dict[str, str]]] = {}


def _enrich_step_results_with_charts(
    thread_id: str,
    step_results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Enrich step results with base64-encoded chart images.

    This function adds chart_image_base64 to step results without modifying
    the original state, preventing token bloat in conversation history.

    Args:
        thread_id: The analysis thread ID
        step_results: List of step results from the state

    Returns:
        Enriched step results with base64 chart data
    """
    if not step_results:
        return []

    enriched_results = []

    for step_result in step_results:
        # Create a copy to avoid modifying the original state
        enriched = dict(step_result)
        step_num = step_result.get("step_number")
        chart_path = step_result.get("chart_image_path")

        # Check if we already have this chart cached
        if thread_id in _analysis_charts and step_num in _analysis_charts[thread_id]:
            enriched["chart_image_base64"] = _analysis_charts[thread_id][step_num]["base64"]
        elif chart_path and os.path.exists(chart_path):
            # Encode and cache the chart
            base64_data = encode_image_to_base64(chart_path)
            if base64_data:
                # Cache it for future requests
                if thread_id not in _analysis_charts:
                    _analysis_charts[thread_id] = {}
                _analysis_charts[thread_id][step_num] = {
                    "path": chart_path,
                    "base64": base64_data
                }
                enriched["chart_image_base64"] = base64_data
            else:
                enriched["chart_image_base64"] = None
        else:
            enriched["chart_image_base64"] = None

        enriched_results.append(enriched)

    return enriched_results


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
        "requires_analysis": False,  # Will be determined by router
        "analysis_request": None,
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
        # Handle case where content is a list (Gemini multi-part response)
        answer = last_ai_message.content
        if isinstance(answer, list):
            answer = "\n".join(str(part) for part in answer)
        elif not isinstance(answer, str):
            answer = str(answer)

        # Extract tool outputs - both the call metadata AND the actual results
        tool_outputs = []
        for msg in messages:
            # Check for tool results (ToolMessage in LangChain)
            # ToolMessage has .content with the actual tool output
            msg_type = type(msg).__name__
            if msg_type == "ToolMessage":
                try:
                    # Tool results are often JSON strings - try to parse
                    import json
                    content = msg.content
                    if isinstance(content, str):
                        try:
                            parsed = json.loads(content)
                            tool_outputs.append(parsed)
                        except json.JSONDecodeError:
                            # If not JSON, include as-is
                            tool_outputs.append({"raw_output": content})
                    else:
                        tool_outputs.append(content if isinstance(content, dict) else {"raw_output": str(content)})
                except Exception as e:
                    logger.warning("tool_output_parse_error", error=str(e))
            # Also include tool call metadata for reference
            elif hasattr(msg, "tool_calls") and msg.tool_calls:
                for call in msg.tool_calls:
                    tool_outputs.append({
                        "tool_name": call.get("name"),
                        "tool_args": call.get("args"),
                        "call_id": call.get("id"),
                    })

        # Try to extract risk score from tool outputs or state
        risk_probability = result.get("risk_score")

        # Format conversation history for response
        def normalize_content(content):
            """Normalize content - handle list from Gemini API."""
            if isinstance(content, list):
                return "\n".join(str(part) for part in content)
            elif not isinstance(content, str):
                return str(content) if content else ""
            return content
        
        formatted_history = [
            {"role": "human" if isinstance(msg, HumanMessage) else "ai", "content": normalize_content(msg.content)}
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

        # Enrich step results with base64 chart data (if any exist at this stage)
        step_results = result.get("step_results")
        enriched_step_results = None
        if step_results:
            enriched_step_results = _enrich_step_results_with_charts(thread_id, step_results)

        return AnalysisResponse(
            thread_id=thread_id,
            status=result.get("workflow_status", "planning"),
            plan=result.get("plan"),
            step_results=enriched_step_results,
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

        # Enrich step results with base64 chart data
        step_results = result.get("step_results")
        enriched_step_results = None
        if step_results:
            enriched_step_results = _enrich_step_results_with_charts(thread_id, step_results)

        return AnalysisResponse(
            thread_id=thread_id,
            status=result.get("workflow_status", "executing"),
            plan=result.get("plan"),
            step_results=enriched_step_results,
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

        # Enrich step results with base64 chart data
        step_results = values.get("step_results")
        enriched_step_results = None
        if step_results:
            enriched_step_results = _enrich_step_results_with_charts(thread_id, step_results)

        return AnalysisResponse(
            thread_id=thread_id,
            status=values.get("workflow_status", "unknown"),
            plan=values.get("plan"),
            step_results=enriched_step_results,
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


@app.get("/analysis/{thread_id}/chart/{step_number}")
async def get_analysis_chart(thread_id: str, step_number: int):
    """
    Download the chart image for a specific analysis step.

    Returns the PNG image file directly for download or display.

    Args:
        thread_id: The analysis thread ID
        step_number: The step number (1-indexed)

    Returns:
        PNG image file
    """
    try:
        # Check if we have the chart cached
        if thread_id in _analysis_charts and step_number in _analysis_charts[thread_id]:
            chart_path = _analysis_charts[thread_id][step_number]["path"]
            if os.path.exists(chart_path):
                return FileResponse(
                    chart_path,
                    media_type="image/png",
                    filename=f"analysis_step_{step_number}.png"
                )

        # If not cached, retrieve from state
        graph = get_analysis_graph()
        config = {"configurable": {"thread_id": thread_id}}
        state = graph.get_state(config)

        if not state or not state.values:
            raise HTTPException(
                status_code=404,
                detail=f"Analysis thread {thread_id} not found"
            )

        step_results = state.values.get("step_results", [])

        # Find the specific step
        step_result = next(
            (r for r in step_results if r["step_number"] == step_number),
            None
        )

        if not step_result:
            raise HTTPException(
                status_code=404,
                detail=f"Step {step_number} not found in analysis"
            )

        chart_path = step_result.get("chart_image_path")

        if not chart_path:
            raise HTTPException(
                status_code=404,
                detail=f"No chart available for step {step_number}"
            )

        if not os.path.exists(chart_path):
            raise HTTPException(
                status_code=404,
                detail=f"Chart file not found for step {step_number}"
            )

        logger.info(
            "chart_downloaded",
            thread_id=thread_id,
            step_number=step_number,
            path=chart_path
        )

        return FileResponse(
            chart_path,
            media_type="image/png",
            filename=f"analysis_step_{step_number}.png"
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "chart_download_error",
            error=str(exc),
            thread_id=thread_id,
            step_number=step_number
        )
        raise HTTPException(
            status_code=500,
            detail=f"Error downloading chart: {str(exc)}"
        )


@app.delete("/analysis/{thread_id}/charts")
async def delete_analysis_charts(thread_id: str):
    """
    Delete all cached charts and chart files for a specific analysis thread.

    Use this to clean up resources when an analysis is no longer needed.

    Args:
        thread_id: The analysis thread ID

    Returns:
        Status message with number of charts deleted
    """
    try:
        deleted_count = 0
        deleted_files = []

        # Delete cached base64 data and files
        if thread_id in _analysis_charts:
            for step_num, chart_info in _analysis_charts[thread_id].items():
                chart_path = chart_info.get("path")
                if chart_path and os.path.exists(chart_path):
                    try:
                        os.remove(chart_path)
                        deleted_files.append(chart_path)
                        deleted_count += 1
                    except OSError as e:
                        logger.warning(
                            "chart_file_deletion_failed",
                            path=chart_path,
                            error=str(e)
                        )

            # Remove from cache
            del _analysis_charts[thread_id]

        logger.info(
            "charts_deleted",
            thread_id=thread_id,
            deleted_count=deleted_count
        )

        return {
            "status": "deleted",
            "thread_id": thread_id,
            "charts_deleted": deleted_count,
            "files_deleted": deleted_files
        }

    except Exception as exc:
        logger.error("chart_deletion_error", error=str(exc), thread_id=thread_id)
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting charts: {str(exc)}"
        )


@app.delete("/charts/cleanup")
async def cleanup_old_charts(max_age_hours: int = 24):
    """
    Clean up chart files older than specified age.

    This endpoint removes chart files from the filesystem that are older
    than the specified number of hours. Useful for periodic cleanup.

    Args:
        max_age_hours: Maximum age of charts to keep (default: 24 hours)

    Returns:
        Status message with cleanup statistics
    """
    try:
        import time
        from pathlib import Path

        chart_dir = Path(os.getenv("CHART_OUTPUT_DIR", "/tmp/analysis_charts"))

        if not chart_dir.exists():
            return {
                "status": "no_charts_directory",
                "message": f"Chart directory {chart_dir} does not exist"
            }

        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        deleted_count = 0
        deleted_files = []

        # Iterate through all PNG files in the chart directory
        for chart_file in chart_dir.glob("*.png"):
            try:
                file_age = current_time - chart_file.stat().st_mtime
                if file_age > max_age_seconds:
                    chart_file.unlink()
                    deleted_files.append(str(chart_file))
                    deleted_count += 1
            except Exception as e:
                logger.warning(
                    "cleanup_file_error",
                    file=str(chart_file),
                    error=str(e)
                )

        # Clean up cache entries for deleted files
        threads_to_remove = []
        for thread_id, charts in _analysis_charts.items():
            steps_to_remove = []
            for step_num, chart_info in charts.items():
                if chart_info["path"] in deleted_files:
                    steps_to_remove.append(step_num)

            for step_num in steps_to_remove:
                del charts[step_num]

            if not charts:
                threads_to_remove.append(thread_id)

        for thread_id in threads_to_remove:
            del _analysis_charts[thread_id]

        logger.info(
            "old_charts_cleaned",
            deleted_count=deleted_count,
            max_age_hours=max_age_hours
        )

        return {
            "status": "cleaned",
            "deleted_count": deleted_count,
            "max_age_hours": max_age_hours,
            "chart_directory": str(chart_dir)
        }

    except Exception as exc:
        logger.error("cleanup_error", error=str(exc))
        raise HTTPException(
            status_code=500,
            detail=f"Error during cleanup: {str(exc)}"
        )
