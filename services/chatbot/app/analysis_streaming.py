"""
Streaming analysis execution for real-time step-by-step results.

This module provides async generators for streaming analysis workflow
execution, allowing step messages to be yielded as they complete.
"""
import os
import base64
from typing import AsyncGenerator
import structlog
from langchain_core.messages import AIMessage
from .analysis_state import AnalysisState
from .analysis_nodes import (
    schema_reader_node,
    planner_node,
    sql_generator_node,
    data_analyzer_node,
    vision_analyzer_node,
    synthesizer_node,
)

logger = structlog.get_logger()


async def execute_analysis_stream(user_request: str) -> AsyncGenerator[AIMessage, None]:
    """
    Execute analysis workflow and yield messages as each step completes.

    This allows real-time streaming of step results instead of waiting
    for the entire analysis to complete.

    Args:
        user_request: User's analysis query

    Yields:
        AIMessage objects with step results as they complete
    """
    # Initialize analysis state
    analysis_state: AnalysisState = {
        "messages": [],
        "user_request": user_request,
        "schema_info": None,
        "plan": None,
        "current_step": 0,
        "step_results": [],
        "final_summary": None,
        "workflow_status": "planning",
    }

    try:
        # Read schema
        analysis_state = {**analysis_state, **schema_reader_node(analysis_state)}

        # Generate plan
        analysis_state = {**analysis_state, **planner_node(analysis_state)}

        # Auto-approve the plan
        if analysis_state.get("plan"):
            analysis_state["plan"]["approved"] = True
            analysis_state["workflow_status"] = "executing"
            analysis_state["current_step"] = 1

        # Execute each step and yield results immediately
        plan = analysis_state.get("plan", {})
        steps = plan.get("steps", [])

        for step_num in range(1, len(steps) + 1):
            analysis_state["current_step"] = step_num

            # SQL generation
            sql_result = sql_generator_node(analysis_state)
            analysis_state = {**analysis_state, **sql_result}

            # Data analysis
            data_result = data_analyzer_node(analysis_state)
            analysis_state = {**analysis_state, **data_result}

            # Vision analysis
            vision_result = vision_analyzer_node(analysis_state)
            analysis_state = {**analysis_state, **vision_result}

            # Get the completed step result
            step_results = analysis_state.get("step_results", [])
            if step_results and len(step_results) >= step_num:
                step_result = step_results[step_num - 1]

                # Get step description from plan
                step_description = "Analysis"
                if steps and step_num <= len(steps):
                    step_description = steps[step_num - 1].get("description", "Analysis")

                # Read chart if available
                chart_path = step_result.get("chart_image_path")
                chart_base64 = None
                if chart_path and os.path.exists(chart_path):
                    with open(chart_path, 'rb') as f:
                        chart_base64 = base64.b64encode(f.read()).decode('utf-8')

                # Create and yield step message immediately
                step_msg = AIMessage(
                    content=f"✓ Completed Step {step_num}",
                    additional_kwargs={
                        "step_result": {
                            "step_number": step_num,
                            "description": step_description,
                            "chart_type": step_result.get("chart_type", ""),
                            "chart_image_base64": chart_base64,
                            "insights": step_result.get("insights", ""),
                        }
                    }
                )

                logger.info("step_completed_yielding", step_num=step_num, has_chart=chart_base64 is not None)

                # YIELD immediately - don't wait for other steps
                yield step_msg

        # Synthesize final summary
        final_result = synthesizer_node(analysis_state)
        analysis_state = {**analysis_state, **final_result}

        summary = analysis_state.get("final_summary", "Analysis completed.")

        logger.info("analysis_stream_completed", num_steps=len(steps))

        # Yield final summary
        yield AIMessage(content=summary)

    except Exception as exc:
        logger.error("analysis_stream_error", error=str(exc))
        yield AIMessage(
            content=f"I encountered an error during analysis: {str(exc)}"
        )
