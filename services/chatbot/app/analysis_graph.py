"""
LangGraph workflow for multi-step iterative analysis.

This module implements the complete analysis workflow:
1. Schema Reader -> Planner -> Human Review
2. Executor Loop (SQL -> Data Analysis -> Vision) for each step
3. Synthesizer for final summary
"""

from __future__ import annotations

from typing import Literal

import structlog
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from .analysis_nodes import (
    data_analyzer_node,
    planner_node,
    schema_reader_node,
    sql_generator_node,
    synthesizer_node,
    vision_analyzer_node,
)
from .analysis_state import AnalysisState

logger = structlog.get_logger()


def create_analysis_graph():
    """
    Create the LangGraph workflow for multi-step analysis.

    The workflow follows this pattern:
    1. schema_reader: Reads PostgreSQL schema
    2. planner: Creates structured analysis plan
    3. human_review: Pauses for user approval (via interrupt)
    4. executor_loop: Iterates through each analysis step
       - sql_generator: Generates SQL query
       - data_analyzer: Executes query and creates chart
       - vision_analyzer: Analyzes chart and extracts insights
    5. synthesizer: Creates final executive summary

    Returns:
        Compiled LangGraph workflow with checkpointing enabled
    """

    def should_continue_after_review(state: AnalysisState) -> Literal["execute_step", "replan"]:
        """
        Determine next action after human review.

        Args:
            state: Current analysis state

        Returns:
            "execute_step" if plan approved, "replan" if edits requested
        """
        plan = state.get("plan")

        if not plan:
            logger.error("no_plan_in_review")
            return "execute_step"  # Fallback

        if plan.get("user_edits"):
            # User requested changes - need to replan
            return "replan"

        # Plan approved - proceed to execution
        return "execute_step"

    def should_continue_execution(state: AnalysisState) -> Literal["execute_step", "synthesize"]:
        """
        Determine if more steps need execution or if ready to synthesize.

        Args:
            state: Current analysis state

        Returns:
            "execute_step" to continue loop, "synthesize" when all steps done
        """
        plan = state.get("plan")
        current_step = state.get("current_step", 0)

        if not plan:
            return "synthesize"

        total_steps = len(plan.get("steps", []))

        if current_step >= total_steps:
            # All steps completed
            return "synthesize"

        # More steps to execute
        return "execute_step"

    def increment_step(state: AnalysisState) -> dict:
        """
        Increment the current step counter.

        This is a simple node to advance the step counter in the loop.

        Args:
            state: Current analysis state

        Returns:
            Updated state with incremented current_step
        """
        current = state.get("current_step", 0)
        return {"current_step": current + 1}

    def human_review_node(state: AnalysisState) -> dict:
        """
        Human review checkpoint.

        This node serves as an interrupt point where the workflow pauses
        for the user to review and approve/edit the analysis plan.

        The user can:
        - Approve the plan (set plan.approved = True)
        - Request edits (set plan.user_edits with modifications)

        Args:
            state: Current analysis state with plan

        Returns:
            State unchanged (plan modifications happen via external update)
        """
        logger.info("human_review_checkpoint_reached")

        # This is just a checkpoint - the actual review happens
        # via external state updates when the graph is interrupted

        # If we reach this node and plan is not approved, we wait
        plan = state.get("plan", {})

        if not plan.get("approved"):
            # Set a flag indicating we're waiting for review
            return {"workflow_status": "awaiting_approval"}

        # Plan is approved, proceed
        return {"workflow_status": "executing", "current_step": 1}

    # Build the workflow graph
    workflow = StateGraph(AnalysisState)

    # Add all nodes
    workflow.add_node("schema_reader", schema_reader_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("sql_generator", sql_generator_node)
    workflow.add_node("data_analyzer", data_analyzer_node)
    workflow.add_node("vision_analyzer", vision_analyzer_node)
    workflow.add_node("increment_step", increment_step)
    workflow.add_node("synthesizer", synthesizer_node)

    # Define the workflow edges
    workflow.add_edge(START, "schema_reader")
    workflow.add_edge("schema_reader", "planner")
    workflow.add_edge("planner", "human_review")

    # After human review - conditional routing
    workflow.add_conditional_edges(
        "human_review",
        should_continue_after_review,
        {
            "execute_step": "sql_generator",
            "replan": "planner",  # Loop back to planner if edits requested
        },
    )

    # Execution loop for each step
    workflow.add_edge("sql_generator", "data_analyzer")
    workflow.add_edge("data_analyzer", "vision_analyzer")
    workflow.add_edge("vision_analyzer", "increment_step")

    # After incrementing step, check if more steps or synthesize
    workflow.add_conditional_edges(
        "increment_step",
        should_continue_execution,
        {
            "execute_step": "sql_generator",
            "synthesize": "synthesizer",
        },
    )

    # Final node leads to END
    workflow.add_edge("synthesizer", END)

    # Compile with checkpointing for human-in-the-loop
    memory = MemorySaver()
    app = workflow.compile(
        checkpointer=memory,
        interrupt_before=["human_review"],  # Pause before human review
    )

    return app


# Singleton instance
_analysis_graph = None


def get_analysis_graph():
    """Get or create the analysis graph instance."""
    global _analysis_graph
    if _analysis_graph is None:
        _analysis_graph = create_analysis_graph()
    return _analysis_graph
