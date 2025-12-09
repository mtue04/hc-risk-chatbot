"""
LangGraph state machine for HomeCredit Risk Chatbot.

This module implements a conversational AI workflow using LangGraph and Gemini
to provide intelligent credit risk analysis through natural dialogue.
"""

from __future__ import annotations

import os
from typing import Annotated, Literal, TypedDict

import structlog
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .tools import get_risk_prediction, query_applicant_data, generate_feature_plot, predict_hypothetical_applicant, explain_shap_values
try:
    from .chart_tools import analyze_and_visualize, generate_data_report
    CHART_TOOLS_AVAILABLE = True
except ImportError:
    CHART_TOOLS_AVAILABLE = False

logger = structlog.get_logger()


class ChatState(TypedDict):
    """State of the conversation."""

    messages: Annotated[list[BaseMessage], add_messages]
    """Conversation history."""

    session_id: str | None
    """Session identifier for tracking conversations."""

    applicant_id: int | None
    """Current applicant being analyzed (SK_ID_CURR)."""

    risk_score: float | None
    """Most recent risk prediction."""

    last_tool_output: dict | None
    """Output from the most recent tool call."""

    requires_analysis: bool
    """Whether the query requires multi-step data analysis."""

    analysis_request: str | None
    """The extracted analysis request if multi-step analysis is needed."""


def create_chatbot_graph():
    """
    Create the LangGraph state machine for the chatbot.

    Returns:
        Compiled LangGraph workflow
    """
    # Initialize Gemini LLM
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # flash has higher rate limits
    if not gemini_api_key or gemini_api_key == "changeme":
        logger.warning("GEMINI_API_KEY not configured, using mock responses")
        llm = None
    else:
        logger.info(f"Initializing Gemini LLM with model: {gemini_model}")
        llm = ChatGoogleGenerativeAI(
            model=gemini_model,
            google_api_key=gemini_api_key,
            temperature=0.7,
            convert_system_message_to_human=True,
        )

    # Define tools available to the agent
    tools = [
        get_risk_prediction,
        query_applicant_data,
        generate_feature_plot,
        predict_hypothetical_applicant,
        explain_shap_values,  # New SHAP explanation tool
    ]

    # Add chart tools if available
    if CHART_TOOLS_AVAILABLE:
        tools.extend([analyze_and_visualize, generate_data_report])
        logger.info("Chart analysis tools loaded")

    # Bind tools to LLM
    if llm:
        llm_with_tools = llm.bind_tools(tools)
    else:
        llm_with_tools = None

    # System prompt
    SYSTEM_PROMPT = """You are a senior credit risk data analyst at Home Credit with expertise in statistical analysis and data visualization.

Your role is to help users understand credit risk predictions through data-driven insights and appropriate visualizations.

## Available Tools:
1. **get_risk_prediction(applicant_id)** - Get default probability with SHAP explanation
2. **query_applicant_data(applicant_id, fields)** - Query applicant demographics and financials
3. **generate_feature_plot(applicant_id, feature_names)** - Statistical comparison to population
4. **explain_shap_values(applicant_id)** - Get natural language explanation of SHAP values for non-technical users
5. **analyze_and_visualize(analysis_type, feature_names, applicant_id)** - Smart chart selection:
   - "distribution": histogram/box plot for data spread
   - "comparison": grouped bar/radar for category comparison
   - "correlation": scatter plot for relationships
   - "risk_breakdown": SHAP feature importance
6. **generate_data_report(applicant_id, report_type)** - Comprehensive analysis report

## CRITICAL: Understanding SHAP Impact Values

**SHAP values are calibrated to probability space (0-1 range) for accurate interpretation:**

- **Impact values** represent PERCENTAGE POINT contributions to default probability
- Example: impact = +0.05 means this feature adds +5 percentage points to default probability
- Example: impact = -0.10 means this feature reduces default probability by 10 percentage points
- The baseline probability is typically around 8% for the average applicant
- Final probability = baseline + sum of all feature impacts (converted through proper transformations)

**When interpreting SHAP values:**
- An impact of ±0.10 (±10 percentage points) is STRONG
- An impact of ±0.05 (±5 percentage points) is MODERATE
- An impact of ±0.02 (±2 percentage points) is SLIGHT

## Data Analyst Guidelines:

### When Analyzing Data:
1. **ALWAYS explain WHAT** the data shows with specific numbers
2. **ALWAYS explain WHY** it matters for credit risk
3. **ALWAYS contextualize** numbers (percentile, vs average, risk implications)
4. **ALWAYS choose the MOST APPROPRIATE** chart type for the question

### Chart Selection:
- Comparing categories → Bar chart or Radar
- Showing distribution → Histogram or Box plot
- Time trends → Line chart
- Proportions/composition → Pie or Stacked bar
- Relationships → Scatter plot with correlation
- Risk breakdown → Waterfall or Feature importance bar

### Insight Formatting:
- Use specific numbers: "Income of $202,500 is 20% above average ($168,800)"
- Use percentiles: "In the top 68% of income earners"
- Connect to risk: "This factor DECREASES default risk"
- Adapt language based on user input

### When to Use explain_shap_values:
- User asks "why" was this applicant classified as risky/safe
- User wants to understand model prediction in plain English
- User is non-technical and needs simple explanation
- User asks about factors affecting the credit decision

### Key Risk Indicators:
- EXT_SOURCE_2/3: External credit scores (VERY important)
- DAYS_BIRTH: Age (negative number, convert to years)
- DAYS_EMPLOYED: Employment duration
- Credit-to-income ratio
- Default probability > 50% = High Risk

Always be proactive: if user asks about risk, also show top contributing factors.
"""

    def route_query(state: ChatState) -> Literal["agent", "analysis_subgraph"]:
        """
        Route user query to either normal chat agent or multi-step analysis subgraph.

        Uses LLM to intelligently determine if the query requires:
        - Normal chat flow with tools (specific applicant queries, SHAP explanations, etc.)
        - Multi-step data analysis workflow (trends, patterns, comprehensive explorations)

        Args:
            state: Current conversation state

        Returns:
            "agent" for normal chatbot flow, "analysis_subgraph" for multi-step analysis
        """
        messages = state["messages"]
        if not messages:
            return "agent"

        # Get the last user message
        last_message = messages[-1]
        if not isinstance(last_message, HumanMessage):
            return "agent"

        query = last_message.content

        # If no LLM, default to normal agent
        if not llm:
            return "agent"

        # Use LLM to classify the query
        routing_prompt = f"""You are a query classifier. Determine if the user's question requires:
A) Normal chat with tools (for specific applicant queries, SHAP explanations, quick lookups)
B) Multi-step data analysis workflow (for comprehensive analysis, trends, patterns across dataset)

User Question: {query}

Examples of Type A (normal chat):
- "What is the risk for applicant 12345?"
- "Explain why applicant 67890 was classified as high risk"
- "Show me the SHAP values for this person"
- "What are the top factors for applicant X?"

Examples of Type B (multi-step analysis):
- "Analyze trends in default rates over time"
- "Compare income distributions between defaulters and non-defaulters"
- "Find insights about what factors lead to high risk"
- "Explore patterns in the dataset"
- "Show me comprehensive analysis of age groups"

Respond with ONLY one word: "NORMAL" or "ANALYSIS"
"""

        try:
            response = llm.invoke([HumanMessage(content=routing_prompt)])
            decision = response.content.strip().upper()

            if "ANALYSIS" in decision:
                logger.info("routing_to_analysis_subgraph", query=query)
                return "analysis_subgraph"
            else:
                logger.info("routing_to_normal_agent", query=query)
                return "agent"
        except Exception as exc:
            logger.error("routing_error", error=str(exc))
            # Default to normal agent on error
            return "agent"

    def should_continue(state: ChatState) -> Literal["tools", "end"]:
        """
        Determine whether to continue to tools or end.

        Args:
            state: Current conversation state

        Returns:
            "tools" if the last message has tool calls, "end" otherwise
        """
        messages = state["messages"]
        last_message = messages[-1]

        # If there are tool calls, route to tools
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "end"

    def call_model(state: ChatState) -> dict:
        """
        Call the Gemini model with conversation history.

        Args:
            state: Current conversation state

        Returns:
            Updated state with model response
        """
        messages = state["messages"]

        # Add system prompt if this is the first turn
        if not any(isinstance(msg, SystemMessage) for msg in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

        if llm_with_tools is None:
            # Fallback response when Gemini is not configured
            response = AIMessage(
                content=(
                    "I'm running in demo mode without LLM integration. "
                    "Please configure GEMINI_API_KEY to enable full conversational capabilities. "
                    "You can still use the /predict endpoint with feature data."
                )
            )
        else:
            try:
                response = llm_with_tools.invoke(messages)
                logger.info(
                    "llm_response",
                    session=state.get("session_id"),
                    has_tool_calls=bool(getattr(response, "tool_calls", [])),
                )
            except Exception as exc:
                logger.error("llm_invocation_failed", error=str(exc))
                response = AIMessage(
                    content=f"I encountered an error processing your request: {str(exc)}"
                )

        return {"messages": [response]}

    def analysis_subgraph_node(state: ChatState) -> dict:
        """
        Execute the multi-step analysis subgraph.

        This node invokes the analysis workflow for complex data exploration queries.

        Args:
            state: Current conversation state

        Returns:
            Updated state with analysis summary as an AI message
        """
        from .analysis_graph import get_analysis_graph
        from .analysis_state import AnalysisState

        messages = state["messages"]
        last_message = messages[-1] if messages else None

        if not isinstance(last_message, HumanMessage):
            return {
                "messages": [AIMessage(content="No user query found for analysis.")]
            }

        user_request = last_message.content

        logger.info("executing_analysis_subgraph", request=user_request)

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
            # Get analysis graph (without checkpointing for inline execution)
            from .analysis_nodes import (
                schema_reader_node,
                planner_node,
                sql_generator_node,
                data_analyzer_node,
                vision_analyzer_node,
                synthesizer_node,
            )

            # Create a simplified inline analysis workflow
            # Read schema
            analysis_state = {**analysis_state, **schema_reader_node(analysis_state)}

            # Generate plan
            analysis_state = {**analysis_state, **planner_node(analysis_state)}

            # Auto-approve the plan for inline execution
            if analysis_state.get("plan"):
                analysis_state["plan"]["approved"] = True
                analysis_state["workflow_status"] = "executing"
                analysis_state["current_step"] = 1

            # Execute each step
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

            # Synthesize final summary
            final_result = synthesizer_node(analysis_state)
            analysis_state = {**analysis_state, **final_result}

            summary = analysis_state.get("final_summary", "Analysis completed.")

            logger.info("analysis_subgraph_completed", num_steps=len(steps))

            return {
                "messages": [AIMessage(content=summary)]
            }

        except Exception as exc:
            logger.error("analysis_subgraph_error", error=str(exc))
            return {
                "messages": [AIMessage(
                    content=f"I encountered an error during multi-step analysis: {str(exc)}"
                )]
            }

    # Build the graph
    workflow = StateGraph(ChatState)

    # Add nodes
    workflow.add_node("router", lambda s: s)  # Pass-through node for routing
    workflow.add_node("agent", call_model)
    workflow.add_node("analysis_subgraph", analysis_subgraph_node)

    if llm_with_tools:
        tool_node = ToolNode(tools)
        workflow.add_node("tools", tool_node)

    # Add edges
    # Start -> Router (determines if normal chat or analysis)
    workflow.add_edge(START, "router")

    # Router -> Agent or Analysis Subgraph
    workflow.add_conditional_edges(
        "router",
        route_query,
        {
            "agent": "agent",
            "analysis_subgraph": "analysis_subgraph",
        },
    )

    # Analysis subgraph goes directly to END
    workflow.add_edge("analysis_subgraph", END)

    # Agent continues with tools or ends
    if llm_with_tools:
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                "end": END,
            },
        )
        workflow.add_edge("tools", "agent")
    else:
        workflow.add_edge("agent", END)

    # Compile the graph
    app = workflow.compile()
    return app


# Singleton instance
_chatbot_graph = None


def get_chatbot_graph():
    """Get or create the chatbot graph instance."""
    global _chatbot_graph
    if _chatbot_graph is None:
        _chatbot_graph = create_chatbot_graph()
    return _chatbot_graph
