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

from .tools import get_risk_prediction, query_applicant_data, generate_feature_plot, predict_hypothetical_applicant
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


def create_chatbot_graph():
    """
    Create the LangGraph state machine for the chatbot.

    Returns:
        Compiled LangGraph workflow
    """
    # Initialize Gemini LLM
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")  # flash has higher rate limits
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
4. **analyze_and_visualize(analysis_type, feature_names, applicant_id)** - Smart chart selection:
   - "distribution": histogram/box plot for data spread
   - "comparison": grouped bar/radar for category comparison
   - "correlation": scatter plot for relationships
   - "risk_breakdown": SHAP feature importance
5. **generate_data_report(applicant_id, report_type)** - Comprehensive analysis report

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

### Key Risk Indicators:
- EXT_SOURCE_2/3: External credit scores (VERY important)
- DAYS_BIRTH: Age (negative number, convert to years)
- DAYS_EMPLOYED: Employment duration
- Credit-to-income ratio
- Default probability > 50% = High Risk

Always be proactive: if user asks about risk, also show top contributing factors.
"""

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

    # Build the graph
    workflow = StateGraph(ChatState)

    # Add nodes
    workflow.add_node("agent", call_model)
    
    if llm_with_tools:
        tool_node = ToolNode(tools)
        workflow.add_node("tools", tool_node)

    # Add edges
    workflow.add_edge(START, "agent")
    
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
