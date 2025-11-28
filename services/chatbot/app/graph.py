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

from .tools import get_risk_prediction, query_applicant_data, generate_feature_plot

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
    if not gemini_api_key or gemini_api_key == "changeme":
        logger.warning("GEMINI_API_KEY not configured, using mock responses")
        llm = None
    else:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            google_api_key=gemini_api_key,
            temperature=0.7,
            convert_system_message_to_human=True,
        )

    # Define tools available to the agent
    tools = [
        get_risk_prediction,
        query_applicant_data,
        generate_feature_plot,
    ]

    # Bind tools to LLM
    if llm:
        llm_with_tools = llm.bind_tools(tools)
    else:
        llm_with_tools = None

    # System prompt
    SYSTEM_PROMPT = """You are an expert credit risk analyst assistant for Home Credit.

Your role is to help users understand credit risk predictions and applicant data through natural conversation.

You have access to the following tools:
1. get_risk_prediction(applicant_id: int) - Get default probability and explanation for an applicant
2. query_applicant_data(applicant_id: int, fields: list[str]) - Query specific applicant information
3. generate_feature_plot(applicant_id: int, feature_names: list[str]) - Generate visualization

Guidelines:
- When users ask about a specific applicant, use the applicant ID (SK_ID_CURR)
- Always explain predictions in business terms, not just technical metrics
- When showing risk scores, contextualize them (e.g., "This is above/below average")
- Use tools proactively to provide data-driven answers
- If you don't have enough information, ask clarifying questions
- Be concise but thorough in explanations

Remember:
- Default probability > 50% indicates high risk
- Top SHAP contributors show why the model made its prediction
- External credit bureau scores (EXT_SOURCE_*) are very important features
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
