"""
LangGraph state machine for HomeCredit Risk Chatbot.

This module implements a conversational AI workflow using LangGraph and Gemini
to provide intelligent credit risk analysis through natural dialogue.

Features:
- PostgreSQL checkpointing for conversation persistence
- Retry logic with exponential backoff
- Memory management with automatic summarization
- Enhanced state tracking
"""

from __future__ import annotations

import os
from typing import Annotated, List, Literal, TypedDict

import structlog
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .tools import (
    get_risk_prediction,
    query_applicant_data,
    generate_feature_plot,
    compare_applicants,
    explain_risk_factors,
    query_bureau_history,
    get_portfolio_stats,
    what_if_analysis,
)

logger = structlog.get_logger()

# Check if enhanced features are enabled
USE_ENHANCED_FEATURES = os.getenv("USE_ENHANCED_GRAPH", "true").lower() == "true"
ENABLE_CHECKPOINTING = os.getenv("ENABLE_CHECKPOINTING", "true").lower() == "true"

# Configure LangSmith tracing if API key is provided
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
if LANGSMITH_API_KEY and LANGSMITH_API_KEY != "changeme":
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "hc-risk-chatbot")
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
    logger.info("langsmith_tracing_enabled", project=os.environ["LANGCHAIN_PROJECT"])
else:
    logger.info("langsmith_tracing_disabled")


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

    # Enhanced features (used when USE_ENHANCED_FEATURES=true)
    conversation_summary: str | None
    """LLM-generated summary of conversation so far."""

    mentioned_applicants: List[int]
    """List of all applicant IDs discussed in this conversation."""

    key_insights: List[dict]
    """Extracted key facts for quick recall."""

    turn_count: int
    """Number of conversation turns (for triggering summarization)."""


def get_postgres_checkpointer():
    """
    Create PostgreSQL checkpointer for conversation persistence.

    Returns None if PostgreSQL connection fails (fallback to in-memory).
    """
    if not ENABLE_CHECKPOINTING:
        return None

    try:
        from psycopg import Connection
        from langgraph.checkpoint.postgres import PostgresSaver

        postgres_host = os.getenv("POSTGRES_HOST", "postgres")
        postgres_port = os.getenv("POSTGRES_PORT", "5432")
        postgres_db = os.getenv("POSTGRES_DB", "homecredit_db")
        postgres_user = os.getenv("POSTGRES_USER", "hc_admin")
        postgres_password = os.getenv("POSTGRES_PASSWORD", "hc_password")

        conn_string = f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"

        # Create checkpoint table if it doesn't exist
        conn = Connection.connect(conn_string, autocommit=True)

        checkpointer = PostgresSaver(conn)
        checkpointer.setup()  # Create tables if they don't exist

        logger.info("postgres_checkpointer_enabled", db=postgres_db)
        return checkpointer

    except ImportError:
        logger.warning("psycopg3_not_available", fallback="in-memory checkpointing")
        return None
    except Exception as e:
        logger.warning("checkpointer_setup_failed", error=str(e), fallback="in-memory")
        return None


def create_chatbot_graph():
    """
    Create the LangGraph state machine for the chatbot.

    Returns:
        Compiled LangGraph workflow
    """
    # Initialize Gemini LLM
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

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

    # Define tools (all 8 tools, but basic mode only uses first 3)
    all_tools = [
        get_risk_prediction,
        query_applicant_data,
        generate_feature_plot,
        compare_applicants,
        explain_risk_factors,
        query_bureau_history,
        get_portfolio_stats,
        what_if_analysis,
    ]

    # Use subset of tools if enhanced features disabled
    tools = all_tools if USE_ENHANCED_FEATURES else all_tools[:3]

    # Bind tools to LLM
    if llm:
        llm_with_tools = llm.bind_tools(tools)
    else:
        llm_with_tools = None

    # System prompt
    tool_descriptions = """1. get_risk_prediction(applicant_id: int) - Get default probability and explanation for an applicant
2. query_applicant_data(applicant_id: int, fields: list[str]) - Query specific applicant information
3. generate_feature_plot(applicant_id: int, feature_names: list[str]) - Generate visualization"""

    if USE_ENHANCED_FEATURES:
        tool_descriptions += """
4. compare_applicants(applicant_ids: list[int]) - Compare multiple applicants side-by-side
5. explain_risk_factors(applicant_id: int, top_n: int) - Get detailed SHAP waterfall explanation
6. query_bureau_history(applicant_id: int) - Get credit bureau history with trends
7. get_portfolio_stats(risk_threshold: float) - Get aggregate portfolio statistics
8. what_if_analysis(applicant_id: int, changes: dict) - Simulate counterfactual scenarios (e.g., "What if income was $50k higher?")"""

    SYSTEM_PROMPT = f"""You are an expert credit risk analyst assistant for Home Credit.

Your role is to help users understand credit risk predictions and applicant data through natural conversation.

You have access to the following tools:
{tool_descriptions}

Guidelines:
- When users ask about a specific applicant, use the applicant ID (SK_ID_CURR)
- Always explain predictions in business terms, not just technical metrics
- When showing risk scores, contextualize them (e.g., "This is above/below average")
- Use tools proactively to provide data-driven answers
- If you don't have enough information, ask clarifying questions
- Be concise but thorough in explanations
{"- When comparing applicants, use the compare_applicants tool instead of calling get_risk_prediction multiple times" if USE_ENHANCED_FEATURES else ""}

Remember:
- Default probability > 50% indicates high risk
- Top SHAP contributors show why the model made its prediction
- External credit bureau scores (EXT_SOURCE_*) are very important features
{"- Always cite specific numbers when making recommendations" if USE_ENHANCED_FEATURES else ""}
"""

    def should_continue(state: ChatState) -> Literal["tools", "summarize", "end"]:
        """
        Determine whether to continue to tools, summarize, or end.

        Args:
            state: Current conversation state

        Returns:
            "tools" if the last message has tool calls
            "summarize" if conversation is too long (enhanced mode only)
            "end" otherwise
        """
        messages = state["messages"]
        last_message = messages[-1]

        # If there are tool calls, route to tools
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"

        # Check if we should summarize (every 10 turns in enhanced mode)
        if USE_ENHANCED_FEATURES:
            turn_count = state.get("turn_count", 0)
            if turn_count > 0 and turn_count % 10 == 0 and len(messages) > 15:
                logger.info("triggering_summarization", turn_count=turn_count)
                return "summarize"

        return "end"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    def call_model_with_retry(messages: list[BaseMessage]) -> AIMessage:
        """Call LLM with retry logic."""
        if llm_with_tools is None:
            return AIMessage(
                content=(
                    "I'm running in demo mode without LLM integration. "
                    "Please configure GEMINI_API_KEY to enable full conversational capabilities."
                )
            )

        try:
            return llm_with_tools.invoke(messages)
        except Exception as exc:
            logger.error("llm_invocation_failed", error=str(exc))
            # Return specific error messages for different failures
            error_str = str(exc).lower()
            if "quota" in error_str or "429" in error_str:
                return AIMessage(
                    content="⚠️ **API Quota Exceeded**\n\n"
                            "The Gemini API quota has been exceeded. This usually means:\n"
                            "- Daily request limit reached\n"
                            "- Requests per minute limit reached\n\n"
                            "**Solutions:**\n"
                            "- Wait for quota reset (usually 24 hours)\n"
                            "- Use a different API key\n"
                            "- Upgrade to paid tier for higher limits\n\n"
                            f"Error details: {str(exc)[:200]}"
                )
            elif "rate" in error_str:
                return AIMessage(
                    content=f"⚠️ **Rate Limit Exceeded**\n\nPlease wait a moment and try again.\n\nError: {str(exc)[:200]}"
                )
            elif "api" in error_str or "key" in error_str:
                return AIMessage(
                    content=f"⚠️ **API Key Error**\n\nThere's an issue with the API key configuration.\n\nError: {str(exc)[:200]}"
                )
            # For other errors, re-raise to trigger retry
            raise

    def call_model(state: ChatState) -> dict:
        """
        Call the Gemini model with conversation history and memory.

        Args:
            state: Current conversation state

        Returns:
            Updated state with model response
        """
        messages = state["messages"]

        # Add system prompt if this is the first turn
        if not any(isinstance(msg, SystemMessage) for msg in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

        # If we have a conversation summary (enhanced mode), inject it before recent messages
        if USE_ENHANCED_FEATURES and state.get("conversation_summary"):
            summary_msg = SystemMessage(
                content=f"[Conversation summary so far: {state['conversation_summary']}]"
            )
            # Insert summary after system prompt, before recent messages
            messages = messages[:1] + [summary_msg] + messages[-10:]  # Keep only last 10 messages

        try:
            response = call_model_with_retry(messages)
            logger.info(
                "llm_response",
                session=state.get("session_id"),
                has_tool_calls=bool(getattr(response, "tool_calls", [])),
            )
        except Exception as exc:
            logger.error("llm_invocation_failed_all_retries", error=str(exc))
            response = AIMessage(
                content=f"I encountered an error processing your request. Please try again."
            )

        # Update turn count (enhanced mode only)
        updates = {"messages": [response]}
        if USE_ENHANCED_FEATURES:
            updates["turn_count"] = state.get("turn_count", 0) + 1

        return updates

    def summarize_conversation(state: ChatState) -> dict:
        """
        Summarize the conversation to manage context length.

        Args:
            state: Current conversation state

        Returns:
            Updated state with conversation summary
        """
        messages = state["messages"]

        if llm is None or len(messages) < 10:
            return {}  # Skip summarization if no LLM or short conversation

        try:
            # Create summarization prompt
            conversation_text = "\n".join([
                f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
                for msg in messages[:-5]  # Summarize all but last 5 messages
                if isinstance(msg, (HumanMessage, AIMessage))
            ])

            summary_prompt = f"""Summarize this credit risk analysis conversation in 2-3 sentences, focusing on:
- Which applicants were discussed (include IDs)
- Key risk scores and findings
- Important insights or decisions

Conversation:
{conversation_text}

Summary:"""

            summary_response = llm.invoke([HumanMessage(content=summary_prompt)])
            summary = summary_response.content

            logger.info("conversation_summarized", session=state.get("session_id"), length=len(messages))

            return {
                "conversation_summary": summary,
            }

        except Exception as exc:
            logger.error("summarization_failed", error=str(exc))
            return {}  # Continue without summarization

    def extract_insights(state: ChatState) -> dict:
        """
        Extract key insights from tool outputs.

        This runs after tool execution to update the key_insights list.
        """
        if not USE_ENHANCED_FEATURES:
            return {}

        last_tool_output = state.get("last_tool_output")

        if not last_tool_output:
            return {}

        insights = state.get("key_insights", [])
        mentioned_applicants = state.get("mentioned_applicants", [])

        # Extract applicant IDs from tool output
        if "applicant_id" in last_tool_output:
            app_id = last_tool_output["applicant_id"]
            if app_id not in mentioned_applicants:
                mentioned_applicants.append(app_id)

        # Extract risk scores
        if "probability" in last_tool_output:
            insights.append({
                "type": "risk_score",
                "applicant_id": last_tool_output.get("applicant_id"),
                "probability": last_tool_output["probability"],
            })

        return {
            "key_insights": insights[-20:],  # Keep last 20 insights
            "mentioned_applicants": mentioned_applicants,
        }

    # Build the graph
    workflow = StateGraph(ChatState)

    # Add nodes
    workflow.add_node("agent", call_model)

    if USE_ENHANCED_FEATURES:
        workflow.add_node("summarize", summarize_conversation)
        workflow.add_node("extract_insights", extract_insights)

    if llm_with_tools:
        tool_node = ToolNode(tools)
        workflow.add_node("tools", tool_node)

    # Add edges
    workflow.add_edge(START, "agent")

    if llm_with_tools:
        if USE_ENHANCED_FEATURES:
            workflow.add_conditional_edges(
                "agent",
                should_continue,
                {
                    "tools": "tools",
                    "summarize": "summarize",
                    "end": END,
                },
            )
            workflow.add_edge("tools", "extract_insights")
            workflow.add_edge("extract_insights", "agent")
            workflow.add_edge("summarize", END)
        else:
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

    # Compile the graph with checkpointing
    checkpointer = get_postgres_checkpointer()

    if checkpointer:
        app = workflow.compile(checkpointer=checkpointer)
        logger.info("graph_compiled_with_checkpointing", enhanced=USE_ENHANCED_FEATURES)
    else:
        app = workflow.compile()
        logger.info("graph_compiled_without_checkpointing", enhanced=USE_ENHANCED_FEATURES)

    return app


# Singleton instance
_chatbot_graph = None


def get_chatbot_graph():
    """Get or create the chatbot graph instance."""
    global _chatbot_graph
    if _chatbot_graph is None:
        _chatbot_graph = create_chatbot_graph()
    return _chatbot_graph
