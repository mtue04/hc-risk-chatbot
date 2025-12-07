"""
Node implementations for the multi-step analysis workflow.

This module contains all the LangGraph nodes for the iterative analysis process:
1. Schema Reader - Inspects PostgreSQL schema
2. Planner - Creates structured analysis plan
3. Human Review - Pauses for user approval
4. SQL Generator - Generates SQL for each step
5. Data Analyzer - Cleans data and creates visualizations
6. Vision Analyzer - Extracts insights from charts
7. Synthesizer - Creates final executive summary
"""

from __future__ import annotations

import json
import os
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import psycopg2
import structlog
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from .analysis_state import AnalysisState, AnalysisStepResult

# Use non-interactive backend for chart generation
matplotlib.use("Agg")

# Set Seaborn style for better-looking charts
sns.set_theme(style="whitegrid", palette="muted")
sns.set_context("notebook", font_scale=1.1)

logger = structlog.get_logger()

# Database configuration
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "homecredit_db")
POSTGRES_USER = os.getenv("POSTGRES_USER", "hc_admin")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "hc_password")

# Output directory for charts
CHART_OUTPUT_DIR = os.getenv("CHART_OUTPUT_DIR", "/tmp/analysis_charts")
os.makedirs(CHART_OUTPUT_DIR, exist_ok=True)


def get_db_connection():
    """Create a PostgreSQL database connection."""
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )


def get_llm() -> ChatGoogleGenerativeAI | None:
    """Get configured Gemini LLM instance."""
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    if not gemini_api_key or gemini_api_key == "changeme":
        logger.warning("GEMINI_API_KEY not configured")
        return None

    return ChatGoogleGenerativeAI(
        model=gemini_model,
        google_api_key=gemini_api_key,
        temperature=0.7,
        convert_system_message_to_human=True,
    )


def schema_reader_node(state: AnalysisState) -> dict:
    """
    Read PostgreSQL schema information.

    Inspects the home_credit schema to get table and column information
    that will be used by the planner.

    Args:
        state: Current analysis state

    Returns:
        Updated state with schema_info populated
    """
    logger.info("schema_reader_node_started")

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get all tables in home_credit schema
        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'home_credit'
            ORDER BY table_name
        """)
        tables = [row[0] for row in cursor.fetchall()]

        # Get columns for each table
        schema_info = {"tables": tables, "columns": {}}

        for table in tables:
            cursor.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'home_credit' AND table_name = %s
                ORDER BY ordinal_position
            """, (table,))

            columns = [
                {"name": row[0], "type": row[1]}
                for row in cursor.fetchall()
            ]
            schema_info["columns"][table] = columns

        cursor.close()
        conn.close()

        logger.info(
            "schema_info_retrieved",
            num_tables=len(tables),
            tables=tables,
        )

        return {"schema_info": schema_info}

    except Exception as exc:
        logger.error("schema_reader_error", error=str(exc))
        return {
            "schema_info": {"error": str(exc), "tables": [], "columns": {}},
        }


def planner_node(state: AnalysisState) -> dict:
    """
    Generate structured analysis plan based on user request and schema.

    Uses LLM to create a step-by-step analysis plan that includes:
    - Description of what to analyze
    - Whether SQL is needed
    - What type of chart to generate

    Args:
        state: Current analysis state with user_request and schema_info

    Returns:
        Updated state with plan populated
    """
    logger.info("planner_node_started", request=state["user_request"])

    llm = get_llm()
    if not llm:
        # Fallback plan if LLM not configured
        fallback_plan = {
            "steps": [
                {
                    "step_number": 1,
                    "description": "Analyze the data (LLM not configured for detailed planning)",
                    "sql_needed": True,
                    "chart_type": "bar",
                }
            ],
            "user_request": state["user_request"],
            "approved": False,
            "user_edits": None,
        }
        return {
            "plan": fallback_plan,
            "workflow_status": "awaiting_approval",
        }

    # Prepare schema summary for the LLM
    schema_info = state.get("schema_info", {})
    schema_summary = f"Available tables: {', '.join(schema_info.get('tables', []))}\n\n"

    for table, columns in schema_info.get("columns", {}).items():
        col_names = [col["name"] for col in columns[:10]]  # First 10 columns
        schema_summary += f"Table '{table}' columns: {', '.join(col_names)}"
        if len(columns) > 10:
            schema_summary += f" ... ({len(columns)} total columns)"
        schema_summary += "\n"

    # Create planning prompt
    planning_prompt = f"""You are a data analysis planning expert. Your task is to create a structured analysis plan.

USER REQUEST: {state["user_request"]}

DATABASE SCHEMA:
{schema_summary}

Create a step-by-step analysis plan. Each step should:
1. Have a clear description of what to analyze
2. Specify if SQL queries are needed
3. Specify what type of chart to create (line, bar, scatter, histogram, boxplot, or none)

Respond with a JSON object in this exact format:
{{
    "steps": [
        {{
            "step_number": 1,
            "description": "Analyze monthly revenue trends",
            "sql_needed": true,
            "chart_type": "line"
        }},
        {{
            "step_number": 2,
            "description": "Compare user cohorts by age group",
            "sql_needed": true,
            "chart_type": "bar"
        }}
    ]
}}

Keep the plan focused and limit to 3-5 steps. Make each step specific and actionable.
"""

    try:
        messages = [HumanMessage(content=planning_prompt)]
        response = llm.invoke(messages)

        # Extract JSON from response
        content = response.content.strip()

        # Handle markdown code blocks
        if content.startswith("```json"):
            content = content[7:]  # Remove ```json
        if content.startswith("```"):
            content = content[3:]  # Remove ```
        if content.endswith("```"):
            content = content[:-3]  # Remove trailing ```

        content = content.strip()

        # Parse the plan
        plan_data = json.loads(content)

        plan = {
            "steps": plan_data.get("steps", []),
            "user_request": state["user_request"],
            "approved": False,
            "user_edits": None,
        }

        logger.info(
            "analysis_plan_generated",
            num_steps=len(plan["steps"]),
        )

        return {
            "plan": plan,
            "workflow_status": "awaiting_approval",
        }

    except Exception as exc:
        logger.error("planner_node_error", error=str(exc))

        # Fallback to simple plan
        fallback_plan = {
            "steps": [
                {
                    "step_number": 1,
                    "description": f"Analyze: {state['user_request']}",
                    "sql_needed": True,
                    "chart_type": "bar",
                }
            ],
            "user_request": state["user_request"],
            "approved": False,
            "user_edits": None,
        }

        return {
            "plan": fallback_plan,
            "workflow_status": "awaiting_approval",
        }


def sql_generator_node(state: AnalysisState) -> dict:
    """
    Generate SQL query for the current analysis step.

    Uses LLM to create appropriate SQL based on:
    - The step description
    - Available schema
    - Type of analysis needed

    Args:
        state: Current analysis state

    Returns:
        Updated state (SQL will be added to step results in the executor)
    """
    logger.info("sql_generator_node_started", step=state["current_step"])

    plan = state.get("plan")
    if not plan:
        logger.error("no_plan_available")
        return {}

    current_step_num = state["current_step"]
    if current_step_num < 1 or current_step_num > len(plan["steps"]):
        logger.error("invalid_step_number", step=current_step_num)
        return {}

    step_info = plan["steps"][current_step_num - 1]

    if not step_info.get("sql_needed", False):
        # No SQL needed for this step
        return {"_sql_query": None}

    llm = get_llm()
    if not llm:
        # Generate a basic fallback SQL
        fallback_sql = """
        SELECT "TARGET", COUNT(*) as count
        FROM home_credit.application_train
        GROUP BY "TARGET"
        ORDER BY "TARGET"
        LIMIT 100
        """
        return {"_sql_query": fallback_sql}

    # Prepare schema context
    schema_info = state.get("schema_info", {})
    schema_summary = ""
    for table, columns in schema_info.get("columns", {}).items():
        col_list = ", ".join([f'"{col["name"]}"' for col in columns[:15]])
        schema_summary += f"Table home_credit.{table}: {col_list}\n"

    # Create SQL generation prompt
    sql_prompt = f"""You are a PostgreSQL expert. Generate a SQL query for this analysis step.

STEP DESCRIPTION: {step_info['description']}

AVAILABLE SCHEMA:
{schema_summary}

IMPORTANT NOTES:
- All column names must be in double quotes (e.g., "SK_ID_CURR")
- Use the home_credit schema (e.g., FROM home_credit.application_train)
- Limit results to a reasonable number (e.g., LIMIT 1000)
- For time-series analysis, use appropriate date functions
- Return only the SQL query, no explanations

Generate the SQL query:
"""

    try:
        messages = [HumanMessage(content=sql_prompt)]
        response = llm.invoke(messages)

        sql_query = response.content.strip()

        # Clean up markdown code blocks
        if sql_query.startswith("```sql"):
            sql_query = sql_query[6:]
        elif sql_query.startswith("```"):
            sql_query = sql_query[3:]
        if sql_query.endswith("```"):
            sql_query = sql_query[:-3]

        sql_query = sql_query.strip()

        logger.info("sql_generated", step=current_step_num)

        return {"_sql_query": sql_query}

    except Exception as exc:
        logger.error("sql_generator_error", error=str(exc))
        return {"_sql_query": None}


def data_analyzer_node(state: AnalysisState) -> dict:
    """
    Execute SQL query, clean data, and generate visualization.

    This node:
    1. Runs the SQL query against PostgreSQL
    2. Loads data into pandas DataFrame
    3. Performs basic cleaning
    4. Generates appropriate chart based on step specification
    5. Saves chart to disk

    Args:
        state: Current analysis state with SQL query

    Returns:
        Updated state with chart path and data summary
    """
    logger.info("data_analyzer_node_started", step=state["current_step"])

    plan = state.get("plan")
    current_step_num = state["current_step"]
    step_info = plan["steps"][current_step_num - 1]

    sql_query = state.get("_sql_query")

    if not sql_query:
        logger.warning("no_sql_query_available", step=current_step_num)
        return {
            "_chart_path": None,
            "_data_summary": {"error": "No SQL query available"},
        }

    try:
        # Execute SQL and load into DataFrame
        conn = get_db_connection()
        df = pd.read_sql_query(sql_query, conn)
        conn.close()

        logger.info(
            "data_loaded",
            rows=len(df),
            columns=list(df.columns),
        )

        # Basic data summary
        data_summary = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "sample_data": df.head(5).to_dict(orient="records"),
        }

        # Generate chart
        chart_type = step_info.get("chart_type", "bar")
        chart_path = os.path.join(
            CHART_OUTPUT_DIR,
            f"step_{current_step_num}_{chart_type}.png",
        )

        # Create figure with better styling
        fig, ax = plt.subplots(figsize=(12, 7))

        if chart_type == "line":
            # Assume first column is x-axis, second is y-axis
            if len(df.columns) >= 2:
                sns.lineplot(
                    data=df,
                    x=df.columns[0],
                    y=df.columns[1],
                    marker='o',
                    linewidth=2.5,
                    markersize=8,
                    ax=ax
                )
                ax.set_xlabel(df.columns[0], fontsize=12, fontweight='bold')
                ax.set_ylabel(df.columns[1], fontsize=12, fontweight='bold')

        elif chart_type == "bar":
            # Bar chart - first column as categories, second as values
            if len(df.columns) >= 2:
                sns.barplot(
                    data=df,
                    x=df.columns[0],
                    y=df.columns[1],
                    palette="Blues_d",
                    ax=ax
                )
                ax.set_xlabel(df.columns[0], fontsize=12, fontweight='bold')
                ax.set_ylabel(df.columns[1], fontsize=12, fontweight='bold')
                ax.tick_params(axis='x', rotation=45)
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        elif chart_type == "scatter":
            if len(df.columns) >= 2:
                sns.scatterplot(
                    data=df,
                    x=df.columns[0],
                    y=df.columns[1],
                    alpha=0.6,
                    s=100,
                    ax=ax
                )
                ax.set_xlabel(df.columns[0], fontsize=12, fontweight='bold')
                ax.set_ylabel(df.columns[1], fontsize=12, fontweight='bold')

        elif chart_type == "histogram":
            if len(df.columns) >= 1:
                sns.histplot(
                    data=df,
                    x=df.columns[0],
                    bins=30,
                    kde=True,
                    color='steelblue',
                    edgecolor='black',
                    ax=ax
                )
                ax.set_xlabel(df.columns[0], fontsize=12, fontweight='bold')
                ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')

        elif chart_type == "boxplot":
            if len(df.columns) >= 1:
                # Select up to 5 numeric columns for boxplot
                numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]
                if len(numeric_cols) > 0:
                    df_melted = df[numeric_cols].melt(var_name='Feature', value_name='Value')
                    sns.boxplot(
                        data=df_melted,
                        x='Feature',
                        y='Value',
                        palette="Set2",
                        ax=ax
                    )
                    ax.set_xlabel('Features', fontsize=12, fontweight='bold')
                    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
                    ax.tick_params(axis='x', rotation=45)
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        ax.set_title(step_info['description'], fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        logger.info("chart_generated", path=chart_path)

        return {
            "_chart_path": chart_path,
            "_data_summary": data_summary,
        }

    except Exception as exc:
        logger.error("data_analyzer_error", error=str(exc), step=current_step_num)
        return {
            "_chart_path": None,
            "_data_summary": {"error": str(exc)},
        }


def vision_analyzer_node(state: AnalysisState) -> dict:
    """
    Analyze the generated chart using vision capabilities.

    Uses Gemini's vision model to:
    1. Examine the chart image
    2. Extract key insights
    3. Identify patterns, trends, anomalies
    4. Provide business context

    Args:
        state: Current analysis state with chart path

    Returns:
        Updated state with insights added to step results
    """
    logger.info("vision_analyzer_node_started", step=state["current_step"])

    plan = state.get("plan")
    current_step_num = state["current_step"]
    step_info = plan["steps"][current_step_num - 1]

    chart_path = state.get("_chart_path")
    data_summary = state.get("_data_summary", {})
    sql_query = state.get("_sql_query")

    # Create step result
    step_result: AnalysisStepResult = {
        "step_number": current_step_num,
        "sql_query": sql_query,
        "data_summary": data_summary,
        "chart_data": None,
        "chart_image_path": chart_path,
        "insights": "",
    }

    if not chart_path:
        step_result["insights"] = (
            f"Could not generate insights for step {current_step_num}: "
            f"No chart was created. Error: {data_summary.get('error', 'Unknown')}"
        )

        # Add to step_results
        current_results = state.get("step_results", [])
        current_results.append(step_result)

        return {"step_results": current_results}

    llm = get_llm()
    if not llm:
        step_result["insights"] = (
            f"Step {current_step_num}: {step_info['description']}. "
            f"Data summary: {data_summary.get('row_count', 0)} rows analyzed. "
            f"(Vision analysis unavailable - LLM not configured)"
        )

        current_results = state.get("step_results", [])
        current_results.append(step_result)

        return {"step_results": current_results}

    try:
        # For now, use text-based analysis of data summary
        # (Full vision analysis would require Gemini Pro Vision with image input)
        analysis_prompt = f"""Analyze the following data analysis results and provide insights.

ANALYSIS STEP: {step_info['description']}

DATA SUMMARY:
- Rows analyzed: {data_summary.get('row_count', 0)}
- Columns: {', '.join(data_summary.get('columns', []))}
- Sample data: {json.dumps(data_summary.get('sample_data', []), indent=2)}

CHART TYPE: {step_info.get('chart_type', 'unknown')}

Provide a concise analysis (2-3 sentences) covering:
1. Key patterns or trends observed
2. Notable findings or anomalies
3. Business implications

Keep it concise and actionable.
"""

        messages = [HumanMessage(content=analysis_prompt)]
        response = llm.invoke(messages)

        insights = response.content.strip()

        logger.info("insights_generated", step=current_step_num)

        step_result["insights"] = insights

        # Add to step_results
        current_results = state.get("step_results", [])
        current_results.append(step_result)

        return {"step_results": current_results}

    except Exception as exc:
        logger.error("vision_analyzer_error", error=str(exc))

        step_result["insights"] = f"Error generating insights: {str(exc)}"

        current_results = state.get("step_results", [])
        current_results.append(step_result)

        return {"step_results": current_results}


def synthesizer_node(state: AnalysisState) -> dict:
    """
    Synthesize final executive summary from all step insights.

    Combines insights from all analysis steps into a cohesive summary with:
    - Key findings
    - Patterns across steps
    - Recommendations
    - Supporting data references

    Args:
        state: Complete analysis state with all step results

    Returns:
        Updated state with final_summary populated
    """
    logger.info("synthesizer_node_started")

    llm = get_llm()
    step_results = state.get("step_results", [])

    if not step_results:
        summary = "No analysis steps were completed."
        return {
            "final_summary": summary,
            "workflow_status": "completed",
        }

    # Compile all insights
    insights_text = ""
    for result in step_results:
        insights_text += f"\nStep {result['step_number']}: {result['insights']}\n"

    if not llm:
        summary = f"""EXECUTIVE SUMMARY

Original Request: {state['user_request']}

Analysis completed with {len(step_results)} step(s).

{insights_text}

(Detailed synthesis unavailable - LLM not configured)
"""
        return {
            "final_summary": summary,
            "workflow_status": "completed",
        }

    # Create synthesis prompt
    synthesis_prompt = f"""You are a data analyst creating an executive summary.

ORIGINAL REQUEST: {state['user_request']}

ANALYSIS INSIGHTS FROM {len(step_results)} STEP(S):
{insights_text}

Create a comprehensive executive summary that:
1. Starts with key findings (bullet points)
2. Discusses patterns and trends observed
3. Provides actionable recommendations
4. Keeps technical jargon minimal

Format as a professional business summary (3-5 paragraphs).
"""

    try:
        messages = [HumanMessage(content=synthesis_prompt)]
        response = llm.invoke(messages)

        summary = response.content.strip()

        logger.info("final_summary_generated")

        return {
            "final_summary": summary,
            "workflow_status": "completed",
        }

    except Exception as exc:
        logger.error("synthesizer_error", error=str(exc))

        summary = f"""EXECUTIVE SUMMARY

Original Request: {state['user_request']}

{insights_text}

Note: Detailed synthesis encountered an error: {str(exc)}
"""

        return {
            "final_summary": summary,
            "workflow_status": "completed",
        }
