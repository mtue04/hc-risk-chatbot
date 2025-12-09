"""
Intelligent Chart Selection Tools for the LangGraph chatbot.

These tools enable the chatbot to:
1. Automatically select the most appropriate chart type for the analysis
2. Generate visualizations using Plotly
3. Provide statistical insights from the data

Chart Selection Logic:
- distribution -> histogram, box plot
- comparison -> grouped bar, radar
- correlation -> scatter, heatmap  
- trend -> line, area
- composition -> pie, stacked bar, waterfall
- risk_breakdown -> gauge, waterfall, feature importance
"""

from __future__ import annotations

import base64
import io
import os
from typing import Any, Literal

import httpx
import psycopg2
import structlog
import numpy as np
from langchain_core.tools import tool

logger = structlog.get_logger()

# Configuration
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "homecredit_db")
POSTGRES_USER = os.getenv("POSTGRES_USER", "hc_admin")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "hc_password")
MODEL_API_URL = os.getenv("MODEL_API_URL", "http://model_serving:8000")


def get_db_connection():
    """Create a PostgreSQL database connection."""
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )


# Chart type recommendations based on analysis intent
CHART_TYPE_MAP = {
    "distribution": ["histogram", "box_plot", "violin"],
    "comparison": ["bar", "grouped_bar", "radar"],
    "correlation": ["scatter", "heatmap"],
    "trend": ["line", "area"],  
    "composition": ["pie", "stacked_bar", "waterfall"],
    "risk_breakdown": ["gauge", "waterfall", "feature_importance"],
    "percentile": ["bullet", "bar_with_marker"],
}


def generate_insight_text(analysis_type: str, stats: dict, feature_name: str = None) -> str:
    """Generate human-readable insight from statistical analysis."""
    insights = []
    
    if analysis_type == "distribution" and feature_name:
        mean = stats.get("mean", 0)
        median = stats.get("median", 0)
        std = stats.get("std", 0)
        
        if mean > median * 1.1:
            skew_text = "right-skewed (some high outliers)"
        elif mean < median * 0.9:
            skew_text = "left-skewed (some low outliers)"
        else:
            skew_text = "approximately symmetric"
        
        insights.append(f"**{feature_name} Analysis**")
        insights.append(f"- Mean: {mean:,.2f}")
        insights.append(f"- Median: {median:,.2f}")
        insights.append(f"- Std Dev: {std:,.2f}")
        insights.append(f"- Distribution: {skew_text}")
        
    elif analysis_type == "comparison":
        for group, group_stats in stats.items():
            insights.append(f"**{group}**: Mean = {group_stats.get('mean', 0):,.2f}")
            
    elif analysis_type == "correlation":
        corr = stats.get("correlation", 0)
        strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.4 else "weak"
        direction = "positive" if corr > 0 else "negative"
        insights.append(f"Correlation: {corr:.3f} ({strength} {direction})")
        
    elif analysis_type == "risk_breakdown":
        for factor in stats.get("top_factors", [])[:5]:
            impact = factor.get("impact", 0)
            direction = "increases risk" if impact > 0 else "decreases risk"
            insights.append(f"- {factor['feature']}: {direction} ({impact:+.4f})")
    
    return "\n".join(insights) if insights else "Insufficient data for analysis."


@tool
def analyze_and_visualize(
    analysis_type: Literal[
        "distribution", "comparison", "correlation", 
        "trend", "composition", "risk_breakdown"
    ],
    feature_names: list[str],
    applicant_id: int | None = None,
    compare_groups: str | None = None,
) -> dict[str, Any]:
    """
    Perform data analysis and automatically select the best chart type.
    
    Use this tool when user wants to see visualizations or understand data patterns.
    The tool will automatically choose the most appropriate chart type based on 
    the analysis type and return both the chart data and insights.
    
    Args:
        analysis_type: Type of analysis to perform
            - "distribution": Show how values are spread (histogram, box plot)
            - "comparison": Compare between categories (bar, radar)
            - "correlation": Show relationships between features (scatter)
            - "trend": Show changes over time (line chart)
            - "composition": Show parts of a whole (pie, waterfall)
            - "risk_breakdown": SHAP feature importance analysis
        feature_names: List of feature names to analyze (max 15)
        applicant_id: Optional applicant ID for individual analysis
        compare_groups: Optional grouping variable for comparison
    
    Returns:
        Dictionary with:
        - chart_type: The recommended chart type
        - chart_data: Data for rendering the chart
        - insights: Natural language description of findings
        - statistics: Computed statistics
    """
    if len(feature_names) > 15:
        return {
            "error": "Maximum 15 features allowed per analysis",
            "requested": len(feature_names),
        }
    
    feature_names = [f.upper() for f in feature_names]
    recommended_charts = CHART_TYPE_MAP.get(analysis_type, ["bar"])
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        result = {
            "analysis_type": analysis_type,
            "chart_type": recommended_charts[0],
            "alternative_charts": recommended_charts[1:] if len(recommended_charts) > 1 else [],
            "features": feature_names,
        }
        
        if analysis_type == "distribution":
            # Get distribution statistics for each feature
            stats = {}
            for feature in feature_names:
                quoted_feature = f'"{feature}"'
                query = f"""
                    SELECT 
                        AVG({quoted_feature}) as mean,
                        STDDEV({quoted_feature}) as std,
                        MIN({quoted_feature}) as min,
                        MAX({quoted_feature}) as max,
                        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {quoted_feature}) as q25,
                        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY {quoted_feature}) as median,
                        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {quoted_feature}) as q75,
                        COUNT(*) as count
                    FROM home_credit.application_train
                    WHERE {quoted_feature} IS NOT NULL
                """
                cursor.execute(query)
                row = cursor.fetchone()
                
                feature_stats = {
                    "mean": float(row[0]) if row[0] else None,
                    "std": float(row[1]) if row[1] else None,
                    "min": float(row[2]) if row[2] else None,
                    "max": float(row[3]) if row[3] else None,
                    "q25": float(row[4]) if row[4] else None,
                    "median": float(row[5]) if row[5] else None,
                    "q75": float(row[6]) if row[6] else None,
                    "count": int(row[7]) if row[7] else 0,
                }
                
                # Compute histogram bins (20 bins)
                # Use IQR method to determine bin range to handle outliers
                if feature_stats["q25"] is not None and feature_stats["q75"] is not None:
                    iqr = feature_stats["q75"] - feature_stats["q25"]
                    # Use 1.5 * IQR for sensible range (exclude extreme outliers)
                    hist_min = max(feature_stats["min"], feature_stats["q25"] - 1.5 * iqr)
                    hist_max = min(feature_stats["max"], feature_stats["q75"] + 1.5 * iqr)
                else:
                    hist_min = feature_stats["min"]
                    hist_max = feature_stats["max"]
                
                num_bins = 20
                if hist_min is not None and hist_max is not None and hist_max > hist_min:
                    bin_width = (hist_max - hist_min) / num_bins
                    
                    # Query to get histogram bin counts
                    bins_query = f"""
                        SELECT 
                            width_bucket({quoted_feature}, %s, %s, %s) as bucket,
                            COUNT(*) as freq
                        FROM home_credit.application_train
                        WHERE {quoted_feature} IS NOT NULL 
                          AND {quoted_feature} BETWEEN %s AND %s
                        GROUP BY bucket
                        ORDER BY bucket
                    """
                    cursor.execute(bins_query, (hist_min, hist_max, num_bins, hist_min, hist_max))
                    bin_rows = cursor.fetchall()
                    
                    # Build histogram bins array
                    histogram_bins = []
                    bin_counts = {r[0]: r[1] for r in bin_rows}
                    
                    for i in range(1, num_bins + 1):
                        bin_start = hist_min + (i - 1) * bin_width
                        bin_end = hist_min + i * bin_width
                        histogram_bins.append({
                            "bin": i,
                            "range_start": round(bin_start, 2),
                            "range_end": round(bin_end, 2),
                            "count": bin_counts.get(i, 0),
                            "label": f"{bin_start/1000:.0f}K" if bin_start >= 1000 else f"{bin_start:.0f}"
                        })
                    
                    feature_stats["histogram_bins"] = histogram_bins
                
                # Add applicant's value if provided
                if applicant_id:
                    cursor.execute(f"""
                        SELECT {quoted_feature} 
                        FROM home_credit.application_train 
                        WHERE "SK_ID_CURR" = %s
                    """, (applicant_id,))
                    app_row = cursor.fetchone()
                    if app_row:
                        feature_stats["applicant_value"] = float(app_row[0]) if app_row[0] else None
                        # Calculate percentile
                        if app_row[0]:
                            cursor.execute(f"""
                                SELECT COUNT(*) * 100.0 / (
                                    SELECT COUNT(*) FROM home_credit.application_train 
                                    WHERE {quoted_feature} IS NOT NULL
                                )
                                FROM home_credit.application_train
                                WHERE {quoted_feature} <= %s AND {quoted_feature} IS NOT NULL
                            """, (app_row[0],))
                            pct = cursor.fetchone()
                            feature_stats["percentile"] = float(pct[0]) if pct[0] else None
                
                stats[feature] = feature_stats
            
            result["statistics"] = stats
            result["insights"] = generate_insight_text(
                "distribution", 
                stats.get(feature_names[0], {}),
                feature_names[0]
            )
            
        elif analysis_type == "comparison":
            # Compare by TARGET (defaulter vs non-defaulter)
            stats = {"Non-Defaulter (TARGET=0)": {}, "Defaulter (TARGET=1)": {}}
            
            for feature in feature_names:
                quoted_feature = f'"{feature}"'
                cursor.execute(f"""
                    SELECT "TARGET", AVG({quoted_feature}), STDDEV({quoted_feature})
                    FROM home_credit.application_train
                    WHERE {quoted_feature} IS NOT NULL
                    GROUP BY "TARGET"
                """)
                for row in cursor.fetchall():
                    target = "Defaulter (TARGET=1)" if row[0] == 1 else "Non-Defaulter (TARGET=0)"
                    stats[target][feature] = {
                        "mean": float(row[1]) if row[1] else None,
                        "std": float(row[2]) if row[2] else None,
                    }
            
            result["statistics"] = stats
            result["chart_type"] = "grouped_bar"
            result["insights"] = generate_insight_text("comparison", stats)
            
        elif analysis_type == "risk_breakdown":
            # Get SHAP explanation for applicant
            if applicant_id:
                url = f"{MODEL_API_URL}/explain/applicant/{applicant_id}"
                with httpx.Client(timeout=10.0) as client:
                    response = client.get(url)
                    if response.ok:
                        explanation = response.json()
                        result["statistics"] = explanation
                        result["chart_type"] = "feature_importance"
                        
                        # Generate insights from SHAP values
                        shap_values = explanation.get("shap_values", {})
                        sorted_factors = sorted(
                            shap_values.items(), 
                            key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0, 
                            reverse=True
                        )[:5]
                        
                        result["insights"] = generate_insight_text(
                            "risk_breakdown",
                            {"top_factors": [{"feature": k, "impact": v} for k, v in sorted_factors]}
                        )
            else:
                result["error"] = "applicant_id required for risk_breakdown analysis"
                
        elif analysis_type == "correlation":
            if len(feature_names) >= 2:
                if len(feature_names) == 2:
                    # Scatter plot for 2 features - sample data points
                    f1, f2 = f'"{feature_names[0]}"', f'"{feature_names[1]}"'
                    
                    # Get correlation value
                    cursor.execute(f"""
                        SELECT CORR({f1}, {f2})
                        FROM home_credit.application_train
                        WHERE {f1} IS NOT NULL AND {f2} IS NOT NULL
                    """)
                    corr = cursor.fetchone()
                    
                    # Sample 200 points for scatter plot
                    cursor.execute(f"""
                        SELECT {f1}, {f2}
                        FROM home_credit.application_train
                        WHERE {f1} IS NOT NULL AND {f2} IS NOT NULL
                        ORDER BY RANDOM()
                        LIMIT 200
                    """)
                    points = cursor.fetchall()
                    
                    result["statistics"] = {
                        "correlation": float(corr[0]) if corr[0] else None,
                        "feature_x": feature_names[0],
                        "feature_y": feature_names[1],
                    }
                    result["chart_data"] = {
                        "x": [float(p[0]) for p in points],
                        "y": [float(p[1]) for p in points],
                        "labels": [f"Point {i+1}" for i in range(len(points))],
                    }
                    result["chart_type"] = "scatter"
                    result["insights"] = generate_insight_text(
                        "correlation", 
                        {"correlation": float(corr[0]) if corr[0] else 0}
                    )
                else:
                    # Heatmap for 3+ features - correlation matrix
                    n = len(feature_names)
                    correlation_matrix = [[0.0] * n for _ in range(n)]
                    
                    for i, f1 in enumerate(feature_names):
                        for j, f2 in enumerate(feature_names):
                            if i == j:
                                correlation_matrix[i][j] = 1.0
                            elif j > i:
                                q_f1, q_f2 = f'"{f1}"', f'"{f2}"'
                                cursor.execute(f"""
                                    SELECT CORR({q_f1}, {q_f2})
                                    FROM home_credit.application_train
                                    WHERE {q_f1} IS NOT NULL AND {q_f2} IS NOT NULL
                                """)
                                c = cursor.fetchone()
                                val = float(c[0]) if c[0] else 0.0
                                correlation_matrix[i][j] = round(val, 3)
                                correlation_matrix[j][i] = round(val, 3)
                    
                    result["statistics"] = {"correlation_matrix": correlation_matrix}
                    result["chart_data"] = {
                        "values": correlation_matrix,
                        "xLabels": feature_names,
                        "yLabels": feature_names,
                    }
                    result["chart_type"] = "heatmap"
                    result["insights"] = f"Correlation matrix for {len(feature_names)} features computed."
            else:
                result["error"] = "Need at least 2 features for correlation analysis"
        
        cursor.close()
        conn.close()
        
        logger.info(
            "analysis_completed",
            analysis_type=analysis_type,
            chart_type=result.get("chart_type"),
            num_features=len(feature_names),
        )
        
        return result
        
    except psycopg2.Error as exc:
        logger.error("database_error", error=str(exc))
        return {
            "error": f"Database query failed: {str(exc)}",
            "analysis_type": analysis_type,
        }
    except Exception as exc:
        logger.error("analysis_error", error=str(exc))
        return {
            "error": f"Analysis failed: {str(exc)}",
            "analysis_type": analysis_type,
        }


@tool
def generate_data_report(
    applicant_id: int,
    report_type: Literal["summary", "detailed", "risk_focused"] = "summary"
) -> dict[str, Any]:
    """
    Generate a comprehensive data analysis report for an applicant.
    
    Use this when user wants to understand an applicant's overall profile,
    not just the risk score. This combines multiple analyses into one report.
    
    Args:
        applicant_id: The applicant ID (SK_ID_CURR)
        report_type: 
            - "summary": Quick overview with key metrics
            - "detailed": Full analysis with all available data
            - "risk_focused": Focus on risk factors and SHAP
    
    Returns:
        Comprehensive report with multiple insights and visualizations
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Key fields for analysis
        key_fields = [
            "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
            "DAYS_BIRTH", "DAYS_EMPLOYED", "EXT_SOURCE_2", "EXT_SOURCE_3"
        ]
        
        field_list = ", ".join(f'"{f}"' for f in key_fields)
        cursor.execute(f"""
            SELECT {field_list}, "TARGET"
            FROM home_credit.application_train
            WHERE "SK_ID_CURR" = %s
        """, (applicant_id,))
        
        row = cursor.fetchone()
        if not row:
            return {"error": f"Applicant {applicant_id} not found"}
        
        applicant_data = dict(zip(key_fields + ["TARGET"], row))
        
        # Calculate derived metrics
        age_years = abs(applicant_data["DAYS_BIRTH"]) // 365 if applicant_data["DAYS_BIRTH"] else None
        employment_years = abs(applicant_data["DAYS_EMPLOYED"]) // 365 if applicant_data["DAYS_EMPLOYED"] and applicant_data["DAYS_EMPLOYED"] < 0 else 0
        
        # Get population averages for comparison
        cursor.execute(f"""
            SELECT 
                AVG("AMT_INCOME_TOTAL"),
                AVG("AMT_CREDIT"),
                AVG("AMT_ANNUITY")
            FROM home_credit.application_train
        """)
        pop_avg = cursor.fetchone()
        
        # Build report
        report = {
            "applicant_id": applicant_id,
            "report_type": report_type,
            "profile": {
                "age_years": age_years,
                "employment_years": employment_years,
                "income": applicant_data["AMT_INCOME_TOTAL"],
                "credit_amount": applicant_data["AMT_CREDIT"],
                "annuity": applicant_data["AMT_ANNUITY"],
                "actual_target": applicant_data.get("TARGET"),
            },
            "comparison_to_population": {
                "income_vs_avg": round(applicant_data["AMT_INCOME_TOTAL"] / pop_avg[0] * 100, 1) if pop_avg[0] and applicant_data["AMT_INCOME_TOTAL"] else None,
                "credit_vs_avg": round(applicant_data["AMT_CREDIT"] / pop_avg[1] * 100, 1) if pop_avg[1] and applicant_data["AMT_CREDIT"] else None,
            },
            "recommended_charts": ["gauge", "feature_importance", "radar"],
        }
        
        # Generate narrative insights
        insights = [f"**Applicant {applicant_id} Analysis Report**\n"]
        
        if age_years:
            insights.append(f"- Age: {age_years} years")
        if employment_years:
            insights.append(f"- Employment: {employment_years} years")
        
        income_pct = report["comparison_to_population"]["income_vs_avg"]
        if income_pct:
            income_status = "above" if income_pct > 100 else "below"
            insights.append(f"- Income: {abs(income_pct - 100):.0f}% {income_status} average")
        
        if report_type in ["detailed", "risk_focused"]:
            try:
                url = f"{MODEL_API_URL}/explain/applicant/{applicant_id}"
                with httpx.Client(timeout=10.0) as client:
                    response = client.get(url)
                    if response.ok:
                        risk_data = response.json()
                        report["risk_analysis"] = {
                            "probability": risk_data.get("probability"),
                            "prediction": risk_data.get("prediction"),
                            "top_factors": list(risk_data.get("shap_values", {}).items())[:5],
                        }
                        
                        prob = risk_data.get("probability", 0)
                        insights.append(f"\n**Risk Analysis**")
                        insights.append(f"- Default Probability: {prob*100:.1f}%")
                        insights.append(f"- Classification: {risk_data.get('prediction', 'N/A')}")
            except Exception as e:
                logger.warning(f"Failed to get risk prediction: {e}")
        
        report["insights"] = "\n".join(insights)
        
        cursor.close()
        conn.close()
        
        return report
        
    except Exception as exc:
        logger.error("report_generation_error", error=str(exc))
        return {"error": f"Report generation failed: {str(exc)}"}
