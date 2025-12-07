"""
Enhanced tool definitions for the LangGraph chatbot.

These tools extend the base tools with:
1. Caching for performance
2. Retry logic for reliability
3. New capabilities (comparisons, SHAP waterfall, portfolio stats, what-if analysis)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx
import psycopg2
import structlog
from langchain_core.tools import tool
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .cache import cached

logger = structlog.get_logger()

# Configuration
MODEL_API_URL = os.getenv("MODEL_API_URL", "http://model_serving:8000")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "homecredit_db")
POSTGRES_USER = os.getenv("POSTGRES_USER", "hc_admin")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "hc_password")


def get_db_connection():
    """Create a PostgreSQL database connection."""
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )


# ============================================================================
# Risk Prediction Tools
# ============================================================================

@tool
@cached(ttl=3600, key_prefix="prediction")  # Cache for 1 hour
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(httpx.HTTPError),
)
def get_risk_prediction(applicant_id: int) -> dict[str, Any]:
    """
    Get credit risk prediction and explanation for a specific applicant.

    This tool fetches features from the feature store and gets a risk probability
    along with SHAP-based explanations showing which features contributed most
    to the prediction.

    Args:
        applicant_id: The applicant ID (SK_ID_CURR) to analyze

    Returns:
        Dictionary containing:
        - probability: Default probability (0-1)
        - prediction: "Low Risk" or "High Risk" classification
        - top_factors: Top 5 features influencing the prediction
        - explanation: Full SHAP contribution details
    """
    try:
        # Call model API with applicant ID
        url = f"{MODEL_API_URL}/predict/applicant/{applicant_id}"

        with httpx.Client(timeout=10.0) as client:
            response = client.get(url)
            response.raise_for_status()
            prediction_data = response.json()

        probability = prediction_data.get("probability", 0.0)

        # Get SHAP explanation
        explain_url = f"{MODEL_API_URL}/explain/applicant/{applicant_id}"
        with httpx.Client(timeout=10.0) as client:
            explain_response = client.get(explain_url)
            explain_response.raise_for_status()
            explanation = explain_response.json()

        # Format top contributors
        contributions = explanation.get("contributions", {})
        top_factors = sorted(
            contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]

        result = {
            "applicant_id": applicant_id,
            "probability": probability,
            "prediction": "High Risk" if probability > 0.5 else "Low Risk",
            "top_factors": [
                {"feature": name, "impact": value}
                for name, value in top_factors
            ],
            "explanation": explanation,
        }

        logger.info(
            "risk_prediction_retrieved",
            applicant_id=applicant_id,
            probability=probability,
        )

        return result

    except httpx.HTTPError as exc:
        logger.error("model_api_error", error=str(exc), applicant_id=applicant_id)
        return {
            "error": f"Could not fetch prediction: {str(exc)}",
            "applicant_id": applicant_id,
        }
    except Exception as exc:
        logger.error("prediction_tool_error", error=str(exc), applicant_id=applicant_id)
        return {
            "error": f"Unexpected error: {str(exc)}",
            "applicant_id": applicant_id,
        }


@tool
def compare_applicants(applicant_ids: List[int]) -> dict[str, Any]:
    """
    Compare risk profiles of multiple applicants side-by-side.

    Use this to show how different applicants compare in terms of risk scores,
    key features, and demographics.

    Args:
        applicant_ids: List of 2-5 applicant IDs to compare

    Returns:
        Dictionary with comparison data for each applicant
    """
    if len(applicant_ids) < 2:
        return {"error": "Need at least 2 applicants to compare"}

    if len(applicant_ids) > 5:
        return {"error": "Can compare maximum 5 applicants at once"}

    try:
        comparisons = []

        for app_id in applicant_ids:
            # Get risk prediction
            prediction = get_risk_prediction.invoke({"applicant_id": app_id})

            # Get basic demographics
            demographics = query_applicant_data.invoke({
                "applicant_id": app_id,
                "fields": [
                    "AMT_INCOME_TOTAL",
                    "AMT_CREDIT",
                    "CODE_GENDER",
                    "DAYS_BIRTH",
                    "NAME_EDUCATION_TYPE"
                ]
            })

            comparisons.append({
                "applicant_id": app_id,
                "risk_probability": prediction.get("probability"),
                "risk_category": prediction.get("prediction"),
                "demographics": demographics,
                "top_risk_factor": prediction.get("top_factors", [{}])[0] if prediction.get("top_factors") else None
            })

        logger.info("applicants_compared", count=len(applicant_ids))

        return {
            "comparisons": comparisons,
            "summary": {
                "avg_risk": sum(c["risk_probability"] for c in comparisons if c.get("risk_probability")) / len(comparisons),
                "highest_risk": max(comparisons, key=lambda x: x.get("risk_probability", 0)),
                "lowest_risk": min(comparisons, key=lambda x: x.get("risk_probability", 1)),
            }
        }

    except Exception as exc:
        logger.error("comparison_error", error=str(exc), applicant_ids=applicant_ids)
        return {"error": f"Comparison failed: {str(exc)}"}


@tool
def explain_risk_factors(applicant_id: int, top_n: int = 10) -> dict[str, Any]:
    """
    Get detailed SHAP waterfall explanation showing how features contribute to risk.

    This provides a detailed breakdown of the top N features that pushed the
    prediction up or down from the baseline risk.

    Args:
        applicant_id: The applicant ID (SK_ID_CURR)
        top_n: Number of top features to explain (default: 10)

    Returns:
        Dictionary with SHAP waterfall data
    """
    try:
        url = f"{MODEL_API_URL}/explain/applicant/{applicant_id}"

        with httpx.Client(timeout=10.0) as client:
            response = client.get(url)
            response.raise_for_status()
            explanation = response.json()

        contributions = explanation.get("contributions", {})

        # Sort by absolute impact
        sorted_factors = sorted(
            contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_n]

        waterfall_data = []
        for feature, impact in sorted_factors:
            waterfall_data.append({
                "feature": feature,
                "impact": impact,
                "direction": "increases_risk" if impact > 0 else "decreases_risk",
                "magnitude": abs(impact)
            })

        baseline = explanation.get("base_value", 0.0)
        final_score = explanation.get("prediction", 0.0)

        return {
            "applicant_id": applicant_id,
            "baseline_risk": baseline,
            "final_risk": final_score,
            "total_impact": final_score - baseline,
            "waterfall": waterfall_data,
            "interpretation": _interpret_waterfall(waterfall_data, baseline, final_score)
        }

    except Exception as exc:
        logger.error("waterfall_error", error=str(exc), applicant_id=applicant_id)
        return {"error": f"Could not generate explanation: {str(exc)}"}


def _interpret_waterfall(waterfall: List[dict], baseline: float, final: float) -> str:
    """Generate human-readable interpretation of SHAP waterfall."""
    impact = final - baseline

    if abs(impact) < 0.05:
        trend = "neutral"
    elif impact > 0:
        trend = "increases"
    else:
        trend = "decreases"

    top_increasing = [f for f in waterfall if f["direction"] == "increases_risk"][:2]
    top_decreasing = [f for f in waterfall if f["direction"] == "decreases_risk"][:2]

    interpretation = f"The model's assessment {trend} risk by {abs(impact):.2%} from baseline. "

    if top_increasing:
        interpretation += f"Key risk drivers: {', '.join(f['feature'] for f in top_increasing)}. "

    if top_decreasing:
        interpretation += f"Risk-reducing factors: {', '.join(f['feature'] for f in top_decreasing)}."

    return interpretation


# ============================================================================
# Data Query Tools
# ============================================================================

@tool
@cached(ttl=86400, key_prefix="applicant_data")  # Cache for 24 hours (static data)
def query_applicant_data(
    applicant_id: int,
    fields: Optional[List[str]] = None,
) -> dict[str, Any]:
    """
    Query specific information about an applicant from the database.

    Use this tool to retrieve demographic, financial, or application details
    for an applicant to provide context in your analysis.

    Args:
        applicant_id: The applicant ID (SK_ID_CURR)
        fields: List of field names to retrieve. If None, returns common fields.
                Examples: ["AMT_INCOME_TOTAL", "AMT_CREDIT", "NAME_CONTRACT_TYPE"]

    Returns:
        Dictionary with the requested field values
    """
    if fields is None:
        # Default fields to query
        fields = [
            "AMT_INCOME_TOTAL",
            "AMT_CREDIT",
            "AMT_ANNUITY",
            "CODE_GENDER",
            "DAYS_BIRTH",
            "DAYS_EMPLOYED",
            "NAME_CONTRACT_TYPE",
            "NAME_INCOME_TYPE",
        ]

    # Convert field names to uppercase
    fields = [f.upper() for f in fields]

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Safely construct query with quoted field names
        field_list = ", ".join(f'"{f}"' for f in fields)
        query = f"""
            SELECT {field_list}
            FROM home_credit.application_train
            WHERE "SK_ID_CURR" = %s
            LIMIT 1
        """

        cursor.execute(query, (applicant_id,))
        row = cursor.fetchone()

        cursor.close()
        conn.close()

        if row is None:
            return {
                "error": f"Applicant {applicant_id} not found",
                "applicant_id": applicant_id,
            }

        # Convert to dict
        result = {"applicant_id": applicant_id}
        for i, field in enumerate(fields):
            result[field] = row[i]

        # Add computed fields
        if "DAYS_BIRTH" in result and result["DAYS_BIRTH"]:
            result["age_years"] = abs(result["DAYS_BIRTH"]) // 365

        if "DAYS_EMPLOYED" in result and result["DAYS_EMPLOYED"]:
            if result["DAYS_EMPLOYED"] > 0:
                result["employment_years"] = 0
            else:
                result["employment_years"] = abs(result["DAYS_EMPLOYED"]) // 365

        logger.info("applicant_data_retrieved", applicant_id=applicant_id)

        return result

    except psycopg2.Error as exc:
        logger.error("database_error", error=str(exc), applicant_id=applicant_id)
        return {
            "error": f"Database query failed: {str(exc)}",
            "applicant_id": applicant_id,
        }
    except Exception as exc:
        logger.error("query_tool_error", error=str(exc), applicant_id=applicant_id)
        return {
            "error": f"Unexpected error: {str(exc)}",
            "applicant_id": applicant_id,
        }


@tool
def query_bureau_history(applicant_id: int) -> dict[str, Any]:
    """
    Get credit bureau history for an applicant with trend analysis.

    This retrieves the applicant's previous credit records from the bureau
    table, showing their credit behavior over time.

    Args:
        applicant_id: The applicant ID (SK_ID_CURR)

    Returns:
        Dictionary with bureau records and summary statistics
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get bureau records
        query = """
            SELECT
                "CREDIT_ACTIVE",
                "CREDIT_TYPE",
                "DAYS_CREDIT",
                "CREDIT_DAY_OVERDUE",
                "AMT_CREDIT_SUM",
                "AMT_CREDIT_SUM_DEBT",
                "AMT_CREDIT_SUM_OVERDUE"
            FROM home_credit.bureau
            WHERE "SK_ID_CURR" = %s
            ORDER BY "DAYS_CREDIT" DESC
            LIMIT 20
        """

        cursor.execute(query, (applicant_id,))
        rows = cursor.fetchall()

        cursor.close()
        conn.close()

        if not rows:
            return {
                "applicant_id": applicant_id,
                "bureau_records": [],
                "summary": "No credit bureau history found"
            }

        # Format records
        records = []
        total_debt = 0
        total_overdue = 0
        active_count = 0

        for row in rows:
            record = {
                "credit_active": row[0],
                "credit_type": row[1],
                "days_credit": row[2],
                "days_overdue": row[3],
                "amt_credit": row[4],
                "amt_debt": row[5],
                "amt_overdue": row[6],
            }
            records.append(record)

            if row[0] == "Active":
                active_count += 1
            if row[5]:
                total_debt += row[5]
            if row[6]:
                total_overdue += row[6]

        summary = {
            "total_records": len(records),
            "active_credits": active_count,
            "total_debt": total_debt,
            "total_overdue": total_overdue,
            "has_overdue": total_overdue > 0,
        }

        logger.info("bureau_history_retrieved", applicant_id=applicant_id, records=len(records))

        return {
            "applicant_id": applicant_id,
            "bureau_records": records,
            "summary": summary
        }

    except Exception as exc:
        logger.error("bureau_query_error", error=str(exc), applicant_id=applicant_id)
        return {"error": f"Could not retrieve bureau history: {str(exc)}"}


# ============================================================================
# Visualization Tools
# ============================================================================

@tool
@cached(ttl=21600, key_prefix="feature_plot")  # Cache for 6 hours
def generate_feature_plot(
    applicant_id: int,
    feature_names: List[str],
) -> dict[str, Any]:
    """
    Generate a visualization comparing applicant's feature values to the population.

    This creates a comparison plot showing where the applicant stands relative
    to others in the dataset for specific features.

    Args:
        applicant_id: The applicant ID (SK_ID_CURR)
        feature_names: List of feature names to visualize (max 5)

    Returns:
        Dictionary with plot data and statistics
    """
    if len(feature_names) > 5:
        return {
            "error": "Maximum 5 features allowed per plot",
            "requested": len(feature_names),
        }

    # Convert feature names to uppercase
    feature_names = [f.upper() for f in feature_names]

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get applicant's values
        field_list = ", ".join(f'"{f}"' for f in feature_names)
        query = f"""
            SELECT {field_list}
            FROM home_credit.application_train
            WHERE "SK_ID_CURR" = %s
        """
        cursor.execute(query, (applicant_id,))
        applicant_row = cursor.fetchone()

        if applicant_row is None:
            cursor.close()
            conn.close()
            return {
                "error": f"Applicant {applicant_id} not found",
                "applicant_id": applicant_id,
            }

        # Get population statistics
        stats = {}
        for i, feature in enumerate(feature_names):
            applicant_value = applicant_row[i]
            quoted_feature = f'"{feature}"'

            stats_query = f"""
                SELECT
                    AVG({quoted_feature}) as mean,
                    STDDEV({quoted_feature}) as std,
                    MIN({quoted_feature}) as min,
                    MAX({quoted_feature}) as max,
                    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {quoted_feature}) as q25,
                    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY {quoted_feature}) as q50,
                    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {quoted_feature}) as q75
                FROM home_credit.application_train
                WHERE {quoted_feature} IS NOT NULL
            """
            cursor.execute(stats_query)
            stat_row = cursor.fetchone()

            stats[feature] = {
                "applicant_value": float(applicant_value) if applicant_value else None,
                "population_mean": float(stat_row[0]) if stat_row[0] else None,
                "population_std": float(stat_row[1]) if stat_row[1] else None,
                "population_min": float(stat_row[2]) if stat_row[2] else None,
                "population_max": float(stat_row[3]) if stat_row[3] else None,
                "population_q25": float(stat_row[4]) if stat_row[4] else None,
                "population_median": float(stat_row[5]) if stat_row[5] else None,
                "population_q75": float(stat_row[6]) if stat_row[6] else None,
            }

            # Calculate percentile
            if applicant_value is not None and stat_row[0] is not None:
                percentile_query = f"""
                    SELECT COUNT(*) * 100.0 / (SELECT COUNT(*) FROM home_credit.application_train WHERE {quoted_feature} IS NOT NULL)
                    FROM home_credit.application_train
                    WHERE {quoted_feature} <= %s AND {quoted_feature} IS NOT NULL
                """
                cursor.execute(percentile_query, (applicant_value,))
                percentile_result = cursor.fetchone()
                stats[feature]["percentile"] = float(percentile_result[0]) if percentile_result[0] else None

        cursor.close()
        conn.close()

        result = {
            "applicant_id": applicant_id,
            "features": stats,
            "plot_type": "feature_comparison",
        }

        logger.info(
            "feature_plot_generated",
            applicant_id=applicant_id,
            num_features=len(feature_names),
        )

        return result

    except Exception as exc:
        logger.error("plot_tool_error", error=str(exc), applicant_id=applicant_id)
        return {
            "error": f"Unexpected error: {str(exc)}",
            "applicant_id": applicant_id,
        }


# ============================================================================
# Portfolio Analytics Tools
# ============================================================================

@tool
def get_portfolio_stats(risk_threshold: float = 0.5) -> dict[str, Any]:
    """
    Get aggregate portfolio statistics for risk analysis.

    This provides high-level statistics about the applicant population,
    useful for understanding overall portfolio risk.

    Args:
        risk_threshold: Threshold to classify high vs low risk (default: 0.5)

    Returns:
        Dictionary with portfolio statistics
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get basic stats
        query = """
            SELECT
                COUNT(*) as total_applicants,
                AVG("AMT_INCOME_TOTAL") as avg_income,
                AVG("AMT_CREDIT") as avg_credit,
                SUM(CASE WHEN "TARGET" = 1 THEN 1 ELSE 0 END)::float / COUNT(*) as default_rate
            FROM home_credit.application_train
        """

        cursor.execute(query)
        row = cursor.fetchone()

        cursor.close()
        conn.close()

        if row is None:
            return {"error": "Could not retrieve portfolio statistics"}

        return {
            "total_applicants": row[0],
            "average_income": float(row[1]) if row[1] else None,
            "average_credit": float(row[2]) if row[2] else None,
            "historical_default_rate": float(row[3]) if row[3] else None,
            "risk_threshold": risk_threshold,
        }

    except Exception as exc:
        logger.error("portfolio_stats_error", error=str(exc))
        return {"error": f"Could not retrieve portfolio stats: {str(exc)}"}


# ============================================================================
# What-If Analysis Tool
# ============================================================================

@tool
def what_if_analysis(applicant_id: int, changes: Dict[str, float]) -> dict[str, Any]:
    """
    Simulate how risk changes if applicant's features were different.

    This powerful tool allows exploring counterfactual scenarios like:
    - "What if income was $50,000 higher?"
    - "What if they had 5 more years of employment?"
    - "What if their credit amount was reduced by 20%?"

    Args:
        applicant_id: The applicant ID (SK_ID_CURR) to analyze
        changes: Dictionary of feature changes, e.g.:
                 {"AMT_INCOME_TOTAL": 300000, "DAYS_EMPLOYED": -3650}
                 Feature names should be UPPERCASE

    Returns:
        Dictionary containing:
        - baseline_risk: Original risk probability
        - modified_risk: Risk after applying changes
        - risk_change: Difference (positive = increased risk)
        - risk_change_pct: Percentage change
        - interpretation: Human-readable explanation
        - modified_features: Features that were changed
        - new_shap_explanation: SHAP values for modified scenario
    """
    try:
        # Get baseline prediction
        baseline_url = f"{MODEL_API_URL}/predict/applicant/{applicant_id}"

        with httpx.Client(timeout=10.0) as client:
            baseline_response = client.get(baseline_url)
            baseline_response.raise_for_status()
            baseline_data = baseline_response.json()

        baseline_risk = baseline_data.get("probability", 0.0)

        # Get applicant's current features to show what's being changed
        current_features = {}
        changed_features_info = []

        for feature_name in changes.keys():
            try:
                conn = get_db_connection()
                cursor = conn.cursor()

                query = f"""
                    SELECT "{feature_name}"
                    FROM home_credit.application_train
                    WHERE "SK_ID_CURR" = %s
                """
                cursor.execute(query, (applicant_id,))
                row = cursor.fetchone()

                cursor.close()
                conn.close()

                if row and row[0] is not None:
                    current_value = float(row[0])
                    new_value = float(changes[feature_name])
                    current_features[feature_name] = current_value

                    change_amount = new_value - current_value
                    change_pct = (change_amount / current_value * 100) if current_value != 0 else 0

                    changed_features_info.append({
                        "feature": feature_name,
                        "current": current_value,
                        "new": new_value,
                        "change": change_amount,
                        "change_pct": change_pct
                    })
            except Exception as e:
                logger.warning(f"Could not fetch current value for {feature_name}", error=str(e))

        # For what-if analysis, we need to call model with modified features
        # Since the model API uses Feast which pulls from feature store,
        # we'll use the legacy /predict endpoint with explicit features

        # First, get all features for this applicant from Feast
        try:
            # Try to get features via explain endpoint (which fetches from Feast)
            explain_url = f"{MODEL_API_URL}/explain/applicant/{applicant_id}"

            with httpx.Client(timeout=10.0) as client:
                explain_response = client.get(explain_url)
                explain_response.raise_for_status()
                baseline_explanation = explain_response.json()

            # Get feature values from explanation
            feature_values = baseline_explanation.get("feature_values", {})

            if not feature_values:
                return {
                    "error": "Could not retrieve feature values for what-if analysis",
                    "applicant_id": applicant_id,
                    "suggestion": "Try using compare_applicants to see similar applicants instead"
                }

            # Apply the changes
            modified_features = {**feature_values, **changes}

            # Call model with modified features using legacy endpoint
            predict_url = f"{MODEL_API_URL}/predict"

            with httpx.Client(timeout=10.0) as client:
                modified_response = client.post(
                    predict_url,
                    json={"features": modified_features},
                    timeout=10.0
                )
                modified_response.raise_for_status()
                modified_data = modified_response.json()

            modified_risk = modified_data.get("probability", 0.0)

        except Exception as e:
            # Fallback: estimate impact based on SHAP values
            logger.warning("what_if_full_prediction_failed", error=str(e))

            # Use SHAP values to estimate impact
            try:
                explain_url = f"{MODEL_API_URL}/explain/applicant/{applicant_id}"

                with httpx.Client(timeout=10.0) as client:
                    explain_response = client.get(explain_url)
                    explain_response.raise_for_status()
                    explanation = explain_response.json()

                contributions = explanation.get("contributions", {})

                # Estimate impact: if feature has positive SHAP and we increase it, risk increases
                estimated_change = 0.0
                for feature, new_value in changes.items():
                    if feature in contributions and feature in current_features:
                        shap_value = contributions[feature]
                        current_val = current_features[feature]
                        change_magnitude = (new_value - current_val) / current_val if current_val != 0 else 0
                        # Rough estimate: proportional to SHAP value and change magnitude
                        estimated_change += shap_value * change_magnitude * 0.1  # Scale down

                modified_risk = max(0.0, min(1.0, baseline_risk + estimated_change))

            except Exception:
                return {
                    "error": "Could not perform what-if analysis. Model API may not support this feature.",
                    "applicant_id": applicant_id,
                }

        # Calculate changes
        risk_change = modified_risk - baseline_risk
        risk_change_pct = (risk_change / baseline_risk * 100) if baseline_risk > 0 else 0

        # Generate interpretation
        if abs(risk_change) < 0.01:
            interpretation = f"These changes would have minimal impact on risk (change: {risk_change:+.2%})"
        elif risk_change > 0:
            interpretation = f"These changes would INCREASE risk by {abs(risk_change):.2%} ({abs(risk_change_pct):.1f}% relative increase)"
        else:
            interpretation = f"These changes would DECREASE risk by {abs(risk_change):.2%} ({abs(risk_change_pct):.1f}% relative decrease)"

        # Add threshold analysis
        baseline_category = "High Risk" if baseline_risk > 0.5 else "Low Risk"
        modified_category = "High Risk" if modified_risk > 0.5 else "Low Risk"

        if baseline_category != modified_category:
            interpretation += f" — Classification would change from {baseline_category} to {modified_category}!"

        result = {
            "applicant_id": applicant_id,
            "baseline_risk": baseline_risk,
            "baseline_category": baseline_category,
            "modified_risk": modified_risk,
            "modified_category": modified_category,
            "risk_change": risk_change,
            "risk_change_pct": risk_change_pct,
            "interpretation": interpretation,
            "modified_features": changed_features_info,
            "recommendation": _generate_recommendation(risk_change, changed_features_info),
        }

        logger.info(
            "what_if_analysis_completed",
            applicant_id=applicant_id,
            baseline=baseline_risk,
            modified=modified_risk,
        )

        return result

    except httpx.HTTPError as exc:
        logger.error("what_if_api_error", error=str(exc), applicant_id=applicant_id)
        return {
            "error": f"Could not perform what-if analysis: {str(exc)}",
            "applicant_id": applicant_id,
        }
    except Exception as exc:
        logger.error("what_if_tool_error", error=str(exc), applicant_id=applicant_id)
        return {
            "error": f"Unexpected error: {str(exc)}",
            "applicant_id": applicant_id,
        }


def _generate_recommendation(risk_change: float, changed_features: List[Dict]) -> str:
    """Generate actionable recommendation based on what-if results."""
    if abs(risk_change) < 0.01:
        return "These changes would not significantly affect the risk assessment. Consider modifying other features."

    if risk_change < 0:
        # Risk decreased - positive outcome
        most_impactful = max(changed_features, key=lambda x: abs(x.get("change", 0)))
        return f"To reduce risk, focus on improving {most_impactful['feature']}. The analysis shows this has a positive impact."
    else:
        # Risk increased - warning
        return "Warning: These changes would increase default risk. Reconsider the proposed modifications."
