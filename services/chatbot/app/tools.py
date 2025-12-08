"""
Tool definitions for the LangGraph chatbot.

These tools allow the chatbot to:
1. Get risk predictions from the model API
2. Query applicant data from PostgreSQL
3. Generate visualizations
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import psycopg2
import structlog
from langchain_core.tools import tool

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


@tool
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
        # In a full implementation, this would fetch features from Feast
        # and pass them to the model
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
def query_applicant_data(
    applicant_id: int,
    fields: list[str] | None = None,
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

    # Convert field names to uppercase and quote them (PostgreSQL columns are uppercase)
    fields = [f.upper() for f in fields]

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Safely construct query with quoted field names (PostgreSQL requires quotes for uppercase)
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

        # Add some computed/formatted fields for better UX
        if "DAYS_BIRTH" in result and result["DAYS_BIRTH"]:
            result["age_years"] = abs(result["DAYS_BIRTH"]) // 365

        if "DAYS_EMPLOYED" in result and result["DAYS_EMPLOYED"]:
            if result["DAYS_EMPLOYED"] > 0:
                result["employment_years"] = 0  # unemployed
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
def generate_feature_plot(
    applicant_id: int,
    feature_names: list[str],
) -> dict[str, Any]:
    """
    Generate a visualization comparing applicant's feature values to the population.

    This creates a comparison plot showing where the applicant stands relative
    to others in the dataset for specific features.

    Args:
        applicant_id: The applicant ID (SK_ID_CURR)
        feature_names: List of feature names to visualize (max 15)
                      Feature names are case-insensitive (will be converted to uppercase)

    Returns:
        Dictionary with plot data and statistics
    """
    if len(feature_names) > 15:
        return {
            "error": "Maximum 15 features allowed per plot",
            "requested": len(feature_names),
        }

    # Convert feature names to uppercase and prepare quoted versions for SQL
    feature_names = [f.upper() for f in feature_names]

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get applicant's values (quote column names for PostgreSQL uppercase)
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

        # Get population statistics for each feature
        stats = {}
        for i, feature in enumerate(feature_names):
            applicant_value = applicant_row[i]
            quoted_feature = f'"{feature}"'

            # Get population stats
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

    except psycopg2.Error as exc:
        logger.error("database_error", error=str(exc), applicant_id=applicant_id)
        return {
            "error": f"Database query failed: {str(exc)}",
            "applicant_id": applicant_id,
        }
    except Exception as exc:
        logger.error("plot_tool_error", error=str(exc), applicant_id=applicant_id)
        return {
            "error": f"Unexpected error: {str(exc)}",
            "applicant_id": applicant_id,
        }


@tool
def explain_shap_values(applicant_id: int) -> dict[str, Any]:
    """
    Get SHAP values for an applicant and explain them in natural language.

    This tool is designed to help normal users understand model predictions
    by translating SHAP values into plain English explanations about which
    factors increase or decrease credit risk and why.

    Args:
        applicant_id: The applicant ID (SK_ID_CURR) to explain

    Returns:
        Dictionary containing:
        - shap_explanation: Natural language explanation of SHAP values
        - top_factors: List of top factors with human-readable impact
        - probability: Default probability
        - detailed_contributions: Full SHAP contributions by feature
    """
    try:
        # Get SHAP explanation from model API
        explain_url = f"{MODEL_API_URL}/explain/applicant/{applicant_id}"
        with httpx.Client(timeout=10.0) as client:
            explain_response = client.get(explain_url)
            explain_response.raise_for_status()
            explanation = explain_response.json()

        # Get prediction
        pred_url = f"{MODEL_API_URL}/predict/applicant/{applicant_id}"
        with httpx.Client(timeout=10.0) as client:
            pred_response = client.get(pred_url)
            pred_response.raise_for_status()
            prediction_data = pred_response.json()

        probability = prediction_data.get("probability", 0.0)

        # Get SHAP values
        shap_values = explanation.get("shap_values", {})

        # Sort by absolute contribution
        sorted_factors = sorted(
            shap_values.items(),
            key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0,
            reverse=True
        )[:10]

        # Build natural language explanation
        explanation_parts = []
        explanation_parts.append(
            f"**Credit Risk Analysis for Applicant {applicant_id}**\n"
        )
        explanation_parts.append(
            f"The model predicts a **{probability*100:.1f}% chance of default** "
            f"({'HIGH RISK' if probability > 0.5 else 'LOW RISK'}).\n"
        )

        # Get baseline probability if available
        base_prob = explanation.get("base_probability", 0.08)  # Default to ~8% if not available
        explanation_parts.append(
            f"The baseline probability for an average applicant is {base_prob*100:.1f}%.\n"
        )

        explanation_parts.append("\n**Key Factors Affecting This Prediction:**\n")

        # Explain top factors in human language with probability contributions
        for i, (feature, impact) in enumerate(sorted_factors[:5], 1):
            if not isinstance(impact, (int, float)):
                continue

            direction = "INCREASES" if impact > 0 else "DECREASES"
            # Convert impact to percentage points (impact is already in probability space: 0-1)
            impact_pct = abs(impact) * 100

            # Determine strength based on percentage point contribution
            strength = "strongly" if impact_pct > 10 else "moderately" if impact_pct > 5 else "slightly"

            # Make feature names more readable
            readable_name = feature.replace("_", " ").title()

            explanation_parts.append(
                f"{i}. **{readable_name}** {strength} {direction} risk "
                f"(impact: {impact:+.1%} or {impact*100:+.1f} percentage points)"
            )

        explanation_parts.append(
            f"\n\n**Understanding the Impact Values:**\n"
            "Impact values show how much each feature contributes to the default probability "
            "compared to the baseline. For example, an impact of +5 percentage points means "
            "this feature increases the default probability from the baseline by 5%. "
            "These values are calibrated SHAP (SHapley Additive exPlanations) contributions "
            "in probability space for accurate interpretation."
        )

        natural_explanation = "\n".join(explanation_parts)

        # Format top factors with readable impact (in probability space)
        top_factors_formatted = [
            {
                "feature": feature.replace("_", " ").title(),
                "impact": float(impact) if isinstance(impact, (int, float)) else 0,  # Probability contribution (0-1 scale)
                "impact_percentage_points": float(impact * 100) if isinstance(impact, (int, float)) else 0,  # As percentage points
                "direction": "increases_risk" if impact > 0 else "decreases_risk",
                "strength": "strong" if abs(impact) * 100 > 10 else "moderate" if abs(impact) * 100 > 5 else "slight"
            }
            for feature, impact in sorted_factors
            if isinstance(impact, (int, float))
        ]

        result = {
            "applicant_id": applicant_id,
            "probability": probability,
            "risk_level": "HIGH RISK" if probability > 0.5 else "LOW RISK",
            "shap_explanation": natural_explanation,
            "top_factors": top_factors_formatted,
            "detailed_contributions": dict(sorted_factors),
        }

        logger.info(
            "shap_explanation_generated",
            applicant_id=applicant_id,
            probability=probability,
            num_factors=len(top_factors_formatted)
        )

        return result

    except httpx.HTTPError as exc:
        logger.error("shap_explain_api_error", error=str(exc), applicant_id=applicant_id)
        return {
            "error": f"Could not fetch SHAP explanation: {str(exc)}",
            "applicant_id": applicant_id,
        }
    except Exception as exc:
        logger.error("shap_explain_tool_error", error=str(exc), applicant_id=applicant_id)
        return {
            "error": f"Unexpected error: {str(exc)}",
            "applicant_id": applicant_id,
        }


@tool
def predict_hypothetical_applicant(
    income: float,
    credit_amount: float,
    age: int,
    employment_years: int = 0,
    education: str = "Higher education",
    has_car: bool = False,
    has_realty: bool = True,
    children_count: int = 0,
    ext_source_2: float | None = None,
    ext_source_3: float | None = None,
) -> dict[str, Any]:
    """
    Predict credit risk for a hypothetical applicant profile.
    
    Use this when user wants to test "what if" scenarios without an existing
    applicant ID. Great for exploring how different factors affect risk.
    
    Args:
        income: Annual income in the local currency
        credit_amount: Requested loan/credit amount
        age: Applicant age in years
        employment_years: Years at current job (0 if unemployed)
        education: Education level (e.g., "Higher education", "Secondary")
        has_car: Whether applicant owns a car
        has_realty: Whether applicant owns real estate
        children_count: Number of children
        ext_source_2: External credit score 2 (0-1, optional)
        ext_source_3: External credit score 3 (0-1, optional)
    
    Returns:
        Prediction with probability, risk level, and SHAP explanation
    """
    try:
        # Build features dictionary
        features = {
            "AMT_INCOME_TOTAL": float(income),
            "AMT_CREDIT": float(credit_amount),
            "DAYS_BIRTH": -age * 365,  # Convert age to days (negative)
            "DAYS_EMPLOYED": -employment_years * 365 if employment_years > 0 else 365243,  # Special value for unemployed
            "CNT_CHILDREN": children_count,
            "FLAG_OWN_CAR": 1 if has_car else 0,
            "FLAG_OWN_REALTY": 1 if has_realty else 0,
        }
        
        # Add external sources if provided (these are the most important features!)
        if ext_source_2 is not None:
            features["EXT_SOURCE_2"] = float(ext_source_2)
        if ext_source_3 is not None:
            features["EXT_SOURCE_3"] = float(ext_source_3)
        
        # Calculate derived features
        if credit_amount > 0:
            features["CREDIT_INCOME_RATIO"] = credit_amount / income if income > 0 else 10
        
        # Call the hypothetical prediction endpoint
        url = f"{MODEL_API_URL}/predict/hypothetical"
        payload = {
            "features": features,
            "name": f"Hypothetical (Age {age}, Income {income:,.0f})",
            "fill_missing_with_median": True,
        }
        
        with httpx.Client(timeout=15.0) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
        
        # Enhance the response with human-readable summary
        probability = result.get("probability", 0)
        risk_level = result.get("risk_level", "Unknown")
        
        summary = f"""**Hypothetical Profile Analysis**

**Input:**
- Income: {income:,.0f}
- Credit Amount: {credit_amount:,.0f}
- Age: {age}
- Employment: {employment_years} years
- Has Car: {'Yes' if has_car else 'No'}
- Has Realty: {'Yes' if has_realty else 'No'}

**Prediction:**
- Default Probability: **{probability*100:.1f}%**
- Classification: **{risk_level} Risk**
- Debt-to-Income Ratio: {credit_amount/income:.1f}x
"""
        
        result["summary"] = summary
        result["input_features"] = features
        
        logger.info(
            "hypothetical_prediction_completed",
            probability=probability,
            risk_level=risk_level,
        )
        
        return result
        
    except httpx.HTTPError as exc:
        logger.error("hypothetical_api_error", error=str(exc))
        return {
            "error": f"Could not get prediction: {str(exc)}",
            "suggestion": "Make sure model_serving is running",
        }
    except Exception as exc:
        logger.error("hypothetical_tool_error", error=str(exc))
        return {
            "error": f"Unexpected error: {str(exc)}",
        }
