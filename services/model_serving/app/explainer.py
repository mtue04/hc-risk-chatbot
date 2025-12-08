import shap
import numpy as np
import pandas as pd
from scipy.special import expit  # sigmoid function for log-odds to probability conversion
from typing import Dict, List, Tuple, Any
import logging

from app.config import settings
from app.model_loader import model_loader

logger = logging.getLogger(__name__)


class ShapExplainer:    
    _instance = None
    _explainer = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ShapExplainer, cls).__new__(cls)
        return cls._instance
    
    def _init_explainer(self):
        if self._explainer is None and settings.enable_shap:
            logger.info("Initializing SHAP explainer...")
            model = model_loader.load_model()
            
            self._explainer = shap.TreeExplainer(model)
            logger.info("SHAP explainer initialized")
    
    def explain(
        self,
        features: Dict[str, float],
        top_n: int = None
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanation for a single prediction with probability-calibrated contributions.

        This method converts SHAP values from log-odds space to probability space,
        making them interpretable as percentage point contributions to default probability.

        Args:
            features: Feature dictionary
            top_n: Number of top features to include

        Returns:
            Dictionary with probability-calibrated SHAP values and interpretation
        """
        if not settings.enable_shap:
            return {"error": "SHAP explanations are disabled"}

        self._init_explainer()
        top_n = top_n or settings.max_shap_features
        feature_names = model_loader.load_feature_names()
        feature_values = model_loader.prepare_features(features)
        X = np.array(feature_values).reshape(1, -1)

        # Get SHAP values in log-odds space
        shap_values = self._explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Class 1 (default)

        base_value = self._explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1]

        # Convert base value from log-odds to probability
        base_probability = float(expit(base_value))

        # Calculate probability contributions for each feature
        # For each feature, we calculate: P(base + shap_i) - P(base)
        contributions_logodds = dict(zip(feature_names, shap_values[0]))
        contributions_probability = {}

        for feature, shap_logodds in contributions_logodds.items():
            # Probability with this feature's contribution
            prob_with_feature = expit(base_value + shap_logodds)
            # Marginal contribution in probability space (percentage points)
            prob_contribution = prob_with_feature - base_probability
            contributions_probability[feature] = prob_contribution

        # Sort by absolute probability contribution
        sorted_contributions = sorted(
            contributions_probability.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_n]

        # Keep both log-odds and probability contributions for reference
        positive_features = [
            {
                "feature": k,
                "value": features.get(k, 0.0),
                "impact": v,  # Probability contribution
                "impact_logodds": contributions_logodds.get(k, 0.0)  # Original log-odds
            }
            for k, v in sorted_contributions if v > 0
        ]

        negative_features = [
            {
                "feature": k,
                "value": features.get(k, 0.0),
                "impact": v,  # Probability contribution
                "impact_logodds": contributions_logodds.get(k, 0.0)  # Original log-odds
            }
            for k, v in sorted_contributions if v < 0
        ]

        explanation_text = self._generate_explanation_text(
            positive_features,
            negative_features,
            base_probability
        )

        return {
            "base_value": float(base_value),  # Log-odds base value (for reference)
            "base_probability": base_probability,  # Baseline probability
            "shap_values": {k: float(v) for k, v in sorted_contributions},  # Probability contributions
            "shap_values_logodds": {k: float(contributions_logodds[k]) for k, _ in sorted_contributions},  # Original log-odds
            "top_positive_features": positive_features[:5],
            "top_negative_features": negative_features[:5],
            "explanation_text": explanation_text
        }
    
    def _generate_explanation_text(
        self,
        positive_features: List[Dict],
        negative_features: List[Dict],
        base_probability: float
    ) -> str:
        """
        Generate human-readable explanation with probability-calibrated impacts.

        Args:
            positive_features: Features that increase default probability
            negative_features: Features that decrease default probability
            base_probability: Baseline probability (0-1 scale)

        Returns:
            Human-readable explanation text
        """

        explanation_parts = [
            f"The model's baseline default probability is {base_probability*100:.1f}%."
        ]

        if positive_features:
            top_pos = positive_features[0]
            explanation_parts.append(
                f"The strongest factor INCREASING default risk is "
                f"'{top_pos['feature']}' (value: {top_pos['value']:.2f}, "
                f"impact: +{top_pos['impact']*100:.1f} percentage points)."
            )

        if negative_features:
            top_neg = negative_features[0]
            explanation_parts.append(
                f"The strongest factor DECREASING default risk is "
                f"'{top_neg['feature']}' (value: {top_neg['value']:.2f}, "
                f"impact: {top_neg['impact']*100:.1f} percentage points)."
            )

        if len(positive_features) > 1:
            other_pos = [f["feature"] for f in positive_features[1:4]]
            explanation_parts.append(
                f"Other risk-increasing factors: {', '.join(other_pos)}."
            )

        if len(negative_features) > 1:
            other_neg = [f["feature"] for f in negative_features[1:4]]
            explanation_parts.append(
                f"Other risk-decreasing factors: {', '.join(other_neg)}."
            )

        return " ".join(explanation_parts)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get global feature importance from model metadata
        
        Returns:
            Dictionary of feature -> importance score (from top 20)
        """
        if not settings.enable_shap:
            return {}
        
        metadata = model_loader.load_metadata()
        
        if "feature_importance" in metadata and "top_20_features" in metadata["feature_importance"]:
            top_features = metadata["feature_importance"]["top_20_features"]
            return {
                item["feature"]: item["importance"] 
                for item in top_features
            }
        
        return {}


shap_explainer = ShapExplainer()