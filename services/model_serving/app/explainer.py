import shap
import numpy as np
import pandas as pd
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
        Generate SHAP explanation for a single prediction
        
        Args:
            features: Feature dictionary
            top_n: Number of top features to include
        
        Returns:
            Dictionary with SHAP values and interpretation
        """
        if not settings.enable_shap:
            return {"error": "SHAP explanations are disabled"}
        
        self._init_explainer()
        top_n = top_n or settings.max_shap_features
        feature_names = model_loader.load_feature_names()
        feature_values = model_loader.prepare_features(features)
        X = np.array(feature_values).reshape(1, -1)

        shap_values = self._explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Class 1 (default)
        
        base_value = self._explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1]
        
        contributions = dict(zip(feature_names, shap_values[0]))
        sorted_contributions = sorted(
            contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_n]
        
        positive_features = [
            {"feature": k, "value": features.get(k, 0.0), "impact": v}
            for k, v in sorted_contributions if v > 0
        ]
        
        negative_features = [
            {"feature": k, "value": features.get(k, 0.0), "impact": v}
            for k, v in sorted_contributions if v < 0
        ]
        
        explanation_text = self._generate_explanation_text(
            positive_features,
            negative_features,
            base_value
        )
        
        return {
            "base_value": float(base_value),
            "shap_values": {k: float(v) for k, v in sorted_contributions},
            "top_positive_features": positive_features[:5],
            "top_negative_features": negative_features[:5],
            "explanation_text": explanation_text
        }
    
    def _generate_explanation_text(
        self,
        positive_features: List[Dict],
        negative_features: List[Dict],
        base_value: float
    ) -> str:
        """Generate human-readable explanation"""
        
        explanation_parts = [
            f"The model's baseline prediction is {base_value:.3f}."
        ]
        
        if positive_features:
            top_pos = positive_features[0]
            explanation_parts.append(
                f"The strongest factor INCREASING default risk is "
                f"'{top_pos['feature']}' (value: {top_pos['value']:.2f}, "
                f"impact: +{top_pos['impact']:.4f})."
            )
        
        if negative_features:
            top_neg = negative_features[0]
            explanation_parts.append(
                f"The strongest factor DECREASING default risk is "
                f"'{top_neg['feature']}' (value: {top_neg['value']:.2f}, "
                f"impact: {top_neg['impact']:.4f})."
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