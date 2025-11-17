import numpy as np
from typing import Dict, List, Tuple
import logging

from app.config import settings
from app.model_loader import model_loader

logger = logging.getLogger(__name__)


class RiskPredictor:
    """Handles model predictions and risk categorization"""
    
    def __init__(self, threshold: float = None):
        """
        Initialize predictor with classification threshold
        
        Args:
            threshold: Probability threshold for binary classification
        """
        self.threshold = threshold or settings.default_threshold
        self.model = model_loader.load_model()
    
    def predict_proba(self, features: Dict[str, float]) -> float:
        """
        Predict probability of default for a single application
        
        Args:
            features: Dictionary of feature name -> value
        
        Returns:
            Probability of default (class 1)
        """
        is_valid, error_msg = model_loader.validate_features(features)
        if not is_valid:
            raise ValueError(error_msg)
        
        feature_values = model_loader.prepare_features(features)
        
        X = np.array(feature_values).reshape(1, -1)
        
        proba = self.model.predict_proba(X)[0, 1]
        
        return float(proba)
    
    def predict(self, features: Dict[str, float]) -> Tuple[int, float]:
        """
        Predict binary class and probability
        
        Args:
            features: Dictionary of feature name -> value
        
        Returns:
            (prediction, probability) tuple
        """
        proba = self.predict_proba(features)
        prediction = 1 if proba >= self.threshold else 0
        
        return prediction, proba
    
    def predict_batch(
        self,
        features_list: List[Dict[str, float]]
    ) -> List[Tuple[int, float]]:
        """
        Predict for multiple applications
        
        Args:
            features_list: List of feature dictionaries
        
        Returns:
            List of (prediction, probability) tuples
        """
        if len(features_list) > settings.max_batch_size:
            raise ValueError(
                f"Batch size {len(features_list)} exceeds maximum "
                f"{settings.max_batch_size}"
            )
        
        results = []
        for features in features_list:
            try:
                pred, proba = self.predict(features)
                results.append((pred, proba))
            except Exception as e:
                logger.error(f"Prediction failed for features: {e}")
                results.append((None, None))
        
        return results
    
    @staticmethod
    def categorize_risk(probability: float) -> str:
        """
        Categorize risk level based on default probability
        
        Risk levels based on HomeCredit business context:
        - LOW: < 0.15 (much lower than baseline 8.07%)
        - MEDIUM: 0.15 - 0.30 (around baseline to moderate)
        - HIGH: 0.30 - 0.50 (significantly elevated)
        - CRITICAL: >= 0.50 (very high likelihood)
        
        Note: Default threshold from model is around 0.5 (from F1 optimization)
        These categories provide more granular risk assessment for business
        
        Args:
            probability: Probability of default
        
        Returns:
            Risk category string
        """
        if probability < 0.15:
            return "LOW"
        elif probability < 0.30:
            return "MEDIUM"
        elif probability < 0.50:
            return "HIGH"
        else:
            return "CRITICAL"
    
    def predict_with_risk(
        self,
        features: Dict[str, float]
    ) -> Dict[str, any]:
        """
        Complete prediction with risk assessment
        
        Args:
            features: Dictionary of feature name -> value
        
        Returns:
            Dictionary with prediction, probability, risk_level
        """
        prediction, probability = self.predict(features)
        risk_level = self.categorize_risk(probability)
        
        return {
            "prediction": prediction,
            "probability": probability,
            "risk_level": risk_level,
            "threshold": self.threshold
        }


risk_predictor = RiskPredictor()