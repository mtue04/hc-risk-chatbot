"""
Feature service for the Home Credit risk model.

This service groups all 170 features needed by the LightGBM model
for credit default risk prediction.
"""

from feast import FeatureService

from feature_views.credit_risk_features import credit_risk_features

# Feature service for model inference
credit_risk_model_v1 = FeatureService(
    name="credit_risk_model_v1",
    features=[credit_risk_features],
    description="Feature service for LightGBM credit risk model (tuned version, 170 features)",
    tags={"model_version": "1.0", "model_type": "lightgbm"},
)
