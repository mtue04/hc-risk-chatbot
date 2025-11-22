"""
Feast feature repository for Home Credit Risk model.

This repository defines:
- Entities: applicant (identified by SK_ID_CURR)
- Data sources: PostgreSQL features table
- Feature views: credit_risk_features (170 features)
- Feature services: credit_risk_model_v1
"""

# Import all feature definitions to register with Feast
from entities.applicant import applicant
from data_sources.features_source import features_source
from feature_views.credit_risk_features import credit_risk_features
from feature_services.model_features import credit_risk_model_v1

__all__ = [
    "applicant",
    "features_source",
    "credit_risk_features",
    "credit_risk_model_v1",
]
