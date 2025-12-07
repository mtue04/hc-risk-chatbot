"""
Feast client for fetching features from the feature store.

This module provides a wrapper around Feast FeatureStore to fetch
credit risk features for model inference.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

try:
    from feast import FeatureStore
    FEAST_AVAILABLE = True
except ImportError:
    FEAST_AVAILABLE = False
    logging.warning("Feast not installed. Feature store integration disabled.")


logger = logging.getLogger(__name__)


class FeastClient:
    """Client for fetching features from Feast feature store."""

    def __init__(self, repo_path: Optional[str] = None):
        """
        Initialize Feast client.

        Args:
            repo_path: Path to Feast feature repository.
                      Defaults to /feast/feature_repo if not specified.
        """
        if not FEAST_AVAILABLE:
            logger.warning("Feast not available, feature store disabled")
            self.store = None
            return

        if repo_path is None:
            repo_path = os.getenv("FEAST_REPO_PATH", "/feast/feature_repo")

        self.repo_path = Path(repo_path)

        if not self.repo_path.exists():
            logger.warning(f"Feast repo not found at {repo_path}, feature store disabled")
            self.store = None
            return

        try:
            self.store = FeatureStore(repo_path=str(self.repo_path))
            logger.info(f"Feast feature store initialized from {repo_path}")
        except Exception as e:
            logger.error(f"Failed to initialize Feast: {e}")
            self.store = None

    def is_available(self) -> bool:
        """Check if Feast feature store is available."""
        return self.store is not None

    def get_features(
        self,
        applicant_id: int,
        feature_service: str = "credit_risk_model_v1"
    ) -> Optional[Dict[str, float]]:
        """
        Fetch features for a specific applicant from online store.

        Args:
            applicant_id: SK_ID_CURR of the applicant
            feature_service: Name of the feature service to use

        Returns:
            Dictionary mapping feature names to values, or None if unavailable
        """
        if not self.is_available():
            logger.warning("Feast not available, cannot fetch features")
            return None

        try:
            # Fetch features from online store
            entity_rows = [{"SK_ID_CURR": int(applicant_id)}]

            # Get feature service object
            fs = self.store.get_feature_service(feature_service)
            features_response = self.store.get_online_features(
                features=fs,
                entity_rows=entity_rows,
            )

            # Convert to dictionary
            features_dict = features_response.to_dict()

            # Remove SK_ID_CURR from features
            if "SK_ID_CURR" in features_dict:
                del features_dict["SK_ID_CURR"]

            # Extract values (Feast returns lists, we want first element)
            # Replace None/NaN with 0.0 for model compatibility
            result = {}
            for k, v in features_dict.items():
                val = v[0] if isinstance(v, list) else v
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    result[k] = 0.0
                else:
                    result[k] = float(val)

            logger.info(f"Fetched {len(result)} features for applicant {applicant_id}")
            return result

        except Exception as e:
            logger.error(f"Error fetching features from Feast: {e}")
            return None

    def get_features_batch(
        self,
        applicant_ids: List[int],
        feature_service: str = "credit_risk_model_v1"
    ) -> Optional[List[Dict[str, float]]]:
        """
        Fetch features for multiple applicants from online store.

        Args:
            applicant_ids: List of SK_ID_CURR values
            feature_service: Name of the feature service to use

        Returns:
            List of feature dictionaries, or None if unavailable
        """
        if not self.is_available():
            logger.warning("Feast not available, cannot fetch features")
            return None

        try:
            entity_rows = [{"SK_ID_CURR": int(aid)} for aid in applicant_ids]

            # Get feature service object
            fs = self.store.get_feature_service(feature_service)
            features_response = self.store.get_online_features(
                features=fs,
                entity_rows=entity_rows,
            )

            # Convert to list of dictionaries
            features_df = features_response.to_df()

            # Remove SK_ID_CURR
            if "SK_ID_CURR" in features_df.columns:
                features_df = features_df.drop(columns=["SK_ID_CURR"])

            # Fill NaN/None with 0.0 for model compatibility
            features_df = features_df.fillna(0.0)

            # Convert to list of dicts
            result = features_df.to_dict("records")

            logger.info(f"Fetched features for {len(applicant_ids)} applicants")
            return result

        except Exception as e:
            logger.error(f"Error fetching batch features from Feast: {e}")
            return None

    def get_feature_names(self, feature_service: str = "credit_risk_model_v1") -> List[str]:
        """
        Get list of feature names from the feature service.

        Args:
            feature_service: Name of the feature service

        Returns:
            List of feature names
        """
        if not self.is_available():
            return []

        try:
            # Get feature service definition
            fs = self.store.get_feature_service(feature_service)

            # Extract feature names
            feature_names = []
            for feature_view_projection in fs.feature_view_projections:
                for feature in feature_view_projection.features:
                    feature_names.append(feature.name)

            return feature_names

        except Exception as e:
            logger.error(f"Error getting feature names: {e}")
            return []


# Global Feast client instance
_feast_client: Optional[FeastClient] = None


def get_feast_client() -> FeastClient:
    """Get or create global Feast client instance."""
    global _feast_client
    if _feast_client is None:
        _feast_client = FeastClient()
    return _feast_client
