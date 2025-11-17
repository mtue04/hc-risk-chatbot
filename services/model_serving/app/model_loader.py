import joblib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

from app.config import settings

logger = logging.getLogger(__name__)


class ModelLoader:    
    _instance = None
    _model = None
    _metadata = None
    _feature_names = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance
    
    def load_model(self) -> Any:
        if self._model is None:
            model_path = settings.model_dir / settings.model_filename
            
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model file not found: {model_path}. "
                    f"Please ensure the model is trained and saved."
                )
            
            logger.info(f"Loading model from {model_path}")
            self._model = joblib.load(model_path)
            logger.info("Model loaded successfully")
        
        return self._model
    
    def load_metadata(self) -> Dict[str, Any]:
        if self._metadata is None:
            metadata_path = settings.model_dir / settings.metadata_filename
            
            if metadata_path.exists():
                logger.info(f"Loading metadata from {metadata_path}")
                with open(metadata_path, 'r') as f:
                    raw_metadata = json.load(f)
                    
                self._metadata = {
                    "model_name": "HomeCredit LightGBM Tuned Model",
                    "model_version": raw_metadata.get("training_info", {}).get("timestamp", "1.0.0"),
                    "model_type": raw_metadata.get("model_info", {}).get("model_type", "LightGBM Classifier"),
                    "trained_at": raw_metadata.get("training_info", {}).get("training_date"),
                    "best_iteration": raw_metadata.get("model_info", {}).get("best_iteration"),
                    "n_features": raw_metadata.get("model_info", {}).get("n_features", 170),
                    "training_samples": raw_metadata.get("model_info", {}).get("training_samples"),
                    "performance_metrics": raw_metadata.get("performance", {}),
                    "hyperparameters": raw_metadata.get("training_info", {}).get("final_parameters", {}),
                    "feature_importance": raw_metadata.get("feature_importance", {}),
                    "optimization_info": raw_metadata.get("optimization", {})
                }
            else:
                logger.warning(f"Metadata file not found: {metadata_path}")
                self._metadata = {
                    "model_name": "HomeCredit LightGBM Tuned Model",
                    "model_version": "1.0.0",
                    "model_type": "LightGBM Classifier",
                    "n_features": 170
                }
        
        return self._metadata
    
    def load_feature_names(self) -> List[str]:
        if self._feature_names is None:
            feature_path = settings.model_dir / settings.feature_names_filename
            
            if not feature_path.exists():
                raise FileNotFoundError(
                    f"Feature names file not found: {feature_path}"
                )
            
            logger.info(f"Loading feature names from {feature_path}")
            with open(feature_path, 'r') as f:
                self._feature_names = [line.strip() for line in f if line.strip()]
            
            logger.info(f"Loaded {len(self._feature_names)} feature names")
        
        return self._feature_names
    
    def get_model_info(self) -> Dict[str, Any]:
        metadata = self.load_metadata()
        feature_names = self.load_feature_names()
        
        return {
            **metadata,
            "feature_count": len(feature_names),
            "feature_names": feature_names
        }
    
    def validate_features(self, features: Dict[str, float]) -> tuple[bool, Optional[str]]:
        """
        Validate input features against expected feature names
        
        Returns:
            (is_valid, error_message)
        """
        expected_features = set(self.load_feature_names())
        provided_features = set(features.keys())
        
        missing = expected_features - provided_features
        if missing:
            metadata = self.load_metadata()
            top_features = []
            if "feature_importance" in metadata and "top_20_features" in metadata["feature_importance"]:
                top_features = [f["feature"] for f in metadata["feature_importance"]["top_20_features"]]
            
            important_missing = [f for f in top_features if f in missing]
            
            if important_missing:
                return False, (
                    f"Missing {len(missing)} required features, including important ones: "
                    f"{important_missing[:5]}. Model requires all 170 features."
                )
            else:
                return False, (
                    f"Missing {len(missing)} required features: {sorted(list(missing))[:10]}. "
                    f"Model requires all 170 features."
                )
        
        extra = provided_features - expected_features
        if extra:
            logger.warning(f"Unexpected features provided (will be ignored): {list(extra)[:10]}")
        
        return True, None
    
    def prepare_features(self, features: Dict[str, float]) -> List[float]:
        """
        Prepare features in the correct order for model prediction
        
        Args:
            features: Dictionary of feature name -> value
        
        Returns:
            List of feature values in correct order
        """
        feature_names = self.load_feature_names()
        
        feature_values = [features.get(name, 0.0) for name in feature_names]
        
        return feature_values
    
    def reload_model(self):
        logger.info("Reloading model and metadata...")
        self._model = None
        self._metadata = None
        self._feature_names = None
        
        self.load_model()
        self.load_metadata()
        self.load_feature_names()
        
        logger.info("Model reloaded successfully")


model_loader = ModelLoader()