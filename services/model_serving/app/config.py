from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):    
    app_name: str = "HomeCredit Risk Model API"
    app_version: str = "1.0.0"
    api_prefix: str = "/api/v1"
    debug: bool = False
    
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    
    model_dir: Path = Path("/app/models/tuned")
    model_filename: str = "tuned_lgbm_model.pkl"
    metadata_filename: str = "tuned_lgbm_model_metadata.json"
    feature_names_filename: str = "tuned_lgbm_model_feature_names.txt"
    
    default_threshold: float = 0.5
    enable_shap: bool = True
    shap_sample_size: int = 100
    max_shap_features: int = 20
    
    prediction_timeout: float = 30.0
    max_batch_size: int = 1000
    
    enable_metrics: bool = True
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()