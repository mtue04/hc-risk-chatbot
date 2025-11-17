from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Dict, List, Optional, Any
from datetime import datetime


# ============================================
# Request Models
# ============================================

class PredictionRequest(BaseModel):
    """Single prediction request"""
    model_config = ConfigDict(protected_namespaces=())
    
    features: Dict[str, float] = Field(
        ...,
        description="Feature dictionary with feature names as keys",
        example={
            "EXT_SOURCE_1": 0.5,
            "EXT_SOURCE_2": 0.6,
            "EXT_SOURCE_3": 0.7,
            "AMT_CREDIT": 500000,
            "AMT_INCOME_TOTAL": 200000
        }
    )
    application_id: Optional[str] = Field(
        None,
        description="Optional application identifier for tracking"
    )
    explain: bool = Field(
        False,
        description="Include SHAP explanations in response"
    )
    
    @field_validator('features')
    @classmethod
    def validate_features(cls, v):
        if not v:
            raise ValueError("Features dictionary cannot be empty")
        if not all(isinstance(val, (int, float)) for val in v.values()):
            raise ValueError("All feature values must be numeric")
        return v


class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    model_config = ConfigDict(protected_namespaces=())
    
    applications: List[PredictionRequest] = Field(
        ...,
        description="List of prediction requests",
        min_length=1,
        max_length=1000
    )


class ExplainRequest(BaseModel):
    """SHAP explanation request"""
    model_config = ConfigDict(protected_namespaces=())
    
    features: Dict[str, float]
    application_id: Optional[str] = None
    top_n_features: int = Field(
        20,
        ge=1,
        le=100,
        description="Number of top features to return"
    )


# ============================================
# Response Models
# ============================================

class PredictionResponse(BaseModel):
    """Single prediction response"""
    model_config = ConfigDict(protected_namespaces=())
    
    application_id: Optional[str] = None
    prediction: int = Field(..., description="Binary prediction: 0 (good) or 1 (default)")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probability of default")
    risk_level: str = Field(..., description="Risk category: LOW, MEDIUM, HIGH, CRITICAL")
    threshold: float = Field(..., description="Classification threshold used")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    shap_values: Optional[Dict[str, float]] = Field(
        None,
        description="SHAP feature contributions (if explain=True)"
    )
    shap_base_value: Optional[float] = Field(
        None,
        description="SHAP base value (average model output)"
    )


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    model_config = ConfigDict(protected_namespaces=())
    
    predictions: List[PredictionResponse]
    total_count: int
    success_count: int
    failed_count: int
    processing_time_seconds: float


class ShapExplanation(BaseModel):
    """SHAP explanation response"""
    model_config = ConfigDict(protected_namespaces=())
    
    application_id: Optional[str] = None
    prediction: int
    probability: float
    base_value: float = Field(..., description="Model's average prediction")
    
    feature_contributions: Dict[str, float] = Field(
        ...,
        description="SHAP values per feature (sorted by absolute impact)"
    )
    
    top_positive_features: List[Dict[str, Any]] = Field(
        ...,
        description="Features increasing default risk"
    )
    
    top_negative_features: List[Dict[str, Any]] = Field(
        ...,
        description="Features decreasing default risk"
    )
    
    explanation_text: str = Field(
        ...,
        description="Human-readable explanation"
    )


# ============================================
# Model Info Models
# ============================================

class ModelMetadata(BaseModel):
    """Model information response"""
    model_config = ConfigDict(protected_namespaces=())
    
    model_name: str
    model_version: str
    model_type: str
    trained_at: Optional[str] = None
    
    feature_count: int
    training_samples: Optional[int] = None
    
    performance_metrics: Optional[Dict[str, float]] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    
    feature_names: Optional[List[str]] = None


class HealthResponse(BaseModel):
    """Health check response"""
    model_config = ConfigDict(protected_namespaces=())
    
    status: str = Field(..., description="Service status: healthy, degraded, unhealthy")
    version: str
    model_loaded: bool
    uptime_seconds: float
    
    system_info: Optional[Dict[str, Any]] = Field(
        None,
        description="System resource information"
    )


# ============================================
# Error Models
# ============================================

class ErrorResponse(BaseModel):
    """Standard error response"""
    model_config = ConfigDict(protected_namespaces=())
    
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)