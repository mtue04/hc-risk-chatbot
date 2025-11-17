from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional
import joblib
import numpy as np
from pydantic import BaseModel, Field, ConfigDict

# --- Config ---
class Settings:
    app_name = "HC Risk Model Serving"
    app_version = "0.1.0"
    host = "0.0.0.0"
    port = 8000
    debug = True
    log_level = "INFO"
    enable_shap = False  # Toggle SHAP explanations

settings = Settings()

# --- Logging ---
logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

startup_time = time.time()

# --- Models  ---
class FeatureVector(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    features: Dict[str, float] = Field(
        default_factory=dict, 
        description="Feature name → value map."
    )

class BatchRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    records: List[FeatureVector]

class PredictionOutput(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    probability: float
    model_version: Optional[str] = None
    is_stub: bool = False

class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    status: str
    version: str
    model_loaded: bool
    uptime_seconds: float

class ErrorResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    error: str
    detail: str

# --- Model Loader ---
MODEL_PATH = Path(os.getenv("MODEL_PATH", "/app/models/tuned/tuned_lgbm_model.pkl"))

def load_model(path: Path):
    if not path.exists():
        logger.warning(f"Model not found at {path}")
        return None
    try:
        model = joblib.load(path)
        logger.info(f"Model loaded successfully from {path}")
        return model
    except Exception as exc:
        logger.error(f"Failed to load model at {path}: {exc}")
        return None

model = load_model(MODEL_PATH)

# --- FastAPI App ---
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Production-ready API for credit risk prediction",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Exception Handlers ---
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error="Validation Error",
            detail=str(exc)
        ).model_dump()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal Server Error",
            detail="An unexpected error occurred"
        ).model_dump()
    )

# --- Health Endpoint ---
@app.get("/health", response_model=HealthResponse)
def health_check():
    status_str = "healthy" if model is not None else "unhealthy"
    return HealthResponse(
        status=status_str,
        version=settings.app_version,
        model_loaded=model is not None,
        uptime_seconds=time.time() - startup_time
    )

# --- Prediction Logic ---
def score_stub(vector: Dict[str, float]) -> float:
    """Fallback scoring when model is not available"""
    if not vector:
        return 0.5
    values = np.array(list(vector.values()), dtype=float)
    normalized = np.clip(values.mean() / (abs(values).mean() + 1e-6), -1, 1)
    probability = 0.5 + 0.5 * normalized
    return float(np.clip(probability, 0.0, 1.0))

@app.post("/predict", response_model=PredictionOutput)
def predict(payload: FeatureVector):
    """Single prediction endpoint"""
    if model is None:
        logger.warning("Model not loaded, using stub prediction")
        return PredictionOutput(
            probability=score_stub(payload.features), 
            is_stub=True
        )

    if not payload.features:
        raise HTTPException(status_code=400, detail="Missing features.")

    try:
        # Get feature names from model
        if hasattr(model, "feature_name_"):
            ordered_columns = model.feature_name_
        elif hasattr(model, "feature_names_in_"):
            ordered_columns = model.feature_names_in_
        else:
            ordered_columns = list(payload.features.keys())
    except AttributeError:
        ordered_columns = list(payload.features.keys())

    # Prepare feature array
    feature_row = np.array([
        [payload.features.get(col, 0.0) for col in ordered_columns]
    ], dtype=float)

    # Predict
    try:
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(feature_row)[0, 1])
        else:
            prediction = float(model.predict(feature_row)[0])
            probability = float(np.clip(prediction, 0.0, 1.0))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    model_version = os.getenv("MODEL_VERSION", "1.0.0")
    return PredictionOutput(
        probability=probability, 
        model_version=model_version
    )

@app.post("/predict/batch")
def predict_batch(payload: BatchRequest):
    """Batch prediction endpoint"""
    results = [predict(record) for record in payload.records]
    return {"predictions": [result.model_dump() for result in results]}

@app.post("/explain")
def explain(payload: FeatureVector):
    """SHAP explanation endpoint (placeholder)"""
    if not settings.enable_shap or model is None:
        return {
            "message": "Model artifact not available or SHAP disabled. Returning heuristic stub explanation.",
            "contributions": payload.features,
        }
    # TODO: Implement SHAP integration
    return {
        "message": "SHAP integration not yet implemented.",
        "features": payload.features,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )