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

# Import Feast client
try:
    from app.feast_client import get_feast_client
    FEAST_ENABLED = True
except ImportError:
    try:
        from feast_client import get_feast_client
        FEAST_ENABLED = True
    except ImportError:
        FEAST_ENABLED = False
        logging.warning("Feast client not available")

# --- Config ---
try:
    from app.config import settings
    from app.explainer import shap_explainer
    SHAP_AVAILABLE = True
except ImportError:
    try:
        from config import settings
        from explainer import shap_explainer
        SHAP_AVAILABLE = True
    except ImportError:
        # Fallback settings if config module not available
        class Settings:
            app_name = "HC Risk Model Serving"
            app_version = "0.1.0"
            host = "0.0.0.0"
            port = 8000
            debug = True
            log_level = "INFO"
            enable_shap = True
        settings = Settings()
        shap_explainer = None
        SHAP_AVAILABLE = False
        logging.warning("config.py not available, using inline settings")

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

class ApplicantRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    applicant_id: int = Field(..., description="SK_ID_CURR of the applicant")

class BatchApplicantRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    applicant_ids: List[int] = Field(..., description="List of SK_ID_CURR values")

class HypotheticalRequest(BaseModel):
    """Request for predicting risk of a hypothetical/custom applicant profile."""
    model_config = ConfigDict(protected_namespaces=())
    
    features: Dict[str, float] = Field(
        default_factory=dict,
        description="Custom feature values for the hypothetical applicant"
    )
    name: str = Field(
        default="Custom Applicant",
        description="Human-readable label for this hypothetical profile"
    )
    fill_missing_with_median: bool = Field(
        default=True,
        description="If True, missing features will be filled with population median"
    )

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

# --- Feast Client ---
feast_client = None
if FEAST_ENABLED:
    try:
        feast_client = get_feast_client()
        if feast_client.is_available():
            logger.info("Feast feature store integration enabled")
        else:
            logger.warning("Feast feature store not available")
    except Exception as e:
        logger.error(f"Failed to initialize Feast client: {e}")
        feast_client = None

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
    feast_available = feast_client is not None and feast_client.is_available() if FEAST_ENABLED else False

    response = HealthResponse(
        status=status_str,
        version=settings.app_version,
        model_loaded=model is not None,
        uptime_seconds=time.time() - startup_time
    )

    # Add Feast status to response (as extra field)
    response_dict = response.model_dump()
    response_dict["feast_available"] = feast_available

    return response_dict

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
    """SHAP explanation endpoint with detailed feature contributions."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not settings.enable_shap:
        return {
            "message": "SHAP explanations are disabled in config.",
            "contributions": payload.features,
        }
    
    if not SHAP_AVAILABLE or shap_explainer is None:
        return {
            "message": "SHAP explainer not available. Returning feature values as stub.",
            "contributions": payload.features,
        }
    
    try:
        explanation = shap_explainer.explain(payload.features)
        return {
            "applicant_features": payload.features,
            **explanation
        }
    except Exception as e:
        logger.error(f"SHAP explanation error: {e}")
        return {
            "message": f"SHAP explanation failed: {str(e)}",
            "contributions": payload.features,
        }

# --- Hypothetical Prediction Endpoint ---

@app.post("/predict/hypothetical")
def predict_hypothetical(payload: HypotheticalRequest):
    """
    Predict default probability for a hypothetical/custom applicant profile.
    
    This endpoint allows prediction without requiring an existing applicant ID.
    Use this for "what-if" scenario testing with custom feature values.
    
    Returns prediction probability, risk classification, and SHAP explanation.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not payload.features:
        raise HTTPException(
            status_code=400, 
            detail="Features dictionary cannot be empty. Please provide at least some feature values."
        )
    
    # Get required feature names from model
    try:
        if hasattr(model, "feature_name_"):
            required_features = model.feature_name_
        elif hasattr(model, "feature_names_in_"):
            required_features = model.feature_names_in_
        else:
            required_features = list(payload.features.keys())
    except AttributeError:
        required_features = list(payload.features.keys())
    
    # Fill missing features with 0 (or could use population median if available)
    filled_features = {}
    for feature in required_features:
        if feature in payload.features:
            filled_features[feature] = payload.features[feature]
        else:
            filled_features[feature] = 0.0  # Default: fill with 0
    
    # Make prediction
    feature_vector = FeatureVector(features=filled_features)
    prediction_result = predict(feature_vector)
    
    response = {
        "name": payload.name,
        "probability": prediction_result.probability,
        "prediction": "High Risk" if prediction_result.probability > 0.5 else "Low Risk",
        "risk_level": (
            "Low" if prediction_result.probability < 0.3 
            else "Medium" if prediction_result.probability < 0.6 
            else "High"
        ),
        "provided_features": list(payload.features.keys()),
        "total_features_used": len(filled_features),
        "features_filled_with_default": len(required_features) - len(payload.features),
    }
    
    # Add SHAP explanation if available
    if SHAP_AVAILABLE and shap_explainer is not None and settings.enable_shap:
        try:
            explanation = shap_explainer.explain(filled_features)
            response["explanation"] = explanation
        except Exception as e:
            logger.warning(f"SHAP explanation failed for hypothetical: {e}")
            response["explanation"] = {"message": "SHAP explanation not available"}
    
    logger.info(
        f"Hypothetical prediction: {payload.name}, prob={prediction_result.probability:.4f}, features={len(payload.features)}"
    )
    
    return response

# --- Feast-powered Endpoints ---

@app.post("/predict/applicant", response_model=PredictionOutput)
def predict_applicant(payload: ApplicantRequest):
    """
    Predict default probability for an applicant using Feast feature store.

    This endpoint fetches features from the Feast online store based on
    applicant ID (SK_ID_CURR) and returns the prediction.
    """
    if not FEAST_ENABLED or feast_client is None or not feast_client.is_available():
        raise HTTPException(
            status_code=503,
            detail="Feast feature store not available. Use /predict endpoint with explicit features."
        )

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Fetch features from Feast
    features_dict = feast_client.get_features(payload.applicant_id)

    if features_dict is None:
        raise HTTPException(
            status_code=404,
            detail=f"Features not found for applicant {payload.applicant_id}"
        )

    # Convert to FeatureVector and predict
    feature_vector = FeatureVector(features=features_dict)
    return predict(feature_vector)


@app.get("/predict/applicant/{applicant_id}", response_model=PredictionOutput)
def predict_applicant_get(applicant_id: int):
    """
    GET version of predict_applicant - same functionality but via path parameter.
    """
    return predict_applicant(ApplicantRequest(applicant_id=applicant_id))


@app.get("/explain/applicant/{applicant_id}")
def explain_applicant_get(applicant_id: int):
    """
    GET SHAP explanation for an applicant using Feast feature store.
    Returns detailed SHAP values showing feature contributions to the prediction.
    """
    if not FEAST_ENABLED or feast_client is None or not feast_client.is_available():
        return {
            "message": "Feast feature store not available.",
            "contributions": {},
            "applicant_id": applicant_id,
        }

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Fetch features from Feast
    features_dict = feast_client.get_features(applicant_id)

    if features_dict is None:
        raise HTTPException(
            status_code=404,
            detail=f"Features not found for applicant {applicant_id}"
        )

    # Get prediction first
    feature_vector = FeatureVector(features=features_dict)
    prediction_result = predict(feature_vector)
    
    # Get SHAP explanation if available
    if SHAP_AVAILABLE and shap_explainer is not None and settings.enable_shap:
        try:
            explanation = shap_explainer.explain(features_dict)
            return {
                "applicant_id": applicant_id,
                "probability": prediction_result.probability,
                "prediction": "High Risk" if prediction_result.probability > 0.5 else "Low Risk",
                **explanation
            }
        except Exception as e:
            logger.error(f"SHAP explanation failed for applicant {applicant_id}: {e}")
    
    # Fallback: return features as contributions
    sorted_features = sorted(
        [(k, v) for k, v in features_dict.items() if isinstance(v, (int, float))],
        key=lambda x: abs(x[1]), reverse=True
    )[:10]
    return {
        "applicant_id": applicant_id,
        "probability": prediction_result.probability,
        "prediction": "High Risk" if prediction_result.probability > 0.5 else "Low Risk",
        "shap_values": dict(sorted_features),
        "message": "Using feature values as contribution estimates.",
    }


@app.post("/predict/applicant/batch")
def predict_applicant_batch(payload: BatchApplicantRequest):
    """
    Batch prediction for multiple applicants using Feast feature store.

    Fetches features from Feast online store for all applicants and
    returns predictions for each.
    """
    if not FEAST_ENABLED or feast_client is None or not feast_client.is_available():
        raise HTTPException(
            status_code=503,
            detail="Feast feature store not available. Use /predict/batch endpoint."
        )

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Fetch features from Feast
    features_list = feast_client.get_features_batch(payload.applicant_ids)

    if features_list is None:
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch features from Feast"
        )

    # Predict for each applicant
    results = []
    for i, features_dict in enumerate(features_list):
        try:
            feature_vector = FeatureVector(features=features_dict)
            prediction = predict(feature_vector)
            results.append({
                "applicant_id": payload.applicant_ids[i],
                **prediction.model_dump()
            })
        except Exception as e:
            logger.error(f"Prediction failed for applicant {payload.applicant_ids[i]}: {e}")
            results.append({
                "applicant_id": payload.applicant_ids[i],
                "error": str(e)
            })

    return {"predictions": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )