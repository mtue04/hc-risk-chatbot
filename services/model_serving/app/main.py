from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(title="HC Risk Model Serving", version="0.1.0")

MODEL_PATH = Path(os.getenv("MODEL_PATH", "/models/credit_default.pkl"))


class FeatureVector(BaseModel):
    """Incoming feature vector keyed by feature name."""

    features: Dict[str, float] = Field(default_factory=dict, description="Feature name → value map.")


class BatchRequest(BaseModel):
    records: List[FeatureVector]


class PredictionOutput(BaseModel):
    probability: float
    model_version: Optional[str] = None
    is_stub: bool = False


def _load_model(path: Path):
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to load model at {path}: {exc}") from exc


model = _load_model(MODEL_PATH)


@app.get("/health")
def health_check() -> Dict[str, str]:
    status = "ready" if model is not None else "stub"
    return {"status": status}


def _score_stub(vector: Dict[str, float]) -> float:
    if not vector:
        return 0.5
    values = np.array(list(vector.values()), dtype=float)
    normalized = np.clip(values.mean() / (abs(values).mean() + 1e-6), -1, 1)
    probability = 0.5 + 0.5 * normalized
    return float(np.clip(probability, 0.0, 1.0))


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: FeatureVector) -> PredictionOutput:
    if model is None:
        return PredictionOutput(probability=_score_stub(payload.features), is_stub=True)

    if not payload.features:
        raise HTTPException(status_code=400, detail="Missing features.")

    # Preserve feature ordering expected by the trained model.
    try:
        ordered_columns = model.feature_names_in_
    except AttributeError:
        ordered_columns = list(payload.features.keys())

    feature_row = np.array([[payload.features.get(col, 0.0) for col in ordered_columns]], dtype=float)

    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(feature_row)[0, 1])
    else:
        prediction = float(model.predict(feature_row)[0])
        probability = float(np.clip(prediction, 0.0, 1.0))

    model_version = os.getenv("MODEL_VERSION")
    return PredictionOutput(probability=probability, model_version=model_version)


@app.post("/predict/batch")
def predict_batch(payload: BatchRequest):
    results = [predict(record) for record in payload.records]
    return {"predictions": [result.model_dump() for result in results]}


@app.post("/explain")
def explain(payload: FeatureVector):
    if model is None:
        return {
            "message": "Model artifact not available. Returning heuristic stub explanation.",
            "contributions": payload.features,
        }

    # Placeholder until SHAP integration is implemented.
    return {
        "message": "SHAP integration not yet implemented.",
        "features": payload.features,
    }
