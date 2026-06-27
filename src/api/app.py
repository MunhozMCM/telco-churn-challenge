"""FastAPI application — serves the Logistic Regression churn pipeline.

The model is loaded once at startup (lifespan), never per request. Routes are
thin HTTP shells that delegate to the pure service layer.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from loguru import logger

from src.api.middleware import add_process_time_header
from src.api.schemas import CustomerData, PredictionOut
from src.api.service import predict_churn
from src.data.io import load_metadata, load_pipeline
from src.version import MODEL_VERSION

# Structured JSON logging to stdout (ready for log aggregators).
logger.remove()
logger.add(sys.stdout, serialize=True, level="INFO")

ml: dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained pipeline (+ metadata) once on startup."""
    try:
        ml["model"] = load_pipeline()
        try:
            ml["meta"] = load_metadata()
        except FileNotFoundError:
            ml["meta"] = {"model_version": MODEL_VERSION}
        logger.info("model loaded — version {}", ml["meta"].get("model_version"))
    except FileNotFoundError:
        # Start degraded rather than crash; /health will report it and /predict 503s.
        logger.error("model artifact not found — run training first (make train)")
    yield
    ml.clear()


app = FastAPI(
    title="Telco Churn Prediction API",
    version=MODEL_VERSION,
    lifespan=lifespan,
)
app.middleware("http")(add_process_time_header)


@app.get("/health")
async def health():
    """Liveness/readiness — reports whether the model is loaded."""
    loaded = "model" in ml
    return {
        "status": "ok" if loaded else "degraded",
        "model_loaded": loaded,
        "model_version": (ml.get("meta") or {}).get("model_version", MODEL_VERSION),
    }


@app.post("/predict", response_model=PredictionOut)
async def predict(features: CustomerData):
    """Score a single customer's churn risk."""
    model = ml.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model unavailable")
    try:
        return PredictionOut(**predict_churn(model, features))
    except Exception as exc:  # noqa: BLE001 — surface inference errors as 500
        logger.exception("inference failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc
