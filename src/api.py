"""FastAPI service for real-time claim severity scoring.

Usage
-----
    uvicorn src.api:app --host 0.0.0.0 --port 8000

Endpoints
---------
GET  /health    -> liveness probe, no model required.
GET  /metadata  -> model artifact info (version, features, target).
POST /predict   -> score a single claim record.

The API loads the same serialized pipeline (`artifacts/model.joblib`) used
by the batch scoring CLI (`src.score`), so predictions are consistent
between batch and online paths.
"""
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from .validate import VALID_CHANNELS, VALID_STATES

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "artifacts/model.joblib"))

app = FastAPI(
    title="Insurance Claims Severity API",
    description=(
        "Analytical decision-support API for estimating claim severity. "
        "Predictions are not automated claims decisions and require human review."
    ),
    version="0.1.0",
)

_bundle: Optional[dict] = None
_load_error: Optional[str] = None


def _load_bundle() -> Optional[dict]:
    """Lazily load the model bundle, caching the result (and any error)."""
    global _bundle, _load_error
    if _bundle is not None or _load_error is not None:
        return _bundle
    if not MODEL_PATH.exists():
        _load_error = f"Model artifact not found at {MODEL_PATH}. Run `python -m src.train` first."
        return None
    try:
        _bundle = joblib.load(MODEL_PATH)
    except Exception as exc:  # pragma: no cover - defensive guard
        _load_error = f"Failed to load model artifact: {exc}"
        return None
    return _bundle


def _model_version(bundle: dict) -> str:
    try:
        digest = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()[:10]
    except OSError:
        digest = "unknown"
    target = bundle.get("target", "loss")
    return f"{target}-{digest}"


class ClaimRequest(BaseModel):
    """Request schema for a single claim scoring request."""

    age: int = Field(..., ge=18, le=100, description="Policyholder age in years")
    vehicle_age: int = Field(..., ge=0, le=40, description="Vehicle age in years")
    state: str = Field(..., description="Two-letter Indian state code")
    channel: str = Field(..., description="Sales channel: agent, online, branch, broker, direct")

    @field_validator("state")
    @classmethod
    def _check_state(cls, v: str) -> str:
        if v.upper() not in VALID_STATES:
            raise ValueError(f"Unknown state code: {v}. Expected one of {sorted(VALID_STATES)}")
        return v.upper()

    @field_validator("channel")
    @classmethod
    def _check_channel(cls, v: str) -> str:
        if v.lower() not in VALID_CHANNELS:
            raise ValueError(f"Unknown channel: {v}. Expected one of {sorted(VALID_CHANNELS)}")
        return v.lower()


class PredictResponse(BaseModel):
    predicted_loss: float
    model_version: str
    scoring_timestamp: str


@app.get("/health")
def health():
    """Liveness probe. Does not require the model artifact to exist."""
    return {"status": "ok"}


@app.get("/metadata")
def metadata():
    """Return model artifact metadata: version, target, and expected features."""
    bundle = _load_bundle()
    if bundle is None:
        raise HTTPException(status_code=503, detail=_load_error)
    return {
        "model_version": _model_version(bundle),
        "target": bundle.get("target", "loss"),
        "features": bundle.get("features", []),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(claim: ClaimRequest):
    """Score a single claim record and return the predicted severity.

    Prediction is decision support only; it does not represent an
    automated claims adjudication or payout commitment.
    """
    bundle = _load_bundle()
    if bundle is None:
        raise HTTPException(status_code=503, detail=_load_error)

    model = bundle["model"]
    expected_features = bundle.get("features", [])
    row = pd.DataFrame([claim.model_dump()])

    if expected_features:
        missing = set(expected_features) - set(row.columns)
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"Request is missing model features: {sorted(missing)}",
            )
        row = row[expected_features]

    try:
        pred = float(model.predict(row)[0])
    except Exception as exc:  # pragma: no cover - defensive guard
        raise HTTPException(status_code=500, detail="Model failed to score the request.") from exc

    return PredictResponse(
        predicted_loss=max(pred, 0.0),
        model_version=_model_version(bundle),
        scoring_timestamp=datetime.now(timezone.utc).isoformat(),
    )
