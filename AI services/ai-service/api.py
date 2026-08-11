"""
api.py  (v4 - production-audit hardened)

Accepts the diagram's "Sensor Data (From IoT Bin)" payload:
{
  "bin_id": "BIN_001", "fill_level": 72, "battery": 89, "temperature": 32.4,
  "location": {"lat": 23.2599, "lng": 77.4126}, "timestamp": "2026-07-27T10:15:30Z",
  "device_id": "DEV_001"
}
zone_id is NOT sent by the sensor - it lives in the bins table, so it's
accepted as an OPTIONAL extra field, not used by the model.

Audit fixes in this version:
  - Model file loading wrapped in try/except: a corrupted or incompatible
    pkl no longer crashes the ENTIRE app (previously this happened at
    unguarded module-import time, meaning even /health would go down).
  - fill_level / prev_fill_level bounded with Pydantic Field(ge=0, le=100):
    malformed inputs now get a clean 422 instead of silently producing an
    absurd extrapolated prediction (verified case: -999999 input produced
    a 625,866-hour prediction with zero warning before this fix).
  - Output sanity clamp: any prediction beyond a plausible ceiling is
    flagged low-confidence rather than returned as if trustworthy.
  - Basic structured logging added for production visibility/monitoring.

Endpoints:
  POST /predict/time-to-full
  POST /predict/priority-score
  GET  /health   - reports model_loaded status AND any load error, so a
                    bad deploy is diagnosable without digging through logs

Run: uvicorn api:app --reload --port 8000
"""

import os
import logging
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from features import build_features, linear_extrapolation_time_to_full, MAX_SANE_PREDICTION_HOURS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("prediction-service")

app = FastAPI(title="Smart Waste - Prediction Service")

# --- Audit fix: never let a bad model file take down the whole app ---
model_bundle = None
model_load_error = None
try:
    if os.path.exists("time_to_full_model.pkl"):
        model_bundle = joblib.load("time_to_full_model.pkl")
        logger.info(f"Model loaded. feature_columns={getattr(model_bundle, 'feature_columns', 'UNKNOWN')}")
    else:
        logger.warning("time_to_full_model.pkl not found - running in dummy mode")
except Exception as e:
    model_load_error = f"{type(e).__name__}: {e}"
    logger.error(f"Failed to load model, running in degraded mode: {model_load_error}")


class Location(BaseModel):
    lat: float
    lng: float


class BinReading(BaseModel):
    bin_id: str
    fill_level: float = Field(..., ge=0, le=100, description="Percent full, 0-100")
    battery: float | None = None
    temperature: float | None = None
    location: Location | None = None
    timestamp: str
    device_id: str | None = None
    zone_id: str | None = None
    prev_fill_level: float | None = Field(None, ge=0, le=100)
    prev_timestamp: str | None = None
    last_collection_minutes_ago: float | None = None


def _build_features_safe(reading: BinReading):
    try:
        return build_features(
            fill_level=reading.fill_level,
            timestamp=reading.timestamp,
            prev_fill_level=reading.prev_fill_level,
            prev_timestamp=reading.prev_timestamp,
        )
    except ValueError as e:
        logger.warning(f"Bad input for bin_id={reading.bin_id}: {e}")
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model_bundle is not None,
        "model_load_error": model_load_error,
        "feature_columns": getattr(model_bundle, "feature_columns", None) if model_bundle else None,
    }


@app.post("/predict/time-to-full")
def predict_time_to_full(reading: BinReading):
    features_df = _build_features_safe(reading)
    fill_rate = float(features_df["fill_rate_per_hour"].iloc[0])
    extrapolated = linear_extrapolation_time_to_full(reading.fill_level, fill_rate)

    if model_bundle is None:
        if extrapolated is None:
            return {"bin_id": reading.bin_id, "predicted_time_to_full_hours": None,
                     "mode": "dummy", "note": "no model and no rising trend to extrapolate from"}
        return {"bin_id": reading.bin_id, "predicted_time_to_full_hours": round(extrapolated, 1), "mode": "extrapolation"}

    model_hrs = max(0.0, model_bundle.predict_one(features_df))

    confidence = "normal"
    # --- Audit fix: catch absurd out-of-distribution predictions even
    # when there's no extrapolation estimate to compare against (this was
    # the exact gap that let a 625,866-hour prediction through as "normal") ---
    if model_hrs > MAX_SANE_PREDICTION_HOURS:
        confidence = "low"
        logger.warning(f"bin_id={reading.bin_id}: model predicted {model_hrs:.1f}hrs, exceeds sane ceiling of {MAX_SANE_PREDICTION_HOURS}hrs")
    elif extrapolated is not None and extrapolated > 0:
        divergence = abs(model_hrs - extrapolated) / extrapolated
        if divergence > 0.75:
            confidence = "low"

    return {
        "bin_id": reading.bin_id,
        "predicted_time_to_full_hours": round(min(model_hrs, MAX_SANE_PREDICTION_HOURS), 1),
        "extrapolation_estimate_hours": round(extrapolated, 1) if extrapolated is not None else None,
        "confidence": confidence,
        "mode": "model",
    }


@app.post("/predict/priority-score")
def predict_priority_score(reading: BinReading):
    """Score = f(current fill %, predicted time-to-full, time since last collected)."""
    features_df = _build_features_safe(reading)

    if model_bundle is not None:
        predicted_hrs = max(0.0, min(model_bundle.predict_one(features_df), MAX_SANE_PREDICTION_HOURS))
    else:
        predicted_hrs = max(0.0, (100 - reading.fill_level) / 5)

    fill_component = reading.fill_level / 100
    urgency_component = max(0.0, 1 - min(predicted_hrs, 12) / 12)
    since_collected_hrs = max(0.0, (reading.last_collection_minutes_ago or 0) / 60)  # clamp negative input
    staleness_component = min(1.0, since_collected_hrs / 24)

    score = round(100 * (0.5 * fill_component + 0.35 * urgency_component + 0.15 * staleness_component), 1)

    return {
        "bin_id": reading.bin_id,
        "zone_id": reading.zone_id,
        "score": score,
        "reason": {
            "fill_level": reading.fill_level,
            "predicted_time_to_full_hours": round(predicted_hrs, 1),
            "last_collection_minutes_ago": reading.last_collection_minutes_ago,
        },
    }
