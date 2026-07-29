"""
api.py  (v3 - matches architecture diagram's Prediction Service + Priority Engine)

Accepts the diagram's exact "Sensor Data (From IoT Bin)" payload:
{
  "bin_id": "BIN_001",
  "fill_level": 72,
  "battery": 89,
  "temperature": 32.4,
  "location": { "lat": 23.2599, "lng": 77.4126 },
  "timestamp": "2026-07-27T10:15:30Z",
  "device_id": "DEV_001"
}
Note: zone_id is NOT sent by the sensor - it lives in the bins table and is
looked up by the backend, so it's accepted as an OPTIONAL extra field here
(only used by the priority-score endpoint's response, not by the model).

Endpoints:
  POST /predict/time-to-full    -> ETA estimation (hrs), matches "5. PREDICTION SERVICE"
  POST /predict/priority-score  -> matches "7. PRIORITY ENGINE":
                                    Score = f(fill %, predicted time-to-full, time since last collected)
  GET  /health

Run: uvicorn api:app --reload --port 8000
"""

import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

from features import build_features, FEATURE_COLUMNS, linear_extrapolation_time_to_full

app = FastAPI(title="Smart Waste - Prediction Service")

model = joblib.load("time_to_full_model.pkl") if os.path.exists("time_to_full_model.pkl") else None


class Location(BaseModel):
    lat: float
    lng: float


class BinReading(BaseModel):
    bin_id: str
    fill_level: float
    battery: float | None = None
    temperature: float | None = None
    location: Location | None = None
    timestamp: str
    device_id: str | None = None
    zone_id: str | None = None                 # optional - looked up by backend from bins table
    prev_fill_level: float | None = None        # optional - backend supplies from last known reading
    prev_timestamp: str | None = None
    last_collection_minutes_ago: float | None = None  # optional - only used by priority-score


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict/time-to-full")
def predict_time_to_full(reading: BinReading):
    features_df = build_features(
        fill_level=reading.fill_level,
        timestamp=reading.timestamp,
        prev_fill_level=reading.prev_fill_level,
        prev_timestamp=reading.prev_timestamp,
    )
    X = features_df[FEATURE_COLUMNS]
    fill_rate = float(features_df["fill_rate_per_hour"].iloc[0])

    # Fallback heuristic from the team's risk plan: linear extrapolation
    # from the last 2 readings. Used when the model is unavailable, and
    # cross-checked against the model otherwise to flag low-confidence predictions.
    extrapolated = linear_extrapolation_time_to_full(reading.fill_level, fill_rate)

    if model is None:
        if extrapolated is None:
            return {"bin_id": reading.bin_id, "predicted_time_to_full_hours": None,
                     "mode": "dummy", "note": "no model and no rising trend to extrapolate from"}
        return {"bin_id": reading.bin_id, "predicted_time_to_full_hours": round(extrapolated, 1), "mode": "extrapolation"}

    model_hrs = max(0.0, float(model.predict(X)[0]))

    confidence = "normal"
    if extrapolated is not None and extrapolated > 0:
        divergence = abs(model_hrs - extrapolated) / extrapolated
        if divergence > 0.75:
            confidence = "low"  # model and simple extrapolation disagree a lot

    return {
        "bin_id": reading.bin_id,
        "predicted_time_to_full_hours": round(model_hrs, 1),
        "extrapolation_estimate_hours": round(extrapolated, 1) if extrapolated is not None else None,
        "confidence": confidence,
        "mode": "model",
    }


@app.post("/predict/priority-score")
def predict_priority_score(reading: BinReading):
    """
    Score = f(current fill %, predicted time-to-full, time since last collected)
    Weights are a starting point - tune with your team once you see real data.
    """
    X = build_features(
        fill_level=reading.fill_level,
        timestamp=reading.timestamp,
        prev_fill_level=reading.prev_fill_level,
        prev_timestamp=reading.prev_timestamp,
    )[FEATURE_COLUMNS]

    if model is not None:
        predicted_hrs = max(0.0, float(model.predict(X)[0]))
    else:
        predicted_hrs = max(0.0, (100 - reading.fill_level) / 5)

    fill_component = reading.fill_level / 100
    urgency_component = max(0.0, 1 - min(predicted_hrs, 12) / 12)  # 1.0 = full very soon
    since_collected_hrs = (reading.last_collection_minutes_ago or 0) / 60
    staleness_component = min(1.0, since_collected_hrs / 24)  # caps at 1 day

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
