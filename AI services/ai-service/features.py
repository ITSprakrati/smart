"""
features.py  (v3 - matches diagram's "Features Used" list exactly)

Diagram spec:
  Features Used: Fill level trend, Rate of change, Day / Time factor (optional)

Single source of truth used by both training and the API.
"""

from datetime import datetime
import pandas as pd

FEATURE_COLUMNS = ["fill_level", "fill_rate_per_hour", "hour", "day_of_week"]

# Experimental extended set - not in the original diagram spec, but tests
# whether adding "time since last collected" fixes the reset-discontinuity
# problem that hurts the baseline model's R^2.
FEATURE_COLUMNS_V2 = FEATURE_COLUMNS + ["last_collection_minutes_ago"]


def parse_timestamp(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")


def build_features(
    fill_level: float,
    timestamp: str,
    prev_fill_level: float = None,
    prev_timestamp: str = None,
    last_collection_minutes_ago: float = None,
) -> pd.DataFrame:
    """
    Build a single-row feature DataFrame from one reading (+ optionally the
    previous reading, to compute the fill-rate trend). If no previous
    reading is available, fill_rate_per_hour defaults to 0 - accuracy
    improves once the backend supplies the bin's last reading.
    """
    dt = parse_timestamp(timestamp)

    if prev_fill_level is not None and prev_timestamp:
        prev_dt = parse_timestamp(prev_timestamp)
        hours = (dt - prev_dt).total_seconds() / 3600.0
        fill_rate = (fill_level - prev_fill_level) / hours if hours > 0 else 0.0
    else:
        fill_rate = 0.0

    row = {
        "fill_level": fill_level,
        "fill_rate_per_hour": fill_rate,
        "hour": dt.hour,
        "day_of_week": dt.weekday(),
        "last_collection_minutes_ago": last_collection_minutes_ago if last_collection_minutes_ago is not None else 0.0,
    }
    return pd.DataFrame([row])


def linear_extrapolation_time_to_full(fill_level: float, fill_rate_per_hour: float, threshold: float = 90.0):
    """
    Fallback heuristic from the team's own risk plan: "AI/ML model isn't
    accurate enough -> fall back to a simple linear extrapolation from the
    last 2 readings." Returns None if fill isn't currently rising (rate <= 0),
    since extrapolation is meaningless in that case.
    """
    if fill_rate_per_hour is None or fill_rate_per_hour <= 0:
        return None
    remaining = max(0.0, threshold - fill_level)
    return remaining / fill_rate_per_hour
