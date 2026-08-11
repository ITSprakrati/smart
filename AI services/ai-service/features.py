"""
features.py  (v4 - production-audit hardened)

Diagram spec:
  Features Used: Fill level trend, Rate of change, Day / Time factor (optional)

Single source of truth used by both training and the API. Also carries
the model/feature contract (see FeatureContract) so training and serving
can never silently drift apart.

Audit fixes in this version:
  - clip_fill_rate(): filters reset-event artifacts (was distorting OLS fit)
  - clamp_fill_level(): input sanity bound, prevents wild extrapolation
  - FeatureContract: bundles a model with the exact feature list it needs,
    so a future retrain can never silently mismatch what the API sends it
"""

from datetime import datetime
from dataclasses import dataclass
from typing import List
import pandas as pd

FEATURE_COLUMNS = ["fill_level", "fill_rate_per_hour", "hour", "day_of_week"]

# Experimental extended set - not in the original diagram spec, tests whether
# adding "time since last collected" fixes the reset-discontinuity problem.
FEATURE_COLUMNS_V2 = FEATURE_COLUMNS + ["last_collection_minutes_ago"]

MIN_PLAUSIBLE_FILL_RATE = -20.0   # steeper drops than this are collection-reset artifacts, not real trend
MAX_PLAUSIBLE_FILL_RATE = 50.0    # sanity ceiling; real bins don't fill 50%/hr
MIN_PLAUSIBLE_FILL_LEVEL = 0.0
MAX_PLAUSIBLE_FILL_LEVEL = 100.0
MAX_SANE_PREDICTION_HOURS = 24 * 30  # 30 days - beyond this, treat prediction as untrustworthy, not literal


def clip_fill_rate(rate: float) -> float:
    """
    A fill_rate this negative almost always means the bin was just emptied
    between readings (a discrete reset event), not a genuine downward trend.
    Feeding raw reset deltas (observed as low as -198%/hr in real training
    data) into a linear model distorts its coefficients disproportionately.
    Treat as "just reset -> no informative rate signal" instead.
    """
    if rate < MIN_PLAUSIBLE_FILL_RATE or rate > MAX_PLAUSIBLE_FILL_RATE:
        return 0.0
    return rate


def clamp_fill_level(level: float) -> float:
    """Defense in depth: even if API-layer validation is bypassed or missing,
    never feed the model a fill_level outside the physically possible range."""
    return max(MIN_PLAUSIBLE_FILL_LEVEL, min(MAX_PLAUSIBLE_FILL_LEVEL, level))


def parse_timestamp(ts: str) -> datetime:
    """
    Accepts ISO 8601 timestamps with or without milliseconds, e.g. both
    "2026-07-27T10:15:30Z" and JS's Date.toISOString() output
    "2026-07-27T10:15:30.123Z". Raises a clear ValueError on genuinely bad input.
    """
    if not ts or not isinstance(ts, str):
        raise ValueError(f"timestamp must be a non-empty ISO 8601 string, got: {ts!r}")
    normalized = ts.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        raise ValueError(f"Could not parse timestamp: {ts!r}. Expected ISO 8601, e.g. '2026-07-27T10:15:30Z'")


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
    reading is available, fill_rate_per_hour defaults to 0.
    """
    fill_level = clamp_fill_level(fill_level)
    dt = parse_timestamp(timestamp)

    if prev_fill_level is not None and prev_timestamp:
        prev_dt = parse_timestamp(prev_timestamp)
        hours = (dt - prev_dt).total_seconds() / 3600.0
        raw_rate = (fill_level - clamp_fill_level(prev_fill_level)) / hours if hours > 0 else 0.0
        fill_rate = clip_fill_rate(raw_rate)
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
    last 2 readings." Returns None if fill isn't currently rising (rate <= 0).
    """
    if fill_rate_per_hour is None or fill_rate_per_hour <= 0:
        return None
    remaining = max(0.0, threshold - fill_level)
    return remaining / fill_rate_per_hour


@dataclass
class FeatureContract:
    """
    Bundles a trained model together with the EXACT feature list it was
    trained on. Fixes a real audit finding: train_model.py can pick either
    FEATURE_COLUMNS or FEATURE_COLUMNS_V2 as the winning feature set, but
    api.py previously hardcoded FEATURE_COLUMNS regardless - a future
    retrain that picks the extended set would have silently broken every
    prediction (wrong number of features passed to sklearn). Now the model
    file itself carries its required feature list, so serving code always
    asks the artifact what it needs instead of assuming.
    """
    model: object
    feature_columns: List[str]

    def predict_one(self, features_df: pd.DataFrame) -> float:
        X = features_df[self.feature_columns]
        return float(self.model.predict(X)[0])
