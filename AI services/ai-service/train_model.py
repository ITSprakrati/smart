"""
train_model.py  (v3 - matches diagram: "Prediction Model: Linear Regression")

Trains a Linear Regression model predicting time_to_full_hours from
fill level trend features. Time-based train/test split (never random
shuffle - this is time-series data).

Run: python train_model.py
Output: time_to_full_model.pkl
"""

import pandas as pd
import json
import os
from datetime import datetime, timezone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

from features import FEATURE_COLUMNS, FEATURE_COLUMNS_V2

TRAIN_FRACTION_BY_TIME = 0.8


def next_version_number(pattern="model_v{}.pkl"):
    """Finds the next free vN so previous versions are never overwritten."""
    v = 1
    while os.path.exists(pattern.format(v)):
        v += 1
    return v


def time_based_split(df, frac=TRAIN_FRACTION_BY_TIME):
    df = df.sort_values("timestamp_dt")
    cutoff_idx = int(len(df) * frac)
    cutoff_time = df.iloc[cutoff_idx]["timestamp_dt"]
    return df[df["timestamp_dt"] < cutoff_time], df[df["timestamp_dt"] >= cutoff_time]


def train_and_eval(train_df, test_df, feature_cols, label):
    X_train, X_test = train_df[feature_cols], test_df[feature_cols]
    y_train, y_test = train_df["time_to_full_hours"], test_df["time_to_full_hours"]

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)
    print(f"\n[{label}] features={feature_cols}")
    print(f"  MAE={mae:.2f} hrs  RMSE={rmse:.2f} hrs  R2={r2:.3f}")
    for feat, coef in zip(feature_cols, model.coef_):
        print(f"    {feat}: {coef:.4f}")
    print(f"    intercept: {model.intercept_:.4f}")
    return model, {"mae_hours": round(mae, 3), "rmse_hours": round(rmse, 3), "r2": round(r2, 3)}


def main():
    df = pd.read_csv("training_dataset.csv")
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"])

    train_df, test_df = time_based_split(df)
    print(f"Train rows: {len(train_df)}  Test rows: {len(test_df)}")

    baseline_model, baseline_metrics = train_and_eval(train_df, test_df, FEATURE_COLUMNS, "BASELINE (diagram spec)")
    extended_model, extended_metrics = train_and_eval(train_df, test_df, FEATURE_COLUMNS_V2, "EXTENDED (+ last_collection_minutes_ago)")

    print(f"\n{'='*60}")
    print(f"Baseline R2: {baseline_metrics['r2']}   Extended R2: {extended_metrics['r2']}")
    improved = extended_metrics["r2"] > baseline_metrics["r2"]
    winner_label = "EXTENDED" if improved else "BASELINE"
    print(f"Winner: {winner_label} {'(improvement confirmed)' if improved else '(extension did not help - keeping baseline)'}")
    print(f"{'='*60}")

    winner_model = extended_model if improved else baseline_model
    winner_features = FEATURE_COLUMNS_V2 if improved else FEATURE_COLUMNS
    winner_metrics = extended_metrics if improved else baseline_metrics

    version = next_version_number()
    model_filename = f"model_v{version}.pkl"
    joblib.dump(winner_model, model_filename)
    joblib.dump(winner_model, "time_to_full_model.pkl")  # stable pointer the API loads

    metadata = {
        "version": version,
        "filename": model_filename,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "trained_on": "training_dataset.csv",
        "data_source": "simulated",
        "n_train_rows": len(train_df),
        "n_test_rows": len(test_df),
        "features": winner_features,
        "feature_set_label": winner_label,
        "metrics": winner_metrics,
        "baseline_metrics_for_comparison": baseline_metrics,
        "extended_metrics_for_comparison": extended_metrics,
        "coefficients": {feat: round(float(c), 4) for feat, c in zip(winner_features, winner_model.coef_)},
        "intercept": round(float(winner_model.intercept_), 4),
    }
    meta_path = "model_registry.json"
    registry = json.load(open(meta_path)) if os.path.exists(meta_path) else []
    registry.append(metadata)
    with open(meta_path, "w") as f:
        json.dump(registry, f, indent=2)

    print(f"\nSaved: {model_filename} (versioned), time_to_full_model.pkl (latest pointer, uses {winner_label} features)")
    print(f"Logged to {meta_path} - {len(registry)} version(s) recorded so far")


if __name__ == "__main__":
    main()
