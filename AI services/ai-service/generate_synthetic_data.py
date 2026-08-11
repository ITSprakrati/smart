"""
generate_synthetic_data.py  (v3 - matches the team's architecture diagram)

Matches the diagram's Prediction Service spec exactly:
  Model: Linear Regression
  Input: Fill level history
  Output: Time-to-full (hrs)
  Features used: Fill level trend, Rate of change, Day/Time factor (optional)

And the diagram's "Sensor Data (From IoT Bin)" payload:
  { bin_id, fill_level, battery, temperature, location:{lat,lng}, timestamp, device_id }
Note: zone_id is NOT in the raw sensor payload - it lives in the bins table
(Main DB) and gets joined in by the backend, not sent by the sensor itself.

Produces:
  bin_sensor_data.csv    - raw IoT-style readings (matches sensor payload)
  training_dataset.csv   - ML-ready rows: bin_id, timestamp, fill_level,
                            fill_rate_per_hour, hour, day_of_week,
                            time_to_full_hours (target)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

NUM_BINS = 30
DAYS = 21  # extended from 7: a fixed-horizon censoring scheme (see below) needs
           # enough runway that trimming the unobservable tail doesn't eat most of the data
READING_INTERVAL_MIN = 30
START_DATE = datetime(2026, 7, 14, 0, 0, 0)
OVERFLOW_THRESHOLD = 90.0
HORIZON_HOURS = 96  # fixed look-ahead window for computing time_to_full_hours (see below)

# zone kept only as bin metadata (mirrors "zone lives in bins table"), NOT a model feature
ZONES = ["Z_01", "Z_02", "Z_03", "Z_04"]
ZONE_FILL_RATE_PER_HOUR = {"Z_01": 1.2, "Z_02": 2.5, "Z_03": 1.8, "Z_04": 2.0}
ZONE_LOCATION_CENTER = {
    "Z_01": (23.2500, 77.4000), "Z_02": (23.2599, 77.4126),
    "Z_03": (23.2650, 77.3950), "Z_04": (23.2450, 77.4200),
}


def main():
    readings_per_day = int(24 * 60 / READING_INTERVAL_MIN)
    num_readings = DAYS * readings_per_day
    raw_rows = []

    for bin_num in range(1, NUM_BINS + 1):
        bin_id = f"BIN_{1000 + bin_num}"
        device_id = f"DEV_{1000 + bin_num}"
        zone_id = ZONES[bin_num % len(ZONES)]  # metadata only, written to a separate bins.csv
        base_rate = ZONE_FILL_RATE_PER_HOUR[zone_id]
        lat_c, lng_c = ZONE_LOCATION_CENTER[zone_id]
        lat = lat_c + np.random.uniform(-0.003, 0.003)
        lng = lng_c + np.random.uniform(-0.003, 0.003)

        fill_level = np.random.uniform(0, 20)
        battery = np.random.uniform(85, 100)
        collection_indices = set(
            np.random.choice(range(5, num_readings), size=np.random.randint(4, 8), replace=False)
        )
        last_collection_time = START_DATE

        current_time = START_DATE
        for i in range(num_readings):
            hour = current_time.hour
            rate_this_step = base_rate * (READING_INTERVAL_MIN / 60.0)
            noise = np.random.normal(0, 0.6)
            fill_level = float(np.clip(fill_level + rate_this_step + noise, 0, 100))

            if i in collection_indices:
                fill_level = float(np.random.uniform(0, 8))
                last_collection_time = current_time

            battery -= np.random.uniform(0.01, 0.05)
            battery = float(np.clip(battery, 0, 100))

            temperature = 27 + 6 * np.sin((hour - 6) / 24 * 2 * np.pi) + np.random.normal(0, 1)
            last_collection_minutes_ago = (current_time - last_collection_time).total_seconds() / 60.0

            raw_rows.append({
                "bin_id": bin_id,
                "fill_level": round(fill_level, 1),
                "battery": round(battery, 1),
                "temperature": round(float(temperature), 1),
                "location_lat": round(lat, 6),
                "location_lng": round(lng, 6),
                "timestamp": current_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "device_id": device_id,
                "_zone_id_metadata": zone_id,  # kept for the priority-score demo only, not a model feature
                "_last_collection_minutes_ago": round(last_collection_minutes_ago, 1),  # experimental feature
            })
            current_time += timedelta(minutes=READING_INTERVAL_MIN)

    raw_df = pd.DataFrame(raw_rows)
    raw_df.drop(columns=["_zone_id_metadata", "_last_collection_minutes_ago"]).to_csv("bin_sensor_data.csv", index=False)
    print("Saved bin_sensor_data.csv:", raw_df.shape[0], "rows (matches sensor payload schema)")

    # bins.csv - separate metadata table (mirrors Main DB "bins" table: bin_id -> zone)
    bins_meta = raw_df[["bin_id", "_zone_id_metadata"]].drop_duplicates().rename(
        columns={"_zone_id_metadata": "zone_id"})
    bins_meta.to_csv("bins_metadata.csv", index=False)
    print("Saved bins_metadata.csv:", bins_meta.shape[0], "bins")

    # -----------------------------------------------------------------
    # Build training_dataset.csv - ONLY the features the diagram calls for
    # -----------------------------------------------------------------
    raw_df["timestamp_dt"] = pd.to_datetime(raw_df["timestamp"])
    raw_df = raw_df.sort_values(["bin_id", "timestamp_dt"]).reset_index(drop=True)

    # fill rate (trend / rate of change) - the diagram's primary feature
    raw_df["prev_fill"] = raw_df.groupby("bin_id")["fill_level"].shift(1)
    raw_df["prev_time"] = raw_df.groupby("bin_id")["timestamp_dt"].shift(1)
    hrs = (raw_df["timestamp_dt"] - raw_df["prev_time"]).dt.total_seconds() / 3600.0
    raw_df["fill_rate_per_hour"] = np.where(hrs > 0, (raw_df["fill_level"] - raw_df["prev_fill"]) / hrs, 0.0)
    raw_df["fill_rate_per_hour"] = raw_df["fill_rate_per_hour"].fillna(0.0)

    # AUDIT FIX: clip reset-event artifacts (bin just emptied -> huge negative
    # "rate" that isn't a real trend) at the SOURCE, not just at serving time,
    # so train_model.py doesn't need to guess the training data is already clean.
    from features import clip_fill_rate
    n_before_clip = len(raw_df)
    outlier_mask = (raw_df["fill_rate_per_hour"] < -20) | (raw_df["fill_rate_per_hour"] > 50)
    raw_df.loc[outlier_mask, "fill_rate_per_hour"] = 0.0
    print(f"Clipped {outlier_mask.sum()}/{n_before_clip} rows with implausible fill_rate_per_hour at generation time")

    raw_df["hour"] = raw_df["timestamp_dt"].dt.hour
    raw_df["day_of_week"] = raw_df["timestamp_dt"].dt.dayofweek

    # AUDIT FIX (censoring-boundary artifact): the previous version capped
    # the look-ahead window at "however much data is left until the dataset
    # ends" - which means rows near the end of the recording period had a
    # much SHORTER effective observation window than rows near the start.
    # That silently skewed the "exact" (non-censored) subset's label
    # distribution as a function of time: rows late in the timeline could
    # only be labeled "exact" if they crossed 90% almost immediately, since
    # anything with more runway ran out of data and got censored instead.
    # Combined with a time-based train/val/test split, this meant the TEST
    # set's exact-labeled rows were a systematically different, heavily
    # skewed sample from train/val's - not real concept drift, just an
    # artifact of how the window was computed. (Confirmed: this alone was
    # enough to swing test R2 from a healthy 0.55 on validation to -11.6
    # on test with the old logic.)
    #
    # Fix: use a FIXED look-ahead horizon (HORIZON_HOURS) for every row,
    # and only keep rows that have at least a full horizon of future data
    # available - so "censored" always means the same thing (didn't cross
    # 90% within a full, consistent HORIZON_HOURS window), regardless of
    # where in the timeline the row falls.
    def compute_time_to_full(group):
        vals = group["fill_level"].values
        times = group["timestamp_dt"].values
        n = len(vals)
        ttf = np.full(n, np.nan)
        censored = np.zeros(n, dtype=bool)
        has_full_horizon = np.zeros(n, dtype=bool)
        data_end = times[-1]
        horizon_delta = np.timedelta64(HORIZON_HOURS, "h")

        for idx in range(n):
            deadline = times[idx] + horizon_delta
            if deadline > data_end:
                continue  # not enough future data for a fair fixed-horizon label - excluded below
            has_full_horizon[idx] = True
            found = False
            for j in range(idx, n):
                if times[j] > deadline:
                    break
                if vals[j] >= OVERFLOW_THRESHOLD:
                    ttf[idx] = (times[j] - times[idx]) / np.timedelta64(1, "h")
                    found = True
                    break
            if not found:
                ttf[idx] = HORIZON_HOURS
                censored[idx] = True
        return pd.DataFrame(
            {"time_to_full_hours": ttf, "is_censored": censored, "has_full_horizon": has_full_horizon},
            index=group.index,
        )

    ttf_result = raw_df.groupby("bin_id", group_keys=False).apply(compute_time_to_full)
    raw_df["time_to_full_hours"] = ttf_result["time_to_full_hours"]
    raw_df["is_censored"] = ttf_result["is_censored"]
    raw_df["has_full_horizon"] = ttf_result["has_full_horizon"]

    n_excluded = (~raw_df["has_full_horizon"]).sum()
    raw_df = raw_df[raw_df["has_full_horizon"]].copy()
    n_censored = raw_df["is_censored"].sum()
    print(f"Excluded {n_excluded} rows lacking a full {HORIZON_HOURS}h look-ahead window "
          f"(near end of each bin's recorded period - can't compute a fair fixed-horizon label for these)")
    print(f"Of {len(raw_df)} remaining rows: {n_censored} censored ({100*n_censored/len(raw_df):.1f}%), "
          f"label = capped at {HORIZON_HOURS}h (lower bound, not exact)")

    training_df = raw_df[[
        "bin_id", "timestamp", "fill_level", "fill_rate_per_hour", "hour", "day_of_week",
        "_last_collection_minutes_ago", "time_to_full_hours", "is_censored",
    ]].rename(columns={"_last_collection_minutes_ago": "last_collection_minutes_ago"})
    training_df.to_csv("training_dataset.csv", index=False)
    print("Saved training_dataset.csv:", training_df.shape)
    print(training_df.head(5))


if __name__ == "__main__":
    main()
