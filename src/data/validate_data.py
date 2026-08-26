"""
validate_data.py

Schema / quality validation for the raw trip export.

Checks implemented (each maps to a real production failure mode):
  1. Required columns present with correct dtypes.
  2. Missing GPS pings (null pickup/dropoff coordinates).
  3. Invalid / unparseable timestamps.
  4. Out-of-range coordinates (outside the service city's bounding box).
  5. Non-positive / implausible trip durations.
  6. Exact duplicate rows (uniqueness).
  7. Missing weather / temperature values.

Design choice: rows that fail a *hard* check (unparseable timestamp, missing
GPS, non-positive duration, out-of-city coordinates) are dropped, because we
cannot safely impute the label or the join key. Rows that fail a *soft* check
(missing weather) are imputed, because a reasonable default exists. Every
decision is counted and written to a validation report, so nothing is
silently discarded (this report is a Week 1 deliverable artifact).
"""

import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from generate_synthetic_data import CITY_LAT_RANGE, CITY_LON_RANGE

REQUIRED_COLUMNS = [
    "trip_id", "pickup_datetime", "pickup_lat", "pickup_lon",
    "dropoff_lat", "dropoff_lon", "weather", "temperature_c", "trip_duration_min",
]

# allow a little slack around the service area for validity checks
LAT_MIN, LAT_MAX = CITY_LAT_RANGE[0] - 0.05, CITY_LAT_RANGE[1] + 0.05
LON_MIN, LON_MAX = CITY_LON_RANGE[0] - 0.05, CITY_LON_RANGE[1] + 0.05

MAX_PLAUSIBLE_DURATION_MIN = 240  # 4 hours; anything above this is almost certainly bad data


def validate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "input_rows": len(df),
        "checks": {},
    }

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Schema violation: missing required columns {missing_cols}")

    df = df.copy()

    # --- 1. Parse timestamps; invalid/unparseable -> NaT ---
    parsed_ts = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    invalid_ts_mask = parsed_ts.isna()
    report["checks"]["invalid_timestamps"] = int(invalid_ts_mask.sum())
    df["pickup_datetime"] = parsed_ts

    # --- 2. Missing GPS pings ---
    missing_gps_mask = df[["pickup_lat", "pickup_lon", "dropoff_lat", "dropoff_lon"]].isna().any(axis=1)
    report["checks"]["missing_gps_pings"] = int(missing_gps_mask.sum())

    # --- 3. Out-of-range coordinates ---
    out_of_range_mask = (
        ~df["pickup_lat"].between(LAT_MIN, LAT_MAX)
        | ~df["pickup_lon"].between(LON_MIN, LON_MAX)
        | ~df["dropoff_lat"].between(LAT_MIN, LAT_MAX)
        | ~df["dropoff_lon"].between(LON_MIN, LON_MAX)
    )
    # rows already missing GPS shouldn't double count as "out of range"
    out_of_range_mask = out_of_range_mask & ~missing_gps_mask
    report["checks"]["out_of_range_coordinates"] = int(out_of_range_mask.sum())

    # --- 4. Non-positive / implausible duration ---
    bad_duration_mask = (df["trip_duration_min"] <= 0) | (df["trip_duration_min"] > MAX_PLAUSIBLE_DURATION_MIN)
    report["checks"]["invalid_duration"] = int(bad_duration_mask.sum())

    # --- 5. Duplicate rows (exact duplicates across all columns except trip_id) ---
    dup_mask = df.drop(columns=["trip_id"]).duplicated(keep="first")
    report["checks"]["duplicate_rows"] = int(dup_mask.sum())

    # --- 6. Missing weather / temperature (soft check -> impute) ---
    missing_weather_mask = df["weather"].isna()
    missing_temp_mask = df["temperature_c"].isna()
    report["checks"]["missing_weather_imputed"] = int(missing_weather_mask.sum())
    report["checks"]["missing_temperature_imputed"] = int(missing_temp_mask.sum())
    df.loc[missing_weather_mask, "weather"] = "unknown"
    df.loc[missing_temp_mask, "temperature_c"] = df["temperature_c"].median()

    # ------------------------------------------------------------------
    # Drop rows failing any hard check
    # ------------------------------------------------------------------
    hard_fail_mask = invalid_ts_mask | missing_gps_mask | out_of_range_mask | bad_duration_mask | dup_mask
    clean_df = df.loc[~hard_fail_mask].reset_index(drop=True)

    report["rows_dropped"] = int(hard_fail_mask.sum())
    report["output_rows"] = len(clean_df)
    report["pass_rate"] = round(len(clean_df) / len(df), 4)

    return clean_df, report


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[2]
    raw_path = base / "data" / "raw" / "trips_raw.csv"
    clean_path = base / "data" / "processed" / "trips_validated.csv"
    report_path = base / "data" / "processed" / "validation_report.json"

    raw_df = pd.read_csv(raw_path)
    clean_df, report = validate(raw_df)

    clean_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(clean_path, index=False)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nValidated data written to {clean_path}")
