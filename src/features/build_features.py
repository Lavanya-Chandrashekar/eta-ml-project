"""
build_features.py

Turns validated raw trips into a model-ready feature table.

Features engineered:
  - distance_km              : haversine distance between pickup/dropoff (x1.35 road factor)
  - hour_of_day               : 0-23
  - day_of_week                : 0=Mon .. 6=Sun
  - is_weekend                 : bool
  - is_rush_hour                : bool (8-9am, 6-8pm on weekdays)
  - time_of_day_bucket         : morning/afternoon/evening/night (categorical -> one-hot)
  - weather                    : categorical -> one-hot
  - temperature_c              : passthrough numeric

Target: trip_duration_min
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from generate_synthetic_data import haversine_km  # noqa: E402

ROAD_NETWORK_FACTOR = 1.35  # matches the midpoint of the generator's 1.2-1.6 range

CATEGORICAL_COLUMNS = ["weather", "time_of_day_bucket"]
NUMERIC_FEATURE_COLUMNS = [
    "distance_km", "hour_of_day", "day_of_week", "is_weekend",
    "is_rush_hour", "temperature_c",
]
TARGET_COLUMN = "trip_duration_min"


def _time_of_day_bucket(hour: int) -> str:
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 21:
        return "evening"
    return "night"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

    df["distance_km"] = haversine_km(
        df["pickup_lat"], df["pickup_lon"], df["dropoff_lat"], df["dropoff_lon"]
    ) * ROAD_NETWORK_FACTOR

    df["hour_of_day"] = df["pickup_datetime"].dt.hour
    df["day_of_week"] = df["pickup_datetime"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_rush_hour"] = (
        df["hour_of_day"].isin([8, 9, 18, 19, 20]) & (df["is_weekend"] == 0)
    ).astype(int)
    df["time_of_day_bucket"] = df["hour_of_day"].apply(_time_of_day_bucket)

    feature_cols = NUMERIC_FEATURE_COLUMNS + CATEGORICAL_COLUMNS
    out = df[["trip_id"] + feature_cols + [TARGET_COLUMN]].copy()

    # one-hot encode categoricals; keep column set stable & documented so
    # training and serving use an identical schema (this list is saved as an
    # artifact alongside the model in Week 3 to avoid train/serve skew).
    out = pd.get_dummies(out, columns=CATEGORICAL_COLUMNS, prefix=CATEGORICAL_COLUMNS)

    return out


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[2]
    in_path = base / "data" / "processed" / "trips_validated.csv"
    out_path = base / "data" / "processed" / "trips_features.csv"

    df = pd.read_csv(in_path)
    features_df = build_features(df)
    features_df.to_csv(out_path, index=False)

    print(f"Feature table: {features_df.shape[0]} rows x {features_df.shape[1]} columns")
    print(f"Columns: {list(features_df.columns)}")
    print(f"Written to {out_path}")
