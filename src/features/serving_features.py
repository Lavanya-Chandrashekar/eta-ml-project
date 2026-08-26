"""
serving_features.py

Turns a single raw API request into the exact same feature vector shape the
model was trained on. This intentionally reuses the SAME transformation
logic as build_features.py (same haversine distance, same road-network
factor, same time-of-day bucketing) so there is no train/serve skew -- this
is the concrete answer to the Week1 quiz question "why must transformation
parameters be saved with the model artifact": here there's no *fitted*
scaler to save (we don't standardize numeric features for a tree/linear
model), but the feature *schema* (exact column list, incl. one-hot
categories) absolutely must travel with the model, which is why
`best_model_feature_columns.json` is saved alongside `best_model.joblib` in
train.py and loaded once at API startup.
"""
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from generate_synthetic_data import haversine_km  # noqa: E402

ROAD_NETWORK_FACTOR = 1.35
VALID_WEATHER = {"clear", "rain", "fog", "storm", "unknown"}


def _time_of_day_bucket(hour: int) -> str:
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 21:
        return "evening"
    return "night"


def raw_request_to_feature_row(request: dict) -> dict:
    """Build the same numeric/one-hot feature dict build_features.py would produce for one trip."""
    pickup_dt = request["pickup_datetime"]
    if isinstance(pickup_dt, str):
        pickup_dt = datetime.fromisoformat(pickup_dt)

    distance_km = float(haversine_km(
        request["pickup_lat"], request["pickup_lon"],
        request["dropoff_lat"], request["dropoff_lon"],
    )) * ROAD_NETWORK_FACTOR

    hour_of_day = pickup_dt.hour
    day_of_week = pickup_dt.weekday()
    is_weekend = int(day_of_week >= 5)
    is_rush_hour = int(hour_of_day in (8, 9, 18, 19, 20) and not is_weekend)
    bucket = _time_of_day_bucket(hour_of_day)
    weather = request.get("weather", "unknown")
    if weather not in VALID_WEATHER:
        weather = "unknown"

    row = {
        "distance_km": distance_km,
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "is_rush_hour": is_rush_hour,
        "temperature_c": float(request.get("temperature_c", 25.0)),
        "weather_clear": int(weather == "clear"),
        "weather_fog": int(weather == "fog"),
        "weather_rain": int(weather == "rain"),
        "weather_storm": int(weather == "storm"),
        "time_of_day_bucket_afternoon": int(bucket == "afternoon"),
        "time_of_day_bucket_evening": int(bucket == "evening"),
        "time_of_day_bucket_morning": int(bucket == "morning"),
        "time_of_day_bucket_night": int(bucket == "night"),
    }
    return row


def to_model_input(request: dict, feature_columns: list[str]) -> pd.DataFrame:
    """Build a single-row DataFrame with EXACTLY the columns/order the model was trained on."""
    row = raw_request_to_feature_row(request)
    df = pd.DataFrame([row])
    # align to training schema: add any missing columns as 0, drop unexpected ones,
    # and enforce the exact training column order
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    return df[feature_columns]
