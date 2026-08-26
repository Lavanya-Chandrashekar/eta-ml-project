"""
generate_synthetic_data.py

Simulates a raw trip-data export from a delivery/ride-hailing platform.

Design notes (for README / justification writeup):
- We use a synthetic dataset (explicitly permitted by the assignment brief as an
  alternative to the Kaggle NYC Taxi dataset) so that the whole pipeline is
  reproducible without external downloads, and so we can *control* the
  ground-truth data-generating process. That control is what lets us later
  simulate a realistic, targeted drift event in Week 4 (a festival/rush-hour
  surge) and know exactly what changed.
- We deliberately inject realistic data quality problems (missing GPS pings,
  malformed timestamps, impossible values) so the Week 1 validation step has
  real issues to catch, instead of validating already-clean data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

RNG_SEED = 42  # single seed constant used everywhere in this project for reproducibility

# Rough bounding box for a mid-size city (used to generate plausible pickup/drop-off coords)
CITY_LAT_RANGE = (12.90, 13.05)
CITY_LON_RANGE = (77.55, 77.70)

WEATHER_CONDITIONS = ["clear", "rain", "fog", "storm"]
WEATHER_WEIGHTS = [0.70, 0.18, 0.08, 0.04]


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two points, in kilometers."""
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def generate_raw_trips(n_rows: int = 15000, start_date: str = "2025-01-01", days: int = 120, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    start = datetime.fromisoformat(start_date)
    # random second-offsets across the date range
    offsets = rng.integers(0, days * 24 * 3600, size=n_rows)
    pickup_times = [start + timedelta(seconds=int(s)) for s in offsets]

    pickup_lat = rng.uniform(*CITY_LAT_RANGE, n_rows)
    pickup_lon = rng.uniform(*CITY_LON_RANGE, n_rows)
    dropoff_lat = rng.uniform(*CITY_LAT_RANGE, n_rows)
    dropoff_lon = rng.uniform(*CITY_LON_RANGE, n_rows)

    distance_km = haversine_km(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
    # add road-network inefficiency: real trips are ~1.2-1.6x the straight-line distance
    distance_km = distance_km * rng.uniform(1.2, 1.6, n_rows)

    weather = rng.choice(WEATHER_CONDITIONS, size=n_rows, p=WEATHER_WEIGHTS)
    temperature_c = rng.normal(26, 5, n_rows)

    hour_of_day = np.array([t.hour for t in pickup_times])
    is_weekend = np.array([t.weekday() >= 5 for t in pickup_times])

    # --- traffic level depends on hour of day + weekend (this is the "ground truth"
    # relationship the model has to learn, and the one we'll later shift for drift) ---
    rush_hour = np.isin(hour_of_day, [8, 9, 18, 19, 20])
    traffic_multiplier = np.select(
        [rush_hour & ~is_weekend, ~rush_hour & ~is_weekend, is_weekend],
        [1.6, 1.0, 1.15],
        default=1.0,
    )

    weather_multiplier = pd.Series(weather).map({"clear": 1.0, "rain": 1.25, "fog": 1.15, "storm": 1.45}).to_numpy()

    base_speed_kmh = 28.0  # baseline average speed in city traffic
    effective_speed = base_speed_kmh / (traffic_multiplier * weather_multiplier)
    duration_hours = distance_km / effective_speed
    noise = rng.normal(0, 0.05, n_rows)  # +/- driver/route noise
    trip_duration_min = np.clip((duration_hours + noise) * 60, 1, None)

    df = pd.DataFrame({
        "trip_id": [f"T{i:07d}" for i in range(n_rows)],
        "pickup_datetime": pickup_times,
        "pickup_lat": pickup_lat,
        "pickup_lon": pickup_lon,
        "dropoff_lat": dropoff_lat,
        "dropoff_lon": dropoff_lon,
        "weather": weather,
        "temperature_c": temperature_c,
        "trip_duration_min": trip_duration_min,
    })

    # ------------------------------------------------------------------
    # Inject realistic data-quality issues (so Week 1 validation has real
    # work to do). Percentages are intentionally small but nonzero.
    # ------------------------------------------------------------------
    n = len(df)
    idx = rng.permutation(n)

    # 1. Missing GPS pings (~1.5% of rows lose dropoff coordinates)
    missing_gps_idx = idx[: int(0.015 * n)]
    df.loc[missing_gps_idx, ["dropoff_lat", "dropoff_lon"]] = np.nan

    # 2. Invalid/malformed timestamps (~1% become strings that won't parse, or nulls)
    bad_ts_idx = idx[int(0.015 * n): int(0.025 * n)]
    df["pickup_datetime"] = df["pickup_datetime"].astype(object)
    half = len(bad_ts_idx) // 2
    df.loc[bad_ts_idx[:half], "pickup_datetime"] = "not-a-timestamp"
    df.loc[bad_ts_idx[half:], "pickup_datetime"] = None

    # 3. Out-of-range coordinates (~0.5%) e.g. GPS glitch far outside the city
    bad_coord_idx = idx[int(0.025 * n): int(0.03 * n)]
    df.loc[bad_coord_idx, "pickup_lat"] = rng.uniform(-90, 90, len(bad_coord_idx))
    df.loc[bad_coord_idx, "pickup_lon"] = rng.uniform(-180, 180, len(bad_coord_idx))

    # 4. Negative / impossible duration (~0.3%), e.g. clock-sync bug
    bad_dur_idx = idx[int(0.03 * n): int(0.033 * n)]
    df.loc[bad_dur_idx, "trip_duration_min"] = -rng.uniform(1, 10, len(bad_dur_idx))

    # 5. Exact duplicate rows (~0.5%) to also exercise uniqueness checks
    dup_idx = idx[int(0.033 * n): int(0.038 * n)]
    dup_rows = df.loc[dup_idx].copy()
    df = pd.concat([df, dup_rows], ignore_index=True)

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle
    return df


if __name__ == "__main__":
    out_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "trips_raw.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_raw_trips()
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
