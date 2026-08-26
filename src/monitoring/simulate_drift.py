"""
simulate_drift.py

Simulates a batch of "new" trips arriving after deployment, under a
festival / rush-hour surge scenario -- exactly the kind of event the brief
asks us to simulate for Week 4 ("e.g., festival/rush-hour surge").

Design: this deliberately mirrors generate_synthetic_data.generate_raw_trips
(same city bounding box, same haversine distance, same base speed model) so
the ONLY thing that changes is the data-generating process we're shifting --
everything else stays comparable to the training distribution. Three things
shift at once, stacked to represent a real surge event:

  1. Rush-hour trips become far more common (35% -> 65% of all trips), and
     the traffic slowdown during rush hour gets worse (1.6x -> 2.1x), since
     festival crowds compound normal rush-hour congestion.
  2. Storm/rain weather becomes more common (city monsoon window), which
     independently slows trips down further.
  3. Average trip distance increases slightly (people traveling further to
     festival venues rather than short local hops).

Ground-truth trip_duration_min is generated from this SHIFTED process, so
when we score these trips with the model trained on the ORIGINAL
distribution, any accuracy degradation we see is real and attributable to
drift -- not just noise.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from generate_synthetic_data import CITY_LAT_RANGE, CITY_LON_RANGE, haversine_km  # noqa: E402

DRIFT_SEED = 2026  # deliberately different from RNG_SEED=42 used for training data
SURGE_WEATHER_WEIGHTS = [0.45, 0.30, 0.10, 0.15]  # clear, rain, fog, storm (vs. 0.70/0.18/0.08/0.04 baseline)
SURGE_RUSH_HOUR_SHARE = 0.65  # vs. natural ~35% baseline rate
SURGE_RUSH_MULTIPLIER = 2.1  # vs. 1.6 baseline
SURGE_DISTANCE_UPLIFT = 1.20  # +20% average trip distance


def generate_drifted_batch(n_rows: int = 3000, start_date: str = "2025-06-01", seed: int = DRIFT_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    start = datetime.fromisoformat(start_date)
    # concentrate timestamps into a 10-day festival window, weighted toward evenings
    day_offsets = rng.integers(0, 10, size=n_rows)
    is_forced_rush = rng.random(n_rows) < SURGE_RUSH_HOUR_SHARE
    rush_hours = rng.choice([8, 9, 18, 19, 20], size=n_rows)
    other_hours = rng.integers(0, 24, size=n_rows)
    hour_of_day = np.where(is_forced_rush, rush_hours, other_hours)
    minute_offsets = rng.integers(0, 3600, size=n_rows)
    pickup_times = [
        start + timedelta(days=int(d), hours=int(h), seconds=int(m))
        for d, h, m in zip(day_offsets, hour_of_day, minute_offsets)
    ]

    pickup_lat = rng.uniform(*CITY_LAT_RANGE, n_rows)
    pickup_lon = rng.uniform(*CITY_LON_RANGE, n_rows)
    dropoff_lat = rng.uniform(*CITY_LAT_RANGE, n_rows)
    dropoff_lon = rng.uniform(*CITY_LON_RANGE, n_rows)

    distance_km = haversine_km(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
    distance_km = distance_km * rng.uniform(1.2, 1.6, n_rows) * SURGE_DISTANCE_UPLIFT

    weather = rng.choice(["clear", "rain", "fog", "storm"], size=n_rows, p=SURGE_WEATHER_WEIGHTS)
    temperature_c = rng.normal(24, 6, n_rows)  # slightly cooler/wetter than baseline's 26±5

    is_weekend = np.array([t.weekday() >= 5 for t in pickup_times])
    rush_hour = np.isin(hour_of_day, [8, 9, 18, 19, 20])
    traffic_multiplier = np.select(
        [rush_hour & ~is_weekend, ~rush_hour & ~is_weekend, is_weekend],
        [SURGE_RUSH_MULTIPLIER, 1.0, 1.25],
        default=1.0,
    )
    weather_multiplier = pd.Series(weather).map(
        {"clear": 1.0, "rain": 1.25, "fog": 1.15, "storm": 1.45}
    ).to_numpy()

    base_speed_kmh = 28.0
    effective_speed = base_speed_kmh / (traffic_multiplier * weather_multiplier)
    duration_hours = distance_km / effective_speed
    noise = rng.normal(0, 0.05, n_rows)
    trip_duration_min = np.clip((duration_hours + noise) * 60, 1, None)

    df = pd.DataFrame({
        "trip_id": [f"D{i:07d}" for i in range(n_rows)],
        "pickup_datetime": pickup_times,
        "pickup_lat": pickup_lat,
        "pickup_lon": pickup_lon,
        "dropoff_lat": dropoff_lat,
        "dropoff_lon": dropoff_lon,
        "weather": weather,
        "temperature_c": temperature_c,
        "trip_duration_min": trip_duration_min,
    })
    return df


if __name__ == "__main__":
    out_path = Path(__file__).resolve().parents[2] / "monitoring" / "drifted_batch_raw.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_drifted_batch()
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} drifted (festival/rush-hour surge) rows to {out_path}")
    print(f"Rush-hour share: {df['pickup_datetime'].apply(lambda t: t.hour in (8,9,18,19,20)).mean():.2%}")
    print(f"Storm/rain share: {df['weather'].isin(['storm','rain']).mean():.2%}")
