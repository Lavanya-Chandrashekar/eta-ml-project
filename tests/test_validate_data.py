"""
Unit tests for the validation pipeline. Run with:
    python -m unittest discover -s tests -v
(pytest is API-compatible if you have internet to `pip install pytest` on your own machine.)
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "data"))
from validate_data import validate  # noqa: E402


def _valid_row(**overrides):
    row = {
        "trip_id": "T0000001",
        "pickup_datetime": "2025-01-01 10:00:00",
        "pickup_lat": 12.95,
        "pickup_lon": 77.60,
        "dropoff_lat": 12.98,
        "dropoff_lon": 77.63,
        "weather": "clear",
        "temperature_c": 25.0,
        "trip_duration_min": 20.0,
    }
    row.update(overrides)
    return row


class TestValidateData(unittest.TestCase):
    def test_valid_rows_pass_through(self):
        # vary a real field (not just trip_id) so rows aren't flagged as duplicates
        df = pd.DataFrame([
            _valid_row(trip_id=f"T{i}", trip_duration_min=20.0 + i) for i in range(5)
        ])
        clean, report = validate(df)
        self.assertEqual(len(clean), 5)
        self.assertEqual(report["rows_dropped"], 0)

    def test_missing_gps_is_dropped(self):
        df = pd.DataFrame([_valid_row(dropoff_lat=np.nan)])
        clean, report = validate(df)
        self.assertEqual(len(clean), 0)
        self.assertEqual(report["checks"]["missing_gps_pings"], 1)

    def test_invalid_timestamp_is_dropped(self):
        df = pd.DataFrame([_valid_row(pickup_datetime="not-a-date")])
        clean, report = validate(df)
        self.assertEqual(len(clean), 0)
        self.assertEqual(report["checks"]["invalid_timestamps"], 1)

    def test_negative_duration_is_dropped(self):
        df = pd.DataFrame([_valid_row(trip_duration_min=-5.0)])
        clean, report = validate(df)
        self.assertEqual(len(clean), 0)
        self.assertEqual(report["checks"]["invalid_duration"], 1)

    def test_out_of_range_coordinates_dropped(self):
        df = pd.DataFrame([_valid_row(pickup_lat=89.9, pickup_lon=170.0)])
        clean, report = validate(df)
        self.assertEqual(len(clean), 0)
        self.assertEqual(report["checks"]["out_of_range_coordinates"], 1)

    def test_duplicate_rows_deduplicated(self):
        rows = [_valid_row(trip_id="T1"), _valid_row(trip_id="T2")]  # differ only by trip_id
        df = pd.DataFrame(rows)
        clean, report = validate(df)
        self.assertEqual(len(clean), 1)
        self.assertEqual(report["checks"]["duplicate_rows"], 1)

    def test_missing_weather_is_imputed_not_dropped(self):
        df = pd.DataFrame([_valid_row(weather=None)])
        clean, report = validate(df)
        self.assertEqual(len(clean), 1)
        self.assertEqual(clean.loc[0, "weather"], "unknown")
        self.assertEqual(report["checks"]["missing_weather_imputed"], 1)

    def test_missing_schema_column_raises(self):
        df = pd.DataFrame([{"trip_id": "T1"}])
        with self.assertRaises(ValueError):
            validate(df)


if __name__ == "__main__":
    unittest.main()
