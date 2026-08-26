"""
Unit tests for the Week 4 monitoring module. Run with:
    python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "monitoring"))
from monitor import population_stability_index, decide_retrain  # noqa: E402
from simulate_drift import generate_drifted_batch  # noqa: E402


class TestPopulationStabilityIndex(unittest.TestCase):
    def test_identical_distributions_have_near_zero_psi(self):
        rng = np.random.default_rng(0)
        ref = pd.Series(rng.normal(0, 1, 5000))
        cur = pd.Series(rng.normal(0, 1, 5000))
        psi = population_stability_index(ref, cur)
        self.assertLess(psi, 0.05)

    def test_shifted_distribution_has_high_psi(self):
        rng = np.random.default_rng(0)
        ref = pd.Series(rng.normal(0, 1, 5000))
        cur = pd.Series(rng.normal(5, 1, 5000))  # completely shifted
        psi = population_stability_index(ref, cur)
        self.assertGreaterEqual(psi, 0.25)

    def test_binary_feature_psi_reflects_proportion_shift(self):
        ref = pd.Series([0] * 800 + [1] * 200)  # 20% positive
        cur_same = pd.Series([0] * 800 + [1] * 200)
        cur_shifted = pd.Series([0] * 200 + [1] * 800)  # 80% positive
        self.assertLess(population_stability_index(ref, cur_same), 0.05)
        self.assertGreater(population_stability_index(ref, cur_shifted), 0.25)


class TestRetrainDecision(unittest.TestCase):
    def _perf(self, rmse_relative_change):
        return {
            "baseline": {"mae_minutes": 3.0, "rmse_minutes": 4.0},
            "current_on_drifted_batch": {"mae_minutes": 3.0, "rmse_minutes": 4.0 * rmse_relative_change},
            "rmse_relative_change": rmse_relative_change,
            "mae_relative_change": rmse_relative_change,
        }

    def test_no_trigger_when_stable(self):
        feature_drift = {"a": 0.02, "b": 0.03}
        decision = decide_retrain(feature_drift, self._perf(1.05))
        self.assertFalse(decision["retrain_triggered"])

    def test_triggers_on_rmse_degradation_alone(self):
        feature_drift = {"a": 0.02, "b": 0.03}
        decision = decide_retrain(feature_drift, self._perf(1.30))
        self.assertTrue(decision["retrain_triggered"])

    def test_triggers_on_multiple_major_feature_shifts_alone(self):
        feature_drift = {"a": 0.30, "b": 0.40, "c": 0.02}
        decision = decide_retrain(feature_drift, self._perf(1.0))
        self.assertTrue(decision["retrain_triggered"])
        self.assertEqual(set(decision["major_shift_features"]), {"a", "b"})

    def test_single_major_shift_feature_does_not_trigger(self):
        feature_drift = {"a": 0.30, "b": 0.02, "c": 0.02}
        decision = decide_retrain(feature_drift, self._perf(1.0))
        self.assertFalse(decision["retrain_triggered"])


class TestSimulateDrift(unittest.TestCase):
    def test_generates_requested_row_count_with_required_columns(self):
        df = generate_drifted_batch(n_rows=200)
        self.assertEqual(len(df), 200)
        for col in ["trip_id", "pickup_datetime", "pickup_lat", "pickup_lon",
                    "dropoff_lat", "dropoff_lon", "weather", "temperature_c", "trip_duration_min"]:
            self.assertIn(col, df.columns)

    def test_rush_hour_share_is_elevated_vs_natural_baseline(self):
        df = generate_drifted_batch(n_rows=2000)
        hours = pd.to_datetime(df["pickup_datetime"]).dt.hour
        rush_share = hours.isin([8, 9, 18, 19, 20]).mean()
        self.assertGreater(rush_share, 0.5)  # natural baseline is ~5/24 hours if uniform


if __name__ == "__main__":
    unittest.main()
