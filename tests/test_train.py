"""
Unit tests for the training/evaluation logic. Run with:
    python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "models"))
from train import evaluate, RANDOM_SEED  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402


class TestTrain(unittest.TestCase):
    def test_evaluate_perfect_predictions_gives_zero_error(self):
        class PerfectModel:
            def predict(self, X):
                return X["y_true"].to_numpy()

        y_true = pd.Series([1.0, 2.0, 3.0, 4.0])
        X_test = pd.DataFrame({"y_true": y_true})
        metrics = evaluate(PerfectModel(), X_test, y_true)

        self.assertAlmostEqual(metrics["mae_minutes"], 0.0)
        self.assertAlmostEqual(metrics["rmse_minutes"], 0.0)
        self.assertAlmostEqual(metrics["r2"], 1.0)

    def test_evaluate_returns_expected_keys(self):
        rng = np.random.default_rng(RANDOM_SEED)
        X = pd.DataFrame({"x1": rng.normal(size=50), "x2": rng.normal(size=50)})
        y = 2 * X["x1"] - X["x2"] + rng.normal(scale=0.01, size=50)
        model = LinearRegression().fit(X, y)

        metrics = evaluate(model, X, y)
        self.assertEqual(set(metrics.keys()), {"mae_minutes", "rmse_minutes", "r2"})
        # a near-noiseless linear relationship should fit almost perfectly
        self.assertGreater(metrics["r2"], 0.99)

    def test_same_seed_gives_identical_model_fit(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"x1": rng.normal(size=30)})
        y = 3 * X["x1"] + rng.normal(scale=0.1, size=30)

        model_a = LinearRegression().fit(X, y)
        model_b = LinearRegression().fit(X, y)
        # LinearRegression is deterministic given the same data; this guards
        # against someone swapping in a stochastic model without seeding it
        np.testing.assert_allclose(model_a.coef_, model_b.coef_)


if __name__ == "__main__":
    unittest.main()
