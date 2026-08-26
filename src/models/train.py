"""
train.py

Trains and compares models to predict trip_duration_min, tracking every run.

Experiments run (>= 2 required by the brief; we run 4 for a fair comparison):
  1. linear_regression                      -- simple, interpretable baseline
  2. gradient_boosting_default               -- sklearn HistGradientBoostingRegressor, default depth
  3. gradient_boosting_tuned_shallow         -- fewer/shallower trees (regularized)
  4. gradient_boosting_tuned_deep            -- more leaves, lower learning rate

Note on XGBoost: the brief's example uses "linear regression vs. gradient
boosting" -- we use scikit-learn's HistGradientBoostingRegressor rather than
XGBoost because XGBoost could not be installed in the sandbox this was
developed in (no internet access). HistGradientBoostingRegressor is the same
family of algorithm (histogram-based gradient-boosted trees) and is a
like-for-like substitute; swap in `xgboost.XGBRegressor` with the same
train/log structure if you have it installed and prefer it.

Reproducibility: a single RANDOM_SEED constant seeds numpy, the train/test
split, and every model's internal randomness, and is logged as a parameter
with every run -- see the Week1 quiz question on why reproducibility must be
engineered, not assumed.
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tracking"))
from tracker import ExperimentTracker  # noqa: E402
from version_model import version_model  # noqa: E402 (same dir as this script, on sys.path by default)

RANDOM_SEED = 42
EXPERIMENT_NAME = "eta-prediction"

BASE = Path(__file__).resolve().parents[2]
FEATURES_PATH = BASE / "data" / "processed" / "trips_features.csv"
MODELS_DIR = BASE / "models"
REPORT_PATH = BASE / "reports" / "model_comparison.json"


def load_train_test():
    df = pd.read_csv(FEATURES_PATH)
    X = df.drop(columns=["trip_id", "trip_duration_min"])
    y = df["trip_duration_min"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    return X_train, X_test, y_train, y_test, list(X.columns)


def evaluate(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    return {
        "mae_minutes": float(mean_absolute_error(y_test, preds)),
        "rmse_minutes": float(np.sqrt(mean_squared_error(y_test, preds))),
        "r2": float(r2_score(y_test, preds)),
    }


EXPERIMENTS = [
    {
        "run_name": "linear_regression",
        "params": {"model": "LinearRegression", "seed": RANDOM_SEED},
        "build": lambda: LinearRegression(),
    },
    {
        "run_name": "gradient_boosting_default",
        "params": {
            "model": "HistGradientBoostingRegressor", "max_depth": "None (default)",
            "learning_rate": 0.1, "max_iter": 100, "seed": RANDOM_SEED,
        },
        "build": lambda: HistGradientBoostingRegressor(random_state=RANDOM_SEED),
    },
    {
        "run_name": "gradient_boosting_tuned_shallow",
        "params": {
            "model": "HistGradientBoostingRegressor", "max_depth": 4,
            "learning_rate": 0.1, "max_iter": 150, "seed": RANDOM_SEED,
        },
        "build": lambda: HistGradientBoostingRegressor(
            max_depth=4, learning_rate=0.1, max_iter=150, random_state=RANDOM_SEED
        ),
    },
    {
        "run_name": "gradient_boosting_tuned_deep",
        "params": {
            "model": "HistGradientBoostingRegressor", "max_depth": 8,
            "learning_rate": 0.05, "max_iter": 300, "seed": RANDOM_SEED,
        },
        "build": lambda: HistGradientBoostingRegressor(
            max_depth=8, learning_rate=0.05, max_iter=300, random_state=RANDOM_SEED
        ),
    },
]


def main():
    np.random.seed(RANDOM_SEED)  # belt-and-braces: seed the global numpy RNG too

    X_train, X_test, y_train, y_test, feature_columns = load_train_test()
    MODELS_DIR.mkdir(exist_ok=True)
    REPORT_PATH.parent.mkdir(exist_ok=True)

    tracker = ExperimentTracker(EXPERIMENT_NAME)
    results = []

    for exp in EXPERIMENTS:
        model = exp["build"]()
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)

        model_path = MODELS_DIR / f"{exp['run_name']}.joblib"
        joblib.dump(model, model_path)

        run_params = {**exp["params"], "n_train": len(X_train), "n_test": len(X_test)}
        with tracker.start_run(exp["run_name"]) as run:
            run.log_params(run_params)
            run.log_metrics(metrics)
            run.log_artifact(model_path)

        results.append({"run_name": exp["run_name"], "params": run_params, "metrics": metrics})
        print(f"[{exp['run_name']}] MAE={metrics['mae_minutes']:.3f}min "
              f"RMSE={metrics['rmse_minutes']:.3f}min R2={metrics['r2']:.4f}")

    # pick best by RMSE (lower is better) -- documented choice: RMSE penalizes
    # large ETA misses more heavily than MAE, which matters more for user trust
    # than being right on average across many small trips.
    best = min(results, key=lambda r: r["metrics"]["rmse_minutes"])

    report = {
        "experiment_name": EXPERIMENT_NAME,
        "random_seed": RANDOM_SEED,
        "feature_columns": feature_columns,
        "selection_metric": "rmse_minutes (lower is better)",
        "results": results,
        "best_model": best["run_name"],
        "best_model_metrics": best["metrics"],
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    # promote the winning model to a stable path used by the serving API
    best_model_path = MODELS_DIR / f"{best['run_name']}.joblib"
    promoted_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(joblib.load(best_model_path), promoted_path)
    (MODELS_DIR / "best_model_feature_columns.json").write_text(json.dumps(feature_columns, indent=2))

    version_entry = version_model(promoted_path, "latest-best-model", best["metrics"], {"run_name": best["run_name"]})

    print(f"\nBest model: {best['run_name']} (RMSE={best['metrics']['rmse_minutes']:.3f} min)")
    print(f"Promoted to {promoted_path}")
    print(f"Comparison report written to {REPORT_PATH}")
    print(f"Model version manifest updated: sha256={version_entry['sha256'][:12]}...")


if __name__ == "__main__":
    main()
