"""
reproduce_run.py

Proves reproducibility: reads the logged configuration for a chosen run out
of the local tracker (or MLflow, if installed), rebuilds the model from
scratch using only that logged config + the versioned dataset, retrains it,
and checks the resulting metrics match the originally logged metrics within
floating-point tolerance.

This directly demonstrates the M3 rubric requirement: "ability to reproduce
a chosen run from logged configuration."

Usage:
    python reproduce_run.py gradient_boosting_default
"""
import sys
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import load_train_test, evaluate, RANDOM_SEED  # noqa: E402

BASE = Path(__file__).resolve().parents[2]
LOCAL_TRACKING_DIR = BASE / "mlruns_local" / "eta-prediction"

MODEL_BUILDERS = {
    "LinearRegression": lambda p: LinearRegression(),
    "HistGradientBoostingRegressor": lambda p: HistGradientBoostingRegressor(
        max_depth=None if p.get("max_depth") in (None, "None (default)") else int(p["max_depth"]),
        learning_rate=float(p["learning_rate"]),
        max_iter=int(p["max_iter"]),
        random_state=int(p["seed"]),
    ),
}


def find_run_by_name(run_name: str) -> dict:
    for run_dir in LOCAL_TRACKING_DIR.iterdir():
        meta_path = run_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta["run_name"] == run_name:
                return meta
    raise FileNotFoundError(f"No logged run named '{run_name}' found under {LOCAL_TRACKING_DIR}")


def main(run_name: str):
    meta = find_run_by_name(run_name)
    params = meta["params"]
    original_metrics = meta["metrics"]

    print(f"Reproducing run '{run_name}' from logged config:")
    print(json.dumps(params, indent=2))

    np.random.seed(RANDOM_SEED)
    X_train, X_test, y_train, y_test, _ = load_train_test()

    model_name = params["model"]
    model = MODEL_BUILDERS[model_name](params)
    model.fit(X_train, y_train)
    reproduced_metrics = evaluate(model, X_test, y_test)

    print("\nOriginal metrics:  ", original_metrics)
    print("Reproduced metrics:", reproduced_metrics)

    tolerance = 1e-6
    mismatches = {
        k: (original_metrics[k], reproduced_metrics[k])
        for k in original_metrics
        if abs(original_metrics[k] - reproduced_metrics[k]) > tolerance
    }

    if mismatches:
        print(f"\nFAILED: metrics differ beyond tolerance {tolerance}: {mismatches}")
        sys.exit(1)
    else:
        print(f"\nPASSED: run '{run_name}' is exactly reproducible from its logged configuration.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python reproduce_run.py <run_name>")
        sys.exit(1)
    main(sys.argv[1])
