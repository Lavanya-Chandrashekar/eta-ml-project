"""
score_batch.py

Runs the (validated, feature-built) drifted batch through the deployed model
-- reusing the exact same validate_data / build_features code Week 1 used,
so the scored batch went through the identical path production traffic
would -- and logs prediction vs. ground-truth-actual per row.

In real production you wouldn't have the ground-truth duration at prediction
time (that's the whole point of drift monitoring -- you have to detect
degradation before you get labels back). Here, because we're simulating,
we keep the actual trip_duration_min alongside each prediction so
monitor.py can compute real accuracy metrics on the drifted batch, not just
feature drift. Predictions themselves are made the exact same way api/app.py
would score a live request.
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "src" / "data"))
sys.path.insert(0, str(BASE / "src" / "features"))
from validate_data import validate  # noqa: E402
from build_features import build_features  # noqa: E402

MODEL_PATH = BASE / "models" / "best_model.joblib"
FEATURE_COLUMNS_PATH = BASE / "models" / "best_model_feature_columns.json"
DRIFTED_RAW_PATH = BASE / "monitoring" / "drifted_batch_raw.csv"
OUT_PATH = BASE / "monitoring" / "predictions_vs_actuals.jsonl"


def score_batch(raw_df: pd.DataFrame) -> pd.DataFrame:
    model = joblib.load(MODEL_PATH)
    feature_columns = json.loads(FEATURE_COLUMNS_PATH.read_text())
    model_version = MODEL_PATH.stat().st_mtime_ns  # same versioning convention as api/app.py

    clean_df, validation_report = validate(raw_df)
    features_df = build_features(clean_df)

    X = features_df.reindex(columns=feature_columns, fill_value=0)
    start = time.perf_counter()
    preds = model.predict(X)
    latency_ms = round((time.perf_counter() - start) * 1000 / max(len(X), 1), 3)  # avg per-row

    scored = features_df.copy()
    scored["predicted_duration_min"] = preds
    scored["model_version"] = model_version
    scored["scored_at"] = datetime.now(timezone.utc).isoformat()
    scored["avg_latency_ms"] = latency_ms

    print(f"Validation on drifted batch: {validation_report['output_rows']}/{validation_report['input_rows']} "
          f"rows passed (pass_rate={validation_report['pass_rate']:.2%})")
    return scored


if __name__ == "__main__":
    if not DRIFTED_RAW_PATH.exists():
        raise FileNotFoundError(
            f"{DRIFTED_RAW_PATH} not found -- run simulate_drift.py first to generate the drifted batch."
        )

    raw_df = pd.read_csv(DRIFTED_RAW_PATH)
    scored = score_batch(raw_df)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    feature_columns = json.loads(FEATURE_COLUMNS_PATH.read_text())  # log ALL model features (incl. one-hot), for PSI
    log_cols = (
        ["trip_id", "predicted_duration_min", "trip_duration_min", "model_version", "scored_at", "avg_latency_ms"]
        + feature_columns
    )
    with open(OUT_PATH, "w") as f:
        for _, row in scored[log_cols].rename(columns={"trip_duration_min": "actual_duration_min"}).iterrows():
            f.write(json.dumps(row.to_dict(), default=str) + "\n")

    mae = (scored["predicted_duration_min"] - scored["trip_duration_min"]).abs().mean()
    print(f"Scored {len(scored)} rows -> {OUT_PATH}")
    print(f"Drifted-batch MAE: {mae:.2f} min")
