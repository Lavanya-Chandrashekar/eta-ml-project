"""
monitor.py

Week 4 / M5 deliverable: compares the scored drifted batch (from
score_batch.py) against the training-time reference distribution and
baseline test metrics, computes monitoring signals, and applies a
documented, rule-based retraining trigger.

Monitoring signals computed
----------------------------
1. Feature drift -- Population Stability Index (PSI) per feature, comparing
   the reference (training feature table) distribution to the current
   (post-deployment / drifted) batch. PSI is the standard industry metric
   for this (used widely in credit-risk model monitoring, which translates
   directly to this banking-adjacent use case):
       PSI < 0.10            -> no significant shift
       0.10 <= PSI < 0.25     -> moderate shift, watch
       PSI >= 0.25            -> major shift, action needed
   Reference bins are built from deciles of the TRAINING distribution (not
   the current batch), which is what makes PSI directional and comparable
   run over run.

2. Performance drift -- MAE/RMSE on the current batch (we have ground truth
   here because this is a simulation) vs. the baseline test-set metrics
   recorded in reports/model_comparison.json for the deployed model.

Retraining trigger (rule-based, documented here so it's auditable)
--------------------------------------------------------------------
Retrain is triggered if EITHER:
  (a) RMSE on the current batch exceeds 1.25x the baseline test RMSE
      (>=25% relative degradation in prediction accuracy), OR
  (b) 2 or more features show PSI >= 0.25 (major population shift)
This mirrors a common production pattern: trigger on either a *direct*
accuracy signal (a) or a *leading* indicator (b) that would explain future
accuracy loss even before enough labelled data exists to measure it
directly. Two independent conditions reduce false triggers from a single
noisy metric.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE = Path(__file__).resolve().parents[2]
REFERENCE_FEATURES_PATH = BASE / "data" / "processed" / "trips_features.csv"
MODEL_COMPARISON_PATH = BASE / "reports" / "model_comparison.json"
PREDICTIONS_PATH = BASE / "monitoring" / "predictions_vs_actuals.jsonl"
REPORT_PATH = BASE / "monitoring" / "monitoring_report.json"
DECISION_PATH = BASE / "monitoring" / "RETRAIN_DECISION.json"
PLOT_PATH = BASE / "monitoring" / "monitoring_report.png"

RMSE_DEGRADATION_TRIGGER = 1.25  # 25% relative worsening
PSI_MAJOR_SHIFT = 0.25
PSI_MODERATE_SHIFT = 0.10
N_MAJOR_SHIFT_FEATURES_TO_TRIGGER = 2


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """PSI of `current` against `reference`, using reference-derived bin edges."""
    reference = reference.dropna().astype(float)
    current = current.dropna().astype(float)

    unique_vals = np.unique(reference)
    if len(unique_vals) <= bins:
        # low-cardinality / binary feature (e.g. one-hot flags): bin on exact values
        edges = np.concatenate([[-np.inf], unique_vals[:-1] + 1e-9, [np.inf]])
    else:
        quantiles = np.linspace(0, 1, bins + 1)
        edges = np.unique(np.quantile(reference, quantiles))
        edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = np.where(ref_counts == 0, 1e-4, ref_counts / ref_counts.sum())
    cur_pct = np.where(cur_counts == 0, 1e-4, cur_counts / cur_counts.sum())

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def compute_feature_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame, feature_columns: list[str]) -> dict:
    return {
        col: round(population_stability_index(reference_df[col], current_df[col]), 4)
        for col in feature_columns
        if col in reference_df.columns and col in current_df.columns
    }


def compute_performance_drift(predictions_df: pd.DataFrame, baseline_metrics: dict) -> dict:
    y_true = predictions_df["actual_duration_min"]
    y_pred = predictions_df["predicted_duration_min"]
    current = {
        "mae_minutes": float(mean_absolute_error(y_true, y_pred)),
        "rmse_minutes": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }
    return {
        "baseline": baseline_metrics,
        "current_on_drifted_batch": current,
        "rmse_relative_change": round(current["rmse_minutes"] / baseline_metrics["rmse_minutes"], 4),
        "mae_relative_change": round(current["mae_minutes"] / baseline_metrics["mae_minutes"], 4),
    }


def decide_retrain(feature_drift: dict, performance_drift: dict) -> dict:
    major_shift_features = [f for f, psi in feature_drift.items() if psi >= PSI_MAJOR_SHIFT]
    moderate_shift_features = [f for f, psi in feature_drift.items() if PSI_MODERATE_SHIFT <= psi < PSI_MAJOR_SHIFT]

    rmse_trigger = performance_drift["rmse_relative_change"] >= RMSE_DEGRADATION_TRIGGER
    psi_trigger = len(major_shift_features) >= N_MAJOR_SHIFT_FEATURES_TO_TRIGGER

    triggered = rmse_trigger or psi_trigger
    reasons = []
    if rmse_trigger:
        reasons.append(
            f"RMSE degraded {performance_drift['rmse_relative_change']:.2f}x baseline "
            f"(threshold {RMSE_DEGRADATION_TRIGGER}x)"
        )
    if psi_trigger:
        reasons.append(
            f"{len(major_shift_features)} features with major distribution shift (PSI>={PSI_MAJOR_SHIFT}): "
            f"{major_shift_features} (threshold: {N_MAJOR_SHIFT_FEATURES_TO_TRIGGER})"
        )
    if not triggered:
        reasons.append("Neither trigger condition met -- no retrain recommended at this time.")

    return {
        "retrain_triggered": triggered,
        "reasons": reasons,
        "major_shift_features": major_shift_features,
        "moderate_shift_features": moderate_shift_features,
        "policy": {
            "rmse_degradation_trigger": RMSE_DEGRADATION_TRIGGER,
            "psi_major_shift_threshold": PSI_MAJOR_SHIFT,
            "n_major_shift_features_to_trigger": N_MAJOR_SHIFT_FEATURES_TO_TRIGGER,
        },
    }


def plot_monitoring_report(feature_drift: dict, performance_drift: dict, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    features = list(feature_drift.keys())
    psis = list(feature_drift.values())
    colors = ["#c0392b" if p >= PSI_MAJOR_SHIFT else "#e67e22" if p >= PSI_MODERATE_SHIFT else "#27ae60" for p in psis]
    axes[0].barh(features, psis, color=colors)
    axes[0].axvline(PSI_MODERATE_SHIFT, color="gray", linestyle="--", linewidth=1, label="moderate (0.10)")
    axes[0].axvline(PSI_MAJOR_SHIFT, color="black", linestyle="--", linewidth=1, label="major (0.25)")
    axes[0].set_xlabel("PSI")
    axes[0].set_title("Feature drift (PSI vs. training reference)")
    axes[0].legend(fontsize=8)

    metrics = ["mae_minutes", "rmse_minutes"]
    baseline_vals = [performance_drift["baseline"][m] for m in metrics]
    current_vals = [performance_drift["current_on_drifted_batch"][m] for m in metrics]
    x = np.arange(len(metrics))
    width = 0.35
    axes[1].bar(x - width / 2, baseline_vals, width, label="baseline (test set)", color="#2980b9")
    axes[1].bar(x + width / 2, current_vals, width, label="current (drifted batch)", color="#c0392b")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["MAE (min)", "RMSE (min)"])
    axes[1].set_title("Performance drift")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def run_monitoring():
    reference_df = pd.read_csv(REFERENCE_FEATURES_PATH)
    model_comparison = json.loads(MODEL_COMPARISON_PATH.read_text())
    feature_columns = model_comparison["feature_columns"]
    best_model_name = model_comparison["best_model"]
    baseline_metrics = model_comparison["best_model_metrics"]

    predictions_df = pd.DataFrame(
        [json.loads(line) for line in PREDICTIONS_PATH.read_text().splitlines() if line.strip()]
    )

    feature_drift = compute_feature_drift(reference_df, predictions_df, feature_columns)
    performance_drift = compute_performance_drift(predictions_df, baseline_metrics)
    decision = decide_retrain(feature_drift, performance_drift)

    report = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "deployed_model": best_model_name,
        "n_scored_rows": len(predictions_df),
        "feature_drift_psi": feature_drift,
        "performance_drift": performance_drift,
        "retrain_decision": decision,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    DECISION_PATH.write_text(json.dumps(decision, indent=2))
    plot_monitoring_report(feature_drift, performance_drift, PLOT_PATH)

    return report


if __name__ == "__main__":
    report = run_monitoring()
    print(json.dumps(report, indent=2))
    print(f"\nMonitoring report -> {REPORT_PATH}")
    print(f"Retrain decision  -> {DECISION_PATH}")
    print(f"Plot              -> {PLOT_PATH}")
