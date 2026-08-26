"""
plot_comparison.py

Renders the model comparison report (reports/model_comparison.json) as a
static PNG for the Week 2 deliverable ("short model comparison report").

Design choices (per project data-viz conventions):
  - Small multiples (MAE panel, RMSE panel) rather than a dual-axis chart --
    the two metrics are on different scales, so one shared y-axis would be
    misleading.
  - Fixed categorical color per model, consistent across both panels, drawn
    from the validated 4-slot palette (blue/orange/aqua/teal-green/yellow).
  - Direct value labels on every bar -- required here because two of the
    four colors fall under the 3:1 contrast floor against the light surface
    (validated via scripts/validate_palette.js), so labels carry meaning
    color alone can't guarantee.
  - No legend: x-axis tick labels already name each model directly.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[2]
REPORT_PATH = BASE / "reports" / "model_comparison.json"
OUT_PATH = BASE / "reports" / "model_comparison.png"

# validated categorical palette, fixed order (references/palette.md, slots 1-4)
MODEL_COLORS = {
    "linear_regression": "#2a78d6",             # blue
    "gradient_boosting_default": "#eb6834",      # orange
    "gradient_boosting_tuned_shallow": "#1baf7a",  # aqua
    "gradient_boosting_tuned_deep": "#eda100",   # yellow
}
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
BEST_MARKER_COLOR = "#0ca30c"  # status: good


def main():
    report = json.loads(REPORT_PATH.read_text())
    results = report["results"]
    best_name = report["best_model"]

    names = [r["run_name"] for r in results]
    labels = [n.replace("_", "\n") for n in names]
    mae = [r["metrics"]["mae_minutes"] for r in results]
    rmse = [r["metrics"]["rmse_minutes"] for r in results]
    colors = [MODEL_COLORS[n] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor=SURFACE)
    fig.suptitle(
        "Model comparison — ETA prediction (Week 2)",
        color=INK_PRIMARY, fontsize=13, fontweight="bold", x=0.02, ha="left",
    )

    for ax, values, title, unit in [
        (axes[0], mae, "Mean Absolute Error", "min"),
        (axes[1], rmse, "Root Mean Squared Error", "min"),
    ]:
        ax.set_facecolor(SURFACE)
        bars = ax.bar(labels, values, color=colors, width=0.6, zorder=3)

        for bar, val, name in zip(bars, values, names):
            ax.text(
                bar.get_x() + bar.get_width() / 2, val + max(values) * 0.02,
                f"{val:.2f}", ha="center", va="bottom",
                color=INK_PRIMARY, fontsize=9, fontweight="bold",
            )
            if name == best_name:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, -max(values) * 0.06,
                    "★ best", ha="center", va="top",
                    color=BEST_MARKER_COLOR, fontsize=8, fontweight="bold",
                )

        ax.set_title(f"{title} ({unit})", color=INK_SECONDARY, fontsize=10, loc="left")
        ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color(BASELINE)
        ax.tick_params(axis="x", colors=INK_SECONDARY, labelsize=8)
        ax.tick_params(axis="y", colors=INK_MUTED, labelsize=8)
        ax.set_ylim(0, max(values) * 1.25)

    n_test = results[0]["params"].get("n_test", "?")
    fig.text(
        0.02, 0.01,
        f"Selection metric: {report['selection_metric']}  |  seed={report['random_seed']}  |  test set n={n_test}",
        color=INK_MUTED, fontsize=7,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(OUT_PATH, dpi=150, facecolor=SURFACE)
    print(f"Chart written to {OUT_PATH}")


if __name__ == "__main__":
    main()
