"""
tracker.py

A minimal experiment tracker with an MLflow-compatible interface.

Why this exists: the course brief calls out MLflow for experiment tracking,
and this code is written to *use real MLflow whenever it's installed*. But
this pipeline also needs to run in network-restricted environments (like the
sandbox this was originally built in) where `pip install mlflow` isn't
possible. So at import time we detect whether MLflow is available:

  - If MLflow IS installed: every call below transparently delegates to the
    real `mlflow` module. Run `mlflow ui` afterwards to see the standard
    MLflow dashboard (params, metrics, run comparison) -- nothing here
    changes that experience.
  - If MLflow is NOT installed: we fall back to a small local tracker that
    writes the same information (params, metrics, artifacts, timestamps) to
    JSON files under `mlruns_local/<experiment>/<run_id>/`, and prints a
    console summary. This keeps params/metrics/reproducibility fully logged
    even without the dependency.

Either way, `train.py` calls the exact same four methods
(start_run / log_params / log_metrics / log_artifact), so switching between
the two backends requires zero changes to training code.
"""

import json
import shutil
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

try:
    import mlflow as _mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    _mlflow = None
    MLFLOW_AVAILABLE = False

LOCAL_TRACKING_DIR = Path(__file__).resolve().parents[2] / "mlruns_local"


class _LocalRun:
    """MLflow-compatible run handle backed by local JSON files."""

    def __init__(self, experiment_name: str, run_name: str):
        self.run_id = uuid.uuid4().hex[:12]
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.dir = LOCAL_TRACKING_DIR / experiment_name / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.params = {}
        self.metrics = {}
        self.started_at = datetime.now(timezone.utc).isoformat()

    def log_params(self, params: dict):
        self.params.update({k: str(v) for k, v in params.items()})
        (self.dir / "params.json").write_text(json.dumps(self.params, indent=2))

    def log_metrics(self, metrics: dict):
        self.metrics.update({k: float(v) for k, v in metrics.items()})
        (self.dir / "metrics.json").write_text(json.dumps(self.metrics, indent=2))

    def log_artifact(self, path):
        path = Path(path)
        artifacts_dir = self.dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        shutil.copy2(path, artifacts_dir / path.name)

    def end(self):
        meta = {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "experiment_name": self.experiment_name,
            "started_at": self.started_at,
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "params": self.params,
            "metrics": self.metrics,
        }
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"[tracker:local] run '{self.run_name}' ({self.run_id}) logged to {self.dir}")


class ExperimentTracker:
    """
    Usage:
        tracker = ExperimentTracker("eta-prediction")
        with tracker.start_run("linear_regression") as run:
            run.log_params({"model": "LinearRegression"})
            run.log_metrics({"rmse": 4.2, "mae": 3.1})
            run.log_artifact(Path("models/linreg.joblib"))
    """

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        if MLFLOW_AVAILABLE:
            _mlflow.set_experiment(experiment_name)

    @contextmanager
    def start_run(self, run_name: str):
        if MLFLOW_AVAILABLE:
            with _mlflow.start_run(run_name=run_name) as mlflow_run:
                yield _MLflowRunAdapter(mlflow_run)
        else:
            run = _LocalRun(self.experiment_name, run_name)
            try:
                yield run
            finally:
                run.end()

    def list_runs(self) -> list[dict]:
        """Return a list of {run_name, params, metrics} dicts across all backends we can read."""
        if MLFLOW_AVAILABLE:
            runs = _mlflow.search_runs(experiment_names=[self.experiment_name])
            return runs.to_dict("records")
        exp_dir = LOCAL_TRACKING_DIR / self.experiment_name
        if not exp_dir.exists():
            return []
        out = []
        for run_dir in sorted(exp_dir.iterdir()):
            meta_path = run_dir / "meta.json"
            if meta_path.exists():
                out.append(json.loads(meta_path.read_text()))
        return out


class _MLflowRunAdapter:
    """Adapts real mlflow's run object to the same log_params/log_metrics/log_artifact surface."""

    def __init__(self, mlflow_run):
        self._run = mlflow_run

    def log_params(self, params: dict):
        _mlflow.log_params(params)

    def log_metrics(self, metrics: dict):
        _mlflow.log_metrics(metrics)

    def log_artifact(self, path):
        _mlflow.log_artifact(str(path))
