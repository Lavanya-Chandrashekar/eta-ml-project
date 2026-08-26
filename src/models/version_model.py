"""
version_model.py

Mirrors src/data/version_dataset.py, but for the trained model artifact:
records a content hash + metadata for the promoted best_model.joblib in a
committed manifest (models/MODEL_VERSIONS.json), even though the .joblib
binary itself is gitignored (see .gitignore for the reasoning). This keeps
the same "reproducible + auditable lineage" story going from data -> model.

Run automatically at the end of train.py; can also be run standalone:
    python version_model.py --tag v1-best-model
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from version_dataset import file_sha256, current_git_commit  # noqa: E402

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MANIFEST_PATH = MODELS_DIR / "MODEL_VERSIONS.json"


def version_model(model_path: Path, tag: str, metrics: dict, params: dict) -> dict:
    model_path = model_path.resolve()
    digest = file_sha256(model_path)

    entry = {
        "tag": tag,
        "file": model_path.name,
        "sha256": digest,
        "size_bytes": model_path.stat().st_size,
        "metrics": metrics,
        "params": params,
        "git_commit": current_git_commit(),
    }

    manifest = {"versions": []}
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text())

    manifest["versions"] = [v for v in manifest["versions"] if v["tag"] != tag]
    manifest["versions"].append(entry)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    return entry


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    report = json.loads((MODELS_DIR.parent / "reports" / "model_comparison.json").read_text())
    entry = version_model(
        MODELS_DIR / "best_model.joblib",
        args.tag,
        report["best_model_metrics"],
        {"run_name": report["best_model"]},
    )
    print(json.dumps(entry, indent=2))
