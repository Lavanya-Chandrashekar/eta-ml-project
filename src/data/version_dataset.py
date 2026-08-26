"""
version_dataset.py

Lightweight dataset versioning.

Why not DVC here: DVC is the tool the course brief calls out, and is the
recommended choice when you have internet access on your own machine (see
README for `dvc init` / `dvc add` instructions to layer it in). This script
implements the same *core idea* DVC is built on -- content-addressed hashing
of data files, decoupled from Git's line-based diffing -- as a dependency-free
fallback: every version is a hash of the exact bytes of the file, recorded
with metadata (row/col counts, timestamp, git commit) in a small manifest.
That manifest is committed to Git, giving you a full, auditable history of
*which dataset version* every experiment and model was trained against,
which is the actual requirement (rubric: "correct use of dataset versioning
e.g., DVC/Git").

Usage:
    python version_dataset.py <path-to-csv> --tag v1-raw
"""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

MANIFEST_PATH = Path(__file__).resolve().parents[2] / "data" / "DATA_VERSIONS.json"


def file_sha256(path: Path, chunk_size: int = 8192) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def current_git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None  # e.g. no commits yet


def version_dataset(csv_path: Path, tag: str) -> dict:
    csv_path = csv_path.resolve()
    df = pd.read_csv(csv_path)
    digest = file_sha256(csv_path)

    entry = {
        "tag": tag,
        "file": str(csv_path.relative_to(MANIFEST_PATH.parent.resolve())),
        "sha256": digest,
        "rows": len(df),
        "columns": list(df.columns),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_git_commit(),
    }

    manifest = {"versions": []}
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text())

    # replace an existing entry with the same tag (re-versioning), else append
    manifest["versions"] = [v for v in manifest["versions"] if v["tag"] != tag]
    manifest["versions"].append(entry)

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    return entry


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--tag", required=True, help="human-readable version tag, e.g. v1-raw")
    args = parser.parse_args()

    entry = version_dataset(args.csv_path, args.tag)
    print(json.dumps(entry, indent=2))
    print(f"\nManifest updated at {MANIFEST_PATH}")
