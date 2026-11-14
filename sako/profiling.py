"""Transfer checked artifacts from model diagnostics."""

import hashlib
from pathlib import Path
import re


def workspace_path(value, *, output=False):
    root = Path(__file__).resolve().parents[1]
    path = Path(value)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    path.relative_to((root / "runs").resolve() if output else root)
    return path


def _artifact(path):
    root = Path(__file__).resolve().parents[1]
    data = path.read_bytes()
    return {"path": path.relative_to(root).as_posix(), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def collect_profile_artifacts(worker, result, run_dir):
    """Fetch saved evidence and verify the worker's recorded hashes."""
    root = Path(worker.root).resolve()
    run_dir = Path(run_dir).resolve()
    run_dir.relative_to(root)
    saved = []
    for artifact in result.get("artifacts", []):
        relative = artifact["path"]
        destination = (root / relative).resolve()
        destination.relative_to(run_dir)
        if not re.fullmatch(r"[a-f0-9]{64}", artifact.get("sha256", "")):
            raise ValueError("Invalid profiler artifact hash.")
        if not destination.is_file() or hashlib.sha256(destination.read_bytes()).hexdigest() != artifact["sha256"]:
            worker.get(relative, destination)
        content = destination.read_bytes()
        if len(content) != artifact["bytes"] or hashlib.sha256(content).hexdigest() != artifact["sha256"]:
            raise ValueError("The downloaded profiling evidence changed.")
        saved.append(artifact)
    return saved
