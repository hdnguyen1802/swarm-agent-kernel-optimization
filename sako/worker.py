"""One serialized worker per GPU. Agent code never sets benchmark results."""

import argparse
import contextlib
import os
import traceback
from pathlib import Path

from .core import read_json, write_json, load_config, candidate_id, score_cases, digest, implementation_hash


@contextlib.contextmanager
def gpu_lock():
    import fcntl
    path = Path.home() / "sako/locks" / ("gpu-" + os.environ.get("CUDA_VISIBLE_DEVICES", "0") + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def execute(job):
    from .gpu import doctor, evaluate_fixture
    from .model import profile, measure_model
    config = load_config(job["config"])
    action = job["action"]
    if action == "doctor":
        return doctor()
    if action == "profile":
        return profile(config, job["profile_dir"])
    catalog = read_json(Path(job["profile_dir"]) / "tasks.json")
    if catalog["config"]["model_id"] != config["model_id"] or catalog["config"]["dtype"] != config["dtype"]:
        raise ValueError("Profile and evaluation model differ.")
    if action == "evaluate":
        candidate = read_json(job["candidate"])
        task = next(t for t in catalog["tasks"] if t["id"] == candidate["task_id"])
        cases = evaluate_fixture(candidate, task, Path(job["profile_dir"]) / (task["id"] + ".pt"), config)
        result = {"scope": "operation", "candidate_id": candidate_id(candidate), "cases": cases,
                  "decision": score_cases(cases, config)}
        if len(cases) != len(task["cases"]):
            result["decision"]["accepted"] = False
            result["decision"]["reasons"].append("Not all fixture cases passed.")
        return result
    if action == "model":
        return measure_model(config, read_json(job["manifest"]), catalog, job["candidate_root"])
    raise ValueError(f"Unknown worker action: {action}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    job = read_json(args.job)
    result = {"job_hash": digest(job), "ok": False}
    try:
        with gpu_lock():
            result.update(result=execute(job), ok=True)
            import torch
            result["provenance"] = {"implementation_hash": implementation_hash(),
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(0), "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    except Exception as error:
        result.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc()[-20000:])
    write_json(args.output, result)
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
