"""Resume an existing search with paced SSH starts and bounded recovery.

The worker retains its job identity and uncertain-dispatch recovery rules.
This driver does not change Triton source, config, test limits, or token usage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import threading
import time

from sako.agent import create_agent
from sako.core import digest, implementation_hash, load_config, read_json
from sako.preparation import prepare
from sako.search import make_manifest
from sako.store import Store
from sako.swarm import SwarmLoop
from sako.transport import TransferError, Worker


class PacedWorker(Worker):
    """Space SSH/SCP starts across all instances in this process."""
    _pace_lock = threading.Lock()
    _last_start = None

    def __init__(self, config, root=None, *, minimum_seconds=3.):
        if not math.isfinite(minimum_seconds) or minimum_seconds < 3:
            raise ValueError("Space SSH/SCP starts by at least three seconds.")
        super().__init__(config, root=root)
        self.minimum_seconds = minimum_seconds

    def _logged_command(self, command, timeout, *, env=None, log_name=None, input_text=None):
        executable = Path(str(command[0])).stem.lower()
        if executable in {"ssh", "scp"}:
            self._pace_connection()
        options = {"env": env, "log_name": log_name}
        if input_text is not None:
            options["input_text"] = input_text
        return super()._logged_command(command, timeout, **options)

    def _pace_connection(self):
        with PacedWorker._pace_lock:
            now = time.monotonic()
            delay = (max(0., PacedWorker._last_start + self.minimum_seconds - now)
                     if PacedWorker._last_start is not None else 0.)
            while delay > 0:
                chunk = min(20., delay)
                time.sleep(chunk)
                delay -= chunk
            PacedWorker._last_start = time.monotonic()


def _append_record(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _pending_jobs(run):
    return [{"path": path.relative_to(run).as_posix(), **{
                key: value.get(key) for key in ("job_id", "job_hash", "status", "dispatch_attempts")}}
            for path in sorted((run / "jobs").glob("*.pending.json"))
            for value in [read_json(path)]]


def _cooldown(run, seconds):
    """Leave STOP responsive during recovery waits."""
    remaining = seconds
    while remaining > 0:
        if (run / "STOP").exists():
            return False
        step = min(1., remaining)
        time.sleep(step)
        remaining -= step
    return not (run / "STOP").exists()


def resume_paced(config_path, run, profile, *, minimum_seconds=3., attempts=3):
    if type(attempts) is not int or not 1 <= attempts <= 3:
        raise ValueError("Use one to three controller recovery attempts.")
    config_path, run, profile = Path(config_path).resolve(), Path(run).resolve(), Path(profile).resolve()
    if not (run / "state.sqlite").is_file():
        raise ValueError("This driver resumes an existing token ledger.")
    config = load_config(config_path)
    if config != load_config(run / "config.json"):
        raise ValueError("Use the saved run config without changing its limits.")
    store = Store(run / "state.sqlite")
    worker = PacedWorker(config, minimum_seconds=minimum_seconds)
    session = str(time.time_ns())
    directory = run / "transport_recovery"
    before = store.budget_snapshot(config["agent"])
    start = {"kind": "transport_pacing", "utc": time.time(), "minimum_seconds": minimum_seconds,
             "attempts": attempts, "cooldown_seconds": [20, 40], "config_hash": digest(config),
             "implementation_hash": implementation_hash(), "script_sha256": hashlib.sha256(
                 Path(__file__).read_bytes()).hexdigest(), "usage_before": before,
             "max_total_tokens": config["agent"]["max_total_tokens"],
             "pending_jobs_before": _pending_jobs(run)}
    _append_record(directory / f"{session}-pacing.json", start)
    last = {"status": "stopped", "usage": store.usage()}
    for attempt in range(1, attempts + 1):
        if (run / "STOP").exists():
            _append_record(directory / f"{session}-stopped-{attempt:02d}.json",
                           {"status": "stopped", "usage": store.usage(), "utc": time.time()})
            return {"status": "stopped", "usage": store.usage()}
        record = {"attempt": attempt, "utc": time.time(), "usage_before": store.usage(),
                  "pending_jobs_before": _pending_jobs(run)}
        try:
            prepare(config, run, profile, worker)
            if (run / "STOP").exists():
                last = {"status": "stopped", "usage": store.usage()}
            else:
                loop = SwarmLoop(config, run, profile, create_agent, lambda _: worker, make_manifest)
                last = loop.run()
        except TransferError as error:
            last = {"status": "infrastructure_failure", "usage": store.usage(),
                    "error": str(error), "job_id": getattr(error, "job_id", None)}
        record.update(status=last["status"], usage_after=store.usage(),
                      pending_jobs_after=_pending_jobs(run))
        if last.get("error"):
            record["error"] = last["error"]
        if last.get("job_id"):
            record["job_id"] = last["job_id"]
        _append_record(directory / f"{session}-attempt-{attempt:02d}.json", record)
        print(json.dumps({"attempt": attempt, "status": last["status"], "usage": store.usage()}), flush=True)
        if last["status"] != "infrastructure_failure" or attempt == attempts:
            return last
        if not _cooldown(run, 20 * (2 ** (attempt - 1))):
            return {"status": "stopped", "usage": store.usage()}
    return last


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--minimum-seconds", type=float, default=3.)
    parser.add_argument("--attempts", type=int, default=3)
    args = parser.parse_args()
    result = resume_paced(args.config, args.run, args.profile,
                          minimum_seconds=args.minimum_seconds, attempts=args.attempts)
    print(json.dumps({"status": result["status"], "usage": result.get("usage")}, indent=2))


if __name__ == "__main__":
    main()
