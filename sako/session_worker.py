"""Serve checked jobs on one SSH stream. This is not a security sandbox."""

import argparse
import contextlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import threading
import time
import traceback
import uuid

from .core import DEFAULT_RESIDENT_MAX_JOBS, digest, implementation_hash, read_json, write_json
from .model_session import ModelSession
from .session_protocol import MAX_FRAME, MAX_FILE, VERSION, pack, unpack, publish, workspace_file
from .worker import gpu_lock, run_payload


class SessionWorker:
    def __init__(self, root, max_jobs=DEFAULT_RESIDENT_MAX_JOBS, runner=None):
        self.root = Path(root).resolve()
        self.source = implementation_hash()
        self.session_id = uuid.uuid4().hex
        self.max_jobs, self.jobs = max_jobs, 0
        self.resources = ModelSession()
        self.runner = runner or run_payload
        self.context = contextlib.ExitStack()
        self.locked = False
        self.stop = False
        self.ledger = self.root / "runs/resident-jobs.sqlite"
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        with self.db() as db:
            db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, hash TEXT, result TEXT, state TEXT, session TEXT)")

    @contextlib.contextmanager
    def db(self):
        connection = sqlite3.connect(self.ledger, timeout=30)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def hello(self):
        return {"protocol": VERSION, "implementation_hash": self.source,
                "pid": os.getpid(), "session_id": self.session_id}

    def checked_result(self, path, job, compact=False):
        payload = read_json(path)
        if payload.get("job_hash") != digest(job) or payload.get("provenance", {}).get("implementation_hash") != self.source:
            raise ValueError("The saved job result has different source or inputs.")
        compact = compact and job.get("action") == "evaluate" and payload.get("ok")
        log = path.with_name(path.name.replace(".result.json", ".worker.log"))
        if compact:
            # The controller still verifies the actual speed and accuracy samples. Detailed diagnostics stay at the worker.
            value = payload["result"]
            for case in value.get("cases", []):
                for field in ("baseline_wall", "candidate_wall", "setup_times"):
                    case.pop(field, None)
                rank = case.get("search_reference", {})
                if all(side in case and rank.get(side) == case[side] for side in ("baseline", "candidate")):
                    for side in ("baseline","candidate"):
                        del rank[side]
                    rank["same_timing_pair"] = True
                for tuner in case.get("autotuning", []):
                    for trial in tuner.get("harness_trials", []):
                        trial.pop("samples_ms", None)
                        trial.pop("setup", None)
            for tuner in value.get("automatic_tuning", {}).get("kernels", []):
                for trial in tuner.get("harness_trials", []):
                    trial.pop("samples_ms", None)
                    trial.pop("setup", None)
            payload["remote_evidence"] = {"result": path.relative_to(self.root).as_posix(),
                "log": log.relative_to(self.root).as_posix() if log.exists() else None}
        body = json.dumps(payload, separators=(",",":"), allow_nan=False).encode() if compact else path.read_bytes()
        response = {"status": "complete", "result": pack(body),
                "session": {**self.hello(), "jobs": self.jobs,
                            "retire": self.resources.retire or self.jobs >= self.max_jobs}}
        if compact:
            return response
        if log.is_file() and log.stat().st_size <= MAX_FILE:
            response["worker_log"] = pack(log.read_bytes())
        elif log.is_file():
            response["worker_log_path"] = log.relative_to(self.root).as_posix()
        return response

    def handle(self, request):
        if request.get("implementation_hash") != self.source or implementation_hash() != self.source:
            self.stop = True
            raise ValueError("Session source changed. Use the matching saved source or a new run.")
        kind = request.get("kind")
        if kind == "close":
            self.stop = True
            return {"status": "closed"}
        if kind == "get":
            path = workspace_file(self.root, request["path"])
            if path.stat().st_size > MAX_FILE:
                return {"status": "large_file", "bytes": path.stat().st_size}
            return {"status": "file", "file": pack(path.read_bytes())}
        if kind not in {"run", "recover"}:
            raise ValueError("Unknown session request.")
        job_id, job = request.get("job_id"), request.get("job")
        if not isinstance(job_id, str) or len(job_id) != 32 or any(c not in "0123456789abcdef" for c in job_id):
            raise ValueError("Invalid session job ID.")
        if not isinstance(job, dict) or job.get("implementation_hash") != self.source:
            raise ValueError("The job source differs from this session.")
        result_path = workspace_file(self.root, request["result_path"])
        job_path = workspace_file(self.root, request["job_path"])
        if job_path.name != job_id + ".json" or result_path != job_path.with_name(job_id + ".result.json"):
            raise ValueError("The job paths do not match its ID.")
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT hash,result,state FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row and (row[0] != digest(job) or row[1] != str(result_path)):
                raise ValueError("This job ID belongs to different inputs.")
            if row:
                if result_path.is_file():
                    return self.checked_result(result_path, job, request.get("compact_result", False))
                return {"status": "pending", "state": row[2], "job_id": job_id}
            if kind == "recover":
                return {"status": "not_started", "job_id": job_id}
            db.execute("INSERT INTO jobs VALUES (?,?,?,?,?)",
                       (job_id, digest(job), str(result_path), "running", self.session_id))
        timeout = request.get("timeout_seconds")
        if type(timeout) not in (int, float) or not 0 < timeout <= 86400:
            raise ValueError("Invalid job timeout.")
        def expire():
            try:
                write_json(result_path, {"job_hash": digest(job), "ok": False,
                    "failure_kind": "worker", "error": "Worker exceeded the fixed job time limit.",
                    "provenance": {"implementation_hash": self.source}})
                with self.db() as db:
                    db.execute("UPDATE jobs SET state='timeout' WHERE id=?", (job_id,))
            finally:
                os._exit(124)
        watchdog = threading.Timer(timeout, expire)
        watchdog.daemon = True
        watchdog.start()
        started = time.perf_counter()
        try:
            files = request.get("files", [])
            if not isinstance(files, list) or len(files) > 128:
                raise ValueError("Invalid session file bundle.")
            decoded = [(workspace_file(self.root, item["path"]), unpack(item)) for item in files]
            if len({path for path, _ in decoded}) != len(decoded):
                raise ValueError("Duplicate session file path.")
            for path, data in decoded:
                publish(path, data)
            if read_json(job_path) != job:
                raise ValueError("The published job changed.")
            wait_start = time.perf_counter()
            if not self.locked:
                self.context.enter_context(gpu_lock())
                self.locked = True
            lock_wait = time.perf_counter() - wait_start
            from .gpu import build_log
            self.resources.log_path = job_path.with_suffix(".worker.log")
            with build_log(self.resources.log_path):
                payload = self.runner(job, session=self.resources)
            if implementation_hash() != self.source:
                payload = {"ok": False, "job_hash": digest(job), "unverified_result": payload,
                           "failure_kind": "infrastructure", "error": "Source changed while the job ran.",
                           "provenance": {"implementation_hash": self.source}}
                self.resources.retire = True
            payload["session_times"] = {"gpu_lock_wait_s": lock_wait, "request_total_s": time.perf_counter() - started}
        except Exception as error:
            self.resources.retire = True
            payload = {"job_hash": digest(job), "ok": False, "failure_kind": "infrastructure",
                       "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc(),
                       "provenance": {"implementation_hash": self.source}}
        finally:
            watchdog.cancel()
        write_json(result_path, payload)
        with self.db() as db:
            db.execute("UPDATE jobs SET state='complete' WHERE id=?", (job_id,))
        self.jobs += 1
        response = self.checked_result(result_path, job, request.get("compact_result", False))
        self.stop = response["session"]["retire"]
        return response

    def close(self):
        self.resources.close_fixtures()
        self.context.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-jobs", type=int, default=DEFAULT_RESIDENT_MAX_JOBS)
    parser.add_argument("--idle-seconds", type=float, default=600)
    args = parser.parse_args()
    if not 1 <= args.max_jobs <= 128:
        parser.error("Use one to 128 jobs per process.")
    if not 30 <= args.idle_seconds <= 3600:
        parser.error("Use an idle limit from 30 to 3600 seconds.")
    server = SessionWorker(Path.cwd(), args.max_jobs)
    # Keep a protocol descriptor separate from candidate/compiler stdout.
    with os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1) as output:
        output.write(json.dumps(server.hello()) + "\n")
        try:
            while not server.stop:
                idle = threading.Timer(args.idle_seconds, lambda: os._exit(0))
                idle.daemon = True
                idle.start()
                try:
                    line = sys.stdin.buffer.readline(MAX_FRAME + 1)
                finally:
                    idle.cancel()
                if not line:
                    break
                if len(line) > MAX_FRAME:
                    raise ValueError("Session request exceeds the frame limit.")
                try:
                    response = server.handle(json.loads(line))
                except Exception as error:
                    response = {"status": "error", "error": f"{type(error).__name__}: {error}",
                                "traceback": traceback.format_exc()}
                output.write(json.dumps(response, allow_nan=False) + "\n")
        finally:
            server.close()


if __name__ == "__main__":
    main()
