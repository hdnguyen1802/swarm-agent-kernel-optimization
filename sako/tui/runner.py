"""Launch the existing harness commands. A run can outlive its terminal."""

from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys

from filelock import FileLock, Timeout

from ..core import digest, implementation_hash, load_config, read_json, write_json
from ..transport import Worker
from .colab import ColabSession, GPUS, cli_command
from .runs import read_if_present, run_path, saved_config

ROOT = Path(__file__).resolve().parents[2]
MODES = {
    "search": "Search and test the model",
    "baseline": "Profile and measure the baseline",
    "doctor": "Verify the GPU worker",
    "resume": "Resume the saved search",
    "integrate": "Test the kept kernels in the model",
}
STAGES = {
    "sync": "Sync source", "prepare": "Prepare profile", "profile": "Profile model",
    "baseline": "Measure baseline", "search": "Search kernels",
    "integrate": "Test complete model", "report": "Write report", "doctor": "Check GPU and software",
}
NEW_MODES = ("search", "baseline", "doctor")


def probe_gpu(config, root):
    worker = Worker(config, root)
    try:
        query = ["nvidia-smi", "-i", str(int(config["worker"]["gpu"])),
                 "--query-gpu=name,memory.total", "--format=csv,noheader"]
        args = query if worker.local else [*worker.ssh(), shlex.join(query)]
        result = subprocess.run(args, cwd=root, stdin=subprocess.DEVNULL, capture_output=True,
                                encoding="utf-8", errors="replace", timeout=28)
        if result.returncode or not result.stdout.strip():
            raise ValueError((result.stderr.strip() or result.stdout.strip() or "GPU check failed.")[-650:])
        return result.stdout.strip()
    finally:
        worker.close()


def commands(mode, run):
    if mode == "doctor":
        return ["sync", "doctor"]
    if mode == "baseline":
        return ["sync", "profile", "baseline"]
    if mode == "integrate":
        return ["profile", "integrate", "report"]
    stages = ["sync", "prepare"]
    if mode == "search" or not (run / "baseline.json").is_file():
        stages.append("baseline")
    return [*stages, "search", "integrate", "report"]


def can_integrate(result):
    kept = result.get("kernel_manifest") or result.get("manifest") or {}
    if result.get("performance_scope") == "cached_decode":
        return (result.get("status") in {"completed_kernel_search", "budget_exhausted"}
                and result.get("integration_status") in {"pending", "no_finalists"})
    return (result.get("status") in {"completed_kernel_search", "accepted", "budget_exhausted"}
            and bool(result.get("integration_status")) and bool(kept.get("entries")))


def validate_saved(run, profile, mode):
    config_path = saved_config(run)
    if config_path is None:
        raise ValueError("This run has no saved config. You can still view its results.")
    config = load_config(config_path)
    catalog = None
    if (profile / "tasks.json").exists():
        stamp = read_if_present(profile / "preparation.json")
        if stamp.get("implementation_hash") != implementation_hash():
            raise ValueError("The saved profile uses different source code. Start a new run or view this run.")
        catalog = read_json(profile / "tasks.json")
        if stamp.get("catalog_hash") != digest(catalog):
            raise ValueError("The saved catalog has changed. Start a new run or view this run.")
    db_path = run / "state.sqlite"
    if db_path.is_file():
        with closing(sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True, timeout=0.25)) as db:
            row = db.execute("SELECT value FROM state WHERE key='fingerprint'").fetchone()
        if row and json.loads(row[0]) != digest({"config": config, "catalog": catalog,
                                                  "implementation": implementation_hash()}):
            raise ValueError("The saved config or profile differs from this search. Use the original run settings.")
    if mode == "integrate" and not (config.get("kernel_search", {}).get("enabled")
                                    and (profile / "tasks.json").is_file() and (run / "result.json").is_file()):
        raise ValueError("This action requires a saved kernel search and its profile.")
    return config_path


def controller_busy(run):
    lock = run / "ui" / "controller.lock"
    if not lock.exists():
        return False
    try:
        with FileLock(str(lock), timeout=0):
            return False
    except Timeout:
        return True


def launch(root, run, profile, mode, config, note="", colab=None, *, keep_session=False):
    run = run_path(root, run)
    for path in (run, profile):
        relative = path.resolve().relative_to(root).as_posix()
        if (colab or config["worker"].get("transport") != "local") and not re.fullmatch(r"[A-Za-z0-9_./-]+", relative):
            raise ValueError("The worker requires folder names with letters A-Z, digits, dashes, dots, or underscores.")
    if mode not in MODES:
        raise ValueError("Unknown run mode.")
    if colab:
        if colab not in GPUS or mode not in NEW_MODES:
            raise ValueError("Use a new run for a new Colab session.")
        if profile.is_dir() and any(profile.iterdir()):
            raise ValueError("Use a fresh profile for the new Colab GPU.")
        cli_command(root)
        for tool in ("ssh", "scp", "ssh-keygen"):
            if not shutil.which(tool):
                raise ValueError(f"Install OpenSSH. The controller needs {tool}.")
    if mode in NEW_MODES:
        run.mkdir(parents=True, exist_ok=False)
        config_path = run / "launch-config.json"
        if not colab:
            write_json(config_path, config)
    else:
        if controller_busy(run):
            raise ValueError("A controller is already working on this run.")
        config_path = validate_saved(run, profile, mode)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    attempt = run / "ui" / ("attempt-" + stamp)
    attempt.mkdir(parents=True)
    if colab:
        write_json(attempt / "requested-config.json", config)
    write_json(attempt / "plan.json", {"mode": mode, "config": config_path.relative_to(root).as_posix(),
                                       "profile": profile.relative_to(root).as_posix(), "note": note, "colab": colab,
                                       "keep_session": keep_session})
    write_json(attempt / "progress.json", {"status": "Starting", "stage": MODES[mode]})
    options = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
               if os.name == "nt" else {"start_new_session": True})
    try:
        with (attempt / "controller.log").open("ab") as log:
            process = subprocess.Popen([sys.executable, "-u", "-m", "sako.tui.runner", str(attempt)],
                                       cwd=root, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       env={**os.environ, "PYTHONIOENCODING": "utf-8"}, **options)
    except OSError as error:
        write_json(attempt / "progress.json", {"status": "Failed", "detail": str(error)})
        raise
    return process, attempt


def execute(attempt, root=ROOT, invoke=subprocess.run, session_factory=ColabSession):
    plan = read_json(attempt / "plan.json")
    run = attempt.parent.parent
    mode = plan["mode"]
    stop = run / "STOP"
    progress = attempt / "progress.json"
    current = None

    def stage(label):
        nonlocal current
        current = label
        write_json(progress, {"status": "Running", "stage": current})
        print(f"\n[{current}]", flush=True)

    try:
        with FileLock(str(run / "ui" / "controller.lock"), timeout=0):
            write_json(run / "ui" / "current.json", {"attempt": attempt.name})
            if mode == "resume":
                stop.unlink(missing_ok=True)
                if (attempt / "stop-requested").exists():
                    stop.touch()
            common = ["--config", plan["config"], "--run", run.relative_to(root).as_posix(),
                      "--profile", plan["profile"]]
            result = {}
            errors = []
            failed_stage = None
            session = None
            if plan.get("colab"):
                session = session_factory(root, attempt, plan["colab"])
            try:
                if session and not stop.exists():
                    stage("Request Colab " + GPUS[plan["colab"]][0])
                    session.allocate()
                    if not stop.exists():
                        stage("Verify assigned GPU")
                        config = session.connect(read_json(attempt / "requested-config.json"))
                        write_json(root / plan["config"], config)
                for step in commands(mode, run):
                    if stop.exists():
                        break
                    if step == "integrate" and mode != "integrate" and not can_integrate(result):
                        continue
                    stage(STAGES[step])
                    args = [sys.executable, "-u", "-m", "sako", step, *common]
                    if step == "sync":
                        # The existing sync command builds one bundle in .local/.
                        with FileLock(str(root / ".local" / "ui-sync.lock")):
                            completed = invoke(args, cwd=root)
                    else:
                        completed = invoke(args, cwd=root)
                    if completed.returncode:
                        raise RuntimeError(f"{current} returned exit code {completed.returncode}. See the log.")
                    if session and step == "sync" and not stop.exists():
                        stage("Install project on Colab")
                        session.install()
                    if step == "search":
                        result = read_json(run / "result.json")
                        if result.get("status") in {"infrastructure_failure", "request_failure"}:
                            raise RuntimeError(f"Search stopped with {result['status']}. See the log.")
            except Exception as error:
                failed_stage = current
                errors.append(f"{current}. {error}")
            finally:
                if session:
                    try:
                        if session.config:
                            stage("Save GPU evidence")
                            session.collect()
                    except Exception as error:
                        failed_stage = failed_stage or current
                        errors.append(f"Evidence download failed. {error}")
                    finally:
                        if plan.get("keep_session"):
                            stage("Keep Colab session active")
                        else:
                            try:
                                try:
                                    stage("Release Colab session")
                                finally:
                                    session.release()
                            except Exception as error:
                                failed_stage = failed_stage or current
                                errors.append(str(error))
            if errors:
                current = failed_stage
                raise RuntimeError("\n".join(errors))
            write_json(progress, {"status": "Stopped" if stop.exists() else "Finished", "stage": current,
                                  "outcome": result.get("status")})
    except Timeout:
        write_json(progress, {"status": "Busy", "detail": "Another controller owns this run."})
        return 1
    except Exception as error:
        write_json(progress, {"status": "Failed", "stage": current, "detail": str(error)})
        print(str(error), file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt", type=Path)
    raise SystemExit(execute(parser.parse_args().attempt.resolve()))
