"""Command line interface for the controller and GPU worker."""

import argparse
import json
from pathlib import Path

from .core import load_config, read_json, write_json, candidate_id
from .transport import Worker

ROOT = Path(__file__).resolve().parents[1]


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def main():
    parser = argparse.ArgumentParser(prog="sako")
    parser.add_argument("command", choices=["ui", "sync", "doctor", "prepare", "profile", "baseline", "model-check",
                                             "evaluate", "timing-audit", "search", "integrate", "status", "report", "api-check", "export", "fetch-evidence"])
    parser.add_argument("--config", default="configs/project.json")
    parser.add_argument("--run", help="Run folder. With ui, open a saved run or suggest a new folder.")
    parser.add_argument("--profile", help="Prepared profile directory; defaults to <run>/profile.")
    parser.add_argument("--candidate")
    parser.add_argument("--manifest")
    parser.add_argument("--job-id", help="Worker job ID for fetch-evidence.")
    args = parser.parse_args()
    if args.command == "ui":
        try:
            from .tui.app import HarnessApp
        except ModuleNotFoundError as error:
            if error.name not in {"textual", "rich", "filelock"}:
                raise
            parser.exit(2, 'Install the terminal interface with python -m pip install -e ".[ui]".\n')
        try:
            app = HarnessApp(ROOT, Path(args.config), args.run, args.profile)
        except (OSError, ValueError) as error:
            parser.error(str(error))
        app.run()
        return
    config = load_config(args.config, historical=args.command in ("status", "report", "export"))
    run = Path(args.run or "runs/main").resolve()
    profile = Path(args.profile).resolve() if args.profile else run / "profile"
    relative(run)
    relative(profile)
    run.mkdir(parents=True, exist_ok=True)
    worker = Worker(config)
    if args.command == "fetch-evidence":
        import re
        if not re.fullmatch("[0-9a-f]{32}", args.job_id or ""):
            parser.error("fetch-evidence requires a 32-character job ID")
        records = list(run.glob("**/jobs/" + args.job_id + ".result.json"))
        if len(records) != 1:
            parser.error("No unique result for this job in the run")
        payload = read_json(records[0])
        files = payload.get("remote_evidence") or {}
        if not files:
            print("Full result already available at " + str(records[0]))
            return
        try:
            folder = records[0].parent / "raw"
            worker.get(files["result"], folder / records[0].name, job_hash=payload["job_hash"])
            if files.get("log"):
                worker.get(files["log"], folder / Path(files["log"]).name)
        finally:
            worker.close()
        result = {"raw_evidence": str(folder / records[0].name)}
    elif args.command == "sync":
        result = worker.sync()
    elif args.command == "doctor":
        result = worker.run("doctor", run)
        write_json(run / "doctor.json", result)
    elif args.command in ("prepare", "profile"):
        from .preparation import prepare
        result = prepare(config, run, profile, worker)
        print(f"Saved {len(result['tasks'])} tasks; selected {len(result['selected_task_ids'])}. "
              f"Revision: {result['config']['revision']}. Profile: {json.dumps(relative(profile))}")
        return
    elif args.command == "baseline" and config.get("measurement", {}).get("mode") == "adaptive":
        result = worker.run("baseline", run, profile_dir=relative(profile))
        write_json(run / "baseline.json", result)
    elif args.command in ("baseline", "model-check"):
        if args.command == "model-check":
            if not args.manifest:
                parser.error("model-check requires --manifest")
            path = Path(args.manifest).resolve()
            for entry in read_json(path)["entries"]:
                source = run / "candidates" / (entry["candidate_id"] + ".json")
                worker.put(source, relative(source))
        else:
            path = run / "empty_manifest.json"
            write_json(path, {"entries": []})
        worker.put(path, relative(path))
        result = worker.run("model", run, profile_dir=relative(profile), manifest=relative(path),
                            candidate_root=relative(run / "candidates"))
        write_json(run / ("baseline.json" if args.command == "baseline" else "model-check.json"), result)
    elif args.command in ("evaluate", "timing-audit"):
        if not args.candidate:
            parser.error(args.command + " requires --candidate")
        candidate = read_json(args.candidate)
        path = run / "candidates" / (candidate_id(candidate) + ".json")
        write_json(path, candidate)
        worker.put(path, relative(path))
        result = worker.run(args.command.replace("-", "_"), run, profile_dir=relative(profile), candidate=relative(path))
        write_json(run / ("evaluation.json" if args.command == "evaluate" else "timing-audit.json"), result)
    elif args.command == "search":
        from .search import search
        from .preparation import prepare
        prepare(config, run, profile, worker)
        result = search(config, run, profile, worker=worker)
    elif args.command == "integrate":
        from .integration import integrate
        try:
            result = integrate(config, run, profile, worker)
        finally:
            worker.close()
    elif args.command in ("status", "report"):
        from .store import Store
        store = Store(run / "state.sqlite")
        if args.command == "report":
            from .reporting import write_swarm_report
            report = write_swarm_report(run, config, profile_dir=profile, store=store)
            result = {"status": report["status"], "budget": report["budget"],
                      "accepted_model_gain": report["accepted_model_gain"],
                      "report": str(run / "report.md"), "details": str(run / "report.json")}
        else:
            result = {"status": store.get("status", "not_started"), "usage": store.usage(),
                      "budget": store.budget_snapshot(config["agent"]),
                      "active_task": store.get("active_task"), "task_progress": store.get("task_progress", {}),
                      "stop": store.get("swarm_stop"), "candidates": store.candidate_summaries()}
    elif args.command == "api-check":
        from .agent import create_agent
        from .store import Store
        small = {**config["agent"], "max_output_tokens": 128}
        result = create_agent(small, Store(run / "api.sqlite"), run).call(
            "connection_test", 'Return {"connected":true}.', "Return only the requested JSON.")
        write_json(run / "api-check.json", result)
    elif args.command == "export":
        import shutil
        final = read_json(run / "result.json")
        if not (final.get("model_result") or {}).get("decision", {}).get("accepted"):
            raise ValueError("Export requires a measured complete model gain.")
        target = run / "export"
        target.mkdir(exist_ok=True)
        write_json(target / "manifest.json", final["manifest"])
        write_json(target / "model_result.json", final["model_result"])
        for entry in final["manifest"]["entries"]:
            name = entry["candidate_id"] + ".json"
            shutil.copy2(run / "candidates" / name, target / name)
        result = {"export": str(target)}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
