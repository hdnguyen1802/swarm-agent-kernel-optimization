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
    parser.add_argument("command", choices=["references", "sync", "doctor", "profile", "baseline", "model-check",
                                             "evaluate", "search", "status", "report", "api-check", "export"])
    parser.add_argument("--config", default="configs/project.json")
    parser.add_argument("--run", default="runs/main")
    parser.add_argument("--profile", default="runs/profile")
    parser.add_argument("--candidate")
    parser.add_argument("--manifest")
    parser.add_argument("--review-failures", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    run, profile = Path(args.run).resolve(), Path(args.profile).resolve()
    relative(run)
    relative(profile)
    run.mkdir(parents=True, exist_ok=True)
    worker = Worker(config)
    if args.command == "references":
        from .references import build_index
        result = build_index()
    elif args.command == "sync":
        result = worker.sync()
    elif args.command == "doctor":
        result = worker.run("doctor", run)
        write_json(run / "doctor.json", result)
    elif args.command == "profile":
        result = worker.run("profile", run, profile_dir=relative(profile))
        write_json(profile / "tasks.json", result)
        print(f"Saved {len(result['tasks'])} tasks. Revision: {result['config']['revision']}")
        return
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
    elif args.command == "evaluate":
        if not args.candidate:
            parser.error("evaluate requires --candidate")
        candidate = read_json(args.candidate)
        path = run / "candidates" / (candidate_id(candidate) + ".json")
        write_json(path, candidate)
        worker.put(path, relative(path))
        result = worker.run("evaluate", run, profile_dir=relative(profile), candidate=relative(path))
        write_json(run / "evaluation.json", result)
    elif args.command == "search":
        from .search import search
        result = search(config, run, profile, reviewer=args.review_failures)
    elif args.command in ("status", "report"):
        from .store import Store
        store = Store(run / "state.sqlite")
        result = {"usage": store.usage(), "generation": store.get("generation", 0),
                  "candidates": [{k: row[k] for k in ("id", "task", "island", "score", "accepted")}
                                 for row in store.candidates()]}
        if args.command == "report":
            lines = ["# SAKO search report", "", f"API calls: {result['usage']['calls']}", "",
                     "| Candidate | Task | Island | Kernel speed ratio | Passed kernel gates |",
                     "|---|---|---|---:|---|"]
            lines.extend(f"| {r['id']} | {r['task']} | {r['island']} | {r['score']} | {bool(r['accepted'])} |"
                         for r in result["candidates"])
            final = read_json(run / "result.json") if (run / "result.json").exists() else {}
            model = final.get("model_result")
            lines.extend(["", "Complete model decision: " + json.dumps(model["decision"] if model else "Not measured.")])
            (run / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    elif args.command == "api-check":
        from .agent import Gemini
        from .store import Store
        small = {**config["agent"], "max_output_tokens": 128}
        result = Gemini(small, Store(run / "api.sqlite"), run).call(
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
