"""Eight writers, four islands, and deterministic measured selection."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

from .agent import Gemini
from .core import write_json, read_json, validate_candidate, select_nonoverlapping, digest, implementation_hash
from .references import retrieve
from .store import Store, BudgetExhausted
from .transport import Worker

SYSTEM = "You assist a CUDA kernel search. Return JSON. Measurements decide correctness and speed. Use short, clear English."


def make_manifest(store, catalog):
    entries, seen = [], set()
    tasks = {t["id"]: t for t in catalog["tasks"]}
    for row in store.candidates():
        if not row["accepted"] or row["task"] in seen:
            continue
        seen.add(row["task"])
        task = tasks[row["task"]]
        entries.append({"task_id": row["task"], "candidate_id": row["id"], "paths": task["paths"],
                        "priority": task["priority_ms"] * (1 - 1 / row["score"])})
    return {"entries": select_nonoverlapping(entries)}


def search(config, run_dir, profile_dir, reviewer=False):
    root = Path(__file__).resolve().parents[1]
    run_dir, profile_dir = Path(run_dir).resolve(), Path(profile_dir).resolve()
    catalog = read_json(profile_dir / "tasks.json")
    store = Store(run_dir / "state.sqlite")
    fingerprint = digest({"config": config, "catalog": catalog, "implementation": implementation_hash()})
    if store.get("fingerprint", fingerprint) != fingerprint:
        raise ValueError("Code, config, or task catalog changed. Use a new run directory.")
    store.set("fingerprint", fingerprint)
    client, worker = Gemini(config["agent"], store, run_dir), Worker(config)
    writer_system = Path(__file__).with_name("prompts").joinpath("writer.txt").read_text(encoding="utf-8")
    config_agent = config["agent"]
    tasks = sorted(catalog["tasks"], key=lambda task: task["priority_ms"], reverse=True)
    tasks_by_id = {t["id"]: t for t in tasks}
    profile_relative = profile_dir.relative_to(root).as_posix()
    start = store.get("generation", 0)
    status = "complete"

    def support(role, value):
        try:
            result = client.call(role, json.dumps(value), SYSTEM)
            if not isinstance(result, dict):
                raise ValueError(f"{role} must return a JSON object.")
            return result
        except BudgetExhausted:
            raise
        except Exception as error:
            result = {"notes": f"{role} unavailable: {error}"}
            store.set(role + "_error", result)
            return result

    def write_one(slot, generation, plan):
        island = config_agent["islands"][slot // 2]
        order = [key for key in plan.get("task_order", []) if key in tasks_by_id]
        order.extend(t["id"] for t in tasks if t["id"] not in order)
        task = tasks_by_id[order[(generation * 4 + slot // 2) % len(order)]]
        rows = store.candidates(task["id"])
        migrate = generation % config_agent["migration_interval"] == 0
        parents = [r for r in rows if r["accepted"] and (migrate or r["island"] == island)]
        parent = parents[0] if parents else None
        recent = sorted(rows, key=lambda r: r["generation"], reverse=True)[:3]
        feedback = [{"id": r["id"], "strategy": r["source"]["strategy"],
                     "result": json.dumps(r["result"])[-7000:]} for r in recent]
        prompt = {"task": task, "island": island, "writer": slot, "generation": generation,
                  "persistent_plan": plan, "parent": parent["source"] if parent else None,
                  "feedback": feedback, "references": retrieve(island + " " + task["kind"], task["kind"])}
        # Remove old observations before source, task, or persistent plan.
        while len(json.dumps(prompt)) > config_agent["context_chars"] and prompt["feedback"]:
            prompt["feedback"].pop()
        while len(json.dumps(prompt)) > config_agent["context_chars"] and prompt["references"]:
            prompt["references"].pop()
        candidate = client.call(f"writer_{slot}_{island}", json.dumps(prompt), writer_system)
        validate_candidate(candidate)
        if candidate["task_id"] != task["id"]:
            raise ValueError("Writer changed the assigned task.")
        return candidate, island, parent["id"] if parent else None

    try:
        for generation in range(start, config_agent["generations"]):
            if (run_dir / "STOP").exists():
                status = "stopped"
                break
            print(f"Generation {generation + 1}: preparing eight writers.", flush=True)
            plan = support("coordinator", {"request": "Return task_order (task IDs) and plan (short text).",
                "previous_plan": store.get("plan"), "analysis": store.get("analysis"),
                "tasks": [{k: t[k] for k in ("id", "kind", "priority_ms")} for t in tasks]})
            store.set("plan", plan)
            write_json(run_dir / "plan.json", plan)
            exhausted = False
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(write_one, slot, generation, plan) for slot in range(8)]
                for future in as_completed(futures):
                    try:
                        candidate, island, parent = future.result()
                        key, fresh = store.add(candidate, island, generation, parent)
                        if not fresh and next(r for r in store.candidates() if r["id"] == key)["result"] is not None:
                            continue
                        path = run_dir / "candidates" / (key + ".json")
                        write_json(path, candidate)
                        worker.put(path, path.relative_to(root).as_posix())
                        print(f"Testing {candidate['task_id']} from {island}.", flush=True)
                        try:
                            result = worker.run("evaluate", run_dir, profile_dir=profile_relative,
                                                candidate=path.relative_to(root).as_posix())
                        except Exception as error:
                            result = {"decision": {"accepted": False, "speedup": None}, "error": str(error)[-12000:]}
                        store.record(key, result)
                    except BudgetExhausted:
                        exhausted = True
                    except Exception as error:
                        write_json(run_dir / "errors" / f"{generation}-{futures.index(future)}.json", {"error": str(error)})
            if exhausted:
                status = "budget_exhausted"
                break
            ranking = [{"id": r["id"], "task": r["task"], "island": r["island"],
                        "accepted": r["accepted"], "score": r["score"],
                        "error": str(r["result"].get("error", ""))[-1500:] if r["result"] else "pending"}
                       for r in store.candidates()]
            analysis = support("performance_analyst", {"request": "Explain measured failures and next kernel experiments.",
                                                       "ranking": ranking[:40]})
            store.set("analysis", analysis)
            manifest = make_manifest(store, catalog)
            integration = support("integration_agent", {"request": "Review replacement interactions. Return notes. Do not change scores.",
                                                         "manifest": manifest, "ranking": ranking[:40]})
            write_json(run_dir / "integration_notes.json", integration)
            if reviewer and generation > 0 and not manifest["entries"]:
                review = support("failure_reviewer", {"request": "Find a shared cause in these failures.", "ranking": ranking[:16]})
                store.set("review", review)
            write_json(run_dir / "manifest.json", manifest)
            current = digest(manifest)
            stall = store.get("stall", 0) + 1 if current == store.get("last_manifest") else 0
            store.set("last_manifest", current)
            store.set("stall", stall)
            store.set("generation", generation + 1)
            if stall >= config_agent["stall_generations"]:
                status = "stalled"
                break
    except BudgetExhausted:
        status = "budget_exhausted"
    manifest = make_manifest(store, catalog)
    write_json(run_dir / "manifest.json", manifest)
    result = {"status": status, "usage": store.usage(), "manifest": manifest, "model_result": None}
    if manifest["entries"] and status != "stopped":
        path = run_dir / "manifest.json"
        worker.put(path, path.relative_to(root).as_posix())
        print("Testing the complete model against torch.compile.", flush=True)
        previous = store.get("model_measurement")
        if previous and previous["manifest_hash"] == digest(manifest):
            result["model_result"] = previous["result"]
        else:
            result["model_result"] = worker.run("model", run_dir, profile_dir=profile_relative,
                manifest=path.relative_to(root).as_posix(), candidate_root=(run_dir / "candidates").relative_to(root).as_posix())
            store.set("model_measurement", {"manifest_hash": digest(manifest), "result": result["model_result"]})
    write_json(run_dir / "result.json", result)
    return result
