"""Launch the Triton swarm and assemble measured model candidates."""

from .agent import create_agent
from .core import select_nonoverlapping
from .transport import Worker


def make_manifest(store, catalog):
    best = store.get("best_model_measurement")
    active = store.get("active_task")
    entries = [dict(entry) for entry in (best or {}).get("manifest", {}).get("entries", [])
               if entry["task_id"] != active]
    seen = {entry["task_id"] for entry in entries}
    tasks = {t["id"]: t for t in catalog["tasks"]}
    blocked = store.get("blocked_model_candidates", {})
    rejected = store.get("rejected_model_candidates", {})
    rows = sorted(store.candidates(), key=lambda r: (r["score"] or 0, r["generation"]), reverse=True)
    for row in rows:
        if (not row["accepted"] or row["task"] in seen or (active is not None and row["task"] != active)
                or row["id"] in blocked or row["id"] in rejected):
            continue
        seen.add(row["task"])
        task = tasks[row["task"]]
        entries.append({"task_id": row["task"], "candidate_id": row["id"], "paths": task["paths"],
                        "priority": task["priority_ms"] * (1 - 1 / row["score"])})
    # If no new replacement qualifies, retain the last accepted active replacement.
    if active not in seen:
        entries.extend(dict(entry) for entry in (best or {}).get("manifest", {}).get("entries", [])
                       if entry["task_id"] == active)
    return {"entries": select_nonoverlapping(entries)}


def search(config, run_dir, profile_dir, worker=None):
    from .swarm import SwarmLoop
    if config["agent"].get("search_mode", "swarm") != "swarm":
        raise ValueError("Only swarm search is supported. Start a new run with a current config.")
    factory = (lambda _: worker) if worker is not None else Worker
    return SwarmLoop(config, run_dir, profile_dir, create_agent, factory, make_manifest).run()
