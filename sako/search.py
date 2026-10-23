"""Eight writers, four islands, and deterministic measured selection."""

from .agent import Gemini
from .core import select_nonoverlapping
from .references import retrieve
from .transport import Worker


def make_manifest(store, catalog):
    entries, seen = [], set()
    tasks = {t["id"]: t for t in catalog["tasks"]}
    rows = sorted(store.candidates(), key=lambda r: (r["generation"], r["score"] or 0), reverse=True)
    for row in rows:
        if (not row["accepted"] or row["task"] in seen
                or row["id"] in store.get("blocked_model_candidates", {})):
            continue
        seen.add(row["task"])
        task = tasks[row["task"]]
        entries.append({"task_id": row["task"], "candidate_id": row["id"], "paths": task["paths"],
                        "priority": task["priority_ms"] * (1 - 1 / row["score"])})
    return {"entries": select_nonoverlapping(entries)}


def search(config, run_dir, profile_dir, reviewer=False):
    from .loop import SearchLoop
    return SearchLoop(config, run_dir, profile_dir, Gemini, Worker, retrieve,
                      make_manifest, reviewer=reviewer).run()
