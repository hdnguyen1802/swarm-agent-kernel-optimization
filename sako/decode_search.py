"""Rank decode kernels on their original compiled operations. Keep a shortlist for each target."""

from collections import defaultdict
import statistics

from .core import digest, summary
from .shape_selection import groups, validate_entries
from .store import operation_correct

FINALIST_LIMIT = 4


def record_candidate(store, task, identity, result, config):
    """Keep a small index of passing routes. The store retains every full trial separately."""
    if not operation_correct(result) or not result.get("search_score") or result.get("reference_state_intact") is False:
        return
    rows = result["cases"]
    options, total_reference = [], 0.
    for key, decision in result.get("shape_decisions", {}).items():
        pairs = [rows[i]["search_reference"] for i in decision["indices"]]
        base = [summary(p["baseline"]["samples_ms"]) for p in pairs]
        new = [summary(p["candidate"]["samples_ms"]) for p in pairs]
        reference = statistics.mean(x["median_ms"] for x in base)
        total_reference += reference
        if (not decision["accepted"] or any(a["median_ms"] < config["kernel_search"]["min_speedup"] * b["median_ms"]
                or b["p95_ms"] > a["p95_ms"] * config["max_p95_ratio"] for a, b in zip(base, new))):
            continue
        paths = decision.get("paths", task["paths"])
        saved = statistics.mean(a["median_ms"] - b["median_ms"] for a, b in zip(base, new))
        for path in paths:
            options.append({"path": path, "signature": decision["signature"], "saving": saved / len(paths)})
    if not options:
        return
    for option in options:
        option["saving"] *= task["priority_ms"] / total_reference
    index = store.get("decode_candidates:" + task["id"], {})
    index[identity] = {"id": identity, "routes": options, "saving": sum(r["saving"] for r in options)}
    ranks = store.get("decode_local_ranks", {})
    ranks[task["id"]] = [r["id"] for r in sorted(index.values(), key=lambda r: (-r["saving"], r["id"]))[:FINALIST_LIMIT]]
    store.set_many({"decode_candidates:" + task["id"]: index, "decode_local_ranks": ranks})


def routes(entries, tasks):
    result = {}
    for entry in entries:
        task = tasks[entry["task_id"]]
        available = groups(task) if "signatures" not in entry else {}
        for path in entry["paths"]:
            signatures = entry.get("signatures") or [g["signature"] for g in available.values()
                                                       if path in g.get("paths", task["paths"])]
            for signature in signatures:
                key = (task["id"], path, digest(signature))
                result[key] = (entry["candidate_id"], signature, 0.)
    return result


def manifest(routes_by_key):
    by_path = defaultdict(dict)
    for (task, path, sig), (identity, signature, _) in sorted(routes_by_key.items()):
        by_path[task, identity, path][sig] = signature
    entries = {}
    for (task, identity, path), signatures in by_path.items():
        key = task, identity, tuple(sorted(signatures))
        entry = entries.setdefault(key, {"task_id": task, "candidate_id": identity, "paths": [],
                                               "signatures": [signatures[s] for s in key[2]]})
        entry["paths"].append(path)
    return {"entries": list(entries.values())}


def overlay(base, addition):
    changed_paths = {(task, path) for task, path, _ in addition}
    result = {k: v for k, v in base.items() if not any(
        (k[1] == path and k[0] != task) or k[1].startswith(path + ".") or path.startswith(k[1] + ".")
        for task, path in changed_paths)}
    return {**result, **addition}


def target_finalists(store, catalog, task_id, incumbent, *, candidate_ids=None):
    """Build up to four distinct challenges for one target on the same accepted base."""
    tasks = {t["id"]: t for t in catalog["tasks"]}
    base = routes(incumbent["entries"], tasks)
    seen = {digest(manifest(base))}
    ranked = sorted(store.get("decode_candidates:" + task_id, {}).values(),
                    key=lambda row: (-row["saving"], row["id"]))
    finalists = []
    for row in ranked:
        if candidate_ids is not None and row["id"] not in candidate_ids:
            continue
        addition = {(task_id, r["path"], digest(r["signature"])): (row["id"], r["signature"], r["saving"])
                    for r in row["routes"]}
        proposal = manifest(overlay(base, addition))
        identity = digest(proposal)
        if identity in seen:
            continue
        validate_entries(proposal["entries"], tasks)
        finalists.append({"rank": len(finalists) + 1, "candidate_id": row["id"], "manifest": proposal,
                          "estimated_target_saving_ms": row["saving"]})
        seen.add(identity)
        if len(finalists) == FINALIST_LIMIT:
            break
    return finalists


def model_metrics(best, trials, previous, catalog=None):
    """Use the same timed blocks for both references. Retain historical provenance on a failed final."""
    measured = best or next((t for t in trials if t["result"].get("cases") and
                             all(c.get("candidate") for c in t["result"]["cases"])), None)
    if not measured:
        if previous:
            return previous.get("decode_metrics", [])
        rows = []
        for record in (catalog or {}).get("starting_baseline", {}).get("workloads", []):
            ms = record["generation_metrics"]["mean_decode_ms"]["median_ms"]
            rows.append({"workload": record["workload"], "ms_per_token": ms,
                         "tokens_per_second": 1000 * record["workload"]["batch_size"] / ms, "original_ms_per_token": ms,
                         "previous_ms_per_token": ms, "vs_original_compile": 1., "vs_previous_accepted": 1.,
                         "pairs": 0, "source": "original_starting_reference"})
            eager = next((r for r in catalog["starting_baseline"].get("pytorch_eager", {}).get("workloads", [])
                          if r["workload"] == record["workload"]), None)
            if eager:
                eager_ms = eager["generation_metrics"]["mean_decode_ms"]["median_ms"]
                rows[-1].update(pytorch_ms_per_token=eager_ms, vs_pytorch=eager_ms / ms,
                                compile_vs_pytorch=eager_ms / ms)
        return rows
    rows = []
    for case in measured["result"]["cases"]:
        metrics = case["generation_metrics"]
        base = metrics["baseline"]["mean_decode_ms"]["median_ms"]
        old = metrics.get("incumbent", metrics["baseline"])["mean_decode_ms"]["median_ms"]
        selected = metrics["candidate"]["mean_decode_ms"]["median_ms"] if best else old
        rows.append({"workload": case["workload"], "ms_per_token": selected,
                     "tokens_per_second": 1000 * case["workload"]["batch_size"] / selected,
                     "original_ms_per_token": base, "previous_ms_per_token": old,
                     "vs_original_compile": base / selected, "vs_previous_accepted": old / selected,
                     "pairs": case["sampling"].get("assessment", {}).get("pairs", 0),
                     "timed_runs": case["sampling"].get("timed_runs", len(case["candidate"].get("samples_ms", []))),
                     "source": case["sampling"].get("method", "final_paired_decode")})
        eager = metrics.get("pytorch_eager")
        if eager:
            ms = eager["mean_decode_ms"]["median_ms"]
            rows[-1].update(pytorch_ms_per_token=ms, vs_pytorch=ms / selected,
                            compile_vs_pytorch=ms / base)
    return rows
