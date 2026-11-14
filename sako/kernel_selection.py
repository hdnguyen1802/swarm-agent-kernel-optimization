"""Keep measured kernel gains separate from full model acceptance."""

import math
import statistics

from .core import digest, score_cases, summary, select_nonoverlapping
from .store import operation_correct


def kernel_first(config):
    return config.get("kernel_search", {}).get("enabled", False)


def kernel_config(config):
    rules = {**config, "min_speedup": config.get("kernel_search", {}).get("min_speedup", 1.01)}
    if config.get("generation_objective") == "decode":
        rules["trials"] = 5
    return rules


def search_score(result, anchor=None, *, mode=None):
    """Rank trials with paired compiled timings or a saved fixed reference."""
    cases = result.get("cases", [])
    try:
        if mode == "paired_compile":
            # A kept-kernel acceptance pair has a different reference.
            if not cases:
                return None
            return math.exp(statistics.mean(math.log(
                summary(case["search_reference"]["baseline"]["samples_ms"])["median_ms"] /
                summary(case["search_reference"]["candidate"]["samples_ms"])["median_ms"])
                for case in cases))
        if mode not in (None, "fixed_anchor"):
            raise ValueError("Unknown search score reference.")
        if not cases or anchor is None or len(anchor) != len(cases):
            return None
        return statistics.geometric_mean(summary([a])["median_ms"] / summary(c["candidate"]["samples_ms"])["median_ms"]
                                         for a, c in zip(anchor, cases))
    except (KeyError, TypeError, ValueError):
        return None


def record_archive_score(store, task, result, baseline_ids, *, trials=1, mode="fixed_anchor"):
    """Record the configured ranking evidence separately from the keep decision."""
    cases = result.get("cases", [])
    result["search_score"] = None
    if (not operation_correct(result) or result.get("reference_state_intact") is False
            or not cases or len(cases) != len(task["cases"])
            or any(c.get("case_index", i) != i for i, c in enumerate(cases))):
        return
    if result.get("performance_scope") == "cached_decode":
        from .shape_selection import case_signature
        if mode != "paired_compile":
            return
        wanted = [i for i,c in enumerate(task["cases"]) if c.get("generation_step", {}).get("phase") == "decode"]
        def layout(i):
            c = task["cases"][i]
            return c.get("generation_step", {}).get("phase"), digest(case_signature(c, task.get("scalars", ())))
        initial = result["stages"]["initial_case_indices"]
        if (wanted != result["timed_case_indices"] or {layout(i) for i in initial} != {layout(i) for i in range(len(cases))}
                or len(initial) != len({layout(i) for i in initial})):
            return
        cases = [cases[i] for i in wanted]
        peer = result.get("archive_measurement")
        if peer and (not peer.get("correct") or not all(c.get("search_reference", {}).get("hardware_fingerprint") ==
                        peer.get("hardware_fingerprint") for c in cases)):
            return
    for case in cases:
        for side in ("baseline", "candidate"):
            samples = (case.get(side) or {}).get("samples_ms")
            if not isinstance(samples, list) or len(samples) < trials:
                return
    if mode == "paired_compile":
        for case in cases:
            reference = case.get("search_reference", {})
            if reference.get("kind") != "paired_compile":
                return
            for side in ("baseline", "candidate"):
                samples = (reference.get(side) or {}).get("samples_ms")
                if not isinstance(samples, list) or len(samples) < trials:
                    return
        result["search_score"] = search_score({"cases": cases}, mode=mode)
        return
    if mode != "fixed_anchor":
        raise ValueError("Unknown search score reference.")
    saved = store.get("kernel_score_anchors", {})
    anchors = list(saved.get(task["id"], [None] * len(cases)))
    if len(anchors) != len(cases):
        raise ValueError("The archive reference and the task have different fixture counts.")
    original_cases = set() if baseline_ids else set(range(len(cases)))
    if isinstance(baseline_ids, dict) and baseline_ids and any("timing_reference" not in c for c in cases):
        from .shape_selection import groups
        original_cases = {i for shape, group in groups(task).items() if not baseline_ids.get(shape)
                          for i in group["indices"]}
    for i, case in enumerate(cases):
        # Older workers only identified the baseline at the assignment level.
        kind = case.get("timing_reference", "torch_compile" if i in original_cases else "kept_kernel")
        if anchors[i] is None and kind == "torch_compile":
            try:
                anchors[i] = summary(case["baseline"]["samples_ms"])["median_ms"]
            except (KeyError, TypeError, ValueError):
                continue
        if anchors[i] is not None:
            case["search_reference"] = {"kind": "fixed_anchor", "baseline_median_ms": anchors[i],
                                        "acceptance_evidence": False}
    if saved.get(task["id"]) != anchors:
        saved[task["id"]] = anchors
        store.set("kernel_score_anchors", saved)
    result["search_score"] = search_score(result, anchors)


def keep_kernel(store, task, candidate_id, config):
    """Promote only against the current kept baseline. Replays cannot roll it back."""
    if candidate_id in store.get("blocked_model_candidates", {}):
        return False
    from .shape_selection import enabled
    if enabled(config):
        return keep_shapes(store, task, candidate_id, config)
    kept = store.get("kernel_baselines", {})
    current = kept.get(task["id"], {}).get("candidate_id")
    if current == candidate_id:
        return False
    row = next(r for r in store.candidates(task["id"]) if r["id"] == candidate_id)
    result = row["result"] or {}
    if ("timing_baseline" not in result or result["timing_baseline"].get("candidate_id") != current
            or not result.get("decision", {}).get("accepted") or not operation_correct(result)
            or result.get("reference_state_intact") is False
            or len(result.get("cases", [])) != len(task["cases"])
            or not score_cases(result.get("cases", []), kernel_config(config))["accepted"]):
        return False
    kept[task["id"]] = {"candidate_id": candidate_id, "previous_candidate_id": current,
        "decision": result["decision"], "search_score": row["score"],
        "scope": "kernel_only", "model_status": "not_tested"}
    store.set("kernel_baselines", kept)
    return True


def keep_shapes(store, task, candidate_id, config):
    from .shape_selection import decide
    kept = store.get("kernel_baselines", {})
    current = kept.get(task["id"], {"shape_scoped": True, "shapes": {}})
    if not current.get("shape_scoped"):
        raise ValueError("Shape selection changed. Use a new run.")
    row = next(r for r in store.candidates(task["id"]) if r["id"] == candidate_id)
    result = row["result"] or {}
    if not operation_correct(result) or result.get("reference_state_intact") is False:
        return False
    decisions = decide(result, task, kernel_config(config))
    baselines = result.get("timing_baseline", {}).get("shapes", {})
    changed = False
    for key, decision in decisions.items():
        previous = current["shapes"].get(key, {}).get("candidate_id")
        decode = result.get("performance_scope") == "cached_decode"
        if decode and (not result.get("search_score") or decision["speedup"] is None or
                       decision["speedup"] <= current["shapes"].get(key, {}).get("decision", {}).get("speedup", 0)):
            continue
        if (key not in baselines or (not decode and baselines[key] != previous) or previous == candidate_id
                or not decision["accepted"]):
            continue
        current["shapes"][key] = {"candidate_id": candidate_id, "previous_candidate_id": previous,
            "signature": decision["signature"], "decision": decision, "model_status": "not_tested",
            **({"paths": decision["paths"]} if "paths" in decision else {})}
        changed = True
    if changed:
        kept[task["id"]] = current
        store.set("kernel_baselines", kept)
    return changed


def kernel_manifest(store, catalog, *, exclude_tasks=(), region_choices=None):
    from .path_selection import manifest_allowed
    tasks = {t["id"]: t for t in catalog["tasks"]}
    entries = []
    restrictions = store.get("candidate_scope_restrictions", {})
    for task_id, kept in store.get("kernel_baselines", {}).items():
        if task_id in exclude_tasks:
            continue
        task = tasks[task_id]
        if kept.get("shape_scoped"):
            grouped = {}
            candidates = {row["id"]: row for row in store.candidates(task_id)}
            for shape in kept["shapes"].values():
                paths = shape.get("paths", task["paths"])
                proposal = {"task_id": task_id, "candidate_id": shape["candidate_id"],
                            "paths": paths, "signatures": [shape["signature"]]}
                if shape["candidate_id"] in restrictions:
                    proposal["scope_revision"] = restrictions[shape["candidate_id"]]["failure_result_hash"]
                if not paths or not manifest_allowed(store, {"entries": [proposal]}):
                    continue
                row = candidates[shape["candidate_id"]]
                if not row["correct"] or not shape["decision"]["accepted"]:
                    raise ValueError("The kept shape has no passing operation result.")
                entry = grouped.setdefault((row["id"], tuple(paths)), {"task_id": task_id, "candidate_id": row["id"],
                    "paths": paths, "priority": task["priority_ms"], "signatures": []})
                if "scope_revision" in proposal:
                    entry["scope_revision"] = proposal["scope_revision"]
                entry["signatures"].append(shape["signature"])
            by_signatures = {}
            for entry in grouped.values():
                key = (entry["candidate_id"], tuple(sorted(digest(s) for s in entry["signatures"])))
                if key in by_signatures:
                    by_signatures[key]["paths"].extend(entry["paths"])
                else:
                    by_signatures[key] = {**entry, "paths": list(entry["paths"])}
            entries.extend(by_signatures.values())
            continue
        paths = kept.get("paths", task["paths"])
        if not manifest_allowed(store, {"entries": [{"task_id": task_id,
                "candidate_id": kept["candidate_id"], "paths": paths}]}):
            continue
        row = next(r for r in store.candidates(task_id) if r["id"] == kept["candidate_id"])
        if not row["accepted"] or not row["correct"]:
            raise ValueError("The kept kernel has no passing operation result.")
        entries.append({"task_id": task_id, "candidate_id": row["id"], "paths": paths,
                        "priority": task["priority_ms"]})
    # Shape variants of one task travel together. Parent/child alternatives
    # remain separate until a complete model comparison chooses a region.
    choices = store.get("kernel_region_choices", {}) if region_choices is None else region_choices
    excluded = {key for region in catalog.get("hierarchy", {}).get("regions", [])
                if region["id"] in choices for key in region["subsection_ids"]
                if key not in choices[region["id"]]}
    entries = [e for e in entries if e["task_id"] not in excluded]
    representatives = {}
    for entry in entries:
        representative = representatives.setdefault(entry["task_id"], {**entry, "paths": []})
        representative["paths"].extend(path for path in entry["paths"] if path not in representative["paths"])
    selected_tasks = {e["task_id"] for e in select_nonoverlapping(list(representatives.values()))}
    allowed_alternatives = {key for region in catalog.get("hierarchy", {}).get("regions", [])
                            if region.get("alternative_task_ids") for key in region["subsection_ids"]}
    if set(representatives) - selected_tasks - allowed_alternatives:
        raise ValueError("Kept kernel replacements overlap.")
    selected = [e for e in entries if e["task_id"] in selected_tasks]
    from .shape_selection import validate_entries
    validate_entries(selected, tasks)
    return {"entries": selected}


def accepted_model_trials(config):
    return config.get("model_trial_base", "kept") == "accepted"


def model_trial_manifest(store, catalog, local):
    """Add a region to a verified model, without importing failed kernel keeps."""
    import copy
    from .integration_repairs import safe_measurement
    best = store.get("best_model_measurement")
    if not best or not best["result"].get("decision", {}).get("accepted") or not safe_measurement(store, best):
        return copy.deepcopy(local)
    paths = [path for entry in local["entries"] for path in entry["paths"]]
    def overlaps(path):
        return any(path == other or path.startswith(other + ".") or other.startswith(path + ".") for other in paths)
    base = []
    for entry in best["manifest"]["entries"]:
        remaining = [path for path in entry["paths"] if not overlaps(path)]
        if remaining:
            base.append({**copy.deepcopy(entry), "paths": remaining})
    result = {"entries": base + copy.deepcopy(local["entries"])}
    from .shape_selection import validate_entries
    validate_entries(result["entries"], {task["id"]: task for task in catalog["tasks"]})
    return result


def region_manifests(store, catalog, region, *, config=None, include_accepted=True):
    """Compare whole-parent and child designs without double-counting work."""
    choices = store.get("kernel_region_choices", {})
    alternatives = [region.get("leaf_task_ids", region["subsection_ids"])]
    alternatives.extend([key] for key in region.get("alternative_task_ids", []))
    seen, result = set(), []
    for task_ids in alternatives:
        clean = accepted_model_trials(config or catalog.get("config", {}))
        manifest = kernel_manifest(store, catalog,
            exclude_tasks=[task["id"] for task in catalog["tasks"] if clean and task["id"] not in region["subsection_ids"]],
            region_choices={**choices, region["id"]: task_ids})
        manifests = ([manifest, model_trial_manifest(store, catalog, manifest)]
                     if clean and include_accepted and manifest["entries"] else [manifest])
        from .core import digest
        for manifest in manifests:
            identity = digest(manifest)
            if identity not in seen:
                result.append((task_ids, manifest))
                seen.add(identity)
    return result
