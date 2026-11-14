"""Test the kept kernels together after kernel search has stopped."""

from pathlib import Path

from .core import candidate_id, digest, implementation_hash, read_json, write_json
from .kernel_selection import (accepted_model_trials, kernel_first, kernel_manifest,
                               model_trial_manifest, region_manifests)
from .store import Store
from .model_acceptance import model_trial_key, reusable_trial


def integrate(config, run_dir, profile_dir, worker):
    root = Path(__file__).resolve().parents[1]
    run, profile = Path(run_dir).resolve(), Path(profile_dir).resolve()
    relative = lambda path: Path(path).relative_to(root).as_posix()
    store = Store(run / "state.sqlite")
    catalog = read_json(profile / "tasks.json")
    expected = digest({"config": config, "catalog": catalog, "implementation": implementation_hash()})
    if not kernel_first(config) or store.get("fingerprint") != expected:
        raise ValueError("Integration requires a matching kernel-first run. Do not reuse a changed run.")
    if store.get("status") == "running":
        raise ValueError("Stop the kernel search before integration.")
    from .measurement import decode_only
    if decode_only(config):
        from .decode_validation import final_combination
        from .reporting import write_swarm_report
        trial = final_combination(run, profile, catalog, store, worker)
        write_swarm_report(run, config, profile_dir=profile, store=store)
        return trial
    manifest = kernel_manifest(store, catalog)
    if not manifest["entries"]:
        raise ValueError("No kept kernels are ready for integration.")
    cache = store.get("model_measurements", {})
    clean = accepted_model_trials(config)
    alternatives = {} if clean else {digest(manifest): manifest}
    regions = list(catalog.get("hierarchy", {}).get("regions", []))
    if clean:
        covered = {key for region in regions for key in region["subsection_ids"]}
        regions.extend({"id": task["id"], "subsection_ids": [task["id"]]}
                       for task in catalog["tasks"] if task["id"] not in covered)
    for region in regions:
        if clean or (region.get("alternative_task_ids") and region["id"] not in store.get("hierarchy_completed_regions", {})):
            for _, choice in region_manifests(store, catalog, region, config=config, include_accepted=False):
                if choice["entries"]:
                    alternatives[digest(choice)] = choice
    measured = None
    def measure(choice, incumbent=None):
        best = store.get("best_model_measurement")
        from .integration_repairs import safe_measurement
        if best and best["manifest"] == choice and safe_measurement(store, best):
            return best
        key = model_trial_key(expected, choice, incumbent)
        trial = reusable_trial(cache, key)
        if trial is None:
            sources = {entry["candidate_id"]: entry for entry in [*choice["entries"], *(incumbent or {}).get("entries", [])]}
            for identity, entry in sources.items():
                path = run / "candidates" / (identity + ".json")
                candidate = read_json(path)
                if candidate_id(candidate) != entry["candidate_id"] or candidate["task_id"] != entry["task_id"]:
                    raise ValueError("The saved candidate does not match the kept kernel.")
                worker.put(path, relative(path))
            path = run / "integrations" / (key + ".manifest.json")
            write_json(path, choice)
            worker.put(path, relative(path))
            options = {}
            if incumbent is not None:
                base_path = run / "integrations" / (key + ".incumbent.json")
                write_json(base_path, incumbent)
                worker.put(base_path, relative(base_path))
                options["incumbent_manifest"] = relative(base_path)
                from .measurement import enabled as adaptive, incumbent_reference
                if adaptive(config):
                    ref_path = run / "integrations" / (key + ".incumbent-reference.json")
                    write_json(ref_path, incumbent_reference(best, catalog))
                    worker.put(ref_path, relative(ref_path))
                    options["incumbent_reference"] = relative(ref_path)
            result = worker.run("model", run, profile_dir=relative(profile), manifest=relative(path),
                                candidate_root=relative(run / "candidates"),
                                feedback_trace_dir=relative(run / "model_checks" / key), **options)
            trial = {"manifest": choice, "result": result, "stage": "deferred_integration",
                     **({"incumbent": incumbent} if incumbent is not None else {})}
            from .deployment_diagnostics import retain_trace_feedback
            retain_trace_feedback(store, key, choice, result)
            write_json(run / "integrations" / (key + ".result.json"), trial)
            cache[key] = trial
            store.set("model_measurements", cache)
        return trial
    tickets = []
    def record(trial):
        from .feedback import save_feedback
        from .integration_repairs import record_failure
        saved = save_feedback(run, trial, name="integration-repair-" + digest(trial["manifest"]))
        artifacts = store.get("evidence_artifacts", [])
        if saved["artifact"] not in artifacts:
            artifacts.append(saved["artifact"])
            store.set("evidence_artifacts", artifacts)
        return record_failure(store, trial, artifact=saved["artifact"], config=config)
    pending_choices = [(choice, clean) for choice in alternatives.values()]
    seen = set(alternatives)
    while pending_choices:
        choice, local_trial = pending_choices.pop(0)
        from .path_selection import manifest_allowed, restricted_manifest
        if not manifest_allowed(store, choice):
            continue
        # Capture the accepted base before this individual trial can replace it.
        combined = model_trial_manifest(store, catalog, choice) if local_trial else choice
        best = store.get("best_model_measurement")
        incumbent = best["manifest"] if clean and best and combined != best["manifest"] else None
        trial = measure(combined, incumbent)
        from .integration_repairs import affected_candidates
        affected = affected_candidates(trial)
        if len(affected) > 1:
            for candidate in sorted(affected):
                isolated = measure({"entries": [e for e in combined["entries"] if e["candidate_id"] == candidate]})
                tickets.extend(record(isolated))
        else:
            tickets.extend(record(trial))
        retry = restricted_manifest(store, combined)
        if retry is not None and digest(retry) not in seen:
            seen.add(digest(retry))
            pending_choices.append((retry, False))
        measured = trial
        store.set("last_model_measurement", measured)
        result = measured["result"]
        from .integration_repairs import safe_measurement
        if result.get("decision", {}).get("accepted") and safe_measurement(store, measured):
            best = store.get("best_model_measurement")
            if (best is None or measured.get("incumbent") == best["manifest"]
                    or result["decision"]["speedup"] > best["result"]["decision"]["speedup"]):
                store.set("best_model_measurement", measured)
    # A failed integration does not erase kernel wins or assign individual blame.
    best = store.get("best_model_measurement")
    selected = best or measured
    final = read_json(run / "result.json")
    pending = [ticket for ticket in tickets if ticket["status"] == "correction_pending"]
    final.update(manifest=best["manifest"] if best else {"entries": []}, tested_manifest=selected["manifest"],
        model_result=selected["result"],
        kernel_manifest=kernel_manifest(store, catalog), integration_status="accepted" if best else
            "repair_pending" if pending else "rejected", integration_repairs=tickets,
        last_integration=measured)
    if best:
        final["status"] = "accepted"
        store.set("status", "accepted")
    elif pending:
        final["status"] = "integration_repair_pending"
        store.set("status", "integration_repair_pending")
    write_json(run / "result.json", final)
    write_json(run / "manifest.json", final["manifest"])
    from .reporting import write_swarm_report
    write_swarm_report(run, config, profile_dir=profile, store=store)
    return measured
