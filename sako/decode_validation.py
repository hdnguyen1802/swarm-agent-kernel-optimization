"""Select a complete model at each target boundary and repeat the final test independently."""

from concurrent.futures import ThreadPoolExecutor

from .core import candidate_id, digest, read_json, write_json
from .decode_devices import decode_devices
from .decode_search import model_metrics, target_finalists
from .operation_names import catalog_names, operation_name
from .store import operation_correct
from .transport import WorkerJobError


def measure_version(run, store, device, stage, proposal, incumbent, candidate=None, *, correctness_only=False):
    """The stage and device bind the saved job. A final replay gets a new timing sample."""
    relative = lambda path: path.relative_to(device.worker.root).as_posix()
    key = digest([store.get("fingerprint"), device.fingerprint, stage, proposal, incumbent,
                  *([{"correctness_only": True}] if correctness_only else [])])
    folder = run / "integrations"
    output = folder / (key + ".result.json")
    if output.exists():
        return read_json(output)
    sources = {e["candidate_id"]: e["task_id"] for e in [*proposal["entries"], *(incumbent or {}).get("entries", [])]}
    for identity, task in sources.items():
        file = run / "candidates" / (identity + ".json")
        value = read_json(file)
        if candidate_id(value) != identity or value["task_id"] != task:
            raise ValueError("The model references a different candidate source.")
        device.worker.put(file, relative(file))
    options = {"correctness_only": True} if correctness_only else {}
    for field, value in (("manifest", proposal), ("incumbent_manifest", incumbent)):
        if value is not None:
            file = folder / (key + "." + field + ".json")
            write_json(file, value)
            device.worker.put(file, relative(file))
            options[field] = relative(file)
    try:
        result = device.worker.run("model", device.directory, profile_dir=relative(device.profile),
            candidate_root=relative(run / "candidates"), feedback_trace_dir=relative(device.directory / "model_checks" / key),
            **options)
        device.verify(result)
    except WorkerJobError as error:
        result = {"scope": "cached_decode", "cases": [], "error": str(error), "worker_report": error.payload,
                  "decision": {"accepted": False, "speedup": None, "reasons": [str(error)]}}
    trial = {"key": key, "stage": stage, "gpu": device.gpu, "hardware_fingerprint": device.fingerprint,
             "manifest": proposal, "incumbent": incumbent, "candidate_id": candidate, "result": result}
    write_json(output, trial)
    return trial


def passed(trial):
    result = trial["result"]
    previous = trial.get("incumbent")
    return (result.get("scope") == "cached_decode" and result["decision"]["accepted"] and operation_correct(result)
            and result.get("reference_state_intact") is not False and (not previous or
                 (result.get("incumbent_comparison") or {}).get("decision", {}).get("accepted") is True))


def trial_summary(trial):
    result = trial["result"]
    comparison = (result.get("incumbent_comparison") or {}).get("decision") or {}
    cases = result.get("cases", [])
    case = cases[0] if cases else {}
    return {"key": trial["key"], "candidate_id": trial.get("candidate_id"), "gpu": trial["gpu"],
            "correct": operation_correct(result), "accepted": bool(passed(trial)),
            "original_ratio": result["decision"].get("speedup"),
            "previous_ratio": comparison.get("speedup") if trial["incumbent"] else result["decision"].get("speedup"),
            "ms_per_token": (case.get("candidate") or {}).get("median_ms"),
            "pairs": (case.get("sampling") or {}).get("assessment", {}).get("pairs", 0),
            "timed_runs": (case.get("sampling") or {}).get("timed_runs", 0),
            "reasons": list(dict.fromkeys([*result["decision"].get("reasons", []), *comparison.get("reasons", [])]))}


def target_boundary(config, run, profile, catalog, store, worker, task_id, visit, reason, *, repair_ids=None):
    """Finish the target before new writers can start on a different target."""
    key = f"{task_id}/{visit}" + ("/repair" if repair_ids is not None else "")
    if store.get("decode_target_boundary") != {"task": task_id, "visit": visit, "reason": reason}:
        store.set("decode_target_boundary", {"task": task_id, "visit": visit, "reason": reason})
    plans = store.get("decode_target_plans", {})
    history = store.get("decode_target_results", {})
    if key in history:
        return history[key]
    if key not in plans:
        accepted = store.get("best_model_measurement") or store.get("decode_previous_model") or {}
        base = accepted.get("manifest", {"entries": []})
        plans[key] = {"task_id": task_id, "visit": visit, "incumbent": base, "reason": reason,
                      "finalists": target_finalists(store, catalog, task_id, base,
                          **({"candidate_ids": repair_ids} if repair_ids is not None else {}))}
        store.set("decode_target_plans", plans)
    plan = plans[key]
    incumbent = plan["incumbent"] if plan["incumbent"]["entries"] else None
    finalists = plan["finalists"]
    trials = []
    measurements = store.get("model_measurements", {})
    if finalists:
        devices = decode_devices(worker, run, profile, catalog)
        name = operation_name(task_id, catalog_names(catalog))
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            for start in range(0, len(finalists), 2):
                group = finalists[start:start+2]
                for offset in range(0, len(group), len(devices)):
                    batch = list(zip(group[offset:], devices))
                    active = [{"rank": item["rank"], "gpu": d.gpu, "candidate_id": item["candidate_id"]} for item,d in batch]
                    store.set("decode_final_progress", {"phase": "target", "target": task_id, "total": len(finalists), "active": active})
                    print(f"Complete model for {name}: " + ", ".join(f"#{r['rank']} on GPU {r['gpu']}" for r in active), flush=True)
                    jobs = [pool.submit(measure_version, run, store, device, "target/" + key, item["manifest"], incumbent,
                                        item["candidate_id"]) for item, device in batch]
                    for job in jobs:
                        trial = job.result()
                        trials.append(trial)
                        measurements[trial["key"]] = trial
                    store.set("model_measurements", measurements)
                if any(passed(t) for t in trials):
                    break
    # Normalize each card's time against its copy of the same accepted model.
    def decode_gain(trial):
        result = trial["result"]
        reference = result["incumbent_comparison"]["decision"] if incumbent else result["decision"]
        return reference["speedup"]
    winner = max((t for t in trials if passed(t)), key=decode_gain, default=None)
    outcome = {"task_id": task_id, "visit": visit, "reason": reason, "trials": [trial_summary(t) for t in trials],
               "skipped_after_pass": [{"rank": f["rank"], "candidate_id": f["candidate_id"]} for f in finalists[len(trials):]],
               "winner": winner["candidate_id"] if winner else None, "winner_measurement": winner["key"] if winner else None,
               "manifest": winner["manifest"] if winner else plan["incumbent"]}
    history[key] = outcome
    updates = {"decode_target_results": history, "decode_final_progress": None}
    if trials:
        updates["last_model_measurement"] = trials[-1]
    if winner:
        updates.update(best_model_measurement=winner, decode_metrics=model_metrics(winner, trials, {}, catalog))
    store.set_many(updates)
    write_json(run / "manifest.json", outcome["manifest"])
    return outcome


def final_combination(run, profile, catalog, store, worker):
    """Always run the selected combination again, after the last target."""
    if store.get("status") in {"running", "request_failure", "infrastructure_failure", "stopped"}:
        raise ValueError("Finish the search and its target checks before the final combined test.")
    best = store.get("best_model_measurement") or store.get("decode_previous_model") or {}
    plan = store.get("decode_combined_plan")
    if plan is None:
        plan = {"manifest": best.get("manifest", {"entries": []}), "incumbent": best.get("incumbent") or None}
        store.set("decode_combined_plan", plan)
    selected, previous = plan["manifest"], plan["incumbent"]
    device = decode_devices(worker, run, profile, catalog)[0]
    store.set("decode_final_progress", {"phase": "combined", "active": [{"rank": 1, "gpu": device.gpu}], "total": 1})
    trial = measure_version(run, store, device, "final_combined", selected, previous)
    all_trials = store.get("model_measurements", {})
    all_trials[trial["key"]] = trial
    confirmed = bool(selected["entries"] and passed(trial))
    summary = trial_summary(trial)
    base = {"manifest": previous or {"entries": []}, "decode_metrics": []}
    fallback = next((t for t in reversed(list(all_trials.values())) if t["manifest"] == base["manifest"] and passed(t)), None)
    if fallback:
        base["decode_metrics"] = model_metrics(fallback, [fallback], {}, catalog)
    metrics = model_metrics(trial if confirmed else None, [trial], base if previous else {}, catalog)
    retained = selected if confirmed else base["manifest"]
    state = "accepted" if confirmed else "retained_previous" if retained["entries"] else "retained_original"
    history = store.get("decode_target_results", {})
    final = read_json(run / "result.json")
    final.update(status=state, integration_status=state, manifest=retained, tested_manifest=selected,
                 model_result=trial["result"], target_validations=history, final_combined=summary, decode_metrics=metrics)
    store.set_many({"status": state, "model_measurements": all_trials, "last_model_measurement": trial,
        "best_model_measurement": trial if confirmed else fallback, "decode_final_progress": None,
        "decode_metrics": metrics, "decode_combined_result": summary})
    write_json(run / "manifest.json", retained)
    write_json(run / "result.json", final)
    return trial
