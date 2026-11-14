"""Explain swarm decisions from saved evaluator records and controller events."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path

from .core import read_json, summary, write_json
from .model_acceptance import workload_policy
from .store import Store


def _safe(value):
    """Keep raw files unchanged while making this derived report valid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


def _records(value):
    if isinstance(value, dict):
        return list(value.values())
    return value if isinstance(value, list) else []


def _validation_policy(config, result=None, scope="model"):
    result = result if isinstance(result, dict) else {}
    # Saved evaluator evidence takes precedence over a later caller's config.
    recorded_workloads = (result.get("config") or {}).get("workloads")
    if recorded_workloads is None:
        recorded_workloads = []
        for case in result.get("cases") or []:
            if not isinstance(case, dict):
                continue
            workload = case.get("workload")
            if workload is not None and workload not in recorded_workloads:
                recorded_workloads.append(workload)
        if not recorded_workloads:
            recorded_workloads = config.get("workloads", [])
    accuracy_scope = (result.get("correctness_scope", config.get("correctness_scope", "full_model"))
                      if scope == "model" else "operation")
    accuracy_reference = (result.get("correctness_reference", config.get("correctness_reference", "torch_compile"))
                          if scope == "model" else "eager")
    generation = any(w.get("mode") == "cached_generation" for w in recorded_workloads)
    return {**workload_policy(recorded_workloads),
            "accuracy_scope": accuracy_scope, "accuracy_reference": accuracy_reference,
            "containing_region_policy": result.get("containing_region_policy", config.get("containing_region_policy", "required")),
            "model_speed_scope": result.get("model_speed_scope", config.get("model_speed_scope", "all_workloads")),
            "target_workloads": result.get("target_workloads"),
            "speed_scope": ("cached_decode" if config.get("generation_objective") == "decode" else
                            "complete_cached_generation" if generation else "complete_model") if scope == "model" else "operation",
            "speed_reference": ((result.get("timing_baseline") or {}).get("kind", "torch_compile")
                                if scope == "operation" else
                                result.get("model_reference", config.get("model_reference", "torch_compile"))),
            "final_output_equivalence": result.get("output_equivalence", "not_tested"
                                                   if accuracy_scope != "full_model" else
                                                   "checked" if result.get("cases") else "not_measured")}


def _shared_state(store):
    """Read optional new tables without creating or modifying them."""
    with store.connect() as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        lessons = []
        if "swarm_lessons" in tables:
            lessons = [json.loads(row[0]) for row in db.execute("SELECT payload FROM swarm_lessons ORDER BY created,id")]
            superseded = {}
            if "swarm_lesson_supersedes" in tables:
                for row in db.execute("SELECT previous_id,new_id FROM swarm_lesson_supersedes"):
                    superseded.setdefault(row[0], []).append(row[1])
            lessons = [{**lesson, "superseded_by": sorted(superseded.get(lesson["id"], []))} for lesson in lessons]
        archives = {row[0]: json.loads(row[1]) for row in db.execute(
            "SELECT key,value FROM state WHERE key LIKE 'swarm_archive:%' ORDER BY key")}
        exhausted = {row[0].split(":", 1)[1]: json.loads(row[1]) for row in db.execute(
            "SELECT key,value FROM state WHERE key LIKE 'repair_exhausted:%' ORDER BY key")}
    return lessons, archives, exhausted


def _case_details(case, index, config, *, adaptive_model=False):
    checks = case.get("checks", [])
    checks = checks if isinstance(checks, list) else []
    details = {
        "index": index, "workload": case.get("workload"),
        "shapes": case.get("shapes"), "case_id": case.get("case_id"),
        "correct": case.get("correct"), "inputs_unchanged": case.get("inputs_unchanged"),
        "numerical_limits": {key: config.get(key) for key in ("atol", "rtol")},
        "checks": checks, "timing": {}, "criteria": [],
        "timing_status": case.get("timing_status"), "timing_reason": case.get("timing_reason"),
    }
    if "sampling" in case:
        details["sampling"] = case["sampling"]
    if "generation_metrics" in case:
        details["generation_metrics"] = case["generation_metrics"]
        details["input"] = case["input"]
    if any(check.get("scope") == "replaced_layers" for check in checks if isinstance(check, dict)):
        details["layer_coverage"] = [{
            "seed": check.get("seed"), "passed": check.get("passed"),
            "expected_paths": check.get("expected_paths", []), "observed_paths": check.get("observed_paths", []),
            "missing_paths": check.get("missing_paths", []),
            "failed_layers": [{key: layer.get(key) for key in ("path", "candidate_id", "call_index", "inputs_unchanged", "check", "acceptance_role")}
                              for layer in check.get("layers", []) if layer.get("passed") is not True],
        } for check in checks if isinstance(check, dict) and check.get("scope") == "replaced_layers"]
    criteria = details["criteria"]
    criteria.append({"name": "correctness", "passed": case.get("correct") is True,
                     "reason": "All required numerical checks must pass."})
    if "inputs_unchanged" in case:
        criteria.append({"name": "inputs_unchanged", "passed": case["inputs_unchanged"] is True,
                         "reason": "The kernel must preserve its input tensors."})
    baseline = candidate = None
    for name in ("baseline", "candidate"):
        source = case.get(name, {})
        try:
            samples = source["samples_ms"]
            measured = summary(samples)
            details["timing"][name] = {key: measured[key] for key in ("median_ms", "p95_ms")}
            details["timing"][name]["sample_count"] = len(samples)
            if name == "baseline":
                baseline = measured
            else:
                candidate = measured
        except (KeyError, TypeError, ValueError):
            details["timing"][name] = {"status": "not_measured_or_invalid"}
    valid = baseline is not None and candidate is not None
    if valid:
        saved = case.get("sampling", {}).get("method") == "saved_reference"
        required_samples = 3 if saved else 5 if adaptive_model else config.get("trials", 5)
        enough = min(len(baseline["samples_ms"]), len(candidate["samples_ms"])) >= required_samples
        ratio = baseline["median_ms"] / candidate["median_ms"]
        p95_ratio = candidate["p95_ms"] / baseline["p95_ms"]
        details["timing"].update(speed_ratio=ratio, p95_ratio=p95_ratio)
        criteria.extend([
            {"name": "sample_count", "passed": enough, "required": required_samples},
            {"name": "minimum_speedup", "passed": ratio >= case.get("speed_requirement", config.get("min_speedup", 1.02)),
             "actual": ratio, "required": case.get("speed_requirement", config.get("min_speedup", 1.02))},
            {"name": "maximum_p95_ratio", "passed": p95_ratio <= config.get("max_p95_ratio", 1.05),
             "actual": p95_ratio, "required": config.get("max_p95_ratio", 1.05)},
        ])
    else:
        criteria.append({"name": "timing", "passed": None,
                         "reason": case.get("timing_reason") or ("Timing was skipped after a correctness failure."
                         if case.get("correct") is not True else "Valid timing samples are missing.")})
    if adaptive_model and case.get("sampling", {}).get("method") != "saved_reference":
        assessment = case.get("sampling", {}).get("assessment", {})
        criteria.append({"name": "adaptive_sampling", "passed": assessment.get("state") == "pass",
                         "state": assessment.get("state", "missing")})
    details["derived_all_gates_pass"] = all(item["passed"] is True for item in criteria)
    return details


def evaluation_details(result, config, *, scope="operation"):
    """Explain the exact saved decision. Derived criteria never grant acceptance."""
    if not isinstance(result, dict):
        return {"status": "not_evaluated", "accepted": False, "reasons": ["No evaluator result is saved."],
                "cases": []}
    decision = result.get("decision", {})
    accepted = decision.get("accepted") is True
    if scope == "operation" and result.get("kernel_min_speedup") is not None:
        config = {**config, "min_speedup": result["kernel_min_speedup"]}
    if scope == "operation" and "confirmation_initial_pairs" in result.get("stages", {}):
        config = {**config, "trials": result["stages"]["confirmation_initial_pairs"]}
    from .measurement import enabled as adaptive
    cases = [_case_details(case, index, config, adaptive_model=scope == "model" and adaptive(config))
             for index, case in enumerate(result.get("cases", []))]
    reasons = list(decision.get("reasons") or [])
    if result.get("error"):
        reasons.append(str(result["error"]))
    if not accepted and not reasons:
        reasons.append("The evaluator did not accept this result. No reason was saved.")
    if accepted and not reasons:
        reasons.append("The evaluator reports that every required acceptance gate passed.")
    shape_decisions = result.get("shape_decisions", {})
    shape_passed = any(value.get("accepted") for value in shape_decisions.values())
    return {
        "status": "accepted" if accepted else "uncertain" if decision.get("timing_status") == "uncertain" else
                  "shape_passed" if shape_passed else "rejected", "accepted": accepted,
        "shape_decisions": shape_decisions, "acceptance_scope": result.get("acceptance_scope"),
        "timing_status": decision.get("timing_status"),
        "sampling_assessments": decision.get("sampling_assessments"),
        "speed_ratio": decision.get("speedup"), "target_speed_ratio": decision.get("target_speedup"),
        "speed_acceptance_scope": decision.get("speed_acceptance_scope",
            result.get("model_speed_scope", config.get("model_speed_scope", "all_workloads"))),
        "reasons": reasons, "cases": cases,
        "worker_provenance": result.get("worker_provenance", result.get("provenance")),
        "gate_disagreement": bool(cases) and accepted != (
            all(case["derived_all_gates_pass"] for case in cases)
            and (result.get("incumbent_comparison") or {}).get("decision", {}).get("accepted", True)),
        "incumbent_comparison": result.get("incumbent_comparison"),
        "error": result.get("error"), "scope": result.get("scope"),
        "validation_policy": _validation_policy(config, result, scope),
        "reference_state_intact": result.get("reference_state_intact"),
        "timing_baseline": result.get("timing_baseline"),
        "search_score": result.get("search_score"),
        "supplemental_correctness": result.get("supplemental_correctness"),
        "supplemental_failure_capture": result.get("supplemental_failure_capture"),
        "supplemental_failure_artifacts": result.get("supplemental_failure_artifacts"),
    }


def _concurrency(events, observed_summary):
    """Count overlap only from saved start/end records, never from writer count."""
    active, maximum, complete = set(), 0, 0
    for event in events:
        key = event.get("assignment_id", event.get("hypothesis_id"))
        if key is None:
            continue
        if event.get("kind") in ("writer_started", "writer_start"):
            active.add(str(key))
            maximum = max(maximum, len(active))
        elif event.get("kind") in ("writer_finished", "writer_failed", "writer_end"):
            if str(key) in active:
                active.remove(str(key))
                complete += 1
    recorded = observed_summary.get("max_active_writers") if isinstance(observed_summary, dict) else None
    return {"max_observed_active_writers": maximum if maximum else recorded,
            "basis": "writer start and end events" if maximum else "saved controller summary" if recorded is not None else "not recorded",
            "completed_writer_intervals": complete, "open_writer_intervals": sorted(active),
            "gpu_tests": "serialized"}


def build_swarm_report(run_dir, config, *, profile_dir=None, store=None):
    """Read persisted state. Do not make API calls or dispatch GPU work."""
    run = Path(run_dir).resolve()
    store = store or Store(run / "state.sqlite")
    final = read_json(run / "result.json") if (run / "result.json").exists() else {}
    progress = store.get("task_progress", final.get("task_progress", {}))
    events = _records(store.get("swarm_events", []))
    plans = _records(store.get("swarm_plans", []))
    from .archive import summarize_parent_use
    assignments = _records(store.get("swarm_assignments", {}))
    started_ids = {event.get("assignment_id") for event in events if event.get("kind") == "writer_started"}
    parent_use = {**summarize_parent_use([a for a in assignments if a.get("id") in started_ids]),
                  "basis": "Assignments with a saved writer_started event.",
                  "all_assignments": summarize_parent_use(assignments)}
    rows, emissions = store.candidates(), store.emissions()
    integration_repairs = store.get("integration_repair_tickets", {})
    raw_models = _records(store.get("model_measurements", {}))
    models = []
    for index, model in enumerate(raw_models):
        manifest = model.get("manifest", {"entries": []})
        models.append({"index": index, "generation": model.get("generation"), "manifest": manifest,
                       "evaluation": evaluation_details(model.get("result"), config, scope="model")})
    candidates = []
    for row in rows:
        tested_models = [model for model in models
                         if any(entry.get("candidate_id") == row["id"] for entry in model["manifest"].get("entries", []))]
        model_status = ("accepted_in_model" if any(m["evaluation"]["accepted"] for m in tested_models)
                        else "rejected_alone" if any(len(m["manifest"].get("entries", [])) == 1 for m in tested_models)
                        else "failed_combination_only" if tested_models else "not_tested")
        own_events = [event for event in events if event.get("candidate_id") == row["id"]]
        proposals = [emission for emission in emissions if emission["candidate"] == row["id"]]
        operation = evaluation_details(row["result"], config)
        if tested_models:
            model_reason = "See the saved complete model checks below."
        elif row["result"] is None:
            model_reason = "The operation evaluation has not finished."
        elif not operation["accepted"]:
            model_reason = ("The complete model test was not scheduled because the operation acceptance gates failed: "
                            + "; ".join(operation["reasons"]))
        else:
            model_reason = ("The kernel passed. Full model checks wait for the integrate command."
                            if config.get("kernel_search", {}).get("enabled") else
                            "The operation passed. No complete model result is saved yet.")
        candidates.append({
            "id": row["id"], "task_id": row["task"], "writer_specialty": row["island"],
            "generation": row["generation"], "parent_id": row["parent"],
            "strategy": row["source"].get("strategy"),
            "kernel_backend": row["source"].get("backend", "triton"),
            "source_file": str(run / "candidates" / (row["id"] + ".json")),
            "operation": operation,
            "model_status": model_status, "model_measurements": [model["index"] for model in tested_models],
            "model_test_reason": model_reason,
            "model_note": "A failed combination does not identify this candidate as the cause."
                          if model_status == "failed_combination_only" else None,
            "proposals": proposals, "events": own_events,
            "integration_repair": integration_repairs.get(row["id"]),
        })
    completed = {entry["task_id"]: entry for entry in progress.get("completed", [])}
    profile = Path(profile_dir) if profile_dir else run / "profile"
    catalog = read_json(profile / "tasks.json") if (profile / "tasks.json").exists() else {}
    queue = progress.get("queue", catalog.get("selected_task_ids", []))
    tasks = []
    for priority, task_id in enumerate(queue, 1):
        own = [candidate for candidate in candidates if candidate["task_id"] == task_id]
        own_models = [model for model in models if any(e.get("task_id") == task_id for e in model["manifest"].get("entries", []))]
        tasks.append({"priority": priority, "task_id": task_id,
                      "status": "unvisited" if not own else "completed" if task_id in completed else "visited",
                      "controller_closed": task_id in completed,
                      "completion": completed.get(task_id), "candidates": len(own),
                      "operation_acceptances": sum(item["operation"]["accepted"] for item in own),
                      "model_checks": len(own_models)})
    best = store.get("best_model_measurement")
    lessons, archives, exhausted = _shared_state(store)
    hardware = read_json(run / "hardware.json") if (run / "hardware.json").exists() else catalog.get("hardware")
    stop = store.get("swarm_stop") or store.get("budget_stop") or store.get("budget_decision")
    report = {
        "schema_version": 1, "run": str(run), "status": store.get("status", final.get("status", "not_started")),
        "stop": stop, "budget": store.budget_snapshot(config["agent"]),
        "profile_token_allocation": store.get("profile_token_allocation", {}),
        "task_token_usage": store.task_token_usage(),
        "limits": {key: config.get(key) for key in
                   ("atol", "rtol", "min_speedup", "kernel_search", "max_p95_ratio", "trials", "repeats", "correctness_seeds", "workloads", "measurement")},
        "hardware": hardware, "tasks": tasks, "unvisited_tasks": [t["task_id"] for t in tasks if t["status"] == "unvisited"],
        "validation_policy": _validation_policy(config, best.get("result") if best else
                                                 raw_models[-1].get("result") if raw_models else None),
        "acceptance_claim": ("Replaced-layer accuracy" if config.get("correctness_scope") == "replaced_layers" else
                             "Complete model accuracy") + " and complete-model speed passed on targeted workloads, with median non-regression elsewhere."
                            if config.get("model_speed_scope") == "targeted_workloads" else
                            "Replaced-layer accuracy and whole-model speed passed."
                            if config.get("correctness_scope") == "replaced_layers"
                            else "Complete model accuracy and speed passed.",
        "candidates": candidates, "model_measurements": models,
        "accepted_model_gain": bool(best and best.get("manifest", {}).get("entries")
                                    and best.get("result", {}).get("decision", {}).get("accepted") is True),
        "best_model_measurement": best, "plans": plans, "events": events,
        "kernel_baselines": store.get("kernel_baselines", {}),
        "tuning_seeds": store.get("tuning_seeds", {}),
        "timing_audits": store.get("timing_audits", {}),
        "hierarchy": catalog.get("hierarchy"),
        "completed_regions": store.get("hierarchy_completed_regions", {}),
        "latest_region_profile": store.get("hierarchy_latest_profile"),
        "integration_status": final.get("integration_status"),
        "integration_repair_tickets": integration_repairs,
        "coordinator_context": {task_id: store.get("coordinator_context:" + task_id)
                                for task_id in queue if store.get("coordinator_context:" + task_id) is not None},
        "parent_use": parent_use,
        "hypotheses": store.get("swarm_hypotheses", {}),
        "task_progress": progress,
        "event_counts": dict(Counter(event.get("kind", "unknown") for event in events)),
        "concurrency": _concurrency(events, store.get("swarm_summary", {})),
        "pending_repairs": store.get("pending_repairs", {}),
        "closed_task_repairs": store.get("closed_task_repairs", {}),
        "exhausted_repairs": exhausted,
        "review_complete": final.get("review_complete"), "incomplete_reviews": final.get("incomplete_reviews", []),
        "model_diagnostics": store.get("swarm_diagnostics", {}),
        "lessons": lessons or store.get("swarm_lessons", []), "candidate_archives": archives,
        "summary": store.get("swarm_summary", {}),
        "notes": ["Operation acceptance does not establish a complete model gain.",
                  "Hypotheses and strategy text are agent proposals. Evaluator records supply test decisions.",
                  "Raw timing samples and complete errors remain in the run files and database."],
    }
    # Include archived measurements when rendering an older run.
    archived = store.get("kernel_profiles", store.get("swarm_profiles"))
    if archived:
        report["profiler_results"] = archived
        report["profile_comparisons"] = store.get("profile_comparisons", {})
    if report["validation_policy"]["declared_workloads"]:
        report["limits"]["workloads"] = report["validation_policy"]["declared_workloads"]
    if config.get("generation_objective") == "decode":
        report["decode_metrics"] = store.get("decode_metrics", [])
        report["decode_starting_reference"] = []
        for w in catalog.get("compiled_profile", {}).get("workload_times", []):
            median = w.get("generation_metrics", {}).get("mean_decode_ms", {}).get("median_ms")
            if median:
                report["decode_starting_reference"].append({"ms_per_token": median,
                    "tokens_per_second": 1000 * w["workload"]["batch_size"] / median})
        report["target_validations"] = store.get("decode_target_results", {})
        report["final_combined"] = store.get("decode_combined_result")
        report["continued_search"] = store.get("continued_search")
        report["notes"].append("Timing starts after the prompt and first output token. Each trial starts with an empty KV cache.")
        report["notes"].append("Each target compares up to four versions with the current accepted model. "
                               "The final combined trial repeats the measurements against the original and the base of the last winning target.")
    elif report["validation_policy"]["speed_scope"] == "complete_cached_generation":
        report["acceptance_claim"] = "The candidate passed the token, score, cache and total generation time checks."
        baseline_file = run / "baseline.json"
        baseline = read_json(baseline_file) if baseline_file.exists() else {}
        report["generation_baseline"] = [case["generation_metrics"]["baseline"] for case in baseline.get("cases", [])
                                         if case.get("generation_metrics")]
        if not report["generation_baseline"]:
            workloads = catalog.get("compiled_profile", {}).get("workload_times", [])
            report["generation_baseline"] = [w["generation_metrics"] for w in workloads if w.get("generation_metrics")]
        report["notes"].append("Total generation time determines speed acceptance. Each tested token must match, "
                               "and the full vocabulary scores and cache tensors must pass the numerical checks.")
    elif report["validation_policy"]["accuracy_scope"] == "replaced_layers":
        report["notes"].append("Accuracy checks compare each replaced layer with its eager reference. "
                               "Speed checks compare the complete model with torch.compile. "
                               "The evaluator does not test final-output numerical equivalence.")
    if not report["accepted_model_gain"]:
        report["acceptance_claim"] = "No candidate passed the required accuracy and complete model speed gates."
    elif report["validation_policy"]["workload_scope"] == "single_workload":
        report["acceptance_claim"] = ("Required accuracy and complete model speed passed for the single declared workload only: "
                                      + json.dumps(report["validation_policy"]["declared_workloads"]) + ".")
    if config.get("generation_objective") == "decode":
        report["acceptance_claim"] = {"accepted": "The selected model passed correctness and both decode speed checks.",
            "retained_previous": "The previous accepted model remains selected.",
            "retained_original": "The original compiled model remains selected."}.get(report["status"], "Final decode validation is pending.")
    report["notes"].append(report["validation_policy"]["generalization"])
    return _safe(report)


def _cell(value):
    return str(value if value is not None else "not recorded").replace("|", "\\|").replace("\n", " ")


def _json(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


def render_swarm_report(report):
    budget = report["budget"]
    lines = ["# Swarm search report", "", f"Status: {report['status']}",
             f"Accepted complete model gain: {report['accepted_model_gain']}",
             f"Acceptance claim: {report['acceptance_claim']}",
             f"Accuracy: {report['validation_policy']['accuracy_scope']} versus {report['validation_policy']['accuracy_reference']}.",
             f"Speed: {report['validation_policy']['speed_scope']} versus {report['validation_policy']['speed_reference']}.",
             f"Workload scope: {report['validation_policy']['workload_scope']}. "
             f"Declared workloads: {json.dumps(report['validation_policy']['declared_workloads'])}.",
             f"Final-output numerical equivalence: {report['validation_policy']['final_output_equivalence']}.",
             f"Tokens: {budget['tokens']} / {budget['max_total_tokens']}. Remaining: {budget['remaining_tokens']}.",
             f"Requests: {budget['generation_calls']} generation and {budget['auxiliary_calls']} metadata.",
             f"Unknown reserved tokens: {budget['unresolved_tokens']}.", "",
             "## Stop decision", "", "```json", _json(report["stop"]), "```", "",
             "## Task coverage", "", "| Priority | Task | Status | Candidates | Operation passes | Model checks |",
             "|---:|---|---|---:|---:|---:|"]
    if report.get("decode_starting_reference"):
        reference = report["decode_starting_reference"][0]
        lines[2:2] = [f"Original compiled decode used {reference['ms_per_token']:.3f} ms/token "
                      f"({reference['tokens_per_second']:.2f} tokens/s).", ""]
    if report.get("decode_metrics"):
        table = ["## Cached decode", ""]
        for r in report["decode_metrics"]:
            eager = r.get("pytorch_ms_per_token")
            batch = r.get("workload", {}).get("batch_size", 1)
            number = lambda value, digits=3: f"{value:.{digits}f}" if value is not None else "—"
            ratio = lambda value: number(value, 4) + "×" if value is not None else "—"
            table.extend(["| Decode | PyTorch | torch.compile | Our optimization |", "|---|---:|---:|---:|",
                          f"| ms/token | {number(eager)} | {number(r['original_ms_per_token'])} | {number(r['ms_per_token'])} |",
                          f"| tokens/s | {number(1000 * batch / eager if eager else None, 2)} | {number(1000 * batch / r['original_ms_per_token'], 2)} | {number(r['tokens_per_second'], 2)} |",
                          f"| Speedup vs PyTorch | {ratio(1 if eager else None)} | {ratio(eager / r['original_ms_per_token'] if eager else None)} | {ratio(eager / r['ms_per_token'] if eager else None)} |", "",
                          f"Validated decode speedup vs torch.compile: {r['vs_original_compile']:.4f}×. "
                          f"Further gain vs the saved accepted model: {r['vs_previous_accepted']:.4f}×.", ""])
        lines[2:2] = [*table, ""]
    if report.get("target_validations"):
        table = ["## Complete model tests by target", "", "| Target | Candidate | GPU | Timed runs per candidate | vs original | vs previous | Result |",
                 "|---|---|---:|---:|---:|---:|---|"]
        for stage in report["target_validations"].values():
            if not stage["trials"]:
                table.append(f"| {stage['task_id']} | — | — | 0 | — | — | No eligible kernel |")
            for t in stage["trials"]:
                ratio = lambda n: f"{n:.4f}×" if isinstance(n, (int, float)) else "—"
                label = "Kept" if t["candidate_id"] == stage["winner"] else "Passed" if t["accepted"] else "Rejected"
                table.append(f"| {stage['task_id']} | {t['candidate_id'][:10]} | {t['gpu']} | {t.get('timed_runs') or t['pairs']} "
                             f"| {ratio(t['original_ratio'])} | {ratio(t['previous_ratio'])} | {label} |")
            for f in stage.get("skipped_after_pass", []):
                table.append(f"| {stage['task_id']} | {f['candidate_id'][:10]} | — | 0 | — | — | Rank {f['rank']} saved after first pair passed |")
        pos = lines.index("## Task coverage")
        lines[pos:pos] = [*table, ""]
    if report["validation_policy"]["speed_scope"] == "complete_cached_generation":
        rows = [("Baseline", "torch.compile", m) for m in report.get("generation_baseline", [])]
        for model in report["model_measurements"]:
            for case in model["evaluation"]["cases"]:
                for role, metrics in case.get("generation_metrics", {}).items():
                    rows.append((f"Model check {model['index']}", role, metrics))
        if rows:
            table = ["## Cached generation", "",
                     "| Measurement | Model | First token (ms) | Mean decode (ms/token) | Total (ms) |",
                     "|---|---|---:|---:|---:|"]
            for name, role, metrics in rows:
                values = [metrics[key]["median_ms"] for key in ("first_token_ms", "mean_decode_ms", "total_ms")]
                table.append(f"| {name} | {role} | " + " | ".join(f"{v:.3f}" for v in values) + " |")
            pos = lines.index("## Task coverage")
            lines[pos:pos] = [*table, ""]
    for task in report["tasks"]:
        lines.append("| " + " | ".join(_cell(task[key]) for key in
                     ("priority", "task_id", "status", "candidates", "operation_acceptances", "model_checks")) + " |")
    lines.extend(["", "## Parallel work", "", "```json", _json(report["concurrency"]), "```", "",
                  "## Parent use", "", "Counts include assignments with a saved writer start.", "",
                  "| Fresh starts | Ordinary archive refinements | Repairs with a parent | Corrections without a parent |",
                  "|---:|---:|---:|---:|",
                  "| " + " | ".join(str(report["parent_use"][key]) for key in (
                      "fresh_starts", "archive_refinements", "repairs", "corrections_without_parent")) + " |", "",
                  "```json", _json(report["parent_use"]), "```", "",
                  "## API use", "", "| Role | Requests | Accounted tokens | Unknown requests |",
                  "|---|---:|---:|---:|"])
    for role, counts in budget["per_role"].items():
        lines.append(f"| {_cell(role)} | {counts['calls']} | {counts['tokens']} | {counts['unknown']} |")
    lines.extend(["", "## Hypothesis status", "",
                  "A hypothesis has one initial attempt and up to two correction attempts. Its closure does not close the task.", "",
                  "| Hypothesis | Task | Status | Attempts | Corrections left | Last outcome | Stagnation effect |",
                  "|---|---|---|---:|---:|---|---|"])
    for hypothesis in _records(report["hypotheses"]):
        outcome = hypothesis.get("last_outcome") or {}
        lines.append("| " + " | ".join(_cell(value) for value in (
            hypothesis.get("id"), hypothesis.get("task_id"), hypothesis.get("status"),
            hypothesis.get("attempt_count"), hypothesis.get("corrections_remaining"),
            outcome.get("kind"), outcome.get("stagnation_action"))) + " |")
    lines.extend(["", "## Candidate decisions", "",
                  "| Candidate | Task | Specialist | Operation decision | Speed ratio | Model status |",
                  "|---|---|---|---|---:|---|"])
    for candidate in report["candidates"]:
        lines.append(f"| {candidate['id']} | {candidate['task_id']} | {_cell(candidate['writer_specialty'])} | "
                     f"{candidate['operation']['status']} | {_cell(candidate['operation'].get('speed_ratio'))} | {candidate['model_status']} |")
    for candidate in report["candidates"]:
        lines.extend(["", f"### Candidate {candidate['id']}", "", f"Parent: {candidate['parent_id'] or 'fresh start'}.",
                      f"Agent strategy: {_cell(candidate['strategy'])}",
                      f"Triton source: [candidate file](<{candidate['source_file']}>).", "",
                      "Operation decision: " + "; ".join(candidate["operation"]["reasons"]),
                      f"Model status: {candidate['model_status']}.", candidate["model_test_reason"]])
        if candidate["model_note"]:
            lines.append(candidate["model_note"])
        if candidate["operation"].get("shape_decisions"):
            lines.extend(["", "Per-shape speed decisions (all accuracy cases must still pass):", "```json",
                          _json(candidate["operation"]["shape_decisions"]), "```"])
        for case in candidate["operation"]["cases"]:
            lines.extend(["", f"Case {case['index']}: shapes={case['shapes']}, workload={case['workload']}.",
                          "```json", _json({key: case[key] for key in ("correct", "checks", "timing", "criteria")}), "```"])
        if candidate["proposals"]:
            lines.extend(["", "Assignments, hypotheses, and parent choices:", "```json", _json(candidate["proposals"]), "```"])
    lines.extend(["", "## Complete model checks", ""])
    for model in report["model_measurements"]:
        lines.extend([f"### Model check {model['index']}", "", "```json", _json(model), "```", ""])
    for title, key in (("Timing controls", "timing_audits"), ("Autotuned seeds", "tuning_seeds"),
                       ("Coordinator plans", "plans"), ("Recorded events", "events"),
                       ("Hypothesis attempts and closure reasons", "hypotheses"),
                       ("Profiler results", "profiler_results"), ("Shared lessons", "lessons"),
                       ("Model diagnostics", "model_diagnostics"),
                       ("Integration repair evidence and remaining attempts", "integration_repair_tickets"),
                       ("Coordinator context sizes and full evidence files", "coordinator_context"),
                       ("Candidate archives", "candidate_archives"),
                       ("Pending repairs", "pending_repairs"), ("Closed repairs", "closed_task_repairs"),
                       ("Repairs at the attempt limit", "exhausted_repairs")):
        if key in report:
            lines.extend(["", f"## {title}", "", "```json", _json(report[key]), "```"])
    lines.extend(["", "## Interpretation limits", "", *["- " + note for note in report["notes"]], ""])
    return "\n".join(lines)


def write_swarm_report(run_dir, config, *, profile_dir=None, store=None):
    report = build_swarm_report(run_dir, config, profile_dir=profile_dir, store=store)
    run = Path(run_dir)
    write_json(run / "report.json", report)
    (run / "report.md").write_text(render_swarm_report(report), encoding="utf-8")
    return report
