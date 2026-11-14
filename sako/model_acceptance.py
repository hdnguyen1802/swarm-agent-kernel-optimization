"""Declare target workloads before timing; score complete model measurements."""

import copy
import math
import statistics

from .core import digest, score_cases


def model_trial_key(run_fingerprint, manifest, incumbent=None):
    """The run fingerprint binds the configuration, catalog, hardware, and evaluator."""
    identity = {"run_fingerprint": run_fingerprint, "manifest": manifest}
    if incumbent is not None:
        identity["incumbent"] = incumbent
    return digest(identity)


def incumbent_decision(cases, config):
    """Require paired non-regression and a strict median improvement over the current model."""
    decision = score_model_cases(cases, {**config, "min_speedup": 1.0,
        "model_speed_scope": "all_workloads"}, config["workloads"])
    improved = any(case.get("candidate", {}).get("median_ms", float("inf")) <
                   case.get("baseline", {}).get("median_ms", 0.) for case in cases)
    if not improved:
        decision["accepted"] = False
        decision["reasons"].append("The candidate did not improve the current model in the paired test.")
    return decision


def reusable_trial(cache, key):
    trial = cache.get(key)
    result = (trial or {}).get("result", {})
    return trial if result.get("decision") and not result.get("error") and not result.get("failure_kind") else None


def workload_policy(workloads):
    """State the declared scope without claiming gains for other input shapes."""
    return {"workload_scope": "single_workload" if len(workloads) == 1 else
                              "multiple_workloads" if workloads else "not_recorded",
            "declared_workloads": copy.deepcopy(workloads),
            "generalization": "Performance on other workloads is not established by this result."}


def target_workloads(config, manifest, catalog):
    if config.get("model_speed_scope", "all_workloads") == "all_workloads":
        return list(config["workloads"])
    from .shape_selection import groups, validate_entries
    tasks = {task["id"]: task for task in catalog["tasks"]}
    validate_entries(manifest["entries"], tasks)
    selected = set()
    for entry in manifest["entries"]:
        task = tasks[entry["task_id"]]
        if "signatures" not in entry:
            return list(config["workloads"])
        wanted = {digest(signature) for signature in entry["signatures"]}
        for key, group in groups(task).items():
            if digest(group["signature"]) not in wanted or ("paths" in group and not set(group["paths"]).intersection(entry["paths"])):
                continue
            for index in group["indices"]:
                workload = task["cases"][index].get("workload")
                if workload not in config["workloads"]:
                    raise ValueError("A targeted shape lacks a configured workload. Create a fresh profile.")
                selected.add(digest(workload))
    return [w for w in config["workloads"] if digest(w) in selected]


def _score_model_cases(cases, config, targets):
    expected = [digest(w) for w in config["workloads"]]
    coverage = [digest(c.get("workload")) for c in cases] == expected
    policy = workload_policy(config["workloads"])
    if config.get("model_speed_scope", "all_workloads") == "all_workloads":
        decision = score_cases(cases, config)
        if not coverage:
            decision["accepted"] = False
            decision["reasons"].append("Every configured workload must be measured once in its declared order.")
        return strict_model_gain({**decision, **policy}, config)
    target_keys = {digest(w) for w in targets}
    reasons, ratios, target_ratios = [], [], []
    if not target_keys or not target_keys.issubset(expected):
        reasons.append("At least one configured target workload is required.")
    if not coverage:
        reasons.append("Every configured workload must be measured once in its declared order.")
    for index, case in enumerate(cases):
        targeted = digest(case.get("workload")) in target_keys
        required = config["min_speedup"] if targeted else 1.0
        decision = score_cases([case], {**config, "min_speedup": required})
        reasons.extend(f"Workload {index}: {reason}" for reason in decision["reasons"])
        if decision["speedup"] is not None:
            ratios.append(decision["speedup"])
            if targeted:
                target_ratios.append(decision["speedup"])
    geometric = lambda values: math.exp(statistics.mean(map(math.log, values))) if values else None
    return strict_model_gain({**policy, "accepted": not reasons, "reasons": reasons, "speedup": geometric(ratios),
            "target_speedup": geometric(target_ratios), "speed_acceptance_scope": "targeted_workloads",
            "target_workloads": targets, "target_min_speedup": config["min_speedup"],
            "other_workloads_min_speedup": 1.0,
            "note": ("Complete model timing for the single declared workload. No performance claim for other workloads."
                     if len(expected) == 1 else
                     "Complete model timing for every workload. Gain is required on declared targets; median non-regression elsewhere.")}, config)


def strict_model_gain(decision, config):
    if config["min_speedup"] == 1.0 and decision.get("speedup") is not None and decision["speedup"] <= 1.0:
        decision["accepted"] = False
        decision["reasons"].append("The complete model did not get faster.")
    return decision


def score_model_cases(cases, config, targets, *, reference_role="baseline"):
    from .measurement import decode_only, enabled, policy, assess_pairs, validate_warmup, saved_reference
    if saved_reference(config):
        from .measurement import checked_decode_record, relative_spread
        from .core import summary
        checked, reasons = [], []
        for index, case in enumerate(cases):
            row = dict(case)
            checked.append(row)
            if not case.get("correct"):
                continue
            try:
                record = case["decode_record"]
                sampling = case["sampling"]
                measured = checked_decode_record(record, record["identity"], config)["mean_decode_ms"]
                if (sampling.get("method") != "saved_reference" or sampling["timed_runs"] != 3
                        or case.get("timing_scope") != "cached_decode" or case["candidate"] != measured
                        or case["baseline"] != case["generation_metrics"]["baseline"]["mean_decode_ms"]
                        or sampling["starting_record"] != record["identity"]["starting_record"]):
                    raise ValueError("The saved decode timing differs from its measured samples.")
                raw = case["generation_metrics"][reference_role]["mean_decode_ms"]["samples_ms"]
                if reference_role == "incumbent":
                    previous = case["accepted_record"]
                    expected = {**record["identity"], "manifest": previous["identity"]["manifest"]}
                    if (checked_decode_record(previous, expected, config)["mean_decode_ms"]["samples_ms"] != raw
                            or previous["record_hash"] != sampling["accepted_record"]):
                        raise ValueError("The accepted timing record differs from the comparison.")
                row["baseline"] = summary(raw)
                if len(raw) != 3 or not all(math.isfinite(x) and x > 0 for x in raw):
                    raise ValueError("Three valid reference samples are required.")
                if max(relative_spread(raw), relative_spread(measured["samples_ms"])) > policy(config)["timing_relative_spread"]:
                    reasons.append(f"Workload {index}: decode samples exceeded the timing spread limit.")
            except (KeyError, ValueError, TypeError) as error:
                reasons.append(str(error))
        decision = _score_model_cases(checked, {**config, "trials": 3}, targets)
        decision["reasons"].extend(reasons)
        decision["accepted"] = not decision["reasons"]
        decision["timing_status"] = "accepted" if decision["accepted"] else "uncertain" if reasons else "rejected"
        return decision
    if reference_role == "incumbent":
        config = {**config, "min_speedup": 1., "model_speed_scope": "all_workloads"}
        cases = [{**c, "baseline": c.get("generation_metrics", {}).get("incumbent", {}).get("mean_decode_ms", {})} for c in cases]
    if not enabled(config):
        return _score_model_cases(cases, config, targets)
    rules = policy(config)
    decision = _score_model_cases(cases, {**config, "trials": rules["model_initial_pairs"]}, targets)
    uncertain = False
    assessments = []
    for index, case in enumerate(cases):
        if case.get("correct") is not True:
            continue
        try:
            recorded = case["sampling"]
            if recorded["policy"] != rules or recorded.get("all_collected_samples_used") is not True:
                raise ValueError("The declared sampling policy is missing or changed.")
            warm = recorded["warmup"]
            expected_roles = {"baseline", "candidate"}
            if decode_only(config) and "incumbent" in case.get("generation_metrics", {}):
                expected_roles.add("incumbent")
            if set(warm) != expected_roles:
                raise ValueError("Warmup evidence is missing.")
            stability = {k: validate_warmup(v, config) for k, v in warm.items()}
            stable = stability[reference_role] and stability["candidate"]
            threshold = (config["min_speedup"] if config.get("model_speed_scope", "all_workloads") == "all_workloads"
                         or case["workload"] in targets else 1.0)
            a, b = case["baseline"]["samples_ms"], case["candidate"]["samples_ms"]
            if decode_only(config):
                metrics = case["generation_metrics"]
                if (case.get("timing_scope") != "cached_decode" or recorded.get("metric") != "mean_decode_ms"
                        or a != metrics[reference_role]["mean_decode_ms"]["samples_ms"]
                        or b != metrics["candidate"]["mean_decode_ms"]["samples_ms"]):
                    raise ValueError("The model decision must use raw decode time per token.")
            assessment = assess_pairs(a, b, config, threshold=threshold, warmup_stable=stable)
            claimed = recorded["rounds"][-1]["incumbent"] if reference_role == "incumbent" else recorded["assessment"]
            if assessment != claimed:
                raise ValueError("The sampling assessment differs from the raw samples.")
            first = assess_pairs(a[:5], b[:5], config, threshold=threshold, warmup_stable=stable)
            needs_more = first["needs_more"]
            if "incumbent" in expected_roles:
                # Both paired comparisons see the same candidate run. Either can require the extra five pairs.
                other = "incumbent" if reference_role == "baseline" else "baseline"
                required = 1. if other == "incumbent" else case["speed_requirement"]
                other_samples = case["generation_metrics"][other]["mean_decode_ms"]["samples_ms"]
                if len(other_samples) != len(a):
                    raise ValueError("The paired model measurements have different lengths.")
                needs_more |= assess_pairs(other_samples[:5], b[:5], config, threshold=required,
                    warmup_stable=stability[other] and stability["candidate"])["needs_more"]
            if (len(a) == 5 and needs_more) or (len(a) == 10 and not needs_more):
                raise ValueError("The collected pair count does not follow the declared extra-sample rule.")
            assessments.append(assessment)
            if assessment["state"] != "pass":
                uncertain |= assessment["state"] == "uncertain"
                decision["accepted"] = False
                decision["reasons"].append(f"Workload {index}: adaptive timing {assessment['state']}.")
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            decision["accepted"] = False
            decision["reasons"].append(f"Workload {index}: invalid adaptive timing evidence: {error}")
    decision["timing_status"] = ("accepted" if decision["accepted"] else
        "uncertain" if uncertain and all(c.get("correct") is True for c in cases) else "rejected")
    decision["sampling_assessments"] = assessments
    return decision
