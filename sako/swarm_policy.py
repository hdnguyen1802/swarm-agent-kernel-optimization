"""Measured admission and validated experiment proposals for the swarm."""

import math
import statistics

from .core import digest

SPECIALTIES = {
    "memory": "Reduce memory transactions and improve coalescing, cache reuse, or vector access.",
    "compute": "Improve arithmetic, reductions, and supported Tensor Core use without changing precision.",
    "scheduling": "Improve tile sizes, block sizes, work distribution, and synchronization.",
    "fusion": "Reduce launches and temporary tensors while preserving all rounding boundaries.",
    "registers": "Reduce register pressure and spills without creating extra traffic or numerical errors.",
}


def coordinator_response_schema(actions=("continue", "move_on", "wait")):
    """Describe the plan at the provider boundary. The controller checks its evidence and policy."""
    from .trial_evidence import OPERATORS

    def array(items):
        return {"type": "ARRAY", "items": items}

    def record(properties, required):
        return {"type": "OBJECT", "properties": properties, "required": required}

    string = {"type": "STRING"}
    integer = {"type": "INTEGER"}
    number = {"type": "NUMBER"}
    evidence = array(string)
    def checks(fields):
        return array(record({"field": {"type": "STRING", "description": fields},
                             "op": {"type": "STRING", "enum": list(OPERATORS)}, "value": number},
                            ["field", "op", "value"]))

    implementation = checks("Use launch.count, launch.I.blocks, launch.I.threads, launch.I.grid.x, "
                            "launch.I.block.x (axes x, y, z), tune.I.J.OPTION, or source.calls.QUALIFIED_CALL. "
                            "Replace I and J with zero-based integers. Example: launch.0.blocks.")
    verification = record({"fixture_index": integer, "implementation": implementation},
                          ["fixture_index", "implementation"])
    hypothesis = record({
        "specialty": {"type": "STRING", "enum": list(SPECIALTIES)}, "parent_id": {"type": "STRING", "nullable": True},
        "objective": {"type": "STRING", "description": "Measured problem and desired outcome. The writer chooses the algorithm."},
        "algorithm": {"type": "STRING", "description": "Optional suggestion for the writer."},
        "experiment_role": {"type": "STRING", "description": "Choose a role from writer_experiments when supplied."},
        "claim": string, "change": string, "expected_metric": string, "evidence": evidence,
        "fresh_start_reason": string, "expected_operation_speedup": number, "verification": verification,
        "merge": record({"parent_ids": array(string), "shape_ids": array(string)}, ["parent_ids", "shape_ids"])},
        ["specialty", "parent_id", "claim", "change", "expected_metric", "evidence"])
    return record({"action": {"type": "STRING", "enum": list(actions)},
                   "reason": string, "evidence": evidence, "hypotheses": array(hypothesis),
                   "archive_skip": record({"reason": string, "evidence": evidence}, ["reason", "evidence"]),
                   "merge_skip": record({"reason": string, "evidence": evidence}, ["reason", "evidence"]),
                   "replan_after": {"type": "STRING", "enum": ["all_results", "next_result"]}},
                  ["action", "reason", "evidence", "hypotheses"])


def profile_shares(catalog, selected, tokens, priorities=None):
    """Apportion integer tokens among eligible GPU costs. Count nested work once."""
    tasks = {task["id"]: task for task in catalog["tasks"]}
    priorities = priorities or {}
    weights = {key: max(0., priorities.get(key, tasks[key]).get("priority_ms", 0.)) for key in selected}
    for region in catalog.get("hierarchy", {}).get("regions", []):
        members = [key for key in selected if key in region["subsection_ids"]]
        if not set(members).intersection(region.get("alternative_task_ids", [])):
            continue
        total = sum(weights[key] for key in members)
        cost = priorities.get(region["id"], region).get("priority_ms", 0.)
        if total > cost > 0:
            for key in members:
                weights[key] *= cost / total
    total = sum(weights.values())
    if not total:
        return {key: 0 for key in selected}
    exact = {key: tokens * value / total for key, value in weights.items()}
    result = {key: int(value) for key, value in exact.items()}
    order = sorted(selected, key=lambda key: (-(exact[key] - result[key]), key))
    for key in order[:tokens - sum(result.values())]:
        result[key] += 1
    return result


def estimate_costs(limits, calls):
    samples = {"writer": [], "coordinator": []}
    for raw in calls:
        row = dict(raw)
        role = row.get("role", "")
        group = "writer" if role.startswith("writer_") and ":" not in role else role
        if (group in samples and row.get("status") == "complete"
                and type(row.get("actual")) is int and row["actual"] > 0):
            samples[group].append(row["actual"])
    return {role: math.ceil(statistics.median(values[-12:]) * 1.15) if values
            else limits["max_output_tokens"] * 2 for role, values in samples.items()}


def admit_experiments(limits, committed, costs, slots, *, task_remaining=None):
    """Reserve the next measured experiment and one coordinator decision."""
    maximum = limits.get("writers", 5)
    if type(maximum) is not int or not 1 <= maximum <= 5:
        raise ValueError("Use one to five specialist writers.")
    if type(slots) is not int or slots < 0 or type(committed) is not int or committed < 0:
        raise ValueError("Use nonnegative integer slots and committed tokens.")
    if task_remaining is not None and type(task_remaining) is not int:
        raise ValueError("The remaining task budget must be an integer.")
    if any(type(costs.get(role)) is not int or costs[role] <= 0 for role in ("writer", "coordinator")):
        raise ValueError("Cost estimates must be positive token counts.")
    if type(limits.get("max_total_tokens")) is not int or limits["max_total_tokens"] <= 0:
        raise ValueError("The token cap must be a positive integer.")
    remaining = max(0, limits["max_total_tokens"] - committed)
    available = min(remaining, max(0, task_remaining)) if task_remaining is not None else remaining
    count = min(slots, maximum, max(0, (available - costs["coordinator"]) // costs["writer"]))
    return {"admit": count > 0, "writer_slots": count, "remaining_tokens": remaining,
            "task_remaining_tokens": task_remaining, "required_tokens": costs["writer"] + costs["coordinator"],
            "review_reserve_tokens": costs["coordinator"],
            "reason": "next_experiments" if count else "task_coverage_reserve" if available < remaining else "token_budget"}


def validate_plan(plan, *, task_id, known_evidence, parent_ids, repair_ids=(), seen=(), max_writers=5,
                  allow_wait=False, archive_refinement_inflight=False, require_archive_trial=False,
                  model_costs=(), fixture_count=None, merge_pairs=(), require_merge_trial=False, portfolio=()):
    """Reject unsupported evidence and duplicate experiments before paid writers run."""
    if type(max_writers) is not int or not 1 <= max_writers <= 5:
        raise ValueError("Use one to five specialist writers.")
    if type(archive_refinement_inflight) is not bool:
        raise ValueError("Archive refinement activity must be true or false.")
    if not isinstance(plan, dict) or plan.get("action") not in {"continue", "move_on", "wait"}:
        raise ValueError("The coordinator must return action continue, move_on, or wait.")
    if plan["action"] == "wait" and not allow_wait:
        raise ValueError("A wait decision requires work in flight.")
    replan = plan.get("replan_after", "all_results")
    if replan not in ("all_results", "next_result"):
        raise ValueError("Select all_results or next_result for replan_after.")
    reason = plan.get("reason")
    evidence = plan.get("evidence", [])
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 4000:
        raise ValueError("The coordinator must give a bounded reason.")
    if (not isinstance(evidence, list) or not evidence
            or any(not isinstance(e, str) or e not in known_evidence for e in evidence)):
        raise ValueError("The coordinator must cite known evidence.")
    raw = plan.get("hypotheses", [])
    if not isinstance(raw, list) or len(raw) > 20:
        raise ValueError("Return at most twenty queued hypotheses.")
    refinement_ids = set(parent_ids) - set(repair_ids)
    hypotheses, rejected = [], []
    fresh_algorithms = set()
    roles_used = set()
    used = set(seen)
    for value in raw:
        if not isinstance(value, dict):
            raise ValueError("A hypothesis must be an object.")
        h = {key: value.get(key) for key in ("specialty", "parent_id", "claim", "change", "expected_metric", "evidence")}
        if portfolio and h["parent_id"] not in repair_ids:
            role = value.get("experiment_role")
            if role is None:
                role = next((o["role"] for o in portfolio if o["parent_id"] == h["parent_id"] and
                             o["role"] not in roles_used and o.get("merge") == value.get("merge")), None)
            option = next((o for o in portfolio if o["role"] == role), None)
            if option is None or role in roles_used or h["parent_id"] != option["parent_id"]:
                raise ValueError("Choose a distinct offered experiment_role and its listed parent_id.")
            if option.get("merge") != value.get("merge"):
                raise ValueError("Use the supplied merge parents and shapes for this experiment.")
            h["experiment_role"] = role
            roles_used.add(role)
        if "objective" in value:
            if not isinstance(value["objective"], str) or not value["objective"].strip() or len(value["objective"]) > 2000:
                raise ValueError("Give a bounded experiment objective.")
            h["objective"] = value["objective"].strip()
        if "verification" in value:
            from .trial_evidence import verification_spec
            h["verification"] = verification_spec(value["verification"])
            if fixture_count is not None and h["verification"]["fixture_index"] >= fixture_count:
                raise ValueError("The selected verification fixture does not exist.")
        if "expected_operation_speedup" in value:
            speed = value["expected_operation_speedup"]
            if type(speed) not in (int, float) or not math.isfinite(speed) or speed <= 0:
                raise ValueError("Use a finite positive expected operation speed ratio.")
            h["expected_operation_speedup"] = speed
            if model_costs and not any("phase_ms" in cost for cost in model_costs):
                estimates = [row["region_ms"] * (1 - 1 / speed) for row in model_costs]
                if not all(math.isfinite(n) for n in estimates):
                    raise ValueError("The expected speed ratio gives an invalid model time.")
                h["estimated_model_saving_ms"] = statistics.mean(estimates)
        algorithm_id = None
        if "algorithm" in value:
            algorithm = value["algorithm"]
            if not isinstance(algorithm, str) or not algorithm.strip() or len(algorithm) > 240:
                raise ValueError("Describe the algorithm mechanism in at most 240 characters.")
            h["algorithm"] = algorithm.strip()
            if h["parent_id"] is None and "objective" not in h:
                algorithm_id = " ".join(algorithm.split()).casefold()
        if h["specialty"] not in SPECIALTIES:
            raise ValueError("Unknown writer specialty.")
        for field in ("claim", "change", "expected_metric"):
            if not isinstance(h[field], str) or not h[field].strip() or len(h[field]) > 2000:
                raise ValueError(f"A hypothesis needs a bounded {field}.")
        if h["parent_id"] is not None and (not isinstance(h["parent_id"], str)
                                            or h["parent_id"] not in {*parent_ids, *repair_ids}):
            raise ValueError("The proposed parent is not eligible for this task.")
        if (not isinstance(h["evidence"], list) or not h["evidence"]
                or any(not isinstance(e, str) or e not in known_evidence for e in h["evidence"])):
            raise ValueError("A hypothesis cites unknown evidence.")
        if h["parent_id"] in refinement_ids and h["parent_id"] not in h["evidence"]:
            raise ValueError("An archive refinement must cite its selected parent as evidence.")
        if value.get("merge") is not None:
            from .recombination import validate_merge
            h["merge"] = validate_merge(value["merge"], h["parent_id"], merge_pairs)
            if not set(h["merge"]["parent_ids"]).issubset(refinement_ids):
                raise ValueError("Merge parents must be ordinary archive parents, not failed repair candidates.")
            if not set(h["merge"]["parent_ids"]).issubset(h["evidence"]):
                raise ValueError("Cite both merge parents as evidence.")
        if h["parent_id"] is None and refinement_ids:
            fresh_reason = value.get("fresh_start_reason")
            if not isinstance(fresh_reason, str) or not fresh_reason.strip() or len(fresh_reason) > 2000:
                raise ValueError("Explain fresh_start_reason when archive parents are available.")
            if not refinement_ids.intersection(h["evidence"]):
                raise ValueError("A fresh start must cite an archive parent that it considered.")
            h["fresh_start_reason"] = fresh_reason
        signature = digest({"task": task_id, "parent": h["merge"]["parent_ids"] if h.get("merge") else h["parent_id"],
                            **({"merge": h["merge"]} if "merge" in h else {}),
                            **({"specialty": h["specialty"]} if "objective" in h else {}),
                            "change": " ".join(h["change"].split()).casefold(),
                            **({"verification": h["verification"]} if "verification" in h else {})})[:24]
        if signature in used:
            rejected.append({"id": signature, "reason": "duplicate_hypothesis"})
            continue
        if algorithm_id in fresh_algorithms:
            rejected.append({"id": signature, "reason": "duplicate_fresh_algorithm"})
            continue
        if algorithm_id is not None:
            fresh_algorithms.add(algorithm_id)
        used.add(signature)
        hypotheses.append({**h, "id": signature})
    if plan["action"] == "continue" and not hypotheses:
        raise ValueError("Continue requires at least one new, distinct hypothesis.")
    if plan["action"] in {"move_on", "wait"} and raw:
        raise ValueError("A move_on or wait plan cannot dispatch hypotheses.")
    merge_skip = plan.get("merge_skip")
    if merge_skip is not None:
        if (not isinstance(merge_skip, dict) or not isinstance(merge_skip.get("reason"), str)
                or not merge_skip["reason"].strip() or len(merge_skip["reason"]) > 4000
                or not isinstance(merge_skip.get("evidence"), list)
                or any(not isinstance(e, str) or e not in known_evidence for e in merge_skip["evidence"])
                or not any(set(p["parent_ids"]).issubset(merge_skip["evidence"]) for p in merge_pairs
                           if not require_merge_trial or not p.get("attempted"))):
            raise ValueError("merge_skip must explain the conflict and cite both measured parents of an available pair.")
        merge_skip = {"reason": merge_skip["reason"], "evidence": list(merge_skip["evidence"])}
    new_merges = [h for h in hypotheses if h.get("merge") and any(
        not p.get("attempted") and set(p["parent_ids"]) == set(h["merge"]["parent_ids"]) for p in merge_pairs)]
    if require_merge_trial and plan["action"] != "wait" and not new_merges and not merge_skip:
        raise ValueError("Propose a merge of a supplied pair, or give merge_skip with measured evidence for both parents.")
    refinements = [h for h in hypotheses if h["parent_id"] in refinement_ids]
    required = bool(refinement_ids) and not archive_refinement_inflight and not portfolio
    if require_archive_trial and required and plan["action"] != "wait" and not refinements:
        raise ValueError("Measure at least one ordinary archive refinement before skipping this archive or moving on.")
    skip = plan.get("archive_skip")
    if skip is not None:
        if (not isinstance(skip, dict) or not isinstance(skip.get("reason"), str)
                or not skip["reason"].strip() or len(skip["reason"]) > 4000
                or not isinstance(skip.get("evidence"), list) or not skip["evidence"]
                or any(not isinstance(e, str) or e not in known_evidence for e in skip["evidence"])
                or not refinement_ids.intersection(skip["evidence"])):
            raise ValueError("archive_skip must explain low value and cite an available archive parent.")
        skip = {"reason": skip["reason"], "evidence": list(skip["evidence"])}
    skip_required = bool(refinement_ids) and (plan["action"] == "move_on" or
                                             required and plan["action"] == "continue" and not refinements)
    if skip_required and skip is None:
        raise ValueError("Include an ordinary archive refinement, or give archive_skip with measured parent evidence.")
    hypotheses.sort(key=lambda h: -h.get("estimated_model_saving_ms", 0))
    if required and refinements:
        # Keep repairs first. Dispatch archive work before fresh work when only
        # one new writer fits, rather than leaving all refinements in the queue.
        hypotheses.sort(key=lambda h: 0 if h["parent_id"] in repair_ids else
                                      1 if h["parent_id"] in refinement_ids else 2)
    return {"action": plan["action"], "reason": reason, "evidence": evidence, "replan_after": replan,
            "merge_skip": merge_skip,
            "hypotheses": hypotheses, "rejected_hypotheses": rejected,
            "archive_decision": {"eligible_parent_ids": sorted(refinement_ids),
                "refinement_inflight": bool(archive_refinement_inflight), "refinement_required": required,
                "planned_refinements": len(refinements),
                "fresh_starts": sum(h["parent_id"] is None for h in hypotheses), "skip": skip},
            "max_active_writers": min(max_writers, len(hypotheses))}


def measured_progress(rows, previous_score):
    """Use saved timing variation as a scheduling noise estimate.

    This estimate changes search allocation only. The evaluator still applies
    every original correctness, minimum speed, and p95 requirement.
    """
    correct = [row for row in rows if row.get("correct") and row.get("score") is not None]
    best = max(correct, key=lambda row: row["score"], default=None)
    if best is None:
        return {"improved": False, "score": 0., "noise_fraction": 0., "candidate_id": None}
    noise = []
    for case in (best.get("result") or {}).get("cases", []):
        errors = []
        for path in ("baseline", "candidate"):
            samples = case.get(path, {}).get("samples_ms", [])
            if len(samples) >= 5 and all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in samples):
                errors.append(statistics.stdev(samples) / (math.sqrt(len(samples)) * statistics.median(samples)))
        if len(errors) == 2:
            noise.append(2 * math.sqrt(sum(error * error for error in errors)))
    threshold = max([.001, *noise])
    return {"improved": best["score"] > previous_score * (1 + threshold),
            "score": best["score"], "noise_fraction": threshold, "candidate_id": best["id"]}
