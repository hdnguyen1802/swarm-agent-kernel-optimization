"""Bound coordinator summaries. Keep complete evidence and writer inputs on disk."""

import copy
import json

from .feedback import compact_feedback



def planning_task(task):
    """Group repeated fixture seeds. Writers retain the complete original task."""
    from .core import digest
    result = copy.deepcopy(task)
    groups = {}
    for index, case in enumerate(task.get("cases", [])):
        signature = _fields(case, ("workload", "shapes", "strides", "dtypes"))
        key = digest(signature)
        group = groups.setdefault(key, {**signature, "case_indices": [], "generation_steps": []})
        group["case_indices"].append(case.get("case_index", index))
        if case.get("generation_step") and case["generation_step"] not in group["generation_steps"]:
            group["generation_steps"].append(copy.deepcopy(case["generation_step"]))
    if groups:
        result["cases"] = list(groups.values())
        result["fixture_count"] = len(task["cases"])
        contract = result.get("contract", {})
        contract.pop("cases", None)
        contract.pop("paths", None)
        validation = contract.get("validation", {})
        if isinstance(validation.get("checks"), dict):
            validation["checks"].pop("examples", None)
        contract["paths_location"] = "task.paths"
        contract["cases_location"] = "task.cases; case_indices refer to original fixture indices"
    evidence = result.get("compiled_evidence", {})
    evidence["generated_kernels"] = [{k: v for k, v in item.items() if k != "source"}
                                      for item in evidence.get("generated_kernels", [])]
    if "groups" in evidence:
        evidence["groups"] = [_fields(group, ("name", "implementation", "median_ms"))
                              for group in evidence["groups"]]
    return result


def _referenced_strings(value):
    if isinstance(value, str):
        return {value}
    if isinstance(value, dict):
        return set().union(*(_referenced_strings(x) for x in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_referenced_strings(x) for x in value), set())
    return set()


def _scope_evidence(result):
    types = result.get("evidence_types", {})
    artifacts = set(types.get("artifact", []))
    if not artifacts:
        return
    values = {k: v for k, v in result.items() if k not in {"known_evidence", "evidence_types"}}
    used = _referenced_strings(values)
    retained = artifacts.intersection(used)
    result["known_evidence"] = [key for key in result.get("known_evidence", []) if key not in artifacts or key in retained]
    result["evidence_types"] = {kind: [key for key in ids if kind != "artifact" or key in retained]
                                for kind, ids in types.items()}


def _fields(value, names):
    return {key: copy.deepcopy(value[key]) for key in names if key in value}


def _brief(value, level=0):
    return compact_feedback(value, max_items=(6, 3, 1)[level],
                            max_text=(600, 320, 120)[level], max_depth=12)


def _operation(value, level):
    if not isinstance(value, dict):
        return value
    result = _fields(value, ("decision", "error", "search_score", "kernel_min_speedup", "acceptance_scope"))
    result = _brief(result, level)
    cases = value.get("cases", [])
    limit = (6, 3, 1)[level]
    result["cases"] = [_brief(_fields(case, (
        "case_index", "seed", "workload", "correct", "max_abs_error", "baseline_ms", "candidate_ms",
        "candidate_p95_ms", "registers_per_thread", "timing_mode", "error")), level) for case in cases[:limit]]
    if len(cases) > limit:
        result["case_count"] = len(cases)
    if isinstance(value.get("shape_decisions"), dict):
        result["shape_decisions"] = {
            key: _brief(_fields(item, ("indices", "accepted", "reasons", "speedup")), level)
            for key, item in value["shape_decisions"].items()}
    if level and value.get("shape_decisions") and all(c.get("correct") for c in cases):
        result.pop("cases", None)
        result["case_count"] = len(cases)
        result["correct_case_count"] = len(cases)
    if isinstance(value.get("timing_baseline"), dict):
        result["timing_baseline"] = _brief(value["timing_baseline"], level)
    return result


def _parent(value, level):
    result = _brief(_fields(value, ("id", "score", "correct", "accepted", "parent_id", "model_status", "projected_from_original",
                                  "hypothesis", "selection_reason", "reuse", "diversity", "code_design", "estimated_model_saving_ms", "phase_savings_ms")), level)
    if level:
        result["hypothesis"] = _brief(_fields(value.get("hypothesis") or {}, ("id", "algorithm", "change")), level)
        diversity = result.get("diversity")
        if isinstance(diversity, dict):
            diversity.pop("change", None)
    result["operation_measurements"] = _operation(value.get("operation_measurements"), level)
    result["model_measurements"] = _model(value.get("model_measurements"), level)
    return result



def _model(value, level):
    if not isinstance(value, dict):
        return value
    result = _brief(_fields(value, ("scope", "decision", "error", "replacements",
                                   "output_equivalence", "reference_state_intact")), level)
    result["cases"] = []
    for case in value.get("cases", [])[:3]:
        row = _fields(case, ("workload", "correct", "baseline_ms", "candidate_ms", "timing_status", "timing_reason", "sampling"))
        checks = case.get("checks", [])
        row["check_count"] = len(checks)
        row["failed_checks"] = _brief([c for c in checks if c.get("passed") is False], level)
        result["cases"].append(row)
    return result


def _hypothesis(value, level):
    result = _fields(value, ("id", "task_id", "status", "origin_assignment", "max_attempts", "max_corrections",
                             "attempt_count", "corrections_used", "corrections_remaining", "last_candidate_id", "close_reason",
                             "implementation_status"))
    result["hypothesis"] = _brief(value.get("hypothesis"), level)
    result["last_outcome"] = _brief(value.get("last_outcome"), level)
    return result


def _summarize(prompt, level):
    result = copy.deepcopy(prompt)
    if isinstance(result.get("hardware"), dict):
        result["hardware"] = _fields(result["hardware"], ("fingerprint", "target", "capability", "facts", "support"))
    hierarchy = result.get("hierarchy")
    if isinstance(hierarchy, dict):
        region = hierarchy.get("active_region")
        if isinstance(region, dict):
            hierarchy["active_region"] = _fields(region, ("id", "kind", "priority_ms", "subsection_ids",
                                                          "leaf_task_ids", "alternative_task_ids"))
        if level > 0:
            hierarchy["latest_priorities"] = {key: _fields(value, ("priority_ms", "excluded_reason"))
                                              for key, value in hierarchy.get("latest_priorities", {}).items()}
    result["parents"] = [_parent(item, level) for item in prompt.get("parents", [])]
    parent_ids = {p["id"] for p in result["parents"]}
    result["operation_results"] = [
        _fields(row, ("id", "task", "generation", "correct", "accepted", "score")) if row.get("id") in parent_ids
        else _brief(row, level) for row in prompt.get("operation_results", [])]
    result["hypothesis_states"] = [_hypothesis(item, level) for item in prompt.get("hypothesis_states", [])]
    if isinstance(result.get("budget"), dict):
        result["budget"].pop("per_role", None)
    result["inflight"] = [_brief(item, level) for item in prompt.get("inflight", [])]
    for key in ("recent_events", "lessons", "invalid_plan"):
        if key in prompt:
            result[key] = _brief(prompt[key], level)
    result["model_result"] = _model(prompt.get("model_result"), level)
    return result


def fit_coordinator_context(prompt, max_chars, *, artifact=None, target_chars=None):
    """Fit JSON without changing contracts, validation rules, or repair budgets.

    Normal summaries keep parent and hypothesis identities. Later
    tiers remove repeated detail. A final fallback removes optional context, never
    the task contract, validation policy, parent identities, or active repairs.
    """
    if type(max_chars) is not int or max_chars <= 0:
        raise ValueError("Use a positive coordinator character limit.")
    target = max_chars if target_chars is None else target_chars
    if type(target) is not int or not 0 < target <= max_chars:
        raise ValueError("The coordinator target must be positive and no greater than its hard limit.")
    before = len(json.dumps(prompt))
    result = None
    for level in range(3):
        result = _summarize(prompt, level)
        result["context_summary"] = {"full_context": copy.deepcopy(artifact), "level": level,
                                     "full_characters": before, "target_chars": target,
                                     "details": "Full evidence remains in the context file."}
        _scope_evidence(result)
        if len(json.dumps(result)) <= target:
            return result
    for key in ("recent_events", "operation_results", "lessons", "model_diagnostic", "example"):
        if len(json.dumps(result)) <= target:
            break
        result.pop(key, None)
    # Closed history can grow even when no candidate remains active. Retain all
    # active state and the original aggregate history count.
    if len(json.dumps(result)) > target:
        result["hypothesis_states"] = [state for state in result.get("hypothesis_states", [])
                                       if state.get("status") not in {"closed", "exhausted", "succeeded"}]
    if len(json.dumps(result)) > max_chars:
        raise ValueError("The required coordinator contract and active state exceed the character limit. Review the saved context file.")
    return result
