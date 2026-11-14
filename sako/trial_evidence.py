"""Test observations declared before a writer submits its candidate."""

import ast
import math
import operator
import re
from collections import Counter

from .core import digest, summary


OPERATORS = {"eq": operator.eq, "lt": operator.lt, "le": operator.le, "gt": operator.gt, "ge": operator.ge}
LAUNCH_FIELD = re.compile(r"launch\.(?:count|\d+\.(?:blocks|threads|(?:grid|block)\.[xyz]))")
SOURCE_FIELD = re.compile(r"source\.calls\.[A-Za-z_]\w*(?:\.\w+)*")
TUNING_FIELD = re.compile(r"tune\.\d+\.\d+\.[A-Za-z_]\w*")


def verification_spec(value):
    if not isinstance(value, dict) or set(value) - {"fixture_index", "implementation"}:
        raise ValueError("Use a fixture_index and implementation checks.")
    fixture = value.get("fixture_index")
    if type(fixture) is not int or fixture < 0:
        raise ValueError("Select a nonnegative fixture_index for the proposed observations.")
    rows = value.get("implementation", [])
    if not isinstance(rows, list) or len(rows) > 8:
        raise ValueError("Use up to eight implementation checks.")
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"field", "op", "value"}:
            raise ValueError("Each observation needs field, op, and value.")
        field = row["field"]
        if not isinstance(field, str) or len(field) > 200 or row["op"] not in OPERATORS:
            raise ValueError("Use a supported observable field and numeric comparison.")
        if type(row["value"]) not in (int, float) or not math.isfinite(row["value"]):
            raise ValueError("Use a finite prediction for each observation.")
        if not (LAUNCH_FIELD.fullmatch(field) or SOURCE_FIELD.fullmatch(field) or TUNING_FIELD.fullmatch(field)):
            raise ValueError(f"Unknown implementation observation: {field}")
    return {"fixture_index": fixture, "implementation": rows}


def validate_writer_design(value, fixture_count, merge_parent_ids=()):
    """Bind a writer's implementation proposal to its submitted source and fixture."""
    required = {"claim", "algorithm", "change", "expected_metric", "reason", "verification"}
    if merge_parent_ids:
        required.add("parent_contributions")
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("The writer design needs claim, algorithm, change, expected_metric, reason and verification.")
    if merge_parent_ids:
        contributions = value["parent_contributions"]
        if (not isinstance(contributions, dict) or set(contributions) != set(merge_parent_ids)
                or any(not isinstance(v, str) or not v.strip() or len(v) > 2000 for v in contributions.values())):
            raise ValueError("Explain the contribution of each supplied merge parent.")
    for field in required - {"verification", "parent_contributions"}:
        if not isinstance(value[field], str) or not value[field].strip() or len(value[field]) > 2000:
            raise ValueError("Each writer design field must contain one to 2,000 characters.")
    verification = verification_spec(value["verification"])
    if verification["fixture_index"] >= fixture_count:
        raise ValueError("The writer selected a fixture outside its task.")
    return {**value, "verification": verification}


def source_calls(candidate, launches):
    """Count call sites reachable from observed Triton entry points. These are source facts."""
    if candidate.get("backend") != "triton":
        return None
    tree = ast.parse(candidate["source"])
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    imports = {a.asname or a.name: a.name for n in tree.body if isinstance(n, ast.Import) for a in n.names}
    pending = {launch["name"] for launch in launches}
    if not pending or not pending.issubset(functions):
        return None
    visited, calls = set(), Counter()
    while pending:
        name = pending.pop()
        visited.add(name)
        for node in ast.walk(ast.Module(body=functions[name].body, type_ignores=[])):
            if isinstance(node, ast.Call):
                call = ast.unparse(node.func)
                root, dot, tail = call.partition(".")
                calls[imports.get(root, root) + dot + tail] += 1
                if call in functions and call not in visited:
                    pending.add(call)
    return calls


def execution_facts(execution):
    if execution.get("status") != "observed":
        return {}
    launches = execution["launches"]
    facts = {"launch.count": len(launches)}
    for i, launch in enumerate(launches):
        for group, total in (("grid", "blocks"), ("block", "threads")):
            dims = launch.get(group)
            if dims:
                facts[f"launch.{i}.{total}"] = math.prod(dims)
                facts.update({f"launch.{i}.{group}.{axis}": n for axis, n in zip("xyz", dims)})
    for i, tuner in enumerate(execution.get("tuners", [])):
        if not tuner.get("invoked"):
            continue
        choices = [tuner["fixed"]] if tuner.get("fixed") else tuner["choices"]
        for j, choice in enumerate(choices):
            facts.update({f"tune.{i}.{j}.{name}": value for name, value in choice["launch_options"].items()
                          if type(value) in (int, float) and math.isfinite(value)})
    return facts


def trial_observations(hypothesis, candidate, result):
    spec = hypothesis.get("verification") or {}
    fixture = spec.get("fixture_index")
    case = next((c for i, c in enumerate(result.get("cases", [])) if c.get("case_index", i) == fixture), {})
    execution = case.get("executions", {}).get("candidate", {})
    launches = execution.get("launches", [])
    facts = execution_facts(execution)
    reference_facts = execution_facts(case.get("executions", {}).get("reference", {}))
    calls = source_calls(candidate, launches) if any(r["field"].startswith("source.") for r in spec.get("implementation", [])) else None
    observations = {"fixture_index": fixture, "requested": hypothesis.get("change"),
                    "submitted_strategy": candidate.get("strategy"), "source_hash": digest(candidate["source"])}
    checks = []
    for row in spec.get("implementation", []):
        field = row["field"]
        actual = facts.get(field)
        if calls is not None and field.startswith("source.calls."):
            actual = calls[field.removeprefix("source.calls.")]
        passed = None if actual is None else OPERATORS[row["op"]](actual, row["value"])
        checks.append({**row, "observed": actual, "unit": None, "passed": passed})
        if field in reference_facts:
            checks[-1]["timing_reference"] = reference_facts[field]
    observations["implementation"] = checks
    observed = checks and all(r["passed"] is True for r in checks)
    indistinguishable = checks and all(r.get("timing_reference") == r["observed"] and r["observed"] is not None for r in checks)
    observations["status"] = ("contradicted" if any(r["passed"] is False for r in checks) else
                              "observed" if observed and not indistinguishable else "unverified")
    if indistinguishable:
        observations["reason"] = "All selected facts also describe the timing reference."
    return observations


def model_opportunity(task, workloads, model_times):
    times = {digest(row["workload"]): row.get("median_ms") for row in model_times}
    costs = task.get("workload_medians_ms", [])
    phases = task.get("compiled_evidence", {}).get("generation_phase_ms", {})
    rows = []
    for i, (workload, cost) in enumerate(zip(workloads, costs)):
        if type(cost) not in (float, int) or cost <= 0:
            continue
        row = {"workload": workload, "region_ms": cost, "model_ms": times.get(digest(workload))}
        if workload.get("mode") == "cached_generation":
            row["phase_ms"] = {phase: values[i] for phase, values in phases.items() if i < len(values)}
        rows.append(row)
    return rows


def measured_opportunity(costs, result):
    """Project each fixture ratio against the original compiled cost for its own workload."""
    by_workload = {digest(row["workload"]): row for row in costs}
    estimates = []
    for i, case in enumerate(result.get("cases", [])):
        cost = by_workload.get(digest(case.get("workload")))
        reference = case.get("search_reference", {})
        if not cost or not case.get("correct"):
            continue
        try:
            if reference.get("kind") == "fixed_anchor":
                baseline = summary([reference["baseline_median_ms"]])["median_ms"]
                candidate = summary(case["candidate"]["samples_ms"])["median_ms"]
            elif reference.get("kind") == "paired_compile":
                candidate = summary(reference["candidate"]["samples_ms"])["median_ms"]
                baseline = summary(reference["baseline"]["samples_ms"])["median_ms"]
            else:
                continue
        except (KeyError, ValueError, TypeError):
            continue
        region_ms = cost["region_ms"]
        phase = case.get("generation_step", {}).get("phase")
        if cost["workload"].get("mode") == "cached_generation":
            region_ms = cost.get("phase_ms", {}).get(phase)
            if region_ms is None:
                continue
        estimates.append({**cost, **({"phase": phase} if phase else {}), "fixture_index": case.get("case_index", i), "operation_speedup": baseline / candidate,
                          "reference_kind": reference["kind"],
                          "estimated_model_saving_ms": region_ms * (1 - candidate / baseline)})
    return estimates



def model_saving(task, result, model_times=()):
    """Estimate saved runtime once per workload and phase, using the slowest captured case."""
    workloads = []
    for case in task.get("cases", []):
        workload = case.get("workload")
        if workload and workload not in workloads:
            workloads.append(workload)
    if (not workloads or len(workloads) != len(task.get("workload_medians_ms", []))
            or not task.get("compiled_evidence", {}).get("generation_phase_ms")):
        return None
    estimates = measured_opportunity(model_opportunity(task, workloads, model_times), result)
    grouped = {}
    for row in estimates:
        key = (digest(row["workload"]), row["phase"])
        grouped.setdefault(key, []).append(row["estimated_model_saving_ms"])
    if not grouped:
        return None
    savings = {key: min(values) for key, values in grouped.items()}
    # Shape routing can use the original operation in a slower phase.
    if result.get("shape_decisions"):
        savings = {key: max(0., value) for key, value in savings.items()}
    per_workload = [sum(value for key, value in savings.items() if key[0] == digest(w)) for w in workloads]
    return {"estimated_model_saving_ms": sum(per_workload) / len(per_workload),
            "phase_savings_ms": [{"workload": w, "phase": phase, "saving_ms": savings[digest(w), phase]}
                                 for w in workloads for phase in ("prefill", "decode")
                                 if (digest(w), phase) in savings],
            "basis": "Original compiled phase costs and paired operation timings."}


def stopping_evidence(reply, measured, assignments, task_id):
    """Build the recorded stopping basis from evaluator facts. Keep the proposed rationale separately."""
    ids = sorted(measured.keys() & set(reply["evidence"]))
    if not ids:
        raise ValueError("A move_on decision must cite a measured candidate from this task.")
    outcomes = [{"candidate_id": key, "accepted": measured[key].get("decision", {}).get("accepted") is True,
                 "timing_baseline": measured[key].get("timing_baseline"),
                 "speedup": measured[key].get("decision", {}).get("speedup")} for key in ids]
    trials = [{"assignment": a["id"], "candidate_id": a["candidate_id"],
               "implementation": (a.get("observations") or {}).get("status", "unverified")}
              for a in assignments.values() if a["task_id"] == task_id and a.get("candidate_id") in ids]
    kept = sum(o["accepted"] for o in outcomes)
    checked = sum(t["implementation"] == "observed" for t in trials)
    return {"agent_reason": reply["reason"], "stopping_basis": {"measured_candidates": outcomes,
            "implementation_checks": trials}, "reason":
            f"The coordinator stopped after citing {len(ids)} measured candidates. {kept} passed the operation gates. "
            f"{checked} trials confirmed their declared observations."}
