"""Trusted deployment controls. Diagnostic timings cannot grant acceptance."""
from collections import defaultdict
import math
from pathlib import Path
from typing import List

from .core import read_json, write_json

_REFERENCE = None


def linear_reference_control():
    """Original linear math behind the same opaque boundary. Never a candidate."""
    global _REFERENCE
    if _REFERENCE is None:
        import torch
        @torch.library.custom_op("sako_diagnostic::linear_reference", mutates_args=())
        def operation(tensors: List[torch.Tensor], scalars: List[float]) -> torch.Tensor:
            return torch.nn.functional.linear(tensors[0], tensors[1], tensors[2] if tensors[2].numel() else None)
        @operation.register_fake
        def fake(tensors, scalars):
            return tensors[0].new_empty((*tensors[0].shape[:-1], tensors[1].shape[0]))
        _REFERENCE = operation
    return _REFERENCE


def trace_summary(data):
    kernels, work = [], []
    runtime, driver, operators = (defaultdict(lambda: [0, 0.]) for _ in range(3))
    activity_counts = defaultdict(int)
    for event in data.get("traceEvents", []):
        duration, start = event.get("dur"), event.get("ts")
        if (event.get("ph") != "X" or type(duration) not in (int, float)
                or type(start) not in (int, float) or not math.isfinite(duration)
                or not math.isfinite(start) or duration < 0):
            continue
        category, name = event.get("cat", ""), event.get("name", "unknown")
        if category == "kernel":
            kernels.append((start, start + duration, name))
        if category in {"kernel", "gpu_memcpy", "gpu_memset"}:
            work.append((start, start + duration))
            activity_counts[category] += 1
        if category == "cuda_runtime":
            runtime[name][0] += 1; runtime[name][1] += duration
        if category == "cuda_driver":
            driver[name][0] += 1; driver[name][1] += duration
        if category == "cpu_op" and name.startswith(("sako::", "sako_diagnostic::")):
            operators[name][0] += 1; operators[name][1] += duration
    intervals = sorted(work)
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    span = merged[-1][1] - merged[0][0] if merged else 0.
    busy = sum(end - start for start, end in merged)
    names = defaultdict(lambda: [0, 0.])
    for start, end, name in kernels:
        names[name][0] += 1; names[name][1] += end - start
    rows = lambda source: [{"name": name, "calls": count, "duration_us": duration}
        for name, (count, duration) in sorted(source.items(), key=lambda item: -item[1][1])]
    return {"kernel_count": len(kernels), "kernel_duration_sum_us": sum(e - s for s, e, _ in kernels),
        "gpu_span_us": span, "gpu_busy_union_us": busy, "gpu_gap_us": max(0., span - busy),
        "kernels": rows(names), "cuda_runtime": rows(runtime), "cuda_driver": rows(driver),
        "gpu_activity_counts": dict(activity_counts), "opaque_cpu_ops": rows(operators),
        "note": "Profiled execution only. Durations can overlap. GPU gaps and CPU events do not by themselves identify a cause."}


def capture_model_traces(baseline, candidate, ids, directory):
    import torch
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    reports = {}
    for name, fn in (("baseline", baseline), ("candidate", candidate)):
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profiler:
            with torch.profiler.record_function("deployment:" + name):
                fn(ids)
                torch.cuda.synchronize()
        path = directory / (name + ".trace.json")
        profiler.export_chrome_trace(str(path))
        reports[name] = {"trace": str(path), **trace_summary(read_json(path))}
    write_json(directory / "summary.json", reports)
    return {"acceptance_evidence": False, "scope": "One warmed complete forward after acceptance timing.",
            "reports": reports}


def performance_trace(directory, decision, cases, config, baseline, candidate, make_inputs):
    """Trace the largest relative speed miss after all accuracy and timing checks."""
    if not directory or decision.get("accepted") or not cases or not all(c.get("correct") for c in cases):
        return None
    from .core import score_cases
    misses = []
    for i, case in enumerate(cases):
        required = case.get("speed_requirement", config["min_speedup"])
        local = score_cases([case], {**config, "min_speedup": required})
        if not local["accepted"] and local["speedup"]:
            misses.append((local["speedup"] / required, i))
    if not misses:
        return None
    workload = cases[min(misses)[1]]["workload"]
    try:
        import torch
        with torch.inference_mode():
            report = capture_model_traces(baseline, candidate, make_inputs(workload), directory)
        return {"status": "measured", "workload": workload, **report}
    except Exception as error:
        return {"status": "unavailable", "workload": workload, "reason": str(error), "acceptance_evidence": False}


def trace_feedback(trace):
    result = {k: trace[k] for k in ("status", "workload", "reason", "acceptance_evidence") if k in trace}
    for role, report in trace.get("reports", {}).items():
        result[role] = {k: report[k] for k in ("trace", "kernel_count", "gpu_span_us", "gpu_busy_union_us", "gpu_gap_us")}
        result[role]["longest_kernels"] = report.get("kernels", [])[:3]
    return result


def retain_trace_feedback(store, trial_key, manifest, result):
    if result.get("performance_trace"):
        store.set("last_performance_trace", {"manifest_hash": trial_key, "decision": result["decision"],
            "replacements": [{k: e[k] for k in ("task_id", "candidate_id")} for e in manifest["entries"]],
            **trace_feedback(result["performance_trace"])})


def checked_compiled_sequence(fn, cases, config, name):
    from .gpu import check_output
    outputs = fn()
    if not isinstance(outputs, (tuple, list)) or len(outputs) != len(cases):
        raise ValueError("The compiled rotating control returned a different number of outputs.")
    checks = [{"path": name, **check_output(output, expected, config)}
              for output, (_, _, expected) in zip(outputs, cases)]
    if not all(check["passed"] for check in checks):
        raise ValueError("The compiled rotating control failed eager accuracy.")
    return checks


def rotating_audit(candidate, task, profile, catalog, config, *, session=None):
    """Rotate real weights for representative paths, outside model acceptance."""
    import torch
    from .gpu import compile_candidate, custom_op, model_operation, reference, check_output, paired_timings
    from .supplemental_correctness import ensure_fixture
    fixture, artifact = ensure_fixture(profile, config, catalog, task, session=session)
    seed = config["correctness_seeds"][0]
    chosen = [case for case in fixture["cases"] if case["origin"] == "representative" and case["seed"] == seed]
    raw = compile_candidate(candidate)
    opaque = custom_op(candidate, task["kind"])
    visible = model_operation(candidate, task["kind"], {**config, "triton_integration": "direct"})
    reports = []
    with torch.inference_mode():
        for workload in config["workloads"]:
            selected = [case for case in chosen if case["workload"] == workload]
            if not selected:
                raise ValueError("Rotating audit lacks representative cases for a workload.")
            cases = []
            for case in selected:
                tensors = [torch.empty_strided(tuple(t.shape), tuple(t.stride()), dtype=t.dtype, device="cuda").copy_(t)
                           for t in case["tensors"]]
                cases.append((tensors, list(case["scalars"]), case["expected"].cuda()))
            saved = [[t.clone() for t in tensors] for tensors, _, _ in cases]
            checks = []
            for tensors, scalars, expected in cases:
                checks.extend(check_output(fn(tensors, scalars), expected, config) for fn in (raw, opaque, visible))
            if not all(check["passed"] for check in checks):
                raise ValueError("A rotating deployment control failed eager accuracy.")
            def sequence(fn):
                return tuple(fn(tensors, scalars) for tensors, scalars, _ in cases)
            compiled = torch.compile(lambda: sequence(visible), fullgraph=True, dynamic=False, mode=config["compile_mode"])
            compiled_reference = torch.compile(lambda: sequence(lambda tensors, scalars: reference(task["kind"], tensors, scalars)),
                fullgraph=True, dynamic=False, mode=config["compile_mode"])
            compiled_checks = []
            for name, fn in (("candidate", compiled), ("reference", compiled_reference)):
                compiled_checks.extend(checked_compiled_sequence(fn, cases, config, name))
            operations = {"raw": lambda: sequence(raw), "opaque": lambda: sequence(opaque), "compiled_direct": compiled}
            timings = {}
            for mode in ("events", "cuda_graph"):
                timings[mode] = {name: paired_timings(compiled_reference, fn, config, mode=mode)
                                 for name, fn in operations.items()}
            intact = all(torch.equal(t, original) for (tensors, _, _), before in zip(cases, saved)
                         for t, original in zip(tensors, before))
            if not intact:
                raise ValueError("The rotating control changed its inputs.")
            reports.append({"workload": workload, "paths": [case["source_path"] for case in selected],
                "checks": checks, "compiled_checks": compiled_checks,
                "inputs_unchanged": intact, "timings": timings,
                "calls_per_sequence": len(cases)})
            del cases, saved, operations, compiled, compiled_reference
    return {"scope": "rotating_representative_weights", "acceptance_evidence": False,
        "artifact": artifact, "cases": reports,
        "note": "Each sequence uses different real layer weights. This is a cache control, not a complete model benchmark."}
