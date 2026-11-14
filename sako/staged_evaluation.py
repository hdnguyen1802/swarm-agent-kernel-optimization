"""Screen a shared design on each real module input before numerical checks."""

from functools import lru_cache
import math
from pathlib import Path
import time

from .core import candidate_id, digest, score_cases, write_json
from .shape_selection import case_signature, decide, groups


def enabled(config):
    return config.get("kernel_search", {}).get("evaluation") == "staged"


def clearly_slower(pair, margin):
    base = pair["baseline"]["samples_ms"]
    trial = pair["candidate"]["samples_ms"]
    return (len(base) == len(trial) >= 5 and
            all(math.isfinite(x) and x > 0 for x in [*base, *trial]) and min(trial) > max(base) * margin)


def initial_indices(cases, focus_paths=(), *, by_layout=False, failures=()):
    """Select one saved example per phase, or per required layout within each phase."""
    ordered = sorted(range(len(cases)), key=lambda i: (i not in failures, cases[i].get("source_path") not in focus_paths))
    phases = {}
    for i in ordered:
        phase = cases[i].get("generation_step", {}).get("phase")
        # A forward workload can have several distinct, non-generation inputs.
        key = (phase, digest(case_signature(cases[i], cases[i].get("scalars", ())))) if by_layout else phase or digest(cases[i]["workload"])
        phases.setdefault(key, i)
    return list(phases.values())


def screen_indices(task, timed, focus_paths):
    """Sample early, middle and late sites for each operation family and actual layout."""
    families = {}
    for i in timed:
        case = task["cases"][i]
        path = case["source_path"]
        family = tuple(part for part in path.split(".") if not part.isdecimal())
        layout = digest(case_signature(case, task.get("scalars", [])))
        key = family, layout, path in focus_paths, case.get("generation_step", {}).get("step")
        families.setdefault(key, []).append(i)
    return sorted({i for sites in families.values() for i in (sites[0], sites[len(sites)//2], sites[-1])})


def measured_split(task, result, margin):
    """Split only across a gap between every observed paired ratio on matching inputs."""
    if len(task["paths"]) < 2 or result.get("stages", {}).get("outcome") != "confirmed":
        return None
    comparisons = {}
    for case, meta in zip(result["cases"], task["cases"]):
        ref = case.get("search_reference", {})
        if not case.get("correct") or ref.get("kind") != "paired_compile":
            continue
        base = ref["baseline"]["samples_ms"]
        new = ref["candidate"]["samples_ms"]
        if len(base) < 5 or len(base) != len(new):
            continue
        key = digest([case_signature(meta, task.get("scalars", [])), meta["workload"],
                      meta.get("generation_step", {}).get("phase")])
        ratios = comparisons.setdefault(key, {}).setdefault(case["source_path"], [])
        ratios.extend(b / n for b, n in zip(base, new))
    best = None
    for key, sites in comparisons.items():
        if set(sites) != set(task["paths"]):
            continue
        order = sorted(sites, key=lambda p: min(sites[p]))
        upper = 0.
        for i in range(1, len(order)):
            upper = max(upper, max(sites[order[i - 1]]))
            lower = min(sites[order[i]])
            gap = lower / upper
            if gap > margin and (not best or gap > best["ratio_gap"]):
                best = {"ratio_gap": gap, "required_gap": margin, "case_signature": key,
                        "candidate_id": result["candidate_id"], "scopes": [
                            {"id": digest(paths)[:12], "paths": paths}
                            for paths in (sorted(order[:i]), sorted(order[i:]))]}
    return best


def focus_measurements(result, paths):
    import statistics
    phases = {}
    for case in (result or {}).get("cases", []):
        if case.get("source_path") not in paths:
            continue
        ref = case.get("search_reference", {})
        if ref.get("kind") != "paired_compile":
            continue
        phase = case.get("generation_step", {}).get("phase", "forward")
        phases.setdefault(phase, []).append(ref["baseline"]["median_ms"] / ref["candidate"]["median_ms"])
    return {phase: {"paths_measured": len(samples), "median_speedup": statistics.median(samples)}
            for phase, samples in phases.items()}


@lru_cache(maxsize=2)
def _saved_inputs(filename, modified, size):
    import torch
    return torch.load(filename, weights_only=True, mmap=True, map_location="cpu")["cases"]


@lru_cache(maxsize=32)
def _compiled_reference(kind, compile_mode, device):
    import torch
    from .gpu import reference
    return torch.compile(lambda tensors, scalars: reference(kind, tensors, scalars),
                         fullgraph=True, dynamic=False, mode=compile_mode)


@lru_cache(maxsize=1)
def _load_generation_reference(filename, modified, size, checksum):
    import torch
    from .transport import _file_hash
    if _file_hash(Path(filename)) != checksum:
        raise ValueError("Saved generation outputs changed. Prepare the reference again.")
    return torch.load(filename, map_location="cpu", weights_only=True)


def prepared_generation(profile, catalog):
    from .core import digest
    original = catalog["starting_baseline"]["workloads"][0]
    record = original["reference_output"]
    if (record["identity_hash"] != digest(original["identity"]) or
            original["record_hash"] != digest({k: v for k, v in original.items() if k != "record_hash"})):
        raise ValueError("The saved model output does not match the baseline settings.")
    path = (Path(profile) / record["file"]).resolve()
    path.relative_to(Path(profile).resolve())
    stat = path.stat()
    ids, layers, scores, storage, extent = _load_generation_reference(str(path), stat.st_mtime_ns, stat.st_size, record["sha256"])
    return ids.cuda(), [(k.cuda(), v.cuda()) for k, v in layers], scores.cuda(), storage, extent


def _validate_fixture(task, cases):
    if not cases or len(cases) != len(task["cases"]):
        raise ValueError("The saved tensors do not cover the module cases.")
    for index, (case, meta) in enumerate(zip(cases, task["cases"])):
        context = {k: meta[k] for k in ("workload", "seed", "generation_step") if k in meta}
        if (case["source_path"] != meta["source_path"] or case["source_path"] not in task["paths"]
                or case.get("eager_output_path") != case["source_path"] or case.get("eager_output_case_index") != index
                or case.get("eager_output_reference") != "original_model" or case.get("eager_output") is None
                or case.get("eager_output_context") != context or any(case.get(k) != v for k, v in context.items())
                or [list(t.shape) for t in case["tensors"]] != meta["shapes"]
                or [list(t.stride()) for t in case["tensors"]] != meta["strides"]
                or [str(t.dtype) for t in case["tensors"]] != meta["dtypes"]):
            raise ValueError("A saved input, output or module path differs from its preparation record.")


class Inputs:
    """Keep two independent device buffers per layout. The saved CPU tensors are read only."""
    def __init__(self, reference_layouts=None):
        self.layouts = {}
        self.reference_layouts = {} if reference_layouts is None else reference_layouts

    def load(self, case):
        import torch
        key = tuple((tuple(t.shape), tuple(t.stride()), t.dtype) for t in case["tensors"])
        for layouts in (self.reference_layouts, self.layouts):
            if key not in layouts:
                layouts[key] = [torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device="cuda") for t in case["tensors"]]
        reference, candidate = self.reference_layouts[key], self.layouts[key]
        for tensor, left, right in zip(case["tensors"], reference, candidate):
            left.copy_(tensor)
            right.copy_(left)
        return reference, candidate


@lru_cache(maxsize=1)
def _reference_workspace(path, modified_ns, size, task_id, hardware_id, mode, device):
    # A resident worker owns one GPU and serializes jobs. Its graphs and input addresses share a lifetime.
    return {}, {}


def _failure_record(profile, task_id, identity, case, check):
    # The immutable .pt fixture already contains this input and its original output.
    root = Path(profile) / "failed-inputs" / task_id
    index = case["case_index"]
    record = {"candidate_id": identity, "case_index": index, "source_path": case["source_path"],
              "fixture": task_id + ".pt", "workload": case["workload"], "seed": case["seed"],
              "generation_step": case.get("generation_step"), "check": check}
    name = root / f"{index}-{identity}.json"
    write_json(name, record)
    return str(name)


def evaluate(candidate, task, profile, config, hardware, baselines, *, job):
    import torch
    from .automatic_tuning import declared, policy as tuning_policy
    from .compat import configure_compiler
    from .gpu import case_identity, check_output, compiler_resources, custom_op, paired_timings
    from .kernel_selection import kernel_config
    from .supplemental_correctness import load_fixture
    configure_compiler(config)
    from .measurement import decode_only
    decode = decode_only(config)
    if decode and baselines:
        raise ValueError("Decode candidates must use the original compiled operation as their reference.")
    declaration = declared(candidate)
    if declaration and declaration["settings"] != tuning_policy(config):
        raise ValueError("The candidate and run use different tuning settings.")
    started = time.perf_counter()
    path = (Path(profile) / (task["id"] + ".pt")).resolve()
    stat = path.stat()
    cases = _saved_inputs(str(path), stat.st_mtime_ns, stat.st_size)
    _validate_fixture(task, cases)
    identity = candidate_id(candidate)
    run = custom_op(candidate, task["kind"], hardware=hardware)
    compiled = _compiled_reference(task["kind"], config["compile_mode"], torch.cuda.current_device())
    sources = {id(value): value for value in baselines.values()}
    compiled_keeps = {key: custom_op(value, task["kind"], hardware=hardware) for key, value in sources.items()}
    kept = {key: compiled_keeps[id(value)] for key, value in baselines.items()}
    by_index = {i: key for key, group in groups(task).items() for i in group["indices"]}
    mode = config["kernel_search"].get("timing_mode", "events")
    rules = kernel_config(config)
    reference_buffers, reference_graphs = (_reference_workspace(str(path), stat.st_mtime_ns, stat.st_size, task["id"],
        hardware["fingerprint"], config["compile_mode"], torch.cuda.current_device()) if decode and mode == "cuda_graph" else ({}, {}))
    buffers = Inputs(reference_buffers)
    graphs = {}
    rows = [{**case_identity(case, i), "source_path": case["source_path"],
             "reference_path": case["source_path"], "shapes": task["cases"][i]["shapes"],
             "reference": "eager_original_model", "correct": None, "checks": [],
             "timing_reference": "kept_kernel" if by_index[i] in kept else "torch_compile"}
            for i, case in enumerate(cases)]
    ids = {key: candidate_id(value) for key, value in sources.items()}
    baseline_ids = {key: ids[id(value)] for key, value in baselines.items()}
    stage_times = {"prepare_s": time.perf_counter() - started}
    result = {"scope": "operation", "candidate_id": identity, "cases": rows, "stage_times": stage_times,
              "kernel_min_speedup": rules["min_speedup"], "acceptance_scope": "measured_module_and_shape",
              "timing_baseline": {"kind": "per_shape", "shapes": {k: baseline_ids.get(k) for k in set(by_index.values())}},
              "stages": {"initial_correctness_calls": 0, "screen_pairs": 5},
              "decision": {"accepted": False, "speedup": None, "reasons": []}}
    timed = [i for i, case in enumerate(cases) if not decode or case.get("generation_step", {}).get("phase") == "decode"]
    if not timed:
        raise ValueError("The fixture has no input for the requested timing phase.")
    if decode:
        result.update(performance_scope="cached_decode", timed_case_indices=timed)
        result["stages"].update(confirmation_initial_pairs=rules["trials"], confirmation_max_pairs=config["trials"])

    def calls(i):
        case = cases[i]
        left, right = buffers.load(case)
        incumbent = kept.get(by_index[i], compiled)
        return left, right, lambda: incumbent(left, case["scalars"]), lambda: run(right, case["scalars"])

    def graph_pair(i, stage):
        # The two persistent buffers have fixed addresses for this layout.
        layout = case_signature(task["cases"][i], task.get("scalars", []))
        key = stage, digest(layout), baseline_ids.get(by_index[i])
        pair = graphs.setdefault(key, {})
        if decode and mode == "cuda_graph":
            repeats = min(3, config["repeats"]) if stage == "screen" else config["repeats"]
            pair["baseline"] = reference_graphs.setdefault((key[1], repeats), {})
        return pair

    def record_check(i, output, left, right):
        check = check_output(output, cases[i]["eager_output"].to("cuda"), config)
        intact = all(torch.equal(x, y) for x, y in zip(left, right))
        rows[i].update(correct=check["passed"] and intact, checks=[check], inputs_unchanged=intact)
        if not rows[i]["correct"]:
            rows[i]["failing_input"] = _failure_record(profile, task["id"], identity, cases[i], check)
        return rows[i]["correct"]

    with torch.inference_mode():
        started = time.perf_counter()
        quick = {**config, "warmup": 1, "trials": 5, "repeats": min(3, config["repeats"])}
        screened = screen_indices(task, timed, job.get("focus_paths", ())) if decode else list(timed)
        def screen(i):
            row = rows[i]
            _, _, base, new = calls(i)
            sample = paired_timings(base, new, quick, mode=mode, prepared=graph_pair(i, "screen"))
            row["speed_screen"] = {"baseline": sample["baseline"], "candidate": sample["candidate"],
                "clearly_slower": clearly_slower(sample, config["max_p95_ratio"]), "acceptance_evidence": False}
        for i in screened:
            screen(i)
        # Confirm that every remaining site is slow before discarding a design with a possible partial gain.
        if decode and all(rows[i]["speed_screen"]["clearly_slower"] for i in screened):
            for i in sorted(set(timed).difference(screened)):
                screen(i)
                screened.append(i)
                if not rows[i]["speed_screen"]["clearly_slower"]:
                    break
        result["stages"]["screen_case_indices"] = screened
        stage_times["speed_screen_s"] = time.perf_counter() - started
        if decode:
            import statistics
            result["screen_speedup"] = statistics.geometric_mean(rows[i]["speed_screen"]["baseline"]["median_ms"] /
                rows[i]["speed_screen"]["candidate"]["median_ms"] for i in screened)
        # Triton caches the configurations selected on the first invocation.
        result["automatic_tuning"] = {"status": "screened", "kernels": run._sako_tuning_metadata()}
        if all(rows[i]["speed_screen"]["clearly_slower"] for i in screened):
            result["stages"]["outcome"] = "screened_slow"
            result["stages"]["correctness_status"] = "not_checked"
            result["decision"]["reasons"] = ["Every measured input was clearly slower in the short speed test."]
            return result
        started = time.perf_counter()
        indices = {int(f.name.split("-", 1)[0]) for f in (Path(profile) / "failed-inputs" / task["id"]).glob("*.json")}
        initial = initial_indices(task["cases"], job.get("focus_paths", ()), by_layout=decode, failures=indices)
        for i in initial:
            left, right, _, new = calls(i)
            with run._sako_observe_resources() as records:
                output = new()
            rows[i]["build"] = compiler_resources(candidate, hardware, records)
            record_check(i, output, left, right)
            result["stages"]["initial_correctness_calls"] += 1
        result["stages"]["initial_case_indices"] = initial
        stage_times["initial_correctness_s"] = time.perf_counter() - started
        if any(not rows[i]["correct"] for i in initial):
            result["stages"]["outcome"] = "initial_accuracy_failed"
            result["decision"]["reasons"] = ["The initial saved input failed its accuracy or integrity check."]
            return result
        if initial:
            rows[initial[0]]["autotuning"] = result["automatic_tuning"]["kernels"]
        # Replay saved local failures before spending time on independent timing pairs.
        for i in sorted(indices.difference(initial)) if not decode else ():
            left, right, _, new = calls(i)
            if not record_check(i, new(), left, right):
                result["stages"]["outcome"] = "saved_accuracy_failed"
                result["decision"]["reasons"] = ["A previously failing input failed again."]
                return result
        supplemental, descriptor = load_fixture(profile, config, job["_catalog"], task) if not decode else (None, None)
        if supplemental:
            from .gpu import evaluate_supplemental
            checked = evaluate_supplemental(candidate, task, supplemental, config, repeats=1,
                                             hardware=hardware)
            result["supplemental_correctness"] = {**checked, "artifact": descriptor}
            if not checked["passed"]:
                result["stages"]["outcome"] = "saved_accuracy_failed"
                result["decision"]["reasons"] = ["An input from a prior model failure failed its accuracy check."]
                return result
        started = time.perf_counter()
        for i in timed:
            row = rows[i]
            left, right, base, new = calls(i)
            output = None
            def measured():
                nonlocal output
                output = new()
                return output
            # Retain the actual timed output for its own path. No extra candidate launch is needed.
            observe = ({"observe_runners": (kept.get(by_index[i]), run)}
                       if i == job.get("observe_fixture") else {})
            prepared = graph_pair(i, "observed" if observe else "confirm")
            row.update(paired_timings(base, measured, config, mode=mode, prepared=prepared,
                                      **({"decision_limits": rules} if decode else {}), **observe))
            if "outputs" in prepared:
                output = prepared["outputs"][1]
            if not record_check(i, output, left, right):
                result["stages"]["outcome"] = "confirmation_accuracy_failed"
                result["decision"]["reasons"] = ["The output of a timed operation failed its saved reference."]
                return result
            if config["kernel_search"].get("score_reference") == "paired_compile":
                ranking = row
                if by_index[i] in kept and not job.get("archive_on_peer"):
                    ranking = paired_timings(lambda: compiled(left, cases[i]["scalars"]), new, config, mode=mode,
                                             prepared=graph_pair(i, "archive"))
                if by_index[i] not in kept or not job.get("archive_on_peer"):
                    row["search_reference"] = {"kind": "paired_compile", "timing_mode": mode,
                        "baseline": ranking["baseline"], "candidate": ranking["candidate"], "acceptance_evidence": False}
        stage_times["confirmation_s"] = time.perf_counter() - started
    result["stages"]["outcome"] = "confirmed"
    result["stages"]["prepared_graph_pairs"] = len(graphs) if mode == "cuda_graph" else 0
    result["automatic_tuning"]["status"] = "checked_and_tuned"
    result["decision"] = score_cases([rows[i] for i in timed], rules)
    result["shape_decisions"] = decide(result, task, rules)
    return result
