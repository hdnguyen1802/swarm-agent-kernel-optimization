"""Trusted Triton build, correctness, and timing functions."""

import os
import contextlib
import logging
import sys
import platform
import subprocess
import time
import traceback
from pathlib import Path
from typing import List

from .core import candidate_id, summary, validate_candidate, digest, write_json
from .hardware import discover_hardware, require_hardware


def doctor(config=None):
    import torch
    import transformers
    import triton
    hardware = discover_hardware(config)
    result = {"python": platform.python_version(), "torch": torch.__version__,
              "transformers": transformers.__version__, "triton": triton.__version__,
              "torch_cuda": torch.version.cuda, "cuda_home": os.environ.get("CUDA_HOME"),
              "hardware": hardware}
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    result.update(gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                  memory_bytes=torch.cuda.get_device_properties(0).total_memory)
    result["nvcc"] = hardware["software"]["compiler_version"]
    if not hardware["support"]["supported"]:
        result["compile_probe"] = False
        return result
    torch._dynamo.config.suppress_errors = False
    x = torch.randn(128, 256, device="cuda", dtype=torch.float16)
    f = torch.compile(lambda a: a.sin() + a, fullgraph=True)
    torch.testing.assert_close(f(x), x.sin() + x, atol=.002, rtol=.002)
    from .timing_controls import control_seed
    probe = control_seed({"id": "device_probe", "cases": [{"shapes": [[32]]}]})
    test_input = torch.arange(32, device="cuda", dtype=torch.float16)
    output = compile_candidate(probe, hardware)([test_input], [])
    torch.testing.assert_close(output, test_input, atol=0., rtol=0.)
    result["candidate_backend"] = "triton"
    hardware["support"].update(compile_probe="passed", runtime_probe="passed")
    result["compile_probe"] = True
    result["driver"] = subprocess.check_output([
        "nvidia-smi", "--query-gpu=uuid,driver_version,memory.used,utilization.gpu",
        "--format=csv,noheader"], text=True).strip()
    return result


def build_identity(candidate, hardware):
    validate_candidate(candidate)
    from .triton_backend import identity
    return identity(candidate, hardware)


def compiler_resources(candidate, hardware, records=None):
    root = Path(__file__).resolve().parents[1]
    path = root / "runs/build" / build_identity(candidate, hardware) / "build.log"
    records = records if isinstance(records, list) else []
    available = [row for row in records if any(row.get(field) is not None for field in
                 ("registers_per_thread", "static_shared_memory_bytes", "reported_spills"))]
    result = {"log_path": path.relative_to(root).as_posix(), "backend": "triton",
              "status": "observed" if available else "unavailable", "resource_reports": available,
              "spill_reports": [row for row in available if (row.get("reported_spills") or 0) > 0],
              "diagnostic_errors": [row for row in records if row.get("error")],
              "note": "Compiler/driver metadata from an untimed call. Resource use alone does not prove a slowdown's cause."}
    if records:
        write_json(path.with_name("resources.json"), {"candidate_id": candidate_id(candidate),
            "source_hash": digest(candidate["source"]), "hardware_fingerprint": hardware["fingerprint"], **result})
    return result


@contextlib.contextmanager
def build_log(path):
    """Capture Python messages and inherited subprocess stdout/stderr."""
    sys.stdout.flush()
    sys.stderr.flush()
    outer_stderr = sys.stderr
    with Path(path).open("a", encoding="utf-8") as log:
        saved = [os.dup(descriptor) for descriptor in (1, 2)]
        try:
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                yield
                log.flush()
        finally:
            # Libraries can create handlers while stderr points at this file.
            # Keep those handlers valid after the job closes its build log.
            try:
                loggers = [logging.getLogger(), *[logger for logger in list(logging.Logger.manager.loggerDict.values())
                                                if isinstance(logger, logging.Logger)]]
                handlers = {handler for logger in loggers for handler in logger.handlers}
                for handler in handlers:
                    # Transformers can bind flush directly to the current file.
                    # Let the handler flush its current stream after this job.
                    if (isinstance(handler, logging.StreamHandler)
                            and getattr(vars(handler).get("flush"), "__self__", None) is log):
                        del handler.flush
                    if isinstance(handler, logging.StreamHandler) and vars(handler).get("stream") is log:
                        handler.setStream(outer_stderr)
            finally:
                for descriptor, backup in zip((1, 2), saved):
                    os.dup2(backup, descriptor)
                    os.close(backup)


def compile_candidate(candidate, hardware=None):
    validate_candidate(candidate)
    hardware = require_hardware(hardware or discover_hardware())
    from .triton_backend import load
    root = Path(__file__).resolve().parents[1]
    name = build_identity(candidate, hardware)
    directory = root / "runs/build" / name
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "hardware.json", hardware)
    write_json(directory / ("request-" + candidate_id(candidate) + ".json"), {
        "candidate_id": candidate_id(candidate), "task_id": candidate["task_id"], "build_identity": name})
    try:
        with build_log(directory / "build.log"):
            return load(candidate, hardware, directory)
    except Exception as error:
        with (directory / "build.log").open("a", encoding="utf-8") as log:
            log.write(traceback.format_exc())
        error.build_log = str(directory / "build.log")
        raise


def reference(kind, tensors, scalars):
    import torch
    import torch.nn.functional as F
    x = tensors[0]
    if kind == "silu":
        return F.silu(x)
    if kind == "linear":
        return F.linear(x, tensors[1], tensors[2] if tensors[2].numel() else None)
    if kind == "embedding":
        return F.embedding(x, tensors[1])
    if kind == "rmsnorm":
        f = x.float()
        return (f * torch.rsqrt(f.square().mean(-1, keepdim=True) + scalars[0])).to(x.dtype) * tensors[1]
    if kind == "mlp":
        return F.linear(F.silu(F.linear(x, tensors[1])) * F.linear(x, tensors[2]), tensors[3])
    if kind == "attention":
        return F.scaled_dot_product_attention(x, tensors[1], tensors[2],
            attn_mask=tensors[3] if tensors[3].numel() else None,
            is_causal=bool(scalars[0]), scale=scalars[1], dropout_p=0.)
    raise ValueError(f"Unknown operation: {kind}")


_CUSTOM_OPERATIONS = {}


def model_operation(candidate, kind, config):
    """Expose validated Triton source to Dynamo; retain eager build logs."""
    mode = config.get("triton_integration", "opaque")
    if mode not in ("opaque", "direct"):
        raise ValueError("Use opaque or direct Triton integration.")
    if mode == "opaque":
        return custom_op(candidate, kind)
    import torch
    if not hasattr(torch, "compiler") or not hasattr(torch.compiler, "is_compiling"):
        raise RuntimeError("Direct Triton integration requires torch.compiler.is_compiling.")
    run = compile_candidate(candidate)
    entry = getattr(run, "_sako_compiler_entry", None)
    if entry is None:
        raise RuntimeError("The validated Triton module has no compiler entry.")
    def operation(tensors, scalars):
        if torch.compiler.is_compiling():
            return entry(tensors, scalars)
        return run(tensors, scalars)
    operation._sako_tuning_metadata = getattr(run, "tuning_metadata", lambda: [])
    operation._sako_integration = "direct"
    return operation


def custom_op(candidate, kind, *, hardware=None):
    import torch
    hardware = require_hardware(hardware or discover_hardware())
    name = "sako::k_" + build_identity(candidate, hardware) + "_" + kind
    key = (name, kind)
    if key in _CUSTOM_OPERATIONS:
        return _CUSTOM_OPERATIONS[key]
    run = compile_candidate(candidate, hardware)

    @torch.library.custom_op(name, mutates_args=())
    def operation(tensors: List[torch.Tensor], scalars: List[float]) -> torch.Tensor:
        return run(tensors, scalars)

    @operation.register_fake
    def fake(tensors, scalars):
        x = tensors[0]
        if kind == "linear":
            return x.new_empty((*x.shape[:-1], tensors[1].shape[0]))
        if kind == "embedding":
            return tensors[1].new_empty((*x.shape, tensors[1].shape[1]))
        return torch.empty_like(x, memory_format=torch.contiguous_format)

    operation._sako_tuning_metadata = getattr(run, "tuning_metadata", lambda: [])
    operation._sako_observe_tuning = getattr(run, "observe_tuning", None)
    operation._sako_observe_resources = getattr(run, "observe_resources", None)
    _CUSTOM_OPERATIONS[key] = operation
    return operation


def check_output(actual, expected, config):
    import math
    import torch
    if not isinstance(actual, torch.Tensor):
        return {"passed": False, "reason": "Output is not a tensor."}
    if actual.shape != expected.shape or actual.dtype != expected.dtype or actual.device != expected.device:
        return {"passed": False, "reason": "Output metadata differs."}
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    error = float((actual.float() - expected.float()).abs().max()) if finite else None
    result = {"passed": finite and bool(torch.allclose(actual, expected, atol=config["atol"], rtol=config["rtol"])),
              "max_abs_error": error, "atol": config["atol"], "rtol": config["rtol"]}
    if not result["passed"]:
        a, b = actual.float(), expected.float()
        allowed = config["atol"] + config["rtol"] * b.abs()
        wrong = (~torch.isfinite(a)) | (~torch.isfinite(b)) | ((a - b).abs() > allowed)
        indices = wrong.flatten().nonzero().flatten()
        result["mismatched_elements"] = indices.numel()
        result["shape"] = list(actual.shape)
        result["examples"] = []
        for index in indices[:5].tolist():
            def number(t):
                value = float(t.flatten()[index])
                return value if math.isfinite(value) else None
            result["examples"].append({"flat_index": index, "actual": number(a),
                                       "expected": number(b), "allowed_abs_error": number(allowed)})
    return result


def _prepare_timing(fn, role, config, repeats, mode, cache, observer, setup):
    import torch
    if cache and "graph" in cache:
        cache["graph"].replay()
        torch.cuda.synchronize()
        setup[role + "_graph_reused"] = True
        return cache["graph"].replay, cache["output"], cache["tuners"]
    start = time.perf_counter()
    for i in range(config["warmup"]):
        fn()
        if i == 0:
            torch.cuda.synchronize()
            setup[role + "_first_warmup_s"] = time.perf_counter() - start
    torch.cuda.synchronize()
    setup["warmup_s"] += time.perf_counter() - start
    if mode == "events":
        return fn, None, None
    try:
        from .launch_evidence import tuning_capture
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(config["warmup"]):
                fn()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        start = time.perf_counter()
        graph = torch.cuda.CUDAGraph()
        observe = tuning_capture(observer) if observer is not None else contextlib.nullcontext()
        with observe as tuners, torch.cuda.graph(graph, stream=stream):
            for _ in range(repeats):
                output = fn()
        torch.cuda.synchronize()
        setup["graph_capture_s"] = setup.get("graph_capture_s", 0.) + time.perf_counter() - start
        start = time.perf_counter()
        for _ in range(max(1, config["warmup"])):
            graph.replay()
        torch.cuda.synchronize()
        setup["graph_replay_warmup_s"] = setup.get("graph_replay_warmup_s", 0.) + time.perf_counter() - start
        cache.update(graph=graph, output=output, tuners=tuners)
        return graph.replay, output, tuners
    except Exception as error:
        from .worker import InfrastructureError
        raise InfrastructureError("CUDA graph capture or replay failed: " + str(error)) from error


def needs_more_kernel_samples(baseline, candidate, rules):
    """Use all trials if the first five vary or lie close to either speed gate."""
    from .measurement import DEFAULTS, relative_spread
    ratios = [a / b for a, b in zip(baseline, candidate)]
    margin = DEFAULTS["near_threshold_ratio"]
    return (relative_spread(ratios) > DEFAULTS["timing_relative_spread"] or
            any(min(ratios) <= limit * (1 + margin) and max(ratios) >= limit * (1 - margin)
                for limit in (rules["min_speedup"], 1. / rules["max_p95_ratio"])))


def _timings(calls, config, repeats, mode, observers, prepared, decision_limits=None):
    import torch
    if mode not in {"events", "cuda_graph"}:
        raise ValueError("Unknown timing mode.")
    setup = {"warmup_s": 0.}
    runners, outputs, tuners = {}, [], {}
    gpu, wall = {}, {}
    for i, (role, fn) in enumerate(calls.items()):
        cache = prepared.setdefault(role, {})
        runner, output, tuner = _prepare_timing(fn, role, config, repeats, mode, cache,
                                                observers[i] if observers else None, setup)
        runners[role] = runner
        outputs.append(output)
        tuners[role] = tuner
        gpu[role], wall[role] = [], []
    if mode == "cuda_graph":
        prepared["outputs"] = outputs
        if all(setup.get(role + "_graph_reused") for role in calls):
            setup["prepared_pair_reused"] = True
    for trial in range(config["trials"]):
        for role in calls if trial % 2 == 0 else reversed(calls):
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            clock = time.perf_counter()
            start.record()
            for _ in range(1 if mode == "cuda_graph" else repeats):
                runners[role]()
            end.record()
            end.synchronize()
            wall[role].append((time.perf_counter() - clock) * 1000 / repeats)
            gpu[role].append(start.elapsed_time(end) / repeats)
        if trial == 4 and decision_limits and not needs_more_kernel_samples(gpu["baseline"], gpu["candidate"], decision_limits):
            break
    result = {role: summary(values) for role, values in gpu.items()}
    result.update({role + "_wall": summary(values) for role, values in wall.items()})
    result.update(setup_times=setup, timing_mode=mode,
                  timing_scope="captured_device_work" if mode == "cuda_graph" else "host_dispatched_operation")
    if observers is not None:
        from .launch_evidence import observe_execution
        start = time.perf_counter()
        result["executions"] = {label: observe_execution(runners[role], observer, captured_tuners=tuners[role],
            repeats=repeats if mode == "cuda_graph" else 1)
            for role, label, observer in zip(calls, ("reference", "candidate"), observers)}
        setup["launch_observation_s"] = time.perf_counter() - start
    return result


def paired_timings(baseline, candidate, config, repeats=None, *, mode="events", observe_runners=None, prepared=None,
                   decision_limits=None):
    return _timings({"baseline": baseline, "candidate": candidate}, config, config["repeats"] if repeats is None else repeats,
                    mode, observe_runners, {} if prepared is None else prepared, decision_limits)


def tuning_timings(candidate, settings):
    return _timings({"candidate": candidate}, settings, settings["repeats"], "cuda_graph", None, {})


def case_identity(case, index):
    return {"case_index": index, "seed": case.get("seed"), "workload": case.get("workload"),
            **({"generation_step": case["generation_step"]} if "generation_step" in case else {})}


def evaluate_supplemental(candidate, task, fixture, config, *, repeats=3, hardware=None):
    """Run trusted representative and failed cases before any candidate timing."""
    import torch
    from .compat import configure_compiler
    from .supplemental_correctness import validate_case
    configure_compiler(config)
    run = custom_op(candidate, task["kind"], **({"hardware": hardware} if hardware else {}))
    results = []
    with torch.inference_mode():
        for case in sorted(fixture["cases"], key=lambda case: case.get("origin") != "integration_failure"):
            validate_case(case, task, config)
            tensors = []
            for source in case["tensors"]:
                tensor = torch.empty_strided(tuple(source.shape), tuple(source.stride()), dtype=source.dtype, device="cuda")
                tensors.append(tensor.copy_(source))
            saved = [tensor.clone() for tensor in tensors]
            metadata = [(tensor.shape, tensor.stride(), tensor.dtype, tensor.device) for tensor in tensors]
            expected = case["expected"].cuda()
            checks = [check_output(run(tensors, case["scalars"]), expected, config) for _ in range(repeats)]
            intact = all((value.shape, value.stride(), value.dtype, value.device) == meta and torch.equal(value, original)
                         for value, original, meta in zip(tensors, saved, metadata))
            results.append({"case_id": case["id"], "source_path": case["source_path"],
                "seed": case["seed"], "workload": case["workload"], "origin": case["origin"],
                "checks": checks, "correct": intact and all(c["passed"] for c in checks),
                "inputs_unchanged": intact, "shapes": case["shapes"], "phase": "supplemental_correctness",
                "reference": "original_eager_module", "timing_status": "not_measured"})
            if not intact:
                break
    return {"passed": bool(results) and len(results) == len(fixture["cases"])
            and all(case["correct"] for case in results), "cases": results,
            "timing_samples_used": False, "required_case_count": len(fixture["cases"])}


def evaluate_fixture(candidate, task, fixture_path, config, *, stage_times=None, baseline_candidate=None,
                     baseline_candidates=None, observe_fixture=None, measure_archive=True, numerical_only=False):
    import torch
    from .compat import configure_compiler
    configure_compiler(config)
    stage_times = stage_times if stage_times is not None else {}
    stage = time.perf_counter()
    fixture = torch.load(fixture_path, map_location="cpu", weights_only=True)
    stage_times["fixture_load_s"] = time.perf_counter() - stage
    eager_reference = config.get("correctness_scope") == "replaced_layers"
    if eager_reference:
        if config.get("correctness_reference") != "eager":
            raise ValueError("Saved eager outputs require correctness_reference=eager.")
        for case_index, case in enumerate(fixture["cases"]):
            expected_context = {"seed": case.get("seed"), "workload": case.get("workload")}
            if (case.get("workload") or {}).get("mode") == "cached_generation":
                metadata = task.get("cases", [])
                if case_index >= len(metadata) or not metadata[case_index].get("generation_step"):
                    raise ValueError("The task catalog has no generation step for this fixture.")
                expected_context = {key: metadata[case_index][key] for key in ("seed", "workload", "generation_step")}
                if any(case.get(key) != value for key, value in expected_context.items()):
                    raise ValueError("The fixture does not match its recorded generation step.")
            if (case.get("eager_output_reference") != "original_model"
                    or not isinstance(case.get("eager_output"), torch.Tensor)
                    or case.get("eager_output_path") != case.get("source_path")
                    or not task.get("paths") or case.get("source_path") != task["paths"][0]
                    or case.get("eager_output_case_index") != case_index
                    or case.get("case_index") != case_index
                    or case.get("eager_output_context") != expected_context
                    or expected_context["seed"] is None or expected_context["workload"] is None):
                raise ValueError("Fixture lacks the matching original eager output. Generate a new profile.")
    stage = time.perf_counter()
    hardware = require_hardware(discover_hardware(config))
    run = custom_op(candidate, task["kind"], hardware=hardware)
    baseline_run = custom_op(baseline_candidate, task["kind"], hardware=hardware) if baseline_candidate is not None else None
    shape_runs = {key: custom_op(value, task["kind"], hardware=hardware) for key, value in (baseline_candidates or {}).items()}
    stage_times["cuda_extension_s"] = time.perf_counter() - stage
    cases = []
    with torch.inference_mode():
        for case_index, case in enumerate(fixture["cases"]):
            stage = time.perf_counter()
            tensors = [t.cuda() for t in case["tensors"]]
            saved = [t.clone() for t in tensors]
            scalars = case["scalars"]
            if baseline_candidates is not None:
                from .shape_selection import case_signature
                baseline_run = shape_runs.get(digest(case_signature(task["cases"][case_index], task.get("scalars", []))))
            base = ((lambda: baseline_run(tensors, scalars)) if baseline_run is not None else
                    torch.compile(lambda: reference(task["kind"], tensors, scalars), fullgraph=True,
                                  mode=config["compile_mode"]))
            new = lambda: run(tensors, scalars)
            expected = case["eager_output"].cuda() if eager_reference else reference(task["kind"], tensors, scalars)
            baseline_diagnostic = check_output(base(), expected, config)
            checks = [] if eager_reference and baseline_run is None else [baseline_diagnostic]
            # Repeated calls detect stateful and unstable outputs.
            checks.extend(check_output(new(), expected, config) for _ in range(2))
            capture_resources = getattr(run, "_sako_observe_resources", None)
            with capture_resources() if capture_resources else contextlib.nullcontext([]) as resource_records:
                checks.append(check_output(new(), expected, config))
            resources = compiler_resources(candidate, hardware, resource_records)
            unchanged = all(torch.equal(a, b) for a, b in zip(tensors, saved))
            correct = unchanged and all(c["passed"] for c in checks)
            item = {**case_identity(case, case_index), "correct": correct, "checks": checks, "inputs_unchanged": unchanged,
                    "timing_reference": "kept_kernel" if baseline_run is not None else "torch_compile",
                    "shapes": [list(t.shape) for t in tensors], "build": resources}
            if eager_reference:
                item.update(reference="eager_original_model", reference_path=case["eager_output_path"],
                            compiled_vs_eager_diagnostic=baseline_diagnostic)
                if baseline_run is not None:
                    item["kept_baseline_check"] = item.pop("compiled_vs_eager_diagnostic")
            item["stage_times"] = {"pre_timing_checks_s": time.perf_counter() - stage}
            if numerical_only:
                item["timing_status"] = "not_measured_initial_correctness"
            if correct and not numerical_only:
                stage = time.perf_counter()
                mode = config.get("kernel_search", {}).get("timing_mode", "events")
                capture = observe_fixture == case_index
                observe = {"observe_runners": (baseline_run, run)} if capture else {}
                item.update(paired_timings(base, new, config, **({"mode": mode} if mode != "events" else {}), **observe))
                if measure_archive and config.get("kernel_search", {}).get("score_reference") == "paired_compile":
                    if baseline_run is None:
                        ranking_pair = item
                    else:
                        original = torch.compile(lambda: reference(task["kind"], tensors, scalars),
                            fullgraph=True, mode=config["compile_mode"])
                        ranking_pair = paired_timings(original, new, config,
                            **({"mode": mode} if mode != "events" else {}))
                    item["search_reference"] = {
                        "baseline": ranking_pair["baseline"], "candidate": ranking_pair["candidate"],
                        "timing_mode": mode, "kind": "paired_compile", "acceptance_evidence": False}
                item["stage_times"]["timing_s"] = time.perf_counter() - stage
                item["inputs_unchanged"] = all(torch.equal(a, b) for a, b in zip(tensors, saved))
                item["correct"] = item["correct"] and item["inputs_unchanged"]
            cases.append(item)
            tuning = getattr(run, "_sako_tuning_metadata", lambda: [])()
            if tuning:
                item["autotuning"] = tuning
            del tensors, saved, expected, base, new
            if not correct:
                break
    return cases
