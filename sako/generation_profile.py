"""Attribute the compiled work of real prefill and decode calls to model operations."""

from collections import defaultdict
from contextlib import contextmanager
import gzip
import json

from .core import digest, write_json
from .generation import GreedyRequest, timing_summary


def profile_request(model, config, workload, workload_index, directory, backend, executed,
                    *, starting=None, hardware=None, tasks=()):
    import torch
    from .compiled_ranking import compiled_records
    from .measurement import enabled as adaptive, policy, saved_workload, warmup, profile_stability

    call = torch.compile(model.forward, backend=backend, fullgraph=True, dynamic=None)
    request = GreedyRequest(model, config, workload, call)
    adaptive_mode = adaptive(config)
    history = []
    check = warm = saved = None
    if adaptive_mode:
        from .generation import check_generation, compiled_request
        saved = saved_workload(starting, model, config, hardware, workload, request.record)
        check = check_generation(compiled_request(model, config, workload), request, config)
        write_json(directory / f"generation-{workload_index}-profile-correctness.json", check)
        if not check["passed"]:
            raise RuntimeError("The profiling compiler path changed generated tokens, scores or cache values.")
        warm = warmup(lambda: request.run(timed=True), config, saved["warmup"]["count"])
        samples = saved["samples"]
        count = policy(config)["profile_max_trials"]
    else:
        for _ in range(config["warmup"]):
            request.run()
        samples = [request.run(timed=True) for _ in range(config["trials"])]
        count = config["trials"]
    # Keep a bounded Kineto trace in memory. The cache lives across all chunks of a request.
    definitions, rows = {}, []
    raw_name = f"generation-{workload_index}-gpu-samples.jsonl.gz"
    with gzip.open(directory / raw_name, "wt", encoding="utf-8") as raw:
        for trial in range(count):
            totals = defaultdict(float)
            plans = {}
            profiler = None

            @contextmanager
            def trace_step(index):
                nonlocal profiler
                if profiler is None:
                    profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA], record_shapes=True)
                    profiler.start()
                executed.clear()
                with torch.profiler.record_function(f"sako::generate::{index}"):
                    yield
                if len(executed) != 1:
                    raise RuntimeError("Expected one compiled model graph for this token.")
                plans[index] = executed[0]["plan"]
                if (index + 1) % 8 and index + 1 != workload["new_tokens"]:
                    return
                torch.cuda.synchronize()
                finished = profiler
                profiler.stop()
                profiler = None
                by_step = defaultdict(list)
                markers = {f"sako::generate::{i}": i for i in plans}
                for event in finished.events():
                    if str(event.device_type).rsplit(".", 1)[-1] != "CPU" or event.self_device_time_total <= 0:
                        continue
                    parent = event
                    while parent is not None and parent.name not in markers:
                        parent = parent.cpu_parent
                    if parent is not None:
                        by_step[markers[parent.name]].append(event)
                for step, plan in plans.items():
                    measured = compiled_records(by_step[step], plan, workload_index, trial)
                    if not measured:
                        raise RuntimeError(f"No GPU events recorded for generation step {step}.")
                    raw_step = []
                    for record in measured:
                        elapsed = record.pop("time_ms")
                        record.pop("trial")
                        record["phase"] = "prefill" if step == 0 else "decode"
                        identity = digest(record)
                        definitions.setdefault(identity, record)
                        totals[identity] += elapsed
                        raw_step.append([identity, elapsed])
                    raw.write(json.dumps({"trial": trial, "step": step, "gpu_samples_ms": raw_step}) + "\n")
                plans.clear()

            try:
                request.run(around=trace_step)
            finally:
                if profiler is not None:
                    profiler.stop()
            rows.extend({**definitions[key], "trial": trial, "time_ms": value} for key, value in totals.items()
                        if config.get("generation_objective") != "decode" or definitions[key]["phase"] == "decode")
            if adaptive_mode and (trial + 1) % policy(config)["profile_batch_trials"] == 0:
                from .measurement import saved_reference
                status = profile_stability(tasks, rows, workload_index, trial + 1, config,
                                           warmup_stable=saved["stable"] if saved_reference(config) else warm["stable"])
                history.append(status)
                if not status["needs_more"]:
                    break
    group_name = f"generation-{workload_index}-gpu-groups.json"
    write_json(directory / group_name, definitions)
    metrics = timing_summary(samples)
    total = metrics["total_ms"]
    if config.get("generation_objective") == "decode":
        from .core import summary
        total = summary([r["mean_decode_ms"] * (workload["new_tokens"] - 1) for r in samples])
    return rows, {"workload": workload, "input": request.record, "samples_ms": total["samples_ms"],
                  "median_ms": total["median_ms"], "generation_metrics": metrics,
                  "gpu_samples": raw_name, "gpu_groups": group_name,
                  **({"starting_times_reused": True, "starting_identity_hash": saved["identity_hash"],
                      "profiling_correctness": check, "profiling_warmup": warm,
                      "profiling": {"trials": trial + 1, "assessments": history, "status": history[-1]["status"]}}
                     if adaptive_mode else {})}
