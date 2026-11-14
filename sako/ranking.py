"""Attribute profiler self GPU time to trusted replacement operations."""

import json
import math
import statistics

from .core import digest

METHOD = "profiler_self_cuda_event_groups"
TARGET = "sako::target::"
CASE = "sako::case::"


def ranking_mode(config):
    return config.get("ranking_mode", "eager_regions")


class OperationScopes:
    """Annotate original eager operations. Do not substitute reference code."""
    original_attention = True

    def __init__(self, paths):
        self.paths, self.active = paths, {}

    def before(self, path):
        import torch
        scope = torch.profiler.record_function(TARGET + self.paths[path] + "::" + path)
        scope.__enter__()
        self.active.setdefault(path, []).append(scope)

    def after(self, path):
        self.active[path].pop().__exit__(None, None, None)

    def close(self):
        # The caller removes hooks first. A failed model forward can leave scopes.
        for scopes in reversed(list(self.active.values())):
            for scope in reversed(scopes):
                scope.__exit__(None, None, None)
        self.active.clear()


def event_record(event):
    """Read only CPU self-device time; adding device events would count it twice."""
    if str(event.device_type).rsplit(".", 1)[-1] != "CPU":
        return None
    name = event.name
    if name.startswith((TARGET, CASE)):
        return None
    elapsed = float(getattr(event, "self_device_time_total", 0.)) / 1000.
    if not math.isfinite(elapsed) or elapsed <= 0:
        return None
    owner, case = None, None
    parent = event.cpu_parent
    while parent is not None:
        if owner is None and parent.name.startswith(TARGET):
            owner = parent.name[len(TARGET):].split("::", 1)
        if parent.name.startswith(CASE):
            case = [int(x) for x in parent.name[len(CASE):].split("::")]
            break
        parent = parent.cpu_parent
    if case is None:
        return None
    shapes = json.loads(json.dumps(event.input_shapes or [], allow_nan=False))
    return {"name": name, "input_shapes": shapes, "time_ms": elapsed,
            "task_id": owner[0] if owner else None, "path": owner[1] if owner else None,
            "workload_index": case[0], "trial": case[1]}


def rank_operations(tasks, records, workload_count, trials):
    """Keep shape groups as evidence; rank targets over all required workloads."""
    groups, totals = {}, {}
    by_id = {task["id"]: task for task in tasks}
    for record in records:
        key = digest([record["task_id"], record["name"], record["input_shapes"]])
        group = groups.setdefault(key, {"name": record["name"], "input_shapes": record["input_shapes"],
            "task_id": record["task_id"], "paths": set(), "calls": 0, "total_gpu_ms": 0.})
        group["calls"] += 1
        group["total_gpu_ms"] += record["time_ms"]
        if record["path"]:
            group["paths"].add(record["path"])
        task = by_id.get(record["task_id"])
        if task and task["kind"] != "mlp":
            identity = (task["id"], record["workload_index"], record["trial"])
            totals[identity] = totals.get(identity, 0.) + record["time_ms"]
    for task in tasks:
        medians = [statistics.median([totals.get((task["id"], workload, trial), 0.)
                   for trial in range(trials)]) for workload in range(workload_count)]
        task.update(priority_method=METHOD, workload_medians_ms=medians, priority_ms=statistics.mean(medians))
        if task["kind"] == "mlp":
            task.update(priority_ms=0., ranking_excluded_reason="composite_region")
        elif any(value <= 0 for value in medians):
            task.update(priority_ms=0., ranking_excluded_reason="missing_profiler_coverage")
    ranked = sorted(groups.values(), key=lambda group: -group["total_gpu_ms"])
    for group in ranked:
        group["paths"] = sorted(group["paths"])
        task = by_id.get(group["task_id"])
        group["replaceable"] = bool(task and not task.get("ranking_excluded_reason"))
        group["exclusion_reason"] = (task.get("ranking_excluded_reason") if task else "no_replacement_contract")
    return {"method": METHOD, "groups": ranked, "samples": records,
            "note": "CPU profiler events supply self GPU time. Device events are not added again. "
                    "Names and shapes form event groups. Matching replacement contracts combine their groups across workloads. "
                    "Composite MLP scopes and unsupported events remain evidence, not targets."}


def profile_operations(model, config, paths, output_dir):
    import torch
    from .model import forward, inputs, install_attention_dispatch, module_kind
    scopes, handles, records = OperationScopes(paths), [], []
    try:
        for path, module in model.named_modules():
            if path in paths and module_kind(module) is not None:
                handles.extend([module.register_forward_pre_hook(lambda mod, args, path=path: scopes.before(path)),
                                module.register_forward_hook(lambda mod, args, output, path=path: scopes.after(path))])
        install_attention_dispatch(timer=scopes)
        with torch.inference_mode():
            for index, workload in enumerate(config["workloads"]):
                ids = inputs(model, workload, config["correctness_seeds"][0])
                for _ in range(config["warmup"]):
                    forward(model, ids)
                torch.cuda.synchronize()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profiler:
                    for trial in range(config["trials"]):
                        with torch.profiler.record_function(CASE + str(index) + "::" + str(trial)):
                            forward(model, ids)
                    torch.cuda.synchronize()
                profiler.export_chrome_trace(str(output_dir / f"ranking-workload-{index}.trace.json"))
                records.extend(record for event in profiler.events() if (record := event_record(event)) is not None)
    finally:
        for handle in handles:
            handle.remove()
        scopes.close()
        install_attention_dispatch()
    return records
