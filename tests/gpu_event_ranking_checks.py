"""Run explicitly on CUDA. Check real profiler attribution and native build cost."""

import copy
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sako.contracts import operation_contract
from sako.core import load_config, read_json, write_json
from sako.hardware import discover_hardware, select_top_tasks
from sako.model import Collector, arguments, forward, inputs, module_kind
from sako.ranking import profile_operations, rank_operations
from sako.worker import gpu_lock
from sako import gpu


class Qwen2MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = torch.nn.Linear(16, 32, bias=False)
        self.up_proj = torch.nn.Linear(16, 32, bias=False)
        self.act_fn = torch.nn.SiLU()
        self.down_proj = torch.nn.Linear(32, 16, bias=False)
    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=32)
        self.embed = torch.nn.Embedding(32, 16)
        self.blocks = torch.nn.Sequential(Qwen2MLP(), Qwen2MLP())
        self.head = torch.nn.Linear(16, 32, bias=True)
    def forward(self, input_ids, **kwargs):
        return (self.head(self.blocks(self.embed(input_ids))),)


def micro(config, output):
    model = TinyModel().eval().cuda().half()
    collector = Collector(config)
    handles, captured = [], {}
    for path, module in model.named_modules():
        kind = module_kind(module)
        if kind is None:
            continue
        def before(mod, args, path=path, kind=kind):
            tensors, scalars = arguments(kind, mod, args[0])
            key = collector.capture(path, kind, tensors, scalars)
            captured[path] = (key, tensors, scalars)
        def after(mod, args, result, path=path, kind=kind):
            key, tensors, scalars = captured[path]
            collector.validate(key, path, gpu.reference(kind, tensors, scalars), result)
        handles.extend([module.register_forward_pre_hook(before), module.register_forward_hook(after)])
    try:
        for workload in config["workloads"]:
            collector.context = {"workload": workload, "seed": config["correctness_seeds"][0]}
            forward(model, inputs(model, workload, config["correctness_seeds"][0]))
    finally:
        for handle in handles:
            handle.remove()
    ids = inputs(model, config["workloads"][0], config["correctness_seeds"][0])
    expected = forward(model, ids).clone()
    records = profile_operations(model, config, collector.path_to_key, output)
    report = rank_operations(list(collector.tasks.values()), records, len(config["workloads"]), config["trials"])
    for key, task in collector.tasks.items():
        task["contract"] = operation_contract(task, collector.validations[key], config)
    selection = select_top_tasks(list(collector.tasks.values()), 4)
    chosen = [collector.tasks[key] for key in selection["selected_task_ids"]]
    assert len(chosen) == 4
    assert all(task["kind"] != "mlp" for task in chosen)
    assert any(any(p.endswith(("gate_proj", "up_proj", "down_proj")) for p in task["paths"]) for task in chosen)
    assert all(task["priority_ms"] > 0 for task in collector.tasks.values() if task["kind"] == "silu")
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in model.modules())
    torch.testing.assert_close(forward(model, ids), expected, rtol=0, atol=0)
    write_json(output / "micro-profile.json", {"passed": True, "report": report, "selection": selection,
        "tasks": list(collector.tasks.values()), "note": "Small validation model. No model speed gain is claimed."})
    print("Real CUDA profiler mapping, nested scopes, SiLU, smaller targets, and unchanged outputs passed.", flush=True)


def builds(config, output):
    hardware = discover_hardware(config)
    saved = read_json("runs/build-validation-rms.json")
    records = []
    for policy in ("validation_full_binding_headers", "tensor_headers_v1"):
        start = time.perf_counter()
        with patch.object(gpu, "INCLUDE_POLICY", policy):
            if policy.startswith("validation_full"):
                with patch.object(gpu, "cuda_source", side_effect=lambda source: source):
                    call = gpu.compile_candidate(saved, hardware)
            else:
                call = gpu.compile_candidate(saved, hardware)
        elapsed = time.perf_counter() - start
        for workload in config["workloads"]:
            for seed in config["correctness_seeds"]:
                torch.manual_seed(seed)
                x = torch.randn(workload["batch_size"], workload["sequence_length"], 3584, device="cuda", dtype=torch.float16)
                weight = torch.randn(3584, device="cuda", dtype=torch.float16)
                actual = call([x, weight], [1e-6])
                expected = gpu.reference("rmsnorm", [x, weight], [1e-6])
                torch.testing.assert_close(actual, expected, atol=config["atol"], rtol=config["rtol"])
        record = {"include_policy": policy, "build_s": elapsed, "correctness_cases": 9}
        records.append(record)
        write_json(output / "build-comparison.json", {"records": records})
        print(str(record), flush=True)
    # A different task with exactly the same source must load the existing binary.
    other = {**saved, "task_id": "another_rms_contract", "name": "same_source_other_task"}
    start = time.perf_counter()
    call = gpu.compile_candidate(other, hardware)
    torch.testing.assert_close(call([x, weight], [1e-6]), expected, atol=config["atol"], rtol=config["rtol"])
    reuse = time.perf_counter() - start
    write_json(output / "build-comparison.json", {"records": records, "cross_task_reuse_s": reuse,
        "note": "One fresh build per policy. Source, hardware, compiler flags, and correctness limits stay fixed."})
    print("Cross-task reuse seconds: " + str(reuse), flush=True)


def main():
    config = load_config("configs/project.json")
    output = Path("runs/event-ranking-micro")
    output.mkdir(parents=True, exist_ok=True)
    with gpu_lock(), torch.inference_mode():
        micro(config, output)
        builds(config, output)


if __name__ == "__main__":
    main()
