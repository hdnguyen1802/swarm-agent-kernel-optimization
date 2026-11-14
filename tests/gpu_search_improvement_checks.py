"""Explicit GPU checks. Never run through CPU unittest discovery."""

from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sako.core import candidate_id, load_config, write_json
from sako.gpu import check_output, custom_op, reference
from sako.model import apply_manifest
from sako.shape_selection import groups
from sako.timing_audit import audit
from triton_fixture_tuning import seed, control_seed
from sako.worker import gpu_lock


def main():
    import torch
    config = load_config(ROOT / "configs/project.json")
    directory = ROOT / "runs/search-improvement-gpu-checks" / uuid.uuid4().hex
    directory.mkdir(parents=True, exist_ok=True)
    print("GPU validation artifacts:", directory, flush=True)
    task = {"id": "linear_control", "kind": "linear", "paths": ["fc"], "cases": [], "scalars": []}
    fixture = {"cases": []}
    with gpu_lock(), torch.inference_mode():
        model = torch.nn.Sequential()
        model.add_module("fc", torch.nn.Linear(96, 64, device="cuda", dtype=torch.float16))
        for workload in config["workloads"]:
            for seed_value in config["correctness_seeds"][:2]:
                generator = torch.Generator(device="cuda").manual_seed(seed_value)
                x = torch.randn((workload["batch_size"], workload["sequence_length"], 96),
                                generator=generator, device="cuda", dtype=torch.float16)
                tensors = [x, model.fc.weight, model.fc.bias]
                metadata = {"shapes": [list(t.shape) for t in tensors], "strides": [list(t.stride()) for t in tensors],
                            "dtypes": [str(t.dtype) for t in tensors], "seed": seed_value, "workload": workload}
                task["cases"].append(metadata)
                fixture["cases"].append({**metadata, "tensors": [t.cpu() for t in tensors], "scalars": [],
                                         "eager_output": model(x).cpu()})
        path = directory / "linear_control.pt"
        torch.save(fixture, path)
        source = seed(task)
        identity = candidate_id(source)
        write_json(directory / (identity + ".json"), source)
        timing = audit(source, task, path, config)
        write_json(directory / "timing-audit.json", timing)
        x = fixture["cases"][0]["tensors"][0].transpose(1, 2)
        control_task = {"id": "strided_control", "kind": "silu", "scalars": [], "cases": [
            {"shapes": [list(x.shape)], "strides": [list(x.stride())], "dtypes": [str(x.dtype)]}]}
        control_path = directory / "strided_control.pt"
        torch.save({"cases": [{"tensors": [x], "scalars": []}]}, control_path)
        copy_timing = audit(control_seed(control_task), control_task, control_path, config, identity_control=True)
        write_json(directory / "strided-timing-audit.json", copy_timing)
        # Check conditional replacement in a real fullgraph compiled module.
        # This ABI test deliberately selects a scope without claiming a speed gain.
        scope = next(iter(groups(task).values()))["signature"]
        manifest = {"entries": [{"task_id": task["id"], "candidate_id": identity,
                                  "paths": ["fc"], "signatures": [scope]}]}
        alternative = torch.nn.Sequential()
        alternative.add_module("fc", model.fc)
        apply_manifest(alternative, manifest, {"tasks": [task]}, directory)
        compiled = torch.compile(alternative, fullgraph=True, dynamic=False, mode=config["compile_mode"])
        checks = []
        for case in fixture["cases"]:
            x = case["tensors"][0].cuda()
            checked = check_output(compiled(x), model(x), config)
            checks.append(checked)
            assert checked["passed"], checked
        # Check the parent seed's FP16 rounding at all operation boundaries.
        parent = seed({"id": "mlp_control", "kind": "mlp", "cases": []})
        operation = custom_op(parent, "mlp")
        parent_checks = []
        for workload in config["workloads"]:
            tensors = [torch.randn((workload["batch_size"], workload["sequence_length"], 96), device="cuda", dtype=torch.float16),
                       torch.randn((64, 96), device="cuda", dtype=torch.float16),
                       torch.randn((64, 96), device="cuda", dtype=torch.float16),
                       torch.randn((96, 64), device="cuda", dtype=torch.float16)]
            checked = check_output(operation(tensors, []), reference("mlp", tensors, []), config)
            parent_checks.append(checked)
            assert checked["passed"], checked
        write_json(directory / "result.json", {"abi_passed": True, "shape_routes": checks,
            "mlp_rounding": parent_checks, "timing_audit_status": timing["status"],
            "strided_control_status": copy_timing["status"],
            "model_gain_claimed": False, "note": "Synthetic ABI checks. Full Qwen model validation remains required."})
        if timing["status"] != "passed" or copy_timing["status"] != "passed":
            raise RuntimeError("Timing controls were unstable. Inspect timing-audit.json before a paid search.")
    print("GPU timing controls, compiled shape routes, and parent rounding passed. No model gain is claimed.")


if __name__ == "__main__":
    main()
