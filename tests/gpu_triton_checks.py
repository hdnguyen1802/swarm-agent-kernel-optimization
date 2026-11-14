"""Run explicitly on the GPU. Do not include this file in CPU test discovery."""
from pathlib import Path
import time
import torch
from sako import gpu
from sako.core import load_config, write_json
from sako.hardware import discover_hardware
from triton_fixtures import RMS_SOURCE, LINEAR_SOURCE
from sako.worker import gpu_lock

def main():
    config = load_config("configs/project.json")
    output = Path("runs/triton-native-validation")
    output.mkdir(parents=True, exist_ok=True)
    hardware = discover_hardware(config)
    results = []
    with gpu_lock(), torch.inference_mode():
        for kind, source in (("rmsnorm", RMS_SOURCE), ("linear", LINEAR_SOURCE)):
            item = {"name": "triton_" + kind, "task_id": "probe_" + kind, "strategy": "Test the installed backend.",
                    "backend": "triton", "references": [], "source": source}
            start = time.perf_counter()
            operation = gpu.custom_op(item, kind)
            load_s = time.perf_counter() - start
            checks = 0
            for workload in config["workloads"]:
                for seed in config["correctness_seeds"]:
                    torch.manual_seed(seed)
                    shape = (workload["batch_size"], workload["sequence_length"], 96)
                    x = torch.randn(shape, device="cuda", dtype=torch.float16)
                    if kind == "rmsnorm":
                        tensors, scalars = [x, torch.randn(96, device="cuda", dtype=torch.float16)], [1e-6]
                    else:
                        tensors = [x, torch.randn(64, 96, device="cuda", dtype=torch.float16),
                                   torch.randn(64, device="cuda", dtype=torch.float16)]
                        scalars = []
                    saved = [tensor.clone() for tensor in tensors]
                    expected = gpu.reference(kind, tensors, scalars)
                    actual = operation(tensors, scalars)
                    torch.testing.assert_close(actual, expected, atol=config["atol"], rtol=config["rtol"])
                    assert all(torch.equal(left, right) for left, right in zip(tensors, saved))
                    checks += 1
            # This checks the opaque operation in a fullgraph compiled caller.
            compiled = torch.compile(lambda: operation(tensors, scalars), fullgraph=True)
            torch.testing.assert_close(compiled(), expected, atol=config["atol"], rtol=config["rtol"])
            torch.cuda.synchronize()
            results.append({"kind": kind, "checks": checks, "fullgraph": True,
                "definition_load_s": load_s, "total_setup_and_checks_s": time.perf_counter() - start,
                "note": "Small synthetic shapes check the ABI. Production fixtures are tested separately."})
    write_json(output / "result.json", {"passed": True, "records": results, "hardware": hardware})
    print("Triton ABI, accuracy, input integrity, and compiled callers passed.")

if __name__ == "__main__":
    main()
