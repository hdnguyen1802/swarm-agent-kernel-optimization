"""Trusted CUDA build, correctness, and timing functions."""

import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import List

from .core import candidate_id, summary, validate_candidate


def doctor():
    import torch
    import transformers
    import triton
    result = {"python": platform.python_version(), "torch": torch.__version__,
              "transformers": transformers.__version__, "triton": triton.__version__,
              "torch_cuda": torch.version.cuda, "cuda_home": os.environ.get("CUDA_HOME")}
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    result.update(gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                  memory_bytes=torch.cuda.get_device_properties(0).total_memory)
    compiler = shutil.which("nvcc")
    if not compiler:
        raise RuntimeError("Set CUDA_HOME and add its bin folder to PATH.")
    result["nvcc"] = subprocess.check_output([compiler, "--version"], text=True).strip()
    if "release 12." not in result["nvcc"]:
        raise RuntimeError("Use CUDA 12.x for sm_70.")
    torch._dynamo.config.suppress_errors = False
    x = torch.randn(128, 256, device="cuda", dtype=torch.float16)
    f = torch.compile(lambda a: a.sin() + a, fullgraph=True)
    torch.testing.assert_close(f(x), x.sin() + x, atol=.002, rtol=.002)
    result["compile_probe"] = True
    result["driver"] = subprocess.check_output([
        "nvidia-smi", "--query-gpu=uuid,driver_version,memory.used,utilization.gpu",
        "--format=csv,noheader"], text=True).strip()
    return result


def compile_candidate(candidate):
    import torch
    from torch.utils.cpp_extension import load_inline
    validate_candidate(candidate)
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0"
    os.environ.setdefault("MAX_JOBS", "2")
    root = Path(__file__).resolve().parents[1]
    name = "sako_" + candidate_id(candidate)
    directory = root / "runs/build" / name
    directory.mkdir(parents=True, exist_ok=True)
    prefix = '''#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <vector>
'''
    declaration = "torch::Tensor sako_run(std::vector<torch::Tensor> inputs, std::vector<double> scalars);"
    library = load_inline(name=name, cpp_sources=declaration,
                          cuda_sources=prefix + candidate["source"], functions=["sako_run"],
                          extra_include_paths=[str(root / "third_party/cutlass/include"),
                                               str(root / "third_party/cutlass/tools/util/include")],
                          extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "-lineinfo"],
                          build_directory=str(directory), verbose=False)
    return library.sako_run


def reference(kind, tensors, scalars):
    import torch
    import torch.nn.functional as F
    x = tensors[0]
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


def custom_op(candidate, kind):
    import torch
    run = compile_candidate(candidate)
    name = "sako::k_" + candidate_id(candidate)

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

    return operation


def check_output(actual, expected, config):
    import torch
    if not isinstance(actual, torch.Tensor):
        return {"passed": False, "reason": "Output is not a tensor."}
    if actual.shape != expected.shape or actual.dtype != expected.dtype or actual.device != expected.device:
        return {"passed": False, "reason": "Output metadata differs."}
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    error = float((actual.float() - expected.float()).abs().max()) if finite else None
    return {"passed": finite and bool(torch.allclose(actual, expected, atol=config["atol"], rtol=config["rtol"])),
            "max_abs_error": error, "atol": config["atol"], "rtol": config["rtol"]}


def paired_timings(baseline, candidate, config, repeats=None):
    """Alternate order. Compilation and warmup are outside the timed region."""
    import torch
    repeats = config["repeats"] if repeats is None else repeats
    for _ in range(config["warmup"]):
        baseline()
        candidate()
    torch.cuda.synchronize()
    gpu = [[], []]
    wall = [[], []]
    for trial in range(config["trials"]):
        for which in ([0, 1] if trial % 2 == 0 else [1, 0]):
            fn = (baseline, candidate)[which]
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            clock = time.perf_counter()
            start.record()
            for _ in range(repeats):
                fn()
            end.record()
            end.synchronize()
            wall[which].append((time.perf_counter() - clock) * 1000 / repeats)
            gpu[which].append(start.elapsed_time(end) / repeats)
    return {"baseline": summary(gpu[0]), "candidate": summary(gpu[1]),
            "baseline_wall": summary(wall[0]), "candidate_wall": summary(wall[1])}


def evaluate_fixture(candidate, task, fixture_path, config):
    import torch
    from .compat import configure_compiler
    configure_compiler(config)
    fixture = torch.load(fixture_path, map_location="cpu", weights_only=True)
    run = custom_op(candidate, task["kind"])
    cases = []
    with torch.inference_mode():
        for case in fixture["cases"]:
            tensors = [t.cuda() for t in case["tensors"]]
            saved = [t.clone() for t in tensors]
            scalars = case["scalars"]
            base = torch.compile(lambda: reference(task["kind"], tensors, scalars), fullgraph=True,
                                 mode=config["compile_mode"])
            new = lambda: run(tensors, scalars)
            expected = reference(task["kind"], tensors, scalars)
            checks = [check_output(base(), expected, config)]
            # Repeated calls detect stateful and unstable outputs.
            checks.extend(check_output(new(), expected, config) for _ in range(3))
            unchanged = all(torch.equal(a, b) for a, b in zip(tensors, saved))
            correct = unchanged and all(c["passed"] for c in checks)
            item = {"correct": correct, "checks": checks, "inputs_unchanged": unchanged,
                    "shapes": [list(t.shape) for t in tensors]}
            if correct:
                item.update(paired_timings(base, new, config))
                item["inputs_unchanged"] = all(torch.equal(a, b) for a, b in zip(tensors, saved))
                item["correct"] = item["correct"] and item["inputs_unchanged"]
            cases.append(item)
            del tensors, saved, expected, base, new
            if not correct:
                break
    return cases
