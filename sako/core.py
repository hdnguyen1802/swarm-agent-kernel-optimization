"""Contracts and scoring. This module does not import GPU libraries."""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from pathlib import Path

KINDS = {"linear", "rmsnorm", "mlp", "embedding", "attention"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def implementation_hash():
    package = Path(__file__).resolve().parent
    files = sorted([*package.glob("*.py"), *package.glob("prompts/*.txt")])
    return digest({p.relative_to(package).as_posix(): p.read_text(encoding="utf-8") for p in files})


def load_config(path):
    c = read_json(path)
    if c.get("dtype") != "float16" or c.get("attention") != "sdpa":
        raise ValueError("Use FP16 and SDPA for this model adapter.")
    if c.get("fullgraph") is not True:
        raise ValueError("A compiled baseline must use fullgraph=true.")
    if c.get("model_reference") != "torch_compile":
        raise ValueError("The complete model reference must be torch_compile.")
    for key in ("warmup", "trials", "repeats", "max_fixture_cases"):
        if type(c.get(key)) is not int or c[key] <= 0:
            raise ValueError(f"Invalid {key}.")
    if c["trials"] < 5 or len(c["correctness_seeds"]) < 2:
        raise ValueError("Use at least five trials and two test seeds.")
    if not c.get("workloads"):
        raise ValueError("Define model workloads.")
    for w in c["workloads"]:
        if set(w) != {"batch_size", "sequence_length"} or any(type(v) is not int or v <= 0 for v in w.values()):
            raise ValueError("Invalid model workload.")
    for key in ("atol", "rtol", "min_speedup", "max_p95_ratio"):
        if type(c.get(key)) not in (int, float) or not math.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f"Invalid {key}.")
    if c["min_speedup"] <= 1 or c["max_p95_ratio"] < 1:
        raise ValueError("Invalid speed gates.")
    a = c["agent"]
    for key in ("writers", "generations", "migration_interval", "max_calls", "max_output_tokens",
                "max_total_tokens", "timeout_seconds", "context_chars", "stall_generations"):
        if type(a.get(key)) is not int or a[key] <= 0:
            raise ValueError(f"Invalid agent limit: {key}.")
    if a["writers"] != 8 or a["islands"] != ["memory", "parallel", "compute", "fusion"]:
        raise ValueError("This search uses eight writers in four islands.")
    if type(a.get("max_parallel_calls")) is not int or not 1 <= a["max_parallel_calls"] <= 8:
        raise ValueError("Use one to eight concurrent API calls.")
    if type(a.get("max_retries")) is not int or not 0 <= a["max_retries"] <= 3:
        raise ValueError("Use zero to three API retries.")
    return c


def validate_candidate(c):
    required = {"name", "task_id", "strategy", "uses_cutlass", "source", "references"}
    if not isinstance(c, dict) or set(c) != required:
        raise ValueError(f"Candidate keys must be {sorted(required)}.")
    for key in ("name", "task_id"):
        if not isinstance(c[key], str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,96}", c[key]):
            raise ValueError(f"Invalid {key}.")
    if type(c["uses_cutlass"]) is not bool or not isinstance(c["references"], list):
        raise ValueError("Declare CUTLASS use and reference paths.")
    if not isinstance(c["strategy"], str) or len(c["strategy"]) > 4000:
        raise ValueError("Invalid strategy.")
    source = c["source"]
    if not isinstance(source, str) or not source.strip() or len(source.encode()) > 131072:
        raise ValueError("CUDA source must be at most 128 KiB.")
    if "sako_run" not in source:
        raise ValueError("Missing sako_run entry point.")
    # This catches common policy errors. It is not a security sandbox.
    banned = r"\b(system|popen|fork|execve|socket|fopen|freopen|ifstream|ofstream|dlopen|cudaDeviceReset)\s*\(|#\s*include\s*[<\"](?:Python|pybind11|unistd|sys/)|PYBIND11_MODULE|\bgetenv\s*\("
    if re.search(banned, source):
        raise ValueError("Source violates the candidate policy.")
    if re.search(r"\b(?:torch|at)::(?:matmul|mm|addmm|linear|softmax|silu|layer_norm|native_layer_norm|scaled_dot_product_attention)\s*\(", source):
        raise ValueError("Use CUDA, not an ATen compute fallback.")
    if "cutlass/" in source and not c["uses_cutlass"]:
        raise ValueError("CUTLASS use must be declared.")
    return c


def candidate_id(c):
    validate_candidate(c)
    return digest({"task_id": c["task_id"], "source": c["source"]})[:24]


def summary(samples):
    if not samples or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in samples):
        raise ValueError("Timing samples must be finite and positive.")
    return {"median_ms": statistics.median(samples),
            "p95_ms": sorted(samples)[math.ceil(len(samples) * .95) - 1], "samples_ms": samples}


def score_cases(cases, config):
    reasons, ratios = [], []
    for i, case in enumerate(cases):
        if case.get("correct") is not True:
            reasons.append(f"Case {i}: correctness failed.")
        try:
            base = summary(case["baseline"]["samples_ms"])
            new = summary(case["candidate"]["samples_ms"])
            if min(len(base["samples_ms"]), len(new["samples_ms"])) < config["trials"]:
                raise ValueError("Missing trials.")
            ratios.append(base["median_ms"] / new["median_ms"])
            if ratios[-1] < config["min_speedup"]:
                reasons.append(f"Case {i}: gain is below {config['min_speedup']}x.")
            if new["p95_ms"] > base["p95_ms"] * config["max_p95_ratio"]:
                reasons.append(f"Case {i}: p95 latency regressed.")
        except (KeyError, TypeError, ValueError):
            reasons.append(f"Case {i}: invalid timing samples.")
    if not cases:
        reasons.append("No measured cases.")
    return {"accepted": not reasons, "reasons": reasons,
            "speedup": math.exp(statistics.mean(map(math.log, ratios))) if ratios else None}


def select_nonoverlapping(entries):
    """A parent replacement excludes child replacements in the same model."""
    chosen, paths = [], []
    for entry in sorted(entries, key=lambda e: e.get("priority", 0), reverse=True):
        current = entry["paths"]
        if any(a == b or a.startswith(b + ".") or b.startswith(a + ".") for a in current for b in paths):
            continue
        chosen.append(entry)
        paths.extend(current)
    return chosen
