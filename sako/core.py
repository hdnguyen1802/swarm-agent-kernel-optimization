"""Contracts and scoring. This module does not import GPU libraries."""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from pathlib import Path

KINDS = {"linear", "rmsnorm", "mlp", "embedding", "attention", "silu"}
CORRECTNESS_DEFAULTS = {"correctness_scope": "full_model", "correctness_reference": "torch_compile"}
DEFAULT_RESIDENT_MAX_JOBS = 64


def correctness_settings(config):
    """Resolve old configs without silently changing their correctness contract."""
    settings = {key: config.get(key, default) for key, default in CORRECTNESS_DEFAULTS.items()}
    pair = (settings["correctness_scope"], settings["correctness_reference"])
    if (any(not isinstance(value, str) for value in pair)
            or pair not in {("full_model", "torch_compile"), ("replaced_layers", "eager")}):
        raise ValueError("Use full_model correctness against torch_compile, or replaced_layers correctness against eager.")
    return settings


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


def require_broad_selection(config):
    """The former legacy value already used broad selection."""
    if config.get("optimization_policy", "legacy") != "legacy":
        raise ValueError("This optimization_policy is no longer supported. Remove it from the launch config "
                         "and create a new run and profile for broad selection.")


def load_config(path, *, historical=False):
    c = read_json(path)
    if not historical:
        require_broad_selection(c)
        c.pop("profiler", None)
    c.pop("optimization_policy", None)
    if c.get("kernel_backend", "triton") != "triton":
        raise ValueError("Only Triton candidates are supported. Use kernel_backend=triton.")
    if c.get("dtype") != "float16" or c.get("attention") not in {"sdpa", "auto"}:
        raise ValueError("Use FP16 and select SDPA or the model default for attention.")
    if c.get("fullgraph") is not True:
        raise ValueError("A compiled baseline must use fullgraph=true.")
    if c.get("model_reference") != "torch_compile":
        raise ValueError("The complete model reference must be torch_compile.")
    correctness_settings(c)
    from .measurement import policy as measurement_policy
    measurement_policy(c)
    if c.get("triton_integration", "opaque") not in ("opaque", "direct"):
        raise ValueError("Use opaque or direct for triton_integration.")
    if c.get("model_trial_base", "kept") not in ("kept", "accepted"):
        raise ValueError("Use kept or accepted for model_trial_base.")
    if c.get("containing_region_policy", "required") not in ("required", "diagnostic"):
        raise ValueError("Use required or diagnostic containing_region_policy.")
    if c.get("model_speed_scope", "all_workloads") not in ("all_workloads", "targeted_workloads"):
        raise ValueError("Use all_workloads or targeted_workloads for model_speed_scope.")
    if c.get("model_speed_scope") == "targeted_workloads" and not c.get("kernel_search", {}).get("shape_scoped"):
        raise ValueError("Targeted model speed requires explicit shape selection.")
    supplemental = c.get("supplemental_correctness", {})
    if (not isinstance(supplemental, dict) or type(supplemental.get("enabled", False)) is not bool
            or type(supplemental.get("max_paths", 3)) is not int or not 1 <= supplemental.get("max_paths", 3) <= 32
            or type(supplemental.get("max_failure_cases", 3)) is not int or not 1 <= supplemental.get("max_failure_cases", 3) <= 32):
        raise ValueError("Use supplemental_correctness.enabled and one to 32 representative paths and failure cases.")
    if c.get("ranking_mode", "eager_regions") not in {"eager_regions", "profiler_events", "compiled_profiler"}:
        raise ValueError("Use eager_regions, profiler_events, or compiled_profiler for ranking_mode.")
    if c.get("measurement", {}).get("mode") == "adaptive" and c.get("ranking_mode") != "compiled_profiler":
        raise ValueError("Adaptive measurement requires compiled_profiler ranking.")
    kernel = c.get("kernel_search", {})
    if (not isinstance(kernel, dict) or type(kernel.get("enabled", False)) is not bool
            or type(kernel.get("min_speedup", 1.01)) not in (int, float)
            or not math.isfinite(kernel.get("min_speedup", 1.01)) or kernel.get("min_speedup", 1.01) <= 1):
        raise ValueError("Use a valid kernel_search setting with min_speedup above one.")
    if kernel.get("enabled") and c["agent"].get("search_mode") != "swarm":
        raise ValueError("Kernel-first selection requires swarm mode.")
    if type(kernel.get("shape_scoped", False)) is not bool or kernel.get("timing_mode", "events") not in {"events", "cuda_graph"}:
        raise ValueError("Use a boolean shape_scoped and events or cuda_graph timing.")
    if kernel.get("shape_scoped") and not kernel.get("enabled"):
        raise ValueError("Shape selection requires kernel-first search.")
    if kernel.get("evaluation", "standard") not in {"standard", "staged"}:
        raise ValueError("Select standard or staged operation evaluation.")
    if c.get("generation_objective") is not None:
        if (c["generation_objective"] != "decode" or kernel.get("evaluation") != "staged"
                or not kernel.get("enabled") or kernel.get("score_reference") != "paired_compile"
                or c.get("measurement", {}).get("mode") != "adaptive"
                or c.get("agent", {}).get("search_mode") != "swarm"
                or not all(w.get("mode") == "cached_generation" and w.get("new_tokens", 0) >= 2 for w in c["workloads"])):
            raise ValueError("Decode search requires cached generation, staged kernel tests, paired compile timings and adaptive swarm search.")
    if kernel.get("evaluation") == "staged" and not (kernel.get("shape_scoped") and
            c.get("correctness_scope") == "replaced_layers" and c.get("model_trial_base") == "accepted"):
        raise ValueError("Staged evaluation requires shape selection, eager layer checks and complete model trials.")
    if kernel.get("score_reference", "fixed_anchor") not in ("fixed_anchor", "paired_compile"):
        raise ValueError("Use fixed_anchor or paired_compile for score_reference.")
    if type(kernel.get("path_scoped", False)) is not bool or kernel.get("path_scoped") and not kernel.get("shape_scoped"):
        raise ValueError("Path scopes require shape selection.")
    for section in ("tuning", "timing_audit"):
        if not isinstance(c.get(section, {}), dict) or type(c.get(section, {}).get("enabled", False)) is not bool:
            raise ValueError(f"Use a boolean {section}.enabled.")
    from .automatic_tuning import policy as tuning_policy
    tuning_policy(c)
    for key in ("warmup", "trials", "repeats", "max_fixture_cases"):
        if type(c.get(key)) is not int or c[key] <= 0:
            raise ValueError(f"Invalid {key}.")
    if c["trials"] < 5 or len(c["correctness_seeds"]) < 2:
        raise ValueError("Use at least five trials and two test seeds.")
    if not c.get("workloads"):
        raise ValueError("Define model workloads.")
    for w in c["workloads"]:
        if isinstance(w, dict) and w.get("mode") == "cached_generation":
            if (set(w) != {"mode", "batch_size", "prompt", "new_tokens"}
                    or type(w["batch_size"]) is not int or w["batch_size"] < 1
                    or not isinstance(w["prompt"], str) or not w["prompt"].strip()
                    or type(w["new_tokens"]) is not int or w["new_tokens"] < 2):
                raise ValueError("Cached generation requires a prompt, a positive batch size and at least two new tokens.")
            if len(c["workloads"]) != 1 or c.get("ranking_mode") != "compiled_profiler":
                raise ValueError("Use a dedicated compiled-profile run for the cached generation workload.")
            if c["max_fixture_cases"] < 3:
                raise ValueError("Generation needs fixture space for prefill, early decode and late decode.")
            continue
        if set(w) != {"batch_size", "sequence_length"} or any(type(v) is not int or v <= 0 for v in w.values()):
            raise ValueError("Invalid model workload.")
    for key in ("atol", "rtol", "min_speedup", "max_p95_ratio"):
        if type(c.get(key)) not in (int, float) or not math.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f"Invalid {key}.")
    if c["min_speedup"] < 1 or c["max_p95_ratio"] < 1:
        raise ValueError("Invalid speed gates.")
    if type(c.get("report_eager_baseline", False)) is not bool:
        raise ValueError("report_eager_baseline must be a boolean.")
    a = c["agent"]
    if not historical:
        from .agent import validate_provider
        validate_provider(a)
    if a.get("budget_allocation", "default") not in {"default", "profile_gpu_time"}:
        raise ValueError("Select default or profile_gpu_time budget allocation.")
    if (type(a.get("coordinator_target_chars", a["context_chars"])) is not int
            or not 0 < a.get("coordinator_target_chars", a["context_chars"]) <= a["context_chars"]):
        raise ValueError("coordinator_target_chars must be positive and no larger than context_chars.")
    for flag in ("search_parent_regions", "performance_refinement", "require_archive_trial", "revisit_with_budget", "candidate_merging"):
        if type(a.get(flag, False)) is not bool:
            raise ValueError(f"Use a boolean {flag}.")
    target_selection = a.get("target_selection", "flat")
    if target_selection not in {"flat", "bottleneck_tree"}:
        raise ValueError("Use flat or bottleneck_tree for target_selection.")
    if target_selection == "bottleneck_tree" and (
            c.get("ranking_mode") != "compiled_profiler" or a.get("search_mode") != "swarm" or not kernel.get("enabled")):
        raise ValueError("Bottleneck trees require compiled profiling and kernel-first swarm search.")
    if target_selection == "flat":
        a.setdefault("top_k_operations", 4)
    a.setdefault("max_stagnant_rounds", 3)
    a.setdefault("adaptive_budget", True)
    a.setdefault("search_mode", "swarm")
    if a["search_mode"] != "swarm":
        raise ValueError("Only swarm search is supported. Use a current config for a new run.")
    if type(a["adaptive_budget"]) is not bool:
        raise ValueError("adaptive_budget must be true or false.")
    for key in ("max_calls",):
        if a.get(key) is None and a["adaptive_budget"]:
            continue
        if type(a.get(key)) is not int or a[key] <= 0:
            raise ValueError(f"Invalid agent limit: {key}.")
    for key in ("writers", "max_output_tokens",
                "max_total_tokens", "timeout_seconds", "context_chars", "top_k_operations",
                "max_stagnant_rounds"):
        if key == "top_k_operations" and target_selection == "bottleneck_tree":
            continue
        if type(a.get(key)) is not int or a[key] <= 0:
            raise ValueError(f"Invalid agent limit: {key}.")
    if not a["adaptive_budget"] or not 1 <= a["writers"] <= 5:
        raise ValueError("The swarm uses an adaptive budget and one to five writer slots.")
    if a["islands"] != ["memory", "compute", "scheduling", "fusion", "registers"]:
        raise ValueError("Use the five supported swarm specialties.")
    a.setdefault("max_correction_attempts", 2)
    if type(a["max_correction_attempts"]) is not int or a["max_correction_attempts"] != 2:
        raise ValueError("Each swarm hypothesis permits one initial attempt and exactly two available correction attempts.")
    for key, default in (("archive_size", 4), ("lesson_limit", 8), ("lesson_context_chars", 12000),
                         ("coordinator_max_input_tokens", 150000)):
        a.setdefault(key, default)
        if type(a[key]) is not int or a[key] <= 0:
            raise ValueError(f"Invalid swarm limit: {key}.")
    if a.get("max_task_experiments") is not None and (
            type(a["max_task_experiments"]) is not int or a["max_task_experiments"] <= 0):
        raise ValueError("max_task_experiments must be positive or null.")
    maximum = 5
    a.setdefault("design_portfolio", False)
    a.setdefault("model_repair_fraction", 0.)
    if type(a["design_portfolio"]) is not bool or type(a["model_repair_fraction"]) not in (int, float) or not 0 <= a["model_repair_fraction"] <= .5:
        raise ValueError("Use a boolean design_portfolio and model_repair_fraction between zero and 0.5.")
    if type(a.get("max_parallel_calls")) is not int or not 1 <= a["max_parallel_calls"] <= maximum:
        raise ValueError(f"Use one to {maximum} concurrent API calls.")
    if type(a.get("max_retries")) is not int or not 0 <= a["max_retries"] <= 3:
        raise ValueError("Use zero to three API retries.")
    if type(a.get("max_repair_attempts")) is not int or not 1 <= a["max_repair_attempts"] <= 10:
        raise ValueError("Use one to ten repair attempts per candidate chain.")
    worker = c["worker"]
    if worker.get("archive_gpu") is not None:
        device = worker["archive_gpu"]
        if (type(device) is not int or device < 0 or device == worker.get("gpu")
                or worker.get("transport") != "ssh" or not kernel.get("enabled")
                or kernel.get("score_reference") != "paired_compile"):
            raise ValueError("Use a different SSH GPU for archive ranking with paired_compile enabled.")
    if type(worker.get("persistent", False)) is not bool:
        raise ValueError("worker.persistent must be true or false.")
    max_jobs = worker.get("resident_max_jobs", DEFAULT_RESIDENT_MAX_JOBS)
    if type(max_jobs) is not int or not 1 <= max_jobs <= 128:
        raise ValueError("Use one to 128 jobs per resident process.")
    idle = worker.get("resident_idle_seconds", 600)
    if type(idle) not in (int, float) or not math.isfinite(idle) or not 30 <= idle <= 3600:
        raise ValueError("Use a resident idle limit from 30 to 3600 seconds.")
    return c


def validate_candidate(c, backend=None):
    required = {"name", "task_id", "strategy", "source", "references", "backend"}
    if not isinstance(c, dict) or set(c) != required:
        raise ValueError(f"Candidate keys must be {sorted(required)}.")
    for key in ("name", "task_id"):
        if not isinstance(c[key], str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,96}", c[key]):
            raise ValueError(f"Invalid {key}.")
    if not isinstance(c["references"], list) or any(not isinstance(p, str) for p in c["references"]):
        raise ValueError("Declare reference paths as a list of strings.")
    if not isinstance(c["strategy"], str) or len(c["strategy"]) > 4000:
        raise ValueError("Invalid strategy.")
    source = c["source"]
    if not isinstance(source, str) or not source.strip() or len(source.encode()) > 131072:
        raise ValueError("Kernel source must be at most 128 KiB.")
    if "sako_run" not in source:
        raise ValueError("Missing sako_run entry point.")
    if c["backend"] != "triton" or backend not in (None, "triton"):
        raise ValueError("Only Triton candidates are supported.")
    from .triton_backend import validate_source
    validate_source(source)
    return c


def candidate_id(c):
    validate_candidate(c)
    return digest({"task_id": c["task_id"], "source": c["source"], "backend": c["backend"]})[:24]


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
