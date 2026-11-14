"""Exercise the staged evaluator and CUDA Graph reuse on real device buffers."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
import torch
from sako.automatic_tuning import prepare
from sako.core import implementation_hash, load_config, write_json
from sako.gpu import paired_timings, reference, tuning_timings
from sako.hardware import discover_hardware
from sako.model import Collector
from sako.staged_evaluation import evaluate
from triton_fixtures import LINEAR_SOURCE


def main():
    folder = ROOT / sys.argv[1]
    folder.mkdir(parents=True, exist_ok=True)
    config = load_config(ROOT / "configs/generation.json")
    decode_config = dict(config)
    config.pop("generation_objective", None)
    config["measurement"] = {"mode": "adaptive"}
    # Each candidate owns its output and graph. Only the compiled reference is shared.
    inputs = torch.ones(32, device="cuda")
    settings = {"warmup": 1, "trials": 5, "repeats": 2}
    shared = {}
    first, second = {"baseline": shared}, {"baseline": shared}
    paired_timings(lambda: inputs + 1, lambda: inputs + 2, settings, mode="cuda_graph", prepared=first)
    inputs.fill_(4)
    timing = paired_timings(lambda: inputs + 1, lambda: inputs * 2, settings, mode="cuda_graph", prepared=second)
    assert timing["setup_times"]["baseline_graph_reused"] and not timing["setup_times"].get("candidate_graph_reused")
    assert all(torch.all(out == n) for out, n in zip(second["outputs"], (5, 8)))
    tuning = tuning_timings(lambda: inputs * 2, settings)
    assert "baseline" not in tuning and len(tuning["candidate"]["samples_ms"]) == 5
    collector = Collector(config)
    weights = {name: torch.randn(64, 64, device="cuda", dtype=torch.float16) for name in ("left", "right")}
    workload = config["workloads"][0]
    for step, rows in enumerate((24, 1)):
        collector.context = {"seed": config["correctness_seeds"][0], "workload": workload,
                             "generation_step": {"phase": "prefill" if step == 0 else "decode", "step": step}}
        for name, weight in weights.items():
            x = torch.randn(1, rows, 64, device="cuda", dtype=torch.float16)
            tensors = [x, weight, x.new_empty(0)]
            key = collector.capture(name, "linear", tensors, [])
            output = reference("linear", tensors, [])
            collector.validate(key, name, output, output)
    task = collector.tasks[key]
    torch.save({"cases": collector.cases[key]}, folder / (key + ".pt"))
    assert len(task["cases"]) == 4
    slow = LINEAR_SOURCE.replace("return output", """other = torch.empty_like(output)
    for i in range(16):
        gemm[(triton.cdiv(m, 32), triton.cdiv(n, 64))](x, weight, bias, other, m, n, k,
            weight.stride(0), weight.stride(1), bias.numel() > 0, 32, 64, 32, num_warps=4, num_stages=2)
    return other""")
    def candidate(source):
        return {"name": "control", "strategy": "Test the evaluator", "task_id": key,
                "backend": "triton", "source": source, "references": []}
    from sako.shape_selection import groups
    baselines = dict.fromkeys(groups(task), candidate(slow))
    hardware = discover_hardware(config)
    job = {"_catalog": {"tasks": [task]}, "focus_paths": ["right"]}
    good, _ = prepare(candidate(LINEAR_SOURCE), config)
    validated = evaluate(good, task, folder, config, hardware, baselines, job=job)
    assert validated["stages"]["outcome"] == "confirmed", validated["stages"]
    assert validated["stages"]["initial_correctness_calls"] == 2
    assert len(validated["shape_decisions"]) == 4 and all(c["correct"] for c in validated["cases"])
    assert any(c["setup_times"].get("prepared_pair_reused") for c in validated["cases"])
    repeated, _ = prepare(candidate(LINEAR_SOURCE + "\n# Repeat the same kernel body in a new module.\n"), config)
    reused = evaluate(repeated, task, folder, config, hardware, baselines, job=job)
    assert reused["stages"]["outcome"] == "confirmed"
    assert any(t.get("cache_reused") for k in reused["automatic_tuning"]["kernels"] for t in k["harness_trials"])
    bad = candidate(LINEAR_SOURCE.replace("acc.to(tl.float16)", "(acc+100).to(tl.float16)"))
    failure = evaluate(bad, task, folder, config, hardware, baselines, job=job)
    assert failure["stages"]["outcome"] == "initial_accuracy_failed"
    assert failure["stages"]["initial_correctness_calls"] == 2
    assert len(list((folder / "failed-inputs" / key).glob("*.json"))) == 2
    rejected = evaluate(candidate(slow), task, folder, config, hardware, {}, job=job)
    assert rejected["stages"]["outcome"] == "screened_slow" and rejected["stages"]["initial_correctness_calls"] == 0
    decode = evaluate(good, task, folder, decode_config, hardware, {}, job=job)
    assert decode["timed_case_indices"] == [i for i,c in enumerate(task["cases"]) if c["generation_step"]["phase"] == "decode"]
    assert all(("speed_screen" in row) == (i in decode["timed_case_indices"]) for i,row in enumerate(decode["cases"]))
    assert decode["stages"]["outcome"] in {"confirmed", "screened_slow"}
    assert decode["stages"]["initial_correctness_calls"] == (2 if decode["stages"]["outcome"] == "confirmed" else 0)
    write_json(folder / "decode.json", decode)
    for name, data in (("correct", validated), ("incorrect", failure), ("slow", rejected)):
        write_json(folder / (name + ".json"), data)
    proof = {"passed": True, "gpu": torch.cuda.get_device_name(), "implementation_hash": implementation_hash(),
             "module_shape_cases": 4, "correct": validated["stages"], "incorrect": failure["stages"],
             "slow": rejected["stages"]}
    write_json(folder / "result.json", proof)
    print(json.dumps(proof), flush=True)


if __name__ == "__main__":
    from sako.worker import gpu_lock
    with gpu_lock(), torch.inference_mode():
        main()
