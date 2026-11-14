"""Declared sampling rules for model timing, profiling and warmup.

Operation races keep their separate trial count. Old configs keep fixed sampling.
"""

import copy
import json
import math
from pathlib import Path
import statistics

from .core import digest, implementation_hash, read_json, summary, write_json


DEFAULTS = {"mode": "adaptive", "version": 1, "baseline_trials": 5,
            "profile_initial_trials": 5, "profile_batch_trials": 5, "profile_max_trials": 15,
            "model_initial_pairs": 5, "model_max_pairs": 10,
            "timing_relative_spread": .02, "near_threshold_ratio": .005,
            "profile_relative_spread": .10, "profile_top_k": 5,
            "warmup_window": 3, "warmup_max_runs": 12, "warmup_relative_spread": .02}


def enabled(config):
    return config.get("measurement", {}).get("mode") == "adaptive"


def decode_only(config):
    return config.get("generation_objective") == "decode"


def timing_metric(config):
    return "mean_decode_ms" if decode_only(config) else "total_ms"


def saved_reference(config):
    return config.get("measurement", {}).get("model_sampling") == "saved_reference"


def policy(config):
    value = config.get("measurement", {})
    if not isinstance(value, dict) or set(value) - (set(DEFAULTS) | {"model_sampling"}):
        raise ValueError("Unknown measurement policy fields.")
    if value.get("mode", "fixed") == "fixed":
        if set(value) - {"mode"}:
            raise ValueError("Fixed sampling uses the original warmup and trials fields.")
        return {"mode": "fixed"}
    defaults = DEFAULTS
    if "model_sampling" in value:
        if not saved_reference(config) or not decode_only(config):
            raise ValueError("Saved model timings require the cached decode workload.")
        defaults = {**DEFAULTS, "baseline_trials": 3, "warmup_max_runs": 3, "warmup_window": 3}
    result = {**defaults, **value}
    if result["mode"] != "adaptive":
        raise ValueError("Use fixed or adaptive measurement mode.")
    for key in ("version", "baseline_trials", "profile_initial_trials", "profile_batch_trials",
                "model_initial_pairs", "model_max_pairs"):
        if type(result[key]) is not int or result[key] != defaults[key]:
            raise ValueError(f"The declared measurement policy requires {key}={defaults[key]}.")
    if saved_reference(config) and (result["warmup_window"] != 3 or result["warmup_max_runs"] != 3):
        raise ValueError("Use exactly three complete-model warmups with saved references.")
    for key, low, high in (("profile_max_trials", 5, 30), ("profile_top_k", 1, 20),
                           ("warmup_window", 3, 6), ("warmup_max_runs", 3, 30)):
        if type(result[key]) is not int or not low <= result[key] <= high:
            raise ValueError(f"Invalid measurement setting: {key}.")
    if result["profile_max_trials"] % 5 or result["warmup_max_runs"] < result["warmup_window"]:
        raise ValueError("Use complete five-trial profile batches and enough warmup runs.")
    for key in ("timing_relative_spread", "near_threshold_ratio", "profile_relative_spread",
                "warmup_relative_spread"):
        if type(result[key]) not in (int, float) or not math.isfinite(result[key]) or not 0 < result[key] <= .25:
            raise ValueError(f"Invalid measurement setting: {key}.")
    return result


def relative_spread(values):
    median = summary(values)["median_ms"]
    return (max(values) - min(values)) / median


def warmup(sample, config, preferred=None):
    """Keep every warmup sample. Stop at the first stable eligible window."""
    rules = policy(config)
    rows = []
    minimum = rules["warmup_window"] if preferred is None else preferred
    if type(minimum) is not int or not rules["warmup_window"] <= minimum <= rules["warmup_max_runs"]:
        raise ValueError("The saved warmup count is outside the declared policy.")
    stable = False
    for _ in range(rules["warmup_max_runs"]):
        rows.append(sample())
        values = [row[timing_metric(config)] for row in rows[-rules["warmup_window"]:]]
        summary(values)
        stable = len(rows) >= minimum and relative_spread(values) <= rules["warmup_relative_spread"]
        if stable:
            break
    return {"count": len(rows), "stable": stable, "runs": rows,
            "saved_count_reused": preferred, "phase": "warmup"}


def validate_warmup(value, config):
    rules = policy(config)
    rows = value["runs"]
    count = value["count"]
    preferred = value["saved_count_reused"]
    minimum = rules["warmup_window"] if preferred is None else preferred
    if (type(count) is not int or count != len(rows) or type(minimum) is not int
            or not rules["warmup_window"] <= minimum <= count <= rules["warmup_max_runs"]
            or value.get("phase") != "warmup"):
        raise ValueError("Invalid warmup run evidence.")
    values = [row[timing_metric(config)] for row in rows]
    summary(values)
    stable = relative_spread(values[-rules["warmup_window"]:]) <= rules["warmup_relative_spread"]
    if value["stable"] is not stable:
        raise ValueError("The warmup decision differs from its raw runs.")
    if not stable and count < rules["warmup_max_runs"]:
        raise ValueError("Warmup stopped before its declared stability rule.")
    return stable


def assess_pairs(baseline, candidate, config, *, threshold=None, warmup_stable=True):
    """Use all pairs and a predeclared conservative interval on log ratios.

    Each of the two planned looks uses a 99% Student-t interval. Critical values
    round up the NIST table at df=4 and df=9. Timing must also pass the spread test.
    This is an estimate under the paired log-ratio model, not an independence claim.
    https://www.itl.nist.gov/div898/handbook/eda/section3/eda3672.htm
    """
    rules = policy(config)
    base, new = summary(baseline), summary(candidate)
    count = len(baseline)
    if len(candidate) != count or count not in (5, 10):
        raise ValueError("Adaptive model timing needs five or ten complete pairs.")
    required = config["min_speedup"] if threshold is None else threshold
    logs = [math.log(a / b) for a, b in zip(baseline, candidate)]
    mean = statistics.mean(logs)
    error = (4.605 if count == 5 else 3.251) * statistics.stdev(logs) / math.sqrt(count)
    lower, upper = math.exp(mean - error), math.exp(mean + error)
    ratio = base["median_ms"] / new["median_ms"]
    spreads = {"baseline": relative_spread(baseline), "candidate": relative_spread(candidate)}
    stable = warmup_stable and all(value <= rules["timing_relative_spread"] for value in spreads.values())
    state = ("pass" if stable and lower > required and ratio >= required else
             "reject" if stable and upper < required and ratio < required else "uncertain")
    near = abs(ratio / required - 1) <= rules["near_threshold_ratio"]
    return {"state": state, "pairs": count, "stable": stable, "relative_spread": spreads,
            "median_speedup": ratio, "required_speedup": required, "interval": [lower, upper],
            "interval_method": "99% Student-t estimate for paired log ratios; two planned looks",
            "near_threshold": near, "needs_more": count == 5 and (near or state == "uncertain")}


def paired_samples(base_sample, candidate_sample, config, *, preferred_warmup=None, threshold=None, incumbent_sample=None):
    rules = policy(config)
    metric = timing_metric(config)
    runners = {"baseline": base_sample, "candidate": candidate_sample}
    if incumbent_sample is not None:
        runners["incumbent"] = incumbent_sample
    warm = {role: warmup(call, config, preferred_warmup) for role, call in runners.items()}
    samples = {role: [] for role in runners}
    rounds, order_log = [], []
    count = rules["model_initial_pairs"]
    while True:
        for index in range(len(samples["baseline"]), count):
            order = list(runners)
            offset = index % len(order)
            order = order[offset:] + order[:offset]
            if len(order) == 3 and (index // 3) % 2:
                order.reverse()
            order_log.append(order)
            for role in order:
                samples[role].append(runners[role]())
        def compare(role, required):
            return assess_pairs([r[metric] for r in samples[role]], [r[metric] for r in samples["candidate"]],
                                config, threshold=required, warmup_stable=warm[role]["stable"] and warm["candidate"]["stable"])
        assessment = compare("baseline", threshold)
        previous = compare("incumbent", 1.) if incumbent_sample is not None else None
        rounds.append({**assessment, **({"incumbent": previous} if previous else {})})
        if not (assessment["needs_more"] or (previous and previous["needs_more"])):
            break
        count = rules["model_max_pairs"]
    metrics = {role: {key: summary([r[key] for r in rows]) for key in rows[0]}
               for role, rows in samples.items()}
    return {"baseline": metrics["baseline"][metric], "candidate": metrics["candidate"][metric],
            "generation_metrics": metrics,
            "sampling": {"policy": rules, "metric": metric, "warmup": warm, "rounds": rounds, "assessment": assessment,
                         "pair_order": order_log, "all_collected_samples_used": True}}


class ForwardRequest:
    """The normal compiled forward path with the existing CUDA event clock."""
    def __init__(self, model, config, workload, call=None):
        import torch
        from .model import forward, inputs
        self.ids = inputs(model, workload, config["correctness_seeds"][-1])
        self.call = call if call is not None else torch.compile(
            lambda ids: forward(model, ids), fullgraph=True, dynamic=False, mode=config["compile_mode"])
        self.record = {"input_ids": self.ids.tolist(), "shape": list(self.ids.shape),
                       "stride": list(self.ids.stride()), "dtype": str(self.ids.dtype), "use_cache": False}

    def run(self, timed=False):
        import torch
        if not timed:
            return self.call(self.ids)
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        self.call(self.ids)
        end.record()
        end.synchronize()
        return {"total_ms": start.elapsed_time(end)}


def normal_request(model, config, workload):
    from .generation import compiled_request, enabled as generating
    from .model import install_attention_dispatch
    # Integration uses this same dispatcher. Set it before measuring the reference,
    # including when the worker runs without a resident ModelSession.
    install_attention_dispatch()
    return compiled_request(model, config, workload) if generating(workload) else ForwardRequest(model, config, workload)


def baseline_identity(model, config, hardware, workload, record):
    import torch
    identity = {"schema": 1, "implementation_hash": implementation_hash(),
            "hardware_fingerprint": hardware["fingerprint"],
            "model_id": config["model_id"], "revision": config["revision"], "dtype": config["dtype"],
            "model_config": model.config.to_dict(), "attention": config["attention"],
            "attention_implementation": getattr(model.config, "_attn_implementation", None),
            "attention_dispatch": "sako_sdpa",
            "workload": copy.deepcopy(workload), "input": copy.deepcopy(record),
            "compile_mode": config["compile_mode"], "fullgraph": config["fullgraph"],
            "dynamic": None if workload.get("mode") == "cached_generation" else False,
            "inference_mode": torch.is_inference_mode_enabled(),
            "training": bool(getattr(model, "training", False)),
            "autocast_enabled": torch.is_autocast_enabled(),
            "math_settings": {"float32_matmul_precision": torch.get_float32_matmul_precision(),
                "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "fp16_reduced_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "deterministic": torch.are_deterministic_algorithms_enabled()},
            "model_reference": config["model_reference"], "measurement_policy": policy(config)}
    # HF configs can use integer label keys. JSON turns those keys into strings.
    # Compare the canonical saved form on both sides of every cache lookup.
    if decode_only(config):
        identity["timing_metric"] = timing_metric(config)
    return json.loads(json.dumps(identity))


def validate_baseline(value, expected):
    if (not isinstance(value, dict) or value.get("identity") != expected
            or value.get("identity_hash") != digest(expected)
            or value.get("record_hash") != digest({k: v for k, v in value.items() if k != "record_hash"})):
        raise ValueError("Saved starting times have different settings or changed content.")
    values = value.get("samples", [])
    if len(values) != expected["measurement_policy"]["baseline_trials"]:
        raise ValueError("Saved starting times lack the required timing samples.")
    timing = summary([r[expected.get("timing_metric", "total_ms")] for r in values])
    if value["samples_ms"] != timing["samples_ms"] or value["median_ms"] != timing["median_ms"]:
        raise ValueError("The starting time summary differs from its raw runs.")
    validate_warmup(value["warmup"], {"measurement": expected["measurement_policy"],
                    "generation_objective": "decode" if expected.get("timing_metric") == "mean_decode_ms" else None})
    return value


def starting_baseline(model, config, hardware, directory):
    """Time the original normal compiler path once. Save five runs per workload."""
    import torch
    with torch.inference_mode():
        measured = _starting_baseline(model, config, hardware, directory)
        if decode_only(config) and config.get("report_eager_baseline", False):
            measured["pytorch_eager"] = starting_eager(model, config, hardware, directory, measured)
        return measured


def starting_eager(model, config, hardware, directory, compiled):
    """Save normal PyTorch timings on the same device, inputs and cache path."""
    import torch
    from .generation import GreedyRequest, check_generation, compiled_request, timing_summary
    file = Path(directory) / "pytorch-baseline.json"
    saved = read_json(file) if file.exists() else {"workloads": []}
    results = []
    for index, workload in enumerate(config["workloads"]):
        request = GreedyRequest(model, config, workload)
        identity = {**baseline_identity(model, config, hardware, workload, request.record),
                    "execution": "pytorch_eager"}
        if index < len(saved["workloads"]):
            cached = validate_baseline(saved["workloads"][index], identity)
            if relative_spread(cached["samples_ms"]) > policy(config)["timing_relative_spread"]:
                raise ValueError("The saved PyTorch baseline timing is unstable. Keep these samples and prepare a new run.")
            results.append(cached)
            continue
        warm = warmup(lambda: request.run(timed=True), config)
        samples = [request.run(timed=True) for _ in range(policy(config)["baseline_trials"])]
        metrics = timing_summary(samples)
        result = {"identity": identity, "identity_hash": digest(identity), "workload": workload,
                  "input": request.record, "warmup": warm, "samples": samples,
                  "samples_ms": metrics["mean_decode_ms"]["samples_ms"],
                  "median_ms": metrics["mean_decode_ms"]["median_ms"], "generation_metrics": metrics,
                  "timing_path": "pytorch_eager",
                  "stable": relative_spread(metrics["mean_decode_ms"]["samples_ms"]) <= policy(config)["timing_relative_spread"]}
        reference = compiled["workloads"][index].get("reference_output")
        if reference:
            expected = torch.load(Path(directory) / reference["file"], weights_only=True, map_location="cuda")
            result["compiled_equivalence"] = check_generation(compiled_request(model, config, workload),
                                                              request, config, expected=expected)
        result["record_hash"] = digest(result)
        results.append(result)
        write_json(file, {"workloads": results})
        if not result["stable"]:
            raise ValueError("The PyTorch baseline timing is unstable. Its raw samples were saved. Prepare a new run.")
    if len(saved["workloads"]) > len(results):
        raise ValueError("Saved PyTorch times have extra workloads.")
    return {"workloads": results}


def _starting_baseline(model, config, hardware, directory):
    file = Path(directory) / "starting_baseline.json"
    saved = read_json(file) if file.exists() else {"workloads": []}
    results = []
    for index, workload in enumerate(config["workloads"]):
        runner = normal_request(model, config, workload)
        identity = baseline_identity(model, config, hardware, workload, runner.record)
        if index < len(saved["workloads"]):
            results.append(validate_baseline(saved["workloads"][index], identity))
            continue
        warm = warmup(lambda: runner.run(timed=True), config)
        samples = [runner.run(timed=True) for _ in range(policy(config)["baseline_trials"])]
        metrics = {key: summary([r[key] for r in samples]) for key in samples[0]}
        result = {"identity": identity, "identity_hash": digest(identity), "workload": workload,
                  "input": runner.record, "warmup": warm, "samples": samples,
                  "samples_ms": metrics[timing_metric(config)]["samples_ms"], "median_ms": metrics[timing_metric(config)]["median_ms"],
                  "generation_metrics": metrics, "timing_path": "normal_torch_compile",
                  "stable": relative_spread(metrics[timing_metric(config)]["samples_ms"]) <= policy(config)["timing_relative_spread"]}
        if workload.get("mode") == "cached_generation" and config.get("kernel_search", {}).get("evaluation") == "staged":
            import torch
            from .generation import observe_request
            from .transport import _file_hash
            target = Path(directory) / f"generation-reference-{index}.pt"
            temporary = target.with_suffix(".pt.tmp")
            torch.save(observe_request(runner), temporary)
            temporary.replace(target)
            result["reference_output"] = {"file": target.name, "sha256": _file_hash(target),
                                          "identity_hash": result["identity_hash"]}
        result["record_hash"] = digest(result)
        results.append(result)
        write_json(file, {"workloads": results})
    if len(saved["workloads"]) > len(results):
        raise ValueError("Saved starting times have extra workloads.")
    return {"workloads": results}


def saved_workload(baseline, model, config, hardware, workload, record):
    index = config["workloads"].index(workload)
    try:
        value = baseline["workloads"][index]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("Adaptive profiling requires saved starting times.") from error
    return validate_baseline(value, baseline_identity(model, config, hardware, workload, record))


def sample_decode(request, config, identity):
    """Save three full warmups and three timed generations from fresh caches."""
    run = lambda: request.run(timed=True)
    warm = warmup(run, config)
    samples = [run() for _ in range(3)]
    record = {"identity": identity, "warmup": warm, "samples": samples}
    record["record_hash"] = digest(record)
    return record


def checked_decode_record(record, identity, config):
    if (record.get("identity") != identity or record.get("record_hash") !=
            digest({k:v for k,v in record.items() if k != "record_hash"}) or len(record.get("samples", [])) != 3):
        raise ValueError("The saved decode measurements have different settings or samples.")
    validate_warmup(record["warmup"], config)
    from .generation import timing_summary
    return timing_summary(record["samples"])


def decode_reference_identity(original, manifest):
    return {"workload_settings": original["identity_hash"], "gpu": original["identity"]["hardware_fingerprint"],
            "starting_record": original["record_hash"],
            "manifest": digest(manifest)}


def saved_decode_time(directory, original, manifest, config, make_request):
    identity = decode_reference_identity(original, manifest)
    path = Path(directory) / (digest(identity) + ".json")
    if path.is_file():
        record = read_json(path)
        metrics = checked_decode_record(record, identity, config)
        return record, metrics, True
    record = sample_decode(make_request(), config, identity)
    metrics = checked_decode_record(record, identity, config)
    write_json(path, record)
    return record, metrics, False


def profile_stability(tasks, records, workload_index, count, config, *, warmup_stable=True):
    """Compare target order in disjoint early and late groups of profile trials."""
    from .compiled_ranking import rank_compiled
    rules = policy(config)
    orders, costs = [], []
    for indices in (list(range(count // 2)), list(range(count // 2, count)), list(range(count))):
        rows = [{**r, "workload_index": 0, "trial": indices.index(r["trial"])}
                for r in records if r["workload_index"] == workload_index and r["trial"] in indices]
        local = copy.deepcopy(tasks)
        rank_compiled(local, rows, 1, len(indices))
        ranked = sorted((t for t in local if t["priority_ms"] > 0), key=lambda t: (-t["priority_ms"], t["id"]))
        orders.append([t["id"] for t in ranked[:rules["profile_top_k"]]])
        costs.append({t["id"]: t["priority_ms"] for t in ranked})
    # Use every trial to check the actual target costs, not only their rank order.
    spreads, per_trial = {}, []
    for trial in range(count):
        rows = [{**r, "workload_index": 0, "trial": 0} for r in records
                if r["workload_index"] == workload_index and r["trial"] == trial]
        local = copy.deepcopy(tasks)
        rank_compiled(local, rows, 1, 1)
        per_trial.append({t["id"]: t["priority_ms"] for t in local})
    for key in orders[-1]:
        values = [cost[key] for cost in per_trial]
        spreads[key] = relative_spread(values) if all(v > 0 for v in values) else None
    stable = warmup_stable and bool(orders[-1]) and orders[0] == orders[1] == orders[2] and all(
        value is not None and value <= rules["profile_relative_spread"] for value in spreads.values())
    return {"stable": stable, "trials": count, "early_targets": orders[0], "late_targets": orders[1],
            "targets": orders[2], "relative_spread": spreads,
            "needs_more": not stable and count < rules["profile_max_trials"],
            "status": "stable" if stable else "collect_more" if count < rules["profile_max_trials"] else "uncertain"}


def saved_warmup(catalog, model, config, workload, record):
    if not enabled(config):
        return None
    baseline = catalog.get("starting_baseline")
    # Validate against the current model input, compiler settings and the catalog's
    # hardware identity. The worker checks that hardware identity before any job.
    value = saved_workload(baseline, model, config, catalog["hardware"], workload, record)
    return value["warmup"]["count"]


def paired_forward(model, baseline, candidate, config, workload, catalog, *, threshold=None):
    base = ForwardRequest(model, config, workload, baseline)
    new = ForwardRequest(model, config, workload, candidate)
    result = paired_samples(lambda: base.run(timed=True), lambda: new.run(timed=True), config,
                            preferred_warmup=saved_warmup(catalog, model, config, workload, base.record),
                            threshold=threshold)
    result.pop("generation_metrics")
    return {**result, "timing_mode": "events", "timing_scope": "complete_model_forward"}


def cached_baseline_report(config, catalog, hardware):
    """Check and expose the prepared reference. Do not load weights or run a model."""
    import torch
    from types import SimpleNamespace
    from .preparation import validate_catalog
    from .generation import enabled as generating, input_record
    validate_catalog(config, catalog, hardware)
    pinned = {**config, "revision": catalog["config"]["revision"]}
    values = catalog.get("starting_baseline", {}).get("workloads", [])
    if len(values) != len(config["workloads"]):
        raise ValueError("The prepared profile lacks the saved starting reference.")
    cases = []
    # This is the same declared math setting used by load_model.
    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.inference_mode():
        for index, (value, workload) in enumerate(zip(values, config["workloads"])):
            identity = value["identity"]
            model_config = copy.deepcopy(identity["model_config"])
            proxy = SimpleNamespace(config=SimpleNamespace(to_dict=lambda: model_config,
                _attn_implementation=identity["attention_implementation"]))
            record = copy.deepcopy(value["input"])
            if generating(workload):
                record.update(input_record(pinned, workload))
            # A prepared catalog fixes the cache implementation and the seeded
            # forward IDs. Source, checkpoint and software identity protect those
            # choices. Generation additionally checks the actual tokenizer IDs.
            saved = validate_baseline(value, baseline_identity(proxy, pinned, hardware, workload, record))
            profiles = catalog.get("compiled_profile", {}).get("workload_times", [])
            check = profiles[index].get("profiling_correctness") if index < len(profiles) else None
            cases.append({"workload": workload, "correct": bool(check and check.get("passed")),
                "checks": [check] if check else [], "baseline": summary(saved["samples_ms"]),
                "generation_metrics": {"baseline": saved["generation_metrics"]}, "input": saved["input"],
                "warmup": saved["warmup"], "starting_identity_hash": saved["identity_hash"]})
    return {"scope": "starting_model_baseline", "config": pinned, "cases": cases,
            "starting_times_reused": True, "additional_timed_runs": 0, "acceptance_evidence": False,
            "timing_path": "normal_torch_compile", "manifest": {"entries": []},
            "decision": {"accepted": False, "speedup": None,
                "reasons": ["Saved starting reference only. No optimized model is tested."]}}


def incumbent_reference(measured, catalog):
    return {"context": {"catalog_hash": digest(catalog), "implementation_hash": implementation_hash(),
                        "hardware_fingerprint": catalog["hardware"]["fingerprint"]},
            "measurement": copy.deepcopy(measured)}


def compare_saved_incumbent(manifest, reference, cases, config, catalog):
    """Compare gains against the same original model without extra GPU runs."""
    from .model_acceptance import score_model_cases, target_workloads
    if reference is None or reference.get("context") != incumbent_reference({}, catalog)["context"]:
        raise ValueError("The accepted model reference has different source, hardware or profile settings.")
    measured = reference["measurement"]
    previous = measured["result"]
    from .preparation import profile_settings
    if (measured["manifest"] != manifest or profile_settings(previous["config"]) != profile_settings(config)
            or not previous["decision"]["accepted"]):
        raise ValueError("The saved accepted model has a different manifest or test settings.")
    decision = score_model_cases(previous["cases"], config, target_workloads(config, manifest, catalog))
    if not decision["accepted"]:
        raise ValueError("The saved accepted model does not pass its declared validation rules.")
    results, reasons = [], []
    uncertain = False
    for old, new in zip(previous["cases"], cases):
        if old["workload"] != new["workload"]:
            raise ValueError("Accepted and proposed model workloads differ.")
        old_logs = [math.log(a / b) for a, b in zip(old["baseline"]["samples_ms"], old["candidate"]["samples_ms"])]
        new_logs = [math.log(a / b) for a, b in zip(new["baseline"]["samples_ms"], new["candidate"]["samples_ms"])]
        difference = statistics.mean(new_logs) - statistics.mean(old_logs)
        # Use df=4, the most conservative critical value allowed by either sample count.
        error = 4.605 * math.sqrt(statistics.variance(old_logs) / len(old_logs)
                                 + statistics.variance(new_logs) / len(new_logs))
        lower, upper = math.exp(difference - error), math.exp(difference + error)
        old_ratio = summary(old["baseline"]["samples_ms"])["median_ms"] / summary(old["candidate"]["samples_ms"])["median_ms"]
        new_ratio = summary(new["baseline"]["samples_ms"])["median_ms"] / summary(new["candidate"]["samples_ms"])["median_ms"]
        old_tail = summary(old["candidate"]["samples_ms"])["p95_ms"] / summary(old["baseline"]["samples_ms"])["p95_ms"]
        new_tail = summary(new["candidate"]["samples_ms"])["p95_ms"] / summary(new["baseline"]["samples_ms"])["p95_ms"]
        state = "pass" if lower > 1 and new_ratio > old_ratio else "reject" if upper < 1 else "uncertain"
        if new_tail > old_tail * config["max_p95_ratio"]:
            state = "reject"
        if state != "pass":
            uncertain |= state == "uncertain"
            reasons.append("The gain over the accepted model is " + state + ".")
        results.append({"workload": new["workload"], "accepted_model_speedup": old_ratio,
                        "proposed_model_speedup": new_ratio, "incremental_speedup": new_ratio / old_ratio,
                        "interval": [lower, upper], "normalized_p95_ratio": new_tail / old_tail, "state": state})
    if len(results) != len(config["workloads"]):
        raise ValueError("The accepted model reference lacks workload coverage.")
    return {"reference": "saved_accepted_model_normalized_to_original", "manifest": manifest, "cases": results,
            "additional_timed_runs": 0,
            "decision": {"accepted": not reasons, "reasons": reasons,
                "timing_status": "accepted" if not reasons else "uncertain" if uncertain else "rejected",
                "speedup": math.exp(statistics.mean(math.log(r["incremental_speedup"]) for r in results))}}
