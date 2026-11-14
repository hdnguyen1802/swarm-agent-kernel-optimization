"""Greedy generation on the selected causal LM. Both sides use the same driver."""

from contextlib import nullcontext
from functools import lru_cache
import inspect
import time

from .core import digest, summary


def enabled(workload):
    return workload.get("mode") == "cached_generation"


def evaluation_seeds(config, workload):
    seeds = config["correctness_seeds"]
    if enabled(workload) and config.get("kernel_search", {}).get("evaluation") == "staged":
        return seeds[:1]
    return seeds


@lru_cache(maxsize=32)
def prompt_tokens(model_id, revision, prompt):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    ids = tuple(tokenizer(prompt, add_special_tokens=True)["input_ids"])
    if not ids:
        raise ValueError("The configured tokenizer returned an empty prompt.")
    return ids


def input_record(config, workload):
    ids = prompt_tokens(config["model_id"], config["revision"], workload["prompt"])
    return {"model_id": config["model_id"], "tokenizer_revision": config["revision"],
            "prompt": workload["prompt"], "prompt_token_ids": list(ids), "prompt_tokens": len(ids),
            "new_tokens": workload["new_tokens"], "decode_steps": workload["new_tokens"] - 1,
            "selection": "greedy_argmax", "use_cache": True}


def sampled_steps(workload):
    return sorted({0, 1, workload["new_tokens"] - 1})


def step_info(index, length):
    return {"step": index, "phase": "prefill" if index == 0 else "decode",
            "cache_length_before": 0 if index == 0 else length + index - 1,
            "cache_length_after": length + index}


def sliding_window(config):
    return None if getattr(config, "use_sliding_window", None) is False else getattr(config, "sliding_window", None)


class GreedyRequest:
    """An empty cache, the tokenizer's prompt, and exactly one model call per new token."""

    def __init__(self, model, config, workload, call=None):
        import torch
        self.model, self.call = model, model if call is None else call
        self.workload = workload
        self.record = input_record(config, workload)
        self.prompt = torch.tensor(self.record["prompt_token_ids"], device=model.device,
                                   dtype=torch.long).expand(workload["batch_size"], -1).contiguous()
        self.positions = torch.arange(self.prompt.shape[1] + workload["new_tokens"], device=model.device)
        self.attention_mask = torch.ones((self.prompt.shape[0], self.positions.numel()),
                                         device=model.device, dtype=torch.long)
        params = inspect.signature(model.forward).parameters
        self.last_logit = {k: 1 for k in ("num_logits_to_keep", "logits_to_keep") if k in params}
        # Use standard static KV storage when the model advertises support. Other models
        # initialize their own cache on prefill and return it for the next call.
        cache_kind = getattr(model.generation_config, "cache_implementation", None)
        text_config = model.config.get_text_config()
        window = sliding_window(text_config)
        self.fixed_cache = None
        if cache_kind is None and getattr(model, "_supports_static_cache", False) and not window:
            cache_kind = "static"
        if cache_kind in {"static", "hybrid", "sliding_window"}:
            from transformers.generation.utils import NEED_SETUP_CACHE_CLASSES_MAPPING
            capacity = self.prompt.shape[1] + workload["new_tokens"] - 1
            self.fixed_cache = NEED_SETUP_CACHE_CLASSES_MAPPING[cache_kind](
                config=text_config, max_batch_size=self.prompt.shape[0], max_cache_len=capacity,
                device=model.device, dtype=model.dtype)
        elif cache_kind is not None:
            raise ValueError(f"The checkpoint requests an unsupported KV cache type: {cache_kind}.")
        self.record["cache"] = type(self.fixed_cache).__name__ if self.fixed_cache is not None else "model_default"

    def run(self, *, around=None, observe=None, timed=False):
        """Timing starts after input preparation and cache reset. It ends when output IDs are ready."""
        import torch
        ids = self.prompt
        length = ids.shape[1]
        cache = self.fixed_cache
        if cache is not None:
            cache.reset()
        if timed:
            torch.cuda.synchronize()
            start = time.perf_counter()
        for index in range(self.workload["new_tokens"]):
            pos = self.positions[:length] if index == 0 else self.positions[length+index-1:length+index]
            with around(index) if around is not None else nullcontext():
                kwargs = self.model.prepare_inputs_for_generation(ids, past_key_values=cache,
                    attention_mask=self.attention_mask[:, :length + index], cache_position=pos, use_cache=True, **self.last_logit)
                actual = kwargs.get("input_ids")
                if actual is None or actual.shape != (ids.shape[0], length if index == 0 else 1):
                    raise ValueError("The model must prefill once and consume one token per cached decode step.")
                output = self.call(**kwargs, return_dict=True)
                cache = getattr(output, "past_key_values", None)
                if cache is None:
                    raise ValueError("The model did not return past_key_values with use_cache=True.")
                logits = output.logits[:, -1, :]
                token = logits.argmax(dim=-1, keepdim=True)
                ids = torch.cat((ids, token), dim=1)
            if timed and index == 0:
                torch.cuda.synchronize()
                first = time.perf_counter()
            if observe is not None:
                observe(index, logits, token, cache)
        if timed:
            torch.cuda.synchronize()
            end = time.perf_counter()
            return {"first_token_ms": (first - start) * 1000,
                    "mean_decode_ms": (end - first) * 1000 / (self.workload["new_tokens"] - 1),
                    "total_ms": (end - start) * 1000}
        return ids, cache


def visit_inputs(model, config, workload, seed, hooks):
    """Install fixture/check hooks only on the prefill, first decode and final decode."""
    from .model import forward, inputs
    context = {"workload": workload, "seed": seed}
    if not enabled(workload):
        with hooks(context):
            return forward(model, inputs(model, workload, seed))
    request = GreedyRequest(model, config, workload)
    steps = sampled_steps(workload)
    def around(index):
        if index in steps:
            return hooks({**context, "generation_step": step_info(index, request.prompt.shape[1])})
        return nullcontext()
    return request.run(around=around)


def cache_layers(cache):
    keys, values = getattr(cache, "key_cache", None), getattr(cache, "value_cache", None)
    if keys is not None and values is not None and len(keys) == len(values):
        layers = list(zip(keys, values))
    elif isinstance(cache, (tuple, list)):
        layers = cache
    else:
        raise ValueError("The model returned a cache without inspectable per-layer KV tensors.")
    if not layers or any(len(pair) != 2 for pair in layers):
        raise ValueError("Expected one key and one value tensor per cache layer.")
    return layers


def observe_request(request):
    import torch
    scores, storage = [], []
    valid_extent = True
    window = sliding_window(request.model.config.get_text_config())
    fixed_sizes = ([pair[0].shape[-2] for pair in cache_layers(request.fixed_cache)]
                   if request.fixed_cache is not None else None)
    def observe(index, logits, token, cache):
        nonlocal valid_extent
        scores.append(logits.detach().clone())
        lengths = [pair[0].shape[-2] for pair in cache_layers(cache)]
        logical_length = request.prompt.shape[1] + index
        if fixed_sizes is not None:
            valid_extent &= lengths == fixed_sizes
        elif window:
            valid_extent &= all(n in (logical_length, min(logical_length, window)) for n in lengths)
        else:
            valid_extent &= all(n == logical_length for n in lengths)
        storage.append(lengths)
    ids, cache = request.run(observe=observe)
    layers = [(key.detach().clone(), value.detach().clone()) for key, value in cache_layers(cache)]
    return ids, layers, torch.stack(scores), storage, valid_extent


def check_generation(base, candidate, config, *, expected=None):
    """Compare every prediction, greedy token and final cache tensor."""
    import torch
    from .gpu import check_output
    ids, layers, expected_scores, storage, base_extent = expected if expected is not None else observe_request(base)
    other_ids, new_layers, observed_scores, other_storage, candidate_extent = observe_request(candidate)
    expected_length = base.prompt.shape[1] + base.workload["new_tokens"] - 1
    cache_checks = []
    for i, (a, b) in enumerate(zip(layers, new_layers)):
        key, value = (check_output(new, ref, config) for ref, new in zip(a, b))
        cache_checks.append({"layer": i, "stored_positions": [int(a[0].shape[-2]), int(b[0].shape[-2])],
                             "key": key, "value": value, "passed": key["passed"] and value["passed"]})
    scores = check_output(observed_scores, expected_scores, config)
    tokens_match = torch.equal(ids, other_ids)
    return {"scope": "cached_generation", "reference": "torch_compile", "logits": scores,
            "greedy_tokens_match": tokens_match, "baseline_ids": ids[:, base.prompt.shape[1]:].tolist(),
            "candidate_ids": other_ids[:, base.prompt.shape[1]:].tolist(), "steps_checked": len(expected_scores),
            "logical_cache_length": expected_length, "cache_storage_matches_each_step": storage == other_storage,
            "cache_extent_valid": base_extent and candidate_extent,
            "cache_layers": cache_checks,
            "passed": bool(scores["passed"] and tokens_match and base_extent and candidate_extent
                           and storage == other_storage and len(layers) == len(new_layers)
                           and all(check["passed"] for check in cache_checks))}


def timing_summary(rows):
    return {key: summary([sample[key] for sample in rows]) for key in rows[0]}


def paired_generation(base, candidate, config, *, preferred_warmup=None, threshold=None, incumbent=None):
    from .measurement import decode_only, enabled as adaptive, paired_samples
    if adaptive(config):
        result = paired_samples(lambda: base.run(timed=True), lambda: candidate.run(timed=True), config,
                                preferred_warmup=preferred_warmup, threshold=threshold,
                                **({"incumbent_sample": lambda: incumbent.run(timed=True)} if incumbent is not None else {}))
        return {**result, "input": base.record, "timing_scope": "cached_decode" if decode_only(config) else "complete_cached_generation",
                "timing_mode": "synchronized_wall_clock"}
    for _ in range(config["warmup"]):
        base.run()
        candidate.run()
    samples = {"baseline": [], "candidate": []}
    for trial in range(config["trials"]):
        order = (("baseline", base), ("candidate", candidate))
        for role, runner in (order if trial % 2 == 0 else reversed(order)):
            samples[role].append(runner.run(timed=True))
    metrics = {role: timing_summary(rows) for role, rows in samples.items()}
    return {"baseline": metrics["baseline"]["total_ms"], "candidate": metrics["candidate"]["total_ms"],
            "generation_metrics": metrics, "timing_scope": "complete_cached_generation",
            "input": base.record, "timing_mode": "synchronized_wall_clock"}


def compiled_request(model, config, workload):
    import torch
    call = torch.compile(model.forward, fullgraph=True, dynamic=None, mode=config["compile_mode"])
    return GreedyRequest(model, config, workload, call)


def compare_models(model, alternative, config, workload, *, catalog=None, threshold=None, prepared=None, incumbent=None,
                   manifest=None, reference_directory=None, accepted_manifest=None, make_accepted=None, correctness_only=False):
    base = compiled_request(model, config, workload)
    candidate = compiled_request(alternative, config, workload)
    check = check_generation(base, candidate, config, **({"expected": prepared} if prepared is not None else {}))
    if correctness_only:
        return check, {"input": base.record, "timing_status": "not_measured",
                       "timing_reason": "Correctness isolation for a failed model candidate."}
    from .measurement import (checked_decode_record, decode_reference_identity, sample_decode, saved_decode_time, saved_reference,
                              saved_warmup, saved_workload)
    if check["passed"] and saved_reference(config):
        original = saved_workload(catalog["starting_baseline"], model, config, catalog["hardware"], workload, base.record)
        identity = decode_reference_identity(original, manifest)
        metrics = {"baseline": timing_summary(original["samples"])}
        eager = catalog["starting_baseline"].get("pytorch_eager")
        if eager:
            from .measurement import baseline_identity, validate_baseline
            saved = eager["workloads"][config["workloads"].index(workload)]
            identity_eager = {**baseline_identity(model, config, catalog["hardware"], workload, base.record),
                              "execution": "pytorch_eager"}
            metrics["pytorch_eager"] = timing_summary(validate_baseline(saved, identity_eager)["samples"])
        previous = None
        reused = True
        if accepted_manifest:
            previous, metrics["incumbent"], reused = saved_decode_time(reference_directory, original, accepted_manifest,
                config, lambda: compiled_request(make_accepted(), config, workload))
        record = sample_decode(candidate, config, identity)
        measured = checked_decode_record(record, identity, config)
        metrics["candidate"] = measured
        from pathlib import Path
        from .core import write_json
        if reference_directory is not None:
            path = Path(reference_directory) / (digest(identity) + ".json")
            if not path.exists():
                write_json(path, record)
        return check, {"input": base.record, "timing_scope": "cached_decode", "timing_mode": "synchronized_wall_clock",
            "baseline": metrics["baseline"]["mean_decode_ms"], "candidate": measured["mean_decode_ms"],
            "generation_metrics": metrics, "decode_record": record, "accepted_record": previous,
            "sampling": {"method": "saved_reference", "metric": "mean_decode_ms", "timed_runs": 3,
                         "warmup": {"candidate": record["warmup"]}, "starting_record": original["record_hash"],
                         "reference_timed_runs": 0 if reused else 3,
                         "accepted_record": previous["record_hash"] if previous else None}}
    preferred = saved_warmup(catalog, model, config, workload, base.record) if check["passed"] and catalog is not None else None
    timings = paired_generation(base, candidate, config, preferred_warmup=preferred, threshold=threshold,
        **({"incumbent": compiled_request(incumbent, config, workload)} if incumbent is not None else {})) if check["passed"] else {
        "input": base.record, "timing_status": "not_measured", "timing_reason": "Generation correctness failed."}
    return check, timings
