"""Untimed numerical diagnosis. These observations cannot accept a model."""

from __future__ import annotations

from pathlib import Path
import time
import traceback

from .core import candidate_id, read_json, write_json
from .profiling import _artifact, workspace_path


def diagnostic_case(config):
    """Select one existing case without changing any evaluation setting."""
    settings = config.get("diagnostics", {})
    workload_index = settings.get("workload_index", 0)
    seed_index = settings.get("seed_index", 0)
    limit = settings.get("max_paths", 128)
    if type(workload_index) is not int or not 0 <= workload_index < len(config["workloads"]):
        raise ValueError("Select an existing diagnostic workload.")
    if type(seed_index) is not int or not 0 <= seed_index < len(config["correctness_seeds"]):
        raise ValueError("Select an existing diagnostic seed.")
    if type(limit) is not int or not 1 <= limit <= 256:
        raise ValueError("Trace one to 256 replacement paths.")
    return {"workload_index": workload_index, "workload": dict(config["workloads"][workload_index]),
            "seed_index": seed_index, "seed": config["correctness_seeds"][seed_index], "max_paths": limit}


def compare_outputs(actual, expected, config):
    """Keep declared numerical checks and exact differences separate."""
    import torch
    from .gpu import check_output
    result = check_output(actual, expected, config)
    if (isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor)
            and actual.shape == expected.shape and actual.dtype == expected.dtype and actual.device == expected.device):
        result["exact_equal"] = bool(torch.equal(actual, expected))
        result["different_elements"] = int(torch.ne(actual, expected).sum().item())
        result["elements"] = int(expected.numel())
    else:
        result["exact_equal"] = False
    return result


def tensor_metadata(tensor):
    return {"shape": list(tensor.shape), "strides": list(tensor.stride()), "dtype": str(tensor.dtype)}


class BoundaryTrace:
    """Compare eager outputs at selected replacement boundaries in call order."""

    def __init__(self, config):
        self.config = config
        self.baseline = {}
        self.visits = {}
        self.records = []

    def capture(self, path, module, args, output):
        # Store activations only. No weights or model state are changed.
        self.baseline.setdefault(path, []).append({
            "input": args[0].detach().cpu(), "output": output.detach().cpu()})

    def compare(self, path, candidate, module, args, output):
        index = self.visits.get(path, 0)
        self.visits[path] = index + 1
        baseline = self.baseline.get(path, [])
        record = {"execution_index": len(self.records), "path": path, "call_index": index,
                  "candidate_id": candidate, "input": tensor_metadata(args[0]), "output": tensor_metadata(output)}
        if index >= len(baseline):
            record.update(status="unmatched", reason="The reference did not execute this replacement boundary.")
        else:
            saved = baseline[index]
            record["input_vs_eager_reference"] = compare_outputs(args[0], saved["input"].to(args[0].device), self.config)
            record["output_vs_eager_reference"] = compare_outputs(output, saved["output"].to(output.device), self.config)
            # The original module sees the exact candidate-trajectory input. This
            # separates local kernel error from accumulated upstream differences.
            expected = module.original(*args)
            record["same_input_local_check"] = compare_outputs(output, expected, self.config)
            record["status"] = "compared"
        self.records.append(record)

    def summary(self):
        compared = [record for record in self.records if record["status"] == "compared"]
        def first(field, predicate):
            record = next((record for record in compared if predicate(record[field])), None)
            return {"path": record["path"], "call_index": record["call_index"],
                    "execution_index": record["execution_index"], "candidate_id": record["candidate_id"],
                    "check": record[field]} if record else None
        missing = [{"path": path, "reference_calls": len(calls), "candidate_calls": self.visits.get(path, 0)}
                   for path, calls in self.baseline.items() if len(calls) != self.visits.get(path, 0)]
        return {"scope": "Eager execution at selected replacement boundaries. Compiled internal boundaries were not observed.",
                "compared_boundaries": len(compared), "unmatched_boundaries": missing,
                "first_exact_difference": first("output_vs_eager_reference", lambda check: not check["exact_equal"]),
                "first_tolerance_failure": first("output_vs_eager_reference", lambda check: not check["passed"]),
                "first_same_input_failure": first("same_input_local_check", lambda check: not check["passed"])}


def trace_paths(manifest, catalog, candidate_root, limit):
    tasks = {task["id"]: task for task in catalog["tasks"]}
    from .layer_checks import replacement_paths
    replacements = replacement_paths(manifest, catalog)
    selected, skipped, seen = [], [], set()
    for entry in manifest["entries"]:
        source = read_json(Path(candidate_root) / (entry["candidate_id"] + ".json"))
        if candidate_id(source) != entry["candidate_id"] or source["task_id"] != entry["task_id"]:
            raise ValueError("The diagnostic candidate identity changed.")
        task = tasks[entry["task_id"]]
        for path in entry["paths"]:
            if path in seen:
                continue
            seen.add(path)
            if task["kind"] == "attention":
                skipped.append({"path": path, "reason": "Attention core tracing is unavailable; full compiled logits are still checked."})
            elif len(selected) >= limit:
                skipped.append({"path": path, "reason": "The diagnostic path limit was reached."})
            else:
                selected.append({"path": path, "candidate_id": entry["candidate_id"], "task_id": entry["task_id"],
                    **({"variants": replacements[path]["variants"]} if "variants" in replacements[path] else {})})
    return selected, skipped


def diagnose_model(config, manifest, catalog, candidate_root, output_dir):
    """Reproduce one fixed case, then inspect eager replacement boundaries."""
    import torch
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from .model import load_model, clone_structure, apply_manifest, forward, inputs
    selection = diagnostic_case(config)
    candidate_root = workspace_path(candidate_root)
    paths, skipped = trace_paths(manifest, catalog, candidate_root, selection["max_paths"])
    output = workspace_path(output_dir, output=True) / ("diagnosis-" + str(time.time_ns()))
    output.mkdir(parents=True, exist_ok=False)
    pinned = {**config, "revision": catalog["config"]["revision"]}
    result = {"scope": "model_boundary_diagnostic", "status": "unavailable", "case": selection,
              "hardware_fingerprint": catalog.get("hardware", {}).get("fingerprint"),
              "model_revision": pinned["revision"], "manifest": manifest,
              "atol": config["atol"], "rtol": config["rtol"], "dtype": config["dtype"],
              "acceptance_evidence": False, "timed": False, "comparisons": {},
              "skipped_paths": skipped, "reasons": [], "artifacts": [],
              "note": "This diagnostic does not replace complete evaluation or identify a compiled internal divergence."}
    trace = BoundaryTrace(config)
    handles = []
    original_attention = ALL_ATTENTION_FUNCTIONS["sdpa"]
    phase = "load_model"
    try:
        model = load_model(pinned)
        alternative = clone_structure(model)
        apply_manifest(alternative, manifest, catalog, candidate_root)
        from .generation import enabled as generation_workload, compiled_request, check_generation
        if generation_workload(selection["workload"]):
            phase = "cached_generation"
            with torch.inference_mode():
                reference_request = compiled_request(model, pinned, selection["workload"])
                replacement_request = compiled_request(alternative, pinned, selection["workload"])
                result["comparisons"]["compiled_candidate_vs_compiled_reference"] = check_generation(
                    reference_request, replacement_request, config)
            result["scope"] = "cached_generation_diagnostic"
        else:
            baseline = torch.compile(lambda ids: forward(model, ids), fullgraph=True, dynamic=False,
                                     mode=config["compile_mode"])
            candidate = torch.compile(lambda ids: forward(alternative, ids), fullgraph=True, dynamic=False,
                                      mode=config["compile_mode"])
            with torch.inference_mode():
                ids = inputs(model, selection["workload"], selection["seed"])
                phase = "compiled_logits"
                expected = baseline(ids)
                actual = candidate(ids)
                result["comparisons"]["compiled_candidate_vs_compiled_reference"] = compare_outputs(actual, expected, config)
                # Hooks are installed only after both compiled outputs have been read.
                phase = "eager_reference_trace"
                for entry in paths:
                    path = entry["path"]
                    handles.append(model.get_submodule(path).register_forward_hook(
                        lambda module, args, value, path=path: trace.capture(path, module, args, value)))
                eager_reference = forward(model, ids)
                for handle in handles:
                    handle.remove()
                handles.clear()
                phase = "eager_candidate_trace"
                for entry in paths:
                    path, key = entry["path"], entry["candidate_id"]
                    def compare(module, args, value, path=path, key=key, variants=entry.get("variants")):
                        if variants is not None:
                            from .model import arguments
                            from .shape_selection import select_variant
                            tensors, scalars = arguments(module.kind, module.original, args[0])
                            key = select_variant(variants, tensors, scalars)
                        trace.compare(path, key, module, args, value)
                    handles.append(alternative.get_submodule(path).register_forward_hook(
                        compare))
                eager_candidate = forward(alternative, ids)
                torch.cuda.synchronize()
                result["comparisons"].update({
                    "eager_reference_vs_compiled_reference": compare_outputs(eager_reference, expected, config),
                    "eager_candidate_vs_eager_reference": compare_outputs(eager_candidate, eager_reference, config),
                    "eager_candidate_vs_compiled_reference": compare_outputs(eager_candidate, expected, config)})
        result["status"] = "partial" if skipped else "complete"
    except Exception as error:
        result["status"] = "partial" if result["comparisons"] or trace.records else "unavailable"
        result["reasons"].append({"phase": phase, "error": f"{type(error).__name__}: {error}"})
        (output / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        for handle in handles:
            handle.remove()
        ALL_ATTENTION_FUNCTIONS["sdpa"] = original_attention
    result["boundary_summary"] = trace.summary()
    write_json(output / "boundary_checks.json", trace.records)
    write_json(output / "diagnostic.json", {key: value for key, value in result.items() if key != "artifacts"})
    result["artifacts"] = [_artifact(path) for path in sorted(output.iterdir()) if path.is_file()]
    return result
