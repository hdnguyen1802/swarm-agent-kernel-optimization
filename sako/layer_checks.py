"""Check replacement outputs on each original layer's inputs."""

from __future__ import annotations

import math
import traceback


def _finite_nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def replacement_paths(manifest, catalog):
    """Validate the exact replacement sites. No path limit is permitted."""
    tasks = {task["id"]: task for task in catalog["tasks"]}
    from .shape_selection import validate_entries
    validate_entries(manifest["entries"], tasks)
    selected = {}
    for entry in manifest["entries"]:
        task = tasks.get(entry["task_id"])
        if task is None or not entry.get("paths"):
            raise ValueError("Every replacement needs a known task and module paths.")
        for path in entry["paths"]:
            if "signatures" in entry:
                value = selected.setdefault(path, {"task_id": entry["task_id"], "candidate_id": entry["candidate_id"],
                                                  "kind": task["kind"], "variants": []})
                value["variants"].append({"candidate_id": entry["candidate_id"], "signatures": entry["signatures"]})
            else:
                selected[path] = {"task_id": entry["task_id"], "candidate_id": entry["candidate_id"], "kind": task["kind"]}
    return selected


class ReferenceStateError(RuntimeError):
    """A rejected candidate left a reference input or weight unsafe to reuse."""


class LayerOutputChecker:
    """Run candidate kernels on each original eager module's actual inputs."""

    def __init__(self, config, selected, operations):
        self.config, self.selected, self.operations = config, selected, operations
        self.records, self.calls, self.attention_inputs = [], {}, {}
        self.context = {}

    def compare(self, path, tensors, scalars, expected, *, attention=False):
        import torch
        from .gpu import check_output
        entry = self.selected[path]
        if "variants" in entry:
            from .shape_selection import select_variant
            key = select_variant(entry["variants"], tensors, scalars)
            if key is None:
                self.calls[path] = self.calls.get(path, 0) + 1
                self.records.append({"path": path, "task_id": entry["task_id"], "candidate_id": None,
                    "kind": entry["kind"], "reference": "eager", "same_input": True, "inputs_unchanged": True,
                    "passed": True, "original_fallback": True, "check": {"passed": True,
                    "reason": "This shape uses the unchanged original layer."}})
                return
            entry = {k: v for k, v in entry.items() if k != "variants"}
            entry["candidate_id"] = key
        call_index = self.calls.get(path, 0)
        self.calls[path] = call_index + 1
        record = {"path": path, **entry, "call_index": call_index,
                  "input_shapes": [list(t.shape) for t in tensors],
                  "input_strides": [list(t.stride()) for t in tensors],
                  "input_dtypes": [str(t.dtype) for t in tensors],
                  "input_scalars": list(scalars),
                  "reference": "eager", "same_input": True, "inputs_unchanged": False,
                  "passed": False}
        metadata = [(tensor.shape, tensor.stride(), tensor.dtype, tensor.device) for tensor in tensors]
        saved = [tensor.detach().clone() for tensor in tensors]
        try:
            actual = self.operations[entry["candidate_id"]](tensors, scalars)
            if attention and isinstance(actual, torch.Tensor) and actual.ndim == 4:
                actual = actual.transpose(1, 2).contiguous()
            record["check"] = check_output(actual, expected, self.config)
            unchanged = [(tensor.shape, tensor.stride(), tensor.dtype, tensor.device) == meta
                         and torch.equal(tensor, old) for tensor, old, meta in zip(tensors, saved, metadata)]
            record["inputs_unchanged"] = all(unchanged)
            record["mutated_input_indices"] = [index for index, value in enumerate(unchanged) if not value]
            record["passed"] = record["check"]["passed"] and record["inputs_unchanged"]
        except Exception as error:
            record.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        finally:
            # Keep later reference layers and all shared model weights unchanged
            # after a rejected mutation. Copies run outside every timed region.
            restored = []
            for index, (tensor, old, meta) in enumerate(zip(tensors, saved, metadata)):
                try:
                    if (tensor.shape, tensor.stride(), tensor.dtype, tensor.device) != meta:
                        record["passed"] = False
                        record.setdefault("restoration_errors", []).append({"input_index": index, "error": "Input metadata changed."})
                    elif not torch.equal(tensor, old):
                        tensor.copy_(old)
                        if not torch.equal(tensor, old):
                            raise RuntimeError("Input restoration did not restore the original values.")
                        restored.append(index)
                except Exception as error:
                    record["passed"] = False
                    record.setdefault("restoration_errors", []).append({"input_index": index, "error": str(error)})
            record["restored_input_indices"] = restored
            self.records.append(record)
        if record.get("restoration_errors"):
            raise ReferenceStateError("The reference state could not be restored after a rejected input mutation.")
        from .supplemental_correctness import record_required_failure
        record_required_failure(record, tensors, scalars, expected, self.context, attention=attention)

    def capture(self, path, kind, tensors, scalars):
        """Adapter used by the existing eager SDPA dispatch."""
        if path in self.selected:
            self.attention_inputs[path] = (tensors, scalars)
        return path

    def validate(self, key, path, functional_output, original_output):
        # The original HF SDPA output is authoritative. The dispatch's separate
        # functional output is not substituted for it.
        if path in self.selected:
            tensors, scalars = self.attention_inputs.pop(path)
            self.compare(path, tensors, scalars, original_output, attention=True)

    def result(self, seed, error=None):
        expected = list(self.selected)
        observed = list(dict.fromkeys(record["path"] for record in self.records))
        missing = [path for path in expected if path not in observed]
        failed = list(dict.fromkeys(record["path"] for record in self.records if not record["passed"]))
        errors = [record.get("check", {}).get("max_abs_error") for record in self.records]
        errors = [value for value in errors if _finite_nonnegative(value)]
        result = {"seed": seed, "scope": "replaced_layers", "reference": "eager",
                  "passed": not missing and not failed and error is None,
                  "reference_state_intact": not any(record.get("restoration_errors") for record in self.records),
                  "expected_paths": expected, "observed_paths": observed,
                  "missing_paths": missing, "failed_paths": failed,
                  "max_abs_error": max(errors) if errors else None,
                  "mismatched_elements": sum(record.get("check", {}).get("mismatched_elements", 0)
                                              for record in self.records),
                  "layers": self.records}
        if error is not None:
            result["error"] = error
        return result


def measure_layer_case(model, manifest, catalog, operations, config, workload, seed, *, alternative=None):
    """Visit every replaced site on an unchanged original eager trajectory."""
    from .model import arguments, install_attention_dispatch
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    selected = replacement_paths(manifest, catalog)
    checker = LayerOutputChecker(config, selected, operations)
    checker.context = {"workload": workload, "seed": seed}
    region_checker = None
    if alternative is not None:
        from .hierarchy import containing_regions
        region_selected, region_operations = {}, {}
        replaced_paths = list(selected)
        for region in containing_regions(catalog, manifest):
            for path in region["paths"]:
                if not any(p.startswith(path + ".") for p in replaced_paths):
                    continue
                key = "region:" + path
                module = alternative.get_submodule(path)
                region_selected[path] = {"task_id": region["id"], "candidate_id": key, "kind": "mlp"}
                region_operations[key] = lambda tensors, scalars, module=module: module(tensors[0])
        region_checker = LayerOutputChecker(config, region_selected, region_operations)
        region_checker.context = {"workload": workload, "seed": seed}
    if not selected:
        return checker.result(seed)
    from contextlib import contextmanager
    from .generation import visit_inputs

    @contextmanager
    def check_step(context):
        checker.context = context
        if region_checker is not None:
            region_checker.context = context
        handles, path_attributes = [], []
        missing_attribute = object()
        original_attention = ALL_ATTENTION_FUNCTIONS["sdpa"]
        try:
            for path, entry in selected.items():
                if entry["kind"] == "attention":
                    module = model.get_submodule(path.removesuffix(".__core__"))
                    path_attributes.append((module, getattr(module, "_sako_path", missing_attribute)))
                    module._sako_path = path.removesuffix(".__core__")
                else:
                    module = model.get_submodule(path)
                    def after(original, args, output, path=path, kind=entry["kind"]):
                        tensors, scalars = arguments(kind, original, args[0])
                        checker.compare(path, tensors, scalars, output)
                        # Return None so the original output continues downstream.
                    handles.append(module.register_forward_hook(after))
            if path_attributes:
                install_attention_dispatch(collector=checker)
            if region_checker is not None:
                for path in region_checker.selected:
                    module = model.get_submodule(path)
                    def region_after(original, args, output, path=path):
                        tensors, scalars = arguments("mlp", original, args[0])
                        region_checker.compare(path, tensors, scalars, output)
                    handles.append(module.register_forward_hook(region_after))
            yield
        finally:
            for handle in handles:
                handle.remove()
            ALL_ATTENTION_FUNCTIONS["sdpa"] = original_attention
            for module, previous in path_attributes:
                if previous is missing_attribute:
                    delattr(module, "_sako_path")
                else:
                    module._sako_path = previous

    error = None
    try:
        visit_inputs(model, config, workload, seed, check_step)
    except Exception as exception:
        error = {"error": f"{type(exception).__name__}: {exception}", "traceback": traceback.format_exc()}
    result = checker.result(seed, error)
    if region_checker is not None and region_checker.selected:
        parent = region_checker.result(seed, error)
        role = "diagnostic" if config.get("containing_region_policy", "required") == "diagnostic" else "required"
        parent["acceptance_role"] = role
        result["region_checks"] = parent
        if role == "required":
            result["passed"] = result["passed"] and parent["passed"]
        result["reference_state_intact"] = result["reference_state_intact"] and parent["reference_state_intact"]
        region_inputs_safe = all(record.get("inputs_unchanged") is True for record in parent["layers"])
        result["passed"] = result["passed"] and result["reference_state_intact"] and region_inputs_safe
        result["layers"].extend({**record, "scope": "containing_region", "acceptance_role": role} for record in parent["layers"])
    return result
