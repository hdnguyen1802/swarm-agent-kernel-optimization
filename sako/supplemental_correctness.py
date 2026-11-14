"""Trusted correctness-only cases. They never replace timing fixtures."""

from contextlib import contextmanager
from contextvars import ContextVar
import copy
import hashlib
from pathlib import Path
import uuid

from .core import digest, implementation_hash, read_json, write_json
from .generation import evaluation_seeds

_CAPTURE = ContextVar("sako_required_failure_capture", default=None)


def enabled(config):
    return config.get("supplemental_correctness", {}).get("enabled", False)


def representative_paths(task, limit=3):
    """Cover early, middle, and late sites while retaining catalog order."""
    paths = task["paths"]
    if len(paths) <= limit:
        return list(paths)
    if limit == 1:
        return [paths[0]]
    indices = {0, len(paths) - 1}
    families = {paths[index].rsplit(".", 1)[-1] for index in indices}
    for slot in range(1, limit - 1):
        target = slot * (len(paths) - 1) / (limit - 1)
        available = [index for index in range(1, len(paths) - 1) if index not in indices]
        fresh = [index for index in available if paths[index].rsplit(".", 1)[-1] not in families]
        index = min(fresh or available, key=lambda index: (abs(index - target), index))
        indices.add(index)
        families.add(paths[index].rsplit(".", 1)[-1])
    indices = sorted(indices)
    return [paths[i] for i in indices]


def binding(config, catalog, task):
    return digest({"implementation_hash": implementation_hash(), "task_id": task["id"],
        "kind": task["kind"], "paths": task["paths"], "scalars": task.get("scalars", []),
        "cases": task.get("cases", []),
        "hardware": catalog.get("hardware", {}).get("fingerprint"),
        "model": {k: catalog["config"].get(k) for k in ("model_id", "revision", "dtype", "attention")},
        "workloads": config["workloads"], "seeds": config["correctness_seeds"],
        "atol": config["atol"], "rtol": config["rtol"],
        "policy": config.get("supplemental_correctness", {})})


def _sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _index(profile):
    path = Path(profile) / "supplemental" / "index.json"
    return read_json(path) if path.exists() else {}


def capture_needed(profile, task_id):
    # This only routes the job. The evaluator still validates the entire fixture.
    return _index(profile).get(task_id, {}).get("coverage_complete") is not True


def load_fixture(profile, config, catalog, task):
    """Validate the trusted index and immutable tensor archive before loading."""
    import torch
    entry = _index(profile).get(task["id"])
    if entry is None:
        return None, None
    if entry.get("binding") != binding(config, catalog, task):
        raise ValueError("Supplemental correctness source, task, hardware, or config changed.")
    root = Path(profile).resolve()
    path = (root / entry["path"]).resolve()
    path.relative_to(root / "supplemental")
    if _sha(path) != entry["sha256"]:
        raise ValueError("The supplemental correctness archive changed.")
    fixture = torch.load(path, map_location="cpu", weights_only=True)
    if fixture.get("binding") != entry["binding"] or fixture.get("task_id") != task["id"]:
        raise ValueError("Supplemental correctness identity differs from its index.")
    cases = fixture.get("cases", [])
    if len(cases) != entry.get("case_count") or len({case.get("id") for case in cases}) != len(cases):
        raise ValueError("Supplemental case count or identity differs from its index.")
    if fixture.get("coverage_complete") != entry.get("coverage_complete"):
        raise ValueError("Supplemental coverage differs from its index.")
    for case in cases:
        validate_case(case, task, config)
    if fixture.get("coverage_complete"):
        paths = representative_paths(task, config.get("supplemental_correctness", {}).get("max_paths", 3))
        from .generation import enabled as generating, sampled_steps
        required = {digest([path, workload, seed, step]) for workload in config["workloads"]
                    for seed in evaluation_seeds(config, workload)
                    for step in (sampled_steps(workload) if generating(workload) else [None])
                    for path in paths_at_step(task, paths, workload, step)}
        observed = {digest([case["source_path"], case["workload"], case["seed"], case.get("generation_step", {}).get("step")]) for case in cases}
        if not required.issubset(observed):
            raise ValueError("Supplemental coverage is missing a required path, workload, or seed.")
    return fixture, entry


def paths_at_step(task, paths, workload, step):
    if step is None:
        return set(paths)
    checks = task["contract"]["validation"]["checks"]
    return {c["path"] for c in checks if c.get("workload") == workload
            and c.get("generation_step", {}).get("step") == step and c["path"] in paths}


def validate_case(case, task, config):
    if (case.get("source_path") not in task["paths"] or case.get("seed") not in config["correctness_seeds"]
            or case.get("workload") not in config["workloads"]
            or case.get("reference") != "original_eager_module"
            or case.get("kind") != task["kind"] or case.get("scalars") != task.get("scalars", [])
            or case.get("origin") not in {"representative", "integration_failure"}
            or type(case.get("call_index")) is not int or case["call_index"] < 0):
        raise ValueError("Supplemental case is outside the trusted task contract.")
    tensors = case.get("tensors", [])
    if (not tensors or case.get("shapes") != [list(t.shape) for t in tensors]
            or case.get("strides") != [list(t.stride()) for t in tensors]
            or case.get("dtypes") != [str(t.dtype) for t in tensors]
            or not hasattr(case.get("expected"), "shape")):
        raise ValueError("Supplemental tensor metadata or eager output is missing.")


class CaptureBuffer:
    def __init__(self, profile, config, catalog):
        self.profile, self.config, self.catalog = Path(profile), config, catalog
        self.tasks = {t["id"]: t for t in catalog["tasks"]}
        self.bundles, self.weights, self.changed = {}, {}, set()
        self.artifacts = {}
        self.failure_summary = {}

    def bundle(self, task_id):
        if task_id not in self.bundles:
            task = self.tasks[task_id]
            previous, _ = load_fixture(self.profile, self.config, self.catalog, task)
            self.bundles[task_id] = previous or {"task_id": task_id, "binding": binding(self.config, self.catalog, task),
                "coverage_complete": False, "cases": []}
            if task["kind"] != "attention":
                for case in self.bundles[task_id]["cases"]:
                    for index, tensor in enumerate(case["tensors"][1:], 1):
                        self.weights.setdefault((task_id, case["source_path"], index), tensor)
        return self.bundles[task_id]

    def failure_record(self, task_id, identity, path, context, call_index, *, saved):
        summary = self.failure_summary.setdefault(task_id, {"saved_cases": [], "uncaptured_count": 0,
            "observed_failure_count": 0, "max_new_failure_cases":
                self.config.get("supplemental_correctness", {}).get("max_failure_cases", 3)})
        summary["observed_failure_count"] += 1
        if saved:
            if not any(case["case_id"] == identity for case in summary["saved_cases"]):
                summary["saved_cases"].append({"case_id": identity, "source_path": path,
                    **copy.deepcopy(context), "call_index": call_index})
        else:
            summary["uncaptured_count"] += 1

    def append(self, task_id, path, tensors, scalars, expected, context, *, origin, call_index=0, attention=False):
        import torch
        task = self.tasks[task_id]
        identity = digest([task_id, path, context, call_index])
        bundle = self.bundle(task_id)
        if any(c["id"] == identity for c in bundle["cases"]):
            if origin == "integration_failure":
                self.failure_record(task_id, identity, path, context, call_index, saved=True)
            return
        if origin == "integration_failure" and self.config.get("kernel_search", {}).get("evaluation") != "staged":
            limit = self.config.get("supplemental_correctness", {}).get("max_failure_cases", 3)
            if sum(case["origin"] == "integration_failure" for case in bundle["cases"]) >= limit:
                self.failure_record(task_id, identity, path, context, call_index, saved=False)
                return
        def cpu_copy(value):
            result = torch.empty_strided(tuple(value.shape), tuple(value.stride()), dtype=value.dtype, device="cpu")
            result.copy_(value.detach())
            return result
        saved = []
        for index, tensor in enumerate(tensors):
            cache_key = (task_id, path, index)
            if task["kind"] != "attention" and index > 0:
                if cache_key not in self.weights:
                    self.weights[cache_key] = cpu_copy(tensor)
                saved.append(self.weights[cache_key])
            else:
                saved.append(cpu_copy(tensor))
        if attention:
            expected = expected.transpose(1, 2).contiguous()
        case = {"id": identity, "source_path": path, "kind": task["kind"], "call_index": call_index,
            **copy.deepcopy(context), "origin": origin, "reference": "original_eager_module",
            "tensors": saved, "scalars": list(scalars), "expected": expected.detach().cpu().clone(),
            "shapes": [list(t.shape) for t in saved], "strides": [list(t.stride()) for t in saved],
            "dtypes": [str(t.dtype) for t in saved]}
        validate_case(case, task, self.config)
        bundle["cases"].append(case)
        self.changed.add(task_id)
        if origin == "integration_failure":
            self.failure_record(task_id, identity, path, context, call_index, saved=True)

    def flush(self):
        import torch
        records = _index(self.profile)
        for task_id in self.changed:
            directory = self.profile / "supplemental"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / (task_id + "-" + uuid.uuid4().hex + ".pt")
            torch.save(self.bundles[task_id], path)
            records[task_id] = {"path": path.relative_to(self.profile).as_posix(), "sha256": _sha(path),
                "binding": self.bundles[task_id]["binding"], "case_count": len(self.bundles[task_id]["cases"]),
                "coverage_complete": self.bundles[task_id]["coverage_complete"]}
        if self.changed:
            write_json(self.profile / "supplemental" / "index.json", records)
        self.artifacts = {key: records[key] for key in self.bundles if key in records}


def ensure_fixture(profile, config, catalog, task, *, session=None):
    """Capture representative original eager paths once for this checked source."""
    existing, entry = load_fixture(profile, config, catalog, task)
    if existing is not None and existing.get("coverage_complete"):
        return existing, entry
    import torch
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from .model import load_model, arguments, install_attention_dispatch
    options = {**config, "revision": catalog["config"]["revision"]}
    model = session.get_model(options) if session is not None else load_model(options)
    paths = representative_paths(task, config.get("supplemental_correctness", {}).get("max_paths", 3))
    capture = CaptureBuffer(profile, config, catalog)
    handles, attributes, observed = [], [], set()
    context, counts, pending = {}, {}, {}
    original_attention = ALL_ATTENTION_FUNCTIONS["sdpa"]
    def record(path, tensors, scalars, output, attention=False):
        count = counts.get(path, 0); counts[path] = count + 1; observed.add(path)
        capture.append(task["id"], path, tensors, scalars, output, context,
                       origin="representative", call_index=count, attention=attention)
    class AttentionTap:
        def capture(self, path, kind, tensors, scalars):
            if path in paths: pending[path] = (tensors, scalars)
            return path
        def validate(self, key, path, functional, original):
            if path in pending:
                tensors, scalars = pending.pop(path)
                record(path, tensors, scalars, original, attention=True)
    from .generation import visit_inputs

    @contextmanager
    def capture_step(step_context):
        context.clear()
        context.update(step_context)
        counts.clear()
        observed.clear()
        pending.clear()
        handles.clear()
        attributes.clear()
        try:
            for path in paths:
                if task["kind"] == "attention":
                    module = model.get_submodule(path.removesuffix(".__core__"))
                    attributes.append((module, hasattr(module, "_sako_path"), getattr(module, "_sako_path", None)))
                    module._sako_path = path.removesuffix(".__core__")
                else:
                    def hook(module, args, output, path=path):
                        tensors, scalars = arguments(task["kind"], module, args[0])
                        record(path, tensors, scalars, output)
                    handles.append(model.get_submodule(path).register_forward_hook(hook))
            if task["kind"] == "attention": install_attention_dispatch(AttentionTap())
            yield
            if observed != paths_at_step(task, paths, context["workload"], context.get("generation_step", {}).get("step")):
                raise ValueError("Supplemental capture did not visit every required representative path.")
        finally:
            for handle in handles: handle.remove()
            ALL_ATTENTION_FUNCTIONS["sdpa"] = original_attention
            for module, existed, previous in attributes:
                if existed: module._sako_path = previous
                else: delattr(module, "_sako_path")

    with torch.inference_mode():
        for workload in config["workloads"]:
            for seed in evaluation_seeds(config, workload):
                visit_inputs(model, options, workload, seed, capture_step)
    capture.bundle(task["id"])["coverage_complete"] = True
    capture.changed.add(task["id"])
    capture.flush()
    return load_fixture(profile, config, catalog, task)


@contextmanager
def capture_required_failures(profile, config, catalog):
    if not enabled(config):
        yield None
        return
    capture = CaptureBuffer(profile, config, catalog)
    token = _CAPTURE.set(capture)
    try:
        yield capture
    finally:
        _CAPTURE.reset(token)
        capture.flush()


def record_required_failure(record, tensors, scalars, expected, context, *, attention=False):
    capture = _CAPTURE.get()
    if capture is None or record.get("task_id") not in capture.tasks:
        return
    if (str(record.get("candidate_id", "")).startswith("region:")
            and capture.config.get("containing_region_policy", "required") == "diagnostic"):
        return
    if (record.get("check", {}).get("passed") is not False or record.get("inputs_unchanged") is not True
            or record.get("restoration_errors") or record.get("error")):
        return
    capture.append(record["task_id"], record["path"], tensors, scalars, expected, context,
                   origin="integration_failure", call_index=record["call_index"], attention=attention)
