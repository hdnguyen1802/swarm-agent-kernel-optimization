"""Profile verified operations in Hugging Face causal language models."""

from __future__ import annotations

import copy
from contextlib import contextmanager
import statistics
import traceback
import time
from pathlib import Path

from .core import digest, read_json, write_json
from .gpu import reference, custom_op, paired_timings, check_output
from .hardware import discover_hardware, require_hardware, select_top_tasks
from .contracts import operation_contract
from .generation import evaluation_seeds


def load_model(config):
    import torch
    from transformers import AutoModelForCausalLM
    from .compat import configure_compiler
    configure_compiler(config)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"], revision=config["revision"], torch_dtype=torch.float16,
        device_map={"": "cuda:0"},
        **({"attn_implementation": "sdpa"} if config["attention"] == "sdpa" else {}),
        trust_remote_code=False, use_safetensors=True).eval()
    if getattr(model.config, "is_encoder_decoder", False):
        raise ValueError("The workload requires a decoder-only causal language model.")
    model.config.use_cache = any(w.get("mode") == "cached_generation" for w in config.get("workloads", []))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def inputs(model, workload, seed):
    import torch
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randint(0, model.config.vocab_size,
                         (workload["batch_size"], workload["sequence_length"]),
                         device="cuda", generator=generator)


def forward(model, ids):
    return model(input_ids=ids, use_cache=False, return_dict=False)[0]


def clone_structure(module):
    """Copy module containers. Share weights to fit two graphs on one GPU."""
    clone = copy.copy(module)
    clone._modules = {key: clone_structure(value) if value is not None else None
                      for key, value in module._modules.items()}
    clone._parameters = module._parameters.copy()
    clone._buffers = module._buffers.copy()
    clone._forward_hooks = module._forward_hooks.copy()
    clone._forward_pre_hooks = module._forward_pre_hooks.copy()
    if hasattr(module, "config"):
        clone.config = copy.copy(module.config)
    return clone


def transposed_linear(module):
    if not hasattr(module, "nf"):
        return False
    from transformers.pytorch_utils import Conv1D
    return isinstance(module, Conv1D)


def module_kind(module):
    import torch
    if isinstance(module, torch.nn.Linear) or transposed_linear(module):
        return "linear"
    if isinstance(module, torch.nn.Embedding):
        return "embedding"
    if isinstance(module, torch.nn.SiLU):
        return "silu"
    # Structural matches nominate contracts. Their actual outputs must pass verification.
    weight = getattr(module, "weight", None)
    if (getattr(weight, "ndim", 0) == 1 and isinstance(getattr(module, "variance_epsilon", None), (int, float))
            and getattr(module, "bias", None) is None):
        return "rmsnorm"
    if isinstance(getattr(module, "act_fn", None), torch.nn.SiLU):
        projections = [getattr(module, key, None) for key in ("gate_proj", "up_proj", "down_proj")]
        if all(isinstance(p, torch.nn.Linear) and p.bias is None for p in projections):
            return "mlp"
    return None


def arguments(kind, module, x):
    if kind == "silu":
        return [x], []
    if kind == "linear":
        weight = module.weight.T if transposed_linear(module) else module.weight
        return [x, weight, module.bias if module.bias is not None else x.new_empty(0)], []
    if kind == "embedding":
        return [x, module.weight], []
    if kind == "rmsnorm":
        return [x, module.weight], [float(module.variance_epsilon)]
    if kind == "mlp":
        return [x, module.gate_proj.weight, module.up_proj.weight, module.down_proj.weight], []
    raise ValueError(kind)


def task_key(kind, tensors, scalars):
    if kind == "silu":
        return kind + "_" + digest([kind, tensors[0].shape[-1], str(tensors[0].dtype)])[:12]
    geometry = ([list(t.shape) for t in tensors[1:]] if kind != "attention"
                else [tensors[0].shape[1], tensors[0].shape[-1]])
    return kind + "_" + digest([kind, geometry, scalars])[:12]


_DEFAULT_ATTENTION_DISPATCH = None


def install_attention_dispatch(collector=None, timer=None):
    """Use the HF attention interface. Projection and rotary code stay in HF."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.integrations.sdpa_attention import sdpa_attention_forward, repeat_kv
    global _DEFAULT_ATTENTION_DISPATCH
    if collector is None and timer is None and _DEFAULT_ATTENTION_DISPATCH is not None:
        ALL_ATTENTION_FUNCTIONS["sdpa"] = _DEFAULT_ATTENTION_DISPATCH
        return

    def dispatch(module, query, key, value, attention_mask, dropout=0., scaling=None,
                 is_causal=None, **kwargs):
        operation = getattr(module, "_sako_attention", None)
        if collector is not None and hasattr(collector, "module_paths") and module in collector.module_paths:
            module._sako_path = collector.module_paths[module]
        if (operation is None and collector is None and timer is None) or (
                operation is None and (not getattr(module, "_sako_path", None)
                or attention_mask is not None and attention_mask.ndim != 4)):
            return sdpa_attention_forward(module, query, key, value, attention_mask,
                                           dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs)
        if dropout != 0.:
            raise ValueError("SAKO measures inference with zero dropout.")
        if timer is not None and getattr(timer, "original_attention", False):
            path = getattr(module, "_sako_path", "") + ".__core__"
            timer.before(path)
            try:
                return sdpa_attention_forward(module, query, key, value, attention_mask,
                    dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs)
            finally:
                timer.after(path)
        original_query, original_key, original_value = query, key, value
        key = repeat_kv(key, getattr(module, "num_key_value_groups", 1)).contiguous()
        value = repeat_kv(value, getattr(module, "num_key_value_groups", 1)).contiguous()
        query = query.contiguous()
        mask = attention_mask[:, :, :, :key.shape[-2]] if attention_mask is not None else query.new_empty(0)
        causal = (attention_mask is None and query.shape[2] > 1) if is_causal is None else is_causal
        scale = scaling if scaling is not None else query.shape[-1] ** -.5
        tensors, scalars = [query, key, value, mask], [float(causal), float(scale)]
        path = getattr(module, "_sako_path", "") + ".__core__"
        if collector is not None:
            task = collector.capture(path, "attention", tensors, scalars)
            expected = sdpa_attention_forward(module, original_query, original_key, original_value,
                attention_mask, dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs)[0]
            actual = reference("attention", tensors, scalars).transpose(1, 2).contiguous()
            collector.validate(task, path, actual, expected)
            return expected, None
        if timer is not None:
            timer.before(path)
        output = operation(tensors, scalars) if operation is not None else reference("attention", tensors, scalars)
        if timer is not None:
            timer.after(path)
        return output.transpose(1, 2).contiguous(), None

    if collector is None and timer is None:
        _DEFAULT_ATTENTION_DISPATCH = dispatch
    ALL_ATTENTION_FUNCTIONS["sdpa"] = dispatch


def fixture_tensor(tensor, preserve_layout):
    if not preserve_layout:
        return tensor.detach().cpu()
    import torch
    return torch.empty_strided(tensor.shape, tensor.stride(), dtype=tensor.dtype, device="cpu").copy_(tensor.detach())


class Collector:
    def __init__(self, config):
        self.config, self.tasks, self.cases, self.events = config, {}, {}, {}
        self.weights = {}
        self.validations, self.path_to_key = {}, {}
        self.context = {}
        self.pending_eager_outputs = {}
        self.separate_operations = config.get("kernel_search", {}).get("evaluation") == "staged"
        self.captured_shapes = set()

    def validate(self, key, path, actual, expected):
        self.validations.setdefault(key, []).append(
            {"path": path, **self.context, **check_output(actual, expected, self.config)})
        if self.config.get("correctness_scope") == "replaced_layers":
            pending = self.pending_eager_outputs.get((key, path), [])
            if not pending:
                raise ValueError("The eager output has no matching captured module call.")
            case_index, context = pending.pop()
            if context != self.context:
                raise ValueError("The eager output context differs from its captured inputs.")
            if case_index is not None:
                output = (expected.transpose(1, 2).contiguous()
                          if self.tasks[key]["kind"] == "attention" else expected)
                case = self.cases[key][case_index]
                case.update(eager_output=output.detach().cpu().clone(),
                            eager_output_reference="original_model", eager_output_path=path,
                            eager_output_context=context, eager_output_case_index=case_index)

    def capture(self, path, kind, tensors, scalars):
        import torch
        key = task_key(kind, tensors, scalars)
        if self.separate_operations:
            layout = [(tuple(t.shape), tuple(t.stride()), str(t.dtype)) for t in tensors[1:]]
            if kind == "attention":
                layout = [(t.ndim, t.stride(-1), str(t.dtype)) for t in tensors[1:]]
            key = kind + "_" + digest([key, layout, str(tensors[0].dtype)])[:12]
        self.path_to_key[path] = key
        task = self.tasks.setdefault(key, {"id": key, "kind": kind, "paths": [], "scalars": scalars,
                                          "cases": [], "priority_ms": 0.})
        if path not in task["paths"]:
            task["paths"].append(path)
        cases = self.cases.setdefault(key, [])
        captured_index = None
        if self.separate_operations:
            task["separate_operations"] = True
            signature = tuple((tuple(t.shape), tuple(t.stride()), str(t.dtype)) for t in tensors)
            capture_key = (key, path, digest(self.context["workload"]), self.context["seed"],
                           self.context.get("generation_step", {}).get("phase"), signature)
            capture = capture_key not in self.captured_shapes
            self.captured_shapes.add(capture_key)
        else:
            duplicate_step = "generation_step" in self.context and any(
                case["seed"] == self.context["seed"] and case["workload"] == self.context["workload"]
                and case.get("generation_step") == self.context["generation_step"] for case in cases)
            capture = path == task["paths"][0] and not duplicate_step and len(cases) < self.config["max_fixture_cases"]
        if capture:
            cpu = []
            for i, tensor in enumerate(tensors):
                if kind != "attention" and i > 0:
                    storage = ((tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype))
                               if self.separate_operations else (key, path, i))
                    if storage not in self.weights:
                        self.weights[storage] = fixture_tensor(tensor, self.separate_operations)
                    cpu.append(self.weights[storage])
                else:
                    cpu.append(fixture_tensor(tensor, self.separate_operations))
            case_index = len(cases)
            captured_index = case_index
            cases.append({"tensors": cpu, "scalars": scalars, "case_index": case_index,
                          **copy.deepcopy(self.context)})
            task["cases"].append({"case_index": case_index, **copy.deepcopy(self.context),
                                   "shapes": [list(t.shape) for t in tensors],
                                   "dtypes": [str(t.dtype) for t in tensors],
                                   "strides": [list(t.stride()) for t in tensors]})
            if self.config.get("correctness_scope") == "replaced_layers":
                cases[-1]["source_path"] = path
                task["cases"][-1]["source_path"] = path
        if self.config.get("correctness_scope") == "replaced_layers":
            self.pending_eager_outputs.setdefault((key, path), []).append(
                (captured_index, copy.deepcopy(self.context)))
        return key


class RegionTimer:
    """Record events without fixture copies or reference recomputation."""
    def __init__(self, path_to_key):
        self.path_to_key = path_to_key
        self.active, self.starts, self.events = False, {}, []
        self.workload_index, self.trial = None, None

    def before(self, path):
        if self.active:
            import torch
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self.starts.setdefault(path, []).append(event)

    def after(self, path):
        if self.active:
            import torch
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            start = self.starts[path].pop()
            self.events.append((self.path_to_key[path], path, self.workload_index, self.trial, start, end))

    def results(self):
        results = {}
        for key, path, workload, trial, start, end in self.events:
            results.setdefault(key, []).append({"path": path, "workload_index": workload,
                "trial": trial, "time_ms": start.elapsed_time(end)})
        return results


def region_priority(samples, workload_count):
    totals = {}
    for sample in samples:
        key = (sample["workload_index"], sample["trial"])
        totals[key] = totals.get(key, 0.) + sample["time_ms"]
    medians = []
    for workload in range(workload_count):
        values = [time for (index, _), time in totals.items() if index == workload]
        if not values:
            return 0., []
        medians.append(statistics.median(values))
    return statistics.mean(medians), medians


def compiled_profile(model, config, output_dir, compiled=None):
    """Measure the real compiled baseline. Do not invent region attribution."""
    import torch
    if compiled is None:
        compiled = torch.compile(lambda ids: forward(model, ids), fullgraph=True, dynamic=False,
                                 mode=config["compile_mode"])
    results = []
    with torch.inference_mode():
        for index, workload in enumerate(config["workloads"]):
            ids = inputs(model, workload, config["correctness_seeds"][0])
            for _ in range(config["warmup"]):
                compiled(ids)
            torch.cuda.synchronize()
            samples = []
            for _ in range(config["trials"]):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                compiled(ids)
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profiler:
                compiled(ids)
                torch.cuda.synchronize()
            trace = output_dir / f"compiled-workload-{index}.trace.json"
            profiler.export_chrome_trace(str(trace))
            records = [{"name": event.key, "count": event.count,
                        "self_gpu_time_us": float(getattr(event, "self_device_time_total", 0.)),
                        "gpu_time_us": float(getattr(event, "device_time_total", 0.)),
                        "self_cpu_time_us": float(event.self_cpu_time_total)}
                       for event in profiler.key_averages()]
            results.append({"workload": workload, "samples_ms": samples,
                            "median_ms": statistics.median(samples), "trace": trace.name, "events": records})
    return {"status": "passed", "fullgraph": True, "workloads": results,
            "region_attribution": "unavailable",
            "note": "Compiled fusion does not provide a verified mapping to replacement regions. "
                    "Ranking uses a separate eager profile, with the method recorded in the task catalog."}


def profile(config, output_dir, *, session=None):
    import torch
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from huggingface_hub import HfApi
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pinned = copy.deepcopy(config)
    pinned["revision"] = HfApi().model_info(config["model_id"], revision=config["revision"]).sha
    hardware = require_hardware(discover_hardware(config))
    write_json(output_dir / "hardware.json", hardware)
    model = load_model(pinned) if session is None else session.get_model(pinned)
    from .measurement import enabled as adaptive, starting_baseline
    starting = starting_baseline(model, pinned, hardware, output_dir) if adaptive(pinned) else None
    collector = Collector(config)
    from .generation import visit_inputs
    collector.module_paths = {module: path for path, module in model.named_modules()}
    incompatible_calls = set()

    @contextmanager
    def capture_step(context):
        collector.context = context
        handles, starts = [], {}
        original_attention = ALL_ATTENTION_FUNCTIONS["sdpa"]
        for path, module in model.named_modules():
            kind = module_kind(module)
            if kind is None:
                continue

            def before(mod, args, kwargs, path=path, kind=kind):
                if kwargs or len(args) != 1 or not isinstance(args[0], torch.Tensor):
                    incompatible_calls.add(path)
                    return
                tensors, scalars = arguments(kind, mod, args[0])
                key = collector.capture(path, kind, tensors, scalars)
                starts[path] = (key, tensors, scalars)

            def after(mod, args, output, path=path):
                if path not in starts:
                    return
                key, tensors, scalars = starts.pop(path)
                if not isinstance(output, torch.Tensor):
                    incompatible_calls.add(path)
                    collector.pending_eager_outputs.pop((key, path), None)
                    return
                collector.validate(key, path, reference(collector.tasks[key]["kind"], tensors, scalars), output)

            handles.extend([module.register_forward_pre_hook(before, with_kwargs=True), module.register_forward_hook(after)])
        try:
            install_attention_dispatch(collector)
            yield
        finally:
            for handle in handles:
                handle.remove()
            ALL_ATTENTION_FUNCTIONS["sdpa"] = original_attention

    with torch.inference_mode():
        for workload in config["workloads"]:
            for seed in evaluation_seeds(config, workload)[:2]:
                visit_inputs(model, pinned, workload, seed, capture_step)
        torch.cuda.synchronize()
    original_attention = ALL_ATTENTION_FUNCTIONS["sdpa"]
    from .ranking import ranking_mode, profile_operations, rank_operations
    compiled = None
    if ranking_mode(config) == "compiled_profiler":
        from .compiled_ranking import profile_compiled
        ranking, compiled = profile_compiled(model, pinned, list(collector.tasks.values()), output_dir,
            **({"starting": starting, "hardware": hardware} if starting is not None else {}))
        write_json(output_dir / "operation_profile.json", ranking)
        timings = {}
    elif ranking_mode(config) == "profiler_events":
        records = profile_operations(model, pinned, collector.path_to_key, output_dir)
        ranking = rank_operations(list(collector.tasks.values()), records, len(config["workloads"]), config["trials"])
        write_json(output_dir / "operation_profile.json", ranking)
        timings = {}
    else:
        timer, handles = RegionTimer(collector.path_to_key), []
        for path, module in model.named_modules():
            if module_kind(module) is not None:
                handles.extend([module.register_forward_pre_hook(lambda mod, args, path=path: timer.before(path)),
                                module.register_forward_hook(lambda mod, args, output, path=path: timer.after(path))])
        try:
            install_attention_dispatch(timer=timer)
            with torch.inference_mode():
                for index, workload in enumerate(config["workloads"]):
                    ids = inputs(model, workload, config["correctness_seeds"][0])
                    timer.active = False
                    for _ in range(config["warmup"]):
                        forward(model, ids)
                    torch.cuda.synchronize()
                    timer.workload_index, timer.active = index, True
                    for trial in range(config["trials"]):
                        timer.trial = trial
                        forward(model, ids)
                    timer.active = False
                    torch.cuda.synchronize()
        finally:
            timer.active = False
            for handle in handles:
                handle.remove()
            ALL_ATTENTION_FUNCTIONS["sdpa"] = original_attention
        timings = timer.results()
    write_json(output_dir / "region_timings.json", timings)
    for key, task in collector.tasks.items():
        if ranking_mode(config) == "eager_regions":
            task["priority_ms"], task["workload_medians_ms"] = region_priority(timings.get(key, []), len(config["workloads"]))
            task["priority_method"] = "warmed_eager_region_cuda_events"
        validations = collector.validations.get(key, [])
        validations.extend({"path": path, "passed": False, "reason": "Unsupported module call signature or output."}
                           for path in task["paths"] if path in incompatible_calls)
        task["contract"] = operation_contract(task, validations, config)
        write_json(output_dir / (key + ".contract.json"), task["contract"])
        torch.save({"cases": collector.cases[key]}, output_dir / (key + ".pt"))
    try:
        if compiled is None:
            compiled = compiled_profile(model, pinned, output_dir,
                                        compiled=session.get_baseline(pinned) if session is not None else None)
    except Exception:
        (output_dir / "compiled_profile_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    write_json(output_dir / "compiled_profile.json", compiled)
    from .hierarchy import enabled, select_hierarchy
    selection = (select_hierarchy(list(collector.tasks.values()),
                                 include_parents=config["agent"].get("search_parent_regions", False)) if enabled(config) else
                 select_top_tasks(list(collector.tasks.values()), config["agent"].get("top_k_operations", 4)))
    result = {"config": pinned, "hardware": hardware, "tasks": list(collector.tasks.values()),
              **({"starting_baseline": starting} if starting is not None else {}),
              "selected_task_ids": selection["selected_task_ids"], "selection": selection,
              **({"hierarchy": selection["hierarchy"]} if enabled(config) else {}),
              "compiled_profile": {"status": compiled["status"], "region_attribution": compiled["region_attribution"],
                                   "workload_times": [{k: w[k] for k in ("workload", "median_ms", "input", "generation_metrics", "gpu_samples", "gpu_groups",
                                                         "profiling", "profiling_correctness", "starting_times_reused", "starting_identity_hash") if k in w}
                                                      for w in compiled.get("workloads", [])],
                                   "report": "compiled_profile.json", "note": compiled["note"]},
              "ranking_mode": ranking_mode(config),
              "ranking_report": "operation_profile.json" if ranking_mode(config) != "eager_regions" else "region_timings.json",
              "note": "The ranking method is explicit. Compiled mode measures remaining compiled GPU work and protects fusion boundaries. "
                      "Eager outputs remain the accuracy reference. Complete compiled model tests decide model acceptance."}
    write_json(output_dir / "tasks.json", result)
    return result


def apply_manifest(model, manifest, task_catalog, candidate_root, *, operations=None, config=None):
    import torch
    from .core import candidate_id
    tasks = {t["id"]: t for t in task_catalog["tasks"]}
    from .shape_selection import select_variant, validate_entries
    validate_entries(manifest["entries"], tasks)

    class Replacement(torch.nn.Module):
        def __init__(self, original, kind, operation, variants=None):
            super().__init__()
            self.original, self.kind, self.operation = original, kind, operation
            self.variants = variants

        def forward(self, x):
            tensors, scalars = arguments(self.kind, self.original, x)
            if self.variants is not None:
                key = select_variant(self.variants, tensors, scalars)
                if key is None:
                    return self.original(x)
                return self.operation[key](tensors, scalars)
            return self.operation(tensors, scalars)

    operations = {} if operations is None else operations
    install_attention_dispatch()
    routed = {}
    for entry in manifest["entries"]:
        candidate = read_json(Path(candidate_root) / (entry["candidate_id"] + ".json"))
        if candidate_id(candidate) != entry["candidate_id"] or candidate["task_id"] != entry["task_id"]:
            raise ValueError("The replacement candidate identity changed.")
        task = tasks[candidate["task_id"]]
        operation = operations.get(entry["candidate_id"])
        if operation is None:
            options = task_catalog.get("config", {}) if config is None else config
            if options.get("triton_integration", "opaque") == "opaque":
                operation = custom_op(candidate, task["kind"])
            else:
                from .gpu import model_operation
                operation = model_operation(candidate, task["kind"], options)
        operations[entry["candidate_id"]] = operation
        for path in entry["paths"]:
            if path not in task["paths"]:
                raise ValueError("Manifest path is outside the task.")
            if "signatures" in entry:
                routed.setdefault((path, task["kind"]), []).append({"candidate_id": entry["candidate_id"],
                                                                  "signatures": entry["signatures"]})
            elif task["kind"] == "attention":
                model.get_submodule(path.removesuffix(".__core__"))._sako_attention = operation
            else:
                parent, _, name = path.rpartition(".")
                owner = model.get_submodule(parent) if parent else model
                setattr(owner, name, Replacement(getattr(owner, name), task["kind"], operation))
    for (path, kind), variants in routed.items():
        if kind == "attention":
            def route(tensors, scalars, variants=variants, kind=kind):
                key = select_variant(variants, tensors, scalars)
                return operations[key](tensors, scalars) if key is not None else reference(kind, tensors, scalars)
            model.get_submodule(path.removesuffix(".__core__"))._sako_attention = route
        else:
            parent, _, name = path.rpartition(".")
            owner = model.get_submodule(parent) if parent else model
            setattr(owner, name, Replacement(getattr(owner, name), kind, operations, variants))
    return model


def measure_model(config, manifest, catalog, candidate_root, *, session=None, trace_dir=None, reference_control=False,
                  feedback_trace_dir=None, incumbent_manifest=None, incumbent_reference=None, prepared_generation=None,
                  reference_directory=None, correctness_only=False):
    from .generation import enabled as generation_workload
    if any(generation_workload(w) for w in config.get("workloads", [])):
        return measure_model_layer_outputs(config, manifest, catalog, candidate_root, session=session,
                                          **({"correctness_only": True} if correctness_only else {}),
                                          reference_control=reference_control, incumbent_manifest=incumbent_manifest,
                                          incumbent_reference=incumbent_reference,
                                          **({"reference_directory": reference_directory} if reference_directory is not None else {}),
                                          **({"prepared_generation": prepared_generation} if prepared_generation is not None else {}))
    if config.get("correctness_scope") == "replaced_layers":
        if config.get("correctness_reference") != "eager":
            raise ValueError("Replacement-layer correctness requires the eager reference.")
        return measure_model_layer_outputs(config, manifest, catalog, candidate_root, session=session,
            **({"trace_dir": trace_dir} if trace_dir is not None else {}),
            **({"feedback_trace_dir": feedback_trace_dir} if feedback_trace_dir is not None else {}),
            **({"reference_control": True} if reference_control else {}),
            **({"incumbent_manifest": incumbent_manifest, "incumbent_reference": incumbent_reference}
               if incumbent_manifest is not None else {}))
    if reference_control:
        raise ValueError("The linear deployment control requires replaced-layer correctness.")
    import torch
    from .model_acceptance import target_workloads, score_model_cases
    targets = target_workloads(config, manifest, catalog)
    config = {**config, "revision": catalog["config"]["revision"]}
    model = load_model(config) if session is None else session.get_model(config)
    alternative = clone_structure(model)
    apply_manifest(alternative, manifest, catalog, candidate_root, config=config)
    baseline = (torch.compile(lambda ids: forward(model, ids), fullgraph=True, dynamic=False,
                              mode=config["compile_mode"]) if session is None else session.get_baseline(config))
    candidate = torch.compile(lambda ids: forward(alternative, ids), fullgraph=True, dynamic=False,
                              mode=config["compile_mode"])
    results = []
    with torch.inference_mode():
        for workload in config["workloads"]:
            checks = []
            eager_diagnostics = []
            for seed in evaluation_seeds(config, workload):
                ids = inputs(model, workload, seed)
                eager = forward(model, ids)
                expected = baseline(ids)
                eager_diagnostics.append(check_output(expected, eager, config))
                checks.append({**check_output(candidate(ids), expected, config), "seed": seed})
            item = {"workload": workload, "correct": all(c["passed"] for c in checks), "checks": checks,
                    "reference": "torch_compile", "compiled_vs_eager_diagnostics": eager_diagnostics}
            if item["correct"]:
                from .measurement import enabled as adaptive, paired_forward
                item.update(paired_forward(model, baseline, candidate, config, workload, catalog,
                    threshold=config["min_speedup"] if workload in targets else 1.0) if adaptive(config) else
                    paired_timings(lambda: baseline(ids), lambda: candidate(ids), config, repeats=1))
            item.update(speed_target=workload in targets,
                        speed_requirement=config["min_speedup"] if workload in targets else 1.0)
            results.append(item)
    from .deployment_diagnostics import performance_trace
    decision = score_model_cases(results, config, targets)
    comparison = compare_incumbent(model, alternative, incumbent_manifest, catalog, candidate_root, config, results,
                                       reference=incumbent_reference)
    decision = require_incumbent_gain(decision, comparison)
    trace = performance_trace(feedback_trace_dir, decision, results, config, baseline, candidate,
                              lambda workload: inputs(model, workload, config["correctness_seeds"][-1]))
    return {"scope": "complete_model_forward", "compiled": True, "fullgraph": True,
            "config": config, "cases": results, "decision": decision, "performance_trace": trace,
            "incumbent_comparison": comparison,
            "model_speed_scope": config.get("model_speed_scope", "all_workloads"), "target_workloads": targets,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "manifest": manifest}


def measure_model_layer_outputs(config, manifest, catalog, candidate_root, *, session=None, trace_dir=None,
                                reference_control=False, feedback_trace_dir=None, incumbent_manifest=None, incumbent_reference=None,
                                prepared_generation=None, reference_directory=None, correctness_only=False):
    """Gate local eager outputs, then time both complete compiled models."""
    import torch
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from .layer_checks import measure_layer_case, replacement_paths
    from .model_acceptance import target_workloads, score_model_cases
    targets = target_workloads(config, manifest, catalog)
    selected = replacement_paths(manifest, catalog)
    config = {**config, "revision": catalog["config"]["revision"]}
    original_attention = ALL_ATTENTION_FUNCTIONS["sdpa"]
    results = []
    reference_intact = True
    stage_times = {}
    try:
        stage = time.perf_counter()
        model = load_model(config) if session is None else session.get_model(config)
        stage_times["model_resources_s"] = time.perf_counter() - stage
        stage = time.perf_counter()
        alternative = clone_structure(model)
        operations = {}
        if reference_control:
            from .deployment_diagnostics import linear_reference_control
            tasks = {t["id"]: t for t in catalog["tasks"]}
            if any(tasks[entry["task_id"]]["kind"] != "linear" for entry in manifest["entries"]):
                raise ValueError("The trusted deployment control supports linear operations only.")
            operations = {entry["candidate_id"]: linear_reference_control() for entry in manifest["entries"]}
        apply_manifest(alternative, manifest, catalog, candidate_root, operations=operations, config=config)
        stage_times["candidate_setup_s"] = time.perf_counter() - stage
        with torch.inference_mode():
            stage = time.perf_counter()
            for workload in config["workloads"]:
                checks = []
                for seed in evaluation_seeds(config, workload):
                    if reference_intact:
                        from .hierarchy import enabled
                        options = {"alternative": alternative} if enabled(config) else {}
                        check = measure_layer_case(model, manifest, catalog, operations, config, workload, seed, **options)
                        reference_intact = check.get("reference_state_intact", True)
                    else:
                        check = {"seed": seed, "scope": "replaced_layers", "reference": "eager",
                                 "passed": False, "not_measured": True, "reference_state_intact": False,
                                 "reason": "A previous rejected mutation left the reference state unsafe to reuse.",
                                 "expected_paths": list(selected), "observed_paths": [], "missing_paths": list(selected),
                                 "failed_paths": [], "layers": []}
                    checks.append(check)
                item = {"workload": workload, "correct": all(check["passed"] for check in checks),
                        "checks": checks, "reference": "eager_replaced_layers", "timing_reference": "torch_compile",
                        "speed_target": workload in targets,
                        "speed_requirement": config["min_speedup"] if workload in targets else 1.0}
                results.append(item)
            stage_times["layer_accuracy_s"] = time.perf_counter() - stage
            # Keep all local evidence, but do not compile or time a model that
            # already failed a required accuracy check on another workload.
            ready = reference_intact and all(item["correct"] for item in results)
            timing_reason = ("The reference state could not be restored." if not reference_intact else
                             "Complete model timing skipped because a required layer check failed.")
            stage = time.perf_counter()
            from .generation import enabled as generation_workload, compare_models
            generating = generation_workload(config["workloads"][0])
            from .measurement import decode_only, saved_reference
            previous_model = None
            def make_accepted():
                accepted = clone_structure(model)
                apply_manifest(accepted, incumbent_manifest, catalog, candidate_root, config=config)
                return accepted
            if ready and decode_only(config) and incumbent_manifest and not saved_reference(config):
                previous_model = make_accepted()
            if ready and not generating:
                # Every local hook has been removed before compilation or timing.
                baseline = (torch.compile(lambda ids: forward(model, ids), fullgraph=True, dynamic=False,
                                          mode=config["compile_mode"]) if session is None else session.get_baseline(config))
                candidate = torch.compile(lambda ids: forward(alternative, ids), fullgraph=True, dynamic=False,
                                          mode=config["compile_mode"])
            for item in results:
                if not ready:
                    item.update(timing_status="not_measured", timing_reason=timing_reason)
                elif generating:
                    from .measurement import enabled as adaptive
                    check, timings = compare_models(model, alternative, config, item["workload"],
                        **({"correctness_only": True} if correctness_only else {}),
                        **({"manifest": manifest, "reference_directory": reference_directory, "accepted_manifest": incumbent_manifest,
                            "make_accepted": make_accepted} if saved_reference(config) else {}),
                        **({"incumbent": previous_model} if previous_model is not None else {}),
                        **({"prepared": prepared_generation} if prepared_generation is not None else {}),
                        **({"catalog": catalog, "threshold": item["speed_requirement"]} if adaptive(config) else {}))
                    item["checks"].append(check)
                    item["correct"] = check["passed"]
                    item.update(timings)
                else:
                    ids = inputs(model, item["workload"], config["correctness_seeds"][-1])
                    from .measurement import enabled as adaptive, paired_forward
                    item.update(paired_forward(model, baseline, candidate, config, item["workload"], catalog,
                        threshold=item["speed_requirement"]) if adaptive(config) else
                        paired_timings(lambda: baseline(ids), lambda: candidate(ids), config, repeats=1))
            stage_times["full_model_timing_s"] = time.perf_counter() - stage
        decision = (score_model_cases(results, config, targets) if ready else {
            "accepted": False, "speedup": None,
            "reasons": [f"Case {index}: correctness failed." for index, item in enumerate(results)
                        if not item["correct"]] + [timing_reason]})
        comparison = compare_incumbent(model, alternative, incumbent_manifest, catalog, candidate_root, config, results,
                                       reference=incumbent_reference)
        decision = require_incumbent_gain(decision, comparison)
        trace = None
        if reference_control:
            decision["accepted"] = False
            decision["reasons"].append("Original-math deployment control. This diagnostic cannot grant acceptance.")
        if ready and not generating and trace_dir is not None:
            from .deployment_diagnostics import capture_model_traces
            with torch.inference_mode():
                ids = inputs(model, config["workloads"][0], config["correctness_seeds"][-1])
                trace = capture_model_traces(baseline, candidate, ids, trace_dir)
        feedback_trace = None
        if ready and not generating and not reference_control:
            from .deployment_diagnostics import performance_trace
            feedback_trace = performance_trace(feedback_trace_dir, decision, results, config, baseline, candidate,
                lambda workload: inputs(model, workload, config["correctness_seeds"][-1]))
        return {"scope": "cached_decode" if decode_only(config) else "complete_cached_generation" if generating else "complete_model_forward", "compiled": ready, "fullgraph": True,
                "performance_trace": feedback_trace,
                "deployment_trace": trace,
                "reference_control": reference_control,
                "correctness_scope": "replaced_layers", "correctness_reference": "eager",
                "model_reference": "torch_compile", "output_equivalence": (
                    "checked_logits_tokens_and_cache" if generating and ready else "not_tested"),
                "model_speed_scope": config.get("model_speed_scope", "all_workloads"), "target_workloads": targets,
                "containing_region_policy": config.get("containing_region_policy", "required"),
                "reference_state_intact": reference_intact,
                "stage_times": stage_times,
                "acceptance_scope": ("Layer outputs, logits, tokens and cache checked. Speed uses cached decode in ms per token."
                                     if decode_only(config) else "Eager layer accuracy, compiled logits, greedy tokens, KV cache and total generation time."
                                     if generating else "Eager replacement-layer accuracy and complete compiled model speed."),
                "config": config, "cases": results, "decision": decision, "incumbent_comparison": comparison,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "manifest": manifest}
    finally:
        ALL_ATTENTION_FUNCTIONS["sdpa"] = original_attention

def require_incumbent_gain(decision, comparison):
    if comparison is None or comparison["decision"]["accepted"]:
        return decision
    return {**decision, "accepted": False,
            **({"timing_status": comparison["decision"]["timing_status"]} if "timing_status" in comparison["decision"] else {}),
            "reasons": decision["reasons"] +
            ["Current model comparison: " + reason for reason in comparison["decision"]["reasons"]]}


def compare_incumbent(model, candidate, manifest, catalog, candidate_root, config, cases, *, reference=None):
    """Compare the complete candidate with the accepted model on the same GPU."""
    if manifest is None or not all(case["correct"] and case.get("baseline") for case in cases):
        return None
    from .measurement import enabled as adaptive, compare_saved_incumbent, decode_only
    if decode_only(config):
        from .model_acceptance import score_model_cases
        comparison = score_model_cases(cases, config, config["workloads"], reference_role="incumbent")
        return {"reference": "saved_accepted_model" if cases[0].get("accepted_record") else "previous_accepted_model",
                "manifest": manifest, "decision": comparison,
                "cases": [{"workload": c["workload"], "baseline": c["generation_metrics"]["incumbent"]["mean_decode_ms"],
                           "candidate": c["candidate"]} for c in cases]}
    if adaptive(config):
        return compare_saved_incumbent(manifest, reference, cases, config, catalog)
    import torch
    from .generation import enabled, compiled_request, paired_generation
    from .model_acceptance import incumbent_decision
    incumbent = clone_structure(model)
    apply_manifest(incumbent, manifest, catalog, candidate_root, config=config)
    measured = []
    with torch.inference_mode():
        if not any(enabled(w) for w in config["workloads"]):
            base = torch.compile(lambda ids: forward(incumbent, ids), fullgraph=True, dynamic=False,
                                 mode=config["compile_mode"])
            new = torch.compile(lambda ids: forward(candidate, ids), fullgraph=True, dynamic=False,
                                mode=config["compile_mode"])
        for workload in config["workloads"]:
            if enabled(workload):
                from .measurement import enabled as adaptive, saved_warmup
                base_request = compiled_request(incumbent, config, workload)
                new_request = compiled_request(candidate, config, workload)
                timing = paired_generation(base_request, new_request, config,
                    **({"threshold": 1.0, "preferred_warmup": saved_warmup(catalog, model, config, workload,
                        base_request.record)} if adaptive(config) else {}))
            else:
                ids = inputs(model, workload, config["correctness_seeds"][-1])
                from .measurement import enabled as adaptive, paired_forward
                timing = (paired_forward(model, base, new, config, workload, catalog, threshold=1.0) if adaptive(config)
                          else paired_timings(lambda: base(ids), lambda: new(ids), config, repeats=1))
            measured.append({"workload": workload, "correct": True, **timing})
    return {"manifest": manifest, "reference": "accepted_model", "cases": measured,
            "decision": incumbent_decision(measured, config)}
