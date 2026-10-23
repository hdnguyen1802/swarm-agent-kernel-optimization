"""Qwen2 model adapter. It keeps the original weights and forward workload."""

from __future__ import annotations

import copy
from pathlib import Path

from .core import digest, read_json, write_json, score_cases
from .gpu import reference, custom_op, paired_timings, check_output


def load_model(config):
    import torch
    from transformers import AutoModelForCausalLM
    from .compat import configure_compiler
    configure_compiler(config)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"], revision=config["revision"], torch_dtype=torch.float16,
        attn_implementation="sdpa", device_map={"": "cuda:0"},
        trust_remote_code=False, use_safetensors=True).eval()
    if model.config.model_type != "qwen2" or model.config.use_sliding_window:
        raise ValueError("This adapter requires Qwen2 without sliding attention.")
    model.config.use_cache = False
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


def module_kind(module):
    import torch
    if isinstance(module, torch.nn.Linear):
        return "linear"
    if isinstance(module, torch.nn.Embedding):
        return "embedding"
    return {"Qwen2RMSNorm": "rmsnorm", "Qwen2MLP": "mlp"}.get(type(module).__name__)


def arguments(kind, module, x):
    if kind == "linear":
        return [x, module.weight, module.bias if module.bias is not None else x.new_empty(0)], []
    if kind == "embedding":
        return [x, module.weight], []
    if kind == "rmsnorm":
        return [x, module.weight], [float(module.variance_epsilon)]
    if kind == "mlp":
        return [x, module.gate_proj.weight, module.up_proj.weight, module.down_proj.weight], []
    raise ValueError(kind)


def task_key(kind, tensors, scalars):
    geometry = ([list(t.shape) for t in tensors[1:]] if kind != "attention"
                else [tensors[0].shape[1], tensors[0].shape[-1]])
    return kind + "_" + digest([kind, geometry, scalars])[:12]


def install_attention_dispatch(collector=None):
    """Use the HF attention interface. Projection and rotary code stay in HF."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.integrations.sdpa_attention import sdpa_attention_forward, repeat_kv

    def dispatch(module, query, key, value, attention_mask, dropout=0., scaling=None,
                 is_causal=None, **kwargs):
        operation = getattr(module, "_sako_attention", None)
        if operation is None and collector is None:
            return sdpa_attention_forward(module, query, key, value, attention_mask,
                                           dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs)
        if dropout != 0.:
            raise ValueError("SAKO measures inference with zero dropout.")
        key = repeat_kv(key, module.num_key_value_groups).contiguous()
        value = repeat_kv(value, module.num_key_value_groups).contiguous()
        query = query.contiguous()
        mask = attention_mask[:, :, :, :key.shape[-2]] if attention_mask is not None else query.new_empty(0)
        causal = (attention_mask is None and query.shape[2] > 1) if is_causal is None else is_causal
        tensors, scalars = [query, key, value, mask], [float(causal), float(scaling)]
        if collector is not None:
            collector.capture(module._sako_path + ".__core__", "attention", tensors, scalars)
        output = operation(tensors, scalars) if operation is not None else reference("attention", tensors, scalars)
        return output.transpose(1, 2).contiguous(), None

    ALL_ATTENTION_FUNCTIONS["sdpa"] = dispatch


class Collector:
    def __init__(self, config):
        self.config, self.tasks, self.cases, self.events = config, {}, {}, {}
        self.weights = {}

    def capture(self, path, kind, tensors, scalars):
        import torch
        key = task_key(kind, tensors, scalars)
        task = self.tasks.setdefault(key, {"id": key, "kind": kind, "paths": [], "scalars": scalars,
                                          "cases": [], "priority_ms": 0.})
        if path not in task["paths"]:
            task["paths"].append(path)
        cases = self.cases.setdefault(key, [])
        if path == task["paths"][0] and len(cases) < self.config["max_fixture_cases"]:
            cpu = []
            for i, tensor in enumerate(tensors):
                # Share CPU weight storage across cases in each fixture.
                if kind != "attention" and i > 0:
                    if (key, i) not in self.weights:
                        self.weights[key, i] = tensor.detach().cpu()
                    cpu.append(self.weights[key, i])
                else:
                    cpu.append(tensor.detach().cpu())
            cases.append({"tensors": cpu, "scalars": scalars})
            task["cases"].append({"shapes": [list(t.shape) for t in tensors],
                                   "dtypes": [str(t.dtype) for t in tensors],
                                   "strides": [list(t.stride()) for t in tensors]})
        return key


def profile(config, output_dir):
    import torch
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from huggingface_hub import HfApi
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pinned = copy.deepcopy(config)
    pinned["revision"] = HfApi().model_info(config["model_id"], revision=config["revision"]).sha
    model = load_model(pinned)
    collector = Collector(config)
    handles, starts = [], {}
    original_attention = ALL_ATTENTION_FUNCTIONS["sdpa"]
    for path, module in model.named_modules():
        if type(module).__name__ == "Qwen2Attention":
            module._sako_path = path
        kind = module_kind(module)
        if kind is None:
            continue

        def before(mod, args, path=path, kind=kind):
            tensors, scalars = arguments(kind, mod, args[0])
            key = collector.capture(path, kind, tensors, scalars)
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            starts[path] = (key, event)

        def after(mod, args, output, path=path):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            key, start = starts[path]
            collector.events.setdefault(key, []).append((start, end))

        handles.extend([module.register_forward_pre_hook(before), module.register_forward_hook(after)])
    try:
        install_attention_dispatch(collector)
        with torch.inference_mode():
            for workload in config["workloads"]:
                for seed in config["correctness_seeds"][:2]:
                    forward(model, inputs(model, workload, seed))
        torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()
        ALL_ATTENTION_FUNCTIONS["sdpa"] = original_attention
    for key, task in collector.tasks.items():
        times = [a.elapsed_time(b) for a, b in collector.events.get(key, [])]
        if times:
            import statistics
            task["priority_ms"] = statistics.median(times) * len(task["paths"])
        else:
            # Attention has no module hook around its core. Use measured fixtures later.
            task["priority_ms"] = 1.
        torch.save({"cases": collector.cases[key]}, output_dir / (key + ".pt"))
    result = {"config": pinned, "tasks": list(collector.tasks.values()),
              "note": "Hook times guide search only. Nested module times overlap."}
    write_json(output_dir / "tasks.json", result)
    return result


def apply_manifest(model, manifest, task_catalog, candidate_root):
    import torch
    tasks = {t["id"]: t for t in task_catalog["tasks"]}

    class Replacement(torch.nn.Module):
        def __init__(self, original, kind, operation):
            super().__init__()
            self.original, self.kind, self.operation = original, kind, operation

        def forward(self, x):
            tensors, scalars = arguments(self.kind, self.original, x)
            return self.operation(tensors, scalars)

    install_attention_dispatch()
    for entry in manifest["entries"]:
        candidate = read_json(Path(candidate_root) / (entry["candidate_id"] + ".json"))
        task = tasks[candidate["task_id"]]
        operation = custom_op(candidate, task["kind"])
        for path in entry["paths"]:
            if path not in task["paths"]:
                raise ValueError("Manifest path is outside the task.")
            if task["kind"] == "attention":
                model.get_submodule(path.removesuffix(".__core__"))._sako_attention = operation
            else:
                parent, _, name = path.rpartition(".")
                owner = model.get_submodule(parent) if parent else model
                setattr(owner, name, Replacement(getattr(owner, name), task["kind"], operation))
    return model


def measure_model(config, manifest, catalog, candidate_root):
    import torch
    config = {**config, "revision": catalog["config"]["revision"]}
    model = load_model(config)
    alternative = clone_structure(model)
    apply_manifest(alternative, manifest, catalog, candidate_root)
    baseline = torch.compile(lambda ids: forward(model, ids), fullgraph=True, dynamic=False,
                             mode=config["compile_mode"])
    candidate = torch.compile(lambda ids: forward(alternative, ids), fullgraph=True, dynamic=False,
                              mode=config["compile_mode"])
    results = []
    with torch.inference_mode():
        for workload in config["workloads"]:
            checks = []
            eager_diagnostics = []
            for seed in config["correctness_seeds"]:
                ids = inputs(model, workload, seed)
                eager = forward(model, ids)
                expected = baseline(ids)
                eager_diagnostics.append(check_output(expected, eager, config))
                checks.append({**check_output(candidate(ids), expected, config), "seed": seed})
            item = {"workload": workload, "correct": all(c["passed"] for c in checks), "checks": checks,
                    "reference": "torch_compile", "compiled_vs_eager_diagnostics": eager_diagnostics}
            if item["correct"]:
                item.update(paired_timings(lambda: baseline(ids), lambda: candidate(ids), config, repeats=1))
            results.append(item)
    return {"scope": "complete_model_forward", "compiled": True, "fullgraph": True,
            "config": config, "cases": results, "decision": score_cases(results, config),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "manifest": manifest}
