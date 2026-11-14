"""Load audited Python source for Triton. This is not a security sandbox."""

import ast
import math
import importlib.util
import sys
import time
import traceback
from collections import UserDict
from contextlib import contextmanager

from .core import digest, write_json

POLICY = "triton_python_v3"
_LOADED = {}


def config_data(config):
    return {"parameters": dict(config.kwargs), "num_warps": config.num_warps,
            "num_stages": config.num_stages, "launch_options": dict(config.all_kwargs())}


class ObservedChoices(UserDict):
    """Keep native autotuning behavior while recording the keys actually read."""

    def __init__(self, cache):
        self.data = cache
        self.used = {}
        self.invoked = False

    def observe_run(self, run):
        def invoke(*args, **kwargs):
            self.invoked = True
            return run(*args, **kwargs)
        return invoke

    def __getitem__(self, key):
        config = super().__getitem__(key)
        self.used[key] = config
        return config


def validate_source(source):
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise ValueError("Invalid Triton Python source: " + str(error)) from error
    imports = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            raise ValueError("Use module imports: torch, triton, and triton.language as tl.")
        if isinstance(node, ast.Import):
            for entry in node.names:
                if entry.name not in {"torch", "triton", "triton.language"}:
                    raise ValueError("Triton imports are limited to torch, triton, and triton.language.")
                imports[entry.asname or entry.name.split(".")[0]] = entry.name
    blocked = {"exec", "eval", "compile", "open", "input", "__import__", "globals", "locals", "getattr", "setattr",
               "delattr", "vars", "breakpoint", "print", "set_printoptions", "manual_seed", "seed", "load", "save",
               "synchronize", "do_bench", "perf_report", "time", "perf_counter", "Event", "sleep", "set_device",
               "set_default_dtype", "set_default_device", "register_hook", "set_stream", "cuda", "cpu", "numpy", "item"}
    allocators = {"torch.empty", "torch.empty_like", "torch.empty_strided"}
    entries = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "sako_run"]
    if len(entries) != 1 or [a.arg for a in entries[0].args.args] != ["tensors", "scalars"]:
        raise ValueError("Define sako_run(tensors, scalars).")
    def qualified(node):
        if isinstance(node, ast.Name):
            return imports.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return qualified(node.value) + "." + node.attr
        return ""
    jit_names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)
                 and any(qualified(d) == "triton.jit" for d in n.decorator_list)}
    if not jit_names:
        raise ValueError("Define at least one @triton.jit kernel.")
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and qualified(node.func) == "triton.autotune":
            values = {k.arg: k.value for k in node.keywords}
            configs = values.get("configs")
            if isinstance(configs, ast.Name):
                configs = next((n.value for n in tree.body if isinstance(n, ast.Assign)
                                and any(isinstance(t, ast.Name) and t.id == configs.id for t in n.targets)), None)
            if not isinstance(configs, (ast.List, ast.Tuple)) or not 1 <= len(configs.elts) <= 16:
                raise ValueError("Autotuning requires one to 16 explicit configurations.")
            if any(not isinstance(c, ast.Call) or qualified(c.func) != "triton.Config" for c in configs.elts):
                raise ValueError("Use explicit triton.Config entries for bounded tuning.")
            if set(values) - {"configs", "key", "warmup", "rep", "use_cuda_graph"}:
                raise ValueError("Autotune hooks and custom timing functions are not permitted.")
            for name, upper in (("warmup", 25), ("rep", 100)):
                value = values.get(name)
                if value is not None and (not isinstance(value, ast.Constant) or type(value.value) not in (int, float)
                                          or not 0 < value.value <= upper):
                    raise ValueError("Autotune warmup and repetition time must be bounded constants.")
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and (node.id in blocked or node.id.startswith("__")):
            raise ValueError("Triton source violates the candidate policy: " + node.id)
        if isinstance(node, ast.Attribute):
            if (node.attr in blocked and qualified(node) != "triton.language.load") or node.attr.startswith("__"):
                raise ValueError("Triton source violates the candidate policy: " + node.attr)
            if node.attr == "bfloat16" or node.attr.startswith("float8"):
                raise ValueError("Preserve FP16 precision. BF16 and FP8 are not permitted.")
        if isinstance(node, (ast.Global, ast.Nonlocal, ast.ClassDef, ast.AsyncFunctionDef, ast.With, ast.Try)):
            raise ValueError("Do not store state or suppress candidate errors.")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            raise ValueError("Use tl.dot inside a Triton kernel, not a Torch compute fallback.")
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(part, (ast.Attribute, ast.Subscript)) for target in targets for part in ast.walk(target)):
                raise ValueError("Do not mutate host objects, tensors, or output caches. Use tl.store in the kernel.")
        if isinstance(node, ast.Call):
            name = qualified(node.func)
            if name == "triton.Config" and any(kw.arg not in {"kwargs", "num_warps", "num_stages", "num_ctas", "maxnreg"}
                                               for kw in node.keywords):
                raise ValueError("Tuning configurations cannot install hooks or custom benchmark functions.")
            if name.startswith("torch.") and name not in allocators:
                raise ValueError("Torch may allocate outputs only. Use Triton for computation.")
            if name.startswith("triton.") and not name.startswith("triton.language.") and name not in {
                    "triton.jit", "triton.autotune", "triton.heuristics", "triton.Config", "triton.cdiv", "triton.next_power_of_2"}:
                raise ValueError("Unsupported Triton host API: " + name)
            for keyword in node.keywords:
                if keyword.arg == "allow_tf32" and not (isinstance(keyword.value, ast.Constant) and keyword.value.value is False):
                    raise ValueError("TF32 is not permitted.")
                if keyword.arg == "input_precision" and not (isinstance(keyword.value, ast.Constant) and keyword.value.value == "ieee"):
                    raise ValueError("Use IEEE input precision.")
    # Host wrappers can allocate and read metadata. Tensor methods must not do compute.
    for function in [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name not in jit_names]:
        for node in ast.walk(function):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                name = qualified(node.func)
                if not name.startswith(("torch.", "triton.")) and node.func.attr not in {
                        "numel", "size", "stride", "dim", "new_empty", "element_size", "is_contiguous"}:
                    raise ValueError("Host tensor methods may read metadata or allocate only.")
    for node in tree.body:
        if not isinstance(node, (ast.Import, ast.FunctionDef, ast.Assign, ast.AnnAssign, ast.Expr)):
            raise ValueError("Use imports, constants, and function definitions at module scope.")
        if isinstance(node, ast.Expr) and not isinstance(node.value, ast.Constant):
            raise ValueError("Do not execute kernels at module import.")
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            if any(isinstance(n, ast.Call) and qualified(n.func) != "triton.Config" for n in ast.walk(node)):
                raise ValueError("Module constants cannot execute candidate code.")
    return tree


def identity(candidate, hardware):
    return "sako_triton_" + digest({"source": candidate["source"], "hardware": hardware["fingerprint"], "policy": POLICY})[:24]


def load(candidate, hardware, directory):
    name = identity(candidate, hardware)
    path = directory / "candidate.py"
    path.write_text(candidate["source"], encoding="utf-8")
    write_json(directory / "source-policy.json", {"policy": POLICY, "backend": "triton",
        "source_hash": digest(candidate["source"]), "hardware_fingerprint": hardware["fingerprint"]})
    if name in _LOADED:
        return _LOADED[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        entry = module.sako_run
    except Exception:
        sys.modules.pop(name, None)
        raise
    first = True
    def invoke(tensors, scalars):
        nonlocal first
        from .gpu import build_log
        try:
            if first:
                start = time.perf_counter()
                with build_log(directory / "build.log"):
                    output = entry(tensors, scalars)
                write_json(directory / "first-launch.json", {"host_seconds": time.perf_counter() - start,
                    "note": "Includes first specialization and launch. Later shapes can compile separately. Not a pure compiler timer."})
                first = False
                return output
            return entry(tensors, scalars)
        except Exception as error:
            with (directory / "build.log").open("a", encoding="utf-8") as log:
                log.write(traceback.format_exc())
            error.build_log = str(directory / "build.log")
            error.tuning_log = str(directory / "tuning-trials.json")
            raise
    def tuners():
        from triton.runtime.autotuner import Autotuner
        from triton.runtime.jit import KernelInterface
        for key, value in vars(module).items():
            while isinstance(value, KernelInterface):
                if isinstance(value, Autotuner):
                    yield key, value
                value = getattr(value, "fn", None)

    def tuning_metadata():
        reports = []
        for key, value in tuners():
            reports.append({"kernel": key, "config_count": len(value.configs),
                "choices": [{"key": [str(v) for v in shape], "parameters": dict(config.kwargs),
                             "num_warps": config.num_warps, "num_stages": config.num_stages}
                            for shape, config in value.cache.items()],
                "last_trials": [{**config_data(config),
                    "timing": [v if math.isfinite(v) else str(v)
                               for v in (timing if isinstance(timing, (tuple, list)) else [timing])],
                    "finite": all(math.isfinite(v) for v in (timing if isinstance(timing, (tuple, list)) else [timing]))}
                    for config, timing in getattr(value, "configs_timings", {}).items()],
                "harness_trials": [trial for trial in harness_trials if trial["kernel"] == key],
                "note": "Autotune timings select parameters. Independent paired trials decide acceptance."})
        return reports

    @contextmanager
    def observe_tuning():
        records, saved = [], []
        inherited = object()
        try:
            for key, tuner in tuners():
                saved.append((key, tuner, tuner.cache, vars(tuner).get("run", inherited)))
                tuner.cache = ObservedChoices(tuner.cache)
                tuner.run = tuner.cache.observe_run(tuner.run)
            yield records
            for key, tuner, _, _ in saved:
                choices = []
                for shape, config in tuner.cache.used.items():
                    serializable = all(type(v) in (int, str, float, bool, type(None)) for v in shape)
                    choices.append({"cache_key": list(shape) if serializable else None, **config_data(config)})
                records.append({"kernel": key, "invoked": tuner.cache.invoked, "choices": choices,
                    "fixed": config_data(tuner.configs[0]) if len(tuner.configs) == 1 else None})
        finally:
            for _, tuner, cache, previous_run in reversed(saved):
                tuner.cache = cache
                if previous_run is inherited:
                    vars(tuner).pop("run", None)
                else:
                    tuner.run = previous_run

    from .automatic_tuning import declared, install_benchmarks, base_configurations
    declaration = declared(candidate)
    harness_trials = install_benchmarks(tuners, declaration, directory, hardware) if declaration else []
    invoke.base_configurations = lambda: base_configurations(tuners)
    invoke.observe_tuning = observe_tuning
    invoke.tuning_metadata = tuning_metadata
    from .triton_resources import observe_resources
    invoke.observe_resources = lambda: observe_resources(module)
    # Only the trusted integration bridge uses this validated source entry.
    # The logging wrapper stays on the eager path, outside Dynamo tracing.
    invoke._sako_compiler_entry = entry
    _LOADED[name] = invoke
    return invoke
