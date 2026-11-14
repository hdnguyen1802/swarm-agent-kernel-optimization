"""Add bounded launch tuning without changing the writer's algorithm."""

import ast
from contextlib import contextmanager
import math
import statistics

from .core import digest, read_json, write_json

MARKER = "SAKO_HARNESS_TUNING"
DEFAULTS = {"enabled": False, "max_configs": 6, "warmup": 2, "trials": 7, "repeats": 10}


def policy(config):
    settings = {**DEFAULTS, **config.get("tuning", {})}
    if type(settings["enabled"]) is not bool:
        raise ValueError("Use a boolean tuning.enabled.")
    for key, lower, upper in (("max_configs", 2, 16), ("warmup", 1, 20), ("trials", 5, 15), ("repeats", 1, 100)):
        if type(settings[key]) is not int or not lower <= settings[key] <= upper:
            raise ValueError(f"Use tuning.{key} from {lower} to {upper}.")
    return settings


def qualified(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return qualified(node.value) + "." + node.attr
    return ""


def prepare(candidate, config):
    settings = policy(config)
    if not settings["enabled"]:
        return candidate, {"enabled": False}
    from .triton_backend import validate_source
    tree = validate_source(candidate["source"])
    aliases = {entry.asname or entry.name.split(".")[0]: entry.name if entry.asname else entry.name.split(".")[0]
               for node in tree.body if isinstance(node, ast.Import) for entry in node.names}
    triton_name = next((entry.asname or "triton" for node in tree.body if isinstance(node, ast.Import)
                        for entry in node.names if entry.name == "triton"), "triton")
    def name_of(node):
        name = qualified(node)
        head, dot, tail = name.partition(".")
        return aliases.get(head, head) + (dot + tail if dot else "")
    tree.body = [node for node in tree.body if not (isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == MARKER for target in node.targets))]
    constants = {target.id: node.value for node in tree.body if isinstance(node, ast.Assign)
                 for target in node.targets if isinstance(target, ast.Name)}
    def literal(node):
        return ast.literal_eval(constants.get(node.id, node) if isinstance(node, ast.Name) else node)
    changes = []
    for function in [node for node in tree.body if isinstance(node, ast.FunctionDef)]:
        if not any(name_of(d) == "triton.jit" for d in function.decorator_list):
            continue
        autotune = next((d for d in function.decorator_list if isinstance(d, ast.Call)
                         and name_of(d.func) == "triton.autotune"), None)
        if autotune is not None:
            for keyword in autotune.keywords:
                if keyword.arg == "configs":
                    values = constants.get(keyword.value.id) if isinstance(keyword.value, ast.Name) else keyword.value
                    keyword.value = ast.List(elts=values.elts[:settings["max_configs"]], ctx=ast.Load())
            changes.append({"kernel": function.name, "kind": "writer_block_configs", "bounded": True})
            continue
        launches = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Subscript) and qualified(node.func.value) == function.name]
        constexpr = [arg.arg for arg in function.args.args if name_of(arg.annotation) == "triton.language.constexpr"]
        if not launches or not constexpr:
            changes.append({"kernel": function.name, "status": "skipped",
                            "reason": "No direct launch or no constexpr shape key. Existing launch remains unchanged."})
            continue
        bases = []
        try:
            for launch in launches:
                values = {keyword.arg: literal(keyword.value) for keyword in launch.keywords
                          if keyword.arg in {"num_warps", "num_stages"}}
                bases.append((values.get("num_warps", 4), values.get("num_stages", 3)))
        except (ValueError, TypeError):
            bases = []
        if not bases or len(set(bases)) != 1 or any(type(v) is not int for v in bases[0]):
            changes.append({"kernel": function.name, "status": "skipped",
                            "reason": "Launch settings vary at runtime. Declare legal triton.Config choices to tune them."})
            continue
        warps, stages = bases[0]
        choices = list(dict.fromkeys([(warps, stages), (warps, 1), (warps, 2),
                                     (max(1, warps // 2), stages), (min(8, warps * 2), stages), (warps, 3)]))
        configs = [ast.Call(func=ast.Attribute(value=ast.Name(id=triton_name, ctx=ast.Load()), attr="Config", ctx=ast.Load()),
                    args=[ast.Dict(keys=[], values=[])], keywords=[ast.keyword(arg="num_warps", value=ast.Constant(w)),
                    ast.keyword(arg="num_stages", value=ast.Constant(s))]) for w, s in choices[:settings["max_configs"]]]
        for launch in launches:
            launch.keywords = [kw for kw in launch.keywords if kw.arg not in {"num_warps", "num_stages"}]
        decorator = ast.Call(func=ast.Attribute(value=ast.Name(id=triton_name, ctx=ast.Load()), attr="autotune", ctx=ast.Load()),
            args=[], keywords=[ast.keyword(arg="configs", value=ast.List(elts=configs, ctx=ast.Load())),
                               ast.keyword(arg="key", value=ast.List(elts=[ast.Constant(key) for key in constexpr], ctx=ast.Load()))])
        function.decorator_list.insert(0, decorator)
        changes.append({"kernel": function.name, "kind": "harness_launch_configs", "keys": constexpr,
                        "configs": [{"num_warps": w, "num_stages": s} for w, s in choices[:settings["max_configs"]]]})
    details = {"enabled": True, "version": 1, "writer_source_hash": digest(candidate["source"]),
               "settings": settings, "kernels": changes}
    tree.body.append(ast.Assign(targets=[ast.Name(id=MARKER, ctx=ast.Store())], value=ast.parse(repr(details), mode="eval").body))
    source = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    validate_source(source)
    return {**candidate, "source": source}, details


def declared(candidate):
    for node in ast.parse(candidate["source"]).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == MARKER for t in node.targets):
            value = ast.literal_eval(node.value)
            return value if isinstance(value, dict) and value.get("version") == 1 else None
    return None


@contextmanager
def base_configurations(tuners):
    """Check the first legal configuration before any tuning benchmark."""
    saved = []
    try:
        for _, tuner in tuners():
            saved.append((tuner, tuner.configs, tuner.cache))
            tuner.configs, tuner.cache = tuner.configs[:1], {}
        yield
    finally:
        for tuner, configs, cache in reversed(saved):
            tuner.configs, tuner.cache = configs, cache


def install_benchmarks(tuners, declaration, directory, hardware=None):
    """Keep raw samples for every tried configuration. Never alter acceptance timing."""
    from triton.compiler.errors import CompilationError
    from triton.runtime.errors import OutOfResources
    from .triton_backend import config_data
    history = []
    settings = policy({"tuning": declaration["settings"]})
    for name, tuner in tuners():
        def benchmark(*args, config, _tuner=tuner, _name=name, **meta):
            from .gpu import tuning_timings
            conflicts = meta.keys() & config.kwargs.keys()
            if conflicts:
                raise ValueError("Conflicting tuning parameters: " + ", ".join(sorted(conflicts)))
            kwargs = {**meta, **config.all_kwargs()}
            invoke = lambda: _tuner.fn.run(*args, **kwargs)
            inputs = {**dict(zip(_tuner.arg_names, args)), **meta}
            key = {name: value for name, value in inputs.items() if type(value) in (int, float, str, bool)}
            record = {"kernel": _name, **config_data(config), "key": key, "acceptance_evidence": False,
                      "timing_mode": "cuda_graph", "samples_ms": []}
            cached_path = None
            try:
                if hardware and getattr(_tuner.fn, "cache_key", None):
                    layout = {}
                    for name, value in inputs.items():
                        if hasattr(value, "stride") and hasattr(value, "dtype"):
                            layout[name] = {"shape": list(value.shape), "stride": list(value.stride()),
                                            "dtype": str(value.dtype), "alignment": value.data_ptr() % 128}
                        elif name == "grid":
                            layout[name] = list(value({**inputs, **config.all_kwargs()})) if callable(value) else value
                        elif type(value) in (int, float, str, bool, type(None)):
                            layout[name] = value
                        else:
                            break
                    else:
                        identity = digest({"kernel": _tuner.fn.cache_key, "hardware": hardware["fingerprint"],
                                           "layout": layout, "config": config_data(config), "settings": settings})
                        cached_path = directory.parent / "tuning-cache" / (identity + ".json")
                if cached_path and cached_path.is_file():
                    try:
                        cached = read_json(cached_path)
                        saved = cached["record"]
                        samples = saved["samples_ms"]
                        if (cached["hash"] == digest(saved) and len(samples) == settings["trials"] and
                                all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in samples)):
                            record.update(saved, kernel=_name, cache_reused=True, cache_artifact=str(cached_path))
                            ordered = sorted(samples)
                            return [statistics.median(samples), ordered[int(.2*(len(samples)-1))],
                                    ordered[int(.8*(len(samples)-1))]]
                    except (ValueError, KeyError, TypeError):
                        pass
                from .triton_resources import kernel_resources
                record["resources"] = kernel_resources(_name, invoke())
                timed = tuning_timings(invoke, settings)
                samples = timed["candidate"]["samples_ms"]
                record.update(samples_ms=samples, setup=timed.get("setup_times", {}), status="measured")
                if cached_path:
                    write_json(cached_path, {"record": record, "hash": digest(record)})
                ordered = sorted(samples)
                return [statistics.median(samples), ordered[int(.2 * (len(samples) - 1))], ordered[int(.8 * (len(samples) - 1))]]
            except Exception as error:
                record.update(status="failed", error=f"{type(error).__name__}: {error}")
                if not isinstance(error, (CompilationError, OutOfResources)):
                    raise
                return [math.inf, math.inf, math.inf]
            finally:
                history.append(record)
                write_json(directory / "tuning-trials.json", {"declaration": declaration, "trials": history})
        tuner._bench = benchmark
    return history
