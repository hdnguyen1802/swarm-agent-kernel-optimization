"""Rank remaining compiled GPU work using compiler source maps.

This adapter reads Inductor diagnostic output. Unknown formats fail closed.
It never inserts module hooks or changes fusion to obtain layer timings.
"""

import ast
from collections import defaultdict
import json
from pathlib import Path
import re
import statistics

from .compiled_opportunities import launch_kind
from .core import digest, require_current_settings, write_json
from .ranking import CASE

METHOD = "compiled_profiler_source_groups_v1"


def module_path(expression):
    """Normalize Dynamo module paths, without evaluating Python expressions."""
    expression = re.sub(r"^L\[['\"][^'\"]+['\"]\]", "", expression)
    expression = expression.replace("._modules", "")
    expression = re.sub(r"\[['\"]([^'\"]+)['\"]\]", r".\1", expression)
    expression = re.sub(r"\[(\d+)\]", r".\1", expression)
    return expression.strip(".")


def graph_sources(graph, module_names):
    result = {}
    inputs = {node: index for index, node in enumerate(n for n in graph.nodes if n.op == "placeholder")}
    for node in graph.nodes:
        if node.op not in {"call_function", "call_method", "call_module"}:
            continue
        paths = [module_path(value[0]) for value in node.meta.get("nn_module_stack", {}).values()]
        paths = [path for path in paths if path in module_names]
        value = node.meta.get("example_value")
        result[node.name] = {"name": node.name, "target": str(node.target),
            "path": max(paths, key=lambda p: (len(p.split('.')), len(p))) if paths else None,
            "shape": [str(x) for x in getattr(value, "shape", [])]}
        if str(node.target) in {"triton_kernel_wrapper_mutation", "triton_kernel_wrapper_functional"}:
            binding = user_triton_binding(node, inputs)
            if binding:
                result[node.name]["user_triton"] = binding
    return result


def user_triton_binding(node, inputs):
    """Record tensor input positions from the actual FX call and kernel signature."""
    from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table
    from torch.fx import Node
    kernel = kernel_side_table.get_kernel(node.kwargs["kernel_idx"])
    # Autotuner and Heuristics wrap the JIT function. Read metadata only.
    while hasattr(kernel, "fn") and not (hasattr(kernel, "arg_names") and hasattr(kernel, "__name__")):
        kernel = kernel.fn
    if not hasattr(kernel, "arg_names") or not hasattr(kernel, "__name__"):
        return None
    arguments = node.kwargs.get("kwargs", {})
    bindings = [{"argument": index, "input": inputs[arguments[name]]}
                for index, name in enumerate(kernel.arg_names)
                if name in arguments and isinstance(arguments[name], Node) and arguments[name] in inputs]
    return {"kernel": kernel.__name__, "inputs": bindings} if bindings else None


def user_triton_origins(call, tree, node, event, nodes):
    """Match a user launch by kernel name and exact graph-input positions.

    Inductor can omit source comments for higher-order Triton calls. The compiled
    callable keeps its FX input order. Use those identities, never launch order
    or tensor shape, to recover an owner. Ambiguous paths remain unmapped.
    """
    definitions = [n for n in tree.body if isinstance(n, ast.Assign)
                   and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                   and n.targets[0].id == event and isinstance(n.value, ast.Call)
                   and ast.unparse(n.value.func) == "async_compile.triton"
                   and n.value.args and isinstance(n.value.args[0], ast.Constant)]
    if len(definitions) != 1:
        return []
    name = definitions[0].value.args[0].value
    unpack = [n for n in call.body if isinstance(n, ast.Assign) and len(n.targets) == 1
              and isinstance(n.targets[0], (ast.Tuple, ast.List))
              and isinstance(n.value, ast.Name) and n.value.id == "args"
              and all(isinstance(item, ast.Name) for item in n.targets[0].elts)]
    if len(unpack) != 1 or unpack[0].lineno >= node.lineno:
        return []
    names = {item.id: index for index, item in enumerate(unpack[0].targets[0].elts)}
    # Ignore names that the compiler has assigned again before this launch.
    for statement in ast.walk(call):
        if isinstance(statement, ast.Name) and isinstance(statement.ctx, ast.Store):
            if unpack[0].lineno < statement.lineno < node.lineno:
                names.pop(statement.id, None)
    matches = []
    for key, origin in nodes.items():
        binding = origin.get("user_triton", {})
        required = binding.get("inputs", [])
        if binding.get("kernel") != name or not required:
            continue
        if all(item["argument"] < len(node.args)
               and isinstance(node.args[item["argument"]], ast.Name)
               and names.get(node.args[item["argument"]].id) == item["input"] for item in required):
            matches.append(key)
    paths = {nodes[key].get("path") for key in matches}
    return matches if len(paths) == 1 and None not in paths else []


def launch_plan(source, nodes, artifact):
    """Read call order and source comments. Do not execute generated text."""
    lines = source.splitlines()
    tree = ast.parse(source)
    calls = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "call"]
    # PyTorch 2.11 puts the same launch body in an unbased Runner class.
    # Read only this observed wrapper. Do not search arbitrary helper scopes.
    for wrapper in tree.body:
        if isinstance(wrapper, ast.ClassDef) and wrapper.name == "Runner" and not wrapper.bases:
            calls.extend(n for n in wrapper.body if isinstance(n, ast.FunctionDef)
                         and n.name == "call" and [arg.arg for arg in n.args.args] == ["self", "args"])
    if not calls:
        raise ValueError("Inductor output has no call function.")
    if len(calls) != 1:
        raise ValueError("Inductor output has multiple call entry points.")
    call = calls[0]
    user_kernels = {node.targets[0].id for node in ast.walk(tree)
                    if isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name) and isinstance(node.value, ast.Call)
                    and ast.unparse(node.value.func) == "async_compile.triton"}
    plan = []
    current = []
    statements = {node.lineno: node for node in ast.walk(call) if isinstance(node, ast.Call)
                  and (ast.unparse(node.func).startswith(("extern_kernels.", "torch.ops.aten."))
                       or (ast.unparse(node.func).endswith(".run") and
                           (ast.unparse(node.func).startswith("triton_")
                            or ast.unparse(node.func)[:-4] in user_kernels)))}
    for number in range(call.lineno, call.end_lineno + 1):
        match = re.search(r"# Topologically Sorted Source Nodes: \[([^]]*)\]", lines[number - 1])
        if match:
            current = [name.strip() for name in match[1].split(',') if name.strip()]
        node = statements.get(number)
        if node is None:
            continue
        name = ast.unparse(node.func)
        if name.endswith(".run") and (name.startswith("triton_") or name[:-4] in user_kernels):
            event = name[:-4]
            origins = current
            if not origins and event in user_kernels:
                origins = user_triton_origins(call, tree, node, event, nodes)
        elif name.startswith("extern_kernels.") or name.startswith("torch.ops.aten."):
            op = name.split('.')[1] if name.startswith("extern_kernels.") else name.split('.')[3]
            event = "aten::" + op
            origins = current
            # Inductor comments can include upstream, already executed sources.
            # For library GEMM calls, only the linear/mm source owns this launch.
            if op in {"mm", "addmm", "bmm"}:
                origins = [key for key in current if re.search(r"\b(linear|matmul|mm|addmm|bmm|baddbmm)\b",
                           nodes.get(key, {}).get("target", ""))]
        else:
            continue
        plan.append({"event": event, "implementation": ("user_triton" if event in user_kernels
                      and not event.startswith("triton_") else launch_kind(event)),
                     "source_nodes": [nodes[key] for key in origins if key in nodes],
                     "mapping_complete": bool(origins) and all(key in nodes for key in origins),
                     "artifact": artifact, "line": number})
    return plan


def compiled_records(events, plan, workload_index, trial):
    """Match each event class by exact name, count, and launch order."""
    expected, observed = defaultdict(list), defaultdict(list)
    for launch in plan:
        expected[launch["event"]].append(launch)
    for event in events:
        if str(event.device_type).rsplit('.', 1)[-1] != "CPU":
            continue
        elapsed = float(getattr(event, "self_device_time_total", 0.)) / 1000.
        if elapsed > 0:
            observed[event.name].append((event, elapsed))
    records = []
    for name, values in observed.items():
        launches = expected.get(name, [])
        exact = len(launches) == len(values)
        for index, (event, elapsed) in enumerate(values):
            launch = launches[index] if exact else {}
            records.append({**launch, "name": name, "time_ms": elapsed,
                "input_shapes": json.loads(json.dumps(event.input_shapes or [])),
                "alignment": "exact_name_count_order" if exact else "unmapped_event_count",
                "mapping_complete": bool(exact and launch.get("mapping_complete")),
                "workload_index": workload_index, "trial": trial})
    return records


def inside(path, root):
    return path is not None and (path == root or path.startswith(root + '.'))


def require_mapped_targets(ranking, tasks, output_dir):
    if not ranking["total_gpu_ms"] or not any(task["priority_ms"] > 0 for task in tasks):
        write_json(Path(output_dir) / "operation_profile.json", ranking)
        raise RuntimeError("Compiled profile has no safe, mapped replacement targets. See operation_profile.json.")


def rank_compiled(tasks, records, workload_count, trials, *, config=None):
    """Select only contracts that contain complete measured fusion groups."""
    if config is not None:
        require_current_settings(config)
    def trial_count(workload):
        return trials[workload] if isinstance(trials, dict) else trials
    groups = {}
    for record in records:
        key = digest([record["name"], record.get("source_nodes"), record.get("artifact"), record.get("line"),
                      record["workload_index"], record.get("input_shapes"), record.get("phase")])
        group = groups.setdefault(key, {k: v for k, v in record.items() if k not in {"time_ms", "trial"}})
        group.setdefault("samples_ms", []).append(record["time_ms"])
        group.setdefault("trial_totals_ms", [0.] * trial_count(record["workload_index"]))[record["trial"]] += record["time_ms"]
    groups = list(groups.values())
    for group in groups:
        group["implementation"] = group.get("implementation", launch_kind(group["name"]))
        sources = group.get("source_nodes", [])
        owners = [(task, path) for task in tasks for path in task["paths"]
                  if group["mapping_complete"] and sources and all(inside(n["path"], path) for n in sources)]
        owner = max(owners, key=lambda pair: len(pair[1]), default=None)
        group["task_id"] = owner[0]["id"] if owner else None
        group["path"] = owner[1] if owner else None
        group["median_ms"] = statistics.median(group["trial_totals_ms"])
        group["exclusion_reason"] = None if owner else "unmapped_or_unsupported_fusion_boundary"
    for task in tasks:
        contained, crossing = [], []
        for group in groups:
            sources = group.get("source_nodes", [])
            touches = any(inside(n["path"], path) for n in sources for path in task["paths"])
            fits = group["mapping_complete"] and sources and any(
                all(inside(n["path"], path) for n in sources) for path in task["paths"])
            if fits:
                contained.append(group)
            elif touches:
                crossing.append(group)
        medians = [statistics.median([sum(g["trial_totals_ms"][trial] for g in contained
                   if g["workload_index"] == workload) for trial in range(trial_count(workload))])
                   for workload in range(workload_count)]
        reason = None
        missing_paths = [{"path": path, "workload_index": workload} for path in task["paths"]
                         for workload in range(workload_count) if not any(g["workload_index"] == workload
                         and all(inside(n["path"], path) for n in g.get("source_nodes", [])) for g in contained)]
        if any("phase" in g for g in contained) and any(
                other["id"] != task["id"] and set(task["paths"]).intersection(other["paths"]) for other in tasks):
            reason = "multiple_contracts_at_same_compiled_path"
        elif crossing:
            reason = "cross_contract_fusion"
        elif task["kind"] == "mlp" and not any(g["task_id"] == task["id"] for g in contained):
            reason = "composite_region_without_cross_operation_fusion"
        elif any(value <= 0 for value in medians):
            reason = "missing_compiled_profiler_coverage"
        elif missing_paths:
            reason = "missing_compiled_path_coverage"
        task.update(priority_method=METHOD, workload_medians_ms=medians,
                    priority_ms=0. if reason else statistics.mean(medians))
        if reason:
            task["ranking_excluded_reason"] = reason
        else:
            task.pop("ranking_excluded_reason", None)
        task["compiled_evidence"] = {"method": METHOD, "contained_groups": len(contained),
            "crossing_groups": len(crossing), "fusion_boundary_safe": not crossing,
            "missing_path_count": len(missing_paths), "missing_paths": missing_paths[:8],
            "groups": [{**{k: g.get(k) for k in ("name", "implementation", "input_shapes", "median_ms", "artifact", "line")},
                        "source_nodes": g.get("source_nodes", [])[:8], "source_node_count": len(g.get("source_nodes", []))}
                       for g in sorted(contained + crossing, key=lambda g: -g["median_ms"])[:6]],
            "report": "operation_profile.json", "timing_scope": "complete compiled model",
            "note": "These are profiled GPU costs, not predicted gains. Do not break existing fusion."}
        if any("phase" in group for group in contained):
            task["compiled_evidence"]["generation_phase_ms"] = {phase: [
                statistics.median(sum(g["trial_totals_ms"][i] for g in contained
                    if g["workload_index"] == w and g.get("phase") == phase) for i in range(trial_count(w)))
                for w in range(workload_count)] for phase in ("prefill", "decode")}
    return {"method": METHOD, "model_execution": "torch_compile", "groups": groups, "samples": records,
        "mapped_gpu_ms": sum(r["time_ms"] for r in records if r.get("mapping_complete")),
        "total_gpu_ms": sum(r["time_ms"] for r in records),
        "note": "Rank complete supported contracts by compiled GPU time. Keep unsupported fused groups as evidence. "
                "Never divide one fused kernel time among its source operations. Eager outputs remain the accuracy reference."}


def profile_compiled(model, config, tasks, output_dir, *, starting=None, hardware=None):
    import torch
    from .compat import configure_compiler
    from .model import forward, inputs
    configure_compiler(config)
    from .measurement import enabled as adaptive, policy, ForwardRequest, saved_workload, warmup, profile_stability
    adaptive_mode = adaptive(config)
    if adaptive_mode and (starting is None or hardware is None):
        raise ValueError("Adaptive profiling requires the saved normal compiler baseline and hardware identity.")
    trial_counts = {}
    output_dir = Path(output_dir)
    debug = output_dir / "compiler"
    captures, records, results, kernel_sources = [], [], [], {}
    workload_index = 0
    executed = []
    options = dict(torch._inductor.list_mode_options(config["compile_mode"], dynamic=False))
    options.update({"trace.enabled": True, "triton.unique_kernel_names": True, "fx_graph_cache": False})

    def backend(graph, examples):
        directory = debug / f"graph-{len(captures)}"
        directory.mkdir(parents=True, exist_ok=True)
        nodes = graph_sources(graph.graph, set(dict(model.named_modules())))
        write_json(directory / "source-map.json", nodes)
        compiled = torch._inductor.compile(graph, examples, options={**options, "trace.debug_dir": str(directory.resolve())})
        files = list(directory.rglob("output_code.py"))
        if len(files) != 1:
            raise RuntimeError("Compiled ranking requires exactly one Inductor output file per full graph.")
        code = files[0]
        artifact = code.relative_to(output_dir).as_posix()
        plan = launch_plan(code.read_text(encoding="utf-8"), nodes, artifact)
        if not plan:
            raise RuntimeError("Compiled ranking could not read a launch plan. No eager ranking fallback is allowed.")
        for node in ast.walk(ast.parse(code.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                    and ast.unparse(node.value.func) == "async_compile.triton"
                    and len(node.value.args) >= 2 and isinstance(node.value.args[1], ast.Constant)):
                name = node.targets[0].id
                kernel_sources[(artifact, name)] = str(node.value.args[1].value)
        captures.append({"workload_index": workload_index, "shape": list(config["workloads"][workload_index].values()),
                         "plan": plan, "artifact": artifact})
        if config["workloads"][workload_index].get("mode") == "cached_generation":
            capture = captures[-1]
            def measured_graph(*args):
                executed.append(capture)
                return compiled(*args)
            return measured_graph
        return compiled

    from .generation import enabled as generation_workload
    compiled = torch.compile(lambda ids: forward(model, ids), backend=backend, fullgraph=True, dynamic=False)
    with torch.inference_mode():
        for workload_index, workload in enumerate(config["workloads"]):
            if generation_workload(workload):
                from .generation_profile import profile_request
                measurements, timing = profile_request(model, config, workload, workload_index, output_dir, backend, executed,
                    **({"starting": starting, "hardware": hardware, "tasks": tasks} if adaptive_mode else {}))
                records.extend(measurements)
                trial_counts[workload_index] = timing.get("profiling", {}).get("trials", config["trials"])
                results.append(timing)
                continue
            ids = inputs(model, workload, config["correctness_seeds"][-1 if adaptive_mode else 0])
            start = len(captures)
            saved = warm = check = None
            history = []
            if adaptive_mode:
                runner = ForwardRequest(model, config, workload, compiled)
                saved = saved_workload(starting, model, config, hardware, workload, runner.record)
                normal = ForwardRequest(model, config, workload)
                from .gpu import check_output
                check = check_output(runner.run(), normal.run(), config)
                write_json(output_dir / f"compiled-workload-{workload_index}-correctness.json", check)
                if not check["passed"]:
                    raise RuntimeError("The profiling compiler path changed the model output.")
                warm = warmup(lambda: runner.run(timed=True), config, saved["warmup"]["count"])
            else:
                for _ in range(config["warmup"]):
                    compiled(ids)
            torch.cuda.synchronize()
            # Static shape reuse may reuse the last graph. Never borrow another shape's plan.
            shape = list(workload.values())
            current = captures[start:] or [g for g in captures if g["shape"] == shape][-1:]
            if len(current) != 1:
                raise RuntimeError("Compiled ranking requires one verified static graph for each workload.")
            plan = current[0]["plan"]
            samples = list(saved["samples_ms"]) if adaptive_mode else []
            if not adaptive_mode:
                for _ in range(config["trials"]):
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record(); compiled(ids); end.record(); end.synchronize()
                    samples.append(begin.elapsed_time(end))
            for trial in range(policy(config)["profile_max_trials"] if adaptive_mode else config["trials"]):
                # A separate trace for each sample makes event-to-call ownership explicit.
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profiler:
                    with torch.profiler.record_function(CASE + str(workload_index) + "::" + str(trial)):
                        compiled(ids)
                    torch.cuda.synchronize()
                records.extend(compiled_records(profiler.events(), plan, workload_index, trial))
                if trial == 0:
                    profiler.export_chrome_trace(str(output_dir / f"compiled-workload-{workload_index}.trace.json"))
                if adaptive_mode and (trial + 1) % policy(config)["profile_batch_trials"] == 0:
                    status = profile_stability(tasks, records, workload_index, trial + 1, config, warmup_stable=warm["stable"])
                    history.append(status)
                    if not status["needs_more"]:
                        break
            trial_counts[workload_index] = trial + 1
            results.append({"workload": workload, "samples_ms": samples, "median_ms": statistics.median(samples),
                **({"starting_times_reused": True, "starting_identity_hash": saved["identity_hash"],
                    "profiling_correctness": check, "profiling_warmup": warm,
                    "profiling": {"trials": trial + 1, "assessments": history, "status": history[-1]["status"]}}
                   if adaptive_mode else {})})
    write_json(output_dir / "compiled_source_maps.json", captures)
    ranking = rank_compiled(tasks, records, len(config["workloads"]),
                            trial_counts if adaptive_mode else config["trials"], config=config)
    require_mapped_targets(ranking, tasks, output_dir)
    for task in tasks:
        examples = []
        seen = set()
        # Small fused kernels may not enter the six largest timing groups.
        # Still show their implementation so writers can see existing fusion.
        for group in sorted(ranking["groups"], key=lambda g: -g["median_ms"]):
            nodes = group.get("source_nodes", [])
            if not group["mapping_complete"] or not nodes or not any(
                    all(inside(node["path"], path) for node in nodes) for path in task["paths"]):
                continue
            identity = (group["artifact"], group["name"])
            source = kernel_sources.get(identity)
            if source and identity not in seen:
                examples.append({"name": group["name"], "artifact": group["artifact"],
                                 "source": source[:6000], "truncated": len(source) > 6000})
                seen.add(identity)
            if len(examples) == 2:
                break
        task["compiled_evidence"]["generated_kernels"] = examples
    return ranking, {"status": "passed",
        "search_ready": True, "fullgraph": True, "workloads": results,
        "region_attribution": METHOD, "note": ranking["note"],
        "diagnostic_options": options, "source_maps": "compiled_source_maps.json"}
