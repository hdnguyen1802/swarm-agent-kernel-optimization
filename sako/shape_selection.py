"""Select measured shapes without changing accuracy or model speed limits."""

import copy

from .core import digest, score_cases


def enabled(config):
    return config.get("kernel_search", {}).get("shape_scoped", False)


def case_signature(case, scalars=()):
    fields = ("shapes", "strides", "dtypes")
    if any(not isinstance(case.get(k), list) or not case[k] for k in fields):
        raise ValueError("Shape selection requires saved shapes, strides, and dtypes.")
    if len({len(case[k]) for k in fields}) != 1:
        raise ValueError("Tensor metadata has different lengths.")
    return {**{k: copy.deepcopy(case[k]) for k in fields}, "scalars": list(scalars)}


def groups(task):
    result = {}
    for index, case in enumerate(task["cases"]):
        signature = case_signature(case, task.get("scalars", []))
        path = case.get("source_path") if task.get("separate_operations") else None
        if task.get("separate_operations") and path not in task["paths"]:
            raise ValueError("A measured operation needs its original module path.")
        key = digest([path, signature]) if path else digest(signature)
        group = result.setdefault(key, {"signature": signature, "indices": [], **({"paths": [path]} if path else {})})
        group["indices"].append(index)
    return result


def decide(result, task, config):
    """All accuracy cases must pass. Apply speed gates to each complete shape."""
    cases = result.get("cases", [])
    complete = (len(cases) == len(task["cases"]) and bool(cases)
                and all(case.get("correct") is True and case.get("inputs_unchanged", True)
                        for case in cases) and result.get("reference_state_intact") is not False)
    decisions = {}
    decode = result.get("performance_scope") == "cached_decode"
    if decode:
        from .store import operation_correct
        complete = len(cases) == len(task["cases"]) and operation_correct(result) and result.get("reference_state_intact") is not False
    for key, group in groups(task).items():
        if decode:
            indices = [i for i in group["indices"] if task["cases"][i].get("generation_step", {}).get("phase") == "decode"]
            if not indices:
                continue
            group = {**group, "indices": indices}
        selected = [cases[i] for i in group["indices"] if i < len(cases)]
        decision = score_cases(selected, config)
        if not complete:
            decision["accepted"] = False
            decision["reasons"].append("Every required accuracy case must pass before retaining a shape.")
        decisions[key] = {**group, **decision}
    return decisions


def matches(tensors, scalars, signature):
    """Use only tensor metadata. Dynamo can guard this static shape branch."""
    if len(tensors) != len(signature["shapes"]) or tuple(scalars) != tuple(signature["scalars"]):
        return False
    for index, tensor in enumerate(tensors):
        if (tuple(tensor.shape) != tuple(signature["shapes"][index])
                or tuple(tensor.stride()) != tuple(signature["strides"][index])
                or str(tensor.dtype) != signature["dtypes"][index]):
            return False
    return True


def select_variant(variants, tensors, scalars):
    for variant in variants:
        if "signatures" not in variant or any(matches(tensors, scalars, s) for s in variant["signatures"]):
            return variant["candidate_id"]
    return None


def validate_entries(entries, tasks):
    """Only disjoint, catalog-backed shapes may share a replacement path."""
    seen = []
    for entry in entries:
        task = tasks.get(entry["task_id"])
        if task is None or not entry.get("paths"):
            raise ValueError("Every replacement needs a known task and module paths.")
        signatures = entry.get("signatures")
        keys = None
        if signatures is not None:
            if not isinstance(signatures, list) or not signatures:
                raise ValueError("A shape replacement needs at least one signature.")
            keys = {digest(s) for s in signatures}
            available = groups(task)
            if len(keys) != len(signatures) or not keys.issubset({digest(g["signature"]) for g in available.values()}):
                raise ValueError("The replacement shapes differ from the verified catalog.")
        for path in entry["paths"]:
            if path not in task["paths"]:
                raise ValueError("Manifest path is outside the task.")
            if keys and task.get("separate_operations") and not keys.issubset({
                    digest(g["signature"]) for g in available.values() if path in g["paths"]}):
                raise ValueError("The replacement shape was not measured on this module path.")
            for previous, previous_task, previous_keys in seen:
                overlap = path == previous or path.startswith(previous + ".") or previous.startswith(path + ".")
                if overlap and (path != previous or previous_task != entry["task_id"]
                                or keys is None or previous_keys is None or keys & previous_keys):
                    raise ValueError("Duplicate replacement paths or shapes overlap; replacements must be nonoverlapping.")
            seen.append((path, entry["task_id"], keys))
