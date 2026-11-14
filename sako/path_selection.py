"""Retain only paths with complete eager checks for an exact input signature."""

import copy

from .core import digest


def manifest_allowed(store, manifest):
    """A failed source may return only inside its recorded, narrower scope."""
    blocked = store.get("blocked_model_candidates", {})
    restrictions = store.get("candidate_scope_restrictions", {})
    for entry in manifest.get("entries", []):
        key = entry["candidate_id"]
        if key not in blocked and key not in restrictions:
            continue
        rule = restrictions.get(key, {})
        shapes = rule.get("shape_paths", {})
        if (entry.get("task_id") != rule.get("task_id") or not entry.get("paths")
                or not entry.get("signatures")
                or entry.get("scope_revision") != rule.get("failure_result_hash")):
            return False
        for signature in entry["signatures"]:
            if not set(entry["paths"]).issubset(shapes.get(digest(signature), [])):
                return False
    return True


def restricted_manifest(store, manifest):
    """Build a smaller trial after an attributable failure changes its scope.

    The caller must measure this manifest before accepting it. Every retry
    removes at least one path/signature pair. A revision alone cannot cause a
    retry loop.
    """
    restrictions = store.get("candidate_scope_restrictions", {})
    blocked = store.get("blocked_model_candidates", {})
    if not any(entry["candidate_id"] in restrictions for entry in manifest["entries"]):
        return None
    entries = []
    for entry in manifest["entries"]:
        rule = restrictions.get(entry["candidate_id"])
        if rule is None:
            if entry["candidate_id"] not in blocked:
                entries.append(copy.deepcopy(entry))
            continue
        if rule["task_id"] != entry["task_id"]:
            continue
        grouped = {}
        for signature in entry.get("signatures", []):
            allowed = rule["shape_paths"].get(digest(signature), [])
            paths = tuple(path for path in entry["paths"] if path in allowed)
            if paths:
                item = grouped.setdefault(paths, {**copy.deepcopy(entry), "paths": list(paths),
                    "signatures": [], "scope_revision": rule["failure_result_hash"]})
                item["signatures"].append(copy.deepcopy(signature))
        entries.extend(grouped.values())
    def scopes(values):
        return {(entry["task_id"], entry["candidate_id"], path, digest(signature))
                for entry in values for path in entry["paths"]
                for signature in entry.get("signatures", [None])}
    narrowed = {**copy.deepcopy(manifest), "entries": entries}
    if not entries or not scopes(entries) < scopes(manifest["entries"]):
        return None
    return narrowed if manifest_allowed(store, narrowed) else None


def unsafe_evidence(value, *, diagnostic=False):
    """Diagnostic errors do not block, but mutations always block salvage."""
    if not isinstance(value, dict):
        return False
    diagnostic = diagnostic or value.get("acceptance_role") == "diagnostic"
    if (value.get("reference_state_intact") is False or value.get("restoration_errors")
            or value.get("inputs_unchanged") is False):
        return True
    if value.get("not_measured") is True and not diagnostic:
        return True
    for key in ("cases", "checks", "layers", "check", "region_checks"):
        children = value.get(key)
        if isinstance(children, dict) and unsafe_evidence(children, diagnostic=diagnostic):
            return True
        if isinstance(children, list) and any(unsafe_evidence(child, diagnostic=diagnostic) for child in children):
            return True
    return False


def _signature(layer):
    keys = ("input_shapes", "input_strides", "input_dtypes", "input_scalars")
    if any(key not in layer for key in keys):
        return None
    return digest(dict(zip(("shapes", "strides", "dtypes", "scalars"),
                           (layer[key] for key in keys))))


def _overlap(left, right):
    return left == right or left.startswith(right + ".") or right.startswith(left + ".")


def passing_paths(measured, candidate_id, signature, config, failures):
    """Return provisional paths. Missing evidence never counts as a pass.

    All configured cases and seeds must be present. Each retained path must
    pass every observed call for this signature. Model speed still needs a
    fresh measurement after the manifest removes failed paths.
    """
    result = measured["result"]
    if (not config.get("kernel_search", {}).get("path_scoped")
            or result.get("correctness_scope") != "replaced_layers"
            or result.get("correctness_reference") != "eager"
            or unsafe_evidence(result)):
        return []
    seeds = config.get("correctness_seeds", [])
    workloads = config.get("workloads", [])
    if not seeds or not workloads:
        return []
    key = digest(signature)
    paths = list(dict.fromkeys(path for entry in measured["manifest"]["entries"]
        if entry["candidate_id"] == candidate_id
        and any(digest(value) == key for value in entry.get("signatures", []))
        for path in entry["paths"]))
    if not paths:
        return []
    cases = result.get("cases", [])
    # Duplicate or missing cases cannot prove the configured coverage.
    if len(cases) != len(workloads) or {digest(c.get("workload")) for c in cases} != {digest(w) for w in workloads}:
        return []
    evidence = {path: [] for path in paths}
    for case in cases:
        from .generation import evaluation_seeds
        seeds = evaluation_seeds(config, case["workload"])
        checks = [c for c in case.get("checks", []) if c.get("scope") != "cached_generation"]
        if len(checks) != len(seeds) or {c.get("seed") for c in checks} != set(seeds):
            return []
        by_seed = {}
        for check in checks:
            if (check.get("error") or check.get("missing_paths") or check.get("not_measured")
                    or check.get("reference") != "eager" or check.get("scope") != "replaced_layers"):
                return []
            if not set(paths).issubset(check.get("expected_paths", [])) or not set(paths).issubset(check.get("observed_paths", [])):
                return []
            layers = [layer for layer in check.get("layers", []) if layer.get("scope") != "containing_region"]
            # Never salvage a source after an input mutation, even if restore
            # succeeded. A numerical error and a mutation are different faults.
            if any(layer.get("inputs_unchanged") is not True or layer.get("error") for layer in layers):
                return []
            by_seed[check["seed"]] = layers
        # A workload is relevant when any call selected this exact signature.
        relevant = any(layer.get("candidate_id") == candidate_id and _signature(layer) == key
                       for layers in by_seed.values() for layer in layers if layer.get("path") in paths)
        if not relevant:
            continue
        for path in paths:
            passed = True
            for seed, layers in by_seed.items():
                selected = [layer for layer in layers if layer.get("path") == path
                            and layer.get("candidate_id") == candidate_id and _signature(layer) == key]
                passed = passed and bool(selected) and all(
                    layer.get("passed") is True and layer.get("same_input") is True
                    and layer.get("reference") == "eager" and not layer.get("original_fallback")
                    and layer.get("check", {}).get("passed") is True for layer in selected)
                if any(f.get("seed") == seed and f.get("workload") == case["workload"]
                       and _overlap(path, f["path"]) for f in failures):
                    passed = False
            evidence[path].append(passed)
    return [path for path, checks in evidence.items() if checks and all(checks)]
