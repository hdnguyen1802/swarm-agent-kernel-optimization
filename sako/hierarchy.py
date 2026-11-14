"""Select one compiled bottleneck at a time and search safe subsections."""

import copy

from .compiled_ranking import inside
from .hardware import select_top_tasks


def enabled(config):
    return config.get("agent", {}).get("target_selection", "flat") == "bottleneck_tree"


def contains(parent, child):
    return all(any(inside(path, root) for root in parent["paths"]) for path in child["paths"])


def select_hierarchy(tasks, *, include_parents=False):
    """Keep a full region plan. Only one region is active during search."""
    valid = [t for t in tasks if t.get("contract", {}).get("validated")
             and t.get("priority_ms", 0) > 0 and not t.get("ranking_excluded_reason")]
    roots = select_top_tasks(valid, max(1, len(valid)))["selected_task_ids"]
    by_id = {t["id"]: t for t in tasks}
    regions, selected = [], []
    for key in roots:
        root = by_id[key]
        descendants = [t for t in valid if t["id"] != key and contains(root, t)]
        # Search the smallest supported contracts. Preserve any fused group that
        # cannot fit a child contract as evidence at the containing region.
        leaves = [t for t in descendants if not any(t["id"] != other["id"] and contains(t, other)
                  for other in descendants)]
        children = select_top_tasks(leaves, max(1, len(leaves)))["selected_task_ids"] if leaves else [key]
        covered = sum(by_id[child]["priority_ms"] for child in children)
        alternatives = [key] if include_parents and children != [key] else []
        search_tasks = children + alternatives
        region = {"id": key, "kind": root["kind"], "paths": root["paths"],
            "priority_ms": root["priority_ms"], "subsection_ids": search_tasks,
            "leaf_task_ids": children, "alternative_task_ids": alternatives,
            "covered_gpu_ms": covered, "unassigned_gpu_ms": max(0., root["priority_ms"] - covered),
            "unassigned_reason": "Fused work without a smaller verified contract remains on the compiled baseline."}
        regions.append(region)
        for child in search_tasks:
            by_id[child]["hierarchy_context"] = {"region_id": key, "region_kind": root["kind"],
                "region_gpu_ms": root["priority_ms"], "subsection_gpu_ms": by_id[child]["priority_ms"],
                "siblings": search_tasks, "unassigned_gpu_ms": region["unassigned_gpu_ms"],
                "search_scope": "parent_alternative" if child in alternatives else "subsection",
                "policy": ("Optimize the complete parent contract. Test wider fusion and preserve original rounding. "
                           "Parent and child replacements are measured as separate model alternatives." if child in alternatives else
                           "Optimize only the assigned operation. Preserve its boundary and existing fusion. "
                           "Different shapes can use different kernel designs.")}
        selected.extend(search_tasks)
    return {"selected_task_ids": selected, "requested_count": None, "available_count": len(selected),
        "policy": "One bottleneck region at a time, then its safe subsections. No fixed top-K limit.",
        "hierarchy": {"schema_version": 1, "regions": regions,
                      "initial_region_id": regions[0]["id"] if regions else None}}


def region_for(catalog, task_id):
    return next((r for r in catalog.get("hierarchy", {}).get("regions", []) if task_id in r["subsection_ids"]), None)


def validate_search_overlaps(catalog, tasks, selected):
    """Overlap is allowed only between a declared parent alternative and its leaves."""
    allowed = set()
    for region in catalog.get("hierarchy", {}).get("regions", []):
        for parent in region.get("alternative_task_ids", []):
            if parent != region["id"] or parent not in tasks:
                raise ValueError("Invalid parent alternative in the profile.")
            for child in region.get("leaf_task_ids", []):
                if child not in tasks or not contains(tasks[parent], tasks[child]):
                    raise ValueError("The parent does not contain its declared subsection.")
                allowed.add((parent, child))
    for index, key in enumerate(selected):
        for other in selected[:index]:
            overlap = any(inside(a, b) or inside(b, a) for a in tasks[key]["paths"] for b in tasks[other]["paths"])
            if overlap and (key, other) not in allowed and (other, key) not in allowed:
                raise ValueError("Selected operation regions overlap outside a declared alternative.")


def validate_queue(catalog, progress):
    expected, queue = catalog["selected_task_ids"], progress["queue"]
    if len(queue) != len(expected) or set(queue) != set(expected) or len(set(queue)) != len(queue):
        raise ValueError("The hierarchy queue changed its subsection set.")
    completed = [r["task_id"] for r in progress.get("completed", [])]
    if completed != queue[:progress["index"]]:
        raise ValueError("The hierarchy queue changed completed work.")
    order = [region_for(catalog, key)["id"] for key in queue]
    closed, previous = set(), None
    for root in order:
        if root != previous:
            if root in closed:
                raise ValueError("Hierarchy regions must remain contiguous.")
            closed.add(root)
            previous = root


def reorder_remaining(catalog, progress, priorities):
    """Only reorder future regions. Never reset completed work or its evidence."""
    result = copy.deepcopy(progress)
    tail = progress["queue"][progress["index"]:]
    regions = [r for r in catalog["hierarchy"]["regions"] if any(k in tail for k in r["subsection_ids"])]
    regions.sort(key=lambda r: (-priorities.get(r["id"], {}).get("priority_ms", r["priority_ms"]), r["id"]))
    result["queue"] = progress["queue"][:progress["index"]] + [key for r in regions for key in r["subsection_ids"] if key in tail]
    validate_queue(catalog, result)
    return result


def containing_regions(catalog, manifest):
    """Return supported parent boundaries for composed eager accuracy checks."""
    paths = [path for entry in manifest["entries"] for path in entry["paths"]]
    return [r for r in catalog.get("hierarchy", {}).get("regions", []) if r["kind"] == "mlp"
            and r["subsection_ids"] != [r["id"]]
            and any(path != root and inside(path, root) for path in paths for root in r["paths"])]
