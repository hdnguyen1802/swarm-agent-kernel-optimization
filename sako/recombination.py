"""Find two measured Triton designs that improve a shared input shape."""

import ast
import itertools
import math

from .core import digest
from .shape_selection import groups


def merge_key(parent_ids, shape_ids):
    return digest([sorted(parent_ids), sorted(shape_ids)])[:24]


def implementation(source):
    """Ignore comments and formatting when checking for unchanged parent code."""
    return ast.dump(ast.parse(source), include_attributes=False)


def improved_shapes(row, task):
    result = row.get("result") or {}
    if not row.get("correct") or row.get("source", {}).get("backend") != "triton":
        return {}
    try:
        catalog = groups(task)
    except (KeyError, ValueError):
        return {}
    decisions = result.get("shape_decisions")
    if decisions is None and result.get("decision", {}).get("accepted") is True:
        decisions = {key: result["decision"] for key in catalog}
    return {key: {"signature": catalog[key]["signature"], "speedup": value["speedup"]}
            for key, value in (decisions or {}).items()
            if key in catalog and value.get("accepted") is True
            and type(value.get("speedup")) in (int, float)
            and math.isfinite(value["speedup"]) and value["speedup"] > 1}


def merge_pairs(task, archive, rows, assignments=()):
    """Use the active archive, its measured shapes, and saved assignment history."""
    by_id = {row["id"]: row for row in rows if row["task"] == task["id"]}
    eligible = [parent for parent in archive if parent["id"] in by_id]
    used = {}
    for assignment in assignments:
        merge = assignment.get("hypothesis", {}).get("merge")
        if merge:
            used.setdefault(tuple(sorted(merge["parent_ids"])), set()).update(merge["shape_ids"])
    pairs = []
    for a, b in itertools.combinations(eligible, 2):
        left, right = by_id[a["id"]], by_id[b["id"]]
        if implementation(left["source"]["source"]) == implementation(right["source"]["source"]):
            continue
        first, second = improved_shapes(left, task), improved_shapes(right, task)
        shared = sorted(first.keys() & second.keys())
        if not shared:
            continue
        parents = sorted([a["id"], b["id"]])
        remaining = sorted(set(shared) - used.get(tuple(parents), set()))
        shapes = remaining or shared
        if task.get("separate_operations"):
            shapes = shapes[:4]
            shared = shapes
        pairs.append({"id": merge_key(parents, shapes), "parent_ids": parents, "shape_ids": shapes,
                      "shapes": {key: first[key]["signature"] for key in shared},
                      "parent_shape_speedups": {a["id"]: {key: first[key]["speedup"] for key in shared},
                                                b["id"]: {key: second[key]["speedup"] for key in shared}},
                      "specialties": [a.get("hypothesis", {}).get("specialty"), b.get("hypothesis", {}).get("specialty")],
                      "attempted": not remaining,
                      "source_characters": len(left["source"]["source"]) + len(right["source"]["source"])})
    return sorted(pairs, key=lambda p: (p["attempted"], p["specialties"][0] == p["specialties"][1], p["id"]))


def validate_merge(value, primary, pairs):
    if not isinstance(value, dict) or set(value) != {"parent_ids", "shape_ids"}:
        raise ValueError("A merge needs parent_ids and shape_ids.")
    parents, shapes = value["parent_ids"], value["shape_ids"]
    if (not isinstance(parents, list) or len(parents) != 2
            or any(not isinstance(p, str) for p in parents) or len(set(parents)) != 2 or primary not in parents):
        raise ValueError("A merge needs two distinct archive parents, including parent_id.")
    if (not isinstance(shapes, list) or not shapes or any(not isinstance(s, str) for s in shapes)
            or len(set(shapes)) != len(shapes)):
        raise ValueError("A merge needs distinct shared shape IDs.")
    pair = next((p for p in pairs if set(p["parent_ids"]) == set(parents)), None)
    if pair is None or not set(shapes).issubset(pair["shape_ids"]):
        raise ValueError("Both merge parents must improve each selected shape in the same task and archive.")
    return {"parent_ids": sorted(parents), "shape_ids": sorted(shapes)}
