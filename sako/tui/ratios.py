"""Explain each speed ratio with its measured scope and comparison target."""

import json
import math
import statistics

from ..core import digest


def speed_text(ratio):
    if type(ratio) not in (int, float) or not math.isfinite(ratio) or ratio <= 0:
        return "not measured"
    change = 100 * (1 - 1 / ratio)
    direction = "less time" if change > 0 else "more time" if change < 0 else "same time"
    return f"{ratio:.3f}× · {abs(change):.1f}% {direction}"


def baseline_name(baseline, shape_id=None):
    baseline = baseline or {}
    kind = baseline.get("kind")
    if kind == "per_shape":
        shapes = baseline.get("shapes") or {}
        if shape_id not in shapes:
            return "recorded baseline"
        identity = shapes[shape_id]
        return "kept " + str(identity)[:12] if identity else "torch.compile"
    if kind == "torch_compile":
        return "torch.compile"
    if kind == "kept_kernel":
        identity = baseline.get("candidate_id")
        return "kept " + str(identity)[:12] if identity else "kept kernel"
    return "recorded baseline"


def _median(cases, indices, field):
    values = [cases[i].get(field) for i in indices if i < len(cases)]
    if values and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in values):
        return statistics.median(values)
    return None


def candidate_shapes(db, identity, selected=(), model_status="not used"):
    # Read summary fields only. Raw traces and timing arrays stay in the database.
    row = db.execute("""
        SELECT json_extract(result,'$.shape_decisions'), json_extract(result,'$.timing_baseline'),
               json_extract(result,'$.decision'),
               (SELECT json_group_array(json_object(
                   'baseline_ms',json_extract(c.value,'$.baseline.median_ms'),
                   'candidate_ms',json_extract(c.value,'$.candidate.median_ms')))
                FROM json_each(result,'$.cases') AS c)
        FROM candidates WHERE id=? AND result IS NOT NULL
    """, (identity,)).fetchone()
    if not row:
        return []
    shapes, baseline, decision, cases = [json.loads(v) if v else {} for v in row]
    used = set()
    used_paths = {}
    all_used = False
    for entry in selected:
        if entry.get("candidate_id") == identity:
            if "signatures" not in entry:
                all_used = True
            used.update(digest(s) for s in entry.get("signatures", []))
            for signature in entry.get("signatures", []):
                used_paths.setdefault(digest(signature), set()).update(entry.get("paths", []))
    rows = []
    for key, value in (shapes or {"all": {**decision, "indices": list(range(len(cases)))}}).items():
        signature = value.get("signature") or {}
        label = " ; ".join("×".join(str(n) for n in shape) for shape in signature.get("shapes", []))
        if value.get("paths"):
            label = value["paths"][0] + "  " + label
        indices = value.get("indices", [])
        rows.append({"shape_id": key, "shape": label or "all recorded shapes",
                     "speedup": value.get("speedup"), "accepted": value.get("accepted"),
                     "baseline": baseline_name(baseline, key),
                     "baseline_ms": _median(cases, indices, "baseline_ms"),
                     "candidate_ms": _median(cases, indices, "candidate_ms"),
                     "model_use": "used" if all_used or key in used or
                     bool(used_paths.get(digest(signature), set()).intersection(value.get("paths", []))) else model_status,
                     "reason": (value.get("reasons") or [""])[0]})
    return rows


def timing_text(row):
    if row.get("baseline_ms") is None or row.get("candidate_ms") is None:
        return "—"
    return f"{row['baseline_ms'] * 1000:.2f} → {row['candidate_ms'] * 1000:.2f}"
