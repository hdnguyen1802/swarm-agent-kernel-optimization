"""Describe code structure and divide writer work among measured designs."""

import ast
from collections import Counter
from functools import lru_cache

from .core import digest


def launch_stages(body):
    """Count sequential launch sites, taking the larger side of a branch."""
    count = 0
    for node in body:
        if isinstance(node, ast.If):
            count += max(launch_stages(node.body), launch_stages(node.orelse))
        elif isinstance(node, (ast.For, ast.While, ast.With)):
            count += launch_stages(node.body)
        else:
            count += sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Subscript) for n in ast.walk(node))
    return count


@lru_cache(maxsize=512)
def describe(source):
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
        t.id == "SAKO_HARNESS_TUNING" for t in node.targets))]
    calls = Counter(node.func.attr for node in ast.walk(tree)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute))
    launches = sum(isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript) for node in ast.walk(tree))
    entry = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "sako_run"), None)
    sequential = launch_stages(entry.body) if entry else launches
    stages = "multiple stages" if sequential > 1 else "one stage"
    arithmetic = " + ".join(name for name, used in (("matrix product", calls["dot"]),
        ("reduction", calls["sum"] or calls["max"] or calls["atomic_add"]),
        ("activation", calls["exp"] or calls["sigmoid"] or calls["exp2"])) if used) or "elementwise"
    features = {"launch_sites": launches, "sequential_launch_sites": sequential, "dot": calls["dot"], "sum": calls["sum"],
                "atomic_add": calls["atomic_add"], "loops": sum(isinstance(n, (ast.For, ast.While)) for n in ast.walk(tree)),
                "activation": sum(calls[name] for name in ("exp", "sigmoid", "tanh", "exp2"))}
    names = {}
    class Structure(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            node.decorator_list = []
            node.name = names.setdefault(node.name, "v" + str(len(names)))
            return self.generic_visit(node)

        def visit_Name(self, node):
            node.id = names.setdefault(node.id, "v" + str(len(names)))
            return node

        def visit_arg(self, node):
            node.arg = names.setdefault(node.arg, "v" + str(len(names)))
            return self.generic_visit(node)

        def visit_Constant(self, node):
            return ast.Constant(value="number") if type(node.value) in (int, float) else node

        def visit_Expr(self, node):
            return None if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str) else self.generic_visit(node)
    structure = ast.dump(Structure().visit(tree), include_attributes=False)
    return {"group": digest(structure)[:16], "label": stages + " / " + arithmetic, "features": features}


def writer_options(archive, pairs, work, capacity):
    """Offer distinct objectives while writers retain control of implementation."""
    if not archive:
        return [{"role": "initial_" + str(i), "parent_id": None, "objective": objective}
                for i, objective in enumerate(("Reduce memory traffic.", "Change work distribution.",
                                  "Improve reduction order.", "Reduce temporary storage.",
                                  "Explore a different algorithm."))][:capacity]
    best = max(archive, key=lambda a: (a.get("estimated_model_saving_ms", a["score"]), a["id"]))
    other = next((a for a in archive[1:] if (a.get("code_design") or {}).get("group") !=
                  (best.get("code_design") or {}).get("group")), None)
    options = [
        {"role": "refine_memory", "parent_id": best["id"], "objective": "Refine the best design's data access."},
        {"role": "refine_work", "parent_id": best["id"], "objective": "Refine the best design's work distribution."},
        {"role": "alternate_design", "parent_id": other["id"] if other else None,
         "objective": "Develop a competitive design with a different code structure."},
        {"role": "new_design", "parent_id": None, "objective": "Explore an algorithm with a different structure."},
        {"role": "combine", "parent_id": pairs[0]["parent_ids"][0] if pairs else best["id"],
         "objective": "Combine measured strengths or address numerical error.",
         **({"merge": {k:pairs[0][k] for k in ("parent_ids", "shape_ids")}} if pairs else {})}]
    active = {a.get("hypothesis", {}).get("experiment_role") for a in work
              if a["status"] not in {"completed", "failed", "budget_deferred"}}
    counts = Counter(a.get("hypothesis", {}).get("experiment_role") for a in work)
    return sorted((o for o in options if o["role"] not in active),
                  key=lambda o: counts[o["role"]])[:capacity]
