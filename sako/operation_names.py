"""Use model paths to name optimization targets in human-readable output."""

from functools import lru_cache
import re

from .core import read_json


PROJECTIONS = {"q_proj": "query", "k_proj": "key", "v_proj": "value", "o_proj": "output",
               "gate_proj": "gate", "up_proj": "up", "down_proj": "down"}


def operation_name(task, names):
    if not task:
        return "—"
    return names.get(task) or re.sub(r"_[0-9a-f]{8,}$", "", task).replace("_", " ")


def catalog_names(catalog):
    names = {}
    for task in [*catalog.get("hierarchy", {}).get("regions", []), *catalog.get("tasks", [])]:
        paths = task.get("paths") or []
        leaves = list(dict.fromkeys(path.split(".")[-1] for path in paths))
        if leaves and all(leaf in PROJECTIONS for leaf in leaves):
            parts = [name for leaf, name in PROJECTIONS.items() if leaf in leaves]
            name = " + ".join(parts).capitalize() + (" projections" if len(parts) > 1 else " projection")
        else:
            parts = [{"lm_head": "Output head", "__core__": "Attention core"}.get(leaf, leaf.replace("_", " "))
                     for leaf in leaves]
            name = " + ".join(parts).capitalize() if parts else operation_name(task["id"], {}).capitalize()
        names[task["id"]] = name
    return names


@lru_cache(maxsize=4)
def read_names(catalog_path, mtime_ns, size):
    return catalog_names(read_json(catalog_path))
