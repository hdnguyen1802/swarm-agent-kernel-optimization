"""Keep full evidence on disk and supply bounded summaries to agents."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .core import write_json


def compact_feedback(value, *, max_items=6, max_text=1200, max_depth=8, **_):
    """Limit prompt size without changing the saved measurement evidence."""
    def compact(item, depth):
        if depth > max_depth:
            return {"omitted": "Nested details remain in the evidence file."}
        if isinstance(item, dict):
            result = {}
            for key, child in item.items():
                if key in {"samples_ms", "config", "stdout", "stderr", "traceback"}:
                    if child:
                        result[key + "_omitted"] = True
                    continue
                if len(result) >= 48:
                    result["fields_omitted"] = len(item) - len(result)
                    break
                result[key] = compact(child, depth + 1)
            return result
        if isinstance(item, (list, tuple)):
            # Failures come first, but every entry keeps its own identity fields.
            entries = list(item)
            if entries and all(isinstance(x, dict) for x in entries):
                entries.sort(key=lambda x: x.get("correct", x.get("passed", True)) is not False)
            result = [compact(child, depth + 1) for child in entries[:max_items]]
            if len(entries) > max_items:
                result.append({"items_omitted": len(entries) - max_items})
            return result
        if isinstance(item, str) and len(item) > max_text:
            return item[:max_text] + "\n[More text remains in the evidence file.]"
        return item
    return compact(value, 0)


def artifact_descriptor(directory, path):
    root, target = Path(directory).resolve(), Path(path).resolve()
    relative = target.relative_to(root)
    data = target.read_bytes()
    return {"path": relative.as_posix(), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def save_feedback(directory, value, *, name, metadata=None):
    """Save immutable evidence before producing its compact prompt summary."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError("Use a simple evidence name.")
    encoded = json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")
    identifier = hashlib.sha256(encoded).hexdigest()[:20]
    path = Path(directory) / "evidence" / f"{name}-{identifier}.json"
    if not path.exists():
        write_json(path, value)
    return {**(metadata or {}), "summary": compact_feedback(value),
            "artifact": artifact_descriptor(directory, path)}
