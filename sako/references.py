"""Build a small, pinned source index and retrieve useful excerpts."""

import re
import subprocess
from pathlib import Path

from .core import read_json, write_json, digest

ROOT = Path(__file__).resolve().parents[1]


def build_index(root=ROOT):
    root = Path(root)
    source = root / "third_party/cutlass"
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    catalog = read_json(root / "references/catalog.json")
    if revision != catalog["commit"]:
        raise ValueError("CUTLASS does not match the pinned commit.")
    chunks = []
    for entry in catalog["sources"]:
        path = source / entry["path"]
        lines = path.read_text(encoding="utf-8").splitlines()
        for start in range(0, len(lines), 100):
            body = "\n".join(lines[max(0, start - 15):start + 100])
            chunks.append({**entry, "start_line": max(0, start - 15) + 1, "text": body,
                           "commit": revision, "sha256": digest(body)})
    write_json(root / "references/index.json", {"commit": revision, "chunks": chunks})
    return {"commit": revision, "chunks": len(chunks)}


def retrieve(query, kind, limit=3, root=ROOT):
    path = Path(root) / "references/index.json"
    if not path.exists():
        build_index(root)
    index = read_json(path)
    words = set(re.findall(r"[a-z0-9_]+", (query + " " + kind).lower()))
    scored = []
    for chunk in index["chunks"]:
        if 70 not in chunk["architectures"]:
            continue
        tokens = set(re.findall(r"[a-z0-9_]+", chunk["text"].lower()))
        score = len(tokens & words) + 8 * (kind in chunk["kinds"])
        scored.append((score, chunk))
    return [c for _, c in sorted(scored, key=lambda item: item[0], reverse=True)[:limit]]
