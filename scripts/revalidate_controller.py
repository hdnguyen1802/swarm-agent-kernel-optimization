"""Audit a controller-only update at a stopped search boundary.

This script does not run a model or make an API request. It keeps the existing
token ledger and numerical evidence. Each search mode has a strict file allowlist.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sqlite3
import re
import tarfile
import time

from sako.core import digest, load_config, read_json, write_json
from sako.hardware import hardware_fingerprint
from sako.preparation import profile_settings, validate_catalog


SWARM_ALLOWED_CHANGES = {"swarm.py", "swarm_policy.py"}
LEDGER_TABLES = ("calls", "candidates", "candidate_islands", "emissions", "candidate_parents")
EVIDENCE_DIRS = ("calls", "candidates", "evidence", "model_checks", "selections", "migrations", "decisions", "jobs",
                 "profiles", "diagnostics", "lessons", "plans", "response_normalizations", "transport_recovery")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _normal_text(data):
    return data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")


def _source_name(name):
    parts = PurePosixPath(name).parts
    return (len(parts) == 2 and parts[0] == "sako" and parts[1].endswith(".py")
            or len(parts) == 3 and parts[:2] == ("sako", "prompts") and parts[2].endswith(".txt"))


def _archive_sources(raw):
    sources, configs = {}, {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        for member in archive.getmembers():
            name = member.name.removeprefix("./")
            if not _source_name(name) and name != "configs/project.json":
                continue
            source_key = str(PurePosixPath(name).relative_to("sako")) if _source_name(name) else None
            if not member.isfile() or source_key in sources or name in configs:
                raise ValueError("The source archive contains a duplicate or non-file entry.")
            value = archive.extractfile(member).read()
            if name == "configs/project.json":
                configs[name] = json.loads(value)
            else:
                sources[source_key] = _normal_text(value)
    if not sources or "configs/project.json" not in configs:
        raise ValueError("The archive needs sako source and the original project config.")
    return sources, configs["configs/project.json"]


def _current_sources(root):
    package = root / "sako"
    return {path.relative_to(package).as_posix(): path.read_text(encoding="utf-8")
            for path in sorted([*package.glob("*.py"), *package.glob("prompts/*.txt")])}


def _ledger_snapshot(db):
    tables = {}
    definitions = {row["name"]: row["sql"] for row in db.execute(
        "SELECT name,sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' AND name != 'state'")}
    if not set(LEDGER_TABLES) <= set(definitions):
        raise ValueError("The run is missing required ledger tables.")
    for table, definition in sorted(definitions.items()):
        rows = [dict(row) for row in db.execute('SELECT * FROM "' + table.replace('"', '""') + '"')]
        rows.sort(key=lambda row: json.dumps(row, sort_keys=True))
        tables[table] = {"count": len(rows), "hash": digest(rows), "schema_hash": digest(definition)}
    usage = dict(db.execute("SELECT COUNT(*) calls, COALESCE(SUM(COALESCE(actual,reserved)),0) tokens, "
                            "COALESCE(MAX(id),0) last_call_id FROM calls").fetchone())
    return {"tables": tables, "usage": usage}


def _config_provenance(original, saved, catalog):
    """Permit only the recorded main-to-commit resolution for a swarm archive."""
    if original == saved:
        return {"mode": "exact", "archive_config_hash": digest(original)}
    pinned = saved.get("revision")
    resolved = catalog.get("config", {}).get("revision")
    without_revision = lambda config: {key: value for key, value in config.items() if key != "revision"}
    if (original.get("revision") != "main" or not isinstance(pinned, str)
            or re.fullmatch(r"[0-9a-f]{40}", pinned) is None or resolved != pinned
            or without_revision(original) != without_revision(saved)):
        raise ValueError("The run config differs from the original archived config.")
    return {"mode": "profile_pinned_main", "archive_revision": "main", "run_revision": pinned,
            "catalog_revision": resolved, "archive_config_hash": digest(original),
            "note": "Only revision differs. The original catalog, preparation stamp, and run fingerprint bind the commit."}


def _swarm_boundary(state, generation, db):
    assignments = state.get("swarm_assignments", {})
    next_assignment = state.get("swarm_next_assignment", 0)
    counts = state.get("swarm_task_counts", {})
    if (not isinstance(assignments, dict) or type(next_assignment) is not int or next_assignment < 0
            or set(assignments) != {str(number) for number in range(next_assignment)}
            or generation > next_assignment):
        raise ValueError("The swarm assignment sequence is invalid.")
    progress = state.get("task_progress", {})
    queue = progress.get("queue", [])
    finished = Counter()
    identities = set()
    for number, assignment in assignments.items():
        if (not isinstance(assignment, dict) or assignment.get("generation") != int(number)
                or assignment.get("task_id") not in queue or not isinstance(assignment.get("id"), str)
                or assignment["id"] in identities):
            raise ValueError("The swarm assignment identity or task is invalid.")
        identities.add(assignment["id"])
        if assignment.get("status") not in {"completed", "failed"}:
            raise ValueError("The swarm still has unfinished assignments.")
        if assignment["status"] == "completed":
            candidate = db.execute("SELECT task,source,result FROM candidates WHERE id=?",
                                   (assignment.get("candidate_id"),)).fetchone()
            if (state.get("reviewed:" + number) is not True or candidate is None or candidate["result"] is None
                    or candidate["task"] != assignment["task_id"]
                    or json.loads(candidate["source"]) != assignment.get("candidate")):
                raise ValueError("Completed swarm assignments need saved evaluation and review evidence.")
            finished[assignment["task_id"]] += 1
        elif assignment.get("candidate") or not assignment.get("error"):
            raise ValueError("A failed swarm assignment needs its failure record and no unevaluated candidate.")
    if (not isinstance(counts, dict) or any(type(value) is not int or value < 0 for value in counts.values())
            or {key: value for key, value in counts.items() if value} != dict(finished)):
        raise ValueError("The swarm task counts differ from completed assignments.")
    completed = progress.get("completed", [])
    index = progress.get("index")
    if (type(index) is not int or not 0 <= index <= len(queue)
            or [item.get("task_id") for item in completed] != queue[:index]
            or any(item.get("rounds") != finished[item["task_id"]] for item in completed)
            or progress.get("round", 0) != (finished[queue[index]] if index < len(queue) else 0)
            or any(task not in queue[:min(index + 1, len(queue))] for task in finished)):
        raise ValueError("The swarm progress differs from completed assignments.")
    if finished and generation != next_assignment:
        raise ValueError("The swarm generation does not cover the saved assignment sequence.")
    return {"assignments": len(assignments), "completed": sum(finished.values()),
            "failed": len(assignments) - sum(finished.values()), "task_counts": dict(finished)}


def _evidence_snapshot(run):
    files = {}
    for folder in EVIDENCE_DIRS:
        for path in sorted((run / folder).rglob("*")):
            if path.is_file():
                files[path.relative_to(run).as_posix()] = _sha(path.read_bytes())
    return {"count": len(files), "hash": digest(files)}


def _exclusive_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def revalidate_controller(run, archive, doctor, *, project_root=None):
    run, archive, doctor = Path(run).resolve(), Path(archive).resolve(), Path(doctor).resolve()
    root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
    if not (run / "STOP").is_file():
        raise ValueError("Keep the STOP file in place during controller revalidation.")
    config_raw = (run / "config.json").read_bytes()
    config = load_config(run / "config.json")
    catalog_path, stamp_path = run / "profile/tasks.json", run / "profile/preparation.json"
    catalog_raw, stamp_raw = catalog_path.read_bytes(), stamp_path.read_bytes()
    catalog, stamp = json.loads(catalog_raw), json.loads(stamp_raw)
    archive_raw = archive.read_bytes()
    previous_sources, original_config = _archive_sources(archive_raw)
    config_provenance = _config_provenance(original_config, json.loads(config_raw), catalog)
    current_sources = _current_sources(root)
    if set(previous_sources) != set(current_sources):
        raise ValueError("The package file set changed.")
    changed = sorted(name for name in current_sources if current_sources[name] != previous_sources[name])
    allowed = SWARM_ALLOWED_CHANGES
    if not changed or not set(changed) <= allowed:
        raise ValueError("Only swarm.py and swarm_policy.py can change in this audit.")
    old_hash, new_hash = digest(previous_sources), digest(current_sources)
    settings_hash, catalog_hash = digest(profile_settings(config)), digest(catalog)
    if (stamp.get("implementation_hash") != old_hash or stamp.get("settings_hash") != settings_hash
            or stamp.get("catalog_hash") != catalog_hash):
        raise ValueError("The original preparation source, settings, or catalog no longer matches.")
    fresh_doctor = read_json(doctor)
    if "ok" in fresh_doctor:
        if fresh_doctor.get("ok") is not True or not isinstance(fresh_doctor.get("result"), dict):
            raise ValueError("The doctor worker job failed.")
        fresh_doctor = {**fresh_doctor["result"], "worker_provenance": fresh_doctor.get("provenance", {})}
    hardware = fresh_doctor.get("hardware", {})
    expected_hardware = catalog.get("hardware", {}).get("fingerprint")
    if (not expected_hardware or hardware.get("fingerprint") != expected_hardware
            or hardware_fingerprint(hardware) != expected_hardware
            or stamp.get("hardware_fingerprint") != expected_hardware):
        raise ValueError("The doctor and profile hardware fingerprints differ.")
    if (fresh_doctor.get("worker_provenance", {}).get("implementation_hash") != new_hash
            or fresh_doctor.get("compile_probe") is not True
            or hardware.get("support", {}).get("supported") is not True):
        raise ValueError("Use a fresh successful doctor result from the updated worker.")
    validate_catalog(config, catalog, hardware)
    before_evidence = _evidence_snapshot(run)
    db = sqlite3.connect(run / "state.sqlite", timeout=30)
    db.row_factory = sqlite3.Row
    stamp_changed, committed = False, False
    try:
        db.execute("BEGIN IMMEDIATE")
        original_state = {row["key"]: row["value"] for row in db.execute("SELECT key,value FROM state")}
        state = {key: json.loads(value) for key, value in original_state.items()}
        if state.get("status") != "stopped":
            raise ValueError("The run must stop before controller revalidation.")
        generation = state.get("generation")
        if type(generation) is not int or generation < 0:
            raise ValueError("The saved generation is invalid.")
        if db.execute("SELECT COUNT(*) FROM calls WHERE status='reserved'").fetchone()[0]:
            raise ValueError("API requests are still in flight.")
        progress = state.get("task_progress", {})
        boundary = _swarm_boundary(state, generation, db)
        old_fingerprint = digest({"config": config, "catalog": catalog, "implementation": old_hash})
        if state.get("fingerprint") != old_fingerprint:
            raise ValueError("The original config, catalog, or implementation fingerprint changed.")
        before_ledger = _ledger_snapshot(db)
        result = read_json(run / "result.json")
        expected_usage = {key: before_ledger["usage"][key] for key in ("calls", "tokens")}
        if (result.get("status") != "stopped" or result.get("usage") != expected_usage
                or result.get("task_progress") != progress):
            raise ValueError("The stopped result does not match the saved ledger and progress.")
        history = state.get("controller_updates", [])
        epoch = len(history) + 1
        audit_path = run / "controller_updates" / f"epoch-{epoch:04d}.json"
        backup = stamp_path.with_name(f"preparation.controller-epoch-{epoch:04d}.before.json")
        new_fingerprint = digest({"config": config, "catalog": catalog, "implementation": new_hash})
        audit = {"kind": "controller_update_authorized", "epoch": epoch, "utc": time.time(),
                 "changed_files": ["sako/" + name for name in changed],
                 "old_implementation_hash": old_hash, "new_implementation_hash": new_hash,
                 "old_fingerprint": old_fingerprint, "new_fingerprint": new_fingerprint,
                 "archive": str(archive), "archive_sha256": _sha(archive_raw),
                 "search_mode": "swarm", "boundary": boundary,
                 "config_provenance": config_provenance, "state_before_hash": digest(original_state),
                 "config_hash": digest(config), "config_sha256": _sha(config_raw),
                 "catalog_hash": catalog_hash, "catalog_sha256": _sha(catalog_raw),
                 "settings_hash": settings_hash, "preparation_sha256": _sha(stamp_raw),
                 "doctor": str(doctor), "doctor_sha256": _sha(doctor.read_bytes()),
                 "hardware_fingerprint": expected_hardware, "generation": generation,
                 "max_total_tokens": config["agent"]["max_total_tokens"],
                 "last_call_id": before_ledger["usage"]["last_call_id"],
                 "ledger_before": before_ledger, "evidence_before": before_evidence,
                 "preparation_backup": str(backup)}
        # Recheck all source and input snapshots under the database lock.
        if (_current_sources(root) != current_sources or (run / "config.json").read_bytes() != config_raw
                or catalog_path.read_bytes() != catalog_raw or stamp_path.read_bytes() != stamp_raw
                or _evidence_snapshot(run) != before_evidence):
            raise ValueError("Source or evidence changed during revalidation.")
        with backup.open("xb") as handle:
            handle.write(stamp_raw)
        _exclusive_json(audit_path, audit)
        audit_ref = {"epoch": epoch, "path": audit_path.relative_to(run).as_posix(),
                     "old_implementation_hash": old_hash, "new_implementation_hash": new_hash}
        updated_stamp = {**stamp, "implementation_hash": new_hash,
                         "revalidated_from": old_hash,
                         "controller_update_history": [*stamp.get("controller_update_history", []), audit_ref]}
        write_json(stamp_path, updated_stamp)
        stamp_changed = True
        updates = {"fingerprint": new_fingerprint, "controller_updates": [*history, audit_ref]}
        for key, value in updates.items():
            db.execute("INSERT OR REPLACE INTO state VALUES(?,?)", (key, json.dumps(value)))
        expected_state = {**original_state, **{key: json.dumps(value) for key, value in updates.items()}}
        after_state = {row["key"]: row["value"] for row in db.execute("SELECT key,value FROM state")}
        if (_ledger_snapshot(db) != before_ledger or _evidence_snapshot(run) != before_evidence
                or after_state != expected_state):
            raise ValueError("The ledger or evidence changed during the update.")
        db.commit()
        committed = True
        applied = {"kind": "controller_update_applied", "epoch": epoch, "utc": time.time(),
                   "audit_sha256": _sha(audit_path.read_bytes()), "ledger_after": before_ledger,
                   "state_after_hash": digest(after_state), "changed_state_keys": sorted(updates),
                   "evidence_after": before_evidence, "preparation_sha256": _sha(stamp_path.read_bytes())}
        _exclusive_json(audit_path.with_name(f"epoch-{epoch:04d}.applied.json"), applied)
        return {**audit, "applied": True, "audit_path": str(audit_path)}
    except Exception:
        if not committed:
            db.rollback()
            if stamp_changed:
                stamp_path.write_bytes(stamp_raw)
        raise
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--doctor", required=True)
    args = parser.parse_args()
    result = revalidate_controller(args.run, args.archive, args.doctor)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
