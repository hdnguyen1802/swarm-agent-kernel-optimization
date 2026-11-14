"""Read the controller's saved evidence without creating or changing a run."""

from __future__ import annotations

from contextlib import closing
from functools import lru_cache
from pathlib import Path
import json
import math
import sqlite3

from ..core import read_json
from .ratios import candidate_shapes
from ..operation_names import read_names


def run_path(root, value):
    """Accept a folder name or a path under runs/."""
    if not str(value).strip():
        raise ValueError("Enter a run folder.")
    base = (root / "runs").resolve()
    path = Path(value)
    if not path.is_absolute():
        path = (root if path.parts and path.parts[0] == "runs" else base) / path
    path = path.resolve()
    if path == base or not path.is_relative_to(base):
        raise ValueError("Choose a folder inside runs/.")
    return path


def saved_config(run):
    for name in ("launch-config.json", "config.json"):
        path = run / name
        if path.is_file():
            return path
    return None


def list_runs(root):
    base = root / "runs"
    if not base.is_dir():
        return []
    folders = [p for p in base.iterdir() if p.is_dir() and
               any((p / name).exists() for name in
                   ("state.sqlite", "result.json", "report.md", "doctor.json", "config.json", "launch-config.json", "ui"))]
    return sorted(folders, key=lambda p: p.stat().st_mtime, reverse=True)


def latest_attempt(run):
    directory = run / "ui"
    current = read_if_present(directory / "current.json").get("attempt", "")
    if current.startswith("attempt-") and current == Path(current).name and (directory / current).is_dir():
        return directory / current
    return max((p for p in directory.glob("attempt-*") if p.is_dir()), default=None)


def read_if_present(path):
    return read_json(path) if path.is_file() else {}


def speed_ratio(value):
    if type(value) in (int, float) and math.isfinite(value) and value > 0:
        return f"{value:.3f}×"
    return "—"


def decision_outcome(decision, entries):
    if decision.get("accepted") is False:
        return "No gain accepted"
    if decision.get("accepted") is True and entries:
        ratio = speed_ratio(decision.get("speedup"))
        return "Accepted" if ratio == "—" else ratio + "  ·  accepted"
    return "Not measured"


@lru_cache(maxsize=8)
def _result_summary(path, stamp, size):
    # Keep the small summary in memory. A complete result can contain MB of samples.
    result = read_json(path)
    decision = (result.get("model_result") or {}).get("decision") or {}
    entries = (result.get("manifest") or {}).get("entries")
    return result.get("status"), decision_outcome(decision, entries)


def read_events(db, offset, tables, *, compact=False):
    """Read new swarm records in commit order. Leave raw trials in the database."""
    row = db.execute("SELECT json_array_length(value) FROM state WHERE key='swarm_events'").fetchone()
    count = (row[0] or 0) if row else 0
    start = offset if offset <= count else 0
    if start == count:
        return start, count, []
    records = db.execute("""
        SELECT json_remove(e.value, '$.alternatives', '$.subsection_decisions', '$.manifest',
                           '$.adopted_manifest', '$.shape_decisions', '$.evidence', '$.outcome'),
               json_array_length(e.value, '$.manifest.entries')
        FROM state AS s, json_each(s.value) AS e
        WHERE s.key='swarm_events' AND e.type='object' AND e.key >= ? ORDER BY e.key
    """, (start,))
    events = [dict(json.loads(value), manifest_entries=entries) for value, entries in records]
    if any(event.get("kind") == "coordinator_decision" for event in events):
        records = db.execute("""
            SELECT json_extract(p.value,'$.id'), json_extract(p.value,'$.hypotheses')
            FROM state AS s, json_each(s.value) AS p WHERE s.key='swarm_plans'
            AND p.type='object' AND json_type(p.value,'$.hypotheses')='array'
        """)
        plans = {identity: json.loads(proposals) for identity, proposals in records}
        for event in events:
            if event.get("kind") == "coordinator_decision":
                event["strategies"] = plans.get(event.get("plan_id"), [])
    if "candidates" in tables:
        for event in events:
            if event.get("kind") == "candidate_evaluated":
                row = db.execute(
                    "SELECT json_extract(result,'$.timing_baseline'), substr(json_extract(result,'$.error'),1,240), "
                    "(SELECT COUNT(*) FROM json_each(result,'$.shape_decisions') WHERE json_extract(value,'$.accepted')=1), "
                    "(SELECT COUNT(*) FROM json_each(result,'$.shape_decisions')) "
                    "FROM candidates WHERE id=?", (event.get("candidate_id"),)).fetchone()
                if row:
                    event.update(baseline=json.loads(row[0]) if row[0] else None, error=row[1],
                                 shapes_passed=row[2], shapes_tested=row[3])
                    if not compact or row[3] <= 2:
                        event["shape_results"] = candidate_shapes(db, event.get("candidate_id"))
    return start, count, events


def run_snapshot(run, attempt=None, *, event_offset=None, compact=False):
    snapshot = dict(status=None, task=None, tokens=None, writers=None, planning=False, tested=None,
                    passed=None, partial=None, pending=None, kernel="—", model="Not measured",
                    gpu=None, memory=None, candidates=[], candidate_details={}, stage={}, colab={}, warning=None,
                    task_position=None, events=None, shortlist={}, candidate_gpus={}, decode_metrics=[], final_progress=None,
                    operation_names={}, selected_candidates=[])
    try:
        attempt = attempt or latest_attempt(run)
        profile = run / "profile"
        if attempt:
            snapshot["stage"] = read_if_present(attempt / "progress.json")
            snapshot["colab"] = read_if_present(attempt / "colab.json")
            configured = read_if_present(attempt / "plan.json").get("profile")
            if configured and not (profile / "tasks.json").exists():
                profile = run.parent.parent / configured
        catalog_path = profile / "tasks.json"
        if catalog_path.is_file():
            stat = catalog_path.stat()
            snapshot["operation_names"] = read_names(catalog_path, stat.st_mtime_ns, stat.st_size)
        hardware = read_if_present(run / "hardware.json")
        if not hardware:
            hardware = read_if_present(run / "doctor.json").get("hardware") or {}
        facts = hardware.get("facts") or {}
        colab = snapshot["colab"]
        snapshot.update(gpu=facts.get("name") or colab.get("gpu"),
                        memory=facts.get("total_memory_bytes") or (colab.get("memory_mib") or 0) * 2**20)
        path = run / "result.json"
        if path.is_file():
            stat = path.stat()
            snapshot["status"], snapshot["model"] = _result_summary(path, stat.st_mtime_ns, stat.st_size)
        db_path = run / "state.sqlite"
        if not db_path.is_file():
            return snapshot
        with closing(sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True, timeout=0.25)) as db:
            db.execute("BEGIN")
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "state" in tables:
                state = {k: json.loads(v) for k, v in db.execute(
                    "SELECT key, CASE WHEN key='task_progress' THEN json_extract(value,'$.queue') ELSE value END "
                    "FROM state WHERE key IN ('status','active_task','task_progress')") if v is not None}
                snapshot.update(status=state.get("status") or snapshot["status"], task=state.get("active_task"))
                queue = state.get("task_progress") or []
                if snapshot["task"] in queue:
                    snapshot["task_position"] = f"{queue.index(snapshot['task']) + 1}/{len(queue)}"
                if event_offset is not None:
                    snapshot["events"] = read_events(db, event_offset, tables, compact=compact)
                extra = {k: json.loads(v) for k,v in db.execute(
                    "SELECT key,value FROM state WHERE key IN ('decode_local_ranks','decode_metrics','decode_final_progress')")}
                snapshot["shortlist"] = {identity: rank for ids in extra.get("decode_local_ranks", {}).values()
                                         for rank, identity in enumerate(ids, 1)}
                snapshot["decode_metrics"] = extra.get("decode_metrics", [])
                snapshot["final_progress"] = extra.get("decode_final_progress")
                # Read only the evaluator decision. Some measurements hold many MB of raw trials.
                row = db.execute(
                    "SELECT json_extract(value,'$.result.decision'), json_array_length(value,'$.manifest.entries') "
                    "FROM state WHERE key IN ('best_model_measurement','last_model_measurement') "
                    "AND json_type(value,'$.result.decision')='object' "
                    "ORDER BY key LIMIT 1").fetchone()
                if row:
                    snapshot["model"] = decision_outcome(json.loads(row[0]), row[1])
            if "calls" in tables:
                snapshot["tokens"], snapshot["writers"], snapshot["planning"] = db.execute(
                    "SELECT COALESCE(SUM(COALESCE(actual,reserved)),0), "
                    "COUNT(CASE WHEN status='reserved' AND role GLOB 'writer_*' "
                    "AND instr(role, ':')=0 THEN 1 END), "
                    "COUNT(CASE WHEN status='reserved' AND role='coordinator' THEN 1 END) FROM calls").fetchone()
            if "candidates" in tables:
                selected = []
                model_states = {}
                if "state" in tables:
                    row = db.execute("SELECT json_extract(value,'$.manifest.entries') FROM state "
                                     "WHERE key='best_model_measurement'").fetchone()
                    selected = json.loads(row[0]) if row and row[0] else []
                    for key, label in (("rejected_model_candidates", "model rejected"),
                                       ("blocked_model_candidates", "model incorrect")):
                        for identity, in db.execute("SELECT e.key FROM state AS s,json_each(s.value) AS e WHERE s.key=?", (key,)):
                            model_states[identity] = label
                selected_ids = sorted({entry["candidate_id"] for entry in selected if entry.get("candidate_id")})
                snapshot["selected_candidates"] = selected_ids
                tested, passed, pending, best = db.execute(
                    "SELECT COUNT(result), COUNT(CASE WHEN result IS NOT NULL AND accepted=1 THEN 1 END), "
                    "COUNT(*)-COUNT(result), MAX(CASE WHEN accepted=1 AND "
                    "json_type(result,'$.decision.speedup') IN ('real','integer') "
                    "THEN json_extract(result,'$.decision.speedup') END) FROM candidates").fetchone()
                partial, shape_best = db.execute(
                    "SELECT COUNT(*), MAX((SELECT MAX(json_extract(value,'$.speedup')) "
                    "FROM json_each(result,'$.shape_decisions') WHERE json_extract(value,'$.accepted')=1)) "
                    "FROM candidates WHERE result IS NOT NULL AND accepted!=1 "
                    "AND EXISTS (SELECT 1 FROM json_each(result,'$.shape_decisions') "
                    "WHERE json_extract(value,'$.accepted')=1)").fetchone()
                best = max((x for x in (best, shape_best) if x is not None), default=None)
                snapshot.update(tested=tested, passed=passed, partial=partial, pending=pending, kernel=speed_ratio(best))
                rows = db.execute(
                    "SELECT id, task, island, CASE WHEN result IS NULL THEN 'pending' "
                    "WHEN json_extract(result,'$.stages.outcome')='screened_slow' THEN 'slow' "
                    "WHEN json_extract(result,'$.stages.outcome') LIKE '%accuracy_failed' THEN 'incorrect' "
                    "WHEN accepted=1 THEN 'passed' "
                    "WHEN EXISTS (SELECT 1 FROM json_each(result,'$.shape_decisions') WHERE json_extract(value,'$.accepted')=1) "
                    "THEN 'partial ' || (SELECT COUNT(*) FROM json_each(result,'$.shape_decisions') WHERE json_extract(value,'$.accepted')=1) "
                    "|| '/' || (SELECT COUNT(*) FROM json_each(result,'$.shape_decisions')) "
                    "ELSE 'failed' END, CASE WHEN ? AND json_extract(result,'$.timing_baseline.kind')='per_shape' "
                    "AND json_extract(result,'$.performance_scope') IS NOT 'cached_decode' THEN NULL ELSE "
                    "COALESCE(json_extract(result,'$.search_score'),json_extract(result,'$.decision.speedup'),json_extract(result,'$.screen_speedup')) END, "
                    "substr(COALESCE(json_extract(result,'$.error'),json_extract(result,'$.decision.reasons[0]')),1,240) "
                    ", json_extract(result,'$.evaluation_device.gpu') FROM candidates "
                    "ORDER BY (id IN (SELECT value FROM json_each(?))) DESC, rowid DESC LIMIT ?",
                    (compact, json.dumps(selected_ids), max(30, len(selected_ids)))).fetchall()
                snapshot["candidates"] = [(identity, task, writer, outcome, speed_ratio(ratio), reason)
                                          for identity, task, writer, outcome, ratio, reason, _ in rows]
                snapshot["candidate_gpus"] = {r[0]: r[-1] for r in rows if r[-1] is not None}
                snapshot["candidate_details"] = {
                    row[0]: candidate_shapes(db, row[0], selected, model_states.get(row[0], "not used"))
                    for row in snapshot["candidates"]} if not compact else {}
    except (OSError, sqlite3.Error, ValueError, TypeError, AttributeError) as error:
        snapshot["warning"] = f"Could not read part of this run. {error}"
    return snapshot


def text_preview(path, *, tail=False, limit=12000):
    if path is None or not path.is_file():
        return ""
    with path.open("rb") as stream:
        if tail:
            stream.seek(max(0, path.stat().st_size - limit))
        data = stream.read(limit)
    return data.decode("utf-8-sig", errors="replace").strip()
