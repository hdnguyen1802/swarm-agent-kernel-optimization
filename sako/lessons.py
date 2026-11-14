"""Scoped conclusions backed by saved evaluator records.

Agent explanations remain hypotheses. Only stored measurements set lesson status.
Raw measurements and old conclusions remain available after a correction.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import statistics
import time
import uuid
from threading import Lock

from .core import digest
from .store import operation_correct


_EVIDENCE_WRITE_LOCK = Lock()


def lesson_context(task, hardware):
    """Use the full hardware fingerprint and the exact operation contract."""
    fingerprint = hardware.get("fingerprint") if isinstance(hardware, dict) else None
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("Lessons require a measured hardware fingerprint.")
    contract = task.get("contract") or {}
    cases = contract.get("cases", task.get("cases", []))
    normalized = [{key: case[key] for key in ("shapes", "dtypes", "strides", "workload") if key in case}
                  for case in cases]
    unique = {json.dumps(case, sort_keys=True): case for case in normalized}
    specification = {key: contract.get(key) for key in
                     ("functional_source", "numerics", "output", "atol", "rtol", "inputs_mutated")}
    specification.update(kind=task["kind"], cases=[unique[key] for key in sorted(unique)],
                         scalars=contract.get("scalar_values", task.get("scalars", [])))
    context = {"task_id": task["id"], "kind": task["kind"], "hardware_fingerprint": fingerprint,
               "contract_signature": digest(specification)}
    return context, digest(context)


def _finite(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def _text(value, limit=600):
    return str(value)[:limit] if value is not None else None


def _write_immutable(path, value):
    # Windows can reject a read while another thread replaces the same file.
    # One controller owns these records. Serialize its publish-and-compare steps.
    with _EVIDENCE_WRITE_LOCK:
        _write_immutable_locked(path, value)


def _write_immutable_locked(path, value):
    """Concurrent writers use separate temporary files for the same record."""
    encoded = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError("The immutable lesson evidence changed.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def hypothesis_summary(hypothesis):
    if isinstance(hypothesis, str):
        hypothesis = {"claim": hypothesis}
    hypothesis = hypothesis if isinstance(hypothesis, dict) else {}
    return {"label": "agent_hypothesis", **{key: _text(hypothesis[key], 600)
            for key in ("id", "specialty", "objective", "claim", "algorithm", "change", "expected_metric",
                        "design_owner", "reason") if key in hypothesis}}


def _validation_scope(result, scope):
    """Keep accuracy and timing contracts distinct in shared conclusions."""
    config = result.get("config") if isinstance(result.get("config"), dict) else {}
    accuracy_scope = result.get("correctness_scope", config.get("correctness_scope", "full_model")) if scope == "model" else "operation"
    accuracy_reference = result.get("correctness_reference", config.get("correctness_reference", "torch_compile")) if scope == "model" else "eager"
    speed_reference = result.get("model_reference", config.get("model_reference", "torch_compile"))
    output_equivalence = result.get("output_equivalence", "checked" if accuracy_scope == "full_model" else "not_tested")
    return {"correctness_scope": _text(accuracy_scope, 120), "correctness_reference": _text(accuracy_reference, 120),
            "speed_scope": "complete_model" if scope == "model" else "operation",
            "speed_reference": _text(speed_reference, 120), "model_reference": _text(speed_reference, 120),
            "output_equivalence": _text(output_equivalence, 120)}


def result_summary(result, scope="operation"):
    """Describe measured outcomes without inferring a cause for a change."""
    if result is None:
        return "untested", "unmeasured", {"correct": None, "accepted": None}
    if not isinstance(result, dict):
        raise ValueError("The stored evaluator result must be an object.")
    infrastructure = result.get("failure_kind") == "infrastructure" or (
        isinstance(result.get("worker_report"), dict)
        and result["worker_report"].get("failure_kind") == "infrastructure")
    decision = result.get("decision") or {}
    correct = operation_correct(result)
    accepted = decision.get("accepted") is True and correct
    observations = {"correct": None if infrastructure else correct, "accepted": accepted,
                    **_validation_scope(result, scope),
                    "speedup": _finite(decision.get("speedup")),
                    "reasons": [_text(x, 240) for x in decision.get("reasons", [])[:8]],
                    "case_count": len(result.get("cases", [])), "cases": []}
    if result.get("error"):
        observations["error"] = _text(result["error"])
    if result.get("acceptance_scope"):
        observations["acceptance_scope"] = _text(result["acceptance_scope"], 600)
    for field in ("timing_baseline", "kernel_min_speedup", "search_score"):
        if field in result:
            observations[field] = result[field]
    for index, case in enumerate(result.get("cases", [])[:6]):
        item = {"index": index, "correct": case.get("correct"),
                **{key: case[key] for key in ("workload", "shapes", "dtypes", "strides", "reference", "timing_reference") if key in case}}
        checks = case.get("checks", [])
        errors = [_finite(check.get("max_abs_error")) for check in checks]
        item.update(checks=len(checks), failed_checks=sum(check.get("passed") is not True for check in checks),
                    max_abs_error=max((value for value in errors if value is not None), default=None),
                    failed_seeds=[check["seed"] for check in checks if check.get("passed") is not True
                                  and "seed" in check][:8])
        for name in ("baseline", "candidate"):
            timing = case.get(name) or {}
            samples = timing.get("samples_ms") or []
            valid = bool(samples) and all(_finite(x) is not None and x > 0 for x in samples)
            item[name] = {"sample_count": len(samples),
                          "median_ms": statistics.median(samples) if valid else _finite(timing.get("median_ms")),
                          "p95_ms": sorted(samples)[math.ceil(.95 * len(samples)) - 1] if valid
                          else _finite(timing.get("p95_ms"))}
        observations["cases"].append(item)
    if infrastructure:
        return "untested", "infrastructure_failure", observations
    if not correct:
        return "failed", "correctness_failure" if result.get("cases") else "execution_failure", observations
    if accepted:
        classification = ("layer_accuracy_and_complete_model_speed_passed" if scope == "model"
                          and observations["correctness_scope"] == "replaced_layers" else
                          "complete_model_gain" if scope == "model" else "operation_gate_passed")
        return "supported", classification, observations
    return "failed", "performance_gate_failed", observations


class LessonStore:
    def __init__(self, store, directory=None):
        self.store = store
        self.directory = Path(directory) if directory else None
        with store.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS swarm_lessons (
                    id TEXT PRIMARY KEY, scope_key TEXT, candidate_id TEXT,
                    scope TEXT, status TEXT, created REAL, payload TEXT);
                CREATE TABLE IF NOT EXISTS swarm_lesson_supersedes (
                    previous_id TEXT, new_id TEXT, PRIMARY KEY(previous_id,new_id));
                CREATE TABLE IF NOT EXISTS swarm_lesson_evidence (
                    sha256 TEXT PRIMARY KEY, payload TEXT);
            ''')

    def _row(self, candidate_id, task):
        row = next((row for row in self.store.candidates(task["id"]) if row["id"] == candidate_id), None)
        if row is None:
            raise ValueError("The lesson candidate does not belong to this task.")
        return row

    def record(self, task, hardware, candidate_id, hypothesis=None,
               scope="operation", result=None, evidence=None, supersedes=(), trial_observations=None):
        if scope not in {"operation", "model"}:
            raise ValueError("Use operation or model lesson scope.")
        context, scope_key = lesson_context(task, hardware)
        row = self._row(candidate_id, task)
        evidence_records = []
        raw_evidence = {}
        if scope == "operation":
            saved = row["result"]
            if result is not None and digest(result) != digest(saved):
                raise ValueError("The lesson differs from the saved evaluator result.")
            result = saved
            evidence_records.append({"kind": "candidate_result", "candidate_id": candidate_id,
                                     "sha256": digest(saved)})
            raw_evidence[digest(saved)] = saved
        else:
            matches = []
            for key, measurement in self.store.get("model_measurements", {}).items():
                entries = measurement.get("manifest", {}).get("entries", [])
                if (any(entry.get("candidate_id") == candidate_id and entry.get("task_id") == task["id"]
                        for entry in entries) and (result is None or digest(result) in
                        {digest(measurement), digest(measurement.get("result"))})):
                    matches.append((measurement.get("generation", -1), key, measurement))
            if not matches:
                raise ValueError("The model lesson has no matching saved measurement.")
            _, key, measurement = sorted(matches, key=lambda item: (item[0], item[1]))[-1]
            result = measurement["result"]
            evidence_records.append({"kind": "model_measurement", "measurement_id": key,
                                     "sha256": digest(measurement),
                                     "candidate_ids": [entry["candidate_id"] for entry in
                                                       measurement["manifest"]["entries"]]})
            raw_evidence[digest(measurement)] = measurement
        if evidence:
            items = evidence if isinstance(evidence, list) else [evidence]
            for item in items[:8]:
                if isinstance(item, dict) and isinstance(item.get("path"), str):
                    evidence_records.append({key: item[key] for key in ("path", "sha256", "bytes") if key in item})
        status, classification, observations = result_summary(result, scope)
        if trial_observations is not None:
            observations["tested_change"] = trial_observations
        note = "This status describes the measured candidate. It does not prove the agent's causal explanation."
        if scope == "model":
            if observations.get("correctness_scope") == "replaced_layers":
                note = ("This status covers eager replacement-layer accuracy and complete compiled model speed. "
                        "Final model output equivalence was not tested. Agent explanations remain hypotheses."
                        if observations.get("output_equivalence") == "not_tested" else
                        "This status covers the declared layer accuracy and complete model speed gates. "
                        "Final output equivalence is a separate observation. Agent explanations remain hypotheses.")
            observations["attribution"] = ("combination_only" if len(evidence_records[0]["candidate_ids"]) > 1
                                           else "single_replacement")
        record = {"schema_version": 1, "scope_key": scope_key, "context": context,
                  "candidate_id": candidate_id, "parent_id": row.get("parent"), "scope": scope,
                  "status": status, "classification": classification, "status_note": note,
                  "hypothesis": hypothesis_summary(hypothesis), "observations": observations,
                  "evidence": evidence_records}
        identifier = digest(record)[:24]
        record["id"] = identifier
        supersedes = list(dict.fromkeys(supersedes))
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for previous in supersedes:
                old = db.execute("SELECT scope_key,scope,status,created,payload FROM swarm_lessons WHERE id=?", (previous,)).fetchone()
                if (old is None or old["scope_key"] != scope_key or old["scope"] != scope
                        or status == "untested" or previous == identifier):
                    raise ValueError("A correction needs prior evidence in the same measured scope.")
                old_observations = json.loads(old["payload"])["observations"]
                old_contract = _validation_scope(old_observations, scope)
                if any(old_contract[key] != observations.get(key) for key in
                       ("correctness_scope", "correctness_reference", "speed_scope", "speed_reference")):
                    raise ValueError("A lesson cannot supersede evidence from a different accuracy or speed contract.")
                # An existing newer record must not be pointed back at an older record.
                current = db.execute("SELECT created FROM swarm_lessons WHERE id=?", (identifier,)).fetchone()
                if current is not None and current["created"] <= old["created"]:
                    raise ValueError("A correction must point to newer evidence.")
            db.execute("INSERT OR IGNORE INTO swarm_lessons VALUES(?,?,?,?,?,?,?)",
                       (identifier, scope_key, candidate_id, scope, status, time.time(), json.dumps(record, allow_nan=False)))
            for sha256, raw in raw_evidence.items():
                db.execute("INSERT OR IGNORE INTO swarm_lesson_evidence VALUES(?,?)",
                           (sha256, json.dumps(raw, allow_nan=False)))
            for previous in supersedes:
                db.execute("INSERT OR IGNORE INTO swarm_lesson_supersedes VALUES(?,?)", (previous, identifier))
        if self.directory:
            path = self.directory / "lessons" / (identifier + ".json")
            _write_immutable(path, record)
            for sha256, raw in raw_evidence.items():
                path = self.directory / "lessons" / "evidence" / (sha256 + ".json")
                _write_immutable(path, raw)
        return record

    def all(self):
        with self.store.connect() as db:
            rows = db.execute("SELECT payload FROM swarm_lessons ORDER BY created,id").fetchall()
            edges = db.execute("SELECT previous_id,new_id FROM swarm_lesson_supersedes ORDER BY new_id").fetchall()
        superseded = {}
        for edge in edges:
            superseded.setdefault(edge["previous_id"], []).append(edge["new_id"])
        return [{**json.loads(row["payload"]), "superseded_by": superseded.get(json.loads(row["payload"])["id"], [])}
                for row in rows]

    def retrieve(self, task, hardware, limit=8, max_chars=12000):
        if type(limit) is not int or limit < 0 or type(max_chars) is not int or max_chars < 2:
            raise ValueError("Use a nonnegative lesson limit and a valid character budget.")
        _, scope_key = lesson_context(task, hardware)
        matching = [row for row in reversed(self.all()) if row["scope_key"] == scope_key and not row["superseded_by"]]
        # Interleave failures and wins so one class cannot erase the other from recall.
        groups = [[row for row in matching if row["status"] == status] for status in ("failed", "supported", "untested")]
        ordered = []
        while any(groups):
            for group in groups:
                if group:
                    ordered.append(group.pop(0))
        selected = []
        for row in ordered:
            if len(selected) >= limit:
                break
            if len(json.dumps([*selected, row], ensure_ascii=False)) <= max_chars:
                selected.append(row)
        return selected
