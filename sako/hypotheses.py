"""Keep a finite correction budget for each logical swarm hypothesis.

This module never advances a task or changes a benchmark limit. Reassignment and
resume keep the same hypothesis identity and the same initial-plus-two budget.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
import json
import math
import time

from .core import digest


MAX_CORRECTIONS = 2
MAX_ATTEMPTS = 1 + MAX_CORRECTIONS
STATE_KEY = "swarm_hypotheses"
TERMINAL = {"succeeded", "exhausted", "closed"}


def _positive_samples(timing):
    samples = timing.get("samples_ms") if isinstance(timing, dict) else None
    return (isinstance(samples, list) and bool(samples)
            and all(type(value) in (int, float) and math.isfinite(value) and value > 0 for value in samples))


def _excluded_measurement(value):
    """Skipped or unsafe evaluation state cannot establish search stagnation."""
    from .path_selection import unsafe_evidence
    return unsafe_evidence(value)


def _numerical_failure(check):
    return (isinstance(check, dict) and not check.get("error") and not _excluded_measurement(check)
            and check.get("acceptance_role") != "diagnostic"
            and check.get("passed") is False
            and type(check.get("max_abs_error")) in (int, float)
            and math.isfinite(check["max_abs_error"]) and check["max_abs_error"] >= 0)


def _measured_check(check):
    if (not isinstance(check, dict) or check.get("error") or _excluded_measurement(check)
            or check.get("acceptance_role") == "diagnostic"):
        return False
    if check.get("missing_paths"):
        return False
    expected, observed = check.get("expected_paths"), check.get("observed_paths")
    if isinstance(expected, list) and isinstance(observed, list) and any(p not in observed for p in expected):
        return False
    if isinstance(check.get("region_checks"), dict) and _measured_check(check["region_checks"]):
        return True
    if check.get("scope") == "cached_generation":
        if type(check.get("steps_checked")) is not int or check["steps_checked"] <= 0:
            return False
        if _numerical_failure(check.get("logits")):
            return True
        if any(_numerical_failure(layer.get(side)) for layer in check.get("cache_layers", [])
               for side in ("key", "value")):
            return True
        return (check.get("greedy_tokens_match") is False and
                bool(check.get("baseline_ids")) and bool(check.get("candidate_ids")))
    if check.get("scope") != "replaced_layers" and "layers" not in check:
        return _numerical_failure(check)
    # Aggregate passed=False can mean missing coverage, a mutated input, or a
    # compiler error. Its maximum error can still come from a passing layer.
    layers = check.get("layers")
    if not isinstance(layers, list) or check.get("missing_paths"):
        return False
    expected, observed = check.get("expected_paths"), check.get("observed_paths")
    if (isinstance(expected, list) and isinstance(observed, list)
            and any(path not in observed for path in expected)):
        return False
    for layer in layers:
        if (not isinstance(layer, dict) or layer.get("error") or layer.get("passed") is not False
                or layer.get("acceptance_role") == "diagnostic"
                or layer.get("inputs_unchanged") is not True or layer.get("same_input") is False):
            continue
        path = layer.get("path")
        if not isinstance(path, str) or not path:
            continue
        if any(key in check and (not isinstance(check[key], list) or path not in check[key])
               for key in ("expected_paths", "observed_paths")):
            continue
        if _numerical_failure(layer.get("check")):
            return True
    return False


def measured_gpu_outcome(result):
    """Require saved numerical checks or actual timing samples, not an LLM score."""
    if not isinstance(result, dict) or result.get("error") or _excluded_measurement(result):
        return False
    if result.get("stages", {}).get("outcome") == "screened_slow":
        from .staged_evaluation import clearly_slower
        rows = result.get("cases", [])
        if result.get("performance_scope") == "cached_decode":
            indices = [i for i,c in enumerate(rows) if c.get("generation_step", {}).get("phase") == "decode"]
            if indices != result.get("timed_case_indices"):
                return False
            rows = [rows[i] for i in indices]
        return bool(rows) and all(c.get("speed_screen", {}).get("clearly_slower") and
            clearly_slower(c["speed_screen"], 1.) for c in rows)
    cases = result.get("cases")
    if not isinstance(cases, list) or not cases:
        return False
    for case in cases:
        if not isinstance(case, dict):
            continue
        checks = case.get("checks")
        checked = isinstance(checks, list) and any(_measured_check(check) for check in checks)
        timed = (case.get("correct") is True and _positive_samples(case.get("baseline"))
                 and _positive_samples(case.get("candidate")))
        if checked or timed:
            return True
    return False


def classify_fault(stage="operation", result=None, error=None):
    """Classify a trusted pipeline outcome without applying a task-wide stop.

    Numerical and timing outcomes can affect stagnation. API, source, compilation,
    runtime, transport, and budget faults cannot. Transfer recovery keeps the same
    attempt because the original GPU job may still exist.
    """
    if not isinstance(stage, str):
        raise ValueError("The failure stage must be text.")
    stage = stage.casefold()
    if isinstance(result, dict) and isinstance(result.get("result"), dict) and "cases" not in result:
        result = result["result"]
    worker = (result.get("worker_report") or {}) if isinstance(result, dict) else {}
    if not isinstance(worker, dict):
        worker = {}
    text = " ".join(str(value) for value in (
        error or "", result.get("error", "") if isinstance(result, dict) else "", worker.get("error", ""))).casefold()
    name = type(error).__name__ if error is not None else ""
    if stage in {"budget", "stopped"} or name in {"BudgetExhausted", "TaskBudgetExhausted"}:
        kind, retry_same = "budget_or_stop", True
    elif stage in {"transfer", "transport"} or name == "TransferError":
        kind, retry_same = "transfer_error", True
    elif (stage in {"infrastructure", "profile"} or worker.get("failure_kind") == "infrastructure"
          or isinstance(result, dict) and result.get("failure_kind") == "infrastructure"):
        kind, retry_same = "infrastructure_error", True
    elif (stage in {"compile", "build"} or bool(worker.get("build_log"))
          or any(marker in text for marker in ("cuda build failed", "error building extension",
                                               "compilation failed", "nvcc fatal", "ptxas fatal"))):
        kind, retry_same = "compile_error", False
    elif stage in {"source", "schema", "validation", "context", "references"} or (
            stage in {"writer", "api", "generation"} and name in {"ValueError", "TypeError", "JSONDecodeError"}):
        kind, retry_same = "source_error", False
    elif stage in {"writer", "api", "generation"}:
        kind, retry_same = "api_error", False
    elif error is not None or isinstance(result, dict) and result.get("error"):
        kind, retry_same = "execution_error", False
    elif isinstance(result, dict) and result.get("stages", {}).get("outcome") == "screened_slow" and measured_gpu_outcome(result):
        return {"kind": "speed_screen_rejection", "stage": stage, "correction_needed": False,
                "resume_same_attempt": False, "measured_gpu_outcome": True, "counts_for_stagnation": True,
                "accepted": False, "correct": None}
    else:
        measured = measured_gpu_outcome(result)
        cases = result.get("cases", []) if isinstance(result, dict) else []
        checked_failure = isinstance(cases, list) and any(
            isinstance(case, dict) and case.get("correct") is False for case in cases)
        if measured or checked_failure:
            from .store import operation_correct
            correct = operation_correct(result)
            accepted = correct and (result.get("decision") or {}).get("accepted") is True
            return {"kind": "accepted" if accepted else "performance_rejection" if correct else "correctness_failure",
                    "stage": stage, "correction_needed": not accepted, "resume_same_attempt": False,
                    "measured_gpu_outcome": measured, "counts_for_stagnation": measured,
                    "accepted": accepted, "correct": correct}
        kind, retry_same = "unmeasured_outcome", False
    return {"kind": kind, "stage": stage, "correction_needed": not retry_same,
            "resume_same_attempt": retry_same, "measured_gpu_outcome": False,
            "counts_for_stagnation": False, "accepted": False, "correct": None}


def _identifier(value, label):
    if not isinstance(value, str) or not value.strip() or len(value) > 160:
        raise ValueError(f"Use a nonempty {label} of at most 160 characters.")
    return value


def _copy(value):
    # Reject non-JSON and nonfinite values before storing durable state.
    return json.loads(json.dumps(value, allow_nan=False))


class HypothesisBook:
    """Atomic state for logical hypotheses, attempts, and pending corrections."""

    def __init__(self, store, *, performance_as_refinement=False):
        self.store = store
        self.performance_as_refinement = performance_as_refinement

    @contextmanager
    def _edit(self):
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT value FROM state WHERE key=?", (STATE_KEY,)).fetchone()
            records = json.loads(row[0]) if row else {}
            yield records
            db.execute("INSERT OR REPLACE INTO state VALUES(?,?)", (STATE_KEY, json.dumps(records, allow_nan=False)))

    @staticmethod
    def _attempt(records, assignment_id):
        for record in records.values():
            for attempt in record["attempts"]:
                if attempt["assignment_id"] == assignment_id:
                    return record, attempt
        raise ValueError("The assignment has no registered hypothesis attempt.")

    @staticmethod
    def _counts(record):
        count = len(record["attempts"])
        record.update(attempt_count=count, corrections_used=max(0, count - 1),
                      corrections_remaining=max(0, MAX_CORRECTIONS - max(0, count - 1)))

    def register(self, task_id, hypothesis, origin_assignment=None):
        _identifier(task_id, "task ID")
        if not isinstance(hypothesis, dict):
            raise ValueError("A logical hypothesis must be an object.")
        hypothesis = _copy(hypothesis)
        identifier = _identifier(hypothesis.get("id"), "hypothesis ID")
        if origin_assignment is not None:
            _identifier(origin_assignment, "origin assignment ID")
        origin = {"task_id": task_id, "hypothesis": hypothesis}
        with self._edit() as records:
            if identifier in records:
                record = records[identifier]
                if record["origin_hash"] != digest(origin):
                    raise ValueError("The logical hypothesis origin changed. A correction must keep its original identity.")
                if origin_assignment is not None and record.get("origin_assignment") not in {None, origin_assignment}:
                    raise ValueError("The hypothesis already has a different origin assignment.")
                if origin_assignment is not None and record.get("origin_assignment") is None:
                    record["origin_assignment"] = origin_assignment
            else:
                record = {"schema_version": 1, "id": identifier, **origin, "origin_hash": digest(origin),
                          "origin_assignment": origin_assignment, "status": "ready", "attempts": [],
                          "max_attempts": MAX_ATTEMPTS, "max_corrections": MAX_CORRECTIONS,
                          "attempt_count": 0, "corrections_used": 0, "corrections_remaining": MAX_CORRECTIONS,
                          "last_outcome": None, "last_candidate_id": None, "close_reason": None,
                          "created": time.time()}
                records[identifier] = record
            return copy.deepcopy(record)

    def begin(self, hypothesis_id, assignment_id, writer=None):
        _identifier(hypothesis_id, "hypothesis ID")
        _identifier(assignment_id, "assignment ID")
        writer = _copy(writer)
        with self._edit() as records:
            if hypothesis_id not in records:
                raise ValueError("Register the logical hypothesis before its first attempt.")
            record = records[hypothesis_id]
            for other in records.values():
                existing = next((a for a in other["attempts"] if a["assignment_id"] == assignment_id), None)
                if existing is None:
                    continue
                if other["id"] != hypothesis_id:
                    raise ValueError("This assignment belongs to a different hypothesis.")
                if existing["status"] == "paused" and record["status"] not in TERMINAL:
                    existing["status"] = "running"
                    record["status"] = "running"
                # Existing IDs are a resume, never another paid correction attempt.
                return copy.deepcopy(existing)
            if record["status"] in TERMINAL or len(record["attempts"]) >= MAX_ATTEMPTS:
                raise ValueError("This hypothesis is closed. Its initial-plus-two budget cannot reset.")
            if any(a["status"] in {"running", "paused"} for a in record["attempts"]):
                raise ValueError("Recover the current attempt before assigning a correction.")
            number = len(record["attempts"])
            attempt = {"assignment_id": assignment_id, "hypothesis_id": hypothesis_id,
                       "attempt_index": number, "attempt_number": number + 1,
                       "is_correction": number > 0, "writer": writer, "status": "running",
                       "started": time.time(), "finished": None, "outcome": None,
                       "outcome_hash": None, "interruptions": []}
            record["attempts"].append(attempt)
            record["status"] = "running"
            record["origin_assignment"] = record.get("origin_assignment") or assignment_id
            self._counts(record)
            return copy.deepcopy(attempt)

    def finish(self, assignment_id, *, stage="operation", result=None, error=None,
               evidence=(), candidate_id=None, improved=None, observations=None):
        """Finish once at the last pipeline stage reached by this attempt."""
        _identifier(assignment_id, "assignment ID")
        if candidate_id is not None:
            _identifier(candidate_id, "candidate ID")
        if improved is not None and type(improved) is not bool:
            raise ValueError("Measured progress must be true, false, or unknown.")
        classification = classify_fault(stage, result, error)
        evidence = _copy(list(evidence) if isinstance(evidence, (list, tuple)) else [evidence])
        raw_error = error if error is not None else result.get("error") if isinstance(result, dict) else None
        error_text = str(raw_error) if raw_error is not None else None
        outcome = {**classification, "error": error_text[:2000] if error_text else None,
                   "candidate_id": candidate_id, "evidence": evidence,
                   "result_hash": digest(result) if result is not None else None,
                   "improved": improved,
                   "stagnation_action": "unchanged" if not classification["measured_gpu_outcome"] else
                                        "reset" if improved is True else "increment"}
        if observations is not None:
            outcome["trial_observations"] = _copy(observations)
            if observations["status"] == "contradicted":
                outcome["stagnation_action"] = "unchanged"
        outcome_hash = digest(outcome)
        with self._edit() as records:
            record, attempt = self._attempt(records, assignment_id)
            if attempt["outcome_hash"] is not None:
                if attempt["outcome_hash"] != outcome_hash:
                    raise ValueError("A finished attempt cannot change its recorded outcome.")
                return copy.deepcopy(record)
            if record["status"] in TERMINAL:
                raise ValueError("A closed hypothesis cannot accept another attempt result.")
            if classification["resume_same_attempt"]:
                interruption = {"reason": error_text or classification["kind"],
                                "stage": stage, "evidence": evidence, "kind": classification["kind"]}
                if interruption not in attempt["interruptions"]:
                    attempt["interruptions"].append(interruption)
                attempt["status"], record["status"] = "paused", "paused"
                record["last_interruption"] = interruption
                return copy.deepcopy(record)
            change_missed = bool(observations and observations["status"] == "contradicted"
                                 and classification["kind"] in {"accepted", "performance_rejection"})
            attempt.update(status="succeeded" if classification["accepted"] and not change_missed else "failed",
                           outcome=outcome, outcome_hash=outcome_hash, finished=time.time())
            record["last_outcome"] = outcome
            if observations is not None:
                record["implementation_status"] = observations["status"]
            if candidate_id is not None:
                record["last_candidate_id"] = candidate_id
            self._counts(record)
            if classification["kind"] == "speed_screen_rejection":
                record["status"] = "closed"
                record["close_reason"] = {"code": "speed_screen_rejection", "evidence": evidence,
                    "reason": "All measured inputs lost the short speed comparison."}
            elif change_missed:
                record["status"] = "correction_pending" if len(record["attempts"]) < MAX_ATTEMPTS else "exhausted"
                if record["status"] == "exhausted":
                    record["close_reason"] = {"code": "implementation_checks_failed", "evidence": evidence,
                        "reason": "The measured implementation did not meet the declared observations."}
            elif (classification["accepted"] and observations and observations["status"] == "unverified"
                  and ("verification" in record["hypothesis"] or observations.get("design_owner") == "writer")):
                record["status"] = "closed"
                record["close_reason"] = {"code": "candidate_only", "evidence": evidence,
                    "reason": "The candidate passed. Its declared mechanism still needs a verified observation."}
            elif classification["accepted"]:
                record["status"] = "succeeded"
            elif self.performance_as_refinement and classification["kind"] == "performance_rejection":
                # Valid measured code is an archive parent. Error repairs keep
                # their two-attempt limit; tuning can branch under a new measured
                # hypothesis and the same run-wide token cap.
                record["status"] = "closed"
                record["close_reason"] = {"code": "performance_refinement", "reason":
                    "The candidate is correct. Further tuning must use the measured archive, not an error repair.",
                    "evidence": evidence}
            elif len(record["attempts"]) < MAX_ATTEMPTS:
                record["status"] = "correction_pending"
            else:
                record["status"] = "exhausted"
                record["close_reason"] = {"code": "correction_limit", "reason":
                    "The initial attempt and both correction attempts failed for this hypothesis.",
                    "last_fault": classification["kind"], "evidence": evidence}
            return copy.deepcopy(record)

    def pause(self, assignment_id, reason, evidence=(), stage="infrastructure"):
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Give the reason for pausing this attempt.")
        if stage not in {"infrastructure", "transfer", "transport", "budget", "stopped"}:
            raise ValueError("Only a recoverable interruption can pause the current attempt.")
        return self.finish(assignment_id, stage=stage, error=reason, evidence=evidence)

    def defer_for_task_budget(self, assignment_id, reason):
        """Close a writer attempt that stopped before dispatch when its target ran out of tokens."""
        with self._edit() as records:
            record, attempt = self._attempt(records, assignment_id)
            if attempt["status"] == "budget_deferred":
                return
            if (attempt["status"] != "paused" or attempt["outcome_hash"] is not None or
                    (record.get("last_interruption") or {}).get("stage") != "budget"):
                raise ValueError("Only a writer paused by its token budget can be deferred.")
            outcome = {"stage": "budget", "kind": "task_budget_deferred", "accepted": False,
                       "correct": None, "candidate_id": None, "measured_gpu_outcome": False,
                       "stagnation_action": "unchanged", "reason": reason}
            attempt.update(status="budget_deferred", outcome=outcome, outcome_hash=digest(outcome), finished=time.time())
            record.update(status="closed", last_outcome=outcome, close_reason={"code":"task_budget_deferred","reason":reason})

    def integration_failure(self, candidate_id, result, evidence=()):
        """Add later validation evidence without rewriting the operation outcome.

        The original hypothesis keeps its identity and correction count. A model
        check can invalidate a kernel keep after its operation attempt finished.
        """
        record = self.for_candidate(candidate_id)
        if record is None:
            return None
        classification = classify_fault("model", result)
        if classification["kind"] != "correctness_failure" or not classification["measured_gpu_outcome"]:
            return None
        identity = digest({"candidate_id": candidate_id, "result": result})
        with self._edit() as records:
            current = records[record["id"]]
            followups = current.setdefault("integration_failures", [])
            if any(item["id"] == identity for item in followups):
                return copy.deepcopy(current)
            # Never redirect a correction which is already running on a newer
            # source. Its writer must finish against its saved parent first.
            if any(a["status"] in {"running", "paused"} for a in current["attempts"]):
                return None
            outcome = {**classification, "candidate_id": candidate_id, "evidence": _copy(list(evidence)),
                       "result_hash": digest(result), "followup_id": identity,
                       "improved": False, "stagnation_action": "unchanged"}
            followups.append({"id": identity, "outcome": outcome, "utc": time.time()})
            current["last_outcome"] = outcome
            current["last_candidate_id"] = candidate_id
            self._counts(current)
            if len(current["attempts"]) < MAX_ATTEMPTS:
                current["status"] = "correction_pending"
                current.pop("close_reason", None)
            else:
                current["status"] = "exhausted"
                current["close_reason"] = {"code": "correction_limit", "reason":
                    "Integration failed after the initial attempt and both corrections.", "evidence": _copy(list(evidence))}
            return copy.deepcopy(current)

    def close(self, hypothesis_id, reason, evidence=()):
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 2000:
            raise ValueError("Give a bounded reason for closing this hypothesis.")
        evidence = _copy(list(evidence) if isinstance(evidence, (list, tuple)) else [evidence])
        with self._edit() as records:
            if hypothesis_id not in records:
                raise ValueError("The hypothesis is not registered.")
            record = records[hypothesis_id]
            if record["status"] in TERMINAL:
                return copy.deepcopy(record)
            if any(a["status"] in {"running", "paused"} for a in record["attempts"]):
                raise ValueError("Finish or recover the active attempt before closing the hypothesis.")
            record["status"] = "closed"
            record["close_reason"] = {"code": "closed_by_controller", "reason": reason, "evidence": evidence}
            return copy.deepcopy(record)

    def get(self, hypothesis_id):
        return copy.deepcopy(self.store.get(STATE_KEY, {}).get(hypothesis_id))

    def all(self, task_id=None):
        rows = self.store.get(STATE_KEY, {}).values()
        return copy.deepcopy(sorted((r for r in rows if task_id is None or r["task_id"] == task_id),
                                    key=lambda r: (r["created"], r["id"])))

    def attempt(self, assignment_id):
        """Look up an assignment without changing its attempt count."""
        try:
            _, attempt = self._attempt(self.store.get(STATE_KEY, {}), assignment_id)
        except ValueError:
            return None
        return copy.deepcopy(attempt)

    def for_candidate(self, candidate_id, task_id=None):
        """Find the latest hypothesis outcome that actually evaluated this source."""
        matches = []
        for record in self.all(task_id):
            related = [attempt for attempt in record["attempts"]
                       if (attempt.get("outcome") or {}).get("candidate_id") == candidate_id]
            if related:
                matches.append((max(attempt["finished"] or 0 for attempt in related), record["created"], record))
        return max(matches, key=lambda item: (item[0], item[1]))[2] if matches else None

    def outstanding(self, task_id=None):
        """Return corrections only. Paused jobs must recover their original attempt."""
        return [r for r in self.all(task_id) if r["status"] == "correction_pending"]
