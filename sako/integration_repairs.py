"""Keep exact integration failures and reopen only attributable writer work."""

import copy
import math

from .core import digest
from .hypotheses import HypothesisBook
from .path_selection import manifest_allowed, passing_paths, unsafe_evidence
from .store import operation_correct


def safe_measurement(store, measured):
    """A cached model result cannot restore a source invalidated by later evidence."""
    result = measured["result"]
    return (operation_correct(result) and result.get("reference_state_intact") is not False
            and manifest_allowed(store, measured["manifest"]))


def numerical_failures(result):
    """Read failed paths from trusted layer checks, including containing regions."""
    failures, seen = [], set()
    if unsafe_evidence(result):
        return failures
    for case in result.get("cases", []):
        def collect(check):
            if (not isinstance(check, dict) or check.get("reference_state_intact") is False
                    or check.get("acceptance_role") == "diagnostic"
                    or check.get("not_measured") or check.get("restoration_errors") or check.get("error")
                    or check.get("missing_paths")):
                return
            if check.get("scope") == "cached_generation" and check.get("passed") is False:
                detail = check.get("logits") or {}
                failure = {"path": "", "seed": check.get("seed"), "workload": case.get("workload"),
                    "scope": "model_output", "reference": check.get("reference"),
                    "check": {"passed": False, "max_abs_error": detail.get("max_abs_error"),
                              "logits": copy.deepcopy(detail), "greedy_tokens_match": check.get("greedy_tokens_match"),
                              "cache_extent_valid": check.get("cache_extent_valid"),
                              "failed_cache_layers": [copy.deepcopy(v) for v in check.get("cache_layers", [])
                                                      if v.get("passed") is False]},
                    "candidate_id": None}
                failures.append(failure)
                return
            expected, observed = check.get("expected_paths"), check.get("observed_paths")
            if isinstance(expected, list) and isinstance(observed, list) and any(p not in observed for p in expected):
                return
            for layer in check.get("layers", []):
                if layer.get("acceptance_role") == "diagnostic":
                    continue
                detail = layer.get("check", {})
                if (layer.get("passed") is False and detail.get("passed") is False
                        and layer.get("same_input") is True and layer.get("inputs_unchanged") is True
                        and not layer.get("error") and not detail.get("not_measured")
                        and type(detail.get("max_abs_error")) in (float, int)
                        and math.isfinite(detail["max_abs_error"])
                        and isinstance(layer.get("path"), str)):
                    failure = {"path": layer["path"], "seed": check.get("seed"),
                        "workload": case.get("workload"), "check": copy.deepcopy(detail),
                        "reference": check.get("reference"), "input_shapes": layer.get("input_shapes"),
                        "input_strides": layer.get("input_strides"), "input_dtypes": layer.get("input_dtypes"),
                        "input_scalars": layer.get("input_scalars"), "candidate_id": layer.get("candidate_id"),
                        "scope": "containing_region" if check.get("_containing_region")
                            or layer.get("scope") == "containing_region" else "replaced_layer"}
                    identity = digest({key: failure[key] for key in ("path", "seed", "workload", "check")})
                    if identity not in seen:
                        seen.add(identity)
                        failures.append(failure)
            region = check.get("region_checks")
            if isinstance(region, dict):
                collect({**region, "_containing_region": True})
        for check in case.get("checks", []):
            collect(check)
    return failures


def affected_candidates(measured):
    failures = numerical_failures(measured["result"])
    affected = set()
    for entry in measured["manifest"]["entries"]:
        for failure in failures:
            # Direct checks identify the exact routed source. A containing
            # region can implicate several child sources and needs isolation.
            if failure["scope"] == "model_output":
                affected.add(entry["candidate_id"])
                continue
            source = failure.get("candidate_id")
            if (failure["scope"] == "replaced_layer" and source is not None
                    and source != entry["candidate_id"]):
                continue
            if any(path == failure["path"] or path.startswith(failure["path"] + ".")
                   or failure["path"].startswith(path + ".") for path in entry["paths"]):
                affected.add(entry["candidate_id"])
    return affected


def record_failure(store, measured, *, artifact=None, config=None, changed_candidate=None, isolation=None):
    """Record one attributable failure. Combined failures need isolated checks."""
    failures = numerical_failures(measured["result"])
    targets = affected_candidates(measured)
    if changed_candidate is not None:
        old = {e["candidate_id"] for e in (measured.get("incumbent") or {}).get("entries", [])}
        changed = {e["candidate_id"] for e in measured["manifest"]["entries"]} - old
        isolated = {e["candidate_id"] for e in (isolation or {}).get("manifest", {}).get("entries", [])}
        if (changed != {changed_candidate} or isolated != changed or changed_candidate not in targets or
                not (operation_correct(isolation["result"]) or numerical_failures(isolation["result"]))):
            raise ValueError("A model repair needs an isolated measurement of its only changed candidate.")
        targets = {changed_candidate}
    if not failures or len(targets) != 1:
        return []
    candidate_id = next(iter(targets))
    entries = [e for e in measured["manifest"]["entries"] if e["candidate_id"] == candidate_id]
    if len({e["task_id"] for e in entries}) != 1:
        return []
    task_id = entries[0]["task_id"]
    tickets = store.get("integration_repair_tickets", {})
    previous = tickets.get(candidate_id)
    result_hash = digest(measured["result"])
    if previous and previous["result_hash"] == result_hash and previous.get("status") != "awaiting_isolation":
        return [previous]
    evidence = [artifact] if artifact else ["model:" + digest(measured["manifest"])]
    book = HypothesisBook(store)
    record = book.integration_failure(candidate_id, measured["result"], evidence)
    ticket = {"candidate_id": candidate_id, "task_id": task_id, "stage": "model",
        "result_hash": result_hash, "result": measured["result"], "manifest": measured["manifest"],
        "failures": failures, "evidence": evidence,
        "hypothesis_id": record["id"] if record else None,
        "status": "correction_pending" if record and record["status"] == "correction_pending" else
                  "exhausted" if record and record["status"] == "exhausted" else "requires_review"}
    if isolation is not None:
        ticket.update(isolation_key=isolation["key"], isolated_correct=operation_correct(isolation["result"]),
                      accepted_manifest=measured.get("incumbent"),
                      attribution="interaction_with_accepted_model" if operation_correct(isolation["result"]) else "changed_candidate")
    kept = store.get("kernel_baselines", {})
    current = kept.get(task_id)
    retired = []
    retained = []
    restrictions = store.get("candidate_scope_restrictions", {})
    old_restriction = restrictions.get(candidate_id)
    allowed = {}
    if current and current.get("shape_scoped"):
        for key, shape in list(current["shapes"].items()):
            if shape["candidate_id"] == candidate_id:
                retired.append({"shape_key": key, "keep": copy.deepcopy(shape)})
                paths = ([] if any(f["scope"] == "model_output" for f in failures) else
                         passing_paths(measured, candidate_id, shape["signature"], config or {}, failures))
                signature_key = digest(shape["signature"])
                paths = [path for path in paths if path in shape.get("paths", paths)]
                if old_restriction is not None:
                    paths = [path for path in paths if path in old_restriction.get("shape_paths", {}).get(signature_key, [])]
                if paths:
                    shape.update(paths=paths, model_status="path_recheck_pending")
                    allowed.setdefault(signature_key, []).extend(paths)
                    retained.append({"shape_key": key, "paths": paths, "model_status": "path_recheck_pending"})
                else:
                    current["shapes"].pop(key)
        if not current["shapes"]:
            kept.pop(task_id)
    elif current and current.get("candidate_id") == candidate_id:
        retired.append({"keep": kept.pop(task_id)})
    ticket["retired_keeps"] = (previous or {}).get("retired_keeps", []) + retired
    ticket["provisional_scopes"] = retained
    if allowed:
        restrictions[candidate_id] = {"task_id": task_id, "shape_paths": allowed,
            "failure_result_hash": result_hash, "requires_fresh_model_check": True}
    else:
        restrictions.pop(candidate_id, None)
    tickets[candidate_id] = ticket
    blocked = store.get("blocked_model_candidates", {})
    blocked[candidate_id] = measured
    rejected = store.get("rejected_model_candidates", {})
    rejected[candidate_id] = measured
    best = store.get("best_model_measurement")
    if best and any(e["candidate_id"] == candidate_id for e in best["manifest"]["entries"]):
        invalidated = store.get("invalidated_model_measurements", {})
        invalidated[digest(best["manifest"])] = best
        store.set_many({"best_model_measurement": None, "invalidated_model_measurements": invalidated})
    pending = store.get("pending_repairs", {})
    if record and record["status"] == "correction_pending":
        attempts = record["attempts"]
        slot = attempts[-1].get("writer", {}).get("slot", 0)
        pending[candidate_id] = {"candidate_id": candidate_id, "task_id": task_id, "stage": "model",
            "result": measured["result"], "evidence": evidence, "manifest": measured["manifest"],
            "repairs_used": record["corrections_used"], "slot": slot}
    else:
        pending.pop(candidate_id, None)
    store.set_many({"integration_repair_tickets": tickets, "kernel_baselines": kept,
        "blocked_model_candidates": blocked, "rejected_model_candidates": rejected, "pending_repairs": pending,
        "candidate_scope_restrictions": restrictions})
    return [ticket]
