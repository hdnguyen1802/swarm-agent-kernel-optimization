"""Use reserved task tokens to correct failures after model finalist tests."""

from .core import digest, read_json
from .decode_devices import decode_devices
from .decode_validation import measure_version, target_boundary
from .integration_repairs import numerical_failures, record_failure
from .store import BudgetExhausted, operation_correct
from .transport import TransferError


def repair_failures(result):
    from .hypotheses import classify_fault
    numerical = numerical_failures(result)
    if numerical:
        return numerical
    fault = classify_fault("model", result)
    if fault["kind"] in {"compile_error", "execution_error"}:
        return [{"scope": "model_execution", "kind": fault["kind"],
                 "error": result.get("error"), "worker_error": (result.get("worker_report") or {}).get("error")}]
    return []


def remember_failures(store, task_id, trials, winner):
    tickets = store.get("integration_repair_tickets", {})
    rejected = store.get("rejected_model_candidates", {})
    for trial in trials:
        identity = trial["candidate_id"]
        if not trial["result"].get("decision", {}).get("accepted"):
            rejected[identity] = trial
        failures = repair_failures(trial["result"])
        if failures:
            tickets.setdefault(identity, {"candidate_id": identity, "task_id": task_id, "stage": "model",
                "result_hash": digest(trial["result"]), "failures": failures, "manifest": trial["manifest"],
                "accepted_manifest": trial["incumbent"], "measurement_key": trial["key"],
                "status": "saved_after_sibling_pass" if winner else "awaiting_isolation"})
    store.set_many({"integration_repair_tickets": tickets, "rejected_model_candidates": rejected})


def repair_target(loop, summary, visit, reason):
    key = f"{loop.active_task}/{visit}"
    records = loop.store.get("decode_repair_work", {})
    state = records.setdefault(key, {"assignments": {}, "repaired_candidates": [], "complete": False})
    if state["complete"]:
        return
    measurements = loop.store.get("model_measurements", {})
    trials = [measurements[t["key"]] for t in summary["trials"] if t["key"] in measurements]
    failed = [t for t in trials if repair_failures(t["result"])]
    # Keep the exact model failures even when a sibling passes or repair funding runs out.
    remember_failures(loop.store, loop.active_task, trials, summary["winner"])
    loop.store.set("budget_reserve", None)
    loop.store.set("phase_budget", {"task_id": loop.active_task, "token_ceiling": loop.task_limits[loop.active_task]})
    devices = decode_devices(loop.worker, loop.directory, loop.profile, loop.catalog) if failed and not summary["winner"] else []
    for trial in failed if devices else ():
        if (loop.directory / "STOP").exists():
            return
        identity = trial["candidate_id"]
        saved = state["assignments"].get(identity)
        assignment = loop.work.get(str(saved)) if saved is not None else None
        left = min(loop.task_limits[loop.active_task], loop.limits["max_total_tokens"]) - loop.store.usage()["tokens"]
        if assignment is None and left < loop.costs()["writer"]:
            break
        device = devices[len(state["assignments"]) % len(devices)] if assignment is None else next(
            d for d in devices if d.gpu == assignment["evaluation_gpu"])
        if assignment is None:
            proposal = {"entries": [e for e in trial["manifest"]["entries"] if e["candidate_id"] == identity]}
            isolated = measure_version(loop.directory, loop.store, device, "isolation/" + key, proposal, None,
                                       identity, correctness_only=True)
            measurements[isolated["key"]] = isolated
            loop.store.set("model_measurements", measurements)
            if not (operation_correct(isolated["result"]) or repair_failures(isolated["result"])):
                continue
            artifact = loop.evidence_report({"combined": trial, "isolated": isolated}, "decode-repair-" + identity)
            # The combined evidence and its accepted base remain intact in the ticket.
            if numerical_failures(trial["result"]) and (
                    operation_correct(isolated["result"]) or numerical_failures(isolated["result"])):
                record_failure(loop.store, trial, artifact=artifact["artifact"], config=loop.config,
                               changed_candidate=identity, isolation=isolated)
            hypothesis = {"id": "model-repair-" + digest([key, identity])[:20], "specialty": "compute",
                "parent_id": identity, "claim": "Complete-model validation exposed a failure.",
                "change": "Repair the failure using the isolated and combined results.",
                "expected_metric": "Pass the original score and cache limits with the accepted kernels.",
                "evidence": [identity], "experiment_role": "model_repair"}
            loop.hypotheses.register(loop.active_task, hypothesis)
            tickets = loop.store.get("integration_repair_tickets", {})
            tickets[identity].update(status="correction_pending", hypothesis_id=hypothesis["id"],
                isolation_key=isolated["key"], isolated_correct=operation_correct(isolated["result"]))
            loop.store.set("integration_repair_tickets", tickets)
            context = {"failed_values": numerical_failures(trial["result"]),
                "execution_error": trial["result"].get("error"),
                "limits": {k: loop.config[k] for k in ("atol", "rtol")},
                "isolated_correct": operation_correct(isolated["result"]),
                "isolated_failures": repair_failures(isolated["result"]),
                "accepted_kernels": [{"task_id": e["task_id"], "candidate_id": e["candidate_id"],
                    "source": read_json(loop.directory / "candidates" / (e["candidate_id"] + ".json"))["source"]}
                    for e in {e["candidate_id"]: e for e in (trial["incumbent"] or {}).get("entries", [])}.values()]}
            pending = loop.store.get("pending_repairs", {})
            pending[identity] = {"candidate_id": identity, "task_id": loop.active_task, "stage": "model",
                "result": trial["result"], "manifest": trial["manifest"], "evidence": [artifact["artifact"]],
                "repairs_used": 0, "slot": 0, "decode_context": context}
            loop.store.set("pending_repairs", pending)
            loop.pending = pending
            loop.blocked = loop.store.get("blocked_model_candidates", {})
            assignment = loop.assignment(hypothesis, 0, {"id": None})
            loop.save_work(assignment, evaluation_gpu=device.gpu)
            state["assignments"][identity] = assignment["generation"]
            loop.store.set("decode_repair_work", records)
            loop.event("integration_repair", task_id=loop.active_task, candidate_id=identity,
                       isolated_correct=context["isolated_correct"], reserved_tokens=loop.repair_reserve())
        try:
            if assignment["status"] not in {"completed", "failed"}:
                loop.write_swarm(assignment)
                loop.process_candidate(assignment, device)
        except BudgetExhausted as error:
            loop.reject_writer(assignment, error)
            loop.save_work(assignment, status="budget_deferred")
            loop.hypotheses.defer_for_task_budget(assignment["id"], str(error))
            break
        except TransferError as error:
            if not assignment.get("candidate"):
                loop.save_work(assignment, status="planned", error=str(error))
            loop.hypotheses.pause(assignment["id"], str(error), stage="transfer")
            raise
        except Exception as error:
            if assignment.get("candidate"):
                raise
            loop.reject_writer(assignment, error)
        candidate = assignment.get("candidate_id")
        if candidate and candidate not in state["repaired_candidates"]:
            state["repaired_candidates"].append(candidate)
        loop.store.set("decode_repair_work", records)
    if state["repaired_candidates"]:
        outcome = target_boundary(loop.config, loop.directory, loop.profile, loop.catalog, loop.store, loop.worker,
            loop.active_task, visit, reason, repair_ids=state["repaired_candidates"])
        measured = loop.store.get("model_measurements", {})
        remember_failures(loop.store, loop.active_task, [measured[t["key"]] for t in outcome["trials"]
                          if t.get("key") in measured], outcome["winner"])
        tickets = loop.store.get("integration_repair_tickets", {})
        for original, number in state["assignments"].items():
            identity = loop.work[str(number)].get("candidate_id")
            checked = next((t for t in outcome["trials"] if t["candidate_id"] == identity), None)
            tickets[original].update(status="corrected" if checked and checked["correct"] else "repair_failed",
                                     corrected_candidate_id=identity, repair_model_result=checked)
        loop.store.set("integration_repair_tickets", tickets)
        loop.event("target_model_repair_selection", task_id=loop.active_task, winner=outcome["winner"])
    state["complete"] = True
    loop.store.set("decode_repair_work", records)
