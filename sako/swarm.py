"""A coordinator and up to five independent writers share measured lessons."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import copy
import json
import threading
import time
import traceback

from .core import candidate_id, correctness_settings, digest, read_json, validate_candidate, write_json
from .feedback import compact_feedback
from .hypotheses import HypothesisBook, measured_gpu_outcome
from .loop import SearchLoop, SYSTEM
from .store import BudgetExhausted, TaskBudgetExhausted, operation_correct
from .transport import TransferError
from .swarm_policy import SPECIALTIES, admit_experiments, estimate_costs, validate_plan, profile_shares
from .kernel_selection import kernel_first, kernel_config, keep_kernel, kernel_manifest
from .model_acceptance import workload_policy


class SwarmLoop(SearchLoop):
    def __init__(self, *args, **kwargs):
        self.state_lock = threading.RLock()
        super().__init__(*args, **kwargs)
        from .archive import CandidateArchive
        from .lessons import LessonStore
        self.decode_search = self.config.get("generation_objective") == "decode"
        self.archive = CandidateArchive(self.store, limit=self.limits.get("archive_size", 4), allow_history=self.decode_search)
        self.lessons = LessonStore(self.store, self.directory)
        self.review_lock = threading.Lock()
        self.review_context = threading.local()
        self.work = self.store.get("swarm_assignments", {})
        self.events = self.store.get("swarm_events", [])
        self.hardware = self.catalog.get("hardware", {})
        self.next_id = self.store.get("swarm_next_assignment", 0)
        self.plan_queue = self.store.get("swarm_queue", [])
        self.task_starts = self.store.get("swarm_task_starts", {})
        self.task_limits = self.store.get("swarm_task_limits", {})
        self.weighted_budget = self.limits.get("budget_allocation") == "profile_gpu_time"
        if self.weighted_budget and not self.store.get("profile_token_allocation"):
            self.store.set("profile_token_allocation", profile_shares(self.catalog, self.progress["queue"],
                                                                      self.limits["max_total_tokens"]))
        self.task_counts = self.store.get("swarm_task_counts", {})
        self.hypotheses = HypothesisBook(self.store,
            performance_as_refinement=kernel_first(self.config) and self.limits.get("performance_refinement", False))
        self.validation_policy = {**correctness_settings(self.config),
            **workload_policy(self.config.get("workloads", [])),
            "speed_scope": "complete_model_forward", "speed_reference": "torch_compile",
            "kernel_backend": self.config.get("kernel_backend", "triton"),
            "containing_region_policy": self.config.get("containing_region_policy", "required"),
            "model_speed_scope": self.config.get("model_speed_scope", "all_workloads"),
            "model_target_rule": "Declare workloads from the manifest before timing.",
            "other_workloads_min_speedup": 1.0 if self.config.get("model_speed_scope") == "targeted_workloads" else self.config["min_speedup"],
            "supplemental_correctness": self.config.get("supplemental_correctness", {})}
        if any(w.get("mode") == "cached_generation" for w in self.config.get("workloads", [])):
            self.validation_policy.update(speed_scope="complete_cached_generation", model_timing="complete_cached_generation",
                generation_checks="Compare greedy tokens, prediction scores and KV caches before timing.")
        if kernel_first(self.config):
            self.validation_policy.update(speed_scope="operation", speed_reference="current_kept_kernel",
                kernel_min_speedup=kernel_config(self.config)["min_speedup"], model_integration="deferred",
                initial_speed_reference="torch_compile", model_min_speedup=self.config["min_speedup"],
                shape_scoped=self.config.get("kernel_search", {}).get("shape_scoped", False),
                timing_mode=self.config.get("kernel_search", {}).get("timing_mode", "events"))
        if self.decode_search:
            from .decode_search import FINALIST_LIMIT
            self.validation_policy.update(model_timing="cached_decode", speed_reference="original_torch_compile",
                                          complete_model_trials_per_target=FINALIST_LIMIT, model_integration="after_each_target",
                                          final_combined_test=True, speed_scope="decode_operation")

    def evidence_report(self, value, name, **metadata):
        with self.state_lock:
            return super().evidence_report(value, name, **metadata)

    def process_candidate(self, assignment, device=None):
        # Serialize controller changes. Independent GPU jobs release this lock while they run.
        with self.review_lock:
            self.review_context.active = True
            self.review_context.device = device
            try:
                return self._process_candidate(assignment)
            finally:
                self.review_context.active = False
                self.review_context.device = None

    def worker_io(self, call, *args, **kwargs):
        if not getattr(self.review_context, "active", False):
            return call(*args, **kwargs)
        return self.call_outside_review(call, *args, **kwargs)

    def call_outside_review(self, call, *args, **kwargs):
        """Let the other thread use controller state while this request waits for I/O."""
        self.review_lock.release()
        try:
            return call(*args, **kwargs)
        finally:
            self.review_lock.acquire()

    def event(self, kind, **fields):
        with self.state_lock:
            event = {"id": len(self.events), "kind": kind, "utc": time.time(), **fields}
            self.events.append(event)
            self.store.set("swarm_events", self.events)
            return event

    def save_work(self, a, **fields):
        with self.state_lock:
            a.update(fields)
            self.work[str(a["generation"])] = a
            self.store.set("swarm_assignments", self.work)

    def costs(self):
        with self.store.connect() as db:
            costs = estimate_costs(self.limits, db.execute("SELECT * FROM calls ORDER BY id").fetchall())
        for role, floor in self.store.get("request_reservation_floors", {}).items():
            if role in costs:
                costs[role] = max(costs[role], floor)
        return costs

    def remaining_task_tokens(self):
        return self.task_limits[self.active_task] - self.repair_reserve() - self.store.usage()["tokens"]

    def repair_reserve(self):
        share = self.task_limits[self.active_task] - self.task_starts[self.active_task]
        if not self.decode_search:
            return 0
        costs = self.costs()
        search_cost = costs["writer"] + costs["coordinator"]
        return min(int(share * self.limits.get("model_repair_fraction", 0.)),
                   max(0, share - search_cost))

    def start_task(self):
        self.restore_integration_siblings()
        from .hierarchy import enabled, region_for
        if enabled(self.config):
            done = {r["task_id"] for r in self.progress["completed"]}
            for region in self.catalog["hierarchy"]["regions"]:
                if set(region["subsection_ids"]).issubset(done):
                    self.finish_region(region)
        if self.active_task is None:
            return
        self.begin_task()
        key = self.active_task
        if key not in self.task_starts:
            used = self.store.usage()["tokens"]
            later = len(self.progress["queue"]) - self.progress["index"]
            self.task_starts[key] = used
            remaining = max(0, self.limits["max_total_tokens"] - used)
            if self.weighted_budget:
                tail = self.progress["queue"][self.progress["index"]:]
                updated = (self.store.get("hierarchy_latest_profile") or {}).get("task_priorities")
                if self.decode_search and self.store.get("continued_search"):
                    allocations = self.store.get("profile_token_allocation")
                    spent = self.store.task_token_usage()
                    weights = {t: {"priority_ms": max(0, allocations.get(t, 0) - spent.get(t, 0))} for t in tail}
                    if any(x["priority_ms"] for x in weights.values()):
                        updated = weights
                share = profile_shares(self.catalog, tail, remaining, updated)[key]
            else:
                share = remaining if enabled(self.config) else remaining // later
            self.task_limits[key] = used + share
            self.store.set_many({"swarm_task_starts": self.task_starts, "swarm_task_limits": self.task_limits})
            region = region_for(self.catalog, key) if enabled(self.config) else None
            self.event("task_started", task_id=key, token_ceiling=self.task_limits[key], allocated_tokens=share,
                       region_id=region["id"] if region else None,
                       reason="profile_gpu_time" if self.weighted_budget else "focus_active_bottleneck" if region else "reserve_later_priorities")
        if self.weighted_budget:
            self.store.set("phase_budget", {"task_id": key, "token_ceiling": self.task_limits[key] - self.repair_reserve()})

    def finish_region(self, region):
        """Verify composed replacements before moving to another bottleneck."""
        if self.config.get("generation_objective") == "decode":
            return
        from .hierarchy import reorder_remaining
        completed = self.store.get("hierarchy_completed_regions", {})
        if region["id"] in completed:
            return
        from .kernel_selection import accepted_model_trials, model_trial_manifest, region_manifests
        from .path_selection import manifest_allowed, restricted_manifest
        measured = None
        comparisons = []
        repair_tickets = []
        clean = accepted_model_trials(self.config)
        alternatives = list(region_manifests(self.store, self.catalog, region, config=self.config,
                                             include_accepted=not clean))
        seen = {digest(manifest) for _, manifest in alternatives}
        while alternatives:
            task_ids, manifest = alternatives.pop(0)
            if not any(e["task_id"] in region["subsection_ids"] for e in manifest["entries"]):
                continue
            if not manifest_allowed(self.store, manifest):
                self.event("region_alternative_skipped", region_id=region["id"], reason="A source failed a prior numerical check.",
                           manifest=manifest)
                continue
            base = self.store.get("best_model_measurement")
            combined = model_trial_manifest(self.store, self.catalog, manifest) if clean else manifest
            incumbent = base["manifest"] if clean and base and combined != base["manifest"] else None
            trial = self._measure_manifest(combined, self.next_id, **({"incumbent": incumbent} if incumbent else {}))
            repair_tickets.extend(self.integration_repairs(trial))
            retry = restricted_manifest(self.store, manifest)
            if retry is not None and digest(retry) not in seen:
                seen.add(digest(retry))
                alternatives.append(([key for key in task_ids if any(e["task_id"] == key for e in retry["entries"])], retry))
                self.event("region_scope_recheck", region_id=region["id"], manifest=retry,
                           reason="Excluded failed paths. The smaller scope requires a new complete model check.")
            comparisons.append({"task_ids": task_ids, "measurement": trial})
            def quality(value):
                from .integration_repairs import safe_measurement
                result = value["result"]
                safe = safe_measurement(self.store, value)
                return (safe and bool(result.get("decision", {}).get("accepted")), safe,
                        (result.get("decision", {}).get("speedup") or 0) if safe else 0)
            if measured is None or quality(trial) > quality(measured):
                measured = trial
                if not clean or quality(trial)[0]:
                    choices = self.store.get("kernel_region_choices", {})
                    choices[region["id"]] = task_ids
                    self.store.set("kernel_region_choices", choices)
            self.event("region_alternative", region_id=region["id"], task_ids=task_ids,
                       decision=trial["result"].get("decision"), manifest=combined)
        if measured is not None:
            self.store.set("last_model_measurement", measured)
            from .integration_repairs import safe_measurement
            if measured["result"].get("decision", {}).get("accepted") and safe_measurement(self.store, measured):
                previous = self.store.get("best_model_measurement")
                if (previous is None or measured.get("incumbent") == previous["manifest"]
                        or measured["result"]["decision"]["speedup"] > previous["result"]["decision"]["speedup"]):
                    self.store.set("best_model_measurement", measured)
            self.event("region_integration", region_id=region["id"],
                       decision=measured["result"].get("decision"), manifest=measured["manifest"])
        repair_tasks = {ticket["task_id"] for ticket in repair_tickets if ticket["status"] == "correction_pending"}
        if repair_tasks:
            self.reopen_integration_tasks(repair_tasks, region["id"])
            return
        best = self.store.get("best_model_measurement")
        adopted = best["manifest"] if best else {"entries": []}
        identity = digest(adopted)
        original_identity = digest({"entries": []})
        if self.active_task is not None and identity != self.store.get("hierarchy_profiled_manifest", original_identity):
            path = self.directory / "hierarchy" / (identity + ".manifest.json")
            write_json(path, adopted)
            self.worker.put(path, self.relative(path))
            result = self.worker.run("reprofile", self.directory, profile_dir=self.relative(self.profile),
                manifest=self.relative(path), candidate_root=self.relative(self.directory / "candidates"),
                output_dir=self.relative(self.directory / "hierarchy" / identity))
            report = self.evidence_report(result, "hierarchy-reprofile-" + identity)
            self.progress = reorder_remaining(self.catalog, self.progress, result["task_priorities"])
            self.store.set_many({"task_progress": self.progress, "active_task": self.active_task,
                "hierarchy_profiled_manifest": identity, "hierarchy_latest_profile": result})
            self.event("region_reprofiled", region_id=region["id"], evidence=report["artifact"],
                       queue=self.progress["queue"], model_state=result["model_state"])
        completed[region["id"]] = {"region_id": region["id"], "subsection_ids": region["subsection_ids"],
            "reason": "Every subsection has a recorded search decision.",
            "subsection_decisions": [d for d in self.progress["completed"] if d["task_id"] in region["subsection_ids"]],
            "integration_decision": measured["result"]["decision"] if measured else None,
            "alternatives": comparisons,
            "profile_action": "reuse_unchanged_model_profile" if identity == original_identity else "profile_accepted_model",
            "adopted_manifest": adopted}
        self.store.set("hierarchy_completed_regions", completed)
        self.event("region_completed", **completed[region["id"]])

    def integration_repairs(self, measured):
        """Attribute numerical failures before giving a writer a correction."""
        from .integration_repairs import affected_candidates, record_failure
        affected = affected_candidates(measured)
        trials = [measured]
        if len(affected) > 1:
            # A combined failure does not prove that every member is wrong.
            trials = [self._measure_manifest({"entries": [e for e in measured["manifest"]["entries"]
                      if e["candidate_id"] == key]}, self.next_id) for key in sorted(affected)]
        tickets = []
        for trial in trials:
            report = None
            if affected_candidates(trial):
                report = self.evidence_report(trial, "integration-repair-" + digest(trial["manifest"]))
            tickets.extend(record_failure(self.store, trial, artifact=report["artifact"] if report else None,
                                          config=self.config))
        self.pending = self.store.get("pending_repairs", {})
        self.blocked = self.store.get("blocked_model_candidates", {})
        for ticket in tickets:
            self.event("integration_repair", task_id=ticket["task_id"], candidate_id=ticket["candidate_id"],
                       hypothesis_id=ticket["hypothesis_id"], status=ticket["status"],
                       failed_paths=[f["path"] for f in ticket["failures"]], evidence=ticket["evidence"])
        return tickets

    def reopen_integration_tasks(self, task_ids, region_id=None):
        self.reopen_tasks(task_ids)
        self.event("integration_tasks_reopened", region_id=region_id, task_ids=[
            key for key in self.progress["queue"] if key in task_ids],
            reason="A measured integration failure needs a bounded writer correction.")

    def reopen_tasks(self, task_ids, *, best_score=0.):
        """Reopen selected tasks while preserving other decisions and archive evidence."""
        queue = self.progress["queue"]
        rewind = min(queue.index(key) for key in task_ids)
        preserved = self.store.get("integration_preserved_completions", {})
        for key in task_ids:
            preserved.pop(key, None)
        for decision in self.progress["completed"]:
            if queue.index(decision["task_id"]) >= rewind and decision["task_id"] not in task_ids:
                preserved[decision["task_id"]] = decision
        self.progress.update(index=rewind, round=0,
                             stagnant=0, best_operation_score=best_score,
                             completed=[d for d in self.progress["completed"] if queue.index(d["task_id"]) < rewind])
        from .hierarchy import region_for
        regions = {r["id"] for key in task_ids for r in [region_for(self.catalog, key)] if r}
        complete = self.store.get("hierarchy_completed_regions", {})
        for key in regions:
            complete.pop(key, None)
        self.plan_queue = []
        self.store.set_many({"task_progress": self.progress, "active_task": self.active_task, "swarm_queue": [],
            "integration_preserved_completions": preserved, "hierarchy_completed_regions": complete,
            "swarm_task_limits": self.task_limits})

    def revisit_with_budget(self):
        """After a pass, select a supported task for a new measured experiment."""
        if not (kernel_first(self.config) and self.limits.get("revisit_with_budget")):
            return False
        visits = self.progress.get("budget_visits", {})
        priorities = (self.store.get("hierarchy_latest_profile") or {}).get("task_priorities", {})
        cap = self.limits.get("max_task_experiments")
        choices = []
        for key in self.progress["queue"]:
            priority = priorities.get(key, self.by_id[key])
            if priority.get("excluded_reason") or priority["priority_ms"] <= 0:
                continue
            if cap is not None and sum(a["task_id"] == key for a in self.work.values()) >= cap:
                continue
            parents = self.archive.refresh(self.by_id[key], self.hardware)
            best = parents[0] if parents else None
            count = visits.get(key, {}).get("count", 0)
            potential = priority["priority_ms"] / (count + 1)
            saving = (best or {}).get("estimated_model_saving_ms", 0.)
            choices.append((-potential, -saving, key, best["id"] if best else None,
                            best["score"] if best else 0., count))
        if not choices:
            return False
        budget = self.store.budget_snapshot(self.limits)
        admission = admit_experiments(self.limits, budget["tokens"], self.costs(), 1)
        self.store.set("budget_decision", {**admission, "stage": "revisit"})
        if not admission["admit"]:
            raise BudgetExhausted("The remaining tokens cannot fund another writer and coordinator review.")
        _, _, key, parent, score, count = min(choices)
        remaining = budget["remaining_tokens"]
        if self.weighted_budget:
            amounts = profile_shares(self.catalog, [row[2] for row in choices], remaining, priorities)
            affordable = [row for row in choices if amounts[row[2]] >= admission["required_tokens"]]
            # Pool residual fractions only after every target has had a turn.
            _, _, key, parent, score, count = min(affordable or choices)
            remaining = remaining if not affordable else amounts[key]
        visit = {"count": count + 1, "first_generation": self.next_id, "parent_id": parent}
        self.progress["budget_visits"] = {**visits, key: visit}
        self.task_starts[key] = budget["tokens"]
        self.store.set("swarm_task_starts", self.task_starts)
        self.task_limits[key] = budget["tokens"] + remaining
        self.reopen_tasks({key}, best_score=score)
        self.event("task_revisited", task_id=key, **visit, remaining_tokens=budget["remaining_tokens"])
        return True

    def restore_integration_siblings(self):
        """Advance over saved sibling decisions without another writer request."""
        preserved = self.store.get("integration_preserved_completions", {})
        while self.active_task in preserved:
            key = self.active_task
            decision = preserved.pop(key)
            self.advance(decision["reason"], self.next_id)
            self.progress["completed"][-1] = decision
            self.store.set_many({"task_progress": self.progress, "integration_preserved_completions": preserved})
            self.event("integration_sibling_restored", task_id=key, reason="Reuse the saved stopping decision.")

    def resume_integration_repairs(self):
        """An explicit resumed run can repair a check made after search stopped."""
        tasks = {ticket["task_id"] for ticket in self.store.get("integration_repair_tickets", {}).values()
                 if ticket["status"] == "correction_pending" and ticket.get("hypothesis_id")
                 and (self.hypotheses.get(ticket["hypothesis_id"]) or {}).get("status") == "correction_pending"}
        if tasks:
            self.reopen_integration_tasks(tasks)

    def diagnose_model(self, measured):
        if not self.config.get("diagnostics", {}).get("enabled", False):
            return None
        results = self.store.get("swarm_diagnostics", {})
        if self.active_task in results:
            return results[self.active_task]
        path = self.directory / "diagnostics" / (self.active_task + ".manifest.json")
        write_json(path, measured["manifest"])
        self.worker_io(self.worker.put, path, self.relative(path))
        try:
            result = self.worker_io(self.worker.run, "diagnose_model", self.directory, profile_dir=self.relative(self.profile),
                manifest=self.relative(path), candidate_root=self.relative(self.directory / "candidates"),
                output_dir=self.relative(self.directory / "diagnostics" / self.active_task))
            from .profiling import collect_profile_artifacts
            self.worker_io(collect_profile_artifacts, self.worker, result, self.directory)
        except TransferError:
            raise
        except Exception as error:
            result = {"status": "unavailable", "acceptance_evidence": False, "reason": str(error)}
        results[self.active_task] = result
        self.store.set("swarm_diagnostics", results)
        self.event("model_diagnostic", task_id=self.active_task, status=result.get("status"),
                   summary=result.get("boundary_summary"), evidence=result.get("artifacts", []))
        return result

    def recall(self):
        return self.lessons.retrieve(self.by_id[self.active_task], self.hardware,
            limit=self.limits.get("lesson_limit", 8), max_chars=self.limits.get("lesson_context_chars", 12000))

    def diagnostic_feedback(self, task_id):
        result = self.store.get("swarm_diagnostics", {}).get(task_id)
        if result is None:
            return None
        return {key: compact_feedback(result.get(key)) for key in
                ("scope", "status", "case", "comparisons", "boundary_summary", "reasons", "artifacts", "acceptance_evidence")}

    @staticmethod
    def operation_feedback(result):
        summary = SearchLoop.operation_feedback(result)
        if summary is not None:
            for small, raw in zip(summary["cases"], result.get("cases", [])):
                build = raw.get("build", {})
                for field in ("reference", "reference_path", "timing_reference"):
                    if field in raw:
                        small[field] = raw[field]
                small["compiler_resources"] = compact_feedback({
                    "resource_reports": build.get("resource_reports", [])[:4],
                    "spill_reports": build.get("spill_reports", [])[:4]})
        return summary

    def planning_epoch(self):
        task_work = [a for a in self.work.values() if a["task_id"] == self.active_task]
        return [sum(a["status"] == "completed" for a in task_work),
                sum(a["status"] == "failed" for a in task_work)]

    def model_costs(self, task_id):
        from .trial_evidence import model_opportunity
        updated = self.store.get("hierarchy_latest_profile") or {}
        task = updated.get("task_priorities", {}).get(task_id, self.by_id[task_id])
        times = updated.get("workload_times", self.catalog.get("compiled_profile", {}).get("workload_times", []))
        return model_opportunity(task, self.config.get("workloads", []), times)

    def search_focus(self):
        divided = self.store.get("operation_split:" + self.active_task)
        if not divided:
            return None
        history = [p for p in self.store.get("swarm_plans", []) if p.get("task_id") == self.active_task]
        scopes = [s for s in divided["scopes"] if not any(
            p.get("focus_id") == s["id"] and p["action"] == "move_on" for p in history)]
        return min(scopes, key=lambda s: sum(p.get("focus_id") == s["id"] for p in history), default=None)

    def kept_summary(self, task):
        kept = self.store.get("kernel_baselines", {}).get(task["id"])
        if not task.get("separate_operations") or not kept:
            return kept
        winners = {}
        for value in kept.get("shapes", {}).values():
            entry = winners.setdefault(value["candidate_id"], {"module_shapes": 0, "example_paths": []})
            entry["module_shapes"] += 1
            for path in value["paths"]:
                if path not in entry["example_paths"] and len(entry["example_paths"]) < 3:
                    entry["example_paths"].append(path)
        return {"winning_candidates": winners}

    def coordinator(self, capacity, *, overlap_review=False):
        from .archive import summarize_parent_use
        from .coordinator_context import fit_coordinator_context
        from .feedback import save_feedback
        from .evidence_registry import coordinator_evidence
        epoch = self.planning_epoch()
        task = self.by_id[self.active_task]
        focus = self.search_focus()
        archive = self.archive.refresh(task, self.hardware)
        lessons = self.recall()
        hypothesis_states = self.hypotheses.all(self.active_task)
        active_states = [h for h in hypothesis_states if h["status"] not in {"closed", "exhausted", "succeeded"}]
        closed_states = [h for h in hypothesis_states if h["status"] in {"closed", "exhausted", "succeeded"}]
        from .coordinator_context import planning_task
        coordinator_task = planning_task(self.task_prompts[self.active_task])
        evidence = coordinator_task.get("compiled_evidence", {})
        if "generated_kernels" in evidence:
            evidence["generated_kernels"] = [{k: v for k, v in entry.items() if k != "source"}
                                              for entry in evidence["generated_kernels"]]
        repair_ids = [k for k, r in self.pending.items() if r["task_id"] == self.active_task]
        task_work = [a for a in self.work.values() if a["task_id"] == self.active_task]
        usage = summarize_parent_use(task_work)
        from .recombination import merge_pairs
        pairs = (merge_pairs(task, archive, self.store.candidates(self.active_task), task_work)
                 if self.limits.get("candidate_merging", True) else [])
        # Keep both complete sources in the writer request. Do not offer a pair
        # whose source alone cannot fit the writer context.
        pairs = [p for p in pairs if p["source_characters"] + len(json.dumps(self.task_prompts[self.active_task])) + 12000
                 <= self.limits["context_chars"]]
        pairs = pairs[:self.limits["writers"]]
        from .designs import writer_options
        portfolio = (writer_options(archive, pairs, task_work, capacity)
                     if self.limits.get("design_portfolio", False) else [])
        if self.limits.get("design_portfolio") and not portfolio:
            epochs = self.store.get("swarm_planning_epochs", {})
            epochs[self.active_task] = epoch
            self.store.set_many({"swarm_planning_epochs": epochs, "replan_after:" + self.active_task: "all_results"})
            return {"action": "wait", "hypotheses": [], "id": None}
        merge_inflight = any(a["hypothesis"].get("merge") and a["status"] not in {"completed", "failed"}
                             for a in task_work)
        require_merge = any(not p["attempted"] for p in pairs) and not merge_inflight and not portfolio
        refinement_inflight = any(value["active_refinements"] for value in usage["by_parent"].values())
        refinement_ids = [p["id"] for p in archive if p["id"] not in repair_ids]
        historical = []
        if self.config.get("generation_objective") == "decode":
            known = {p["id"] for p in archive}
            historical = [r for r in self.store.candidate_summaries() if r["task"] == task["id"] and r["id"] not in known]
        require_trial = (kernel_first(self.config) and self.limits.get("require_archive_trial", False)
                         and usage["archive_refinements"] == 0)
        rejected_models = self.store.get("rejected_model_candidates", {})
        best_model = self.store.get("best_model_measurement")
        best_model_ids = {e["candidate_id"] for e in (best_model or {}).get("manifest", {}).get("entries", [])}
        from .trial_evidence import model_saving, model_opportunity
        original_costs = model_opportunity(task, self.config.get("workloads", []),
            self.catalog.get("compiled_profile", {}).get("workload_times", []))
        archive = [{**p, "reuse": usage["by_parent"].get(p["id"], {
            "archive_refinements": 0, "repairs": 0, "active_refinements": 0, "completed_refinements": 0}),
            "operation_measurements": self.operation_feedback(self.row(p["id"])["result"]),
            "projected_from_original": model_saving(task, self.row(p["id"])["result"]),
            "model_measurements": self.model_feedback(rejected_models.get(p["id"]) or
                                                      (best_model if p["id"] in best_model_ids else None))}
            for p in archive]
        example_parent = refinement_ids[0] if refinement_ids else None
        candidates = self.store.candidates(self.active_task)
        measured = {r["id"]: r["result"] for r in candidates if measured_gpu_outcome(r.get("result"))}
        visit = self.progress.get("budget_visits", {}).get(self.active_task)
        visit_measured = measured
        revisit_parent = None
        if visit:
            visit_measured = {r["id"]: measured[r["id"]] for r in candidates
                              if r["id"] in measured and r["generation"] >= visit["first_generation"]}
            parent = visit.get("parent_id")
            parent_info = next((p for p in archive if p["id"] == parent), {})
            useful_parent = parent_info.get("estimated_model_saving_ms", parent_info.get("score", 0.) - 1.) > 0
            if parent in refinement_ids and useful_parent and not any(
                    a["generation"] >= visit["first_generation"] and a.get("parent_id") == parent
                    and a.get("candidate_id") in visit_measured for a in task_work):
                revisit_parent = parent
            if refinement_ids and not visit_measured:
                require_trial = True
        assignments = {a["id"]: {k: a.get(k) for k in ("id", "task_id", "candidate_id", "observations")}
                       for a in task_work}
        registry = coordinator_evidence(self.active_task, self.hardware,
            candidates, lessons, self.evidence,
            self.diagnostic_feedback(self.active_task))
        inflight = [{k: a.get(k) for k in ("id", "hypothesis", "status", "candidate_id")}
                    for a in self.work.values() if a["task_id"] == self.active_task
                    and a["status"] not in {"completed", "failed"}]
        allowed_actions = ["continue"]
        if inflight:
            allowed_actions.append("wait")
        if not revisit_parent and not (refinement_ids and require_trial and not refinement_inflight) and visit_measured:
            allowed_actions.append("move_on")
        prompt = {"request": "Select distinct, testable GPU kernel hypotheses using measured evidence. Return JSON with "
            "action (choose from allowed_actions), reason, evidence (known IDs), and hypotheses. Each hypothesis needs "
            "objective, specialty, parent_id (eligible ID or null for a new design), claim, change, expected_metric, and evidence. "
            "Set an evidence-based problem and desired outcome. The writer owns the algorithm, tile sizes, launch "
            "settings and implementation checks. Algorithm and change details are suggestions the writer may replace. "
            "Return at most five new hypotheses per call. Use the available writer slots, but do not invent work to fill them. "
            "Use wait with no hypotheses when work is in flight and new evidence is needed. "
            "A failed parent may be used only for its listed repair. Give repairs priority. Explain why each proposed change "
            "can help. Existing in-flight hypotheses already have writers. Do not duplicate them. Share conclusions, not "
            "conversations. Do not change precision, workload, tolerance, budget, or test rules. "
            "Use measured results. Write Triton Python kernels. "
            "If progress has stopped, explain move_on "
            "using measured evidence. Explain why another concrete experiment has low expected value. "
            "API, source, and compile errors are not evidence of a performance plateau. "
            "Each hypothesis has one initial attempt and two corrections. A failed hypothesis does not close the operation. "
            "Corrections keep the same hypothesis ID even if the writer changes. "
            "Use a measured archive parent as the default starting point. Compare its timings, model result, and reuse history. "
            "Prefer a promising parent that has not yet been refined. Keep different designs alive. "
            "When ordinary archive parents exist and no ordinary refinement is in flight, include at least one ordinary "
            "archive refinement in a continue plan. Repairs do not satisfy this rule. "
            "Cite each selected parent in that hypothesis's evidence. For each fresh start, give fresh_start_reason and "
            "cite an available parent you considered. Explain why the new design needs a fresh start. "
            "To skip all ordinary refinements or move on, supply archive_skip with a concrete low-value reason and "
            "evidence that cites an available ordinary parent. Do not invent a refinement just to fill slots. "
            "Assign independent objectives or specialist perspectives. Use model_costs and phase costs "
            "to prioritize saved model time. Supplied algorithms and verification checks are advisory. "
            "Writers declare their own concrete algorithm and observable checks with the source. "
            "Use merge_pairs to combine ideas from two complete Triton candidates for shared measured shapes. "
            "Set hypothesis.merge={parent_ids:[A,B],shape_ids:[shared IDs]} and parent_id to A or B. Cite both parents. "
            "The writer receives both complete sources and measurements and chooses which ideas can work together. "
            "Do not assume that speed ratios multiply or that the merged kernel will improve. "
            "When merge_policy.trial_required is true, propose an untried pair or give merge_skip with a concrete "
            "conflict or low-value reason and evidence for both parents. Merged kernels use the same tests. "
            "Queue independent work now. The controller waits locally for the batch. Use replan_after=next_result only "
            "when an intermediate result can open independent work. A speed result applies only to the submitted code. "
            "A succeeded candidate with unverified implementation_status leaves the proposed mechanism untested.",
            "task": coordinator_task, "hardware": self.hardware, "allowed_actions": allowed_actions,
            "validation_policy": self.validation_policy,
            "model_costs": self.model_costs(self.active_task),
            "previous_model_trace": self.store.get("last_performance_trace"),
            "kept_kernel": self.kept_summary(task),
            "specialties": SPECIALTIES, "available_writer_slots": capacity, "parents": archive,
            "merge_pairs": pairs, "merge_policy": {"trial_required": require_merge, "inflight": merge_inflight,
                "acceptance": "Unchanged correctness, operation speed and complete model tests."},
            "archive_policy": {"eligible_parent_ids": refinement_ids,
                "must_measure_one_refinement": bool(refinement_ids) and require_trial,
                "refinement_inflight": refinement_inflight,
                "refinement_required": bool(refinement_ids) and not refinement_inflight,
                "untried_parent_ids": [p for p in refinement_ids if not usage["by_parent"].get(p, {}).get("archive_refinements")],
                "skip_format": {"archive_skip": {"reason": "Explain why available parents have low value.",
                    "evidence": refinement_ids[:1]}}},
            "parent_selection_counts": {**usage, "by_parent": {
                p["id"]: usage["by_parent"][p["id"]] for p in archive if p["id"] in usage["by_parent"]}},
            "lessons": lessons, "inflight": inflight, "known_evidence": sorted(registry.ids()),
            "evidence_types": registry.types(),
            "hypothesis_states": compact_feedback(active_states + closed_states[-self.limits.get("lesson_limit", 8):]),
            "closed_hypothesis_count": len(closed_states),
            "operation_results": self.review_ranking([r for r in candidates if r["result"] is not None]),
            "pending_repairs": self.repair_feedback(self.pending),
            "model_result": self.model_feedback(self.store.get("last_model_measurement")),
            "model_diagnostic": self.diagnostic_feedback(self.active_task),
            "progress": self.progress,
            "recent_events": compact_feedback([e for e in self.events if e.get("task_id") == self.active_task][-10:]),
            "budget": self.store.budget_snapshot(self.limits), "task_remaining_tokens": self.remaining_task_tokens(),
            "example": {"action": "continue", "reason": "Explain with evidence.",
                "evidence": ["contract:" + self.active_task], "hypotheses": [{"specialty": "memory", "parent_id": None,
                    "objective": "Reduce the measured runtime cost of this operation.",
                    "claim": "State a specific suspected limit.", "change": "State one concrete code change.",
                    "expected_metric": "State the measured effect to check.", "evidence": ["contract:" + self.active_task]}]}}
        from .hierarchy import enabled, region_for
        if visit:
            prompt["request"] += (" This is a funded revisit. Use the earlier results to choose a distinct change. "
                "Moving on requires citing a new GPU measurement from this visit.")
            if revisit_parent:
                prompt["request"] += f" Refine parent {revisit_parent} with a new change before moving on."
        if enabled(self.config):
            prompt["request"] += (" This is a bottleneck tree. Read hierarchy_context.search_scope. "
                "A parent_alternative permits wider fusion of the complete parent contract; a subsection keeps its boundary. "
                "Prefer hypotheses about measured limits "
                "of this subsection and its required shapes. move_on closes this subsection. The controller "
                "finishes sibling subsections before selecting another bottleneck. "
                "There is no fixed top-four queue or equal budget split among future regions.")
            prompt["hierarchy"] = {"active_region": region_for(self.catalog, self.active_task),
                "completed_regions": list(self.store.get("hierarchy_completed_regions", {})),
                "current_subsection": self.active_task,
                "latest_priorities": self.store.get("hierarchy_latest_profile", {}).get("task_priorities", {}),
                "unvisited_regions": [{"id": r["id"], "kind": r["kind"], "priority_ms": r["priority_ms"]}
                    for r in self.catalog["hierarchy"]["regions"]
                    if not any(a["task_id"] in r["subsection_ids"] for a in self.work.values())]}
        if self.weighted_budget:
            prompt["budget_allocation"] = {"basis": "Measured GPU time of eligible targets",
                "current_tokens_remaining": self.remaining_task_tokens(),
                "instruction": "Work within this target's share. Unused tokens pass to later targets."}
        if self.config.get("generation_objective") == "decode":
            prompt["request"] += (" Optimize cached decode. Each measured ratio compares with torch.compile. "
                "Refine promising kernels after local wins. At the end of this target, up to four candidates run "
                "in the complete model. The validated winner is the base for the next target. "
                "Any saved source below can be a parent, including a failed design that you can improve.")
            prompt["other_saved_parents"] = [{"id": r["id"], "score": r["score"], "writer": r["island"]} for r in historical]
        if focus:
            from .staged_evaluation import focus_measurements
            prompt["operation_search"] = {"focus_id": focus["id"], "example_paths": focus["paths"][:6],
                "path_count": len(focus["paths"]), "parent_results": {
                    parent["id"]: focus_measurements(self.row(parent["id"])["result"], focus["paths"]) for parent in archive}}
            prompt["request"] += " Plan for this measured subset. Other subsets have separate decisions and share the code archive."
        example = prompt["example"]["hypotheses"][0]
        if example_parent:
            example.update(parent_id=example_parent, evidence=[example_parent],
                           change="Refine the supplied implementation using one measured limit.")
        def fit_request(value, name):
            artifact = save_feedback(self.directory, value, name=name)["artifact"]
            registry.add(artifact["path"], "artifact")
            value = {**value, "known_evidence": sorted(registry.ids()), "evidence_types": registry.types()}
            fitted = fit_coordinator_context(value, self.limits["context_chars"], artifact=artifact,
                target_chars=self.limits.get("coordinator_target_chars"))
            self.store.set("coordinator_context:" + self.active_task, {
                "full_context": artifact, "full_characters": len(json.dumps(value)),
                "sent_characters": len(json.dumps(fitted)), "limit": self.limits["context_chars"],
                "target": self.limits.get("coordinator_target_chars", self.limits["context_chars"])})
            return fitted
        if portfolio:
            prompt["writer_experiments"] = portfolio
            prompt["request"] += (" Assign each new hypothesis a distinct experiment_role from writer_experiments "
                "and its listed parent_id. Use the different objectives to avoid duplicate work. "
                "Choose your own algorithm within that objective. Describe new code changes, not just new launch "
                "settings. Automatic tuning already tests small setting variations and reuses matching results.")
        full_prompt = prompt
        prompt = fit_request(full_prompt, "coordinator-context")
        key = "swarm_plan_request:" + digest(prompt)
        saved = self.store.get(key)
        if saved is None:
            seen = [a["hypothesis"]["id"] for a in self.work.values() if a["task_id"] == self.active_task]
            request = prompt
            validation_errors = []
            for attempt in range(3):
                args = ("coordinator", json.dumps(request), SYSTEM)
                reply = (self.call_outside_review(self.client.call, *args) if overlap_review
                         else self.client.call(*args))
                try:
                    saved = validate_plan(reply, task_id=self.active_task, known_evidence=set(request["known_evidence"]),
                        parent_ids=[p["id"] for p in [*archive, *historical]], repair_ids=repair_ids, seen=seen,
                        max_writers=self.limits["writers"], allow_wait=bool(inflight),
                        archive_refinement_inflight=refinement_inflight, require_archive_trial=require_trial,
                        model_costs=prompt.get("model_costs", []), fixture_count=len(task.get("cases", [])),
                        merge_pairs=pairs, require_merge_trial=require_merge, portfolio=portfolio)
                    if revisit_parent and saved["action"] != "wait" and not any(
                            h["parent_id"] == revisit_parent for h in saved["hypotheses"]):
                        raise ValueError(f"The revisit requires a new refinement of {revisit_parent}.")
                    if saved["action"] == "move_on":
                        from .trial_evidence import stopping_evidence
                        if visit and visit_measured.keys().isdisjoint(reply["evidence"]):
                            raise ValueError("Propose a distinct experiment or cite a measured candidate from this revisit.")
                        saved.update(stopping_evidence(reply, measured, assignments, self.active_task))
                    break
                except (ValueError, TypeError) as error:
                    rejected = save_feedback(self.directory, reply, name="coordinator-rejected-plan")["artifact"]
                    registry.add(rejected["path"], "artifact")
                    self.event("coordinator_plan_rejected", task_id=self.active_task, reason=str(error),
                               attempt=attempt + 1, evidence=rejected)
                    reason = str(error)
                    if reason in validation_errors or attempt == 2:
                        raise
                    validation_errors.append(reason)
                    request = fit_request({**full_prompt, "invalid_plan": reply, "invalid_plan_evidence": rejected,
                        "validation_error": reason, "validation_errors": list(validation_errors),
                        "retry_instruction": "Choose a permitted action and resolve every listed constraint. Return the complete plan."},
                        "coordinator-retry-context")
            plans = self.store.get("swarm_plans", [])
            saved.update(id=len(plans), task_id=self.active_task, utc=time.time(),
                         lesson_ids=[l.get("id", l.get("lesson_id")) for l in lessons])
            if focus:
                saved["focus_id"] = focus["id"]
                for hypothesis in saved["hypotheses"]:
                    hypothesis["focus_id"] = focus["id"]
            plans.append(saved)
            self.store.set_many({key: saved, "swarm_plans": plans})
            write_json(self.directory / "plans" / f"plan-{saved['id']:04d}.json", saved)
        self.event("coordinator_decision", task_id=self.active_task, plan_id=saved["id"],
                   action=saved["action"], reason=saved["reason"], evidence=saved["evidence"],
                   archive_decision=saved.get("archive_decision"),
                   hypothesis_count=len(saved["hypotheses"]))
        epochs = self.store.get("swarm_planning_epochs", {})
        epochs[self.active_task] = epoch
        self.store.set_many({"swarm_planning_epochs": epochs, "replan_after:" + self.active_task:
            "all_results" if saved["action"] == "wait" else saved.get("replan_after", "all_results")})
        return saved

    def assignment(self, hypothesis, slot, plan):
        proposal = hypothesis
        record = self.hypotheses.get(hypothesis["id"])
        if record is None and hypothesis.get("parent_id") in self.pending:
            record = self.hypotheses.for_candidate(hypothesis["parent_id"], self.active_task)
        if record is None:
            record = self.hypotheses.register(self.active_task, hypothesis)
        elif record["task_id"] != self.active_task:
            raise ValueError("The hypothesis belongs to another operation.")
        if record["status"] not in {"ready", "correction_pending"}:
            raise ValueError("This hypothesis is closed or already has an active attempt.")
        hypothesis = {**copy.deepcopy(record["hypothesis"]), "specialty": proposal["specialty"]}
        parent = hypothesis["parent_id"]
        previous = None
        if record["attempts"]:
            previous_id = record["attempts"][-1]["assignment_id"]
            previous = next((a for a in self.work.values() if a["id"] == previous_id), None)
            parent = record.get("last_candidate_id") or parent
        repair = self.pending.get(parent)
        if record["attempts"] and repair is None:
            repair = {"candidate_id": parent, "task_id": self.active_task,
                      "stage": (record.get("last_outcome") or {}).get("stage", "writer"),
                      "repairs_used": record["corrections_used"],
                      "result": compact_feedback(record.get("last_outcome")), "slot": slot}
        approved_parent = None
        if parent:
            approved_parent = self.archive.parent(parent, self.by_id[self.active_task], self.hardware, repair=repair is not None)
        merge_parents = []
        if hypothesis.get("merge") and repair is None:
            from .recombination import merge_pairs, validate_merge
            task = self.by_id[self.active_task]
            active = self.archive.refresh(task, self.hardware)
            validate_merge(hypothesis["merge"], parent,
                           merge_pairs(task, active, self.store.candidates(self.active_task)))
            for identity in hypothesis["merge"]["parent_ids"]:
                row = self.archive.parent(identity, task, self.hardware)
                summary = next(p for p in active if p["id"] == identity)
                merge_parents.append({"id": identity, "source": copy.deepcopy(row["source"]),
                                      "measurements": self.operation_feedback(row["result"]),
                                      "design": summary.get("hypothesis"), "model_status": summary.get("model_status")})
        number = self.next_id
        attempt_index = len(record["attempts"])
        self.next_id += 1
        lesson_ids = [l.get("id", l.get("lesson_id")) for l in self.recall()]
        reason = "repair" if repair else "archive_merge" if merge_parents else "shared_archive" if parent else "fresh_start"
        provided_ids = [p["id"] for p in merge_parents] if merge_parents else [parent] if parent else []
        a = {"id": f"experiment-{number:05d}", "generation": number, "task_id": self.active_task,
             "slot": slot, "island": hypothesis["specialty"], "hypothesis": hypothesis,
             "hypothesis_id": record["id"], "proposal_id": proposal["id"],
             "attempt_number": attempt_index + 1, "correction_attempt": attempt_index,
             "previous_attempt": {"error": previous.get("error"),
                 "rejected_response": previous.get("rejected_response"),
                 "evidence": previous.get("failure_evidence"), "tested_change": previous.get("observations")} if previous else None,
             "parent_id": parent, "repair": copy.deepcopy(repair), "example_ids": [],
             "approved_parent_source": approved_parent["source"] if approved_parent else None,
             "merge_parents": merge_parents,
             "repairs_used": attempt_index,
             "status": "planned", "plan_id": plan.get("id"), "lesson_ids": lesson_ids,
             "selection": {"reason": reason, "round": number + 1, "generation": number, "slot": slot,
                 "receiving_island": hypothesis["specialty"], "task_id": self.active_task,
                 "candidate_id": parent, "parent_ids": provided_ids,
                 "provided_parent_ids": provided_ids, "source_island": self.row(parent)["island"] if parent else None,
                 "hypothesis": hypothesis, "lesson_ids": lesson_ids, "migration_event": None}}
        # Save the complete assignment and the next ID atomically before its
        # attempt starts. Resume can begin a saved assignment after any crash.
        with self.state_lock:
            self.work[str(number)] = a
            self.store.set_many({"swarm_assignments": self.work, "swarm_next_assignment": self.next_id})
        self.hypotheses.begin(record["id"], a["id"], {"slot": slot, "specialty": hypothesis["specialty"]})
        return a

    def write_swarm(self, a):
        if a.get("candidate"):
            return a["candidate"]
        if a.get("prompt"):
            # Resume the exact saved request. New lessons belong to the next experiment.
            return self.request_writer(a, a["prompt"])
        task = self.by_id[a["task_id"]]
        lessons = self.lessons.retrieve(task, self.hardware, limit=self.limits.get("lesson_limit", 8),
            max_chars=self.limits.get("lesson_context_chars", 12000))
        parent = self.row(a["parent_id"]) if a["parent_id"] else None
        if parent:
            if parent["task"] != task["id"] or parent["source"] != a.get("approved_parent_source"):
                raise ValueError("The approved parent source or task changed.")
            if parent["id"] in self.store.get("blocked_model_candidates", {}) and not a["repair"]:
                raise ValueError("The approved parent now requires an explicit model repair.")
        for supplied in a.get("merge_parents", []):
            row = self.row(supplied["id"])
            if row["task"] != task["id"] or row["source"] != supplied["source"]:
                raise ValueError("An approved merge parent source or task changed.")
            if row["id"] in self.store.get("blocked_model_candidates", {}):
                raise ValueError("A merge parent now requires an explicit model repair.")
        prompt = {"task": self.task_prompts[task["id"]], "hardware": self.hardware,
                  "validation_policy": self.validation_policy,
                  "automatic_tuning": self.config.get("tuning", {"enabled": False}),
                  "model_costs": self.model_costs(task["id"]),
                  "previous_model_trace": self.store.get("last_performance_trace"),
                  "kept_kernel": self.kept_summary(task),
                  "writer": a["slot"], "island": a["island"], "generation": a["generation"],
                  "hypothesis": a["hypothesis"], "island_strategy": SPECIALTIES[a["island"]],
                  "attempt_number": a["attempt_number"], "correction_attempt": a["correction_attempt"],
                  "max_correction_attempts": 2, "previous_attempt": compact_feedback(a.get("previous_attempt")),
                  "design_authority": {"owner": "writer", "objective": a["hypothesis"].get("objective") or a["hypothesis"].get("claim", "Improve the measured operation."),
                      "instruction": "Choose the algorithm and launch settings from the task and measured evidence. "
                          "You may replace coordinator suggestions. Explain your design and reason. "
                          "Preserve the operation, precision, inputs, limits and supplied parent identity. "
                          "Return candidate and design in one JSON object."},
                  "writer_mode": "Repair the exact supplied failure using your own design." if a["repair"] else
                      "Refine the supplied archive parent. Choose the method and settings from its measured results." if parent else
                      "Design an implementation for the measured objective.",
                  "parent_selection": a["selection"], "parent": parent["source"] if parent else None,
                  "merge_parents": copy.deepcopy(a.get("merge_parents", [])),
                  "parent_measurements": self.operation_feedback(parent["result"]) if parent else None,
                  "repair": self.repair_feedback(a["repair"]), "lessons": lessons,
                  "source_rejections": [{"reason": e.get("reason"), "assignment_id": e.get("assignment_id")}
                      for e in self.events if e.get("kind") == "writer_rejected" and e.get("task_id") == a["task_id"]][-3:],
                  "model_diagnostic": self.diagnostic_feedback(a["task_id"]),
                  }
        if self.limits.get("design_portfolio"):
            prompt["experiment_role"] = a["hypothesis"].get("experiment_role")
            prompt["tuning_instruction"] = ("Use the parent's measured tuning choices. Put small legal block and "
                "launch variations in triton.autotune. Make API experiments change the algorithm, data access, "
                "work distribution or numerical behavior. Matching tuning results can be reused.")
        if (a.get("repair") or {}).get("decode_context"):
            prompt["model_failure"] = copy.deepcopy(a["repair"]["decode_context"])
        focus_id = a["hypothesis"].get("focus_id")
        if focus_id:
            from .staged_evaluation import focus_measurements
            division = self.store.get("operation_split:" + task["id"])
            focus = next(s for s in division["scopes"] if s["id"] == focus_id)
            prompt["operation_search"] = {"focus_id": focus_id, "example_paths": focus["paths"][:6],
                "parent_performance": focus_measurements(parent["result"] if parent else {}, focus["paths"]),
                "objective": "Optimize this measured subset. Preserve the shared operation contract."}
        if a.get("merge_parents"):
            # Both complete sources already appear in merge_parents.
            prompt["parent"] = None
            prompt["parent_measurements"] = None
            prompt["writer_mode"] = ("Combine compatible ideas from both supplied candidates for the shared shapes. "
                "Choose the algorithm and settings. Explain each parent's contribution and any conflict. "
                "Write one new Triton implementation. Do not return either parent unchanged or route "
                "between the two parents and call that a merged improvement.")
            prompt["merge_shapes"] = a["hypothesis"]["merge"]["shape_ids"]
        if a.get("repair"):
            from .hypotheses import classify_fault
            failure = a["repair"].get("result", {})
            fault = classify_fault(a["repair"].get("stage", "operation"), failure)
            if fault["kind"] == "correctness_failure":
                prompt["writer_mode"] = "Correct the saved numerical failure before further performance tuning."
                prompt["repair_objective"] = {
                    "priority": "numerical_correctness", "preserve_hypothesis_identity": True,
                    "instruction": "Explain the numerical cause and the code change that addresses it. "
                        "Use the exact failed boundary, inputs, and numerical limits. Do not merely repeat the "
                        "old performance hypothesis. A tile change is useful only if it changes the failing "
                    "numerical behavior. Trusted failing cases must pass before a kernel can be kept. "
                        "The full model test still decides any model speed claim.",
                    "failed_cases": [compact_feedback(case) for case in failure.get("cases", [])
                        if case.get("correct") is False and case.get("phase") == "supplemental_correctness"][:3]}
        for field in ("lessons",):
            while len(json.dumps(prompt)) > self.limits["context_chars"] and prompt[field]:
                prompt[field].pop()
        if len(json.dumps(prompt)) > self.limits["context_chars"]:
            raise ValueError("The writer context exceeds its character limit.")
        return self.request_writer(a, prompt)

    def request_writer(self, a, prompt):
        parent = self.row(a["parent_id"]) if a["parent_id"] else None
        for supplied in a.get("merge_parents", []):
            row = self.row(supplied["id"])
            if row["task"] != a["task_id"] or row["source"] != supplied["source"]:
                raise ValueError("An approved merge parent changed before writer dispatch.")
            if row["id"] in self.store.get("blocked_model_candidates", {}):
                raise ValueError("A saved merge parent now requires a model repair.")
        self.save_work(a, status="running", prompt=prompt)
        self.event("writer_started", assignment_id=a["id"], task_id=a["task_id"], slot=a["slot"],
                   hypothesis_id=a["hypothesis"]["id"], parent_id=a["parent_id"],
                   merge_parent_ids=[p["id"] for p in a.get("merge_parents", [])])
        self.event("lesson_used", assignment_id=a["id"], task_id=a["task_id"], slot=a["slot"],
                   lesson_ids=[l.get("id", l.get("lesson_id")) for l in prompt["lessons"]])
        try:
            response = self.client.call(f"writer_{a['slot']}_{a['island']}", json.dumps(prompt), self.system)
            # Preserve both the writer design and source before validation.
            response_report = self.evidence_report(response, "writer-response-" + a["id"], assignment_id=a["id"])
            self.save_work(a, rejected_response=compact_feedback(response), response_evidence=response_report["artifact"])
            candidate = response
            if isinstance(response, dict) and set(response) == {"candidate", "design"}:
                from .trial_evidence import validate_writer_design
                design = validate_writer_design(response["design"], len(self.by_id[a["task_id"]].get("cases", [])),
                    merge_parent_ids=[p["id"] for p in a.get("merge_parents", [])])
                self.save_work(a, writer_design=design, selection={**a["selection"], "writer_design": design})
                candidate = response["candidate"]
            elif a.get("merge_parents"):
                raise ValueError("A merge must return candidate and design, including parent_contributions for both parents.")
            if isinstance(candidate, dict) and isinstance(candidate.get("strategy"), str) and len(candidate["strategy"]) > 4000:
                self.evidence_report(candidate, "long-strategy-" + a["id"], assignment_id=a["id"])
                candidate = {**candidate, "strategy": candidate["strategy"][:1900] +
                    "\n[Full explanation retained in the raw response.]\n" + candidate["strategy"][-1900:]}
            validate_candidate(candidate, self.config.get("kernel_backend", "triton"))
            if candidate["task_id"] != a["task_id"]:
                raise ValueError("The writer changed the operation.")
            if candidate_id(candidate) in {p["id"] for p in a.get("merge_parents", [])} | ({parent["id"]} if parent else set()):
                raise ValueError("The writer returned unchanged parent code.")
            if a.get("merge_parents"):
                from .recombination import implementation
                if any(implementation(candidate["source"]) == implementation(p["source"]["source"])
                       for p in a["merge_parents"]):
                    raise ValueError("A merge must change the implementation; comments and formatting do not count.")
            from .automatic_tuning import prepare as prepare_tuning
            candidate, tuning = prepare_tuning(candidate, self.config)
            validate_candidate(candidate, self.config.get("kernel_backend", "triton"))
            self.save_work(a, status="generated", candidate=candidate, automatic_tuning=tuning)
            return candidate
        except Exception as error:
            from .agent import AuthenticationUnavailable
            report = self.evidence_report({"error": str(error), "traceback": traceback.format_exc(),
                "response_evidence": a.get("response_evidence")}, "writer-error-" + a["id"], assignment_id=a["id"])
            self.save_work(a, status="failed", error=f"{type(error).__name__}: {error}",
                failure_kind="authentication_before_dispatch" if isinstance(error, AuthenticationUnavailable) else None,
                failure_evidence=report["artifact"])
            raise
        finally:
            self.event("writer_finished", assignment_id=a["id"], task_id=a["task_id"], slot=a["slot"],
                       hypothesis_id=a["hypothesis"]["id"], status=a["status"])

    def _process_candidate(self, a):
        candidate = a["candidate"]
        key, fresh = self.store.add(candidate, a["island"], a["generation"], a["parent_id"])
        self.store.record_emission(a["generation"], a["slot"], key, a["island"], not fresh, a["selection"])
        self.save_work(a, candidate_id=key)
        self.evaluate(a, candidate, a["generation"])
        row = self.row(key)
        split_key = "operation_split:" + a["task_id"]
        if self.by_id[a["task_id"]].get("separate_operations") and not self.store.get(split_key):
            from .staged_evaluation import measured_split
            split = measured_split(self.by_id[a["task_id"]], row["result"], self.config["max_p95_ratio"])
            if split:
                self.store.set(split_key, split)
                self.event("operation_search_split", task_id=a["task_id"], **split)
        if a["status"] != "evaluated":
            self.save_work(a, status="evaluated")
            self.event("candidate_evaluated", assignment_id=a["id"], task_id=a["task_id"], slot=a["slot"],
                       candidate_id=key, decision=row["result"].get("decision"), correct=row["correct"],
                       stages=row["result"].get("stages", {}))
        from .trial_evidence import trial_observations
        tested_hypothesis = {**a["hypothesis"], **a.get("writer_design", {})}
        if a.get("writer_design"):
            tested_hypothesis["design_owner"] = "writer"
        observations = a.get("observations") or trial_observations(tested_hypothesis, candidate, row["result"])
        if a.get("writer_design"):
            observations.update(design_owner="writer", coordinator_requested=a["hypothesis"].get("change"),
                                design_reason=a["writer_design"]["reason"])
        self.save_work(a, observations=observations)
        lesson = self.lessons.record(self.by_id[a["task_id"]], self.hardware, key,
            hypothesis=tested_hypothesis, scope="operation", trial_observations=observations)
        self.archive.refresh(self.by_id[a["task_id"]], self.hardware)
        measured = None
        kept = False
        if kernel_first(self.config):
            if self.config.get("generation_objective") == "decode":
                from .decode_search import record_candidate
                record_candidate(self.store, self.by_id[a["task_id"]], key, row["result"], self.config)
            kept = keep_kernel(self.store, self.by_id[a["task_id"]], key, self.config)
            if kept:
                self.event("kernel_kept", task_id=a["task_id"], candidate_id=key,
                    baseline=row["result"].get("timing_baseline"), decision=row["result"]["decision"],
                    shape_decisions=row["result"].get("shape_decisions"),
                    acceptance_scope=row["result"].get("acceptance_scope", "kernel_only"))
            if kept and self.config.get("generation_objective") != "decode" and (self.config.get("model_trial_base") == "accepted"
                         or (a.get("repair") or {}).get("stage") == "model"):
                # Validate promotions while the hypothesis can still use its correction attempts.
                # A local timing win can fail after scores accumulate across the model.
                if self.config.get("model_trial_base") == "accepted":
                    # Restrict before resolving parent/child alternatives. A
                    # retained parent must not hide the repaired child source.
                    choices = dict(self.store.get("kernel_region_choices", {}))
                    for region in self.catalog.get("hierarchy", {}).get("regions", []):
                        if a["task_id"] in region["subsection_ids"]:
                            choices[region["id"]] = [a["task_id"]]
                    repair_manifest = kernel_manifest(self.store, self.catalog,
                        exclude_tasks=[task["id"] for task in self.catalog["tasks"] if task["id"] != a["task_id"]],
                        region_choices=choices)
                else:
                    repair_manifest = kernel_manifest(self.store, self.catalog)
                from .kernel_selection import model_trial_manifest
                base = self.store.get("best_model_measurement")
                if self.config.get("model_trial_base") == "accepted":
                    repair_manifest = model_trial_manifest(self.store, self.catalog, repair_manifest)
                incumbent = base["manifest"] if base and repair_manifest != base["manifest"] else None
                measured = self._measure_manifest(repair_manifest, a["generation"],
                    **({"incumbent": incumbent} if incumbent else {}))
                self.store.set("last_model_measurement", measured)
                from .integration_repairs import safe_measurement
                if measured["result"].get("decision", {}).get("accepted") and safe_measurement(self.store, measured):
                    self.store.set("best_model_measurement", measured)
                self.lessons.record(self.by_id[a["task_id"]], self.hardware, key,
                    hypothesis=tested_hypothesis, scope="model", result=measured["result"])
                self.event("model_repair_checked" if a.get("repair") else "model_checked", assignment_id=a["id"], task_id=a["task_id"], candidate_id=key,
                           decision=measured["result"].get("decision"), manifest=measured["manifest"])
        if row["accepted"] and not kernel_first(self.config):
            # Always test this exact new candidate, not a higher scoring older sibling.
            original = self.manifest_factory
            def exact_manifest(store, catalog):
                best = store.get("best_model_measurement")
                entries = [dict(e) for e in (best or {}).get("manifest", {}).get("entries", []) if e["task_id"] != a["task_id"]]
                entries.append({"task_id": a["task_id"], "candidate_id": key,
                                "paths": self.by_id[a["task_id"]]["paths"],
                                "priority": self.by_id[a["task_id"]]["priority_ms"]})
                return {"entries": entries}
            self.manifest_factory = exact_manifest
            try:
                measured = self.measure_model(a["generation"], [a])
            finally:
                self.manifest_factory = original
            self.lessons.record(self.by_id[a["task_id"]], self.hardware, key,
                hypothesis=tested_hypothesis, scope="model", result=measured["result"])
            self.event("model_checked", assignment_id=a["id"], task_id=a["task_id"], candidate_id=key,
                       decision=measured["result"].get("decision"), manifest=measured["manifest"])
            if not operation_correct(measured["result"]):
                self.diagnose_model(measured)
        old_best = self.progress["best_operation_score"]
        from .swarm_policy import measured_progress
        signal = measured_progress([row], old_best)
        if kernel_first(self.config):
            # Progress guides exploration. Acceptance still uses every speed and
            # accuracy gate. Do not count a useful, but not yet winning, step as
            # stagnation. Replayed measurements cannot create new progress.
            signal["improved"] = row["generation"] == a["generation"] and (kept or signal["improved"])
        final_result = measured["result"] if measured else row["result"]
        if not a.get("review"):
            counted = measured_gpu_outcome(final_result)
            if observations["status"] == "contradicted":
                counted = False
            if kernel_first(self.config):
                counted = counted and row["generation"] == a["generation"]
            progress = {**self.progress, "round": self.progress["round"] + 1,
                "best_operation_score": max(old_best, signal["score"]),
                "stagnant": (0 if signal["improved"] else self.progress["stagnant"] + 1)
                            if counted else self.progress["stagnant"]}
            self.save_work(a, review={"progress": progress, "improved": bool(signal["improved"]),
                "task_count": self.task_counts.get(a["task_id"], 0) + 1,
                "stage": "model" if measured else "operation"})
        review = a["review"]
        record = self.hypotheses.finish(a["id"], stage=review["stage"], result=final_result,
            candidate_id=key, improved=review["improved"], evidence=[key], observations=observations)
        if record["status"] == "paused":
            self.save_work(a, status="evaluated", review=None)
            reason = record.get("last_interruption", {}).get("reason", "The worker needs infrastructure recovery.")
            raise TransferError(reason)
        self.hypothesis_event(record, a)
        if measured and kernel_first(self.config) and not operation_correct(measured["result"]):
            self.integration_repairs(measured)
        elif measured and operation_correct(measured["result"]):
            tickets = self.store.get("integration_repair_tickets", {})
            for ticket in tickets.values():
                if ticket.get("hypothesis_id") == record["id"] and ticket["status"] == "correction_pending":
                    ticket.update(status="corrected", corrected_candidate_id=key)
            self.store.set("integration_repair_tickets", tickets)
        self.progress = review["progress"]
        self.task_counts[a["task_id"]] = review["task_count"]
        with self.state_lock:
            a.update(status="completed", operation_accepted=bool(row["accepted"]),
                     model_accepted=bool(measured and measured["result"].get("decision", {}).get("accepted")))
            self.work[str(a["generation"])] = a
            self.store.set_many({"swarm_assignments": self.work, "task_progress": self.progress,
                "swarm_task_counts": self.task_counts, "generation": self.next_id, f"reviewed:{a['generation']}": True})
        return lesson

    def hypothesis_event(self, record, assignment):
        outcome = record.get("last_outcome") or {}
        if record["status"] in {"succeeded", "exhausted", "closed"}:
            candidate_ids = {(a.get("outcome") or {}).get("candidate_id") for a in record["attempts"]}
            self.pending = {k: v for k, v in self.pending.items() if k not in candidate_ids}
            self.persist_repairs()
        self.event("hypothesis_outcome", task_id=record["task_id"], assignment_id=assignment["id"],
            hypothesis_id=record["id"], status=record["status"], attempt_number=assignment["attempt_number"],
            corrections_remaining=record["corrections_remaining"], outcome=outcome,
            close_reason=record.get("close_reason"))

    def queue_corrections(self):
        """Give unfinished hypotheses two bounded corrections before new work."""
        ready = self.hypotheses.outstanding(self.active_task)
        ids = {record["id"] for record in ready}
        fresh = [q for q in self.plan_queue if q["hypothesis"]["id"] not in ids
                 and (self.hypotheses.get(q["hypothesis"]["id"]) or {}).get("status", "ready") == "ready"]
        self.plan_queue = [{"hypothesis": r["hypothesis"], "plan_id": None} for r in ready] + fresh
        self.store.set("swarm_queue", self.plan_queue)

    def reject_writer(self, a, error=None):
        """Save the failure before closing its attempt, so resume never repeats it."""
        fault = a.get("writer_fault")
        if fault is None:
            message = f"{type(error).__name__}: {error}" if error is not None else a.get("error", "Writer failed.")
            if isinstance(error, BudgetExhausted) or message.startswith(("BudgetExhausted:", "TaskBudgetExhausted:")):
                self.save_work(a, status="planned", error=message)
                self.hypotheses.pause(a["id"], message, stage="budget")
                self.event("writer_paused", task_id=a["task_id"], assignment_id=a["id"], reason=message[:2000])
                return
            stage = "source" if isinstance(error, (ValueError, TypeError)) or message.startswith(
                ("ValueError:", "TypeError:", "JSONDecodeError:")) else "api"
            report = self.evidence_report({"error": message,
                "traceback": traceback.format_exc() if error is not None else None,
                "response_evidence": a.get("response_evidence")}, "writer-failure-" + a["id"])
            fault = {"stage": stage, "error": message, "evidence": [report["artifact"]]}
            self.save_work(a, status="failed", error=message, writer_fault=fault, failure_evidence=report["artifact"])
        record = self.hypotheses.finish(a["id"], **fault)
        self.event("writer_rejected", assignment_id=a["id"], task_id=a["task_id"], reason=fault["error"][:2000])
        self.hypothesis_event(record, a)

    def stop_record(self, code, reason, *, source="controller", **extra):
        record = {"code": code, "reason": reason, "decision_source": source,
                  "task_id": self.active_task, "remaining_tokens": self.limits["max_total_tokens"] - self.store.usage()["tokens"],
                  "utc": time.time(), **extra}
        self.store.set("swarm_stop", record)
        self.event("stop_decision", **record)
        return record

    def validate_decode_target(self, reason):
        from .decode_validation import target_boundary
        visit_record = self.progress.get("budget_visits", {}).get(self.active_task)
        if visit_record and not any(a["task_id"] == self.active_task and
                                    a["generation"] >= visit_record["first_generation"] and a.get("candidate_id")
                                    for a in self.work.values()):
            self.event("target_model_selection_skipped", task_id=self.active_task,
                       reason="This visit produced no new kernel measurement.")
            return
        visit = self.progress.get("budget_visits", {}).get(self.active_task, {}).get("count", 0)
        summary = target_boundary(self.config, self.directory, self.profile, self.catalog, self.store, self.worker,
                                  self.active_task, visit, reason)
        self.event("target_model_selection", task_id=self.active_task, winner=summary["winner"],
                   trials=[{k: trial[k] for k in ("candidate_id", "accepted", "previous_ratio", "original_ratio")}
                           for trial in summary["trials"]])
        if self.limits.get("model_repair_fraction", 0.) and not (self.directory / "STOP").exists():
            from .decode_repairs import repair_target
            repair_target(self, summary, visit, reason)

    def finish_task(self, reason, source="controller", evidence=None):
        from .hierarchy import enabled, region_for
        region = region_for(self.catalog, self.active_task) if enabled(self.config) else None
        if reason == "task_coverage_reserve":
            for a in self.work.values():
                if a["task_id"] == self.active_task and a["status"] in {"planned", "budget_deferred"} and not a.get("candidate"):
                    attempt = self.hypotheses.attempt(a["id"])
                    if attempt and attempt["status"] in {"paused", "budget_deferred"}:
                        self.hypotheses.defer_for_task_budget(a["id"], "The next request exceeded this task's allocation.")
                    self.save_work(a, status="budget_deferred", error="The next request exceeded this task's allocation.")
        if self.config.get("generation_objective") == "decode":
            self.validate_decode_target(reason)
        for record in self.hypotheses.all(self.active_task):
            if record["status"] not in {"succeeded", "exhausted", "closed"}:
                self.hypotheses.close(record["id"], reason[:2000], evidence=evidence or [])
        self.event("task_finished", task_id=self.active_task, reason=reason,
                   decision_source=source, evidence=evidence or [])
        tickets = self.store.get("integration_repair_tickets", {})
        for ticket in tickets.values():
            if (ticket["task_id"] == self.active_task and ticket["status"] == "correction_pending"
                    and (self.hypotheses.get(ticket["hypothesis_id"]) or {}).get("status") == "closed"):
                ticket.update(status="deferred", closed_reason=reason)
        self.store.set("integration_repair_tickets", tickets)
        self.plan_queue = []
        self.store.set("swarm_queue", [])
        self.advance(reason, self.next_id)
        if self.config.get("generation_objective") == "decode":
            self.store.set("decode_target_boundary", None)
        if region and all(key in {d["task_id"] for d in self.progress["completed"]} for key in region["subsection_ids"]):
            self.finish_region(region)

    def run(self):
        if self.store.get("status") in {"accepted", "completed_no_gain", "completed_kernel_search", "retained_previous", "retained_original"}:
            return read_json(self.directory / "result.json")
        status, terminal_error = "completed_no_gain", None
        self.store.set_many({"status": "running", "budget_reserve": None})
        def recover_auth_failures():
            # A sibling future can fail after another transfer stops the loop.
            # Recover its saved typed failure before classifying writer errors.
            for assignment in self.work.values():
                if (assignment["status"] != "failed"
                        or assignment.get("failure_kind") != "authentication_before_dispatch"):
                    continue
                attempt = self.hypotheses.attempt(assignment["id"])
                if attempt is not None and attempt["outcome_hash"] is None:
                    self.save_work(assignment, status="planned")
                    self.hypotheses.pause(assignment["id"], assignment["error"], stage="infrastructure")
        try:
            recover_auth_failures()
            self.resume_integration_repairs()
            devices = [None]
            if self.config.get("generation_objective") == "decode" and self.config.get("worker", {}).get("archive_gpu") is not None:
                from .decode_devices import decode_devices
                devices = decode_devices(self.worker, self.directory, self.profile, self.catalog)
            with (ThreadPoolExecutor(max_workers=self.limits["writers"]) as pool,
                  ThreadPoolExecutor(max_workers=len(devices), thread_name_prefix="gpu-review") as gpu):
                while True:
                    if (self.directory / "STOP").exists():
                        status = "stopped"
                        self.stop_record(status, "The user STOP file is present.")
                        break
                    if self.active_task is None:
                        self.start_task()  # Finish any region interrupted at the end of the pass.
                        if self.active_task is None and not self.revisit_with_budget():
                            break
                    self.start_task()
                    if self.active_task is None:
                        continue
                    boundary = self.store.get("decode_target_boundary") if self.decode_search else None
                    current_visit = self.progress.get("budget_visits", {}).get(self.active_task, {}).get("count", 0)
                    if boundary and boundary["task"] == self.active_task and boundary["visit"] == current_visit:
                        self.finish_task(boundary["reason"])
                        continue
                    from .hierarchy import enabled
                    updated = self.store.get("hierarchy_latest_profile", {}).get("task_priorities", {}).get(self.active_task)
                    if enabled(self.config) and updated and (updated.get("excluded_reason") or updated["priority_ms"] <= 0):
                        self.finish_task("The updated compiled profile cannot verify this subsection: " +
                                         str(updated.get("excluded_reason")), "controller")
                        continue
                    futures, closing = {}, None
                    pending_gpu = deque()
                    reviews = {}
                    for a in self.work.values():
                        attempt = self.hypotheses.attempt(a["id"])
                        if a["task_id"] == self.active_task and attempt is None:
                            attempt = self.hypotheses.begin(a["hypothesis_id"], a["id"],
                                {"slot": a["slot"], "specialty": a["island"]})
                        if a["task_id"] == self.active_task and a["status"] == "failed" and attempt and not attempt["outcome_hash"]:
                            self.reject_writer(a)
                    saved = [a for a in self.work.values() if a["task_id"] == self.active_task
                             and a["status"] not in {"completed", "failed", "budget_deferred"}]
                    # Finish a partially reviewed candidate before selecting another baseline.
                    for a in sorted(saved, key=lambda a: (not bool(a.get("candidate_id")), a["generation"])):
                        if a.get("candidate"):
                            pending_gpu.append(a)
                        else:
                            futures[pool.submit(self.write_swarm, a)] = a
                    stop_file = self.directory / "STOP"
                    while True:
                        busy = {slot for _, slot in reviews.values()}
                        in_review = {candidate_id(a["candidate"]) for a, _ in reviews.values()}
                        for slot, device in enumerate(devices):
                            if slot in busy:
                                continue
                            a = next((item for item in pending_gpu if candidate_id(item["candidate"]) not in in_review and
                                      (not device or item.get("evaluation_gpu") in (None, device.gpu))), None)
                            if a is None:
                                break
                            pending_gpu.remove(a)
                            in_review.add(candidate_id(a["candidate"]))
                            if device:
                                # Reconnect to the assigned GPU for a pending durable job.
                                self.save_work(a, evaluation_gpu=device.gpu)
                            future = gpu.submit(self.process_candidate, a, device) if device else gpu.submit(self.process_candidate, a)
                            reviews[future] = (a, slot)
                        outstanding = bool(futures or reviews or pending_gpu)
                        if stop_file.exists():
                            closing = ("stopped", "controller", [])
                        cap = self.limits.get("max_task_experiments")
                        attempted = sum(a["task_id"] == self.active_task for a in self.work.values())
                        if not closing and cap is not None and attempted >= cap:
                            closing = ("task_experiment_limit", "controller", [])
                        costs = self.costs()
                        # Include estimated calls that have not yet reached the API reservation.
                        usage = self.store.budget_snapshot(self.limits)
                        unreserved = max(0, len(futures) * costs["writer"] - usage.get("inflight_tokens", 0))
                        admission = admit_experiments(self.limits, usage["tokens"] + unreserved, costs,
                            self.limits["writers"] - len(futures), task_remaining=self.remaining_task_tokens() - unreserved)
                        self.store.set("budget_decision", {**admission, "task_id": self.active_task, "cost_estimates": costs})
                        self.store.set("budget_reserve", {"protected_tokens": costs["coordinator"], "allowed_roles": ["coordinator"]})
                        if not outstanding and not admission["admit"] and not closing:
                            closing = (admission["reason"], "controller", [])
                        new_evidence = self.planning_epoch() != self.store.get("swarm_planning_epochs", {}).get(self.active_task)
                        if not closing and admission["admit"] and (self.plan_queue or not outstanding or new_evidence):
                            with self.review_lock:
                                self.queue_corrections()
                            may_replan = not outstanding or (new_evidence and
                                self.store.get("replan_after:" + self.active_task) == "next_result")
                            ready = any(f.done() for f in (*futures, *reviews))
                            if not self.plan_queue and may_replan and not ready and not stop_file.exists():
                                try:
                                    with self.review_lock:
                                        plan = self.coordinator(admission["writer_slots"], overlap_review=True)
                                except TaskBudgetExhausted:
                                    closing = ("task_coverage_reserve", "controller", [])
                                    plan = {"action": "wait", "hypotheses": [], "id": None}
                                except (ValueError, TypeError) as error:
                                    if not outstanding:
                                        raise
                                    # Do not discard in-flight results because a planning reply is invalid.
                                    self.event("coordinator_deferred", task_id=self.active_task, reason=str(error))
                                    epochs = self.store.get("swarm_planning_epochs", {})
                                    epochs[self.active_task] = self.planning_epoch()
                                    self.store.set("swarm_planning_epochs", epochs)
                                    plan = {"action": "wait", "hypotheses": [], "id": None}
                                if plan["action"] == "move_on" and plan.get("focus_id") and self.search_focus():
                                    self.plan_queue = []
                                    self.store.set("swarm_queue", [])
                                elif plan["action"] == "move_on":
                                    closing = (plan["reason"], "coordinator", plan["evidence"])
                                else:
                                    self.plan_queue = [{"hypothesis": h, "plan_id": plan["id"]} for h in plan["hypotheses"]]
                                    self.store.set("swarm_queue", self.plan_queue)
                            if stop_file.exists():
                                closing = ("stopped", "controller", [])
                            if not closing:
                                # Recompute after the planning request consumed tokens.
                                costs = self.costs()
                                usage = self.store.budget_snapshot(self.limits)
                                unreserved = max(0, len(futures) * costs["writer"] - usage.get("inflight_tokens", 0))
                                admission = admit_experiments(self.limits, usage["tokens"] + unreserved, costs,
                                    self.limits["writers"] - len(futures), task_remaining=self.remaining_task_tokens() - unreserved)
                                free = [s for s in range(self.limits["writers"]) if s not in {a["slot"] for a in futures.values()}]
                                count = admission["writer_slots"]
                                if cap is not None:
                                    count = min(count, max(0, cap - attempted))
                                for slot in free[:count]:
                                    if not self.plan_queue:
                                        break
                                    queued = self.plan_queue[0]
                                    try:
                                        with self.review_lock:
                                            if stop_file.exists():
                                                closing = ("stopped", "controller", [])
                                                break
                                            if any(f.done() and f.exception() is not None for f in reviews):
                                                break
                                            a = self.assignment(queued["hypothesis"], slot, {"id": queued["plan_id"]})
                                    except ValueError as error:
                                        self.event("hypothesis_rejected", task_id=self.active_task,
                                                   hypothesis_id=queued["hypothesis"]["id"], reason=str(error))
                                    else:
                                        futures[pool.submit(self.write_swarm, a)] = a
                                    self.plan_queue.pop(0)
                                    self.store.set("swarm_queue", self.plan_queue)
                        if not futures and not reviews:
                            if pending_gpu:
                                continue
                            if closing:
                                reason, source, evidence = closing
                                if reason == "stopped":
                                    status = "stopped"
                                    self.stop_record(status, "The user STOP file is present.")
                                elif reason == "token_budget":
                                    status = "budget_exhausted"
                                    self.stop_record(status, "The next experiment and review do not fit the token cap.",
                                                     required_tokens=admission["required_tokens"])
                                else:
                                    self.finish_task(reason, source, evidence)
                                break
                            continue
                        waiting = [*reviews, *futures]
                        ready, _ = wait(waiting, timeout=1, return_when=FIRST_COMPLETED)
                        # Handle a GPU failure before scheduling a sibling. Break ties
                        # between simultaneous writer completions in submission order.
                        for future in sorted(ready, key=lambda f: (0, reviews[f][0]["generation"]) if f in reviews
                                             else (1, futures[f]["generation"])):
                            is_review = future in reviews
                            if is_review:
                                a, _ = reviews.pop(future)
                            else:
                                a = futures.pop(future)
                            try:
                                future.result()
                                if not is_review:
                                    pending_gpu.append(a)
                            except BudgetExhausted as error:
                                self.save_work(a, status="planned", error=str(error))
                                self.hypotheses.pause(a["id"], str(error), stage="budget")
                                self.event("writer_rejected", assignment_id=a["id"], task_id=a["task_id"], reason=str(error))
                                closing = ("task_coverage_reserve" if isinstance(error, TaskBudgetExhausted)
                                           else "token_budget", "controller", [])
                            except TransferError as error:
                                from .agent import AuthenticationUnavailable
                                if isinstance(error, AuthenticationUnavailable):
                                    # No generation request was sent. Resume this
                                    # assignment without using a correction attempt.
                                    self.save_work(a, status="planned", error=str(error))
                                    self.hypotheses.pause(a["id"], str(error), stage="infrastructure")
                                    raise
                                self.hypotheses.pause(a["id"], str(error), stage="transfer")
                                raise
                            except Exception as error:
                                if a.get("candidate"):
                                    raise
                                with self.review_lock:
                                    self.reject_writer(a, error)
                    if status in {"stopped", "budget_exhausted"}:
                        break
                if self.active_task is None and status not in {"stopped", "budget_exhausted"}:
                    status = ("completed_kernel_search" if kernel_first(self.config) else
                              "accepted" if self.store.get("best_model_measurement") else "completed_no_gain")
                    self.stop_record(status, "All priority tasks reached a recorded stopping decision.")
        except BudgetExhausted as error:
            status, terminal_error = "budget_exhausted", str(error)
        except TransferError as error:
            # The executor has joined all sibling writers at this point.
            recover_auth_failures()
            status, terminal_error = "infrastructure_failure", str(error)
        except Exception as error:
            status, terminal_error = "request_failure", f"{type(error).__name__}: {error}"
            self.evidence_report({"error": terminal_error, "traceback": traceback.format_exc()}, "swarm-failure")
        finally:
            self.store.set_many({"budget_reserve": None, "phase_budget": None})
            # Budget failures can occur while writer futures finish. Their saved
            # outputs cost no further API tokens. Always give them GPU results.
            if status == "budget_exhausted" and not (self.directory / "STOP").exists():
                try:
                    for assignment in self.work.values():
                        if assignment["status"] in {"generated", "evaluated"} and assignment.get("candidate"):
                            self.event("evaluation_drain", assignment_id=assignment["id"],
                                       task_id=assignment["task_id"], reason="Finish saved output without API calls.")
                            device = next((d for d in devices if d and d.gpu == assignment.get("evaluation_gpu")), devices[0])
                            if device:
                                self.process_candidate(assignment, device)
                            else:
                                self.process_candidate(assignment)
                    if self.config.get("generation_objective") == "decode" and self.active_task:
                        self.validate_decode_target("token_budget")
                except Exception as error:
                    status, terminal_error = "infrastructure_failure", f"Saved evaluation drain failed: {error}"
            if callable(getattr(self.worker, "close", None)):
                self.worker.close()
        if terminal_error:
            self.stop_record(status, terminal_error)
        best_model = self.store.get("best_model_measurement")
        selected = best_model or self.store.get("last_model_measurement")
        incomplete = [a["id"] for a in self.work.values() if a["status"] not in {"completed", "failed", "budget_deferred"}]
        result = {"status": status, "usage": self.store.usage(), "budget": self.store.budget_snapshot(self.limits),
            "task_progress": self.progress, "active_task": self.active_task, "pending_repairs": self.pending,
            "closed_task_repairs": self.store.get("closed_task_repairs", {}), "review_complete": not incomplete,
            "incomplete_reviews": incomplete, "budget_decision": self.store.get("budget_decision"),
            "swarm_stop": self.store.get("swarm_stop"), "hardware": self.hardware,
            "manifest": (kernel_manifest(self.store, self.catalog) if kernel_first(self.config) else
                         selected["manifest"] if selected else {"entries": []}),
            "model_result": selected["result"] if selected else None}
        if kernel_first(self.config):
            result.update(kernel_baselines=self.store.get("kernel_baselines", {}),
                integration_status="pending" if result["manifest"]["entries"] else "no_kept_kernels")
            from .hierarchy import enabled
            if enabled(self.config):
                result["hierarchy"] = self.catalog["hierarchy"]
                result["completed_regions"] = self.store.get("hierarchy_completed_regions", {})
                result["kernel_manifest"] = result["manifest"]
                if selected:
                    result.update(manifest=best_model["manifest"] if best_model else {"entries": []},
                        tested_manifest=selected["manifest"], integration_status="accepted" if
                        self.store.get("best_model_measurement") else "rejected")
        if self.config.get("generation_objective") == "decode":
            previous = self.store.get("decode_previous_model") or {}
            accepted = (best_model or previous).get("manifest", {"entries": []})
            result.update(kernel_manifest=accepted, manifest=accepted, target_validations=self.store.get("decode_target_results", {}),
                          integration_status="pending", performance_scope="cached_decode")
        self.store.set_many({"status": status, "task_progress": self.progress,
                            "swarm_summary": {"experiments": len(self.work), "completed": sum(a["status"] == "completed" for a in self.work.values()),
                                              "max_writers": self.limits["writers"]}})
        write_json(self.directory / "manifest.json", result["manifest"])
        write_json(self.directory / "result.json", result)
        from .reporting import write_swarm_report
        write_swarm_report(self.directory, self.config, profile_dir=self.profile, store=self.store)
        return result
