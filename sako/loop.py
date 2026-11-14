"""Shared state, evaluation, and feedback for the Triton swarm."""

import copy
import traceback
from pathlib import Path

from .core import digest, implementation_hash, read_json, require_broad_selection, select_nonoverlapping, validate_candidate, write_json
from .feedback import compact_feedback, save_feedback
from .operation_names import catalog_names, operation_name
from .store import Store, operation_correct
from .transport import TransferError

SYSTEM = "You assist a GPU kernel search. Return JSON. Measurements decide correctness and speed. Use short, clear English."


class SearchLoop:
    def __init__(self, config, run_dir, profile_dir, client_factory, worker_factory,
                 manifest_factory):
        self.root = Path(__file__).resolve().parents[1]
        self.directory, self.profile = Path(run_dir).resolve(), Path(profile_dir).resolve()
        self.catalog = read_json(self.profile / "tasks.json")
        self.operation_names = catalog_names(self.catalog)
        require_broad_selection(config)
        require_broad_selection(self.catalog.get("config", {}))
        self.config, self.limits = config, config["agent"]
        self.store = Store(self.directory / "state.sqlite")
        fingerprint = digest({"config": config, "catalog": self.catalog, "implementation": implementation_hash()})
        if self.store.get("fingerprint", fingerprint) != fingerprint:
            raise ValueError("Code, config, hardware, or task catalog changed. Use a new run directory.")
        self.client = client_factory(self.limits, self.store, self.directory)
        self.worker = worker_factory(config)
        self.manifest_factory = manifest_factory
        prompt_file = "triton_writer.txt"
        self.system = Path(__file__).with_name("prompts").joinpath(prompt_file).read_text(encoding="utf-8")
        self.tasks = sorted(self.catalog["tasks"], key=lambda t: (-t["priority_ms"], t["id"]))
        self.by_id = {t["id"]: t for t in self.tasks}
        selected = self.catalog.get("selected_task_ids")
        if selected is None:
            entries = [{"task_id": t["id"], "paths": t["paths"], "priority": t["priority_ms"]}
                       for t in self.tasks]
            selected = [e["task_id"] for e in select_nonoverlapping(entries)]
        if len(selected) != len(set(selected)) or any(key not in self.by_id for key in selected):
            raise ValueError("The profile contains an invalid task selection.")
        from .hierarchy import enabled
        if not enabled(config):
            selected = selected[:self.limits.get("top_k_operations", 4)]
        chosen_regions = [{"task_id": key, "paths": self.by_id[key]["paths"],
                           "priority": self.by_id[key]["priority_ms"]} for key in selected]
        if len(select_nonoverlapping(chosen_regions)) != len(selected):
            if enabled(config) and self.limits.get("search_parent_regions", False):
                from .hierarchy import validate_search_overlaps
                validate_search_overlaps(self.catalog, self.by_id, selected)
            else:
                raise ValueError("Selected operation regions overlap.")
        for key in selected:
            contract = self.by_id[key].get("contract")
            if contract is not None and contract.get("validated") is not True:
                raise ValueError("The explicit operation contract has not passed verification.")
        self.store.set("fingerprint", fingerprint)
        self.progress = self.store.get("task_progress", {
            "queue": selected, "index": 0, "round": 0, "stagnant": 0,
            "best_operation_score": 0., "completed": []})
        if enabled(config):
            from .hierarchy import validate_queue
            validate_queue(self.catalog, self.progress)
        elif self.progress["queue"] != selected:
            raise ValueError("The saved task queue differs from the profile.")
        self.pending = self.store.get("pending_repairs", {})
        self.owners = self.store.get("owners", {})
        self.blocked = self.store.get("blocked_model_candidates", {})
        self.evidence = self.store.get("evidence_artifacts", [])
        self.task_prompts = {key: self.prepare_task_prompt(self.by_id[key]) for key in selected}

    @property
    def active_task(self):
        index = self.progress["index"]
        return self.progress["queue"][index] if index < len(self.progress["queue"]) else None

    def relative(self, path):
        return Path(path).relative_to(self.root).as_posix()

    def worker_io(self, call, *args, **kwargs):
        return call(*args, **kwargs)

    def evidence_report(self, value, name, **metadata):
        report = save_feedback(self.directory, value, name=name, metadata=metadata)
        artifact = report["artifact"]
        if artifact not in self.evidence:
            self.evidence.append(artifact)
            self.store.set("evidence_artifacts", self.evidence)
        return report

    def prepare_task_prompt(self, task):
        """Keep the exact operation; put repeated validation details in evidence."""
        report = self.evidence_report(task, "contract-" + task["id"], task_id=task["id"])
        prompt = copy.deepcopy(task)
        prompt["evidence"] = report["artifact"]
        contract = prompt.get("contract", {})
        validation = contract.get("validation", {})
        checks = validation.get("checks", [])
        if checks:
            validation["checks"] = {"count": len(checks),
                "passed": sum(check.get("passed") is True for check in checks),
                "failed": sum(check.get("passed") is not True for check in checks),
                "max_abs_error": max((check.get("max_abs_error", 0.) for check in checks
                                      if isinstance(check.get("max_abs_error", 0.), (int, float))), default=0.),
                "examples": compact_feedback(checks, max_items=3)}
        if task.get("separate_operations"):
            from .shape_selection import case_signature
            examples = {}
            for i, case in enumerate(task["cases"]):
                key = (digest(case_signature(case, task.get("scalars", []))),
                       case.get("generation_step", {}).get("phase"))
                examples.setdefault(key, {**case, "case_index": i})
            prompt["cases"] = list(examples.values())
            prompt["fixture_count"] = len(task["cases"])
            contract.pop("cases", None)
            prompt["case_scope"] = "Representative signatures. The evaluator measures each real module path."
        for owner in (prompt, contract):
            paths = owner.get("paths", [])
            if len(paths) > 8:
                owner["path_count"] = len(paths)
                owner["paths"] = paths[:8]
                owner["paths_note"] = "First eight module paths. Complete paths remain in the contract evidence."
        return prompt


    def row(self, key):
        row = self.store.candidate(key)
        if row is None:
            raise ValueError("Unknown candidate: " + str(key))
        return row

    def persist_repairs(self):
        self.store.set_many({"pending_repairs": self.pending, "owners": self.owners,
                             "blocked_model_candidates": self.blocked})

    def queue(self, slot, key, stage, result, used, manifest=None):
        used = max(used, self.store.get("repairs_used:" + key, 0))
        task_id = self.row(key)["task"]
        report = self.evidence_report(result, "repair-" + key + "-" + stage,
                                      task_id=task_id, candidate_id=key, stage=stage)
        repair = {"candidate_id": key, "task_id": task_id, "stage": stage,
                  "result": self.model_feedback(result) if stage in {"model", "model_performance", "interaction"}
                            else report["summary"], "evidence": report["artifact"],
                  "repairs_used": used, "manifest": manifest, "slot": slot}
        if used >= self.limits["max_repair_attempts"]:
            self.pending.pop(key, None)
            self.store.set("repair_exhausted:" + key, repair)
        else:
            self.pending[key] = repair
        self.persist_repairs()


    def evaluate(self, assignment, candidate, generation):
        validate_candidate(candidate, self.config.get("kernel_backend", "triton"))
        device = getattr(getattr(self, "review_context", None), "device", None)
        worker, directory, profile = ((device.worker, device.directory, device.profile) if device else
                                      (self.worker, self.directory, self.profile))
        key, _ = self.store.add(candidate, assignment["island"], generation, assignment["parent_id"])
        self.owners.setdefault(key, assignment["slot"])
        self.store.set("repairs_used:" + key, max(assignment["repairs_used"], self.store.get("repairs_used:" + key, 0)))
        path = self.directory / "candidates" / (key + ".json")
        write_json(path, candidate)
        self.worker_io(worker.put, path, self.relative(path))
        result = self.row(key)["result"]
        if result is None:
            from .kernel_selection import kernel_first, kernel_config, record_archive_score
            fields = {}
            focus_id = assignment.get("hypothesis", {}).get("focus_id")
            if focus_id:
                division = self.store.get("operation_split:" + candidate["task_id"])
                fields["focus_paths"] = next(s["paths"] for s in division["scopes"] if s["id"] == focus_id)
            observation = assignment.get("writer_design", assignment.get("hypothesis", {})).get("verification", {})
            if observation.get("implementation"):
                fields["observe_fixture"] = observation["fixture_index"]
            if kernel_first(self.config):
                from .shape_selection import enabled as shape_enabled
                shaped = shape_enabled(self.config)
                baselines = self.store.get("kernel_evaluation_baselines", {})
                if key not in baselines:
                    kept = self.store.get("kernel_baselines", {}).get(candidate["task_id"], {})
                    if self.config.get("generation_objective") == "decode":
                        kept = {}
                    baselines[key] = ({s: v["candidate_id"] for s, v in kept.get("shapes", {}).items()}
                                      if shaped else kept.get("candidate_id"))
                    self.store.set("kernel_evaluation_baselines", baselines)
                baseline_id = baselines[key]
                if shaped:
                    fields["baseline_shapes"] = {}
                    uploaded = set()
                    for shape, identity in (baseline_id or {}).items():
                        baseline_path = self.directory / "candidates" / (identity + ".json")
                        if identity not in uploaded:
                            self.worker_io(self.worker.put, baseline_path, self.relative(baseline_path))
                            uploaded.add(identity)
                        fields["baseline_shapes"][shape] = self.relative(baseline_path)
                elif baseline_id:
                    baseline_path = self.directory / "candidates" / (baseline_id + ".json")
                    self.worker_io(self.worker.put, baseline_path, self.relative(baseline_path))
                    fields["baseline_candidate"] = self.relative(baseline_path)
            print(f"Testing {operation_name(candidate['task_id'], self.operation_names)} from {assignment['island']}.", flush=True)
            try:
                result = self.worker_io(worker.run, "evaluate", directory, profile_dir=self.relative(profile),
                                         candidate=self.relative(path), **fields)
                if device:
                    device.verify(result)
            except TransferError:
                raise
            except Exception as error:
                result = {"decision": {"accepted": False, "speedup": None}, "error": str(error),
                          "traceback": traceback.format_exc(), "worker_report": getattr(error, "payload", None)}
            if kernel_first(self.config):
                record_archive_score(self.store, self.by_id[candidate["task_id"]], result, baseline_id,
                                     trials=kernel_config(self.config)["trials"],
                                     mode=self.config.get("kernel_search", {}).get("score_reference", "fixed_anchor"))
            self.evidence_report(result, "operation-" + key, task_id=candidate["task_id"], candidate_id=key)
            self.store.record(key, result)
        if assignment["repair"]:
            self.pending.pop(assignment["repair"]["candidate_id"], None)
        if key in self.blocked:
            previous = self.blocked[key]
            self.queue(assignment["slot"], key, "model", previous["result"],
                       assignment["repairs_used"], previous["manifest"])
        elif not operation_correct(result) and result.get("stages", {}).get("outcome") != "screened_slow":
            self.queue(assignment["slot"], key, "operation", result, assignment["repairs_used"])
        self.persist_repairs()
        return key

    def _measure_manifest(self, manifest, generation, *, incumbent=None):
        from .model_acceptance import model_trial_key, reusable_trial
        from .integration_repairs import safe_measurement
        best = self.store.get("best_model_measurement")
        if best and best["manifest"] == manifest and safe_measurement(self.store, best):
            return best
        manifest_hash = model_trial_key(self.store.get("fingerprint"), manifest, incumbent)
        cache = self.store.get("model_measurements", {})
        cached = reusable_trial(cache, manifest_hash)
        if cached:
            return cached
        path = self.directory / "model_checks" / (manifest_hash + ".manifest.json")
        write_json(path, manifest)
        self.worker_io(self.worker.put, path, self.relative(path))
        options = {}
        if incumbent is not None:
            base_path = self.directory / "model_checks" / (manifest_hash + ".incumbent.json")
            write_json(base_path, incumbent)
            self.worker_io(self.worker.put, base_path, self.relative(base_path))
            options["incumbent_manifest"] = self.relative(base_path)
            from .measurement import enabled as adaptive, incumbent_reference
            if adaptive(self.config):
                ref_path = self.directory / "model_checks" / (manifest_hash + ".incumbent-reference.json")
                write_json(ref_path, incumbent_reference(best, self.catalog))
                self.worker_io(self.worker.put, ref_path, self.relative(ref_path))
                options["incumbent_reference"] = self.relative(ref_path)
        print("Testing the complete model inside the search loop.", flush=True)
        try:
            result = self.worker_io(self.worker.run, "model", self.directory, profile_dir=self.relative(self.profile),
                manifest=self.relative(path), candidate_root=self.relative(self.directory / "candidates"),
                feedback_trace_dir=self.relative(self.directory / "model_checks" / manifest_hash), **options)
        except TransferError:
            raise
        except Exception as error:
            result = {"decision": {"accepted": False, "speedup": None}, "error": str(error),
                      "traceback": traceback.format_exc(), "worker_report": getattr(error, "payload", None)}
        measured = {"manifest": manifest, "result": result, "generation": generation,
                    **({"incumbent": incumbent} if incumbent is not None else {})}
        from .deployment_diagnostics import retain_trace_feedback
        retain_trace_feedback(self.store, manifest_hash, manifest, result)
        write_json(self.directory / "model_checks" / (manifest_hash + ".result.json"), measured)
        self.evidence_report(measured, "model-" + manifest_hash, stage="model")
        cache[manifest_hash] = measured
        self.store.set("model_measurements", cache)
        return measured

    def measure_model(self, generation, assigned):
        manifest = self.manifest_factory(self.store, self.catalog)
        if not manifest["entries"]:
            return None
        measured = self._measure_manifest(manifest, generation)
        self.store.set("last_model_measurement", measured)
        if measured["result"].get("decision", {}).get("accepted"):
            covered = {e["candidate_id"] for e in manifest["entries"]}
            self.pending = {k: r for k, r in self.pending.items() if k not in covered}
            best = self.store.get("best_model_measurement")
            if not best or (measured["result"]["decision"].get("speedup") or 0) > (best["result"]["decision"].get("speedup") or 0):
                self.store.set("best_model_measurement", measured)
        else:
            rejected = self.store.get("rejected_model_candidates", {})
            combinations = self.store.get("rejected_model_combinations", {})
            combinations[digest(manifest)] = measured
            combination_correct = (operation_correct(measured["result"])
                                   and measured["result"].get("reference_state_intact") is not False)
            for entry in manifest["entries"]:
                if entry["task_id"] != self.active_task:
                    continue
                key = entry["candidate_id"]
                if len(manifest["entries"]) == 1:
                    individual = measured
                elif combination_correct:
                    # The combination already passed accuracy. Another complete
                    # model test is not needed to establish that it missed speed.
                    individual = None
                else:
                    individual = self._measure_manifest({"entries": [entry]}, generation)
                rejected[key] = measured
                incorrect = individual is not None and not operation_correct(individual["result"])
                if incorrect:
                    self.blocked[key] = individual
                stage = "model" if incorrect else "interaction" if len(manifest["entries"]) > 1 else "model_performance"
                feedback = {"combination": measured["result"],
                            "individual": individual["result"] if individual is not None else None}
                if individual is None:
                    feedback["individual_test"] = {"status": "not_measured", "reason":
                        "The combination passed accuracy. An isolated test is not required for this speed rejection."}
                self.queue(self.owners[key], key, stage, feedback,
                           self.store.get("repairs_used:" + key, 0), manifest)
            self.store.set_many({"rejected_model_candidates": rejected, "rejected_model_combinations": combinations})
        self.persist_repairs()
        return measured

    def begin_task(self):
        task_id = self.active_task
        self.store.set("active_task", task_id)
        self.store.set("phase_budget", None)

    def advance(self, reason, next_generation):
        task_id = self.active_task
        self.progress["completed"].append({"task_id": task_id, "rounds": self.progress["round"], "reason": reason})
        deferred = self.store.get("closed_task_repairs", {})
        deferred.update({k: {**r, "closed_reason": reason} for k, r in self.pending.items() if r["task_id"] == task_id})
        self.pending = {k: r for k, r in self.pending.items() if r["task_id"] != task_id}
        self.progress.update(index=self.progress["index"] + 1, round=0, stagnant=0, best_operation_score=0.)
        self.store.set_many({"task_progress": self.progress, "generation": next_generation,
                             "phase_budget": None, "active_task": self.active_task,
                             "closed_task_repairs": deferred, "pending_repairs": self.pending})


    @staticmethod
    def operation_feedback(result):
        """Send decision metrics. Keep complete samples and diagnostics in evidence."""
        if not isinstance(result, dict):
            return None
        decision = dict(result.get("decision", {}))
        reasons = decision.get("reasons", [])
        if len(reasons) > 3:
            decision.update(reasons=reasons[:3], reason_count=len(reasons))
        summary = {"decision": decision, "cases": []}
        if result.get("error"):
            summary["error"] = str(result["error"])[:800]
        for field in ("timing_baseline", "kernel_min_speedup", "search_score", "acceptance_scope", "shape_decisions"):
            if field in result:
                summary[field] = result[field]
        if "stages" in result:
            summary["stages"] = result["stages"]
            summary["shape_decisions"] = {k: {field: v[field] for field in ("paths", "accepted", "speedup") if field in v}
                                          for k, v in list(result.get("shape_decisions", {}).items())[:4]}
            summary.pop("timing_baseline", None)
        for case in result.get("cases", [])[:6]:
            checks = case.get("checks", [])
            errors = [check.get("max_abs_error", 0.) for check in checks]
            max_error = None if any(error is None for error in errors) else max(errors, default=0.)
            registers = [r["registers_per_thread"] for r in case.get("build", {}).get("resource_reports", [])
                         if isinstance(r.get("registers_per_thread"), int)]
            shared = [r["static_shared_memory_bytes"] for r in case.get("build", {}).get("resource_reports", [])
                      if isinstance(r.get("static_shared_memory_bytes"), int)]
            spills = [r["reported_spills"] for r in case.get("build", {}).get("resource_reports", [])
                      if isinstance(r.get("reported_spills"), int)]
            summary["cases"].append({"case_index": case.get("case_index"), "seed": case.get("seed"),
                **({"source_path": case["source_path"]} if case.get("source_path") else {}),
                "workload": case.get("workload"), "correct": case.get("correct"),
                "max_abs_error": max_error,
                "baseline_ms": case.get("baseline", {}).get("median_ms"),
                "candidate_ms": case.get("candidate", {}).get("median_ms"),
                "candidate_p95_ms": case.get("candidate", {}).get("p95_ms"),
                "registers_per_thread": max(registers, default=None),
                **({"shared_memory_bytes": max(shared)} if shared else {}),
                **({"reported_spills": max(spills)} if spills else {}),
                **({"resource_status": case["build"]["status"]} if "status" in case.get("build", {}) else {}),
                "timing_mode": case.get("timing_mode"),
                "autotuning": [{**{k: v for k, v in tuning.items() if k in {"kernel", "config_count", "choices"}},
                    **({"failed_configurations": [{"parameters": t["parameters"], "num_warps": t["num_warps"],
                        "num_stages": t["num_stages"], "error": t["error"][:400]}
                        for t in tuning["harness_trials"] if t.get("error")][-3:]}
                       if any(t.get("error") for t in tuning.get("harness_trials", [])) else {})}
                               for tuning in case.get("autotuning", [])]})
        return summary

    def review_ranking(self, rows=None):
        if rows is None:
            rows = self.store.candidates(self.active_task)
        selected = {r["id"]: r for r in sorted(rows, key=lambda r: (-r["generation"], r["island"], r["id"]))[:self.limits["writers"]]}
        for island in self.limits["islands"]:
            best = next((r for r in rows if r["island"] == island and r["correct"]), None)
            if best:
                selected.setdefault(best["id"], best)
        return [{"id": r["id"], "task": r["task"], "island": r["island"], "generation": r["generation"],
                 "correct": r["correct"], "accepted": r["accepted"], "score": r["score"],
                 "result": self.operation_feedback(r["result"])} for r in selected.values()]

    @staticmethod
    def model_feedback(value):
        """Keep seed checks and a few examples without repeating complete reports."""
        if not isinstance(value, dict):
            return None
        source = value
        for _ in range(3):
            nested = source.get("combination", source.get("result"))
            if not isinstance(nested, dict):
                break
            source = nested
        decision = dict(source.get("decision", {}))
        if len(decision.get("reasons", [])) > 3:
            decision.update(reason_count=len(decision["reasons"]), reasons=decision["reasons"][:3])
        result = {"scope": source.get("scope"), "compiled": source.get("compiled"),
                  "decision": decision, "cases": []}
        if source.get("incumbent_comparison"):
            comparison = source["incumbent_comparison"]
            result["incumbent_comparison"] = {
                "decision": comparison["decision"],
                "reference": comparison.get("reference"),
                "cases": [copy.deepcopy(case) if "incremental_speedup" in case else
                          {"workload": case["workload"], "baseline_ms": case["baseline"]["median_ms"],
                           "candidate_ms": case["candidate"]["median_ms"]} for case in comparison["cases"]]}
        if source.get("performance_trace"):
            from .deployment_diagnostics import trace_feedback
            result["performance_trace"] = trace_feedback(source["performance_trace"])
        if isinstance(value.get("individual_test"), dict):
            result["individual_test"] = copy.deepcopy(value["individual_test"])
        for field in ("correctness_scope", "correctness_reference", "model_reference", "output_equivalence", "reference_state_intact"):
            if field in source:
                result[field] = source[field]

        def numerical_error(value):
            import math
            return value if type(value) in (int, float) and math.isfinite(value) else None

        def path_count(values):
            return sum(1 if isinstance(item, str) else item.get("items_omitted", 0)
                       if isinstance(item, dict) and type(item.get("items_omitted")) is int else 0
                       for item in values)

        def layer_feedback(check, examples=False, depth=0):
            def retain_region(entry):
                region = check.get("region_checks")
                if isinstance(region, dict):
                    if depth < 2:
                        entry["region_checks"] = layer_feedback(region, examples=examples, depth=depth + 1)
                    else:
                        entry["region_details_omitted"] = True
                return entry

            if "layers" not in check and "failed_layers" in check:
                entry = {key: copy.deepcopy(check[key]) for key in (
                    "seed", "passed", "scope", "expected_path_count", "observed_path_count", "missing_path_count",
                    "missing_paths", "layer_check_count", "failed_layer_count", "max_abs_error", "layer_details_omitted",
                    "failed_layer_count_is_minimum", "reference_state_intact", "not_measured", "reason", "error") if key in check}
                entry["failed_layers"] = copy.deepcopy(check["failed_layers"][:2])
                if not examples:
                    for layer in entry["failed_layers"]:
                        layer.pop("examples", None)
                return retain_region(entry)
            raw_layers = check.get("layers", [])
            layers = [layer for layer in raw_layers if isinstance(layer, dict) and isinstance(layer.get("path"), str)]
            omitted = sum(layer.get("items_omitted", 0) for layer in raw_layers
                          if isinstance(layer, dict) and type(layer.get("items_omitted")) is int)
            failed = [layer for layer in layers if layer.get("passed") is not True
                      or layer.get("inputs_unchanged") is False]
            failed.sort(key=lambda layer: (numerical_error((layer.get("check") or {}).get("max_abs_error")) is None,
                                          numerical_error((layer.get("check") or {}).get("max_abs_error")) or 0.), reverse=True)
            errors = [numerical_error((layer.get("check") or {}).get("max_abs_error")) for layer in layers]
            entry = {"seed": check.get("seed"), "passed": check.get("passed"), "scope": "replaced_layers",
                     "expected_path_count": path_count(check.get("expected_paths", [])),
                     "observed_path_count": path_count(check.get("observed_paths", [])),
                     "missing_path_count": path_count(check.get("missing_paths", [])),
                     "missing_paths": [path for path in check.get("missing_paths", []) if isinstance(path, str)][:3],
                     "layer_check_count": len(layers) + omitted, "failed_layer_count": len(failed),
                     "max_abs_error": None if any(error is None for error in errors) else max(errors, default=None),
                     "failed_layers": []}
            if omitted:
                entry.update(layer_details_omitted=omitted, failed_layer_count_is_minimum=True)
            for field in ("reference_state_intact", "not_measured"):
                if field in check:
                    entry[field] = check[field]
            if check.get("reason"):
                entry["reason"] = str(check["reason"])[:400]
            for index, layer in enumerate(failed[:2]):
                numerical = layer.get("check") or {}
                brief = {key: layer.get(key) for key in
                         ("path", "candidate_id", "kind", "call_index", "passed", "inputs_unchanged")}
                brief.update({key: numerical_error(numerical.get(key)) for key in ("max_abs_error", "mismatched_elements", "atol", "rtol")})
                for field in ("scope", "same_input", "reference", "input_shapes", "input_strides", "input_dtypes"):
                    if field in layer:
                        brief[field] = copy.deepcopy(layer[field])
                if any(isinstance(numerical.get(key), dict) for key in ("max_abs_error", "atol", "rtol")):
                    brief["numerical_details_omitted"] = True
                if examples and index == 0 and isinstance(numerical.get("examples"), list):
                    brief["examples"] = [{key: example.get(key) for key in
                        ("flat_index", "actual", "expected", "allowed_abs_error")}
                        for example in numerical["examples"][:2]]
                if layer.get("error"):
                    brief["error"] = str(layer["error"])[:300]
                entry["failed_layers"].append(brief)
            if check.get("error"):
                entry["error"] = str(check["error"])[:300]
            return retain_region(entry)
        manifest = value.get("manifest", source.get("manifest", {})) or {}
        if manifest.get("entries"):
            result["replacements"] = [{"task_id": e.get("task_id"), "candidate_id": e.get("candidate_id"),
                                       "module_count": len(e.get("paths", []))} for e in manifest["entries"][:4]]
        if source.get("error"):
            result["error"] = str(source["error"])[:800]
        for case in source.get("cases", [])[:3]:
            checks = case.get("checks", [])
            brief = {"workload": case.get("workload"), "correct": case.get("correct"),
                     **({"sampling": {"assessment": case["sampling"].get("assessment"),
                                      "rounds": case["sampling"].get("rounds")}} if case.get("sampling") else {}),
                     "reference": case.get("reference"), "checks": [],
                     "baseline_ms": (case.get("baseline") or {}).get("median_ms"),
                     "candidate_ms": (case.get("candidate") or {}).get("median_ms")}
            if case.get("timing_status"):
                brief["timing_status"] = case["timing_status"]
            if case.get("timing_reason"):
                brief["timing_reason"] = str(case["timing_reason"])[:400]
            ranked = [layer_feedback(check) if check.get("scope") == "replaced_layers" else check for check in checks]
            def failure_rank(check):
                choices = [check]
                nested = check.get("region_checks")
                while isinstance(nested, dict):
                    choices.append(nested)
                    nested = nested.get("region_checks")
                def rank(value):
                    errors = [layer.get("max_abs_error") for layer in value.get("failed_layers", [])]
                    error = (None if any(item is None for item in errors) else max(errors)) if errors else value.get("max_abs_error")
                    return (value.get("passed") is False, bool(value.get("failed_layer_count")),
                            value.get("missing_path_count", 0), error is None, error or 0.)
                return max(rank(value) for value in choices)
            worst = max(range(len(checks)), key=lambda i: failure_rank(ranked[i]), default=None)
            for index, check in enumerate(checks):
                if check.get("scope") == "replaced_layers":
                    entry = layer_feedback(check, examples=index == worst)
                    if entry["seed"] is None:
                        entry["seed"] = case.get("seed")
                    brief["checks"].append(entry)
                    continue
                if check.get("scope") == "cached_generation":
                    entry = {key: check.get(key) for key in ("scope", "passed", "steps_checked",
                        "greedy_tokens_match", "cache_extent_valid", "logical_cache_length")}
                    entry["logits"] = compact_feedback(check.get("logits"))
                    entry["failed_cache_layers"] = compact_feedback([x for x in check.get("cache_layers", [])
                                                                    if x.get("passed") is False])
                    brief["checks"].append(entry)
                    continue
                entry = {key: check.get(key) for key in ("seed", "passed", "max_abs_error",
                                                          "mismatched_elements", "atol", "rtol")}
                if entry["seed"] is None:
                    entry["seed"] = case.get("seed")
                if index == worst and check.get("examples"):
                    entry["examples"] = [{key: example.get(key) for key in
                        ("flat_index", "actual", "expected", "allowed_abs_error")}
                        for example in check["examples"][:2]]
                brief["checks"].append(entry)
            diagnostics = [c.get("max_abs_error") for c in case.get("compiled_vs_eager_diagnostics", [])
                           if isinstance(c.get("max_abs_error"), (int, float))]
            if diagnostics:
                brief["compiled_vs_eager_max_abs_error"] = max(diagnostics)
            if case.get("error"):
                brief["error"] = str(case["error"])[:600]
            result["cases"].append(brief)
        return result

    @staticmethod
    def repair_feedback(value):
        """Keep repair ownership and the worst seed; raw evidence stays unchanged."""
        if value is None:
            return None
        if not isinstance(value, dict):
            return {}
        if "candidate_id" not in value:
            return {key: SearchLoop.repair_feedback(item) for key, item in value.items()}
        result = {key: copy.deepcopy(value.get(key)) for key in
                  ("candidate_id", "task_id", "stage", "slot", "repairs_used", "evidence")}
        model = SearchLoop.model_feedback(value.get("result", {})) or {}
        checks = [(case, check) for case in model.get("cases", []) for check in case.get("checks", [])]
        boundaries = []
        for case, check in checks:
            boundaries.append((case, check, "replacement"))
            region = check.get("region_checks")
            while isinstance(region, dict):
                boundaries.append((case, region, "containing_region"))
                region = region.get("region_checks")
        def failure_rank(item):
            _, check, boundary = item
            errors = [layer.get("max_abs_error") for layer in check.get("failed_layers", [])]
            error = (None if any(value is None for value in errors) else max(errors)) if errors else check.get("max_abs_error")
            return (check.get("passed") is False, bool(check.get("failed_layer_count")),
                    check.get("missing_path_count", 0), error is None, error or 0.,
                    boundary == "containing_region")
        worst = max(boundaries, key=failure_rank, default=None)
        brief = {"decision": model.get("decision", {}),
                 "failed_checks": sum(check.get("passed") is False for _, check in checks)}
        if worst:
            case, check, boundary = worst
            brief.update(workload=case.get("workload"), **check)
            if boundary == "containing_region":
                brief["failure_boundary"] = boundary
        for field in ("correctness_scope", "correctness_reference", "model_reference", "output_equivalence", "reference_state_intact"):
            if field in model:
                brief[field] = model[field]
        if model.get("error"):
            brief["error"] = model["error"]
        case_errors = [case["error"] for case in model.get("cases", []) if case.get("error")]
        if case_errors:
            brief["case_error"] = case_errors[0]
        result["result"] = brief
        manifest = value.get("manifest") or {}
        if manifest.get("entries"):
            result["replacements"] = [{"task_id": e.get("task_id"), "candidate_id": e.get("candidate_id"),
                                       "module_count": len(e.get("paths", []))} for e in manifest["entries"][:4]]
        return result
