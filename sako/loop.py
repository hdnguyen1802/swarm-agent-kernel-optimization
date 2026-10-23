"""Run tests inside the search and keep failed tasks assigned for repair."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

from .core import (candidate_id, digest, implementation_hash, read_json,
                   validate_candidate, write_json)
from .store import BudgetExhausted, Store

SYSTEM = "You assist a CUDA kernel search. Return JSON. Measurements decide correctness and speed. Use short, clear English."


def test_feedback(value):
    """Keep test failures and summaries. Store raw timing arrays separately."""
    if isinstance(value, dict):
        return {k: test_feedback(v) for k, v in value.items() if k not in ("samples_ms", "config")}
    if isinstance(value, list):
        return [test_feedback(v) for v in value]
    return value


class SearchLoop:
    def __init__(self, config, run_dir, profile_dir, client_factory, worker_factory,
                 retrieve, manifest_factory, reviewer=False):
        self.root = Path(__file__).resolve().parents[1]
        self.directory, self.profile = Path(run_dir).resolve(), Path(profile_dir).resolve()
        self.catalog = read_json(self.profile / "tasks.json")
        self.config, self.limits = config, config["agent"]
        self.store = Store(self.directory / "state.sqlite")
        fingerprint = digest({"config": config, "catalog": self.catalog, "implementation": implementation_hash()})
        if self.store.get("fingerprint", fingerprint) != fingerprint:
            raise ValueError("Code, config, or task catalog changed. Use a new run directory.")
        self.store.set("fingerprint", fingerprint)
        self.client = client_factory(self.limits, self.store, self.directory)
        self.worker = worker_factory(config)
        self.retrieve, self.manifest_factory, self.reviewer = retrieve, manifest_factory, reviewer
        self.system = Path(__file__).with_name("prompts").joinpath("writer.txt").read_text(encoding="utf-8")
        self.tasks = sorted(self.catalog["tasks"], key=lambda t: t["priority_ms"], reverse=True)
        self.by_id = {t["id"]: t for t in self.tasks}
        self.pending = self.store.get("pending_repairs", {})
        self.owners = self.store.get("owners", {})
        self.blocked = self.store.get("blocked_model_candidates", {})

    def relative(self, path):
        return Path(path).relative_to(self.root).as_posix()

    def support(self, role, data):
        try:
            result = self.client.call(role, json.dumps(data), SYSTEM)
            if not isinstance(result, dict):
                raise ValueError(f"{role} must return an object.")
            return result
        except BudgetExhausted:
            raise
        except Exception as error:
            result = {"notes": f"{role} unavailable: {error}"}
            self.store.set(role + "_error", result)
            return result

    def row(self, key):
        return next(r for r in self.store.candidates() if r["id"] == key)

    def persist_repairs(self):
        self.store.set("pending_repairs", self.pending)
        self.store.set("owners", self.owners)
        self.store.set("blocked_model_candidates", self.blocked)

    def queue(self, slot, key, stage, result, used, manifest=None):
        used = max(used, self.store.get("repairs_used:" + key, 0))
        repair = {"candidate_id": key, "task_id": self.row(key)["task"], "stage": stage,
                  "result": test_feedback(result), "repairs_used": used, "manifest": manifest, "slot": slot}
        if used >= self.limits["max_repair_attempts"]:
            self.pending.pop(key, None)
            self.store.set("repair_exhausted:" + key, repair)
        else:
            self.pending[key] = repair
        self.persist_repairs()

    def assignments(self, generation, plan):
        saved = self.store.get(f"assignments:{generation}")
        if saved is not None:
            return saved
        order = [k for k in plan.get("task_order", []) if k in self.by_id]
        order.extend(t["id"] for t in self.tasks if t["id"] not in order)
        assigned = []
        for slot in range(8):
            repairs = [r for r in self.pending.values() if r["slot"] == slot]
            repairs.sort(key=lambda r: r["stage"] != "model")
            repair = repairs[0] if repairs else None
            island = self.limits["islands"][slot // 2]
            task_id = repair["task_id"] if repair else order[(generation * 4 + slot // 2) % len(order)]
            rows = self.store.candidates(task_id)
            migrate = generation % self.limits["migration_interval"] == 0
            parents = [r for r in rows if r["id"] not in self.blocked and r["accepted"]
                       and (migrate or r["island"] == island)]
            # Failed source is useful for repair even when no candidate passed yet.
            parent = self.row(repair["candidate_id"]) if repair else (parents[0] if parents else None)
            if parent is None and rows:
                parent = max(rows, key=lambda r: r["generation"])
            assigned.append({"slot": slot, "island": island, "task_id": task_id,
                             "parent_id": parent["id"] if parent else None, "repair": repair,
                             "repairs_used": repair["repairs_used"] + 1 if repair else 0})
        self.store.set(f"assignments:{generation}", assigned)
        return assigned

    def write(self, assignment, generation, plan):
        task = self.by_id[assignment["task_id"]]
        rows = self.store.candidates(task["id"])
        recent = sorted(rows, key=lambda r: r["generation"], reverse=True)[:3]
        prompt = {"task": task, "island": assignment["island"], "writer": assignment["slot"],
                  "generation": generation, "persistent_plan": plan,
                  "parent": self.row(assignment["parent_id"])["source"] if assignment["parent_id"] else None,
                  "repair": assignment["repair"],
                  "instruction": "Repair the supplied parent for the same task. Address the failing tests first."
                                 if assignment["repair"] else "Explore a CUDA implementation for this task.",
                  "feedback": [{"id": r["id"], "strategy": r["source"]["strategy"],
                                "result": test_feedback(r["result"])} for r in recent],
                  "references": self.retrieve(assignment["island"] + " " + task["kind"], task["kind"])}
        # Never drop the failed source or its current test report to fit old history.
        for field in ("feedback", "references"):
            while len(json.dumps(prompt)) > self.limits["context_chars"] and prompt[field]:
                prompt[field].pop()
        candidate = self.client.call(f"writer_{assignment['slot']}_{assignment['island']}",
                                     json.dumps(prompt), self.system)
        validate_candidate(candidate)
        if candidate["task_id"] != task["id"]:
            raise ValueError("Writer changed the assigned task.")
        return candidate

    def evaluate(self, assignment, candidate, generation):
        key, _ = self.store.add(candidate, assignment["island"], generation, assignment["parent_id"])
        self.owners.setdefault(key, assignment["slot"])
        self.store.set("repairs_used:" + key, max(assignment["repairs_used"], self.store.get("repairs_used:" + key, 0)))
        path = self.directory / "candidates" / (key + ".json")
        write_json(path, candidate)
        self.worker.put(path, self.relative(path))
        row = self.row(key)
        result = row["result"]
        if result is None:
            print(f"Testing {candidate['task_id']} from {assignment['island']}.", flush=True)
            try:
                result = self.worker.run("evaluate", self.directory, profile_dir=self.relative(self.profile),
                                         candidate=self.relative(path))
            except Exception as error:
                result = {"decision": {"accepted": False, "speedup": None}, "error": str(error)[-12000:]}
            self.store.record(key, result)
        if assignment["repair"]:
            self.pending.pop(assignment["repair"]["candidate_id"], None)
        if key in self.blocked:
            previous = self.blocked[key]
            self.queue(assignment["slot"], key, "model", previous["result"],
                       assignment["repairs_used"], previous["manifest"])
        elif not result.get("decision", {}).get("accepted"):
            self.queue(assignment["slot"], key, "operation", result, assignment["repairs_used"])
        self.persist_repairs()
        return key

    def measure_model(self, generation, assigned):
        manifest = self.manifest_factory(self.store, self.catalog)
        if not manifest["entries"]:
            return None
        manifest_hash = digest(manifest)
        cache = self.store.get("model_measurements", {})
        if manifest_hash in cache:
            measured = cache[manifest_hash]
        else:
            path = self.directory / "model_checks" / (manifest_hash + ".manifest.json")
            write_json(path, manifest)
            self.worker.put(path, self.relative(path))
            print("Testing the complete model inside the search loop.", flush=True)
            try:
                result = self.worker.run("model", self.directory, profile_dir=self.relative(self.profile),
                    manifest=self.relative(path), candidate_root=self.relative(self.directory / "candidates"))
            except Exception as error:
                result = {"decision": {"accepted": False, "speedup": None}, "error": str(error)[-12000:]}
            measured = {"manifest": manifest, "result": result, "generation": generation}
            write_json(self.directory / "model_checks" / (manifest_hash + ".result.json"), measured)
            cache[manifest_hash] = measured
            self.store.set("model_measurements", cache)
        self.store.set("last_model_measurement", measured)
        if measured["result"].get("decision", {}).get("accepted"):
            covered = {e["task_id"] for e in manifest["entries"]}
            self.pending = {k: r for k, r in self.pending.items() if r["task_id"] not in covered}
            best = self.store.get("best_model_measurement")
            if not best or (measured["result"]["decision"].get("speedup") or 0) > (best["result"]["decision"].get("speedup") or 0):
                self.store.set("best_model_measurement", measured)
        else:
            for entry in manifest["entries"]:
                key = entry["candidate_id"]
                self.blocked[key] = measured
                slot = self.owners[key]
                self.queue(slot, key, "model", measured["result"], self.store.get("repairs_used:" + key, 0), manifest)
        self.persist_repairs()
        return measured

    def run(self):
        if self.store.get("status") == "accepted":
            return read_json(self.directory / "result.json")
        status = "generation_limit_reached"
        try:
            for generation in range(self.store.get("generation", 0), self.limits["generations"]):
                if (self.directory / "STOP").exists():
                    status = "stopped"
                    break
                print(f"Generation {generation + 1}: writing or repairing eight tasks.", flush=True)
                plan = self.store.get(f"plan:{generation}")
                if plan is None:
                    plan = self.support("coordinator", {"request": "Return task_order (task IDs) and plan (short text).",
                        "previous_plan": self.store.get("plan"), "analysis": self.store.get("analysis"),
                        "pending_repairs": self.pending,
                        "tasks": [{k: t[k] for k in ("id", "kind", "priority_ms")} for t in self.tasks]})
                    self.store.set(f"plan:{generation}", plan)
                self.store.set("plan", plan)
                write_json(self.directory / "plan.json", plan)
                assigned = self.assignments(generation, plan)
                submissions = self.store.get(f"submissions:{generation}", {})
                exhausted = False
                with ThreadPoolExecutor(max_workers=8) as pool:
                    futures = {pool.submit(self.write, a, generation, plan): a for a in assigned
                               if str(a["slot"]) not in submissions}
                    for future in as_completed(futures):
                        a = futures[future]
                        try:
                            candidate = future.result()
                            key, _ = self.store.add(candidate, a["island"], generation, a["parent_id"])
                            submissions[str(a["slot"])] = key
                            self.store.set(f"submissions:{generation}", submissions)
                        except BudgetExhausted:
                            exhausted = True
                        except Exception as error:
                            write_json(self.directory / "errors" / f"{generation}-{a['slot']}.json", {"error": str(error)})
                for a in assigned:
                    key = submissions.get(str(a["slot"]))
                    if key:
                        self.evaluate(a, self.row(key)["source"], generation)
                # Even the last round and a partially exhausted round get a model check.
                self.measure_model(generation, assigned)
                self.store.set("generation", generation + 1)
                if exhausted:
                    status = "budget_exhausted"
                    break
                ranking = [{"id": r["id"], "task": r["task"], "island": r["island"],
                            "accepted": r["accepted"], "score": r["score"], "result": test_feedback(r["result"])}
                           for r in self.store.candidates()]
                self.store.set("analysis", self.support("performance_analyst", {
                    "request": "Explain failures and next experiments.", "ranking": ranking[:40],
                    "model_result": test_feedback(self.store.get("last_model_measurement"))}))
                integration = self.support("integration_agent", {"request": "Review test failures and replacement interactions.",
                    "model_result": test_feedback(self.store.get("last_model_measurement")), "pending_repairs": self.pending})
                write_json(self.directory / "integration_notes.json", integration)
                if self.reviewer and self.pending:
                    self.store.set("review", self.support("failure_reviewer", {"pending_repairs": self.pending}))
                if self.store.get("best_model_measurement") and not self.pending:
                    status = "accepted"
                    break
        except BudgetExhausted:
            status = "budget_exhausted"
        selected = self.store.get("best_model_measurement") or self.store.get("last_model_measurement")
        if self.store.get("best_model_measurement") and not self.pending and status == "generation_limit_reached":
            status = "accepted"
        result = {"status": status, "usage": self.store.usage(), "pending_repairs": self.pending,
                  "manifest": selected["manifest"] if selected else {"entries": []},
                  "model_result": selected["result"] if selected else None}
        write_json(self.directory / "manifest.json", result["manifest"])
        write_json(self.directory / "result.json", result)
        self.store.set("status", status)
        return result
