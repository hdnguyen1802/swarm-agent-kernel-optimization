"""Exercise incremental keeps, replay safety, and deferred integration."""

import copy
from pathlib import Path
import unittest

import test_swarm_loop as fixtures
from sako.core import candidate_id, digest, implementation_hash, read_json, score_cases, write_json
from sako.integration import integrate
from sako.kernel_selection import keep_kernel, kernel_config, search_score
from sako.reporting import evaluation_details


class PairedWorker(fixtures.SwarmWorker):
    def __init__(self):
        super().__init__()
        self.latencies = [8., 7.96, 7.9, 7.95]
        self.baselines = []

    def run(self, action, run_dir, **fields):
        if action != "evaluate":
            return super().run(action, run_dir, **fields)
        self.calls.append((action, fields))
        root = Path(__file__).resolve().parents[1]
        source = read_json(root / fields["candidate"])
        def trial_index(item):
            # Writer labels survive the harness source preparation. Manually
            # supplied repair fixtures retain their original comment label.
            return (int(item["name"].rsplit("_", 1)[-1]) if item["name"].startswith("kernel_")
                    else int(item["source"].rsplit(" ", 1)[-1]))
        index = trial_index(source)
        baseline = read_json(root / fields["baseline_candidate"]) if fields.get("baseline_candidate") else None
        baseline_index = trial_index(baseline) if baseline else None
        elapsed = self.latencies[baseline_index] if baseline else 10.
        self.baselines.append(baseline_index)
        cases = [{"correct": True, "inputs_unchanged": True,
                  "baseline": {"samples_ms": [elapsed] * 15},
                  "candidate": {"samples_ms": [self.latencies[index]] * 15}}]
        return {"scope": "operation", "candidate_id": candidate_id(source), "cases": cases,
                "kernel_min_speedup": 1.01, "acceptance_scope": "kernel_only",
                "timing_baseline": {"kind": "kept_kernel" if baseline else "torch_compile",
                    "candidate_id": candidate_id(baseline) if baseline else None},
                "decision": score_cases(cases, {"trials": 15, "min_speedup": 1.01, "max_p95_ratio": 1.05})}


class KernelFirstTests(unittest.TestCase):
    loop = fixtures.SwarmLoopTests.loop
    writers = fixtures.SwarmLoopTests.writers

    def setUp(self):
        fixtures.SwarmLoopTests.setUp(self)
        self.config["kernel_search"] = {"enabled": True, "min_speedup": 1.01}
        self.config["agent"]["max_task_experiments"] = 4
        self.worker = PairedWorker()
        self.catalog_path = self.directory / "profile/tasks.json"
        catalog = read_json(self.catalog_path)
        catalog["tasks"][0]["cases"] = [{}]
        catalog["config"] = self.config
        write_json(self.catalog_path, catalog)

    def search(self):
        result = self.loop().run()
        self.assertEqual(result["status"], "completed_kernel_search")
        self.assertEqual(self.worker.baselines, [None, 0, 0, 2])
        self.assertEqual([a for a, _ in self.worker.calls], ["evaluate"] * 4)
        return result

    def test_races_current_keep_and_defers_all_model_work(self):
        result = self.search()
        loop = self.loop()
        rows = sorted(loop.store.candidates(), key=lambda row: row["generation"])
        self.assertEqual([bool(row["accepted"]) for row in rows], [True, False, True, False])
        self.assertEqual(result["manifest"]["entries"][0]["candidate_id"], rows[2]["id"])
        self.assertEqual(result["integration_status"], "pending")
        self.assertIsNone(result["model_result"])
        self.assertGreater(rows[2]["score"], rows[0]["score"])
        self.assertAlmostEqual(rows[2]["result"]["decision"]["speedup"], 8 / 7.9)
        self.assertAlmostEqual(rows[2]["score"], 10 / 7.9)
        self.assertEqual(evaluation_details(rows[2]["result"], self.config)["gate_disagreement"], False)
        # A duplicate old winner cannot replace the current winner during replay.
        task = loop.by_id["rmsnorm_1"]
        self.assertFalse(keep_kernel(loop.store, task, rows[0]["id"], self.config))
        before = len(self.worker.calls)
        self.loop().run()
        self.assertEqual(len(self.worker.calls), before)

    def test_funded_revisit_requires_a_new_measurement_and_survives_stop(self):
        self.config["agent"].update(max_task_experiments=3, require_archive_trial=False)
        rejected_old_citation = []
        def plan(data, number):
            parents = data["archive_policy"]["eligible_parent_ids"]
            parent = parents[0] if parents else None
            evidence = [parent or "contract:" + data["task"]["id"]]
            if "move_on" in data["allowed_actions"]:
                latest = max(data["operation_results"], key=lambda row: row["generation"])["id"]
                if data["progress"].get("budget_visits"):
                    if data.get("validation_error"):
                        rejected_old_citation.append(data["validation_error"])
                    else:
                        latest = min(data["operation_results"], key=lambda row: row["generation"])["id"]
                return {"action": "move_on", "reason": "Review another task.", "evidence": [latest],
                        "archive_skip": {"reason": "The latest trial has been measured.", "evidence": [parent]},
                        "hypotheses": []}
            return {"action": "continue", "reason": "Test the next design.", "evidence": evidence,
                    "hypotheses": [{"specialty": "memory", "parent_id": parent, "claim": f"Reduce reads {number}.",
                    "change": f"Use mapping {number}.", "expected_metric": "Lower latency.", "evidence": evidence}]}
        self.client.plan_callback = plan
        loop = self.loop()
        stop = loop.directory / "STOP"
        self.client.writer_callback = lambda data: stop.touch() if data["generation"] == 1 else None
        self.assertEqual(loop.run()["status"], "stopped")
        self.assertEqual(len(self.worker.baselines), 2)
        self.assertEqual(loop.progress["budget_visits"][loop.active_task]["first_generation"], 1)
        stop.unlink()
        result = self.loop().run()
        self.assertEqual(result["status"], "completed_kernel_search")
        self.assertEqual(self.worker.baselines, [None, 0, 0])
        self.assertEqual(result["task_progress"]["budget_visits"]["rmsnorm_1"]["count"], 2)
        self.assertTrue(any("measured candidate from this revisit" in reason for reason in rejected_old_citation))

    def test_end_of_pass_requires_budget_for_writer_and_review(self):
        from sako.store import BudgetExhausted
        loop = self.loop()
        loop.advance("First pass finished.", 0)
        reservation = loop.store.reserve("coordinator", loop.limits["max_total_tokens"] - 1, loop.limits)
        loop.store.finish_call(reservation, loop.limits["max_total_tokens"] - 1)
        with self.assertRaises(BudgetExhausted):
            loop.revisit_with_budget()
        self.assertIsNone(loop.active_task)

    def test_integration_failure_keeps_kernel_wins_and_caches_the_result(self):
        result = self.search()
        kept = copy.deepcopy(result["kernel_baselines"])
        self.worker.model_correct = False
        measured = integrate(self.config, self.directory / "run", self.directory / "profile", self.worker)
        self.assertFalse(measured["result"]["decision"]["accepted"])
        loop = self.loop()
        self.assertEqual(loop.store.get("kernel_baselines"), kept)
        self.assertIsNone(loop.store.get("best_model_measurement"))
        self.assertEqual(loop.store.get("blocked_model_candidates", {}), {})
        before = len(self.worker.calls)
        integrate(self.config, self.directory / "run", self.directory / "profile", self.worker)
        self.assertEqual(len(self.worker.calls), before)

    def test_only_later_model_success_sets_model_acceptance(self):
        self.search()
        self.worker.model_correct = True
        integrate(self.config, self.directory / "run", self.directory / "profile", self.worker)
        final = read_json(self.directory / "run/result.json")
        self.assertEqual(final["status"], "accepted")
        self.assertEqual(final["integration_status"], "accepted")
        self.assertEqual(sum(a == "model" for a, _ in self.worker.calls), 1)

    def test_kernel_gate_keeps_all_case_accuracy_p95_and_trial_rules(self):
        config = kernel_config(self.config)
        self.assertEqual(config["min_speedup"], 1.01)
        self.assertEqual(self.config["min_speedup"], 1.0)
        case = {"correct": True, "baseline": {"samples_ms": [101.] * 15},
                "candidate": {"samples_ms": [100.] * 15}}
        self.assertTrue(score_cases([case], config)["accepted"])
        for changed in ({"correct": False}, {"candidate": {"samples_ms": [100.1] * 15}},
                        {"candidate": {"samples_ms": [100.] * 14}},
                        {"candidate": {"samples_ms": [100.] * 14 + [200.]}}):
            self.assertFalse(score_cases([{**case, **changed}], config)["accepted"])
        self.assertFalse(score_cases([case, {**case, "correct": False}], config)["accepted"])

    def test_integration_rejects_changed_contract_and_active_search(self):
        self.search()
        loop = self.loop()
        loop.store.set("status", "running")
        with self.assertRaisesRegex(ValueError, "Stop the kernel search"):
            integrate(self.config, self.directory / "run", self.directory / "profile", self.worker)
        loop.store.set("status", "completed_kernel_search")
        config = {**self.config, "atol": .9}
        with self.assertRaisesRegex(ValueError, "matching kernel-first"):
            integrate(config, self.directory / "run", self.directory / "profile", self.worker)


if __name__ == "__main__":
    unittest.main()
