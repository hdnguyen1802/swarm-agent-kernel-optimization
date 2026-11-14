"""Verify report claims against saved evaluator and controller evidence."""
from triton_fixtures import RMS_SOURCE

import copy
import json
from pathlib import Path
import tempfile
import unittest

from sako.core import write_json
from sako.reporting import build_swarm_report, evaluation_details, write_swarm_report
from sako.store import Store


class SwarmReportingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name)
        self.store = Store(self.run / "state.sqlite")
        self.config = {"agent": {"max_total_tokens": 1_000_000, "search_mode": "swarm"},
                       "atol": .03, "rtol": .02, "trials": 5, "min_speedup": 1.02,
                       "max_p95_ratio": 1.05}
        self.store.set("task_progress", {"queue": ["rmsnorm_1", "linear_2"], "completed": []})

    def candidate(self, task="rmsnorm_1", suffix="a"):
        candidate = {"name": "candidate_" + suffix, "task_id": task, "strategy": "Test vector loads.",
                     "backend": "triton", "references": [], "source": RMS_SOURCE + "# " + suffix}
        key, _ = self.store.add(candidate, "memory", 0, None)
        return key

    def measured(self, accepted=True):
        return {"scope": "operation", "cases": [{"shapes": [[1, 128, 3584]], "correct": True,
            "inputs_unchanged": True, "checks": [{"seed": 1729, "passed": True, "max_abs_error": 0.0}],
            "baseline": {"samples_ms": [10.] * 5}, "candidate": {"samples_ms": [5.] * 5}}],
            "decision": {"accepted": accepted, "speedup": 2., "reasons": []}}

    def test_operation_gain_does_not_become_model_gain(self):
        key = self.candidate()
        self.store.record(key, self.measured())
        report = build_swarm_report(self.run, self.config, store=self.store)
        self.assertTrue(report["candidates"][0]["operation"]["accepted"])
        self.assertEqual(report["candidates"][0]["model_status"], "not_tested")
        self.assertFalse(report["accepted_model_gain"])
        self.assertEqual(report["unvisited_tasks"], ["linear_2"])
        self.assertIsNone(report["concurrency"]["max_observed_active_writers"])

    def test_model_combination_failure_does_not_assign_individual_blame(self):
        first, second = self.candidate(), self.candidate("linear_2", "b")
        for key in (first, second):
            self.store.record(key, self.measured())
        failed = {"cases": [{"correct": False, "checks": [{"seed": 1729, "passed": False,
                                                            "max_abs_error": .6}]}],
                  "decision": {"accepted": False, "reasons": ["Case 0: correctness failed."]}}
        self.store.set("model_measurements", {"one": {"generation": 0, "result": failed,
            "manifest": {"entries": [{"candidate_id": first, "task_id": "rmsnorm_1"},
                                       {"candidate_id": second, "task_id": "linear_2"}]}}})
        report = build_swarm_report(self.run, self.config, store=self.store)
        for candidate in report["candidates"]:
            self.assertEqual(candidate["model_status"], "failed_combination_only")
            self.assertIn("does not identify", candidate["model_note"])
        case = report["model_measurements"][0]["evaluation"]["cases"][0]
        self.assertEqual(case["criteria"][-1]["reason"], "Timing was skipped after a correctness failure.")

    def test_case_report_explains_speed_and_tail_failures(self):
        result = self.measured(False)
        case = result["cases"][0]
        case["candidate"]["samples_ms"] = [9.9, 9.9, 9.9, 9.9, 12.]
        result["decision"]["reasons"] = ["Case 0: gain is below 1.02x.", "Case 0: p95 latency regressed."]
        details = evaluation_details(result, self.config)
        criteria = {item["name"]: item for item in details["cases"][0]["criteria"]}
        self.assertFalse(criteria["minimum_speedup"]["passed"])
        self.assertFalse(criteria["maximum_p95_ratio"]["passed"])
        self.assertAlmostEqual(criteria["maximum_p95_ratio"]["actual"], 1.2)
        self.assertEqual(details["reasons"], result["decision"]["reasons"])
        self.assertFalse(details["gate_disagreement"])
        key = self.candidate()
        self.store.record(key, result)
        report = build_swarm_report(self.run, self.config, store=self.store)
        self.assertIn("operation acceptance gates failed", report["candidates"][0]["model_test_reason"])

    def test_saved_best_model_survives_later_rejection(self):
        key = self.candidate()
        accepted = {"manifest": {"entries": [{"candidate_id": key, "task_id": "rmsnorm_1"}]},
                    "generation": 0, "result": self.measured()}
        failed = copy.deepcopy(accepted)
        failed["generation"] = 1
        failed["result"]["decision"].update(accepted=False, reasons=["Later result failed."])
        self.store.set("model_measurements", {"earlier": accepted, "later": failed})
        self.store.set("best_model_measurement", accepted)
        self.store.set("last_model_measurement", failed)
        report = build_swarm_report(self.run, self.config, store=self.store)
        self.assertTrue(report["accepted_model_gain"])
        self.assertEqual(report["candidates"][0]["model_status"], "accepted_in_model")

    def test_event_overlap_and_stop_are_measured_not_inferred(self):
        self.store.set("swarm_events", [
            {"kind": "writer_started", "assignment_id": "a"},
            {"kind": "writer_started", "assignment_id": "b"},
            {"kind": "writer_finished", "assignment_id": "a"},
            {"kind": "writer_started", "assignment_id": "c"},
            {"kind": "writer_failed", "assignment_id": "b"},
            {"kind": "writer_finished", "assignment_id": "c"}])
        stop = {"code": "budget_reserve", "reason": "The next request cannot fit.",
                "remaining_tokens": 100, "required_tokens": 500, "decision_source": "controller"}
        self.store.set("swarm_stop", stop)
        self.store.set("status", "budget_exhausted")
        report = write_swarm_report(self.run, self.config, store=self.store)
        self.assertEqual(report["concurrency"]["max_observed_active_writers"], 2)
        self.assertEqual(report["concurrency"]["completed_writer_intervals"], 3)
        self.assertEqual(report["stop"], stop)
        self.assertEqual(json.loads((self.run / "report.json").read_text())["stop"], stop)
        text = (self.run / "report.md").read_text()
        self.assertIn("The next request cannot fit.", text)
        self.assertIn("Unknown reserved tokens: 0", text)

    def test_invalid_numbers_do_not_mutate_raw_evidence(self):
        key = self.candidate()
        raw = self.measured(False)
        raw["cases"][0]["checks"][0].update(passed=False, max_abs_error=float("inf"))
        raw["cases"][0]["correct"] = False
        self.store.record(key, raw)
        report = write_swarm_report(self.run, self.config, store=self.store)
        self.assertEqual(report["candidates"][0]["operation"]["cases"][0]["checks"][0]["max_abs_error"], "inf")
        self.assertEqual(self.store.candidates()[0]["result"]["cases"][0]["checks"][0]["max_abs_error"], float("inf"))
        json.loads((self.run / "report.json").read_text())

    def test_empty_manifest_control_is_not_a_model_gain(self):
        self.store.set("best_model_measurement", {"manifest": {"entries": []}, "result": self.measured()})
        report = build_swarm_report(self.run, self.config, store=self.store)
        self.assertFalse(report["accepted_model_gain"])

    def test_budget_closed_task_without_candidates_is_still_unvisited(self):
        self.store.set("task_progress", {"queue": ["rmsnorm_1"], "completed": [
            {"task_id": "rmsnorm_1", "rounds": 0, "reason": "task_coverage_reserve"}]})
        report = build_swarm_report(self.run, self.config, store=self.store)
        self.assertEqual(report["unvisited_tasks"], ["rmsnorm_1"])
        self.assertTrue(report["tasks"][0]["controller_closed"])

    def test_lessons_and_archives_remain_scoped_and_retain_corrections(self):
        with self.store.connect() as db:
            db.executescript("""
                CREATE TABLE swarm_lessons (id TEXT PRIMARY KEY, created REAL, payload TEXT);
                CREATE TABLE swarm_lesson_supersedes (previous_id TEXT, new_id TEXT);
            """)
            db.execute("INSERT INTO swarm_lessons VALUES(?,?,?)", ("old", 1, json.dumps(
                {"id": "old", "context": {"hardware_fingerprint": "gpu-a"}, "status": "failed"})))
            db.execute("INSERT INTO swarm_lessons VALUES(?,?,?)", ("new", 2, json.dumps(
                {"id": "new", "context": {"hardware_fingerprint": "gpu-a"}, "status": "supported"})))
            db.execute("INSERT INTO swarm_lesson_supersedes VALUES(?,?)", ("old", "new"))
        self.store.set("swarm_archive:scope-a", {"context": {"hardware_fingerprint": "gpu-a"}, "limit": 4})
        report = build_swarm_report(self.run, self.config, store=self.store)
        self.assertEqual(len(report["lessons"]), 2)
        self.assertEqual(report["lessons"][0]["superseded_by"], ["new"])
        self.assertEqual(report["candidate_archives"]["swarm_archive:scope-a"]["limit"], 4)

    def test_model_diagnostics_and_repair_stop_reasons_are_retained(self):
        diagnosis = {"status": "available", "scope": "eager_boundary_trace",
                     "note": "This trace does not identify the first compiled divergence.",
                     "artifacts": [{"path": "diagnostics/trace.json", "sha256": "abc"}]}
        self.store.set("swarm_diagnostics", {"rmsnorm_1": diagnosis})
        self.store.set("repair_exhausted:candidate-a", {"candidate_id": "candidate-a", "repairs_used": 3,
                                                       "stage": "model", "task_id": "rmsnorm_1"})
        self.store.set("closed_task_repairs", {"candidate-b": {"closed_reason": "coordinator_move_on"}})
        report = write_swarm_report(self.run, self.config, store=self.store)
        self.assertEqual(report["model_diagnostics"]["rmsnorm_1"], diagnosis)
        self.assertEqual(report["exhausted_repairs"]["candidate-a"]["repairs_used"], 3)
        self.assertEqual(report["closed_task_repairs"]["candidate-b"]["closed_reason"], "coordinator_move_on")

    def test_closed_hypothesis_does_not_claim_that_task_closed(self):
        hypotheses = {"failed-method": {"id": "failed-method", "task_id": "rmsnorm_1", "status": "closed",
                                        "close_reason": "Two correction attempts failed."},
                      "active-method": {"id": "active-method", "task_id": "rmsnorm_1", "status": "active"}}
        self.store.set("swarm_hypotheses", hypotheses)
        report = write_swarm_report(self.run, self.config, store=self.store)
        self.assertEqual(report["hypotheses"], hypotheses)
        self.assertFalse(report["tasks"][0]["controller_closed"])
        self.assertEqual(report["task_progress"]["completed"], [])
        self.assertIn("Two correction attempts failed.", (self.run / "report.md").read_text())

    def test_replaced_layer_accuracy_does_not_claim_final_output_equivalence(self):
        self.config.update(correctness_scope="replaced_layers", correctness_reference="eager", model_reference="torch_compile")
        key = self.candidate()
        measured = self.measured()
        measured.update(correctness_scope="replaced_layers", correctness_reference="eager",
                        model_reference="torch_compile", output_equivalence="not_tested", scope="model")
        measured["cases"][0]["checks"] = [{"seed": 1729, "passed": True, "scope": "replaced_layers",
            "expected_paths": ["model.norm"], "observed_paths": ["model.norm"], "missing_paths": [],
            "layers": [{"path": "model.norm", "candidate_id": key, "kind": "rmsnorm", "call_index": 0,
                        "passed": True, "inputs_unchanged": True, "check": {"passed": True, "max_abs_error": 0.}}]}]
        record = {"manifest": {"entries": [{"candidate_id": key, "task_id": "rmsnorm_1"}]},
                  "generation": 0, "result": measured}
        self.store.set("model_measurements", {"test": record})
        self.store.set("best_model_measurement", record)
        report = write_swarm_report(self.run, self.config, store=self.store)
        self.assertEqual(report["acceptance_claim"], "Replaced-layer accuracy and whole-model speed passed.")
        self.assertEqual(report["validation_policy"]["final_output_equivalence"], "not_tested")
        evaluation = report["model_measurements"][0]["evaluation"]
        self.assertEqual(evaluation["validation_policy"]["accuracy_reference"], "eager")
        self.assertEqual(evaluation["validation_policy"]["speed_reference"], "torch_compile")
        self.assertEqual(evaluation["cases"][0]["layer_coverage"][0]["observed_paths"], ["model.norm"])
        self.assertIn("Final-output numerical equivalence: not_tested", (self.run / "report.md").read_text())

    def test_layer_rejection_keeps_missing_paths_and_failure_examples(self):
        result = self.measured(False)
        result.update(correctness_scope="replaced_layers", correctness_reference="eager", scope="model")
        result["cases"][0]["correct"] = False
        result["cases"][0]["checks"] = [{"seed": 1729, "passed": False, "scope": "replaced_layers",
            "expected_paths": ["model.layer0", "model.layer1"], "observed_paths": ["model.layer0"],
            "missing_paths": ["model.layer1"], "layers": [{"path": "model.layer0", "candidate_id": "candidate",
                "call_index": 0, "passed": False, "inputs_unchanged": True,
                "check": {"passed": False, "max_abs_error": .6, "examples": [{"actual": 1.6, "expected": 1.}]}}]}]
        details = evaluation_details(result, self.config, scope="model")
        coverage = details["cases"][0]["layer_coverage"][0]
        self.assertEqual(coverage["missing_paths"], ["model.layer1"])
        self.assertEqual(coverage["failed_layers"][0]["check"]["max_abs_error"], .6)
        self.assertEqual(details["validation_policy"]["final_output_equivalence"], "not_tested")


if __name__ == "__main__":
    unittest.main()
