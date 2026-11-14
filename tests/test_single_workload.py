"""A declared workload controls evaluation and bounds every performance claim."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from sako.core import load_config
from sako.model import measure_model
from sako.model_acceptance import score_model_cases
from sako.preparation import validate_catalog
from sako.reporting import build_swarm_report, evaluation_details
from sako.store import Store
import test_layer_evaluation as fixtures


ROOT = Path(__file__).resolve().parents[1]


class SingleWorkloadScopeTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "configs/project.json")
        self.config["measurement"] = {"mode": "fixed"}
        self.workload = self.config["workloads"][0]

    def case(self, workload=None, latency=9.5):
        return {"workload": workload or self.workload, "correct": True,
                "baseline": {"samples_ms": [10.] * self.config["trials"]},
                "candidate": {"samples_ms": [latency] * self.config["trials"]}}

    def test_only_the_declared_target_is_needed_for_model_acceptance(self):
        self.assertEqual(self.config["workloads"], [{"batch_size": 1, "sequence_length": 128}])
        result = score_model_cases([self.case()], self.config, [self.workload])
        self.assertTrue(result["accepted"])
        self.assertEqual(result["workload_scope"], "single_workload")
        self.assertEqual(result["declared_workloads"], [self.workload])
        self.assertIn("No performance claim for other workloads", result["note"])

    def test_absent_duplicate_or_unrelated_measurement_cannot_pass(self):
        unrelated = {"batch_size": 1, "sequence_length": 512}
        for mode in ("all_workloads", "targeted_workloads"):
            config = {**self.config, "model_speed_scope": mode}
            for cases in ([], [self.case(unrelated)], [self.case(), self.case()],
                          [self.case(), self.case(unrelated)]):
                with self.subTest(mode=mode, cases=cases):
                    result = score_model_cases(cases, config, [self.workload])
                    self.assertFalse(result["accepted"])
                    self.assertTrue(any("Every configured workload" in r for r in result["reasons"]))

    def test_accuracy_gain_and_tail_limits_still_apply(self):
        wrong = self.case()
        wrong["correct"] = False
        tail = self.case()
        tail["candidate"]["samples_ms"][-1] = 20.
        for case in (wrong, tail, self.case(latency=10.)):
            with self.subTest(case=case):
                self.assertFalse(score_model_cases([case], self.config, [self.workload])["accepted"])

    def test_a_different_declared_workload_can_be_optimized_separately(self):
        other = {"batch_size": 2, "sequence_length": 128}
        config = {**self.config, "workloads": [other]}
        result = score_model_cases([self.case(other)], config, [other])
        self.assertTrue(result["accepted"])
        self.assertEqual(result["declared_workloads"], [other])

    def test_old_multi_workload_profile_cannot_be_reused(self):
        old = copy.deepcopy(self.config)
        old["workloads"].append({"batch_size": 1, "sequence_length": 512})
        hardware = {"fingerprint": "test-gpu"}
        with self.assertRaisesRegex(ValueError, "workloads"):
            validate_catalog(self.config, {"config": old, "hardware": hardware}, hardware)

    def test_report_records_target_without_relabelling_old_results(self):
        measured = {"cases": [self.case()], "decision": {"accepted": True, "speedup": 10 / 9.5}}
        policy = evaluation_details(measured, self.config, scope="model")["validation_policy"]
        self.assertEqual(policy["declared_workloads"], [self.workload])
        self.assertEqual(policy["workload_scope"], "single_workload")
        measured["cases"].append(self.case({"batch_size": 1, "sequence_length": 512}))
        policy = evaluation_details(measured, self.config, scope="model")["validation_policy"]
        self.assertEqual(policy["workload_scope"], "multiple_workloads")
        self.assertEqual(len(policy["declared_workloads"]), 2)

    def test_old_report_keeps_its_saved_workloads_after_default_changes(self):
        old = copy.deepcopy(self.config)
        old["workloads"].append({"batch_size": 1, "sequence_length": 512})
        result = {"config": old, "cases": [self.case(w) for w in old["workloads"]],
                  "decision": {"accepted": True, "speedup": 10 / 9.5}}
        trial = {"manifest": {"entries": [{"task_id": "linear", "candidate_id": "old"}]}, "result": result}
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "state.sqlite")
            store.set("model_measurements", {"old": trial})
            store.set("best_model_measurement", trial)
            report = build_swarm_report(directory, self.config, store=store)
        self.assertEqual(report["validation_policy"]["workload_scope"], "multiple_workloads")
        self.assertEqual(report["limits"]["workloads"], old["workloads"])
        self.assertNotIn("single declared workload", report["acceptance_claim"])


class SingleWorkloadModelTests(unittest.TestCase):
    setUp = fixtures.LayerEvaluationTests.setUp
    environment = fixtures.LayerEvaluationTests.environment
    manifest = fixtures.LayerEvaluationTests.manifest

    def test_model_checks_all_seeds_and_replaced_layers_then_times_one_workload(self):
        self.config = load_config(ROOT / "configs/project.json")
        self.config["measurement"] = {"mode": "fixed"}
        manifest, catalog = self.manifest(28)
        model, alternative = fixtures.Model(28), fixtures.Model(28)
        self.torch.compile = Mock(side_effect=lambda fn, **kwargs: fn)
        visited = []
        def operation(tensors, scalars):
            visited.append(tensors[1].value)
            return fixtures.Tensor(tensors[0].value * tensors[1].value + .001)
        def install(model, manifest, catalog, root, *, operations, config=None):
            operations["candidate"] = operation
        def timings(baseline, candidate, config, repeats):
            baseline()
            candidate()
            return {"baseline": {"samples_ms": [10.] * config["trials"]},
                    "candidate": {"samples_ms": [9.5] * config["trials"]}}
        with self.environment(), patch("sako.model.load_model", return_value=model), \
                patch("sako.model.clone_structure", return_value=alternative), \
                patch("sako.model.apply_manifest", side_effect=install), \
                patch("sako.model.paired_timings", side_effect=timings) as timed, \
                patch("sako.model.check_output", side_effect=AssertionError("Use eager layer checks.")):
            result = measure_model(self.config, manifest, catalog, "unused")
        self.assertTrue(result["decision"]["accepted"])
        self.assertEqual(len(result["cases"]), 1)
        self.assertEqual([c["seed"] for c in result["cases"][0]["checks"]], [1729, 2718, 3141])
        self.assertEqual(len(visited), 28 * 3)
        self.assertEqual(len(set(visited)), 28)
        timed.assert_called_once()
        self.assertEqual(self.torch.compile.call_count, 2)
        self.assertTrue(all(c.kwargs["fullgraph"] for c in self.torch.compile.call_args_list))
        self.assertEqual(result["correctness_reference"], "eager")
        self.assertEqual(result["model_reference"], "torch_compile")
        self.assertEqual(result["output_equivalence"], "not_tested")


if __name__ == "__main__":
    unittest.main()
