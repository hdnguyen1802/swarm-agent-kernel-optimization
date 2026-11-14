"""Targets come from the manifest, never from a favorable timing result."""
import copy
import unittest

from sako.model_acceptance import score_model_cases, target_workloads
from sako.shape_selection import case_signature
from sako.reporting import evaluation_details


class ModelAcceptanceScopeTests(unittest.TestCase):
    def setUp(self):
        self.workloads = [{"batch_size": 1, "sequence_length": 128},
                          {"batch_size": 1, "sequence_length": 512}]
        self.config = {"workloads": self.workloads, "model_speed_scope": "targeted_workloads",
                       "min_speedup": 1.02, "trials": 5, "max_p95_ratio": 1.05}
        self.task = {"id": "linear", "paths": ["layers.0.proj"], "cases": [
            {"shapes": [[1, w["sequence_length"], 16], [16, 16]],
             "strides": [[w["sequence_length"] * 16, 16, 1], [16, 1]],
             "dtypes": ["torch.float16", "torch.float16"], "workload": w}
            for w in self.workloads]}
        self.manifest = {"entries": [{"task_id": "linear", "candidate_id": "candidate",
            "paths": self.task["paths"], "signatures": [case_signature(self.task["cases"][0])]}]}

    def cases(self, target=9.5, other=10.):
        return [{"workload": w, "correct": True, "baseline": {"samples_ms": [10.] * 5},
                 "candidate": {"samples_ms": [value] * 5}}
                for w, value in zip(self.workloads, [target, other])]

    def test_scope_is_frozen_from_manifest_and_catalog(self):
        targets = target_workloads(self.config, self.manifest, {"tasks": [self.task]})
        self.assertEqual(targets, self.workloads[:1])
        result = score_model_cases(self.cases(), self.config, targets)
        self.assertTrue(result["accepted"])
        self.assertAlmostEqual(result["target_speedup"], 10 / 9.5)
        self.assertLess(result["speedup"], result["target_speedup"])

    def test_any_positive_gain_passes_but_equal_speed_or_wrong_answers_do_not(self):
        config = {**self.config, "min_speedup": 1.0}
        self.assertTrue(score_model_cases(self.cases(target=9.999), config, self.workloads[:1])["accepted"])
        self.assertFalse(score_model_cases(self.cases(target=10.), config, self.workloads[:1])["accepted"])
        wrong = self.cases(target=9.)
        wrong[0]["correct"] = False
        self.assertFalse(score_model_cases(wrong, config, self.workloads[:1])["accepted"])

    def test_non_target_regression_still_rejects(self):
        result = score_model_cases(self.cases(other=10.01), self.config, self.workloads[:1])
        self.assertFalse(result["accepted"])
        self.assertTrue(any("1.0x" in x for x in result["reasons"]))

    def test_target_gain_cannot_be_replaced_by_a_gain_elsewhere(self):
        result = score_model_cases(self.cases(target=10., other=8.), self.config, self.workloads[:1])
        self.assertFalse(result["accepted"])

    def test_missing_target_or_workload_rejects(self):
        for cases, targets in [(self.cases(), []), (self.cases()[:1], self.workloads[:1]),
                               (self.cases()[::-1], self.workloads[:1])]:
            with self.subTest(cases=cases, targets=targets):
                self.assertFalse(score_model_cases(cases, self.config, targets)["accepted"])

    def test_targeted_scope_keeps_correctness_and_p95_gates(self):
        cases = self.cases()
        cases[1]["correct"] = False
        self.assertFalse(score_model_cases(cases, self.config, self.workloads[:1])["accepted"])
        cases = self.cases()
        cases[0]["candidate"]["samples_ms"][-1] = 20.
        self.assertFalse(score_model_cases(cases, self.config, self.workloads[:1])["accepted"])

    def test_legacy_scope_still_requires_gain_everywhere(self):
        config = {**self.config, "model_speed_scope": "all_workloads"}
        self.assertEqual(target_workloads(config, self.manifest, {"tasks": [self.task]}), self.workloads)
        self.assertFalse(score_model_cases(self.cases(), config, self.workloads)["accepted"])

    def test_unconditional_replacement_targets_every_workload(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["entries"][0].pop("signatures")
        self.assertEqual(target_workloads(self.config, manifest, {"tasks": [self.task]}), self.workloads)

    def test_missing_workload_mapping_cannot_choose_a_target(self):
        self.task["cases"][0].pop("workload")
        with self.assertRaisesRegex(ValueError, "fresh profile"):
            target_workloads(self.config, self.manifest, {"tasks": [self.task]})

    def test_report_preserves_target_and_non_target_thresholds(self):
        cases = self.cases()
        for i, case in enumerate(cases):
            case["speed_requirement"] = 1.02 if i == 0 else 1.
        decision = score_model_cases(cases, self.config, self.workloads[:1])
        report = evaluation_details({"decision": decision, "cases": cases}, self.config, scope="model")
        self.assertFalse(report["gate_disagreement"])
        self.assertEqual(report["target_speed_ratio"], decision["target_speedup"])
        thresholds = [[x["required"] for x in c["criteria"] if x["name"] == "minimum_speedup"]
                      for c in report["cases"]]
        self.assertEqual(thresholds, [[1.02], [1.]])


if __name__ == "__main__":
    unittest.main()
