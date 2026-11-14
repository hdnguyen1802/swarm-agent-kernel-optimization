"""Keep layer failure identity and coverage in bounded writer feedback."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from sako.feedback import compact_feedback, save_feedback
from sako.loop import SearchLoop


def layer_result():
    paths = [f"model.layers.{index}.input_layernorm" for index in range(57)]
    cases = []
    for workload in ({"batch_size": 1, "sequence_length": 128},
                     {"batch_size": 1, "sequence_length": 512},
                     {"batch_size": 2, "sequence_length": 128}):
        checks = []
        for offset, seed in enumerate((1729, 2718, 3141)):
            layers = []
            for index, path in enumerate(paths):
                passed = index not in (0, 56)
                layers.append({"path": path, "candidate_id": "a" * 24, "kind": "rmsnorm", "call_index": 0,
                    "passed": passed, "inputs_unchanged": True,
                    "check": {"passed": passed, "max_abs_error": .001 if passed else .6 + offset * .01,
                              "atol": .03, "rtol": .02, "mismatched_elements": 0 if passed else 100,
                              "examples": [{"flat_index": n, "actual": 1.6, "expected": 1.,
                                            "allowed_abs_error": .05} for n in range(100)]}})
            checks.append({"scope": "replaced_layers", "seed": seed, "passed": False,
                           "expected_paths": paths, "observed_paths": paths, "missing_paths": [], "layers": layers})
        cases.append({"workload": workload, "correct": False, "reference": "eager_replaced_layers",
                      "timing_reference": "torch_compile", "checks": checks})
    return {"scope": "complete_model_forward", "compiled": True, "correctness_scope": "replaced_layers",
            "correctness_reference": "eager", "model_reference": "torch_compile", "output_equivalence": "not_tested",
            "cases": cases, "decision": {"accepted": False, "reasons": ["Case 0: correctness failed."]}}


class LayerFeedbackTests(unittest.TestCase):
    def test_all_seed_summaries_keep_failed_paths_and_worst_layer_errors(self):
        raw = layer_result()
        before = copy.deepcopy(raw)
        feedback = SearchLoop.model_feedback(raw)
        self.assertEqual(raw, before)
        self.assertEqual(feedback["correctness_reference"], "eager")
        self.assertEqual(feedback["model_reference"], "torch_compile")
        self.assertEqual(feedback["output_equivalence"], "not_tested")
        self.assertEqual(len(feedback["cases"]), 3)
        for case in feedback["cases"]:
            for check in case["checks"]:
                self.assertEqual(check["expected_path_count"], 57)
                self.assertEqual(check["observed_path_count"], 57)
                self.assertEqual(check["missing_path_count"], 0)
                self.assertEqual(check["failed_layer_count"], 2)
                self.assertEqual(len(check["failed_layers"]), 2)
                self.assertEqual({layer["path"] for layer in check["failed_layers"]},
                                 {"model.layers.0.input_layernorm", "model.layers.56.input_layernorm"})
                for layer in check["failed_layers"]:
                    self.assertEqual(layer["candidate_id"], "a" * 24)
                    self.assertEqual(layer["atol"], .03)
                    self.assertLessEqual(len(layer.get("examples", [])), 2)
        self.assertLess(len(json.dumps(feedback)), 14000)
        self.assertEqual(SearchLoop.model_feedback(feedback), feedback)

    def test_missing_coverage_is_visible_and_unknown_error_stays_unknown(self):
        raw = layer_result()
        raw["cases"][0]["checks"][0]["missing_paths"] = ["model.norm"]
        raw["cases"][0]["checks"][0]["layers"][0]["check"]["max_abs_error"] = None
        summary = SearchLoop.model_feedback(raw)
        check = summary["cases"][0]["checks"][0]
        self.assertEqual(check["missing_paths"], ["model.norm"])
        self.assertEqual(check["missing_path_count"], 1)
        self.assertIsNone(check["max_abs_error"])
        self.assertIsNone(check["failed_layers"][0]["max_abs_error"])

    def test_generic_old_feedback_with_omission_markers_cannot_crash(self):
        old = compact_feedback({"combination": layer_result(), "individual": layer_result()})
        summary = SearchLoop.model_feedback(old)
        check = summary["cases"][0]["checks"][0]
        self.assertEqual(check["expected_path_count"], 57)
        self.assertEqual(check["layer_check_count"], 57)
        self.assertTrue(check["failed_layer_count_is_minimum"])
        self.assertTrue(check["failed_layers"][0]["numerical_details_omitted"])
        self.assertIsNone(check["max_abs_error"])

    def test_queue_saves_typed_layer_errors_and_preserves_full_raw_evidence(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        loop = SearchLoop.__new__(SearchLoop)
        loop.store = {"repairs_used:candidate": 0}
        loop.pending, loop.owners, loop.blocked = {}, {}, {}
        loop.limits = {"max_repair_attempts": 3}
        loop.row = lambda _: {"task": "rmsnorm_1"}
        loop.evidence_report = lambda value, name, **metadata: save_feedback(directory, value, name=name, metadata=metadata)
        loop.persist_repairs = lambda: None
        raw = {"combination": layer_result(), "individual": layer_result()}
        before = copy.deepcopy(raw)
        loop.queue(0, "candidate", "model", raw, 0)
        queued = loop.pending["candidate"]
        feedback = SearchLoop.repair_feedback(queued)
        self.assertEqual(raw, before)
        self.assertEqual(feedback["result"]["correctness_reference"], "eager")
        self.assertEqual(feedback["result"]["model_reference"], "torch_compile")
        self.assertAlmostEqual(feedback["result"]["max_abs_error"], .62)
        self.assertEqual(feedback["result"]["failed_layers"][0]["candidate_id"], "a" * 24)
        self.assertEqual(feedback["result"]["failed_checks"], 9)
        self.assertLess(len(json.dumps(feedback)), 2500)
        saved = json.loads((directory / queued["evidence"]["path"]).read_text())
        self.assertEqual(saved, raw)
        self.assertEqual(len(saved["combination"]["cases"][0]["checks"][0]["layers"]), 57)

    def test_unsafe_reference_state_keeps_skipped_check_and_timing_reasons(self):
        raw = layer_result()
        raw["reference_state_intact"] = False
        case = raw["cases"][0]
        case.update(timing_status="not_measured", timing_reason="The reference state could not be restored.")
        check = case["checks"][0]
        check.update(reference_state_intact=False, not_measured=True,
                     reason="An earlier replacement changed input metadata.", layers=[])
        summary = SearchLoop.model_feedback(raw)
        self.assertFalse(summary["reference_state_intact"])
        self.assertEqual(summary["cases"][0]["timing_reason"], case["timing_reason"])
        self.assertTrue(summary["cases"][0]["checks"][0]["not_measured"])
        self.assertEqual(summary["cases"][0]["checks"][0]["reason"], check["reason"])


if __name__ == "__main__":
    unittest.main()
