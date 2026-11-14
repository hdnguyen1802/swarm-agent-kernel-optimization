"""Keep containing-region failures usable in bounded writer repair prompts."""

import copy
import json
import unittest

from sako.loop import SearchLoop


EXAMPLE = {"flat_index": 1729, "actual": .7783203125, "expected": .8251953125,
           "allowed_abs_error": .046503905206918716}


def region_failure():
    linear_paths = [f"model.layers.{i}.mlp.{projection}" for i in range(28)
                    for projection in ("gate_proj", "up_proj")]
    region_paths = [f"model.layers.{i}.mlp" for i in range(28)]

    def layer(path, passed, kind):
        numerical = {"passed": passed, "max_abs_error": .0625 if passed else 4.,
                     "mismatched_elements": 0 if passed else 1, "atol": .03, "rtol": .02}
        if not passed:
            numerical["examples"] = [dict(EXAMPLE) for _ in range(100)]
        return {"path": path, "candidate_id": "region:" + path if kind == "mlp" else "candidate",
                "kind": kind, "call_index": 0, "passed": passed, "same_input": True,
                "inputs_unchanged": True, "reference": "eager", "input_shapes": [[1, 128, 3584]],
                "input_strides": [[458752, 3584, 1]], "input_dtypes": ["torch.float16"], "check": numerical}

    region = {"seed": 1729, "scope": "replaced_layers", "reference": "eager", "passed": False,
              "reference_state_intact": True, "expected_paths": region_paths, "observed_paths": region_paths,
              "missing_paths": [], "failed_paths": [region_paths[26]], "max_abs_error": 4.,
              "mismatched_elements": 1,
              "layers": [layer(path, i != 26, "mlp") for i, path in enumerate(region_paths)]}
    check = {"seed": 1729, "scope": "replaced_layers", "reference": "eager", "passed": False,
             "reference_state_intact": True, "expected_paths": linear_paths, "observed_paths": linear_paths,
             "missing_paths": [], "failed_paths": [], "max_abs_error": .0625, "mismatched_elements": 0,
             "layers": [layer(path, True, "linear") for path in linear_paths], "region_checks": region}
    return {"scope": "complete_model_forward", "compiled": True,
            "correctness_scope": "replaced_layers", "correctness_reference": "eager",
            "model_reference": "torch_compile", "output_equivalence": "not_tested",
            "reference_state_intact": True,
            "decision": {"accepted": False, "speedup": None, "reasons": ["Case 0: correctness failed."]},
            "cases": [{"workload": {"batch_size": 1, "sequence_length": 128}, "correct": False,
                       "checks": [check], "timing_status": "not_measured",
                       "timing_reason": "A required layer check failed."}]}


def repair(raw):
    return {"candidate_id": "candidate", "task_id": "gate_and_up", "stage": "model", "slot": 0,
            "repairs_used": 1, "evidence": {"path": "evidence/model-failure.json", "sha256": "a" * 64},
            "result": {"combination": raw, "individual": raw}}


class RegionRepairFeedbackTests(unittest.TestCase):
    def test_model_summary_preserves_nested_failure_and_exact_examples(self):
        raw = region_failure()
        original = copy.deepcopy(raw)
        feedback = SearchLoop.model_feedback(raw)
        check = feedback["cases"][0]["checks"][0]
        self.assertEqual(check["failed_layer_count"], 0)
        nested = check["region_checks"]
        self.assertFalse(nested["passed"])
        self.assertEqual(nested["failed_layer_count"], 1)
        failed = nested["failed_layers"][0]
        self.assertEqual(failed["path"], "model.layers.26.mlp")
        self.assertEqual(failed["examples"], [EXAMPLE, EXAMPLE])
        self.assertTrue(failed["same_input"])
        self.assertTrue(failed["inputs_unchanged"])
        self.assertEqual(failed["reference"], "eager")
        self.assertEqual(failed["input_shapes"], [[1, 128, 3584]])
        self.assertEqual(raw, original)
        self.assertLess(len(json.dumps(feedback)), 4000)

    def test_compacting_an_existing_summary_keeps_the_same_region_evidence(self):
        feedback = SearchLoop.model_feedback(region_failure())
        self.assertEqual(SearchLoop.model_feedback(feedback), feedback)
        first = SearchLoop.repair_feedback(repair(region_failure()))
        second = SearchLoop.repair_feedback(repair(feedback))
        self.assertEqual(first, second)

    def test_repair_prompt_selects_containing_region_instead_of_passing_linears(self):
        raw = region_failure()
        pending = repair(raw)
        original = copy.deepcopy(pending)
        feedback = SearchLoop.repair_feedback(pending)
        failure = feedback["result"]
        self.assertEqual(failure["failure_boundary"], "containing_region")
        self.assertEqual(failure["seed"], 1729)
        self.assertEqual(failure["workload"], {"batch_size": 1, "sequence_length": 128})
        self.assertEqual(failure["failed_layers"][0]["path"], "model.layers.26.mlp")
        self.assertEqual(failure["failed_layers"][0]["examples"][0], EXAMPLE)
        self.assertEqual(failure["correctness_reference"], "eager")
        self.assertEqual(failure["model_reference"], "torch_compile")
        self.assertEqual(failure["failed_checks"], 1)
        self.assertEqual(feedback["evidence"], pending["evidence"])
        self.assertEqual(feedback["repairs_used"], 1)
        self.assertEqual(pending, original)
        self.assertLess(len(json.dumps(feedback)), 3000)

    def test_failed_region_wins_over_larger_errors_in_passing_replacements(self):
        raw = region_failure()
        passing = copy.deepcopy(raw["cases"][0]["checks"][0])
        passing.update(seed=2718, passed=True)
        passing.pop("region_checks")
        for layer in passing["layers"]:
            layer["check"]["max_abs_error"] = 8.
        raw["cases"][0]["checks"].append(passing)
        feedback = SearchLoop.repair_feedback(repair(raw))["result"]
        self.assertEqual(feedback["seed"], 1729)
        self.assertEqual(feedback["failure_boundary"], "containing_region")
        self.assertEqual(feedback["failed_layers"][0]["examples"][0], EXAMPLE)

    def test_missing_or_unsafe_region_coverage_remains_visible(self):
        raw = region_failure()
        region = raw["cases"][0]["checks"][0]["region_checks"]
        region.update(layers=[], observed_paths=[], missing_paths=region["expected_paths"],
                      reference_state_intact=False, not_measured=True, reason="Reference restoration failed.")
        nested = SearchLoop.model_feedback(raw)["cases"][0]["checks"][0]["region_checks"]
        self.assertEqual(nested["missing_path_count"], 28)
        self.assertFalse(nested["reference_state_intact"])
        self.assertTrue(nested["not_measured"])
        self.assertEqual(nested["reason"], "Reference restoration failed.")

    def test_merged_region_records_keep_boundary_and_rank_failed_values(self):
        raw = region_failure()
        first = raw["cases"][0]["checks"][0]
        first["layers"].extend({**record, "scope": "containing_region"}
                               for record in first["region_checks"]["layers"])
        later = copy.deepcopy(first)
        later["seed"] = later["region_checks"]["seed"] = 3141
        later["region_checks"]["layers"][0]["check"]["max_abs_error"] = 8.
        later["region_checks"]["layers"][26]["check"]["max_abs_error"] = .5
        later["layers"][-28: ] = [{**record, "scope": "containing_region"}
                                   for record in later["region_checks"]["layers"]]
        raw["cases"][0]["checks"].append(later)
        failure = SearchLoop.repair_feedback(repair(raw))["result"]
        self.assertEqual(failure["seed"], 1729)
        self.assertEqual(failure["failure_boundary"], "containing_region")
        self.assertEqual(failure["failed_layers"][0]["path"], "model.layers.26.mlp")
        self.assertEqual(failure["failed_layers"][0]["examples"][0], EXAMPLE)


if __name__ == "__main__":
    unittest.main()
