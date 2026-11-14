"""Keep model failures small in prompts and retain complete saved evidence."""

import copy
import json
import unittest

from sako.loop import SearchLoop


def model_result():
    cases = []
    for index, workload in enumerate((
            {"batch_size": 1, "sequence_length": 128},
            {"batch_size": 1, "sequence_length": 512},
            {"batch_size": 2, "sequence_length": 128})):
        checks = []
        for offset, seed in enumerate((1729, 2718, 3141)):
            checks.append({"seed": seed, "passed": False,
                "max_abs_error": .6 + .01 * index + .001 * offset,
                "atol": .03, "rtol": .02, "mismatched_elements": 30000 + index,
                "shape": [workload["batch_size"], workload["sequence_length"], 152064],
                "examples": [{"flat_index": n, "actual": .01, "expected": .06,
                              "allowed_abs_error": .0312} for n in range(100)]})
        cases.append({"workload": workload, "correct": False, "reference": "torch_compile",
                      "checks": checks, "compiled_vs_eager_diagnostics": copy.deepcopy(checks),
                      "baseline": {"median_ms": 2., "samples_ms": [2.] * 1500},
                      "candidate": {"median_ms": 1., "samples_ms": [1.] * 1500}})
    manifest = {"entries": [{"task_id": "rmsnorm_0", "candidate_id": "a" * 24,
                              "paths": [f"model.layers.{i}.input_layernorm" for i in range(57)]}]}
    return {"scope": "complete_model_forward", "compiled": True, "fullgraph": True,
            "config": {"weights": "private weight metadata " * 1000, "atol": .03, "rtol": .02},
            "cases": cases, "manifest": manifest,
            "worker_provenance": {"full_environment": "raw evidence " * 2000},
            "decision": {"accepted": False, "speedup": None,
                         "reasons": ["Case 0: correctness failed."]}}


class ModelFeedback(unittest.TestCase):
    def test_model_summary_keeps_workloads_seed_checks_and_bounded_examples(self):
        raw = model_result()
        before = copy.deepcopy(raw)
        measured = {"manifest": raw["manifest"], "result": raw, "generation": 4}
        summary = SearchLoop.model_feedback(measured)
        self.assertEqual(raw, before)
        self.assertEqual(summary["decision"], raw["decision"])
        self.assertEqual(len(summary["cases"]), 3)
        for case, original in zip(summary["cases"], raw["cases"]):
            self.assertEqual(case["workload"], original["workload"])
            self.assertFalse(case["correct"])
            self.assertEqual(len(case["checks"]), 3)
            for check, expected in zip(case["checks"], original["checks"]):
                self.assertEqual(check["seed"], expected["seed"])
                self.assertEqual(check["max_abs_error"], expected["max_abs_error"])
                self.assertEqual(check["atol"], expected["atol"])
                self.assertEqual(check["rtol"], expected["rtol"])
                self.assertFalse(check["passed"])
                self.assertLessEqual(len(check.get("examples", [])), 2)
        encoded = json.dumps(summary)
        self.assertLess(len(encoded), 6000)
        for omitted in ("private weight metadata", "samples_ms", "full_environment",
                        "model.layers.56.input_layernorm"):
            self.assertNotIn(omitted, encoded)
        self.assertEqual(SearchLoop.model_feedback(raw), summary)

    def test_model_summary_keeps_unknown_nonfinite_error_and_raw_data(self):
        raw = model_result()
        raw["cases"][0]["checks"][0]["max_abs_error"] = None
        before = copy.deepcopy(raw)
        summary = SearchLoop.model_feedback(raw)
        self.assertIsNone(summary["cases"][0]["checks"][0]["max_abs_error"])
        self.assertFalse(summary["cases"][0]["correct"])
        self.assertEqual(raw, before)

    def test_four_pending_repairs_keep_ownership_and_small_numerical_evidence(self):
        raw = model_result()
        pending = {}
        for slot in range(4):
            key = f"{slot:024x}"
            pending[key] = {"candidate_id": key, "task_id": "rmsnorm_0", "stage": "model",
                            "slot": slot, "repairs_used": slot,
                            "evidence": {"path": f"evidence/repair-{key}.json",
                                         "sha256": "a" * 64, "bytes": 65000},
                            "manifest": copy.deepcopy(raw["manifest"]),
                            "result": {"combination": copy.deepcopy(raw), "individual": copy.deepcopy(raw)}}
        before = copy.deepcopy(pending)
        summary = SearchLoop.repair_feedback(pending)
        self.assertEqual(set(summary), set(pending))
        for key, original in pending.items():
            item = summary[key]
            for field in ("candidate_id", "task_id", "stage", "slot", "repairs_used", "evidence"):
                self.assertEqual(item[field], original[field])
            evidence = json.dumps(item["result"])
            self.assertIn("max_abs_error", evidence)
            self.assertIn("workload", evidence)
            self.assertIn("seed", evidence)
            def errors(value):
                if isinstance(value, dict):
                    for field, child in value.items():
                        if field == "max_abs_error" and isinstance(child, (int, float)):
                            yield child
                        else:
                            yield from errors(child)
                elif isinstance(value, list):
                    for child in value:
                        yield from errors(child)
            self.assertEqual(max(errors(item["result"])), max(errors(raw)))
        encoded = json.dumps(summary)
        self.assertLess(len(encoded), 8000)
        for omitted in ("private weight metadata", "samples_ms", "full_environment",
                        "model.layers.56.input_layernorm"):
            self.assertNotIn(omitted, encoded)
        self.assertEqual(pending, before)


if __name__ == "__main__":
    unittest.main()
