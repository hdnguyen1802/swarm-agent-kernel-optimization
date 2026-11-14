"""Diagnostic regions do not hide or create required numerical failures."""

import unittest

from sako.hypotheses import measured_gpu_outcome
import test_hypotheses as fixtures


class DiagnosticHypothesisTests(unittest.TestCase):
    setUp = fixtures.HypothesisPolicyTests.setUp
    register = fixtures.HypothesisPolicyTests.register

    def accepted_parent(self):
        self.register()
        self.book.begin("hypothesis-a", "assignment-a")
        self.book.finish("assignment-a", stage="operation", result=fixtures.measured_result(accepted=True),
                         candidate_id="candidate-a")

    def test_skipped_diagnostic_keeps_a_required_failure_and_its_repair(self):
        self.accepted_parent()
        result = fixtures.layer_result()
        check = result["cases"][0]["checks"][0]
        check["region_checks"] = {"acceptance_role": "diagnostic", "not_measured": True,
                                   "passed": False, "layers": []}
        check["layers"].append({"acceptance_role": "diagnostic", "path": "model",
                                "passed": False, "not_measured": True})
        self.assertTrue(measured_gpu_outcome(result))
        record = self.book.integration_failure("candidate-a", result)
        self.assertEqual(record["status"], "correction_pending")
        self.assertEqual(record["attempt_count"], 1)
        self.assertEqual(record["corrections_remaining"], 2)

    def test_diagnostic_numerical_failure_does_not_reopen_a_parent(self):
        self.accepted_parent()
        result = fixtures.layer_result(passed=True)
        check = result["cases"][0]["checks"][0]
        diagnostic = fixtures.layer_result()["cases"][0]["checks"][0]
        diagnostic["acceptance_role"] = "diagnostic"
        check["region_checks"] = diagnostic
        check["layers"].extend({**layer, "acceptance_role": "diagnostic"} for layer in diagnostic["layers"])
        self.assertFalse(measured_gpu_outcome(result))
        self.assertIsNone(self.book.integration_failure("candidate-a", result))
        self.assertEqual(self.book.get("hypothesis-a")["status"], "succeeded")

    def test_unsafe_diagnostics_still_exclude_the_measurement(self):
        for unsafe in ({"reference_state_intact": False}, {"restoration_errors": ["Input changed."]},
                       {"inputs_unchanged": False}):
            with self.subTest(unsafe=unsafe):
                result = fixtures.layer_result()
                result["cases"][0]["checks"][0]["region_checks"] = {
                    "acceptance_role": "diagnostic", "layers": [{**unsafe}]}
                self.assertFalse(measured_gpu_outcome(result))

    def test_required_unmeasured_evidence_still_excludes_the_measurement(self):
        result = fixtures.layer_result()
        result["cases"][0]["checks"][0]["region_checks"] = {"not_measured": True}
        self.assertFalse(measured_gpu_outcome(result))


if __name__ == "__main__":
    unittest.main()
