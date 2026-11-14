"""Keep repair state visible without claiming a model gain."""

import unittest
import test_swarm_reporting as fixtures
from sako.reporting import build_swarm_report, render_swarm_report


class RepairReportingTests(unittest.TestCase):
    setUp = fixtures.SwarmReportingTests.setUp
    candidate = fixtures.SwarmReportingTests.candidate
    measured = fixtures.SwarmReportingTests.measured
    def test_exact_repair_and_context_evidence_are_visible_without_acceptance(self):
        key = self.candidate()
        self.store.record(key, self.measured())
        ticket = {"candidate_id": key, "status": "correction_pending", "hypothesis_id": "original",
                  "failures": [{"path": "model.layers.26.mlp", "seed": 1729,
                      "check": {"actual": .7783203125, "expected": .8251953125,
                                "allowed_abs_error": .046503905206918716}}]}
        context = {"full_characters": 90480, "characters": 62903, "limit": 90000,
                   "artifact": {"path": "evidence/coordinator-context.json"}}
        self.store.set_many({"integration_repair_tickets": {key: ticket},
                             "coordinator_context:rmsnorm_1": context,
                             "status": "integration_repair_pending"})
        report = build_swarm_report(self.run, self.config, store=self.store)
        self.assertEqual(report["integration_repair_tickets"][key], ticket)
        self.assertEqual(report["candidates"][0]["integration_repair"], ticket)
        self.assertEqual(report["coordinator_context"]["rmsnorm_1"], context)
        self.assertFalse(report["accepted_model_gain"])
        text = render_swarm_report(report)
        self.assertIn("model.layers.26.mlp", text)
        self.assertIn("Coordinator context sizes", text)


if __name__ == "__main__":
    unittest.main()
