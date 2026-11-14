"""Bound parallel writers by distinct experiments and measured call costs."""

import copy
import unittest

from sako.swarm_policy import admit_experiments, estimate_costs, validate_plan


class SwarmPolicyTests(unittest.TestCase):
    def setUp(self):
        self.limits = {"writers": 5, "max_total_tokens": 1_000_000, "max_output_tokens": 8192}
        self.costs = {"writer": 10_000, "coordinator": 5_000}
        self.parent = "a" * 24
        self.evidence = {"hardware:gpu-a", "contract:task-a", self.parent}

    def hypothesis(self, number=0, **changes):
        result = {"specialty": "memory", "parent_id": self.parent,
                  "claim": "This data layout may reduce memory traffic.",
                  "change": "Test vector width " + str(number + 1),
                  "expected_metric": "Memory transactions", "evidence": [self.parent]}
        return {**result, **changes}

    def plan(self, hypotheses, **changes):
        return {"action": "continue", "reason": "Test the measured memory limit.",
                "evidence": ["contract:task-a"], "hypotheses": hypotheses, **changes}

    def validate(self, plan, **changes):
        options = {"task_id": "task-a", "known_evidence": self.evidence,
                   "parent_ids": [self.parent], "max_writers": 5, **changes}
        return validate_plan(plan, **options)

    def test_one_to_five_hypotheses_use_one_to_five_slots(self):
        for count in range(1, 6):
            with self.subTest(count=count):
                plan = self.validate(self.plan([self.hypothesis(n) for n in range(count)]))
                admission = admit_experiments(self.limits, 0, self.costs, plan["max_active_writers"])
                self.assertEqual(plan["max_active_writers"], count)
                self.assertEqual(admission["writer_slots"], count)
        queued = self.validate(self.plan([self.hypothesis(n) for n in range(8)]))
        self.assertEqual(len(queued["hypotheses"]), 8)
        self.assertEqual(queued["max_active_writers"], 5)

    def test_single_experiment_fits_without_a_three_round_reserve(self):
        result = admit_experiments(self.limits, 985_000, self.costs, 5)
        self.assertTrue(result["admit"])
        self.assertEqual(result["writer_slots"], 1)
        self.assertEqual(result["required_tokens"], 15_000)
        self.assertEqual(result["review_reserve_tokens"], 5_000)
        result = admit_experiments(self.limits, 985_001, self.costs, 5)
        self.assertFalse(result["admit"])
        self.assertEqual(result["reason"], "token_budget")

    def test_task_coverage_can_reduce_slots_without_resetting_global_usage(self):
        result = admit_experiments(self.limits, 200_000, self.costs, 5, task_remaining=25_000)
        self.assertEqual(result["writer_slots"], 2)
        self.assertEqual(result["remaining_tokens"], 800_000)
        exhausted = admit_experiments(self.limits, 200_000, self.costs, 5, task_remaining=14_999)
        self.assertFalse(exhausted["admit"])
        self.assertEqual(exhausted["reason"], "task_coverage_reserve")

    def test_slots_never_exceed_five_or_configured_limit(self):
        self.assertEqual(admit_experiments(self.limits, 0, self.costs, 100)["writer_slots"], 5)
        self.assertEqual(admit_experiments({**self.limits, "writers": 2}, 0, self.costs, 5)["writer_slots"], 2)
        self.assertFalse(admit_experiments(self.limits, 0, self.costs, 0)["admit"])
        for costs in ({"writer": 0, "coordinator": 1}, {"writer": 1, "coordinator": -1}):
            with self.assertRaises(ValueError):
                admit_experiments(self.limits, 0, costs, 5)
        with self.assertRaises(ValueError):
            self.validate(self.plan([self.hypothesis()]), max_writers=6)

    def test_duplicate_transformations_ignore_reworded_claims_and_spacing(self):
        first = self.hypothesis(change="Use float4 loads")
        duplicate = self.hypothesis(specialty="compute", claim="A different explanation of the same change.",
                                    change="  USE   FLOAT4\nLOADS ")
        plan = self.validate(self.plan([first, duplicate, self.hypothesis(2)]))
        self.assertEqual(len(plan["hypotheses"]), 2)
        self.assertEqual(plan["rejected_hypotheses"][0]["reason"], "duplicate_hypothesis")
        with self.assertRaisesRegex(ValueError, "distinct hypothesis"):
            self.validate(self.plan([first]), seen=[plan["hypotheses"][0]["id"]])

    def test_unknown_parent_and_evidence_are_rejected_before_generation(self):
        for hypothesis in (self.hypothesis(parent_id="b" * 24),
                           self.hypothesis(evidence=["made-up-result"]),
                           self.hypothesis(parent_id={"forged": "object"}),
                           self.hypothesis(evidence=[{"forged": "object"}])):
            with self.subTest(hypothesis=hypothesis):
                with self.assertRaises(ValueError):
                    self.validate(self.plan([hypothesis]))
        repair = "b" * 24
        plan = self.validate(self.plan([self.hypothesis(parent_id=repair)], archive_skip={
            "reason": "Repair the known failure before testing another correct parent.",
            "evidence": [self.parent]}), repair_ids=[repair])
        self.assertEqual(plan["hypotheses"][0]["parent_id"], repair)

    def test_move_on_needs_evidence_and_cannot_start_writers(self):
        plan = self.validate(self.plan([], action="move_on", archive_skip={
            "reason": "Measured parent variants have low value at the current model speed gate.",
            "evidence": [self.parent]}))
        self.assertEqual(plan["max_active_writers"], 0)
        for invalid in (self.plan([], action="move_on", evidence=[]),
                        self.plan([self.hypothesis()], action="move_on"),
                        self.plan([]), self.plan([], action="invent_result")):
            with self.assertRaises(ValueError):
                self.validate(invalid)

    def test_wait_requires_running_work_and_dispatches_no_hypotheses(self):
        waiting = self.plan([], action="wait", reason="Wait for the two pending measurements.")
        plan = self.validate(waiting, allow_wait=True)
        self.assertEqual(plan["action"], "wait")
        self.assertEqual(plan["hypotheses"], [])
        self.assertEqual(plan["max_active_writers"], 0)
        self.assertEqual(plan["reason"], waiting["reason"])
        with self.assertRaises(ValueError):
            self.validate(waiting)
        with self.assertRaises(ValueError):
            self.validate(waiting, allow_wait=False)

    def test_wait_cannot_dispatch_work_or_cite_unknown_evidence(self):
        for invalid in (self.plan([self.hypothesis()], action="wait"),
                        self.plan([], action="wait", evidence=["missing-profile"]),
                        self.plan([], action="wait", evidence=[])):
            with self.assertRaises(ValueError):
                self.validate(invalid, allow_wait=True)

    def test_cost_estimates_ignore_metadata_failures_and_old_samples(self):
        calls = [{"role": "writer_0_memory", "status": "complete", "actual": 1_000_000}]
        calls.extend({"role": "writer_1_compute", "status": "complete", "actual": 100}
                     for _ in range(12))
        calls.extend([
            {"role": "writer_1_compute:count_tokens", "status": "complete", "actual": 500_000},
            {"role": "writer_1_compute", "status": "failed", "actual": 500_000},
            {"role": "coordinator", "status": "complete", "actual": 200},
            {"role": "coordinator", "status": "reserved", "actual": None}])
        original = copy.deepcopy(calls)
        self.assertEqual(estimate_costs(self.limits, calls), {"writer": 115, "coordinator": 230})
        self.assertEqual(calls, original)
        self.assertEqual(estimate_costs(self.limits, []), {"writer": 16384, "coordinator": 16384})


if __name__ == "__main__":
    unittest.main()
