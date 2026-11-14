"""Bound coordinator requests while retaining decisions and full evidence."""

import copy
import json
from pathlib import Path
import unittest

from sako.coordinator_context import fit_coordinator_context
from sako.core import read_json
import test_swarm_loop as fixtures


def large_prompt():
    parents = []
    for index in range(4):
        parents.append({"id": f"parent-{index}", "score": 1.1, "correct": True, "accepted": False,
            "model_status": "untested", "hypothesis": {"id": f"hyp-{index}", "change": "Test a larger tile."},
            "reuse": {"archive_refinements": index}, "operation_measurements": {
                "decision": {"accepted": False, "reasons": ["Some shapes are slow."], "speedup": 1.1},
                "cases": [{"case_index": case, "seed": 1729 + case, "workload": {"batch_size": 1, "sequence_length": 128},
                    "correct": True, "baseline_ms": .2, "candidate_ms": .18, "candidate_p95_ms": .19,
                    "autotuning": [{"choices": "full tuning detail " * 300}]} for case in range(6)]},
            "model_measurements": None})
    states = [{"id": f"hyp-{index}", "task_id": "task", "status": "correction_pending",
        "hypothesis": {"id": f"hyp-{index}", "parent_id": f"parent-{index % 4}", "change": "Fix rounding."},
        "attempt_count": 2, "max_attempts": 3, "max_corrections": 2, "corrections_used": 1,
        "corrections_remaining": 1, "last_candidate_id": f"parent-{index % 4}",
        "attempts": [{"raw_writer_response": "source " * 5000}], "last_outcome": {"stage": "model"}}
        for index in range(5)]
    return {"request": "Select hypotheses using measured evidence.",
        "task": {"id": "task", "contract": {"validated": True, "operation": "x @ weight.T"}},
        "validation_policy": {"correctness_reference": "eager", "kernel_min_speedup": 1.01},
        "parents": parents,
        "hypothesis_states": states, "closed_hypothesis_count": 10,
        "known_evidence": [f"parent-{i}" for i in range(4)], "pending_repairs": {
            "parent-0": {"candidate_id": "parent-0", "repairs_used": 1, "stage": "model",
                         "evidence": {"path": "evidence/numerical-error.json", "sha256": "f" * 64}}},
        "archive_policy": {"eligible_parent_ids": [f"parent-{i}" for i in range(4)]},
        "budget": {"tokens": 100000, "max_total_tokens": 1500000, "remaining_tokens": 1400000}}


class CoordinatorContextTests(unittest.TestCase):
    def test_large_parents_and_history_fit_without_changing_raw_evidence(self):
        raw = large_prompt()
        before = copy.deepcopy(raw)
        self.assertGreater(len(json.dumps(raw)), 90000)
        artifact = {"path": "evidence/full-context.json", "sha256": "f" * 64, "bytes": 300000}
        result = fit_coordinator_context(raw, 90000, artifact=artifact)
        self.assertLessEqual(len(json.dumps(result)), 90000)
        self.assertEqual(raw, before)
        self.assertEqual(result["task"], raw["task"])
        self.assertEqual(result["validation_policy"], raw["validation_policy"])
        self.assertEqual(result["context_summary"]["full_context"], artifact)
        self.assertEqual([p["id"] for p in result["parents"]], [p["id"] for p in raw["parents"]])
        self.assertEqual(result["parents"][0]["operation_measurements"]["cases"][0]["candidate_ms"], .18)

    def test_active_repair_ids_counters_and_exact_repair_evidence_survive_pressure(self):
        raw = large_prompt()
        result = fit_coordinator_context(raw, 9000)
        self.assertLessEqual(len(json.dumps(result)), 9000)
        self.assertEqual(result["pending_repairs"], raw["pending_repairs"])
        self.assertEqual(result["archive_policy"], raw["archive_policy"])
        for actual, original in zip(result["hypothesis_states"], raw["hypothesis_states"]):
            for key in ("id", "status", "attempt_count", "max_attempts", "max_corrections", "corrections_used",
                        "corrections_remaining", "last_candidate_id"):
                self.assertEqual(actual[key], original[key])

    def test_retry_detail_is_bounded_under_the_same_character_limit(self):
        raw = large_prompt()
        raw.update(invalid_plan={"reason": "untrusted long explanation " * 10000, "hypotheses": []},
                   validation_error="The coordinator must cite known evidence.",
                   retry_instruction="Correct the plan.")
        result = fit_coordinator_context(raw, 10000)
        self.assertLessEqual(len(json.dumps(result)), 10000)
        self.assertEqual(result["validation_error"], raw["validation_error"])
        self.assertTrue(result["invalid_plan"]["reason"].startswith("untrusted"))


    def test_impossible_core_context_fails_with_saved_evidence_instruction(self):
        raw = large_prompt()
        raw["task"]["contract"]["exact_operation"] = "core contract " * 10000
        with self.assertRaisesRegex(ValueError, "required coordinator contract"):
            fit_coordinator_context(raw, 9000)


class CoordinatorRequestTests(unittest.TestCase):
    setUp = fixtures.SwarmLoopTests.setUp
    loop = fixtures.SwarmLoopTests.loop

    def test_requests_save_full_evidence_and_fit_even_after_a_plan_rejection(self):
        self.config["agent"]["context_chars"] = 90000
        loop = self.loop()
        loop.store.set("kernel_profiles", {"obsolete": {"task_id": loop.active_task}})
        def plan(data, number):
            if number == 0:
                return {"action": "continue", "reason": "invalid " * 20000, "evidence": ["unknown"], "hypotheses": []}
            return {"action": "continue", "reason": "Test the measured limit.",
                "evidence": ["contract:" + loop.active_task], "hypotheses": [{"specialty": "memory", "parent_id": None,
                "claim": "Coalesce loads.", "change": "Change load order.", "expected_metric": "Memory bandwidth.",
                "evidence": ["contract:" + loop.active_task]}]}
        self.client.plan_callback = plan
        loop.start_task()
        result = loop.coordinator(1)
        self.assertEqual(result["action"], "continue")
        self.assertEqual(self.client.plan_number, 2)
        for role, request in self.client.prompts:
            self.assertEqual(role, "coordinator")
            self.assertLessEqual(len(json.dumps(request)), self.config["agent"]["context_chars"])
            descriptor = request["context_summary"]["full_context"]
            evidence = read_json(loop.directory / descriptor["path"])
            self.assertNotIn("profiles", evidence)
            self.assertNotIn("obsolete", request["known_evidence"])
        retry = self.client.prompts[1][1]
        rejected = retry["invalid_plan_evidence"]
        raw_reply = read_json(loop.directory / rejected["path"])
        self.assertEqual(raw_reply["reason"], "invalid " * 20000)
        self.assertIn(rejected["path"], retry["known_evidence"])
        raw_retry = read_json(loop.directory / retry["context_summary"]["full_context"]["path"])
        self.assertEqual(raw_retry["invalid_plan"], raw_reply)
        self.assertLess(len(retry["invalid_plan"]["reason"]), len(raw_reply["reason"]))


if __name__ == "__main__":
    unittest.main()
