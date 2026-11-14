"""Prompt references and scheduling must use the same measured state."""

import json
from concurrent.futures import Future
import threading
import unittest
from unittest.mock import patch

from sako.coordinator_context import fit_coordinator_context
from sako.evidence_registry import coordinator_evidence
from sako.swarm_policy import validate_plan
import test_swarm_loop as fixtures
from test_coordinator_context import large_prompt


def hypothesis(change="Change load order.", algorithm=None):
    value = {"specialty": "memory", "parent_id": None, "claim": "Test measured limits.",
             "change": change, "expected_metric": "Lower latency.", "evidence": ["contract:task"]}
    if algorithm is not None:
        value["algorithm"] = algorithm
    return value


class EvidenceRegistryTests(unittest.TestCase):

    def test_validator_uses_measured_evidence_ids(self):
        registry = coordinator_evidence("task", {"fingerprint": "gpu"}, [{"id": "parent"}],
                                        [{"id": "lesson"}], [{"path": "evidence/raw.json"}])
        self.assertIn("parent", registry.types()["candidate"])
        plan = {"action": "move_on", "reason": "Measured trials show low expected value.",
                "evidence": ["parent", "baseline:task"], "hypotheses": [],
                "archive_skip": {"reason": "Retain this measured design.", "evidence": ["parent"]}}
        self.assertEqual(validate_plan(plan, task_id="task", known_evidence=registry.ids(),
                                       parent_ids=["parent"])["action"], "move_on")
        plan["evidence"].append("invented")
        with self.assertRaisesRegex(ValueError, "known evidence"):
            validate_plan(plan, task_id="task", known_evidence=registry.ids(), parent_ids=["parent"])

    def test_explicit_duplicate_fresh_algorithms_are_not_parallel_diversity(self):
        plan = {"action": "continue", "reason": "Compare algorithms.", "evidence": ["contract:task"],
            "hypotheses": [hypothesis("Tile 32.", "Split K"), hypothesis("Tile 64.", " split   k ")]}
        result = validate_plan(plan, task_id="task", known_evidence={"contract:task"}, parent_ids=[])
        self.assertEqual(len(result["hypotheses"]), 1)
        self.assertEqual(result["rejected_hypotheses"][0]["reason"], "duplicate_fresh_algorithm")
        plan["hypotheses"][1]["algorithm"] = "Persistent work scheduling"
        result = validate_plan(plan, task_id="task", known_evidence={"contract:task"}, parent_ids=[])
        self.assertEqual(result["hypotheses"][0]["algorithm"], "Split K")

    def test_soft_target_compacts_more_but_never_truncates_required_contract(self):
        raw = large_prompt()
        initial = fit_coordinator_context(raw, 90000)
        target = len(json.dumps(initial)) - 1000
        small = fit_coordinator_context(raw, 90000, target_chars=target)
        self.assertLessEqual(len(json.dumps(small)), target)
        self.assertEqual(small["task"], raw["task"])
        self.assertEqual(small["pending_repairs"], raw["pending_repairs"])
        raw["task"]["contract"]["detail"] = "required" * 3000
        retained = fit_coordinator_context(raw, 90000, target_chars=5000)
        self.assertGreater(len(json.dumps(retained)), 5000)
        self.assertLessEqual(len(json.dumps(retained)), 90000)
        self.assertEqual(retained["task"], raw["task"])


class CoordinatorProtocolTests(unittest.TestCase):
    setUp = fixtures.SwarmLoopTests.setUp
    loop = fixtures.SwarmLoopTests.loop

    def test_auth_failure_resumes_the_same_assignment_without_a_correction(self):
        from sako.agent import AuthenticationUnavailable
        generations = []
        def writer(data):
            generations.append(data["generation"])
            if len(generations) == 1:
                raise AuthenticationUnavailable("Token refresh failed before generation.")
        self.client.writer_callback = writer
        first_loop = self.loop()
        first = first_loop.run()
        self.assertEqual(first["status"], "infrastructure_failure")
        assignment = next(iter(first_loop.work.values()))
        self.assertEqual(assignment["status"], "planned")
        record = first_loop.hypotheses.get(assignment["hypothesis_id"])
        self.assertEqual(record["status"], "paused")
        self.assertEqual(record["attempt_count"], 1)
        self.assertIsNone(record["attempts"][0]["outcome_hash"])
        self.assertEqual(record["last_interruption"]["stage"], "infrastructure")
        resumed = self.loop()
        second = resumed.run()
        self.assertEqual(second["status"], "completed_no_gain", second.get("swarm_stop"))
        self.assertEqual(generations, [0, 0])
        self.assertEqual(len(resumed.work), 1)
        record = resumed.hypotheses.get(assignment["hypothesis_id"])
        self.assertEqual(record["attempt_count"], 1)
        self.assertEqual(record["corrections_used"], 0)
        self.assertEqual(sum(action == "evaluate" for action, _ in self.worker.calls), 1)

    def test_parallel_auth_failures_resume_without_losing_the_other_writer(self):
        from sako.agent import AuthenticationUnavailable
        self.config["agent"].update(writers=2, max_task_experiments=2)
        self.client.width = 2
        barrier = threading.Barrier(2)
        failures, generations = {0, 1}, []
        def writer(data):
            generation = data["generation"]
            generations.append(generation)
            if generation in failures:
                failures.remove(generation)
                barrier.wait(timeout=5)
                raise AuthenticationUnavailable("Token refresh failed before generation.")
        self.client.writer_callback = writer
        first = self.loop().run()
        self.assertEqual(first["status"], "infrastructure_failure")
        resumed = self.loop()
        second = resumed.run()
        self.assertEqual(second["status"], "completed_no_gain", second.get("swarm_stop"))
        self.assertEqual(sorted(generations), [0, 0, 1, 1])
        self.assertEqual(len(resumed.work), 2)
        self.assertEqual(sum(action == "evaluate" for action, _ in self.worker.calls), 2)
        for record in resumed.hypotheses.all():
            self.assertEqual(record["attempt_count"], 1)
            self.assertEqual(record["corrections_used"], 0)


    def test_distinct_plan_errors_can_be_corrected_without_repeating_the_first(self):
        loop = self.loop()
        def plan(data, number):
            evidence = ["contract:" + data["task"]["id"]]
            h = {**hypothesis(), "evidence": evidence}
            reply = {"action": "continue", "reason": "Test one change.", "evidence": evidence, "hypotheses": [h]}
            if number == 0:
                reply["action"] = "invalid"
            elif number == 1:
                h["verification"] = [{"fixture_index": 0, "field": "launch.count", "op": "eq", "value": 1}]
            else:
                self.assertEqual(len(data["validation_errors"]), 2)
                self.assertEqual(data["allowed_actions"], ["continue"])
            return reply
        self.client.plan_callback = plan
        loop.start_task()
        self.assertEqual(loop.coordinator(1)["action"], "continue")
        self.assertEqual(self.client.plan_number, 3)

    def test_finished_writers_are_evaluated_before_another_planning_call(self):
        self.config["agent"].update(max_task_experiments=None, require_archive_trial=False)
        self.client.width = 2
        class ImmediatePool:
            def __init__(self, **_): pass
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def submit(self, fn, *args):
                future = Future()
                try: future.set_result(fn(*args))
                except Exception as error: future.set_exception(error)
                return future
        def plan(data, number):
            if number == 0:
                values = []
                for index in range(2):
                    h = hypothesis("Measured design " + str(index))
                    h["evidence"] = ["contract:" + data["task"]["id"]]
                    values.append(h)
                return {"action": "continue", "reason": "Test two designs.",
                        "evidence": values[0]["evidence"], "hypotheses": values}
            self.assertEqual(sum(action == "evaluate" for action, _ in self.worker.calls), 2,
                             "Generated candidates must be measured before buying another plan.")
            parent = data["parents"][0]["id"]
            return {"action": "move_on", "reason": "The designs were measured.", "evidence": [parent],
                    "hypotheses": [], "archive_skip": {"reason": "No further useful trial.", "evidence": [parent]}}
        self.client.plan_callback = plan
        def one_at_a_time(futures, **_):
            first = next(iter(futures))
            return {first}, set(futures) - {first}
        loop = self.loop()
        # This test covers the scheduler. Numerical corrections have separate tests.
        loop.hypotheses.performance_as_refinement = True
        with patch("sako.swarm.ThreadPoolExecutor", ImmediatePool), patch("sako.swarm.wait", one_at_a_time):
            result = loop.run()
        self.assertEqual(result["status"], "completed_no_gain", result.get("swarm_stop"))
        self.assertEqual(self.client.plan_number, 2)


if __name__ == "__main__":
    unittest.main()
