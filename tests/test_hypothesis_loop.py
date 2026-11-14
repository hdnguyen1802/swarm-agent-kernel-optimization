"""Failures belong to individual hypotheses and do not prove a GPU plateau."""

from collections import Counter
import json
from pathlib import Path
import threading
import unittest
from unittest.mock import patch

from sako.core import read_json
from sako.store import BudgetExhausted, Store
from sako.transport import TransferError
import test_swarm_loop as fixtures

ROOT = fixtures.ROOT


class HypothesisLoopTests(unittest.TestCase):
    loop = fixtures.SwarmLoopTests.loop
    writers = fixtures.SwarmLoopTests.writers

    def setUp(self):
        fixtures.SwarmLoopTests.setUp(self)
        self.config["agent"].update(writers=1, max_task_experiments=None, max_stagnant_rounds=3)
        self.counts = Counter()
        self.count_lock = threading.Lock()

    def plan(self, data, labels):
        evidence = ["contract:" + data["task"]["id"]]
        return {"action": "continue", "reason": "Test distinct hypotheses with separate failure records.",
                "evidence": evidence, "hypotheses": [
                    {"specialty": list(data["specialties"])[index % 5], "parent_id": None,
                     "claim": "Test method " + label, "change": "Use method " + label,
                     "expected_metric": "Lower median latency.", "evidence": evidence}
                    for index, label in enumerate(labels)]}

    def count_writer(self, data):
        identity = data["hypothesis"]["id"]
        with self.count_lock:
            self.counts[identity] += 1
            return self.counts[identity]

    def stop_after_evaluations(self, count):
        original = self.worker.run

        def run(action, run_dir, **fields):
            result = original(action, run_dir, **fields)
            if action == "evaluate" and sum(kind == "evaluate" for kind, _ in self.worker.calls) == count:
                (Path(run_dir) / "STOP").touch()
            return result

        self.worker.run = run

    def state(self):
        return Store(self.directory / "run/state.sqlite")

    def test_unrelated_failures_do_not_close_the_operation(self):
        self.config["agent"]["writers"] = 4
        self.worker.operation_accepted = True
        self.worker.model_correct = True
        def plan(data, number):
            if number and data["inflight"]:
                return {"action": "wait", "reason": "Wait for the active corrections.",
                        "evidence": ["contract:" + data["task"]["id"]], "hypotheses": []}
            return self.plan(data, ["a", "b", "c", "healthy"])

        self.client.plan_callback = plan

        def writer(data):
            attempt = self.count_writer(data)
            if "healthy" not in data["hypothesis"]["claim"] and attempt == 1:
                raise ValueError("This hypothesis returned invalid CUDA source.")

        self.client.writer_callback = writer
        self.stop_after_evaluations(4)
        result = self.loop().run()
        self.assertEqual(result["status"], "stopped")
        self.assertEqual(result["active_task"], "rmsnorm_1")
        self.assertEqual(result["task_progress"]["completed"], [])
        self.assertEqual(sorted(self.counts.values()), [1, 2, 2, 2])
        self.assertEqual(sum(action == "evaluate" for action, _ in self.worker.calls), 4)
        self.assertFalse(any(event.get("reason") == "writer_failure_limit"
                             for event in self.state().get("swarm_events", [])))

    def test_initial_attempt_and_two_corrections_keep_one_hypothesis(self):
        self.worker.operation_accepted = True
        self.worker.model_correct = True
        self.client.plan_callback = lambda data, number: self.plan(data, ["correctable"])

        def writer(data):
            if self.count_writer(data) <= 2:
                raise ValueError("The entry point is missing.")

        self.client.writer_callback = writer
        self.stop_after_evaluations(1)
        result = self.loop().run()
        self.assertEqual(result["status"], "stopped")
        self.assertEqual(list(self.counts.values()), [3])
        prompts = self.writers()
        self.assertEqual(len(prompts), 3)
        self.assertEqual(len({prompt["hypothesis"]["id"] for prompt in prompts}), 1)
        self.assertIn("entry point is missing", json.dumps(prompts[1]))
        self.assertIn("entry point is missing", json.dumps(prompts[2]))
        self.assertEqual(sum(action == "evaluate" for action, _ in self.worker.calls), 1)

    def test_exhaustion_closes_only_failed_hypothesis_and_reassigns_slot(self):
        def plan(data, number):
            return self.plan(data, ["bad"] if number == 0 else ["replacement"])

        self.client.plan_callback = plan

        def writer(data):
            self.count_writer(data)
            if data["hypothesis"]["claim"].endswith("bad"):
                raise ValueError("CUDA source is invalid for this hypothesis.")

        self.client.writer_callback = writer
        self.stop_after_evaluations(1)
        result = self.loop().run()
        self.assertEqual(result["status"], "stopped")
        self.assertEqual(result["task_progress"]["completed"], [])
        prompts = self.writers()
        bad = [prompt for prompt in prompts if prompt["hypothesis"]["claim"].endswith("bad")]
        other = [prompt for prompt in prompts if prompt["hypothesis"]["claim"].endswith("replacement")]
        self.assertEqual(len(bad), 3)
        self.assertEqual(len(other), 1)
        records = self.state().get("swarm_hypotheses", {})
        self.assertEqual(len(records), 2)
        bad_record = records[bad[0]["hypothesis"]["id"]]
        self.assertIn(bad_record.get("status"), {"closed", "exhausted"})

    def test_resume_generated_correction_does_not_repeat_paid_or_gpu_work(self):
        self.client.plan_callback = lambda data, number: self.plan(data, ["resumable"])

        def writer(data):
            if self.count_writer(data) == 1:
                raise ValueError("The source needs a correction.")

        self.client.writer_callback = writer
        self.worker.fail_upload = True
        self.stop_after_evaluations(1)
        first = self.loop().run()
        self.assertEqual(first["status"], "infrastructure_failure")
        self.assertEqual(len(self.writers()), 2)
        self.assertEqual(len(self.worker.calls), 0)
        second = self.loop().run()
        self.assertEqual(second["status"], "stopped")
        self.assertEqual(len(self.writers()), 2)
        self.assertEqual(sum(action == "evaluate" for action, _ in self.worker.calls), 1)

    def test_api_and_source_failures_do_not_increment_stagnation(self):
        self.client.plan_callback = lambda data, number: self.plan(data, ["errors"])

        def writer(data):
            attempt = self.count_writer(data)
            if attempt == 3:
                (self.directory / "run/STOP").touch()
            if attempt == 1:
                raise RuntimeError("The provider request failed.")
            raise ValueError("The generated source violates the schema.")

        self.client.writer_callback = writer
        result = self.loop().run()
        self.assertEqual(result["status"], "stopped")
        self.assertEqual(result["task_progress"]["stagnant"], 0)
        self.assertEqual(self.worker.calls, [])

    def test_compile_failure_does_not_increment_stagnation(self):
        self.client.plan_callback = lambda data, number: self.plan(data, ["compile"])

        def run(action, run_dir, **fields):
            self.worker.calls.append((action, fields))
            (Path(run_dir) / "STOP").touch()
            return {"error": "nvcc compilation failed.", "decision": {"accepted": False, "speedup": None},
                    "worker_report": {"failure_kind": "worker", "error": "Compiler returned a nonzero exit."}}

        self.worker.run = run
        result = self.loop().run()
        self.assertEqual(result["status"], "stopped")
        self.assertEqual(result["task_progress"]["stagnant"], 0)

    def test_measured_correct_but_slow_candidate_increments_stagnation(self):
        self.client.plan_callback = lambda data, number: self.plan(data, ["measured"])
        self.stop_after_evaluations(1)
        loop = self.loop()
        loop.progress["best_operation_score"] = 1.0
        loop.store.set("task_progress", loop.progress)
        result = loop.run()
        self.assertEqual(result["status"], "stopped")
        self.assertEqual(result["task_progress"]["stagnant"], 1)
        self.assertEqual(sum(action == "evaluate" for action, _ in self.worker.calls), 1)

    def test_resume_reconciles_writer_failure_after_transfer_exit(self):
        self.config["agent"]["writers"] = 2
        self.worker.operation_accepted = True
        self.worker.model_correct = True
        self.client.plan_callback = lambda data, number: self.plan(data, ["first", "late"])
        transfer_failed = threading.Event()

        def writer(data):
            attempt = self.count_writer(data)
            if data["hypothesis"]["claim"].endswith("late") and attempt == 1:
                self.assertTrue(transfer_failed.wait(5))
                raise ValueError("The late writer returned invalid source after transfer stopped the controller.")

        self.client.writer_callback = writer

        def put(*_):
            if not transfer_failed.is_set():
                transfer_failed.set()
                raise TransferError("The first upload failed while another writer was active.")

        self.worker.put = put
        self.stop_after_evaluations(2)
        first = self.loop().run()
        self.assertEqual(first["status"], "infrastructure_failure")
        self.assertEqual(len(self.writers()), 2)
        self.assertEqual(self.worker.calls, [])
        second = self.loop().run()
        self.assertEqual(second["status"], "stopped")
        self.assertEqual(len(self.writers()), 3)
        self.assertEqual(sorted(self.counts.values()), [1, 2])
        self.assertEqual(sum(action == "evaluate" for action, _ in self.worker.calls), 2)
        records = self.state().get("swarm_hypotheses", {})
        self.assertTrue(all(record["last_outcome"]["accepted"] for record in records.values()))
        self.assertTrue(all(record["status"] == "succeeded" for record in records.values()))
        self.assertTrue(all(record["implementation_status"] == "unverified" for record in records.values()))

    def test_compiler_error_cannot_support_measured_move_on(self):
        def plan(data, number):
            if number == 0:
                return self.plan(data, ["compile-only"])
            return {"action": "move_on", "reason": "Only compilation failures exist.",
                    "evidence": [data["operation_results"][0]["id"]], "hypotheses": []}

        self.client.plan_callback = plan

        def run(action, run_dir, **fields):
            self.worker.calls.append((action, fields))
            return {"error": "nvcc compilation failed.", "decision": {"accepted": False, "speedup": None}}

        self.worker.run = run
        result = self.loop().run()
        self.assertEqual(result["status"], "request_failure")
        self.assertEqual(result["task_progress"]["completed"], [])
        self.assertEqual(result["task_progress"]["stagnant"], 0)
        self.assertIn("measured candidate", result["swarm_stop"]["reason"])
        self.assertEqual(sum(action == "evaluate" for action, _ in self.worker.calls), 3)

    def test_resume_after_assignment_commit_before_attempt_begin(self):
        self.client.plan_callback = lambda data, number: self.plan(data, ["atomic-resume"])
        self.stop_after_evaluations(1)
        loop = self.loop()
        with patch.object(loop.hypotheses, "begin", side_effect=RuntimeError("Interrupted after assignment commit.")):
            first = loop.run()
        self.assertEqual(first["status"], "request_failure")
        self.assertEqual(self.writers(), [])
        assignments = self.state().get("swarm_assignments")
        self.assertEqual(list(assignments), ["0"])
        self.assertEqual(self.state().get("swarm_next_assignment"), 1)
        second = self.loop().run()
        self.assertEqual(second["status"], "stopped")
        self.assertEqual(len(self.writers()), 1)
        self.assertEqual(self.client.plan_number, 1)
        self.assertEqual(sum(action == "evaluate" for action, _ in self.worker.calls), 1)
        records = self.state().get("swarm_hypotheses")
        self.assertEqual(len(records), 1)
        self.assertEqual(next(iter(records.values()))["attempt_count"], 1)

    def test_worker_infrastructure_result_preserves_unreviewed_attempt(self):
        self.client.plan_callback = lambda data, number: self.plan(data, ["infrastructure"])

        def run(action, run_dir, **fields):
            self.worker.calls.append((action, fields))
            return {"failure_kind": "infrastructure", "error": "The worker environment differs.",
                    "decision": {"accepted": False, "speedup": None}}

        self.worker.run = run
        result = self.loop().run()
        self.assertEqual(result["status"], "infrastructure_failure")
        self.assertEqual(result["task_progress"]["stagnant"], 0)
        self.assertEqual(result["task_progress"]["round"], 0)
        state = self.state()
        self.assertIsNot(state.get("reviewed:0"), True)
        assignment = state.get("swarm_assignments")["0"]
        self.assertEqual(assignment["status"], "evaluated")
        self.assertFalse(assignment.get("review"))
        hypothesis = next(iter(state.get("swarm_hypotheses").values()))
        self.assertEqual(hypothesis["status"], "paused")
        self.assertEqual(hypothesis["attempt_count"], 1)
        self.assertEqual(hypothesis["corrections_used"], 0)
        self.assertIsNone(hypothesis["last_outcome"])

    def test_late_budget_failure_resumes_same_attempt_without_api_fault(self):
        self.config["agent"]["writers"] = 2
        self.worker.operation_accepted = True
        self.worker.model_correct = True
        self.client.plan_callback = lambda data, number: self.plan(data, ["first", "late-budget"])
        transfer_failed = threading.Event()

        def writer(data):
            attempt = self.count_writer(data)
            if data["hypothesis"]["claim"].endswith("late-budget") and attempt == 1:
                self.assertTrue(transfer_failed.wait(5))
                raise BudgetExhausted("The late request cannot reserve its tokens.")

        self.client.writer_callback = writer

        def put(*_):
            if not transfer_failed.is_set():
                transfer_failed.set()
                raise TransferError("The first upload ended the controller while the other writer waited.")

        self.worker.put = put
        self.stop_after_evaluations(2)
        first = self.loop().run()
        self.assertEqual(first["status"], "infrastructure_failure")
        second = self.loop().run()
        self.assertEqual(second["status"], "stopped")
        late = [data for data in self.writers() if data["hypothesis"]["claim"].endswith("late-budget")]
        self.assertEqual(len(late), 2)
        self.assertEqual(late[0]["generation"], late[1]["generation"])
        self.assertEqual(late[0]["attempt_number"], 1)
        self.assertEqual(late[1]["attempt_number"], 1)
        records = self.state().get("swarm_hypotheses")
        self.assertTrue(all(record["attempt_count"] == 1 and record["corrections_used"] == 0 for record in records.values()))
        self.assertTrue(all(record["last_outcome"]["kind"] == "accepted" for record in records.values()))
        self.assertEqual(len(self.state().get("swarm_assignments")), 2)
        self.assertEqual(sum(action == "evaluate" for action, _ in self.worker.calls), 2)


if __name__ == "__main__":
    unittest.main()
