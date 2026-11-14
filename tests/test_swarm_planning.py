"""Do not spend coordinator calls while pending writers supply no new evidence."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import time
import unittest

import test_swarm_loop as fixtures
from sako.core import read_json


class SwarmPlanningTests(unittest.TestCase):
    def setUp(self):
        fixtures.SwarmLoopTests.setUp(self)
        # These tests isolate coordinator scheduling. Successful hypotheses do
        # not consume the remaining experiment slots with automatic corrections.
        self.worker.operation_accepted = True
        self.worker.model_correct = True

    def loop(self):
        return fixtures.SwarmLoopTests.loop(self)

    def writers(self):
        return fixtures.SwarmLoopTests.writers(self)

    def test_one_writer_starts_queued_ideas_while_the_gpu_is_busy(self):
        self.config["agent"].update(writers=1, max_task_experiments=3)
        self.client.width = 3
        gpu_started, release, all_written = (threading.Event() for _ in range(3))
        run_job = self.worker.run
        def run(action, directory, **fields):
            if action == "evaluate" and not gpu_started.is_set():
                gpu_started.set()
                if not release.wait(10):
                    raise AssertionError("The test did not release the GPU.")
            return run_job(action, directory, **fields)
        def writer(data):
            if data["generation"] == 2:
                all_written.set()
        self.worker.run = run
        self.client.writer_callback = writer
        loop = self.loop()
        with ThreadPoolExecutor(max_workers=1) as runner:
            future = runner.submit(loop.run)
            try:
                self.assertTrue(gpu_started.wait(5))
                self.assertTrue(all_written.wait(5), "The GPU must not occupy a writer slot.")
                self.assertEqual(self.client.plan_number, 1)
                self.assertFalse(future.done())
            finally:
                release.set()
            result = future.result(timeout=15)
        self.assertEqual(sum(action == "evaluate" for action, _ in self.worker.calls), 3)
        self.assertTrue(result["review_complete"])
        self.assertEqual(result["status"], "accepted")


    def test_planner_and_second_gpu_trial_can_run_at_the_same_time(self):
        self.config["agent"].update(writers=1, max_task_experiments=3)
        sibling_written, second_trial, release = (threading.Event() for _ in range(3))
        trial_count = 0
        run_job = self.worker.run
        def run(action, directory, **fields):
            nonlocal trial_count
            if action == "evaluate":
                trial_count += 1
                if trial_count == 1:
                    self.assertTrue(sibling_written.wait(8))
                elif trial_count == 2:
                    second_trial.set()
                    self.assertTrue(release.wait(8))
            return run_job(action, directory, **fields)
        def plan(data, number):
            parents = data["archive_policy"]["eligible_parent_ids"]
            parent = parents[0] if parents else None
            evidence = [parent or "contract:" + data["task"]["id"]]
            if number:
                self.assertTrue(second_trial.wait(5), "The planner must not hold the GPU.")
                try:
                    self.assertEqual(len(data["operation_results"]), 1)
                    self.assertTrue(data["lessons"])
                finally:
                    release.set()
            return {"action": "continue", "reason": "Test independent approaches.", "evidence": evidence,
                    "replan_after": "next_result", "hypotheses": [
                        {"specialty": "memory", "parent_id": parent, "claim": f"Reduce reads {number}-{i}.",
                         "change": f"Change indexing {number}-{i}.", "expected_metric": "Lower latency.",
                         "evidence": evidence} for i in range(1 if number else 2)]}
        self.worker.run = run
        self.client.plan_callback = plan
        self.client.writer_callback = lambda data: sibling_written.set() if data["generation"] == 1 else None
        result = self.loop().run()
        self.assertEqual(result["status"], "accepted", result["swarm_stop"])
        self.assertEqual((trial_count, self.client.plan_number), (3, 2))

    def test_resume_reviews_the_queue_without_repeating_either_writer(self):
        from sako.transport import TransferError
        self.config["agent"].update(writers=1, max_task_experiments=2)
        self.client.width = 2
        sibling_written = threading.Event()
        self.client.writer_callback = lambda data: sibling_written.set() if data["generation"] == 1 else None
        run_job = self.worker.run
        def disconnect(action, directory, **fields):
            if action == "evaluate":
                self.assertTrue(sibling_written.wait(8))
                raise TransferError("The evaluation needs recovery.")
            return run_job(action, directory, **fields)
        self.worker.run = disconnect
        self.assertEqual(self.loop().run()["status"], "infrastructure_failure")
        self.worker.run = run_job
        result = self.loop().run()
        self.assertEqual(result["status"], "accepted")
        self.assertTrue(result["review_complete"])
        self.assertEqual(len(self.writers()), 2)
        self.assertEqual(sum(action == "evaluate" for action, _ in self.worker.calls), 2)

    def pending_initial_writers(self, cap):
        self.config["agent"].update(writers=5, max_task_experiments=cap)
        self.client.width = 2
        started = threading.Event()
        release = threading.Event()
        started_ids = set()
        start_lock = threading.Lock()
        planning_observations = []

        def writer(data):
            if data["generation"] < 2:
                with start_lock:
                    started_ids.add(data["generation"])
                    if len(started_ids) == 2:
                        started.set()
                if not release.wait(8):
                    raise AssertionError("The test did not release the initial writers.")

        def plan(data, number):
            results = data.get("operation_results", [])
            planning_observations.append({"results": results, "lessons": data.get("lessons", []),
                                          "initial_writers_released": release.is_set()})
            evidence = [results[0]["id"]] if results else ["contract:" + data["task"]["id"]]
            ordinary = data.get("archive_policy", {}).get("eligible_parent_ids", [])
            parent = ordinary[0] if ordinary else None
            if parent:
                evidence = [parent]
            width = self.client.width if number == 0 else 1
            return {"action": "continue", "reason": "Test distinct implementations using saved evidence.",
                    "evidence": evidence, "hypotheses": [
                        {"specialty": list(data["specialties"])[index], "parent_id": parent,
                         "claim": f"Method {number}-{index} may reduce memory traffic.",
                         "change": f"Use method {number}-{index}.", "expected_metric": "Median latency.",
                         "evidence": evidence} for index in range(width)]}

        self.client.writer_callback = writer
        self.client.plan_callback = plan
        loop = self.loop()
        with ThreadPoolExecutor(max_workers=1) as runner:
            future = runner.submit(loop.run)
            try:
                self.assertTrue(started.wait(5), "Both initial writers must start.")
                # The loop's normal wait timeout is one second. Leave unused slots
                # available across that timeout without supplying any new result.
                time.sleep(1.2)
                with self.client.lock:
                    before_release = sum(role == "coordinator" for role, _ in self.client.prompts)
            finally:
                release.set()
            result = future.result(timeout=12)
        return result, before_release, planning_observations

    def test_initial_two_writers_do_not_trigger_extra_plans_before_release(self):
        result, before_release, plans = self.pending_initial_writers(cap=2)
        self.assertEqual(before_release, 1)
        self.assertEqual(len(plans), 1)
        self.assertEqual(len(self.writers()), 2)
        self.assertEqual(result["status"], "accepted")

    def test_unused_slots_wait_for_evidence_then_plan_with_results_and_lessons(self):
        self.config["profiler"] = {"enabled": True}  # Old run settings cannot dispatch a removed job.
        result, before_release, plans = self.pending_initial_writers(cap=3)
        self.assertEqual(before_release, 1, "Free slots alone must not trigger another paid plan.")
        self.assertEqual(len(plans), 2)
        self.assertTrue(plans[1]["initial_writers_released"])
        self.assertTrue(plans[1]["results"])
        self.assertTrue(plans[1]["lessons"])
        self.assertEqual(len(self.writers()), 3)
        self.assertEqual(result["status"], "accepted")
        ids = {row["id"] for row in plans[1]["results"]}
        self.assertEqual(len(ids), 2)
        self.assertFalse(any(action == "profile_kernel" for action, _ in self.worker.calls))

    def test_wait_plan_is_not_polled_while_sibling_runs(self):
        self.config["agent"].update(writers=5, max_task_experiments=3)
        waiting_plan = threading.Event()
        release_sibling = threading.Event()

        def writer(data):
            if data["generation"] == 1 and not release_sibling.wait(8):
                raise AssertionError("The test did not release the pending sibling.")

        def plan(data, number):
            ordinary = data.get("archive_policy", {}).get("eligible_parent_ids", [])
            parent = ordinary[0] if ordinary else None
            evidence = [data["operation_results"][0]["id"]] if data["operation_results"] else [
                "contract:" + data["task"]["id"]]
            if parent:
                evidence = [parent]
            if number == 1:
                self.assertTrue(data["inflight"])
                self.assertTrue(data["lessons"])
                waiting_plan.set()
                return {"action": "wait", "reason": "Wait for the remaining sibling's measured result.",
                        "evidence": evidence, "hypotheses": []}
            return {"action": "continue", "reason": "Test a new measured hypothesis.", "evidence": evidence,
                    "replan_after": "next_result",
                    "hypotheses": [{"specialty": "memory", "parent_id": parent,
                        "claim": f"Method {number}-{index} may improve execution.",
                        "change": f"Try a different memory method {number}-{index}.",
                        "expected_metric": "Median latency.", "evidence": evidence}
                        for index in range(2 if number == 0 else 1)]}

        self.client.writer_callback = writer
        self.client.plan_callback = plan
        loop = self.loop()
        with ThreadPoolExecutor(max_workers=1) as runner:
            future = runner.submit(loop.run)
            try:
                self.assertTrue(waiting_plan.wait(5), "The coordinator must reach its evidence-backed wait decision.")
                time.sleep(1.2)
                with self.client.lock:
                    before_release = sum(role == "coordinator" for role, _ in self.client.prompts)
            finally:
                release_sibling.set()
            result = future.result(timeout=12)
        self.assertEqual(before_release, 2)
        self.assertEqual(self.client.plan_number, 3)
        self.assertEqual(len(self.writers()), 3)
        self.assertEqual(result["status"], "accepted")

    def test_default_batch_waits_locally_after_one_writer_finishes(self):
        from concurrent.futures import wait
        from unittest.mock import patch
        self.config["agent"].update(writers=2, max_task_experiments=None)
        release, waiting = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        self.client.width = 2
        self.worker.operation_accepted = self.worker.model_correct = True
        self.client.writer_callback = lambda data: release.wait(8) if data["generation"] else None
        def wait_for_writers(futures, **options):
            if len(futures) == 1 and any(action == "evaluate" for action, _ in self.worker.calls):
                waiting.set()
            return wait(futures, **options)
        loop = self.loop()
        with patch("sako.swarm.wait", wait_for_writers), ThreadPoolExecutor(max_workers=1) as runner:
            future = runner.submit(loop.run)
            try:
                self.assertTrue(waiting.wait(5))
                self.assertEqual(self.client.plan_number, 1)
            finally:
                (loop.directory / "STOP").touch()
                release.set()
            future.result(timeout=12)
        self.assertEqual(len(self.writers()), 2)


if __name__ == "__main__":
    unittest.main()
