"""Exercise independent writers, durable results, and real acceptance boundaries."""
from triton_fixtures import RMS_SOURCE

import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from sako.core import load_config, read_json, write_json
from sako.reporting import build_swarm_report
from sako.search import make_manifest
from sako.store import BudgetExhausted, Store
from sako.swarm import SwarmLoop
from sako.transport import TransferError


ROOT = Path(__file__).resolve().parents[1]


class SwarmClient:
    def __init__(self, width=1):
        self.width = width
        self.prompts = []
        self.lock = threading.Lock()
        self.writer_callback = None
        self.plan_callback = None
        self.plan_number = 0
        self.active = self.maximum = 0

    def call(self, role, prompt, system):
        data = json.loads(prompt)
        with self.lock:
            self.prompts.append((role, data))
        if role == "coordinator":
            number = self.plan_number
            self.plan_number += 1
            if self.plan_callback:
                return self.plan_callback(data, number)
            hypotheses = []
            ordinary = data.get("archive_policy", {}).get("eligible_parent_ids", [])
            parent = ordinary[0] if ordinary else None
            for index, specialty in enumerate(list(data["specialties"])[:self.width]):
                hypotheses.append({"specialty": specialty, "parent_id": parent,
                    "claim": f"Test independent method {number}-{index}.",
                    "change": f"Change the implementation method {number}-{index}.",
                    "expected_metric": "Lower median latency.",
                    "evidence": [parent] if parent else ["contract:" + data["task"]["id"]]})
            return {"action": "continue", "reason": "Compare distinct methods.",
                    "evidence": ["contract:" + data["task"]["id"]], "hypotheses": hypotheses}
        if not role.startswith("writer_"):
            raise AssertionError("The swarm called an old support role: " + role)
        with self.lock:
            self.active += 1
            self.maximum = max(self.maximum, self.active)
        try:
            if self.writer_callback:
                self.writer_callback(data)
            number = data["generation"]
            return {"name": "kernel_" + str(number), "task_id": data["task"]["id"],
                    "strategy": "Test the assigned method.", "backend": "triton", "references": [],
                    "source": RMS_SOURCE + '# candidate ' + str(number)}
        finally:
            with self.lock:
                self.active -= 1


class SwarmWorker:
    def __init__(self):
        self.calls = []
        self.fail_upload = False
        self.fail_model_transfer = False
        self.operation_accepted = False
        self.model_correct = False
        self.stop_after_model = False

    def put(self, *_):
        if self.fail_upload:
            self.fail_upload = False
            raise TransferError("A temporary upload failure.")

    def run(self, action, run_dir, **fields):
        self.calls.append((action, fields))
        if action == "model" and self.fail_model_transfer:
            self.fail_model_transfer = False
            raise TransferError("The existing model job needs result recovery.")
        correct = True if action == "evaluate" else self.model_correct
        accepted = self.operation_accepted if action == "evaluate" else self.model_correct
        if action == "model" and self.stop_after_model:
            (Path(run_dir) / "STOP").touch()
        return {"scope": "operation" if action == "evaluate" else "model",
                "cases": [{"correct": correct, "inputs_unchanged": True,
                           "checks": [{"seed": 1729, "passed": correct, "max_abs_error": 0 if correct else .6}],
                           **({"baseline": {"samples_ms": [10.] * 15},
                               "candidate": {"samples_ms": [5. if accepted else 10.] * 15}} if correct else {})}],
                "decision": {"accepted": accepted, "speedup": 2. if accepted else 1. if correct else None,
                             "reasons": [] if accepted else ["Gain is below the gate." if correct else "Numerical checks failed."]}}


class SwarmLoopTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "runs").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=ROOT / "runs")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.config = load_config(ROOT / "configs/project.json")
        self.config["measurement"] = {"mode": "fixed"}
        self.config.pop("model_trial_base", None)
        self.config["agent"]["target_selection"] = "flat"
        self.config["agent"]["design_portfolio"] = False
        self.config["kernel_backend"] = "triton"
        # These tests retain the earlier immediate integration policy.
        self.config["kernel_search"] = {"enabled": False}
        self.config["agent"].update(search_mode="swarm", adaptive_budget=True,
            writers=5, islands=["memory", "compute", "scheduling", "fusion", "registers"],
            max_parallel_calls=5, top_k_operations=1, max_task_experiments=1)
        self.config["profiler"] = {"enabled": False}
        self.config["diagnostics"] = {"enabled": False}
        self.client, self.worker = SwarmClient(), SwarmWorker()
        task = {"id": "rmsnorm_1", "kind": "rmsnorm", "paths": ["model.norm"],
                "priority_ms": 4., "cases": [], "contract": {"validated": True}}
        write_json(self.directory / "profile/tasks.json", {"config": self.config, "tasks": [task],
                   "selected_task_ids": ["rmsnorm_1"], "hardware": {"fingerprint": "test-gpu"}})

    def loop(self):
        return SwarmLoop(self.config, self.directory / "run", self.directory / "profile",
                         lambda *_: self.client, lambda *_: self.worker,
                         make_manifest)

    def writers(self):
        return [data for role, data in self.client.prompts if role.startswith("writer_")]

    def test_coordinator_cannot_skip_a_task_using_only_contract_evidence(self):
        self.client.plan_callback = lambda data, number: {
            "action": "move_on", "reason": "No measured candidate exists.",
            "evidence": ["contract:" + data["task"]["id"]], "hypotheses": []}
        result = self.loop().run()
        self.assertEqual(result["status"], "request_failure")
        self.assertIn("measured candidate", result["swarm_stop"]["reason"])
        self.assertEqual(self.client.plan_number, 2)
        self.assertEqual(self.writers(), [])
        self.assertEqual(result["task_progress"]["completed"], [])

    def test_five_writers_overlap_and_old_support_roles_do_not_run(self):
        self.config["agent"]["max_task_experiments"] = 5
        self.client.width = 5
        barrier = threading.Barrier(5)
        self.client.writer_callback = lambda _: barrier.wait(timeout=5)
        result = self.loop().run()
        self.assertEqual(result["status"], "completed_no_gain")
        self.assertEqual(len(self.writers()), 5)
        self.assertEqual(self.client.maximum, 5)
        report = build_swarm_report(self.directory / "run", self.config)
        self.assertEqual(report["concurrency"]["max_observed_active_writers"], 5)
        self.assertEqual({role for role, _ in self.client.prompts if not role.startswith("writer_")}, {"coordinator"})
        self.assertEqual(len(report["lessons"]), 5)

    def test_free_writer_receives_new_lessons_before_slow_sibling_finishes(self):
        self.config["agent"].update(writers=2, max_task_experiments=3)
        self.client.width = 2
        refill_started = threading.Event()
        slow_was_released = []

        def writer(data):
            if data["generation"] == 1:
                slow_was_released.append(refill_started.wait(5))
            elif data["generation"] == 2:
                self.assertTrue(data["lessons"])
                refill_started.set()

        self.client.writer_callback = writer
        result = self.loop().run()
        self.assertEqual(result["status"], "completed_no_gain")
        self.assertEqual(slow_was_released, [True])
        self.assertEqual(len(self.writers()), 3)
        events = Store(self.directory / "run/state.sqlite").get("swarm_events")
        started = next(i for i, event in enumerate(events) if event["kind"] == "writer_started" and event["assignment_id"] == "experiment-00002")
        finished = next(i for i, event in enumerate(events) if event["kind"] == "writer_finished" and event["assignment_id"] == "experiment-00001")
        self.assertLess(started, finished)

    def test_resume_generated_candidate_does_not_repeat_writer_or_operation(self):
        self.config["agent"]["writers"] = 1
        self.worker.fail_upload = True
        first = self.loop().run()
        self.assertEqual(first["status"], "infrastructure_failure")
        self.assertEqual(len(self.writers()), 1)
        self.assertEqual(len(self.worker.calls), 0)
        second = self.loop().run()
        self.assertEqual(second["status"], "completed_no_gain")
        self.assertEqual(len(self.writers()), 1)
        self.assertEqual(sum(action == "evaluate" for action, _ in self.worker.calls), 1)
        self.assertTrue(second["review_complete"])

    def test_resume_after_model_transfer_uses_saved_operation_result(self):
        self.config["agent"]["writers"] = 1
        self.worker.operation_accepted = True
        self.worker.fail_model_transfer = True
        first = self.loop().run()
        self.assertEqual(first["status"], "infrastructure_failure")
        second = self.loop().run()
        self.assertEqual(second["status"], "completed_no_gain")
        self.assertEqual(len(self.writers()), 1)
        self.assertEqual(sum(action == "evaluate" for action, _ in self.worker.calls), 1)
        self.assertEqual(sum(action == "model" for action, _ in self.worker.calls), 2)
        self.assertFalse(second["model_result"]["decision"]["accepted"])

    def test_correct_operation_bad_model_preserves_repair_on_stop(self):
        self.config["agent"].update(writers=1, max_task_experiments=None)
        self.worker.operation_accepted = True
        self.worker.stop_after_model = True
        result = self.loop().run()
        self.assertEqual(result["status"], "stopped")
        self.assertTrue(result["pending_repairs"])
        self.assertFalse(result["model_result"]["decision"]["accepted"])
        report = build_swarm_report(self.directory / "run", self.config)
        self.assertFalse(report["accepted_model_gain"])
        self.assertEqual(report["candidates"][0]["model_status"], "rejected_alone")
        self.assertEqual(report["stop"]["decision_source"], "controller")

    def test_experiment_cap_limits_first_parallel_dispatch(self):
        self.client.width = 5
        result = self.loop().run()
        self.assertEqual(len(self.writers()), 1)
        self.assertEqual(result["task_progress"]["completed"][0]["reason"], "task_experiment_limit")

    def test_budget_failure_keeps_exhausted_hypothesis_and_exact_stop_reason(self):
        self.config["agent"].update(writers=1, max_task_experiments=None)
        self.worker.operation_accepted = True

        def plan(data, number):
            if number:
                raise BudgetExhausted("The next review cannot fit the hard cap.")
            evidence = ["contract:" + data["task"]["id"]]
            return {"action": "continue", "reason": "Test one method.", "evidence": evidence,
                    "hypotheses": [{"specialty": "memory", "parent_id": None,
                        "claim": "Vector loads may reduce transactions.", "change": "Use aligned vector loads.",
                        "expected_metric": "Lower median latency.", "evidence": evidence}]}

        self.client.plan_callback = plan
        result = self.loop().run()
        self.assertEqual(result["status"], "budget_exhausted")
        records = Store(self.directory / "run/state.sqlite").get("swarm_hypotheses", {})
        self.assertEqual(len(records), 1)
        hypothesis = next(iter(records.values()))
        self.assertEqual(hypothesis["status"], "exhausted")
        self.assertEqual(hypothesis["attempt_count"], 3)
        self.assertEqual(result["pending_repairs"], {})
        self.assertEqual(result["active_task"], "rmsnorm_1")
        self.assertEqual(result["task_progress"]["completed"], [])
        self.assertIn("next review cannot fit", result["swarm_stop"]["reason"])

    def test_pre_request_failure_is_bounded_and_recorded(self):
        self.config["agent"].update(writers=1, max_task_experiments=3, max_stagnant_rounds=2)
        loop = self.loop()

        def failed_retrieval(*args, **kwargs):
            raise ValueError("The reference context cannot be read.")

        loop.write_swarm = failed_retrieval
        result = loop.run()
        self.assertEqual(result["status"], "completed_no_gain")
        self.assertEqual(len(self.writers()), 0)
        self.assertEqual(len(loop.work), 3)
        self.assertTrue(all(work["status"] == "failed" for work in loop.work.values()))
        self.assertEqual(result["task_progress"]["completed"][0]["reason"], "task_experiment_limit")
        records = loop.store.get("swarm_hypotheses", {})
        self.assertEqual(len(records), 1)
        self.assertEqual(next(iter(records.values()))["status"], "exhausted")


if __name__ == "__main__":
    unittest.main()
