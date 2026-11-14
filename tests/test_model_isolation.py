"""Avoid extra model jobs after measured combinations pass accuracy."""
from triton_fixtures import RMS_SOURCE

import copy
from pathlib import Path
import tempfile
import unittest

from sako.core import load_config, read_json, write_json
from sako.loop import SearchLoop

ROOT = Path(__file__).resolve().parents[1]


def outcome(correct=True, accepted=False):
    return {"cases": [{"correct": correct, "checks": [{"passed": correct, "max_abs_error": 0. if correct else 1.}],
                       **({"baseline": {"samples_ms": [2.] * 15},
                           "candidate": {"samples_ms": [2.] * 15}} if correct else {})}],
            "decision": {"accepted": accepted, "speedup": 1. if correct else None,
                         "reasons": ["Gain is below the gate." if correct else "Numerical failure."]}}


class Worker:
    def __init__(self):
        self.calls = []
        self.combination, self.individual = outcome(), outcome(accepted=True)
    def put(self, *args):
        pass
    def run(self, action, directory, **fields):
        assert action == "model"
        manifest = read_json(ROOT / fields["manifest"])
        self.calls.append(manifest)
        return copy.deepcopy(self.combination if len(manifest["entries"]) > 1 else self.individual)


class ModelIsolationTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "runs").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=ROOT / "runs")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.config = load_config(ROOT / "configs/project.json")
        self.config["measurement"] = {"mode": "fixed"}
        self.config["agent"]["target_selection"] = "flat"
        tasks = [{"id": name, "kind": "rmsnorm", "paths": [name], "priority_ms": 2 - index,
                  "cases": [], "contract": {"validated": True}} for index, name in enumerate(("old", "new"))]
        write_json(self.directory / "profile/tasks.json", {"config": self.config, "tasks": tasks,
                   "selected_task_ids": ["old", "new"], "hardware": {"fingerprint": "test-gpu"}})
        self.worker = Worker()
        self.entries = []
        self.loop = SearchLoop(self.config, self.directory / "run", self.directory / "profile",
            lambda *_: None, lambda *_: self.worker, lambda *_: {"entries": self.entries})
        self.loop.progress["index"] = 1
        for index, task in enumerate(tasks):
            candidate = {"name": task["id"], "task_id": task["id"], "source": RMS_SOURCE,
                         "strategy": "Test the operation.", "backend": "triton", "references": []}
            key, _ = self.loop.store.add(candidate, "memory", index, None)
            self.loop.owners[key] = index
            self.entries.append({"task_id": task["id"], "candidate_id": key,
                                 "paths": task["paths"], "priority": task["priority_ms"]})
        self.key = self.entries[-1]["candidate_id"]
        self.best = {"manifest": {"entries": self.entries[:1]}, "result": outcome(accepted=True), "generation": -1}
        self.loop.store.set("best_model_measurement", self.best)

    def test_speed_rejection_skips_isolated_job_and_keeps_full_combination_evidence(self):
        result = self.loop.measure_model(2, [])
        self.assertEqual(self.worker.calls, [{"entries": self.entries}])
        self.assertFalse(result["result"]["decision"]["accepted"])
        self.assertEqual(self.loop.store.get("best_model_measurement"), self.best)
        self.assertFalse(self.loop.blocked)
        repair = self.loop.pending[self.key]
        self.assertEqual(repair["stage"], "interaction")
        self.assertEqual(repair["result"]["individual_test"]["status"], "not_measured")
        saved = read_json(self.directory / "run" / repair["evidence"]["path"])
        self.assertIsNone(saved["individual"])
        self.assertEqual(saved["combination"], self.worker.combination)
        self.assertEqual(len(self.loop.store.get("model_measurements")), 1)
        self.loop.measure_model(2, [])
        self.assertEqual(len(self.worker.calls), 1)

    def test_accuracy_failure_still_tests_new_entry_alone(self):
        self.worker.combination = outcome(correct=False)
        self.loop.measure_model(2, [])
        self.assertEqual(self.worker.calls, [{"entries": self.entries}, {"entries": self.entries[-1:]}])
        self.assertFalse(self.loop.blocked)
        self.assertEqual(self.loop.pending[self.key]["stage"], "interaction")

    def test_isolated_accuracy_failure_blocks_only_the_new_entry(self):
        self.worker.combination = self.worker.individual = outcome(correct=False)
        self.loop.measure_model(2, [])
        self.assertEqual(len(self.worker.calls), 2)
        self.assertEqual(set(self.loop.blocked), {self.key})
        self.assertEqual(self.loop.pending[self.key]["stage"], "model")
        self.assertEqual(self.loop.store.get("best_model_measurement"), self.best)

    def test_unknown_or_unsafe_combination_is_not_a_speed_only_rejection(self):
        for result in ({"error": "A worker failure.", "decision": {"accepted": False}},
                       {**outcome(), "reference_state_intact": False}):
            with self.subTest(result=result):
                self.worker.combination = result
                self.loop.store.set("model_measurements", {})
                self.worker.calls.clear()
                self.loop.measure_model(2, [])
                self.assertEqual(len(self.worker.calls), 2)

    def test_single_entry_reuses_its_own_measurement(self):
        self.entries[:] = self.entries[-1:]
        self.worker.individual = outcome()
        self.loop.measure_model(2, [])
        self.assertEqual(len(self.worker.calls), 1)
        repair = self.loop.pending[self.key]
        self.assertEqual(repair["stage"], "model_performance")
        saved = read_json(self.directory / "run" / repair["evidence"]["path"])
        self.assertEqual(saved["combination"], saved["individual"])
        self.assertNotIn("individual_test", saved)


if __name__ == "__main__":
    unittest.main()
