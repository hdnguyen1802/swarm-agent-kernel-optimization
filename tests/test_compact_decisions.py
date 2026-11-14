from triton_fixtures import RMS_SOURCE
"""Keep scheduling context small and validate exact evidence aliases."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from sako.core import load_config, write_json
from sako.loop import SearchLoop
from sako.search import make_manifest
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[1]


def operation_result(score=1.2):
    cases = []
    for index in range(6):
        cases.append({
            "case_index": index, "seed": 1729 + index,
            "workload": {"batch_size": 1, "sequence_length": 128},
            "correct": True,
            "checks": [{"passed": True, "max_abs_error": .0001},
                       {"passed": True, "max_abs_error": .0002}],
            "baseline": {"median_ms": 2., "p95_ms": 2.1, "samples_ms": [2.] * 150},
            "candidate": {"median_ms": 2. / score, "p95_ms": 2. / score,
                          "samples_ms": [2. / score] * 150},
            "build": {"resource_reports": [{"registers_per_thread": 48},
                                             {"registers_per_thread": 96}],
                      "traceback": "full compiler evidence " * 3000}})
    return {"cases": cases, "decision": {"accepted": True, "speedup": score, "reasons": []}}


class CompactOperationFeedback(unittest.TestCase):
    def test_summary_keeps_measurements_and_does_not_change_raw_results(self):
        raw = operation_result()
        before = copy.deepcopy(raw)
        result = SearchLoop.operation_feedback(raw)
        self.assertEqual(raw, before)
        self.assertEqual(result["decision"], raw["decision"])
        self.assertEqual(len(result["cases"]), 6)
        case = result["cases"][0]
        self.assertTrue(case["correct"])
        self.assertEqual(case["max_abs_error"], .0002)
        self.assertEqual(case["baseline_ms"], 2.)
        self.assertAlmostEqual(case["candidate_ms"], 2. / 1.2)
        self.assertEqual(case["registers_per_thread"], 96)
        encoded = json.dumps(result)
        self.assertNotIn("samples_ms", encoded)
        self.assertNotIn("full compiler evidence", encoded)
        self.assertLess(len(encoded), 6000)

    def test_error_summary_keeps_the_reason_with_a_bounded_message(self):
        raw = {"error": "Compilation failed: " + "x" * 30000 + " final raw detail",
               "traceback": "raw traceback " * 3000,
               "decision": {"accepted": False, "speedup": None}}
        result = SearchLoop.operation_feedback(raw)
        self.assertIn("Compilation failed", json.dumps(result))
        self.assertLess(len(json.dumps(result)), 3000)
        self.assertTrue(raw["error"].endswith("final raw detail"))

    def test_nonfinite_output_remains_an_explicit_numerical_failure(self):
        raw = {"cases": [{"correct": False, "checks": [
            {"passed": True, "max_abs_error": 0.},
            {"passed": False, "max_abs_error": None}]}],
            "decision": {"accepted": False, "speedup": None}}
        result = SearchLoop.operation_feedback(raw)
        self.assertFalse(result["cases"][0]["correct"])
        self.assertIsNone(result["cases"][0]["max_abs_error"])


class CompactRanking(unittest.TestCase):
    def setUp(self):
        (ROOT / "runs").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=ROOT / "runs")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.config = load_config(ROOT / "configs/project.json")
        self.config["agent"].update(max_calls=100, top_k_operations=1, target_selection="flat")
        task = {"id": "linear_0", "kind": "linear", "paths": ["model.layer0"],
                "priority_ms": 4., "cases": [], "contract": {"validated": True}}
        write_json(self.directory / "profile/tasks.json", {
            "config": self.config, "tasks": [task], "selected_task_ids": ["linear_0"],
            "hardware": {"fingerprint": "test-gpu"}})
        self.client, self.worker = Mock(), Mock()
        self.loop = SearchLoop(
            self.config, self.directory / "run", self.directory / "profile",
            lambda *args: self.client, lambda *args: self.worker,
            make_manifest)

    def test_ranking_keeps_latest_round_and_best_of_each_island(self):
        expected = set()
        all_rows = {}
        for generation, score in [(0, 1.4), (1, 1.2), (2, 1.1)]:
            for slot, island in enumerate(self.config["agent"]["islands"]):
                candidate = {"name": f"candidate_{generation}_{slot}", "task_id": "linear_0",
                             "backend": "triton", "references": [], "strategy": "Tune tile size.",
                             "source": RMS_SOURCE + f"# {generation} {slot}"}
                key, _ = self.loop.store.add(candidate, island, generation, None)
                raw = operation_result(score)
                self.loop.store.record(key, raw)
                all_rows[key] = raw
                if generation in (0, 2):
                    expected.add(key)
        result = self.loop.review_ranking()
        self.assertEqual({row["id"] for row in result}, expected)
        self.assertEqual(len(result), 10)
        self.assertLess(len(json.dumps(result)), 24000)
        self.assertNotIn("full compiler evidence", json.dumps(result))
        for row in self.loop.store.candidates():
            self.assertEqual(row["result"], all_rows[row["id"]])





if __name__ == "__main__":
    unittest.main()
