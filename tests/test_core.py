from triton_fixtures import RMS_SOURCE
import copy
import math
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sako.core import candidate_id, load_config, score_cases, select_nonoverlapping, validate_candidate
from sako.store import Store, BudgetExhausted, TaskBudgetExhausted, operation_correct


class Contracts(unittest.TestCase):
    def setUp(self):
        self.config = load_config(Path(__file__).resolve().parents[1] / "configs/project.json")
        self.candidate = {"name": "test", "task_id": "linear_1", "strategy": "tile", "backend": "triton",
                          "source": RMS_SOURCE, "references": []}

    def test_project_retains_paired_compile_ranking_in_active_and_historical_configs(self):
        path = Path(__file__).resolve().parents[1] / "configs/project.json"
        self.assertEqual(self.config["kernel_search"]["score_reference"], "paired_compile")
        self.assertEqual(load_config(path, historical=True)["kernel_search"]["score_reference"], "paired_compile")

    def test_names_do_not_defeat_deduplication(self):
        other = {**self.candidate, "name": "renamed", "strategy": "different words"}
        self.assertEqual(candidate_id(self.candidate), candidate_id(other))
        self.assertNotEqual(candidate_id(self.candidate), candidate_id({**other, "task_id": "linear_2"}))

    def test_false_correctness_and_bad_samples_fail(self):
        samples = [2.] * self.config["trials"]
        case = {"correct": True, "baseline": {"samples_ms": samples},
                "candidate": {"samples_ms": [1.] * len(samples)}}
        self.assertTrue(score_cases([case], self.config)["accepted"])
        case["correct"] = False
        self.assertFalse(score_cases([case], self.config)["accepted"])
        case["correct"] = True
        case["candidate"]["samples_ms"][0] = float("nan")
        self.assertFalse(score_cases([case], self.config)["accepted"])
        self.assertFalse(score_cases([], self.config)["accepted"])

    def test_missing_trials_and_one_slow_case_fail(self):
        case = {"correct": True, "baseline": {"samples_ms": [2.] * 15},
                "candidate": {"samples_ms": [1.] * 14}}
        self.assertFalse(score_cases([case], self.config)["accepted"])
        case["candidate"]["samples_ms"] = [3.] * 15
        self.assertFalse(score_cases([case], self.config)["accepted"])

    def test_parent_and_child_cannot_both_be_replaced(self):
        selected = select_nonoverlapping([
            {"paths": ["model.layers.0.mlp"], "priority": 5},
            {"paths": ["model.layers.0.mlp.up_proj"], "priority": 3},
            {"paths": ["model.layers.0.self_attn.__core__"], "priority": 2},
            {"paths": ["model.layers.0.self_attn.q_proj"], "priority": 1}])
        self.assertEqual(len(selected), 3)

    def test_policy_rejects_common_escapes_and_false_labels(self):
        for source in ('sako_run(); system("bad");', 'sako_run(); torch::matmul(a,b);',
                       '#include <cutlass/gemm/device/gemm.h>\nsako_run();'):
            with self.assertRaises(ValueError):
                validate_candidate({**self.candidate, "source": source})


class Persistence(unittest.TestCase):
    def test_task_share_reserves_calls_and_tokens_for_later_priorities(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "state.sqlite")
            config = {"max_calls": 12, "max_total_tokens": 1200}
            store.set("phase_budget", {"task_id": "first", "call_ceiling": 2, "token_ceiling": 300})
            first = store.reserve("writer", 200, config)
            with self.assertRaises(TaskBudgetExhausted):
                store.reserve("writer", 101, config)
            store.finish_call(first, 20)
            store.reserve("writer", 100, config)
            with self.assertRaises(TaskBudgetExhausted):
                store.reserve("analyst", 1, config)
            store.set("phase_budget", {"task_id": "second", "call_ceiling": 6, "token_ceiling": 600})
            self.assertIsInstance(store.reserve("writer", 200, config), int)

    def test_correctness_is_separate_from_speed_acceptance(self):
        self.assertTrue(operation_correct({"cases": [{"correct": True}],
                         "decision": {"accepted": False, "speedup": .8}}))
        self.assertFalse(operation_correct({"cases": [{"correct": False}],
                          "decision": {"accepted": True, "speedup": 2.}}))
        self.assertFalse(operation_correct({"error": "Build failed."}))

    def test_parallel_reservations_obey_budget_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite"
            store = Store(path)
            config = {"max_calls": 3, "max_total_tokens": 1000}
            def reserve(_):
                try:
                    return store.reserve("writer", 300, config)
                except BudgetExhausted:
                    return None
            with ThreadPoolExecutor(max_workers=8) as executor:
                calls = list(executor.map(reserve, range(8)))
            self.assertEqual(sum(x is not None for x in calls), 3)
            self.assertEqual(Store(path).usage(), {"calls": 3, "tokens": 900})
            store.finish_call(next(x for x in calls if x), 20)
            self.assertEqual(store.usage()["tokens"], 620)
            with self.assertRaises(BudgetExhausted):
                Store(path).reserve("writer", 1, config)

    def test_results_and_plan_survive_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.sqlite"
            store = Store(path)
            store.set("plan", {"text": "test"})
            self.assertEqual(Store(path).get("plan"), {"text": "test"})


if __name__ == "__main__":
    unittest.main()
