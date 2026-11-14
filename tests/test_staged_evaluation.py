from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from sako.core import digest
from sako.generation import evaluation_seeds
from sako.hypotheses import classify_fault
from sako.shape_selection import groups, validate_entries
from sako.staged_evaluation import clearly_slower, initial_indices, measured_split
from sako.store import Store, TaskBudgetExhausted
from sako.swarm_policy import profile_shares


class StagedEvaluationTests(unittest.TestCase):
    def task(self):
        cases = [{"source_path": path, "generation_step": {"phase": phase},
                  "shapes": [[length, 16]], "strides": [[16, 1]], "dtypes": ["fp16"],
                  "workload": {"mode": "cached_generation"}}
                 for path in ("first", "second") for length, phase in ((7, "prefill"), (1, "decode"))]
        return {"id": "same_math", "separate_operations": True, "paths": ["first", "second"], "cases": cases}

    def test_percentage_allocation_preserves_small_targets_and_total(self):
        catalog = {"tasks": [{"id": key, "priority_ms": cost} for key, cost in
                             (("parent", 40), ("a", 10), ("b", 30), ("other", 60))],
                   "hierarchy": {"regions": [{"id": "parent", "priority_ms": 40, "subsection_ids": ["a", "b", "parent"],
                                              "alternative_task_ids": ["parent"]}]}}
        self.assertEqual(profile_shares(catalog, ["a", "b", "other"], 101), {"a": 10, "b": 30, "other": 61})
        self.assertEqual(profile_shares(catalog, ["parent", "a", "b", "other"], 100),
                         {"parent": 20, "a": 5, "b": 15, "other": 60})
        with TemporaryDirectory() as root:
            store = Store(Path(root) / "state.sqlite")
            store.set("phase_budget", {"task_id": "a", "token_ceiling": 10})
            store.reserve("writer", 8, {"max_total_tokens": 100})
            with self.assertRaises(TaskBudgetExhausted):
                store.reserve("writer", 3, {"max_total_tokens": 100})
            self.assertEqual(store.task_token_usage(), {"a": 8})

    def test_noisy_or_partial_screen_survives(self):
        def screen(samples):
            return {"baseline": {"samples_ms": [1.] * 5}, "candidate": {"samples_ms": samples}}
        fast = screen([.6] * 5)
        slow = screen([1.4] * 5)
        noisy = screen([1.4, 1.51, .99, 1.55, 1.7])
        self.assertFalse(clearly_slower(noisy, 1.05))
        self.assertFalse(all(clearly_slower(p, 1.05) for p in (fast, slow)))
        self.assertTrue(clearly_slower(slow, 1.05))
        res = {"stages": {"outcome": "screened_slow"}, "cases": [{"correct": None,
               "speed_screen": {**slow, "clearly_slower": True}}]}
        self.assertEqual(classify_fault(result=res)["kind"], "speed_screen_rejection")

    def test_each_path_can_keep_a_different_source_and_split_by_same_reference(self):
        task = self.task()
        assignments = groups(task)
        self.assertEqual(len(assignments), 4)
        entries = [{"task_id": task["id"], "candidate_id": group["paths"][0], "paths": group["paths"],
                    "signatures": [group["signature"]]} for group in assignments.values()]
        validate_entries(entries, {task["id"]: task})
        data = {"candidate_id": "tested", "stages": {"outcome": "confirmed"}, "cases": []}
        for case in task["cases"]:
            ratio = 1.3 if case["source_path"] == "first" else .93
            data["cases"].append({"source_path": case["source_path"], "correct": True, "search_reference": {
                "kind": "paired_compile", "baseline": {"samples_ms": [ratio] * 5}, "candidate": {"samples_ms": [1.] * 5}}})
        split = measured_split(task, data, 1.05)
        self.assertEqual({tuple(scope["paths"]) for scope in split["scopes"]}, {("first",), ("second",)})
        for row in data["cases"]:
            row["search_reference"]["baseline"]["samples_ms"] = [.92, 1.07, 1.25, 1.001, 1.08]
        self.assertIsNone(measured_split(task, data, 1.05))

    def test_fixed_generation_uses_one_seed_and_two_initial_checks(self):
        task = self.task()
        config = {"kernel_search": {"evaluation": "staged"}, "correctness_seeds": [1, 2, 3]}
        self.assertEqual(initial_indices(task["cases"], ["second"]), [2, 3])
        self.assertEqual(evaluation_seeds(config, {"mode": "cached_generation"}), [1])
        self.assertEqual(evaluation_seeds(config, {"sequence_length": 8}), [1, 2, 3])
