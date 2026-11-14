import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sako.store import BudgetExhausted, TaskBudgetExhausted, Store


class TokenBudgetStore(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "state.sqlite"
        self.store = Store(self.path)
        self.config = {"max_calls": None, "max_total_tokens": 1000}

    def test_rejected_task_reservation_updates_admission_without_spending_tokens(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from sako.swarm import SwarmLoop
        from sako.swarm_policy import admit_experiments
        self.store.set('phase_budget', {'task_id':'gate', 'token_ceiling':250})
        before = {'writer':140, 'coordinator':100}
        self.assertTrue(admit_experiments(self.config, 0, before, 1, task_remaining=250)['admit'])
        with self.assertRaises(TaskBudgetExhausted):
            self.store.reserve('coordinator', 300, self.config)
        self.assertEqual(self.store.usage()['tokens'], 0)
        self.assertEqual(self.store.usage()['calls'], 0)
        with patch('sako.swarm.estimate_costs', return_value=dict(before)):
            costs = SwarmLoop.costs(SimpleNamespace(store=self.store, limits=self.config))
        self.assertEqual(costs['coordinator'], 300)
        self.assertFalse(admit_experiments(self.config, 0, costs, 1, task_remaining=250)['admit'])
        self.store.set('phase_budget', {'task_id':'gate', 'token_ceiling':500})
        call = self.store.reserve('coordinator', 300, self.config)
        self.assertEqual(self.store.usage()['tokens'], 300)
        self.store.finish_call(call, 200)
        self.assertEqual(self.store.usage()['tokens'], 200)

    def test_rejection_floors_keep_the_largest_request_for_each_role(self):
        self.store.set('phase_budget', {'task_id':'gate', 'token_ceiling':100})
        for role, amount in [('coordinator',400), ('coordinator',300), ('writer_memory',250), ('writer_compute',350)]:
            with self.assertRaises(TaskBudgetExhausted):
                self.store.reserve(role, amount, self.config)
        self.assertEqual(self.store.get('request_reservation_floors'), {'coordinator':400,'writer':350})
        self.assertEqual(self.store.usage(), {'calls':0,'tokens':0})

    def test_optional_call_cap_does_not_end_token_budget(self):
        for index in range(15):
            self.store.finish_call(self.store.reserve("writer", 50, self.config), 5)
        self.assertEqual(self.store.usage(), {"calls": 15, "tokens": 75})

    def test_metadata_is_separate_from_generation_cap(self):
        self.config["max_calls"] = 1
        self.store.finish_call(self.store.reserve("analyst:count_tokens", 0, self.config), 0)
        self.store.finish_call(self.store.reserve("writer", 80, self.config), 12)
        self.store.finish_call(self.store.reserve("analyst:models.get", 0, self.config), 0)
        with self.assertRaises(BudgetExhausted):
            self.store.reserve("writer", 1, self.config)
        stats = self.store.call_stats()
        self.assertEqual(stats["generation_calls"], 1)
        self.assertEqual(stats["auxiliary_calls"], 2)

    def test_concurrent_reservations_cannot_spend_review_reserve(self):
        self.store.set("budget_reserve", {"protected_tokens": 400,
                                          "allowed_roles": ["performance_analyst", "writer_repair"]})
        def reserve(_):
            try:
                return self.store.reserve("writer", 200, self.config)
            except BudgetExhausted:
                return None
        with ThreadPoolExecutor(max_workers=8) as pool:
            calls = list(pool.map(reserve, range(8)))
        self.assertEqual(len([call for call in calls if call]), 3)
        self.assertIsInstance(self.store.reserve("performance_analyst", 200, self.config), int)
        self.assertIsInstance(self.store.reserve("writer_repair", 200, self.config), int)
        with self.assertRaises(BudgetExhausted):
            self.store.reserve("writer_repair", 1, self.config)

    def test_unknown_outcomes_remain_committed_after_resume(self):
        unknown = self.store.reserve("writer", 400, self.config)
        self.store.finish_call(unknown, None, "failed")
        self.store.reserve("writer", 300, self.config)
        known = self.store.reserve("analyst", 200, self.config)
        self.store.finish_call(known, 50)
        stats = Store(self.path).budget_snapshot(self.config)
        self.assertEqual(stats["reported_tokens"], 50)
        self.assertEqual(stats["unresolved_tokens"], 700)
        self.assertEqual(stats["inflight_tokens"], 300)
        self.assertEqual(stats["tokens"], 750)
        self.assertEqual(stats["remaining_tokens"], 250)
        with self.assertRaises(BudgetExhausted):
            Store(self.path).reserve("writer", 251, self.config)

    def test_repair_writer_cannot_spend_the_final_review_reserve(self):
        self.store.set("budget_reserve", {"protected_tokens": 600,
            "allowed_roles": ["performance_analyst"],
            "role_protected_tokens": {"writer_repair": 200}})
        with self.assertRaises(BudgetExhausted):
            self.store.reserve("writer_fresh", 500, self.config)
        self.store.reserve("writer_repair", 800, self.config)
        with self.assertRaises(BudgetExhausted):
            self.store.reserve("writer_repair", 1, self.config)
        self.store.reserve("performance_analyst", 200, self.config)
        self.assertEqual(self.store.usage()["tokens"], 1000)

    def test_role_permissions_cannot_raise_the_hard_cap(self):
        self.store.set("budget_reserve", {"protected_tokens": 400, "allowed_roles": ["writer"]})
        with self.assertRaises(BudgetExhausted):
            self.store.reserve("writer", 1001, self.config)


if __name__ == "__main__":
    unittest.main()
