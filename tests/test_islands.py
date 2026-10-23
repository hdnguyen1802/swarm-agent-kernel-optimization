import copy
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from sako.core import load_config, write_json, read_json
from sako.islands import ParentPolicy, ParentSelectionError, STRATEGIES
from sako.loop import SearchLoop
from sako.store import Store

ROOT = Path(__file__).resolve().parents[1]


class IslandPolicy(unittest.TestCase):
    def setUp(self):
        (ROOT / "runs").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=ROOT / "runs")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.config = load_config(ROOT / "configs/project.json")
        self.store = Store(self.directory / "state.sqlite")
        self.policy = ParentPolicy(self.store, self.config["agent"], self.directory, ["linear_0", "linear_1"])

    def add(self, name, island="parallel", task="linear_0", score=1.2, accepted=True, generation=0):
        candidate = {"name": name, "task_id": task, "source": "sako_run(); // " + name,
                     "strategy": "Test an experiment.", "uses_cutlass": False, "references": []}
        key, _ = self.store.add(candidate, island, generation, None)
        self.store.record(key, {"decision": {"accepted": accepted, "speedup": score}})
        return key, candidate

    def test_empty_island_starts_fresh_despite_foreign_candidates(self):
        self.add("foreign", score=1.9)
        a = self.policy.select(0, 1, "linear_0", None)
        self.assertIsNone(a["parent_id"])
        self.assertEqual(a["selection"]["reason"], "fresh_start")
        self.assertIsNone(a["selection"]["source_island"])

    def test_normal_round_selects_own_best_parent(self):
        self.add("foreign", score=3.)
        self.add("own_slow", island="memory", score=1.1)
        best, _ = self.add("own_best", island="memory", score=1.3)
        a = self.policy.select(0, 1, "linear_0", None)
        self.assertEqual(a["parent_id"], best)
        self.assertEqual(a["selection"]["reason"], "own_island")
        self.assertEqual(a["selection"]["source_island"], "memory")

    def test_failed_or_blocked_local_candidates_do_not_trigger_foreign_fallback(self):
        self.add("foreign", score=2.)
        self.add("failed", island="memory", accepted=False)
        blocked, _ = self.add("blocked", island="memory")
        self.store.set("blocked_model_candidates", {blocked: {"result": "failed"}})
        self.assertEqual(self.policy.select(0, 1, "linear_0", None)["selection"]["reason"], "fresh_start")

    def test_migration_matches_task_and_uses_one_writer_per_island(self):
        foreign, _ = self.add("foreign", task="linear_1")
        self.assertIsNone(self.policy.plan_migration(0, {}))
        self.assertIsNone(self.policy.plan_migration(1, {}))
        event = self.policy.plan_migration(2, {})
        self.assertEqual(len(event["transfers"]), 4)
        a = self.policy.select(1, 2, "linear_0", None)
        self.assertEqual(a["parent_id"], foreign)
        self.assertEqual(a["task_id"], "linear_1")
        self.assertEqual(a["selection"]["migration_event"], "after_round_2")
        self.assertEqual(a["selection"]["source_island"], "parallel")
        self.assertEqual(self.policy.select(0, 2, "linear_0", None)["selection"]["reason"], "fresh_start")
        self.assertEqual(self.policy.select(1, 3, "linear_0", None)["selection"]["reason"], "fresh_start")

    def test_no_eligible_migrant_is_logged(self):
        failed, _ = self.add("failed", accepted=False)
        blocked, _ = self.add("blocked")
        self.add("unknown", task="unknown_task")
        self.store.set("blocked_model_candidates", {blocked: {}})
        event = self.policy.plan_migration(2, {})
        self.assertTrue(all(t["reason"] == "no_eligible_candidate" for t in event["transfers"]))
        self.assertEqual(self.policy.select(1, 2, "linear_0", None)["selection"]["migration_skipped"],
                         "no_eligible_candidate")

    def test_repair_keeps_source_task_and_writer_over_planned_migrant(self):
        self.add("migrant", task="linear_1")
        key, _ = self.add("failed_own", island="memory", accepted=False)
        event = self.policy.plan_migration(2, {})
        repair = {"candidate_id": key, "task_id": "linear_0", "repairs_used": 0, "slot": 1}
        a = self.policy.select(1, 2, "linear_1", repair)
        self.assertEqual((a["parent_id"], a["task_id"], a["slot"]), (key, "linear_0", 1))
        self.assertEqual(a["selection"]["reason"], "repair")
        self.assertEqual(a["selection"]["migration_skipped"], "repair_priority")
        self.assertIsNone(a["selection"]["migration_event"])
        self.assertEqual(self.policy.plan_migration(2, {}), event)

    def test_pending_repair_skips_migration_at_planning(self):
        self.add("migrant")
        event = self.policy.plan_migration(2, {"failed": {"slot": 1}})
        transfer = next(t for t in event["transfers"] if t["slot"] == 1)
        self.assertEqual(transfer["reason"], "repair_priority")
        self.assertEqual(transfer["status"], "skipped")

    def test_foreign_parent_without_authorization_fails_before_gemini(self):
        foreign, _ = self.add("foreign")
        a = self.policy.select(0, 1, "linear_0", None)
        a["parent_id"] = foreign
        a["selection"].update(candidate_id=foreign, source_island="memory", origin_island="parallel", reason="own_island")
        loop = object.__new__(SearchLoop)
        loop.parent_policy, loop.client = self.policy, Mock()
        with self.assertRaisesRegex(ParentSelectionError, "Foreign parent"):
            loop.write(a, 1, {})
        loop.client.call.assert_not_called()

    def test_migration_requires_saved_event_and_correct_round(self):
        self.add("foreign")
        self.policy.plan_migration(2, {})
        a = self.policy.select(1, 2, "linear_0", None)
        self.store.set("migration:after_round_2", None)
        with self.assertRaises(ParentSelectionError):
            self.policy.validate(a, 2)
        a["selection"].update(generation=3, round=4)
        with self.assertRaisesRegex(ParentSelectionError, "schedule"):
            self.policy.validate(a, 3)

    def test_tampered_migrant_task_or_writer_is_rejected(self):
        self.add("foreign")
        self.policy.plan_migration(2, {})
        a = self.policy.select(1, 2, "linear_0", None)
        wrong = copy.deepcopy(a)
        wrong["task_id"] = wrong["selection"]["task_id"] = "linear_1"
        with self.assertRaises(ParentSelectionError):
            self.policy.validate(wrong, 2)
        wrong = copy.deepcopy(a)
        wrong["slot"] = wrong["selection"]["slot"] = 0
        with self.assertRaises(ParentSelectionError):
            self.policy.validate(wrong, 2)

    def test_resume_keeps_event_and_decision_instead_of_picking_new_parent(self):
        old, _ = self.add("original", score=1.2)
        event = self.policy.plan_migration(2, {})
        a = self.policy.select(1, 2, "linear_0", None)
        self.store.set("assignments:2", [a])
        self.policy.save([a], 2)
        self.add("later_better", score=2., generation=1)
        resumed = ParentPolicy(Store(self.directory / "state.sqlite"), self.config["agent"],
                               self.directory, ["linear_0", "linear_1"])
        self.assertEqual(resumed.plan_migration(2, {}), event)
        saved = resumed.store.get("assignments:2")
        resumed.save(saved, 2)
        self.assertEqual(saved[0]["parent_id"], old)
        record = read_json(self.directory / "selections/round-0003.json")["selections"][0]
        self.assertEqual(record, a["selection"])

    def test_completed_assignment_stays_auditable_after_parent_is_blocked(self):
        key, _ = self.add("migrant")
        self.policy.plan_migration(2, {})
        a = self.policy.select(1, 2, "linear_0", None)
        self.store.set("submissions:2", {"1": "already_submitted"})
        self.store.set("blocked_model_candidates", {key: {}})
        self.policy.save([a], 2)
        with self.assertRaisesRegex(ParentSelectionError, "not eligible"):
            self.policy.validate(a, 2)

    def test_independent_duplicate_is_tracked_without_silent_transfer(self):
        key, candidate = self.add("same_source")
        fresh = self.policy.select(0, 0, "linear_0", None)
        self.assertEqual(fresh["selection"]["reason"], "fresh_start")
        duplicate, inserted = self.store.add(candidate, "memory", 0, None)
        self.assertFalse(inserted)
        self.assertEqual(key, duplicate)
        self.store.record_emission(0, 0, key, "memory", True, fresh["selection"])
        a = self.policy.select(0, 1, "linear_0", None)
        self.assertEqual(a["selection"]["reason"], "own_island")
        self.assertEqual(a["selection"]["source_island"], "memory")
        self.assertEqual(a["selection"]["origin_island"], "parallel")
        self.assertEqual(Store(self.directory / "state.sqlite").emissions()[0]["duplicate"], 1)

    def test_each_island_receives_distinct_experiment_instructions(self):
        tasks = [{"id": "linear_0", "kind": "linear", "priority_ms": 1., "paths": [], "cases": []}]
        write_json(self.directory / "profile/tasks.json", {"config": self.config, "tasks": tasks})
        client = Mock()
        client.call.return_value = {"name": "example", "task_id": "linear_0", "source": "sako_run();",
                                    "strategy": "Experiment.", "uses_cutlass": False, "references": []}
        loop = SearchLoop(self.config, self.directory, self.directory / "profile", lambda *args: client,
                          lambda *args: Mock(), lambda *args: [], lambda *args: {})
        assigned = loop.assignments(0, {"task_order": []})
        for slot in (0, 2, 4, 6):
            loop.write(assigned[slot], 0, {})
        prompts = [json.loads(c.args[1]) for c in client.call.call_args_list]
        self.assertEqual({p["island"] for p in prompts}, set(STRATEGIES))
        self.assertEqual(len({p["island_strategy"]["goal"] for p in prompts}), 4)
        self.assertTrue(all(len(p["island_strategy"]["experiments"]) == 3 for p in prompts))
        self.assertTrue(all(p["parent"] is None and p["parent_selection"]["reason"] == "fresh_start" for p in prompts))

    def test_three_round_routine_uses_recorded_migration(self):
        tasks = [{"id": f"linear_{i}", "kind": "linear", "priority_ms": 8-i,
                  "paths": ["model.shared"], "cases": []} for i in range(8)]
        self.config["agent"]["generations"] = 3
        write_json(self.directory / "profile/tasks.json", {"config": self.config, "tasks": tasks})
        prompts = []
        class Client:
            def call(client, role, prompt, system):
                data = json.loads(prompt)
                if not role.startswith("writer_"):
                    return {"task_order": [], "notes": "Read measurements."}
                prompts.append(data)
                name = f"writer_{data['writer']}_g{data['generation']}"
                return {"name": name, "task_id": data["task"]["id"], "source": "sako_run(); // " + name,
                        "strategy": "One experiment.", "uses_cutlass": False, "references": []}
        class Worker:
            def put(worker, *args):
                pass
            def run(worker, action, *args, **kwargs):
                if action == "evaluate":
                    candidate = read_json(ROOT / kwargs["candidate"])
                    slot = int(candidate["name"].split("_")[1])
                    return {"decision": {"accepted": True, "speedup": 1.3 if slot % 2 == 0 else 1.2}}
                return {"decision": {"accepted": False, "speedup": .9}, "cases": [{"correct": True}]}
        from sako.search import make_manifest
        loop = SearchLoop(self.config, self.directory, self.directory / "profile", lambda *a: Client(),
                          lambda *a: Worker(), lambda *a: [], make_manifest)
        result = loop.run()
        third = [p for p in prompts if p["generation"] == 2]
        migrated = [p for p in third if p["parent_selection"]["reason"] == "migration"]
        self.assertEqual(len(prompts), 24)
        self.assertEqual(len(migrated), 4)
        self.assertTrue(all(p["writer"] % 2 == 1 for p in migrated))
        self.assertTrue(all(p["parent"]["task_id"] == p["task"]["id"] for p in migrated))
        self.assertTrue(all(p["parent_selection"]["migration_event"] == "after_round_2" for p in migrated))
        self.assertFalse(any(p["parent_selection"]["reason"] == "migration" for p in prompts if p["generation"] < 2))
        self.assertEqual(len(self.store.emissions()), 24)
        self.assertEqual(result["duplicate_proposals"], 0)
        self.assertTrue((self.directory / "selections/round-0003.json").exists())

    def test_report_includes_selections_even_when_no_proposal_returned(self):
        self.add("migrant")
        self.policy.plan_migration(2, {})
        a = self.policy.select(1, 2, "linear_0", None)
        self.policy.save([a], 2)
        from sako.cli import main
        output = io.StringIO()
        with patch("sys.argv", ["sako", "report", "--run", str(self.directory),
                               "--config", str(ROOT / "configs/project.json")]), contextlib.redirect_stdout(output):
            main()
        result = json.loads(output.getvalue())
        self.assertEqual(result["parent_selections"], [a["selection"]])
        self.assertEqual(result["candidate_emissions"], [])
        self.assertEqual(result["migration_events"][0]["id"], "after_round_2")
        report = (self.directory / "report.md").read_text()
        self.assertIn("migration", report)
        self.assertIn("after_round_2", report)


if __name__ == "__main__":
    unittest.main()
