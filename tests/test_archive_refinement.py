"""Use measured parents before spending all writer slots on fresh designs."""

import copy
import unittest

import test_swarm_loop as fixtures
from sako.archive import summarize_parent_use
from sako.core import digest
from sako.reporting import build_swarm_report
from sako.store import Store
from sako.swarm_policy import validate_plan


class ArchivePlanTests(unittest.TestCase):
    def hypothesis(self, parent="parent-a", change="Change the tile size.", **fields):
        return {"specialty": "scheduling", "parent_id": parent, "claim": "This tile may improve reuse.",
                "change": change, "expected_metric": "Lower median latency.",
                "evidence": [parent or "parent-a"], **fields}

    def plan(self, hypotheses, **fields):
        return {"action": "continue", "reason": "Compare measured designs.",
                "evidence": ["parent-a"], "hypotheses": hypotheses, **fields}

    def validate(self, plan, **fields):
        return validate_plan(plan, task_id="task", parent_ids=["parent-a", "parent-b"],
                             known_evidence={"parent-a", "parent-b", "contract:task"}, **fields)

    def fresh(self, **fields):
        return self.hypothesis(None, **{"change": "Use a new reduction design.",
            "fresh_start_reason": "The parent uses a serial reduction. Test a separate reduction design.", **fields})

    def test_all_fresh_plan_needs_a_measured_reason_to_skip_archive(self):
        with self.assertRaisesRegex(ValueError, "ordinary archive refinement"):
            self.validate(self.plan([self.fresh()]))
        skip = {"reason": "Both stored tile designs repeat the measured cache limit. Test a different reduction.",
                "evidence": ["parent-a", "parent-b"]}
        plan = self.validate(self.plan([self.fresh()], archive_skip=skip))
        self.assertEqual(plan["reason"], "Compare measured designs.")
        self.assertEqual(plan["archive_decision"]["skip"], skip)
        self.assertEqual(plan["archive_decision"]["planned_refinements"], 0)

    def test_fresh_start_must_explain_the_choice_and_consider_an_actual_parent(self):
        for fresh in (self.hypothesis(None), self.fresh(evidence=["contract:task"]),
                      self.fresh(fresh_start_reason="")):
            with self.subTest(fresh=fresh), self.assertRaises(ValueError):
                self.validate(self.plan([self.hypothesis(), fresh]))

    def test_refinement_is_dispatched_before_fresh_design_when_one_slot_fits(self):
        original = self.plan([self.fresh(), self.hypothesis()])
        saved = copy.deepcopy(original)
        plan = self.validate(original, max_writers=1)
        self.assertEqual(plan["hypotheses"][0]["parent_id"], "parent-a")
        self.assertIsNone(plan["hypotheses"][1]["parent_id"])
        self.assertEqual(plan["max_active_writers"], 1)
        self.assertEqual(plan["archive_decision"]["planned_refinements"], 1)
        self.assertEqual(original, saved)

    def test_repairs_do_not_satisfy_ordinary_refinement_requirement(self):
        with self.assertRaisesRegex(ValueError, "ordinary archive refinement"):
            self.validate(self.plan([self.hypothesis()]), repair_ids=["parent-a"])
        plan = self.validate(self.plan([self.hypothesis("parent-b"), self.hypothesis()]),
                             repair_ids=["parent-a"])
        self.assertEqual([h["parent_id"] for h in plan["hypotheses"]], ["parent-a", "parent-b"])
        self.assertEqual(plan["archive_decision"]["planned_refinements"], 1)

    def test_active_refinement_allows_fresh_work_without_filling_every_slot_with_reuse(self):
        plan = self.validate(self.plan([self.fresh()]), archive_refinement_inflight=True)
        self.assertFalse(plan["archive_decision"]["refinement_required"])
        self.assertEqual(plan["archive_decision"]["fresh_starts"], 1)
        self.assertIsNone(plan["archive_decision"]["skip"])

    def test_duplicate_refinement_cannot_hide_an_all_fresh_plan(self):
        first = self.hypothesis()
        seen = digest({"task": "task", "parent": "parent-a", "change": first["change"].casefold()})[:24]
        with self.assertRaisesRegex(ValueError, "ordinary archive refinement"):
            self.validate(self.plan([first, self.fresh()]), seen=[seen])

    def test_refinement_must_cite_its_own_measured_parent(self):
        with self.assertRaisesRegex(ValueError, "selected parent"):
            self.validate(self.plan([self.hypothesis(evidence=["parent-b"])]))

    def test_move_on_must_explain_why_available_parents_have_low_value(self):
        moving = self.plan([], action="move_on")
        with self.assertRaisesRegex(ValueError, "ordinary archive refinement"):
            self.validate(moving)
        with self.assertRaisesRegex(ValueError, "ordinary archive refinement"):
            self.validate(moving, archive_refinement_inflight=True)
        moving["archive_skip"] = {"reason": "The measured parent variants fail the full model speed gate.",
                                  "evidence": ["parent-a"]}
        self.assertEqual(self.validate(moving)["action"], "move_on")
        waiting = self.validate(self.plan([], action="wait"), allow_wait=True)
        self.assertEqual(waiting["action"], "wait")

    def test_skip_cannot_use_unknown_or_contract_only_evidence(self):
        for skip in ({"reason": "Low value.", "evidence": ["missing"]},
                     {"reason": "Low value.", "evidence": ["contract:task"]},
                     {"reason": "", "evidence": ["parent-a"]},
                     {"reason": "Low value.", "evidence": [{"forged": "parent-a"}]}):
            with self.subTest(skip=skip), self.assertRaises(ValueError):
                self.validate(self.plan([self.fresh()], archive_skip=skip))

    def test_no_ordinary_parent_does_not_force_false_refinement(self):
        plan = self.validate(self.plan([self.hypothesis(None)]), repair_ids=["parent-a", "parent-b"])
        self.assertFalse(plan["archive_decision"]["refinement_required"])
        self.assertEqual(plan["archive_decision"]["eligible_parent_ids"], [])

    def test_usage_counts_separate_repairs_and_corrections_without_source(self):
        work = [
            {"parent_id": None, "selection": {"reason": "fresh_start"}},
            {"parent_id": "a", "status": "completed", "selection": {"reason": "shared_archive"}},
            {"parent_id": "a", "status": "running", "selection": {"reason": "shared_archive"}},
            {"parent_id": "a", "status": "running", "selection": {"reason": "repair"}},
            {"parent_id": None, "repair": {"stage": "source"}, "selection": {"reason": "repair"}},
        ]
        use = summarize_parent_use(work)
        self.assertEqual([use[k] for k in ("fresh_starts", "archive_refinements", "repairs", "corrections_without_parent")],
                         [1, 2, 1, 1])
        self.assertEqual(use["by_parent"]["a"]["active_refinements"], 1)
        self.assertEqual(use["by_parent"]["a"]["completed_refinements"], 1)


class ArchiveLoopTests(unittest.TestCase):
    loop = fixtures.SwarmLoopTests.loop
    writers = fixtures.SwarmLoopTests.writers

    def setUp(self):
        fixtures.SwarmLoopTests.setUp(self)
        self.config["agent"].update(writers=1, max_task_experiments=3)
        self.worker.operation_accepted = self.worker.model_correct = True

    def state(self):
        return Store(self.directory / "run/state.sqlite")

    def test_successful_parent_receives_ordinary_refinements_and_full_source(self):
        result = self.loop().run()
        self.assertEqual(result["status"], "accepted")
        writers = self.writers()
        self.assertIsNone(writers[0]["parent"])
        for prompt in writers[1:]:
            self.assertEqual(prompt["parent_selection"]["reason"], "shared_archive")
            parent = next(row for row in self.state().candidates() if row["id"] == prompt["parent_selection"]["candidate_id"])
            self.assertEqual(prompt["parent"], parent["source"])
            self.assertIn("Refine the supplied archive parent", prompt["writer_mode"])
        report = build_swarm_report(self.directory / "run", self.config)
        self.assertEqual(report["parent_use"]["fresh_starts"], 1)
        self.assertEqual(report["parent_use"]["archive_refinements"], 2)
        self.assertEqual(report["parent_use"]["repairs"], 0)
        plans = [data for role, data in self.client.prompts if role == "coordinator"]
        self.assertTrue(plans[1]["archive_policy"]["refinement_required"])
        self.assertIsNotNone(plans[1]["example"]["hypotheses"][0]["parent_id"])
        self.assertTrue(plans[1]["parents"][0]["operation_measurements"])
        self.assertTrue(plans[1]["parents"][0]["model_measurements"])
        self.assertEqual(plans[2]["parent_selection_counts"]["archive_refinements"], 1)

    def test_invalid_all_fresh_plan_is_corrected_before_the_next_writer_runs(self):
        self.config["agent"]["max_task_experiments"] = 2

        def coordinator(data, number):
            parents = data["archive_policy"]["eligible_parent_ids"]
            evidence = parents[:1] or ["contract:" + data["task"]["id"]]
            hypothesis = {"specialty": "memory", "parent_id": parents[0] if parents and number > 1 else None,
                "claim": "This memory method may reduce latency.", "change": "Try method " + str(number),
                "expected_metric": "Latency.", "evidence": evidence,
                "fresh_start_reason": "Test a separate layout after comparing the stored parent."}
            return {"action": "continue", "reason": "Test the next measured method.",
                    "evidence": evidence, "hypotheses": [hypothesis]}

        self.client.plan_callback = coordinator
        self.assertEqual(self.loop().run()["status"], "accepted")
        self.assertEqual(self.client.plan_number, 3)
        self.assertEqual(len(self.writers()), 2)
        self.assertEqual(self.writers()[1]["parent_selection"]["reason"], "shared_archive")
        rejections = [event for event in self.state().get("swarm_events") if event["kind"] == "coordinator_plan_rejected"]
        self.assertEqual(len(rejections), 1)
        self.assertIn("ordinary archive refinement", rejections[0]["reason"])

    def test_explicit_archive_skip_is_kept_in_the_plan_and_report(self):
        self.config["agent"]["max_task_experiments"] = 2

        def coordinator(data, number):
            parents = data["archive_policy"]["eligible_parent_ids"]
            evidence = parents[:1] or ["contract:" + data["task"]["id"]]
            hypothesis = {"specialty": "memory", "parent_id": None, "claim": "A separate layout may help.",
                "change": "Try layout " + str(number), "expected_metric": "Latency.", "evidence": evidence,
                "fresh_start_reason": "The parent uses the original layout. Compare a separate layout design."}
            plan = {"action": "continue", "reason": "Compare a distinct design.",
                    "evidence": evidence, "hypotheses": [hypothesis]}
            if parents:
                plan["archive_skip"] = {"reason": "Use the recorded parent timings as the control for a separate layout.",
                                        "evidence": [parents[0]]}
            return plan

        self.client.plan_callback = coordinator
        self.assertEqual(self.loop().run()["status"], "accepted")
        report = build_swarm_report(self.directory / "run", self.config)
        self.assertEqual(report["parent_use"]["fresh_starts"], 2)
        self.assertEqual(report["parent_use"]["archive_refinements"], 0)
        skip = report["plans"][1]["archive_decision"]["skip"]
        self.assertTrue(skip["reason"])
        self.assertEqual(skip["evidence"], [report["candidates"][0]["id"]])

    def test_one_remaining_experiment_uses_archive_even_when_plan_lists_fresh_first(self):
        self.config["agent"]["max_task_experiments"] = 2

        def coordinator(data, number):
            parent_ids = data["archive_policy"]["eligible_parent_ids"]
            evidence = parent_ids[:1] or ["contract:" + data["task"]["id"]]
            base = {"specialty": "memory", "parent_id": None, "claim": "Test a distinct memory method.",
                    "change": f"Method {number}", "expected_metric": "Latency.", "evidence": evidence}
            hypotheses = [base]
            if number:
                hypotheses = [{**base, "fresh_start_reason": "Test a separate layout after comparing the stored parent."},
                              {**base, "parent_id": parent_ids[0], "change": "Improve the archived layout."}]
            return {"action": "continue", "reason": "Compare designs using the current measurements.",
                    "evidence": evidence, "hypotheses": hypotheses}

        self.client.plan_callback = coordinator
        result = self.loop().run()
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(len(self.writers()), 2)
        self.assertEqual(self.writers()[1]["parent_selection"]["reason"], "shared_archive")

    def test_generated_archive_refinement_resumes_without_another_writer_call(self):
        self.config["agent"]["max_task_experiments"] = 2
        original = self.worker.run
        calls = 0

        def run(action, run_dir, **fields):
            nonlocal calls
            result = original(action, run_dir, **fields)
            if action == "model":
                calls += 1
                if calls == 1:
                    self.worker.fail_upload = True
            return result

        self.worker.run = run
        first = self.loop().run()
        self.assertEqual(first["status"], "infrastructure_failure")
        prompt = copy.deepcopy(self.writers()[1])
        second = self.loop().run()
        self.assertEqual(second["status"], "accepted")
        self.assertEqual(len(self.writers()), 2)
        self.assertEqual(self.writers()[1], prompt)
        self.assertEqual(sum(action == "evaluate" for action, _ in self.worker.calls), 2)
        self.assertEqual(build_swarm_report(self.directory / "run", self.config)["parent_use"]["archive_refinements"], 1)

    def test_reports_do_not_count_a_queued_assignment_as_a_started_writer(self):
        loop = self.loop()
        loop.store.set("swarm_assignments", {
            "0": {"id": "started", "parent_id": "parent", "selection": {"reason": "shared_archive"}},
            "1": {"id": "queued", "parent_id": "parent", "selection": {"reason": "shared_archive"}},
        })
        loop.store.set("swarm_events", [{"kind": "writer_started", "assignment_id": "started"}])
        report = build_swarm_report(self.directory / "run", self.config)
        self.assertEqual(report["parent_use"]["archive_refinements"], 1)
        self.assertEqual(report["parent_use"]["all_assignments"]["archive_refinements"], 2)


if __name__ == "__main__":
    unittest.main()
