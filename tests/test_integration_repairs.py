"""Keep numerical integration repairs exact, finite, and separate per source."""
from triton_fixtures import RMS_SOURCE

import copy
import unittest
from unittest.mock import patch

from sako.core import candidate_id, digest, read_json, write_json
from sako.hierarchy import select_hierarchy, validate_queue
from sako.hypotheses import HypothesisBook, measured_gpu_outcome
from sako.integration import integrate
from sako.integration_repairs import numerical_failures, record_failure
from sako.kernel_selection import keep_kernel, kernel_manifest
from sako.store import Store
from test_kernel_first import PairedWorker
import test_swarm_loop as fixtures


def failing_model(path="block.mlp", seed=1729):
    detail = {"passed": False, "max_abs_error": 4., "mismatched_elements": 1,
        "atol": .03, "rtol": .02, "examples": [{"flat_index": 1729,
        "actual": .7783203125, "expected": .8251953125, "allowed_abs_error": .046503905206918716}]}
    layer = {"path": path, "passed": False, "inputs_unchanged": True, "same_input": True,
             "check": detail, "input_shapes": [[1, 128, 3584]]}
    region = {"seed": seed, "scope": "replaced_layers", "reference": "eager", "passed": False,
        "expected_paths": [path], "observed_paths": [path], "missing_paths": [],
        "failed_paths": [path], "layers": [layer], "max_abs_error": 4., "reference_state_intact": True}
    outer = {"seed": seed, "scope": "replaced_layers", "reference": "eager", "passed": False,
        "expected_paths": [path + ".gate"], "observed_paths": [path + ".gate"], "missing_paths": [],
        "failed_paths": [], "layers": [{"path": path + ".gate", "passed": True, "same_input": True,
            "inputs_unchanged": True, "check": {"passed": True, "max_abs_error": .015625}}],
        "region_checks": region, "reference_state_intact": True}
    return {"scope": "model", "correctness_scope": "replaced_layers", "correctness_reference": "eager",
        "reference_state_intact": True, "cases": [{"workload": {"batch_size": 1, "sequence_length": 128},
        "correct": False, "checks": [outer]}], "decision": {"accepted": False, "speedup": None,
        "reasons": ["A containing MLP check failed. Timing was skipped."]}}


class IntegrationRepairTests(unittest.TestCase):
    setUp = fixtures.SwarmLoopTests.setUp
    loop = fixtures.SwarmLoopTests.loop

    def prepare(self, extra_sibling=False, include_parent=False):
        self.config["agent"].update(target_selection="bottleneck_tree", max_task_experiments=None,
                                    performance_refinement=True, search_parent_regions=include_parent)
        self.config["kernel_search"] = {"enabled": True, "min_speedup": 1.01}
        self.worker = PairedWorker()
        tasks = [{"id": "mlp", "kind": "mlp", "paths": ["block.mlp"], "priority_ms": 10.,
                  "cases": [{}], "contract": {"validated": True}},
                 {"id": "gate", "kind": "linear", "paths": ["block.mlp.gate"], "priority_ms": 8.,
                  "cases": [{}], "contract": {"validated": True}},
                 {"id": "head", "kind": "linear", "paths": ["head"], "priority_ms": 2.,
                  "cases": [{}], "contract": {"validated": True}}]
        if extra_sibling:
            tasks.append({"id": "down", "kind": "linear", "paths": ["block.mlp.down"], "priority_ms": 4.,
                          "cases": [{}], "contract": {"validated": True}})
        selected = select_hierarchy(tasks, include_parents=include_parent)
        catalog = {"config": self.config, "tasks": tasks, "hardware": {"fingerprint": "test-gpu"},
                   "selected_task_ids": selected["selected_task_ids"], "hierarchy": selected["hierarchy"]}
        write_json(self.directory / "profile/tasks.json", catalog)
        loop = self.loop()
        loop.start_task()
        plan = loop.coordinator(1)
        assignment = loop.assignment(plan["hypotheses"][0], 0, plan)
        assignment["candidate"] = {"name": "initial", "task_id": "gate", "strategy": "Measured source.",
            "backend": "triton", "references": [], "source": RMS_SOURCE + '# candidate 0'}
        loop.process_candidate(assignment)
        manifest = kernel_manifest(loop.store, loop.catalog)
        self.assertEqual(len(manifest["entries"]), 1)
        return loop, assignment, manifest

    def test_nested_region_failure_is_a_measured_gpu_outcome(self):
        result = failing_model()
        self.assertTrue(measured_gpu_outcome(result))
        failures = numerical_failures(result)
        self.assertEqual(failures[0]["path"], "block.mlp")
        self.assertEqual(failures[0]["scope"], "containing_region")
        self.assertEqual(failures[0]["check"]["examples"][0]["actual"], .7783203125)
        result["cases"][0]["checks"][0]["region_checks"]["reference_state_intact"] = False
        self.assertFalse(measured_gpu_outcome(result))
        self.assertEqual(numerical_failures(result), [])

    def test_merged_region_failure_is_not_duplicated_or_mislabelled(self):
        result = failing_model()
        check = result["cases"][0]["checks"][0]
        check["layers"].extend({**layer, "scope": "containing_region"}
                               for layer in check["region_checks"]["layers"])
        failures = numerical_failures(result)
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["scope"], "containing_region")
        check["region_checks"]["reference_state_intact"] = False
        self.assertEqual(numerical_failures(result), [])

    def test_missing_boundary_coverage_cannot_queue_a_writer_repair(self):
        loop, assignment, manifest = self.prepare()
        result = failing_model()
        result["cases"][0]["checks"][0]["region_checks"]["missing_paths"] = ["block.mlp"]
        result["cases"][0]["checks"][0]["region_checks"]["observed_paths"] = []
        self.assertEqual(record_failure(loop.store, {"manifest": manifest, "result": result}), [])
        self.assertEqual(loop.hypotheses.get(assignment["hypothesis_id"])["status"], "succeeded")

    def test_followup_preserves_attempt_outcome_and_correction_count(self):
        loop, assignment, manifest = self.prepare()
        book = loop.hypotheses
        before = copy.deepcopy(book.attempt(assignment["id"]))
        result = failing_model()
        for repeat in range(2):
            record = book.integration_failure(assignment["candidate_id"], result, ["evidence.json"])
            self.assertEqual(record["status"], "correction_pending")
            self.assertEqual(record["attempt_count"], 1)
        after = book.attempt(assignment["id"])
        self.assertEqual(after["outcome_hash"], before["outcome_hash"])
        self.assertEqual(after["outcome"], before["outcome"])
        self.assertEqual(len(record["integration_failures"]), 1)
        for index in (1, 2):
            name = "correction-" + str(index)
            book.begin(record["id"], name, {"slot": 0})
            record = book.finish(name, stage="model", result=result, candidate_id="corrected-" + str(index))
        self.assertEqual(record["status"], "exhausted")
        self.assertEqual(record["corrections_remaining"], 0)
        with self.assertRaisesRegex(ValueError, "cannot reset"):
            book.begin(record["id"], "correction-3")

    def test_failure_retires_only_its_keep_and_preserves_full_evidence(self):
        loop, assignment, manifest = self.prepare()
        keeps = loop.store.get("kernel_baselines")
        keeps["head"] = {"candidate_id": "unrelated", "decision": {"accepted": True}}
        loop.store.set("kernel_baselines", keeps)
        measured = {"manifest": manifest, "result": failing_model()}
        ticket = record_failure(loop.store, measured)[0]
        self.assertEqual(ticket["status"], "correction_pending")
        self.assertEqual(loop.store.get("kernel_baselines"), {"head": keeps["head"]})
        self.assertEqual(ticket["retired_keeps"][0]["keep"], keeps["gate"])
        self.assertEqual(ticket["result"], measured["result"])
        self.assertFalse(keep_kernel(loop.store, loop.by_id["gate"], assignment["candidate_id"], self.config))
        again = record_failure(loop.store, measured)[0]
        self.assertEqual(again, ticket)
        self.assertEqual(loop.hypotheses.get(assignment["hypothesis_id"])["attempt_count"], 1)

    def test_region_failure_reopens_source_before_region_completion(self):
        loop, assignment, manifest = self.prepare()
        loop._measure_manifest = lambda *args: {"manifest": manifest, "result": failing_model()}
        loop.finish_task("Measured exploration ended.", "coordinator", [assignment["candidate_id"]])
        self.assertEqual(loop.active_task, "gate")
        self.assertEqual(loop.progress["completed"], [])
        self.assertNotIn("mlp", loop.store.get("hierarchy_completed_regions", {}))
        self.assertEqual(loop.hypotheses.get(assignment["hypothesis_id"])["status"], "correction_pending")
        loop.queue_corrections()
        queued = loop.plan_queue[0]
        repair = loop.assignment(queued["hypothesis"], 1, {"id": None})
        self.assertEqual(repair["hypothesis_id"], assignment["hypothesis_id"])
        self.assertEqual(repair["correction_attempt"], 1)
        self.assertEqual(repair["repair"]["stage"], "model")
        self.assertEqual(repair["repair"]["result"]["cases"][0]["checks"][0]["region_checks"]["failed_paths"], ["block.mlp"])
        validate_queue(loop.catalog, loop.progress)

    def test_numerical_repair_rechecks_model_and_keeps_its_two_attempt_limit(self):
        loop, assignment, manifest = self.prepare()
        def measure(m, *_):
            measured = {"manifest": m, "result": failing_model()}
            cache = loop.store.get("model_measurements", {})
            cache[digest(m)] = measured
            loop.store.set("model_measurements", cache)
            return measured
        loop._measure_manifest = measure
        loop.integration_repairs({"manifest": manifest, "result": failing_model()})
        for index in (1, 2):
            loop.queue_corrections()
            repair = loop.assignment(loop.plan_queue[0]["hypothesis"], 0, {"id": None})
            repair["candidate"] = {**assignment["candidate"], "source": RMS_SOURCE + '# candidate ' + str(index)}
            loop.process_candidate(repair)
            self.assertTrue(any(e["kind"] == "model_repair_checked" and e["assignment_id"] == repair["id"] for e in loop.events))
        record = loop.hypotheses.get(assignment["hypothesis_id"])
        self.assertEqual(record["status"], "exhausted")
        self.assertEqual(record["attempt_count"], 3)
        self.assertEqual(loop.store.get("kernel_baselines", {}), {})
        self.assertEqual(loop.pending, {})

    def test_child_repair_is_measured_when_a_parent_keep_and_choice_remain(self):
        self.config['model_trial_base'] = 'accepted'
        loop, assignment, manifest = self.prepare(include_parent=True)
        loop.integration_repairs({'manifest': manifest, 'result': failing_model()})
        parent = {**assignment['candidate'], 'name': 'kept_parent', 'task_id': 'mlp'}
        parent_id, _ = loop.store.add(parent, 'fusion', 99, None)
        loop.store.record(parent_id, {'correct': True, 'decision': {'accepted': True, 'speedup': 1.2}})
        keeps = loop.store.get('kernel_baselines', {})
        keeps['mlp'] = {'candidate_id': parent_id}
        loop.store.set_many({'kernel_baselines': keeps, 'kernel_region_choices': {'mlp': ['mlp']}})
        loop.queue_corrections()
        repair = loop.assignment(loop.plan_queue[0]['hypothesis'], 0, {'id': None})
        repair['candidate'] = {**assignment['candidate'], 'source': RMS_SOURCE + '# candidate 1'}
        tested = []
        def measure(local, *_):
            tested.append(copy.deepcopy(local))
            self.assertEqual({e['task_id'] for e in local['entries']}, {'gate'})
            self.assertEqual({e['candidate_id'] for e in local['entries']}, {repair['candidate_id']})
            measured = {'manifest': local, 'result': {'correct': True, 'reference_state_intact': True,
                'cases': [{'correct': True, 'checks': [{'seed': 1729, 'passed': True, 'max_abs_error': 0.}],
                    'baseline': {'samples_ms': [10.] * 15}, 'candidate': {'samples_ms': [11.] * 15}}],
                'decision': {'accepted': False, 'speedup': 10. / 11., 'reasons': ['The model speed gate failed.']}}}
            cache = loop.store.get('model_measurements', {})
            cache[digest(local)] = measured
            loop.store.set('model_measurements', cache)
            return measured
        loop._measure_manifest = measure
        loop.process_candidate(repair)
        self.assertEqual(len(tested), 1)
        self.assertEqual(loop.store.get('kernel_baselines')['mlp'], keeps['mlp'])
        self.assertEqual(loop.store.get('kernel_region_choices'), {'mlp': ['mlp']})
        self.assertIsNone(loop.store.get('best_model_measurement'))

    def test_manual_integration_writes_a_durable_repair_for_resume(self):
        loop, assignment, manifest = self.prepare()
        loop.store.set("status", "request_failure")
        write_json(self.directory / "run/result.json", {"status": "request_failure"})
        original = self.worker.run
        self.worker.run = lambda action, *args, **kw: failing_model() if action == "model" else original(action, *args, **kw)
        integrate(self.config, self.directory / "run", self.directory / "profile", self.worker)
        final = read_json(self.directory / "run/result.json")
        self.assertEqual(final["integration_status"], "repair_pending")
        self.assertEqual(final["status"], "integration_repair_pending")
        self.assertEqual(final["manifest"], {"entries": []})
        resumed = self.loop()
        resumed.queue_corrections()
        self.assertEqual(resumed.plan_queue[0]["hypothesis"]["id"], assignment["hypothesis_id"])
        self.assertEqual(resumed.pending[assignment["candidate_id"]]["stage"], "model")
        descriptor = resumed.pending[assignment["candidate_id"]]["evidence"][0]
        self.assertIn(descriptor, resumed.store.get("evidence_artifacts"))
        self.assertTrue((self.directory / "run" / descriptor["path"]).is_file())

    def test_complete_run_repairs_region_failure_then_continues_other_tasks(self):
        # Use prepare only for the catalog. The run below starts from a clean
        # store so its initial writer, transition, correction, and next task all
        # execute through the real scheduler.
        self.prepare()
        self.directory = self.directory / "fresh"
        self.directory.mkdir()
        old_catalog = read_json(self.directory.parent / "profile/tasks.json")
        write_json(self.directory / "profile/tasks.json", old_catalog)
        # This case tests the first pass, including its model repair.
        self.config["agent"].update(max_task_experiments=3, require_archive_trial=False, revisit_with_budget=False)
        self.client = fixtures.SwarmClient()
        model_calls = []
        original = self.worker.run
        def run(action, *args, **kwargs):
            if action == "model":
                model_calls.append(kwargs)
                if len(model_calls) == 1:
                    return failing_model()
                self.worker.model_correct = True
            if action == "reprofile":
                return {"task_priorities": {}, "model_state": "accepted_replacements"}
            return original(action, *args, **kwargs)
        self.worker.run = run
        def plan(data, number):
            parents = data["archive_policy"]["eligible_parent_ids"]
            if parents:
                return {"action": "move_on", "reason": "Proceed to the measured region check.",
                    "evidence": [parents[0]], "hypotheses": [], "archive_skip": {
                        "reason": "Check composition before further work.", "evidence": [parents[0]]}}
            return {"action": "continue", "reason": "Measure the assigned operation.",
                "evidence": ["contract:" + data["task"]["id"]], "hypotheses": [{"specialty": "memory",
                    "parent_id": None, "claim": "Improve memory access.", "change": "Use coalesced loads.",
                    "expected_metric": "Lower latency.", "evidence": ["contract:" + data["task"]["id"]]}]}
        self.client.plan_callback = plan
        result = self.loop().run()
        self.assertEqual(result["status"], "completed_kernel_search")
        self.assertEqual(result["integration_status"], "accepted")
        prompts = [data for role, data in self.client.prompts if role.startswith("writer_")]
        self.assertEqual([p["task"]["id"] for p in prompts], ["gate", "gate", "head"])
        self.assertIsNone(prompts[0]["repair"])
        repair = prompts[1]["repair"]
        self.assertEqual(repair["stage"], "model")
        self.assertEqual(repair["result"]["failed_layers"][0]["path"], "block.mlp")
        self.assertEqual(repair["result"]["failed_layers"][0]["examples"][0]["actual"], .7783203125)
        state = Store(self.directory / "run/state.sqlite")
        self.assertTrue(all(t["status"] == "corrected" for t in state.get("integration_repair_tickets").values()))
        self.assertEqual(set(state.get("hierarchy_completed_regions")), {"mlp", "head"})

    def test_completed_queue_reopens_pending_task_and_skips_saved_siblings(self):
        loop, assignment, manifest = self.prepare()
        loop.progress.update(index=2, completed=[{"task_id": "gate", "rounds": 1, "reason": "Old gate decision."},
                                                {"task_id": "head", "rounds": 2, "reason": "Old head decision."}])
        loop.store.set("task_progress", loop.progress)
        record_failure(loop.store, {"manifest": manifest, "result": failing_model()})
        loop.resume_integration_repairs()
        validate_queue(loop.catalog, loop.progress)
        self.assertEqual(loop.active_task, "gate")
        self.assertEqual(loop.store.get("integration_preserved_completions")["head"]["rounds"], 2)
        # A later return to the already completed head uses its old decision.
        loop.progress.update(index=1, completed=[{"task_id": "gate", "reason": "Corrected.", "rounds": 1}])
        loop.restore_integration_siblings()
        self.assertIsNone(loop.active_task)
        validate_queue(loop.catalog, loop.progress)
        self.assertEqual(loop.progress["completed"][-1]["reason"], "Old head decision.")

    def test_task_experiment_cap_defers_repair_without_an_extra_gate_writer(self):
        loop, assignment, manifest = self.prepare()
        self.config["agent"]["max_task_experiments"] = 1
        loop._measure_manifest = lambda m, *_: {"manifest": m, "result": failing_model()}
        loop.finish_task("Check integration.", "coordinator", [assignment["candidate_id"]])
        self.assertEqual(loop.active_task, "gate")
        result = loop.run()
        self.assertEqual(result["status"], "completed_kernel_search")
        self.assertEqual(loop.hypotheses.get(assignment["hypothesis_id"])["attempt_count"], 1)
        self.assertEqual(loop.store.get("integration_repair_tickets")[assignment["candidate_id"]]["status"], "deferred")
        self.assertFalse(any(role.startswith("writer_") and prompt["task"]["id"] == "gate"
                             for role, prompt in self.client.prompts))

    def test_accepted_repair_is_retained_if_search_stops_before_region_completion(self):
        loop, assignment, manifest = self.prepare()
        loop.integration_repairs({"manifest": manifest, "result": failing_model()})
        loop.queue_corrections()
        repair = loop.assignment(loop.plan_queue[0]["hypothesis"], 0, {"id": None})
        repair["candidate"] = {**assignment["candidate"], "source": RMS_SOURCE + '# candidate 1'}
        self.worker.model_correct = True
        loop.process_candidate(repair)
        self.assertEqual(loop.store.get("best_model_measurement")["manifest"]["entries"][0]["candidate_id"], repair["candidate_id"])
        (loop.directory / "STOP").touch()
        result = loop.run()
        self.assertEqual(result["status"], "stopped")
        self.assertEqual(result["integration_status"], "accepted")
        self.assertEqual(result["manifest"]["entries"][0]["candidate_id"], repair["candidate_id"])

    def test_later_precomputed_alternative_cannot_restore_a_blocked_source(self):
        loop, assignment, manifest = self.prepare()
        calls = []
        def measure(m, *_):
            calls.append(m)
            return {"manifest": m, "result": failing_model() if len(calls) == 1 else {
                "decision": {"accepted": True, "speedup": 2.}, "correct": True}}
        loop._measure_manifest = measure
        second = copy.deepcopy(manifest)
        second["entries"][0]["priority"] = 99.
        with patch("sako.kernel_selection.region_manifests", return_value=[(["gate"], manifest), (["gate"], second)]):
            loop.finish_region(loop.catalog["hierarchy"]["regions"][0])
        self.assertEqual(len(calls), 1)
        self.assertIsNone(loop.store.get("best_model_measurement"))

    def test_restored_same_region_siblings_still_finish_the_reopened_region(self):
        loop, assignment, manifest = self.prepare(extra_sibling=True)
        loop.progress.update(index=3, completed=[{"task_id": key, "rounds": 1, "reason": "Saved decision."}
                                               for key in loop.progress["queue"]])
        loop.store.set("task_progress", loop.progress)
        record_failure(loop.store, {"manifest": manifest, "result": failing_model()})
        loop.resume_integration_repairs()
        loop.progress.update(index=1, completed=[{"task_id": "gate", "rounds": 2, "reason": "Corrected."}])
        visited = []
        original = loop.finish_region
        def finish(region):
            visited.append(region["id"])
            return original(region)
        loop.finish_region = finish
        loop.start_task()
        self.assertIsNone(loop.active_task)
        self.assertIn("mlp", visited)
        self.assertIn("mlp", loop.store.get("hierarchy_completed_regions"))
        validate_queue(loop.catalog, loop.progress)

    def test_a_later_reopened_sibling_cannot_remain_on_the_skip_list(self):
        loop, assignment, manifest = self.prepare(extra_sibling=True)
        loop.progress.update(index=3, completed=[{"task_id": key, "rounds": 1, "reason": "Saved decision."}
                                               for key in loop.progress["queue"]])
        loop.reopen_integration_tasks({"gate"})
        self.assertIn("down", loop.store.get("integration_preserved_completions"))
        loop.reopen_integration_tasks({"gate", "down"})
        self.assertNotIn("down", loop.store.get("integration_preserved_completions"))
        loop.progress.update(index=1, completed=[{"task_id": "gate", "rounds": 2, "reason": "Corrected."}])
        loop.restore_integration_siblings()
        self.assertEqual(loop.active_task, "down")


if __name__ == "__main__":
    unittest.main()
