"""Test measured lessons and diverse, shared candidate parents."""
from triton_fixtures import RMS_SOURCE

import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import unittest

from sako.archive import CandidateArchive
from sako.core import digest
from sako.lessons import LessonStore
from sako.store import Store


class SwarmMemoryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name)
        self.store = Store(self.path / "state.sqlite")
        self.lessons = LessonStore(self.store, self.path)
        self.hardware = {"fingerprint": "hardware-and-software-a"}
        self.task = {"id": "rmsnorm_0", "kind": "rmsnorm", "scalars": [1e-6],
                     "cases": [{"shapes": [[2, 1024], [1024]], "dtypes": ["float16", "float16"],
                                "strides": [[1024, 1], [1]]}],
                     "contract": {"numerics": "FP32 reduction and FP16 rounding", "atol": .03, "rtol": .02}}

    def candidate(self, number, score=1.2, *, specialty="memory", change="vector loads", correct=True,
                  accepted=True, registers=32, measured=True):
        candidate = {"name": "candidate_" + str(number), "task_id": self.task["id"],
            "source": RMS_SOURCE + "# " + str(number) + " */ }",
            "strategy": "Agent explanation, not a measured fact.", "references": [], "backend": "triton"}
        key, _ = self.store.add(candidate, specialty, number, None)
        result = {"cases": [{"correct": correct, "checks": [{"passed": correct, "max_abs_error": 0. if correct else .7}],
                    "baseline": {"samples_ms": [2.] * 15}, "candidate": {"samples_ms": [2. / score] * 15},
                    "build": {"resource_reports": [{"registers_per_thread": registers}]}}],
                  "decision": {"accepted": accepted, "speedup": score, "reasons": [] if accepted else ["Test failed."]}}
        if measured:
            self.store.record(key, result)
        hypothesis = {"id": "hypothesis-" + str(number), "specialty": specialty,
                      "claim": "This change may reduce execution time.", "change": change,
                      "expected_metric": "memory throughput"}
        self.store.record_emission(number, 0, key, specialty, False,
                                  {"hypothesis": hypothesis, "parent_ids": []})
        return key, result, hypothesis

    def test_lesson_uses_saved_result_and_does_not_accept_hypothesis_as_fact(self):
        key, result, hypothesis = self.candidate(1)
        raw = copy.deepcopy(result)
        lesson = self.lessons.record(self.task, self.hardware, key, hypothesis=hypothesis)
        self.assertEqual(lesson["status"], "supported")
        self.assertEqual(lesson["classification"], "operation_gate_passed")
        self.assertEqual(lesson["hypothesis"]["label"], "agent_hypothesis")
        self.assertEqual(lesson["scope"], "operation")
        self.assertNotIn("samples_ms", json.dumps(lesson))
        self.assertEqual(self.store.candidates()[0]["result"], raw)
        forged = {**result, "decision": {"accepted": True, "speedup": 99.}}
        with self.assertRaisesRegex(ValueError, "saved evaluator"):
            self.lessons.record(self.task, self.hardware, key, result=forged)

    def test_retrieval_requires_matching_hardware_shape_stride_and_numerics(self):
        key, _, hypothesis = self.candidate(1)
        self.lessons.record(self.task, self.hardware, key, hypothesis=hypothesis)
        self.assertEqual(len(self.lessons.retrieve(self.task, self.hardware)), 1)
        self.assertEqual(self.lessons.retrieve(self.task, {"fingerprint": "different-software"}), [])
        for field, value in (("shapes", [[3, 1024], [1024]]), ("strides", [[1, 3], [1]]),
                             ("dtypes", ["bfloat16", "bfloat16"])):
            changed = copy.deepcopy(self.task)
            changed["cases"][0][field] = value
            self.assertEqual(self.lessons.retrieve(changed, self.hardware), [])
        changed = copy.deepcopy(self.task)
        changed["contract"]["atol"] = .3
        self.assertEqual(self.lessons.retrieve(changed, self.hardware), [])

    def test_model_lesson_requires_saved_manifest_and_preserves_combination_scope(self):
        key, result, hypothesis = self.candidate(1)
        with self.assertRaisesRegex(ValueError, "matching saved"):
            self.lessons.record(self.task, self.hardware, key, scope="model", result=result)
        bad = copy.deepcopy(result)
        bad["cases"][0]["correct"] = False
        bad["decision"] = {"accepted": False, "speedup": None, "reasons": ["Correctness failed."]}
        measurement = {"generation": 1, "manifest": {"entries": [
            {"task_id": self.task["id"], "candidate_id": key},
            {"task_id": "linear_0", "candidate_id": "b" * 24}]}, "result": bad}
        self.store.set("model_measurements", {"model-1": measurement})
        lesson = self.lessons.record(self.task, self.hardware, key, scope="model", result=bad,
                                     hypothesis=hypothesis)
        self.assertEqual(lesson["status"], "failed")
        self.assertEqual(lesson["observations"]["attribution"], "combination_only")
        forged = copy.deepcopy(bad)
        forged["decision"]["accepted"] = True
        with self.assertRaises(ValueError):
            self.lessons.record(self.task, self.hardware, key, scope="model", result=forged)

    def test_layer_accuracy_and_full_model_speed_remain_separate_in_lessons(self):
        key, result, hypothesis = self.candidate(1)
        raw = copy.deepcopy(result)
        raw.update(correctness_scope="replaced_layers", correctness_reference="eager",
                   model_reference="torch_compile", output_equivalence="not_tested",
                   acceptance_scope="Eager replacement-layer accuracy and complete compiled model speed.")
        raw["cases"][0].update(reference="eager_replaced_layers", timing_reference="torch_compile")
        measured = {"generation": 1, "manifest": {"entries": [{"task_id": self.task["id"], "candidate_id": key}]},
                    "result": raw}
        self.store.set("model_measurements", {"layer-model": measured})
        before = copy.deepcopy(measured)
        lesson = self.lessons.record(self.task, self.hardware, key, scope="model", result=raw, hypothesis=hypothesis)
        observations = lesson["observations"]
        self.assertEqual(lesson["status"], "supported")
        self.assertEqual(lesson["classification"], "layer_accuracy_and_complete_model_speed_passed")
        self.assertEqual(observations["correctness_scope"], "replaced_layers")
        self.assertEqual(observations["correctness_reference"], "eager")
        self.assertEqual(observations["speed_scope"], "complete_model")
        self.assertEqual(observations["speed_reference"], "torch_compile")
        self.assertEqual(observations["output_equivalence"], "not_tested")
        self.assertEqual(observations["cases"][0]["reference"], "eager_replaced_layers")
        self.assertEqual(observations["cases"][0]["timing_reference"], "torch_compile")
        self.assertIn("equivalence was not tested", lesson["status_note"])
        self.assertEqual(self.store.get("model_measurements")["layer-model"], before)
        recalled = self.lessons.retrieve(self.task, self.hardware)
        self.assertEqual(recalled[0]["observations"]["output_equivalence"], "not_tested")

    def test_accuracy_contract_change_cannot_supersede_a_full_model_failure(self):
        key, result, hypothesis = self.candidate(1)
        bad = copy.deepcopy(result)
        bad["cases"][0]["correct"] = False
        bad["decision"]["accepted"] = False
        manifest = {"entries": [{"task_id": self.task["id"], "candidate_id": key}]}
        self.store.set("model_measurements", {"legacy": {"generation": 1, "manifest": manifest, "result": bad}})
        prior = self.lessons.record(self.task, self.hardware, key, scope="model", result=bad, hypothesis=hypothesis)
        self.assertEqual(prior["observations"]["correctness_scope"], "full_model")
        self.assertEqual(prior["observations"]["correctness_reference"], "torch_compile")
        self.assertEqual(prior["observations"]["output_equivalence"], "checked")
        local = copy.deepcopy(result)
        local.update(correctness_scope="replaced_layers", correctness_reference="eager",
                     model_reference="torch_compile", output_equivalence="not_tested")
        self.store.set("model_measurements", {"new": {"generation": 2, "manifest": manifest, "result": local}})
        with self.assertRaisesRegex(ValueError, "different accuracy or speed"):
            self.lessons.record(self.task, self.hardware, key, scope="model", result=local,
                                hypothesis=hypothesis, supersedes=[prior["id"]])
        self.assertEqual(len(self.lessons.all()), 1)
        self.assertEqual(self.lessons.all()[0]["superseded_by"], [])

    def test_operation_lesson_does_not_claim_final_model_equivalence(self):
        key, _, _ = self.candidate(1)
        lesson = self.lessons.record(self.task, self.hardware, key)
        observations = lesson["observations"]
        self.assertEqual(observations["correctness_scope"], "operation")
        self.assertEqual(observations["correctness_reference"], "eager")
        self.assertEqual(observations["speed_scope"], "operation")
        self.assertEqual(observations["speed_reference"], "torch_compile")
        self.assertEqual(observations["output_equivalence"], "not_tested")

    def test_stable_ids_concurrent_writes_and_raw_evidence_survive_correction(self):
        key, result, hypothesis = self.candidate(1, accepted=False)
        with ThreadPoolExecutor(max_workers=5) as pool:
            records = list(pool.map(lambda _: self.lessons.record(self.task, self.hardware, key,
                                                                 hypothesis=hypothesis), range(5)))
        first = records[0]
        self.assertEqual({row["id"] for row in records}, {first["id"]})
        self.assertEqual(len(self.lessons.all()), 1)
        second_key, _, second_hypothesis = self.candidate(2)
        second = self.lessons.record(self.task, self.hardware, second_key, hypothesis=second_hypothesis,
                                     supersedes=[first["id"]])
        self.assertEqual([row["id"] for row in self.lessons.retrieve(self.task, self.hardware)], [second["id"]])
        self.assertEqual(len(self.lessons.all()), 2)
        with self.store.connect() as db:
            raw = json.loads(db.execute("SELECT payload FROM swarm_lesson_evidence WHERE sha256=?",
                                        (digest(result),)).fetchone()[0])
        self.assertEqual(raw, result)
        self.assertTrue((self.path / "lessons" / (first["id"] + ".json")).is_file())
        with self.assertRaises(ValueError):
            self.lessons.record(self.task, self.hardware, key, hypothesis=hypothesis, supersedes=[second["id"]])

    def test_unmeasured_or_cross_scope_record_cannot_supersede_measured_record(self):
        key, _, hypothesis = self.candidate(1)
        first = self.lessons.record(self.task, self.hardware, key, hypothesis=hypothesis)
        unknown, _, _ = self.candidate(2, measured=False)
        with self.assertRaises(ValueError):
            self.lessons.record(self.task, self.hardware, unknown, supersedes=[first["id"]])
        with self.assertRaises(ValueError):
            self.lessons.record(self.task, {"fingerprint": "different-gpu"}, key, supersedes=[first["id"]])


    def test_recall_bounds_and_infrastructure_failure_is_not_a_cuda_trap(self):
        for number in range(8):
            key, _, hypothesis = self.candidate(number, accepted=number % 2 == 0)
            self.lessons.record(self.task, self.hardware, key, hypothesis=hypothesis)
        records = self.lessons.retrieve(self.task, self.hardware, limit=3, max_chars=5000)
        self.assertLessEqual(len(records), 3)
        self.assertLessEqual(len(json.dumps(records, ensure_ascii=False)), 5000)
        self.assertEqual({row["status"] for row in records}, {"failed", "supported"})
        key, _, _ = self.candidate(9)
        self.store.record(key, {"error": "SSH reset", "failure_kind": "infrastructure"})
        lesson = self.lessons.record(self.task, self.hardware, key)
        self.assertEqual(lesson["status"], "untested")
        self.assertIsNone(lesson["observations"]["correct"])

    def test_archive_keeps_best_and_slower_distinct_designs(self):
        best, _, _ = self.candidate(1, 2.)
        same, _, _ = self.candidate(2, 1.99)
        compute, _, _ = self.candidate(3, 1.8, specialty="compute", change="warp reduction")
        slow, _, _ = self.candidate(4, .95, specialty="scheduling", change="fewer threads", accepted=False)
        archive = CandidateArchive(self.store, limit=3)
        selected = archive.refresh(self.task, self.hardware)
        self.assertEqual(selected[0]["id"], best)
        self.assertEqual({row["id"] for row in selected}, {best, compute, slow})
        self.assertNotIn(same, {row["id"] for row in selected})
        self.assertEqual(archive.parent(slow, self.task, self.hardware)["island"], "scheduling")
        self.assertTrue(all(row["model_status"] == "untested" for row in selected))

    def test_archive_blocks_model_failures_except_explicit_repair(self):
        key, _, _ = self.candidate(1, 2.)
        other, _, _ = self.candidate(2, 1.5, specialty="compute")
        self.store.set("blocked_model_candidates", {key: {"result": "failed"}})
        self.store.set("best_model_measurement", {"manifest": {"entries": [{"candidate_id": other}]},
                                                  "result": {"decision": {"accepted": True}}})
        before = self.store.get("best_model_measurement")
        archive = CandidateArchive(self.store)
        self.assertEqual([row["id"] for row in archive.refresh(self.task, self.hardware)], [other])
        with self.assertRaises(ValueError):
            archive.parent(key, self.task, self.hardware)
        self.assertEqual(archive.parent(key, self.task, self.hardware, repair=True)["id"], key)
        self.assertEqual(self.store.get("best_model_measurement"), before)

    def test_archive_context_changes_and_invalid_scores_are_rejected(self):
        good, _, _ = self.candidate(1)
        wrong, _, _ = self.candidate(2, correct=False, accepted=False)
        nonfinite, result, _ = self.candidate(3)
        result["decision"]["speedup"] = float("inf")
        self.store.record(nonfinite, result)
        archive = CandidateArchive(self.store)
        self.assertEqual([row["id"] for row in archive.refresh(self.task, self.hardware)], [good])
        with self.assertRaises(ValueError):
            archive.parent(wrong, self.task, self.hardware)
        with self.assertRaises(ValueError):
            archive.refresh(self.task, {"fingerprint": "other-device"})


if __name__ == "__main__":
    unittest.main()
