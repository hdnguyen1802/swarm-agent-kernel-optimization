"""Bound corrections per logical hypothesis across every pipeline stage."""

from concurrent.futures import ThreadPoolExecutor
import copy
from pathlib import Path
import tempfile
import unittest

from sako.hypotheses import HypothesisBook, classify_fault, measured_gpu_outcome
from sako.store import BudgetExhausted, Store
from sako.transport import TransferError


def measured_result(*, accepted=False, correct=True):
    return {"scope": "operation", "cases": [{"correct": correct,
        "checks": [{"passed": correct, "max_abs_error": 0. if correct else .6}],
        **({"baseline": {"samples_ms": [2.] * 15}, "candidate": {"samples_ms": [1. if accepted else 2.] * 15}}
           if correct else {})}],
        "decision": {"accepted": accepted, "speedup": 2. if accepted else 1. if correct else None}}


def layer_result(*, passed=False, missing=False):
    local = {"path": "model.norm", "passed": passed, "inputs_unchanged": True, "same_input": True,
             "check": {"passed": passed, "max_abs_error": .001 if passed else .6, "atol": .03, "rtol": .02}}
    seed = {"seed": 1729, "scope": "replaced_layers", "reference": "eager", "passed": passed and not missing,
            "max_abs_error": local["check"]["max_abs_error"], "layers": [local],
            "expected_paths": ["model.norm"] + (["model.mlp"] if missing else []),
            "observed_paths": ["model.norm"], "missing_paths": ["model.mlp"] if missing else []}
    return {"correctness_scope": "replaced_layers", "correctness_reference": "eager",
            "cases": [{"correct": seed["passed"], "checks": [seed]}],
            "decision": {"accepted": False, "speedup": None}}


class HypothesisPolicyTests(unittest.TestCase):
    def test_only_an_unfunded_writer_can_close_without_a_gpu_trial(self):
        record = self.book.register("task", self.hypothesis)
        self.book.begin(record["id"], "first")
        self.book.pause("first", "SSH unavailable", stage="transfer")
        with self.assertRaises(ValueError):
            self.book.defer_for_task_budget("first", "target allocation")
        self.book.begin(record["id"], "first")
        self.book.pause("first", "request exceeds allocation", stage="budget")
        self.book.defer_for_task_budget("first", "target allocation")
        self.book.defer_for_task_budget("first", "target allocation")
        ended = self.book.close(record["id"], "task finished")
        self.assertEqual(ended["status"], "closed")
        self.assertEqual(ended["attempts"][0]["status"], "budget_deferred")
        self.assertFalse(ended["last_outcome"]["measured_gpu_outcome"])

    def test_accepted_timing_does_not_close_a_contradicted_change(self):
        record = self.book.register("task", {**self.hypothesis, "verification": {"fixture_index": 0}})
        self.book.begin(record["id"], "first")
        first = self.book.finish("first", candidate_id="candidate-a", result=measured_result(accepted=True),
                                 observations={"status": "contradicted"})
        self.assertEqual(first["status"], "correction_pending")
        self.assertTrue(first["last_outcome"]["accepted"])
        self.assertEqual(first["last_outcome"]["stagnation_action"], "unchanged")
        self.book.begin(record["id"], "second")
        second = self.book.finish("second", candidate_id="candidate-b", result=measured_result(accepted=True),
                                  observations={"status": "observed"})
        self.assertEqual(second["status"], "succeeded")
        self.assertEqual(second["implementation_status"], "observed")

    def test_unverified_writer_method_does_not_become_a_verified_coordinator_claim(self):
        record = self.register()
        self.book.begin(record["id"], "first")
        result = self.book.finish("first", candidate_id="candidate-a", result=measured_result(accepted=True),
                                  observations={"status": "unverified", "design_owner": "writer"})
        self.assertEqual(result["status"], "closed")
        self.assertEqual(result["close_reason"]["code"], "candidate_only")
        self.assertTrue(result["last_outcome"]["accepted"])
        self.assertEqual(result["hypothesis"], record["hypothesis"])

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.store = Store(Path(temporary.name) / "state.sqlite")
        self.book = HypothesisBook(self.store)
        self.hypothesis = {"id": "hypothesis-a", "parent_id": None, "specialty": "memory",
                           "claim": "Vector loads may reduce memory traffic.", "change": "Use vector loads.",
                           "expected_metric": "Memory transactions.", "evidence": ["contract:task-a"]}

    def register(self, identifier="hypothesis-a", task="task-a"):
        return self.book.register(task, {**self.hypothesis, "id": identifier})

    def test_one_initial_attempt_and_two_corrections_share_all_failure_stages(self):
        self.register()
        for index, stage in enumerate(("writer", "source", "compile")):
            attempt = self.book.begin("hypothesis-a", f"assignment-{index}", writer=index)
            self.assertEqual(attempt["attempt_index"], index)
            self.assertEqual(attempt["is_correction"], index > 0)
            record = self.book.finish(f"assignment-{index}", stage=stage, error=RuntimeError("The attempt failed."),
                                      evidence=[f"evidence/error-{index}.json"])
            self.assertEqual(record["attempt_count"], index + 1)
            self.assertEqual(record["corrections_used"], index)
            self.assertEqual(record["last_outcome"]["stagnation_action"], "unchanged")
            self.assertEqual(record["status"], "correction_pending" if index < 2 else "exhausted")
        self.assertEqual(record["corrections_remaining"], 0)
        self.assertEqual(record["close_reason"]["code"], "correction_limit")
        self.assertEqual(self.book.outstanding(), [])
        with self.assertRaisesRegex(ValueError, "cannot reset"):
            self.book.begin("hypothesis-a", "assignment-3", writer="different-specialist")

    def test_reassignment_and_resume_keep_identity_attempts_and_error_evidence(self):
        original = self.register()
        self.book.begin(original["id"], "assignment-0", writer="memory")
        failed = self.book.finish("assignment-0", stage="compile", error="CUDA build failed.",
                                  candidate_id="candidate-a", evidence=[{"path": "compile.log", "sha256": "a" * 64}])
        self.book = HypothesisBook(self.store)
        outstanding = self.book.outstanding("task-a")
        self.assertEqual(len(outstanding), 1)
        self.assertEqual(outstanding[0]["last_candidate_id"], "candidate-a")
        self.assertEqual(outstanding[0]["last_outcome"]["error"], "CUDA build failed.")
        self.assertEqual(outstanding[0]["last_outcome"]["evidence"], failed["last_outcome"]["evidence"])
        reassigned = self.book.begin(original["id"], "assignment-1", writer="registers")
        self.assertEqual(reassigned["hypothesis_id"], original["id"])
        self.assertEqual(reassigned["attempt_number"], 2)
        self.assertEqual(self.book.begin(original["id"], "assignment-1", writer="registers"), reassigned)
        self.assertEqual(self.book.get(original["id"])["attempt_count"], 2)

    def test_failure_does_not_advance_operation_or_modify_other_hypotheses(self):
        self.register()
        self.register("hypothesis-b")
        progress = {"index": 0, "round": 0, "stagnant": 0, "completed": []}
        self.store.set("task_progress", progress)
        self.book.begin("hypothesis-b", "assignment-b", writer=1)
        other = self.book.get("hypothesis-b")
        for index in range(3):
            self.book.begin("hypothesis-a", f"assignment-a-{index}", writer=0)
            self.book.finish(f"assignment-a-{index}", stage="api", error="Provider request failed.")
        self.assertEqual(self.store.get("task_progress"), progress)
        self.assertEqual(self.book.get("hypothesis-b"), other)
        self.assertEqual(self.book.get("hypothesis-a")["status"], "exhausted")

    def test_finished_outcome_is_idempotent_and_cannot_be_overwritten(self):
        self.register()
        self.book.begin("hypothesis-a", "assignment-0")
        result = measured_result(accepted=True)
        first = self.book.finish("assignment-0", stage="model", result=result, candidate_id="candidate-a", improved=True)
        self.assertEqual(first["status"], "succeeded")
        self.assertEqual(first["last_outcome"]["stagnation_action"], "reset")
        second = HypothesisBook(self.store).finish("assignment-0", stage="model", result=result,
                                                  candidate_id="candidate-a", improved=True)
        self.assertEqual(first, second)
        self.assertEqual(self.book.for_candidate("candidate-a")["id"], "hypothesis-a")
        self.assertEqual(self.book.attempt("assignment-0")["status"], "succeeded")
        self.assertIsNone(self.book.for_candidate("missing"))
        self.assertIsNone(self.book.attempt("missing"))
        with self.assertRaisesRegex(ValueError, "cannot change"):
            self.book.finish("assignment-0", stage="model", result=measured_result(correct=False))

    def test_pending_gpu_job_and_budget_pause_recover_same_attempt(self):
        self.register()
        initial = self.book.begin("hypothesis-a", "assignment-0", writer=0)
        for error in (TransferError("Recover job abc."), BudgetExhausted("No API reservation fits.")):
            paused = self.book.finish("assignment-0", stage="operation", error=error, evidence=["job:abc"])
            self.assertEqual(paused["status"], "paused")
            self.assertEqual(paused["attempt_count"], 1)
            self.assertEqual(self.book.outstanding(), [])
            with self.assertRaisesRegex(ValueError, "Recover"):
                self.book.begin("hypothesis-a", "new-assignment")
            resumed = self.book.begin("hypothesis-a", "assignment-0", writer=0)
            self.assertEqual(resumed["attempt_index"], initial["attempt_index"])
        finished = self.book.finish("assignment-0", result=measured_result(correct=False), candidate_id="candidate-a")
        self.assertEqual(finished["status"], "correction_pending")
        self.assertEqual(finished["corrections_used"], 0)
        self.assertEqual(len(finished["attempts"][0]["interruptions"]), 2)

    def test_atomic_begin_prevents_two_simultaneous_corrections(self):
        self.register()
        self.book.begin("hypothesis-a", "assignment-0")
        self.book.finish("assignment-0", stage="source", error="Invalid JSON.")

        def begin(number):
            try:
                return self.book.begin("hypothesis-a", f"correction-{number}", writer=number)
            except ValueError:
                return None

        with ThreadPoolExecutor(max_workers=5) as pool:
            attempts = list(pool.map(begin, range(5)))
        self.assertEqual(sum(attempt is not None for attempt in attempts), 1)
        self.assertEqual(self.book.get("hypothesis-a")["attempt_count"], 2)

    def test_registration_cannot_reset_origin_or_exhausted_budget(self):
        self.register()
        for index in range(3):
            self.book.begin("hypothesis-a", f"assignment-{index}")
            self.book.finish(f"assignment-{index}", stage="source", error="Invalid source.")
        self.assertEqual(self.register()["status"], "exhausted")
        with self.assertRaisesRegex(ValueError, "origin changed"):
            self.book.register("task-a", {**self.hypothesis, "change": "Pretend this is a new initial attempt."})
        with self.assertRaisesRegex(ValueError, "origin changed"):
            self.book.register("task-b", self.hypothesis)

    def test_close_only_pending_hypothesis_and_keep_its_history(self):
        self.register()
        self.register("hypothesis-b")
        self.book.begin("hypothesis-a", "assignment-0")
        with self.assertRaises(ValueError):
            self.book.close("hypothesis-a", "Do not discard running work.")
        self.book.finish("assignment-0", result=measured_result(), candidate_id="candidate-a")
        record = self.book.close("hypothesis-a", "Measured gain has low value.", evidence=["candidate-a"])
        self.assertEqual(record["status"], "closed")
        self.assertEqual(len(record["attempts"]), 1)
        self.assertEqual(record["close_reason"]["evidence"], ["candidate-a"])
        self.assertEqual(self.book.get("hypothesis-b")["status"], "ready")
        self.assertEqual(self.book.outstanding(), [])

    def test_stagnation_requires_actual_numerical_or_timing_evidence(self):
        for stage, error in (("api", RuntimeError("429")), ("writer", ValueError("Bad schema")),
                             ("source", "Illegal source"), ("compile", "CUDA build failed"),
                             ("operation", RuntimeError("CUDA launch failed"))):
            with self.subTest(stage=stage):
                outcome = classify_fault(stage, error=error)
                self.assertFalse(outcome["counts_for_stagnation"])
                self.assertTrue(outcome["correction_needed"])
        numeric = classify_fault("model", result=measured_result(correct=False))
        self.assertEqual(numeric["kind"], "correctness_failure")
        self.assertTrue(numeric["counts_for_stagnation"])
        slow = classify_fault("operation", result=measured_result())
        self.assertEqual(slow["kind"], "performance_rejection")
        self.assertTrue(slow["counts_for_stagnation"])
        self.assertFalse(measured_gpu_outcome({"decision": {"accepted": True, "speedup": 100}}))
        self.assertFalse(measured_gpu_outcome({"cases": [{"correct": True, "checks": [{"passed": True}]}]}))
        malformed = measured_result()
        malformed["cases"][0]["candidate"]["samples_ms"] = [float("nan")]
        self.assertFalse(measured_gpu_outcome(malformed))

    def test_compile_error_inside_worker_report_never_counts_as_plateau(self):
        result = {"error": "Worker failed.", "worker_report": {"error": "RuntimeError: CUDA build failed.",
                  "build_log": "runs/build/compiler.log"}, "decision": {"accepted": False}}
        before = copy.deepcopy(result)
        outcome = classify_fault("operation", result=result)
        self.assertEqual(outcome["kind"], "compile_error")
        self.assertFalse(outcome["counts_for_stagnation"])
        self.assertEqual(result, before)

    def test_nested_layer_numerical_failure_counts_as_measured_gpu_outcome(self):
        result = layer_result(passed=False)
        before = copy.deepcopy(result)
        self.assertTrue(measured_gpu_outcome(result))
        outcome = classify_fault("model", result=result)
        self.assertEqual(outcome["kind"], "correctness_failure")
        self.assertTrue(outcome["counts_for_stagnation"])
        self.assertEqual(result, before)

    def test_passing_layers_without_timings_and_missing_coverage_do_not_count(self):
        self.assertFalse(measured_gpu_outcome(layer_result(passed=True)))
        missing = layer_result(passed=True, missing=True)
        self.assertFalse(missing["cases"][0]["checks"][0]["passed"])
        self.assertIsNotNone(missing["cases"][0]["checks"][0]["max_abs_error"])
        self.assertFalse(measured_gpu_outcome(missing))
        self.assertFalse(classify_fault("model", result=missing)["counts_for_stagnation"])
        self.assertFalse(measured_gpu_outcome(layer_result(passed=False, missing=True)),
                         "A partial trace must not stand in for a complete GPU outcome.")
        missing["cases"][0].update(baseline={"samples_ms": [2.] * 15}, candidate={"samples_ms": [1.] * 15})
        self.assertFalse(measured_gpu_outcome(missing), "Stale timings cannot override a missing-coverage case.")

    def test_nested_errors_mutation_and_unknown_paths_do_not_count_as_numerical_progress(self):
        for mutation in ("compile_error", "runtime_error", "input_mutation", "unobserved", "seed_error", "nonfinite", "missing_layers"):
            with self.subTest(mutation=mutation):
                result = layer_result()
                seed = result["cases"][0]["checks"][0]
                layer = seed["layers"][0]
                if mutation == "compile_error":
                    layer["error"] = "CUDA build failed."
                elif mutation == "runtime_error":
                    layer["error"] = "CUDA launch failed."
                elif mutation == "input_mutation":
                    layer["inputs_unchanged"] = False
                elif mutation == "unobserved":
                    seed["observed_paths"] = []
                elif mutation == "seed_error":
                    seed["error"] = "The trace did not complete."
                elif mutation == "missing_layers":
                    seed["layers"] = None
                else:
                    layer["check"]["max_abs_error"] = float("inf")
                self.assertFalse(measured_gpu_outcome(result))
                self.assertFalse(classify_fault("model", result=result)["counts_for_stagnation"])

    def test_nested_pass_with_valid_complete_timings_counts_for_stagnation(self):
        result = layer_result(passed=True)
        result["cases"][0].update(baseline={"samples_ms": [2.] * 15}, candidate={"samples_ms": [2.] * 15})
        self.assertTrue(measured_gpu_outcome(result))
        self.assertEqual(classify_fault("model", result=result)["kind"], "performance_rejection")

    def test_skipped_or_unsafe_flags_at_each_evaluator_level_never_count(self):
        flags = (("not_measured", True), ("reference_state_intact", False),
                 ("restoration_errors", [{"input_index": 0, "error": "Cannot restore the input."}]))
        for level in ("result", "case", "seed", "layer", "check"):
            for flag, value in flags:
                with self.subTest(level=level, flag=flag):
                    result = layer_result()
                    case = result["cases"][0]
                    seed = case["checks"][0]
                    layer = seed["layers"][0]
                    target = {"result": result, "case": case, "seed": seed,
                              "layer": layer, "check": layer["check"]}[level]
                    target[flag] = copy.deepcopy(value)
                    self.assertFalse(measured_gpu_outcome(result))
                    self.assertFalse(classify_fault("model", result=result)["counts_for_stagnation"])

    def test_unsafe_later_case_cannot_reuse_earlier_valid_timing_for_stagnation(self):
        result = measured_result()
        result["cases"].append({"correct": False, "checks": [
            {"seed": 2718, "scope": "replaced_layers", "passed": False, "not_measured": True,
             "reference_state_intact": False, "reason": "The reference state could not be restored.", "layers": []}]})
        self.assertFalse(measured_gpu_outcome(result))
        self.assertFalse(classify_fault("model", result=result)["counts_for_stagnation"])

    def test_explicit_safe_flags_do_not_hide_real_gpu_outcomes(self):
        for result in (measured_result(), layer_result()):
            result.update(not_measured=False, reference_state_intact=True, restoration_errors=[])
            result["cases"][0].update(reference_state_intact=True, restoration_errors=[])
            self.assertTrue(measured_gpu_outcome(result))


if __name__ == "__main__":
    unittest.main()
