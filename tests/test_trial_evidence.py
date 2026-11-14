import contextlib
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sako.deployment_diagnostics import performance_trace
from sako.launch_evidence import graph_execution, one_invocation, trace_launches
from sako.swarm_policy import validate_plan
from sako.trial_evidence import measured_opportunity, stopping_evidence, trial_observations, verification_spec


def launch(name="kernel", blocks=7):
    return {"name": name, "grid": [blocks, 1, 1], "block": [128, 1, 1]}


class LaunchEvidenceTests(unittest.TestCase):
    def test_graph_match_uses_the_submission_and_retains_other_timed_launches(self):
        def host(name, correlation):
            return {"cat": "cuda_runtime", "ph": "X", "name": name, "args": {"correlation": correlation}}

        def gpu(name, correlation, blocks, timestamp):
            return {"cat": "kernel", "ph": "X", "name": name, "ts": timestamp,
                    "args": {"correlation": correlation, "grid": [blocks, 1, 1], "block": [128, 1, 1]}}

        # Both the host and the graph submit a kernel with the same name.
        events = [host("cudaLaunchKernel", 3), host("cudaGraphLaunch", 4), gpu("work", 3, 1, 0),
                  gpu("work", 4, 7, 1), gpu("end", 4, 1, 2), gpu("work", 4, 7, 3), gpu("end", 4, 1, 4)]
        result = graph_execution({"traceEvents": events}, 2)
        self.assertEqual(result["launches"], [launch("work", 7), launch("end", 1)])
        self.assertEqual(result["graph_replay"], {"operations_per_replay": 2, "graph_node_launches": 4,
                         "additional_timed_launches_per_replay": [launch("work", 1)]})
        for ambiguous in (events[1:], events + [host("cuGraphLaunch", 10)], events[:-1]):
            with self.assertRaises(ValueError):
                graph_execution({"traceEvents": ambiguous}, 2)

    def test_launches_are_ordered_and_repeated_calls_have_the_same_geometry(self):
        trace = {"traceEvents": [
            {"cat": "kernel", "ph": "X", "name": name, "args": {"grid": [size, 1, 1],
                "block": [128, 1, 1], "correlation": index}} for name, size, index in
            [("reduce", 1, 21), ("work", 56, 20)]]}
        kernels = [launch("work", 56), launch("reduce", 1)]
        self.assertEqual(trace_launches(trace), kernels)
        self.assertEqual(one_invocation(kernels * 3, 3), kernels)
        with self.assertRaises(ValueError):
            one_invocation(kernels + list(reversed(kernels)), 2)


class PredictionTests(unittest.TestCase):
    def setUp(self):
        self.candidate = {"backend": "triton", "source": "import triton.language as t\ndef kernel():\n t.store()\n"
                          "def unused():\n t.sum()\n", "strategy": "A different implementation."}
        self.result = {"cases": [{"case_index": 0, "executions": {
            "candidate": {"status": "observed", "launches": [launch()]}}}]}
        self.spec = verification_spec({"fixture_index": 0, "implementation": [
            {"field": "source.calls.triton.language.sum", "op": "ge", "value": 1},
            {"field": "launch.0.blocks", "op": "eq", "value": 7}]})

    def observe(self):
        return trial_observations({"verification": self.spec, "change": "Use a reduction."}, self.candidate,
                                  self.result)

    def test_unlaunched_source_cannot_confirm_the_assigned_change(self):
        evidence = self.observe()
        self.assertEqual(evidence["status"], "contradicted")
        self.assertEqual([r["observed"] for r in evidence["implementation"]], [0, 7])
        self.assertEqual(evidence["submitted_strategy"], self.candidate["strategy"])
        self.result = {"cases": []}
        self.assertEqual(self.observe()["status"], "unverified")

    def test_stopping_reason_can_only_report_evaluator_observations(self):
        reply = {"reason": "All designs saturated memory.", "evidence": ["new"]}
        measured = {"new": {"decision": {"accepted": False, "speedup": .9}}}
        result = stopping_evidence(reply, measured, {}, "task")
        self.assertNotIn("saturated", result["reason"])
        self.assertEqual(result["agent_reason"], reply["reason"])
        self.assertEqual(result["stopping_basis"]["measured_candidates"][0]["candidate_id"], "new")

    def test_selected_tuning_options_can_distinguish_equal_launch_grids(self):
        self.spec["implementation"] = [{"field": "tune.0.0.num_stages", "op": "eq", "value": 4}]
        runs = self.result["cases"][0]["executions"]
        runs["candidate"]["tuners"] = [{"invoked": True, "choices": [{"launch_options": {"num_stages": 4}}]}]
        runs["reference"] = copy.deepcopy(runs["candidate"])
        self.assertEqual(self.observe()["status"], "unverified")
        runs["reference"]["tuners"][0]["choices"][0]["launch_options"]["num_stages"] = 2
        observed = self.observe()
        self.assertEqual(observed["status"], "observed")
        self.assertEqual(observed["implementation"][0]["timing_reference"], 2)
        runs["candidate"]["tuners"][0]["invoked"] = False
        self.assertEqual(self.observe()["status"], "unverified")


    def test_dispatch_ranks_predicted_model_savings_and_preserves_evidence_scope(self):
        h = {"specialty": "memory", "parent_id": None, "claim": "Test the grid.", "change": "A",
             "expected_metric": "Reduce time.", "evidence": ["contract"], "expected_operation_speedup": 1.25}
        reply = {"action": "continue", "reason": "Test both.", "evidence": ["contract"],
                 "hypotheses": [h, {**h, "change": "B", "expected_operation_speedup": 2}]}
        plan = validate_plan(reply, task_id="task", known_evidence={"contract"}, parent_ids=[],
                             model_costs=[{"region_ms": 2}])
        self.assertEqual([h["change"] for h in plan["hypotheses"]], ["B", "A"])
        self.assertEqual(plan["hypotheses"][0]["estimated_model_saving_ms"], 1)
        self.assertEqual(plan["replan_after"], "all_results")
        case = {"workload": {"b": 2}, "correct": True, "search_reference": {"kind": "paired_compile",
                "baseline": {"samples_ms": [2]}, "candidate": {"samples_ms": [1]}}}
        estimates = measured_opportunity([{"workload": {"b": 2}, "region_ms": 4, "model_ms": 10}],
                                         {"cases": [case, {**case, "workload": {"b": 1}}]})
        self.assertEqual([r["estimated_model_saving_ms"] for r in estimates], [2])
        case.update(baseline={"samples_ms": [1.1]}, candidate={"samples_ms": [1.]},
                    search_reference={"kind": "fixed_anchor", "baseline_median_ms": 2.})
        estimates = measured_opportunity([{"workload": {"b": 2}, "region_ms": 4}], {"cases": [case]})
        self.assertEqual(estimates[0]["estimated_model_saving_ms"], 2)
        self.assertEqual(estimates[0]["reference_kind"], "fixed_anchor")

    def test_trace_selects_the_workload_that_missed_speed(self):
        config = {"min_speedup": 1.02, "trials": 1, "max_p95_ratio": 1.05}
        cases = [{"workload": {"b": i}, "correct": True, "baseline": {"samples_ms": [1]},
                  "candidate": {"samples_ms": [ms]}} for i, ms in enumerate([.5, 1.2])]
        with patch.dict("sys.modules", {"torch": SimpleNamespace(inference_mode=contextlib.nullcontext)}), \
             patch("sako.deployment_diagnostics.capture_model_traces", return_value={"reports": {}}) as capture_trace:
            trace = performance_trace("directory", {"accepted": False}, cases, config, "base", "new", lambda w: w["b"])
            capture_trace.assert_called_once_with("base", "new", 1, "directory")
            self.assertEqual(trace["workload"], {"b": 1})
            self.assertIsNone(performance_trace("directory", {"accepted": True}, cases, config, None, None, None))
            bad = copy.deepcopy(cases)
            bad[0]["correct"] = False
            self.assertIsNone(performance_trace("directory", {"accepted": False}, bad, config, None, None, None))
