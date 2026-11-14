"""Trusted extra cases must preserve the timing contract and exact inputs."""

import contextlib
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from sako.core import read_json, write_json
from sako.gpu import evaluate_supplemental
from sako.supplemental_correctness import (CaptureBuffer, binding, load_fixture,
    capture_required_failures, ensure_fixture, record_required_failure, representative_paths, validate_case, _CAPTURE)


class Tensor:
    def __init__(self, value, shape=(2, 3), strides=(7, 1), dtype="torch.float16", device="cpu"):
        self.value, self.shape, self.strides, self.dtype, self.device = value, shape, strides, dtype, device
    def stride(self): return self.strides
    def detach(self): return self
    def clone(self): return copy.deepcopy(self)
    def cpu(self): return self.clone()
    def cuda(self):
        result = self.clone(); result.device = "cuda"; return result
    def copy_(self, other): self.value = other.value; return self


def compare(actual, expected, config):
    error = abs(actual.value - expected.value)
    return {"passed": error <= config["atol"] + config["rtol"] * abs(expected.value),
            "max_abs_error": error, "atol": config["atol"], "rtol": config["rtol"]}


class SupplementalCorrectnessTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.profile = Path(temporary.name)
        self.config = {"workloads": [{"batch_size": 1, "sequence_length": 128}],
            "correctness_seeds": [1729, 2718], "atol": .03, "rtol": .02,
            "supplemental_correctness": {"enabled": True, "max_paths": 3}}
        self.task = {"id": "linear_1", "kind": "linear", "paths": ["layer0.gate_proj", "layer1.up_proj"],
            "scalars": [], "cases": [{"seed": 1729}, {"seed": 2718}]}
        self.catalog = {"tasks": [self.task], "hardware": {"fingerprint": "gpu-A"},
            "config": {"model_id": "test", "revision": "commit", "dtype": "float16", "attention": "sdpa"}}
        self.context = {"seed": 1729, "workload": self.config["workloads"][0]}
        self.saved = {}
        def save(value, path):
            self.saved[str(path)] = copy.deepcopy(value)
            Path(path).write_bytes(b"trusted-tensor-archive")
        self.torch = SimpleNamespace(Tensor=Tensor, inference_mode=contextlib.nullcontext,
            empty_strided=lambda shape, stride, dtype, device: Tensor(0, shape, stride, dtype, device),
            equal=lambda a, b: a.value == b.value and a.shape == b.shape,
            save=save, load=Mock(side_effect=lambda path, **kwargs: copy.deepcopy(self.saved[str(path)])))
        stack = contextlib.ExitStack()
        stack.enter_context(patch.dict("sys.modules", {"torch": self.torch}))
        stack.enter_context(patch("sako.supplemental_correctness.implementation_hash", return_value="source-A"))
        stack.enter_context(patch("sako.compat.configure_compiler"))
        self.addCleanup(stack.close)

    def capture(self, *, path="layer0.gate_proj", value=3, origin="representative", context=None):
        capture = CaptureBuffer(self.profile, self.config, self.catalog)
        capture.append(self.task["id"], path, [Tensor(value), Tensor(2)], [], Tensor(value * 2),
            context or self.context, origin=origin)
        capture.flush()
        return load_fixture(self.profile, self.config, self.catalog, self.task)

    def test_representatives_cover_depth_and_projection_families(self):
        paths = [f"model.layers.{depth}.self_attn.{kind}_proj" for depth in range(28) for kind in ("q", "k", "v")]
        chosen = representative_paths({"paths": paths})
        self.assertEqual(chosen[0], paths[0])
        self.assertEqual(chosen[-1], paths[-1])
        self.assertIn(int(chosen[1].split(".")[2]), {13, 14})
        self.assertEqual({path.rsplit(".", 1)[-1] for path in chosen}, {"q_proj", "k_proj", "v_proj"})

    def test_capture_preserves_exact_strides_and_standard_timing_cases(self):
        original = copy.deepcopy(self.task)
        fixture, entry = self.capture()
        case = fixture["cases"][0]
        self.assertEqual(case["strides"], [[7, 1], [7, 1]])
        self.assertEqual(case["expected"].value, 6)
        self.assertEqual(entry["case_count"], 1)
        self.assertEqual(self.task, original)
        self.torch.load.assert_called_with(self.profile / entry["path"], map_location="cpu", weights_only=True)

    def test_binding_changes_with_source_hardware_inputs_and_limits(self):
        original = binding(self.config, self.catalog, self.task)
        for field, value in (("correctness_seeds", [9]), ("atol", .04),
                             ("workloads", [{"batch_size": 2, "sequence_length": 128}])):
            self.assertNotEqual(original, binding({**self.config, field: value}, self.catalog, self.task))
        catalog = {**self.catalog, "hardware": {"fingerprint": "gpu-B"}}
        self.assertNotEqual(original, binding(self.config, catalog, self.task))
        self.assertNotEqual(original, binding(self.config, self.catalog, {**self.task, "paths": ["other"]}))
        with patch("sako.supplemental_correctness.implementation_hash", return_value="source-B"):
            self.assertNotEqual(original, binding(self.config, self.catalog, self.task))

    def test_representative_capture_covers_all_seeds_and_paths_then_uses_cache(self):
        hooks = {}
        def register(path, hook):
            hooks[path] = hook
            return SimpleNamespace(remove=lambda: hooks.pop(path))
        modules = {path: SimpleNamespace(weight=Tensor(index + 2),
            register_forward_hook=lambda hook, path=path: register(path, hook))
            for index, path in enumerate(self.task["paths"])}
        model = SimpleNamespace(get_submodule=lambda path: modules[path])
        session = SimpleNamespace(get_model=lambda config: model)
        attention = {"sdpa": object()}
        def forward(model, seed):
            for index, path in enumerate(self.task["paths"]):
                x = Tensor(seed + index)
                hooks[path](modules[path], [x], Tensor(x.value * modules[path].weight.value))
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.dict("sys.modules", {"transformers.modeling_utils":
                SimpleNamespace(ALL_ATTENTION_FUNCTIONS=attention)}))
            stack.enter_context(patch("sako.model.arguments", side_effect=lambda kind, module, x: ([x, module.weight], [])))
            stack.enter_context(patch("sako.model.inputs", side_effect=lambda model, workload, seed: seed))
            calls = stack.enter_context(patch("sako.model.forward", side_effect=forward))
            fixture, entry = ensure_fixture(self.profile, self.config, self.catalog, self.task, session=session)
            self.assertEqual(calls.call_count, 2)
            self.assertEqual(len(fixture["cases"]), 4)
            self.assertTrue(fixture["coverage_complete"])
            self.assertEqual(hooks, {})
            cached, cached_entry = ensure_fixture(self.profile, self.config, self.catalog, self.task, session=session)
            self.assertEqual(calls.call_count, 2)
            self.assertEqual(cached_entry, entry)
            self.assertEqual(len(cached["cases"]), 4)

    def test_corrupt_archive_fails_before_tensor_load(self):
        _, entry = self.capture()
        self.torch.load.reset_mock()
        (self.profile / entry["path"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "archive changed"):
            load_fixture(self.profile, self.config, self.catalog, self.task)
        self.torch.load.assert_not_called()

    def test_archive_path_escape_fails_before_tensor_load(self):
        self.capture()
        index_path = self.profile / "supplemental/index.json"
        index = read_json(index_path)
        index[self.task["id"]]["path"] = "../outside.pt"
        write_json(index_path, index)
        self.torch.load.reset_mock()
        with self.assertRaises(ValueError):
            load_fixture(self.profile, self.config, self.catalog, self.task)
        self.torch.load.assert_not_called()

    def test_untrusted_case_path_seed_scalars_and_metadata_fail(self):
        fixture, _ = self.capture()
        case = fixture["cases"][0]
        for changes in ({"source_path": "other"}, {"seed": 999}, {"scalars": [0.1]},
                        {"workload": {"batch_size": 99}}, {"strides": [[1], [1]]},
                        {"dtypes": ["torch.float32"]}, {"call_index": -1}, {"origin": "agent"}):
            with self.assertRaises(ValueError):
                validate_case({**case, **changes}, self.task, self.config)

    def test_replay_preserves_stride_and_never_produces_timing_samples(self):
        fixture, _ = self.capture()
        def operation(tensors, scalars):
            self.assertTrue(all(t.stride() == (7, 1) and t.device == "cuda" for t in tensors))
            return Tensor(tensors[0].value * 2)
        with patch("sako.gpu.custom_op", return_value=operation), \
                patch("sako.gpu.check_output", side_effect=compare), patch("sako.gpu.paired_timings") as timing:
            result = evaluate_supplemental({}, self.task, fixture, self.config)
        self.assertTrue(result["passed"])
        self.assertFalse(result["timing_samples_used"])
        self.assertEqual(result["cases"][0]["timing_status"], "not_measured")
        self.assertNotIn("baseline", result["cases"][0])
        timing.assert_not_called()

    def test_failure_capture_cap_keeps_all_failure_counts_and_reuses_weights(self):
        self.config["supplemental_correctness"]["max_failure_cases"] = 2
        capture = CaptureBuffer(self.profile, self.config, self.catalog)
        for path in self.task["paths"]:
            for seed in self.config["correctness_seeds"]:
                capture.append(self.task["id"], path, [Tensor(seed), Tensor(2)], [], Tensor(seed * 2),
                    {**self.context, "seed": seed}, origin="integration_failure")
        capture.flush()
        fixture, _ = load_fixture(self.profile, self.config, self.catalog, self.task)
        self.assertEqual(len(fixture["cases"]), 2)
        self.assertIs(fixture["cases"][0]["tensors"][1], fixture["cases"][1]["tensors"][1])
        summary = capture.failure_summary[self.task["id"]]
        self.assertEqual(summary["observed_failure_count"], 4)
        self.assertEqual(summary["uncaptured_count"], 2)
        self.assertEqual(len(summary["saved_cases"]), 2)
        self.assertEqual(len(capture.artifacts), 1)

    def test_exact_layer_failure_is_saved_with_its_original_inputs_and_output(self):
        from sako.layer_checks import LayerOutputChecker
        path = self.task["paths"][1]
        checker = LayerOutputChecker(self.config,
            {path: {"task_id": self.task["id"], "candidate_id": "candidate", "kind": "linear"}},
            {"candidate": lambda tensors, scalars: Tensor(99)})
        checker.context = self.context
        tensors = [Tensor(3), Tensor(2)]
        with patch("sako.gpu.check_output", side_effect=compare):
            with capture_required_failures(self.profile, self.config, self.catalog) as capture:
                checker.compare(path, tensors, [], Tensor(6))
        self.assertFalse(checker.records[0]["passed"])
        fixture, _ = load_fixture(self.profile, self.config, self.catalog, self.task)
        case = fixture["cases"][0]
        self.assertEqual(case["origin"], "integration_failure")
        self.assertEqual(case["source_path"], path)
        self.assertEqual(case["tensors"][0].value, 3)
        self.assertEqual(case["strides"][0], [7, 1])
        self.assertEqual(case["expected"].value, 6)
        self.assertEqual(capture.failure_summary[self.task["id"]]["uncaptured_count"], 0)

    def test_saved_failure_runs_before_representative_cases(self):
        self.capture()
        fixture, _ = self.capture(path=self.task["paths"][1], value=4, origin="integration_failure")
        values = []
        def operation(tensors, scalars):
            values.append(tensors[0].value)
            return Tensor(tensors[0].value * 2)
        with patch("sako.gpu.custom_op", return_value=operation), patch("sako.gpu.check_output", side_effect=compare):
            result = evaluate_supplemental({}, self.task, fixture, self.config)
        self.assertTrue(result["passed"])
        self.assertEqual(values, [4, 4, 4, 3, 3, 3])

    def test_numerical_failure_and_input_mutation_are_required_failures(self):
        fixture, _ = self.capture()
        for mutate in (False, True):
            def operation(tensors, scalars):
                if mutate: tensors[0].value = 999
                return Tensor(99 if not mutate else 6)
            with patch("sako.gpu.custom_op", return_value=operation), patch("sako.gpu.check_output", side_effect=compare):
                result = evaluate_supplemental({}, self.task, fixture, self.config)
            self.assertFalse(result["passed"])
            self.assertEqual(result["cases"][0]["inputs_unchanged"], not mutate)

    def test_only_required_numerical_failures_enter_replay_archive(self):
        capture = SimpleNamespace(tasks={self.task["id"]: self.task},
            config={"containing_region_policy": "diagnostic"}, append=Mock())
        record = {"task_id": self.task["id"], "candidate_id": "candidate", "path": self.task["paths"][0],
            "call_index": 0, "check": {"passed": False}, "inputs_unchanged": True}
        token = _CAPTURE.set(capture)
        try:
            record_required_failure(record, [Tensor(3)], [], Tensor(6), self.context)
            self.assertEqual(capture.append.call_count, 1)
            for changes in ({"candidate_id": "region:mlp"}, {"inputs_unchanged": False},
                            {"check": {"passed": True}}, {"error": "compile error"}):
                record_required_failure({**record, **changes}, [Tensor(3)], [], Tensor(6), self.context)
            self.assertEqual(capture.append.call_count, 1)
        finally:
            _CAPTURE.reset(token)

    def test_worker_skips_operation_timing_after_supplemental_failure(self):
        from sako.worker import execute
        candidate = {"task_id": self.task["id"]}
        failure = {"correct": False, "checks": [{"passed": False, "max_abs_error": 2.}],
            "inputs_unchanged": True, "phase": "supplemental_correctness"}
        supplemental = {"passed": False, "cases": [failure], "timing_samples_used": False}
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("sako.worker.load_config", return_value=self.config))
            stack.enter_context(patch("sako.worker.read_json", side_effect=[self.catalog, candidate]))
            stack.enter_context(patch("sako.worker.validate_candidate"))
            stack.enter_context(patch("sako.worker.candidate_id", return_value="candidate-id"))
            stack.enter_context(patch("sako.preparation.validate_catalog"))
            stack.enter_context(patch("sako.hardware.discover_hardware", return_value={}))
            stack.enter_context(patch("sako.supplemental_correctness.ensure_fixture", return_value=({}, {"sha256": "saved"})))
            stack.enter_context(patch("sako.gpu.evaluate_supplemental", return_value=supplemental))
            timing = stack.enter_context(patch("sako.gpu.evaluate_fixture"))
            result = execute({"config": "config", "action": "evaluate", "profile_dir": "profile", "candidate": "candidate"})
        self.assertFalse(result["decision"]["accepted"])
        self.assertEqual(result["cases"], [failure])
        self.assertIn("supplemental_checks_s", result["stage_times"])
        timing.assert_not_called()

    def test_numerical_repair_priority_is_separate_from_performance_hypothesis(self):
        from sako.swarm import SwarmLoop
        hypothesis = {"id": "same-hypothesis", "hypothesis": "Use a larger tile for speed."}
        failure = {"cases": [{"correct": False, "phase": "supplemental_correctness",
            "source_path": "layer1.up_proj", "checks": [{"passed": False, "max_abs_error": .5}]}]}
        assignment = {"task_id": "task", "parent_id": None, "island": "compute", "slot": 0,
            "generation": 1, "hypothesis": hypothesis, "attempt_number": 2, "correction_attempt": 1,
            "repair": {"stage": "operation", "result": failure}, "selection": {}}
        loop = SimpleNamespace(by_id={"task": {"id": "task", "kind": "linear"}},
            lessons=SimpleNamespace(retrieve=lambda *a, **kw: []), hardware={},
            limits={"context_chars": 100000}, config={"kernel_backend": "triton"},
            retrieve=lambda *a, **kw: [], task_prompts={"task": {}}, validation_policy={}, store={},
            repair_feedback=lambda x: x, diagnostic_feedback=lambda x: {}, model_costs=lambda _: [], kept_summary=lambda _: None,
            profiler_unavailable=None, events=[],
            request_writer=lambda a, prompt: prompt)
        prompt = SwarmLoop.write_swarm(loop, assignment)
        self.assertEqual(prompt["repair_objective"]["priority"], "numerical_correctness")
        self.assertTrue(prompt["repair_objective"]["preserve_hypothesis_identity"])
        self.assertEqual(prompt["hypothesis"], hypothesis)
        self.assertEqual(prompt["repair_objective"]["failed_cases"][0]["source_path"], "layer1.up_proj")


if __name__ == "__main__":
    unittest.main()
