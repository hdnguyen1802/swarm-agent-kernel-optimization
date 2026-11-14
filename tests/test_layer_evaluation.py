import contextlib
import copy
import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from sako.layer_checks import LayerOutputChecker, measure_layer_case, replacement_paths
from sako.model import measure_model


class Tensor:
    dtype, device = "float16", "cuda"
    def __init__(self, value, shape=(1,)):
        self.value, self.shape = value, shape
    @property
    def ndim(self): return len(self.shape)
    def stride(self):
        sizes, stride = [], 1
        for length in reversed(self.shape):
            sizes.append(stride)
            stride *= length
        return tuple(reversed(sizes))
    def detach(self): return self
    def clone(self): return copy.deepcopy(self)
    def copy_(self, other): self.value = other.value; return self
    def transpose(self, left, right):
        shape = list(self.shape)
        shape[left], shape[right] = shape[right], shape[left]
        return Tensor(self.value, tuple(shape))
    def contiguous(self): return self


def check(actual, expected, config):
    if not isinstance(actual, Tensor) or actual.shape != expected.shape or actual.dtype != expected.dtype:
        return {"passed": False, "reason": "Output metadata differs."}
    finite = math.isfinite(actual.value) and math.isfinite(expected.value)
    error = abs(actual.value - expected.value) if finite else None
    return {"passed": finite and error <= config["atol"] + config["rtol"] * abs(expected.value),
            "max_abs_error": error, "atol": config["atol"], "rtol": config["rtol"]}


class Module:
    def __init__(self, weight):
        self.weight, self.hooks = Tensor(weight), []
    def __call__(self, tensor):
        output = Tensor(tensor.value * self.weight.value)
        for hook in tuple(self.hooks):
            hook(self, (tensor,), output)
        return output
    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        return SimpleNamespace(remove=lambda: self.hooks.remove(hook))


class Model:
    def __init__(self, count=2):
        self.modules = {f"layer{index}": Module(1.01 + index / 1000) for index in range(count)}
        self.last_output = None
    def get_submodule(self, path): return self.modules[path]
    def __call__(self, tensor):
        for module in self.modules.values():
            tensor = module(tensor)
        self.last_output = tensor
        return tensor


class LayerEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.config = {"atol": .03, "rtol": .02, "compile_mode": "default", "revision": "main",
                       "correctness_scope": "replaced_layers", "correctness_reference": "eager",
                       "model_reference": "torch_compile", "workloads": [{"batch_size": 1, "sequence_length": 128},
                                                                              {"batch_size": 2, "sequence_length": 128}],
                       "correctness_seeds": [1729, 2718, 3141], "trials": 5, "min_speedup": 1.02,
                       "max_p95_ratio": 1.05}
        self.attention = {"sdpa": object()}
        self.torch = SimpleNamespace(Tensor=Tensor, equal=lambda a, b: a.shape == b.shape and a.value == b.value,
                                     inference_mode=contextlib.nullcontext,
                                     cuda=SimpleNamespace(max_memory_allocated=lambda: 123))

    def manifest(self, count, kind="linear"):
        paths = [f"layer{index}" for index in range(count)]
        return {"entries": [{"task_id": "task", "candidate_id": "candidate", "paths": paths}]}, {
            "config": {"revision": "pinned"}, "tasks": [{"id": "task", "kind": kind, "paths": paths}]}

    def environment(self):
        stack = contextlib.ExitStack()
        stack.enter_context(patch.dict("sys.modules", {"torch": self.torch,
            "transformers.modeling_utils": SimpleNamespace(ALL_ATTENTION_FUNCTIONS=self.attention)}))
        stack.enter_context(patch("sako.model.arguments", side_effect=lambda kind, module, x: ([x, module.weight], [])))
        stack.enter_context(patch("sako.model.inputs", return_value=Tensor(1)))
        stack.enter_context(patch("sako.model.forward", side_effect=lambda model, ids: model(ids)))
        stack.enter_context(patch("sako.gpu.check_output", side_effect=check))
        return stack

    def test_every_layer_uses_its_own_actual_weight_and_original_input(self):
        model = Model(28)
        manifest, catalog = self.manifest(28)
        visited = []
        def operation(tensors, scalars):
            visited.append((tensors[0].value, tensors[1].value))
            # A small local perturbation must not feed the next reference layer.
            return Tensor(tensors[0].value * tensors[1].value + .001)
        with self.environment():
            result = measure_layer_case(model, manifest, catalog, {"candidate": operation}, self.config,
                                        self.config["workloads"][0], 1729)
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["layers"]), 28)
        self.assertEqual(len({weight for _, weight in visited}), 28)
        expected_input = 1.
        for actual_input, weight in visited:
            self.assertEqual(actual_input, expected_input)
            expected_input *= weight
        self.assertEqual(model.last_output.value, expected_input)
        self.assertTrue(all(not module.hooks for module in model.modules.values()))

    def test_later_layer_failure_cannot_hide_behind_first_layer_pass(self):
        model = Model(57)
        manifest, catalog = self.manifest(57)
        def operation(tensors, scalars):
            error = 2 if tensors[1].value == model.modules["layer40"].weight.value else 0
            return Tensor(tensors[0].value * tensors[1].value + error)
        with self.environment():
            result = measure_layer_case(model, manifest, catalog, {"candidate": operation}, self.config,
                                        self.config["workloads"][0], 1729)
        self.assertFalse(result["passed"])
        self.assertEqual(result["failed_paths"], ["layer40"])
        self.assertEqual(len(result["observed_paths"]), 57)

    def test_mutation_is_rejected_and_reference_weights_are_restored(self):
        model = Model(2)
        manifest, catalog = self.manifest(2)
        before = [module.weight.value for module in model.modules.values()]
        def operation(tensors, scalars):
            output = Tensor(tensors[0].value * tensors[1].value)
            tensors[1].value = 999.
            return output
        with self.environment():
            result = measure_layer_case(model, manifest, catalog, {"candidate": operation}, self.config,
                                        self.config["workloads"][0], 1729)
        self.assertFalse(result["passed"])
        self.assertEqual([module.weight.value for module in model.modules.values()], before)
        self.assertEqual(result["layers"][0]["mutated_input_indices"], [1])
        self.assertEqual(result["layers"][0]["restored_input_indices"], [1])

    def test_wrong_shape_nonfinite_and_missing_path_fail(self):
        manifest, catalog = self.manifest(2)
        for operation in (lambda tensors, scalars: Tensor(1, (2,)), lambda tensors, scalars: Tensor(float("nan"))):
            with self.environment():
                result = measure_layer_case(Model(2), manifest, catalog, {"candidate": operation}, self.config,
                                            self.config["workloads"][0], 1729)
            self.assertFalse(result["passed"])
        with self.environment():
            result = measure_layer_case(Model(1), manifest, catalog, {}, self.config, self.config["workloads"][0], 1729)
        self.assertFalse(result["passed"])
        self.assertIn("layer1", result["missing_paths"])

    def test_failed_restoration_aborts_the_original_trajectory(self):
        model = Model(2)
        manifest, catalog = self.manifest(2)
        def cannot_restore(other):
            raise RuntimeError("Device copy failed.")
        model.modules["layer0"].weight.copy_ = cannot_restore
        visited = []
        def operation(tensors, scalars):
            visited.append(tensors[1].value)
            output = Tensor(tensors[0].value * tensors[1].value)
            tensors[1].value = 999.
            return output
        with self.environment():
            result = measure_layer_case(model, manifest, catalog, {"candidate": operation}, self.config,
                                        self.config["workloads"][0], 1729)
        self.assertFalse(result["passed"])
        self.assertFalse(result["reference_state_intact"])
        self.assertEqual(len(visited), 1)
        self.assertIn("layer1", result["missing_paths"])
        self.assertTrue(result["layers"][0]["restoration_errors"])
        self.assertTrue(all(not module.hooks for module in model.modules.values()))

    def test_attention_compares_original_sdpa_output_and_restores_dispatch(self):
        module = SimpleNamespace()
        model = SimpleNamespace(get_submodule=lambda path: module)
        manifest = {"entries": [{"task_id": "attention", "candidate_id": "candidate", "paths": ["attn.__core__"]}]}
        catalog = {"tasks": [{"id": "attention", "kind": "attention", "paths": ["attn.__core__"]}]}
        original_dispatch = self.attention["sdpa"]
        collector = []
        def install(*, collector):
            self.attention["sdpa"] = collector
        def forward(model, ids):
            active = self.attention["sdpa"]
            path = module._sako_path + ".__core__"
            tensors = [Tensor(2, (1, 2, 3, 4))]
            key = active.capture(path, "attention", tensors, [1., .5])
            # Deliberately wrong functional output: the original is authoritative.
            active.validate(key, path, Tensor(999, (1, 3, 2, 4)), Tensor(10, (1, 3, 2, 4)))
            return Tensor(10)
        with self.environment(), patch("sako.model.install_attention_dispatch", side_effect=install), \
                patch("sako.model.forward", side_effect=forward):
            result = measure_layer_case(model, manifest, catalog, {"candidate": lambda tensors, scalars: Tensor(10, (1, 2, 3, 4))},
                                        self.config, self.config["workloads"][0], 1729)
        self.assertTrue(result["passed"])
        self.assertIs(self.attention["sdpa"], original_dispatch)
        self.assertFalse(hasattr(module, "_sako_path"))

    def test_empty_manifest_has_vacuous_layer_pass(self):
        with self.environment():
            result = measure_layer_case(Model(), {"entries": []}, {"tasks": []}, {}, self.config,
                                        self.config["workloads"][0], 1729)
        self.assertTrue(result["passed"])
        self.assertEqual(result["expected_paths"], [])
        self.assertEqual(result["layers"], [])

    def test_full_model_speed_has_no_final_numeric_gate_and_uses_all_cases(self):
        model, alternative = Model(), Model()
        manifest, catalog = self.manifest(2)
        pairs, local_calls = [], []
        self.torch.compile = lambda function, **kwargs: function
        def local(model, manifest, catalog, operations, config, workload, seed):
            local_calls.append((workload.copy(), seed))
            return {"seed": seed, "passed": True, "scope": "replaced_layers", "layers": []}
        def timings(baseline, candidate, config, repeats):
            pairs.append((baseline().value, candidate().value, repeats))
            return {"baseline": {"samples_ms": [2.] * 5}, "candidate": {"samples_ms": [1.] * 5}}
        def install(alternative, manifest, catalog, root, *, operations, config=None):
            alternative.modules["layer0"].weight.value = 99.
        original_config = copy.deepcopy(self.config)
        with self.environment(), patch("sako.model.load_model", return_value=model), \
                patch("sako.model.clone_structure", return_value=alternative), \
                patch("sako.model.apply_manifest", side_effect=install), \
                patch("sako.layer_checks.measure_layer_case", side_effect=local), \
                patch("sako.model.paired_timings", side_effect=timings), \
                patch("sako.model.check_output", side_effect=AssertionError("No final-output numerical gate is permitted.")):
            result = measure_model(self.config, manifest, catalog, "unused")
        self.assertEqual(len(local_calls), 6)
        self.assertEqual([seed for _, seed in local_calls], [1729, 2718, 3141] * 2)
        self.assertEqual(len(pairs), 2)
        self.assertTrue(all(a != b for a, b, _ in pairs))
        self.assertTrue(result["decision"]["accepted"])
        self.assertEqual(result["output_equivalence"], "not_tested")
        self.assertEqual(result["correctness_reference"], "eager")
        self.assertEqual(result["model_reference"], "torch_compile")
        self.assertEqual(self.config, original_config)

    def test_failed_local_case_is_not_timed(self):
        model = Model()
        manifest, catalog = self.manifest(2)
        self.torch.compile = lambda function, **kwargs: function
        with self.environment(), patch("sako.model.load_model", return_value=model), \
                patch("sako.model.clone_structure", return_value=Model()), \
                patch("sako.model.apply_manifest"), \
                patch("sako.layer_checks.measure_layer_case", return_value={"passed": False}), \
                patch("sako.model.paired_timings") as timings:
            result = measure_model(self.config, manifest, catalog, "unused")
        timings.assert_not_called()
        self.assertFalse(result["decision"]["accepted"])

    def test_one_failed_seed_skips_all_compilation_and_timing_but_keeps_all_checks(self):
        manifest, catalog = self.manifest(2)
        for failed_workload in self.config["workloads"]:
            with self.subTest(failed_workload=failed_workload):
                local_calls = []
                def local(model, manifest, catalog, operations, config, workload, seed):
                    local_calls.append((workload, seed))
                    passed = workload != failed_workload or seed != 2718
                    return {"seed": seed, "passed": passed, "max_abs_error": 0. if passed else 1.}
                self.torch.compile = Mock(side_effect=AssertionError("Do not compile a rejected model."))
                session = SimpleNamespace(get_model=lambda config: Model(), get_baseline=Mock())
                with self.environment(), patch("sako.model.clone_structure", return_value=Model()), \
                        patch("sako.model.apply_manifest"), \
                        patch("sako.layer_checks.measure_layer_case", side_effect=local), \
                        patch("sako.model.paired_timings") as timings:
                    result = measure_model(self.config, manifest, catalog, "unused", session=session)
                self.assertEqual(len(local_calls), 6)
                self.assertEqual([seed for _, seed in local_calls], self.config["correctness_seeds"] * 2)
                self.torch.compile.assert_not_called()
                session.get_baseline.assert_not_called()
                timings.assert_not_called()
                self.assertFalse(result["compiled"])
                self.assertFalse(result["decision"]["accepted"])
                self.assertIsNone(result["decision"]["speedup"])
                self.assertEqual(sum(not case["correct"] for case in result["cases"]), 1)
                for case in result["cases"]:
                    self.assertEqual(len(case["checks"]), 3)
                    self.assertEqual(case["timing_status"], "not_measured")
                    self.assertIn("required layer check failed", case["timing_reason"])
                    self.assertNotIn("baseline", case)
                    self.assertNotIn("candidate", case)
                self.assertNotIn("invalid timing samples", str(result["decision"]))
                from sako.hypotheses import measured_gpu_outcome
                self.assertTrue(measured_gpu_outcome(result))

    def test_wrong_reference_and_overlapping_paths_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "eager"):
            measure_model({**self.config, "correctness_reference": "torch_compile"}, {}, {}, "unused")
        manifest, catalog = self.manifest(1)
        manifest["entries"][0]["paths"].append("layer0")
        with self.assertRaises(ValueError):
            replacement_paths(manifest, catalog)

    def test_restoration_failure_prevents_all_later_checks_and_all_timing(self):
        model = Model()
        manifest, catalog = self.manifest(2)
        self.torch.compile = lambda function, **kwargs: function
        sequence = [{"passed": True, "reference_state_intact": True}] * 3 + [
            {"passed": False, "reference_state_intact": False}]
        with self.environment(), patch("sako.model.load_model", return_value=model), \
                patch("sako.model.clone_structure", return_value=Model()), \
                patch("sako.model.apply_manifest"), \
                patch("sako.layer_checks.measure_layer_case", side_effect=sequence) as local, \
                patch("sako.model.paired_timings") as timings:
            result = measure_model(self.config, manifest, catalog, "unused")
        self.assertEqual(local.call_count, 4)
        timings.assert_not_called()
        self.assertFalse(result["decision"]["accepted"])
        self.assertFalse(result["reference_state_intact"])
        self.assertTrue(all(case["timing_status"] == "not_measured" for case in result["cases"]))
        self.assertEqual(len(result["cases"][1]["checks"]), 3)
        self.assertTrue(all(check["not_measured"] for check in result["cases"][1]["checks"][1:]))


if __name__ == "__main__":
    unittest.main()
