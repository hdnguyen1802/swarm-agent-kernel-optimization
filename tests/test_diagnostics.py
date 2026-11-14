from triton_fixtures import RMS_SOURCE
import contextlib
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sako.core import candidate_id, read_json, write_json
from sako.diagnostics import BoundaryTrace, diagnose_model, diagnostic_case, trace_paths


ROOT = Path(__file__).resolve().parents[1]


class Tensor:
    shape, dtype, device = (1,), "float16", "cuda"
    def __init__(self, value): self.value = value
    def stride(self): return (1,)
    def detach(self): return self
    def cpu(self): return Tensor(self.value)
    def to(self, device): return Tensor(self.value)


def comparison(actual, expected, config):
    difference = abs(actual.value - expected.value)
    return {"passed": difference <= config["atol"] + config["rtol"] * abs(expected.value),
            "max_abs_error": difference, "exact_equal": actual.value == expected.value,
            "atol": config["atol"], "rtol": config["rtol"]}


class Module:
    def __init__(self, function):
        self.function, self.hooks = function, []
    def __call__(self, tensor):
        output = Tensor(self.function(tensor.value))
        for hook in tuple(self.hooks):
            hook(self, (tensor,), output)
        return output
    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        return SimpleNamespace(remove=lambda: self.hooks.remove(hook))


class Model:
    def __init__(self):
        self.modules = {"layer0": Module(lambda x: x + 1), "layer1": Module(lambda x: x * 10)}
        self.weights = (1., 10.)
    def get_submodule(self, path): return self.modules[path]
    def __call__(self, tensor):
        for module in self.modules.values():
            tensor = module(tensor)
        return tensor


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "runs").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=ROOT / "runs")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.config = {"model_id": "fixed-model", "revision": "main", "dtype": "float16",
                       "compile_mode": "default", "atol": .05, "rtol": .0001,
                       "workloads": [{"batch_size": 1, "sequence_length": 128}, {"batch_size": 1, "sequence_length": 512}],
                       "correctness_seeds": [1729, 2718]}
        self.source = {"name": "test", "task_id": "linear_0", "strategy": "Diagnostic test.",
                       "backend": "triton", "references": [], "source": RMS_SOURCE}
        self.key = candidate_id(self.source)
        write_json(self.directory / (self.key + ".json"), self.source)
        self.manifest = {"entries": [{"task_id": "linear_0", "candidate_id": self.key, "paths": ["layer0", "layer1"]}]}
        self.catalog = {"config": {"revision": "fixed-revision"}, "hardware": {"fingerprint": "fixed-gpu"},
                        "tasks": [{"id": "linear_0", "kind": "linear", "paths": ["layer0", "layer1"]}]}

    def test_selects_existing_case_without_changing_config_or_limits(self):
        before = copy.deepcopy(self.config)
        case = diagnostic_case(self.config)
        self.assertEqual(case["seed"], 1729)
        self.assertEqual(case["workload"], self.config["workloads"][0])
        case["workload"]["batch_size"] = 7
        self.assertEqual(self.config, before)
        for settings in ({"workload_index": 2}, {"seed_index": -1}, {"max_paths": 257}):
            with self.assertRaises(ValueError):
                diagnostic_case({**self.config, "diagnostics": settings})

    def test_trace_separates_accumulated_error_from_local_kernel_error(self):
        trace = BoundaryTrace(self.config)
        trace.capture("layer0", None, (Tensor(1),), Tensor(2))
        trace.capture("layer1", None, (Tensor(2),), Tensor(20))
        with patch("sako.diagnostics.compare_outputs", side_effect=comparison):
            trace.compare("layer0", self.key, SimpleNamespace(original=Module(lambda x: x + 1)), (Tensor(1),), Tensor(2.01))
            trace.compare("layer1", self.key, SimpleNamespace(original=Module(lambda x: x * 10)), (Tensor(2.01),), Tensor(20.1))
        summary = trace.summary()
        self.assertEqual(summary["first_exact_difference"]["path"], "layer0")
        self.assertEqual(summary["first_tolerance_failure"]["path"], "layer1")
        self.assertIsNone(summary["first_same_input_failure"])
        self.assertIn("Compiled internal boundaries were not observed", summary["scope"])

    def test_repeated_calls_match_in_execution_order_and_report_missing_boundary(self):
        trace = BoundaryTrace(self.config)
        trace.capture("layer0", None, (Tensor(1),), Tensor(2))
        trace.capture("layer0", None, (Tensor(2),), Tensor(3))
        with patch("sako.diagnostics.compare_outputs", side_effect=comparison):
            trace.compare("layer0", self.key, SimpleNamespace(original=Module(lambda x: x + 1)), (Tensor(1),), Tensor(2))
        self.assertEqual(trace.summary()["unmatched_boundaries"], [
            {"path": "layer0", "reference_calls": 2, "candidate_calls": 1}])
        self.assertIsNone(trace.summary()["first_exact_difference"])

    def test_path_limit_and_attention_scope_are_explicit(self):
        selected, skipped = trace_paths(self.manifest, self.catalog, self.directory, 1)
        self.assertEqual(len(selected), 1)
        self.assertEqual(skipped[0]["path"], "layer1")
        self.catalog["tasks"][0]["kind"] = "attention"
        selected, skipped = trace_paths(self.manifest, self.catalog, self.directory, 128)
        self.assertEqual(selected, [])
        self.assertEqual(len(skipped), 2)

    def test_changed_candidate_identity_is_rejected_before_model_execution(self):
        changed = {**self.source, "source": RMS_SOURCE + '# changed */ }'}
        write_json(self.directory / (self.key + ".json"), changed)
        with self.assertRaisesRegex(ValueError, "identity"):
            trace_paths(self.manifest, self.catalog, self.directory, 128)

    def test_complete_diagnostic_keeps_weights_limits_and_acceptance_separate(self):
        reference, alternative = Model(), Model()
        initial_weights = (reference.weights, alternative.weights)
        original_config = copy.deepcopy(self.config)
        original_manifest = copy.deepcopy(self.manifest)
        events = []
        attention = {"sdpa": object()}
        attention_before = attention["sdpa"]
        def apply(model, *args):
            for index, path in enumerate(("layer0", "layer1")):
                original = model.modules[path]
                replacement = Module((lambda x: x + 1.01) if index == 0 else (lambda x: x * 10))
                replacement.original = original
                model.modules[path] = replacement
            attention["sdpa"] = "temporary-dispatch"
        def compile(function, **settings):
            self.assertEqual(settings, {"fullgraph": True, "dynamic": False, "mode": "default"})
            def call(tensor):
                self.assertTrue(all(not module.hooks for model in (reference, alternative) for module in model.modules.values()))
                events.append("compiled")
                return function(tensor)
            return call
        torch = SimpleNamespace(compile=compile, inference_mode=contextlib.nullcontext,
                                cuda=SimpleNamespace(synchronize=lambda: None))
        with patch.dict("sys.modules", {"torch": torch, "transformers.modeling_utils": SimpleNamespace(ALL_ATTENTION_FUNCTIONS=attention)}), \
                patch("sako.model.load_model", return_value=reference) as load, \
                patch("sako.model.clone_structure", return_value=alternative), \
                patch("sako.model.apply_manifest", side_effect=apply), \
                patch("sako.model.forward", side_effect=lambda model, ids: model(ids)), \
                patch("sako.model.inputs", return_value=Tensor(1)) as inputs, \
                patch("sako.diagnostics.compare_outputs", side_effect=comparison):
            result = diagnose_model(self.config, self.manifest, self.catalog, self.directory, self.directory / "diagnostic")
        self.assertEqual(result["status"], "complete")
        self.assertEqual(events, ["compiled", "compiled"])
        self.assertFalse(result["acceptance_evidence"])
        self.assertFalse(result["timed"])
        self.assertNotIn("decision", result)
        self.assertFalse(result["comparisons"]["compiled_candidate_vs_compiled_reference"]["passed"])
        self.assertEqual(result["boundary_summary"]["first_tolerance_failure"]["path"], "layer1")
        self.assertEqual(load.call_args.args[0]["revision"], "fixed-revision")
        inputs.assert_called_once_with(reference, self.config["workloads"][0], 1729)
        self.assertEqual(self.config, original_config)
        self.assertEqual(self.manifest, original_manifest)
        self.assertEqual((reference.weights, alternative.weights), initial_weights)
        self.assertIs(attention["sdpa"], attention_before)
        self.assertTrue(all(not module.hooks for model in (reference, alternative) for module in model.modules.values()))
        raw = next(artifact for artifact in result["artifacts"] if artifact["path"].endswith("boundary_checks.json"))
        self.assertEqual(len(read_json(ROOT / raw["path"])), 2)


if __name__ == "__main__":
    unittest.main()
