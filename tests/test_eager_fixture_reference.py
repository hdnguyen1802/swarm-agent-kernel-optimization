import contextlib
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sako.gpu import evaluate_fixture
from sako.model import Collector


class Tensor:
    dtype, device = "float16", "cuda"
    def __init__(self, value, shape=(1, 2)):
        self.value, self.shape = value, shape
    def detach(self): return self
    def cpu(self): return self.clone()
    def cuda(self): return self.clone()
    def clone(self): return copy.deepcopy(self)
    def stride(self): return tuple(1 for _ in self.shape)
    def transpose(self, left, right):
        shape = list(self.shape)
        shape[left], shape[right] = shape[right], shape[left]
        return Tensor(self.value, tuple(shape))
    def contiguous(self): return self


def compare(actual, expected, config):
    difference = abs(actual.value - expected.value)
    return {"passed": actual.shape == expected.shape and difference <= config["atol"] + config["rtol"] * abs(expected.value),
            "max_abs_error": difference, "atol": config["atol"], "rtol": config["rtol"]}


class EagerFixtureReferenceTests(unittest.TestCase):
    def setUp(self):
        self.config = {"max_fixture_cases": 2, "atol": .03, "rtol": .02, "compile_mode": "default",
                       "correctness_scope": "replaced_layers", "correctness_reference": "eager"}
        self.context = {"seed": 1729, "workload": {"batch_size": 1, "sequence_length": 128}}
        self.torch = SimpleNamespace(Tensor=Tensor, inference_mode=contextlib.nullcontext,
                                     compile=lambda function, **kwargs: function,
                                     equal=lambda left, right: left.value == right.value and left.shape == right.shape)

    def test_capture_saves_original_output_and_does_not_overwrite_from_other_layer(self):
        collector = Collector(self.config)
        collector.context = copy.deepcopy(self.context)
        tensors = [Tensor(3), Tensor(2), Tensor(0)]
        expected = Tensor(6)
        with patch.dict("sys.modules", {"torch": self.torch}), patch("sako.model.check_output", side_effect=compare):
            key = collector.capture("layer0", "linear", tensors, [])
            collector.validate(key, "layer0", Tensor(6.001), expected)
            expected.value = 999
            other = collector.capture("layer1", "linear", [Tensor(3), Tensor(7), Tensor(0)], [])
            collector.validate(other, "layer1", Tensor(21), Tensor(21))
        self.assertEqual(key, other)
        self.assertEqual(len(collector.cases[key]), 1)
        saved = collector.cases[key][0]
        self.assertEqual(saved["eager_output"].value, 6)
        self.assertEqual(saved["eager_output_path"], "layer0")
        self.assertEqual(saved["eager_output_context"], self.context)
        self.assertEqual(saved["source_path"], "layer0")

    def test_context_and_case_index_bind_each_output_to_its_inputs(self):
        collector = Collector(self.config)
        with patch.dict("sys.modules", {"torch": self.torch}), patch("sako.model.check_output", side_effect=compare):
            for seed, value in ((1729, 4), (2718, 7)):
                collector.context = {**copy.deepcopy(self.context), "seed": seed}
                key = collector.capture("layer0", "linear", [Tensor(value), Tensor(2)], [])
                collector.validate(key, "layer0", Tensor(value * 2), Tensor(value * 2))
        cases = collector.cases[key]
        self.assertEqual([case["eager_output"].value for case in cases], [8, 14])
        self.assertEqual([case["eager_output_context"]["seed"] for case in cases], [1729, 2718])
        self.assertEqual([case["eager_output_case_index"] for case in cases], [0, 1])

    def test_changed_context_cannot_attach_an_unrelated_output(self):
        collector = Collector(self.config)
        collector.context = copy.deepcopy(self.context)
        with patch.dict("sys.modules", {"torch": self.torch}), patch("sako.model.check_output", side_effect=compare):
            key = collector.capture("layer0", "linear", [Tensor(3), Tensor(2)], [])
            collector.context["seed"] = 2718
            with self.assertRaisesRegex(ValueError, "context"):
                collector.validate(key, "layer0", Tensor(6), Tensor(6))
        self.assertNotIn("eager_output", collector.cases[key][0])

    def test_attention_original_output_is_saved_in_candidate_api_layout(self):
        collector = Collector(self.config)
        collector.context = copy.deepcopy(self.context)
        tensors = [Tensor(1, (1, 2, 3, 4)) for _ in range(4)]
        with patch.dict("sys.modules", {"torch": self.torch}), patch("sako.model.check_output", side_effect=compare):
            key = collector.capture("attention.__core__", "attention", tensors, [1., .5])
            collector.validate(key, "attention.__core__", Tensor(8, (1, 3, 2, 4)), Tensor(8, (1, 3, 2, 4)))
        self.assertEqual(collector.cases[key][0]["eager_output"].shape, (1, 2, 3, 4))
        self.assertEqual(collector.cases[key][0]["eager_output"].value, 8)

    def fixture(self):
        return {"cases": [{"tensors": [Tensor(1)], "scalars": [], "case_index": 0,
                           **copy.deepcopy(self.context), "source_path": "layer0",
                           "eager_output": Tensor(20), "eager_output_reference": "original_model",
                           "eager_output_path": "layer0", "eager_output_case_index": 0,
                           "eager_output_context": copy.deepcopy(self.context)}]}

    def evaluate(self, fixture, config, **options):
        self.torch.load = lambda *args, **kwargs: fixture
        stack = contextlib.ExitStack()
        stack.enter_context(patch.dict("sys.modules", {"torch": self.torch}))
        stack.enter_context(patch("sako.compat.configure_compiler"))
        self.resources = stack.enter_context(patch("sako.gpu.compiler_resources", return_value={}))
        self.hardware = {"support": {"supported": True}}
        self.discovery = stack.enter_context(patch("sako.gpu.discover_hardware", return_value=self.hardware))
        stack.enter_context(patch("sako.gpu.reference", return_value=Tensor(99)))
        stack.enter_context(patch("sako.gpu.check_output", side_effect=compare))
        self.addCleanup(stack.close)
        return evaluate_fixture({"name": "candidate"}, {"kind": "linear", "paths": ["layer0"]}, "unused.pt", config, **options)

    def test_one_hardware_snapshot_serves_candidate_baseline_and_resources(self):
        with patch("sako.gpu.custom_op", return_value=lambda tensors, scalars: Tensor(20)) as build, \
                patch("sako.gpu.paired_timings", return_value={}):
            result = self.evaluate(self.fixture(), self.config, baseline_candidate={"name": "kept"})
        self.assertTrue(result[0]["correct"])
        self.discovery.assert_called_once_with(self.config)
        self.assertEqual(build.call_count, 2)
        self.assertTrue(all(call.kwargs["hardware"] is self.hardware for call in build.call_args_list))
        self.assertIs(self.resources.call_args.args[1], self.hardware)

    def test_kept_kernel_does_not_replace_the_original_eager_accuracy_reference(self):
        for baseline_value in (20, 99):
            calls = [lambda tensors, scalars: Tensor(20), lambda tensors, scalars: Tensor(baseline_value)]
            with patch("sako.gpu.custom_op", side_effect=calls), \
                    patch("sako.gpu.paired_timings", return_value={}) as timing:
                result = self.evaluate(self.fixture(), self.config, baseline_candidate={"name": "kept"})
            self.assertEqual(result[0]["correct"], baseline_value == 20)
            self.assertEqual(result[0]["timing_reference"], "kept_kernel")
            self.assertEqual(result[0]["reference"], "eager_original_model")
            self.assertEqual(result[0]["kept_baseline_check"]["passed"], baseline_value == 20)
            self.assertEqual(timing.call_count, int(baseline_value == 20))

    def test_compiled_baseline_difference_is_only_a_diagnostic_in_new_mode(self):
        calls = []
        def operation(tensors, scalars):
            calls.append(True)
            return Tensor(20)
        with patch("sako.gpu.custom_op", return_value=operation), \
                patch("sako.gpu.paired_timings", return_value={}) as timing:
            result = self.evaluate(self.fixture(), self.config)
        self.assertTrue(result[0]["correct"])
        self.assertEqual(len(result[0]["checks"]), 3)
        self.assertEqual(len(calls), 3)
        self.assertFalse(result[0]["compiled_vs_eager_diagnostic"]["passed"])
        self.assertEqual(result[0]["reference"], "eager_original_model")
        timing.assert_called_once()

    def test_saved_eager_output_is_required_and_context_mismatch_is_rejected(self):
        for change in (lambda case: case.pop("eager_output"),
                       lambda case: case.update(eager_output_path="layer1"),
                       lambda case: case.update(eager_output_context={**self.context, "seed": 2718}),
                       lambda case: case.update(eager_output_case_index=1)):
            fixture = self.fixture()
            change(fixture["cases"][0])
            with patch("sako.gpu.custom_op") as compile_candidate:
                with self.assertRaisesRegex(ValueError, "new profile"):
                    self.evaluate(fixture, self.config)
                compile_candidate.assert_not_called()

    def test_legacy_mode_uses_functional_output_and_keeps_four_checks(self):
        config = {key: value for key, value in self.config.items() if key not in {"correctness_scope", "correctness_reference"}}
        with patch("sako.gpu.custom_op", return_value=lambda tensors, scalars: Tensor(20)), \
                patch("sako.gpu.paired_timings") as timing:
            result = self.evaluate(self.fixture(), config)
        self.assertFalse(result[0]["correct"])
        self.assertEqual(len(result[0]["checks"]), 4)
        self.assertTrue(result[0]["checks"][0]["passed"])
        self.assertNotIn("compiled_vs_eager_diagnostic", result[0])
        timing.assert_not_called()

    def test_first_trial_times_the_original_once(self):
        config = {**self.config, "kernel_search": {"score_reference": "paired_compile"}}
        samples = {"baseline": {"samples_ms": [10.]}, "candidate": {"samples_ms": [8.]}}
        with patch("sako.gpu.custom_op", return_value=lambda tensors, scalars: Tensor(20)), \
                patch("sako.gpu.paired_timings", return_value=samples) as timing:
            result = self.evaluate(self.fixture(), config)
        timing.assert_called_once()
        self.assertEqual(result[0]["timing_reference"], "torch_compile")
        self.assertEqual(result[0]["baseline"], samples["baseline"])
        self.assertEqual(result[0]["search_reference"]["baseline"], samples["baseline"])
        self.assertEqual(result[0]["search_reference"]["candidate"], samples["candidate"])
        self.assertFalse(result[0]["search_reference"]["acceptance_evidence"])

    def test_fixed_anchor_config_keeps_one_timing_pair(self):
        config = {**self.config, "kernel_search": {"score_reference": "fixed_anchor"}}
        samples = {"baseline": {"samples_ms": [10.]}, "candidate": {"samples_ms": [8.]}}
        with patch("sako.gpu.custom_op", return_value=lambda tensors, scalars: Tensor(20)), \
                patch("sako.gpu.paired_timings", return_value=samples) as timing, \
                patch.object(self.torch, "compile") as compile_reference:
            result = self.evaluate(self.fixture(), config, baseline_candidate={"name": "kept"})
        timing.assert_called_once()
        compile_reference.assert_not_called()
        self.assertEqual(result[0]["timing_reference"], "kept_kernel")
        self.assertEqual(result[0]["baseline"]["samples_ms"], [10.])
        self.assertEqual(result[0]["candidate"]["samples_ms"], [8.])
        self.assertNotIn("search_reference", result[0])

    def test_paired_compile_ranking_uses_a_separate_original_pair_after_a_keep(self):
        for mode in ("events", "cuda_graph"):
            with self.subTest(mode=mode):
                config = {**self.config, "kernel_search": {"score_reference": "paired_compile", "timing_mode": mode}}
                primary = {"baseline": {"samples_ms": [7.]}, "candidate": {"samples_ms": [8.]}}
                ranking = {"baseline": {"samples_ms": [12.]}, "candidate": {"samples_ms": [8.5]}}
                with patch("sako.gpu.custom_op", return_value=lambda tensors, scalars: Tensor(20)), \
                        patch("sako.gpu.paired_timings", side_effect=[primary, ranking]) as timing:
                    result = self.evaluate(self.fixture(), config, baseline_candidate={"name": "kept"})
                self.assertEqual(timing.call_count, 2)
                self.assertEqual(result[0]["baseline"], primary["baseline"])
                self.assertEqual(result[0]["candidate"], primary["candidate"])
                ref = result[0]["search_reference"]
                self.assertEqual(ref["baseline"], ranking["baseline"])
                self.assertEqual(ref["candidate"], ranking["candidate"])
                self.assertEqual(ref["kind"], "paired_compile")
                self.assertEqual(ref["timing_mode"], mode)
                self.assertFalse(ref["acceptance_evidence"])
                self.assertEqual(timing.call_args_list[1].kwargs, {"mode": mode} if mode != "events" else {})


if __name__ == "__main__":
    unittest.main()
