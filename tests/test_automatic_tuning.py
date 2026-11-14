import ast
import contextlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sako.automatic_tuning import prepare, declared, base_configurations, policy
from sako.core import load_config, write_json
from sako.worker import evaluate_operation
from triton_fixtures import LINEAR_SOURCE, RMS_SOURCE


def candidate(source=LINEAR_SOURCE):
    return {"name": "tuning", "task_id": "linear_1", "backend": "triton", "source": source,
            "strategy": "Keep the algorithm and tune legal launch settings.", "references": []}


class AutomaticTuningTests(unittest.TestCase):
    config = {"tuning": {"enabled": True, "max_configs": 6}}

    def test_injection_keeps_algorithm_and_grid_and_keys_by_constexpr(self):
        original = candidate()
        changed, details = prepare(original, self.config)
        before = ast.parse(original["source"])
        after = ast.parse(changed["source"])
        old_kernel = next(n for n in before.body if isinstance(n, ast.FunctionDef) and n.name == "gemm")
        new_kernel = next(n for n in after.body if isinstance(n, ast.FunctionDef) and n.name == "gemm")
        self.assertEqual([ast.dump(n) for n in old_kernel.body], [ast.dump(n) for n in new_kernel.body])
        self.assertEqual(original["source"], LINEAR_SOURCE)
        kernel = details["kernels"][0]
        self.assertEqual(kernel["configs"][0], {"num_warps": 4, "num_stages": 2})
        self.assertIn("M", kernel["keys"])
        self.assertIn("SW0", kernel["keys"])
        self.assertGreater(len(kernel["configs"]), 1)
        self.assertEqual(declared(changed), details)

    def test_disabled_policy_keeps_exact_source(self):
        original = candidate()
        self.assertIs(prepare(original, {})[0], original)
        self.assertIsNone(declared(original))

    def test_tuning_supports_valid_import_aliases(self):
        source = LINEAR_SOURCE.replace("import triton\n", "import triton as tr\n").replace("@triton.jit", "@tr.jit").replace("triton.cdiv", "tr.cdiv")
        changed, details = prepare(candidate(source), self.config)
        self.assertIn("@tr.autotune", changed["source"])
        self.assertEqual(details["kernels"][0]["kind"], "harness_launch_configs")

    def test_dynamic_launch_is_retained_with_an_explicit_skip_reason(self):
        source = LINEAR_SOURCE.replace("num_warps=4", "num_warps=(4 if m > 1 else 2)")
        changed, details = prepare(candidate(source), self.config)
        self.assertEqual(details["kernels"][0]["status"], "skipped")
        self.assertNotIn("@triton.autotune", changed["source"])
        self.assertIn("num_warps=4 if m > 1 else 2", changed["source"])

    def test_writer_block_choices_are_bounded_and_preserved(self):
        decorator = "@triton.autotune(configs=[triton.Config({}, num_warps=2), triton.Config({}, num_warps=4)], key=['M', 'N', 'K'])\n"
        source = LINEAR_SOURCE.replace("@triton.jit", decorator + "@triton.jit")
        changed, details = prepare(candidate(source), self.config)
        self.assertEqual(details["kernels"][0]["kind"], "writer_block_configs")
        self.assertEqual(changed["source"].count("@triton.autotune"), 1)

    def test_base_context_restores_cache_and_configs_even_after_failure(self):
        configs, cache = ["base", "alternative"], {"shape": "alternative"}
        tuner = SimpleNamespace(configs=configs, cache=cache)
        with self.assertRaisesRegex(RuntimeError, "failed"):
            with base_configurations(lambda: [("kernel", tuner)]):
                self.assertEqual(tuner.configs, ["base"])
                self.assertEqual(tuner.cache, {})
                raise RuntimeError("failed")
        self.assertIs(tuner.configs, configs)
        self.assertIs(tuner.cache, cache)

    def test_tuning_limits_do_not_change_acceptance_limits(self):
        for change in ({"max_configs": 17}, {"trials": 1}, {"repeats": 0}, {"warmup": False}):
            with self.assertRaises(ValueError):
                policy({"tuning": {"enabled": True, **change}})
        config = load_config(Path(__file__).resolve().parents[1] / "configs/generation.json")
        self.assertEqual(config["trials"], 15)
        self.assertEqual(config["agent"]["max_total_tokens"], 1000000)
        self.assertEqual(config["min_speedup"], 1.0)


class TuningCorrectnessGateTests(unittest.TestCase):
    def evaluate(self, initial_correct):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        config = load_config(Path(__file__).resolve().parents[1] / "configs/validation.json")
        config.update(tuning={"enabled": True}, supplemental_correctness={"enabled": False}, kernel_search={"enabled": False})
        item, _ = prepare(candidate(RMS_SOURCE), config)
        path = directory / "candidate.json"
        write_json(path, item)
        job = {"candidate": str(path), "profile_dir": str(directory)}
        catalog = {"tasks": [{"id": "linear_1", "kind": "rmsnorm", "cases": [{}], "paths": ["norm"]}]}
        cases = [{"correct": initial_correct, "inputs_unchanged": True}]
        timed = [{"correct": True, "baseline": {"samples_ms": [2.] * config["trials"]},
                  "candidate": {"samples_ms": [1.] * config["trials"]}}]
        function = SimpleNamespace(base_configurations=contextlib.nullcontext)
        with patch("sako.gpu.compile_candidate", return_value=function), \
                patch("sako.gpu.evaluate_fixture", side_effect=[cases, timed]) as evaluator:
            result = evaluate_operation(job, config, catalog, fixture_only=True)
        return result, evaluator

    def test_wrong_initial_output_prevents_tuning_and_timing(self):
        result, evaluator = self.evaluate(False)
        self.assertEqual(evaluator.call_count, 1)
        self.assertTrue(evaluator.call_args.kwargs["numerical_only"])
        self.assertFalse(result["decision"]["accepted"])
        self.assertEqual(result["automatic_tuning"]["status"], "blocked_by_initial_correctness")

    def test_correct_initial_output_still_requires_independent_acceptance_pairs(self):
        result, evaluator = self.evaluate(True)
        self.assertEqual(evaluator.call_count, 2)
        self.assertTrue(evaluator.call_args_list[0].kwargs["numerical_only"])
        self.assertNotIn("numerical_only", evaluator.call_args_list[1].kwargs)
        self.assertTrue(result["decision"]["accepted"])
        self.assertTrue(result["automatic_tuning"]["initial_correctness"]["cases"][0]["correct"])


class BackendFailureReportTests(unittest.TestCase):
    def test_worker_error_keeps_its_traceback_and_enters_the_correction_path(self):
        import test_swarm_loop as fixtures
        fixtures.SwarmLoopTests.setUp(self)
        loop = fixtures.SwarmLoopTests.loop(self)
        loop.start_task()
        plan = loop.coordinator(1)
        assignment = loop.assignment(plan["hypotheses"][0], 0, plan)
        assignment["candidate"] = candidate(RMS_SOURCE)
        assignment["candidate"]["task_id"] = assignment["task_id"]
        with patch.object(self.worker, "run", side_effect=RuntimeError("test compile failure")):
            loop.process_candidate(assignment)
        result = loop.store.candidate(assignment["candidate_id"])["result"]
        self.assertIn("test compile failure", result["error"])
        self.assertIn("RuntimeError: test compile failure", result["traceback"])
        self.assertFalse(result["decision"]["accepted"])
