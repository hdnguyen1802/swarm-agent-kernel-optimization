import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from sako.core import load_config
from sako.generation import input_record, paired_generation, prompt_tokens, sampled_steps, step_info
from sako.preparation import profile_settings
from sako.reporting import build_swarm_report, render_swarm_report
from sako.trial_evidence import measured_opportunity, model_opportunity


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config("configs/generation.json")
        self.workload = self.config["workloads"][0]

    def test_checkpoint_and_workload_determine_profile_identity(self):
        changed = {**self.config, "model_id": "org/another-causal-lm", "revision": "other"}
        with TemporaryDirectory() as folder:
            file = Path(folder) / "model.json"
            file.write_text(json.dumps(changed), encoding="utf-8")
            self.assertEqual(load_config(file)["model_id"], changed["model_id"])
        self.assertNotEqual(profile_settings(changed), profile_settings(self.config))
        self.assertNotEqual(profile_settings(load_config("configs/project.json")), profile_settings(self.config))

    def test_record_uses_selected_tokenizer_revision_and_prompt(self):
        from types import SimpleNamespace
        factory = Mock()
        factory.from_pretrained.side_effect = [lambda *a, **k: {"input_ids": [4, 7]},
                                               lambda *a, **k: {"input_ids": [9]}]
        prompt_tokens.cache_clear()
        with patch.dict("sys.modules", {"transformers": SimpleNamespace(AutoTokenizer=factory)}):
            for model, expected in (("org/a", [4, 7]), ("org/b", [9])):
                cfg = {**self.config, "model_id": model, "revision": "fixed"}
                record = input_record(cfg, self.workload)
                self.assertEqual(record["prompt_token_ids"], expected)
                self.assertEqual(record["prompt_tokens"], len(expected))
                self.assertEqual(record["decode_steps"], 127)
                factory.from_pretrained.assert_called_with(model, revision="fixed", trust_remote_code=False)
        prompt_tokens.cache_clear()

    def test_capture_phases_and_logical_cache_lengths(self):
        self.assertEqual(sampled_steps(self.workload), [0, 1, 127])
        self.assertEqual(sampled_steps({"new_tokens": 2}), [0, 1])
        lengths = [step_info(step, 17) for step in sampled_steps(self.workload)]
        self.assertEqual([x["phase"] for x in lengths], ["prefill", "decode", "decode"])
        self.assertEqual([x["cache_length_before"] for x in lengths], [0, 17, 143])
        self.assertEqual([x["cache_length_after"] for x in lengths], [17, 18, 144])

    def test_keep_gate_uses_total_time_and_retains_each_metric(self):
        order = []
        class Runner:
            record = {"prompt_tokens": 19}
            def __init__(self, name, total):
                self.name, self.total = name, total
            def run(self, timed=False):
                if timed:
                    order.append(self.name)
                return {"first_token_ms": 5., "mean_decode_ms": 0.5, "total_ms": self.total}
        result = paired_generation(Runner("a", 100.), Runner("b", 90.), {"warmup": 1, "trials": 3})
        self.assertEqual(order, list("abbaab"))
        self.assertEqual(result["baseline"]["samples_ms"], [100.] * 3)
        self.assertEqual(result["candidate"]["samples_ms"], [90.] * 3)
        self.assertEqual(result["generation_metrics"]["candidate"]["mean_decode_ms"]["samples_ms"], [0.5] * 3)

    def test_prefill_gain_only_uses_the_measured_prefill_cost(self):
        task = {"workload_medians_ms": [100.], "compiled_evidence": {
            "generation_phase_ms": {"prefill": [4.], "decode": [96.]}}}
        costs = model_opportunity(task, [self.workload], [{"workload": self.workload, "median_ms": 200.}])
        case = {"workload": self.workload, "correct": True, "search_reference": {"kind": "paired_compile",
                "baseline": {"samples_ms": [2.]}, "candidate": {"samples_ms": [1.]}}}
        self.assertEqual(measured_opportunity(costs, {"cases": [case]}), [])
        case["generation_step"] = step_info(0, 19)
        row, = measured_opportunity(costs, {"cases": [case]})
        self.assertEqual(row["estimated_model_saving_ms"], 2.)

    def test_report_shows_profile_baseline_with_no_accepted_candidate(self):
        metrics = {k: {"median_ms": v} for k, v in (("first_token_ms", 7.5), ("mean_decode_ms", 4.),
                                                   ("total_ms", 515.5))}
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "profile").mkdir()
            (root / "profile/tasks.json").write_text(json.dumps({"compiled_profile": {
                "workload_times": [{"workload": self.workload, "generation_metrics": metrics}]}}))
            report = build_swarm_report(root, self.config)
            historical = build_swarm_report(root, {k:v for k,v in self.config.items() if k != "generation_objective"})
        self.assertFalse(report["accepted_model_gain"])
        self.assertIn("4.000 ms/token (250.00 tokens/s)", render_swarm_report(report))
        self.assertNotIn("515.500", render_swarm_report(report))
        self.assertIn("| Baseline | torch.compile | 7.500 | 4.000 | 515.500 |", render_swarm_report(historical))
