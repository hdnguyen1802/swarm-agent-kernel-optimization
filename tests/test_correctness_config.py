"""Bind declared correctness scope to config validation and saved profiles."""

import copy
from pathlib import Path
import tempfile
import unittest

from sako.core import correctness_settings, digest, load_config, read_json, write_json
from sako.preparation import PROFILE_FIELDS, profile_settings, validate_catalog


ROOT = Path(__file__).resolve().parents[1]


class CorrectnessConfiguration(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "config.json"
        self.project = read_json(ROOT / "configs/project.json")
        self.legacy = read_json(ROOT / "configs/validation.json")

    def load(self, config):
        write_json(self.path, config)
        return load_config(self.path)

    def test_default_keeps_numerical_limits_and_uses_the_new_model_speed_gate(self):
        config = self.load(self.project)
        self.assertEqual(correctness_settings(config), {
            "correctness_scope": "replaced_layers", "correctness_reference": "eager"})
        self.assertEqual(config["model_reference"], "torch_compile")
        self.assertEqual(config["min_speedup"], 1.0)
        self.assertEqual(config["agent"]["max_total_tokens"], 1_000_000)
        self.assertEqual(config["agent"]["max_correction_attempts"], 2)
        self.assertEqual(config["workloads"], self.legacy["workloads"][:1])
        for key in ("atol", "rtol", "dtype", "correctness_seeds", "warmup", "trials",
                    "repeats", "fullgraph", "max_p95_ratio"):
            self.assertEqual(config[key], self.legacy[key], key)

    def test_missing_keys_keep_legacy_semantics_without_mutating_saved_config(self):
        config = self.load(self.legacy)
        self.assertNotIn("correctness_scope", config)
        self.assertNotIn("correctness_reference", config)
        self.assertEqual(correctness_settings(config), {
            "correctness_scope": "full_model", "correctness_reference": "torch_compile"})
        self.assertEqual(config["agent"]["max_repair_attempts"], 3)
        self.assertEqual(config["agent"]["max_correction_attempts"], 2)
        explicit = {**config, "correctness_scope": "full_model", "correctness_reference": "torch_compile"}
        self.assertEqual(profile_settings(config), profile_settings(explicit))

    def test_unknown_or_mixed_correctness_policies_are_rejected(self):
        for scope, reference in (("replaced_layers", "torch_compile"), ("full_model", "eager"),
                                 ("unknown", "eager"), ("replaced_layers", "unknown"),
                                 (None, "eager"), ([], "eager")):
            with self.subTest(scope=scope, reference=reference):
                changed = {**self.project, "correctness_scope": scope, "correctness_reference": reference}
                with self.assertRaisesRegex(ValueError, "correctness"):
                    self.load(changed)
        with self.assertRaisesRegex(ValueError, "model reference"):
            self.load({**self.project, "model_reference": "eager"})

    def test_partial_new_policy_does_not_silently_change_legacy_reference(self):
        changed = copy.deepcopy(self.project)
        del changed["correctness_reference"]
        with self.assertRaises(ValueError):
            self.load(changed)
        changed = copy.deepcopy(self.project)
        del changed["correctness_scope"]
        with self.assertRaises(ValueError):
            self.load(changed)

    def test_swarm_correction_limit_is_two_and_legacy_repair_limit_stays_separate(self):
        missing = copy.deepcopy(self.project)
        del missing["agent"]["max_correction_attempts"]
        self.assertEqual(self.load(missing)["agent"]["max_correction_attempts"], 2)
        for value in (0, 1, 3, True, 2., "2", None):
            changed = copy.deepcopy(self.project)
            changed["agent"]["max_correction_attempts"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "correction"):
                self.load(changed)
        self.assertEqual(self.load(self.legacy)["agent"]["max_repair_attempts"], 3)

    def test_profile_hash_binds_correctness_scope_and_reference(self):
        self.assertIn("correctness_scope", PROFILE_FIELDS)
        self.assertIn("correctness_reference", PROFILE_FIELDS)
        legacy = self.load(self.legacy)
        changed = {**legacy, "correctness_scope": "replaced_layers", "correctness_reference": "eager"}
        self.assertNotEqual(digest(profile_settings(legacy)), digest(profile_settings(changed)))
        hardware = {"fingerprint": "same-device-and-software"}
        catalog = {"hardware": hardware, "config": copy.deepcopy(legacy)}
        validate_catalog(legacy, catalog, hardware)
        with self.assertRaisesRegex(ValueError, "correctness_scope"):
            validate_catalog(changed, catalog, hardware)
        explicit_legacy = {**legacy, "correctness_scope": "full_model", "correctness_reference": "torch_compile"}
        validate_catalog(explicit_legacy, catalog, hardware)


if __name__ == "__main__":
    unittest.main()
