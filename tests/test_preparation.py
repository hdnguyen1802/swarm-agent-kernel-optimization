"""Preparation must bind the profile to its device, source, and test contract."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sako.core import load_config, read_json, write_json
from sako.preparation import prepare, validate_catalog

ROOT = Path(__file__).resolve().parents[1]


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / "run"
        self.profile = self.run / "profile"
        self.config = load_config(ROOT / "configs/validation.json")
        self.config["revision"] = "main"
        self.hardware = {"fingerprint": "device-a", "target": "sm_70"}
        self.catalog = {"config": copy.deepcopy(self.config), "hardware": self.hardware,
                        "selected_task_ids": ["linear_1"], "tasks": [{"id": "linear_1"}]}
        self.catalog["config"]["revision"] = "a" * 40
        self.events = []
        outer = self

        class Worker:
            root = outer.root

            def run(self, action, directory, **kwargs):
                outer.events.append(action)
                if action == "doctor":
                    return {"hardware": outer.hardware}
                if action == "profile":
                    return copy.deepcopy(outer.catalog)
                raise AssertionError(action)

        self.worker = Worker()

    def test_device_check_precedes_profile_and_resume_keeps_catalog(self):
        catalog = prepare(self.config, self.run, self.profile, self.worker)
        self.assertEqual(self.events, ["doctor", "profile"])
        self.assertEqual(catalog["selected_task_ids"], ["linear_1"])
        self.events.clear()
        self.assertEqual(prepare(self.config, self.run, self.profile, self.worker), catalog)
        self.assertEqual(self.events, ["doctor"])

    def test_changed_gpu_cannot_reuse_profile(self):
        prepare(self.config, self.run, self.profile, self.worker)
        self.hardware = {"fingerprint": "device-b", "target": "sm_80"}
        self.events.clear()
        with self.assertRaisesRegex(ValueError, "hardware or software changed"):
            prepare(self.config, self.run, self.profile, self.worker)
        self.assertEqual(self.events, ["doctor"])

    def test_changed_test_limits_cannot_reuse_profile(self):
        prepare(self.config, self.run, self.profile, self.worker)
        changed = copy.deepcopy(self.config)
        changed["atol"] *= 2
        with self.assertRaisesRegex(ValueError, "atol"):
            prepare(changed, self.run, self.profile, self.worker)

    def test_changed_source_or_catalog_requires_new_profile(self):
        prepare(self.config, self.run, self.profile, self.worker)
        with patch("sako.preparation.implementation_hash", return_value="different"):
            with self.assertRaisesRegex(ValueError, "Profile source"):
                prepare(self.config, self.run, self.profile, self.worker)
        catalog = read_json(self.profile / "tasks.json")
        catalog["selected_task_ids"] = ["tampered"]
        write_json(self.profile / "tasks.json", catalog)
        with self.assertRaisesRegex(ValueError, "catalog changed"):
            prepare(self.config, self.run, self.profile, self.worker)

    def test_explicit_commit_must_match(self):
        config = copy.deepcopy(self.config)
        config["revision"] = "b" * 40
        with self.assertRaisesRegex(ValueError, "revisions differ"):
            validate_catalog(config, self.catalog, self.hardware)

    def test_no_eligible_tasks_does_not_save_catalog(self):
        self.catalog["selected_task_ids"] = []
        with self.assertRaisesRegex(ValueError, "no eligible"):
            prepare(self.config, self.run, self.profile, self.worker)
        self.assertFalse((self.profile / "tasks.json").exists())


if __name__ == "__main__":
    unittest.main()
