from triton_fixtures import RMS_SOURCE
import copy
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from scripts.revalidate_controller import revalidate_controller, _ledger_snapshot
from sako.core import digest, read_json, write_json
from sako.hardware import hardware_fingerprint
from sako.preparation import profile_settings
from sako.store import Store
from sako.lessons import LessonStore


class RevalidationFixture:
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.run = self.root / "runs/test"
        self.run.mkdir(parents=True)
        (self.run / "STOP").touch()
        self.config = read_json(Path(__file__).resolve().parents[1] / "configs/project.json")
        self.config["revision"] = "a" * 40
        self.old_sources = {"loop.py": "old_loop = True\n",
            "core.py": "core = True\n", "gpu.py": "evaluator = True\n", "model.py": "model = True\n",
            "hardware.py": "hardware = True\n", "store.py": "store = True\n", "agent.py": "agent = True\n",
            "prompts/triton_writer.txt": "Keep precision.\n"}
        self.current_sources = {**self.old_sources, "loop.py": "new_loop = True\n"}
        for name, value in self.current_sources.items():
            path = self.root / "sako" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value, encoding="utf-8")
        self.old_hash, self.new_hash = digest(self.old_sources), digest(self.current_sources)
        hardware = {"target": "sm_70", "capability": [7, 0], "facts": {}, "software": {}, "features": {},
                    "support": {"supported": True}}
        hardware["fingerprint"] = hardware_fingerprint(hardware)
        self.catalog = {"config": copy.deepcopy(self.config), "hardware": hardware, "tasks": []}
        self.stamp = {"implementation_hash": self.old_hash, "settings_hash": digest(profile_settings(self.config)),
                      "catalog_hash": digest(self.catalog), "hardware_fingerprint": hardware["fingerprint"]}
        write_json(self.run / "config.json", self.config)
        write_json(self.run / "profile/tasks.json", self.catalog)
        write_json(self.run / "profile/preparation.json", self.stamp)
        self.stamp_raw = (self.run / "profile/preparation.json").read_bytes()
        self.archive = self.run / "source-initial.tar.gz"
        self.make_archive()
        self.doctor = self.run / "doctor-new.result.json"
        write_json(self.doctor, {"ok": True, "result": {"hardware": hardware, "compile_probe": True},
                                 "provenance": {"implementation_hash": self.new_hash}})
        self.store = Store(self.run / "state.sqlite")
        call = self.store.reserve("writer", 200, self.config["agent"])
        self.store.finish_call(call, 100)
        self.progress = {"queue": ["linear_0"], "index": 0, "round": 1, "completed": []}
        self.store.set_many({"status": "stopped", "generation": 1, "task_progress": self.progress,
            "round_evaluated:0": {"task_id": "linear_0"}, "reviewed:0": True,
            "fingerprint": digest({"config": self.config, "catalog": self.catalog, "implementation": self.old_hash})})
        write_json(self.run / "result.json", {"status": "stopped", "usage": self.store.usage(),
                                              "task_progress": self.progress})
        write_json(self.run / "calls/00001.json", {"raw": "Keep this API record."})
        write_json(self.run / "evidence/failed.json", {"raw_samples": [1, 2, 3], "correct": False})

    def make_archive(self):
        with tarfile.open(self.archive, "w:gz") as archive:
            for name, value in {**{"sako/" + k: v for k, v in self.old_sources.items()},
                                "configs/project.json": json.dumps(self.config)}.items():
                raw = value.encode()
                info = tarfile.TarInfo(name)
                info.size = len(raw)
                archive.addfile(info, io.BytesIO(raw))

    def apply(self):
        return revalidate_controller(self.run, self.archive, self.doctor, project_root=self.root)

    def ledger(self):
        with self.store.connect() as db:
            return _ledger_snapshot(db)









class SwarmControllerRevalidation(unittest.TestCase):
    make_archive = RevalidationFixture.make_archive
    apply = RevalidationFixture.apply
    ledger = RevalidationFixture.ledger

    def setUp(self):
        RevalidationFixture.setUp(self)
        self.config = read_json(Path(__file__).resolve().parents[1] / "configs/project.json")
        self.config["revision"] = "a" * 40
        self.old_sources.update({"swarm.py": "old_swarm = True\n", "swarm_policy.py": "old_policy = True\n",
                                 "profiling.py": "profiling = True\n", "diagnostics.py": "diagnostics = True\n",
                                 "lessons.py": "lessons = True\n", "archive.py": "archive = True\n"})
        self.current_sources = {**self.old_sources, "swarm.py": "new_swarm = True\n",
                                "swarm_policy.py": "new_policy = True\n"}
        for name, value in self.current_sources.items():
            (self.root / "sako" / name).write_text(value, encoding="utf-8")
        self.old_hash, self.new_hash = digest(self.old_sources), digest(self.current_sources)
        self.catalog["config"] = copy.deepcopy(self.config)
        self.stamp.update(implementation_hash=self.old_hash, settings_hash=digest(profile_settings(self.config)),
                          catalog_hash=digest(self.catalog))
        write_json(self.run / "config.json", self.config)
        write_json(self.run / "profile/tasks.json", self.catalog)
        write_json(self.run / "profile/preparation.json", self.stamp)
        self.stamp_raw = (self.run / "profile/preparation.json").read_bytes()
        self.make_archive()
        doctor = read_json(self.doctor)
        doctor["provenance"]["implementation_hash"] = self.new_hash
        write_json(self.doctor, doctor)
        candidate = {"name": "trial", "task_id": "linear_0", "strategy": "Test a new tile.",
                     "backend": "triton", "references": [], "source": RMS_SOURCE}
        key, _ = self.store.add(candidate, "memory", 0, None)
        self.store.record(key, {"correct": True, "decision": {"accepted": False, "speedup": 1.0}})
        assignments = {
            "0": {"id": "experiment-00000", "generation": 0, "status": "completed", "task_id": "linear_0",
                  "candidate_id": key, "candidate": candidate},
            "1": {"id": "experiment-00001", "generation": 1, "status": "failed", "task_id": "linear_0",
                  "error": "Writer returned an invalid schema."}}
        self.store.set_many({"generation": 2, "swarm_next_assignment": 2, "swarm_assignments": assignments,
                            "swarm_task_counts": {"linear_0": 1}, "reviewed:0": True,
                            "fingerprint": digest({"config": self.config, "catalog": self.catalog,
                                                   "implementation": self.old_hash})})
        LessonStore(self.store)
        with self.store.connect() as db:
            db.execute("INSERT INTO swarm_lesson_evidence VALUES(?,?)", ("raw-hash", '{"correct":false}'))
            db.execute("INSERT INTO swarm_lessons VALUES(?,?,?,?,?,?,?)", (
                "lesson-1", "scope-1", key, "operation", "failed", 1., '{"id":"lesson-1"}'))
            db.execute("INSERT INTO swarm_lesson_supersedes VALUES(?,?)", ("lesson-old", "lesson-1"))
        for folder in ("profiles", "diagnostics", "lessons", "plans"):
            write_json(self.run / folder / "raw.json", {"raw": folder})
        self.store.set("kernel_profiles", {"baseline:linear_0": {"status": "unavailable"}})
        self.store.set("swarm_queue", [{"hypothesis": {"id": "queued"}, "plan_id": 0}])
        self.store.set("swarm_task_limits", {"linear_0": 1000000})
        self.original_state = self.state()

    def state(self):
        with self.store.connect() as db:
            return {row["key"]: row["value"] for row in db.execute("SELECT key,value FROM state")}

    def archive_with_config(self, config):
        with tarfile.open(self.archive, "w:gz") as archive:
            values = {**{"sako/" + k: v for k, v in self.old_sources.items()},
                      "configs/project.json": json.dumps(config)}
            for name, value in values.items():
                raw = value.encode()
                info = tarfile.TarInfo(name)
                info.size = len(raw)
                archive.addfile(info, io.BytesIO(raw))

    def test_swarm_update_preserves_failed_assignments_lessons_and_evidence(self):
        ledger = self.ledger()
        files = {str(path): path.read_bytes() for folder in ("profiles", "diagnostics", "lessons", "plans")
                 for path in (self.run / folder).rglob("*.json")}
        result = self.apply()
        self.assertEqual(result["changed_files"], ["sako/swarm.py", "sako/swarm_policy.py"])
        self.assertEqual(result["boundary"], {"assignments": 2, "completed": 1, "failed": 1,
                                             "task_counts": {"linear_0": 1}})
        self.assertEqual(self.ledger(), ledger)
        for path, raw in files.items():
            self.assertEqual(Path(path).read_bytes(), raw)
        changes = {key for key in set(self.original_state) | set(self.state())
                   if self.original_state.get(key) != self.state().get(key)}
        self.assertEqual(changes, {"fingerprint", "controller_updates"})
        self.assertEqual(result["ledger_before"]["tables"]["swarm_lesson_evidence"]["count"], 1)
        self.assertEqual(result["max_total_tokens"], self.config["agent"]["max_total_tokens"])

    def test_archived_main_alias_must_match_catalog_commit_and_all_other_settings(self):
        original = copy.deepcopy(self.config)
        original["revision"] = "main"
        self.archive_with_config(original)
        result = self.apply()
        self.assertEqual(result["config_provenance"]["mode"], "profile_pinned_main")
        self.assertEqual(result["config_provenance"]["catalog_revision"], "a" * 40)
        self.assertEqual(read_json(self.run / "config.json"), self.config)

    def test_another_alias_or_changed_budget_cannot_use_revision_exception(self):
        for change in ({"revision": "release"}, {"revision": "main", "atol": .1}):
            original = {**self.config, **change}
            self.archive_with_config(original)
            with self.assertRaisesRegex(ValueError, "original archived config"):
                self.apply()

    def test_swarm_evaluator_profiler_diagnostics_and_prompt_changes_are_rejected(self):
        for name in ("gpu.py", "model.py", "profiling.py", "diagnostics.py", "lessons.py", "prompts/triton_writer.txt", "loop.py"):
            with self.subTest(name=name):
                path = self.root / "sako" / name
                original = path.read_bytes()
                path.write_bytes(original + b"# changed\n")
                with self.assertRaisesRegex(ValueError, "Only swarm.py"):
                    self.apply()
                path.write_bytes(original)
        self.assertEqual(self.state(), self.original_state)

    def test_generated_running_or_unreviewed_assignment_rejects_audit(self):
        original = self.store.get("swarm_assignments")
        for status in ("planned", "running", "generated", "evaluated"):
            assignments = copy.deepcopy(original)
            assignments["0"]["status"] = status
            self.store.set("swarm_assignments", assignments)
            with self.assertRaisesRegex(ValueError, "unfinished assignments"):
                self.apply()
        self.store.set("swarm_assignments", original)
        self.store.set("reviewed:0", False)
        with self.assertRaisesRegex(ValueError, "evaluation and review"):
            self.apply()

    def test_swarm_counts_and_assignment_sequence_must_match(self):
        self.store.set("swarm_task_counts", {"linear_0": 2})
        with self.assertRaisesRegex(ValueError, "task counts"):
            self.apply()
        self.store.set("swarm_task_counts", {"linear_0": 1})
        self.store.set("swarm_next_assignment", 3)
        with self.assertRaisesRegex(ValueError, "assignment sequence"):
            self.apply()

    def test_reserved_api_call_blocks_swarm_audit(self):
        self.store.reserve("writer_0_memory", 200, self.config["agent"])
        with self.assertRaisesRegex(ValueError, "in flight"):
            self.apply()

    def test_swarm_requires_stop_file_and_same_saved_result(self):
        (self.run / "STOP").unlink()
        with self.assertRaisesRegex(ValueError, "STOP"):
            self.apply()
        (self.run / "STOP").touch()
        result = read_json(self.run / "result.json")
        result["usage"]["tokens"] += 1
        write_json(self.run / "result.json", result)
        with self.assertRaisesRegex(ValueError, "stopped result"):
            self.apply()


if __name__ == "__main__":
    unittest.main()
