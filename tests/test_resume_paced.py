from triton_fixtures import RMS_SOURCE
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.resume_paced import PacedWorker, resume_paced
from sako.agent import Gemini
from sako.core import digest, read_json, validate_candidate, write_json
from sako.store import Store


class PacedResume(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.run = Path(self.directory.name)
        self.profile = self.run / "profile"
        self.profile.mkdir()
        self.config = read_json(Path(__file__).resolve().parents[1] / "configs/project.json")
        self.config_path = self.run / "config.json"
        write_json(self.config_path, self.config)
        self.store = Store(self.run / "state.sqlite")
        call = self.store.reserve("writer", 200, self.config["agent"])
        self.store.finish_call(call, 100)
        self.pending = self.run / "jobs/existing.pending.json"
        write_json(self.pending, {"job_id": "a" * 32, "job_hash": "unchanged", "status": "awaiting_result",
                                  "dispatch_attempts": 1})
        self.pending_raw = self.pending.read_bytes()
        PacedWorker._last_start = None
        self.addCleanup(setattr, PacedWorker, "_last_start", None)


    def test_pacing_is_shared_and_preserves_command_arguments(self):
        first = PacedWorker(self.config, minimum_seconds=3.)
        second = PacedWorker(self.config, minimum_seconds=3.)
        clock = {"now": 0.}
        def sleep(seconds):
            clock["now"] += seconds
        starts = []
        def logged(command, timeout, **kwargs):
            starts.append((clock["now"], command, timeout, kwargs))
            return "unchanged result"
        environment = {"CUDA_VISIBLE_DEVICES": "0"}
        with patch("scripts.resume_paced.time.monotonic", side_effect=lambda: clock["now"]), \
             patch("scripts.resume_paced.time.sleep", side_effect=sleep), \
             patch("sako.transport.Worker._logged_command", side_effect=logged):
            self.assertEqual(first._logged_command(["ssh", "same job"], 120, env=environment, log_name="job-id"),
                             "unchanged result")
            second._logged_command(["scp", "same payload"], 600, log_name="transfer-id")
            first._logged_command(["python", "local"], 20)
            first._logged_command(["ssh", "same result download"], 600)
        self.assertEqual([row[0] for row in starts], [0., 3., 3., 6.])
        self.assertEqual(starts[0][1:3], (["ssh", "same job"], 120))
        self.assertIs(starts[0][3]["env"], environment)
        self.assertEqual(starts[0][3]["log_name"], "job-id")

    def test_recovery_is_bounded_and_keeps_ledger_and_pending_job(self):
        before = self.store.usage()
        results = [{"status": status, "usage": before}
                   for status in ("infrastructure_failure", "infrastructure_failure", "budget_exhausted")]
        with patch("scripts.resume_paced.prepare") as prepare, \
             patch("scripts.resume_paced.SwarmLoop") as loop, \
             patch("scripts.resume_paced._cooldown", return_value=True) as cooldown:
            loop.return_value.run.side_effect = results
            result = resume_paced(self.config_path, self.run, self.profile)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(loop.return_value.run.call_count, 3)
        self.assertEqual([call.args[1] for call in cooldown.call_args_list], [20, 40])
        workers = [call.args[3] for call in prepare.call_args_list]
        self.assertTrue(all(worker is workers[0] for worker in workers))
        self.assertTrue(all(call.args[4](self.config) is workers[0] for call in loop.call_args_list))
        self.assertEqual(self.store.usage(), before)
        self.assertEqual(self.pending.read_bytes(), self.pending_raw)
        records = [read_json(path) for path in sorted((self.run / "transport_recovery").glob("*-attempt-*.json"))]
        self.assertEqual([row["status"] for row in records], [row["status"] for row in results])
        self.assertTrue(all(row["pending_jobs_after"][0]["job_id"] == "a" * 32 for row in records))
        # An API or budget failure must not cause a controller recovery retry.
        with patch("scripts.resume_paced.prepare"), patch("scripts.resume_paced.SwarmLoop") as loop, \
             patch("scripts.resume_paced._cooldown") as cooldown:
            loop.return_value.run.return_value = {"status": "request_failure", "usage": before}
            self.assertEqual(resume_paced(self.config_path, self.run, self.profile)["status"], "request_failure")
        cooldown.assert_not_called()
        self.assertEqual(loop.return_value.run.call_count, 1)

    def test_existing_stop_prevents_prepare_and_paid_requests(self):
        stop = self.run / "STOP"
        stop.write_text("Keep stopped.")
        before = self.store.usage()
        with patch("scripts.resume_paced.prepare") as prepare, patch("scripts.resume_paced.SwarmLoop") as loop:
            self.assertEqual(resume_paced(self.config_path, self.run, self.profile)["status"], "stopped")
        prepare.assert_not_called()
        loop.assert_not_called()
        self.assertEqual(stop.read_text(), "Keep stopped.")
        self.assertEqual(self.store.usage(), before)
        self.assertEqual(read_json(self.config_path)["agent"]["max_total_tokens"], self.config["agent"]["max_total_tokens"])





if __name__ == "__main__":
    unittest.main()
