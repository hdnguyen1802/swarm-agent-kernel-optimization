import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sako.core import digest, implementation_hash, read_json, write_json
from sako.process import CommandTimeout, run_command
from sako.transport import Worker, TransferError, WorkerJobError, prune_package_sources


def config():
    return {"worker": {"transport": "ssh", "host": "test@example", "identity": "key",
                       "root": "sako/project", "python": "../venv/bin/python", "gpu": 0,
                       "cuda_home": "/usr/local/cuda", "timeout_seconds": 30,
                       "transfer_attempts": 3, "transfer_timeout_seconds": 5,
                       "transfer_deadline_seconds": 20}}


class TransferTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.worker = Worker(config(), root=self.root)
        self.source = self.root / "candidate.json"
        self.source.write_text('{"source":"complete"}')
        self.hash = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.enterContext(patch("sako.transport.time.sleep"))

    def completed(self, command, returncode=0, stdout="", stderr=""):
        return subprocess.CompletedProcess(command, returncode, stdout, stderr)

    def test_upload_retries_transient_failure_and_commits_after_hash(self):
        copied, committed = [], []
        def execute(command, **kwargs):
            if command[0] == "scp":
                copied.append(command[-1])
                if len(copied) == 1:
                    return self.completed(command, 255, stderr="Connection reset by peer")
            if "sha256sum" in command[-1]:
                return self.completed(command, stdout=self.hash + "  file\n")
            if "mv -f" in command[-1]:
                committed.append(command[-1])
            return self.completed(command)
        with patch("sako.transport.run_command", side_effect=execute):
            self.worker.put(self.source, "runs/candidate.json")
        self.assertEqual(len(copied), 2)
        self.assertEqual(copied[0], copied[1])
        self.assertTrue(copied[0].endswith(".part"))
        self.assertEqual(len(committed), 1)
        reports = list(self.worker.log_directory.glob("*.transfer.json"))
        report = read_json(reports[0])
        self.assertEqual(len(report["attempts"]), 2)
        self.assertEqual(report["status"], "complete")

    def test_permission_failure_does_not_retry_and_keeps_full_error(self):
        full_error = "Permission denied " + "x" * 30000
        with patch("sako.transport.run_command", side_effect=lambda c, **k: self.completed(c, 255, stderr=full_error)) as execute:
            with self.assertRaises(TransferError) as raised:
                self.worker.put(self.source, "runs/candidate.json")
        self.assertEqual(execute.call_count, 1)
        self.assertTrue(raised.exception.artifacts)
        commands = [read_json(p) for p in self.worker.log_directory.glob("*.json") if not p.name.endswith("transfer.json")]
        self.assertEqual(commands[0]["stderr"], full_error)

    def test_transient_retry_count_is_bounded(self):
        with patch("sako.transport.run_command", side_effect=lambda c, **k: self.completed(c, 255, stderr="Connection timed out")) as execute:
            with self.assertRaises(TransferError):
                self.worker.put(self.source, "runs/candidate.json")
        self.assertEqual(execute.call_count, 3)

    def test_download_checks_hash_before_replacing_existing_file(self):
        target = self.root / "result.json"
        target.write_text("old")
        downloads = []
        def execute(command, **kwargs):
            if command[0] == "scp":
                downloads.append(command[-1])
                self.assertEqual(target.read_text(), "old")
                Path(command[-1]).write_bytes(b"bad" if len(downloads) == 1 else self.source.read_bytes())
                return self.completed(command)
            return self.completed(command, stdout=self.hash + "  file\n")
        with patch("sako.transport.run_command", side_effect=execute):
            self.worker.get("runs/result.json", target)
        self.assertEqual(len(downloads), 2)
        self.assertEqual(target.read_bytes(), self.source.read_bytes())

    def test_unknown_job_result_is_recovered_without_resubmitting(self):
        directory = self.root / "runs" / "test"
        downloads = []
        def get(relative, destination, **kwargs):
            downloads.append(relative)
            if len(downloads) == 1:
                raise TransferError("Network temporarily unavailable.")
            job_path = Path(str(destination).replace(".result.json", ".json"))
            job = read_json(job_path)
            write_json(destination, {"job_hash": digest(job), "ok": True, "result": {"done": True},
                                     "provenance": {"implementation_hash": implementation_hash()}})
        with patch.object(self.worker, "put") as put, \
             patch.object(self.worker, "_logged_command", return_value=self.completed([])) as launch, \
             patch.object(self.worker, "get", side_effect=get):
            with self.assertRaises(TransferError):
                self.worker.run("evaluate", directory, candidate="candidate.json")
            self.assertTrue(self.worker.run("evaluate", directory, candidate="candidate.json")["done"])
        self.assertEqual(launch.call_count, 1)
        self.assertEqual(put.call_count, 2)
        self.assertEqual(downloads[0], downloads[1])
        self.assertFalse(list((directory / "jobs").glob("*.pending.json")))

    def test_stale_worker_code_is_rejected(self):
        directory = self.root / "runs" / "test"
        def get(relative, destination, **kwargs):
            write_json(destination, {"job_hash": kwargs["job_hash"], "ok": True, "result": {},
                                     "provenance": {"implementation_hash": "stale"}})
        with patch.object(self.worker, "put"), \
             patch.object(self.worker, "_logged_command", return_value=self.completed([])), \
             patch.object(self.worker, "get", side_effect=get):
            with self.assertRaisesRegex(TransferError, "differs"):
                self.worker.run("doctor", directory)

    def test_transfer_path_cannot_escape_worker_root(self):
        with self.assertRaises(ValueError):
            self.worker.put(self.source, "../outside.json")
        with self.assertRaises(ValueError):
            self.worker.get("/outside.json", self.root / "file")

    def test_attempt_timeout_and_total_deadline_are_bounded(self):
        clock = [0.]
        observed = []
        self.worker.transfer_deadline = 3
        def execute(command, **kwargs):
            observed.append(kwargs["timeout"])
            clock[0] += 3
            raise CommandTimeout(command, kwargs["timeout"], "partial output", "connection stalled")
        with patch("sako.transport.time.monotonic", side_effect=lambda: clock[0]), \
             patch("sako.transport.run_command", side_effect=execute):
            with self.assertRaises(TransferError):
                self.worker.put(self.source, "runs/candidate.json")
        self.assertEqual(observed, [3.])
        commands = [read_json(p) for p in self.worker.log_directory.glob("*.json") if not p.name.endswith("transfer.json")]
        self.assertEqual(commands[0]["stdout"], "partial output")

    def test_worker_error_retains_full_payload(self):
        directory = self.root / "runs" / "test"
        traceback = "failure\n" * 10000
        def get(relative, destination, **kwargs):
            write_json(destination, {"job_hash": kwargs["job_hash"], "ok": False,
                                     "error": "Compilation failed.", "traceback": traceback,
                                     "provenance": {"implementation_hash": implementation_hash()}})
        with patch.object(self.worker, "put"), \
             patch.object(self.worker, "_logged_command", return_value=self.completed([])), \
             patch.object(self.worker, "get", side_effect=get):
            with self.assertRaises(WorkerJobError) as raised:
                self.worker.run("evaluate", directory)
        self.assertEqual(raised.exception.payload["traceback"], traceback)
        artifact = raised.exception.artifacts[0]
        self.assertEqual(read_json(self.root / artifact["path"])["traceback"], traceback)

    def test_worker_infrastructure_failure_is_not_a_kernel_error(self):
        directory = self.root / "runs" / "test"
        def get(relative, destination, **kwargs):
            write_json(destination, {"job_hash": kwargs["job_hash"], "ok": False,
                                     "error": "Source mismatch.", "failure_kind": "infrastructure",
                                     "provenance": {"implementation_hash": implementation_hash()}})
        with patch.object(self.worker, "put"), \
             patch.object(self.worker, "_logged_command", return_value=self.completed([])), \
             patch.object(self.worker, "get", side_effect=get):
            with self.assertRaises(TransferError):
                self.worker.run("evaluate", directory)

    def test_failed_result_also_requires_current_worker_provenance(self):
        for provenance in ({}, {"implementation_hash": "old"}):
            directory = self.root / "runs" / ("missing" if not provenance else "old")
            def get(relative, destination, **kwargs):
                write_json(destination, {"job_hash": kwargs["job_hash"], "ok": False,
                                         "error": "Candidate failed.", "provenance": provenance})
            with patch.object(self.worker, "put"), \
                 patch.object(self.worker, "_logged_command", return_value=self.completed([])), \
                 patch.object(self.worker, "get", side_effect=get):
                with self.assertRaisesRegex(TransferError, "differs"):
                    self.worker.run("evaluate", directory)


class DispatchRecoveryTests(unittest.TestCase):
    setUp = TransferTests.setUp
    completed = TransferTests.completed

    def complete_result(self, relative, destination, **kwargs):
        write_json(destination, {"job_hash": kwargs["job_hash"], "ok": True, "result": {"done": True},
                                 "provenance": {"implementation_hash": implementation_hash()}})

    def pending(self, directory):
        path = next((directory / "jobs").glob("*.pending.json"))
        return path, read_json(path)

    def test_banner_failure_retries_same_job_then_succeeds(self):
        directory = self.root / "runs" / "retry"
        commands = []
        def execute(command, **kwargs):
            commands.append(command)
            client_log = Path(command[command.index("-E") + 1])
            client_log.write_text("Connection timed out during banner exchange\n" if len(commands) == 1
                                  else "Authenticated to target using publickey.\n")
            return self.completed(command, 255 if len(commands) == 1 else 0)
        with patch.object(self.worker, "put"), patch("sako.transport.run_command", side_effect=execute), \
             patch.object(self.worker, "get", side_effect=self.complete_result) as get:
            result = self.worker.run("evaluate", directory, candidate="candidate.json")
        self.assertTrue(result["done"])
        self.assertEqual(len(commands), 2)
        self.assertEqual(commands[0][-1], commands[1][-1])
        self.assertEqual(get.call_count, 1)
        self.assertEqual(len(list(self.worker.log_directory.glob("*-dispatch-*.json"))), 2)
        self.assertEqual(len(list(self.worker.log_directory.glob("*-dispatch-*.ssh.log"))), 2)

    def test_three_total_start_attempts_remain_bounded_across_resume(self):
        directory = self.root / "runs" / "exhausted"
        def execute(command, **kwargs):
            Path(command[command.index("-E") + 1]).write_text("ssh: connect to host target port 22: Connection refused\n")
            return self.completed(command, 255)
        with patch.object(self.worker, "put"), patch("sako.transport.run_command", side_effect=execute) as launch, \
             patch.object(self.worker, "get") as get:
            with self.assertRaisesRegex(TransferError, "3 total attempts"):
                self.worker.run("evaluate", directory)
            _, pending = self.pending(directory)
            self.assertEqual(pending["dispatch_attempts"], 3)
            self.assertEqual(pending["status"], "not_dispatched")
            with self.assertRaisesRegex(TransferError, "3 total attempts"):
                self.worker.run("evaluate", directory)
        self.assertEqual(launch.call_count, 3)
        get.assert_not_called()
        logs = [item["log"] for item in pending["dispatch_history"]]
        self.assertEqual(len(set(logs)), 3)
        self.assertTrue(all((self.root / path).exists() for path in logs))

    def test_uncertain_disconnect_only_recovers_result(self):
        for index, diagnostics in enumerate(("Connection reset by peer\n", "Authenticated to target.\nConnection timed out during banner exchange\n", "")):
            directory = self.root / "runs" / f"uncertain-{index}"
            def execute(command, **kwargs):
                Path(command[command.index("-E") + 1]).write_text(diagnostics)
                # Remote stderr must not be confused with a local OpenSSH diagnostic.
                return self.completed(command, 255, stderr="Connection timed out during banner exchange\n")
            with patch.object(self.worker, "put"), patch("sako.transport.run_command", side_effect=execute) as launch, \
                 patch.object(self.worker, "get", side_effect=TransferError("result pending")):
                with self.assertRaises(TransferError):
                    self.worker.run("evaluate", directory)
                _, pending = self.pending(directory)
                self.assertEqual(pending["status"], "awaiting_result")
                with patch.object(self.worker, "get", side_effect=self.complete_result):
                    self.assertTrue(self.worker.run("evaluate", directory)["done"])
            self.assertEqual(launch.call_count, 1)

    def test_local_timeout_does_not_repeat_dispatch(self):
        directory = self.root / "runs" / "timeout"
        def execute(command, **kwargs):
            raise CommandTimeout(command, kwargs["timeout"], "", "")
        with patch.object(self.worker, "put"), patch("sako.transport.run_command", side_effect=execute) as launch, \
             patch.object(self.worker, "get", side_effect=TransferError("result pending")):
            with self.assertRaises(TransferError):
                self.worker.run("evaluate", directory)
            with patch.object(self.worker, "get", side_effect=self.complete_result):
                self.assertTrue(self.worker.run("evaluate", directory)["done"])
        self.assertEqual(launch.call_count, 1)

    def test_known_not_dispatched_state_resumes_with_same_job_id(self):
        directory = self.root / "runs" / "interrupted"
        commands = []
        def execute(command, **kwargs):
            commands.append(command)
            Path(command[command.index("-E") + 1]).write_text("Connection timed out during banner exchange\n" if len(commands) == 1 else "")
            return self.completed(command, 255 if len(commands) == 1 else 0)
        with patch.object(self.worker, "put"), patch("sako.transport.run_command", side_effect=execute):
            with patch("sako.transport.time.sleep", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    self.worker.run("evaluate", directory)
            _, pending = self.pending(directory)
            self.assertEqual(pending["dispatch_attempts"], 1)
            self.assertEqual(pending["status"], "not_dispatched")
            with patch.object(self.worker, "get", side_effect=self.complete_result):
                self.assertTrue(self.worker.run("evaluate", directory)["done"])
        self.assertEqual(len(commands), 2)
        self.assertEqual(commands[0][-1], commands[1][-1])

    def test_old_log_cannot_authorize_a_second_dispatch(self):
        directory = self.root / "runs" / "legacy"
        commands = []
        def execute(command, **kwargs):
            commands.append(command)
            return self.completed(command, 255 if len(commands) == 1 else 0)
        with patch.object(self.worker, "put"), patch("sako.transport.run_command", side_effect=execute) as launch:
            with patch.object(self.worker, "get", side_effect=TransferError("result pending")):
                with self.assertRaises(TransferError):
                    self.worker.run("evaluate", directory)
            pending_path, pending = self.pending(directory)
            pending.pop("dispatch_attempts")
            pending.pop("dispatch_history")
            pending["status"] = "dispatched"
            write_json(pending_path, pending)
            legacy_path = self.worker.log_directory / (pending["job_id"] + ".json")
            write_json(legacy_path, {"command": self.worker.ssh() + [commands[0][-1]], "status": "complete",
                "returncode": 255, "stdout": "", "stderr": "Connection timed out during banner exchange\nConnection to target port 22 timed out\n"})
            with patch.object(self.worker, "get", side_effect=self.complete_result):
                self.assertTrue(self.worker.run("evaluate", directory)["done"])
        self.assertEqual(launch.call_count, 1)
        self.assertTrue(legacy_path.exists())
        self.assertEqual(len(list(self.worker.log_directory.glob("*-dispatch-2-*.json"))), 0)

    def test_missing_log_and_altered_job_do_not_dispatch(self):
        directory = self.root / "runs" / "legacy-unknown"
        with patch.object(self.worker, "put"), \
             patch.object(self.worker, "_logged_command", return_value=self.completed([], 255)) as launch:
            with patch.object(self.worker, "get", side_effect=TransferError("result pending")):
                with self.assertRaises(TransferError):
                    self.worker.run("evaluate", directory)
                pending_path, pending = self.pending(directory)
                pending["status"] = "dispatched"
                write_json(pending_path, pending)
                with self.assertRaises(TransferError):
                    self.worker.run("evaluate", directory)
            job_path = directory / "jobs" / (pending["job_id"] + ".json")
            job = read_json(job_path)
            job["implementation_hash"] = "different"
            write_json(job_path, job)
            with self.assertRaisesRegex(TransferError, "source"):
                self.worker.run("evaluate", directory)
        self.assertEqual(launch.call_count, 1)

    def test_authentication_failure_stops_after_one_attempt(self):
        directory = self.root / "runs" / "auth"
        def execute(command, **kwargs):
            Path(command[command.index("-E") + 1]).write_text("test@example: Permission denied (publickey).\n")
            return self.completed(command, 255)
        with patch.object(self.worker, "put"), patch("sako.transport.run_command", side_effect=execute) as launch, \
             patch.object(self.worker, "get") as get:
            with self.assertRaisesRegex(TransferError, "authentication"):
                self.worker.run("evaluate", directory)
        self.assertEqual(launch.call_count, 1)
        get.assert_not_called()


class ProcessEvidence(unittest.TestCase):
    def test_timeout_preserves_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timeout.log"
            with self.assertRaises(CommandTimeout) as raised:
                run_command([sys.executable, "-c", "import time; print('x'*30000, flush=True); time.sleep(10)"],
                            timeout=1, log_path=path)
            self.assertIn("x" * 30000, raised.exception.stdout)
            self.assertIn("x" * 30000, path.read_text())


class PackageSourceSync(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.package = self.root / "sako"
        (self.package / "prompts").mkdir(parents=True)
        for name in ("__init__.py", "worker.py", "stale.py", "prompts/current.txt", "prompts/stale.txt",
                     "weights.pt", "notes.md"):
            (self.package / name).write_text("saved content", encoding="utf-8")
        self.expected = ["__init__.py", "worker.py", "prompts/current.txt"]

    def test_sync_removes_stale_sources_and_preserves_data(self):
        evidence = self.root / "runs" / "trial.json"
        evidence.parent.mkdir()
        evidence.write_text('{"timings": [1, 2, 3]}')
        prune_package_sources(self.root, self.expected)
        for name in self.expected + ["weights.pt", "notes.md"]:
            self.assertEqual((self.package / name).read_text(), "saved content")
        self.assertFalse((self.package / "stale.py").exists())
        self.assertFalse((self.package / "prompts/stale.txt").exists())
        self.assertEqual(json.loads(evidence.read_text()), {"timings": [1, 2, 3]})
        prune_package_sources(self.root, self.expected)

    def test_bad_or_incomplete_manifest_cannot_remove_sources(self):
        outside = self.root / "outside.py"
        outside.write_text("keep")
        for expected in ([], ["../outside.py"], [str(outside)], ["..\\outside.py"],
                         [*self.expected, "missing.py"]):
            with self.subTest(expected=expected), self.assertRaises((ValueError, FileNotFoundError)):
                prune_package_sources(self.root, expected)
            self.assertTrue((self.package / "stale.py").exists())
            self.assertTrue((self.package / "prompts/stale.txt").exists())
            self.assertEqual(outside.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
