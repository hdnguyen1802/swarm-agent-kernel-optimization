"""Check stream reuse, exact job recovery, and resident state integrity."""

import contextlib
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sako.core import digest, implementation_hash, read_json, write_json
from sako.session_protocol import pack, unpack, workspace_file
from sako.session_worker import SessionWorker
from sako.transport import Worker, TransferError

ROOT = Path(__file__).resolve().parents[1]


class SessionProtocolTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "runs").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "runs")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.count = 0
        self.enterContext(patch("sako.session_worker.gpu_lock", return_value=contextlib.nullcontext()))
        def runner(job, session):
            self.count += 1
            print("Full worker log marker.")
            return {"ok": True, "job_hash": digest(job), "result": {"done": True},
                    "provenance": {"implementation_hash": implementation_hash()}}
        self.server = SessionWorker(self.root, runner=runner)
        self.addCleanup(self.server.close)

    def packet(self, identifier="a" * 32):
        job = {"action": "evaluate", "implementation_hash": implementation_hash(), "config": "runs/test/config.json"}
        raw = (json.dumps(job) + "\n").encode()
        return {"kind": "run", "job_id": identifier, "job": job,
                "implementation_hash": implementation_hash(), "timeout_seconds": 20,
                "job_path": "runs/test/jobs/" + identifier + ".json",
                "result_path": "runs/test/jobs/" + identifier + ".result.json",
                "files": [{"path": "runs/test/jobs/" + identifier + ".json", **pack(raw)}]}

    def test_repeated_job_and_reconnect_return_original_saved_result(self):
        packet = self.packet()
        first = self.server.handle(packet)
        second = self.server.handle(packet)
        self.assertEqual(first["result"], second["result"])
        self.assertEqual(self.count, 1)
        self.assertIn(b"Full worker log marker", unpack(first["worker_log"]))
        new = SessionWorker(self.root, runner=lambda *a, **k: self.fail("A recovered job ran twice."))
        self.addCleanup(new.close)
        recovered = new.handle({**packet, "kind": "recover", "files": []})
        self.assertEqual(recovered["result"], first["result"])

    def test_same_id_cannot_change_inputs(self):
        packet = self.packet()
        self.server.handle(packet)
        altered = copy.deepcopy(packet)
        altered["job"]["action"] = "model"
        with self.assertRaisesRegex(ValueError, "different inputs"):
            self.server.handle(altered)
        self.assertEqual(self.count, 1)

    def test_running_or_interrupted_job_is_never_submitted_again(self):
        packet = self.packet()
        result = str(workspace_file(self.root, packet["result_path"]))
        with self.server.db() as db:
            db.execute("INSERT INTO jobs VALUES (?,?,?,?,?)",
                       (packet["job_id"], digest(packet["job"]), result, "running", "old-session"))
        self.assertEqual(self.server.handle(packet)["status"], "pending")
        self.assertEqual(self.server.handle({**packet, "kind": "recover"})["status"], "pending")
        self.assertEqual(self.count, 0)

    def test_recover_can_prove_job_was_not_claimed(self):
        packet = self.packet()
        self.assertEqual(self.server.handle({**packet, "kind": "recover"})["status"], "not_started")
        self.assertEqual(self.server.handle(packet)["status"], "complete")
        self.assertEqual(self.count, 1)

    def test_hash_corruption_fails_before_evaluation_and_preserves_error(self):
        packet = self.packet()
        packet["files"][0]["sha256"] = "0" * 64
        response = self.server.handle(packet)
        payload = json.loads(unpack(response["result"]))
        self.assertFalse(payload["ok"])
        self.assertIn("hash differs", payload["error"])
        self.assertEqual(self.count, 0)
        self.assertTrue(response["session"]["retire"])

    def test_source_change_blocks_reuse(self):
        with patch("sako.session_worker.implementation_hash", return_value="new-source"):
            with self.assertRaisesRegex(ValueError, "source changed"):
                self.server.handle(self.packet())
        self.assertEqual(self.count, 0)

    def test_small_files_use_same_checked_stream_and_path_cannot_escape(self):
        path = self.root / "runs/test/evidence.json"
        write_json(path, {"test": "complete"})
        result = self.server.handle({"kind": "get", "path": "runs/test/evidence.json",
                                     "implementation_hash": implementation_hash()})
        self.assertEqual(json.loads(unpack(result["file"])), {"test": "complete"})
        for name in ("../secret.json", "runs/../../secret.json", "/absolute", ".git/config"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                workspace_file(self.root, name)

    def test_job_limit_retires_process_between_completed_jobs(self):
        self.server.max_jobs = 1
        response = self.server.handle(self.packet())
        self.assertTrue(response["session"]["retire"])
        self.assertTrue(self.server.stop)

    def test_watchdog_saves_a_failed_result_that_recovery_does_not_repeat(self):
        packet = self.packet()
        class Timer:
            daemon = False
            def __init__(self, seconds, callback): self.callback=callback
            def start(self): self.callback()
        with patch("sako.session_worker.threading.Timer",Timer), \
                patch("sako.session_worker.os._exit",side_effect=SystemExit(124)):
            with self.assertRaises(SystemExit):
                self.server.handle(packet)
        response=self.server.handle({**packet,"kind":"recover"})
        payload=json.loads(unpack(response["result"]))
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["failure_kind"],"worker")
        self.assertEqual(self.count,0)


class StreamTests(unittest.TestCase):
    """Run the real framed pipe in a child process with a fake GPU evaluator."""
    def setUp(self):
        (ROOT / "runs").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "runs")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = {"worker": {"transport": "local", "python": sys.executable, "gpu": 0,
            "timeout_seconds": 20, "persistent": True, "resident_max_jobs": 16,
            "transfer_timeout_seconds": 5, "transfer_deadline_seconds": 20, "transfer_attempts": 2}}
        self.worker = Worker(self.config, root=self.root)
        self.addCleanup(self.worker.close)
        code = f'''
import sys,os,json,contextlib
sys.path.insert(0,{str(ROOT)!r})
from pathlib import Path
from sako.core import digest,implementation_hash
import sako.session_worker as sw
sw.gpu_lock=lambda:contextlib.nullcontext()
def run(job,session):
    p=Path('calls.txt')
    p.write_text(p.read_text()+'x' if p.exists() else 'x')
    return {{'ok':True,'job_hash':digest(job),'result':{{'done':True}},'provenance':{{'implementation_hash':implementation_hash()}}}}
sw.run_payload=run
sw.main()
'''
        self.worker._session_command = lambda: [sys.executable, "-u", "-c", code]
        self.run = self.root / "runs/test"

    def test_two_jobs_and_download_share_one_process_and_connection(self):
        path = self.run / "candidate.json"
        content={"source":"kept"*20000}
        write_json(path, content)
        self.worker.put(path, "runs/test/candidate.json")
        first = self.worker.run("doctor", self.run)
        hello = dict(self.worker._resident.hello)
        second = self.worker.run("doctor", self.run)
        self.assertTrue(first["done"] and second["done"])
        self.assertEqual(self.worker._resident.hello, hello)
        self.assertEqual(self.worker._resident.connection_count, 1)
        self.worker.get("runs/test/candidate.json", self.root / "download.json")
        self.assertEqual(read_json(self.root / "download.json"), content)
        self.assertEqual((self.root / "calls.txt").read_text(), "xx")

    def test_lost_response_recovers_same_job_without_another_execution(self):
        client = self.worker._session()
        original = client.rpc
        lost = False
        def rpc(request, timeout):
            nonlocal lost
            result = original(request, timeout)
            if request["kind"] == "run" and not lost:
                lost = True
                client.close()
                raise TransferError("Connection closed after result was saved.")
            return result
        with patch.object(client, "rpc", side_effect=rpc), patch("sako.session_client.time.sleep"):
            self.assertTrue(self.worker.run("doctor", self.run)["done"])
        self.assertEqual((self.root / "calls.txt").read_text(), "x")
        self.assertFalse(list((self.run / "jobs").glob("*.pending.json")))

    def test_saved_packet_corruption_stops_resume(self):
        client = self.worker._session()
        with patch.object(client, "rpc", side_effect=TransferError("Unavailable.")):
            with self.assertRaises(TransferError):
                self.worker.run("doctor", self.run)
        packet = next((self.run / "jobs").glob("*.request.json"))
        value = read_json(packet)
        value["job"]["action"] = "changed"
        write_json(packet, value)
        with self.assertRaisesRegex(TransferError, "request changed"):
            self.worker.run("doctor", self.run)
        self.assertFalse((self.root / "calls.txt").exists())

    def test_close_releases_child_process(self):
        self.worker.run("doctor", self.run)
        process = self.worker._resident.process
        self.worker.close()
        self.assertIsNotNone(process.poll())

    def test_file_download_retries_are_bounded_and_preserve_old_file(self):
        client=self.worker._session()
        target=self.root/"old.json"
        target.write_text("old")
        with patch.object(client,"rpc",side_effect=TransferError("Connection reset.")) as rpc, \
                patch("sako.session_client.time.sleep"):
            with self.assertRaises(TransferError):
                client.get("runs/test/file.json",target)
        self.assertEqual(rpc.call_count,2)
        self.assertEqual(target.read_text(),"old")

    def test_bootstrap_upload_uses_one_checked_ssh_command(self):
        import subprocess,base64
        config=copy.deepcopy(self.config)
        config["worker"].update(transport="ssh",host="test@example",identity="key",root="sako/project")
        worker=Worker(config,root=self.root)
        self.addCleanup(worker.close)
        path=self.root/"source.tar.gz"
        path.write_bytes(b"source-bytes")
        with patch("sako.transport.run_command",return_value=subprocess.CompletedProcess([],0,"","")) as run:
            worker.put(path,"project.tar.gz")
        self.assertEqual(run.call_count,1)
        self.assertEqual(base64.b64decode(run.call_args.kwargs["input_text"]),b"source-bytes")
        command=run.call_args.args[0][-1]
        self.assertIn("sha256sum -c",command)
        self.assertLess(command.index("sha256sum -c"),command.index("mv -f"))


if __name__ == "__main__":
    unittest.main()
