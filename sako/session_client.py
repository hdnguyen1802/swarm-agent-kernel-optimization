"""Reuse one SSH connection. Recover saved jobs without duplicate execution."""

import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
import uuid

from .core import digest, implementation_hash, read_json, write_json
from .session_protocol import MAX_FRAME, VERSION, pack, unpack, publish, workspace_file
from .transport import TransferError, WorkerJobError, _IntegrityFailure, _transient


class SessionClient:
    def __init__(self, owner):
        self.owner = owner
        self.source = implementation_hash()
        self.process = None
        self.stderr = None
        self.responses = None
        self.staged = {}
        self.lock = threading.RLock()
        self.hello = None
        self.connection_count = 0

    def stage(self, source, relative):
        workspace_file(self.owner.root, relative)
        data = Path(source).read_bytes()
        self.staged[relative] = {"path": relative, **pack(data)}

    def _receive(self, timeout):
        try:
            response = self.responses.get(timeout=timeout)
        except queue.Empty as error:
            raise TransferError("The session response timed out. Recover the saved job.") from error
        if isinstance(response, Exception):
            raise TransferError("The SSH session closed. Recover the saved job.") from response
        if not isinstance(response, dict):
            raise TransferError("Invalid session response.")
        return response

    def connect(self):
        if implementation_hash() != self.source:
            self.close()
            raise TransferError("Controller source changed during the session. Use a new run.")
        if self.process is not None and self.process.poll() is None:
            return
        self.close()
        self.owner._pace_connection()
        started = time.perf_counter()
        path = self.owner.log_directory / ("session-" + uuid.uuid4().hex + ".ssh.log")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stderr = path.open("wb")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(self.owner.settings["gpu"]), PYTHONUNBUFFERED="1")
        options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
        self.process = subprocess.Popen(self.owner._session_command(), cwd=self.owner.root, env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr, bufsize=0, **options)
        self.connection_count += 1
        responses = self.responses = queue.Queue()
        process = self.process
        def read():
            try:
                while True:
                    line = process.stdout.readline(MAX_FRAME + 1)
                    if not line:
                        raise EOFError("SSH stream ended.")
                    if len(line) > MAX_FRAME:
                        raise ValueError("Session response exceeds its byte limit.")
                    responses.put(json.loads(line))
            except Exception as error:
                responses.put(error)
        threading.Thread(target=read, daemon=True).start()
        try:
            self.hello = self._receive(min(30, self.owner.settings["timeout_seconds"]))
            if (self.hello.get("protocol") != VERSION
                    or self.hello.get("implementation_hash") != self.source):
                raise TransferError("Worker code differs from the controller. Sync before a new run.")
            write_json(path.with_suffix(".json"), {**self.hello, "handshake_s": time.perf_counter() - started,
                                                 "command": self.owner._session_command()})
        except Exception as error:
            self.close()
            detail = path.read_text(encoding="utf-8", errors="replace")
            raise TransferError(str(error) + "\n" + detail) from error

    def rpc(self, request, timeout):
        self.connect()
        request = {**request, "implementation_hash": self.source}
        encoded = (json.dumps(request, allow_nan=False) + "\n").encode("utf-8")
        if len(encoded) > MAX_FRAME:
            raise TransferError("Session request exceeds its byte limit.")
        try:
            remaining = memoryview(encoded)
            while remaining:
                written = self.process.stdin.write(remaining)
                if not written:
                    raise BrokenPipeError("The SSH request stream closed.")
                remaining = remaining[written:]
            self.process.stdin.flush()
            response = self._receive(timeout)
        except (OSError, TransferError) as error:
            self.close()
            raise TransferError("SSH connection closed or timed out. Recover the same job.") from error
        if response.get("status") == "error":
            raise TransferError(response.get("error", "Session failed."))
        return response

    def get(self, relative, destination, *, job_hash=None):
        with self.lock:
            started = time.monotonic()
            last = None
            for attempt in range(self.owner.transfer_attempts):
                remaining = self.owner.transfer_deadline - (time.monotonic() - started)
                if remaining <= 0:
                    break
                try:
                    response = self.rpc({"kind": "get", "path": relative}, min(self.owner.transfer_timeout, remaining))
                    if response.get("status") == "large_file":
                        return self.owner._get_scp(relative, destination, job_hash=job_hash)
                    if response.get("status") != "file":
                        raise TransferError("Session did not return the requested file.")
                    try:
                        data = unpack(response["file"])
                    except ValueError as error:
                        raise _IntegrityFailure(str(error)) from error
                    if job_hash is not None and json.loads(data).get("job_hash") != job_hash:
                        raise TransferError("Worker result does not match the submitted job.")
                    publish(Path(destination), data)
                    return Path(destination)
                except (TransferError, _IntegrityFailure) as error:
                    last = error
                    if not _transient(error) or attempt + 1 == self.owner.transfer_attempts:
                        break
                    time.sleep(min(2 ** attempt, max(0, self.owner.transfer_deadline - (time.monotonic() - started))))
            raise TransferError("Session download failed within its retry bound: " + str(last)) from last

    def run(self, action, run_dir, fields):
        with self.lock:
            return self._run(action, run_dir, fields)

    def _run(self, action, run_dir, fields):
        root = self.owner.root
        directory = Path(run_dir).resolve()
        directory.relative_to(root)
        directory.mkdir(parents=True, exist_ok=True)
        self.owner.log_directory = directory / "jobs/transfers"
        config_path = directory / "jobs" / ("config-" + digest(self.owner.config) + ".json")
        write_json(directory / "config.json", self.owner.config)
        write_json(config_path, self.owner.config)
        config_relative = config_path.relative_to(root).as_posix()
        job = {"action": action, "config": config_relative, "implementation_hash": self.source, **fields}
        identity = digest({"protocol": VERSION, "job": job, "config": self.owner.config})
        pending_path = directory / "jobs" / (identity + ".pending.json")
        pending = read_json(pending_path) if pending_path.exists() else None
        if pending and pending.get("job_hash") != digest(job):
            raise TransferError("The saved session job changed.")
        job_id = pending["job_id"] if pending else uuid.uuid4().hex
        job_path = directory / "jobs" / (job_id + ".json")
        result_path = directory / "jobs" / (job_id + ".result.json")
        packet_path = directory / "jobs" / (job_id + ".request.json")
        if pending is None:
            write_json(job_path, job)
            self.stage(config_path, config_relative)
            self.stage(job_path, job_path.relative_to(root).as_posix())
            # Include every small model input in the same request. Fixtures stay
            # on the worker; weights never travel through this protocol.
            if fields.get("profile_dir"):
                path = root / fields["profile_dir"] / "tasks.json"
                if path.is_file():
                    self.stage(path, path.relative_to(root).as_posix())
            for field in ("candidate", "manifest"):
                if fields.get(field):
                    path = workspace_file(root, fields[field])
                    self.stage(path, fields[field])
            if fields.get("manifest") and fields.get("candidate_root"):
                for entry in read_json(root / fields["manifest"])["entries"]:
                    path = workspace_file(root, fields["candidate_root"] + "/" + entry["candidate_id"] + ".json")
                    self.stage(path, path.relative_to(root).as_posix())
            packet = {"kind": "run", "job_id": job_id, "job": job,
                "job_path": job_path.relative_to(root).as_posix(),
                "result_path": result_path.relative_to(root).as_posix(),
                "files": list(self.staged.values()), "timeout_seconds": self.owner.settings["timeout_seconds"]}
            if action == "evaluate" and not self.owner.local and self.owner.config.get("generation_objective") == "decode":
                packet["compact_result"] = True
            write_json(packet_path, packet)
            pending = {"job_id": job_id, "job_hash": digest(job), "transport": "resident",
                       "packet_hash": digest(packet), "status": "ready"}
            write_json(pending_path, pending)
        else:
            packet = read_json(packet_path)
            if digest(packet) != pending.get("packet_hash") or packet.get("job") != job or read_json(job_path) != job:
                raise TransferError("The saved session request changed.", job_id=job_id)
        start = time.perf_counter()
        last = None
        for attempt in range(self.owner.transfer_attempts):
            try:
                if pending["status"] == "ready":
                    pending["status"] = "requesting"
                    write_json(pending_path, pending)
                    response = self.rpc(packet, self.owner.settings["timeout_seconds"] + 30)
                else:
                    response = self.rpc({**packet, "kind": "recover", "files": []}, self.owner.transfer_timeout)
                    if response.get("status") == "not_started":
                        # Atomic server claims deduplicate the same job ID, even
                        # if an older connection reaches it after this check.
                        response = self.rpc(packet, self.owner.settings["timeout_seconds"] + 30)
                if response.get("status") != "complete":
                    raise TransferError("The original session job is still pending. Do not submit another job.", job_id=job_id)
                data = unpack(response["result"])
                payload = json.loads(data)
                if payload.get("job_hash") != digest(job) or payload.get("provenance", {}).get("implementation_hash") != self.source:
                    raise TransferError("Worker result has different source or inputs.", job_id=job_id)
                evidence = payload.get("remote_evidence")
                if evidence:
                    if evidence["result"] != result_path.relative_to(root).as_posix():
                        raise TransferError("Remote evidence points to a different job.")
                    for case in payload["result"].get("cases", []):
                        rank = case.get("search_reference", {})
                        if rank.pop("same_timing_pair", False):
                            rank.update(baseline=case["baseline"], candidate=case["candidate"])
                    payload["result"]["remote_evidence"] = evidence
                    write_json(result_path, payload)
                else:
                    publish(result_path, data)
                if implementation_hash() != self.source:
                    raise TransferError("Controller source changed while the job ran. The raw result is saved.", job_id=job_id)
                if response.get("worker_log"):
                    publish(job_path.with_suffix(".worker.log"), unpack(response["worker_log"]))
                elif response.get("worker_log_path"):
                    self.owner._get_scp(response["worker_log_path"], job_path.with_suffix(".worker.log"))
                self.staged.clear()
                pending_path.unlink(missing_ok=True)
                write_json(directory / "jobs" / (job_id + ".transport.json"), {
                    "mode": "resident", "wall_s": time.perf_counter() - start,
                    "connections": self.connection_count, "session": response.get("session"),
                    "file_bytes": sum(f["bytes"] for f in packet["files"])})
                if (response.get("session") or {}).get("retire"):
                    self.close()
                if not payload.get("ok"):
                    if payload.get("failure_kind") == "infrastructure":
                        raise TransferError(payload.get("error", "Worker infrastructure failed."), job_id=job_id)
                    from .feedback import artifact_descriptor
                    raise WorkerJobError(payload, artifact_descriptor(root, result_path))
                result = payload["result"]
                if isinstance(result, dict):
                    result = {**result, "worker_provenance": payload["provenance"],
                              "worker_total_s": payload.get("worker_total_s"),
                              "session_times": payload.get("session_times"),
                              "transport_wall_s": time.perf_counter() - start}
                return result
            except WorkerJobError:
                raise
            except (TransferError, ValueError) as error:
                last = error
                if not _transient(error) or attempt + 1 == self.owner.transfer_attempts:
                    break
                time.sleep(min(2 ** attempt, 5))
        raise TransferError(str(last), job_id=job_id) from last

    def close(self):
        process, self.process = self.process, None
        if process is not None:
            try:
                process.stdin.close()
                process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                process.kill()
                process.wait(timeout=5)
            finally:
                process.stdout.close()
        if self.stderr is not None:
            self.stderr.close()
            self.stderr = None
