"""Transfer jobs through SSH. Keep cloud credentials on the controller."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shlex
import tarfile
import time
import uuid
import atexit
import base64
from pathlib import Path, PurePosixPath

from .core import DEFAULT_RESIDENT_MAX_JOBS, read_json, write_json, digest, implementation_hash
from .feedback import artifact_descriptor
from .process import CommandTimeout, run_command


def prune_package_sources(root, expected):
    """Remove stale package sources after sync. Preserve data and run evidence."""
    root = Path(root).resolve()
    if (root / "sako").is_symlink():
        raise ValueError("The package folder must not be a symlink.")
    package = (root / "sako").resolve(strict=True)
    package.relative_to(root)
    expected = set(expected)
    if not expected:
        raise ValueError("The package source list must not be empty.")
    for name in expected:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ValueError("Invalid package source path.")
        source = package / name
        source.resolve(strict=True).relative_to(package)
        if not source.is_file():
            raise ValueError("Missing synchronized source: " + name)
    sources = [*package.glob("*.py"), *package.glob("prompts/*.txt")]
    # Check every path before removing any source.
    for source in sources:
        source.resolve().relative_to(package)
    for source in sources:
        if source.relative_to(package).as_posix() not in expected:
            source.unlink()


class TransferError(RuntimeError):
    """An infrastructure failure. It must not consume a CUDA repair attempt."""

    def __init__(self, message, *, artifacts=None, job_id=None):
        super().__init__(message)
        self.artifacts, self.job_id = artifacts or [], job_id


class WorkerJobError(RuntimeError):
    """A worker failure with the full report available as evidence."""

    def __init__(self, payload, artifact):
        super().__init__(payload.get("error", "Worker failed.") + "\nDetails: " + artifact["path"])
        self.payload, self.artifacts = payload, [artifact]


class _CommandFailure(RuntimeError):
    def __init__(self, completed):
        super().__init__(completed.stderr or completed.stdout or f"Exit code {completed.returncode}.")
        self.completed = completed


class _IntegrityFailure(RuntimeError):
    pass


def _file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _transient(error):
    if isinstance(error, (TimeoutError, _IntegrityFailure)):
        return True
    message = str(error).lower()
    permanent = ("permission denied", "host key verification failed", "remote host identification has changed",
                 "no space left", "read-only file system", "not a regular file", "authentication failed",
                 "bad permissions", "invalid format", "no such file or directory")
    if any(text in message for text in permanent):
        return False
    return any(text in message for text in (
        "connection reset", "connection timed out", "connection refused", "connection closed",
        "lost connection", "broken pipe", "network is unreachable", "no route to host",
        "temporary failure in name resolution", "operation timed out", "connection timeout"))


def _dispatch_failure(returncode, stdout, client_diagnostics):
    """Classify only explicit failures written by the local OpenSSH client."""
    if returncode != 255 or stdout.strip():
        return None
    lines = [line.strip() for line in client_diagnostics.splitlines()]
    # An authenticated connection might already have started the command.
    if any("Authenticated to " in line or "Sending command:" in line for line in lines):
        return None
    permanent = ("Host key verification failed.", "REMOTE HOST IDENTIFICATION HAS CHANGED")
    if any(any(message in line for message in permanent)
           or re.search(r"Permission denied \([a-z0-9,@_-]+\)\.?$", line, re.IGNORECASE)
           for line in lines):
        return {"retryable": False, "reason": "SSH authentication or host-key check failed before dispatch."}
    for line in lines:
        if (line == "Connection timed out during banner exchange"
                or re.fullmatch(r"ssh: connect to host .+ port \d+: (Connection refused|Connection timed out|Network is unreachable|No route to host)", line)
                or re.fullmatch(r"kex_exchange_identification: (read: Connection reset by peer|Connection closed by remote host)", line)
                or re.fullmatch(r"ssh: Could not resolve hostname .+: Temporary failure in name resolution", line)):
            return {"retryable": True, "reason": line}
    return None


class Worker:
    def __init__(self, config, root=None):
        self.config = config
        self.settings = config["worker"]
        self.root = Path(root or Path(__file__).resolve().parents[1]).resolve()
        self.local = self.settings["transport"] == "local"
        self.log_directory = self.root / ".local" / "transport"
        if not self.local:
            if "port" in self.settings and (type(self.settings["port"]) is not int
                                             or not 1 <= self.settings["port"] <= 65535):
                raise ValueError("Use an SSH port from 1 to 65535.")
            if not re.fullmatch(r"[A-Za-z0-9_.@-]+", self.settings["host"]) or self.settings["host"].startswith("-"):
                raise ValueError("Invalid SSH host.")
            path = PurePosixPath(self.settings["root"])
            if path.is_absolute() or ".." in path.parts or not re.fullmatch(r"[A-Za-z0-9_./-]+", str(path)):
                raise ValueError("Use a simple relative server path.")
        self.transfer_attempts = self.settings.get("transfer_attempts", 3)
        self.transfer_timeout = self.settings.get("transfer_timeout_seconds", 600)
        self.transfer_deadline = self.settings.get("transfer_deadline_seconds", 1800)
        if type(self.transfer_attempts) is not int or not 1 <= self.transfer_attempts <= 5:
            raise ValueError("Use one to five transfer attempts.")
        if self.transfer_timeout <= 0 or self.transfer_deadline <= 0:
            raise ValueError("Transfer time limits must be positive.")
        self._resident = None
        self._archive_worker = None
        if self.settings.get("persistent", False):
            atexit.register(self.close)

    def _session(self):
        if self._resident is None:
            from .session_client import SessionClient
            self._resident = SessionClient(self)
        return self._resident

    def close(self):
        if self._resident is not None:
            self._resident.close()
        if self._archive_worker is not None:
            self._archive_worker.close()

    def _pace_connection(self):
        """The paced driver overrides this for a new persistent SSH stream."""

    def _session_command(self):
        args = [self.settings["python"], "-u", "-m", "sako.session_worker", "--max-jobs",
                str(self.settings.get("resident_max_jobs", DEFAULT_RESIDENT_MAX_JOBS)), "--idle-seconds",
                str(self.settings.get("resident_idle_seconds", 600))]
        if self.local:
            return args
        cuda = self.settings["cuda_home"]
        command = (f"cd {shlex.quote(self.settings['root'])} && "
            f"export CUDA_HOME={shlex.quote(cuda)} PATH={shlex.quote(cuda + '/bin')}:\"$PWD/../.venv/bin:$PATH\" "
            f"CUDA_VISIBLE_DEVICES={int(self.settings['gpu'])} "
            "TORCHINDUCTOR_CACHE_DIR=\"$PWD/../cache/inductor\" TRITON_CACHE_DIR=\"$PWD/../cache/triton\" "
            "TORCH_EXTENSIONS_DIR=\"$PWD/../cache/extensions\" HF_HUB_DISABLE_XET=1 TORCHINDUCTOR_COMPILE_THREADS=2; "
            + " ".join(shlex.quote(value) for value in args))
        return self.ssh() + [command]

    def _connection_options(self, port_flag):
        options = ["-i", str(Path(self.settings["identity"]).expanduser()),
                   "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=20"]
        if self.settings.get("ssh_config"):
            options.extend(["-F", str(Path(self.settings["ssh_config"]).expanduser().resolve())])
        if "port" in self.settings:
            options.extend([port_flag, str(self.settings["port"])])
        if self.settings.get("known_hosts"):
            path = Path(self.settings["known_hosts"]).expanduser().resolve().as_posix()
            if any(character in path for character in ('"', "\n", "\r")):
                raise ValueError("Invalid SSH host key file path.")
            options.extend(["-o", f'UserKnownHostsFile="{path}"'])
        return options

    def ssh(self):
        return ["ssh", *self._connection_options("-p"), self.settings["host"]]

    def scp(self):
        return ["scp", "-q", *self._connection_options("-P")]

    def _remote_path(self, relative):
        path = PurePosixPath(relative)
        if (path.is_absolute() or ".." in path.parts or not path.parts
                or not re.fullmatch(r"[A-Za-z0-9_./-]+", str(path))):
            raise ValueError("Use a simple relative transfer path.")
        return str(PurePosixPath(self.settings["root"]) / path)

    def _logged_command(self, command, timeout, *, env=None, log_name=None, input_text=None):
        path = self.log_directory / ((log_name or uuid.uuid4().hex) + ".json")
        record = {"command": command, "timeout_seconds": timeout, "status": "pending"}
        write_json(path, record)
        try:
            options = {"env": env}
            if input_text is not None:
                options["input_text"] = input_text
            completed = run_command(command, timeout=timeout, **options)
        except Exception as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}",
                          stdout=getattr(error, "stdout", ""), stderr=getattr(error, "stderr", ""))
            write_json(path, record)
            raise
        record.update(status="complete", returncode=completed.returncode,
                      stdout=completed.stdout, stderr=completed.stderr)
        write_json(path, record)
        return completed

    def command(self, command, timeout=120):
        result = self._logged_command(self.ssh() + [command], timeout)
        if result.returncode:
            raise _CommandFailure(result)
        return result.stdout

    def _transfer(self, operation, callback):
        transfer_id = uuid.uuid4().hex
        report = self.log_directory / (transfer_id + ".transfer.json")
        start = time.monotonic()
        attempts = []
        last_error = None
        for number in range(1, self.transfer_attempts + 1):
            remaining = self.transfer_deadline - (time.monotonic() - start)
            if remaining <= 0:
                last_error = TimeoutError("The transfer deadline expired.")
                break
            attempt_end = time.monotonic() + min(self.transfer_timeout, remaining)
            steps = []

            def step(command, *, input_text=None):
                timeout = attempt_end - time.monotonic()
                if timeout <= 0:
                    raise TimeoutError("The transfer attempt timed out.")
                name = f"{transfer_id}-{number}-{len(steps) + 1}"
                steps.append(name + ".json")
                options = {"log_name": name}
                if input_text is not None:
                    options["input_text"] = input_text
                result = self._logged_command(command, timeout, **options)
                if result.returncode:
                    raise _CommandFailure(result)
                return result.stdout

            entry = {"attempt": number, "commands": steps}
            try:
                callback(step)
                entry["status"] = "complete"
                attempts.append(entry)
                write_json(report, {"operation": operation, "status": "complete", "attempts": attempts})
                return
            except Exception as error:
                last_error = error
                retry = _transient(error)
                entry.update(status="failed", error=f"{type(error).__name__}: {error}", transient=retry)
                attempts.append(entry)
                write_json(report, {"operation": operation, "status": "failed", "attempts": attempts})
                if not retry or number == self.transfer_attempts:
                    break
                delay = min(2 ** number + random.uniform(0, .5),
                            self.transfer_deadline - (time.monotonic() - start))
                if delay <= 0:
                    break
                time.sleep(delay)
        write_json(report, {"operation": operation, "status": "exhausted", "attempts": attempts,
                            "error": str(last_error)})
        paths = [report, *[self.log_directory / name for item in attempts for name in item["commands"]]]
        evidence = [artifact_descriptor(self.root, path) for path in paths if path.exists()]
        raise TransferError(f"{operation} failed after {len(attempts)} attempt(s). Details: "
                            + report.relative_to(self.root).as_posix(), artifacts=evidence) from last_error

    def put(self, source, relative):
        if self.settings.get("persistent", False) and relative.startswith("runs/"):
            from .session_protocol import MAX_FILE
            if Path(source).stat().st_size <= MAX_FILE:
                self._session().stage(source, relative)
                return
        if self.local:
            return
        source = Path(source).resolve()
        remote = self._remote_path(relative)
        expected = _file_hash(source)
        temporary = remote + "." + uuid.uuid4().hex + ".part"

        if self.settings.get("persistent", False):
            # Bootstrap the evaluator before its Python stream is available.
            # Keep hash checking and atomic publication in one SSH request.
            data = base64.b64encode(source.read_bytes()).decode("ascii")
            check_line = expected + "  " + temporary + "\n"
            command = ("mkdir -p -- " + shlex.quote(str(PurePosixPath(remote).parent)) +
                " && base64 -d > " + shlex.quote(temporary) +
                " && printf %s " + shlex.quote(check_line) + " | sha256sum -c - >/dev/null" +
                " && mv -f -- " + shlex.quote(temporary) + " " + shlex.quote(remote))
            self._transfer("Upload " + relative, lambda step: step(self.ssh() + [command], input_text=data))
            return

        def upload(step):
            step(self.ssh() + ["mkdir -p -- " + shlex.quote(str(PurePosixPath(remote).parent))])
            step(self.scp() + [str(source), self.settings["host"] + ":" + temporary])
            raw = step(self.ssh() + ["sha256sum -- " + shlex.quote(temporary)])
            if not raw.split() or raw.split()[0] != expected:
                raise _IntegrityFailure("The uploaded file hash differs.")
            step(self.ssh() + ["mv -f -- " + shlex.quote(temporary) + " " + shlex.quote(remote)])
        self._transfer("Upload " + relative, upload)

    def get(self, relative, destination, *, job_hash=None):
        if self.settings.get("persistent", False):
            return self._session().get(relative, destination, job_hash=job_hash)
        return self._get_scp(relative, destination, job_hash=job_hash)

    def _get_scp(self, relative, destination, *, job_hash=None):
        """Fetch one existing artifact. This method never launches a worker job."""
        destination = Path(destination)
        if self.local:
            return destination
        remote = self._remote_path(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".part")

        def download(step):
            raw = step(self.ssh() + ["sha256sum -- " + shlex.quote(remote)])
            expected = raw.split()[0] if raw.split() else ""
            if not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise _IntegrityFailure("The remote file hash is invalid.")
            step(self.scp() + [self.settings["host"] + ":" + remote, str(temporary)])
            if _file_hash(temporary) != expected:
                raise _IntegrityFailure("The downloaded file hash differs.")
            if job_hash is not None and read_json(temporary).get("job_hash") != job_hash:
                raise ValueError("Worker result does not match the submitted job.")
            temporary.replace(destination)
        self._transfer("Download " + relative, download)
        return destination

    def sync(self):
        """Use an allowlist. Do not transfer .git, runs, credentials, or weights."""
        self.close()
        self._resident = None
        if self.local:
            return {"transport": "local"}
        archive = self.root / ".local/project.tar.gz"
        archive.parent.mkdir(exist_ok=True)
        allowed = [self.root / n for n in ("sako", "configs", "scripts", "tests")]
        with tarfile.open(archive, "w:gz") as tar:
            for top in allowed:
                for path in top.rglob("*"):
                    if path.is_file() and "__pycache__" not in path.parts and path.suffix in {".py", ".json", ".txt", ".sh"}:
                        tar.add(path, arcname=path.relative_to(self.root).as_posix())
            tar.add(self.root / "pyproject.toml", arcname="pyproject.toml")
        self.put(archive, "project.tar.gz")
        remote = shlex.quote(self.settings["root"])
        package = self.root / "sako"
        expected = sorted(p.relative_to(package).as_posix()
                          for p in [*package.glob("*.py"), *package.glob("prompts/*.txt")])
        cleanup = "from sako.transport import prune_package_sources; import json, sys; prune_package_sources('.', json.loads(sys.argv[1]))"
        self.command(f"cd {remote} && tar -xzf project.tar.gz && " +
                     shlex.join([self.settings["python"], "-c", cleanup, json.dumps(expected)]))
        return {"transport": "ssh", "root": self.settings["root"], "kernel_backend": "triton"}

    def _dispatch_artifacts(self, pending):
        paths = [self.root / item[key] for item in pending.get("dispatch_history", [])
                 for key in ("log", "client_log") if item.get(key)]
        return [artifact_descriptor(self.root, path) for path in paths if path.exists()]

    def _dispatch_remote(self, pending, pending_path, remote_command, timeout):
        """Retry proven startup failures. Never repeat an uncertain remote command."""
        limit = min(self.transfer_attempts, 3)
        history = pending.setdefault("dispatch_history", [])
        used = pending.get("dispatch_attempts", 0)
        if type(used) is not int or not 0 <= used <= 3:
            raise TransferError("Invalid saved dispatch attempt count.", job_id=pending["job_id"])
        for attempt in range(used + 1, limit + 1):
            stem = f"{pending['job_id']}-dispatch-{attempt}-{uuid.uuid4().hex[:8]}"
            client_path = self.log_directory / (stem + ".ssh.log")
            client_path.parent.mkdir(parents=True, exist_ok=True)
            entry = {"attempt": attempt, "status": "dispatching",
                     "log": (self.log_directory / (stem + ".json")).relative_to(self.root).as_posix(),
                     "client_log": client_path.relative_to(self.root).as_posix()}
            history.append(entry)
            pending.update(status="dispatching", dispatch_attempts=attempt)
            write_json(pending_path, pending)
            command = self.ssh()
            # -E separates local client diagnostics from remote stdout/stderr.
            command[1:1] = ["-E", str(client_path), "-o", "LogLevel=VERBOSE"]
            try:
                completed = self._logged_command(command + [remote_command], timeout + 30, log_name=stem)
            except CommandTimeout:
                entry["status"] = pending["status"] = "awaiting_result"
                write_json(pending_path, pending)
                return
            except OSError as error:
                entry.update(status="not_dispatched", retryable=False, reason=f"SSH could not start: {error}")
                pending["status"] = "not_dispatched"
                write_json(pending_path, pending)
                raise TransferError("SSH could not start. See the saved command log.",
                                    artifacts=self._dispatch_artifacts(pending), job_id=pending["job_id"]) from error
            try:
                diagnostics = client_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                diagnostics = ""
            failure = _dispatch_failure(completed.returncode, completed.stdout, diagnostics)
            if failure is None:
                entry["status"] = pending["status"] = "awaiting_result"
                write_json(pending_path, pending)
                return
            entry.update(status="not_dispatched", **failure)
            pending["status"] = "not_dispatched"
            write_json(pending_path, pending)
            if not failure["retryable"]:
                raise TransferError(failure["reason"], artifacts=self._dispatch_artifacts(pending), job_id=pending["job_id"])
            if attempt < limit:
                time.sleep(min(2 ** attempt, 5) + random.uniform(0, .5))
        raise TransferError(f"SSH startup failed before dispatch after {pending['dispatch_attempts']} total attempts. "
                            "See the saved client logs.", artifacts=self._dispatch_artifacts(pending), job_id=pending["job_id"])

    def archive_peer(self):
        if self._archive_worker is None:
            settings = dict(self.settings)
            settings["gpu"] = settings.pop("archive_gpu")
            self._archive_worker = type(self)({**self.config, "worker": settings}, self.root)
        return self._archive_worker

    def run(self, action, run_dir, **fields):
        share = action == "profile" or (action == "evaluate" and self.config.get("generation_objective") != "decode")
        if share and self.settings.get("archive_gpu") is not None:
            from .archive_gpu import run_paired
            return run_paired(self, self.archive_peer(), action, Path(run_dir).resolve(), fields)
        return self._run_action(action, run_dir, **fields)

    def _run_action(self, action, run_dir, **fields):
        if self.settings.get("persistent", False):
            return self._session().run(action, run_dir, fields)
        run_dir = Path(run_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        run_dir.relative_to(self.root)
        self.log_directory = run_dir / "jobs" / "transfers"
        config_path = run_dir / "config.json"
        relative = config_path.relative_to(self.root).as_posix()
        local_implementation = implementation_hash()
        job = {"action": action, "config": relative, "implementation_hash": local_implementation, **fields}
        identity = digest({"job": job, "config": self.config})
        pending_path = run_dir / "jobs" / (identity + ".pending.json")
        pending = read_json(pending_path) if pending_path.exists() else None
        if pending is not None and (pending.get("job_hash") != digest(job)
                or not re.fullmatch(r"[0-9a-f]{32}", pending.get("job_id", ""))):
            raise TransferError("The saved dispatch does not match the current job.")
        job_id = pending["job_id"] if pending else uuid.uuid4().hex
        job_path = run_dir / "jobs" / (job_id + ".json")
        result_path = run_dir / "jobs" / (job_id + ".result.json")
        job_relative, result_relative = (p.relative_to(self.root).as_posix() for p in (job_path, result_path))
        is_new = pending is None
        if pending is not None:
            if pending.get("result") != result_relative or not job_path.exists() or read_json(job_path) != job:
                raise TransferError("The saved job source or result path changed.", job_id=job_id)
            if pending.get("status") not in {"ready", "dispatched", "dispatching", "not_dispatched", "awaiting_result"}:
                raise TransferError("The saved dispatch state is invalid.", job_id=job_id)
        else:
            write_json(config_path, self.config)
            self.put(config_path, relative)
            write_json(job_path, job)
            self.put(job_path, job_relative)
            pending = {"job_id": job_id, "job_hash": digest(job), "status": "ready",
                       "result": result_relative, "dispatch_attempts": 0, "dispatch_history": []}
            write_json(pending_path, pending)
        timeout = self.settings["timeout_seconds"]
        command = [self.settings["python"], "-m", "sako.worker", "--job", job_relative, "--output", result_relative]
        if self.local and is_new:
            pending["status"] = "dispatched"
            write_json(pending_path, pending)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(self.settings["gpu"])
            try:
                self._logged_command(command, timeout, env=env, log_name=job_id)
            except CommandTimeout:
                if not result_path.exists():
                    raise TransferError("Worker timed out without a result. Recover the existing job.", job_id=job_id)
            except OSError as error:
                raise TransferError("Worker could not start. See the saved command log.", job_id=job_id) from error
        elif not self.local:
            cuda = self.settings["cuda_home"]
            command_text = " ".join(shlex.quote(x) for x in command)
            remote_command = (f"cd {shlex.quote(self.settings['root'])} && "
                f"export CUDA_HOME={shlex.quote(cuda)} PATH={shlex.quote(cuda + '/bin')}:\"$PWD/../.venv/bin:$PATH\" "
                f"CUDA_VISIBLE_DEVICES={int(self.settings['gpu'])} "
                "TORCHINDUCTOR_CACHE_DIR=\"$PWD/../cache/inductor\" TRITON_CACHE_DIR=\"$PWD/../cache/triton\" "
                "TORCH_EXTENSIONS_DIR=\"$PWD/../cache/extensions\" HF_HUB_DISABLE_XET=1 TORCHINDUCTOR_COMPILE_THREADS=2; "
                f"timeout --signal=TERM --kill-after=10 {timeout} {command_text}")
            if pending["status"] in {"ready", "not_dispatched"}:
                self._dispatch_remote(pending, pending_path, remote_command, timeout)
            # Dispatching, dispatched, and awaiting_result are all uncertain outcomes.
            # Only retrieve their existing result.
            try:
                self.get(result_relative, result_path, job_hash=digest(job))
            except TransferError as error:
                error.job_id = job_id
                raise
        if not result_path.exists():
            raise TransferError("Worker has no saved result. Recover the existing job.", job_id=job_id)
        payload = read_json(result_path)
        if payload.get("job_hash") != digest(job):
            raise TransferError("Worker result does not match the submitted job.", job_id=job_id)
        if payload.get("provenance", {}).get("implementation_hash") != local_implementation:
            raise TransferError("Worker code differs from the controller. Sync the project.", job_id=job_id)
        pending_path.unlink(missing_ok=True)
        if not payload["ok"]:
            if payload.get("failure_kind") == "infrastructure":
                raise TransferError(payload.get("error", "Worker infrastructure failed."),
                                    artifacts=[artifact_descriptor(self.root, result_path)], job_id=job_id)
            raise WorkerJobError(payload, artifact_descriptor(self.root, result_path))
        result = payload["result"]
        if isinstance(result, dict) and "provenance" in payload:
            result = {**result, "worker_provenance": payload["provenance"],
                      "worker_total_s": payload.get("worker_total_s"), "session_times": payload.get("session_times")}
        return result
