"""Manage one named Colab runtime and its private CLI state."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid

from ..core import read_json, write_json


# Each choice specifies the CLI accelerator and the machine shape.
GPUS = {
    "a100-40": ("A100 40 GB", "A100", False),
    "a100-80": ("A100 80 GB", "A100", True),
    "l4": ("L4", "L4", False),
    "t4": ("T4", "T4", False),
    "g4": ("G4 / RTX PRO 6000", "G4", False),
    "h100": ("H100", "H100", False),
}


def cli_command(root):
    if override := os.environ.get("SAKO_COLAB_CLI"):
        executable = shutil.which(override)
        if not executable:
            raise ValueError("SAKO_COLAB_CLI must name the Colab executable.")
        return [executable]
    private_cli = root / ".local" / "run_colab_cli.py"
    private_python = root / ".local/colab-cli-env" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if private_cli.is_file() and private_python.is_file():
        return [str(private_python), str(private_cli)]
    if importlib.util.find_spec("colab_cli") is not None:
        return [sys.executable, "-m", "colab_cli.cli"]
    if executable := shutil.which("colab"):
        return [executable]
    raise ValueError('Install the Colab CLI with python -m pip install -e ".[colab]". Then sign in with colab usage.')


def verify_gpu(choice, name, memory_mib):
    _, accelerator, _ = GPUS[choice]
    expected = "RTX PRO 6000" if accelerator == "G4" else accelerator
    if not re.search(r"\b" + expected + r"\b", name, re.IGNORECASE):
        raise ValueError(f"Colab assigned {name}. The request was for {GPUS[choice][0]}.")
    if accelerator == "A100":
        gib = 80 if choice == "a100-80" else 40
        if not gib * 970 <= memory_mib <= gib * 1030:
            raise ValueError(f"Colab assigned {name} with {memory_mib} MiB. The request was for {gib} GB.")


def error_message(action, output):
    low = output.lower()
    if "service unavailable" in low:
        return "Colab returned Service Unavailable. Try this GPU later or select another GPU."
    if "rejected accelerator" in low or "allocation refused" in low:
        return "Colab refused the GPU request. Check the account quota or select another GPU."
    if any(s in low for s in ("credential", "unauthorized", "invalid_grant", "sign in", "eof when reading")):
        return "Colab could not sign in. Sign in with the same Colab CLI, then retry."
    # Provider tracebacks can contain signed URLs. Keep them out of run evidence.
    return f"Colab could not complete {action}."


class ColabSession:
    def __init__(self, root, attempt, choice, *, resume_cleanup=False):
        self.root, self.attempt, self.choice = root, attempt, choice
        self.receipt = attempt / "colab.json"
        if resume_cleanup:
            self.record = read_json(self.receipt)
        else:
            self.record = {"name": "sako-" + uuid.uuid4().hex, "requested": GPUS[choice][0],
                           "choice": choice, "allocation_attempted": False, "released": False}
        self.name = self.record["name"]
        if not re.fullmatch(r"sako-[0-9a-f]{32}", self.name):
            raise ValueError("Invalid saved Colab session name.")
        self.private = root / ".local" / "colab-sessions" / self.name
        self.cli = [*cli_command(root), "--config", str(self.private / "sessions.json")]
        self.config = None

    def save(self, **values):
        self.record.update(values)
        write_json(self.receipt, self.record)

    def call(self, *args, timeout=90):
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "COLAB_CLI_REQUEST_TIMEOUT": "55",
               "COLAB_CLI_ASSIGN_RETRIES": "0"}
        try:
            response = subprocess.run([*self.cli, *args], cwd=self.root, env=env,
                                      stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout,
                                      encoding="utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Colab timed out during {args[0]}.") from None
        if response.returncode:
            raise RuntimeError(error_message(args[0], response.stdout + response.stderr))
        return response.stdout

    def sessions(self):
        output = self.call("sessions")
        rows = re.findall(r"^\[([^]]+)\] (\S+) \| Hardware:", output, re.MULTILINE)
        if not rows and "No active sessions found on server" not in output:
            raise RuntimeError("Colab did not return a session list.")
        if not rows and not re.search(r"^Active assignments: 0\s*$", self.call("usage"), re.MULTILINE):
            raise RuntimeError("Colab could not confirm the account's session count.")
        return rows

    def allocate(self):
        self.private.mkdir(mode=0o700, parents=True, exist_ok=False)
        self.save()
        before = self.sessions()
        _, accelerator, high_ram = GPUS[self.choice]
        self.save(allocation_attempted=True, before=[self.endpoint_hash(endpoint) for _, endpoint in before])
        args = ["new", "-s", self.name, "--gpu", accelerator]
        if high_ram:
            args.append("--high-mem")
        self.call(*args, timeout=120)
        self.save(allocated=True)
        owned = dict(self.sessions()).get(self.name)
        if not owned:
            raise RuntimeError("Colab did not confirm the named session.")
        self.save(endpoint_hash=self.endpoint_hash(owned))

    @staticmethod
    def endpoint_hash(endpoint):
        return hashlib.sha256(endpoint.encode()).hexdigest()

    def connect(self, base):
        from scripts.configure_colab import host_key_record

        script = self.root / "scripts/colab_gpu_info.py"
        for retry in range(2):
            try:
                output = self.call("exec", "-s", self.name, "-f", str(script), "--timeout", "60")
                marker = next(line[9:] for line in output.splitlines() if line.startswith("SAKO_GPU="))
                info = json.loads(marker)
                break
            except (RuntimeError, StopIteration, ValueError):
                if retry:
                    raise RuntimeError("Could not read the GPU facts from the new Colab runtime.") from None
                time.sleep(2)
        gpu, memory = info["gpu"], info["memory_mib"]
        self.save(gpu=gpu, memory_mib=memory)
        verify_gpu(self.choice, gpu, memory)
        record = host_key_record({**info, "host": self.name, "port": 22, "user": "root"})
        hosts = self.private / "known_hosts"
        hosts.write_text(record, encoding="utf-8")
        identity = self.private / "id_ed25519"
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(identity)],
                       check=True, stdin=subprocess.DEVNULL, timeout=20)
        # In proxy mode an unnamed connection requires an existing session. The
        # private store contains exactly this allocation. A missing name passed
        # with -s would instead ask the CLI to create another runtime.
        args = [*self.cli, "ssh", "--proxy-mode", "-i", str(identity)]
        proxy = subprocess.list2cmdline(args) if os.name == "nt" else shlex.join(args)
        if "\n" in proxy or "\r" in proxy:
            raise ValueError("Invalid path in the Colab SSH configuration.")
        ssh_config = self.private / "ssh_config"
        ssh_config.write_text(f"Host {self.name}\n  IdentitiesOnly yes\n  ProxyCommand "
                              + proxy.replace("%", "%%") + "\n", encoding="utf-8")
        worker = {**base["worker"], "transport": "ssh", "host": "root@" + self.name, "port": 22, "gpu": 0,
                  "identity": str(identity), "known_hosts": str(hosts), "ssh_config": str(ssh_config),
                  "root": "sako-colab/project", "python": "/content/sako-venv/bin/python",
                  "cuda_home": "/usr/local/cuda", "persistent": True}
        self.config = {**base, "worker": worker}
        return self.config

    def install(self):
        script = self.root / "scripts/colab_install.py"
        output = self.call("exec", "-s", self.name, "-f", str(script), "--timeout", "1800", timeout=1830)
        payload = next((line.removeprefix("SAKO_INSTALL=") for line in output.splitlines()
                        if line.startswith("SAKO_INSTALL=")), None)
        if payload is None:
            raise RuntimeError("Colab did not return the installer result.")
        result = json.loads(payload)
        print(result["log"], flush=True)
        if result["exit_code"] != 0:
            raise RuntimeError(f"The GPU package installer returned exit code {result['exit_code']}. See the log.")

    def collect(self):
        # This path also works after a worker install or SSH failure.
        output = self.call("exec", "-s", self.name, "-f", str(self.root / "scripts/colab_collect.py"),
                           "--timeout", "180", timeout=210)
        payload = next((line.removeprefix("SAKO_ARCHIVE=") for line in output.splitlines()
                        if line.startswith("SAKO_ARCHIVE=")), None)
        if payload is None:
            raise RuntimeError("Colab did not return an evidence checksum.")
        info = json.loads(payload)
        temp = self.attempt / "gpu-evidence.tar.gz.part"
        for attempt in range(2):
            try:
                self.call("download", "-s", self.name, "content/sako-gpu-evidence.tar.gz", str(temp), timeout=300)
                digest = hashlib.sha256()
                with temp.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest() != info["sha256"] or temp.stat().st_size != info["bytes"]:
                    raise ValueError("The evidence checksum differs.")
                target = self.attempt / "gpu-evidence.tar.gz"
                temp.replace(target)
                self.save(evidence=target.name, evidence_sha256=info["sha256"], evidence_bytes=info["bytes"])
                return
            except (OSError, RuntimeError, ValueError):
                if attempt:
                    raise

    def release(self):
        if not self.record.get("allocation_attempted") or self.record.get("released"):
            return
        error = None
        for retry in range(3):
            try:
                rows = self.sessions()
                named = dict(rows).get(self.name)
                expected = self.record.get("endpoint_hash")
                if named and expected and self.endpoint_hash(named) != expected:
                    raise RuntimeError("The named Colab session has changed.")
                if named:
                    self.record["endpoint_hash"] = self.endpoint_hash(named)
                    self.call("stop", "-s", self.name)
                    rows = self.sessions()
                endpoints = {self.endpoint_hash(endpoint) for _, endpoint in rows}
                if self.name in dict(rows) or self.record.get("endpoint_hash") in endpoints:
                    raise RuntimeError("Colab still lists the test session.")
                if not self.record.get("endpoint_hash") and endpoints.difference(self.record.get("before", [])):
                    raise RuntimeError("Colab lists a new session with no confirmed owner. Check Colab sessions.")
                self.save(released=True, release_error=None)
                for name in ("ssh_config", "id_ed25519", "id_ed25519.pub"):
                    (self.private / name).unlink(missing_ok=True)
                return
            except (OSError, RuntimeError) as exc:
                error = str(exc)
                if retry < 2:
                    time.sleep(1 + retry)
        self.save(release_error=error)
        raise RuntimeError(f"Could not confirm release of {self.name}. {error}")
