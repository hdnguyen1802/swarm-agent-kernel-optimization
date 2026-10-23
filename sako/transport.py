"""Transfer jobs through SSH. Keep cloud credentials on the controller."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import tarfile
import uuid
from pathlib import Path, PurePosixPath

from .core import read_json, write_json, digest
from .process import run_command


class Worker:
    def __init__(self, config, root=None):
        self.config = config
        self.settings = config["worker"]
        self.root = Path(root or Path(__file__).resolve().parents[1])
        self.local = self.settings["transport"] == "local"
        if not self.local:
            import re
            if not re.fullmatch(r"[A-Za-z0-9_.@-]+", self.settings["host"]) or self.settings["host"].startswith("-"):
                raise ValueError("Invalid SSH host.")
            path = PurePosixPath(self.settings["root"])
            if path.is_absolute() or ".." in path.parts or not re.fullmatch(r"[A-Za-z0-9_./-]+", str(path)):
                raise ValueError("Use a simple relative server path.")

    def ssh(self):
        return ["ssh", "-i", str(Path(self.settings["identity"]).expanduser()),
                "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", self.settings["host"]]

    def command(self, command, timeout=120):
        result = run_command(self.ssh() + [command], timeout=timeout)
        if result.returncode:
            raise RuntimeError(result.stderr[-12000:] or result.stdout[-12000:])
        return result.stdout

    def put(self, source, relative):
        if self.local:
            return
        remote = str(PurePosixPath(self.settings["root"]) / relative)
        self.command("mkdir -p " + shlex.quote(str(PurePosixPath(remote).parent)))
        command = ["scp", "-q", "-i", str(Path(self.settings["identity"]).expanduser()),
                   "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", str(source),
                   self.settings["host"] + ":" + remote]
        result = run_command(command, timeout=600)
        if result.returncode:
            raise RuntimeError(result.stderr[-4000:])

    def sync(self):
        """Use an allowlist. Do not transfer .git, runs, credentials, or weights."""
        if self.local:
            return {"transport": "local"}
        archive = self.root / ".local/project.tar.gz"
        archive.parent.mkdir(exist_ok=True)
        allowed = [self.root / n for n in ("sako", "configs", "references", "scripts", "tests")]
        with tarfile.open(archive, "w:gz") as tar:
            for top in allowed:
                for path in top.rglob("*"):
                    if path.is_file() and "__pycache__" not in path.parts and path.suffix in {".py", ".json", ".txt", ".sh"}:
                        tar.add(path, arcname=path.relative_to(self.root).as_posix())
            tar.add(self.root / "pyproject.toml", arcname="pyproject.toml")
        self.put(archive, "project.tar.gz")
        remote = shlex.quote(self.settings["root"])
        self.command(f"cd {remote} && tar -xzf project.tar.gz")
        commit = read_json(self.root / "references/catalog.json")["commit"]
        self.command(f"cd {remote} && mkdir -p third_party && "
                     "(test -d third_party/cutlass/.git || git clone --filter=blob:none --no-checkout "
                     "https://github.com/NVIDIA/cutlass.git third_party/cutlass) && "
                     f"git -C third_party/cutlass fetch --depth 1 origin {shlex.quote(commit)} && "
                     f"git -C third_party/cutlass checkout --detach {shlex.quote(commit)}", timeout=600)
        return {"transport": "ssh", "root": self.settings["root"], "cutlass_commit": commit}

    def run(self, action, run_dir, **fields):
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        job_id = uuid.uuid4().hex
        config_path = run_dir / "config.json"
        write_json(config_path, self.config)
        relative = config_path.relative_to(self.root).as_posix()
        self.put(config_path, relative)
        job = {"action": action, "config": relative, **fields}
        job_path = run_dir / "jobs" / (job_id + ".json")
        result_path = run_dir / "jobs" / (job_id + ".result.json")
        write_json(job_path, job)
        job_relative, result_relative = (p.relative_to(self.root).as_posix() for p in (job_path, result_path))
        self.put(job_path, job_relative)
        timeout = self.settings["timeout_seconds"]
        command = [self.settings["python"], "-m", "sako.worker", "--job", job_relative, "--output", result_relative]
        if self.local:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(self.settings["gpu"])
            completed = run_command(command, timeout=timeout, env=env)
        else:
            cuda = self.settings["cuda_home"]
            # The remote timeout kills the job even if the SSH client exits.
            command_text = " ".join(shlex.quote(x) for x in command)
            remote_command = (f"cd {shlex.quote(self.settings['root'])} && "
                f"export CUDA_HOME={shlex.quote(cuda)} PATH={shlex.quote(cuda + '/bin')}:\"$PWD/../.venv/bin:$PATH\" "
                f"CUDA_VISIBLE_DEVICES={int(self.settings['gpu'])} "
                "TORCHINDUCTOR_CACHE_DIR=\"$PWD/../cache/inductor\" TRITON_CACHE_DIR=\"$PWD/../cache/triton\" "
                "TORCH_EXTENSIONS_DIR=\"$PWD/../cache/extensions\" HF_HUB_DISABLE_XET=1 TORCHINDUCTOR_COMPILE_THREADS=2; "
                f"timeout --signal=TERM --kill-after=10 {timeout} {command_text}")
            completed = run_command(self.ssh() + [remote_command], timeout=timeout + 30)
            (run_dir / "jobs" / (job_id + ".log")).write_text(completed.stdout + completed.stderr, encoding="utf-8")
            try:
                raw = self.command("cat " + shlex.quote(str(PurePosixPath(self.settings["root"]) / result_relative)))
            except RuntimeError as error:
                raise RuntimeError(f"Worker exited with code {completed.returncode} without a result. "
                                   + completed.stderr[-6000:]) from error
            result_path.write_text(raw, encoding="utf-8")
        (run_dir / "jobs" / (job_id + ".log")).write_text(completed.stdout + completed.stderr, encoding="utf-8")
        if not result_path.exists():
            raise RuntimeError(f"Worker did not return a result. {completed.stderr[-3000:]}")
        payload = read_json(result_path)
        if payload.get("job_hash") != digest(job):
            raise RuntimeError("Worker result does not match the submitted job.")
        if not payload["ok"]:
            raise RuntimeError(payload.get("error", "Worker failed."))
        return payload["result"]
