"""Run commands without a shell and stop failed worker groups."""

from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path


class CommandTimeout(TimeoutError):
    """Keep complete command output when a process exceeds its time limit."""

    def __init__(self, command, timeout, stdout, stderr, log_path=None):
        super().__init__(f"Command exceeded {timeout} seconds.")
        self.command, self.stdout, self.stderr = command, stdout, stderr
        self.log_path = str(log_path) if log_path else None


def _save_output(log_path, stdout, stderr):
    if log_path is not None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("STDOUT\n" + stdout + "\nSTDERR\n" + stderr, encoding="utf-8")


def run_command(command: list[str], *, input_text: str | None = None,
                timeout: float = 120, env: dict | None = None,
                log_path: str | Path | None = None) -> subprocess.CompletedProcess:
    if not command or any(not isinstance(item, str) or not item for item in command):
        raise ValueError("Provide a command as a non-empty list of strings.")
    if timeout <= 0:
        raise ValueError("Command timeout must be positive.")
    options = {"start_new_session": True} if os.name != "nt" else {
        "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP
    }
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                               env=env, **options)
    try:
        stdout, stderr = process.communicate(input=input_text, timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           capture_output=True, check=False)
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.kill()
        stdout, stderr = process.communicate()
        _save_output(log_path, stdout, stderr)
        raise CommandTimeout(command, timeout, stdout, stderr, log_path) from None
    _save_output(log_path, stdout, stderr)
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
