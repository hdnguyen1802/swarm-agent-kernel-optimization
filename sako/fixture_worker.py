"""Run native fixture trials without loading the resident reference model."""

import contextlib
import json
import multiprocessing
from multiprocessing.connection import wait
import os
from pathlib import Path
import threading
import traceback

from .core import read_json
from .worker import InfrastructureError, evaluate_operation


def serve(connection):
    # The resident worker owns the GPU lock and the job watchdog. Stop this
    # process too if the owner exits during a native call or a job timeout.
    parent = multiprocessing.parent_process()

    def stop_with_parent():
        wait([parent.sentinel])
        os._exit(1)

    threading.Thread(target=stop_with_parent, daemon=True).start()
    from .gpu import build_log
    with connection:
        while True:
            try:
                job, config, log_path = json.loads(connection.recv_bytes())
            except EOFError:
                return
            try:
                with build_log(log_path) if log_path else contextlib.nullcontext():
                    catalog = read_json(Path(job["profile_dir"]) / "tasks.json")
                    result = evaluate_operation(job, config, catalog, fixture_only=True)
                    import torch
                    torch.cuda.empty_cache()
                reply = {"result": result}
            except Exception as error:
                reply = {"error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc(),
                         "infrastructure": isinstance(error, InfrastructureError),
                         "build_log": str(error.build_log) if getattr(error, "build_log", None) else None}
            connection.send_bytes(json.dumps(reply).encode())


class FixtureWorker:
    def __init__(self):
        # Fork would inherit the parent's initialized CUDA state.
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=serve, args=(child,), name="sako-fixtures")
        try:
            self.process.start()
        except BaseException:
            self.connection.close()
            raise
        finally:
            child.close()

    def evaluate(self, job, config, log_path):
        try:
            self.connection.send_bytes(json.dumps([job, config, str(log_path) if log_path else None]).encode())
            reply = json.loads(self.connection.recv_bytes())
        except (EOFError, OSError) as error:
            raise InfrastructureError("The fixture worker disconnected during evaluation.") from error
        if "error" in reply:
            kind = InfrastructureError if reply["infrastructure"] else RuntimeError
            error = kind(f"{reply['error']}\n{reply['traceback']}")
            error.build_log = reply["build_log"]
            raise error
        return reply["result"]

    def close(self):
        self.connection.close()
        self.process.join(timeout=3)
        if self.process.is_alive():
            self.process.kill()
            self.process.join()
        self.process.close()
