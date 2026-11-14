from types import SimpleNamespace
import logging
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sako.loop import SearchLoop
from sako.triton_resources import kernel_resources, observe_resources
from sako.gpu import build_log


class TritonResourceTests(unittest.TestCase):
    def test_library_flush_does_not_keep_a_closed_job_file(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = logging.getLogger("sako-test-library-flush")
            logger.propagate = False
            logger.setLevel(logging.WARNING)
            first, second = Path(directory) / "first.log", Path(directory) / "second.log"
            try:
                with build_log(first):
                    handler = logging.StreamHandler()
                    handler.flush = sys.stderr.flush
                    logger.addHandler(handler)
                    logger.warning("first job")
                handler.flush()
                with build_log(second):
                    logger.warning("second job")
                self.assertIn("second job", second.read_text())
                self.assertNotIn("Logging error", second.read_text())
            finally:
                logger.handlers.clear()
                if "handler" in locals():
                    vars(handler).pop("flush", None)

    def test_nested_build_logs_leave_new_handlers_valid_for_the_next_job(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = logging.getLogger("sako-test-nested-build-stream")
            logger.propagate = False
            logger.setLevel(logging.WARNING)
            paths = [Path(directory) / name for name in ("outer.log", "inner.log", "next.log")]
            try:
                with build_log(paths[0]):
                    with build_log(paths[1]):
                        handler = logging.StreamHandler()
                        logger.addHandler(handler)
                        logger.warning("inner job")
                    self.assertFalse(handler.stream.closed)
                    logger.warning("outer job")
                self.assertFalse(handler.stream.closed)
                with build_log(paths[2]):
                    logger.warning("next job")
                for path, expected in zip(paths, ("outer job", "inner job", "next job")):
                    self.assertIn(expected, path.read_text())
            finally:
                logger.handlers.clear()

    def test_logger_created_in_a_job_does_not_keep_a_closed_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "first.log", Path(directory) / "second.log"
            logger = logging.getLogger("sako-test-build-stream")
            logger.propagate = False
            logger.setLevel(logging.WARNING)
            try:
                with build_log(first):
                    handler = logging.StreamHandler()
                    logger.addHandler(handler)
                    logger.warning("first job")
                self.assertFalse(handler.stream.closed)
                with build_log(second):
                    logger.warning("second job")
                self.assertIn("first job", first.read_text())
                self.assertIn("second job", second.read_text())
                self.assertNotIn("Logging error", second.read_text())
            finally:
                logger.handlers.clear()
    def compiled(self):
        return SimpleNamespace(name="gemm", hash="compiled-1", n_regs=48, n_spills=4,
            metadata=SimpleNamespace(shared=4096, num_warps=4, num_stages=2))

    def test_reads_actual_compiled_values_and_does_not_invent_missing_values(self):
        values = kernel_resources("gemm", self.compiled())
        self.assertEqual(values["registers_per_thread"], 48)
        self.assertEqual(values["static_shared_memory_bytes"], 4096)
        self.assertEqual(values["reported_spills"], 4)
        missing = kernel_resources("unknown", SimpleNamespace())
        self.assertIsNone(missing["registers_per_thread"])
        self.assertIsNone(missing["reported_spills"])

    def test_observer_restores_launch_method_and_retains_distinct_specializations(self):
        compiled = self.compiled()
        class Kernel:
            def run(self, fail=False):
                if fail:
                    raise RuntimeError("native failure")
                return compiled
        kernel = Kernel()
        with patch("sako.triton_resources.kernel_objects", return_value=iter([("gemm", kernel)])):
            with observe_resources(None) as records:
                self.assertIs(kernel.run(), compiled)
                kernel.run()
        self.assertNotIn("run", vars(kernel))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["registers_per_thread"], 48)
        with patch("sako.triton_resources.kernel_objects", return_value=iter([("gemm", kernel)])):
            with self.assertRaisesRegex(RuntimeError, "native failure"):
                with observe_resources(None):
                    kernel.run(True)
        self.assertNotIn("run", vars(kernel))

    def test_writer_feedback_includes_resources_for_a_slow_correct_result(self):
        result = {"cases": [{"correct": True, "checks": [], "build": {"status": "observed",
            "resource_reports": [kernel_resources("gemm", self.compiled())]}}],
            "decision": {"accepted": False, "speedup": .8}}
        feedback = SearchLoop.operation_feedback(result)["cases"][0]
        self.assertEqual(feedback["registers_per_thread"], 48)
        self.assertEqual(feedback["shared_memory_bytes"], 4096)
        self.assertEqual(feedback["reported_spills"], 4)
        self.assertEqual(feedback["resource_status"], "observed")
