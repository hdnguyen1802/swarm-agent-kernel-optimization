"""Worker failures must preserve complete evidence in the result file."""

import contextlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sako.core import digest, read_json, write_json
from sako.worker import main, execute, InfrastructureError


class WorkerEvidenceTests(unittest.TestCase):
    def test_stale_worker_source_is_an_infrastructure_failure(self):
        with self.assertRaisesRegex(InfrastructureError, "source differ"):
            execute({"implementation_hash": "stale", "config": "unused"})

    def test_long_failure_keeps_full_traceback_and_job_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job = {"action": "evaluate", "candidate": "candidate.json"}
            write_json(root / "job.json", job)
            message = "first-marker-" + "x" * 40000 + "-last-marker"
            with patch("sys.argv", ["worker", "--job", str(root / "job.json"),
                                    "--output", str(root / "result.json")]), \
                    patch("sako.worker.gpu_lock", return_value=contextlib.nullcontext()), \
                    patch("sako.worker.execute", side_effect=RuntimeError(message)):
                with self.assertRaises(SystemExit):
                    main()
            result = read_json(root / "result.json")
            self.assertFalse(result["ok"])
            self.assertEqual(result["job_hash"], digest(job))
            self.assertIn(message, result["traceback"])
            self.assertEqual(result["error"], "RuntimeError: " + message)


if __name__ == "__main__":
    unittest.main()
