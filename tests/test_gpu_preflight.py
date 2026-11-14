from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from sako.core import read_json
from scripts.gpu_preflight import preflight


class GpuPreflightTests(unittest.TestCase):
    def test_checks_cuda_output_and_saves_the_result(self):
        for code, expected in [(0, "passed"), (1, "triton_check_failed")]:
            with self.subTest(code=code), tempfile.TemporaryDirectory() as root:
                def command(args, **kwargs):
                    if args[0] == "nvidia-smi":
                        return subprocess.CompletedProcess(args, 0, "Test GPU,GPU-test,40960,555.0,8.0", "")
                    self.assertEqual(Path(args[1]).name, "probe.py")
                    return subprocess.CompletedProcess(args, code, "", "")

                with patch("scripts.gpu_preflight.run_command", side_effect=command):
                    result = preflight(root)
                self.assertEqual(result["status"], expected)
                self.assertEqual(read_json(Path(result["output_dir"]) / "summary.json"), result)
