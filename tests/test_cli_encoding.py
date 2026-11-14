"""The Windows console may not support the workspace path's Unicode text."""

import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sako.cli import main


class CliEncodingTests(unittest.TestCase):
    def test_profile_summary_works_with_a_legacy_console_encoding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "profile-\ubb38\uc11c"
            output = io.BytesIO()
            console = io.TextIOWrapper(output, encoding="cp1252", errors="strict")
            with patch("sako.cli.ROOT", root), \
                    patch("sako.cli.Worker"), \
                    patch("sako.preparation.prepare", return_value={"tasks": [{"id": "a"}],
                        "selected_task_ids": ["a"], "config": {"revision": "a" * 40}}), \
                    patch("sys.argv", ["sako", "profile", "--run", str(run)]), \
                    patch("sys.stdout", console):
                main()
            console.flush()
            self.assertIn(b"selected 1", output.getvalue())
            self.assertIn(b"\\ubb38", output.getvalue())
            console.detach()


if __name__ == "__main__":
    unittest.main()
