import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from sako.profiling import collect_profile_artifacts


ROOT = Path(__file__).resolve().parents[1]


class DiagnosticArtifactTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "runs").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=ROOT / "runs")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def test_artifact_fetch_checks_scope_hash_and_size(self):
        content = b"measured evidence"
        destination = self.directory / "profiles/raw.log"
        def get(relative, path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        worker = SimpleNamespace(root=ROOT, get=get)
        artifact = {"path": destination.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
        self.assertEqual(collect_profile_artifacts(worker, {"artifacts": [artifact]}, self.directory), [artifact])
        with self.assertRaises(ValueError):
            collect_profile_artifacts(worker, {"artifacts": [{**artifact, "path": "../outside.log"}]}, self.directory)
        with self.assertRaises(ValueError):
            collect_profile_artifacts(worker, {"artifacts": [{**artifact, "bytes": 0}]}, self.directory)
