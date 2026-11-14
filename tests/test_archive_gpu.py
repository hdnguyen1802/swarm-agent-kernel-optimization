from triton_fixtures import RMS_SOURCE
import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier
import unittest
from unittest.mock import Mock, patch

from sako.archive_gpu import check_catalogs, run_paired
from sako.core import candidate_id, load_config, write_json
from sako.transport import TransferError


class ArchiveGPUTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.directory = self.root / "runs" / "trial"
        self.config = load_config(Path(__file__).resolve().parents[1] / "configs/project.json")
        self.primary = {"hardware": {"target": "sm_70", "software": {}, "fingerprint": "primary",
                                     "facts": {"uuid": "gpu-a", "name": "V100"}},
                        "tasks": [{"id": "linear_1", "cases": [{"seed": 22}]}], "config": self.config}
        self.archive = copy.deepcopy(self.primary)
        self.archive["hardware"].update(facts={"uuid": "gpu-b", "name": "V100"}, fingerprint="archive")

    def test_eager_reference_finishes_before_the_other_gpu_profiles(self):
        events = []
        primary = Mock(root=self.root, config={**self.config, 'report_eager_baseline':True}, settings={'gpu':0})
        peer = Mock(root=self.root, config=self.config, settings={'gpu':1})
        def first(*args, **kwargs):
            events.extend(['primary starts', 'primary finishes'])
            return self.primary
        def second(*args, **kwargs):
            self.assertEqual(events[-1], 'primary finishes')
            events.append('peer starts')
            return self.archive
        primary._run_action.side_effect = first
        with patch('sako.archive_gpu.prepare', side_effect=second):
            self.assertEqual(run_paired(primary, peer, 'profile', self.directory, {}), self.primary)
        self.assertEqual(events, ['primary starts', 'primary finishes', 'peer starts'])

    def test_rejects_wrong_device_or_inputs(self):
        with self.assertRaises(TransferError):
            check_catalogs(self.primary, self.primary)
        self.archive["tasks"][0]["cases"][0]["seed"] += 1
        with self.assertRaises(TransferError):
            check_catalogs(self.primary, self.archive, "linear_1")

    def test_secondary_rank_cannot_change_keep_decision(self):
        primary = Mock(root=self.root, settings={"gpu": 0})
        archive = Mock(root=self.root, settings={"gpu": 1})
        profile = self.directory / "profile"
        other_profile = self.directory / "archive-gpu-1" / "profile"
        write_json(profile / "tasks.json", self.primary)
        write_json(other_profile / "tasks.json", self.archive)
        write_json(other_profile / "preparation.json", {})
        source = {"name": "test", "task_id": "linear_1", "strategy": "tile", "backend": "triton",
                  "source": RMS_SOURCE, "references": []}
        path = self.directory / "candidates" / "test.json"
        write_json(path, source)
        case = {"case_index": 0, "seed": 22, "correct": True, "candidate": {"samples_ms": [11]}}
        barrier = Barrier(2, timeout=5)
        def measure(device, **kwargs):
            barrier.wait()
            return {"candidate_id": candidate_id(source), "worker_total_s": 1,
                    "worker_provenance": {"visible_devices": str(device)},
                    "cases": [{**case, **kwargs}], "decision": {"accepted": device == 1}}
        primary._run_action.side_effect = lambda *a, **kw: measure(0)
        archive.run.side_effect = lambda *a, **kw: measure(1, search_reference={
            "kind": "paired_compile", "baseline": {"samples_ms": [10]}, "candidate": {"samples_ms": [5]}})
        fields = {"profile_dir": profile.relative_to(self.root).as_posix(),
                  "candidate": path.relative_to(self.root).as_posix(), "baseline_shapes": {"shape": "kept"}}
        result = run_paired(primary, archive, "evaluate", self.directory, fields)
        self.assertFalse(result["decision"]["accepted"])
        self.assertEqual(result["cases"][0]["candidate"], {"samples_ms": [11]})
        self.assertEqual(result["cases"][0]["search_reference"]["candidate"], {"samples_ms": [5]})
        self.assertEqual(result["cases"][0]["search_reference"]["hardware_fingerprint"], "archive")
        self.assertTrue(primary._run_action.call_args.kwargs["archive_on_peer"])
        self.assertNotIn("baseline_shapes", archive.run.call_args.kwargs)
