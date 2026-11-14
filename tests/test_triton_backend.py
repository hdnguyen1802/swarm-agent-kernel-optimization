"""Check backend selection, source policy, cache identity, and prompt routing."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sako.core import candidate_id, load_config, validate_candidate
from sako.gpu import build_identity
from sako.preparation import profile_settings, validate_catalog
from sako.triton_backend import validate_source
from triton_fixtures import RMS_SOURCE, LINEAR_SOURCE


def candidate(source=RMS_SOURCE):
    return {"name": "test", "task_id": "rmsnorm_1", "strategy": "Test Triton.", "backend": "triton",
            "references": [], "source": source}


class TritonBackendTests(unittest.TestCase):
    def test_examples_use_the_explicit_triton_entry_and_validate(self):
        for source in (RMS_SOURCE, LINEAR_SOURCE):
            self.assertEqual(validate_candidate(candidate(source), "triton")["backend"], "triton")

    def test_backend_mismatch_and_unknown_fields_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Triton|backend"):
            validate_candidate(candidate(), "cuda")
        with self.assertRaisesRegex(ValueError, "keys"):
            validate_candidate({**candidate(), "extra": True})
        with self.assertRaisesRegex(ValueError, "Triton|backend"):
            validate_candidate({k: v for k, v in candidate().items() if k != "backend"}, "triton")

    def test_rejects_compute_fallbacks_io_precision_changes_and_error_suppression(self):
        fragments = ["import os", "import subprocess", "from torch import matmul", "open('x')",
                     "eval('x')", "torch.matmul(x, weight)", "x.matmul(weight)", "x @ weight",
                     "x.cpu()", "x.item()", "torch.cuda.synchronize()", "triton.testing.do_bench(fn)",
                     "tl.dot(x, y, allow_tf32=True)", "x.to(tl.bfloat16)", "globals()",
                     "cache[0] = output", "rms.saved_output = output",
                     "try:\n        return x\n    except Exception:\n        return x"]
        for fragment in fragments:
            source = RMS_SOURCE.replace("    return output", "    " + fragment + "\n    return output")
            with self.subTest(fragment=fragment), self.assertRaises(ValueError):
                validate_source(source)

    def test_forbids_import_time_execution_and_requires_a_kernel(self):
        for source in (RMS_SOURCE + "\nx = torch.empty((1,))\n", RMS_SOURCE.replace("@triton.jit", ""),
                       RMS_SOURCE.replace("sako_run(tensors, scalars)", "sako_run(x)")):
            with self.assertRaises(ValueError):
                validate_source(source)

    def test_identity_is_bound_to_backend_source_and_device_but_not_task_name(self):
        item = candidate()
        hardware = {"fingerprint": "v100-triton31"}
        self.assertEqual(build_identity(item, hardware), build_identity({**item, "task_id": "other"}, hardware))
        self.assertNotEqual(build_identity(item, hardware), build_identity(item, {"fingerprint": "other-version"}))
        self.assertNotEqual(candidate_id(item), candidate_id(candidate(RMS_SOURCE + "\n# change\n")))
        self.assertTrue(build_identity(item, hardware).startswith("sako_triton_"))

    def test_prepared_profiles_cannot_cross_backend_modes(self):
        hardware = {"fingerprint": "device"}
        config = {"kernel_backend": "triton", "agent": {"top_k_operations": 4}}
        self.assertEqual(profile_settings(config)["kernel_backend"], "triton")
        with self.assertRaisesRegex(ValueError, "backend changed"):
            validate_catalog(config, {"hardware": hardware, "config": {"kernel_backend": "cuda"}}, hardware)

    def test_all_launch_configs_use_triton(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(load_config(root / "configs/project.json")["kernel_backend"], "triton")
        for path in (root / "configs").glob("*.json"):
            self.assertEqual(load_config(path)["kernel_backend"], "triton")


if __name__ == "__main__":
    unittest.main()
