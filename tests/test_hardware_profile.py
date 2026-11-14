from triton_fixtures import RMS_SOURCE
import copy
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sako.contracts import operation_contract
from sako.gpu import build_identity, compile_candidate, case_identity
from sako.hardware import capability_features, discover_hardware, hardware_fingerprint, select_top_tasks
from sako.model import region_priority, Collector


def hardware(capability=(7, 0), compiler_targets="sm_70 sm_80 sm_90", *, config=None):
    props = SimpleNamespace(major=capability[0], minor=capability[1], name="Test GPU",
                            total_memory=32 * 1024 ** 3, multi_processor_count=80)
    cuda = SimpleNamespace(is_available=lambda: True, init=lambda: None,
                           current_device=lambda: 0, get_device_properties=lambda _: props)
    torch = SimpleNamespace(cuda=cuda, empty=lambda *a, **k: None,
                            __version__="test-torch", version=SimpleNamespace(cuda="12.3"))
    def command(args):
        if "--list-gpu-code" in args:
            return compiler_targets
        return "test-version"
    with patch.dict("sys.modules", {"torch": torch}), patch("sako.hardware._command", side_effect=command), \
            patch("sako.hardware.shutil.which", return_value="/usr/bin/nvcc"), \
            patch("sako.hardware._version", return_value="test"), \
            patch("sako.hardware.query_driver_attributes", return_value=(
                {"uuid": "GPU-test", "warp_size": 32, "max_threads_per_block": 1024,
                 "driver_api_version": 12030}, {})):
        return discover_hardware(config)


def task(name, paths, time, valid=True):
    return {"id": name, "paths": paths, "priority_ms": time,
            "priority_method": "warmed_eager_region_cuda_events", "contract": {"validated": valid}}


class HardwareAndProfile(unittest.TestCase):
    def test_triton_does_not_require_an_nvcc_target(self):
        profile = hardware(compiler_targets="sm_80", config={"kernel_backend": "triton"})
        self.assertTrue(profile["support"]["supported"])

    def test_queries_actual_target_and_reports_unknown_limits(self):
        profile = hardware((8, 0))
        self.assertEqual(profile["target"], "sm_80")
        self.assertEqual(profile["facts"]["warp_size"], 32)
        self.assertIsNone(profile["facts"]["max_blocks_per_sm"])
        self.assertTrue(profile["support"]["supported"])
        self.assertTrue(profile["features"]["hardware_async_copy_global_to_shared"])

    def test_fingerprint_rejects_gpu_and_software_changes(self):
        first = hardware()
        changed = copy.deepcopy(first)
        changed["software"]["torch"] = "new-version"
        changed["fingerprint"] = hardware_fingerprint(changed)
        self.assertNotEqual(first["fingerprint"], changed["fingerprint"])
        changed = copy.deepcopy(first)
        changed["transient_free_memory_bytes"] = 0
        self.assertEqual(hardware_fingerprint(first), hardware_fingerprint(changed))

    def test_build_key_changes_when_gpu_changes(self):
        candidate = {"name": "probe", "task_id": "task", "source": RMS_SOURCE,
                     "backend": "triton", "references": [], "strategy": "Test."}
        self.assertNotEqual(build_identity(candidate, hardware()), build_identity(candidate, hardware((8, 0))))

    def test_compile_failure_keeps_full_log_and_exposes_path(self):
        candidate = {"name": "probe", "task_id": "task", "source": RMS_SOURCE,
                     "backend": "triton", "references": [], "strategy": "Test."}
        def fail(*args, **kwargs):
            os.write(1, ("complete compiler output " * 2000).encode())
            raise RuntimeError("Compilation failed.")
        with tempfile.TemporaryDirectory() as root:
            modules = {"torch": SimpleNamespace(), "torch.utils": SimpleNamespace(),
                       "torch.utils.cpp_extension": SimpleNamespace(load_inline=fail)}
            profile = hardware()
            with patch("sako.triton_backend.load", side_effect=fail), patch("sako.gpu.__file__", str(Path(root) / "sako/gpu.py")):
                with self.assertRaises(RuntimeError) as failure:
                    compile_candidate(candidate, profile)
            log = Path(failure.exception.build_log).read_text(encoding="utf-8")
            self.assertIn("complete compiler output " * 2000, log)
            self.assertIn("Compilation failed.", log)

    def test_failure_identity_preserves_fixture_seed_and_workload(self):
        case = {"seed": 1729, "workload": {"batch_size": 1, "sequence_length": 512}}
        self.assertEqual(case_identity(case, 2), {"case_index": 2, **case})

    def test_capture_keeps_reproduction_metadata_in_fixture_and_catalog(self):
        class Tensor:
            shape, dtype = (2, 4), "torch.float16"
            def detach(self): return self
            def cpu(self): return self
            def stride(self): return (4, 1)
        collector = Collector({"max_fixture_cases": 2})
        collector.context = {"seed": 1729, "workload": {"batch_size": 1, "sequence_length": 512}}
        with patch.dict("sys.modules", {"torch": SimpleNamespace()}):
            key = collector.capture("model.linear", "linear", [Tensor(), Tensor(), Tensor()], [])
        collector.context["workload"]["sequence_length"] = 128
        for case in (collector.cases[key][0], collector.tasks[key]["cases"][0]):
            self.assertEqual(case["seed"], 1729)
            self.assertEqual(case["workload"]["sequence_length"], 512)
            self.assertEqual(case["case_index"], 0)

    def test_selection_excludes_parent_child_overlap_and_failed_contract(self):
        tasks = [task("parent", ["model.0.mlp"], 10),
                 task("child", ["model.0.mlp.gate_proj"], 9),
                 task("failed", ["model.0.attn"], 20, False),
                 task("next", ["model.0.norm"], 3), task("last", ["model.1.norm"], 2)]
        result = select_top_tasks(tasks, 2)
        self.assertEqual(result["selected_task_ids"], ["parent", "next"])
        self.assertEqual({r["task_id"]: r["reason"] for r in result["excluded"]},
                         {"failed": "contract_failed", "child": "overlapping_region", "last": "top_k_limit"})

    def test_task_priority_uses_call_totals_and_equal_workload_weight(self):
        samples = [{"workload_index": 0, "trial": 0, "time_ms": 2},
                   {"workload_index": 0, "trial": 0, "time_ms": 3},
                   {"workload_index": 0, "trial": 1, "time_ms": 7},
                   {"workload_index": 1, "trial": 0, "time_ms": 10}]
        priority, medians = region_priority(samples, 2)
        self.assertEqual(medians, [6., 10.])
        self.assertEqual(priority, 8.)
        self.assertEqual(region_priority(samples, 3), (0., []))

    def test_contract_requires_comparison_with_original(self):
        candidate = {"id": "rms", "kind": "rmsnorm", "cases": [], "paths": ["model.norm"], "scalars": [1e-6]}
        config = {"atol": .03, "rtol": .02}
        self.assertFalse(operation_contract(candidate, [], config)["validated"])
        contract = operation_contract(candidate, [{"passed": True}], config)
        self.assertTrue(contract["validated"])
        self.assertIn("normalized.to(x.dtype) * weight", contract["functional_source"])
        self.assertFalse(operation_contract(candidate, [{"passed": False}], config)["validated"])
        self.assertEqual(contract["atol"], .03)


    def test_specialized_instruction_permission_is_not_inferred_from_gpu_age(self):
        self.assertFalse(capability_features([12, 0])["architecture_specific_instructions"])
        self.assertFalse(capability_features([7, 0])["hardware_async_copy_global_to_shared"])


if __name__ == "__main__":
    unittest.main()
