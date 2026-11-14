"""Test broad selection with the project config and archived configs."""
import tempfile
import unittest
from pathlib import Path

from sako.compiled_ranking import launch_plan, rank_compiled, require_mapped_targets
from sako.core import load_config, read_json, write_json
from sako.hierarchy import select_hierarchy
from sako.preparation import validate_catalog
from test_compiled_ranking import record, task


PROJECT = Path(__file__).resolve().parents[1] / "configs/project.json"


class BroadSelectionTests(unittest.TestCase):
    def test_project_prioritizes_a_mixed_mlp_and_queues_its_gemm(self):
        config = load_config(PROJECT)
        tasks = [task("norm", "rmsnorm", ["norm"]), task("mlp", "mlp", ["block.mlp"]),
                 task("gate", "linear", ["block.mlp.gate"])]
        samples = [record("triton_norm", ["norm"], 2.),
                   record("aten::mm", ["block.mlp.gate"], 9.),
                   record("triton_fused_silu_mul", ["block.mlp", "block.mlp.act"], 1.)]

        report = rank_compiled(tasks, samples, 1, 1, config=config)
        selection = select_hierarchy(tasks, include_parents=config["agent"]["search_parent_regions"])

        self.assertEqual([t["priority_ms"] for t in tasks], [2., 10., 9.])
        self.assertEqual(selection["hierarchy"]["initial_region_id"], "mlp")
        self.assertEqual(selection["selected_task_ids"], ["gate", "norm"])
        self.assertEqual(selection["hierarchy"]["regions"][0]["alternative_task_ids"], [])
        self.assertEqual(report["total_gpu_ms"], 12.)

    def test_selects_deepest_verified_parts_and_preserves_an_indivisible_region(self):
        config = load_config(PROJECT)
        tasks = [task("encoder", "block", ["encoder"]),
                 task("mix", "transform", ["encoder.mix"]),
                 task("left", "linear", ["encoder.mix.left"]),
                 task("right", "linear", ["encoder.mix.right"]),
                 task("coupled", "transform", ["decoder.coupled"]),
                 task("inner", "linear", ["decoder.coupled.inner"])]
        samples = [record("mm_left", ["encoder.mix.left"], 4.),
                   record("mm_right", ["encoder.mix.right"], 7.),
                   record("fused", ["decoder.coupled.inner", "decoder.coupled"], 3.)]
        rank_compiled(tasks, samples, 1, 1, config=config)
        plan = select_hierarchy(tasks, include_parents=config["agent"]["search_parent_regions"])
        self.assertEqual(plan["selected_task_ids"], ["right", "left", "coupled"])
        self.assertEqual(tasks[-1]["ranking_excluded_reason"], "cross_contract_fusion")

    def test_all_crossing_fusion_saves_measurements_and_stops(self):
        tasks = [task("norm", "rmsnorm", ["block.norm"])]
        report = rank_compiled(tasks, [record("triton_fused_add_norm", ["block", "block.norm"], 3.)], 1, 1)
        self.assertEqual(tasks[0]["ranking_excluded_reason"], "cross_contract_fusion")
        self.assertEqual(select_hierarchy(tasks)["selected_task_ids"], [])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "no safe, mapped"):
                require_mapped_targets(report, tasks, directory)
            self.assertEqual(read_json(Path(directory) / "operation_profile.json")["total_gpu_ms"], 3.)

    def test_retired_setting_cannot_select_strict_or_reuse_a_strict_profile(self):
        config = read_json(PROJECT)
        self.assertNotIn("optimization_policy", config)
        strict = {**config, "optimization_policy": "compiled_opportunities"}
        hardware = {"fingerprint": "device-a"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old-config.json"
            write_json(path, strict)
            with self.assertRaisesRegex(ValueError, "Remove it"):
                load_config(path)
            self.assertEqual(load_config(path, historical=True), load_config(PROJECT))
            self.assertEqual(read_json(path), strict)
            write_json(path, {**config, "optimization_policy": "legacy"})
            self.assertEqual(load_config(path), load_config(PROJECT))
        with self.assertRaisesRegex(ValueError, "new run and profile"):
            validate_catalog(config, {"config": strict, "hardware": hardware}, hardware)

    def test_launch_provenance_is_explicit_for_agent_context(self):
        source = '''def call(args):
    # Topologically Sorted Source Nodes: [norm], Original ATen: [aten.mean]
    triton_red_norm_0.run(x, out, stream=get_stream(0))
    # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.mm]
    extern_kernels.mm(x, w, out=y)
'''
        nodes = {"norm": {"path": "norm", "target": "mean"},
                 "linear": {"path": "fc", "target": "linear"}}
        launches = launch_plan(source, nodes, "output_code.py")
        self.assertEqual([launch["implementation"] for launch in launches],
                         ["generated_triton", "vendor_gemm"])

if __name__ == "__main__":
    unittest.main()
