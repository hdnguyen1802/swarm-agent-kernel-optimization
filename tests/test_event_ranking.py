"""Check profiler attribution, shape groups, and disjoint smaller targets."""
import copy
from types import SimpleNamespace
import unittest

from sako.core import load_config
from sako.hardware import select_top_tasks
from sako.model import arguments, task_key
from sako.preparation import profile_settings, validate_catalog
from sako.ranking import CASE, TARGET, METHOD, event_record, rank_operations


def event(name, elapsed=0., parent=None, device="CPU", shapes=None):
    return SimpleNamespace(name=name, self_device_time_total=elapsed, cpu_parent=parent,
                           device_type=device, input_shapes=shapes)


class EventRankingTests(unittest.TestCase):
    def test_only_self_gpu_time_is_used_and_nearest_target_owns_it(self):
        case = event(CASE + "1::2")
        mlp = event(TARGET + "mlp::model.0.mlp", 9000., case)
        linear = event(TARGET + "linear::model.0.mlp.up_proj", 3000., mlp)
        operator = event("aten::mm", 3000., linear, shapes=[[128, 16], [16, 32]])
        record = event_record(operator)
        self.assertEqual(record["task_id"], "linear")
        self.assertEqual(record["path"], "model.0.mlp.up_proj")
        self.assertEqual((record["time_ms"], record["workload_index"], record["trial"]), (3., 1, 2))
        self.assertIsNone(event_record(mlp))
        self.assertIsNone(event_record(linear))
        self.assertIsNone(event_record(event("gemm", 3000., operator, "CUDA")))

    def test_unmapped_gpu_work_remains_evidence_and_invalid_times_are_ignored(self):
        case = event(CASE + "0::0")
        record = event_record(event("aten::mul", 1000., case, shapes=[[1, 8]]))
        self.assertIsNone(record["task_id"])
        self.assertEqual(record["time_ms"], 1.)
        for value in (0., -1., float("nan"), float("inf")):
            self.assertIsNone(event_record(event("aten::mm", value, case)))
        self.assertIsNone(event_record(event("aten::mm", 3000.)))

    def test_shapes_stay_separate_repeated_paths_combine_and_mlp_is_not_selected(self):
        tasks = [{"id": key, "kind": kind, "paths": paths, "contract": {"validated": True}}
                 for key, kind, paths in (("mlp", "mlp", ["model.0.mlp"]),
                    ("linear", "linear", ["model.0.mlp.gate_proj", "model.0.mlp.up_proj"]),
                    ("silu", "silu", ["model.0.mlp.act_fn"]))]
        records = []
        for workload in range(2):
            for trial in range(3):
                for path in tasks[1]["paths"]:
                    records.append({"task_id": "linear", "path": path, "name": "aten::mm",
                        "input_shapes": [[128 * (workload + 1), 16]], "time_ms": 2. * (workload + 1),
                        "workload_index": workload, "trial": trial})
                records.extend({"task_id": key, "path": path, "name": "aten::mul", "input_shapes": [[8]],
                    "time_ms": value, "workload_index": workload, "trial": trial}
                    for key, path, value in (("mlp", "model.0.mlp", 1.), ("silu", "model.0.mlp.act_fn", .5)))
        report = rank_operations(tasks, records, 2, 3)
        self.assertEqual(tasks[1]["priority_ms"], 6.)
        self.assertEqual(tasks[1]["workload_medians_ms"], [4., 8.])
        self.assertEqual(tasks[1]["priority_method"], METHOD)
        groups = [g for g in report["groups"] if g["task_id"] == "linear"]
        self.assertEqual(len(groups), 2)
        self.assertTrue(all(g["calls"] == 6 and len(g["paths"]) == 2 for g in groups))
        selection = select_top_tasks(tasks, 4)
        self.assertEqual(selection["selected_task_ids"], ["linear", "silu"])
        self.assertEqual(selection["excluded"][0]["reason"], "composite_region")

    def test_missing_workload_cannot_rank_as_complete_target(self):
        task = {"id": "linear", "kind": "linear", "paths": ["m"], "contract": {"validated": True}}
        record = {"task_id": "linear", "path": "m", "name": "aten::mm", "input_shapes": [[8]],
                  "time_ms": 2., "workload_index": 0, "trial": 0}
        rank_operations([task], [record], 2, 1)
        self.assertEqual(task["priority_ms"], 0.)
        self.assertEqual(select_top_tasks([task])["selected_task_ids"], [])

    def test_silu_contract_groups_geometry_without_merging_different_hidden_widths(self):
        tensor = SimpleNamespace(shape=(1, 128, 32), dtype="float16")
        self.assertEqual(arguments("silu", None, tensor), ([tensor], []))
        other = SimpleNamespace(shape=(2, 512, 32), dtype="float16")
        self.assertEqual(task_key("silu", [tensor], []), task_key("silu", [other], []))
        other.shape = (2, 512, 64)
        self.assertNotEqual(task_key("silu", [tensor], []), task_key("silu", [other], []))

    def test_ranking_mode_is_bound_to_profile_but_old_configs_resolve_to_legacy(self):
        old = {"correctness_scope": "replaced_layers", "correctness_reference": "eager", "agent": {}}
        new = {**old, "ranking_mode": "profiler_events"}
        self.assertNotEqual(profile_settings(old), profile_settings(new))
        catalog = {"hardware": {"fingerprint": "gpu"}, "config": old}
        with self.assertRaisesRegex(ValueError, "ranking mode changed"):
            validate_catalog(new, catalog, {"fingerprint": "gpu"})


if __name__ == "__main__":
    unittest.main()
