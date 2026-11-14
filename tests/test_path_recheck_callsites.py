"""Exercise fresh model checks after a saved source loses one path or shape."""
from triton_fixtures import RMS_SOURCE

import copy
import unittest

import test_swarm_loop as fixtures
from test_selection_measurement_fixes import setup_case
from sako.core import digest, implementation_hash, read_json, write_json
from sako.hierarchy import select_hierarchy
from sako.integration import integrate
from sako.kernel_selection import kernel_manifest


class PathRecheckCallSiteTests(unittest.TestCase):
    setUp = fixtures.SwarmLoopTests.setUp
    loop = fixtures.SwarmLoopTests.loop

    def prepare(self, failures):
        settings, task, signatures, _, memory = setup_case()
        self.config.update(settings)
        self.config["kernel_search"].update(enabled=True, shape_scoped=True, min_speedup=1.01)
        self.config["agent"]["target_selection"] = "bottleneck_tree"
        task.update(kind="linear", contract={"validated": True})
        selected = select_hierarchy([task])
        catalog = {"config": self.config, "tasks": [task], "hardware": {"fingerprint": "test-gpu"},
                   "selected_task_ids": selected["selected_task_ids"], "hierarchy": selected["hierarchy"]}
        write_json(self.directory / "profile/tasks.json", catalog)
        loop = self.loop()
        candidate = {"name": "measured_gate", "task_id": "gate", "strategy": "Keep measured paths.",
                     "backend": "triton", "references": [], "source": RMS_SOURCE}
        key, _ = loop.store.add(candidate, "memory", 0, None)
        loop.store.record(key, {"correct": True, "decision": {"accepted": True, "speedup": 1.25}})
        write_json(self.directory / "run/candidates" / (key + ".json"), candidate)
        keeps = memory.get("kernel_baselines")
        for shape in keeps["gate"]["shapes"].values():
            shape["candidate_id"] = key
        loop.store.set("kernel_baselines", keeps)
        loop.progress.update(index=1, completed=[{"task_id": "gate", "reason": "Measured kernel gain."}])
        loop.store.set("task_progress", loop.progress)
        loop.store.set("status", "completed_kernel_search")
        write_json(self.directory / "run/result.json", {"status": "completed_kernel_search"})
        self.model_calls = []
        def run(action, run_dir, **fields):
            self.assertEqual(action, "model")
            manifest = read_json(fixtures.ROOT / fields["manifest"])
            self.model_calls.append(copy.deepcopy(manifest))
            # No earlier failed result can establish a model gain.
            self.assertIsNone(loop.store.get("best_model_measurement"))
            failure = failures[len(self.model_calls) - 1] if len(self.model_calls) <= len(failures) else None
            self.assertLessEqual(len(self.model_calls), len(failures) + 1)
            paths = list(dict.fromkeys(path for entry in manifest["entries"] for path in entry["paths"]))
            cases = []
            for index, (workload, signature) in enumerate(zip(self.config["workloads"], signatures)):
                checks = []
                for seed in self.config["correctness_seeds"]:
                    layers = []
                    for path in paths:
                        active = any(path in e["paths"] and signature in e["signatures"] for e in manifest["entries"])
                        passed = not (active and failure == (index, path) and seed == self.config["correctness_seeds"][-1])
                        layers.append({"path": path, "candidate_id": key if active else None,
                            "reference": "eager", "same_input": True, "inputs_unchanged": True,
                            "passed": passed, "original_fallback": not active,
                            "input_shapes": signature["shapes"], "input_strides": signature["strides"],
                            "input_dtypes": signature["dtypes"], "input_scalars": signature["scalars"],
                            "check": {"passed": passed, "max_abs_error": 0. if passed else 4.}})
                    checks.append({"seed": seed, "scope": "replaced_layers", "reference": "eager",
                        "expected_paths": paths, "observed_paths": paths, "missing_paths": [],
                        "layers": layers, "passed": all(layer["passed"] for layer in layers)})
                cases.append({"workload": workload, "checks": checks, "correct": all(c["passed"] for c in checks)})
            accepted = all(case["correct"] for case in cases)
            if accepted:
                for case in cases:
                    case.update(baseline={"samples_ms": [10.] * self.config["trials"]},
                                candidate={"samples_ms": [8.] * self.config["trials"]})
            return {"scope": "complete_model_forward", "correctness_scope": "replaced_layers",
                    "correctness_reference": "eager", "reference_state_intact": True, "cases": cases,
                    "decision": {"accepted": accepted, "speedup": 1.25 if accepted else None}}
        self.worker.run = run
        return loop, catalog["hierarchy"]["regions"][0]

    def seed_old_narrow_cache(self, loop):
        stale = copy.deepcopy(kernel_manifest(loop.store, loop.catalog))
        original = stale["entries"][0]
        stale["entries"] = [{**copy.deepcopy(original), "paths": ["block0.gate"],
                              "signatures": [original["signatures"][0]]},
                             {**copy.deepcopy(original), "signatures": [original["signatures"][1]]}]
        measured = {"manifest": stale, "result": {"correct": True,
            "decision": {"accepted": True, "speedup": 99.}}}
        # Both production call sites use their own cache identity.
        cache = {digest({"manifest": stale, "hardware": loop.catalog.get("hardware"),
                         "config": loop.catalog.get("config")}): measured,
                 digest({"manifest": stale, "config": self.config, "hardware": loop.catalog.get("hardware"),
                         "implementation": implementation_hash()}): measured}
        loop.store.set("model_measurements", cache)

    def check_success(self, loop):
        self.assertEqual(len(self.model_calls), 2)
        broad, narrow = self.model_calls
        self.assertNotEqual(digest(broad), digest(narrow))
        self.assertTrue(all("scope_revision" in e for e in narrow["entries"]))
        without_revision = {"entries": [{k: v for k, v in entry.items() if k != "scope_revision"}
                                         for entry in narrow["entries"]]}
        self.assertTrue(any(t["manifest"] == without_revision and t["result"]["decision"]["speedup"] == 99.
                            for t in loop.store.get("model_measurements").values()))
        best = loop.store.get("best_model_measurement")
        self.assertEqual(best["manifest"], narrow)
        self.assertEqual(best["result"]["decision"]["speedup"], 1.25)
        trials = list(loop.store.get("model_measurements").values())
        self.assertTrue(any(t["manifest"] == broad and not t["result"]["decision"]["accepted"] for t in trials))
        self.assertTrue(any(t["manifest"] == narrow and t["result"]["decision"]["accepted"] for t in trials))

    def test_finish_region_tests_narrow_scope_before_acceptance(self):
        loop, region = self.prepare([(0, "block1.gate")])
        self.seed_old_narrow_cache(loop)
        loop.finish_region(region)
        self.check_success(loop)
        self.assertEqual(sum(e["kind"] == "region_scope_recheck" for e in loop.events), 1)
        loop.finish_region(region)
        self.assertEqual(len(self.model_calls), 2)

    def test_deferred_integration_tests_narrow_scope_before_acceptance(self):
        loop, _ = self.prepare([(0, "block1.gate")])
        self.seed_old_narrow_cache(loop)
        integrate(self.config, self.directory / "run", self.directory / "profile", self.worker)
        self.check_success(loop)
        self.assertEqual(read_json(self.directory / "run/result.json")["integration_status"], "accepted")
        integrate(self.config, self.directory / "run", self.directory / "profile", self.worker)
        self.assertEqual(len(self.model_calls), 2)

    def test_finish_region_stops_after_each_remaining_scope_fails(self):
        failures = [(0, "block1.gate"), (0, "block0.gate"), (1, "block1.gate"), (1, "block0.gate")]
        loop, region = self.prepare(failures)
        loop.finish_region(region)
        self.assertEqual(len(self.model_calls), 4)
        self.assertEqual(len({digest(m) for m in self.model_calls}), 4)
        self.assertIsNone(loop.store.get("best_model_measurement"))
        self.assertEqual(kernel_manifest(loop.store, loop.catalog), {"entries": []})

    def test_deferred_integration_stops_after_each_remaining_scope_fails(self):
        failures = [(0, "block1.gate"), (0, "block0.gate"), (1, "block1.gate"), (1, "block0.gate")]
        loop, _ = self.prepare(failures)
        integrate(self.config, self.directory / "run", self.directory / "profile", self.worker)
        self.assertEqual(len(self.model_calls), 4)
        self.assertEqual(len({digest(m) for m in self.model_calls}), 4)
        self.assertIsNone(loop.store.get("best_model_measurement"))
        self.assertEqual(kernel_manifest(loop.store, loop.catalog), {"entries": []})
        self.assertEqual(read_json(self.directory / "run/result.json")["integration_status"], "rejected")


if __name__ == "__main__":
    unittest.main()
