"""Check current score references and safe, exact path fallbacks without a GPU."""

import copy
import unittest

from sako.core import digest
from sako.integration_repairs import affected_candidates, numerical_failures, record_failure, safe_measurement
from sako.kernel_selection import kernel_manifest, record_archive_score, search_score
from sako.path_selection import manifest_allowed, passing_paths, restricted_manifest


class MemoryStore:
    def __init__(self, state=None):
        self.state = copy.deepcopy(state or {})

    def get(self, key, default=None):
        return copy.deepcopy(self.state.get(key, default))

    def set(self, key, value):
        self.state[key] = copy.deepcopy(value)

    def set_many(self, values):
        self.state.update(copy.deepcopy(values))

    def candidates(self, task):
        return [{"id": "source", "correct": True, "accepted": True}]


def setup_case():
    paths = ["block0.gate", "block1.gate"]
    signatures = [{"shapes": [[1, size, 8], [8, 8]], "strides": [[size * 8, 8, 1], [8, 1]],
                   "dtypes": ["torch.float16", "torch.float16"], "scalars": []} for size in (128, 512)]
    workloads = [{"batch_size": 1, "sequence_length": size} for size in (128, 512)]
    config = {"kernel_search": {"path_scoped": True}, "correctness_seeds": [1, 2], "workloads": workloads}
    task = {"id": "gate", "paths": paths, "priority_ms": 8, "scalars": [],
            "cases": [{k: v for k, v in s.items() if k != "scalars"} for s in signatures]}
    manifest = {"entries": [{"task_id": "gate", "candidate_id": "source", "paths": paths,
                             "signatures": signatures, "priority": 8}]}
    cases = []
    for i, (workload, signature) in enumerate(zip(workloads, signatures)):
        checks = []
        for seed in config["correctness_seeds"]:
            layers = []
            for path in paths:
                passed = not (i == 0 and seed == 2 and path == paths[1])
                layers.append({"path": path, "candidate_id": "source", "same_input": True,
                    "inputs_unchanged": True, "passed": passed, "reference": "eager",
                    "input_shapes": signature["shapes"], "input_strides": signature["strides"],
                    "input_dtypes": signature["dtypes"], "input_scalars": signature["scalars"],
                    "check": {"passed": passed, "max_abs_error": 0. if passed else 4.}})
            checks.append({"seed": seed, "scope": "replaced_layers", "reference": "eager",
                "expected_paths": paths, "observed_paths": paths, "missing_paths": [], "layers": layers})
        cases.append({"workload": workload, "checks": checks, "correct": i != 0})
    measured = {"manifest": manifest, "result": {"scope": "model", "correctness_scope": "replaced_layers",
        "correctness_reference": "eager", "reference_state_intact": True, "cases": cases,
        "decision": {"accepted": False}}}
    store = MemoryStore({"kernel_baselines": {"gate": {"shape_scoped": True, "shapes": {
        digest(s): {"signature": s, "candidate_id": "source", "decision": {"accepted": True},
                    "model_status": "not_tested"} for s in signatures}}}})
    return config, task, signatures, measured, store


class CurrentReferenceTests(unittest.TestCase):
    def test_recorded_paired_ranking_uses_fresh_evidence_and_keeps_primary_samples(self):
        store = MemoryStore({"kernel_score_anchors": {"gate": [1000.]}})
        task = {"id": "gate", "cases": [{}]}
        result = {"cases": [{"correct": True, "baseline": {"samples_ms": [7.] * 5},
            "candidate": {"samples_ms": [8.] * 5}, "search_reference": {"kind": "paired_compile",
                "baseline": {"samples_ms": [10.] * 5}, "candidate": {"samples_ms": [8.5] * 5}}}]}
        before = copy.deepcopy(result["cases"])
        record_archive_score(store, task, result, "kept", trials=5, mode="paired_compile")
        self.assertAlmostEqual(result["search_score"], 10 / 8.5)
        self.assertEqual(result["cases"], before)
        self.assertEqual(store.get("kernel_score_anchors"), {"gate": [1000.]})
        for change in ("missing", "short", "invalid"):
            value = copy.deepcopy(result)
            ref = value["cases"][0]["search_reference"]
            if change == "missing":
                value["cases"][0].pop("search_reference")
            elif change == "short":
                ref["candidate"]["samples_ms"].pop()
            else:
                ref["baseline"]["samples_ms"][0] = float("nan")
            record_archive_score(store, task, value, "kept", trials=5, mode="paired_compile")
            self.assertIsNone(value["search_score"])

    def test_archive_uses_the_same_reference_for_each_shape_as_keeps_change(self):
        store = MemoryStore()
        task = {"id": "gate", "cases": [{}, {}]}
        def case(reference, baseline, candidate):
            return {"correct": True, "timing_reference": reference,
                    "baseline": {"samples_ms": [baseline]}, "candidate": {"samples_ms": [candidate]}}
        # A missing original time for a kept shape cannot be reconstructed from its keep ratio.
        first = {"cases": [case("torch_compile", 20, 10), case("kept_kernel", 9, 3)]}
        record_archive_score(store, task, first, {"b": "previous"})
        self.assertIsNone(first["search_score"])
        self.assertEqual(store.get("kernel_score_anchors")["gate"], [20, None])
        second = {"cases": [case("kept_kernel", 10, 8), case("torch_compile", 18, 9)]}
        record_archive_score(store, task, second, {"a": "first"})
        self.assertAlmostEqual(second["search_score"], (20 / 8 * 18 / 9) ** .5)
        third = {"cases": [case("kept_kernel", 8, 7), case("torch_compile", 22, 9)]}
        record_archive_score(store, task, third, {"a": "second"})
        self.assertGreater(third["search_score"], second["search_score"])
        self.assertEqual(store.get("kernel_score_anchors")["gate"], [20, 18])
        self.assertEqual(third["cases"][1]["search_reference"]["baseline_median_ms"], 18)
        third["cases"].pop()
        record_archive_score(store, task, third, {})
        self.assertIsNone(third["search_score"])

    def test_archive_score_uses_current_original_pair_and_preserves_keep_pair(self):
        result = {"cases": [{"baseline": {"samples_ms": [7.] * 5},
            "candidate": {"samples_ms": [8.] * 5},
            "search_reference": {"baseline": {"samples_ms": [10.] * 5},
                                 "candidate": {"samples_ms": [8.] * 5}}}]}
        old = copy.deepcopy(result)
        self.assertAlmostEqual(search_score(result, [1000.], mode="paired_compile"), 1.25)
        self.assertEqual(result, old)
        self.assertAlmostEqual(search_score(result, [16.]), 2.)

    def test_archive_score_does_not_fall_back_to_a_stale_anchor(self):
        result = {"cases": [{"baseline": {"samples_ms": [8.]}, "candidate": {"samples_ms": [4.]}}]}
        self.assertIsNone(search_score(result, [1000.], mode="paired_compile"))
        result["cases"][0]["search_reference"] = {"baseline": {"samples_ms": [float("nan")]},
                                                     "candidate": {"samples_ms": [4.]}}
        self.assertIsNone(search_score(result, mode="paired_compile"))


class PathSelectionTests(unittest.TestCase):
    def test_safe_paths_remain_provisional_and_need_a_new_manifest_check(self):
        config, task, signatures, measured, store = setup_case()
        old = copy.deepcopy(measured)
        store.set("best_model_measurement", measured)
        ticket = record_failure(store, measured, config=config)[0]
        self.assertEqual(measured, old)
        self.assertIsNone(store.get("best_model_measurement"))
        shapes = store.get("kernel_baselines")["gate"]["shapes"]
        self.assertEqual(shapes[digest(signatures[0])]["paths"], ["block0.gate"])
        self.assertEqual(shapes[digest(signatures[1])]["paths"], task["paths"])
        self.assertTrue(all(s["model_status"] == "path_recheck_pending" for s in shapes.values()))
        self.assertEqual(len(ticket["provisional_scopes"]), 2)
        narrowed = kernel_manifest(store, {"tasks": [task]})
        self.assertEqual(len(narrowed["entries"]), 2)
        self.assertNotEqual(digest(narrowed), digest(measured["manifest"]))
        self.assertFalse(manifest_allowed(store, measured["manifest"]))
        self.assertTrue(manifest_allowed(store, narrowed))
        retry = restricted_manifest(store, measured["manifest"])
        self.assertEqual(retry, narrowed)
        self.assertIsNone(restricted_manifest(store, retry))
        self.assertTrue(safe_measurement(store, {"manifest": narrowed, "result": {"correct": True}}))
        self.assertEqual(record_failure(store, measured, config=config)[0], ticket)
        widened = copy.deepcopy(narrowed)
        widened["entries"][0]["paths"] = task["paths"]
        self.assertFalse(manifest_allowed(store, widened))
        stale = copy.deepcopy(narrowed)
        stale["entries"][0]["scope_revision"] = "old-failure"
        self.assertFalse(manifest_allowed(store, stale))

    def test_incomplete_seed_or_shape_metadata_cannot_establish_a_safe_subset(self):
        for mutation in ("seed", "scalar", "workload", "unchanged", "fallback"):
            with self.subTest(mutation=mutation):
                config, task, signatures, measured, store = setup_case()
                check = measured["result"]["cases"][0]["checks"][0]
                if mutation == "seed":
                    measured["result"]["cases"][0]["checks"].pop()
                elif mutation == "workload":
                    measured["result"]["cases"].pop()
                elif mutation == "scalar":
                    check["layers"][0].pop("input_scalars")
                elif mutation == "fallback":
                    check["layers"][0]["original_fallback"] = True
                else:
                    check["layers"][0]["inputs_unchanged"] = False
                self.assertEqual(passing_paths(measured, "source", signatures[0], config,
                                               numerical_failures(measured["result"])), [])

    def test_legacy_run_retires_all_paths(self):
        _, _, _, measured, store = setup_case()
        record_failure(store, measured)
        self.assertEqual(store.get("kernel_baselines"), {})
        self.assertEqual(store.get("candidate_scope_restrictions"), {})
        self.assertIsNone(restricted_manifest(store, measured["manifest"]))

    def test_later_failure_shrinks_scope_and_cannot_restore_an_old_pass(self):
        config, task, signatures, measured, store = setup_case()
        record_failure(store, measured, config=config)
        previous = kernel_manifest(store, {"tasks": [task]})
        check = measured["result"]["cases"][0]["checks"][1]
        check["layers"][0].update(passed=False, check={"passed": False, "max_abs_error": 5.})
        check["layers"][1].update(passed=True, check={"passed": True, "max_abs_error": 0.})
        record_failure(store, measured, config=config)
        shapes = store.get("kernel_baselines")["gate"]["shapes"]
        self.assertNotIn(digest(signatures[0]), shapes)
        self.assertEqual(shapes[digest(signatures[1])]["paths"], task["paths"])
        self.assertFalse(manifest_allowed(store, previous))
        self.assertTrue(manifest_allowed(store, kernel_manifest(store, {"tasks": [task]})))

    def test_exact_source_identity_avoids_blame_for_another_shape(self):
        _, _, _, measured, _ = setup_case()
        measured["manifest"]["entries"].append({**measured["manifest"]["entries"][0], "candidate_id": "other"})
        self.assertEqual(affected_candidates(measured), {"source"})

    def test_diagnostic_numerical_failure_does_not_retire_a_direct_source(self):
        _, _, _, measured, _ = setup_case()
        layer = measured["result"]["cases"][0]["checks"][1]["layers"][1]
        layer["acceptance_role"] = "diagnostic"
        layer["scope"] = "containing_region"
        self.assertEqual(numerical_failures(measured["result"]), [])
        self.assertEqual(affected_candidates(measured), set())

    def test_diagnostic_missing_measurement_cannot_hide_a_direct_failure(self):
        _, _, _, measured, _ = setup_case()
        check = measured["result"]["cases"][0]["checks"][1]
        check["region_checks"] = {"acceptance_role": "diagnostic", "not_measured": True,
                                   "layers": [{"check": {"not_measured": True}}]}
        self.assertEqual(len(numerical_failures(measured["result"])), 1)
        check["region_checks"]["inputs_unchanged"] = False
        self.assertEqual(numerical_failures(measured["result"]), [])


if __name__ == "__main__":
    unittest.main()
