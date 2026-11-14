"""Regression checks for useful progress, shape keeps, tuning, and timing."""
from triton_fixtures import RMS_SOURCE

import contextlib
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sako.core import candidate_id, digest, score_cases, validate_candidate, write_json, read_json
from sako.gpu import paired_timings
from sako.hierarchy import select_hierarchy
from sako.model import apply_manifest
from sako.hypotheses import HypothesisBook
from sako.kernel_selection import keep_kernel, kernel_manifest, region_manifests
from sako.shape_selection import groups, decide, select_variant, validate_entries
from sako.store import Store
from sako.swarm_policy import validate_plan
from sako.timing_audit import control_check
from triton_fixture_tuning import seed, control_seed, CONFIGS
from sako.triton_backend import load as load_triton, identity as triton_identity, _LOADED

import test_swarm_loop as swarm_fixtures
from test_kernel_first import PairedWorker


def task(key="linear"):
    cases = []
    for m in (128, 512):
        for seed_value in (1729, 2718):
            cases.append({"seed": seed_value, "shapes": [[1, m, 4], [4, 4], [0]],
                          "strides": [[m * 4, 4, 1], [4, 1], [1]], "dtypes": ["torch.float16"] * 3})
    return {"id": key, "kind": "linear", "paths": ["block.fc"], "cases": cases, "scalars": [],
            "priority_ms": 10., "contract": {"validated": True}}


def candidate(number, task_id="linear"):
    return {"name": "candidate_" + str(number), "task_id": task_id, "strategy": "Measured test.",
            "backend": "triton", "references": [], "source": RMS_SOURCE + "# " + str(number)}


CONFIG = {"trials": 15, "min_speedup": 1.02, "max_p95_ratio": 1.05,
          "kernel_search": {"enabled": True, "min_speedup": 1.01, "shape_scoped": True}}


def result(task_value, elapsed, baseline_ids=None, base=None):
    cases = [{"case_index": i, "correct": True, "inputs_unchanged": True,
              "baseline": {"samples_ms": [(base or [10.] * 4)[i]] * 15},
              "candidate": {"samples_ms": [v] * 15}} for i, v in enumerate(elapsed)]
    value = {"cases": cases, "decision": score_cases(cases, {**CONFIG, "min_speedup": 1.01}),
             "timing_baseline": {"shapes": baseline_ids or {k: None for k in groups(task_value)}}}
    value["shape_decisions"] = decide(value, task_value, {**CONFIG, "min_speedup": 1.01})
    return value


class ShapeSelectionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = Store(Path(directory.name) / "state.sqlite")
        self.task = task()

    def record(self, number, elapsed, **kwargs):
        key, _ = self.store.add(candidate(number), "memory", number, None)
        self.store.record(key, result(self.task, elapsed, **kwargs))
        return key

    def test_keep_only_winning_shapes_and_compare_with_their_current_baselines(self):
        first = self.record(0, [8., 8., 12., 12.])
        self.assertTrue(keep_kernel(self.store, self.task, first, CONFIG))
        keys = list(groups(self.task))
        current = self.store.get("kernel_baselines")["linear"]["shapes"]
        self.assertEqual(set(current), {keys[0]})
        second = self.record(1, [9., 9., 8., 8.],
                             baseline_ids={keys[0]: first, keys[1]: None}, base=[8., 8., 10., 10.])
        self.assertTrue(keep_kernel(self.store, self.task, second, CONFIG))
        manifest = kernel_manifest(self.store, {"tasks": [self.task]})
        self.assertEqual(len(manifest["entries"]), 2)
        self.assertEqual({v["candidate_id"] for v in manifest["entries"]}, {first, second})
        self.assertTrue(all(len(v["signatures"]) == 1 for v in manifest["entries"]))
        self.assertFalse(keep_kernel(self.store, self.task, first, CONFIG))
        # A result measured against the original cannot replace a newer keep.
        stale = self.record(2, [7., 7., 7., 7.])
        self.assertFalse(keep_kernel(self.store, self.task, stale, CONFIG))

    def test_one_accuracy_failure_prevents_every_shape_keep(self):
        value = result(self.task, [8.] * 4)
        value["cases"][3]["correct"] = False
        decisions = decide(value, self.task, {**CONFIG, "min_speedup": 1.01})
        self.assertFalse(any(v["accepted"] for v in decisions.values()))
        value["cases"] = value["cases"][:2]
        self.assertFalse(any(v["accepted"] for v in decide(value, self.task, CONFIG).values()))

    def test_shape_requires_both_seeds_and_p95_limit(self):
        value = result(self.task, [8., 10.1, 8., 8.])
        value["cases"][2]["candidate"]["samples_ms"][-1] = 12.
        self.assertFalse(any(v["accepted"] for v in decide(value, self.task, {**CONFIG, "min_speedup": 1.01}).values()))

    def test_routes_match_exact_metadata_and_unknown_shapes_keep_original(self):
        signature = next(iter(groups(self.task).values()))["signature"]
        variants = [{"candidate_id": "one", "signatures": [signature]}]
        tensors = [SimpleNamespace(shape=tuple(s), dtype=d, stride=lambda v=v: tuple(v))
                   for s, d, v in zip(signature["shapes"], signature["dtypes"], signature["strides"])]
        self.assertEqual(select_variant(variants, tensors, []), "one")
        tensors[0].shape = (1, 256, 4)
        self.assertIsNone(select_variant(variants, tensors, []))
        tensors[0].shape = (1, 128, 4)
        tensors[0].stride = lambda: (512, 1, 128)
        self.assertIsNone(select_variant(variants, tensors, []))

    def test_duplicate_or_unverified_shape_replacements_fail_closed(self):
        signature = next(iter(groups(self.task).values()))["signature"]
        entry = {"task_id": "linear", "candidate_id": "one", "paths": self.task["paths"], "signatures": [signature]}
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_entries([entry, entry], {"linear": self.task})
        altered = copy.deepcopy(entry)
        altered["signatures"][0]["shapes"][0][1] = 1024
        with self.assertRaisesRegex(ValueError, "catalog"):
            validate_entries([altered], {"linear": self.task})

    def test_model_routes_two_candidates_and_preserves_the_original_fallback(self):
        first, second = candidate(10), candidate(11)
        folder = Path(self.store.path).parent
        for value in (first, second):
            write_json(folder / (candidate_id(value) + ".json"), value)
        signatures = [v["signature"] for v in groups(self.task).values()]
        entries = [{"task_id": "linear", "candidate_id": candidate_id(c), "paths": ["block.fc"],
                    "signatures": [signature]} for c, signature in zip((first, second), signatures)]
        class Module:
            def __call__(self, x): return self.forward(x)
        original = lambda x: "original"
        block = SimpleNamespace(fc=original)
        model = SimpleNamespace(get_submodule=lambda path: block)
        def tensor(shape, strides):
            return SimpleNamespace(shape=tuple(shape), stride=lambda: tuple(strides), dtype="torch.float16")
        weight, bias = tensor([4, 4], [4, 1]), tensor([0], [1])
        with patch.dict("sys.modules", {"torch": SimpleNamespace(nn=SimpleNamespace(Module=Module))}), \
                patch("sako.model.install_attention_dispatch"), \
                patch("sako.model.arguments", side_effect=lambda kind, module, x: ([x, weight, bias], [])):
            apply_manifest(model, {"entries": entries}, {"tasks": [self.task]}, folder,
                           operations={candidate_id(first): lambda *args: "small", candidate_id(second): lambda *args: "large"})
            self.assertEqual(block.fc(tensor([1, 128, 4], [512, 4, 1])), "small")
            self.assertEqual(block.fc(tensor([1, 512, 4], [2048, 4, 1])), "large")
            self.assertEqual(block.fc(tensor([1, 256, 4], [1024, 4, 1])), "original")

    def test_parent_and_children_remain_separate_manifest_choices(self):
        child = self.task
        parent = {**task("mlp"), "kind": "mlp", "paths": ["block"], "priority_ms": 20.}
        hierarchy = select_hierarchy([parent, child], include_parents=True)["hierarchy"]
        kept = {}
        for i, t in enumerate((child, parent)):
            value = candidate(i, t["id"])
            identity, _ = self.store.add(value, "compute", i, None)
            self.store.record(identity, result(t, [8.] * 4))
            kept[t["id"]] = {"candidate_id": identity}
        self.store.set("kernel_baselines", kept)
        variants = region_manifests(self.store, {"tasks": [parent, child], "hierarchy": hierarchy}, hierarchy["regions"][0])
        self.assertEqual([{e["task_id"] for e in m["entries"]} for _, m in variants], [{"linear"}, {"mlp"}])


class ProgressRegressionTests(unittest.TestCase):
    setUp = swarm_fixtures.SwarmLoopTests.setUp
    loop = swarm_fixtures.SwarmLoopTests.loop

    def test_progress_toward_baseline_resets_stagnation_without_accepting_kernel(self):
        self.config["kernel_search"] = {"enabled": True, "min_speedup": 1.01}
        self.config["agent"].update(max_task_experiments=2, performance_refinement=True)
        self.worker = PairedWorker()
        self.worker.latencies = [20., 12.5]
        path = self.directory / "profile/tasks.json"
        catalog = read_json(path)
        catalog["tasks"][0]["cases"] = [{}]
        catalog["config"] = self.config
        write_json(path, catalog)
        self.loop().run()
        assignments = Store(self.directory / "run/state.sqlite").get("swarm_assignments")
        last = max(assignments.values(), key=lambda a: a["generation"])
        self.assertAlmostEqual(last["review"]["progress"]["best_operation_score"], .8)
        self.assertTrue(last["review"]["improved"])
        self.assertEqual(last["review"]["progress"]["stagnant"], 0)
        self.assertFalse(last["operation_accepted"])
        self.assertEqual(Store(self.directory / "run/state.sqlite").get("kernel_baselines", {}), {})
        self.assertEqual(last["selection"]["reason"], "shared_archive")

    def test_legacy_seed_flags_cannot_run_a_template_before_the_coordinator(self):
        self.config["kernel_backend"] = "triton"
        self.config["kernel_search"] = {"enabled": True, "shape_scoped": True, "min_speedup": 1.01}
        self.config["tuning"] = {"enabled": True}
        self.config["timing_audit"] = {"enabled": True}
        self.config["profiler"] = {"enabled": False}
        t = task()
        write_json(self.directory / "profile/tasks.json", {"config": self.config, "tasks": [t],
            "selected_task_ids": [t["id"]], "hardware": {"fingerprint": "test-gpu"}})
        loop = self.loop()
        loop.start_task()
        self.assertEqual(self.worker.calls, [])
        self.assertEqual(loop.store.candidates(), [])
        self.assertEqual(loop.store.get("tuning_seeds", {}), {})
        loop.coordinator(1)
        self.assertEqual(self.client.prompts[0][0], "coordinator")
        self.assertEqual(self.worker.calls, [])

    def test_triton_writer_gets_no_builtin_optimization_template(self):
        self.config["kernel_backend"] = "triton"
        self.config["profiler"] = {"enabled": False}
        t = task()
        write_json(self.directory / "profile/tasks.json", {"config": self.config, "tasks": [t],
            "selected_task_ids": [t["id"]], "hardware": {"fingerprint": "test-gpu"}})
        loop = self.loop()
        loop.start_task()
        plan = loop.coordinator(1)
        assignment = loop.assignment(plan["hypotheses"][0], 0, plan)
        with patch.object(loop, "request_writer", side_effect=lambda a, prompt: prompt):
            prompt = loop.write_swarm(assignment)
            self.assertNotIn("references", prompt)
        self.assertIsNone(prompt["parent"])
        self.assertIn("Write the first candidate", loop.system)
        self.assertEqual(loop.store.candidates(), [])

    def test_complete_model_gates_choose_safe_children_over_a_regressing_parent(self):
        t = task()
        p = {**task("mlp"), "kind": "mlp", "paths": ["block"], "priority_ms": 20.}
        plan = select_hierarchy([p, t], include_parents=True)
        self.config["kernel_search"] = {"enabled": True, "min_speedup": 1.01}
        self.config["agent"].update(target_selection="bottleneck_tree", search_parent_regions=True)
        write_json(self.directory / "profile/tasks.json", {"config": self.config, "tasks": [p, t],
            "selected_task_ids": plan["selected_task_ids"], "hierarchy": plan["hierarchy"],
            "hardware": {"fingerprint": "test-gpu"}})
        loop = self.loop()
        kept = {}
        for i, value in enumerate((t, p)):
            source = candidate(i, value["id"])
            key, _ = loop.store.add(source, "compute", i, None)
            loop.store.record(key, result(value, [8.] * 4))
            kept[value["id"]] = {"candidate_id": key}
        loop.store.set("kernel_baselines", kept)
        loop.progress.update(index=2, completed=[{"task_id": "linear"}, {"task_id": "mlp"}])
        measured = []
        def measure(manifest, generation):
            child = manifest["entries"][0]["task_id"] == "linear"
            measured.append(manifest)
            return {"manifest": manifest, "result": {"cases": [{"correct": True}],
                "decision": {"accepted": child, "speedup": 1.04 if child else 1.1}}}
        loop._measure_manifest = measure
        loop.finish_region(plan["hierarchy"]["regions"][0])
        self.assertEqual(len(measured), 2)
        self.assertEqual(loop.store.get("kernel_region_choices")["mlp"], ["linear"])
        self.assertEqual(loop.store.get("best_model_measurement")["result"]["decision"]["speedup"], 1.04)


class SearchPolicyTests(unittest.TestCase):
    def test_parents_are_alternatives_without_double_counting_region_cost(self):
        parent = {**task("mlp"), "kind": "mlp", "paths": ["block"], "priority_ms": 20.}
        child = task()
        plan = select_hierarchy([parent, child], include_parents=True)
        self.assertEqual(plan["selected_task_ids"], ["linear", "mlp"])
        self.assertEqual(plan["hierarchy"]["regions"][0]["covered_gpu_ms"], 10.)
        self.assertEqual(parent["hierarchy_context"]["search_scope"], "parent_alternative")


    def test_autotuned_seed_is_valid_and_has_bounded_shape_keys(self):
        value = seed(task())
        validate_candidate(value, "triton")
        self.assertEqual(value["source"].count("triton.Config("), len(CONFIGS))
        self.assertIn("key=['M', 'N', 'K', 'SW0', 'SW1', 'HAS_BIAS']", value["source"])
        altered = {**value, "source": value["source"].replace("rep=20", "rep=20000")}
        with self.assertRaisesRegex(ValueError, "bounded"):
            validate_candidate(altered)
        altered = {**value, "source": value["source"].replace("warmup=5", "pre_hook=hook, warmup=5")}
        with self.assertRaisesRegex(ValueError, "hooks"):
            validate_candidate(altered)
        parent = seed({**task(), "kind": "mlp"})
        validate_candidate(parent, "triton")
        self.assertIn("activation = (gate / (1.0 + tl.exp(-gate))).to(tl.float16)", parent["source"])

    def test_archive_skip_cannot_avoid_the_first_measured_refinement(self):
        plan = {"action": "move_on", "reason": "Low expected value.", "evidence": ["parent"], "hypotheses": [],
                "archive_skip": {"reason": "No more tiles.", "evidence": ["parent"]}}
        with self.assertRaisesRegex(ValueError, "Measure at least one"):
            validate_plan(plan, task_id="task", known_evidence={"parent"}, parent_ids=["parent"], require_archive_trial=True)
        self.assertEqual(validate_plan(plan, task_id="task", known_evidence={"parent"}, parent_ids=["parent"])["action"], "move_on")


class TimingTests(unittest.TestCase):
    def test_same_source_control_detects_bias_and_excess_noise(self):
        self.assertTrue(control_check({"baseline": {"samples_ms": [1.] * 15}, "candidate": {"samples_ms": [1.] * 15}})["passed"])
        self.assertFalse(control_check({"baseline": {"samples_ms": [1.] * 15}, "candidate": {"samples_ms": [1.2] * 15}})["passed"])
        self.assertFalse(control_check({"baseline": {"samples_ms": [.1, 2.] * 8}, "candidate": {"samples_ms": [1.] * 16}})["passed"])

    def test_graph_trials_replay_captured_work_without_repeating_python_dispatch(self):
        state = SimpleNamespace(clock=0., active=None, calls=[0, 0], replays=0)
        class Event:
            def __init__(self, **kwargs): pass
            def record(self): self.clock = state.clock
            def synchronize(self): pass
            def elapsed_time(self, other): return other.clock - self.clock
        class Graph:
            def __init__(self): self.costs = []; self.replayed = False
            def replay(self):
                state.clock += sum(self.costs) + (100. if not self.replayed else 0.)
                self.replayed = True
                state.replays += 1
        @contextlib.contextmanager
        def capture(graph, **kwargs):
            state.active = graph
            try: yield
            finally: state.active = None
        stream = SimpleNamespace(wait_stream=lambda *_: None)
        cuda = SimpleNamespace(Event=Event, CUDAGraph=Graph, graph=capture, Stream=lambda: stream,
                               current_stream=lambda: stream, stream=lambda _: contextlib.nullcontext(), synchronize=lambda: None)
        def operation(index):
            def run():
                cost = 2. if index == 0 else 1.
                state.calls[index] += 1
                state.clock += cost
                if state.active is not None: state.active.costs.append(cost)
                return object()
            return run
        with patch.dict("sys.modules", {"torch": SimpleNamespace(cuda=cuda)}):
            measured = paired_timings(operation(0), operation(1), {"warmup": 2, "trials": 5, "repeats": 10}, mode="cuda_graph")
        self.assertEqual(state.calls, [14, 14])
        self.assertEqual(state.replays, 14)
        self.assertEqual(measured["baseline"]["samples_ms"], [2.] * 5)
        self.assertEqual(measured["candidate"]["samples_ms"], [1.] * 5)
        self.assertEqual(measured["timing_scope"], "captured_device_work")

    def test_tuning_log_preserves_failed_configurations_as_json_safe_evidence(self):
        import json
        class Config:
            def __init__(self, kwargs, num_warps=4, num_stages=2):
                self.kwargs, self.num_warps, self.num_stages = kwargs, num_warps, num_stages
            def all_kwargs(self):
                return {**self.kwargs, "num_warps": self.num_warps, "num_stages": self.num_stages}
        class Autotuner:
            def __init__(self, configs):
                self.configs = configs
                self.cache = {(128, 4, 4): configs[0]}
                self.configs_timings = {configs[0]: [1., .9, 1.1], configs[1]: [float("inf")] * 3}
            def run(self, key):
                return self.cache[key]
        triton = SimpleNamespace(Config=Config, jit=lambda fn: fn,
                                 autotune=lambda **kw: lambda fn: Autotuner(kw["configs"]))
        language = SimpleNamespace(constexpr=object())
        triton.language = language
        source = seed(task())
        hardware = {"fingerprint": "test-tuning"}
        name = triton_identity(source, hardware)
        self.addCleanup(_LOADED.pop, name, None)
        with tempfile.TemporaryDirectory() as tmp, patch.dict("sys.modules", {
                "torch": SimpleNamespace(), "triton": triton, "triton.language": language,
                "triton.runtime.jit": SimpleNamespace(KernelInterface=Autotuner),
                "triton.runtime.autotuner": SimpleNamespace(Autotuner=Autotuner)}):
            operation = load_triton(source, hardware, Path(tmp))
            report = operation.tuning_metadata()[0]
            json.dumps(report, allow_nan=False)
            import sys
            tuner = getattr(sys.modules[name], report["kernel"])
            cache = tuner.cache
            with operation.observe_tuning() as chosen:
                self.assertIs(tuner.run((128, 4, 4)), tuner.configs[0])
            self.assertTrue(chosen[0]["invoked"])
            self.assertNotIn("run", vars(tuner))
            self.assertIs(tuner.cache, cache)
        self.assertEqual(len(report["last_trials"]), 2)
        self.assertFalse(report["last_trials"][1]["finite"])
        self.assertEqual(report["last_trials"][1]["timing"], ["inf"] * 3)


if __name__ == "__main__":
    unittest.main()
