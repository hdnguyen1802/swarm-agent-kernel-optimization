import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import patch

from sako.core import candidate_id, digest, implementation_hash, load_config, read_json, write_json
from sako.decode_devices import DecodeDevice
from sako.decode_search import target_finalists
from sako.decode_validation import target_boundary
from sako.integration import integrate
from sako.hypotheses import classify_fault, measured_gpu_outcome
from sako.measurement import paired_samples
from sako.model_acceptance import score_model_cases
from sako.staged_evaluation import initial_indices
from sako.store import Store


def signature(n):
    return {"shapes": [[1,n]], "strides": [[n,1]], "dtypes": ["torch.float16"], "scalars": []}


class DecodeSearchTests(unittest.TestCase):
    def test_first_pair_pass_and_second_pair_fallback_use_the_same_base(self):
        from triton_fixtures import RMS_SOURCE
        root = Path(__file__).resolve().parents[1]
        (root / "runs").mkdir(exist_ok=True)
        config = load_config(root / "configs/generation.json")
        workload = config["workloads"][0]
        barrier = Barrier(2, timeout=8)
        calls, scores = [], {}
        final_passes = True

        def worker(gpu):
            def execute(action, directory, **fields):
                self.assertEqual(action, "model")
                selected = read_json(root / fields["manifest"])
                base = read_json(root / fields["incumbent_manifest"]) if "incumbent_manifest" in fields else {"entries":[]}
                chosen = [e["candidate_id"] for e in selected["entries"]]
                incumbent = [e["candidate_id"] for e in base["entries"]]
                calls.append((gpu, chosen, incumbent))
                if final_passes:
                    barrier.wait()
                new = 30. - sum(scores[c] for c in chosen)
                old = 30. - sum(scores[c] for c in incumbent)
                metrics = {r:{"mean_decode_ms":{"median_ms":ms}} for r,ms in (("baseline",30.),("candidate",new),("incumbent",old))}
                return {"scope":"cached_decode", "worker_provenance":{"visible_devices":str(gpu)}, "reference_state_intact":True,
                    "decision":{"accepted":final_passes and old > new, "speedup":30./new,"reasons":[]},
                    "incumbent_comparison":{"decision":{"accepted":old > new,"speedup":old/new}},
                    "cases":[{"workload":workload,"correct":True,"candidate":{"median_ms":new},"generation_metrics":metrics,
                              "sampling":{"assessment":{"pairs":5}}}]}
            return SimpleNamespace(root=root, put=lambda *_:None, run=execute)
        with TemporaryDirectory(dir=root/"runs") as folder:
            run, profile = Path(folder), Path(folder)/"profile"
            store = Store(run/"state.sqlite")
            tasks = [{"id":k, "paths":[k], "cases":[dict(signature(1),source_path=k)], "separate_operations":True}
                     for k in ("alpha","beta")]
            catalog = {"tasks":tasks, "hardware":{"fingerprint":"first"}}
            write_json(profile/"tasks.json",catalog)
            write_json(run/"result.json",{"status":"budget_exhausted"})
            store.set("fingerprint",digest({"config":config,"catalog":catalog,"implementation":implementation_hash()}))
            best = {}
            for task in tasks:
                name = task["id"]
                index = {}
                for num in range(5):
                    value = {"task_id":name, "name":"test", "strategy":"test", "backend":"triton","references":[],
                             "source": RMS_SOURCE + f"\n# variant {num}"}
                    identity = candidate_id(value)
                    write_json(run/"candidates"/(identity+".json"),value)
                    scores[identity] = -1. if name == "beta" and num >= 3 else num + 1.
                    if num == (4 if name == "alpha" else 2):
                        best[name] = identity
                    index[identity] = {"id":identity, "routes":[{"path":name,"signature":signature(1),"saving":num+1.}], "saving":num+1.}
                store.set("decode_candidates:"+name,index)
            devices = [DecodeDevice(i,worker(i),run if i == 0 else run/"peer", profile if i == 0 else run/"peer/profile",str(i))
                       for i in range(2)]
            with patch("sako.decode_validation.decode_devices",return_value=devices), patch("sako.reporting.write_swarm_report"):
                for task in tasks:
                    result = target_boundary(config,run,profile,catalog,store,devices[0].worker,task["id"],0,"finished")
                    self.assertEqual(result["winner"],best[task["id"]])
                    self.assertEqual(len(result["trials"]),2 if task["id"] == "alpha" else 4)
                    self.assertEqual([entry["rank"] for entry in result["skipped_after_pass"]],
                                     [3, 4] if task["id"] == "alpha" else [])
                    target_boundary(config,run,profile,catalog,store,devices[0].worker,task["id"],0,"finished")
                self.assertEqual(len(calls),6)
                self.assertTrue(all(old == [] for _,_,old in calls[:2]))
                self.assertTrue(all(best["alpha"] in chosen and old == [best["alpha"]] for _,chosen,old in calls[2:]))
                self.assertEqual([sum(g == i for g,_,_ in calls) for i in (0,1)], [3,3])
                self.assertEqual(len(store.get("decode_target_plans")["alpha/0"]["finalists"]), 4)
                final_passes = False
                final = integrate(config,run,profile,devices[0].worker)
                self.assertEqual({e["candidate_id"] for e in final["manifest"]["entries"]},set(best.values()))
                self.assertEqual(store.get("status"),"retained_previous")
                self.assertEqual(store.get("best_model_measurement")["candidate_id"],best["alpha"])
                integrate(config,run,profile,devices[0].worker)
                self.assertEqual(len(calls),7)
                self.assertEqual(len(store.get("model_measurements")),7)

    def test_skipped_prompt_rows_do_not_count_as_numerical_failures(self):
        screen = {"clearly_slower": True, "baseline": {"samples_ms":[1.]*5}, "candidate":{"samples_ms":[2.]*5}}
        rows = [{"generation_step":{"phase":"prefill"}, "correct": True, "inputs_unchanged": True},
                {"generation_step":{"phase":"prefill"}, "correct": None},
                {"generation_step":{"phase":"decode"}, "correct": True, "inputs_unchanged": True,
                 "baseline":{"samples_ms":[3.]*5}, "candidate":{"samples_ms":[2.]*5}, "speed_screen": screen}]
        result = {"performance_scope":"cached_decode", "cases":rows, "timed_case_indices":[2],
                  "stages":{"outcome":"screened_slow"}}
        self.assertTrue(measured_gpu_outcome(result))
        self.assertEqual(classify_fault(result=result)["kind"], "speed_screen_rejection")
        result.update(stages={"outcome":"confirmed", "initial_correctness_calls":2, "initial_case_indices":[0,2]},
                      decision={"accepted":True})
        self.assertEqual(classify_fault(result=result)["kind"], "accepted")

    def test_one_gpu_checks_both_ranks_and_keeps_the_model_when_all_fail(self):
        accepted = {"entries":[{"task_id":"old","candidate_id":"previous"}]}
        plans = [{"rank":i, "candidate_id":str(i), "manifest":{"entries":[{"candidate_id":str(i)}]}}
                 for i in range(1,5)]
        device = SimpleNamespace(gpu=0)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = Store(root/"state.sqlite")
            store.set("best_model_measurement", {"manifest":accepted})
            def reject(run, store, gpu, stage, proposal, incumbent, identity):
                self.assertEqual(incumbent, accepted)
                return {"key":identity, "candidate_id":identity, "incumbent":incumbent, "manifest":proposal, "gpu":0,
                        "result":{"scope":"cached_decode","cases":[],"decision":{"accepted":False,"reasons":["slow"]}}}
            with patch("sako.decode_validation.target_finalists", return_value=plans), \
                 patch("sako.decode_validation.decode_devices", return_value=[device]), \
                 patch("sako.decode_validation.measure_version", side_effect=reject):
                outcome = target_boundary({}, root, root, {}, store, None, "mlp", 0, "done")
            self.assertEqual([row["candidate_id"] for row in outcome["trials"]], ["1","2","3","4"])
            self.assertEqual(outcome["manifest"], accepted)
            self.assertIsNone(outcome["winner"])
            self.assertEqual(outcome["skipped_after_pass"], [])

    def test_speed_uses_decode_and_both_references_share_one_candidate_run(self):
        c = load_config(Path(__file__).resolve().parents[1] / "configs/generation.json")
        c["measurement"] = {"mode": "adaptive"}
        w = c["workloads"][0]
        def sample(per_token, total):
            return lambda: {"mean_decode_ms": per_token, "total_ms": total}
        measured = paired_samples(sample(20, 5000), sample(15, 7000), c, incumbent_sample=sample(15.01, 4000))
        case = {**measured, "correct": True, "workload": w, "speed_requirement": c["min_speedup"], "timing_scope": "cached_decode"}
        self.assertEqual(len(case["candidate"]["samples_ms"]), 10)
        self.assertTrue(score_model_cases([case], c, [w])["accepted"])
        self.assertTrue(score_model_cases([case], c, [w], reference_role="incumbent")["accepted"])
        case["baseline"] = case["generation_metrics"]["baseline"]["total_ms"]
        self.assertFalse(score_model_cases([case], c, [w])["accepted"])

    def test_one_call_for_each_layout_even_with_many_sites(self):
        cases = [{**signature(n), "generation_step": {"phase": phase}, "source_path": path}
                 for path in ("a","b") for n,phase in ((24,"prefill"),(1,"decode"),(7,"decode"))]
        self.assertEqual(initial_indices(cases, ["b"], by_layout=True), [3,4,5])

    def test_saved_times_follow_the_device_and_manifest_and_accuracy_blocks_timing(self):
        from unittest.mock import Mock
        from sako.generation import compare_models
        from sako.measurement import saved_decode_time
        c = load_config(Path(__file__).resolve().parents[1] / "configs/generation.json")
        w = c["workloads"][0]
        base = SimpleNamespace(record={}, run=Mock(side_effect=AssertionError("Original ran again.")))
        def runner(ms):
            return SimpleNamespace(record={}, run=Mock(return_value={"mean_decode_ms": ms, "total_ms": 1000.}))
        previous, first, next_version = runner(9.), runner(8.), runner(7.9)
        original = {"identity": {"hardware_fingerprint": "gpu0"}, "identity_hash": "settings",
                    "record_hash": "original", "samples": [{"mean_decode_ms":10., "total_ms":1000.}] * 3}
        with TemporaryDirectory() as folder:
            with patch("sako.measurement.saved_workload", return_value=original), \
                 patch("sako.generation.compiled_request", side_effect=lambda model,*args: model), \
                 patch("sako.generation.check_generation", return_value={"passed": True}) as check:
                def measure(request, version, current):
                    _, timing = compare_models(base, request, c, w, catalog={"starting_baseline":{}, "hardware":{}},
                        manifest={"v":version}, accepted_manifest={"v":current}, reference_directory=folder,
                        make_accepted=lambda: previous)
                    return {**timing, "correct":True, "workload":w}
                case = measure(first, 2, 1)
                self.assertEqual((previous.run.call_count, first.run.call_count), (6, 6))
                self.assertEqual(case["sampling"]["reference_timed_runs"], 3)
                row = measure(next_version, 3, 2)
                self.assertEqual(row["sampling"]["reference_timed_runs"], 0)
                self.assertEqual(row["accepted_record"]["record_hash"], case["decode_record"]["record_hash"])
                self.assertEqual((base.run.call_count, previous.run.call_count, next_version.run.call_count), (0,6,6))
                self.assertTrue(score_model_cases([row], c, [w])["accepted"])
                self.assertTrue(score_model_cases([row], c, [w], reference_role="incumbent")["accepted"])
                self.assertFalse(score_model_cases([row], {**c, "min_speedup":1.02}, [w], reference_role="incumbent")["accepted"])
                peer = {**original, "identity":{"hardware_fingerprint":"gpu1"}}
                self.assertFalse(saved_decode_time(folder, peer, {"v":2}, c, lambda: previous)[2])
                self.assertTrue(saved_decode_time(folder, peer, {"v":2}, c, lambda: None)[2])
                self.assertEqual(previous.run.call_count, 12)
                check.return_value = {"passed": False}
                check_result, failed = compare_models(base, next_version, c, w)
                self.assertFalse(check_result["passed"])
                self.assertEqual(failed["timing_status"], "not_measured")
                self.assertEqual(next_version.run.call_count, 6)

    def test_top_four_for_a_target_preserve_prior_paths_and_keep_all_sources(self):
        task = {"id":"a","paths":["a"],"cases":[dict(signature(n),source_path="a") for n in (24,1)],"separate_operations":True}
        with TemporaryDirectory() as folder:
            store = Store(Path(folder)/"state.sqlite")
            previous = {"entries":[{"task_id":"a","candidate_id":"old","paths":["a"],"signatures":[signature(24),signature(1)]}]}
            index = {str(i): {"id":str(i),"saving":10-i,"routes":[{"path":"a","signature":signature(1),"saving":10-i}]} for i in range(6)}
            store.set("decode_candidates:a",index)
            plans = target_finalists(store,{"tasks":[task]},"a",previous)
            self.assertEqual([p["candidate_id"] for p in plans],["0","1","2","3"])
            self.assertEqual(len(store.get("decode_candidates:a")),6)
            self.assertTrue(all(any(e["candidate_id"] == "old" and e["signatures"] == [signature(24)]
                                    for e in p["manifest"]["entries"]) for p in plans))
