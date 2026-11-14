"""Protect structural grouping, experiment variety and failure attribution."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from sako.designs import describe, writer_options
from sako.integration_repairs import record_failure
from sako.store import Store


class DesignSearchTests(unittest.TestCase):
    def test_compile_errors_and_failed_repairs_remain_available_as_model_feedback(self):
        from sako.decode_repairs import repair_failures, remember_failures
        compile_error = {"error":"Unsupported: call_method GetAttrVariable best_config kwargs",
                         "decision":{"accepted":False}}
        self.assertEqual(repair_failures(compile_error)[0]["scope"],"model_execution")
        self.assertEqual(repair_failures({"error":"connection lost","failure_kind":"infrastructure"}),[])
        with TemporaryDirectory() as tmp:
            store=Store(Path(tmp)/"state.sqlite")
            trial={"key":"repair-test","candidate_id":"fixed","manifest":{"entries":[]},
                   "incumbent":None,"result":compile_error}
            remember_failures(store,"task",[trial],None)
            self.assertEqual(store.get("rejected_model_candidates")["fixed"],trial)
            self.assertEqual(store.get("integration_repair_tickets")["fixed"]["status"],"awaiting_isolation")

    def budget_loop(self):
        from test_swarm_loop import SwarmLoopTests
        fixture = SwarmLoopTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.config["generation_objective"] = "decode"
        fixture.config["kernel_search"] = {"enabled":True}
        fixture.config["agent"].update(model_repair_fraction=.2,revisit_with_budget=True)
        return fixture.loop()

    def test_small_budget_preserves_one_experiment_before_repair_reserve(self):
        loop = self.budget_loop()
        task = loop.active_task
        loop.task_starts[task] = 953675
        loop.task_limits[task] = 1000000
        loop.costs = lambda: {"writer":25000,"coordinator":20000}
        self.assertEqual(loop.repair_reserve(),1325)
        loop.task_starts[task] = 0
        self.assertEqual(loop.repair_reserve(),200000)

    def test_revisit_resets_the_start_of_its_token_share(self):
        loop = self.budget_loop()
        task = loop.active_task
        loop.weighted_budget = False
        loop.task_starts[task] = 0
        loop.costs = lambda: {"writer":10000,"coordinator":10000}
        number = loop.store.reserve("writer_0_memory",700000,loop.limits)
        loop.store.finish_call(number,700000)
        self.assertTrue(loop.revisit_with_budget())
        self.assertEqual(loop.task_starts[task],700000)
        self.assertEqual(loop.store.get("swarm_task_starts")[task],700000)

    def test_empty_revisit_does_not_repeat_model_tests(self):
        loop = self.budget_loop()
        task = loop.active_task
        loop.progress["budget_visits"] = {task: {"count":1,"first_generation":0}}
        with patch("sako.decode_validation.target_boundary") as measure:
            loop.validate_decode_target("token_budget")
        measure.assert_not_called()
        self.assertEqual(loop.events[-1]["kind"],"target_model_selection_skipped")

    def test_controller_dispatches_the_offered_experiments(self):
        from test_swarm_loop import SwarmLoopTests
        fixture = SwarmLoopTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.config["agent"].update(design_portfolio=True,max_task_experiments=10)
        fixture.worker.operation_accepted = fixture.worker.model_correct = True
        def plan(data, number):
            hypotheses=[]
            parents=data["archive_policy"]["eligible_parent_ids"]
            for option in data["writer_experiments"]:
                parent=option["parent_id"]
                hypotheses.append({"specialty":"memory","parent_id":parent,"experiment_role":option["role"],
                    "objective":option["objective"],"claim":"The measured design can improve.",
                    "change":option["objective"] + " Experiment " + str(number),
                    "expected_metric":"Reduce operation latency.","evidence":[parent] if parent else
                        parents[:1] or ["contract:"+data["task"]["id"]],
                    "fresh_start_reason":"Test another structure after considering the archive.",
                    **({"merge":option["merge"]} if option.get("merge") else {})})
            return {"action":"continue","reason":"Test the different offered changes.",
                    "evidence":["contract:"+data["task"]["id"]],"hypotheses":hypotheses}
        fixture.client.plan_callback=plan
        loop=fixture.loop()
        result=loop.run()
        self.assertEqual(result["status"],"accepted",result.get("swarm_stop"))
        roles=[a["hypothesis"].get("experiment_role") for a in loop.work.values()]
        self.assertEqual(len(roles),10)
        self.assertEqual(len(set(roles[:5])),5)

    def test_structure_ignores_descriptions_and_settings_but_keeps_branches_and_reductions(self):
        source = "def sako_run(x):\n    kernel[(1,)](x, num_warps=4)\n"
        self.assertEqual(describe(source)["group"], describe("# other design\n" + source.replace("=4", "=8"))["group"])
        staged = source + "    reduce[(1,)](x)\n"
        self.assertNotEqual(describe(source)["group"], describe(staged)["group"])
        branch = "def sako_run(x):\n    if x:\n        first[(1,)](x)\n    else:\n        second[(1,)](x)\n"
        self.assertEqual(describe(branch)["features"]["sequential_launch_sites"], 1)
        self.assertEqual(describe(staged)["features"]["sequential_launch_sites"], 2)

    def test_parallel_work_refines_two_best_paths_and_another_design(self):
        archive = [{"id":"best","score":2.,"code_design":{"group":"a"}},
                   {"id":"same","score":1.99,"code_design":{"group":"a"}},
                   {"id":"other","score":1.8,"code_design":{"group":"b"}}]
        options = writer_options(archive, [], [], 5)
        self.assertEqual([o["parent_id"] for o in options[:4]], ["best","best","other",None])
        self.assertEqual(len({o["role"] for o in options}), 5)
        active = [{"status":"running","hypothesis":{"experiment_role":options[0]["role"]}}]
        self.assertNotIn(options[0]["role"], [o["role"] for o in writer_options(archive, [], active, 4)])

    def test_isolated_combination_failure_only_blocks_the_changed_source(self):
        with TemporaryDirectory() as tmp:
            store = Store(Path(tmp)/"state.sqlite")
            old = {"candidate_id":"kept","task_id":"first","paths":["first"]}
            new = {"candidate_id":"new","task_id":"second","paths":["second"]}
            accepted = {"manifest":{"entries":[old]},"result":{"cases":[{"correct":True}]}}
            store.set("best_model_measurement", accepted)
            trial = {"candidate_id":"new","manifest":{"entries":[old,new]}, "incumbent":{"entries":[old]},
                "result":{"cases":[{"correct":False,"checks":[{"scope":"cached_generation","passed":False,
                    "logits":{"passed":False,"max_abs_error":.1,"atol":.01,"examples":[{"actual":1.1,"expected":1.}]} }]}]}}
            isolated = {"key":"isolation","manifest":{"entries":[new]},"result":{"cases":[{"correct":True}]}}
            ticket = record_failure(store, trial, changed_candidate="new", isolation=isolated)[0]
            self.assertEqual(ticket["attribution"],"interaction_with_accepted_model")
            self.assertEqual(set(store.get("blocked_model_candidates")), {"new"})
            self.assertEqual(store.get("best_model_measurement"), accepted)
            self.assertEqual(ticket["failures"][0]["check"]["logits"]["examples"][0]["actual"],1.1)
            with self.assertRaises(ValueError):
                record_failure(store, trial, changed_candidate="kept", isolation=isolated)

    def test_isolation_checks_correctness_without_timing(self):
        from sako.generation import compare_models
        request = Mock(record={})
        with patch("sako.generation.compiled_request", return_value=request), \
             patch("sako.generation.check_generation", return_value={"passed":True}), \
             patch("sako.generation.paired_generation", side_effect=AssertionError("Timed isolation")):
            check, result = compare_models(None,None,{}, {},correctness_only=True)
        self.assertTrue(check["passed"])
        self.assertEqual(result["timing_status"],"not_measured")
        request.run.assert_not_called()

    def test_repair_isolated_before_writer_and_resume_keeps_completed_work(self):
        self.repair_path()

    def test_compiler_failure_is_isolated_and_sent_to_repair_writer(self):
        self.repair_path(compile_error=True)

    def repair_path(self, compile_error=False):
        from sako.core import write_json
        from sako.decode_repairs import repair_target
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = Store(root/"state.sqlite")
            original = {"candidate_id":"old", "task_id":"a", "paths":["a"]}
            changed = {"candidate_id":"broken", "task_id":"b", "paths":["b"]}
            result = {"cases":[{"correct":False,"checks":[{"scope":"cached_generation","passed":False,
                       "logits":{"passed":False,"max_abs_error":.1,"examples":[{"actual":.2,"expected":.1}]}}]}]}
            if compile_error:
                result = {"error":"Unsupported: best_config.kwargs inside torch.compile",
                          "decision":{"accepted":False},"cases":[]}
            trial = {"key":"trial", "candidate_id":"broken", "manifest":{"entries":[original,changed]},
                     "incumbent":{"entries":[original]}, "result":result}
            store.set("model_measurements", {"trial":trial})
            write_json(root/"candidates/old.json", {"source":"accepted kernel source"})
            loop = SimpleNamespace(active_task="b", store=store, directory=root, profile=root, catalog={},
                config={"atol":.01,"rtol":.01}, limits={"max_total_tokens":10000}, task_limits={"b":10000}, work={},
                worker=Mock(), hypotheses=Mock(), costs=lambda:{"writer":100}, repair_reserve=lambda:2000,
                evidence_report=lambda *args:{"artifact":{"path":"failure.json"}}, event=Mock())
            order=[]
            def assign(h, slot, plan):
                a={"generation":0,"status":"planned","repair":loop.pending["broken"]}
                loop.work["0"]=a
                return a
            loop.assignment=assign
            loop.save_work=lambda a,**kwargs:a.update(kwargs)
            def write(a):
                order.append("write")
                self.assertEqual(a["repair"]["decode_context"]["isolated_correct"],not compile_error)
                if compile_error:
                    self.assertIn("Unsupported",a["repair"]["decode_context"]["execution_error"])
                self.assertEqual(a["repair"]["decode_context"]["accepted_kernels"][0]["source"],"accepted kernel source")
            loop.write_swarm=write
            def evaluate(a,d):
                order.append("evaluate")
                a.update(status="completed",candidate_id="fixed")
            loop.process_candidate=evaluate
            def isolate(*args,**kwargs):
                self.assertTrue(kwargs["correctness_only"])
                order.append("isolate")
                return {"key":"isolated","manifest":{"entries":[changed]},
                        "result":result if compile_error else {"cases":[{"correct":True}]}}
            def validate(*args,**kwargs):
                self.assertEqual(kwargs["repair_ids"],["fixed"])
                order.append("validate")
                return {"winner":"fixed","trials":[{"candidate_id":"fixed","correct":True}]}
            summary={"winner":None,"trials":[{"key":"trial"}]}
            with patch("sako.decode_repairs.decode_devices",return_value=[SimpleNamespace(gpu=0)]), \
                 patch("sako.decode_repairs.measure_version",side_effect=isolate), \
                 patch("sako.decode_repairs.target_boundary",side_effect=validate):
                repair_target(loop,summary,0,"test")
                repair_target(loop,summary,0,"test")
            self.assertEqual(order,["isolate","write","evaluate","validate"])
            self.assertEqual(store.get("integration_repair_tickets")["broken"]["status"],"corrected")
