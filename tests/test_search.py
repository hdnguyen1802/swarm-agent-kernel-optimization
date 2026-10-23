import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sako.core import load_config, write_json, read_json
from sako.search import search
from sako.store import Store
from sako.store import BudgetExhausted

ROOT = Path(__file__).resolve().parents[1]


class FakeGemini:
    calls = []
    prompts = []

    def __init__(self, *args):
        pass

    def call(self, role, prompt, system):
        self.calls.append(role)
        data = json.loads(prompt)
        self.prompts.append((role, data))
        if role == "coordinator":
            return {"task_order": [], "plan": "Test the kernel."}
        if not role.startswith("writer_"):
            return {"notes": "Use measured results."}
        return {"name": role, "task_id": data["task"]["id"], "uses_cutlass": False, "references": [],
                "strategy": data["island"], "source": "sako_run(); // " + role + str(data["generation"])}


class FakeWorker:
    calls = []
    model_failures = 0
    operation_failures = False
    stop_after_model = False

    def __init__(self, *args):
        pass

    def put(self, *args):
        pass

    def run(self, action, *args, **kwargs):
        self.calls.append(action)
        if action == "evaluate" and self.operation_failures:
            c = read_json(ROOT / kwargs["candidate"])
            if c["source"].endswith("0"):
                return {"cases": [{"correct": False, "checks": [{"max_abs_error": .078125}]}],
                        "decision": {"accepted": False, "speedup": None}}
        if action == "model" and self.stop_after_model:
            (Path(args[0]) / "STOP").touch()
        if action == "model" and self.model_failures:
            type(self).model_failures -= 1
            return {"cases": [{"workload": {"batch_size": 1, "sequence_length": 128},
                    "correct": False, "checks": [{"passed": False, "max_abs_error": .078125}]}],
                    "decision": {"accepted": False, "speedup": None, "reasons": ["Correctness failed."]}}
        return {"decision": {"accepted": True, "speedup": 1.2}, "scope": action}


class SearchLoop(unittest.TestCase):
    def setUp(self):
        FakeGemini.calls, FakeGemini.prompts, FakeWorker.calls = [], [], []
        FakeWorker.model_failures = 0
        FakeWorker.operation_failures = False
        FakeWorker.stop_after_model = False

    def test_eight_writers_feedback_selection_and_resume(self):
        (ROOT / "runs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as directory:
            root = Path(directory)
            config = load_config(ROOT / "configs/project.json")
            config["agent"]["generations"] = 2
            tasks = [{"id": f"linear_{i}", "kind": "linear", "paths": [f"model.layer{i}"],
                      "priority_ms": 4-i, "cases": [], "scalars": []} for i in range(4)]
            write_json(root / "profile/tasks.json", {"config": config, "tasks": tasks})
            FakeGemini.calls, FakeWorker.calls = [], []
            with patch("sako.search.Gemini", FakeGemini), patch("sako.search.Worker", FakeWorker), \
                 patch("sako.search.retrieve", return_value=[]):
                result = search(config, root / "run", root / "profile")
                self.assertEqual(len([r for r in FakeGemini.calls if r.startswith("writer_")]), 8)
                self.assertEqual(len(result["manifest"]["entries"]), 4)
                self.assertTrue(result["model_result"]["decision"]["accepted"])
                count = len(FakeGemini.calls)
                search(config, root / "run", root / "profile")
                self.assertEqual(len(FakeGemini.calls), count)
                self.assertEqual(FakeWorker.calls.count("model"), 1)
            rows = Store(root / "run/state.sqlite").candidates()
            self.assertEqual(len(rows), 8)

    def repair_run(self, failure, stop=False, limit=3):
        (ROOT / "runs").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=ROOT / "runs")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        config = load_config(ROOT / "configs/project.json")
        config["agent"].update(generations=2, max_repair_attempts=limit)
        tasks = [{"id": f"linear_{i}", "kind": "linear", "paths": [f"model.layer{i}"],
                  "priority_ms": 8-i, "cases": [], "scalars": []} for i in range(8)]
        write_json(root / "profile/tasks.json", {"config": config, "tasks": tasks})
        FakeWorker.model_failures = failure if type(failure) is int else 0
        FakeWorker.operation_failures = failure == "operation"
        FakeWorker.stop_after_model = stop
        self.enterContext(patch("sako.search.Gemini", FakeGemini))
        self.enterContext(patch("sako.search.Worker", FakeWorker))
        self.enterContext(patch("sako.search.retrieve", return_value=[]))
        result = search(config, root / "run", root / "profile")
        return root, config, result

    def assert_repair_prompts(self, root, stage):
        repairs = [p for role, p in FakeGemini.prompts if role.startswith("writer_") and p.get("repair")]
        self.assertTrue(repairs)
        for p in repairs:
            self.assertEqual(p["repair"]["stage"], stage)
            self.assertEqual(p["task"]["id"], p["repair"]["task_id"])
            self.assertEqual(p["parent"], read_json(root / "run/candidates" / (p["repair"]["candidate_id"] + ".json")))
            self.assertIn(".078125", json.dumps(p["repair"]["result"]))
        return repairs

    def test_model_failure_returns_source_and_errors_to_same_writer(self):
        root, _, result = self.repair_run(1)
        repairs = self.assert_repair_prompts(root, "model")
        for p in repairs:
            old = next(q for role, q in FakeGemini.prompts if role.startswith("writer_")
                       and q["writer"] == p["writer"] and q["generation"] == 0)
            self.assertEqual(old["task"]["id"], p["task"]["id"])
        self.assertEqual(FakeWorker.calls.count("model"), 2)

        self.assertTrue(result["model_result"]["decision"]["accepted"])
        blocked = Store(root / "run/state.sqlite").get("blocked_model_candidates")
        self.assertTrue(blocked)
        self.assertTrue(all(e["candidate_id"] not in blocked for e in result["manifest"]["entries"]))

    def test_operation_failure_is_repaired_without_accepted_parent(self):
        root, _, result = self.repair_run("operation")
        self.assert_repair_prompts(root, "operation")
        self.assertTrue(result["model_result"]["decision"]["accepted"])

    def test_stop_and_resume_keep_model_repair_assignment(self):
        root, config, result = self.repair_run(1, stop=True)
        self.assertEqual(result["status"], "stopped")
        self.assertTrue(result["pending_repairs"])
        (root / "run/STOP").unlink()
        FakeWorker.stop_after_model = False
        result = search(config, root / "run", root / "profile")
        self.assert_repair_prompts(root, "model")
        self.assertTrue(result["model_result"]["decision"]["accepted"])

    def test_repair_limit_and_failed_model_cannot_pass(self):
        root, _, result = self.repair_run(2, limit=1)
        self.assertFalse(result["model_result"]["decision"]["accepted"])
        self.assertTrue(all(r["task_id"] in {f"linear_{i}" for i in range(4, 8)}
                            for r in result["pending_repairs"].values()))
        self.assertEqual(result["status"], "generation_limit_reached")
        self.assertEqual(FakeWorker.calls.count("model"), 2)

        # Returning the same exhausted source must not reset its repair count.
        from sako.loop import SearchLoop as Controller
        from sako.search import make_manifest
        controller = Controller(load_config(ROOT / "configs/project.json"), root / "unused",
            root / "profile", FakeGemini, FakeWorker, lambda *args: [], make_manifest)
        controller.store = Store(root / "run/state.sqlite")
        controller.limits["max_repair_attempts"] = 1
        controller.pending = result["pending_repairs"].copy()
        exhausted = next(r for r in controller.store.candidates() if r["generation"] == 1
                         and r["task"] == "linear_0")
        controller.queue(0, exhausted["id"], "model", result["model_result"], 0)
        self.assertNotIn(exhausted["id"], controller.pending)

    def test_budget_stops_repairs_and_keeps_failures(self):
        original = FakeGemini.call
        def call(client, role, prompt, system):
            data = json.loads(prompt)
            if role.startswith("writer_") and data["generation"] > 0:
                raise BudgetExhausted("Test budget exhausted.")
            return original(client, role, prompt, system)
        with patch.object(FakeGemini, "call", call):
            root, _, result = self.repair_run(2)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertTrue(result["pending_repairs"])
        self.assertFalse(result["model_result"]["decision"]["accepted"])

    def test_model_worker_error_is_returned_for_repair(self):
        original = FakeWorker.run
        failed = []
        def run(worker, action, *args, **kwargs):
            if action == "model" and not failed:
                failed.append(True)
                raise RuntimeError("Model compiler failed.")
            return original(worker, action, *args, **kwargs)
        with patch.object(FakeWorker, "run", run):
            _, _, result = self.repair_run(0)
        repairs = [p for role, p in FakeGemini.prompts if role.startswith("writer_") and p.get("repair")]
        self.assertTrue(repairs)
        self.assertTrue(all(p["repair"]["result"]["error"] == "Model compiler failed." for p in repairs))
        self.assertTrue(result["model_result"]["decision"]["accepted"])

    def test_later_failure_preserves_best_model_and_all_repair_tasks(self):
        original = FakeWorker.run
        measurements = []
        def run(worker, action, *args, **kwargs):
            if action == "evaluate":
                c = read_json(ROOT / kwargs["candidate"])
                if c["task_id"] == "linear_3" and c["source"].endswith("0"):
                    return {"decision": {"accepted": False, "speedup": None}}
            if action == "model":
                worker.calls.append(action)
                manifest = read_json(ROOT / kwargs["manifest"])
                measurements.append(manifest)
                return {"decision": {"accepted": len(measurements) == 1, "speedup": 1.2},
                        "manifest": manifest}
            return original(worker, action, *args, **kwargs)
        with patch.object(FakeWorker, "run", run):
            _, _, result = self.repair_run(0)
        self.assertEqual(len(measurements), 2)
        self.assertEqual(result["manifest"], measurements[0])
        self.assertEqual(result["model_result"]["manifest"], measurements[0])
        self.assertTrue(result["model_result"]["decision"]["accepted"])
        self.assertEqual({r["candidate_id"] for r in result["pending_repairs"].values()},
                         {e["candidate_id"] for e in measurements[1]["entries"]})


if __name__ == "__main__":
    unittest.main()
