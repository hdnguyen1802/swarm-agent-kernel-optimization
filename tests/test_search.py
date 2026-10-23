import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sako.core import load_config, write_json, read_json
from sako.search import search
from sako.store import Store

ROOT = Path(__file__).resolve().parents[1]


class FakeGemini:
    calls = []

    def __init__(self, *args):
        pass

    def call(self, role, prompt, system):
        self.calls.append(role)
        data = json.loads(prompt)
        if role == "coordinator":
            return {"task_order": [], "plan": "Test the kernel."}
        if not role.startswith("writer_"):
            return {"notes": "Use measured results."}
        return {"name": role, "task_id": data["task"]["id"], "uses_cutlass": False, "references": [],
                "strategy": data["island"], "source": "sako_run(); // " + role + str(data["generation"])}


class FakeWorker:
    calls = []

    def __init__(self, *args):
        pass

    def put(self, *args):
        pass

    def run(self, action, *args, **kwargs):
        self.calls.append(action)
        return {"decision": {"accepted": True, "speedup": 1.2}, "scope": action}


class SearchLoop(unittest.TestCase):
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
                self.assertEqual(len([r for r in FakeGemini.calls if r.startswith("writer_")]), 16)
                self.assertEqual(len(result["manifest"]["entries"]), 4)
                self.assertTrue(result["model_result"]["decision"]["accepted"])
                count = len(FakeGemini.calls)
                search(config, root / "run", root / "profile")
                self.assertEqual(len(FakeGemini.calls), count)
                self.assertEqual(FakeWorker.calls.count("model"), 1)
            rows = Store(root / "run/state.sqlite").candidates()
            self.assertEqual(len(rows), 16)
            self.assertTrue(any(r["parent"] for r in rows))


if __name__ == "__main__":
    unittest.main()
