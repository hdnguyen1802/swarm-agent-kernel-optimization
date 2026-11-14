import json
import tempfile
import unittest
from pathlib import Path

from sako.feedback import compact_feedback, save_feedback


class FeedbackEvidence(unittest.TestCase):
    def test_large_errors_remain_complete_on_disk(self):
        with tempfile.TemporaryDirectory() as directory:
            value = {"error": "x" * 50000, "traceback": "trace" * 10000,
                     "cases": [{"correct": False, "seed": 1729, "max_abs_error": .125}],
                     "candidate": {"samples_ms": list(range(10000)), "median_ms": 4.5}}
            summary = save_feedback(directory, value, name="failed-kernel", metadata={"task_id": "linear"})
            stored = json.loads((Path(directory) / summary["artifact"]["path"]).read_text())
            self.assertEqual(stored, value)
            self.assertLess(len(json.dumps(summary)), 3000)
            self.assertEqual(summary["summary"]["cases"][0]["seed"], 1729)
            self.assertTrue(summary["summary"]["traceback_omitted"])

    def test_summary_keeps_failure_before_success(self):
        result = compact_feedback({"cases": [{"correct": True, "seed": 1},
                                  {"correct": False, "seed": 2}]}, max_items=1)
        self.assertEqual(result["cases"][0]["seed"], 2)


if __name__ == "__main__":
    unittest.main()
