"""Candidate lookup must not decode unrelated source or timing records."""

import tempfile
import unittest
from pathlib import Path

from sako.store import Store
from triton_fixtures import RMS_SOURCE


class CandidateReadTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.store = Store(Path(temporary.name) / "state.sqlite")
        self.key, _ = self.store.add({"task_id": "linear_1", "source": RMS_SOURCE, "backend": "triton",
                                      "name": "lookup", "strategy": "Read the saved record.", "references": []},
                                     "memory", 1, None)
        self.store.record(self.key, {"cases": [{"correct": True}],
                                     "decision": {"accepted": False, "speedup": .9}})

    def test_direct_lookup_preserves_the_complete_record(self):
        self.assertEqual(self.store.candidate(self.key), self.store.candidates()[0])
        self.assertTrue(self.store.candidate(self.key)["correct"])
        self.assertFalse(self.store.candidate(self.key)["accepted"])
        self.assertIsNone(self.store.candidate("missing"))
        self.assertIsNone(self.store.candidate("' OR 1=1 --"))

    def test_lookup_and_status_do_not_decode_unrelated_payloads(self):
        with self.store.connect() as database:
            database.execute("INSERT INTO candidates VALUES(?,?,?,?,?,?,?,?,?)",
                             ("other", "linear_2", "compute", 2, None, "invalid", "invalid", 1., 0))
        self.assertEqual(self.store.candidate(self.key)["id"], self.key)
        summaries = self.store.candidate_summaries()
        self.assertEqual([row["id"] for row in summaries], [self.key, "other"])
        self.assertTrue(all(set(row) == {"id", "task", "island", "score", "accepted"} for row in summaries))
