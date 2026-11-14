import unittest


class TimingProgress(unittest.TestCase):
    def row(self, samples, score=1.02, correct=True):
        return {"id": "candidate", "correct": correct, "score": score,
                "result": {"cases": [{"baseline": {"samples_ms": [1.] * len(samples)},
                                       "candidate": {"samples_ms": samples}}]}}

    def test_noise_does_not_extend_an_unproductive_search(self):
        from sako.swarm_policy import measured_progress
        noisy = self.row([.5, 1., 1.5] * 5)
        self.assertFalse(measured_progress([noisy], 1.)["improved"])
        self.assertTrue(measured_progress([self.row([1.] * 15)], 1.)["improved"])

    def test_wrong_outputs_cannot_count_as_search_progress(self):
        from sako.swarm_policy import measured_progress
        self.assertFalse(measured_progress([self.row([1.] * 15, score=10., correct=False)], 1.)["improved"])


if __name__ == "__main__":
    unittest.main()
