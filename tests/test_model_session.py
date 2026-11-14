from triton_fixtures import RMS_SOURCE
import copy
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sako.model_session import ModelSession
from sako.core import write_json
from sako.worker import run_payload


def serve_cpu_fixture(connection):
    """Use the real child protocol and operation gates with a CPU timing stub."""
    from sako.fixture_worker import serve
    def time_fixture(*args, **kwargs):
        return [{"correct": True, "worker_pid": os.getpid(), "baseline": {"samples_ms": [2.]},
                 "candidate": {"samples_ms": [1.]}}]
    torch = SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None))
    with patch("sako.gpu.evaluate_fixture", side_effect=time_fixture), patch.dict("sys.modules", {"torch": torch}):
        serve(connection)


class Tensor:
    dtype = "float16"
    def __init__(self, values):
        self.values = list(values)
        self.shape = (len(self.values),)
    def detach(self): return self
    def to(self, **kwargs): return copy.deepcopy(self)
    def cpu(self): return copy.deepcopy(self)
    def reshape(self, *args): return self
    def numel(self): return len(self.values)
    def element_size(self): return 2
    def __getitem__(self, index): return Tensor(self.values[index])


class Model:
    def __init__(self):
        self.weights = {"weight": Tensor([1., 2.]), "other": Tensor([3.])}
    def named_parameters(self): return self.weights.items()


class ModelSessionTests(unittest.TestCase):
    def test_cached_baseline_does_not_scan_weights_without_model_access(self):
        from sako.model_session import ModelSession
        session = ModelSession()
        session.model = object()
        session.job_stats = {"model_accessed": False}
        with patch.object(session, 'verify', side_effect=AssertionError('Metadata job scanned weights.')):
            self.assertTrue(session.verify_after_job('baseline'))
        self.assertEqual(session.job_stats['integrity_check'], 'metadata_only')
        session.job_stats['model_accessed'] = True
        with patch.object(session, 'verify', return_value=False) as verify:
            self.assertFalse(session.verify_after_job('baseline'))
        verify.assert_called_once()

    def setUp(self):
        self.model = Model()
        self.session = ModelSession()
        self.session.begin()
        self.torch = SimpleNamespace(equal=lambda a,b:a.values==b.values,
            compile=lambda function, **kwargs:function,
            cuda=SimpleNamespace(empty_cache=lambda:None, get_device_name=lambda n:"test"),
            __version__="test",version=SimpleNamespace(cuda="test"))
        self.enterContext(patch.dict("sys.modules", {"torch":self.torch}))
        self.enterContext(patch("sako.model.install_attention_dispatch"))
        self.config = {"model_id":"test", "revision":"pinned", "compile_mode":"default"}

    def test_model_and_compiled_reference_reuse_without_reloading_weights(self):
        with patch("sako.model.load_model", return_value=self.model) as load, \
                patch.object(self.torch,"compile",wraps=self.torch.compile) as compile:
            self.session.get_model(self.config)
            first = self.session.get_baseline(self.config)
            self.assertFalse(self.session.job_stats["model_cache_hit"])
            self.assertFalse(self.session.job_stats["baseline_cache_hit"])
            self.assertTrue(self.session.verify())
            self.session.begin()
            second = self.session.get_baseline(self.config)
            self.assertIs(first,second)
            self.assertTrue(self.session.job_stats["model_cache_hit"])
            self.assertTrue(self.session.job_stats["baseline_cache_hit"])
            self.assertEqual(load.call_count,1)
            self.assertEqual(compile.call_count,1)

    def test_changed_reference_contract_drops_model_and_compiled_callable(self):
        with patch("sako.model.load_model", side_effect=[self.model,Model()]) as load:
            first=self.session.get_baseline(self.config)
            second=self.session.get_baseline({**self.config,"revision":"other-pinned"})
            self.assertIsNot(first,second)
            self.assertEqual(load.call_count,2)

    def test_exact_parameter_check_catches_native_write_without_version_counter(self):
        with patch("sako.model.load_model",return_value=self.model):
            self.session.get_model(self.config)
        self.model.weights["other"].values[0] += .00001
        self.assertFalse(self.session.verify())
        self.assertTrue(self.session.retire)

    def test_changed_parameter_metadata_fails_state_check(self):
        with patch("sako.model.load_model",return_value=self.model):
            self.session.get_model(self.config)
        self.model.weights["weight"].shape=(1,2)
        self.assertFalse(self.session.verify())

    def test_saved_fixtures_run_in_a_reused_process_without_a_weight_scan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "tasks.json", {"tasks": [{"id": "op", "kind": "linear", "cases": [{}]}]})
            write_json(root / "candidate.json", {"name": "trial", "task_id": "op", "strategy": "Test the fixture.",
                       "backend": "triton", "references": [], "source": RMS_SOURCE})
            job = {"action": "evaluate", "profile_dir": str(root), "candidate": str(root / "candidate.json"),
                   "config": "test"}
            config = {"trials": 1, "min_speedup": 1.01, "max_p95_ratio": 1.05}
            self.session.log_path = root / "worker.log"
            with patch("sako.model.load_model", return_value=self.model):
                self.session.get_model(self.config)
            with patch("sako.fixture_worker.serve", serve_cpu_fixture), \
                    patch("sako.worker.load_config", return_value=config), patch("sako.preparation.validate_catalog"), \
                    patch("sako.hardware.discover_hardware", return_value={}), patch.object(self.session, "verify") as scan:
                try:
                    first = run_payload(job, session=self.session)
                    second = run_payload(job, session=self.session)
                finally:
                    self.session.close_fixtures()
                scan.assert_not_called()
            for reply in (first, second):
                self.assertTrue(reply["ok"], reply.get("error"))
                self.assertTrue(reply["result"]["decision"]["accepted"])
                self.assertEqual(reply["result"]["resident_cache"]["integrity_check"], "fixture_only")
            pid = first["result"]["cases"][0]["worker_pid"]
            self.assertNotEqual(pid, os.getpid())
            self.assertEqual(pid, second["result"]["cases"][0]["worker_pid"])
            self.assertIs(self.session.model, self.model)

    def test_fixture_capture_and_same_process_native_jobs_still_check_all_weights(self):
        with patch("sako.model.load_model", return_value=self.model):
            self.session.get_model(self.config)
        with patch.object(self.session, "verify", wraps=self.session.verify) as scan:
            self.session.begin()
            self.session.get_baseline(self.config)
            self.session.job_stats["fixture_worker"] = True
            self.assertTrue(self.session.verify_after_job("evaluate"))
            scan.assert_called_once()
            self.session.begin()
            self.model.weights["other"].values[0] += 1
            self.assertFalse(self.session.verify_after_job("evaluate"))
            self.assertTrue(self.session.retire)

    def test_unsafe_state_cannot_accept_saved_timings(self):
        value={"reference_state_intact":True,"decision":{"accepted":True,"reasons":[]},
               "cases":[{"correct":True,"baseline":{"samples_ms":[2.]*5},"candidate":{"samples_ms":[1.]*5}}]}
        with patch("sako.worker.execute",return_value=value),patch.object(self.session,"verify",return_value=False):
            result=run_payload({"action":"model"},session=self.session)
        self.assertFalse(result["result"]["decision"]["accepted"])
        self.assertFalse(result["result"]["cases"][0]["correct"])
        self.assertEqual(result["result"]["cases"][0]["candidate"]["samples_ms"],[1.]*5)
        self.assertTrue(self.session.retire)

    def test_integrity_exception_cannot_leave_success_flag_true(self):
        value={"decision":{"accepted":True,"reasons":[]},"cases":[{"correct":True}]}
        with patch("sako.worker.execute",return_value=value), \
                patch.object(self.session,"verify",side_effect=RuntimeError("CUDA state cannot be read.")):
            result=run_payload({"action":"model"},session=self.session)
        self.assertFalse(result["ok"])
        self.assertTrue(self.session.retire)


if __name__=="__main__":
    unittest.main()
