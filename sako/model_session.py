"""Keep one trusted reference model and compiled callable resident."""

import gc
import time
import sys

from .core import digest
from .preparation import PROFILE_FIELDS


class ModelSession:
    def __init__(self):
        self.key = self.model = self.baseline = None
        self.shadow = []
        self.retire = False
        self.job_stats = {}
        self.fixture_worker = None
        self.log_path = None

    def begin(self):
        torch = sys.modules.get("torch")
        reset = getattr(getattr(torch, "cuda", None), "reset_peak_memory_stats", None)
        if callable(reset):
            reset()
        self.job_stats = {"model_cache_hit": False, "baseline_cache_hit": False,
                          "model_accessed": False, "fixture_worker": False,
                          "model_load_s": 0., "baseline_wrapper_s": 0., "integrity_check_s": 0.}

    def get_model(self, config):
        self.close_fixtures()
        self.job_stats["model_accessed"] = True
        import torch
        from .model import load_model, install_attention_dispatch
        key = digest({field: config.get(field) for field in PROFILE_FIELDS})
        install_attention_dispatch()
        if self.key == key and self.model is not None:
            self.job_stats["model_cache_hit"] = True
            return self.model
        self.clear()
        start = time.perf_counter()
        self.model = load_model(config)
        # Exact CPU copies protect the resident reference after native code runs.
        # Do not keep a second full copy on the GPU.
        self.shadow = []
        for name, value in self.model.named_parameters():
            if getattr(value, "is_cuda", False):
                copy = torch.empty(value.shape, dtype=value.dtype, device="cpu", pin_memory=True)
                copy.copy_(value.detach(), non_blocking=True)
            else:
                copy = value.detach().to(device="cpu", copy=True)
            self.shadow.append((name, copy))
        if any(getattr(value, "is_cuda", False) for _, value in self.model.named_parameters()):
            torch.cuda.synchronize()
        self.key = key
        self.job_stats["model_load_s"] += time.perf_counter() - start
        return self.model

    def get_baseline(self, config):
        self.close_fixtures()
        self.job_stats["model_accessed"] = True
        import torch
        from .model import forward, install_attention_dispatch
        key = digest({field: config.get(field) for field in PROFILE_FIELDS})
        model = self.model if self.key == key and self.model is not None else self.get_model(config)
        install_attention_dispatch()
        if self.baseline is not None:
            self.job_stats["baseline_cache_hit"] = True
            self.job_stats["model_cache_hit"] = True
            return self.baseline
        start = time.perf_counter()
        self.baseline = torch.compile(lambda ids: forward(model, ids), fullgraph=True,
                                      dynamic=False, mode=config["compile_mode"])
        self.job_stats["baseline_wrapper_s"] += time.perf_counter() - start
        return self.baseline

    def evaluate_fixtures(self, job, config):
        from .fixture_worker import FixtureWorker
        if self.fixture_worker is None:
            import torch
            torch.cuda.empty_cache()
            self.fixture_worker = FixtureWorker()
        self.job_stats["fixture_worker"] = True
        return self.fixture_worker.evaluate(job, config, self.log_path)

    def close_fixtures(self):
        if self.fixture_worker is not None:
            self.fixture_worker.close()
            self.fixture_worker = None

    def verify_after_job(self, action):
        if action in {"doctor", "baseline"} and not self.job_stats["model_accessed"]:
            self.job_stats["integrity_check"] = "metadata_only"
            return True
        if action == "evaluate" and not self.job_stats["model_accessed"] and (
                self.model is None or self.job_stats["fixture_worker"]):
            self.job_stats["integrity_check"] = "fixture_only"
            return True
        self.job_stats["integrity_check"] = "exact" if self.model is not None else "no_resident_model"
        return self.verify()

    def verify(self):
        """Check exact parameter values outside every timed region."""
        if self.model is None:
            return True
        import torch
        start = time.perf_counter()
        try:
            current = dict(self.model.named_parameters())
            if set(current) != {name for name, _ in self.shadow}:
                self.retire = True
                return False
            flags, buffers = [], {}
            for name, expected in self.shadow:
                actual = current[name].detach()
                if actual.shape != expected.shape or actual.dtype != expected.dtype:
                    self.retire = True
                    return False
                actual, expected = actual.reshape(-1), expected.reshape(-1)
                step = max(1, 64 * 1024 * 1024 // expected.element_size())
                if getattr(actual, "is_cuda", False):
                    key = (actual.dtype, actual.device)
                    if key not in buffers:
                        buffers[key] = torch.empty(step, dtype=actual.dtype, device=actual.device)
                    buffer = buffers[key]
                    for offset in range(0, expected.numel(), step):
                        count = min(step, expected.numel() - offset)
                        part = buffer[:count]
                        part.copy_(expected[offset:offset + count], non_blocking=True)
                        # These scalar tests stay on the GPU. Read one combined
                        # result after all exact element comparisons finish.
                        flags.append(torch.eq(actual[offset:offset + count], part).all())
                    continue
                for offset in range(0, expected.numel(), step):
                    if not torch.equal(actual[offset:offset + step].cpu(), expected[offset:offset + step]):
                        self.retire = True
                        return False
            if flags and not bool(torch.stack(flags).all().item()):
                self.retire = True
                return False
            return True
        except Exception:
            self.retire = True
            raise
        finally:
            self.job_stats["integrity_check_s"] += time.perf_counter() - start

    def clear(self):
        self.close_fixtures()
        had_model = self.model is not None
        self.model = self.baseline = self.key = None
        self.shadow = []
        if had_model:
            import torch
            gc.collect()
            torch.cuda.empty_cache()
