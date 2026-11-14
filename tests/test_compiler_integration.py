"""Keep build diagnostics outside a compiler-visible Triton launch path."""
import contextlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from sako.core import candidate_id, write_json
from sako.gpu import model_operation
from sako.model import apply_manifest
from sako.triton_backend import load
from triton_fixtures import RMS_SOURCE


def candidate(source=RMS_SOURCE):
    return {"name": "compile_path", "task_id": "rms", "strategy": "Test compiler visibility.",
            "backend": "triton", "references": [], "source": source}


class CompilerIntegrationTests(unittest.TestCase):
    def test_eager_checks_use_logged_runner_and_trace_uses_source_entry(self):
        entry = Mock(return_value="traced")
        run = Mock(return_value="logged")
        run._sako_compiler_entry = entry
        run.tuning_metadata = lambda: [{"kernel": "rms"}]
        compiling = {"value": False}
        torch = SimpleNamespace(compiler=SimpleNamespace(is_compiling=lambda: compiling["value"]))
        with patch.dict("sys.modules", {"torch": torch}), patch("sako.gpu.compile_candidate", return_value=run):
            operation = model_operation(candidate(), "rmsnorm", {"triton_integration": "direct"})
            tensors, scalars = [object()], [.001]
            self.assertEqual(operation(tensors, scalars), "logged")
            entry.assert_not_called()
            compiling["value"] = True
            self.assertEqual(operation(tensors, scalars), "traced")
        self.assertEqual(run.call_count, 1)
        entry.assert_called_once_with(tensors, scalars)
        self.assertEqual(operation._sako_tuning_metadata(), [{"kernel": "rms"}])

    def test_opaque_triton_uses_custom_operation_registration(self):
        item = candidate()
        with patch("sako.gpu.custom_op", return_value="opaque") as custom:
            self.assertEqual(model_operation(item, "rmsnorm", {"triton_integration": "opaque"}), "opaque")
            custom.assert_called_once_with(item, "rmsnorm")

    def test_missing_compiler_entry_fails_without_opaque_fallback(self):
        torch = SimpleNamespace(compiler=SimpleNamespace(is_compiling=lambda: True))
        with patch.dict("sys.modules", {"torch": torch}), \
             patch("sako.gpu.compile_candidate", return_value=lambda *args: None), \
             patch("sako.gpu.custom_op") as wrapped:
            with self.assertRaisesRegex(RuntimeError, "compiler entry"):
                model_operation(candidate(), "rmsnorm", {"triton_integration": "direct"})
        wrapped.assert_not_called()

    def test_direct_mode_does_not_bypass_source_validation(self):
        torch = SimpleNamespace(compiler=SimpleNamespace(is_compiling=lambda: False))
        modules = {"torch": torch, "torch.utils": SimpleNamespace(),
                   "torch.utils.cpp_extension": SimpleNamespace(load_inline=Mock())}
        with patch.dict("sys.modules", modules), patch("sako.gpu.discover_hardware") as discover:
            with self.assertRaisesRegex(ValueError, "imports"):
                model_operation(candidate(RMS_SOURCE + "\nimport os\n"), "rmsnorm", {"triton_integration": "direct"})
        discover.assert_not_called()

    def test_loader_exposes_the_exact_source_entry_and_keeps_first_launch_record(self):
        fake_triton = SimpleNamespace(jit=lambda f: f)
        modules = {"torch": SimpleNamespace(), "triton": fake_triton,
                   "triton.language": SimpleNamespace(constexpr=object())}
        source = '''import torch
import triton
import triton.language as tl
@triton.jit
def kernel(X):
    pass
def sako_run(tensors, scalars):
    return tensors[0]
'''
        # Test loading only. Runtime correctness still belongs to the evaluator.
        fake_triton.language = modules["triton.language"]
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", modules), \
             patch("sako.gpu.build_log", return_value=contextlib.nullcontext()):
            path = Path(directory)
            run = load(candidate(source), {"fingerprint": "test-source-entry"}, path)
            value = object()
            self.assertIs(run._sako_compiler_entry([value], []), value)
            self.assertFalse((path / "first-launch.json").exists())
            self.assertIs(run([value], []), value)
            self.assertTrue((path / "first-launch.json").exists())

    def test_apply_manifest_uses_explicit_config_and_preserves_fallback(self):
        class Module:
            def __call__(self, x): return self.forward(x)
        item = candidate()
        key = candidate_id(item)
        original = Mock(return_value="original")
        block = SimpleNamespace(fc=original)
        model = SimpleNamespace(get_submodule=lambda path: block)
        signature = {"shapes": [[1, 4]], "strides": [[4, 1]], "dtypes": ["torch.float16"], "scalars": []}
        task = {"id": "rms", "kind": "rmsnorm", "paths": ["block.fc"], "cases": [{k: v for k, v in signature.items() if k != "scalars"}]}
        manifest = {"entries": [{"task_id": "rms", "candidate_id": key, "paths": ["block.fc"], "signatures": [signature]}]}
        op = Mock(return_value="direct")
        def tensor(shape): return SimpleNamespace(shape=shape, stride=lambda: (4, 1), dtype="torch.float16")
        with tempfile.TemporaryDirectory() as directory:
            write_json(Path(directory) / (key + ".json"), item)
            with patch.dict("sys.modules", {"torch": SimpleNamespace(nn=SimpleNamespace(Module=Module))}), \
                 patch("sako.model.install_attention_dispatch"), \
                 patch("sako.model.arguments", side_effect=lambda kind, mod, x: ([x], [])), \
                 patch("sako.gpu.model_operation", return_value=op) as direct, patch("sako.model.custom_op") as opaque:
                operations = {}
                apply_manifest(model, manifest, {"config": {"triton_integration": "opaque"}, "tasks": [task]},
                               directory, config={"triton_integration": "direct"}, operations=operations)
                self.assertEqual(block.fc(tensor((1, 4))), "direct")
                self.assertEqual(block.fc(tensor((2, 4))), "original")
                self.assertIs(operations[key], op)
                direct.assert_called_once_with(item, "rmsnorm", {"triton_integration": "direct"})
                opaque.assert_not_called()


if __name__ == "__main__":
    unittest.main()
