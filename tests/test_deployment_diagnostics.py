import unittest
from unittest.mock import patch
import test_layer_evaluation as f
from sako.model import measure_model
from sako.deployment_diagnostics import trace_summary, checked_compiled_sequence


class DeploymentTraceTests(unittest.TestCase):
    def test_overlap_is_not_double_counted_as_busy_time(self):
        events = [{"ph": "X", "cat": "kernel", "name": "gemm", "ts": start, "dur": duration}
                  for start, duration in [(1, 3), (2, 3), (7, 2)]]
        result = trace_summary({"traceEvents": events})
        self.assertEqual(result["kernel_count"], 3)
        self.assertEqual(result["kernel_duration_sum_us"], 8)
        self.assertEqual(result["gpu_span_us"], 8)
        self.assertEqual(result["gpu_busy_union_us"], 6)
        self.assertEqual(result["gpu_gap_us"], 2)

    def test_invalid_and_cpu_events_do_not_create_gpu_work(self):
        events = [{"ph": "X", "cat": "cpu_op", "name": "sako::kernel", "ts": 0, "dur": 20},
                  {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": 1, "dur": 3},
                  {"ph": "X", "cat": "kernel", "name": "bad", "ts": 2, "dur": float('nan')}]
        result = trace_summary({"traceEvents": events})
        self.assertEqual(result["kernel_count"], 0)
        self.assertEqual(result["gpu_gap_us"], 0)
        self.assertEqual(result["opaque_cpu_ops"][0]["duration_us"], 20)
        self.assertEqual(result["cuda_runtime"][0]["calls"], 1)

    def test_empty_trace_cannot_show_a_gain(self):
        result = trace_summary({})
        self.assertEqual(result["kernel_count"], 0)
        self.assertNotIn("accepted", result)

    def test_memory_work_is_not_counted_as_idle_and_driver_calls_are_visible(self):
        events = [{'ph': 'X', 'cat': cat, 'name': name, 'ts': start, 'dur': duration}
                  for cat, name, start, duration in [
                      ('kernel', 'gemm', 0, 2), ('gpu_memcpy', 'copy', 2, 3),
                      ('gpu_memset', 'zero', 6, 2), ('cuda_driver', 'cuLaunchKernel', 0, 1),
                      ('cpu_op', 'sako_diagnostic::linear_reference', 0, 2)]]
        result = trace_summary({'traceEvents': events})
        self.assertEqual(result['kernel_count'], 1)
        self.assertEqual(result['gpu_gap_us'], 1)
        self.assertEqual(result['gpu_busy_union_us'], 7)
        self.assertEqual(result['cuda_driver'][0]['calls'], 1)
        self.assertEqual(result['opaque_cpu_ops'][0]['name'], 'sako_diagnostic::linear_reference')

    def test_compiled_sequence_checks_all_outputs_before_timing(self):
        expected = [f.Tensor(1.), f.Tensor(2.)]
        cases = [(None, None, value) for value in expected]
        with patch('sako.gpu.check_output', side_effect=lambda a, b, c: {'passed': a.value == b.value}) as check:
            result = checked_compiled_sequence(lambda: tuple(expected), cases, {}, 'candidate')
            self.assertEqual(len(result), 2)
            self.assertEqual(check.call_count, 2)
            with self.assertRaisesRegex(ValueError, 'eager accuracy'):
                checked_compiled_sequence(lambda: (f.Tensor(1.), f.Tensor(99.)), cases, {}, 'candidate')
        with self.assertRaisesRegex(ValueError, 'number of outputs'):
            checked_compiled_sequence(lambda: (), cases, {}, 'reference')


class DeploymentControlTests(unittest.TestCase):
    setUp = f.LayerEvaluationTests.setUp
    environment = f.LayerEvaluationTests.environment
    manifest = f.LayerEvaluationTests.manifest

    def test_original_math_control_never_grants_candidate_acceptance(self):
        model = f.Model()
        manifest, catalog = self.manifest(2)
        self.torch.compile = lambda fn, **kw: fn
        sentinel = object()
        with self.environment(), patch('sako.model.load_model', return_value=model), \
             patch('sako.model.clone_structure', return_value=f.Model()), \
             patch('sako.model.apply_manifest') as install, \
             patch('sako.deployment_diagnostics.linear_reference_control', return_value=sentinel), \
             patch('sako.layer_checks.measure_layer_case', return_value={'passed': True}), \
             patch('sako.model.paired_timings', return_value={'baseline': {'samples_ms': [2.] * 5},
                                                            'candidate': {'samples_ms': [1.] * 5}}):
            result = measure_model(self.config, manifest, catalog, 'unused', reference_control=True)
        self.assertEqual(install.call_args.kwargs['operations'], {'candidate': sentinel})
        self.assertTrue(result['reference_control'])
        self.assertFalse(result['decision']['accepted'])
        self.assertTrue(any('cannot grant acceptance' in reason for reason in result['decision']['reasons']))


if __name__ == '__main__':
    unittest.main()
