"""Check saved timing identity and the declared five-then-ten sampling rules."""

import copy
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sako.core import load_config, read_json, write_json
from sako.measurement import (assess_pairs, baseline_identity, compare_saved_incumbent,
    incumbent_reference, paired_samples, policy, profile_stability, starting_baseline, warmup)
from sako.model_acceptance import score_model_cases
from sako.compiled_ranking import rank_compiled


class MeasurementPolicyTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(Path(__file__).resolve().parents[1] / 'configs/project.json')
        self.config['measurement'] = {'mode': 'adaptive'}
        self.config['min_speedup'] = 1.02
        self.config['revision'] = 'a' * 40
        self.config['model_speed_scope'] = 'all_workloads'
        self.workload = self.config['workloads'][0]

    def result(self, a=100., b=90.):
        return paired_samples(lambda: {'total_ms': a}, lambda: {'total_ms': b}, self.config)

    def case(self, result):
        return {'workload': self.workload, 'correct': True, **result}

    def test_clear_gain_uses_five_alternating_pairs(self):
        result = self.result()
        self.assertEqual(result['baseline']['samples_ms'], [100.] * 5)
        self.assertEqual(result['candidate']['samples_ms'], [90.] * 5)
        self.assertEqual(result['sampling']['pair_order'], [
            ['baseline', 'candidate'], ['candidate', 'baseline'], ['baseline', 'candidate'],
            ['candidate', 'baseline'], ['baseline', 'candidate']])
        self.assertEqual(result['sampling']['assessment']['state'], 'pass')
        self.assertTrue(score_model_cases([self.case(result)], self.config, [self.workload])['accepted'])

    def test_close_result_adds_five_pairs_and_keeps_all_samples(self):
        result = self.result(b=100. / 1.022)
        self.assertEqual(len(result['baseline']['samples_ms']), 10)
        self.assertEqual(len(result['sampling']['rounds']), 2)
        self.assertTrue(result['sampling']['rounds'][0]['near_threshold'])
        self.assertEqual(result['sampling']['assessment']['state'], 'pass')
        self.assertTrue(score_model_cases([self.case(result)], self.config, [self.workload])['accepted'])

    def test_noisy_result_stays_uncertain_after_ten_pairs(self):
        # Three warmups precede the timed samples. No outlier is removed.
        values = iter([100.] * 3 + [100., 103., 100., 99., 100.] + [100.] * 5)
        result = paired_samples(lambda: {'total_ms': next(values)}, lambda: {'total_ms': 90.}, self.config)
        self.assertEqual(result['baseline']['samples_ms'][1], 103.)
        self.assertEqual(len(result['baseline']['samples_ms']), 10)
        self.assertEqual(result['sampling']['assessment']['state'], 'uncertain')
        decision = score_model_cases([self.case(result)], self.config, [self.workload])
        self.assertFalse(decision['accepted'])
        self.assertEqual(decision['timing_status'], 'uncertain')

    def test_interval_crossing_the_limit_cannot_pass(self):
        status = assess_pairs([100.] * 10, [98., 98.2, 98.1, 98.05, 98.15] * 2, self.config)
        self.assertEqual(status['state'], 'uncertain')
        self.assertLess(status['interval'][0], 1.02)
        self.assertGreater(status['interval'][1], 1.02)

    def test_correctness_failure_blocks_a_fast_result(self):
        case = {**self.case(self.result()), 'correct': False}
        self.assertFalse(score_model_cases([case], self.config, [self.workload])['accepted'])

    def test_generation_accuracy_failure_never_starts_pairs(self):
        from sako.generation import compare_models
        runner = SimpleNamespace(record={'cache': 'StaticCache'})
        with patch('sako.generation.compiled_request', return_value=runner), \
             patch('sako.generation.check_generation', return_value={'passed': False}), \
             patch('sako.generation.paired_generation') as timed:
            check, result = compare_models(None, None, self.config, self.workload, catalog={})
        self.assertFalse(check['passed'])
        self.assertEqual(result['timing_status'], 'not_measured')
        timed.assert_not_called()

    def test_report_uses_model_pair_count_and_displays_uncertain(self):
        from sako.reporting import evaluation_details
        result = self.result()
        case = {**self.case(result), 'input': {}}
        model = {'cases': [case], 'decision': score_model_cases([case], self.config, [self.workload])}
        report = evaluation_details(model, self.config, scope='model')
        self.assertFalse(report['gate_disagreement'])
        self.assertEqual(report['cases'][0]['timing']['baseline']['sample_count'], 5)
        model['decision'] = {'accepted': False, 'timing_status': 'uncertain', 'reasons': ['Timing is uncertain.']}
        model['cases'][0]['sampling']['assessment']['state'] = 'uncertain'
        self.assertEqual(evaluation_details(model, self.config, scope='model')['status'], 'uncertain')

    def test_missing_policy_or_changed_raw_data_cannot_pass(self):
        result = self.result()
        missing = copy.deepcopy(result)
        missing.pop('sampling')
        self.assertFalse(score_model_cases([self.case(missing)], self.config, [self.workload])['accepted'])
        result['candidate']['samples_ms'][0] = 120.
        self.assertFalse(score_model_cases([self.case(result)], self.config, [self.workload])['accepted'])

    def test_warmup_uses_first_stable_window_and_logs_runs_separately(self):
        values = iter([1000., 100., 100.2, 100.1, 100.])
        result = warmup(lambda: {'total_ms': next(values)}, self.config)
        self.assertEqual(result['count'], 4)
        self.assertTrue(result['stable'])
        self.assertEqual(result['runs'][0]['total_ms'], 1000.)
        self.assertEqual(result['phase'], 'warmup')

    def test_exhausted_unstable_warmup_blocks_acceptance(self):
        index = [0]
        def sample():
            index[0] += 1
            return {'total_ms': 100. if index[0] % 2 else 110.}
        result = paired_samples(sample, lambda: {'total_ms': 90.}, self.config)
        self.assertEqual(result['sampling']['warmup']['baseline']['count'], 12)
        self.assertEqual(result['sampling']['assessment']['state'], 'uncertain')
        self.assertFalse(score_model_cases([self.case(result)], self.config, [self.workload])['accepted'])

    def fake_identity_torch(self):
        return SimpleNamespace(get_float32_matmul_precision=lambda: 'highest',
            inference_mode=nullcontext, is_inference_mode_enabled=lambda: True, is_autocast_enabled=lambda: False,
            are_deterministic_algorithms_enabled=lambda: False,
            backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(
                allow_tf32=False, allow_fp16_reduced_precision_reduction=True)),
                cudnn=SimpleNamespace(allow_tf32=True)))

    def test_starting_times_reuse_only_exact_settings_and_never_retime_on_hit(self):
        record = {'prompt_token_ids': [3, 9], 'cache': 'StaticCache', 'new_tokens': 128}
        model = SimpleNamespace(config=SimpleNamespace(to_dict=lambda: {
            'model_type': 'generic', 'id2label': {0: 'LABEL_0', 1: 'LABEL_1'}},
                                                      _attn_implementation='sdpa'))
        calls = []
        values = iter([1000., 100., 100.2, 100.1] + [100.] * 5)
        class Runner:
            def __init__(self):
                self.record = copy.deepcopy(record)
            def run(self, timed=False):
                calls.append(timed)
                return {'total_ms': next(values)}
        with TemporaryDirectory() as folder, patch.dict('sys.modules', {'torch': self.fake_identity_torch()}), \
             patch('sako.measurement.normal_request', side_effect=lambda *a: Runner()):
            hardware = {'fingerprint': 'gpu-and-software-a'}
            first = starting_baseline(model, self.config, hardware, folder)
            self.assertEqual(len(calls), 9)
            second = starting_baseline(model, self.config, hardware, folder)
            self.assertEqual(first, second)
            self.assertEqual(len(calls), 9)
            self.assertEqual(len(first['workloads'][0]['samples']), 5)
            from sako.measurement import cached_baseline_report
            catalog = {'config': self.config, 'hardware': hardware, 'starting_baseline': first,
                'compiled_profile': {'workload_times': [{'profiling_correctness': {'passed': True}}]}}
            report = cached_baseline_report(self.config, catalog, hardware)
            self.assertEqual(report['additional_timed_runs'], 0)
            self.assertEqual(report['cases'][0]['baseline']['samples_ms'], [100.] * 5)
            self.assertFalse(report['decision']['accepted'])
            self.assertEqual(len(calls), 9)
            changes = [('model_id', 'org/other'), ('revision', 'b' * 40), ('dtype', 'float32'),
                       ('compile_mode', 'reduce-overhead'), ('attention', 'sdpa')]
            for key, value in changes:
                if self.config[key] == value:
                    value = 'other-value'
                with self.subTest(setting=key), self.assertRaisesRegex(ValueError, 'different settings'):
                    starting_baseline(model, {**self.config, key: value}, hardware, folder)
            with self.assertRaisesRegex(ValueError, 'different settings'):
                starting_baseline(model, self.config, {'fingerprint': 'gpu-b'}, folder)
            record['cache'] = 'DynamicCache'
            with self.assertRaisesRegex(ValueError, 'different settings'):
                starting_baseline(model, self.config, hardware, folder)
            record['cache'] = 'StaticCache'
            record['prompt_token_ids'] = [3, 10]
            with self.assertRaisesRegex(ValueError, 'different settings'):
                starting_baseline(model, self.config, hardware, folder)
            record['prompt_token_ids'] = [3, 9]
            with patch('sako.measurement.implementation_hash', return_value='changed-source'), \
                 self.assertRaisesRegex(ValueError, 'different settings'):
                starting_baseline(model, self.config, hardware, folder)
            self.assertEqual(len(calls), 9)

    def test_eager_reference_reuses_only_matching_inputs_hardware_and_settings(self):
        from sako.measurement import starting_eager
        config = load_config(Path(__file__).resolve().parents[1] / 'configs/generation.json')
        config['revision'] = 'a' * 40
        record = {'prompt_token_ids': [3, 9], 'cache': 'StaticCache', 'new_tokens': 128}
        model = SimpleNamespace(config=SimpleNamespace(to_dict=lambda: {'model_type': 'generic'},
                                                       _attn_implementation='sdpa'))
        calls = []
        class Runner:
            def __init__(self, *args):
                self.record = copy.deepcopy(record)
            def run(self, timed=False):
                calls.append(timed)
                return {'total_ms': 130., 'mean_decode_ms': 1., 'first_token_ms': 3.}
        with TemporaryDirectory() as folder, patch.dict('sys.modules', {'torch': self.fake_identity_torch()}), \
             patch('sako.generation.GreedyRequest', Runner):
            compiled = {'workloads': [{}]}
            hardware = {'fingerprint': 'gpu-a'}
            first = starting_eager(model, config, hardware, folder, compiled)
            second = starting_eager(model, config, hardware, folder, compiled)
            self.assertEqual(first, second)
            self.assertEqual(len(calls), 6)
            self.assertEqual(first['workloads'][0]['identity']['execution'], 'pytorch_eager')
            self.assertEqual(first['workloads'][0]['warmup']['count'], 3)
            with self.assertRaisesRegex(ValueError, 'different settings'):
                starting_eager(model, config, {'fingerprint': 'gpu-b'}, folder, compiled)
            record['prompt_token_ids'] = [4, 9]
            with self.assertRaisesRegex(ValueError, 'different settings'):
                starting_eager(model, config, hardware, folder, compiled)

    def test_unstable_eager_reference_keeps_samples_and_cannot_be_reused(self):
        from sako.measurement import starting_eager
        config = load_config(Path(__file__).resolve().parents[1] / 'configs/generation.json')
        model = SimpleNamespace(config=SimpleNamespace(to_dict=lambda: {'model_type':'generic'},
                                                       _attn_implementation='sdpa'))
        values = iter([1.,1.,1.,1.,1.1,1.])
        calls = []
        class Runner:
            record = {'cache':'StaticCache'}
            def __init__(self, *args):
                pass
            def run(self, timed=False):
                calls.append(timed)
                return {'mean_decode_ms':next(values),'total_ms':130.,'first_token_ms':3.}
        with TemporaryDirectory() as folder, patch.dict('sys.modules', {'torch':self.fake_identity_torch()}), \
             patch('sako.generation.GreedyRequest', Runner):
            for _ in range(2):
                with self.assertRaisesRegex(ValueError, 'unstable'):
                    starting_eager(model, config, {'fingerprint':'gpu-a'}, folder, {'workloads':[{}]})
            self.assertEqual(len(calls), 6)
            saved = read_json(Path(folder)/'pytorch-baseline.json')['workloads'][0]
            self.assertEqual(saved['samples_ms'], [1.,1.1,1.])
            self.assertFalse(saved['stable'])

    def record(self, key, path, trial, elapsed, workload=0):
        return {'name': key, 'time_ms': elapsed, 'input_shapes': [], 'mapping_complete': True,
                'source_nodes': [{'name': key, 'path': path}], 'workload_index': workload,
                'trial': trial, 'artifact': 'graph.py', 'line': 1}

    def test_profile_stability_uses_disjoint_trial_groups(self):
        tasks = [{'id': key, 'kind': 'linear', 'paths': [key]} for key in ('a', 'b')]
        rows = [self.record(key, key, trial, cost) for trial in range(5)
                for key, cost in (('a', 10. + trial * .01), ('b', 5.))]
        status = profile_stability(tasks, rows, 0, 5, self.config)
        self.assertTrue(status['stable'])
        self.assertFalse(status['needs_more'])
        for row in rows:
            if row['trial'] >= 2:
                row['time_ms'] = 5. if row['name'] == 'a' else 10.
        status = profile_stability(tasks, rows, 0, 5, self.config)
        self.assertFalse(status['stable'])
        self.assertTrue(status['needs_more'])

    def test_different_profile_counts_do_not_pad_missing_trials_with_zero(self):
        tasks = [{'id': 'a', 'kind': 'linear', 'paths': ['a']}]
        rows = ([self.record('a', 'a', i, 2., 0) for i in range(5)] +
                [self.record('a', 'a', i, 3., 1) for i in range(10)])
        rank_compiled(tasks, rows, 2, {0: 5, 1: 10})
        self.assertEqual(tasks[0]['workload_medians_ms'], [2., 3.])

    def test_saved_accepted_gain_needs_no_extra_gpu_runs_and_requires_matching_context(self):
        manifest = {'entries': []}
        previous = {'manifest': manifest, 'result': {'config': self.config,
            'cases': [self.case(self.result(b=90.))]}}
        previous['result']['decision'] = score_model_cases(previous['result']['cases'], self.config, [self.workload])
        catalog = {'hardware': {'fingerprint': 'a'}, 'config': self.config, 'tasks': []}
        reference = incumbent_reference(previous, catalog)
        result = compare_saved_incumbent(manifest, reference, [self.case(self.result(b=89.))], self.config, catalog)
        self.assertTrue(result['decision']['accepted'])
        self.assertEqual(result['additional_timed_runs'], 0)
        rejected = compare_saved_incumbent(manifest, reference, [self.case(self.result(b=91.))], self.config, catalog)
        self.assertFalse(rejected['decision']['accepted'])
        with self.assertRaisesRegex(ValueError, 'different source, hardware'):
            compare_saved_incumbent(manifest, reference, [], self.config,
                                    {**catalog, 'hardware': {'fingerprint': 'b'}})

    def test_policy_counts_are_declared_and_cannot_be_silently_reduced(self):
        self.assertEqual(policy(self.config)['model_initial_pairs'], 5)
        with self.assertRaises(ValueError):
            policy({**self.config, 'measurement': {'mode': 'adaptive', 'baseline_trials': 3}})
        self.assertEqual(policy({}), {'mode': 'fixed'})

    def test_standard_configs_adopt_adaptive_policy_and_keep_operation_trials(self):
        root = Path(__file__).resolve().parents[1]
        for name in ('project.json', 'generation.json'):
            value = load_config(root / 'configs' / name)
            self.assertEqual(value['measurement']['mode'], 'adaptive')
            self.assertEqual(value['trials'], 15)
            self.assertEqual(value['agent']['max_total_tokens'], 1000000)
        invalid = {**self.config, 'ranking_mode': 'eager_regions'}
        with TemporaryDirectory() as folder:
            f = Path(folder) / 'invalid.json'
            write_json(f, invalid)
            with self.assertRaisesRegex(ValueError, 'compiled_profiler'):
                load_config(f)

    def test_terminal_baseline_command_reuses_reference_instead_of_model_comparison(self):
        from sako import cli
        from unittest.mock import Mock
        from contextlib import redirect_stdout
        import io
        root = Path(__file__).resolve().parents[1]
        (root / 'runs').mkdir(exist_ok=True)
        with TemporaryDirectory(dir=root / 'runs') as folder:
            base = Path(folder)
            write_json(base / 'config.json', self.config)
            worker = Mock()
            worker.run.return_value = {'starting_times_reused': True, 'additional_timed_runs': 0}
            argv = ['sako', 'baseline', '--config', str(base / 'config.json'), '--run', str(base / 'run')]
            with patch('sys.argv', argv), patch('sako.cli.Worker', return_value=worker), redirect_stdout(io.StringIO()):
                cli.main()
            self.assertEqual(worker.run.call_args.args[0], 'baseline')
            self.assertEqual(worker.run.call_count, 1)
            self.assertEqual(read_json(base / 'run/baseline.json')['additional_timed_runs'], 0)

    def test_saved_baseline_worker_action_never_loads_model_weights(self):
        from sako.worker import execute
        with TemporaryDirectory() as folder:
            base = Path(folder)
            write_json(base / 'config.json', self.config)
            hardware = {'fingerprint': 'hardware-a'}
            write_json(base / 'profile/tasks.json', {'config': self.config, 'hardware': hardware})
            with patch('sako.hardware.discover_hardware', return_value=hardware), \
                 patch('sako.measurement.cached_baseline_report', return_value={'additional_timed_runs': 0}) as cached, \
                 patch('sako.model.load_model', side_effect=AssertionError('Read-only baseline loaded weights.')):
                result = execute({'action': 'baseline', 'config': str(base / 'config.json'),
                                  'profile_dir': str(base / 'profile')})
            self.assertEqual(result['additional_timed_runs'], 0)
            cached.assert_called_once()


if __name__ == '__main__':
    unittest.main()
