"""A kernel keep must not import an earlier model-speed failure."""
from triton_fixtures import RMS_SOURCE
import copy
import unittest

import test_swarm_loop as fixtures
from sako.core import read_json, write_json
from sako.hierarchy import select_hierarchy
from sako.integration import integrate
from sako.kernel_selection import kernel_manifest, model_trial_manifest, region_manifests


class CleanModelTrialsTests(unittest.TestCase):
    setUp = fixtures.SwarmLoopTests.setUp
    loop = fixtures.SwarmLoopTests.loop

    def prepare(self, decisions):
        self.config.update(model_trial_base='accepted')
        self.config['kernel_search'] = {'enabled': True, 'min_speedup': 1.01}
        self.config['agent']['target_selection'] = 'bottleneck_tree'
        tasks = [{'id': key, 'kind': 'linear', 'paths': [key + '.proj'],
                  'priority_ms': float(3-index), 'cases': [{}], 'contract': {'validated': True}}
                 for index, key in enumerate(['gate', 'attention', 'head'])]
        selected = select_hierarchy(tasks)
        catalog = {'config': self.config, 'tasks': tasks, 'hardware': {'fingerprint': 'test-gpu'},
                   'selected_task_ids': selected['selected_task_ids'], 'hierarchy': selected['hierarchy']}
        write_json(self.directory / 'profile/tasks.json', catalog)
        loop = self.loop()
        keeps = {}
        for task in tasks:
            candidate = {'name': task['id'], 'task_id': task['id'], 'strategy': 'Measured operation.',
                         "backend": "triton", 'references': [], 'source': RMS_SOURCE + "# " + task['id']}
            key, _ = loop.store.add(candidate, 'memory', 0, None)
            loop.store.record(key, {'correct': True, 'decision': {'accepted': True, 'speedup': 1.2}})
            write_json(self.directory / 'run/candidates' / (key + '.json'), candidate)
            keeps[task['id']] = {'candidate_id': key}
        loop.store.set('kernel_baselines', keeps)
        loop.store.set('status', 'completed_kernel_search')
        write_json(self.directory / 'run/result.json', {'status': 'completed_kernel_search'})
        # Region completion does not need a reprofile in these final-state tests.
        loop.progress.update(index=3, completed=[{'task_id': t['id']} for t in tasks])
        self.model_calls = []
        def run(action, run_dir, **fields):
            self.assertEqual(action, 'model')
            manifest = read_json(fixtures.ROOT / fields['manifest'])
            keys = tuple(sorted(e['task_id'] for e in manifest['entries']))
            self.model_calls.append(keys)
            accepted, speedup = decisions.get(keys, (False, .9))
            cases = [{'workload': w, 'correct': True, 'checks': [],
                      'baseline': {'samples_ms': [10.] * self.config['trials']},
                      'candidate': {'samples_ms': [10. / speedup] * self.config['trials']}}
                     for w in self.config['workloads']]
            return {'reference_state_intact': True, 'cases': cases,
                    'decision': {'accepted': accepted, 'speedup': speedup,
                                 'reasons': [] if accepted else ['The model speed gate failed.']}}
        self.worker.run = run
        return loop, catalog['hierarchy']['regions'], copy.deepcopy(keeps)

    def test_region_completion_does_not_carry_a_rejected_keep(self):
        loop, regions, kept = self.prepare({('attention',): (True, 1.1)})
        loop.finish_region(regions[0])
        self.assertIsNone(loop.store.get('best_model_measurement'))
        self.assertNotIn('gate', loop.store.get('kernel_region_choices', {}))
        loop.finish_region(regions[1])
        self.assertEqual(self.model_calls, [('gate',), ('attention',)])
        self.assertEqual({e['task_id'] for e in loop.store.get('best_model_measurement')['manifest']['entries']}, {'attention'})
        self.assertEqual(loop.store.get('kernel_baselines'), kept)
        self.assertEqual({e['task_id'] for e in kernel_manifest(loop.store, loop.catalog)['entries']}, set(kept))

    def test_regions_test_the_combination_with_the_accepted_base(self):
        loop, regions, kept = self.prepare({('attention',): (True, 1.1), ('head',): (True, 1.03)})
        loop.finish_region(regions[1])
        loop.finish_region(regions[2])
        self.assertEqual(self.model_calls, [('attention',), ('attention', 'head')])
        self.assertEqual({e['task_id'] for e in loop.store.get('best_model_measurement')['manifest']['entries']}, {'attention'})
        self.assertEqual(loop.store.get('kernel_baselines'), kept)

    def test_deferred_integration_tests_each_region_and_preserves_accepted_state(self):
        loop, _, kept = self.prepare({('attention',): (True, 1.1), ('head',): (True, 1.03)})
        integrate(self.config, self.directory / 'run', self.directory / 'profile', self.worker)
        self.assertEqual(self.model_calls, [('gate',), ('attention',), ('attention', 'head')])
        final = read_json(self.directory / 'run/result.json')
        self.assertEqual({e['task_id'] for e in final['manifest']['entries']}, {'attention'})
        self.assertEqual(final['integration_status'], 'accepted')
        self.assertEqual(loop.store.get('kernel_baselines'), kept)
        before = list(self.model_calls)
        integrate(self.config, self.directory / 'run', self.directory / 'profile', self.worker)
        self.assertEqual(self.model_calls, before + [('attention', 'gate')])

    def test_no_accepted_base_means_only_region_local_trials(self):
        loop, regions, _ = self.prepare({})
        choices = region_manifests(loop.store, loop.catalog, regions[1], config=self.config)
        self.assertEqual([{e['task_id'] for e in manifest['entries']} for _, manifest in choices], [{'attention'}])

    def test_invalidated_accepted_state_is_not_reused(self):
        loop, _, _ = self.prepare({})
        manifest = kernel_manifest(loop.store, loop.catalog)
        entry = manifest['entries'][0]
        loop.store.set('best_model_measurement', {'manifest': {'entries': [entry]},
            'result': {'correct': True, 'reference_state_intact': False, 'decision': {'accepted': True}}})
        local = {'entries': [manifest['entries'][1]]}
        self.assertEqual(model_trial_manifest(loop.store, loop.catalog, local), local)

    def test_deferred_integration_reuses_the_same_search_measurements(self):
        loop, regions, _ = self.prepare({('attention',): (True, 1.1)})
        for region in regions:
            loop.finish_region(region)
        before = list(self.model_calls)
        self.assertEqual(len(before), 3)
        integrate(self.config, self.directory / 'run', self.directory / 'profile', self.worker)
        self.assertEqual(self.model_calls, before + [('attention', 'gate')])


if __name__ == '__main__':
    unittest.main()
