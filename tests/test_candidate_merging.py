"""Two measured parents supply ideas, but never grant acceptance to their child."""

import copy
import json
import unittest

import test_swarm_loop as fixtures
from triton_fixtures import RMS_SOURCE
from sako.archive import summarize_parent_use
from sako.core import read_json, write_json
from sako.recombination import merge_pairs
from sako.shape_selection import groups
from sako.swarm_policy import validate_plan


class CandidateMergingTests(unittest.TestCase):
    setUp = fixtures.SwarmLoopTests.setUp
    loop = fixtures.SwarmLoopTests.loop

    def prepare(self):
        self.config['kernel_search'] = {'enabled': True, 'shape_scoped': True, 'min_speedup': 1.01}
        catalog = read_json(self.directory / 'profile/tasks.json')
        catalog['config'] = self.config
        task = catalog['tasks'][0]
        task.update(scalars=[1e-6], cases=[{'shapes': [[2, 8], [8]], 'strides': [[8, 1], [1]],
                                          'dtypes': ['torch.float16', 'torch.float16']}])
        write_json(self.directory / 'profile/tasks.json', catalog)
        loop = self.loop()
        loop.start_task()
        shape, = groups(task)
        ids = []
        for index, (specialty, warps) in enumerate((('memory', 2), ('scheduling', 8))):
            source = {'name': specialty, 'task_id': task['id'], 'backend': 'triton', 'references': [],
                      'strategy': 'Measured design.', 'source': RMS_SOURCE.replace('num_warps=4', f'num_warps={warps}')}
            identity, _ = loop.store.add(source, specialty, index, None)
            loop.store.record(identity, {'cases': [{'correct': True, 'inputs_unchanged': True,
                'baseline': {'samples_ms': [2.] * 15}, 'candidate': {'samples_ms': [1.5] * 15}}],
                'shape_decisions': {shape: {**groups(task)[shape], 'accepted': True, 'speedup': 4 / 3}},
                'decision': {'accepted': True, 'speedup': 4 / 3}})
            ids.append(identity)
        parents = loop.archive.refresh(task, loop.hardware)
        loop.next_id = len(ids)
        pairs = merge_pairs(task, parents, loop.store.candidates(task['id']))
        return loop, task, ids, shape, pairs

    def plan(self, loop, ids, shape, pairs, **changes):
        hypothesis = {'specialty': 'fusion', 'parent_id': ids[0], 'objective': 'Combine compatible measured ideas.',
            'claim': 'The two changes can work together.', 'change': 'Combine data access and work distribution.',
            'expected_metric': 'Less elapsed time.', 'evidence': ids,
            'merge': {'parent_ids': ids, 'shape_ids': [shape]}}
        value = {'action': 'continue', 'reason': 'Measure the combined design.', 'evidence': ids,
                 'hypotheses': [hypothesis], **changes}
        return validate_plan(value, task_id=loop.active_task, known_evidence=set(ids), parent_ids=ids,
                             merge_pairs=pairs, require_merge_trial=True)

    def writer(self, loop, mode='new'):
        seen = []
        def call(role, prompt, system):
            value = json.loads(prompt)
            seen.append(value)
            source = RMS_SOURCE if mode == 'new' else value['merge_parents'][0]['source']['source'] + '\n# comment only\n'
            return {'candidate': {'name': 'combined', 'task_id': loop.active_task, 'backend': 'triton',
                                 'source': source, 'strategy': 'Combine compatible ideas.', 'references': []},
                    'design': {'claim': 'The ideas can work together.', 'algorithm': 'One reduction per row',
                        'change': 'Combine the measured approaches.', 'expected_metric': 'Less elapsed time.',
                        'reason': 'The shared shape permits both changes.',
                        'verification': {'fixture_index': 0, 'implementation': []},
                        'parent_contributions': {p['id']: 'Reuse its measured approach.' for p in value['merge_parents']}}}
        self.client.call = call
        return seen

    def test_sources_measurements_lineage_and_exact_resume_survive(self):
        loop, task, ids, shape, pairs = self.prepare()
        plan = self.plan(loop, ids, shape, pairs)
        assignment = loop.assignment(plan['hypotheses'][0], 0, plan)
        seen = self.writer(loop)
        child = loop.write_swarm(assignment)
        self.assertEqual({p['id'] for p in seen[0]['merge_parents']}, set(ids))
        for parent in seen[0]['merge_parents']:
            self.assertEqual(parent['source'], loop.row(parent['id'])['source'])
            self.assertIn('cases', parent['measurements'])
        self.assertEqual(seen[0]['merge_shapes'], [shape])
        resumed = self.loop()
        self.assertEqual(resumed.write_swarm(resumed.work[str(assignment['generation'])]), child)
        self.assertEqual(len(seen), 1)
        loop.process_candidate(assignment)
        with loop.store.connect() as db:
            parents = {r[0] for r in db.execute('SELECT parent FROM candidate_parents WHERE candidate=?',
                                               (assignment['candidate_id'],))}
        self.assertEqual(parents, set(ids))
        self.assertEqual(summarize_parent_use(loop.work)['archive_merges'], 1)
        self.assertIsNone(loop.store.get('best_model_measurement'))
        self.assertTrue(any(action == 'evaluate' for action, _ in self.worker.calls))

    def test_pair_requires_the_same_measured_signature_and_correct_parents(self):
        loop, task, ids, shape, pairs = self.prepare()
        self.assertEqual(len(pairs), 1)
        changed = copy.deepcopy(loop.row(ids[1])['result'])
        changed['shape_decisions'] = {'different_stride': changed['shape_decisions'][shape]}
        loop.store.record(ids[1], changed)
        self.assertEqual(merge_pairs(task, loop.archive.refresh(task, loop.hardware), loop.store.candidates(task['id'])), [])
        loop.store.set('blocked_model_candidates', {ids[0]: {'reason': 'Generation accuracy failed.'}})
        self.assertNotIn(ids[0], {p['id'] for p in loop.archive.refresh(task, loop.hardware)})

    def test_both_parents_and_a_shared_shape_must_be_validated_and_cited(self):
        loop, _, ids, shape, pairs = self.prepare()
        for merge in ({'parent_ids': [ids[0], ids[0]], 'shape_ids': [shape]},
                      {'parent_ids': ids, 'shape_ids': ['other_shape']},
                      {'parent_ids': [ids[0], 'unknown'], 'shape_ids': [shape]}):
            with self.subTest(merge=merge), self.assertRaises(ValueError):
                self.plan(loop, ids, shape, pairs, hypotheses=[{'specialty': 'memory', 'parent_id': ids[0],
                    'claim': 'Combine.', 'change': 'Combine.', 'expected_metric': 'Latency.',
                    'evidence': ids, 'merge': merge}])
        incomplete = {'specialty': 'memory', 'parent_id': ids[0], 'claim': 'Combine.', 'change': 'Combine.',
                      'expected_metric': 'Latency.', 'evidence': [ids[0]], 'merge': {'parent_ids': ids, 'shape_ids': [shape]}}
        with self.assertRaisesRegex(ValueError, 'both merge parents'):
            self.plan(loop, ids, shape, pairs, hypotheses=[incomplete])

    def test_coordinator_must_try_an_untried_pair_or_explain_a_measured_conflict(self):
        loop, _, ids, shape, pairs = self.prepare()
        with self.assertRaisesRegex(ValueError, 'Propose a merge'):
            self.plan(loop, ids, shape, pairs, action='move_on', hypotheses=[],
                      archive_skip={'reason': 'Low value.', 'evidence': ids})
        plan = self.plan(loop, ids, shape, pairs, action='move_on', hypotheses=[],
                         archive_skip={'reason': 'Low value.', 'evidence': ids},
                         merge_skip={'reason': 'Measured designs require conflicting layouts.', 'evidence': ids})
        self.assertEqual(plan['action'], 'move_on')

    def test_comment_only_merge_is_rejected_before_gpu_evaluation(self):
        loop, _, ids, shape, pairs = self.prepare()
        plan = self.plan(loop, ids, shape, pairs)
        assignment = loop.assignment(plan['hypotheses'][0], 0, plan)
        self.writer(loop, 'unchanged')
        with self.assertRaisesRegex(ValueError, 'comments and formatting'):
            loop.write_swarm(assignment)
        self.assertFalse(self.worker.calls)

    def test_new_model_failure_blocks_a_saved_merge_parent_before_writer_dispatch(self):
        loop, _, ids, shape, pairs = self.prepare()
        plan = self.plan(loop, ids, shape, pairs)
        assignment = loop.assignment(plan['hypotheses'][0], 0, plan)
        loop.store.set('blocked_model_candidates', {ids[1]: {'reason': 'Model failed.'}})
        seen = self.writer(loop)
        with self.assertRaisesRegex(ValueError, 'repair'):
            loop.write_swarm(assignment)
        self.assertFalse(seen)

    def test_coordinator_receives_the_pair_and_must_select_a_valid_merge(self):
        loop, _, ids, shape, pairs = self.prepare()
        plan = self.plan(loop, ids, shape, pairs)
        seen = []
        def call(role, prompt, system):
            self.assertEqual(role, 'coordinator')
            seen.append(json.loads(prompt))
            return plan
        self.client.call = call
        selected = loop.coordinator(1)
        self.assertTrue(seen[0]['merge_policy']['trial_required'])
        self.assertEqual(set(seen[0]['merge_pairs'][0]['parent_ids']), set(ids))
        self.assertEqual(selected['hypotheses'][0]['merge']['shape_ids'], [shape])


if __name__ == '__main__':
    unittest.main()
