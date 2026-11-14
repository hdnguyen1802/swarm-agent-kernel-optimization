"""A writer can test its own implementation while the task contract stays fixed."""
from triton_fixtures import RMS_SOURCE

import json
import unittest

from sako.core import read_json, write_json
from sako.swarm_policy import validate_plan
import test_swarm_loop as fixtures


class WriterAutonomyTests(unittest.TestCase):
    setUp = fixtures.SwarmLoopTests.setUp
    loop = fixtures.SwarmLoopTests.loop

    def test_writer_changes_algorithm_and_owns_its_observable_checks(self):
        path = self.directory / 'profile/tasks.json'
        catalog = read_json(path)
        catalog['tasks'][0]['cases'] = [{}, {}]
        write_json(path, catalog)
        loop = self.loop()
        loop.start_task()
        plan = loop.coordinator(1)
        idea = plan['hypotheses'][0]
        idea['verification'] = {'fixture_index': 0, 'implementation': [
            {'field': 'launch.0.grid.z', 'op': 'ge', 'value': 16}]}
        assignment = loop.assignment(idea, 0, plan)
        original_hypothesis = loop.hypotheses.get(assignment['hypothesis_id'])['hypothesis']
        seen = []
        original_call = self.client.call
        def call(role, prompt, system):
            if not role.startswith('writer_'):
                return original_call(role, prompt, system)
            seen.append(json.loads(prompt))
            return {'candidate': {'name': 'writer_choice', 'task_id': loop.active_task,
                'strategy': 'Use one reduction without a workspace.', "backend": "triton",
                'references': [], 'source': RMS_SOURCE + '# writer choice'},
                'design': {'claim': 'The reduction is small.', 'algorithm': 'One block per row',
                    'change': 'Use one reduction launch.', 'expected_metric': 'Lower latency.',
                    'reason': 'A split would add a workspace and another launch.',
                    'verification': {'fixture_index': 1, 'implementation': [
                        {'field': 'launch.0.grid.z', 'op': 'eq', 'value': 1}]}}}
        self.client.call = call
        original_run = self.worker.run
        def run(action, run_dir, **fields):
            result = original_run(action, run_dir, **fields)
            if action == 'evaluate':
                result['cases'].append(json.loads(json.dumps(result['cases'][0])))
                result['cases'][1]['executions'] = {
                    'candidate': {'status': 'observed', 'launches': [{'name': 'one', 'grid': [8, 1, 1], 'block': [32, 1, 1]}]},
                    'reference': {'status': 'observed', 'launches': [{'name': 'split', 'grid': [8, 1, 16], 'block': [32, 1, 1]}]}}
            return result
        self.worker.run = run
        candidate = loop.write_swarm(assignment)
        self.assertNotIn('design', candidate)
        self.assertEqual(assignment['writer_design']['algorithm'], 'One block per row')
        self.assertEqual(seen[0]['design_authority']['owner'], 'writer')
        loop.process_candidate(assignment)
        self.assertEqual(self.worker.calls[0][1]['observe_fixture'], 1)
        self.assertEqual(assignment['observations']['status'], 'observed')
        self.assertEqual(assignment['observations']['design_owner'], 'writer')
        self.assertEqual(assignment['observations']['requested'], 'Use one reduction launch.')
        self.assertEqual(loop.hypotheses.get(assignment['hypothesis_id'])['hypothesis'], original_hypothesis)
        self.assertTrue(loop.row(assignment['candidate_id'])['result']['cases'][0]['correct'])
        parent, = loop.archive.refresh(loop.by_id[loop.active_task], loop.hardware)
        self.assertEqual(parent['hypothesis']['algorithm'], 'One block per row')
        self.assertEqual(parent['hypothesis']['change'], 'Use one reduction launch.')
        self.assertEqual(parent['hypothesis']['design_owner'], 'writer')

    def test_same_goal_can_be_examined_from_different_specialties(self):
        base = {'objective': 'Reduce decode memory traffic.', 'parent_id': None,
                'claim': 'Weights dominate.', 'change': 'Reduce decode memory traffic.',
                'expected_metric': 'Latency.', 'evidence': ['contract:task']}
        plan = {'action': 'continue', 'reason': 'Compare independent approaches.', 'evidence': ['contract:task'],
                'hypotheses': [{**base, 'specialty': 'memory'}, {**base, 'specialty': 'scheduling'}]}
        result = validate_plan(plan, task_id='task', known_evidence={'contract:task'}, parent_ids=[])
        self.assertEqual(len(result['hypotheses']), 2)
        self.assertNotEqual(result['hypotheses'][0]['id'], result['hypotheses'][1]['id'])


if __name__ == '__main__':
    unittest.main()
