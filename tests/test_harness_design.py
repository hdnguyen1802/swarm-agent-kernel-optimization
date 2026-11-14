"""Regressions from the cached Mistral run and its measured planning costs."""
from triton_fixtures import RMS_SOURCE

import copy
import json
import unittest

from sako.coordinator_context import fit_coordinator_context, planning_task
from sako.core import read_json
from sako.integration import integrate
from sako.integration_repairs import affected_candidates, numerical_failures, record_failure
from sako.loop import SearchLoop
from sako.model import require_incumbent_gain
from sako.model_acceptance import incumbent_decision, model_trial_key
from sako.trial_evidence import model_saving
import test_clean_model_trials as clean_fixtures
import test_integration_repairs as repair_fixtures
import test_kernel_first as kernel_fixtures
import test_swarm_loop as fixtures


def failed_generation():
    return {"reference_state_intact": True, "cases": [{"correct": False, "workload": {},
        "checks": [{"scope": "replaced_layers", "passed": True, "layers": []}] * 3 + [{
            "scope": "cached_generation", "passed": False, "steps_checked": 128,
            "greedy_tokens_match": True, "cache_extent_valid": True,
            "logits": {"passed": False, "max_abs_error": .2, "atol": .03, "rtol": .02},
            "cache_layers": [{"layer": 0, "passed": True}]}]}],
        "decision": {"accepted": False, "speedup": None, "reasons": ["Score check failed."]}}


class ModelAssemblyTests(unittest.TestCase):
    setUp = fixtures.SwarmLoopTests.setUp
    loop = fixtures.SwarmLoopTests.loop
    prepare = clean_fixtures.CleanModelTrialsTests.prepare

    def test_small_addition_gets_a_combined_trial_without_passing_alone(self):
        loop, regions, _ = self.prepare({('attention',): (True, 1.10),
                                        ('head',): (False, 1.001),
                                        ('attention', 'head'): (True, 1.105)})
        loop.finish_region(regions[1])
        loop.finish_region(regions[2])
        self.assertEqual(self.model_calls, [('attention',), ('attention', 'head')])
        best = loop.store.get('best_model_measurement')
        self.assertEqual({e['task_id'] for e in best['manifest']['entries']}, {'attention', 'head'})
        self.assertEqual({e['task_id'] for e in best['incumbent']['entries']}, {'attention'})
        integrate(self.config, self.directory / 'run', self.directory / 'profile', self.worker)
        self.assertEqual(read_json(self.directory / 'run/result.json')['manifest'], best['manifest'])

    def test_revisit_prefers_gpu_cost_over_a_small_targets_large_ratio(self):
        loop, _, _ = self.prepare({})
        loop.by_id['gate']['priority_ms'] = 1000.
        loop.by_id['head']['priority_ms'] = 50.
        for row in loop.store.candidates('head'):
            loop.store.record(row['id'], {'correct': True, 'decision': {'accepted': True, 'speedup': 5.}})
        loop.config['agent']['revisit_with_budget'] = True
        self.assertTrue(loop.revisit_with_budget())
        self.assertEqual(loop.active_task, 'gate')

    def test_incumbent_pair_prevents_regression_despite_global_gain(self):
        cfg = self.config
        cases = [{"workload": w, "correct": True, "baseline": {"median_ms": 10., "p95_ms": 10.,
                   "samples_ms": [10.] * cfg['trials']}, "candidate": {"median_ms": 10.1, "p95_ms": 10.1,
                   "samples_ms": [10.1] * cfg['trials']}} for w in cfg['workloads']]
        pair = incumbent_decision(cases, cfg)
        global_gain = {"accepted": True, "speedup": 1.1, "reasons": []}
        self.assertFalse(require_incumbent_gain(global_gain, {"decision": pair})['accepted'])
        for row in cases:
            row['candidate'] = {"median_ms": 9.99, "p95_ms": 9.99, "samples_ms": [9.99] * cfg['trials']}
        self.assertTrue(incumbent_decision(cases, cfg)['accepted'])
        self.assertNotEqual(model_trial_key('run', {}), model_trial_key('run', {}, {'entries': []}))


class GenerationRepairTests(unittest.TestCase):
    setUp = fixtures.SwarmLoopTests.setUp
    loop = fixtures.SwarmLoopTests.loop
    prepare = repair_fixtures.IntegrationRepairTests.prepare

    def test_generation_score_failure_reopens_the_writer_hypothesis(self):
        loop, assignment, manifest = self.prepare()
        result = failed_generation()
        measured = {"manifest": manifest, "result": result}
        self.assertEqual(affected_candidates(measured), {assignment['candidate_id']})
        ticket, = record_failure(loop.store, measured)
        self.assertEqual(ticket['status'], 'correction_pending')
        self.assertEqual(numerical_failures(result)[0]['scope'], 'model_output')
        self.assertIn(assignment['candidate_id'], loop.store.get('blocked_model_candidates'))
        brief = SearchLoop.model_feedback(measured)
        check = next(x for x in brief['cases'][0]['checks'] if x.get('scope') == 'cached_generation')
        self.assertFalse(check['logits']['passed'])
        self.assertTrue(check['greedy_tokens_match'])


class PromotionTests(unittest.TestCase):
    setUp = kernel_fixtures.KernelFirstTests.setUp
    loop = fixtures.SwarmLoopTests.loop

    def test_a_new_keep_gets_its_generation_check_before_region_completion(self):
        self.config['model_trial_base'] = 'accepted'
        loop = self.loop()
        original = self.worker.run
        def run(action, run_dir, **fields):
            if action == 'model':
                self.worker.calls.append((action, fields))
                return failed_generation()
            return original(action, run_dir, **fields)
        self.worker.run = run
        loop.start_task()
        plan = loop.coordinator(1)
        assignment = loop.assignment(plan['hypotheses'][0], 0, plan)
        assignment['candidate'] = {'name': 'promotion', 'task_id': loop.active_task,
            'strategy': 'Measured operation.', "backend": "triton", 'references': [],
            'source': RMS_SOURCE + '# candidate 0'}
        loop.process_candidate(assignment)
        self.assertEqual([action for action, _ in self.worker.calls], ['evaluate', 'model'])
        self.assertEqual(loop.hypotheses.get(assignment['hypothesis_id'])['status'], 'correction_pending')
        self.assertIn(assignment['candidate_id'], loop.store.get('blocked_model_candidates'))
        self.assertFalse(loop.progress['completed'])


class PlanningCostTests(unittest.TestCase):
    def test_repeated_seeds_keep_exact_fixture_indices_in_two_groups(self):
        cases = [{"case_index": i, "shapes": [[1, 29 if i % 3 == 0 else 1, 4096]],
                  "strides": [[1]], "dtypes": ['torch.float16'], "workload": {"mode": 'cached_generation'},
                  "seed": i // 3} for i in range(6)]
        task = {"cases": cases, "contract": {"cases": cases, "functional_source": 'original operation'}}
        before = copy.deepcopy(task)
        plan = planning_task(task)
        self.assertEqual(task, before)
        self.assertEqual([g['case_indices'] for g in plan['cases']], [[0, 3], [1, 2, 4, 5]])
        self.assertEqual(plan['contract']['functional_source'], 'original operation')

    def test_unused_artifacts_are_removed_but_failure_citations_survive(self):
        ids = ['evidence/file-' + str(i) for i in range(100)]
        prompt = {"task": {}, "validation_policy": {}, "known_evidence": ids + ['contract:x'],
                  "evidence_types": {"artifact": ids, "contract": ['contract:x']},
                  "pending_repairs": {'candidate': {'evidence': {'path': ids[0]}}}}
        output = fit_coordinator_context(prompt, 10000, artifact={'path': ids[1]})
        self.assertEqual(set(output['known_evidence']), {ids[0], ids[1], 'contract:x'})
        self.assertLess(len(json.dumps(output)), len(json.dumps(prompt)))

    def test_decode_savings_are_weighted_once_across_repeated_cases(self):
        workload = {'mode': 'cached_generation', 'batch_size': 1, 'prompt': 'fixed', 'new_tokens': 128}
        task = {'workload_medians_ms': [100.], 'compiled_evidence': {
            'generation_phase_ms': {'prefill': [4.], 'decode': [96.]}},
            'cases': [{'workload': workload}] * 6}
        result = {'shape_decisions': {'shape': {}}, 'cases': []}
        for phase, speed in [('prefill', .5), ('decode', 2.), ('decode', 2.)] * 2:
            result['cases'].append({'workload': workload, 'correct': True, 'generation_step': {'phase': phase},
                'search_reference': {'kind': 'paired_compile', 'baseline': {'samples_ms': [2.]},
                                     'candidate': {'samples_ms': [2. / speed]}}})
        self.assertEqual(model_saving(task, result)['estimated_model_saving_ms'], 48.)


if __name__ == '__main__':
    unittest.main()
