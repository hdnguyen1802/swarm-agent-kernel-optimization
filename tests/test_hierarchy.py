"""Check bottleneck scope, resume, integration, and budget shutdown."""
import copy
import json
import threading
import unittest
from unittest.mock import patch

from sako.core import load_config, write_json
from sako.hierarchy import select_hierarchy, validate_queue, reorder_remaining, containing_regions
from sako.preparation import profile_settings, validate_catalog
from sako.store import BudgetExhausted, Store
import test_swarm_loop as fixtures


def task(key, kind, paths, cost, reason=None):
    value = {"id": key, "kind": kind, "paths": paths, "priority_ms": cost,
             "cases": [], "contract": {"validated": True}}
    if reason:
        value["ranking_excluded_reason"] = reason
    return value


def catalog():
    tasks = [task('mlp', 'mlp', ['block.mlp'], 53.),
             task('gate_up', 'linear', ['block.mlp.gate', 'block.mlp.up'], 34.),
             task('down', 'linear', ['block.mlp.down'], 18.),
             task('act', 'silu', ['block.mlp.act'], 0., 'cross_contract_fusion')]
    tasks += [task('head'+str(i), 'linear', ['head'+str(i)], 4.-i/10) for i in range(5)]
    selected = select_hierarchy(tasks)
    return {"tasks": tasks, "selected_task_ids": selected["selected_task_ids"], "hierarchy": selected["hierarchy"]}


class HierarchyTests(unittest.TestCase):
    def test_highest_region_is_divided_without_a_top_four_limit(self):
        c = catalog()
        self.assertEqual(c['hierarchy']['initial_region_id'], 'mlp')
        self.assertEqual(c['selected_task_ids'][:2], ['gate_up', 'down'])
        self.assertEqual(len(c['selected_task_ids']), 7)
        self.assertNotIn('mlp', c['selected_task_ids'])
        self.assertNotIn('act', c['selected_task_ids'])
        self.assertEqual(c['hierarchy']['regions'][0]['unassigned_gpu_ms'], 1.)

    def test_future_roots_can_change_order_without_repeating_completed_work(self):
        c = catalog();p = {'queue':c['selected_task_ids'], 'index':2,
                           'completed':[{'task_id':'gate_up'}, {'task_id':'down'}]}
        new = reorder_remaining(c, p, {'head4':{'priority_ms':10.}})
        self.assertEqual(new['queue'][:3], ['gate_up', 'down', 'head4'])
        self.assertEqual(new['index'], 2)
        self.assertEqual(p['queue'][2], 'head0')
        validate_queue(c,new)
        new['queue'][0],new['queue'][1]=new['queue'][1],new['queue'][0]
        with self.assertRaisesRegex(ValueError,'completed work'):validate_queue(c,new)

    def test_unknown_duplicate_and_interleaved_subsections_are_rejected(self):
        c=catalog();q=c['selected_task_ids']
        for queue in [q[:-1]+['bad'],q[:-1]+[q[0]],[q[0],q[2],q[1]]+q[3:]]:
            with self.assertRaises(ValueError):validate_queue(c,{'queue':queue,'index':0,'completed':[]})

    def test_parent_boundaries_only_cover_replaced_subsections(self):
        c=catalog()
        self.assertEqual(len(containing_regions(c,{'entries':[{'paths':['block.mlp.down']}]})),1)
        self.assertEqual(containing_regions(c,{'entries':[{'paths':['head0']}]}),[])

    def test_new_default_cap_and_profile_scope_are_explicit(self):
        cfg=load_config('configs/project.json')
        self.assertEqual(cfg['agent']['max_total_tokens'],1000000)
        self.assertEqual(cfg['agent']['target_selection'],'bottleneck_tree')
        old=copy.deepcopy(cfg);old['agent']['target_selection']='flat'
        self.assertNotEqual(profile_settings(old),profile_settings(cfg))
        with self.assertRaisesRegex(ValueError,'target selection changed'):
            validate_catalog(cfg,{'config':old,'hardware':{'fingerprint':'a'}},{'fingerprint':'a'})


class BudgetDrainTests(unittest.TestCase):
    setUp=fixtures.SwarmLoopTests.setUp
    loop=fixtures.SwarmLoopTests.loop

    def test_budget_failure_drains_generated_outputs_without_another_api_request(self):
        self.config['agent'].update(writers=3,max_task_experiments=None)
        self.worker.operation_accepted=True
        self.worker.model_correct=True
        self.client.width=3
        release=threading.Event();self.addCleanup(release.set)
        original_call=self.client.call
        def call(role,prompt,system):
            if role=='coordinator' and self.client.plan_number:
                release.set()
                raise BudgetExhausted('The next request does not fit.')
            reply = original_call(role,prompt,system)
            if role == 'coordinator':
                reply['replan_after'] = 'next_result'
            return reply
        self.client.call=call
        self.client.writer_callback=lambda data: release.wait(5) if data['generation'] else None
        result=self.loop().run()
        self.assertEqual(result['status'],'budget_exhausted')
        self.assertEqual(sum(k=='evaluate' for k,_ in self.worker.calls),3)
        self.assertTrue(result['review_complete'])
        self.assertTrue(any(e['kind']=='evaluation_drain' for e in Store(self.directory/'run/state.sqlite').get('swarm_events')))


class RegionTransitionTests(unittest.TestCase):
    setUp=fixtures.SwarmLoopTests.setUp
    loop=fixtures.SwarmLoopTests.loop

    def test_kept_children_are_integrated_and_only_accepted_state_is_profiled(self):
        from sako.swarm import SwarmLoop
        c=catalog();c['config']=self.config
        self.config['agent']['target_selection']='bottleneck_tree'
        self.config['kernel_search']={'enabled':True,'min_speedup':1.01}
        write_json(self.directory/'profile/tasks.json',c)
        loop=self.loop()
        loop.progress.update(index=2,completed=[{'task_id':'gate_up'},{'task_id':'down'}])
        manifest={'entries':[{'task_id':'down','candidate_id':'x','paths':['block.mlp.down']}]}
        loop._measure_manifest=lambda *args:{'manifest':manifest,'result':{'decision':{'accepted':True,'speedup':1.04}}}
        original=self.worker.run
        def run(action,*args,**kwargs):
            if action=='reprofile':return {'task_priorities':{'head4':{'priority_ms':20.}},'model_state':'accepted_replacements'}
            return original(action,*args,**kwargs)
        self.worker.run=run
        with patch('sako.kernel_selection.region_manifests',return_value=[(['gate_up','down'],manifest)]):
            loop.finish_region(c['hierarchy']['regions'][0])
        self.assertEqual(loop.progress['queue'][2],'head4')
        self.assertEqual(loop.store.get('best_model_measurement')['manifest'],manifest)
        self.assertIn('mlp',loop.store.get('hierarchy_completed_regions'))

    def test_failed_integration_retains_original_model_and_does_not_reprofile_it(self):
        c=catalog();c['config']=self.config
        self.config['agent']['target_selection']='bottleneck_tree'
        write_json(self.directory/'profile/tasks.json',c)
        loop=self.loop();loop.progress.update(index=2,completed=[{'task_id':'gate_up'},{'task_id':'down'}])
        manifest={'entries':[{'task_id':'down','candidate_id':'x','paths':['block.mlp.down']}]}
        loop._measure_manifest=lambda *args:{'manifest':manifest,'result':{'decision':{'accepted':False,'speedup':.99}}}
        with patch('sako.kernel_selection.region_manifests',return_value=[(['gate_up','down'],manifest)]):
            loop.finish_region(c['hierarchy']['regions'][0])
        self.assertIsNone(loop.store.get('best_model_measurement'))
        self.assertFalse(any(a=='reprofile' for a,_ in self.worker.calls))
        self.assertEqual(loop.store.get('hierarchy_completed_regions')['mlp']['adopted_manifest'],{'entries':[]})


if __name__=='__main__':unittest.main()
