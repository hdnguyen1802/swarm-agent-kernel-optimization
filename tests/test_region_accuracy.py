"""A local layer pass does not replace a composed region check."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import test_layer_evaluation as f
from sako.layer_checks import measure_layer_case


class Parent:
    def __init__(self, child):
        self.child,self.weight,self.hooks=child,f.Tensor(1.),[]
    def __call__(self,x):
        output=self.child(x)
        for hook in tuple(self.hooks):hook(self,(x,),output)
        return output
    def register_forward_hook(self,hook):
        self.hooks.append(hook)
        return SimpleNamespace(remove=lambda:self.hooks.remove(hook))


class Model:
    def __init__(self):self.leaf=f.Module(1.);self.parent=Parent(self.leaf)
    def get_submodule(self,path):return self.parent if path=='mlp' else self.leaf
    def __call__(self,x):return self.parent(x)


class RegionAccuracyTests(unittest.TestCase):
    setUp=f.LayerEvaluationTests.setUp
    environment=f.LayerEvaluationTests.environment

    def test_composed_region_failure_rejects_a_passing_subsection(self):
        model,alternative=Model(),Model()
        alternative.leaf.weight=f.Tensor(1.2)
        manifest={'entries':[{'task_id':'down','candidate_id':'candidate','paths':['mlp.down']}]}
        catalog={'tasks':[{'id':'down','kind':'linear','paths':['mlp.down']}],
                 'hierarchy':{'regions':[{'id':'parent','kind':'mlp','paths':['mlp'],
                                          'subsection_ids':['down']}]}}
        with self.environment():
            result=measure_layer_case(model,manifest,catalog,{'candidate':lambda ts,ss:f.Tensor(1.001)},
                                      self.config,self.config['workloads'][0],1729,alternative=alternative)
        self.assertTrue(result['layers'][0]['passed'])
        self.assertFalse(result['passed'])
        self.assertEqual(result['region_checks']['failed_paths'],['mlp'])
        self.assertTrue(result['reference_state_intact'])
        self.assertFalse(model.parent.hooks or model.leaf.hooks)

    def test_original_eager_region_is_the_accuracy_reference(self):
        model,alternative=Model(),Model()
        manifest={'entries':[{'task_id':'down','candidate_id':'candidate','paths':['mlp.down']}]}
        catalog={'tasks':[{'id':'down','kind':'linear','paths':['mlp.down']}],
                 'hierarchy':{'regions':[{'id':'parent','kind':'mlp','paths':['mlp'],
                                          'subsection_ids':['down']}]}}
        with self.environment():
            result=measure_layer_case(model,manifest,catalog,{'candidate':lambda ts,ss:f.Tensor(1.001)},
                                      self.config,self.config['workloads'][0],1729,alternative=alternative)
        self.assertTrue(result['passed'])
        self.assertEqual(result['region_checks']['reference'],'eager')
        self.assertEqual(result['layers'][-1]['scope'],'containing_region')

    def test_diagnostic_region_difference_does_not_reject_passing_layer(self):
        model, alternative = Model(), Model()
        alternative.leaf.weight = f.Tensor(1.2)
        manifest = {'entries': [{'task_id': 'down', 'candidate_id': 'candidate', 'paths': ['mlp.down']}]}
        catalog = {'tasks': [{'id': 'down', 'kind': 'linear', 'paths': ['mlp.down']}],
                   'hierarchy': {'regions': [{'id': 'parent', 'kind': 'mlp', 'paths': ['mlp'],
                                             'subsection_ids': ['down']}]}}
        self.config['containing_region_policy'] = 'diagnostic'
        with self.environment():
            result = measure_layer_case(model, manifest, catalog, {'candidate': lambda ts, ss: f.Tensor(1.001)},
                self.config, self.config['workloads'][0], 1729, alternative=alternative)
        self.assertTrue(result['passed'])
        self.assertFalse(result['region_checks']['passed'])
        self.assertEqual(result['region_checks']['acceptance_role'], 'diagnostic')
        self.assertEqual(result['layers'][-1]['acceptance_role'], 'diagnostic')

    def test_diagnostic_region_does_not_remove_direct_layer_gate(self):
        model, alternative = Model(), Model()
        manifest = {'entries': [{'task_id': 'down', 'candidate_id': 'candidate', 'paths': ['mlp.down']}]}
        catalog = {'tasks': [{'id': 'down', 'kind': 'linear', 'paths': ['mlp.down']}],
                   'hierarchy': {'regions': [{'id': 'parent', 'kind': 'mlp', 'paths': ['mlp'],
                                             'subsection_ids': ['down']}]}}
        self.config['containing_region_policy'] = 'diagnostic'
        with self.environment():
            result = measure_layer_case(model, manifest, catalog, {'candidate': lambda ts, ss: f.Tensor(1.2)},
                self.config, self.config['workloads'][0], 1729, alternative=alternative)
        self.assertFalse(result['passed'])
        self.assertIn('mlp.down', result['failed_paths'])

    def test_diagnostic_region_input_mutation_still_rejects(self):
        class MutatingParent(Parent):
            def __call__(self, x):
                x.value *= 2
                return super().__call__(x)
        model, alternative = Model(), Model()
        alternative.parent = MutatingParent(alternative.leaf)
        manifest = {'entries': [{'task_id': 'down', 'candidate_id': 'candidate', 'paths': ['mlp.down']}]}
        catalog = {'tasks': [{'id': 'down', 'kind': 'linear', 'paths': ['mlp.down']}],
                   'hierarchy': {'regions': [{'id': 'parent', 'kind': 'mlp', 'paths': ['mlp'],
                                             'subsection_ids': ['down']}]}}
        self.config['containing_region_policy'] = 'diagnostic'
        with self.environment():
            result = measure_layer_case(model, manifest, catalog, {'candidate': lambda ts, ss: f.Tensor(1.001)},
                self.config, self.config['workloads'][0], 1729, alternative=alternative)
        self.assertFalse(result['passed'])
        self.assertFalse(result['region_checks']['layers'][0]['inputs_unchanged'])


if __name__=='__main__':unittest.main()
