"""Check compiled attribution, fusion ownership, and profile compatibility."""
import copy
from types import SimpleNamespace
import unittest

from sako.compiled_ranking import module_path, launch_plan, compiled_records, rank_compiled, METHOD
from sako.core import load_config
from sako.hardware import select_top_tasks
from sako.preparation import profile_settings, validate_catalog


def task(key, kind, paths):
    return {"id": key, "kind": kind, "paths": paths, "contract": {"validated": True}}


def record(name, paths, elapsed=2., workload=0, trial=0):
    return {"name": name, "time_ms": elapsed, "input_shapes": [[8]], "mapping_complete": True,
            "source_nodes": [{"name": str(i), "path": path} for i, path in enumerate(paths)],
            "workload_index": workload, "trial": trial, "artifact": "compiler/output_code.py", "line": 4}


class CompiledRankingTests(unittest.TestCase):
    def test_user_triton_kernel_keeps_its_owner_after_integration(self):
        source = '''_gemv_kernel_0 = async_compile.triton('_gemv_kernel', 'generated kernel')
def call(args):
    # Topologically Sorted Source Nodes: [user_kernel], Original ATen: []
    _gemv_kernel_0.run(x, weight, out, grid=grid(80))
    unknown_function.run(x)
    return out
'''
        nodes = {'user_kernel': {'name': 'user_kernel', 'target': 'triton_kernel_wrapper_mutation',
                                 'path': 'block.mlp.up_proj'}}
        plan = launch_plan(source, nodes, 'output_code.py')
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0]['event'], '_gemv_kernel_0')
        self.assertTrue(plan[0]['mapping_complete'])
        self.assertEqual(plan[0]['source_nodes'], [nodes['user_kernel']])

    def test_empty_user_comments_use_exact_graph_inputs_without_launch_order(self):
        source = """_gemv_kernel_0 = async_compile.triton('_gemv_kernel', 'generated kernel')
def call(args):
    x, weight_up, weight_gate = args
    args.clear()
    # Topologically Sorted Source Nodes: [], Original ATen: []
    def grid_wrapper(meta):
        return (224, 1, 1)
    _gemv_kernel_0.run(x, weight_gate, gate_out, grid=grid_wrapper)
    # Topologically Sorted Source Nodes: [], Original ATen: []
    _gemv_kernel_0.run(x, weight_up, up_out, grid=grid_wrapper)
"""
        nodes = {key: {'name': key, 'target': 'triton_kernel_wrapper_mutation',
                       'path': path, 'user_triton': {'kernel': '_gemv_kernel',
                       'inputs': [{'argument': 0, 'input': 0}, {'argument': 1, 'input': index}]}}
                 for key, path, index in [('up', 'block.mlp.up', 1), ('gate', 'block.mlp.gate', 2)]}
        plan = launch_plan(source, nodes, 'output_code.py')
        self.assertEqual([p['source_nodes'][0]['path'] for p in plan], ['block.mlp.gate', 'block.mlp.up'])
        self.assertTrue(all(p['mapping_complete'] for p in plan))
        self.assertTrue(all(p['implementation'] == 'user_triton' for p in plan))
        events = [SimpleNamespace(name='_gemv_kernel_0', device_type='CPU',
                  self_device_time_total=1000., input_shapes=[]) for _ in range(2)]
        records = compiled_records(events, plan, 0, 0)
        tasks = [task('projections', 'linear', ['block.mlp.gate', 'block.mlp.up'])]
        rank_compiled(tasks, records, 1, 1)
        self.assertEqual(tasks[0]['priority_ms'], 2.)
        self.assertNotIn('ranking_excluded_reason', tasks[0])

    def test_ambiguous_or_changed_user_inputs_stay_unmapped(self):
        source = """_gemv_kernel_0 = async_compile.triton('_gemv_kernel', 'generated kernel')
def call(args):
    x, weight = args
    # Topologically Sorted Source Nodes: [], Original ATen: []
    _gemv_kernel_0.run(x, weight, out)
"""
        binding = {'kernel': '_gemv_kernel', 'inputs': [{'argument': 1, 'input': 1}]}
        nodes = {key: {'name': key, 'path': path, 'user_triton': binding}
                 for key, path in [('up', 'block.mlp.up'), ('gate', 'block.mlp.gate')]}
        self.assertFalse(launch_plan(source, nodes, 'code.py')[0]['mapping_complete'])
        changed = source.replace('    # Topologically', '    weight = other_tensor\n    # Topologically')
        self.assertFalse(launch_plan(changed, {'up': nodes['up']}, 'code.py')[0]['mapping_complete'])
        self.assertFalse(launch_plan(source, {'up': {**nodes['up'], 'user_triton':
            {'kernel': 'different_kernel', 'inputs': binding['inputs']}}}, 'code.py')[0]['mapping_complete'])

    def test_module_path_handles_parameter_module_and_layer_paths(self):
        self.assertEqual(module_path("L['model'].model.layers[0].mlp.gate_proj"), "model.layers.0.mlp.gate_proj")
        self.assertEqual(module_path('L["self"]._modules["fc"]'), 'fc')

    def test_launch_plan_filters_upstream_origins_and_handles_nested_calls(self):
        source = '''def call(args):
    # Topologically Sorted Source Nodes: [silu, mul], Original ATen: [aten.silu, aten.mul]
    stream = get_stream(0)
    triton_poi_fused_mul_silu_0.run(buf, grid=grid(8), stream=stream)
    # Topologically Sorted Source Nodes: [silu, mul, linear], Original ATen: [aten.silu, aten.mul, aten.mm]
    extern_kernels.mm(buf, reinterpret_tensor(weight, (8, 8), (1, 8), 0), out=out)
    return out
'''
        nodes = {k: {"name": k, "target": v, "path": p} for k, v, p in
                 [('silu', 'silu', 'mlp.act'), ('mul', 'mul', 'mlp'), ('linear', 'linear', 'mlp.down')]}
        plan = launch_plan(source, nodes, "output_code.py")
        self.assertEqual([p['event'] for p in plan], ['triton_poi_fused_mul_silu_0', 'aten::mm'])
        self.assertEqual([n['name'] for n in plan[1]['source_nodes']], ['linear'])

    def test_count_mismatch_stays_unmapped_and_device_events_are_not_added(self):
        def event(name, device='CPU', duration=1000.):
            return SimpleNamespace(name=name, device_type=device, self_device_time_total=duration, input_shapes=[])
        plan = [{"event": 'aten::mm', 'source_nodes': [{'path': 'fc'}], 'mapping_complete': True}]
        values = compiled_records([event('aten::mm'), event('aten::mm'), event('kernel', 'CUDA')], plan, 0, 0)
        self.assertEqual(len(values), 2)
        self.assertTrue(all(not v['mapping_complete'] for v in values))
        values = compiled_records([event('aten::mm'), event('kernel', 'CUDA')], plan, 0, 0)
        self.assertTrue(values[0]['mapping_complete'])
        self.assertEqual(values[0]['time_ms'], 1.)

    def test_pytorch_211_runner_keeps_launch_order_and_source_ownership(self):
        source = '''class Runner:
    def __init__(self):
        extern_kernels.mm(ignore, ignore, out=ignore)
    def call(self, args):
        # Topologically Sorted Source Nodes: [activation], Original ATen: [aten.silu]
        triton_poi_fused_silu_0.run(buf, stream=get_stream(0))
        # Topologically Sorted Source Nodes: [activation, linear], Original ATen: [aten.silu, aten.mm]
        extern_kernels.mm(buf, weight, out=output)
        return output
'''
        nodes = {'activation': {'name': 'activation', 'target': 'silu', 'path': 'mlp.act'},
                 'linear': {'name': 'linear', 'target': 'linear', 'path': 'mlp.down'}}
        plan = launch_plan(source, nodes, 'output_code.py')
        self.assertEqual([p['event'] for p in plan], ['triton_poi_fused_silu_0', 'aten::mm'])
        self.assertEqual([n['path'] for n in plan[1]['source_nodes']], ['mlp.down'])
        self.assertTrue(all(p['mapping_complete'] for p in plan))
        self.assertEqual([p['line'] for p in plan], [6, 8])

    def test_ambiguous_compiler_entry_points_fail_closed(self):
        source = '''def call(args):
    pass
class Runner:
    def call(self, args):
        pass
'''
        with self.assertRaisesRegex(ValueError, 'multiple call entry points'):
            launch_plan(source, {}, 'output_code.py')

    def test_unrecognized_call_wrappers_fail_closed(self):
        sources = ['''class Other:
    def call(self, args):
        pass
''', '''def helper():
    def call(args):
        pass
''', '''class Runner(Unknown):
    def call(self, args):
        pass
''']
        for source in sources:
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, 'no call function'):
                launch_plan(source, {}, 'output_code.py')

    def test_builtin_addmm_keeps_its_projection_owner(self):
        source = ("def call(args):\n"
                  "    # Topologically Sorted Source Nodes: [reshape, x], Original ATen: [aten.addmm]\n"
                  "    extern_kernels.addmm(bias, hidden, weight, out=out)\n")
        nodes = {"reshape": {"target": "view", "path": "other"},
                 "x": {"target": "<built-in method addmm of type object>", "path": "block.projection"}}
        row, = launch_plan(source, nodes, "graph.py")
        self.assertTrue(row["mapping_complete"])
        self.assertEqual(row["source_nodes"], [nodes["x"]])

    def test_cross_boundary_fusion_cannot_be_charged_to_one_layer(self):
        tasks = [task('norm', 'rmsnorm', ['block.norm']), task('head', 'linear', ['head'])]
        records = [record('triton_fused_add_norm', ['block', 'block.norm']), record('aten::mm', ['head'])]
        report = rank_compiled(tasks, records, 1, 1)
        self.assertEqual(tasks[0]['ranking_excluded_reason'], 'cross_contract_fusion')
        self.assertEqual(tasks[0]['priority_ms'], 0.)
        self.assertEqual(tasks[1]['priority_ms'], 2.)
        self.assertIsNone(report['groups'][0]['task_id'])
        self.assertEqual(report['total_gpu_ms'], 4.)

    def test_complete_mlp_contract_can_own_fused_activation_and_multiply(self):
        tasks = [task('mlp', 'mlp', ['block.mlp']), task('act', 'silu', ['block.mlp.act']),
                 task('linear', 'linear', ['block.mlp.gate', 'block.mlp.down'])]
        records = [record('fused', ['block.mlp.act', 'block.mlp']),
                   record('aten::mm', ['block.mlp.gate'], 3.), record('aten::mm', ['block.mlp.down'], 4.)]
        report = rank_compiled(tasks, records, 1, 1)
        self.assertEqual(tasks[0]['priority_ms'], 9.)
        self.assertEqual(tasks[1]['priority_ms'], 0.)
        self.assertEqual(tasks[2]['priority_ms'], 7.)
        self.assertEqual(report['groups'][0]['task_id'], 'mlp')
        self.assertEqual(select_top_tasks(tasks, 4)['selected_task_ids'], ['mlp'])
        self.assertEqual(tasks[0]['priority_method'], METHOD)

    def test_missing_workload_and_unknown_sources_cannot_create_priority(self):
        tasks = [task('fc', 'linear', ['fc'])]
        rank_compiled(tasks, [record('mm', ['fc'])], 2, 1)
        self.assertEqual(tasks[0]['priority_ms'], 0.)
        self.assertEqual(tasks[0]['ranking_excluded_reason'], 'missing_compiled_profiler_coverage')

    def test_partial_path_coverage_cannot_rank_a_group_of_replacements(self):
        tasks=[task('fc','linear',['block.0.fc','block.1.fc'])]
        rank_compiled(tasks,[record('mm',['block.0.fc'])],1,1)
        self.assertEqual(tasks[0]['priority_ms'],0.)
        self.assertEqual(tasks[0]['ranking_excluded_reason'],'missing_compiled_path_coverage')

    def test_new_mode_requires_a_new_profile_and_is_default_in_project(self):
        config = load_config('configs/project.json')
        self.assertEqual(config['ranking_mode'], 'compiled_profiler')
        previous = copy.deepcopy(config)
        previous['ranking_mode'] = 'profiler_events'
        self.assertNotEqual(profile_settings(previous), profile_settings(config))
        with self.assertRaisesRegex(ValueError, 'ranking mode changed'):
            validate_catalog(config, {'hardware': {'fingerprint': 'x'}, 'config': previous}, {'fingerprint': 'x'})


if __name__ == '__main__':
    unittest.main()
