"""Explicit GPU test: real fusion maps, library calls, and safe targets."""
from pathlib import Path
import torch

from sako.core import load_config, write_json
from sako.compiled_ranking import profile_compiled, METHOD
from sako.model import Collector, arguments, forward, inputs, module_kind
from sako.gpu import reference
from sako.contracts import operation_contract
from sako.hardware import select_top_tasks
from sako.hierarchy import select_hierarchy
from tests.gpu_event_ranking_checks import TinyModel


def run():
    config = load_config('configs/project.json')
    output = Path('runs/compiled-ranking-native')
    output.mkdir(parents=True, exist_ok=True)
    model = TinyModel().eval().cuda().half().requires_grad_(False)
    collector, handles, saved = Collector(config), [], {}
    for path, module in model.named_modules():
        kind = module_kind(module)
        if kind is None:
            continue
        def before(mod, args, path=path, kind=kind):
            tensors, scalars = arguments(kind, mod, args[0])
            key = collector.capture(path, kind, tensors, scalars)
            saved[path] = (key, tensors, scalars)
        def after(mod, args, result, path=path, kind=kind):
            key, tensors, scalars = saved[path]
            collector.validate(key, path, reference(kind, tensors, scalars), result)
        handles.extend([module.register_forward_pre_hook(before), module.register_forward_hook(after)])
    with torch.inference_mode():
        try:
            for workload in config['workloads']:
                collector.context = {'workload': workload, 'seed': config['correctness_seeds'][0]}
                forward(model, inputs(model, workload, config['correctness_seeds'][0]))
        finally:
            for handle in handles:
                handle.remove()
        ids = inputs(model, config['workloads'][0], config['correctness_seeds'][0])
        expected = forward(model, ids).clone()
        ranking, compiled = profile_compiled(model, config, list(collector.tasks.values()), output)
        torch.testing.assert_close(forward(model, ids), expected, atol=0, rtol=0)
    for key, task in collector.tasks.items():
        task['contract'] = operation_contract(task, collector.validations[key], config)
    tasks = list(collector.tasks.values())
    selected = select_hierarchy(tasks)
    assert selected['selected_task_ids']
    assert selected['hierarchy']['regions'][0]['kind']=='mlp'
    chosen=[t for t in tasks if t['id'] in selected['selected_task_ids']]
    assert all(t['kind']!='mlp' for t in chosen)
    assert any(any(p.endswith('down_proj') for p in t['paths']) for t in chosen)
    assert ranking['model_execution'] == 'torch_compile'
    assert all(t['priority_method'] == METHOD for t in tasks)
    assert ranking['mapped_gpu_ms'] / ranking['total_gpu_ms'] > .9
    assert any(g['name'] == 'aten::mm' for g in ranking['groups'])
    assert any(g['name'].startswith('triton_') for g in ranking['groups'])
    assert all(t['priority_ms'] == 0 for t in tasks if t['kind'] == 'silu')
    assert any(t['kind'] == 'mlp' and t['priority_ms'] > 0 for t in tasks)
    assert all(t['compiled_evidence']['generated_kernels'] for t in tasks if t['kind'] == 'mlp')
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
    write_json(output/'result.json', {'passed': True, 'ranking': ranking, 'compiled': compiled,
               'tasks': tasks, 'selection': selected, 'note': 'Small GPU validation model. No speed gain is claimed.'})
    print('Compiled fusion mapping and safe target selection passed.', flush=True)


if __name__ == '__main__':
    run()
