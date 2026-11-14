"""Run explicitly on a GPU. Synthetic controls do not establish model gains."""
from collections import OrderedDict
from pathlib import Path
import sys
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sako.core import candidate_id, load_config, write_json
from sako.gpu import check_output, model_operation
from sako.model import apply_manifest
from sako.shape_selection import case_signature
from triton_fixtures import LINEAR_SOURCE
from triton_fixture_tuning import linear_source


def compiler_autotune_source():
    # Torch 2.5 traces configs/key, but rejects non-default timing arguments.
    # Keep the shared eager tuning fixture unchanged. This is an explicit
    # synthetic control for the supported compiler API, not a search candidate.
    source = linear_source()
    timing = ', warmup=5, rep=20, use_cuda_graph=True'
    assert source.count(timing) == 1
    return source.replace(timing, '')


def main():
    import torch
    config = load_config(ROOT / 'configs/project.json')
    config = {**config, 'triton_integration': 'direct'}
    folder = ROOT / 'runs/compiler-integration-checks' / uuid.uuid4().hex
    folder.mkdir(parents=True)
    torch.manual_seed(1729)
    original = torch.nn.Sequential(OrderedDict(fc=torch.nn.Linear(96, 64, device='cuda', dtype=torch.float16)))
    inputs = [torch.randn((1, size, 96), device='cuda', dtype=torch.float16) for size in (128, 256)]
    task = {'id': 'linear_control', 'kind': 'linear', 'paths': ['fc'], 'scalars': [], 'cases': []}
    for x in inputs:
        tensors = [x, original.fc.weight, original.fc.bias]
        task['cases'].append({'shapes': [list(t.shape) for t in tensors],
            'strides': [list(t.stride()) for t in tensors], 'dtypes': [str(t.dtype) for t in tensors]})
    reports, state = [], {}
    def progress(stage, **fields):
        state.clear()
        state.update(stage=stage, **fields)
        write_json(folder / 'progress.json', state)
        print({'compiler_integration_stage': stage, **fields}, flush=True)
    def save(status, **fields):
        result = {'status': status, 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
                  'reports': reports, 'stage': dict(state), 'acceptance_evidence': False,
                  'model_gain': 'not_measured', **fields}
        write_json(folder / 'result.json', result)
        return result
    try:
        with torch.inference_mode():
            for label, source in [('plain', LINEAR_SOURCE), ('autotuned', compiler_autotune_source())]:
                candidate = {'name': 'compiler_' + label, 'task_id': task['id'], 'backend': 'triton',
                    'references': [], 'strategy': 'Synthetic compiler integration test.', 'source': source}
                key = candidate_id(candidate)
                write_json(folder / (key + '.json'), candidate)
                progress(label + ':load', candidate_id=key)
                operation = model_operation(candidate, 'linear', config)
                for index, x in enumerate(inputs):
                    progress(label + ':eager', input_index=index, shape=list(x.shape))
                    # Eager warmup performs build diagnostics before Dynamo sees the entry.
                    check = check_output(operation([x, original.fc.weight, original.fc.bias], []), original(x), config)
                    assert check['passed'], check
                progress(label + ':compile_callable')
                compiled = torch.compile(lambda x: operation([x, original.fc.weight, original.fc.bias], []),
                                         fullgraph=True, dynamic=False, mode=config['compile_mode'])
                checks = []
                for index, x in enumerate(inputs):
                    progress(label + ':compiled', input_index=index, shape=list(x.shape))
                    checks.append(check_output(compiled(x), original(x), config))
                assert all(check['passed'] for check in checks), checks
                progress(label + ':routing_setup')
                manifest = {'entries': [{'task_id': task['id'], 'candidate_id': key,
                    'paths': ['fc'], 'signatures': [case_signature(task['cases'][0])]}]}
                alternative = torch.nn.Sequential(OrderedDict(fc=original.fc))
                apply_manifest(alternative, manifest, {'tasks': [task], 'config': config}, folder, config=config)
                progress(label + ':routing_eager')
                alternative(inputs[0])
                progress(label + ':routing_compile_callable')
                routed = torch.compile(alternative, fullgraph=True, dynamic=False, mode=config['compile_mode'])
                routed_checks = []
                for index, x in enumerate(inputs):
                    progress(label + ':routing_compiled', input_index=index, shape=list(x.shape))
                    routed_checks.append(check_output(routed(x), original(x), config))
                assert all(check['passed'] for check in routed_checks), routed_checks
                reports.append({'source_kind': label, 'integration': 'direct', 'fullgraph': True,
                    'compiled_checks': checks, 'shape_routing_checks': routed_checks,
                    'shape_routing': ['candidate', 'original_fallback']})
                progress(label + ':passed')
                save('running')
            if torch.__version__.split('+')[0] == '2.5.1':
                progress('unsupported_autotune:load')
                unsupported_candidate = {'name': 'compiler_unsupported_autotune', 'task_id': task['id'],
                    'backend': 'triton', 'references': [],
                    'strategy': 'Negative control for unsupported compiler timing arguments.',
                    'source': linear_source()}
                write_json(folder / (candidate_id(unsupported_candidate) + '.json'), unsupported_candidate)
                unsupported = model_operation(unsupported_candidate, 'linear', config)
                progress('unsupported_autotune:eager')
                unsupported([inputs[0], original.fc.weight, original.fc.bias], [])
                progress('unsupported_autotune:compiled')
                rejected = torch.compile(lambda x: unsupported([x, original.fc.weight, original.fc.bias], []),
                                         fullgraph=True, dynamic=False, mode=config['compile_mode'])
                try:
                    rejected(inputs[0])
                except torch._dynamo.exc.Unsupported as error:
                    if 'Only configs and keys are supported for triton.autotune' not in str(error):
                        raise
                    reports.append({'source_kind': 'unsupported_autotune', 'status': 'expected_rejection',
                        'reason': 'Only configs and keys are supported for triton.autotune',
                        'unsupported_arguments': ['warmup=5', 'rep=20', 'use_cuda_graph=True']})
                else:
                    raise AssertionError('Torch 2.5.1 accepted the unsupported autotune control unexpectedly.')
        progress('passed')
        print(save('passed'), flush=True)
    except Exception as error:
        save('failed', error=type(error).__name__ + ': ' + str(error), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
