"""Keep shape ratios, their baselines, and accepted model use separate."""

import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

from sako.core import digest
from sako.store import Store
from sako.tui.ratios import speed_text
from sako.tui.runs import run_snapshot


def evidence(run):
    store = Store(run / 'state.sqlite')
    first = {'shapes': [[2, 8], [8]], 'strides': [[8, 1], [1]],
             'dtypes': ['torch.float16', 'torch.float16'], 'scalars': [1e-6]}
    second = {**first, 'shapes': [[4, 8], [8]]}
    a, b = digest(first), digest(second)
    result = {'decision': {'accepted': False, 'speedup': 1.05},
        'timing_baseline': {'kind': 'per_shape', 'shapes': {a: None, b: 'prior_kernel'}},
        'shape_decisions': {a: {'signature': first, 'indices': [0], 'accepted': True, 'speedup': 1.25},
                            b: {'signature': second, 'indices': [1], 'accepted': False, 'speedup': .8}},
        'cases': [{'correct': True, 'baseline': {'median_ms': .02}, 'candidate': {'median_ms': .016}},
                  {'correct': True, 'baseline': {'median_ms': .03}, 'candidate': {'median_ms': .0375}}]}
    with store.connect() as db:
        db.execute('INSERT INTO candidates(id,task,island,accepted,result) VALUES (?,?,?,?,?)',
                   ('partial', 'rmsnorm', 'memory', 0, json.dumps(result)))
    store.set('best_model_measurement', {'manifest': {'entries': [
        {'candidate_id': 'partial', 'signatures': [first]}]},
        'result': {'decision': {'accepted': True, 'speedup': 1.1}}})
    return a, b


class RatioEvidenceTests(unittest.TestCase):
    def test_selected_candidates_stay_visible_after_thirty_newer_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            evidence(run)
            store = Store(run / 'state.sqlite')
            with store.connect() as db:
                db.executemany('INSERT INTO candidates(id,task,island) VALUES (?,?,?)',
                    [(f'new-{i}', 'rmsnorm', 'memory') for i in range(40)])
            snapshot = run_snapshot(run, compact=True)
            self.assertIsNone(snapshot['warning'])
            self.assertEqual(snapshot['selected_candidates'], ['partial'])
            self.assertEqual(snapshot['candidates'][0][0], 'partial')
            self.assertEqual(len(snapshot['candidates']), 30)

    def test_percent_uses_elapsed_time_and_handles_slow_invalid_and_missing_ratios(self):
        self.assertEqual(speed_text(1.25), '1.250× · 20.0% less time')
        self.assertEqual(speed_text(.8), '0.800× · 25.0% more time')
        for value in (None, 0, -1, float('nan'), float('inf'), True):
            self.assertEqual(speed_text(value), 'not measured')

    def test_shape_baselines_and_model_use_are_not_derived_from_the_aggregate_ratio(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            a, b = evidence(run)
            snapshot = run_snapshot(run)
            self.assertIsNone(snapshot['warning'])
            shapes = {row['shape_id']: row for row in snapshot['candidate_details']['partial']}
            self.assertEqual(shapes[a]['baseline'], 'torch.compile')
            self.assertEqual(shapes[b]['baseline'], 'kept prior_kernel')
            self.assertEqual(shapes[a]['model_use'], 'used')
            self.assertEqual(shapes[b]['model_use'], 'not used')
            self.assertEqual(shapes[a]['candidate_ms'], .016)
            self.assertEqual(shapes[b]['speedup'], .8)
            self.assertEqual(snapshot['model'], '1.100×  ·  accepted')


@unittest.skipUnless(importlib.util.find_spec('textual'), 'Terminal UI dependency is optional.')
class RatioScreenTests(unittest.IsolatedAsyncioTestCase):
    async def test_decode_columns_use_eager_time_as_the_ratio_reference(self):
        from rich.console import Console
        from sako.tui.app import RunView
        from types import SimpleNamespace
        data = {'decode_metrics': [{'original_ms_per_token': 25., 'pytorch_ms_per_token': 40.,
            'ms_per_token': 20., 'tokens_per_second': 50., 'vs_original_compile': 1.25}], 'final_progress': None}
        rendered = io.StringIO()
        Console(width=132, file=rendered, color_system=None).print(RunView.model_summary(SimpleNamespace(data=data)))
        for value in ('PyTorch', 'torch.compile', 'Our optimization', '40.000', '25.000', '20.000', '1.600×', '2.000×'):
            self.assertIn(value, rendered.getvalue())

    async def test_candidates_have_one_row_and_keep_the_model_gain_separate(self):
        from rich.console import Console
        from textual.widgets import Static
        from sako.tui.app import HarnessApp
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / 'runs/trial'
            evidence(run)
            app = HarnessApp(root, 'missing.json', run)
            async with app.run_test(size=(132, 38)) as pilot:
                await pilot.pause()
                await app.workers.wait_for_complete()
                await pilot.press('c')
                await app.workers.wait_for_complete()
                await pilot.pause()
                text = io.StringIO()
                console = Console(width=132, file=text, color_system=None)
                console.print(app.screen.query_one('#content', Static).content)
                rendered = text.getvalue()
                for value in ('torch.compile', '1.100×', 'partial 1/2', 'Local', 'Rank'):
                    self.assertIn(value, rendered)
                self.assertEqual(rendered.count('rmsnorm'), 1)
                self.assertNotIn('1.050×', rendered)


if __name__ == '__main__':
    unittest.main()
