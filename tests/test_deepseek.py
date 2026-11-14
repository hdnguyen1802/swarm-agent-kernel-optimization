"""Exercise provider routing, DeepSeek HTTP requests, and the shared budget."""

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from sako.agent import AuthenticationUnavailable, Gemini, create_agent
from sako.core import load_config, write_json
from sako.deepseek import DeepSeek, DeepSeekRequestError
from sako.store import BudgetExhausted, Store


ROOT = Path(__file__).resolve().parents[1]


def reply(content='{"ok":true}', *, usage=True, finish="stop"):
    return {"choices": [{"message": {"content": content, "reasoning_content": "private reasoning"},
                         "finish_reason": finish}],
            **({"usage": {"total_tokens": 34, "prompt_tokens": 21, "completion_tokens": 13,
                          "prompt_cache_hit_tokens": 7,
                          "completion_tokens_details": {"reasoning_tokens": 5}}} if usage else {})}


def connection(payload=None, *, status=200):
    client = MagicMock()
    response = client.getresponse.return_value
    response.status = status
    response.read.return_value = json.dumps(payload if payload is not None else reply()).encode()
    return client


class DeepSeekTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.run = Path(temporary.name)
        self.config = load_config(ROOT / 'configs/generation-deepseek.json')['agent']
        self.store = Store(self.run / 'state.sqlite')
        self.client = create_agent(self.config, self.store, self.run)
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'test-only-secret'}).start()
        self.transport = patch('sako.deepseek.HTTPSConnection').start()
        self.connection = connection()
        self.transport.return_value = self.connection

    def records(self):
        return [json.loads(p.read_text(encoding='utf-8')) for p in sorted((self.run/'calls').glob('*.json'))]

    def test_provider_factory_routes_both_clients(self):
        self.assertIsInstance(self.client, DeepSeek)
        vertex = load_config(ROOT / 'configs/generation.json')['agent']
        self.assertIsInstance(create_agent(vertex, self.store, self.run), Gemini)
        vertex.pop('provider')
        self.assertIsInstance(create_agent(vertex, self.store, self.run), Gemini)

    def test_posts_json_to_deepseek_without_google_credentials(self):
        with patch.object(Gemini, '_credentials_for_client', side_effect=AssertionError('Google not required')):
            self.assertEqual(self.client.call('writer_0_memory', 'write a kernel', 'system'), {'ok': True})
        self.transport.assert_called_once_with('api.deepseek.com', timeout=self.config['timeout_seconds'])
        args, fields = self.connection.request.call_args
        self.assertEqual(args, ('POST', '/chat/completions'))
        self.assertEqual(fields['headers']['Authorization'], 'Bearer test-only-secret')
        body = json.loads(fields['body'])
        self.assertEqual(body['model'], self.config['model'])
        self.assertEqual(body['response_format'], {'type': 'json_object'})
        self.assertEqual(body['max_tokens'], self.config['max_output_tokens'])
        self.assertEqual(body['thinking'], {'type': 'enabled'})
        self.assertEqual(body['reasoning_effort'], 'low')
        self.assertFalse(body['stream'])
        self.assertEqual([m['role'] for m in body['messages']], ['system', 'user'])
        self.assertIn('JSON', body['messages'][0]['content'])
        self.connection.close.assert_called_once()

    def test_coordinator_receives_the_same_schema_in_count_and_request(self):
        self.client.call('coordinator', '{"allowed_actions":["wait"]}', 'system')
        body = json.loads(self.connection.request.call_args.kwargs['body'])
        system = body['messages'][0]['content']
        self.assertIn('response structure', system)
        self.assertIn('"enum": ["wait"]', system)
        self.assertIn('"type": "object"', system)
        self.assertNotIn('response_schema', body)
        generation = next(r for r in self.records() if r['role'] == 'coordinator' and 'endpoint' not in r)
        counted = self.client._count_input_tokens('{"allowed_actions":["wait"]}', 'system', generation['response_schema'])
        self.assertEqual(generation['reserved_tokens'], counted + self.config['max_output_tokens'])

    def test_local_input_bound_includes_instructions_schema_and_unicode(self):
        schema = {'type': 'OBJECT', 'properties': {'answer': {'type': 'STRING'}}}
        bound = self.client._count_input_tokens('GPU \U0001f680 \u4e2d\u6587', 'system', schema)
        sent = json.dumps(self.client._messages('GPU \U0001f680 \u4e2d\u6587', 'system', schema), ensure_ascii=False).encode()
        self.assertGreater(bound, len(sent))
        self.assertGreater(bound, self.client._count_input_tokens('GPU \U0001f680 \u4e2d\u6587', 'system'))
        self.client.call('writer_0_memory', 'test', 'system')
        counter = next(r for r in self.records() if r.get('endpoint') == 'count_tokens')
        self.assertFalse(counter['request']['remote_request'])
        self.assertEqual(counter['request']['method'], 'utf8_byte_bound')
        self.assertEqual(self.transport.call_count, 1)

    def test_actual_usage_replaces_the_bound_without_double_counting_reasoning(self):
        self.client.call('writer_0_memory', 'test', 'system')
        self.assertEqual(self.store.usage()['tokens'], 34)
        record = next(r for r in self.records() if r['role'] == 'writer_0_memory' and 'endpoint' not in r)
        self.assertEqual(record['usage']['input_tokens'], 21)
        self.assertEqual(record['usage']['output_tokens'], 13)
        self.assertEqual(record['usage']['reasoning_tokens'], 5)
        self.assertEqual(record['usage']['cached_input_tokens'], 7)
        self.assertGreater(record['reserved_tokens'], 34)
        saved = '\n'.join(p.read_text(encoding='utf-8') for p in (self.run/'calls').glob('*.json'))
        self.assertNotIn('test-only-secret', saved)
        self.assertNotIn('private reasoning', saved)

    def test_disabled_thinking_does_not_send_reasoning_effort(self):
        self.config['thinking'] = 'disabled'
        self.client.call('writer_0_memory', 'test', 'system')
        body = json.loads(self.connection.request.call_args.kwargs['body'])
        self.assertEqual(body['thinking'], {'type': 'disabled'})
        self.assertNotIn('reasoning_effort', body)

    def test_missing_key_stops_before_http_without_spending_tokens(self):
        with patch.dict(os.environ, {'DEEPSEEK_API_KEY': ''}):
            with self.assertRaises(AuthenticationUnavailable):
                self.client.call('writer_0_memory', 'test', 'system')
        self.transport.assert_not_called()
        self.assertEqual(self.store.usage()['tokens'], 0)
        generation = next(r for r in self.records() if r['role'] == 'writer_0_memory' and 'endpoint' not in r)
        self.assertIs(generation['dispatch_started'], False)

    def test_custom_key_environment_is_used(self):
        self.config['api_key_env'] = 'SAKO_TEST_KEY'
        with patch.dict(os.environ, {'SAKO_TEST_KEY': 'other-test-secret'}):
            self.client.call('writer_0_memory', 'test', 'system')
        self.assertEqual(self.connection.request.call_args.kwargs['headers']['Authorization'], 'Bearer other-test-secret')

    def test_auth_rejection_is_sanitized_and_does_not_retry(self):
        self.connection.getresponse.return_value.status = 401
        self.connection.getresponse.return_value.read.return_value = b'error test-only-secret'
        with self.assertRaisesRegex(DeepSeekRequestError, 'HTTP 401'):
            self.client.call('writer_0_memory', 'test', 'system')
        self.assertEqual(self.transport.call_count, 1)
        self.assertEqual(self.store.usage()['tokens'], 0)
        self.connection.getresponse.return_value.read.assert_not_called()
        self.assertNotIn('test-only-secret', json.dumps(self.records()))

    def test_rate_limit_uses_shared_backoff_and_counts_only_the_success(self):
        first, second = connection(status=429), connection()
        self.transport.side_effect = [first, second]
        with patch('sako.agent.time.sleep'):
            self.assertTrue(self.client.call('writer_0_memory', 'test', 'system')['ok'])
        self.assertEqual(self.store.usage()['tokens'], 34)
        self.assertEqual(self.transport.call_count, 2)
        self.assertLess(self.client.gate.limit, self.client.gate.maximum)

    def test_transient_failure_cannot_retry_past_the_budget(self):
        reserve = self.client._count_input_tokens('test', 'system') + self.config['max_output_tokens']
        self.config['max_total_tokens'] = reserve + 1
        self.connection.getresponse.return_value.status = 503
        with patch('sako.agent.time.sleep'):
            with self.assertRaises(BudgetExhausted):
                self.client.call('writer_0_memory', 'test', 'system')
        self.assertEqual(self.transport.call_count, 1)
        self.assertEqual(self.store.usage()['tokens'], reserve)

    def test_timeout_keeps_reservation_and_hides_transport_error_details(self):
        self.connection.getresponse.side_effect = TimeoutError('test-only-secret')
        with self.assertRaisesRegex(DeepSeekRequestError, 'TimeoutError'):
            self.client.call('writer_0_memory', 'test', 'system')
        expected = self.client._count_input_tokens('test', 'system') + self.config['max_output_tokens']
        self.assertEqual(self.store.call_stats()['unresolved_tokens'], expected)
        self.assertNotIn('test-only-secret', json.dumps(self.records()))
        self.connection.close.assert_called_once()

    def test_missing_usage_keeps_the_reservation(self):
        self.connection.getresponse.return_value.read.return_value = json.dumps(reply(usage=False)).encode()
        self.assertTrue(self.client.call('writer_0_memory', 'test', 'system')['ok'])
        self.assertEqual(self.store.call_stats()['unresolved_tokens'],
                         self.client._count_input_tokens('test', 'system') + self.config['max_output_tokens'])

    def test_invalid_or_truncated_json_keeps_reported_usage_and_is_not_cached(self):
        for content, finish in [('not JSON', 'stop'), ('{"ok":true}', 'length')]:
            with self.subTest(content=content, finish=finish):
                self.connection.getresponse.return_value.read.return_value = json.dumps(reply(content, finish=finish)).encode()
                with self.assertRaises(ValueError):
                    self.client.call('writer_0_memory', content, 'system')
        self.assertEqual(self.store.usage()['tokens'], 68)
        self.assertTrue(all(r['status'] == 'failed' for r in self.records()
                            if r['role'] == 'writer_0_memory' and 'endpoint' not in r))

    def test_budget_and_context_limits_block_http_dispatch(self):
        self.config['max_total_tokens'] = 1
        with self.assertRaises(BudgetExhausted):
            self.client.call('writer_0_memory', 'test', 'system')
        self.config['max_total_tokens'] = 1_000_000
        self.config['model_context_tokens'] = self.config['max_output_tokens']
        self.client.provider_limits = None
        with self.assertRaisesRegex(ValueError, 'output reserve'):
            self.client.call('writer_0_memory', 'test', 'system')
        self.transport.assert_not_called()

    def test_resume_reuses_a_success_but_does_not_cross_provider_or_settings(self):
        self.client.call('writer_0_memory', 'test', 'system')
        create_agent(self.config, Store(self.run/'state.sqlite'), self.run).call('writer_0_memory', 'test', 'system')
        self.assertEqual(self.transport.call_count, 1)
        config = {**self.config, 'thinking': 'disabled'}
        create_agent(config, self.store, self.run).call('writer_0_memory', 'test', 'system')
        self.assertEqual(self.transport.call_count, 2)
        vertex = load_config(ROOT/'configs/generation.json')['agent']
        vertex.update(model=self.config['model'], adaptive_budget=False)
        with patch.object(Gemini, '_request', return_value=('{"vertex":true}', 20)) as request:
            self.assertEqual(create_agent(vertex, self.store, self.run).call('writer', 'test', 'system'), {'vertex': True})
        request.assert_called_once()


class ProviderConfigurationTests(unittest.TestCase):
    def test_deepseek_template_preserves_gpu_and_numerical_settings(self):
        vertex = load_config(ROOT/'configs/generation.json')
        deepseek = load_config(ROOT/'configs/generation-deepseek.json')
        for k in vertex:
            if k != 'agent':
                self.assertEqual(vertex[k], deepseek[k], k)
        for k in vertex['agent']:
            if k not in {'provider','model','project','location','model_context_tokens','model_output_tokens',
                         'model_limits_source','model_limits_verified'}:
                self.assertEqual(vertex['agent'][k], deepseek['agent'][k], k)

    def test_invalid_provider_credentials_and_model_limits_fail_config_validation(self):
        config = load_config(ROOT/'configs/generation-deepseek.json')
        for change in [{'provider':'other'}, {'api_key':'do-not-save-keys'}, {'api_key_env':'bad name'},
                       {'thinking':'maybe'}, {'reasoning_effort':'medium'}, {'model_context_tokens':None},
                       {'model_output_tokens':True}, {'model_output_tokens':1}]:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'config.json'
                write_json(path, {**config, 'agent':{**config['agent'], **change}})
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_api_check_uses_selected_provider_without_a_gpu_or_google_login(self):
        from sako import cli
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/'runs/check';config=load_config(ROOT/'configs/generation-deepseek.json')
            path=root/'config.json';write_json(path,config)
            http=connection(reply('{"connected":true}'))
            with patch.dict(os.environ,{'DEEPSEEK_API_KEY':'test-only-secret'}), \
                 patch('sako.deepseek.HTTPSConnection',return_value=http), patch.object(cli,'ROOT',root), \
                 patch.object(cli,'Worker') as worker, \
                 patch('sys.argv',['sako','api-check','--config',str(path),'--run',str(run)]), \
                 contextlib.redirect_stdout(io.StringIO()):
                cli.main()
            worker.return_value.run.assert_not_called()
            self.assertEqual(json.loads((run/'api-check.json').read_text()),{'connected':True})

    def test_search_passes_the_provider_factory(self):
        from sako.search import search
        with patch('sako.swarm.SwarmLoop') as loop:
            config=load_config(ROOT/'configs/generation-deepseek.json')
            search(config,Path('runs/test'),Path('runs/test/profile'),worker=object())
        self.assertIs(loop.call_args.args[3],create_agent)


if __name__ == '__main__':
    unittest.main()
