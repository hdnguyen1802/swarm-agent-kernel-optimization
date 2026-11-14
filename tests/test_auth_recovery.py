"""Only a proven pre-dispatch failure releases the token reservation."""
import datetime
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sako.agent import AuthenticationUnavailable, Gemini
from sako.core import load_config
from sako.store import Store


class AuthenticationRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        config = load_config(Path(__file__).resolve().parents[1] / "configs/project.json")["agent"]
        config.update(adaptive_budget=False, max_retries=2)
        self.store = Store(self.run / "state.sqlite")
        self.client = Gemini(config, self.store, self.run)

    def test_transport_auth_retries_are_bounded_and_do_not_spend_tokens(self):
        with patch.object(self.client, "_request", side_effect=AuthenticationUnavailable("offline")) as request, \
             patch("sako.agent.time.sleep"):
            with self.assertRaises(AuthenticationUnavailable):
                self.client.call("writer", "prompt", "system")
        self.assertEqual(request.call_count, 3)
        self.assertEqual(self.store.usage()["tokens"], 0)
        self.assertEqual(self.store.call_stats()["unresolved_tokens"], 0)
        calls = [json.loads(path.read_text()) for path in (self.run / "calls").glob("*.json")]
        self.assertTrue(all(c["dispatch_started"] is False for c in calls))

    def test_invalid_credentials_stop_without_repeated_refresh(self):
        with patch.object(self.client, "_request", side_effect=AuthenticationUnavailable("invalid", retryable=False)) as request:
            with self.assertRaises(AuthenticationUnavailable):
                self.client.call("writer", "prompt", "system")
        self.assertEqual(request.call_count, 1)
        self.assertEqual(self.store.usage()["tokens"], 0)

    def test_unknown_generation_outcome_keeps_reservation(self):
        with patch.object(self.client, "_request", side_effect=TimeoutError("unknown outcome")):
            with self.assertRaises(TimeoutError):
                self.client.call("writer", "prompt", "system")
        self.assertGreater(self.store.call_stats()["unresolved_tokens"], 0)

    def test_auth_recovery_only_counts_successful_generation_tokens(self):
        with patch.object(self.client, "_request", side_effect=[AuthenticationUnavailable("offline"), ('{"ok":true}', 23)]), \
             patch("sako.agent.time.sleep"):
            self.assertTrue(self.client.call("writer", "prompt", "system")["ok"])
        self.assertEqual(self.store.usage()["tokens"], 23)

    def test_parallel_calls_share_one_credential_refresh(self):
        credentials = SimpleNamespace(valid=False, expiry=None)
        refreshes = []
        def refresh(request):
            refreshes.append(request)
            credentials.valid = True
            credentials.expiry = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1)
        credentials.refresh = refresh
        with patch("google.auth.default", return_value=(credentials, "project")) as default:
            with ThreadPoolExecutor(max_workers=5) as pool:
                values = list(pool.map(lambda _: self.client._credentials_for_client(), range(5)))
        self.assertEqual(default.call_count, 1)
        self.assertEqual(len(refreshes), 1)
        self.assertTrue(all(value is credentials for value in values))

    def test_auth_error_log_does_not_include_provider_secret_text(self):
        from google.auth.exceptions import RefreshError
        with patch('google.auth.default', side_effect=RefreshError('secret-provider-detail')), \
             patch.object(self.client, '_request', side_effect=lambda *args: self.client._credentials_for_client()):
            with self.assertRaises(AuthenticationUnavailable):
                self.client.call('writer', 'prompt', 'system')
        saved = next((self.run / 'calls').glob('*.json')).read_text()
        self.assertNotIn('secret-provider-detail', saved)
        self.assertIn('RefreshError', saved)


if __name__ == "__main__":
    unittest.main()
