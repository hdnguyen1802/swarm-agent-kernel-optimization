import tempfile
import unittest
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event, Lock
from unittest.mock import patch

from sako.agent import Gemini
from sako.core import load_config
from sako.store import Store


class QuotaError(RuntimeError):
    code = 429


class AgentLimits(unittest.TestCase):
    def test_coordinator_counts_and_sends_the_same_plan_schema(self):
        config = load_config(Path(__file__).resolve().parents[1] / "configs/project.json")["agent"]
        with tempfile.TemporaryDirectory() as directory:
            client = Gemini(config, Store(Path(directory) / "state.sqlite"), directory)
            with patch.object(client, "_count_input_tokens", return_value=125) as count, \
                 patch.object(client, "_request", return_value=('{"action":"continue"}', 133)) as generate:
                client.call("coordinator", '{"allowed_actions":["continue"]}', "system")
            schema = generate.call_args.args[2]["response_schema"]
            self.assertEqual(schema, count.call_args.kwargs["response_schema"])
            self.assertEqual(schema["properties"]["action"]["enum"], ["continue"])
            hypothesis = schema["properties"]["hypotheses"]["items"]
            checks = hypothesis["properties"]["verification"]
            self.assertEqual(checks["type"], "OBJECT")
            self.assertNotIn("verification", hypothesis["required"])
            self.assertIn("objective", hypothesis["properties"])
            self.assertEqual(checks["properties"]["implementation"]["items"]["required"], ["field", "op", "value"])
            self.assertNotIn("response_schema", client.limits_for_role("writer_0_memory"))

    def test_retry_counts_each_request(self):
        config = load_config(Path(__file__).resolve().parents[1] / "configs/project.json")["agent"]
        config["adaptive_budget"] = False
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "state.sqlite")
            client = Gemini(config, store, directory)
            with patch.object(client, "_request", side_effect=[QuotaError("quota"), ('{"ok":true}', 12)]), \
                 patch("sako.agent.time.sleep") as sleep:
                self.assertTrue(client.call("writer", "test", "system")["ok"])
                self.assertTrue(sleep.called)
            self.assertEqual(store.usage(), {"calls": 2, "tokens": 12})

    def test_invalid_json_keeps_measured_token_use(self):
        config = load_config(Path(__file__).resolve().parents[1] / "configs/project.json")["agent"]
        config["adaptive_budget"] = False
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "state.sqlite")
            client = Gemini(config, store, directory)
            with patch.object(client, "_request", return_value=("invalid JSON", 17)):
                with self.assertRaises(ValueError):
                    client.call("writer", "test", "system")
            self.assertEqual(store.usage(), {"calls": 1, "tokens": 17})
            record = json.loads(next((Path(directory) / "calls").glob("*.json")).read_text())
            self.assertEqual(record["response"], "invalid JSON")
            self.assertIn("JSONDecodeError", record["traceback"])

    def test_requests_obey_four_call_limit(self):
        config = load_config(Path(__file__).resolve().parents[1] / "configs/project.json")["agent"]
        config["adaptive_budget"] = False
        config["max_parallel_calls"] = 4
        with tempfile.TemporaryDirectory() as directory:
            client = Gemini(config, Store(Path(directory) / "state.sqlite"), directory)
            state = {"active": 0, "maximum": 0}
            lock = Lock()
            barrier = Barrier(4)
            def request(*args):
                with lock:
                    state["active"] += 1
                    state["maximum"] = max(state["active"], state["maximum"])
                barrier.wait(timeout=5)
                with lock:
                    state["active"] -= 1
                return '{"ok":true}', 10
            with patch.object(client, "_request", side_effect=request), ThreadPoolExecutor(max_workers=8) as pool:
                result = list(pool.map(lambda n: client.call(str(n), "test", "system"), range(8)))
            self.assertEqual(len(result), 8)
            self.assertEqual(state["maximum"], 4)

    def test_coordinator_counts_tokens_instead_of_characters(self):
        config = load_config(Path(__file__).resolve().parents[1] / "configs/project.json")["agent"]
        config["adaptive_budget"] = True
        config.pop("model_context_tokens", None)
        config.pop("model_output_tokens", None)
        config["coordinator_max_input_tokens"] = 150000
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "state.sqlite")
            client = Gemini(config, store, directory)
            with patch.object(client, "_get_provider_limits", return_value={
                    "input_token_limit": 1000000, "output_token_limit": 65536}), \
                 patch.object(client, "_count_input_tokens", return_value=140000) as counter, \
                 patch.object(client, "_request", return_value=('{"ok":true}', 140100)):
                self.assertTrue(client.call("coordinator", "x" * 300000, "system")["ok"])
            self.assertEqual(counter.call_args.args[1], "system")
            self.assertEqual(store.usage(), {"calls": 3, "tokens": 140100})
            with store.connect() as db:
                row = db.execute("SELECT reserved FROM calls WHERE role='coordinator'").fetchone()
            self.assertEqual(row["reserved"], 140000 + config["max_output_tokens"])

    def test_coordinator_rejects_input_above_token_limit(self):
        config = load_config(Path(__file__).resolve().parents[1] / "configs/project.json")["agent"]
        config["adaptive_budget"] = False
        config.update(coordinator_max_input_tokens=150000, model_context_tokens=1000000)
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "state.sqlite")
            client = Gemini(config, store, directory)
            with patch.object(client, "_count_input_tokens", return_value=150001), \
                 patch.object(client, "_request") as request:
                with self.assertRaisesRegex(ValueError, "150001"):
                    client.call("coordinator", "small", "system")
            request.assert_not_called()
            self.assertEqual(store.usage(), {"calls": 1, "tokens": 0})

    def test_reported_reasoning_tokens_are_not_added_twice(self):
        config = load_config(Path(__file__).resolve().parents[1] / "configs/project.json")["agent"]
        config["adaptive_budget"] = False
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "state.sqlite")
            client = Gemini(config, store, directory)
            usage = {"total_tokens": 35, "input_tokens": 10, "output_tokens": 15, "reasoning_tokens": 10}
            with patch.object(client, "_request", return_value=('{"ok":true}', usage)):
                self.assertTrue(client.call("writer", "test", "system")["ok"])
            self.assertEqual(store.usage()["tokens"], 35)
            record = json.loads(next((Path(directory) / "calls").glob("*.json")).read_text())
            self.assertEqual(record["usage"], usage)

    def test_resume_reuses_a_known_success_without_an_api_request(self):
        config = load_config(Path(__file__).resolve().parents[1] / "configs/project.json")["agent"]
        config["adaptive_budget"] = False
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "state.sqlite")
            with patch.object(Gemini, "_request", return_value=('{"ok":true}', 12)) as request:
                self.assertTrue(Gemini(config, store, directory).call("writer", "test", "system")["ok"])
                self.assertTrue(Gemini(config, Store(Path(directory) / "state.sqlite"), directory).call(
                    "writer", "test", "system")["ok"])
            self.assertEqual(request.call_count, 1)
            self.assertEqual(store.usage(), {"calls": 1, "tokens": 12})

    def test_adaptive_writers_count_input_before_reserving_tokens(self):
        config = load_config(Path(__file__).resolve().parents[1] / "configs/project.json")["agent"]
        config.update(adaptive_budget=True, max_calls=None)
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "state.sqlite")
            client = Gemini(config, store, directory)
            with patch.object(client, "_count_input_tokens", return_value=123), \
                 patch.object(client, "_request", return_value=('{"ok":true}', 140)):
                client.call("writer", "x" * 10000, "system")
            with store.connect() as db:
                reserved = db.execute("SELECT reserved FROM calls WHERE role='writer'").fetchone()[0]
            self.assertEqual(reserved, 123 + config["max_output_tokens"])
            self.assertEqual(store.call_stats()["auxiliary_calls"], 1)

    def test_missing_provider_total_keeps_reservation_even_with_components(self):
        config = load_config(Path(__file__).resolve().parents[1] / "configs/project.json")["agent"]
        config.update(adaptive_budget=True, max_calls=None)
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "state.sqlite")
            client = Gemini(config, store, directory)
            usage = {"total_tokens": None, "input_tokens": 123, "output_tokens": 20, "reasoning_tokens": 8}
            with patch.object(client, "_count_input_tokens", return_value=123), \
                 patch.object(client, "_request", return_value=('{"ok":true}', usage)):
                self.assertTrue(client.call("writer", "test", "system")["ok"])
            self.assertEqual(store.usage()["tokens"], 123 + config["max_output_tokens"])
            self.assertEqual(Store(Path(directory) / "state.sqlite").call_stats()["unresolved_tokens"],
                             123 + config["max_output_tokens"])

    def test_quota_backoff_reduces_concurrency_and_keeps_telemetry(self):
        config = load_config(Path(__file__).resolve().parents[1] / "configs/project.json")["agent"]
        config.update(adaptive_budget=False, retry_base_seconds=1, retry_max_seconds=2, max_parallel_calls=4)
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "state.sqlite")
            client = Gemini(config, store, directory)
            with patch.object(client, "_request", side_effect=[QuotaError("quota"), ('{"ok":true}', 12)]), \
                 patch("sako.agent.random.uniform", return_value=1.2), patch("sako.agent.time.sleep"):
                client.call("writer", "test", "system")
            self.assertEqual(client.gate.limit, 2)
            pacing = store.get("api_pacing")
            self.assertEqual(pacing["effective_parallel_calls"], 2)
            event = json.loads(next((Path(directory) / "api_events").glob("*-retry_scheduled.json")).read_text())
            self.assertEqual(event["code"], 429)
            self.assertEqual(event["delay_seconds"], 1.2)
            self.assertEqual(store.usage(), {"calls": 2, "tokens": 12})

    def test_cooldown_observes_a_later_failure_before_dispatch(self):
        config = load_config(Path(__file__).resolve().parents[1] / "configs/project.json")["agent"]
        config["adaptive_budget"] = False
        with tempfile.TemporaryDirectory() as directory:
            client = Gemini(config, Store(Path(directory) / "state.sqlite"), directory)
            client.not_before = 30.
            def extend(_):
                client.not_before = 45.
            with patch("sako.agent.time.monotonic", return_value=0.), \
                 patch("sako.agent.time.sleep", side_effect=extend) as sleep:
                client._wait_for_cooldown()
            self.assertEqual([call.args[0] for call in sleep.call_args_list], [30., 15.])

    def test_coordinator_reserves_output_headroom(self):
        config = load_config(Path(__file__).resolve().parents[1] / "configs/project.json")["agent"]
        config["adaptive_budget"] = False
        config.update(coordinator_max_input_tokens=150000, model_context_tokens=150000)
        with tempfile.TemporaryDirectory() as directory:
            client = Gemini(config, Store(Path(directory) / "state.sqlite"), directory)
            with patch.object(client, "_count_input_tokens", return_value=149000), \
                 patch.object(client, "_request") as request:
                with self.assertRaisesRegex(ValueError, "output reserve"):
                    client.call("coordinator", "small", "system")
            request.assert_not_called()

    def test_unknown_coordinator_outcome_keeps_counted_reservation(self):
        config = load_config(Path(__file__).resolve().parents[1] / "configs/project.json")["agent"]
        config["adaptive_budget"] = False
        config.update(coordinator_max_input_tokens=150000, model_context_tokens=1000000)
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "state.sqlite")
            client = Gemini(config, store, directory)
            with patch.object(client, "_count_input_tokens", return_value=140000), \
                 patch.object(client, "_request", side_effect=TimeoutError("unknown outcome")):
                with self.assertRaises(TimeoutError):
                    client.call("coordinator", "small", "system")
            self.assertEqual(store.usage()["tokens"], 140000 + config["max_output_tokens"])

    def test_unknown_provider_context_limit_does_not_guess(self):
        config = load_config(Path(__file__).resolve().parents[1] / "configs/project.json")["agent"]
        config["adaptive_budget"] = True
        config.pop("model_context_tokens", None)
        config.pop("model_output_tokens", None)
        with tempfile.TemporaryDirectory() as directory:
            client = Gemini(config, Store(Path(directory) / "state.sqlite"), directory)
            with patch.object(client, "_get_provider_limits", return_value={"input_token_limit": None}), \
                 patch.object(client, "_count_input_tokens", return_value=30), \
                 patch.object(client, "_request") as request:
                with self.assertRaisesRegex(ValueError, "unavailable"):
                    client.call("coordinator", "small", "system")
            request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
