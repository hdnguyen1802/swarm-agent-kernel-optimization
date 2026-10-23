import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Lock
from unittest.mock import patch

from sako.agent import Gemini
from sako.core import load_config
from sako.store import Store


class QuotaError(RuntimeError):
    code = 429


class AgentLimits(unittest.TestCase):
    def test_retry_counts_each_request(self):
        config = load_config(Path(__file__).resolve().parents[1] / "configs/project.json")["agent"]
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
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "state.sqlite")
            client = Gemini(config, store, directory)
            with patch.object(client, "_request", return_value=("invalid JSON", 17)):
                with self.assertRaises(ValueError):
                    client.call("writer", "test", "system")
            self.assertEqual(store.usage(), {"calls": 1, "tokens": 17})

    def test_eight_writers_obey_two_call_limit(self):
        config = load_config(Path(__file__).resolve().parents[1] / "configs/project.json")["agent"]
        with tempfile.TemporaryDirectory() as directory:
            client = Gemini(config, Store(Path(directory) / "state.sqlite"), directory)
            state = {"active": 0, "maximum": 0}
            lock = Lock()
            def request(*args):
                with lock:
                    state["active"] += 1
                    state["maximum"] = max(state["active"], state["maximum"])
                Event().wait(.03)
                with lock:
                    state["active"] -= 1
                return '{"ok":true}', 10
            with patch.object(client, "_request", side_effect=request), ThreadPoolExecutor(max_workers=8) as pool:
                result = list(pool.map(lambda n: client.call(str(n), "test", "system"), range(8)))
            self.assertEqual(len(result), 8)
            self.assertEqual(state["maximum"], 2)


if __name__ == "__main__":
    unittest.main()
