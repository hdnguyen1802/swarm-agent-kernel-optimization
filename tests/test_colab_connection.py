"""Check connection isolation and preserve the complete evaluation contract."""

import base64
import copy
import hashlib
from pathlib import Path
import struct
import tempfile
import unittest

from sako.core import load_config, read_json
from sako.transport import Worker
from scripts.configure_colab import configure, host_key_record


ROOT = Path(__file__).resolve().parents[1]


def connection():
    algorithm = b"ssh-ed25519"
    blob = struct.pack(">I", len(algorithm)) + algorithm + struct.pack(">I", 32) + bytes(range(32))
    return {"host": "example.test", "port": 12345, "user": "root", "python": "/usr/bin/python3",
            "cuda_home": "/usr/local/cuda", "host_key": "ssh-ed25519 " + base64.b64encode(blob).decode(),
            "host_key_fingerprint": "256 SHA256:" + base64.b64encode(hashlib.sha256(blob).digest()).decode().rstrip("=") + " (ED25519)"}


class ColabConnectionTests(unittest.TestCase):
    def test_new_connection_changes_only_worker_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = root / "private_key"
            identity.write_text("test key; no real credentials")
            base = load_config(ROOT / "configs/project.json")
            before = copy.deepcopy(base)
            output = root / "connection with spaces"
            path = configure(base, connection(), identity, output)
            result = load_config(path)
            self.assertEqual(base, before)
            self.assertEqual({k: v for k, v in result.items() if k != "worker"},
                             {k: v for k, v in base.items() if k != "worker"})
            self.assertEqual(result["worker"]["host"], "root@example.test")
            self.assertEqual(result["worker"]["port"], 12345)
            worker = Worker(result)
            self.assertEqual(worker.ssh()[worker.ssh().index("-p") + 1], "12345")
            self.assertEqual(worker.scp()[worker.scp().index("-P") + 1], "12345")
            expected = f'UserKnownHostsFile="{(output / "known_hosts").as_posix()}"'
            for command in (worker.ssh(), worker.scp()):
                self.assertIn(expected, command)
                self.assertIn("StrictHostKeyChecking=yes", command)
            self.assertTrue((output / "known_hosts").read_text().startswith("[example.test]:12345 ssh-ed25519 "))
            with self.assertRaisesRegex(ValueError, "new connection directory"):
                configure(base, connection(), identity, output)

    def test_invalid_or_changed_host_identity_is_rejected(self):
        for changes in ({"host_key_fingerprint": "SHA256:wrong"}, {"host_key": "ssh-ed25519 AAAA"},
                        {"host": "-option"}, {"host": "user@host"}, {"user": "root;command"},
                        {"port": 0}, {"port": 65536}, {"port": True}, {"port": "22"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                host_key_record({**connection(), **changes})

    def test_port_22_uses_plain_host_record(self):
        self.assertTrue(host_key_record({**connection(), "port": 22}).startswith("example.test ssh-ed25519 "))

    def test_missing_runtime_paths_leave_no_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = root / "key"
            identity.touch()
            output = root / "output"
            for changes in ({"python": None}, {"python": "python3"}, {"cuda_home": "/cuda/../wrong"}):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    configure(load_config(ROOT / "configs/project.json"), {**connection(), **changes}, identity, output)
                self.assertFalse(output.exists())

    def test_transport_rejects_invalid_ports_before_dispatch(self):
        for port in (True, "22", -1, 0, 65536):
            config = load_config(ROOT / "configs/project.json")
            config["worker"]["port"] = port
            with self.subTest(port=port), self.assertRaises(ValueError):
                Worker(config)



if __name__ == "__main__":
    unittest.main()
