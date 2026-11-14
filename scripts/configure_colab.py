"""Make a worker config from the Colab notebook output. Keep model tests fixed."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import struct

from sako.core import load_config, read_json, write_json
from sako.transport import Worker


def host_key_record(connection):
    host, user, port = (connection.get(key) for key in ("host", "user", "port"))
    if (not isinstance(host, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", host)
            or not isinstance(user, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", user)):
        raise ValueError("Invalid Colab SSH host or user.")
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("Use an SSH port from 1 to 65535.")
    parts = str(connection.get("host_key", "")).split()
    if len(parts) < 2 or parts[0] != "ssh-ed25519":
        raise ValueError("Supply the Ed25519 host public key from the Colab notebook.")
    try:
        blob = base64.b64decode(parts[1], validate=True)
    except ValueError as error:
        raise ValueError("Invalid SSH host public key.") from error
    algorithm = b"ssh-ed25519"
    prefix = struct.pack(">I", len(algorithm)) + algorithm + struct.pack(">I", 32)
    if not blob.startswith(prefix) or len(blob) != len(prefix) + 32:
        raise ValueError("Invalid Ed25519 host public key data.")
    fingerprint = "SHA256:" + base64.b64encode(hashlib.sha256(blob).digest()).decode().rstrip("=")
    supplied = str(connection.get("host_key_fingerprint", "")).split()
    if fingerprint not in supplied:
        raise ValueError("The host public key does not match the notebook fingerprint.")
    address = host if port == 22 else f"[{host}]:{port}"
    return f"{address} {parts[0]} {parts[1]}\n"


def absolute_remote_path(value, label):
    if not isinstance(value, str) or not value.startswith("/") or "\x00" in value or "\n" in value or "\r" in value:
        raise ValueError(f"Supply the absolute {label} path reported by Colab.")
    if ".." in PurePosixPath(value).parts:
        raise ValueError(f"Invalid {label} path.")
    return value


def configure(base, connection, identity, output_dir):
    """Trust the key supplied through Colab. Do not learn a key from the network."""
    record = host_key_record(connection)
    identity = Path(identity).expanduser().resolve()
    if not identity.is_file():
        raise ValueError("The controller SSH private key is missing.")
    output_dir = Path(output_dir).resolve()
    config_path = output_dir / "config.json"
    known_hosts = output_dir / "known_hosts"
    if config_path.exists() or known_hosts.exists():
        raise ValueError("Use a new connection directory. Existing connection files stay fixed.")
    python = absolute_remote_path(connection.get("python"), "Python")
    cuda = absolute_remote_path(connection.get("cuda_home"), "CUDA toolkit")
    config = copy.deepcopy(base)
    config["worker"].update(transport="ssh", host=f"{connection['user']}@{connection['host']}",
                            port=connection["port"], identity=str(identity), known_hosts=str(known_hosts),
                            root="sako-colab/project", python=python, cuda_home=cuda, gpu=0)
    Worker(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    known_hosts.write_text(record, encoding="utf-8")
    write_json(config_path, config)
    return config_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connection", required=True, help="JSON printed by the Colab SSH notebook.")
    parser.add_argument("--config", default="configs/project.json")
    parser.add_argument("--identity", default=".local/colab_ssh/id_ed25519")
    parser.add_argument("--output-dir", required=True, help="A new directory under .local/colab-connections/.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output_dir).resolve()
    output.relative_to((root / ".local" / "colab-connections").resolve())
    path = configure(load_config(args.config), read_json(args.connection), args.identity, output)
    print("Saved Colab worker config: " + json.dumps(str(path)))
    print("Use a new run and profile. Run the CUDA output check before model tests.")


if __name__ == "__main__":
    main()
