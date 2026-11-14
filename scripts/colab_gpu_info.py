"""Read the GPU and the public SSH host identity inside the allocated runtime."""

import json
from pathlib import Path
import subprocess


def main():
    def output(*command):
        return subprocess.check_output(command, text=True).strip()

    public_key = Path("/etc/ssh/ssh_host_ed25519_key.pub")
    if not public_key.exists():
        subprocess.run(["ssh-keygen", "-A"], check=True)
    gpu, memory = output("nvidia-smi", "-i", "0", "--query-gpu=name,memory.total",
                         "--format=csv,noheader,nounits").rsplit(",", 1)
    print("SAKO_GPU=" + json.dumps({
        "gpu": gpu.strip(), "memory_mib": int(memory), "host_key": public_key.read_text().strip(),
        "host_key_fingerprint": output("ssh-keygen", "-E", "sha256", "-lf", str(public_key)),
    }))


if __name__ == "__main__":
    main()
