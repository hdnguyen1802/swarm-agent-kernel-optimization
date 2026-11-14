"""Collect worker evidence through the authenticated Colab notebook."""

import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


def main():
    project = Path.home() / "sako-colab/project"
    output = Path("/content/sako-gpu-evidence.tar.gz")
    if (project / "runs").is_dir():
        subprocess.run(["tar", "-czf", str(output), "--exclude=*.pt", "--exclude=*.safetensors",
                        "-C", str(project), "runs"], check=True)
    else:
        with tarfile.open(output, "w:gz"):
            pass
    digest = hashlib.sha256()
    with output.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    print("SAKO_ARCHIVE=" + json.dumps({"sha256": digest.hexdigest(), "bytes": output.stat().st_size}))


if __name__ == "__main__":
    main()
