"""Run the project installer inside the assigned Colab session."""

import json
from pathlib import Path
import subprocess


def main():
    project = Path.home() / "sako-colab/project"
    result = subprocess.run(["bash", "scripts/setup_colab.sh"], cwd=project,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            encoding="utf-8", errors="replace")
    print("SAKO_INSTALL=" + json.dumps({"exit_code": result.returncode, "log": result.stdout}), flush=True)


if __name__ == "__main__":
    main()
