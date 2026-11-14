"""Check CUDA execution before model setup. Save every attempt."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
from pathlib import Path
import platform
import re
import sys
import uuid

from sako.core import write_json
from sako.process import CommandTimeout, run_command


SOURCE = """import torch
from sako.gpu import compile_candidate
from sako.timing_controls import control_seed
candidate = control_seed({"id": "preflight", "cases": [{"shapes": [[4096]]}]})
x = torch.arange(4096, device="cuda", dtype=torch.float16)
y = compile_candidate(candidate)([x], [])
torch.cuda.synchronize()
torch.testing.assert_close(y, x, atol=0., rtol=0.)
print("Triton output check passed.")
"""



def preflight(output_root):
    output = Path(output_root) / (datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    result = {"status": "not_checked", "model_gain": "not_tested",
              "python": sys.version, "platform": platform.platform(), "output_dir": str(output)}

    def finish(status):
        result["status"] = status
        write_json(output / "summary.json", result)
        return result

    def command(name, arguments, timeout=30):
        write_json(output / (name + ".command.json"), {"command": arguments, "timeout_seconds": timeout})
        try:
            completed = run_command(arguments, timeout=timeout, log_path=output / (name + ".log"))
        except (OSError, CommandTimeout) as error:
            write_json(output / (name + ".error.json"), {"error": str(error),
                       "stdout": getattr(error, "stdout", ""), "stderr": getattr(error, "stderr", "")})
            return None
        write_json(output / (name + ".result.json"), {"returncode": completed.returncode,
                   "stdout": completed.stdout, "stderr": completed.stderr})
        return completed

    device = command("device", ["nvidia-smi", "-i", "0", "--query-gpu=name,uuid,memory.total,driver_version,compute_cap",
                                "--format=csv,noheader,nounits"])
    if device is None or device.returncode:
        return finish("device_query_failed")
    rows = list(csv.reader(io.StringIO(device.stdout)))
    if len(rows) != 1 or len(rows[0]) != 5 or not re.fullmatch(r"\d+\.\d+", rows[0][-1].strip()):
        return finish("device_query_invalid")
    name, identity, memory, driver, capability = [value.strip() for value in rows[0]]
    target = "sm_" + capability.replace(".", "")
    result.update(gpu=name, uuid=identity, memory_mib=memory, driver=driver, target=target)
    source = output / "probe.py"
    source.write_text(SOURCE, encoding="utf-8")
    checked = command("triton-check", [sys.executable, str(source)], timeout=120)
    if checked is None or checked.returncode:
        return finish("triton_check_failed")
    result["triton_check"] = "passed"
    return finish("passed")


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/colab-preflight")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    output.relative_to((root / "runs").resolve())
    result = preflight(output)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
