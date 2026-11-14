"""Trusted copy diagnostic. Never used as an optimization candidate."""

def control_seed(task):
    """A trusted logical copy checks timing only, including strided inputs."""
    rank = len(task["cases"][0]["shapes"][0]) if task.get("cases") else 1
    parameters = ", ".join(f"D{i}: tl.constexpr, S{i}: tl.constexpr" for i in range(rank))
    offsets = "\n".join(f"    source += (remaining % D{i}) * S{i}\n    remaining = remaining // D{i}"
                        for i in reversed(range(rank)))
    arguments = ", ".join(f"x.shape[{i}], x.stride({i})" for i in range(rank))
    return {"name": "timing_copy_control", "task_id": task["id"], "backend": "triton", "references": ["sako/timing_controls.py"], "strategy": "Calibrate direct and wrapped timing. This is not an optimization candidate.",
        "source": '''import torch
import triton
import triton.language as tl

@triton.jit
def control_copy(X, Y, SIZE: tl.constexpr, PARAMETERS, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    remaining = offset
    source = tl.full((BLOCK,), 0, tl.int64)
OFFSETS
    value = tl.load(X + source, offset < SIZE, 0)
    tl.store(Y + offset, value, offset < SIZE)

def sako_run(tensors, scalars):
    x = tensors[0]
    output = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    control_copy[(triton.cdiv(x.numel(), 256),)](x, output, x.numel(), ARGUMENTS, 256)
    return output
'''.replace("PARAMETERS", parameters).replace("OFFSETS", offsets).replace("ARGUMENTS", arguments)}
