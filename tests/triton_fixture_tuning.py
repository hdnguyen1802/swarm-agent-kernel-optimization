"""Synthetic backend test fixtures. The search does not use these kernels."""
from triton_fixtures import LINEAR_SOURCE, RMS_SOURCE
from sako.timing_controls import control_seed

CONFIGS = (
    (32, 64, 32, 4, 2), (32, 128, 32, 4, 3), (32, 128, 64, 4, 3),
    (64, 64, 32, 4, 3), (64, 128, 32, 4, 3), (64, 128, 64, 4, 3),
    (64, 256, 32, 8, 3), (128, 64, 32, 4, 3), (128, 128, 32, 8, 3),
    (128, 128, 64, 8, 3), (16, 128, 32, 4, 2), (64, 64, 64, 4, 4),
)

def linear_source():
    configs = ",\n        ".join(
        f"triton.Config({{'BM': {m}, 'BN': {n}, 'BK': {k}}}, num_warps={w}, num_stages={s})"
        for m, n, k, w, s in CONFIGS)
    decorator = ("@triton.autotune(configs=[\n        " + configs +
        "\n    ], key=['M', 'N', 'K', 'SW0', 'SW1', 'HAS_BIAS'], warmup=5, rep=20, use_cuda_graph=True)\n")
    source = LINEAR_SOURCE.replace("@triton.jit", decorator + "@triton.jit")
    source = source.replace("gemm[(triton.cdiv(m, 32), triton.cdiv(n, 64))]",
        "gemm[lambda meta: (triton.cdiv(m, meta['BM']), triton.cdiv(n, meta['BN']))]")
    return source.replace("bias.numel() > 0, 32, 64, 32, num_warps=4, num_stages=2)", "bias.numel() > 0)")


def seed(task):
    if task["kind"] not in {"linear", "mlp", "rmsnorm"}:
        return None
    for case in task.get("cases", []):
        shape, strides = case["shapes"][0], case["strides"][0]
        expected = 1
        for extent, stride in zip(reversed(shape), reversed(strides)):
            if extent != 1 and stride != expected:
                return None
            expected *= extent
    return {"name": "autotuned_" + task["kind"] + "_seed", "task_id": task["id"], "backend": "triton",
        "references": ["tests/triton_fixture_tuning.py"],
        "source": RMS_SOURCE if task["kind"] == "rmsnorm" else linear_source() if task["kind"] == "linear" else mlp_source(),
        "strategy": ("Measure 12 bounded tile configurations per matrix shape before acceptance timing. "
                     "Preserve FP16 inputs and FP32 accumulation." if task["kind"] != "rmsnorm" else
                     "Check the RMSNorm control and preserve the required FP16 rounding before weight multiplication.")}


def mlp_source():
    # Keep an accurate parent starting point. Writers can fuse beyond this seed.
    return linear_source().replace("def sako_run(tensors, scalars):", "def linear_part(tensors, scalars):") + '''

@triton.jit
def activation_product(G, U, Y, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    gate = tl.load(G + offset, offset < SIZE, 0).to(tl.float32)
    up = tl.load(U + offset, offset < SIZE, 0).to(tl.float32)
    activation = (gate / (1.0 + tl.exp(-gate))).to(tl.float16).to(tl.float32)
    tl.store(Y + offset, (activation * up).to(tl.float16), offset < SIZE)

def sako_run(tensors, scalars):
    x, gate_weight, up_weight, down_weight = tensors
    bias = torch.empty((0,), dtype=x.dtype, device=x.device)
    gate = linear_part([x, gate_weight, bias], [])
    up = linear_part([x, up_weight, bias], [])
    product = torch.empty_like(gate)
    activation_product[(triton.cdiv(gate.numel(), 256),)](gate, up, product, gate.numel(), 256)
    return linear_part([product, down_weight, bias], [])
'''
