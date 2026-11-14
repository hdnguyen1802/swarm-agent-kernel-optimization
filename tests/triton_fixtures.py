"""Backend test fixtures only. Evaluator checks, not these examples, grant acceptance."""

RMS_SOURCE = '''import torch
import triton
import triton.language as tl

@triton.jit
def rms(X, W, Y, D: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    value = tl.load(X + row * D + col, col < D, 0).to(tl.float32)
    weight = tl.load(W + col, col < D, 0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(value * value, axis=0) / D + EPS)
    normalized = (value * inv).to(tl.float16).to(tl.float32)
    tl.store(Y + row * D + col, (normalized * weight).to(tl.float16), col < D)

def sako_run(tensors, scalars):
    x, weight = tensors
    assert x.is_contiguous() and weight.is_contiguous()
    d = x.shape[-1]
    output = torch.empty_like(x)
    rms[(x.numel() // d,)](x, weight, output, d, scalars[0], triton.next_power_of_2(d), num_warps=4)
    return output
'''

LINEAR_SOURCE = '''import torch
import triton
import triton.language as tl

@triton.jit
def gemm(X, W, BIAS, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
         SW0: tl.constexpr, SW1: tl.constexpr, HAS_BIAS: tl.constexpr,
         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for block in range(tl.cdiv(K, BK)):
        kk = block * BK + ks
        a = tl.load(X + rows[:, None] * K + kk[None, :], (rows[:, None] < M) & (kk[None, :] < K), 0)
        b = tl.load(W + cols[None, :] * SW0 + kk[:, None] * SW1, (cols[None, :] < N) & (kk[:, None] < K), 0)
        acc += tl.dot(a, b, allow_tf32=False)
    if HAS_BIAS:
        acc += tl.load(BIAS + cols, cols < N, 0)[None, :].to(tl.float32)
    tl.store(Y + rows[:, None] * N + cols[None, :], acc.to(tl.float16), (rows[:, None] < M) & (cols[None, :] < N))

def sako_run(tensors, scalars):
    x, weight, bias = tensors
    assert x.is_contiguous()
    m = x.numel() // x.shape[-1]
    n, k = weight.shape
    output = torch.empty((*x.shape[:-1], n), device=x.device, dtype=x.dtype)
    gemm[(triton.cdiv(m, 32), triton.cdiv(n, 64))](x, weight, bias, output, m, n, k,
        weight.stride(0), weight.stride(1), bias.numel() > 0, 32, 64, 32, num_warps=4, num_stages=2)
    return output
'''
