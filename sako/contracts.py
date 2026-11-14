"""Trusted functional contracts for the supported model adapter."""

from .core import digest


SPECIFICATIONS = {
    "silu": {
        "tensors": ["x"], "scalars": [],
        "functional_source": "def operation(x):\n    return F.silu(x)",
        "output": "Shape and dtype: x.shape and x.dtype.",
        "numerics": "Preserve the original SiLU activation. Do not modify the input."},
    "linear": {
        "tensors": ["x", "weight", "bias (empty means absent)"], "scalars": [],
        "functional_source": "def operation(x, weight, bias):\n    return F.linear(x, weight, bias if bias.numel() else None)",
        "output": "Shape: x.shape[:-1] + (weight.shape[0],). Dtype: x.dtype.",
        "numerics": "Preserve the original FP16 linear operation and optional bias."},
    "embedding": {
        "tensors": ["integer_ids", "weight"], "scalars": [],
        "functional_source": "def operation(integer_ids, weight):\n    return F.embedding(integer_ids, weight)",
        "output": "Shape: integer_ids.shape + (weight.shape[1],). Dtype: weight.dtype.",
        "numerics": "Copy the selected weight rows exactly."},
    "rmsnorm": {
        "tensors": ["x", "weight"], "scalars": ["epsilon"],
        "functional_source": "def operation(x, weight, epsilon):\n    f = x.float()\n    normalized = f * torch.rsqrt(f.square().mean(-1, keepdim=True) + epsilon)\n    return normalized.to(x.dtype) * weight",
        "output": "Shape and dtype: x.shape and x.dtype.",
        "numerics": "Accumulate mean squares in FP32. Round normalized x to FP16 before multiplying by FP16 weight."},
    "mlp": {
        "tensors": ["x", "gate_weight", "up_weight", "down_weight"], "scalars": [],
        "functional_source": "def operation(x, gate_weight, up_weight, down_weight):\n    gate = F.silu(F.linear(x, gate_weight))\n    up = F.linear(x, up_weight)\n    return F.linear(gate * up, down_weight)",
        "output": "Shape: x.shape[:-1] + (down_weight.shape[0],). Dtype: x.dtype.",
        "numerics": "Preserve FP16 rounding at every original operation boundary. Projections have no bias."},
    "attention": {
        "tensors": ["q", "k", "v", "mask (empty means absent)"],
        "scalars": ["causal_flag", "scale"],
        "functional_source": "def operation(q, k, v, mask, causal_flag, scale):\n    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask if mask.numel() else None, is_causal=bool(causal_flag), scale=scale, dropout_p=0.0)",
        "output": "Shape: q.shape[:-1] + (v.shape[-1],). Dtype: q.dtype. Layout: [B,H,S,D].",
        "numerics": "Keys and values already contain repeated groups. Preserve causal masking, scale, and zero dropout."},
}


def operation_contract(task, checks, config):
    specification = SPECIFICATIONS[task["kind"]]
    contract = {"schema_version": 1, "task_id": task["id"], "kind": task["kind"],
                **specification, "imports": ["import torch", "import torch.nn.functional as F"],
                "cases": task["cases"], "scalar_values": task["scalars"],
                "paths": task["paths"], "inputs_mutated": False,
                "atol": config["atol"], "rtol": config["rtol"],
                "reference_owner": "trusted_adapter",
                "validated": bool(checks) and all(check["passed"] for check in checks),
                "validation": {"method": "Compare trusted functional output with the original module on captured inputs.",
                               "checks": checks}}
    contract["fingerprint"] = digest(contract)
    return contract
