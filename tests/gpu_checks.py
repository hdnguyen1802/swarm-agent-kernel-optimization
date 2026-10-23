"""Run explicitly on the GPU worker. CPU CI does not import this file."""

import torch
from sako.compat import configure_compiler
from sako.core import load_config
from sako.gpu import check_output
from sako.model import clone_structure


def main():
    config = load_config("configs/project.json")
    configure_compiler(config)
    x = torch.randn(8, 3584, device="cuda", dtype=torch.float16)
    w = torch.randn(3584, device="cuda", dtype=torch.float16)
    def f(a, b):
        z = a.float()
        return (z * torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-6)).half() * b
    compiled = torch.compile(f, fullgraph=True)
    torch.testing.assert_close(compiled(x, w), f(x, w), atol=.03, rtol=.02)
    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.ReLU()).cuda().half()
    clone = clone_structure(model)
    assert model[0] is not clone[0]
    assert model[0].weight is clone[0].weight
    probe = torch.randn(4, 8, device="cuda", dtype=torch.float16)
    torch.testing.assert_close(model(probe), clone(probe))
    assert not check_output(torch.full_like(x, float("nan")), x, config)["passed"]
    failure = check_output(torch.ones_like(x), torch.zeros_like(x), config)
    assert failure["mismatched_elements"] == x.numel()
    assert failure["examples"][0] == {"flat_index": 0, "actual": 1., "expected": 0., "allowed_abs_error": float(torch.tensor(config['atol']))}
    print("GPU precision, shared weights, and invalid-output checks passed.")


if __name__ == "__main__":
    main()
