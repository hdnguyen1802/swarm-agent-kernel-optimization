"""Run explicitly on the GPU. Check the batched exact state comparison."""

import torch
from sako.model_session import ModelSession
from sako.worker import gpu_lock


def main():
    with gpu_lock(), torch.inference_mode():
        model = torch.nn.Linear(16, 16, bias=False).cuda().half()
        session = ModelSession()
        session.begin()
        session.model = model
        expected = torch.empty(model.weight.shape, dtype=model.weight.dtype, device="cpu", pin_memory=True)
        expected.copy_(model.weight, non_blocking=True)
        torch.cuda.synchronize()
        session.shadow = [("weight", expected)]
        assert session.verify()
        model.weight[0, 0] += 1
        assert not session.verify()
        assert session.retire
    print("Exact GPU state check accepts unchanged weights and rejects a native-value change.")


if __name__ == "__main__":
    main()
