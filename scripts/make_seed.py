"""Create a CUTLASS linear candidate for ABI and integration checks."""

import argparse
from sako.core import read_json, write_json

SOURCE = r'''
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/half.h>

__global__ void bias_kernel(half* output, const half* bias, int64_t count, int width) {
  int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) output[i] = __hadd(output[i], bias[i % width]);
}

torch::Tensor sako_run(std::vector<torch::Tensor> inputs, std::vector<double> scalars) {
  auto x = inputs.at(0).contiguous();
  auto w = inputs.at(1).contiguous();
  auto bias = inputs.at(2).contiguous();
  c10::cuda::CUDAGuard guard(x.device());
  int k = w.size(1), n = w.size(0), m = x.numel() / k;
  auto shape = x.sizes().vec(); shape.back() = n;
  auto output = torch::empty(shape, x.options());
  using H = cutlass::half_t;
  using Gemm = cutlass::gemm::device::Gemm<
    H, cutlass::layout::RowMajor, H, cutlass::layout::ColumnMajor,
    H, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm70>;
  typename Gemm::Arguments args({m,n,k},
    {reinterpret_cast<H*>(x.data_ptr<at::Half>()), k},
    {reinterpret_cast<H*>(w.data_ptr<at::Half>()), k},
    {reinterpret_cast<H*>(output.data_ptr<at::Half>()), n},
    {reinterpret_cast<H*>(output.data_ptr<at::Half>()), n}, {1.0f,0.0f});
  auto stream = at::cuda::getCurrentCUDAStream();
  Gemm gemm;
  auto status = gemm(args, nullptr, stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS GEMM failed");
  if (bias.numel()) {
    int64_t count = output.numel();
    bias_kernel<<<(count+255)/256,256,0,stream>>>(
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(bias.data_ptr<at::Half>()),count,n);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="runs/profile/tasks.json")
    parser.add_argument("--output", default="runs/validation/seed.json")
    args = parser.parse_args()
    tasks = read_json(args.profile)["tasks"]
    # A real projection group, not an artificial model or a speed claim.
    task = next(t for t in tasks if t["kind"] == "linear" and any(p.endswith("q_proj") for p in t["paths"]))
    write_json(args.output, {"name": "cutlass_gemm_seed", "task_id": task["id"],
        "strategy": "Use CUTLASS tensor operations and a separate bias kernel.",
        "uses_cutlass": True, "source": SOURCE,
        "references": ["include/cutlass/gemm/device/gemm.h"]})


if __name__ == "__main__":
    main()
