# Third-party sources

The CUTLASS submodule contains NVIDIA code under the BSD-3-Clause license.
Its license remains in `third_party/cutlass/LICENSE.txt`.
The pinned source is [CUTLASS 3.5.1](https://github.com/NVIDIA/cutlass/tree/v3.5.1).

The model adapter calls PyTorch and Hugging Face Transformers.
It follows the attention interface in [Transformers 4.48.3](https://github.com/huggingface/transformers/tree/v4.48.3).
Transformers uses the Apache-2.0 license. Its package retains the license.
SAKO does not copy model weights into this repository.

The project uses the design sources listed in the README. Their reported speed
figures are not SAKO results.
