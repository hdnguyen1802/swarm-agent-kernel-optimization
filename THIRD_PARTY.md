# Third-party sources

SAKO generates Triton kernels. The GPU environment installs Triton with PyTorch.
These dependencies retain their own licenses in their installed packages.

The model adapter calls PyTorch and Hugging Face Transformers.
It follows the attention interface in [Transformers 4.48.3](https://github.com/huggingface/transformers/tree/v4.48.3).
Transformers uses the Apache-2.0 license. Its package retains the license.
SAKO does not copy model weights into this repository.

The project uses the design sources listed in the README. Their reported speed
figures are not SAKO results.
