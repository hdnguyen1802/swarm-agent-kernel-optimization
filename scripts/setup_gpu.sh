#!/usr/bin/env bash
set -euo pipefail
python3 -m venv ../.venv
../.venv/bin/python -m pip install --upgrade pip
# Override these for another GPU only after checking the CUDA/PyTorch support matrix.
SAKO_TORCH_SPEC=${SAKO_TORCH_SPEC:-torch==2.5.1}
SAKO_TORCH_INDEX=${SAKO_TORCH_INDEX:-https://download.pytorch.org/whl/cu121}
../.venv/bin/python -m pip install "$SAKO_TORCH_SPEC" --index-url "$SAKO_TORCH_INDEX"
../.venv/bin/python -m pip install -e . 'transformers==4.48.3' 'numpy==1.26.4' 'accelerate==1.3.0' 'sentencepiece==0.2.3' 'protobuf==7.36.2'
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.3}
export PATH="$CUDA_HOME/bin:$PWD/../.venv/bin:$PATH"
export TORCHINDUCTOR_CACHE_DIR="$PWD/../cache/inductor"
export TRITON_CACHE_DIR="$PWD/../cache/triton"
python -c 'from sako.gpu import doctor; print(doctor())'
