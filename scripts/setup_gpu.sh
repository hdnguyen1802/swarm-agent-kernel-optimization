#!/usr/bin/env bash
set -euo pipefail
python3 -m venv ../.venv
../.venv/bin/python -m pip install --upgrade pip
../.venv/bin/python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
../.venv/bin/python -m pip install -e '.[gpu]'
export CUDA_HOME=/usr/local/cuda-12.3
export PATH="$CUDA_HOME/bin:$PWD/../.venv/bin:$PATH"
export TORCHINDUCTOR_CACHE_DIR="$PWD/../cache/inductor"
export TRITON_CACHE_DIR="$PWD/../cache/triton"
python -c 'from sako.gpu import doctor; print(doctor())'
