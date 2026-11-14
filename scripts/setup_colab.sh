#!/usr/bin/env bash
# Run in the source checkout on a fresh Colab session.
set -euo pipefail
export PIP_DISABLE_PIP_VERSION_CHECK=1
python3 -m pip install --quiet --target /content/sako-installer uv
export PYTHONPATH="/content/sako-installer"
python3 -m uv venv --python 3.12 /content/sako-venv
python3 -m uv pip install --python /content/sako-venv/bin/python '.[gpu]'
