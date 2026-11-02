#!/usr/bin/env bash
# One-shot environment setup for the Fashion-MNIST KNN project.
# Creates .venv next to this script (Python >= 3.10) and installs torch
# (CUDA wheel if available, CPU fallback) plus everything in requirements.txt.
#
# Usage:  bash setup_env.sh            (uses python3)
#         PYTHON=python3.12 bash setup_env.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="${PYTHON:-python3}"
if ! "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "[setup] ERROR: $PYTHON is $("$PYTHON" -V 2>&1); Python >= 3.10 is required." >&2
  echo "[setup] Set PYTHON=/path/to/python3.10+ and re-run." >&2
  exit 1
fi

echo "[setup] creating venv with $("$PYTHON" -V 2>&1)"
"$PYTHON" -m venv .venv
.venv/bin/pip install -q -U pip wheel setuptools

echo "[setup] installing torch (trying cu128 -> cu126 -> cpu wheel indexes)"
TORCH_OK=0
for IDX in cu128 cu126 cpu; do
  if .venv/bin/pip install "torch>=2.3" "torchvision>=0.18" \
      --index-url "https://download.pytorch.org/whl/$IDX"; then
    echo "TORCH_INDEX_OK $IDX"
    TORCH_OK=1
    break
  fi
  echo "TORCH_INDEX_FAIL $IDX"
done
if [ "$TORCH_OK" -ne 1 ]; then
  echo "[setup] falling back to the default PyPI torch wheel"
fi

echo "[setup] installing requirements.txt"
.venv/bin/pip install -r requirements.txt

echo "[setup] verifying"
.venv/bin/python - <<'EOF'
import torch, sklearn, numpy, optuna
print("TORCH", torch.__version__, "| cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), "| capability:", torch.cuda.get_device_capability(0))
print("sklearn", sklearn.__version__, "| numpy", numpy.__version__, "| optuna", optuna.__version__)
EOF
echo "SETUP_DONE"
