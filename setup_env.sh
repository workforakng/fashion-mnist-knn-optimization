#!/usr/bin/env bash
# One-shot environment setup for the Fashion-MNIST KNN project.
# Creates .venv (Python 3.14) and installs torch (CUDA with CPU fallback) + analysis stack.
set -u
cd /home/akng/Downloads/Fashion

echo "[setup] creating venv"
python3.14 -m venv .venv
.venv/bin/pip install -q -U pip wheel setuptools

echo "[setup] installing torch (trying cu130 -> cu128 -> cpu)"
for IDX in cu130 cu128 cpu; do
  if .venv/bin/pip install torch --index-url "https://download.pytorch.org/whl/$IDX"; then
    echo "TORCH_INDEX_OK $IDX"
    break
  fi
  echo "TORCH_INDEX_FAIL $IDX"
done

echo "[setup] installing analysis stack"
.venv/bin/pip install numpy scikit-learn matplotlib pandas optuna joblib tqdm nbformat nbclient ipykernel

echo "[setup] verifying"
.venv/bin/python - <<'EOF'
import torch, sklearn, numpy, optuna
print("TORCH", torch.__version__, "| cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), "| capability:", torch.cuda.get_device_capability(0))
print("sklearn", sklearn.__version__, "| numpy", numpy.__version__, "| optuna", optuna.__version__)
EOF
echo "SETUP_DONE"
