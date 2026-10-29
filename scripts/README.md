# Execution Scripts (`scripts/`)

This directory contains standalone execution entry points, fast smoke testing routines, multi-seed training runners, and knowledge distillation engines.

---

## Script Manifest

### 1. `smoke.py` — Fast Local Pre-Flight Verification
* **Purpose:** Validates environment integrity, dataset loading, tensor shapes, gradient computation, and forward passes in under 3 seconds.
* **Usage:**
  ```bash
  python3 scripts/smoke.py
  ```
* **Checks Performed:**
  * Checks dataset loading and split dimensions.
  * Verifies PCA dimensionality reduction.
  * Validates differentiable Soft-KNN forward pass and temperature softmax outputs.
  * Performs a quick 2-epoch mini-batch training loop to confirm gradient flow.

### 2. `smoke_kernel.py` — Remote/Kaggle Pre-Flight Check
* **Purpose:** Minimal standalone smoke-test script optimized for running in headless Kaggle or Google Colab environments before launching multi-hour training runs.

### 3. `train_conv_seed.py` — Multi-Seed Convolutional Backbone Trainer
* **Purpose:** Trains independent convolutional Soft-KNN models across diversified random seeds (e.g., Seeds 42, 7, 123, 2024, 31415).
* **Outputs:** Saves individual model checkpoints (`softknn_conv_s42.pt`, etc.) into `artifacts/` and logs seed variance reduction metrics into `results/conv_seeds.json`.

### 4. `train_distill.py` — Knowledge Distillation Engine
* **Purpose:** Implements teacher-student representation distillation. Transfers softened ensemble distributions from deep convolutional teacher models into a compact student metric head.
* **Outputs:** Produces distilled student weights (`softknn_conv_distill.pt`) and records distillation convergence in `results/distill.json`.
