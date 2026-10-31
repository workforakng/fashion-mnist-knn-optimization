# Execution Scripts (`scripts/`)

Stand-alone entry points: smoke tests, the multi-seed conv trainer and the distillation trainer. Some of the logged runs were executed on Kaggle GPUs with these scripts: the conv seeds in `results/kaggle_conv_seeds.json` and the distilled students in `results/kaggle_distill.json` (the 89.03 % student reused by ledger stages E23–E25) and `results/kaggle_distill_5teacher.json`. `results/conv_seeds.json` holds a separate run of `train_conv_seed.py` (the artefacts reused by ledger stage E21).

---

## Script Manifest

### 1. `smoke.py` — Local pre-flight check
* **Usage:** `python3 scripts/smoke.py` (downloads Fashion-MNIST into `data/` on first run; about 1 minute on CPU).
* **Checks:**
  * Data loading on a small subset of the 54k / 6k / 10k splits.
  * Augmentation ops (shapes, value range, shift direction) and the 6 TTA views.
  * PCA-32 + KNN.
  * `per_class_topk_logits` against a hand-computed example.
  * Leave-one-out masking (`exclude_self=True`) on random data, for both `predict_probs` and `train_softknn` with augmented bank copies.
  * A 2-epoch Soft-KNN training run, then plain and TTA prediction.

### 2. `smoke_kernel.py` — Kaggle bundle pre-push check
* **Usage:** `python3 scripts/smoke_kernel.py`
* Compiles the bundled sources, tests the data-file mapping, `--help` of both training scripts, the teacher glob and the dual-GPU queue logic.
* Two checks need inputs that are not in the repository and are reported as **SKIP** when missing:
  * the generated Kaggle kernel script — pass its path with `KAGGLE_KERNEL_SCRIPT=/path/to/script.py`;
  * seed artefacts in `artifacts/` (gitignored; produced by `train_conv_seed.py`).

### 3. `train_conv_seed.py` — One conv-metric Soft-KNN seed
* **Usage:** `python3 scripts/train_conv_seed.py --seed 42 [--epochs 40] [--exclude-self]`
* Adam lr 3e-4, gradient clipping 1.0, AMP on CUDA, patience 10, bank refresh every 107 steps. Evaluates at fixed (k = 9, τ = 0.1).
* **Outputs:** `artifacts/softknn_conv_s<seed>.pt` and an entry in `results/conv_seeds.json` (`val_acc`, `test_acc`, `best_train_val`, `epochs_run`).

### 4. `train_distill.py` — Conv-seed ensemble → conv student
* **Usage:** `python3 scripts/train_distill.py [--epochs 35] [--alpha 0.5] [--exclude-self]`
* Teacher = mean τ = 1.0 probabilities of **every** `artifacts/softknn_conv_s*.pt` present (3 or 5 seeds). Student loss = (1 − α)·CE + α·KL(teacher ‖ student).
* By default the teacher is evaluated on training images that are in its own memory bank (self-match; see `src/README.md`). `--exclude-self` uses leave-one-out targets.
* **Outputs:** `artifacts/softknn_conv_distill.pt` and `results/distill.json` (the logged Kaggle runs were saved as `results/kaggle_distill.json` and `results/kaggle_distill_5teacher.json`).
