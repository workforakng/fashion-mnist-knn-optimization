# Source Code Architecture (`src/`)

Modular Python implementation of the Fashion-MNIST pipeline: data loading, classical KNN and PCA, Soft-KNN metric learning, evaluation, and the experiment runner that writes `results/results.json`.

Data split used everywhere in `src/`: the official 60,000 training images are split (stratified, seed 42) into **54,000 fit** and **6,000 validation** images; the official **10,000 test** images are only used for reporting. All tuning (PCA dimension, Optuna, τ, inference (k, τ) grids, early stopping, fusion weights) uses the validation split.

---

## Modules

### 1. `data.py` — IDX loading and splits
* `ensure_downloaded()`: downloads the four official Zalando IDX `.gz` files into `<repo>/data/` if missing and checks their header shapes.
* `load_raw()`: returns `(x_train uint8 [60000, 784], y_train, x_test uint8 [10000, 784], y_test)`.
* `load_splits(val_size=6000, seed=42)`: scales pixels to [0, 1] and returns `(x_fit, y_fit, x_val, y_val, x_test, y_test)` with a stratified 54,000 / 6,000 split; the test set is untouched.
* `CLASS_NAMES`: the 10 class names.

### 2. `classical.py` — KNN, PCA, latency, Optuna, bias-variance (scikit-learn)
* `make_knn(k, metric, weights, p, algorithm, n_jobs)`; `n_jobs` defaults to the `KNN_NJOBS` environment variable (4).
* `eval_knn(clf, xa, ya, xv, yv)`: fit + predict with timing.
* `fit_pca(xa, n_components)`: PCA fitted on the fit split only.
* `pca_sweep(...)`: validation accuracy vs. PCA dimension (not used by the runner, which inlines the sweep in E2).
* `latency_benchmark(...)`: fit/query time per (feature space × brute/kd_tree/ball_tree). One untimed warm-up, then brute is timed 3× (median) and tree backends once. Each row records `n_queries` (all queries are predicted in one call), `n_jobs` and `repeats`.
* `optuna_search(...)`: TPE search over k ∈ {3, 5, …, 15}, weights ∈ {uniform, distance}, metric ∈ {euclidean, manhattan, chebyshev, minkowski(p)} on PCA features (the PCA dimension is fixed, not searched).
* `bias_variance_curve(...)`: train vs. validation accuracy over k.

### 3. `softknn.py` — Differentiable Soft-KNN
* `MetricProjector`: linear 784 → 128 projection (optionally 784 → hidden → 128 with ReLU), first layer warm-started from PCA components, output L2-normalised.
* `ConvMetric`: Conv(1→32, 3×3) → ReLU → MaxPool → Conv(32→64, 3×3) → ReLU → MaxPool → Linear(64·7·7 → 128), output L2-normalised.
* `SoftKNN(metric="linear" | "conv", embed_dim=128, hidden_dim=0)`: `embed()`, `logits_from_z(zq, bank_z, bank_y, k, tau, self_idx=None)`.
* `per_class_topk_logits(dist, bank_y, k, n_classes, tau)`: logit for class c = −(mean of the k smallest distances from the query to bank points of class c) / τ. So k is the number of anchors **per class**.
* `mask_self_matches(dist, self_idx)`: leave-one-out mask (see "Self-match" below).
* `embed_chunked`, `load_softknn` (rebuilds linear, MLP or conv models from a saved state dict), `predict_probs(..., exclude_self=False)`, `predict_probs_tta` (average over 6 views: identity, horizontal flip, four 1-px shifts).

There are **no learnable class prototypes**: the "anchors" are the detached embeddings of all 54,000 training images (a memory bank refreshed once per epoch, or every `bank_refresh_steps` steps). Inference is a brute-force `torch.cdist` against the whole bank.

### 4. `augment.py` — Pure-PyTorch augmentation
* `random_hflip`, `shift`, `random_shift` (±N px with zero padding), `random_erasing`, `random_rotate`, the `Augmenter` pipeline, and `tta_views`.
* There is no random crop.

### 5. `train.py` — Training loop
* `make_optimizer(name, params, lr)`: `adam` (plain `torch.optim.Adam`, no weight decay), `rmsprop`, or `sgd` (momentum 0.9, no Nesterov). There is no AdamW.
* `train_softknn(x_fit, y_fit, x_val, y_val, cfg)`: on-the-fly query augmentation, optional augmented bank copies, periodic bank refresh, optional AMP and gradient clipping, optional distillation loss `(1−α)·CE + α·T²·KL(teacher ‖ student)`, early stopping on validation accuracy (computed against the bank from the last refresh). Returns the best-validation weights; nothing is written to disk. There is no learning-rate schedule.

### 6. `evaluate.py` — Metrics and figures
* `full_metrics(y_true, y_pred)`: accuracy, macro precision / recall / F1, confusion matrix.
* `plot_confusion_matrix(...)` (always writes `figures/12_confusion_matrix.png` by default), `save_fig(...)`, `classification_report_text(...)`.
* Latency benchmarking lives in `classical.py`, not here. Soft-KNN latency is not measured anywhere.

### 7. `experiments.py` — Stage runner
* `python -m src.experiments --scale {tiny,quick,full} --stages E1,E2 [--force] [--exclude-self]` or `--all`.
* Stages E0–E17 and E19–E27 (there is no E18). Each stage appends one entry to `results/results.json`.
* Caveats when re-running:
  * A stage is skipped if **any** ledger entry has its ID, whatever its `scale`.
  * Upstream choices (best PCA dim, Optuna parameters, best k/τ) are kept only in memory, so a new session falls back to defaults.
  * E21 and E23 silently reuse artefacts from `artifacts/` if present (the logged entries reused models trained with `scripts/` on Kaggle).
  * The fusion stages (E22, E24, E25, E27) include whatever artefacts exist in `artifacts/`, so their sources can change between runs.
  * `artifacts/` is gitignored, so stages E10 onward cannot be re-run from a fresh clone without retraining.

---

## Self-match in the memory bank (known issue)

During Soft-KNN training the memory bank is built from the same training images that are used as queries. A training query therefore finds its own (slightly stale) embedding at distance ≈ 0, and also its own augmented copies when `augmented_bank_copies > 0`. Effects:

* `train_acc` / `train_loss` in every training history (and the train curve in `figures/11_softknn_curves.png`) are inflated by roughly 8–10 points.
* The loss partly rewards "stay close to your own bank entry"; validation and test numbers are only slightly affected (about ±0.5 pt), because validation and test images are never in the bank.
* Distillation teachers (E23, E26, `scripts/train_distill.py`) are computed on training images that are in their own bank, so the soft targets are biased toward the hard labels. The E26 classical-KNN teacher component (`predict_proba` with distance weights on its own fit set) is exactly one-hot.

**Opt-in fix.** `cfg["exclude_self"] = True` in `train_softknn`, `predict_probs(..., exclude_self=True)`, `python -m src.experiments --exclude-self`, and `--exclude-self` in `scripts/train_conv_seed.py` / `scripts/train_distill.py` mask each query's own bank indices (leave-one-out, as in NCA). This also applies a leave-one-out KNN for the E26 classical teacher. **The default is off**, so the code reproduces the logged results in `results/results.json`, which were all produced with self-match.
