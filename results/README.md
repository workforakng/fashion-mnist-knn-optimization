# Empirical Results & Experiment Logs (`results/`)

This directory houses complete machine-readable JSON logs for all 28 experiment iterations, seed evaluations, and distillation benchmarks conducted throughout the project.

---

## File Manifest

### 1. `results.json` — Master Experiment Ledger
* **Description:** The comprehensive record of all 28 sequential experiment iterations (E01 to E28).
* **Fields per Experiment:**
  * `experiment_id` / `name`: Identifier of the architecture or trial.
  * `val_acc`: Classification accuracy on the held-out 10,000-sample validation split.
  * `test_acc`: Unbiased classification accuracy on the 10,000-sample test benchmark.
  * `precision_macro`, `recall_macro`, `test_f1_macro`: Macro-averaged performance metrics.
  * `wall_s`: Total execution / wall-clock time in seconds.
  * `timestamp`: Date and time of run completion.
  * `sources` & `weights`: Model components and calibrated mixture coefficients for ensemble experiments (e.g. E24, E25).

### 2. `conv_seeds.json` — Multi-Seed Variance Benchmark
* **Description:** Records accuracy, precision, and variance reduction metrics across independent training seeds (Seeds 42, 7, 123) for the convolutional Soft-KNN backbone.

### 3. `kaggle_conv_seeds.json` — Cloud Multi-Seed Execution Logs
* **Description:** Logs of multi-seed convolutional backbone training executed remotely on Kaggle GPU instances.

### 4. `kaggle_distill.json` — Single-Teacher Distillation Record
* **Description:** Loss convergence, temperature parameter settings, and student test accuracy for knowledge distillation experiments on Kaggle.

### 5. `kaggle_distill_5teacher.json` — Multi-Teacher Ensemble Distillation
* **Description:** Logs tracking knowledge transfer from an ensemble of 5 distinct teacher models into a unified student metric head.
