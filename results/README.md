# Experiment Logs (`results/`)

Machine-readable JSON logs. All splits are 54,000 fit / 6,000 validation (stratified, seed 42) / 10,000 test.

---

## File Manifest

### 1. `results.json` — Experiment ledger
* Written by `src/experiments.py`: one entry per stage run, 28 entries in total.
* Stage IDs run from **E0 to E26**:
  * **E18 does not exist.** The numbering skips from E17 to E19.
  * **E10 (TTA) and E11 (TTA + KNN fusion) appear twice.** Both runs were kept; they differ in the source model:
    * the first E10 used `softknn_aug.pt` (and its E11 fused that model);
    * the second E10 used `softknn_base.pt`.
  * So the 28 entries cover 26 distinct stages.
  * E27 exists in the code but has no ledger entry.
* **Common fields** (not every stage has every field):
  * `id`, `name`, `scale`;
  * `val_acc`: accuracy on the 6,000-image validation split;
  * `test_acc`: accuracy on the 10,000 test images;
  * `test_f1_macro`, `precision_macro`, `recall_macro`, `config`, `history`;
  * `wall_s`, `timestamp`;
  * for fusions: `sources`, `weights`.
* **Example: E5.** E5 stores latency rows. Brute-force query time over all 6,000 validation queries is 2.98 s for raw-784 and 0.418 s for PCA-64, a 7.12× speed-up.
* **Mislabelled field:** the E15 entry says `"source": "softknn_shift.pt"`, but E15 retrains its three seeds with the E6 base config (no augmentation). The code now logs the correct label; the existing entry is left unchanged.
* **Reused artefacts:**
  * E21 did not train anything (`epochs_run: 0`): it reused conv-seed artefacts trained with `scripts/train_conv_seed.py` (35 epochs, AMP), whose numbers are those in `conv_seeds.json`.
  * E23 reused the Kaggle-trained distilled student (`reused_artifact: true`, same numbers as `kaggle_distill.json`).

### 2. `conv_seeds.json` — Conv-metric seeds (the run reused by E21)
* For seeds 42, 7 and 123, it gives `val_acc`, `test_acc`, `best_train_val` and `epochs_run` only.
* Test accuracies are 87.53 %, 86.92 % and 87.57 %.
* It contains no precision or variance statistics.

### 3. `kaggle_conv_seeds.json` — Conv-metric seeds (Kaggle GPU run)
* It has the same fields and seeds as `conv_seeds.json`, but from a separate Kaggle run.
* Its test accuracies are different: 87.44 %, 87.25 % and 87.54 %.
* AMP and non-deterministic CUDA kernels mean the same seed does not give identical results.

### 4. `kaggle_distill.json` — 3-teacher distillation (Kaggle GPU run)
* The student was trained by `scripts/train_distill.py` from the 3 Kaggle seeds.
* Results: val 90.12 %, test 89.03 % (k = 7, τ = 0.1).
* This is the student used in E23–E25.

### 5. `kaggle_distill_5teacher.json` — 5-teacher distillation (Kaggle GPU run)
* The same script with 5 seed teachers.
* Results: val 90.08 %, test 89.01 %.
* It is not used by any ledger stage.
