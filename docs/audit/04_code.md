# 04 — Code correctness and experimental validity audit

Repo: /home/user/fashion-mnist-knn-optimization (HEAD 08a5062). No repo files were edited.

## What I ran
- All 12 .py files parse (`ast.parse`). pyflakes found unused imports `np` (src/evaluate.py:9), `copy` (src/experiments.py:14) and `SoftKNN` (scripts/smoke.py:18), plus dead locals `use_aug` and `state` (src/experiments.py:562 and :566).
- I made a scratch venv (torch 2.15 CPU, sklearn 1.9.1, matplotlib) and ran the code on a **copy** of src/ and scripts/ under the scratchpad.
  - `scripts/smoke.py` passes: SMOKE_OK on CPU, and the IDX download works.
  - `scripts/smoke_kernel.py` fails 2 of 6 checks. It hard-codes `/tmp/opencode/kg-kernel/script.py`, which is not in the repo, and it needs `artifacts/`, which is gitignored.
  - `pip install --dry-run -r requirements.txt` fails: `torch==2.14.1+cu130` is not on PyPI.
- Scratch checks:
  - `selfmatch_check.py` tests teacher self-match and sklearn's distance-weighted predict_proba on its own fit set.
  - `loo_ab.py` A/B-tests training with and without excluding the sample itself, using a modified copy `train_loo.py`.
  - `metrics_check.py` reproduces the classical numbers in paper Tables 1 and 4.
- Scripts and outputs are under `docs/audit/`.

## Test-set leakage verdict (main pipeline)
In `src/` and `scripts/` I found **no hyperparameter chosen directly by test accuracy**. Validation accuracy (6k stratified, seed 42) chooses all of these:
- PCA dim (E2)
- the Optuna (k, metric, weights) search (E3)
- tau (E8) and the inference (k, tau) grids (E12/E16/E19/E23/E26)
- early stopping, in `train.py` against val
- the fusion simplex weights (E11/E15/E17/E22/E24/E25/E27)

PCA is fit on the fit split only (`Ctx.pca_view`, experiments.py:70). Normalisation is /255 for both train and test (data.py:71-72). Test is used only to report.

Remaining caveats:
1. The test number is computed and printed at **every** stage. The ladder was built by hand and re-run (E10/E11 appear twice in the ledger). E27, the "definitive fusion", is in the code but missing from the ledger and the paper. So results may have been selected by test, though the code cannot prove it.
2. Several inference settings are **hard-coded constants with no logged val selection for that model**:
   - (k=9, tau=0.1) is used for every conv seed, the distilled student and the gen-2 teacher.
   - (k=5, tau=0.1) is used for bank-aug.
   - Details are in finding M7.
3. The **notebooks do leak test**. See C2.

---

## CRITICAL

### C1. Many paper table values match no code output; some contradict what the code produces
docs/paper.tex:220-233 (Table 1), :484-514 (Table 4), :265-287 (Table 2), :250.
Running the repo code (scratch `metrics_check.py`, deterministic sklearn, 54k fit / 10k test) gives:

| Paper claim | Code output |
|---|---|
| Euclidean raw k=5: 85.24, F1 0.8510 | 85.24, F1 **0.8518** (ledger E1 also 0.8518) |
| Cosine raw: 85.50 (called best metric) | **85.63** |
| Manhattan raw: 85.15 ("trails marginally") | **85.91**, the *best* raw metric. The paper's conclusion is reversed. |
| PCA d=16 val/test 82.80/82.40 | 84.73/83.42 (d=16 is not in the full-scale sweep at all) |
| PCA d=32 85.10/84.75 | 86.72/85.05 |
| PCA d=64 86.60/86.20 | 87.33/85.63 |
| PCA d=128 86.40/85.95 | 87.17/85.86 |
| Optuna: PCA-64, **k=7, dist. weights**, val 86.85 | ledger E3: **k=5, manhattan, distance**, val 87.82. k=7 euclidean distance gives test 85.77, not 86.43. |
| 64 comps ≈ 86% variance (README 85.3%) | 88.1% |
| Latency raw 1.84 s, PCA 0.26 s, 7.08x | ledger E5: 2.98 s / 0.418 s (7.13x); E1 query_s 3.19 s |

In Table 4, **every learned-model row's Val column is lower than the ledger value, while its Test column matches the ledger exactly**:

| Row | Paper val | Ledger val |
|---|---|---|
| E09 (ledger E6) | 87.40 | 88.22 |
| E11 (E14) | 87.45 | 88.28 |
| E12 (E19) | 88.10 | 89.07 |
| E14 (E21) | 88.50 | 89.72 |
| E15 (E23) | 89.20 | 90.10 |
| E16 (E22) | 89.95 | 90.27 |

The precision/recall/F1 values for E09-E15 are never computed by those stages: E19, E21 and E23 return no macro metrics. No code measures any Soft-KNN latency (0.02-0.29 s).

The paper calls itself "every number traceable to a logged experiment", but roughly half of Table 4 cannot be traced, and some cells contradict the code.

**Fix:** regenerate all tables automatically from `results/results.json` and add stages for raw cosine/Manhattan and for latency.

### C2. The "one-click reproducible" Colab notebook is a different model and protocol, and it tunes on the test set
fashion_knn_colab.ipynb (same code in fashion_knn_clean_pipeline.ipynb); paper.tex:570-581; README §7.

**Test leakage.**
- Cell 4 compares metrics on `X_test_np`.
- Cell 5 sweeps the PCA dims {16,32,64,128} and k ∈ {1,...,15} by `knn.score(X_test_..., y_test_np)`.
- The paper's claimed d=64 / k choices therefore correspond to test-selected values in the notebook.

**Validation leakage.**
- Cell 7 builds `train_aug_dataset = FashionMNIST(train=True)`, i.e. the full 60k, and trains on it with `aug_loader`.
- `val_sub` is a subset of those same 60k images.
- So every validation curve, the "overfitting diagnostic" and the saved `val_acc` are measured on training images.

**Not the paper's method.**
- It uses learnable class **prototypes** (`self.prototypes`, a parametric nearest-class-mean model with BatchNorm, Dropout and GELU), not Soft-KNN over a memory bank.
- The split is a 50k/10k `random_split`, not stratified 54k/6k.
- There is no PCA-Soft-KNN, no ensemble, no distillation, no TTA and no fusion.
- Classical KNN uses only 10k train and 2k test images.
- It cannot reproduce 85.24 → 89.51.

**Other problems.**
- The notebook has no outputs, so it was never executed in the committed form.
- The "activation study" claimed at paper.tex:400-402 is not run anywhere: only `relu` is used in cell 7 and `gelu` in cell 8.
- Its markdown table says "5-Seed" and "Conv + Augmentation" for 87.63; the ledger shows E19 used no augmentation and the paper says 3 seeds.

**Fix:**
- Either make the notebook call `src.experiments` stages, or state honestly that it is a separate demo.
- In any case, move every selection in the notebook to a validation split.
- Build the augmented training set from `train_sub` indices only.

---

## MAJOR

### M1. Soft-KNN training lets each query match itself in the memory bank (label leak inside training)
src/train.py:228 (`bank_x = x_img`), :253/259 (bank = embeddings of all training images), :260-267 (queries are drawn from the same `x_img`); src/softknn.py:50-60.

Without augmentation the query's own anchor sits at distance ≈0, and it is always in its own class's top-k:
- In my check, the self-distance on a fresh bank was ≤ 9.8e-4.
- With a stale bank the distance is small but non-zero.
- With `augmented_bank_copies`, the bank also contains augmented copies of the query (E14).

At test time no self-anchor exists, so train and test behave differently.

Measured on 12k fit / 6k val, 6 epochs (`loo_ab.py`):

| Setting | Final train acc | Best val |
|---|---|---|
| tau=0.5, self kept | 94.4% | 85.87% |
| tau=0.5, self excluded | 85.1% | 85.15% |
| tau=0.1, self kept | 96.5% | 87.42% |
| tau=0.1, self excluded | 88.1% | 87.72% |

**Impact:**
- `train_acc` and `train_loss` in every history are inflated by about 8-10 points. This affects figures 11, 13 and 15 and the paper's / README's "overfitting / learning-curve" readings.
- The gradient partly rewards "be close to your own stale embedding".
- The effect on validation and test is small and mixed (about ±0.5 pt), so the reported test numbers are not invalidated.
- The Soft-KNN objective as trained is not the objective described in the paper's equation (2).

**Fix:** pass `idx` to the logits function and set `dist[i, idx[i]] = +inf` (and for its augmented copies), i.e. leave-one-out NCA style.

### M2. Distillation teachers are computed on training points that are themselves in the bank, and the gen-2 teacher includes the ground-truth labels
src/experiments.py:1005, :1107-1108, :1128-1130; scripts/train_distill.py:353.

`predict_probs(m, bz, ya_t, ctx.xa, ...)` queries the training images against a bank that contains them.

In the scratch check, the teacher's argmax accuracy on train was:
- **94.2%** with self in the bank
- **85.7%** leave-one-out
- **85.9%** on val

So the teacher's soft labels are close to the hard labels plus a self-bias, not the "inter-class similarity structure" the paper describes (paper.tex:434-436).

E26 also adds `KNeighborsClassifier(weights="distance").predict_proba(za)` on its own fit set. That is **exactly one-hot ground truth** for 100% of samples (verified), because a zero distance gets infinite weight. It is weighted 0.1 in the gen-2 teacher.

The gen-2 teacher also uses tau=0.1 (very sharp) where gen-1 used tau=1.0. So the paper's "negative result" explanation (paper.tex:546-549, "128-dim embedding capacity is the binding constraint") is confounded by three changes:
- the teacher temperature
- the self-match
- the label-copy component

The negative result does not support that conclusion.

Also, at tau=1.0 the gen-1 teacher is nearly **uniform**. The mean p(true class) was 0.17 with 10 classes for the linear model, because unit-norm embeddings bound the logits to [-2/tau, 0]. So the KD term behaves mostly like label smoothing.

**Fix:**
- Compute the teacher with leave-one-out (mask the self index), or with k-fold out-of-bank teachers.
- Never use `predict_proba` on the fit set.
- Use the same teacher temperature in both generations.
- Report teacher entropy.

### M3. The E8 tau sweep evaluates every model at tau=0.5, so Figure 10 and the "boundary sharpening" claims are artefacts
src/experiments.py:337-349, together with `_softknn_eval` at :254-257, which uses `getattr(ctx, "best_tau", 0.5)` and is not set until E8 finishes.

Each model is trained with its own tau but scored and entropy-measured at inference tau=0.5 and k=7. The ledger confirms it: mean entropy is about constant (0.927-0.937) for every tau.

The paper claims:
- a "validation sweep over τ ∈ [0.01, 1.0]"
- that "τ < 0.05 saturates gradients"

(paper.tex:314-316). The code only sweeps {0.1, 0.2, 0.3, 0.5, 0.7, 1.0}. There is no 0.01 or 0.05.

**Fix:** pass `tau` through to `_softknn_eval` for E8. E12 does the inference-only sweep correctly, so cite E12 for the sharpening claim.

### M4. The optimizer study in the paper contradicts the code and the ledger
- paper.tex:385-399
- src/train.py:185-186
- src/experiments.py:368-394
- ledger E9

| | Paper | Code / ledger |
|---|---|---|
| SGD | Nesterov, lr 0.01, "≈3× more epochs" | plain momentum 0.9 (no Nesterov), lr 0.1 |
| Winner | Adam family | E9 picks **SGD** (best val 83.85 vs Adam 81.47 / RMSprop 80.83) |

There is no AdamW and no weight decay anywhere in src/. The paper says "all final models use Adam at 3e-4 with clipping", but the bank-aug linear model (largest fusion weight, 0.4) used lr 1e-3 with no clipping (E14).

Also:
- E9 compares `best_val_acc`, taken from the stale-bank training loop at tau=0.5, while `final_val_acc` uses tau=0.1. The two columns are not comparable.
- The learning rates differ by 100x between optimizers, which is not a controlled comparison.

### M5. The ledger is not reproducible: stage skipping ignores scale, run state is not restored, and artefact reuse is silent
src/experiments.py:1247-1250, :172/:431/:600/:710/:957, :841-843, :1013-1032, :911-930.

**Scale is ignored.** `run_stage` skips a stage if any ledger entry has that `id`, regardless of `scale`. A `--scale quick` ledger blocks `--scale full`.

**Cross-session state is lost.** These values exist only in `ctx`: best_pca_dim, best_knn_params, best_k, best_tau, _e6_val, _e7_val, and the tta probabilities. A new session silently falls back to hard-coded defaults. Evidence from the ledger:
- E11 used `knn_params = euclidean/distance`, not the Optuna winner (manhattan).
- The defaults differ between stages: euclidean in E11 versus manhattan in E15, E17 and E22+.

**Artefacts are reused silently.**
- E21 loads `softknn_conv_s*.pt` when it exists. The ledger shows `epochs_run: 0` and wall time 3 s. Those models came from `scripts/train_conv_seed.py` (35 epochs, AMP), not from E21's own config (50 epochs, no AMP), even though the E21 docstring says "3 seeds x 50 epochs".
- E23 likewise reused a Kaggle-trained student (`reused_artifact: true`). Its `teacher_models: 3` counts the local seed files, not the teachers actually used.

**Re-running changes the result.** `_collect_sources` automatically adds `conv_s2024`, `conv_s31415` and `softknn_conv_distill_k5.pt` when they exist. README/scripts say 5 seeds and a 5-teacher student exist. Re-running E22/E24/E25 today would give different sources, weights and test accuracy from the reported 89.51.

**No artefacts are shipped.** `artifacts/` is gitignored, so E10 onward cannot be re-run from the repo.

**Fix:**
- Key the ledger on (id, scale).
- Reload the upstream choices from the ledger.
- Make the fusion sources an explicit list.
- Save the training config into each artefact.
- Make the stage fail rather than silently reuse an artefact.

### M6. Mismatches between the paper and the code (architecture and hyperparameters)

| Paper says | Code / ledger |
|---|---|
| E09 linear Soft-KNN "784→64, τ=0.1" (Table 4) | E6: embed 128, train and eval τ=0.5, k=7 |
| E10 MLP "784→256→64" | hidden 256 → **128** (experiments.py:627) |
| Conv: "AdamW-style, mixed precision, patience 10"; "single best seed reaches 87.63" (paper.tex:412-414) | 87.63 is **E19**: plain Adam, no AMP, patience 8, 30 epochs, and it is *not* one of the three ensemble seeds (87.53/86.92/87.57 from conv_seeds.json, which are 35-epoch AMP runs) |
| MoCo-style bank "halving compute", refresh every 107 steps (paper.tex:318-321) | There is no momentum encoder. The linear models E6/E7/E13/E14 refresh once per epoch; 107 is used only for MLP and conv. |
| "+3.08 over the strongest single learned model" (paper.tex:452) | 89.51−86.43 is measured against the **classical** tuned KNN. Against the distilled student (89.03) the gain is +0.48. |
| Teacher = the seeds whose ensemble scores 88.06 | The 89.03 student (E23 reused artefact = kaggle_distill.json) was trained on Kaggle from **Kaggle** seeds (kaggle_conv_seeds.json: 87.44/87.25/87.54), which are different models |
| Soft-KNN 0.02 s, "short-list candidate pruning" (paper.tex:268-272) | Inference is brute-force `torch.cdist` over all 54k anchors (softknn.py:94). No pruning and no latency measurement exist. |
| Bias-variance k ∈ {1,...,15} (paper.tex:238) | The code sweeps 1..25 (SCALES["full"]). E4's best k=7 is never used downstream. |
| Fig. 4b caption "Test accuracy vs dim" | The plot shows validation accuracy (which is correct). Fix the caption. |
| Negative-results table lists bank aug "+0.03" as a degradation | Not a degradation |
| Reproducibility: "Python 3.11+" | setup_env.sh uses python3.14; README badge says 3.10+ and "PyTorch 2.4"; requirements.txt pins torch 2.14.1 |
| README / src/README: 50k/10k split; functions `load_fashion_mnist`, `get_splits`, `SoftKNNLinear`, `SoftKNNConvBackbone`, `temperature_distance_softmax`; cosine LR schedule; checkpointing in train.py; latency benchmarking in evaluate.py | None of these exist |

### M7. Inference settings hard-coded with no val selection for the model they are applied to
- **(k=9, tau=0.1)** comes from E19's grid on a *different* model (E19 seed-42, 30 epochs). It is applied without re-selection to:
  - all conv seeds (experiments.py:859-861, :917; train_conv_seed.py:286-289)
  - the distilled student in E24/E25/E27 (:1074-1077, :1195-1198, :1226-1229)
  - the gen-2 teacher (:1112, :1122)

  E23's own val grid selected **k=7** for the student, yet the fusions use k=9.
- **(k=5, tau=0.1)** for bank-aug (:937, :1114) comes from E12, which tuned `softknn_base.pt`, not bank-aug.

These are not test-selected in the code, but nothing records why these values were chosen. Re-select (k, tau) per source on val inside each stage and log it.

### M8. Validation used for early stopping is computed against a stale bank
src/train.py:296-298.

`val_probs` uses `bank_z` from the last refresh. For the linear models that is the *start* of the epoch, so the bank was embedded by weights 211 Adam steps older than the query encoder. The best-epoch selection and `best_train_val` therefore score a mismatched model.

This explains the gaps between `best_train_val` and the fresh-bank `val_acc`, for example E20 (0.8378 vs 0.8557) and E9.

**Fix:** re-embed the bank with the current weights before validating.

---

## MINOR

1. **Validation reuse / optimistic val.** The same 6k val split is used for early stopping, the per-source (k, tau) grid, the source inclusion decisions and a simplex search over up to C(16,6)=8008 weight vectors. The reported 90.85% val is therefore biased upward. The test number is still held out, but the paper should not present the val figure as an estimate. Consider a second held-out split or nested CV for the fusion weights.
2. **KL direction docstring.** scripts/train_distill.py:5 says KL(student‖teacher). The code `F.kl_div(log_softmax(student), teacher)` computes KL(teacher‖student), which is the correct Hinton direction and matches the paper. Since T=1, the T² factor and the temperature are no-ops, and the teacher is already a probability vector (not logits/T as the paper's equation implies). If a teacher component is missing, the E26 teacher weights (:1132-1137) are not renormalised, so the soft targets may not sum to 1.
3. **Nondeterminism.** There is no `torch.use_deterministic_algorithms` or cudnn flag in src/. CUDA `cdist`/scatter backward and AMP vary between runs, as seen by seed 42 giving 87.53 locally and 87.44 on Kaggle. Seed 123 early-stopped at epoch 13/14 in both environments (best epoch ≈3), and the script saves no history to check for instability. "Fix the global seed" does not give bit-reproducibility.
4. **Latency methodology** (classical.py:243-279, experiments.py:217-237):
   - The query set is the 6k val set, not "a batch". The y-label says "1k val".
   - `n_jobs` comes from the `KNN_NJOBS` env var (default 4).
   - There is no warm-up, and tree backends get 1 repeat versus 3.
   - Soft-KNN latency is not measured at all; the paper's CPU (sklearn) versus GPU comparison has no code.
5. **E15 dead code and mislabel.** E15 (experiments.py:560-621) computes `use_aug` and `state` and never uses them. It always trains augment=None seeds, but records `"source": "softknn_shift.pt"` in the ledger.
6. **Paper-relevant numbers that diverge between E10 runs.** E10 and E11 have duplicate ledger entries (first source `softknn_aug.pt`, second `softknn_base.pt`). The ledger's "28 experiments" count includes the duplicates.
7. **Linear-only loaders.** E10/E12 (:411-413, :468-470) build `SoftKNN(embed_dim=state["proj.lin.weight"]...)` directly and crash for an MLP or conv artefact. Use `load_softknn`.
8. **E25 docstring** (:1178-1180) says six translations, but `tta_views` gives identity + hflip + four 1-px shifts. The paper is correct here. The hflip view is applied to conv models never trained with flips.
9. **Ledger summary** (:1283) prints `e.get('stage')`, but entries use `id`, so it always prints "?".
10. **Overwritten figure.** Every fusion stage overwrites `figures/12_confusion_matrix.png`, so the figure depends on stage order. The paper's per-class numbers cannot be traced to a logged probability file.
11. **Data download.** data.py:41-51 downloads over HTTP with no checksum; only the header shape is checked.
12. **smoke_kernel.py:**
   - hard-coded `/tmp/opencode/kg-kernel/script.py` (:34)
   - `argv` built but unused in `t_arg_parsing` (:92-101), so only `--help` is tested
   - dead conditional `exp[:-3] if ... else exp[:-3]` (:78)
   - `t_queue_logic` tests a local stub rather than the kernel's queue code
13. **setup_env.sh:**
   - `cd /home/akng/Downloads/Fashion` is hard-coded (:5)
   - it requires `python3.14`
   - `set -u` without `-e`, so a failed install continues
   - torch is installed unpinned, which contradicts requirements.txt
14. **requirements.txt:**
   - `torch==2.14.1+cu130` cannot be installed from PyPI (verified), so the README's `pip install -r requirements.txt` fails
   - the notebooks need `torchvision`, `seaborn` and `jupyter`, which are missing
   - `scipy` is only a transitive dependency
15. **Unused code:** `classical.pca_sweep` and `evaluate.classification_report_text`; the unused imports listed at the top. The notebook has `EPOCHS if "EPOCHS" in locals()`, and its "best model checkpoint" actually saves the last epoch.
16. **Notebook:**
   - The optimizer comparison does not re-seed per candidate, despite the "identical initialization" comment.
   - `list(aug_loader)[:75]` materialises the whole augmented 60k dataset three times.
   - `num_workers=2` with augmentation is not seeded.
17. **Bias-variance train accuracy at k=1 is 1.0 by construction** (self-match). The code comment (classical.py:131) acknowledges this, but any reading of it as "overfitting" in the docs is invalid.

## Items checked and found OK
- IDX parsing and the label/image alignment.
- The stratified split, with the test set untouched.
- PCA and scalers fit on the fit split only.
- `shift()` direction (smoke-tested).
- `per_class_topk_logits` math, including the sign and tau division.
- Soft-target indexing (`soft_targets[idx]` aligned with `x_img` order).
- KL direction in the code.
- AMP: only the embedding runs under autocast; distances and loss are fp32; GradScaler is unscaled before clipping.
- TTA is applied to val and test alike, and never to training queries by accident.
- Augmentation is applied only when `augment` is set, and only in training.
- `predict_probs_tta` divides by the correct number of views (6).
- No test data enters any bank or fit set.
