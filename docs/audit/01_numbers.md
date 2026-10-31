# Audit 01: Numerical consistency of docs/paper.tex

Scope: every number in docs/paper.tex (the paper_member2-4.tex copies differ only at lines 96-98, so every finding applies to all four PDFs). The extracted PDF text has the same numbers as the .tex, so the compiled PDF is up to date.

Sources checked:
- results/results.json (the ledger; values given as id.key)
- results/conv_seeds.json, kaggle_*.json
- the README files
- src/*.py, scripts/*.py
- both notebooks
- the figure PNGs themselves, read as images

Ledger fact: results.json has 28 entries. Their ids are E0-E17 and E19-E26. E10 and E11 each appear twice, and E18 and E27 do not exist. The paper's table ids E01-E18 do not match the ledger ids.

Neither notebook has any saved output. No cell output exists, so no number can be traced to a notebook.

---------------------------------------------------------------------
## CRITICAL

C1. Most of Table 3 (tab:ladder, lines 494-511) cannot be traced to the ledger or is contradicted by it. Only the test accuracies of the learned models and the E24/E25 rows match. Every "Val" value in rows E04-E16 contradicts the ledger:

| Paper row | Paper Val | Ledger val_acc | Ledger key |
|---|---|---|---|
| E04 PCA d=16 | 82.80 | not run: the sweep dims are 10,20,32,48,64,80,96,128 | E2.rows |
| E05 PCA d=32 | 85.10 | 86.72 | E2.rows[dim=32] |
| E06 PCA d=64 | 86.60 | 87.33 | E2.rows[dim=64] |
| E07 PCA d=128 | 86.40 | 87.17 | E2.rows[dim=128] |
| E08 Optuna | 86.85 | 87.82 | E3.val_acc |
| E09 Soft-KNN linear | 87.40 | 88.22 | E6.val_acc |
| E10 MLP | 86.20 | 86.92 | E16.val_acc |
| E11 bank aug | 87.45 | 88.28 | E14.val_acc |
| E12 conv | 88.10 | 89.07 | E19.val_acc |
| E13 conv shift | 85.10 | 85.57 | E20.val_acc |
| E14 3-seed ensemble | 88.50 | 89.72 | E21.val_acc |
| E15 distilled student | 89.20 | 90.10 | E23.val_acc (kaggle_distill.json 90.12) |
| E16 fusion (E22) | 89.95 | 90.27 | E22.val_acc |

   - Rows E01-E03 show Val as "---", but E1.val_acc = 86.13 exists.
   - Test accuracies for E04-E07 (82.40/84.75/86.20/85.95) are not in the ledger. E2 logs only validation accuracy, and no PCA test run exists.
   - Rows E02 and E03 (Manhattan 85.15, Cosine 85.50) have no ledger entry at all. The notebook cell that would compute them runs on a 10k/2k subset and has no output.
   - Every latency in the column (1.84, 3.12, 1.95, 0.08, 0.14, 0.26, 0.48, 0.25, 0.02, 0.03, 0.04, 0.05, 0.05, 0.12, 0.05, 0.21, 0.28, 0.29 s) cannot be traced. The only logged latencies are E1.query_s = 3.185 s and E5 (raw brute 2.979 s, pca-64 brute 0.418 s). The ledger has no Soft-KNN or fusion latency anywhere.
   - Fix: regenerate the table from results.json, or remove the cells that have no logged value.

C2. Macro precision, recall and F1 in Table 3 and Table 1 (lines 228-230, 494-511) mostly cannot be traced, and where they can be checked they are mostly wrong.

| Row | Paper | Ledger | Ledger key |
|---|---|---|---|
| E01 F1 | 0.8510 | 0.8518 | E1.test_f1_macro |
| E08 F1 | 0.8631 | 0.8644 | E3.test_f1_macro |
| E11 F1 | 0.8702 | 0.8703 | E14 |
| E16 Prec | 0.8885 | 0.8863 | E22.precision_macro |
| E16 F1 | 0.8860 | 0.8858 | E22 |

   - E09 F1 0.8698 matches. E17 and E18 match.
   - The ledger has no precision for E01-E15, and no F1 for E02-E05, E07, E10, E12-E15. Those values cannot be traced.
   - Recall in each row equals test accuracy. That is mathematically expected because the test set has 1000 images per class, so recall is consistent.
   - The Macro F1 column of Table 1 (0.8510/0.8501/0.8532) inherits the same problem.
   - Fix: use the ledger values, or "---" where no value was logged.

C3. Latency and speed-up claims (abstract line 133, lines 268-271, Table 2 lines 282-284, Fig. 1 line 350, conclusion line 557) are not supported by the logged measurements.
   - Paper: "7.08x (1.84 s -> 0.26 s)" and "0.02 s ... 92x speed-up".
   - Ledger E5 (brute, median of 3, 6,000 validation queries): raw 2.979 s, PCA-64 0.418 s, a 7.12x speed-up. E1.query_s = 3.185 s.
   - README.md line 25/43 and figures/README.md give 7.14x. The README line 305 gives 7.08x. Three different values exist in the repo.
   - No Soft-KNN latency is logged anywhere, so 0.02 s and 92x cannot be verified (1.84/0.02 = 92.0 is only arithmetic on unlogged numbers).
   - Table 2 assigns 0.02 s to "Soft-KNN (conv backbone)". Table 3 gives 0.02 s to the linear model (E09) and 0.05 s to the conv model (E12), so the paper contradicts itself.
   - Table 2 also gives 0.26 s for "PCA-64, tuned k" at 86.43 %. Table 3 shows 0.26 s for untuned E06 and 0.25 s for tuned E08.
   - Fix: report 2.98 s -> 0.42 s (7.1x) from E5, say that this is 6,000 validation queries on CPU, and either drop the 92x claim or measure it.

C4. The optimizer study (lines 385-399) contradicts the ledger.
   - Ledger E9 (optimizer_comparison): SGD lr = 0.1 (not 0.01) had best_val 0.8385, Adam 0.8147 and RMSprop 0.8083. Ledger key "best": "sgd".
   - Figure 09_optimizer_compare.png shows SGD reaching the highest validation accuracy after 1 epoch and early-stopping first.
   - The paper says SGD needed "roughly 3x more epochs for equivalent validation accuracy" and that adaptive methods "converged fastest". Both statements are contradicted.
   - eta = 0.01 matches only the notebook cell, which has no output. src/train.py:21 uses momentum=0.9 without Nesterov, but the paper says "Nesterov".
   - Fix: report the E9 numbers (SGD lr 0.1: 83.85 %; Adam 1e-3: 81.47 %; RMSprop 1e-3: 80.83 %, under augmented training) and remove the 3x claim.

C5. Per-class accuracies in the confusion-matrix caption (line 521) are wrong.
   - Caption: "Trouser 98.6 %, Sandal 97.8 %, Bag 97.4 %".
   - figures/12_confusion_matrix.png has a diagonal summing to 8951/10000 = 89.51 %, so it is the E25 matrix. Its values are Trouser 979 (97.9 %), Sandal 966 (96.6 %), Bag 985 (98.5 %).
   - Fix: replace the three values with 97.9 %, 96.6 % and 98.5 %. Bag is the strongest class (98.5 %), followed by Trouser (97.9 %).

---------------------------------------------------------------------
## MAJOR

M1. Line 414: "The single best seed reaches 87.63 % test accuracy". Line 417 lists the seeds as 87.53/86.92/87.57.
   - conv_seeds.json and E21.per_seed give seed 42 = 87.53. The 87.63 comes from E19.test_acc, a different seed-42 run (patience 8, 30 epochs, no AMP).
   - So the "best single seed" is not among the three ensemble seeds. The same mix-up affects Table 3 E12, Table 2 and Fig. 1 Stage 3.
   - Fix: describe 87.63 as the E19 single run, separate from the ensemble seeds, or quote 87.57 as the best ensemble seed.

M2. Line 422: "+0.43 gain over the best single seed".
   - 88.06 - 87.57 = +0.49 against the best ensemble seed. +0.43 only works against the 87.63 run, which is not in the ensemble (E21).
   - Fix: write +0.49 (or +0.72 over the seed mean of 87.34).

M3. Line 452: "+3.08 over the strongest single learned model".
   - 89.51 - 86.43 = 3.08, but 86.43 is the Optuna-tuned classical PCA-KNN (E3), which is not a learned model.
   - The strongest single learned model is the distilled student at 89.03 % (E23), giving +0.48. Against the conv model (87.63 %) the gain is +1.88.
   - Fix: write "+0.48 over the strongest single learned model (distilled student)", or "+3.08 over tuned PCA-KNN".

M4. PCA variance at d = 64 (line 250, "64 components retain approx. 86 % of total variance").
   - figures/03_pca_variance.png shows about 0.875-0.88 at d = 64. The ledger gives var90_dim = 84 (E2), which agrees with the figure.
   - README.md (lines 25, 261) and figures/README.md say 85.3 %, a third value.
   - The Fig. 2 caption (line 259) says "the dotted line marks the d = 64 operating point". The actual figure has a dashed line at 90 % variance and no d = 64 marker.
   - Fix: log the variance ratio. Write "approx. 88 %" and "dashed line = 90 % variance (84 components)".

M5. Fig. 2(b) caption (lines 260-261): "Test accuracy versus retained dimensionality ... too many re-introduce distance concentration".
   - 04_pca_sweep.png and E2 plot validation accuracy, not test accuracy.
   - The curve is flat after d = 64 (87.33 -> 87.17 at d = 128, a 0.16-point change). It shows a plateau, not a decline.
   - The README Fig. 3 description (Euclidean, Manhattan and Cosine curves over d in [16,128]) also does not match the image, which shows one Euclidean curve from d = 10 to 128.
   - Fix: change "Test" to "Validation" and soften the claim.

M6. Line 253 and Table 3 E08: "86.43 % after k/weight tuning (Bayesian optimisation over {k, d, metric, weights})" and "PCA-64, k = 7, dist. weights".
   - Ledger E3: d was fixed at 64 and not searched (classical.py optuna_search has no d parameter). The best parameters were k = 5, weights = distance, metric = manhattan, found in 40 trials.
   - k = 7 is the best k from the separate bias-variance sweep (E4.best_k, val 87.37), not the configuration that scored 86.43 on test.
   - README Fig. 1 says "k = 7, Cosine", which is a third version.
   - Fix: write "PCA-64, k = 5, Manhattan, distance weights (40 TPE trials over k, weights, metric)".

M7. Temperature sweep (lines 314-316): "sweep over tau in [0.01, 1.0]: accuracy peaks sharply at tau = 0.1, while tau < 0.05 saturates gradients".
   - Ledger E8 tested only tau in {0.1, 0.2, 0.3, 0.5, 0.7, 1.0}. Validation accuracy falls steadily from 86.53 % to 81.20 % (also visible in 10_tau_sweep.png).
   - 0.1 is the edge of the grid, so it cannot be a "peak". No tau below 0.1 was ever run, so the claim about tau < 0.05 is unverified.
   - E8 was trained with hflip, shift and erase augmentation for 10 epochs, which the paper does not say.
   - The same 0.01 lower bound appears in README Fig. 6, which also adds "(87.10 %)" at tau = 0.1. The logged value is 86.53 %.
   - Fix: state the real grid and say the best value is at the edge of the grid.

M8. Linear Soft-KNN hyperparameters in Table 3 E09 ("784->64 linear, tau = 0.1"), compared with line 323 ("z in R^128") and ledger E6.config (embed_dim 128, train tau 0.5, k 7).
   - At E6 the evaluation tau was also 0.5, because ctx.best_tau is set only later in E8 (src/experiments.py:257).
   - The paper contradicts itself here (64 vs 128), and the ledger supports neither 64 nor tau = 0.1.
   - Fix: write "784->128 linear (PCA-64 warm start), tau = 0.5".

M9. Table 3 E10: "784->256->64 MLP". Ledger E16.config has hidden_dim 256 and embed_dim 128, i.e. 784->256->128.
   - Fix: change to 784->256->128.

M10. Training settings for the conv backbone (lines 412-413): "AdamW-style settings at 3x10^-4 ... mixed precision, early stopping (patience 10)", attributed to the 87.63 % model.
   - E19 (87.63 %) used Adam with no weight decay (train.py:19), patience 8, 30 epochs and no AMP.
   - Patience 10 and AMP apply only to the conv_seeds and distillation runs (scripts/train_conv_seed.py, train_distill.py).
   - "AdamW" does not appear in src or scripts at all.
   - Fix: write "Adam, lr 3e-4, clip 1.0, patience 8 (E19); patience 10 + AMP for the seed and distillation runs".

M11. Line 399: "all final models in this report use Adam ... at 3x10^-4 with gradient-norm clipping".
   - The bank-augmented linear model, which carries the largest fusion weight (0.4 in E24 and E25), was trained at lr 1e-3 with no clipping (E14.config).
   - The E6 linear model also used 1e-3.
   - Fix: limit the claim to the conv, MLP and distilled models.

M12. Dataset split conflicts across the repo.
   - The paper (lines 190-191, 347; 54,000/6,000, stratified, seed 42) matches src/data.py:63 (val_size = 6000) and the experiments full scale (val = 6000).
   - README.md lines 38, 154, 165 and 303, src/README.md, and both notebooks (random_split [50000, 10000], unstratified) say 50k/10k.
   - results/README.md says the validation split has "10,000 samples".
   - The notebook also normalises with mean/std (0.2860, 0.3530), not to [0, 1].
   - The paper itself is correct here. Fix the README files, and note that the notebook does not use the paper's split.

M13. Reproducibility (lines 142-143, 576-580): "the complete pipeline reproduces end-to-end from a single standalone Google Colab notebook ... eleven executable cells".
   - fashion_knn_colab.ipynb has 11 cells, of which only 8 are code.
   - It uses a 10k-train / 2k-test subset for KNN, a 50k/10k split, a 64-D prototype model with BatchNorm and GELU, AdamW, and 8 epochs. That is not the memory-bank Soft-KNN, ensemble, distillation or fusion pipeline.
   - It cannot produce 85.24, 86.43 or 89.51, and it has no outputs.
   - Its own summary table says "5-Seed variance reduction" for 88.06 % (the paper says 3 seeds) and "ConvNet + Augmentation" for 87.63 % (E19 had augment = null).
   - Fix: say "eight code cells; a simplified demonstration". The numbers come from src/experiments.py.

M14. Table 4 (tab:negative), line 537: "Memory-bank augmentation 87.13 %, +0.03 vs. no-aug", listed under a caption "Techniques that degraded accuracy".
   - The numbers are correct (E14 87.13 - E6 87.10 = +0.03), but this technique did not degrade accuracy.
   - Fix: move the row or reword the caption.

M15. Lines 546-549: gen-2 distillation is described as "feeding the gen-1 student back".
   - E26.teacher_weights = {conv_s42 0.2, bankaug_linear 0.4, knn_classical 0.1, conv_distill 0.3}, i.e. the E24 fusion, not the gen-1 student alone.
   - The claim that 128-D capacity "is the binding constraint" has no supporting experiment (no other embedding dimension was tried). It is repeated at line 564.
   - The -2.03 / "loses 2 points" arithmetic is correct.

M16. Line 481: "the complete JSON ledger contains all 28 experiment entries" and "E01-E18" as logged iterations.
   - There are 28 entries but only 26 unique ids. E10 and E11 are duplicated (two TTA runs and two fusion runs), and E18 is missing.
   - The paper's E01-E18 numbering does not match ledger ids. For example, paper E12 = ledger E19 and paper E15 = ledger E23. Paper E01-E07 have no ledger counterpart.
   - results/README.md claims "E01 to E28".
   - Fix: use ledger ids in Table 3, or add a mapping column.

M17. Omitted results that bear on the claims.
   - TTA (line 442, "reducing input-level variance"): E10 on softknn_base lowered test accuracy from 87.10 to 86.74 and val from 88.22 to 87.85. The other E10 run (on the augmented model) improved test accuracy from 81.68 to 82.60. Only the gain is implied in the paper.
   - kaggle_conv_seeds.json gives different results for the same seeds (87.44 / 87.25 / 87.54), and the paper does not say which seed artifacts were used in the fusion.
   - kaggle_distill_5teacher.json (89.01 %, 5 teachers) is not mentioned, while README.md calls the method "multi-teacher".
   - Fix: add a sentence or footnote for each.

---------------------------------------------------------------------
## MINOR

m1. Line 238: "sweep over k in {1, 3, ..., 15}". Ledger E4.ks is 1-25 (odd). Best k = 7 (val 87.37). Fix: write {1, 3, ..., 25}.

m2. Lines 270-271: "short-list candidate pruning maximally effective". No pruning exists in the code. predict_probs uses full torch.cdist against the whole bank (softknn.py:96).

m3. Line 320: "107 steps approx. one half-epoch at batch 256". 54,000/256 = 210.9 steps, so half an epoch is 105.5. This is approximately right. The 87.10 % linear "Result" (E6) did not use bank_refresh_steps (it refreshes once per epoch). "Halving compute" cannot be verified.

m4. Line 400: the activation study (GELU vs ReLU vs LeakyReLU) has no ledger entry, and the notebook never runs a comparison. It cannot be verified.

m5. Line 580: "under one hour on a free Colab T4" cannot be verified. "Python 3.11+" conflicts with the README badge "3.10+".

m6. Table 3 E16 label "6-source fusion" is correct (E22 has 6 sources), but README.md line 246 says "5-Model".

m7. Table 3 E14 "3-seed ensemble" val 88.50 is wrong (see C1). The colab notebook calls the same 88.06 % a "5-Seed" ensemble.

m8. Figure 05_latency.png's y-label says "1k val". At full scale the validation set has 6,000 queries. This affects how any latency is quoted.

m9. Line 432: distillation loss written as KL(sigma(p_T/T) || sigma(q/T)). The code (train.py:105) applies the KL directly to the teacher probabilities, without re-softmaxing p_T/T. With T = 1.0 the numbers are unaffected, but the formula is not what the code does.

m10. results/README.md says conv_seeds.json "records accuracy, precision, and variance reduction metrics". The file holds only val_acc, test_acc, best_train_val and epochs_run.

---------------------------------------------------------------------
## VERIFIED CORRECT (one line each)

- Title and abstract 85.24 % / 85.2 %: E1.test_acc 0.8524.
- Title and abstract 89.51 % / 89.5 %: E25.test_acc 0.8951.
- 90.85 % validation (lines 470, 511, 560): E25.val_acc 0.9085.
- E25 Prec 0.8945, Rec 0.8951, F1 0.8936: E25.
- E24 val/test 90.62/89.36, Prec 0.8927, Rec 0.8936, F1 0.8923: E24.
- E24 weights 0.2/0/0/0.4/0/0.1/0.3 and E25 weights 0/0/0/0.4/0/0.1/0.5: E24.weights, E25.weights.
- Fusion simplex step 0.1: experiments.py _weight_simplex(step=0.1).
- Seven fusion sources, including the classical PCA-KNN: E24/E25 sources.
- 86.43 % tuned PCA-KNN test: E3.test_acc.
- 87.10 % linear Soft-KNN test: E6.test_acc. Its F1 0.8698 matches E6.
- 85.70 % MLP test: E16.test_acc.
- 87.13 % bank-augmented test: E14.test_acc.
- 87.63 % conv test: E19.test_acc (but see M1).
- 84.62 % conv-shift test: E20.test_acc.
- 88.06 % 3-seed ensemble test: E21.test_acc.
- 89.03 % distilled student: E23.test_acc and kaggle_distill.json.
- 88.73 % E22 test and Rec 0.8873: E22.
- Seeds 42/7/123 with tests 87.53/86.92/87.57: conv_seeds.json and E21.per_seed.
- "Exceeding every individual teacher": 89.03 > 87.57.
- 81.68 % on-the-fly augmentation (hflip + shift + erase): E7.
- -5.42 vs no-aug: 81.68 - 87.10.
- -3.01 shift conv: 84.62 - 87.63.
- 87.00 % gen-2 student: E26.test_acc. -2.03: 87.00 - 89.03.
- +4.27 (lines 181, 451): 89.51 - 85.24.
- +0.26 cosine vs Euclidean: 85.50 - 85.24 (arithmetic only; 85.50 cannot be traced).
- +0.67 Soft-KNN over tuned classical: 87.10 - 86.43.
- +1.19 PCA-64 (line 556): 86.43 - 85.24 (but this is Optuna-tuned with k = 5 Manhattan, see M6).
- 1.84/0.26 = 7.08 and 1.84/0.02 = 92: arithmetic correct, inputs not traceable (C3).
- 70,000 / 60,000 / 10,000 images, 10 classes: data.py FILES.
- 54,000 / 6,000 stratified, seed 42: data.py load_splits(val_size=6000, seed=42).
- 5,400 per class: 54,000/10, stratified.
- [0,1] scaling: data.py:69-70.
- 784 = 28x28; conv 28->14->7 with 32/64 channels and a 128-D head: softknn.py ConvMetric.
- ReLU in the shipped backbone: softknn.py:65-66.
- Batch size 256, gradient clipping 1.0, bank refresh 107: E16/E19/E26 configs.
- lr 3e-4 for conv, MLP and distill; lr 1e-3 for linear E6: configs.
- Adam beta1 0.9, beta2 0.999: torch defaults (train.py:19).
- RMSprop alpha 0.99, lr 1e-3: torch default and E9.
- Teacher soft targets at k = 9, tau = 1.0: experiments.py e23 and train_distill.py.
- Distillation alpha = 0.5, T = 1.0: E26.config, train_distill.py defaults.
- TTA with six views (identity, hflip, four 1-px shifts): augment.py tta_views; softknn.py n_views = 6.
- Linear projector PCA warm start: experiments.py _softknn_run passes pca.components_.
- d = 64 chosen as best PCA dimension: E2.best_dim 64.
- Bias-variance peak at moderate k: E4.best_k = 7.
- Global seed 42, multi-seed runs 42/7/123: experiments.py Ctx and E21.
- On-the-fly augmentation hflip + shift + erase: E7.config.augment.
- The four member .tex files differ only at lines 96-98, so every finding applies to all four.
