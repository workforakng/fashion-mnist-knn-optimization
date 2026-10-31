# Audit 02: Figures, tables and diagrams

Scope: docs/paper.tex (621 lines), the compiled PDF docs/241001001213_AksharNathGorain.pdf (6 pages, rendered at 110 dpi), figures/*.png (15 files), figures/README.md, the figure section of the root README.md, src/experiments.py, src/evaluate.py, src/classical.py, src/softknn.py, src/train.py, both notebooks, and results/*.json.
The four member PDFs and .tex files differ only in the name, ID and email lines (checked with diff), so every finding applies to all four.

## 0. Inventory

| Item | In paper? | Where | Label / ref |
|---|---|---|---|
| 01_class_distribution.png | no | - | - |
| 02_samples.png | no | - | - |
| 03_pca_variance.png | yes, Fig. 1(a) | l.257 | fig:pca, ref'd l.249 |
| 04_pca_sweep.png | yes, Fig. 1(b) | l.258 | fig:pca, ref'd l.251 |
| 05_latency.png | no (Sec 3.4 has Table 2 only) | - | - |
| 06_bias_variance.png | no (Sec 3.2 describes it in text) | - | - |
| 07_optuna_history.png | no | - | - |
| 08_augmented.png | no | - | - |
| 09_optimizer_compare.png | no (Sec 5.2 discusses it in text) | - | - |
| 10_tau_sweep.png | no (Sec 4 text claims a sweep) | - | - |
| 11_softknn_curves.png | no | - | - |
| 12_confusion_matrix.png | yes, Fig. 3 | l.519 | **no \label, never referenced** |
| 13_conv_curves.png | no | - | - |
| 14_inference_grid.png | no | - | - |
| 15_conv_shift_curves.png | no | - | - |
| TikZ pipeline | yes, Fig. 2 | l.339-364 | fig:pipeline, ref'd l.332 |
| Table 1 metrics | yes | l.220 | ref'd l.213 |
| Table 2 latency | yes | l.274 | ref'd l.266 |
| Table 3 fusion | yes | l.454 | ref'd l.450 |
| Table 4 ladder | yes | l.484 | ref'd l.479 |
| Table 5 negative | yes | l.528 | **tab:negative never referenced** |

The paper uses 3 of the 15 PNGs. Every \ref has a matching \label, so nothing points to a missing label. The figures are numbered in order of first mention (Fig 1 at l.249, Fig 2 at l.332, Fig 3 never mentioned).

---

## 1. Findings: figures in the paper

### F1. Critical. Fig. 3 (confusion matrix), caption l.520-523: the per-class percentages are wrong
- Problem: the caption says "Trouser 98.6 %, Sandal 97.8 %, Bag 97.4 %". The plotted matrix shows Trouser 979/1000 = **97.9 %**, Sandal 966/1000 = **96.6 %** and Bag 985/1000 = **98.5 %**. None of the three numbers is correct, and the order is wrong too: Bag is the best class, not Trouser. Sneaker (97.0 %) also beats Sandal. The caption also leaves out the most important fact: **Shirt recall is 61.7 %** (160 Shirts are predicted as T-shirt/top).
- Evidence: I read the matrix from the PNG. Every row sums to 1000 and the trace is 8951, which matches the 89.51 % accuracy. Macro precision recomputed from the matrix is 0.8945, which matches E25 in the ledger. So the matrix is the right one and only the caption is wrong. The same wrong numbers appear in README.md l.281 ("Figure 8 Legend").
- Fix: "Diagonal strengths: Bag 98.5 %, Trouser 97.9 %, Sneaker 97.0 %; Shirt is weakest at 61.7 % (16 % of Shirts predicted as T-shirt/top)."

### F2. Major. Fig. 3, l.517-524: no \label, never cited, and it floats away from its section
- Problem: Sec 6.2 "Final Evaluation" has only a heading and the figure environment, with no text. In the PDF the heading sits empty on p.5. The figure floats to the top of p.6, after Sec 7 (Conclusion) has already started. The reader never gets a "Figure 3 shows ...".
- Fix: add `\label{fig:cm}` and one or two sentences in 6.2 that cite it, including the per-class recall and the Shirt row. Use `[!t]` or `[H]` (float package) so the figure stays in 6.2.

### F3. Major. Fig. 3, l.519: illegible at printed size
- Problem: the PNG is 7.0 in wide (figsize 7.5x6.5) and is printed at 0.38\textwidth = 2.60 in, a scale of 0.37. The 8 pt cell annotations print at **about 3.0 pt** and the tick labels at about 3.7 pt. On p.6 at 110 dpi the cell counts cannot be read. The caption quotes percentages, but the figure shows raw counts and no row-normalised view.
- Fix: print at 0.6-0.7\textwidth, or regenerate at figsize about 4x3.6 in with 7-8 pt annotations. Better, plot row-normalised recall (%) so the cells match the caption. The colour bar has no label (it should read "count" or "recall").

### F4. Major. Fig. 1(a), caption l.259: the caption describes a line that does not exist
- Problem: the caption says "the dotted line marks the d=64 operating point". The figure has a **dashed horizontal** line at **90 % variance**, with the legend "90% variance". That line meets the curve at d≈81-84 (ledger `var90_dim = 84`), not at 64. Nothing in the figure marks d=64.
- Evidence: 03_pca_variance.png. Code at src/experiments.py l.161 is `ax.axhline(0.90, ..., ls="--", label="90% variance")`.
- Fix: add `ax.axvline(64, ls=":")` and fix the caption to "dashed line = 90 % variance (reached at d=84); dotted line = d=64".

### F5. Major. Fig. 1(a), text l.250 and figures/README l.12: variance at d=64 is misstated
- Problem: the text says "64 components retain ≈86 %" and figures/README.md says "85.3 % threshold at d=64". The root README (l.254) says "dashed reference line marks the 85.0 % threshold … At d=64 … 85.3 %". I calibrated pixels against the gridlines: x=0→px 122.5, 20→239.5; y=0.9→px 90, 0.8→163.5. The curve at d=64 sits at px row about 103, which is **≈0.88 (88 %)**. This agrees with var90_dim=84 in the ledger: the curve reaches 0.90 only at about 84.
- Fix: state "≈88 %" or compute the exact value from `explained_variance_ratio_[:64].sum()` and print it in the figure.

### F6. Major. Fig. 1(b), caption l.260: says "test accuracy" but the plot shows validation accuracy
- Problem: the y-axis label is "validation accuracy" and the code (experiments.py l.141-152) scores on `ctx.xv`. The caption says "Test accuracy versus retained dimensionality". This is exactly the val/test confusion the protocol section warns against.
- Fix: change the caption to "Validation accuracy (k=5, Euclidean, uniform weights)".

### F7. Major. Fig. 1(b), caption l.260-261: the overfitting claim is not supported by the plotted data
- Problem: the caption says "too many [components] re-introduce distance concentration". The plotted values after the peak are 0.8733 (64), 0.8725 (80), 0.8725 (84), 0.8713 (96) and 0.8717 (128). The total drop is 0.16 pp, about 10 of 6,000 validation images, with no error bars. That is a plateau. The truncated y-axis (0.82-0.875) makes it look like a decline.
- Fix: say "accuracy saturates beyond d≈48-64". Optionally add binomial error bars (±0.43 pp at n=6000) or average over seeds.

### F8. Major. Fig. 1(b), 04_pca_sweep.png: zig-zag line because the x values are unsorted
- Problem: the 90 %-variance point (d=84) is appended after d=128 (experiments.py l.141-150; ledger E2 row order 10, 20, 32, 48, 64, 80, 96, 128, 84). `ax.plot(..., "o-")` connects 128 back to 84, which draws a spurious crossing segment between d=84 and d=128.
- Fix: sort rows by dim before plotting, or plot the 90 %-variance point as a separate marker.

### F9. Major. Fig. 1(b) against text l.251-253 and Table 4 rows E04-E07: the numbers disagree
- Problem: the figure (ledger E2) gives d=32 val 86.72, d=64 val 87.33, d=128 val 87.17, and **has no d=16 point** (the sweep starts at d=10). Table 4 gives E04 "PCA d=16" val 82.80, E05 d=32 val 85.10, E06 d=64 val 86.60, E07 d=128 val 86.40. Every value disagrees with the figure printed next to it, and d=16 was never run (`SCALES['full']['pca_dims'] = [10, 20, 32, 48, 64, 80, 96, 128]`, experiments.py l.43). The text says d=64 lifts the baseline "to 86.43 %", which is a *test* number from a *different* model (Optuna E3: k=5, Manhattan, distance). The figure already shows 87.33 % *val* with untuned k=5 Euclidean.
- Fix: rebuild rows E04-E07 from the ledger (d = 10/20/32/48/64/80/96/128, val only, or run test once if you need it). Drop d=16 or actually run it.

### F10. Major. Fig. 1(a)+(b), l.257-258: illegible at 0.38\textwidth, and the layout wastes space
- Problem: each PNG is about 6.1-6.2 in wide (dpi 150) and is printed at 2.60 in, a scale of 0.42. Ticks at 10 pt print at **about 4.2 pt**, axis labels at about 4.2 pt, titles at about 5 pt and the legend at about 4.2 pt. All of these are below the usual 6-7 pt minimum. On rendered p.3 the tick labels can barely be read. The two images plus `\hfill` leave a gap of about 24 % of \textwidth in the middle. Neither panel carries an "(a)" or "(b)" marker, even though the caption and text use them.
- Fix: use `width=0.49\textwidth` each, or regenerate at figsize about 3.4x2.4 in with 8-9 pt fonts. Add "(a)" and "(b)" in the panel titles or use the subcaption package.

### F11. Minor. Fig. 1 against the caption: the panels have titles the caption does not match
- The panel titles are "PCA explained variance" and "PCA dimension sweep (k=5, Euclidean)". The caption omits k=5, Euclidean and uniform weights, and the text implies the tuned configuration. Put the configuration in the caption.

### F12. Minor. Fig. 2 (TikZ pipeline), l.339-360: arrows and box widths
- The boxes have different natural widths (no `text width`), so the stack looks ragged. With `node distance=2.5mm` the `-{Latex}` arrowheads almost touch the next box (visible on p.3). Fix: set `text width=0.9\textwidth` on the stage style and use node distance 4-5 mm.

### F13. Major. Fig. 2 (TikZ), l.345-356: the diagram does not match the code
- (a) Stage 1 says "70,000 images … stratified split 54k / 6k / 10k". The 10k test set is the official partition, not part of a split of 70k (data.load_splits splits only the 60k train set; paper l.190-192 says the same). Write "60k dev → 54k/6k stratified; official 10k test".
- (b) The arrow from Stage 2 to Stage 3 and the caption ("each stage consumes the previous stage's artefacts", l.335) imply the conv Soft-KNN consumes the PCA-64 projection. `ConvMetric` (src/softknn.py l.62-76) ignores `pca_components`; only the *linear* MetricProjector is PCA-warm-started. Stage 3 names only the conv backbone.
- (c) Stage 4 shows "3-seed ensemble 88.06 % → distilled student 89.03 % → TTA + convex fusion". In the E25 champion every conv seed has weight **0.0** (Table 3, ledger E25 `weights [0,0,0,0.4,0,0.1,0.5]`). The champion is 0.4 × bank-augmented **linear** Soft-KNN + 0.1 × classical PCA-KNN + 0.5 × distilled student. The linear/bank-augmented model (40 % of the champion) and the classical PCA-KNN source (10 %) appear nowhere in Stages 3-4, and no arrow runs from Stage 2 to Stage 4.
- (d) The memory bank (refreshed every 107 steps) is a core component (train.py l.75-78, 87-88) and does not appear.
- Fix: redraw as a DAG. Stage 2 (PCA-KNN) and the linear/bank-augmented Soft-KNN both feed the fusion. The seed ensemble feeds only the distillation teacher. Label the memory bank.

---

## 2. Findings: tables

### T1. Critical. Table 4 (ladder), l.484-514: the Val column is not the ledger's
For each row: paper value, then results.json value (ledger id).
- E01 Val "---" / 86.13 (E1)
- E04 d=16 82.80 / no d=16 run
- E05 d=32 85.10 / 86.72 (E2)
- E06 d=64 86.60 / 87.33 (E2)
- E07 d=128 86.40 / 87.17 (E2)
- E08 Optuna 86.85 / **87.82** (E3)
- E09 Soft-KNN linear 87.40 / 88.22 (E6)
- E10 MLP 86.20 / 86.92 (E16)
- E11 bank-aug 87.45 / 88.28 (E14)
- E12 conv s42 88.10 / 89.07 (E19)
- E13 conv shift 85.10 / 85.57 (E20)
- E14 3-seed ensemble 88.50 / 89.72 (E21)
- E15 distilled 89.20 / 90.10 (E23)
- E16 (E22) 89.95 / 90.27 (E22)
- E17 (E24) 90.62 / 90.62 (E24) ✓
- E18 (E25) 90.85 / 90.85 (E25) ✓

The Test column matches the ledger for E01, E08-E18. The Val numbers for E04-E16 are all 0.3-1.2 pp below the ledger, and E04-E07 test values have no ledger source (the PCA sweep logged val only). The Prec./Rec. values for E01-E16 and every **latency** value in the table have no source in the ledger or the code. There is no Soft-KNN timing code at all (grep for perf_counter finds only classical.eval_knn). E01 F1 is 0.8510 in the paper and 0.85177 in the ledger.
- Fix: regenerate the table programmatically from results.json. Use "n/a" wherever no measurement exists. Drop the latency column, or add the timing code and log the results.

### T2. Major. Table 4: two experiment-ID systems are mixed
- The rows use a private E01-E18 numbering. Ledger IDs appear in parentheses only for E16-E18, and the caption says "The champion (E25) is highlighted" while the row is labelled E18. Table 3 uses ledger IDs (E24/E25). Text l.481 says the ledger has "all 28 experiment entries". In fact results.json has 28 list items with ids E0-E26 (E10 and E11 appear twice, there is **no E18**, and E27 exists in code but was never logged). Paper "E12 conv" is ledger E19, while ledger E12 is the inference grid, so the paper's E-numbers point to different experiments in the ledger.
- Fix: use ledger IDs throughout, or add a mapping column.

### T3. Major. Table 4: the row descriptions contradict the ledger configs
- E08 "PCA-64, k=7, dist. weights" (implies Euclidean). The ledger E3 best_params are {k=5, weights=distance, **metric=manhattan**}, and the Optuna plot (fig 07) best is 0.8782.
- E09 "784→64 linear, τ=0.1". The ledger E6 config is `embed_dim 128, tau 0.5`, matching paper l.323 ("z∈R^128"). The table contradicts its own text.
- E10 "784→256→64 MLP". The ledger E16 config is hidden 256 → **embed 128**.
- E17 "conv + linear + distill". The weights also include classical PCA-KNN (0.1).
- Fix: correct the descriptions from the ledger configs.

### T4. Major. Table 4: the ladder is not monotone, not chronological, and does not add one technique per rung
- Caption and text l.482 claim "Every rung adds exactly one technique". In fact E10 (MLP, 85.70) and E13 (conv shift, 84.62) are regressions. E16 (E22 fusion, 88.73) comes *after* E15 (distilled, 89.03), yet ledger E22 ran before distillation existed (timestamps 23:27 and 00:00) and does not include the student. Readers will take the 88.73 after 89.03 as a regression of the method.
- Fix: order rows by dependency, put ablations or negative variants in a separate block, and drop the "exactly one technique" claim.

### T5. Major. Table 2 (latency), l.274-287: rows mix sources and contradict Table 4 and the ledger
- Raw 784-D brute: paper 1.84 s. Ledger: E1 `query_s` = 3.19 s and E5 brute median = 2.98 s, for 6,000 validation queries.
- PCA-64: paper 0.26 s. Ledger E5 brute = 0.418 s. Ledger ratio = 2.98/0.418 = **7.12×**. Paper 7.08× (= 1.84/0.26). figures/README says 7.14×. That is three different speed-ups.
- The "PCA-64, tuned k" row pairs latency 0.26 s (Table 4 E06, untuned) with accuracy 86.43 % (E08, latency 0.25 s).
- The "Soft-KNN (conv backbone)" row shows 0.02 s, but Table 4 gives the conv model (E12) **0.05 s**. 0.02 s is E09, the *linear* model. The 92× speed-up is therefore computed from the wrong row. With 0.05 s it would be 37×.
- The caption says "(CPU)", but Soft-KNN inference runs on CUDA in the code (`predict_probs(..., device="cuda")`). "Per query batch" is never defined (the ledger batch is 6,000 queries).
- Text l.270-272 credits the speed to "short-list candidate pruning". The code does a brute `torch.cdist` against the whole bank (softknn.py l.88-90), with no pruning.
- Fix: take every number from E5 and add a logged Soft-KNN timing (same device, same query count). State the hardware and batch size in the caption.

### T6. Major. Table 1 (metrics), l.220-233: only one of three rows can be traced, and the latency unit is undefined
- Only the Euclidean row has a ledger entry (E1: test 85.24 ✓, F1 **0.8518** vs 0.8510 in the paper, query 3.19 s vs 1.84 s). Manhattan and Cosine raw-pixel runs do not exist in results.json or experiments.py. The notebooks compute them on a 10k-train/2k-test subset with standardised pixels (fashion_knn_colab.ipynb cell 4), which cannot give these full-data numbers.
- The header says "Latency (s/batch)" and the batch is undefined.
- Bolding 85.50 implies a meaningful win, but +0.26 pp is below the binomial SE of about 0.35 pp at n=10,000.
- Fix: log E1-style runs for Manhattan and Cosine, define the batch, and add ± or a note on significance.

### T7. Major. Table 5 (negative results), l.528-542: the caption contradicts row 2, and the table breaks a sentence
- The caption says "Techniques that *degraded* accuracy", but row 2 (memory-bank augmentation) shows **+0.03**. The text (l.545) calls it "neutral".
- In the PDF (p.5) the table floats into the middle of the sentence "First, *where* [TABLE] augmentation is applied…", because `\begin{table}[htbp]` is placed mid-paragraph right after "rigour:". The label tab:negative is never cited.
- Omitted negatives present in the ledger: TTA on the base model (E10 second entry: test 87.10 → 86.74), shift-augmented *linear* Soft-KNN (E13: 83.66 vs 87.10, −3.44), the MLP metric (85.70 vs linear 87.10), and the 5-teacher distillation (89.01 vs 89.03). The optimizer study (SGD beat Adam, see F15) is also missing.
- The Δ values themselves are correct: 87.10−81.68 = 5.42, 87.63−84.62 = 3.01, 89.03−87.00 = 2.03.
- Fix: retitle to "Techniques that did not improve accuracy". Place the table before the paragraph and cite it with \ref. Add the omitted rows.

### T8. Minor. Table 3 (fusion), l.454-473
- The weights match ledger E24 and E25 exactly ✓, and the Val/Test values match ✓.
- Bolding is inconsistent: 0.4 is bold in both columns and 0.5 is bold, but 0.2 (the largest conv weight in E24) and 0.3 are not. Either bold the maximum per column or nothing.
- Missing context: the distilled-student source is evaluated at k=9, τ=0.1 (experiments.py l.1073-1076), while ledger E23 selected k=7. The table does not say this.
- Text l.452 says "+3.08 over the strongest single learned model". 89.51−3.08 = 86.43, which is the classical PCA-KNN, not a learned model. The strongest single learned model is the distilled student at 89.03 (+0.48).

### T9. Minor. Table 4 layout, l.490
- Fixed-width `p{}` columns justify the text, which produces stretched gaps ("Soft-KNN      (bank aug.)", "6-source      fusion (E22)", p.5). Numbers are left-aligned instead of decimal-aligned. Val/Test are in %, but Prec/Rec/F1 are 0-1 fractions in the same table. "Lat." carries the unit in every cell while the caption also gives it.
- Fix: use `>{\raggedright\arraybackslash}p{}` or tabularx for the text columns and siunitx `S` columns for the numbers. Use one scale (all %).

### T10. Major. All tables and Section 5.4: no uncertainty reported, and sub-noise deltas are presented as gains
- Three seeds exist (conv_seeds.json, kaggle_conv_seeds.json), yet no table reports mean ± sd. Several key deltas are within test-set noise (SE ≈ 0.31 pp at 89 %, n=10k): E25 vs E24 (+0.15), bank-aug vs base (+0.03), cosine vs Euclidean (+0.26).
- Section 5.4 / Table 4 also mix runs. The seed-42 test accuracy is 87.53 (conv_seeds.json) in l.417 but 87.63 (ledger E19) in Table 4 and l.414. The "+0.43 over the best single seed" uses 87.63, but the best of the three ensemble seeds is 123 at 87.57 (gain +0.49).
- Fix: report mean ± sd over seeds for E12, use one run consistently, and mark differences below 2·SE as not significant.

---

## 3. Findings: figures not used in the paper (but in the repo, README and figures/README)

### U1. Critical. Fig 09_optimizer_compare.png contradicts the paper's optimizer verdict (Sec 5.2, l.366-399)
- Problem: the paper says SGD "struggled … requiring roughly 3× more epochs", that RMSprop showed "gradient thrashing", and that "Adaptive methods converged fastest". The plot and ledger E9 show **SGD reaching the highest validation accuracy from epoch 1** (≈0.83-0.84 vs Adam ≈0.74-0.81). Ledger E9 `"best": "sgd"`, with best_val 0.8385 (SGD), 0.8147 (Adam) and 0.8083 (RMSprop). The code's SGD is lr **0.1** with plain momentum 0.9 (train.py l.21-22, experiments.py l.373), not Nesterov with η=0.01 as stated at l.385-387. Adam ran at 1e-3, not 3e-4. All three runs used the harmful on-the-fly augmentation (hflip+shift+erase), the same config Table 5 reports as −5.42 pp.
- The root README l.272 also inverts the colours: it says "SGD (orange) descends slowly, RMSprop (green) oscillates". In the plot SGD is green and best, and RMSprop is orange and smooth. It also calls Adam "AdamW … decay 1e-4"; the legend says "adam" and the code uses `torch.optim.Adam` with no weight decay.
- The figure is left out, and that omission hides the contradiction.
- Fix: either rewrite Sec 5.2 to match the evidence (SGD with lr=0.1 was best in this budget) or rerun the comparison under the stated settings. Include the figure.

### U2. Major. Fig 10_tau_sweep.png does not support "peaks sharply at τ=0.1; τ<0.05 saturates; τ≥0.5 leaks" (l.314-316)
- Problem: the sweep grid is {0.1, 0.2, 0.3, 0.5, 0.7, 1.0} (experiments.py l.338). No τ<0.1 was tested, so the "[0.01, 1.0]" range, the "peak" and the "τ<0.05 saturates" claim are unsupported. Accuracy decreases monotonically from the smallest τ tested (86.53 → 81.20), so the best value sits at the boundary of the grid.
- Every τ model was trained **with the harmful hflip+shift+erase augmentation** at half the epochs (l.341-343). They were *evaluated* through `_softknn_eval`, which uses `ctx.best_tau` (default 0.5, l.256). ctx.best_tau is set only after the loop (l.364), so every model was scored at inference τ=0.5. The "mean pred. entropy" curve therefore does not measure sharpening by τ. It stays flat at 0.927-0.937 on a truncated twin axis, which exaggerates 1 pp of variation. It also contradicts fig 14, where entropy at τ=0.1 is 0.37.
- The figure title "boundary sharpening" and the figures/README text ("peak at τ=0.1") overstate what was measured. The root README l.272 adds "(87.10 %)", but the plotted peak is 86.53 % val, and 87.10 is the E6 *test* accuracy trained at τ=0.5.
- Fix: rerun with τ ∈ {0.01, 0.02, 0.05, 0.1, …}, without augmentation, evaluate at the training τ, and use one y-axis or clearly labelled panels.

### U3. Major. Fig 14_inference_grid.png: shows accuracy does not depend on inference τ, which undercuts the τ narrative; figures/README describes a different figure
- Problem: the left panel has flat lines. Accuracy is invariant to inference τ, as it must be, since argmax(−d/τ) does not depend on τ. The y-axis (0.879-0.883) is truncated, so k-curves that differ by about 20 images look widely separated.
- figures/README l.24 calls this file "Qualitative Prediction Grid: test samples alongside predicted category, ground-truth label, and confidence score". No such image exists in the repo.
- Fix: correct the README description. The paper should state that τ matters only during training.

### U4. Major. Fig 11_softknn_curves.png: train accuracy is inflated by self-matching; README descriptions are wrong
- Train accuracy is 0.947 at epoch 0, while loss is 1.71 and val is 0.876. In train.py the training queries are drawn from `x_img`, which is also the memory bank (`bank_x = x_img`, l.61, l.92-93), so each query's own embedding is among its class's top-k anchors (distance ≈0). The train curve is therefore not a valid overfitting diagnostic.
- figures/README l.21 says "Dual-panel plot of training vs. validation loss and accuracy". There is **no validation loss**. Root README l.281 says "10 epochs, Top/Bottom panel, val loss orange dashed … converge without divergence". The real figure has 8 epochs (0-7), left/right panels, no val loss, and an 8 pp train-val gap.
- Fix: exclude self from the bank (mask the diagonal) when computing train accuracy, add val loss, and correct both READMEs.

### U5. Major. Fig 05_latency.png: mislabelled axis and a scale that hides the claimed speed-up
- The y-label says "query latency, 1k val (s, median of 3)". The code queries the full **6,000-image** validation set (`ctx.xv`, experiments.py l.218-222), and trees run only **one** repeat (classical.py l.69).
- With a linear axis to 80 s, the brute bars that carry the paper's 7× claim (2.98 s vs 0.42 s) are nearly invisible.
- figures/README says 7.14×; the data give 7.12× and the paper says 7.08×.
- Fix: use a log y-axis with value labels, correct the label, and use one speed-up value.

### U6. Minor. Fig 06_bias_variance.png against text l.237-241 and the READMEs
- The plot covers k ∈ {1..25} (13 points), while the text and figures/README say k ∈ {1..15}. Val accuracy at k=5/7/9 is 0.8733/0.8737/0.8735, differences of 1-2 images, so "optimal k=7" is a tie on a truncated axis. The title says weights=uniform, while E08 is presented as distance-weighted. Train accuracy at k=1 = 1.0 comes from self-inclusion; the code docstring documents this, but the figure does not.
- Root README l.263 says "Peak **test** performance (86.43 %) occurs at K=7". The curve is validation and peaks at 87.37 %, and 86.43 is the Optuna test accuracy (k=5).
- Fix: correct the range and the split, and mark the plateau.

### U7. Minor. Fig 07_optuna_history.png
- The best value 0.8782 is E3 val ✓, which contradicts Table 4 E08 val 86.85 and k=7 (T3). Trials are joined by a line, which suggests a trajectory. figures/README promises "objective value improvement", but no best-so-far curve is drawn. The search space did not include d (fixed PCA-64), contradicting l.253 "Bayesian optimisation over {k, d, metric, weights}" (classical.py l.106-112).

### U8. Minor. Fig 02_samples.png has no class labels
- `set_ylabel` is called after `axis("off")` (experiments.py l.111-112), so the row names never render. figures/README says it shows "all 10 garment categories", but they are unlabelled. Fix: use `ax.set_axis_off()` only for ticks and spines, or add text labels.

### U9. Minor. Fig 08_augmented.png
- figures/README l.17 says "horizontal reflections and random crop variations". The code applies hflip p=0.5, a ±2 px shift and random erase. There is no crop, and no flip is visible in this draw. The 8 samples are just `ctx.xa[:8]` (two bags, two boots, two shirts, …), so they are not class-diverse.

### U10. Minor. Figs 13 and 15 (conv curves)
- 13_conv_curves.png is titled "E19", while the paper calls this model E12. Val accuracy is still rising at the last epoch (0.8908 at epoch 29). Training hit the 30-epoch cap and did not early-stop. The paper l.413 says "early stopping (patience 10)"; the code uses patience 8 (l.749).
- 15_conv_shift_curves.png: figures/README l.25 calls it a "Spatial Shift Stress Test … under synthetic horizontal and vertical pixel shifts". It is actually the training curve of shift-augmented training (E20). Val accuracy oscillates between 0.76 and 0.84.
- Both use a different style from the rest (no grid, no panel titles, no legend). Note that the val curves in 11, 13 and 15 use the training τ=0.5 and k=7, not the reported inference settings.

### U11. Minor. Colour and style are inconsistent across figures
- Orange means "validation" in 06, 11, 13 and 15, but "rmsprop" in 09, "entropy" in 10 and "k=5" in 14. Fonts and figure sizes vary (7x4.5, 8x4.5, 11x4, 7.5x6.5). There are no error bars anywhere, even though 3-5 seeds exist. Truncated y-axes are never noted (04, 06, 07, 10, 14).

### U12. Minor. figures/README.md, row 04 (l.13)
- It says "Evaluates Euclidean, Manhattan, and Cosine distance metrics across d ∈ [16,128]". The plot has a single Euclidean curve over d ∈ [10,128]. The root README l.263 repeats this and adds "Test accuracy … peaks at d=64 across all metrics".

### U13. Minor. figures/README row 03 / root README l.254
- Says "85.3 % / 85.0 % threshold". The plotted line is at 90 %, and d=64 corresponds to about 88 % (F5).

### U14. Minor. Notebooks do not regenerate figures/
- fashion_knn_colab.ipynb and fashion_knn_clean_pipeline.ipynb only call `plt.show()` and never `savefig`. They also use a different split (`random_split` 50k/10k, not stratified 54k/6k), normalised pixels, an AdamW+GELU model with embed 64, and their own "Figure 1-5" numbering. The paper's claim (l.570-580) that "everything in this report regenerates" from the notebook does not hold for any figure. All 15 PNGs come from src/experiments.py. 12_confusion_matrix.png is overwritten by every fusion stage (E15, E17, E22, E24, E25 and E27 all call `plot_confusion_matrix` with the default filename). It currently matches E25 only because E25 was the last of those stages run (E27 is not in the ledger).
- Fix: give each confusion matrix a distinct filename, and state that figures come from `python -m src.experiments --scale full --all`.

---

## 4. Code check: what each included figure actually plots

| Figure | Generator | Split | Model / config | Matches caption? |
|---|---|---|---|---|
| Fig 1(a) 03 | e2_pca_sweep l.157-164 | PCA fit on the 54k train set | 128 comps, cumulative ratio | No: line is the 90 % threshold, not d=64 (F4) |
| Fig 1(b) 04 | e2_pca_sweep l.149-155 | **validation** (6k) | k=5, Euclidean, uniform | No: caption says test (F6); unsorted x (F8) |
| Fig 3 12 | evaluate.plot_confusion_matrix via _fusion_eval (E25) | test 10k | E25 TTA fusion, weights [0,0,0,0.4,0,0.1,0.5] | Model/split ✓; per-class numbers ✗ (F1) |
| Fig 2 TikZ | - | - | - | Partly; see F13 |

---

## 5. Severity summary

- Critical (3): F1 (CM caption numbers), T1 (ladder Val/latency/prec not from ledger), U1 (optimizer figure contradicts Sec 5.2).
- Major (20): F2, F3, F4, F5, F6, F7, F8, F9, F10, F13, T2, T3, T4, T5, T6, T7, T10, U2, U3, U4, U5. (U3 counted with U2 as one τ issue.)
- Minor (rest): F11, F12, T8, T9, U6-U14.

The highest-value fixes:
1. Regenerate Tables 1, 2 and 4 from results.json, and log what is missing.
2. Fix the Fig 1 and Fig 3 captions (val vs test, the d=64 line, the per-class numbers).
3. Enlarge the figures to at least 0.49\textwidth or regenerate them at print size.
4. Label Fig 3 and cite it in the text.
5. Rewrite Sec 5.2 and the τ claim, or rerun the experiments, so they match figs 09 and 10.
6. Fix the figures/README rows 03, 04, 05, 06, 09, 11, 14 and 15.
