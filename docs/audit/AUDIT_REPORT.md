# Paper and code audit — consolidated report

**Scope.** This audit covers:
- `docs/paper.tex` and the four member PDFs (the copies differ only in name, ID and email, so every finding applies to all four);
- `figures/`, `results/*.json`, `src/`, `scripts/`, the two notebooks and the READMEs.

**Method.** Five independent audits were run. Each has a detailed file in this folder with line numbers, quotes and suggested rewrites.

| File | Scope |
|---|---|
| `01_numbers.md` | Every number in the paper checked against `results/*.json`, the code and the READMEs |
| `02_figures.md` | All 15 PNGs, the rendered PDF pages, the tables and the TikZ diagram |
| `03_writing.md` | Sentence by sentence: clarity, jargon, consistency, LaTeX and the bibliography |
| `04_code.md` | Correctness, test-set leakage, reproducibility, plus scripts that reran parts of the pipeline |
| `05_claims.md` | Maths, technical claims, novelty, statistical validity |

The main findings were spot-checked a second time against the ledger and the figures (marked ✔).

---

## 0. Bottom line

The headline numbers are real and logged:
- the 85.24 % baseline;
- the 89.51 % test / 90.85 % validation champion (E25);
- every learned model's test accuracy.

The core `src/` pipeline does **not** tune anything on the test set.

**Much of the supporting material in the paper does not match the repository.**
- The validation column of the main table does not match the ledger.
- Some table rows were never run.
- The optimizer study states the opposite of the logged result.
- The latency and speed-up claims were never measured.
- The confusion-matrix caption gives the wrong numbers.
- The "one-click reproducible" Colab notebook runs a different model.

An examiner who opens `results/results.json` will find these within minutes. Fix them before any other edit.

---

## 1. Critical: numbers in the paper that the repo contradicts

| # | Paper (tex line) | What the repo shows | Fix |
|---|---|---|---|
| 1 ✔ | Table 4 (l.494–511): **Val** column, rows E05–E16 | Every row is 0.3–1.2 pt below the ledger. E08 86.85 vs **87.82**; E09 87.40 vs **88.22**; E12 88.10 vs **89.07**; E14 88.50 vs **89.72**; E15 89.20 vs **90.10**; E16 89.95 vs **90.27**. Test values match the ledger exactly. | Regenerate the table from `results.json` |
| 2 ✔ | Table 4 E04 "PCA d=16" | Never run. The sweep used d = 10, 20, 32, 48, 64, 80, 96, 128. The E04–E07 test values have no source. | Use the real sweep rows (validation only) |
| 3 | Table 4 E01–E03 Val "---" | E1 validation = 86.13 is logged | Fill it in |
| 4 | Table 1 / E02–E03 Manhattan 85.15, Cosine 85.50 | Not in the ledger. Rerunning the repo's code gives Manhattan **85.91** (the *best* raw metric) and Cosine 85.63. | Rerun and log. The paragraph explaining why cosine beats Manhattan (l.214–218) then becomes wrong. |
| 5 | Precision/F1 values (Tables 1 and 4) | Precision is not logged for E01–E15. Logged F1 values differ: E01 0.8518 vs 0.8510, E08 0.8644 vs 0.8631, E16 precision 0.8863 vs 0.8885. | Use the logged values; drop untraceable ones |
| 6 | Latency 1.84 s → 0.26 s, 7.08× (l.133, 268, 282–283, 350, 557) | Ledger E5: **2.98 s → 0.418 s = 7.12×**, on 6,000 validation queries. The READMEs say 7.14×. | Use the E5 numbers and state the batch size |
| 7 | Soft-KNN 0.02 s, **92×** (l.269–272, Table 2) | Never measured. Table 2 gives 0.02 s to the conv model, but Table 4 gives it 0.05 s (0.02 s is the linear model). The caption says CPU, but Soft-KNN ran on GPU. The "candidate pruning" explanation describes code that does not exist; `logits_from_z` is a brute-force `cdist` over all 54k images. | Measure it on the same hardware or delete the claim |
| 8 ✔ | Optimizer study (l.366–399): "Adam converged fastest", "SGD needed ~3× more epochs", SGD η=0.01 with Nesterov | Ledger E9: **SGD won** (val 83.85 vs Adam 81.47, RMSprop 80.83). The code used SGD at lr 0.1 with plain momentum, not Nesterov. "3×" and "monotonic loss" have no data. The figure (09) is left out of the paper. | Rewrite the section to match E9, or remove it. Fix the Nesterov sign (look-ahead is θ − γv). |
| 9 ✔ | Fig. 3 caption (l.521): Trouser 98.6, Sandal 97.8, Bag 97.4 | The figure shows **Trouser 97.9, Sandal 96.6, Bag 98.5**, and Shirt is the worst class at **61.7**. README l.281 has the same error. | Correct the numbers and mention Shirt |
| 10 ✔ | τ sweep "τ ∈ [0.01, 1.0], peaks sharply at 0.1, τ<0.05 saturates" (l.314–316) | Ledger E8 tested only 0.1–1.0. Accuracy falls steadily as τ grows, so 0.1 is the *edge of the grid*, not a peak. τ < 0.1 was never tested, and every model in the sweep was scored at τ = 0.5. | Say "best of the tested values (0.1–1.0)", or extend the sweep |
| 11 | PCA "64 components ≈86 % variance" (l.250) | The figure shows ≈88 % (code: 88.1 %); figures/README says 85.3 %. Three different values. | 88 % |
| 12 | Optuna result "k=7, distance weights", "search over {k, d, metric, weights}" (l.253, E08) | Ledger: **k=5, Manhattan, distance**, with d fixed at 64. The README says Cosine. | Report the real winner; d was not searched |
| 13 | Colab notebook "complete pipeline … eleven executable cells … reproduces end-to-end" (l.142, 576–580) | The notebook has **8** code cells and **no saved outputs**. It runs a *different* model (CNN with learnable class prototypes, no memory bank) on a 50k/10k split. It has no ensemble, distillation or fusion; 89.51 % is hard-coded in a markdown cell. **It also picks metric, PCA dim and k on the test set.** | Either make the notebook run `src/` or remove the reproduction claim |

## 2. Critical/Major: contradictions inside the paper

| # | Where | Problem |
|---|---|---|
| 14 ✔ | l.452 "+3.08 over the strongest single learned model" | 89.51 − 3.08 = 86.43, the *classical* KNN. Against the distilled student (89.03) it is **+0.48**; against the conv model, +1.88. |
| 15 ✔ | l.414 vs l.417 | Seed 42 is 87.63 in one place and 87.53 in the next. 87.63 is a separate run (E19). |
| 16 ✔ | l.422 "+0.43 over best single seed" | The best seed in the list is 87.57, so the gain is **+0.49**. |
| 17 ✔ | l.323 vs Table 4 E09 | The text says a 128-D linear projector; the table says 784→64 with τ=0.1. The run was 128-D at τ=0.5. E10 is 784→256→**128**, not 64. |
| 18 | l.324 / l.399 / l.412 | The optimizer is described three ways: Adam 1e-3, "all final models Adam 3e-4 with clipping", "AdamW-style". There is no AdamW in the code. The bank-aug model (fusion weight 0.4) used 1e-3 with no clipping. E19 used patience 8 (not 10) and no mixed precision. |
| 19 ✔ | Table 5 caption "Techniques that **degraded** accuracy" | Row 2 is **+0.03**. That source also has the largest fusion weight (0.4). |
| 20 | l.481–482 "E01–E18 … all 28 entries … every rung adds exactly one technique" | The paper's IDs (E01–E18) don't match the ledger's (E0–E26, with E10/E11 duplicated and E18 missing). Bold "E18" is called "E25". Several rungs change 3–6 things at once. |
| 21 | l.560 "every rung individually ablated" | No ablations were run. Missing controls: hard-vote KNN on the learned embedding; the same conv backbone with a plain softmax head; TTA on/off. |
| 22 | l.556 "PCA-64: +1.19 points" | That figure includes Optuna tuning. PCA alone is E06. |
| 23 | Fig. 2 (TikZ) | Shows the 3-seed ensemble feeding the champion, but in E25 all conv-seed weights are **0.0**. Leaves out the bank-aug and PCA-KNN sources, which carry weight. Draws 54k/6k/10k as one split of 70k. |
| 24 | Splits | The paper (54k/6k) matches `src/data.py`. The README, both notebooks and `results/README.md` say 50k/10k. |

## 3. Major: technical claims that are wrong or unsupported

25. **"Exactly KNN as τ→0" (l.310) is false.**
    - The limit picks the class whose k nearest *same-class* anchors are closest on average. That is a local-mean classifier, not a majority vote.
    - τ does not change the predicted class at all, only the confidence.
    - Here k means k anchors *per class* (90 in total at k=9).
26. **"Gradients through … the top-k selection" (l.312) is false.** Top-k selection has no gradient. The bank is detached, so only the query embedding is trained.
27. **Training images match themselves in the bank** (`src/train.py:228,253-267`).
    - Nothing excludes an image from its own neighbour list, which NCA does exclude.
    - Rerunning shows this inflates train accuracy by about 8–10 pt (94–96 % vs 85–88 %), so the training curves (Fig. 11) are invalid.
    - Validation is barely affected.
28. **Distillation targets leak training labels.**
    - Teacher probabilities are computed on images that are in their own bank: teacher train accuracy is 94.2 % with self-match vs 85.7 % leave-one-out.
    - The gen-2 teacher includes sklearn `predict_proba` on its own fit set, which is exactly one-hot.
    - At τ = 1 the gen-1 teacher is nearly uniform (true-class probability ≈ 0.17), so it acts mostly as label smoothing.
    - So "soft targets encode inter-class structure" (l.435) and "128-D capacity is the binding constraint" (l.548, 564) are not supported.
    - The gen-2 teacher was the E24 fusion, not the gen-1 student.
29. **The distillation equation (l.431) is malformed.**
    - σ is undefined and is applied to *probabilities*.
    - T and τ are used for the same idea.
    - p and q are mixed.
    - `hinton2015` is in the bibliography but never cited.
30. **"MoCo-style memory bank" is the wrong name.** There is no momentum encoder; it is a periodically refreshed bank (Wu et al. 2018). "Doubles/halves compute" is wrong arithmetic. `he2020` is never cited.
31. **Novelty is overstated (l.293–297).**
    - NCA and LMNN are not "parametric prototype methods"; NCA also works over training neighbours.
    - The method is close to Wu, Efros & Yu (ECCV 2018).
    - Related work to cite: ProxyNCA, Prototypical/Matching Networks, Salakhutdinov & Hinton 2007.
32. **No uncertainty is reported.** On 10k test images a single-run difference under about ±0.3–0.6 pt is noise. The +0.26 (cosine), +0.03, +0.15 (TTA), +0.43/0.49 and +0.48 gains are all inside it. Seed spread is 0.65 pt. Add ± or McNemar tests, and stop giving causal explanations for gaps of 0.26 pt.
33. **TTA was never ablated.** Ledger E10 shows TTA *hurt* the linear model (87.10 → 86.74). A horizontal flip at test time sits oddly with the claim that flip-augmentation hurts training.
34. **The validation set is reused for everything**: PCA, Optuna, early stopping, the (k, τ) grids and an ≈8,000-point fusion search. The 90.85 % validation figure is therefore optimistic. Several inference settings (k=9, τ=0.1 for conv seeds and the student) are hard-coded rather than chosen on validation.
35. **The paper overclaims.**
    - "competitive modern vision system" (l.554), "Systematic" (title), "Honest", "same rigour".
    - 89.5 % is about RBF-SVM level on Fashion-MNIST (≈89.7 %, Xiao 2017, worth double-checking) and below small CNNs.
    - "Without departing from the nearest-neighbour paradigm" is arguable once a CNN, an ensemble and fusion are involved.
36. **Explanations with no evidence.**
    - Cosine "removes lighting variation" (these are not photographs).
    - Manhattan "insensitive to diagonal strokes" (and Manhattan actually won when rerun).
    - The "inverted-U" in k (figure not shown; the sweep is k = 1–25, not 1–15).
    - Distance weighting "softens" the vote (it sharpens it).
    - "Too many PCA dims re-introduce concentration" (based on a 0.16-pt plateau).
    - RMSprop "second-order" (it is a second *moment*).
37. **Section 6.3 has a confounded augmentation conclusion.** The −5.42 run applied flip, shift and erase together, to queries only, with short training. "Where augmentation is applied matters" cannot be concluded from it. "The student" (l.544) is the wrong model.

## 4. Graphs and tables (presentation)

38. Only **3 of 15** figures are used (03, 04, 12). Figures 06, 09 and 10 would back up claims the paper makes; 09 and 10 contradict them, which is a problem in itself.
39. **Fig. 1(a)** caption: "dotted line marks d=64". It doesn't; the figure has a dashed line at 90 % variance (d ≈ 84).
40. **Fig. 1(b)** caption says *test* accuracy, but the plot and the code use **validation**. As written, the caption suggests d was chosen on test data. The d=84 point is plotted after d=128, so the line zig-zags. The y-axis is truncated with no note.
41. **Print size:** at 0.38\textwidth, figure text is about 4 pt and the confusion-matrix cell numbers about 3 pt, which can't be read. Use 0.48\textwidth or more, or regenerate the figures at print size. Panels have no (a)/(b) markers.
42. **Fig. 3** has no `\label` and is never referenced. Section 6.2 "Final Evaluation" is an empty heading, and the figure floats onto the Conclusion page.
43. **Table 5** is never `\ref`'d. Line 527 ends in a colon, so the PDF reads "same rigour: Two lessons follow."
44. **Table 1** "Latency (s/batch)": batch size undefined. **Table 2** says CPU but mixes CPU and GPU timings.
45. **Table 4** rows are neither monotone nor chronological: the E22 fusion at 88.73 is listed after the student at 89.03. Each technique has two or three different names.
46. Figure problems in figures/README and the unused figures:
    - Fig. 11 train accuracy is inflated (item 27).
    - Fig. 05's axis says "1k val, median of 3", but 6k queries were used.
    - Fig. 14 is described as a prediction grid that doesn't exist.
    - Fig. 02 has no class labels.
    - Colours are inconsistent across figures; README l.272 swaps SGD/RMSprop colours.
    - `12_confusion_matrix.png` is overwritten by every fusion stage; it matches E25 only because E25 ran last.

## 5. Writing: sentences are not simple

The paper breaks the "simple and easy to understand" rule in most paragraphs. `03_writing.md` gives a rewrite for every flagged sentence. The main patterns:

- **Long sentences.** The abstract averages about 38 words per sentence, and one sentence (l.130–140) is about 100 words. Others: l.554–561 (85 words), l.573–580 (75), l.293–297, 312–316 and 561–565 (55 each). Split every sentence over about 25–30 words; one idea per sentence.
- **Semicolon chains and stacked asides** (l.125–127, 151–155, 195–198, 332–335). Turn them into separate sentences.
- **Inflated words to replace:** champion, ladder/rung, leaderboard, Grand fusion, shipped, Verdict, systematic, principled, archetypal, canonical dissimilarities, spectral energy, operating point, binding constraint, foreshadow, "well calibrated" (misused), "test boundary".
- **Acronyms used without being spelled out:** PCA, MoCo, TTA (used in Fig. 2 before it is defined), MLP, SGD, CE, KL, Optuna, hflip, gen-1/gen-2, MNIST.
- **Undefined or reused symbols:**
  - Undefined: N, C, W, W_o, η, λ, ε, σ, p_T, q_θ.
  - α means two things.
  - d means three things (distance, PCA dims, per-class distance).
  - k means two things (total neighbours vs per class).
  - T and τ are both "temperature" and both 1.0.
  - Eq. 7's ‖·‖₂ notation is invalid.
- **Inconsistent terms:**
  - British and American spelling mixed (optimisation/optimization, neighbour/neighbor, artefacts/artifacts).
  - KNN / k-NN / K-Nearest Neighbors.
  - 784-D / 784-dimensional / 128-dim.
  - Differences written as "points", bare numbers or "gain": pick "pp" and define it once.
- **Order:**
  - Soft-KNN results appear (Table 2, l.269) before Soft-KNN is defined.
  - The optimizer study mentions convolutional filters before the conv backbone is introduced.
  - The pipeline overview (5.1) comes after the stages it summarises.
  - The MLP and bank-augmented models appear in the fusion with no introduction.
  - There is no Related Work section and no Limitations paragraph.
- **Self-praise to delete:** "Honest negative results", "same rigour", "fully reproducible", "competitive modern vision system".
- **Title** (33 words) → e.g. *"Improving k-Nearest Neighbors on Fashion-MNIST with Learned Distances"*. A plain ~150-word abstract is drafted in `03_writing.md` §R.

## 6. Bibliography, title page and LaTeX

47. `cover1967`, `he2020` and `hinton2015` are in the bibliography but **never cited**, at exactly the places they are needed (l.149, 175/318, 425). As a result the in-text numbers jump [1], [5], [6], [7, 8], [10], [9].
48. No citations for Optuna, RMSprop, Nesterov SGD, TTA, PyTorch or scikit-learn. "NeurIPS 17" should be "NIPS 17" for 2004. Some page numbers are missing, and title capitalisation is mixed.
49. **The member PDFs (2–4) show a blank underline instead of an email** (line 98). The source header (l.2–4) names Akshar as the author in all four files. No PDF says it is team work, gives a course or instructor, or names the other members. Four single-author PDFs of identical text may read as duplicated work, so add a team or contribution statement.
50. Repo URLs are repeated (title page and §8), and long raw URLs break mid-word in print ("workforak ng").
51. About 40 lines of dead pandoc/highlighting macros, with a comment pointing to an "Appendix A/B" that doesn't exist. Duplicate `longtable`/`booktabs` loads; unused packages.

## 7. Code and reproducibility

52. `requirements.txt` pins `torch==2.14.1+cu130`, which is not on PyPI, so `pip install -r` fails. `torchvision`, `seaborn` and `jupyter` are missing.
53. `setup_env.sh` hard-codes `cd /home/akng/Downloads/Fashion` and requires Python 3.14. The paper says 3.11+, the README 3.10+.
54. `scripts/smoke_kernel.py` depends on `/tmp/opencode/kg-kernel/script.py` and the gitignored `artifacts/`, so it fails 2 of 6 checks. `scripts/smoke.py` passes.
55. **The ledger can't be regenerated as it stands.**
    - Stage skipping ignores `--scale`.
    - Tuned values are kept only in memory and fall back to defaults in a new session (E11 used Euclidean, not the Optuna winner).
    - E21 and E23 reused artefacts trained elsewhere (Kaggle, 35-epoch AMP); the paper never mentions Kaggle.
    - `_collect_sources` adds whatever artefacts exist, so rerunning E22/E24/E25 today gives different results.
    - `kaggle_conv_seeds.json` reports different accuracies for the same seeds (87.44 / 87.25 / 87.54).
    - E27 "definitive fusion" exists in the code but not in the ledger.
    - E10 and E11 are logged twice.
56. Early stopping compares against a stale bank (`src/train.py:296`). There is no deterministic cuDNN mode, and AMP is on.
57. Latency benchmark: no warm-up, validation queries, thread count set by an env variable. "Median of 3" is not true for the tree-based searches.
58. The `train_distill.py` docstring states the KL direction backwards (the code is correct). E15 has dead variables and a wrong source label. E10 and E12 crash on non-linear artefacts.
59. READMEs: they list functions that don't exist, the 50k/10k split, k=7 cosine, AdamW, 64-D, "5-Model" E22 (it has 6 sources), 7.14×, and 85.3 %.

**Checked and correct:**
- IDX parsing and label alignment; the stratified 54k/6k split; PCA fitted on the training split only.
- The per-class top-k logits; the KL direction in the code; TTA applied only at val/test.
- No test images in any bank or fit set; fusion weights chosen on validation.
- The E24/E25 weights and metrics; seeds 42/7/123 = 87.53/86.92/87.57; the deltas −5.42, −3.01, −2.03, +0.67, +4.27.

---

## Suggested fix order
1. Regenerate Tables 1, 2 and 4 and every in-text number from `results.json` (items 1–12, 14–19). Delete anything that cannot be traced.
2. Rewrite or remove the optimizer study and the τ claims so they match E9/E8, and include figures 09 and 10 if you keep them.
3. Fix the reproducibility claims: make the Colab notebook call `src/`, or say plainly that it is a simplified demo. Disclose the Kaggle runs.
4. Remove the claims that don't hold (items 25–31, 35–37). Add NCA-style related work and ± or significance testing.
5. Fix the figures: correct captions, print at a readable size, label and reference Fig. 3 and Table 5.
6. Simplify the language using the rewrites in `03_writing.md`, and fix the author block on the member PDFs.
7. Fix the code issues: self-match exclusion, requirements, setup script, smoke script.

`docs/audit/checks/` contains the scripts used for the reruns: the self-match A/B test, the teacher self-match check and the metric recheck. Each expects a local copy of the data.
