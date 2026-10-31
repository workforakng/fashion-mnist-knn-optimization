# Audit 05 — Technical statements, math and scientific claims (docs/paper.tex)

Scope: every equation, technical claim and argument in `docs/paper.tex`, cross-checked against `src/`, `scripts/`, `results/*.json`, `README.md` and `fashion_knn_colab.ipynb`. No repo files were edited.

Reference statistics used below: with n = 10,000 test images, the binomial standard error of accuracy is about 0.31 pt at 89.5 %, 0.33 pt at 87.6 % and 0.36 pt at 85.2 %, so the 95 % CI is about ±0.6 pt. Any single-run difference under about 0.6 pt cannot be called a gain without a paired test.

---

## CRITICAL

### C1. The reproducibility claim is false: the Colab notebook runs a different model, split and protocol
- **Lines 142–143, 570–581:** "the complete pipeline reproduces end-to-end from a single standalone Google Colab notebook"; "the complete pipeline ... as eleven executable cells".
- **Evidence in `fashion_knn_colab.ipynb`:**
  - The `DifferentiableSoftKNN` class (cell 6) is a **parametric prototype network**: a CNN with BatchNorm, GELU, Dropout and an extra FC layer, scored against 10 learnable `nn.Parameter` prototypes. It has no memory bank, no nearest-neighbour retrieval and no per-class top-k. That is the very kind of "parametric prototype method" the paper says Soft-KNN is not (l. 296).
  - The split is 50k/10k/10k via `random_split` (paper: stratified 54k/6k). Training uses `train_aug_dataset = FashionMNIST(train=True)`, all 60k images, so **the notebook trains on its own validation images**.
  - The classical KNN cells use a 10k-image training subset and a **2k-image test subset**.
  - There are no seeds 7/123, no distillation, no TTA and no convex fusion. The 89.51 % figure is hard-coded in a markdown table, which also says "5-Seed" ensemble and "State-of-the-Art".
- **Fix:** Either make the notebook call `src/experiments.py` and the `scripts/` (stages E1–E25 on the 54k/6k/10k split), or remove the claim. State that the reported numbers come from `src/` plus Kaggle GPU runs (`results/kaggle_*.json`). The paper never says that the 89.03 % student and its teachers were trained on Kaggle.

### C2. The optimizer study contradicts the ledger and contains invented results and a formula error
- **Lines 366–402.**
  - The ledger entry E9 (`results.json`) has **SGD as the winner**: best val 83.85 % vs Adam 81.47 % vs RMSprop 80.83 %, `"best": "sgd"`. The paper's "Verdict. Adaptive methods converged fastest and smoothest" reverses the logged result.
  - "SGD ... η = 0.01, γ = 0.9", but `src/experiments.py:e9_optimizers` uses `lr=0.1`, and `make_optimizer` uses plain momentum with no `nesterov=True`. The "roughly 3× more epochs" claim and the RMSprop "gradient thrashing" claim have no logged evidence.
  - "anchor coordinates updated too slowly relative to the convolutional filters" (l. 388–389) is conceptually wrong for this implementation. The anchors are detached bank embeddings with **no learnable coordinates** (`embed_chunked` runs under `@torch.no_grad`). E9 also used the *linear* projector, not a conv backbone. This sentence describes the notebook's prototype model, not `src/`.
  - The GELU/ReLU/LeakyReLU "activation study" (l. 400–402) does not exist in `src/` or the ledger. It appears only in the notebook's prototype model.
  - **Nesterov formula sign error (l. 385–386):** with θ_t = θ_{t-1} − v_t, the look-ahead point must be θ_{t-1} − γ v_{t-1}, not θ_{t-1} + γ v_{t-1}.
  - E9 was run with the hflip+shift+erase augmentation that the paper itself reports as harmful. That makes it a weak basis for any optimizer conclusion.
- **Fix:** Report E9 as logged (SGD at lr 0.1 was best under that config) or rerun it on the final conv config with multiple seeds. Correct the NAG formula. Remove the activation study and the "anchor coordinate" argument.

### C3. The latency and "92× speed-up" claims are unsupported, internally inconsistent and mechanistically false
- **Lines 268–272, Table 2 (l. 274–287).** The text says "costs 0.02 s per batch, a 92× speed-up ... because the discriminative embedding makes short-list candidate pruning maximally effective."
  - **No candidate pruning exists.** `SoftKNN.logits_from_z` computes a full `torch.cdist` against all N = 54,000 bank embeddings, then runs `topk` per class. Inference is brute force, O(N·d) with d = 128, plus a conv forward pass.
  - On equal hardware, a 128-D brute-force search plus a CNN forward pass cannot be about 13× faster than a 64-D brute-force search (0.26 s). The 0.02 s figure is plausibly GPU time compared against CPU sklearn time, yet Table 2's caption says "(CPU)".
  - No timing code for Soft-KNN exists in `src/` or the ledger. E5 has only sklearn timings: raw 2.98 s and PCA-64 0.418 s per 6,000 queries, which gives 7.12×, not 1.84 s → 0.26 s.
  - The same conv model is listed at **0.02 s** in Table 2 but **0.05 s** in Table 4 (E12).
  - "per query batch" is never defined: batch size, n_jobs and hardware are all missing.
  - O(N·D) (l. 266) is correct for brute force. The point is that Soft-KNN keeps the same asymptotic cost.
- **Fix:** Time all methods on the same device and the same query set (state the batch size, CPU/GPU model and thread count), and log the timings. Delete the "pruning" explanation. Report the 0.29 s cost of the champion (TTA ×6 views × several models), not the single-model cost, next to the 89.51 % headline.

### C4. Self-matching in the training bank (no leave-one-out) and label leakage into the distillation teachers
- **Lines 297–313 (training objective), 427–436 (distillation), 546–549 (gen-2).**
  - In `src/train.py`, queries are drawn from `x_fit` and the bank is the embedding of the *same* `x_fit`. `per_class_topk_logits` never excludes the query's own index. Each training query therefore always has itself (a stale copy at near-zero distance) among its own-class top-k.
  - NCA (Goldberger 2004) explicitly sets p_ii = 0 to avoid this. The objective actually optimised is not the one in Eq. (5). It also explains train accuracy of about 0.95 against val accuracy of about 0.88 (E6, E19 histories).
  - **Teacher soft targets** (E23, `scripts/train_distill.py`) come from `predict_probs(m, bz, ya_t, xa, ...)`, i.e. Soft-KNN applied to the training points with themselves in the bank. The targets are biased toward the true label.
  - **Gen-2 teacher (E26)** includes `knn_classical` via `clf.predict_proba(za)` on the *training* points with `weights="distance"`. In sklearn, a zero-distance neighbour receives all the weight, so this component is **exactly one-hot ground truth**. Its bankaug and conv components at τ = 0.1 are also near one-hot.
  - This invalidates the mechanistic stories in l. 434–436 ("soft targets encode inter-class similarity structure") and l. 546–549 ("128-dim capacity ... is the binding constraint").
- **Fix:** Mask the self-index in training (leave-one-out) and when computing teacher targets on the training set. Use cross-fitted (out-of-fold) teacher predictions. Rerun E19/E21/E23/E26 and update every downstream number.

### C5. Novelty is misstated and key prior work is missing
- **Lines 174–175, 293–297:** "unlike those parametric prototype methods, however, the anchor set here remains the non-parametric training bank itself."
  - NCA (Goldberger et al. 2004) is **not** a parametric prototype method. It is a stochastic nearest-neighbour softmax over the training set, which is exactly what Soft-KNN is. LMNN (Weinberger & Saul 2009) is also defined over training-set neighbours. The contrast drawn is false.
  - The method is close to **Wu, Efros & Yu, "Improving Generalization via Scalable Neighborhood Component Analysis", ECCV 2018**, which combines a deep CNN, an NCA loss, a memory bank of training embeddings refreshed during training, and kNN inference. Its only differences are the per-class top-k mean and the choice of τ.
  - Further work the paper must cite and compare against:
    - Salakhutdinov & Hinton 2007 (nonlinear NCA)
    - Wu et al. 2018, "Unsupervised feature learning via non-parametric instance discrimination" (memory bank; CVPR 2018)
    - Movshovitz-Attias et al. 2017 (ProxyNCA)
    - Snell et al. 2017 (Prototypical Networks)
    - Vinyals et al. 2016 (Matching Networks)
    - Plötz & Roth 2018 (Neural Nearest Neighbors Networks; continuous kNN relaxation)
    - Frosst, Papernot & Hinton 2019 (soft nearest-neighbour loss)
    - Papernot & McDaniel 2018 (Deep k-NN)
    - Mitani & Hamamoto 2006 (local-mean-based nonparametric classifier; this is exactly the per-class k-mean-distance decision rule, see M1)
- **Fix:** Add a Related Work section. Reframe Soft-KNN as "an NCA-style objective with a per-class k-local-mean rule, in the style of Wu et al. 2018" and claim only what differs.

### C6. "Every number traceable to a logged experiment" is contradicted by the ledger
- **Lines 141, 167–168, 195–198, 479–482, 561 vs `results/results.json`:**
  - E01 val "---", but the ledger has 86.13 %. E01 latency 1.84 s, but the ledger has 3.19 s (E1) and 2.98 s (E5).
  - E02/E03 (Manhattan, cosine) are not in the ledger. They exist only in the notebook, on a 10k-train / 2k-test subset.
  - E04 (d = 16) was never run (the ledger has d ∈ {10, 20, 32, 48, 64, 80, 96, 128}). Test accuracies for E04–E07 are not logged; only val is. Logged val values differ, e.g. d = 32 is 86.72 (table 85.10) and d = 64 is 87.33 (table 86.60).
  - E08 is described as "k = 7, dist. weights", but the ledger E3 best is **k = 5, Manhattan, distance**, val 87.82 (table 86.85).
  - E09 is described as "784→64, τ = 0.1", but E6 config is embed_dim 128 with training τ = 0.5. §4 itself says z ∈ ℝ¹²⁸ (l. 323).
  - E10 is described as "784→256→64", but E16 config is 784→256→128.
  - Val column mismatches: E12 88.10 vs ledger 89.07; E14 88.50 vs 89.72; E15 89.20 vs 90.10; E16 89.95 vs 90.27.
  - The table's E13 is the conv-shift run (ledger E20), while ledger E13 is the linear shift run (test 83.66). E16–E18 are relabelled ledger E22/E24/E25. The "28 entries" include two re-runs of E10/E11 with different source models.
- **Fix:** Regenerate Table 4 and Table 1 programmatically from the ledger. Use ledger IDs. Mark any number that does not come from `src/` (notebook or Kaggle) as such.

---

## MAJOR

### M1. The "exactly KNN as τ→0" claim is wrong; τ has no effect on predictions
- **Line 310:** "Equation (5) is exactly KNN in the limit τ→0".
  - For any τ > 0, argmax_c P(y=c|x) = argmin_c d_c(z). The decision is **τ-invariant**, and τ only rescales confidence. The ledger confirms this: in E16, E19 and E23 the val accuracy is identical across τ ∈ {0.1, 0.3, 0.5} for each k.
  - The limiting rule is the *per-class k-local-mean distance* classifier (Mitani & Hamamoto 2006), not majority-vote k-NN. It equals k-NN only for k = 1.
  - Since k is chosen *per class*, k = 9 in Soft-KNN examines 90 neighbours, while k in classical KNN means k neighbours in total. The same symbol is used for both without comment.
- **Fix:** State that the hard limit is argmin_c d_c (a local-mean rule) and cite Mitani & Hamamoto. Say that τ affects only training dynamics and probability calibration (and so fusion and distillation), not the single-model argmax.

### M2. Gradients do not flow "through the top-k selection" or through the anchors
- **Lines 311–312:** "non-vanishing gradients through both W and the top-k selection."
  - `topk` indices are piecewise constant, so no gradient flows through the selection. Only the selected distance values are differentiable.
  - The bank is computed under `torch.no_grad` and detached (`embed_chunked`), so gradients flow only through the **query** embedding, never through the anchors z_i.
- **Fix:** Rewrite as "gradients flow through the selected query-to-anchor distances into the query encoder; anchor embeddings are treated as constants between refreshes." For a genuinely differentiable selection, cite Plötz & Roth 2018 or differentiable top-k / sorting (Xie et al. 2020; Grover et al. 2019).

### M3. The τ sweep is misreported, and the selected τ was never used for training the final models
- **Lines 314–316.**
  - "validation sweep over τ ∈ [0.01, 1.0] ... τ < 0.05 saturates gradients": E8 swept only {0.1, 0.2, 0.3, 0.5, 0.7, 1.0}. No τ < 0.1 was tested, so the τ < 0.05 claim is invented.
  - E8 used the hflip+shift+erase augmentation (the harmful config) and half the epochs.
  - All final conv models (E19, E21 seeds, E23 student, E26) were **trained with τ = 0.5**. τ = 0.1 is used only at inference, where it cannot change the argmax (M1).
  - A correct mechanism is available. Embeddings lie on the unit sphere, so d_c ∈ [0, 2] and the logit range is bounded by 2/τ. At τ = 0.5 the maximum attainable class probability is 1/(1 + 9e⁻⁴) ≈ 0.86, so cross-entropy cannot go below about 0.15. That is the real reason a large τ under-trains.
- **Fix:** Report the sweep actually run and its config. Rerun the sweep on the final conv config. Use the bounded-logit argument. Either train the finals at the chosen τ or explain why they were not.

### M4. "MoCo-style memory bank" is a misnomer, and the compute arithmetic is wrong
- **Lines 175, 318–321.**
  - MoCo (He et al. 2020) uses a momentum-updated key encoder plus a FIFO queue. The implementation is a detached bank that is periodically fully recomputed. That is the memory-bank idea of Wu et al. 2018, without momentum.
  - The bib entry `he2020` exists but is **never cited**.
  - "Recomputing all N embeddings every step doubles the backward cost ... halving compute" is wrong. A full recompute each step costs N/B ≈ 211 extra batch forwards per step (plus backward through the bank if it were not detached), not 2×. Refreshing every 107 steps adds about 2 full-bank forwards per epoch.
  - The `src/softknn.py` docstring says "refreshed once per epoch". The linear model E6 (the reported 87.10 %) indeed used per-epoch refresh (211 steps); only the MLP/conv runs used 107.
- **Fix:** Call it a "periodically refreshed (stale) memory bank", cite Wu et al. 2018 (and He et al. 2020 only for the contrast), give the correct cost ratio, and state the refresh interval for each experiment.

### M5. The distillation equation is malformed, and its mechanism claim is untested and probably wrong
- **Lines 427–436.**
  - **Notation:** σ is undefined (usually the sigmoid). p_T is a *probability* (the mean of teacher softmaxes), so σ(p_T/T) applies a softmax to probabilities, which is not what the code does: the code uses p_T directly as the target. The CE term uses p_θ but the KL term uses q_θ.
  - The distillation temperature T is conflated with the Soft-KNN temperature τ. The teacher's softness comes from τ = 1.0, and T = 1 makes the T² factor vacuous.
  - KL direction: the code computes KL(p_T ‖ q), which matches the paper, but `scripts/train_distill.py` documents it as "KL(student || teacher)".
  - **Mechanism:** at τ = 1 with d_c ∈ [0, 2], the teacher's maximum attainable confidence is 1/(1 + 9e⁻²) ≈ **0.45**. The soft targets are therefore heavily smoothed, near-uniform distributions. With α = 0.5 the gain is at least as plausibly a **label-smoothing / regularisation effect** (Müller et al. 2019; Yuan et al. 2020, "Revisiting KD via label smoothing regularization") as "inter-class similarity structure".
  - The student also differs from the teachers in seed (100), AMP and epochs, and it is a single run.
  - `hinton2015` is in the bibliography but is never cited.
- **Fix:** Write L = (1−α) CE(y, softmax(s)) + α T² KL(p_T ‖ softmax(s/T)) with s = −d/τ_s, and define p_T = mean_s softmax(−d^{(s)}/τ_T). Cite Hinton et al. 2015. Add controls: (a) uniform label smoothing with matched entropy, (b) a non-distilled student with identical config and seed, and (c) ≥3 student seeds. Cite Furlanello et al. 2018 (Born-Again Networks) for "student exceeds teacher".

### M6. The gen-2 distillation and capacity conclusions are unsupported
- **Lines 546–549, 563–564:** "indicating the 128-dim embedding capacity—not teacher quality—is the binding constraint" and "capacity proved the binding constraint".
  - No experiment varies the embedding dimension.
  - The gen-2 teacher differs from gen-1 in composition (it includes one-hot leaked classical-KNN targets, C4) and in temperature (component τ = 0.1, near one-hot, vs gen-1 τ = 1.0, near-uniform). The obvious explanation is that gen-2 lost the label-smoothing effect.
  - It is a single run with no seed variation.
  - "feeding the gen-1 student back" is also inaccurate: the gen-2 teacher is the E24 fusion (weights 0.2 conv_s42, 0.4 bankaug, 0.1 knn_classical, 0.3 conv_distill), not the gen-1 student alone.
- **Fix:** Downgrade to an observation. Fix the leakage. To make a capacity claim, sweep embed_dim ∈ {64, 128, 256}.

### M7. "Every rung adds exactly one technique" and "every rung individually ablated" are false
- **Lines 335–337, 481–482, 560–561.** Confounded steps:
  - E08→E09 changes the model family, embedding dim (64→128), training τ/k and optimizer all at once.
  - E11→E12 changes linear→conv, lr 1e-3→3e-4, epochs 20→30, patience 4→8, gradient clipping, bank refresh 211→107 and removes the PCA warm start.
  - E14→E15 changes ensemble→distillation, seed 100, AMP and 35 epochs.
  - E17→E18 adds TTA *and* re-optimises the fusion weights (E24→E25).
- Missing ablations, all essential:
  - (a) Hard majority-vote kNN on the learned conv embedding vs the Soft-KNN rule.
  - (b) The same conv backbone with a standard softmax linear head (parametric control).
  - (c) Fusion without the classical PCA-KNN source.
  - (d) TTA per source, with fixed weights.
  - (e) Leave-one-out vs self-included bank.
- **Fix:** Remove the claims or run the ablations as one-factor-at-a-time changes.

### M8. No uncertainty quantification or significance tests
- **Lines 214–217, 417–423, 451–452, 470, 536–539 and the abstract.** Examples:
  - Cosine vs Euclidean +0.26 pt (n = 10k, SE about 0.35 pt).
  - Bank augmentation +0.03.
  - TTA fusion over plain fusion +0.15 pt (15 images).
  - Fusion over distilled student +0.48.
  - Ensemble over best seed +0.49.
- All are within about 1–1.5 SE and come from single runs. The seed spread alone is 86.92–87.57 (0.65 pt range).
- **Fix:** Report mean ± sd over ≥3 seeds for every learned rung, 95 % CIs, and paired McNemar tests (or a paired bootstrap) for adjacent rungs. Remove words like "counter-intuitively outperforms" unless the difference is significant.

### M9. Test and validation reuse (adaptive analysis) contradicts the stated protocol
- **Lines 192, 195–198, 447–448, 470, 560.**
  - "the test set is touched once per reported number" sits alongside a ledger that computes test accuracy at every stage (E1–E26). Design decisions were made sequentially after those results were visible: E10/E11 were re-run with a different source model, and E13's docstring reads "flip/erase proved harmful in E7".
  - The same 6,000 validation images serve PCA dim selection, Optuna (40 trials), early stopping of every network, (k, τ) inference grids, and an exhaustive simplex search over C(16, 6) = 8,008 weight vectors.
  - The headline "90.85 % validation" is therefore a heavily selected, optimistically biased figure and should not be reported as a performance estimate.
  - In fairness, the val–test gap of the fusion (1.34 pt) is similar to that of single conv seeds (about 1.5 pt), so fusion over-fitting is not obviously large. But the val numbers are not unbiased.
- **Fix:** Freeze the full pipeline, then evaluate test once. Report other stages on val only, or use a separate held-out "fusion-val" split (or nested CV) for fusion weights. Drop "(90.85 % validation)" from the abstract and conclusion.

### M10. Arithmetic and claim errors in the reported gains
- **Line 452:** "+3.08 over the strongest single learned model". 89.51 − 3.08 = 86.43, which is the *classical tuned PCA-KNN*, not a learned model. The strongest single learned model is the distilled student at 89.03, so the gain is **+0.48**.
- **Lines 417–423:** "+0.43 gain over the best single seed". The best seed listed is 87.57 (seed 123), so the gain is **+0.49**. 87.63 belongs to the separate E19 run, not the ensemble's seed 42 (87.53).
- **Line 423:** "with zero additional inference-time training" is meaningless. The ensemble costs 3× training and 3× inference, as l. 426 itself admits.
- **Lines 252, 556:** "PCA-64: +1.19 points". 86.43 includes Optuna tuning (Manhattan, distance weights). By the paper's own table, PCA alone (E06, k = 5 Euclidean) gives +0.96.
- **Lines 434–436:** "exceeding every individual teacher" uses one student run.
- **Fix:** Correct the numbers and attribute each gain to its actual source.

### M11. Framing: "competitive modern vision system" and "never departing from the NN paradigm"
- **Title and lines 129–130, 554–555.**
  - The champion is a convex blend: 0.5 distilled conv Soft-KNN, 0.4 linear bank-augmented Soft-KNN and 0.1 PCA-KNN, each with 6-view TTA.
  - Every source is a neighbour-based classifier over an embedding, so "KNN family" is defensible. But the conv embedding is trained discriminatively with supervision, so the system is really *supervised deep metric learning + local-mean kNN*.
  - **No parametric baseline** with the same backbone is reported, so it is unknown whether the kNN head helps or hurts.
  - At 89.51 % the system is about level with the RBF-SVC on raw pixels in the Fashion-MNIST benchmark (≈89.7 %, Xiao et al. 2017). It is below small 2–3-layer CNNs (roughly 90–93 %) and far below ResNets (≈94–96 %) listed in the zalandoresearch/fashion-mnist benchmark table (verify the exact figures before citing).
  - "Competitive modern vision system" is therefore an overclaim, and so is "Systematic Optimization" in the title (see M7).
  - The README and notebook also call it "State-of-the-Art KNN boundary performance" without any citation.
- **Fix:** Add the (b) parametric-head control and the (a) hard-kNN-on-embedding control from M7. Cite the Xiao 2017 baselines. Reword to "a kNN-based classifier on a learned metric reaches 89.5 %, comparable to RBF-SVM and below standard CNNs".

### M12. Curse-of-dimensionality claims are asserted, not measured
- **Lines 124–126, 152–155, 244–246, 259–261, 556.**
  - "every point becomes an almost-equidistant outlier" and "d_max/d_min → 1": Beyer et al.'s result holds under specific conditions (e.g. i.i.d.-like dimensions). Image data has a low intrinsic dimension, and raw-pixel 5-NN already reaches 85 %, which shows that neighbours remain meaningful.
  - The Figure 1(b) caption says "too many re-introduce distance concentration". The ledger shows val 87.33 (d = 64) vs 87.17 (d = 128), a 0.16 pt difference that is within noise, and no concentration statistic was measured. PCA's gain is equally explained by denoising.
  - The Manhattan explanation ("axis-aligned penalty insensitive to diagonal stroke structure", l. 217–218) is unfounded. Aggarwal, Hinneburg & Keim 2001 ("On the surprising behavior of distance metrics in high dimensional space") predict L1 to be *more* robust than L2 in high dimensions.
  - The cosine "lighting-intensity" explanation (l. 215–217) is an untested hypothesis attached to a +0.26 pt, non-significant difference (M8).
- **Fix:** Measure relative contrast (d_max − d_min)/d_min and the nearest/farthest-neighbour distance ratio for raw, PCA-d and learned spaces, and estimate intrinsic dimension. Cite Aggarwal et al. 2001. Phrase explanations as hypotheses.

### M13. TTA claims are unablated and contradicted by the logged evidence
- **Lines 441–443, 180–181.** The text says "reducing input-level variance at zero training cost".
  - No per-source TTA ablation exists, and E25 vs E24 confounds TTA with re-optimised weights.
  - Ledger E10 (second run, base linear model) shows TTA **hurt**: val 88.22 → 87.85 and test 87.10 → 86.74.
  - The TTA views include a horizontal flip, although hflip augmentation in training was reported as harmful (Table 5). The rationale is inconsistent.
  - The `src/experiments.py:e25_tta_fusion` docstring lists the views as "identity, ±1 px horizontal, ±1 px vertical" (no flip), while `tta_views` includes the flip.
  - "Zero training cost" hides a 6× inference cost.
- **Fix:** Ablate TTA per source with fixed fusion weights, report the E10 result, and justify (or drop) the flip view.

### M14. The negative results are confounded and partly mislabelled
- **Table 5 and lines 526–549.**
  - "Memory-bank augmentation +0.03" sits in a table captioned "Techniques that *degraded* accuracy". Yet this source carries the **largest weight (0.4)** in both champion fusions.
  - "corrupting the training batches the *student* sees" (l. 544): E7 has no student.
  - The augmentation failure is confounded by four things:
    - (i) Augmentation is applied only to queries while the bank stays clean. A train/bank mismatch is created, and the self-match shortcut (C4) is removed, so the comparison is not like-for-like.
    - (ii) hflip, shift and erase were applied jointly.
    - (iii) Training was short (20 epochs, patience 4); E20 stopped after 13 epochs with train accuracy 0.78, i.e. under-trained.
    - (iv) Each result is a single run.
  - Shift augmentation is known to help CNNs on Fashion-MNIST, so a strong negative claim ("harmful") needs these confounders removed.
- **Fix:** Retitle Table 5 to "techniques that did not improve accuracy". Ablate the augmentation types separately, train to convergence, and augment the bank and queries consistently.

### M15. Training-recipe statements contradict the code
- **Line 412:** "AdamW-style settings", but `make_optimizer` uses `torch.optim.Adam` with **no weight decay**, so the Eq. (8) decay term is never active.
- **Line 398–399:** "all final models in this report use Adam at 3×10⁻⁴". The bank-aug linear model (weight 0.4 in the champion) and E6 used 1e-3, as l. 324 itself states.
- **Line 413:** "early stopping (patience 10)", but E19 (the 87.63 % model) used patience 8 and 30 epochs. The seeds used patience 10 and 35–40 epochs, and seed 123 stopped at 14 epochs.
- "mixed precision": AMP was used for the seeds and the student but not for E19.
- **Fix:** Add a per-experiment hyperparameter table generated from the ledger configs.

### M16. Contradictions between the paper and README.md
- **Soft-KNN definition:** the README describes "learnable class prototypes" P = {p_1..p_C}, a λ‖θ‖² loss, and inference as "matrix products against 10 learned prototypes". The paper describes a non-parametric training bank and explicitly contrasts itself with prototype methods.
- **PCA:** README says 85.3 % variance at d = 64; paper says ≈86 %.
- **Speed-up:** README 7.14× vs paper 7.08×.
- **Tuned KNN:** README "k = 7, Cosine"; paper "k = 7, dist. weights"; ledger k = 5, Manhattan, distance.
- **Optimizer and embedding:** README says AdamW wd 1e-4 and a 64-D embedding; the code uses Adam with no wd and 128-D.
- **Bank size:** README says "50,000 training points"; the paper and code use 54,000.
- **Augmentation:** README "convolutional backbone paired with regularized spatial stochastic augmentations" vs paper Table 5, where these hurt.
- **Distillation:** README says "multi-teacher distillation"; the 5-teacher run (`kaggle_distill_5teacher.json`, 89.01 %) is not mentioned in the paper.
- **Fix:** Pick one authoritative description (the `src/` implementation) and align the README and notebook to it.

### M17. The Optuna description is incorrect
- **Line 253:** "Bayesian optimisation over {k, d, metric, weights}".
  - d is not searched. It is fixed at 64 from the E2 sweep.
  - The search space also includes Chebyshev and Minkowski with p ∈ [1, 3], which the paper never mentions or defines (no Minkowski formula is given alongside Eqs. 1–3).
  - The selected metric (Manhattan) is not reported in the text.
- **Fix:** State the actual space (k ∈ {3, 5, …, 15}, weights, metric ∈ {L2, L1, L∞, Minkowski p}), TPE with 40 trials, and the selected configuration. Add the Minkowski definition d_p = (Σ|x_i − z_i|^p)^{1/p} if it is kept.

---

## MINOR

1. **l. 240–241:** "Distance weighting (w_i ∝ 1/d_i) further softens the vote." Inverse-distance weighting *sharpens* the vote toward the nearest neighbours (lower effective k, higher variance). Fix the wording.
2. **l. 238:** "k ∈ {1, 3, …, 15}", but the ledger E4 sweeps 1–25. The ledger "inverted-U" peak (k = 7: 87.37 vs k = 5: 87.33) is flat within noise. Say so.
3. **l. 199:** "macro-averaged metrics are well calibrated" misuses "calibrated". Under perfect balance, macro recall equals accuracy (visible in Table 4: Rec = Test). Say "macro and micro metrics coincide/are comparable".
4. **l. 259–261 (Figure 1 caption):**
   - "(a) ... the dotted line marks the d = 64 operating point", but the code (`e2_pca_sweep`) draws a dashed horizontal 90 %-variance line plus a vertical line at the best dimension.
   - "(b) Test accuracy versus retained dimensionality", but the sweep plots *validation* accuracy.
   - The "≈86 %" variance figure is not logged (only var90_dim = 84 is). Log `explained_variance_ratio_[:64].sum()`.
5. **Notation overloads:**
   - d is used for distance (d_1, d_2, d_cos, d_c) and for PCA dimension.
   - k is both the global KNN neighbour count and the per-class anchor count.
   - α is the RMSprop decay and the distillation weight.
   - T (distillation) vs τ (Soft-KNN).
   - C (number of classes) and σ are undefined.
   - N is used for the bank size and for training-set size inside O(N·D).
   - W is the linear projector in Eq. (5) (column convention, Wx), while Eq. (9) uses the row convention vec(·) W_o.
   - Eq. (9) omits ReLU/MaxPool and writes ‖·‖₂ ambiguously. It should be z = f_θ(x)/‖f_θ(x)‖₂ with f_θ defined.
6. **l. 392–395:** RMSprop "second-order tracking" should be "second-*moment* (uncentred variance) tracking". RMSprop is not a second-order method.
7. **l. 323:** "PCA warm-started": `MetricProjector` copies `pca.components_` but does not subtract the PCA mean, so it is not the PCA projection. State that it initialises from the PCA directions only.
8. **l. 304:** The abstract (l. 135) says "softmax over distances". It should be over *negative* distances (as in Eq. 5).
9. **l. 446–448:** The fusion breaks ties by first occurrence (strict `>`) in an enumeration of 8,008 grid points. With step 0.1 and 6k validation images, report the full set of near-optimal weight vectors to show that the weights are not arbitrary. The sources also have very different calibration (τ = 0.1 near one-hot Soft-KNN vs sklearn vote fractions), so the weights are not comparable across sources. Consider temperature-calibrating each source first. Cite Wolpert 1992 (stacked generalisation) / Breiman 1996 for convex combination.
10. **l. 593:** `cover1967` is listed but never cited. Cite it where the KNN rule is introduced (l. 149–151), and give the asymptotic error bound if it is used.
11. **Validation inside training (`src/train.py:296`)** uses the **stale** `bank_z` from the last refresh (up to a full epoch old for the linear model) while queries use the current weights. Early-stopping selection is therefore made on a mismatched bank, which affects model selection. Re-embed the bank before validation.
12. **Ensemble Eq. (10):** the index set of s is undefined. Write s ∈ S = {42, 7, 123}.
13. **Contribution list (l. 183–184):** "second-generation distillation degrade[s]" is better framed as "did not improve" (see M6).
14. **l. 160:** "split into 60,000 development and 10,000 test images" is correct. Note the Colab split differs (C1).

---

## Missing information needed to reproduce

- **Hardware:** CPU model and thread count (`KNN_NJOBS` = 4 by default), GPU model, local vs Kaggle runs. Seeds and the distilled student came from Kaggle (`results/kaggle_*.json`), which the paper never says.
- **Wall-clock time** per stage (available in the ledger `wall_s`) and the definition of "per query batch".
- **Per-model hyperparameters:**
  - epochs and the epoch at which early stopping fired (seed 123 stopped at 14 vs 35);
  - patience;
  - training (k, τ) = (7, 0.5) vs inference (k, τ) = (9, 0.1) or (5, 0.1);
  - bank refresh interval;
  - AMP;
  - gradient clipping;
  - lr per model;
  - PCA warm start (linear only);
  - Optuna trials and search space;
  - student seed 100, gen-2 seed 200;
  - the 5-teacher distillation variant.
- **Determinism:** `src/train.py` does not set `cudnn.deterministic`, and AMP is non-deterministic, so "fix the global seed" does not guarantee bit-reproducibility. Say so.
- **Software versions:** PyTorch, scikit-learn and Optuna versions (only "Python 3.11+, PyTorch (CUDA)" is given).
- **E27 ("grand fusion")** exists in the code but not in the ledger. State whether it was run and why E25 is the champion.

## Citations to add (summary)
- Wu, Efros & Yu, ECCV 2018 (scalable NCA with memory bank)
- Wu, Xiong, Yu & Lin, CVPR 2018 (memory bank)
- Salakhutdinov & Hinton, AISTATS 2007
- Movshovitz-Attias et al., ICCV 2017 (ProxyNCA)
- Snell et al., NeurIPS 2017
- Vinyals et al., NeurIPS 2016
- Plötz & Roth, NeurIPS 2018
- Frosst, Papernot & Hinton, ICML 2019
- Papernot & McDaniel 2018
- Mitani & Hamamoto, Pattern Recognition 2006
- Aggarwal, Hinneburg & Keim, ICDT 2001
- Hinton et al. 2015 (already in the bib, uncited)
- He et al. 2020 (already in the bib, uncited; only for contrast)
- Furlanello et al., ICML 2018 (Born-Again Networks)
- Müller, Kornblith & Hinton, NeurIPS 2019
- Yuan et al., CVPR 2020
- Lakshminarayanan et al., NeurIPS 2017 (deep ensembles)
- Wolpert 1992 / Breiman 1996 (stacking / convex combination)
- Cover & Hart 1967 (already in the bib, uncited)
- Dietterich 1998 / McNemar 1947 (paired tests)
