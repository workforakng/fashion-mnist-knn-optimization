# AGENTS.md — Autonomous Multi-Agent & Kaggle Execution Protocol

> **Project:** Fashion-MNIST K-Nearest Neighbors (KNN) & Soft-KNN Optimization  
> **Environment:** Linux (Ubuntu 24.04), Python 3.14 venv, NVIDIA GPU local / Kaggle Cloud T4/P100  
> **Target:** Push test accuracy towards >90.2% on Fashion-MNIST with rigorous academic documentation, standalone reproduction notebooks, and zero opaque scripts.

---

## 1. Operating Directives & User Constraints

1. **Offload Heavy Computation to Kaggle:**
   - All compute-heavy operations (deep metric training, distillation, multi-seed ensembles) must be executed on Kaggle via Kaggle CLI / remote kernels.
   - Do not saturate the local host CPU/GPU; preserve machine responsiveness and protect thermal limits.

2. **Detached Background Execution & Real-Time Log Visibility:**
   - Launch long-running jobs in detached background processes (e.g. `nohup`, background scripts, or Kaggle async jobs).
   - If an agent or session disconnects, terminates, or receives an interrupt/Escape, running jobs **must not crash**.
   - Always expose live progress logs and status directly in the user-visible chat/terminal so work status is never hidden.

3. **Fault-Tolerant, Pausable & Resumable Checkpointing:**
   - Every experiment, training loop, and evaluation pipeline must save progress incrementally to disk (`results/`, `artifacts/`).
   - Pipelines must detect existing checkpoints and resume seamlessly from the point of interruption without duplicate computation.

4. **Clean, Standalone & Academic Notebooks (No Obfuscation):**
   - **Never** inject opaque blobs, Base64 archives, or convoluted packaging scripts into public notebooks.
   - Notebooks must be readable cell-by-cell, loading Fashion-MNIST directly from Kaggle input (`/kaggle/input/fashionmnist`) or torchvision/open datasets.
   - Use clear, professional, academic terminology. Avoid confusing shorthands (clarify terms like `E01..E25` as numbered Experiment Iterations).

5. **Mandatory Pre-Flight Smoke Testing:**
   - Never push code or submit jobs to Kaggle without running a fast local or remote smoke test on a tiny slice of data.
   - Ensure imports, dimensions, metric computations, and export paths are verified before committing full compute resources.

6. **Storage Hygiene & Resource Auditing:**
   - Keep `.gitignore` updated: exclude `.venv/`, caches, and raw weights.
   - Maintain `storage.md` tracking all disk allocations, model weights, and virtual environment footprints.

---

## 2. Architecture & Algorithmic Ladder

- **Baseline:** Classical KNN with distance metrics (Euclidean, Cosine, Manhattan) across raw pixel space (784-D) and PCA dimensionality reduction.
- **Novelty 1 — Soft-KNN (Differentiable Metric Learning):**
  - Learnable projection head mapping 784-D inputs to a discriminative feature space ($d=64$ or $d=128$).
  - Temperature-scaled softmax over negative Euclidean distances for continuous, differentiable neighbor assignment.
- **Novelty 2 — Convolutional Metric Backbone:**
  - Lightweight ConvNet feature extractor preserving spatial 2D image topology prior to distance computation.
- **Novelty 3 — Teacher-Student Soft-KNN Knowledge Distillation:**
  - Transfer representations from deep teacher models into compact student metric heads for enhanced generalization.
- **Novelty 4 — Multimodal / Multi-Seed Ensemble Fusion:**
  - Probability-level convex combination across diversified seeds and architectures. Current best: **89.51%** test accuracy.

---

## 3. Directory Layout & Key Deliverables

```
├── AGENTS.md                  # Autonomous agent execution protocol & rules
├── README.md                  # Comprehensive project report & team submission guide
├── explain.md                 # Educational concepts, theories, and flowcharts for beginners
├── storage.md                 # Local storage audit and footprint ledger
├── notebooks/                 # Standalone, clean Google Colab / Kaggle notebooks
├── src/
│   ├── data.py                # Dataset loaders & IDX binary parser
│   ├── classical.py           # Standard KNN, PCA & distance sweeps
│   ├── softknn.py             # Differentiable Soft-KNN architectures
│   ├── augment.py             # Data augmentation & test-time augmentation (TTA)
│   ├── train.py               # Soft-KNN training routines
│   └── evaluate.py            # Classification reports, F1-score & confusion matrices
├── scripts/
│   ├── smoke.py               # Pre-flight smoke validation
│   ├── train_conv_seed.py     # Multi-seed convolutional trainer
│   └── train_distill.py       # Knowledge distillation engine
├── results/                   # JSON logs of all experiment runs & metrics
├── figures/                   # Visualizations, ROC curves, confusion matrices
└── artifacts/                 # Saved model weights (.pt)
```

---

## 4. Current State & Remaining Roadmap

- [x] Baseline KNN + PCA evaluation (85.24% baseline, 86.43% tuned)
- [x] Differentiable Soft-KNN implementation (87.10%)
- [x] Convolutional metric backbone (87.63% single model, 88.06% ensemble)
- [x] Knowledge distillation + 7-source weighted fusion (**89.51% test accuracy achieved in E25**)
- [x] Creation of beginner-friendly `explain.md` with conceptual breakdowns
- [x] Creation of machine storage inventory `storage.md`
- [x] Authoring operational guidelines `AGENTS.md`
- [ ] Deliver a clean, standalone, cell-by-cell Kaggle / Colab reproduction notebook (`notebooks/fashion_knn_clean_pipeline.ipynb`)
- [ ] Author top-tier `README.md` complete with problem statement, methodology, evaluation benchmarks, and deployment steps
