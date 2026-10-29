# Source Code Architecture (`src/`)

This directory houses the foundational modular Python implementation for the Fashion-MNIST classification, dimensionality reduction, metric learning, and evaluation pipeline.

---

## Directory Modules

### 1. `data.py` — Ingestion & Binary IDX Parsing
* **Purpose:** Implements custom binary parsers for Zalando's Fashion-MNIST `.gz` archives (`train-images-idx3-ubyte.gz`, `train-labels-idx1-ubyte.gz`, etc.).
* **Key Functions:**
  * `load_fashion_mnist(data_dir)`: Unpacks binary IDX data into standard NumPy arrays and PyTorch tensors.
  * `get_splits(train_ratio=50000, val_ratio=10000)`: Enforces the deterministic 50,000 / 10,000 / 10,000 partitioning protocol with seed reproducibility.

### 2. `classical.py` — Non-Parametric KNN & PCA Analysis
* **Purpose:** Classical baseline implementations using scikit-learn.
* **Key Components:**
  * Euclidean ($L_2$), Manhattan ($L_1$), and Cosine distance metric benchmarks.
  * Principal Component Analysis (PCA) projection and variance preservation sweeps ($d \in [16, 128]$).
  * Hyperparameter grid search over neighbor count $k \in [1, 15]$ and distance weighting strategies (`uniform` vs `distance`).

### 3. `softknn.py` — Differentiable Metric Learning Architectures
* **Purpose:** Core novelty architectures for continuous, backpropagatable nearest-neighbor classification.
* **Key Architectures:**
  * `SoftKNNLinear`: Differentiable linear metric projector mapping $\mathbb{R}^{784} \to \mathbb{R}^{64}$.
  * `SoftKNNConvBackbone`: 2D Convolutional metric feature extractor with Batch Normalization, GELU activations, and Class Prototype Anchors $\mathcal{P} \in \mathbb{R}^{10 \times 64}$.
  * `temperature_distance_softmax(embeddings, prototypes, tau)`: Computes the continuous Boltzmann posterior distribution with temperature scaling parameter $\tau$.

### 4. `augment.py` — Spatial Transformations & TTA
* **Purpose:** Data regularization to expand the metric space beyond rigid pixel alignments.
* **Key Components:**
  * Random horizontal reflection ($p=0.5$).
  * Random spatial crop ($28 \times 28$, padding 2, edge reflection).
  * Test-Time Augmentation (TTA) averaging inference pipeline for uncertainty reduction.

### 5. `train.py` — Optimization Engine
* **Purpose:** Robust training routines for differentiable metric heads.
* **Key Components:**
  * Gradient descent optimization supporting AdamW, SGD with Momentum, and RMSprop.
  * Cosine annealing learning rate schedulers.
  * Automated checkpointing saving best validation weights to `artifacts/`.

### 6. `evaluate.py` — Diagnostic Metrics & Confusion Matrix
* **Purpose:** Unbiased final evaluation on held-out test partitions.
* **Key Components:**
  * Macro-averaged Precision, Recall, and F1-score calculation.
  * Confusion matrix generation identifying class-boundary confusions.
  * Query latency benchmarking comparing raw 784-D vs PCA vs Soft-KNN inference speed.

### 7. `experiments.py` — Multi-Stage Experiment Runner
* **Purpose:** Orchestrates the multi-stage experimental ladder (baseline through multi-seed ensembling) and serializes logs into `results/results.json`.
