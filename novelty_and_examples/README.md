# Systematic Optimization & Metric Learning for Fashion-MNIST KNN

## 5-Line Novelty Summary

Standard K-Nearest Neighbors fails on image data due to the curse of dimensionality and the loss of 2D spatial context. This project makes KNN fully differentiable by computing temperature-scaled softmax probabilities over the nearest exemplars in a dynamic 54,000-sample memory bank rather than relying on fixed class centroids. It replaces raw pixel distances with lightweight 2D convolutional embedding manifolds that naturally preserve spatial neighborhoods and translation invariance. The framework also augments the exemplar memory bank itself to heighten neighbor density without growing parameter count, and distills knowledge from multi-seed ensembles into a fast student model. Finally, it unites classical manifold-projected KNN, bank-augmented metric learning, and multi-view test-time augmentation through optimal decision fusion, elevating test accuracy from an 85.24% baseline to 89.51%.

---

## Condensed Novelty (Ready for Presentation / PPT)

> **Core Idea:** Modernizing classical Nearest Neighbors into a differentiable, deep-learning-compatible classifier for computer vision.

- **Differentiable Soft-KNN:** Instead of rigid voting or static prototypes, it optimizes an end-to-end neural embedding using an active memory bank of all 54,000 training images.
- **2D Spatial Awareness:** Replaces flat pixel vectors with a custom 2D Convolutional metric, preserving spatial structures and overcoming distance concentration.
- **Memory-Bank Augmentation:** Enhances classification confidence by augmenting the stored memory bank itself, increasing exemplar density without adding extra model weights.
- **Ensemble Distillation:** Compresses the predictive power of multi-seed metric models into a single, highly efficient inference student.
- **Hybrid Decision Fusion:** Blends the strengths of classical PCA-KNN, deep metric representations, and multi-view test-time augmentation to reach an 89.51% peak accuracy.

---

## Key Novelty Points in Detail

### 1. Differentiable Memory-Bank Metric Learning (Soft-KNN)
- Traditional KNN has zero gradients almost everywhere due to discrete hard-majority voting. Prototypical approaches oversimplify categories by compressing them into single centroid vectors.
- This project implements an exact differentiable soft nearest-neighbor decision rule: for each query, it takes the average distance to the top-$k$ nearest neighbors of each class stored inside an actively refreshed 54,000-sample memory bank.
- Backpropagation directly tunes query feature embeddings to pull representations closer to correct-class neighborhoods in the latent space.

### 2. Preserving 2D Spatial Geometry & Invariance
- Flattening $28 \times 28$ images into 784-D vectors destroys pixel adjacency, making Euclidean distance sensitive to even 1-pixel translations.
- Incorporating a 2-stage Convolutional Metric Network (`ConvMetric`) projects images into a compact 128-D normalized hypersphere, preserving local textures and translational invariance.

### 3. Exemplar-Space Memory-Bank Augmentation
- Standard data augmentation operates only on incoming query batches.
- This work augments the reference memory bank itself (expanding to 108,000 exemplars with shifted views), substantially increasing neighborhood density and decision boundary sharpness without inflating parameter footprint.

### 4. Cross-Paradigm Ensemble Knowledge Distillation
- Combines soft predictive distributions across diverse multi-seed teacher models into a unified supervision signal.
- Distills the collective knowledge of the ensemble into a single compact convolutional student via a combined KL-divergence and cross-entropy loss, boosting student test accuracy to 89.03%.

### 5. Multi-Paradigm Heterogeneous Decision Fusion
- Rather than discarding classical statistics, the final decision rule fuses predictions across disparate paradigms:
  1. Classical Optuna-tuned Manhattan PCA-64 KNN
  2. Bank-augmented linear metric model
  3. Distilled convolutional student
  4. 6-view Test-Time Augmentation (TTA)
- Simplex-constrained grid search finds optimal fusion weights on validation data, achieving the peak **89.51% test accuracy** (90.85% validation).
