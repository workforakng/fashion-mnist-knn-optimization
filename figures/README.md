# Experimental Figures & Visual Diagnostics (`figures/`)

This directory contains the 15 publication-grade diagnostic plots, empirical curves, and confusion heatmaps generated during the project.

---

## Figure Catalog

| File Name | Figure Title | Description & Analytical Focus |
| :--- | :--- | :--- |
| `01_class_distribution.png` | Class Balance Histogram | Confirms uniform 6,000 observations per category in the development set. |
| `02_samples.png` | Exploratory Sample Gallery | Representative visual grid displaying images from all 10 garment categories. |
| `03_pca_variance.png` | PCA Cumulative Variance Curve | Tracks spectral variance vs. component dimension ($d$), highlighting the 85.3% threshold at $d=64$. |
| `04_pca_sweep.png` | Accuracy across PCA Dimensions | Evaluates Euclidean, Manhattan, and Cosine distance metrics across dimensions $d \in [16, 128]$. |
| `05_latency.png` | Query Latency Benchmark | Measures neighbor query time comparing raw 784-D search vs. 64-D PCA (**7.14× speedup**). |
| `06_bias_variance.png` | Bias-Variance Tradeoff Curve | Plots validation accuracy vs. neighbor count $k \in [1, 15]$, identifying optimal tradeoff at $k=7$. |
| `07_optuna_history.png` | Hyperparameter Optimization Trace | Visualizes objective value improvement across automated parameter search trials. |
| `08_augmented.png` | Data Augmentation Visualizer | Compares raw fashion items against horizontal reflections and random crop variations. |
| `09_optimizer_compare.png` | Optimizer Convergence Benchmark | Empirical training loss descent curves for **AdamW vs. SGD with Momentum vs. RMSprop**. |
| `10_tau_sweep.png` | Softmax Temperature $\tau$ Sensitivity | Validation accuracy as a function of temperature $\tau$, showing the peak at $\tau=0.1$. |
| `11_softknn_curves.png` | Soft-KNN Epoch Learning Curves | Dual-panel plot of training vs. validation loss and accuracy across epochs. |
| `12_confusion_matrix.png` | Test Confusion Matrix Heatmap | Complete 10×10 error matrix on 10,000 test images, detailing correct counts and garment confusions. |
| `13_conv_curves.png` | ConvNet Backbone Learning Dynamics | Loss and accuracy trajectories for the 2D convolutional feature extractor. |
| `14_inference_grid.png` | Qualitative Prediction Grid | Displays test samples alongside predicted category, ground-truth label, and confidence score. |
| `15_conv_shift_curves.png` | Spatial Shift Stress Test Curves | Evaluates representation stability under synthetic horizontal and vertical pixel shifts. |
