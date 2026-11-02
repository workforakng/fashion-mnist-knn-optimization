#!/usr/bin/env python3
"""
Generate a professional, high-impact 6-slide presentation deck (.pptx)
for the Fashion-MNIST KNN Optimization coursework project.
"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
import os

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# Color Palette: Deep Academic Navy, Slate Grey, Crisp Accent Blue
COLOR_BG_DARK = RGBColor(15, 23, 42)      # Deep Navy Slate
COLOR_TEXT_LIGHT = RGBColor(248, 250, 252) # Pure Off-White
COLOR_TEXT_MUTED = RGBColor(148, 163, 184) # Slate Gray
COLOR_PRIMARY = RGBColor(30, 58, 138)     # Navy Primary
COLOR_ACCENT = RGBColor(2, 132, 199)      # Bright Cyan / Blue
COLOR_CARD_BG = RGBColor(241, 245, 249)   # Crisp Light Slate
COLOR_TEXT_DARK = RGBColor(15, 23, 42)    # Dark text

def add_header(slide, title_text, category_text="MACHINE LEARNING COURSEWORK"):
    # Header container
    tb = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(1.1))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    
    p0 = tf.paragraphs[0]
    p0.text = category_text.upper()
    p0.font.size = Pt(11)
    p0.font.bold = True
    p0.font.color.rgb = COLOR_ACCENT
    
    p1 = tf.add_paragraph()
    p1.text = title_text
    p1.font.size = Pt(24)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_PRIMARY
    p1.space_before = Pt(3)

# -------------------------------------------------------------
# SLIDE 1: Title Slide (Dark Theme)
# -------------------------------------------------------------
blank_layout = prs.slide_layouts[6]
slide1 = prs.slides.add_slide(blank_layout)

# Background fill
bg1 = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
bg1.fill.solid()
bg1.fill.fore_color.rgb = COLOR_BG_DARK
bg1.line.color.rgb = COLOR_BG_DARK

# Title box
tb1 = slide1.shapes.add_textbox(Inches(1.2), Inches(1.5), Inches(11.0), Inches(3.0))
tf1 = tb1.text_frame
tf1.word_wrap = True

p_sub = tf1.paragraphs[0]
p_sub.text = "MACHINE LEARNING PROJECT & ALGORITHM OPTIMIZATION"
p_sub.font.size = Pt(13)
p_sub.font.bold = True
p_sub.font.color.rgb = RGBColor(56, 189, 248) # Light cyan

p_title = tf1.add_paragraph()
p_title.text = "Improving k-Nearest Neighbors on Fashion-MNIST\nwith Learned Distances"
p_title.font.size = Pt(34)
p_title.font.bold = True
p_title.font.color.rgb = COLOR_TEXT_LIGHT
p_title.space_before = Pt(12)

p_tag = tf1.add_paragraph()
p_tag.text = "Systematic progression from 85.24 % to 89.51 % Test Accuracy via Metric Learning, Distillation & Convex Fusion"
p_tag.font.size = Pt(16)
p_tag.font.color.rgb = COLOR_TEXT_MUTED
p_tag.space_before = Pt(14)

# Author metadata card
card1 = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.2), Inches(4.8), Inches(10.9), Inches(1.8))
card1.fill.solid()
card1.fill.fore_color.rgb = RGBColor(30, 41, 59)
card1.line.color.rgb = RGBColor(51, 65, 85)

tb_meta = slide1.shapes.add_textbox(Inches(1.5), Inches(5.0), Inches(10.3), Inches(1.4))
tf_m = tb_meta.text_frame
tf_m.word_wrap = True

p_m1 = tf_m.paragraphs[0]
p_m1.text = "TEAM MEMBERS & STUDENT IDS:"
p_m1.font.size = Pt(11)
p_m1.font.bold = True
p_m1.font.color.rgb = RGBColor(56, 189, 248)

p_m2 = tf_m.add_paragraph()
p_m2.text = "• Akshar Nath Gorain (241001001213)      • Sourodyuti Biswas Sanyal (241001001246)\n• Aditi Barman (241001001182)            • Sanjana Mahata (241001001274)"
p_m2.font.size = Pt(13)
p_m2.font.color.rgb = COLOR_TEXT_LIGHT
p_m2.space_before = Pt(4)

p_m3 = tf_m.add_paragraph()
p_m3.text = "Department of Computer Science & Engineering | Techno India University, West Bengal | Batch: BCS 3B"
p_m3.font.size = Pt(11)
p_m3.font.color.rgb = COLOR_TEXT_MUTED
p_m3.space_before = Pt(6)


# -------------------------------------------------------------
# SLIDE 2: Problem Statement & Classical KNN Baselines
# -------------------------------------------------------------
slide2 = prs.slides.add_slide(blank_layout)
add_header(slide2, "1. Problem Statement & Classical Distance Baselines")

# Left Column: Concepts & Protocol
tb2_l = slide2.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(5.8), Inches(5.0))
tf2_l = tb2_l.text_frame
tf2_l.word_wrap = True

def add_bullet(tf, title, body):
    p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
    p.text = f"• {title}: "
    p.font.bold = True
    p.font.size = Pt(13)
    p.font.color.rgb = COLOR_PRIMARY
    run = p.add_run()
    run.text = body
    run.font.bold = False
    run.font.color.rgb = COLOR_TEXT_DARK
    p.space_before = Pt(8)

add_bullet(tf2_l, "Curse of Dimensionality", "In raw 784-D pixel space, distance concentration causes relative distances between nearest and farthest neighbors to converge, degrading discrimination.")
add_bullet(tf2_l, "Loss of 2D Spatial Structure", "Flattening 28x28 images into 1D vectors discards translation invariance and local topological correlations.")
add_bullet(tf2_l, "Inference Latency Bottleneck", "Standard KNN requires O(N·D) brute-force comparisons against 54,000 instances for every single query.")
add_bullet(tf2_l, "Rigorous Experimental Protocol", "60,000 samples partitioned into 54,000 fit and 6,000 validation images (stratified, seed 42). 10,000 test images held out strictly for final reporting.")

# Right Column: Baseline Table & Insight
table_shape2 = slide2.shapes.add_table(4, 4, Inches(7.0), Inches(2.0), Inches(5.5), Inches(2.2))
tbl2 = table_shape2.table
headers2 = ["Metric", "Val (%)", "Test (%)", "Latency"]
data2 = [
    ["Euclidean (L2)", "86.13", "85.24", "4.64 s"],
    ["Manhattan (L1)", "86.97", "85.91", "84.00 s"],
    ["Cosine Distance", "86.18", "85.63", "10.75 s"],
]
for col_idx, h in enumerate(headers2):
    cell = tbl2.cell(0, col_idx)
    cell.text = h
    cell.fill.solid()
    cell.fill.fore_color.rgb = COLOR_PRIMARY
    for p in cell.text_frame.paragraphs:
        p.font.bold = True
        p.font.size = Pt(12)
        p.font.color.rgb = COLOR_TEXT_LIGHT
        p.alignment = PP_ALIGN.CENTER

for row_idx, row in enumerate(data2):
    for col_idx, val in enumerate(row):
        cell = tbl2.cell(row_idx + 1, col_idx)
        cell.text = val
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(241, 245, 249) if row_idx % 2 == 0 else RGBColor(255, 255, 255)
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(12)
            p.font.color.rgb = COLOR_TEXT_DARK
            p.alignment = PP_ALIGN.CENTER
            if col_idx == 2 and "85.91" in val:
                p.font.bold = True

card_box2 = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.0), Inches(4.7), Inches(5.5), Inches(2.0))
card_box2.fill.solid()
card_box2.fill.fore_color.rgb = RGBColor(238, 242, 255)
card_box2.line.color.rgb = RGBColor(199, 210, 254)

tb2_c = slide2.shapes.add_textbox(Inches(7.2), Inches(4.9), Inches(5.1), Inches(1.6))
tf2_c = tb2_c.text_frame
tf2_c.word_wrap = True
p2_c = tf2_c.paragraphs[0]
p2_c.text = "CORE BASELINE FINDINGS:"
p2_c.font.bold = True
p2_c.font.size = Pt(12)
p2_c.font.color.rgb = COLOR_PRIMARY
p2_c2 = tf2_c.add_paragraph()
p2_c2.text = "• Manhattan distance (L1) outperforms Euclidean by +0.67 pp, aligning with high-dimensional norm theory.\n• However, Manhattan query latency is 18x slower (84 s vs 4.6 s).\n• Demonstrates the imperative for learned dimensionality reduction."
p2_c2.font.size = Pt(11)
p2_c2.font.color.rgb = COLOR_TEXT_DARK
p2_c2.space_before = Pt(4)


# -------------------------------------------------------------
# SLIDE 3: PCA Dimension Reduction & Hyperparameter Tuning
# -------------------------------------------------------------
slide3 = prs.slides.add_slide(blank_layout)
add_header(slide3, "2. Dimensionality Reduction & Bayesian Optuna Tuning")

# Add images
if os.path.exists("figures/paper/pca_variance.pdf"):
    # If converted or png available, we add png
    if os.path.exists("figures/03_pca_variance.png"):
        slide3.shapes.add_picture("figures/03_pca_variance.png", Inches(0.8), Inches(1.8), width=Inches(3.8))
    if os.path.exists("figures/06_bias_variance.png"):
        slide3.shapes.add_picture("figures/06_bias_variance.png", Inches(4.8), Inches(1.8), width=Inches(3.8))

tb3_r = slide3.shapes.add_textbox(Inches(8.8), Inches(1.8), Inches(3.8), Inches(5.0))
tf3_r = tb3_r.text_frame
tf3_r.word_wrap = True

add_bullet(tf3_r, "PCA Spectral Compression", "Projecting 784-D -> 64-D preserves 88.1 % of cumulative variance, eliminating high-frequency sensor noise.")
add_bullet(tf3_r, "7.12x Query Acceleration", "Querying 6,000 samples drops from 2.98 s (raw 784-D) to 0.418 s (PCA-64), enabling rapid real-time retrieval.")
add_bullet(tf3_r, "Bias-Variance Peak at k=7", "Sweeping k from 1 to 25 shows classic inverted-U; peak validation accuracy occurs at k=7 (87.37 %).")
add_bullet(tf3_r, "Optuna Optimization (E3)", "40-trial Bayesian search selects k=5, Manhattan distance, and inverse-distance weights (w_i = 1/d_i), reaching 86.43 % test accuracy (+1.19 pp).")


# -------------------------------------------------------------
# SLIDE 4: Differentiable Soft-KNN & ConvNet Metric Backbone
# -------------------------------------------------------------
slide4 = prs.slides.add_slide(blank_layout)
add_header(slide4, "3. Novelty: Soft-KNN & ConvNet Metric Learning")

tb4_l = slide4.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(6.0), Inches(5.2))
tf4_l = tb4_l.text_frame
tf4_l.word_wrap = True

add_bullet(tf4_l, "Why Soft-KNN?", "Discrete majority voting has zero gradient almost everywhere. Soft-KNN formulates a continuous, temperature-scaled Boltzmann neighbor assignment.")
add_bullet(tf4_l, "Memory Bank Architecture", "A 54,000-instance memory bank stores unit-normalized embeddings z = f_theta(x) / ||f_theta(x)|| in S^127, updated periodically.")
add_bullet(tf4_l, "Local Mean Prototype Formulation", "For query z, the mean distance to class c among its m nearest exemplars is:\n  delta_bar_c(x) = (1/m) sum_{i in A_c(x)} ||z - b_i||_2")
add_bullet(tf4_l, "Temperature-Scaled Softmax", "P_tau(y = c | x) = exp(-delta_bar_c / tau) / sum_{c'} exp(-delta_bar_c' / tau)\nTrained end-to-end via cross-entropy loss: L = -log P_tau(y | x).")

# Right column: Architectures & Results
card4_1 = slide4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.2), Inches(1.8), Inches(5.3), Inches(2.4))
card4_1.fill.solid()
card4_1.fill.fore_color.rgb = RGBColor(240, 253, 244) # Light green
card4_1.line.color.rgb = RGBColor(187, 247, 208)

tb4_c1 = slide4.shapes.add_textbox(Inches(7.4), Inches(1.9), Inches(4.9), Inches(2.1))
tf4_c1 = tb4_c1.text_frame
tf4_c1.word_wrap = True
p4_1 = tf4_c1.paragraphs[0]
p4_1.text = "LINEAR SOFT-KNN HEAD (E6):"
p4_1.font.bold = True
p4_1.font.size = Pt(12)
p4_1.font.color.rgb = RGBColor(22, 101, 52)
p4_1_sub = tf4_c1.add_paragraph()
p4_1_sub.text = "• 784 -> 128 linear projection initialized from PCA.\n• Adam optimizer (lr 1e-3, tau=0.5, m=7).\n• Validation Acc: 88.22 % | Test Acc: 87.10 %\n• Outperforms tuned classical PCA-KNN by +0.67 pp."
p4_1_sub.font.size = Pt(11)
p4_1_sub.font.color.rgb = COLOR_TEXT_DARK
p4_1_sub.space_before = Pt(4)

card4_2 = slide4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.2), Inches(4.5), Inches(5.3), Inches(2.4))
card4_2.fill.solid()
card4_2.fill.fore_color.rgb = RGBColor(238, 242, 255) # Light blue
card4_2.line.color.rgb = RGBColor(199, 210, 254)

tb4_c2 = slide4.shapes.add_textbox(Inches(7.4), Inches(4.6), Inches(4.9), Inches(2.1))
tf4_c2 = tb4_c2.text_frame
tf4_c2.word_wrap = True
p4_2 = tf4_c2.paragraphs[0]
p4_2.text = "CONVNET METRIC BACKBONE (E19):"
p4_2.font.bold = True
p4_2.font.size = Pt(12)
p4_2.font.color.rgb = COLOR_PRIMARY
p4_2_sub = tf4_c2.add_paragraph()
p4_2_sub.text = "• 2 stages of 3x3 Convolutions (32, 64 channels), ReLU, 2x2 Max-Pool.\n• Preserves 2D image topology prior to metric lookup.\n• Validation Acc: 89.07 % | Test Acc: 87.63 %\n• Provides the core structural backbone for ensembling."
p4_2_sub.font.size = Pt(11)
p4_2_sub.font.color.rgb = COLOR_TEXT_DARK
p4_2_sub.space_before = Pt(4)


# -------------------------------------------------------------
# SLIDE 5: Multi-Seed Ensemble, Distillation & Convex Fusion
# -------------------------------------------------------------
slide5 = prs.slides.add_slide(blank_layout)
add_header(slide5, "4. Multi-Seed Ensembling, Distillation & Convex Fusion")

# Left Column: Progression Steps
tb5_l = slide5.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(6.0), Inches(5.2))
tf5_l = tb5_l.text_frame
tf5_l.word_wrap = True

add_bullet(tf5_l, "Multi-Seed ConvNet Ensemble (E21)", "Trained across seeds {42, 7, 123}. Individual test accuracies: 87.53 %, 86.92 %, 87.57 % (spread: 0.65 pp). Probability averaging yields 88.06 % (+0.72 pp over mean).")
add_bullet(tf5_l, "Knowledge Distillation (E23)", "Trained student ConvNet using KL-divergence soft targets from the 3-teacher ensemble. Achieves 90.10 % validation and 89.03 % test accuracy (+0.97 pp gain).")
add_bullet(tf5_l, "Test-Time Augmentation (TTA)", "Queries evaluated across 6 perturbations (original, horizontal flip, 4 directional 1-pixel shifts) to smooth boundary sensitivity.")
add_bullet(tf5_l, "7-Source Convex Simplex Search (E25)", "Exhaustive grid search over 8,008 simplex weight vectors (step 0.1) on validation set.")

# Right Column: Simplex Weights Table & Champion Callout
table_shape5 = slide5.shapes.add_table(5, 3, Inches(7.2), Inches(1.8), Inches(5.3), Inches(2.5))
tbl5 = table_shape5.table
headers5 = ["Metric Source", "Role", "E25 Weight"]
data5 = [
    ["Distilled Student (E23)", "High-capacity ConvNet", "w = 0.5 (50 %)"],
    ["Linear Soft-KNN (E14)", "Augmented memory bank", "w = 0.4 (40 %)"],
    ["PCA-64 KNN (E3)", "Classical Manhattan lookup", "w = 0.1 (10 %)"],
    ["Other 4 heads", "Seeds 42, 7, 123, MLP", "w = 0.0 (Pruned)"],
]
for col_idx, h in enumerate(headers5):
    cell = tbl5.cell(0, col_idx)
    cell.text = h
    cell.fill.solid()
    cell.fill.fore_color.rgb = COLOR_PRIMARY
    for p in cell.text_frame.paragraphs:
        p.font.bold = True
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_TEXT_LIGHT
        p.alignment = PP_ALIGN.CENTER

for row_idx, row in enumerate(data5):
    for col_idx, val in enumerate(row):
        cell = tbl5.cell(row_idx + 1, col_idx)
        cell.text = val
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(241, 245, 249) if row_idx % 2 == 0 else RGBColor(255, 255, 255)
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(11)
            p.font.color.rgb = COLOR_TEXT_DARK
            p.alignment = PP_ALIGN.CENTER
            if col_idx == 2 and "w = 0.5" in val:
                p.font.bold = True

card5_champ = slide5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.2), Inches(4.7), Inches(5.3), Inches(2.2))
card5_champ.fill.solid()
card5_champ.fill.fore_color.rgb = RGBColor(254, 242, 242) # Light red/rose
card5_champ.line.color.rgb = RGBColor(254, 202, 202)

tb5_ch = slide5.shapes.add_textbox(Inches(7.4), Inches(4.9), Inches(4.9), Inches(1.8))
tf5_ch = tb5_ch.text_frame
tf5_ch.word_wrap = True
p5_ch1 = tf5_ch.paragraphs[0]
p5_ch1.text = "CHAMPION MODEL METRICS (E25):"
p5_ch1.font.bold = True
p5_ch1.font.size = Pt(13)
p5_ch1.font.color.rgb = RGBColor(185, 28, 28)

p5_ch2 = tf5_ch.add_paragraph()
p5_ch2.text = ">>> UNBIASED TEST ACCURACY: 89.51 % <<<\n>>> VALIDATION ACCURACY: 90.85 % <<<\nMacro F1: 0.8936 | Precision: 0.8944 | Recall: 0.8951"
p5_ch2.font.size = Pt(12)
p5_ch2.font.bold = True
p5_ch2.font.color.rgb = COLOR_TEXT_DARK
p5_ch2.space_before = Pt(4)


# -------------------------------------------------------------
# SLIDE 6: Error Analysis, Confusion Matrix & Conclusions
# -------------------------------------------------------------
slide6 = prs.slides.add_slide(blank_layout)
add_header(slide6, "5. Benchmark Results, Error Diagnostics & Key Takeaways")

# Add Confusion Matrix image if available
if os.path.exists("figures/12_confusion_matrix.png"):
    slide6.shapes.add_picture("figures/12_confusion_matrix.png", Inches(0.8), Inches(1.8), width=Inches(4.8))

tb6_r = slide6.shapes.add_textbox(Inches(5.8), Inches(1.8), Inches(6.8), Inches(5.2))
tf6_r = tb6_r.text_frame
tf6_r.word_wrap = True

add_bullet(tf6_r, "Rigorous Algorithmic Progression", "85.24 % (Baseline E1) -> 86.43 % (PCA-64 E3) -> 87.10 % (Linear Soft-KNN E6) -> 87.63 % (ConvNet E19) -> 89.03 % (Distillation E23) -> 89.51 % (Convex Fusion E25).")
add_bullet(tf6_r, "Diagnostic Confusion Matrix", "Near-perfect recall on distinct silhouettes: Bag (98.5 %), Trouser (97.9 %), Sneaker (97.0 %).")
add_bullet(tf6_r, "Garment Confusion Bottleneck", "Shirt achieves only 61.7 % recall; 16.0 % misclassified as T-shirt/top and 10.4 % as Pullover due to unresolved low-resolution grayscale texture.")
add_bullet(tf6_r, "Academic Takeaway", "Non-parametric instance classifiers can achieve competitive deep-learning performance (>89.5 %) while retaining neighbor-lookup interpretability.")
add_bullet(tf6_r, "Open-Source Reproduction", "Code, audited ledgers, and standalone Colab notebook public on GitHub: github.com/workforakng/fashion-mnist-knn-optimization.")

output_pptx = "docs/presentation.pptx"
prs.save(output_pptx)
print(f"Presentation saved successfully to {output_pptx} (6 slides)")
