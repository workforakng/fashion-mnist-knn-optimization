"""Print-ready vector figures for the paper, driven only by logged results.

Usage:
    python scripts/make_paper_figures.py [--data-dir DIR] [--recompute-pca]

Reads results/results.json (and results/conv_seeds.json) and writes PDFs to
figures/paper/. Existing figures/*.png are left untouched.

Every figure is sized for 0.48\\textwidth on A4 with 1.8 cm margins
(about 8.4 cm = 3.3 in wide), with all text at 8 pt or larger.

The only figure that needs the raw data is pca_variance.pdf: results.json logs
var90_dim but not the per-component variance, so the script fits PCA on the
same 54k training split as src/experiments.py (load_splits(val_size=6000,
seed=42)) and caches the numbers in figures/paper/pca_variance.json. Later
runs reuse that file unless --recompute-pca is given.

The confusion matrix of the E25 champion is not stored in results.json. The
counts below were transcribed from figures/12_confusion_matrix.png (written by
E25, the last fusion stage run); the script checks that the trace equals the
logged E25 test accuracy (0.8951 x 10,000 = 8951) and that every row sums to
1,000.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "figures", "paper")
RESULTS = os.path.join(REPO, "results", "results.json")
SEEDS = os.path.join(REPO, "results", "conv_seeds.json")

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# E25 test confusion matrix (rows = true class, cols = predicted), transcribed
# from figures/12_confusion_matrix.png. Not logged in results.json.
E25_CM = [
    [888, 1, 10, 16, 4, 0, 76, 0, 5, 0],
    [3, 979, 0, 13, 2, 0, 2, 0, 1, 0],
    [15, 2, 849, 8, 88, 0, 38, 0, 0, 0],
    [19, 4, 15, 903, 32, 0, 26, 0, 1, 0],
    [1, 2, 94, 31, 836, 0, 32, 0, 3, 1],
    [0, 0, 0, 0, 0, 966, 0, 24, 0, 10],
    [160, 2, 87, 26, 91, 0, 617, 0, 17, 0],
    [0, 0, 0, 0, 0, 8, 0, 970, 0, 22],
    [1, 0, 0, 3, 4, 1, 2, 3, 985, 1],
    [0, 0, 0, 0, 0, 4, 0, 37, 1, 958],
]

# ---------------------------------------------------------------- style
# Colour-blind-safe categorical order (validated adjacent-pair palette):
# slot 1 blue, slot 2 orange, slot 3 aqua. Roles are fixed across figures:
#   blue   = validation accuracy / the main measured series
#   orange = training accuracy / second series of a pair
#   aqua   = third series (only in the optimizer figure)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
# Sequential blue ramp (light -> dark) for the confusion matrix.
SEQ_BLUE = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
            "#256abf", "#184f95", "#0d366b"]

W = 3.3            # figure width in inches (~8.4 cm)
H = 2.4
N_VAL = 6000       # validation set size (stratified from the 60k pool)
N_TEST = 10000

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK,
    "axes.linewidth": 0.6,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "lines.linewidth": 1.4,
    "lines.markersize": 4,
    "legend.frameon": False,
    "pdf.fonttype": 42,   # embed TrueType so text stays text
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def binom_se(p, n):
    return np.sqrt(np.asarray(p) * (1 - np.asarray(p)) / n)


def mark_truncated_y(ax):
    """Draw a small break mark at the bottom of the y spine (axis does not start at 0)."""
    kw = dict(transform=ax.transAxes, color=INK2, clip_on=False, lw=0.7)
    d = 0.012
    for y0 in (0.0, 0.025):
        ax.plot((-2 * d, 2 * d), (y0 - d, y0 + d), **kw)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"[fig] wrote {os.path.relpath(path, REPO)}")


def by_id(ledger, eid):
    rows = [e for e in ledger if e.get("id") == eid]
    if len(rows) != 1:
        raise KeyError(f"expected exactly one ledger entry {eid}, got {len(rows)}")
    return rows[0]


# ---------------------------------------------------------------- 1. PCA variance
def pca_variance_numbers(data_dir, recompute):
    cache = os.path.join(OUT, "pca_variance.json")
    if os.path.exists(cache) and not recompute:
        with open(cache) as f:
            return json.load(f)
    sys.path.insert(0, REPO)
    import src.data as data
    from sklearn.decomposition import PCA
    if data_dir:
        data.DATA_DIR = data_dir
    xa, _, xv, _, _, _ = data.load_splits(val_size=N_VAL, seed=42)
    pca = PCA(svd_solver="full").fit(xa)     # all 784 components
    cum = np.cumsum(pca.explained_variance_ratio_)
    d90 = int(np.searchsorted(cum, 0.90) + 1)  # same rule as PCA(n_components=0.90)
    nums = {
        "source": "PCA(svd_solver='full') fit on the 54,000-image training split "
                  "(src.data.load_splits(val_size=6000, seed=42)), pixels in [0,1]",
        "n_fit": int(xa.shape[0]),
        "cum_var_at": {str(d): float(cum[d - 1]) for d in
                       (1, 2, 10, 20, 32, 48, 64, 80, 84, 96, 128, 187, 200)},
        "dim_for_90pct": d90,
        "dim_for_95pct": int(np.searchsorted(cum, 0.95) + 1),
        "cumulative_explained_variance": [float(c) for c in cum],
    }
    with open(cache, "w") as f:
        json.dump(nums, f, indent=1)
    print(f"[pca] wrote {os.path.relpath(cache, REPO)}")
    return nums


def fig_pca_variance(nums, ledger):
    cum = np.asarray(nums["cumulative_explained_variance"]) * 100
    d90_logged = by_id(ledger, "E2")["var90_dim"]
    if nums["dim_for_90pct"] != d90_logged:
        print(f"[warn] recomputed 90% dim {nums['dim_for_90pct']} != logged {d90_logged}")
    v64 = cum[63]
    dmax = 200
    x = np.arange(1, dmax + 1)
    fig, ax = plt.subplots(figsize=(W, H))
    ax.plot(x, cum[:dmax], color=BLUE)
    # 90 % threshold (dashed, horizontal) reached at d=84
    ax.axhline(90, color=MUTED, ls="--", lw=0.8)
    ax.text(dmax - 2, 88.8, "90 % threshold", ha="right", va="top", color=INK2)
    ax.plot([d90_logged], [cum[d90_logged - 1]], "o", ms=4, mfc="white", mec=INK2, mew=0.9)
    ax.annotate(f"d = {d90_logged}", (d90_logged, cum[d90_logged - 1]),
                xytext=(d90_logged + 14, 80), color=INK2, va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    # d=64 operating point (dotted, vertical)
    ax.axvline(64, color=INK, ls=":", lw=1.0)
    ax.plot([64], [v64], "o", ms=4, color=BLUE)
    ax.annotate(f"d = 64: {v64:.1f} %", (64, v64), xytext=(72, 62),
                color=INK, va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.set_xlim(0, dmax)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 32, 64, 100, 150, 200])
    ax.set_xlabel("Number of principal components d")
    ax.set_ylabel("Cumulative explained variance (%)")
    save(fig, "pca_variance.pdf")
    return v64


# ---------------------------------------------------------------- 2. PCA sweep
def fig_pca_sweep(ledger):
    e2 = by_id(ledger, "E2")
    rows = sorted(e2["rows"], key=lambda r: r["dim"])
    d = np.array([r["dim"] for r in rows])
    acc = np.array([r["val_acc"] for r in rows])
    se = binom_se(acc, N_VAL)
    best = max(rows, key=lambda r: r["val_acc"])
    fig, ax = plt.subplots(figsize=(W, H))
    ax.errorbar(d, acc * 100, yerr=se * 100, color=BLUE, marker="o", ms=3.5,
                capsize=0, elinewidth=0.8, ecolor=BLUE)
    ax.plot([best["dim"]], [best["val_acc"] * 100], "o", ms=7, mfc="none", mec=INK, mew=0.9)
    ax.annotate(f"best d = {best['dim']}\n{best['val_acc'] * 100:.2f} %",
                (best["dim"], best["val_acc"] * 100), xytext=(best["dim"] + 8, 84.2),
                color=INK, va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.set_xlim(0, 135)
    ax.set_ylim(81.5, 88.2)
    ax.set_xticks([10, 32, 64, 96, 128])
    ax.set_xlabel("Number of principal components d")
    ax.set_ylabel("Validation accuracy (%)")
    mark_truncated_y(ax)
    save(fig, "pca_sweep.pdf")
    return [(int(a), float(b)) for a, b in zip(d, acc)]


# ---------------------------------------------------------------- 3. bias-variance
def fig_bias_variance(ledger):
    e4 = by_id(ledger, "E4")
    ks = np.array(e4["ks"])
    tr = np.array(e4["train_accs"]) * 100
    va = np.array(e4["val_accs"])
    fig, ax = plt.subplots(figsize=(W, H))
    ax.plot(ks, tr, color=ORANGE, marker="s", ms=3, ls="--", lw=1.1, label="Training")
    ax.errorbar(ks, va * 100, yerr=binom_se(va, N_VAL) * 100, color=BLUE, marker="o",
                ms=3.5, capsize=0, elinewidth=0.8, label="Validation")
    bk = e4["best_k"]
    ax.plot([bk], [e4["best_val_acc"] * 100], "o", ms=7, mfc="none", mec=INK, mew=0.9)
    ax.annotate(f"best k = {bk}: {e4['best_val_acc'] * 100:.2f} %",
                (bk, e4["best_val_acc"] * 100), xytext=(13, 85.0), color=INK,
                va="center", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.set_xticks(ks[::2])
    ax.set_xlim(0, 26)
    ax.set_ylim(84, 100.5)
    ax.set_yticks([85, 90, 95, 100])
    ax.set_xlabel("Number of neighbours k")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(loc="upper right", handlelength=1.8)
    mark_truncated_y(ax)
    save(fig, "bias_variance.pdf")
    return list(zip(ks.tolist(), va.tolist()))


# ---------------------------------------------------------------- 4. tau sweep
def fig_tau_sweep(ledger):
    e8 = by_id(ledger, "E8")
    rows = sorted(e8["rows"], key=lambda r: r["tau"])
    t = np.array([r["tau"] for r in rows])
    acc = np.array([r["val_acc"] for r in rows])
    fig, ax = plt.subplots(figsize=(W, H))
    ax.errorbar(t, acc * 100, yerr=binom_se(acc, N_VAL) * 100, color=BLUE, marker="o",
                ms=3.5, capsize=0, elinewidth=0.8)
    ax.plot([t[0]], [acc[0] * 100], "o", ms=7, mfc="none", mec=INK, mew=0.9)
    ax.annotate(f"best tested $\\tau$ = {t[0]:g}: {acc[0] * 100:.2f} %\n(smallest value in the grid)",
                (t[0], acc[0] * 100), xytext=(0.3, 86.6), color=INK, va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.set_xlim(0, 1.05)
    ax.set_xticks(t)
    ax.set_xticklabels([f"{v:g}" for v in t])
    ax.set_ylim(80.4, 87.6)
    ax.set_xlabel(r"Training temperature $\tau$ (grid points only)")
    ax.set_ylabel("Validation accuracy (%)")
    mark_truncated_y(ax)
    save(fig, "tau_sweep.pdf")
    return list(zip(t.tolist(), acc.tolist()))


# ---------------------------------------------------------------- 5. optimizer
def fig_optimizer(ledger):
    e9 = by_id(ledger, "E9")
    rows = e9["rows"]
    names = {"adam": "Adam", "rmsprop": "RMSprop", "sgd": "SGD"}
    labels = [f"{names[r['optimizer']]}\nlr = {r['lr']:g}" for r in rows]
    best = np.array([r["best_val_acc"] for r in rows]) * 100
    final = np.array([r["final_val_acc"] for r in rows]) * 100
    x = np.arange(len(rows))
    w = 0.36
    fig, ax = plt.subplots(figsize=(W, H))
    b1 = ax.bar(x - w / 2 - 0.01, best, w, color=BLUE,
                label="In-training val (best epoch)")
    b2 = ax.bar(x + w / 2 + 0.01, final, w, color=ORANGE,
                label="Re-evaluated after training")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.15,
                    f"{b.get_height():.1f}", ha="center", va="bottom", color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(78.6, 89.5)
    ax.set_ylabel("Validation accuracy (%)")
    ax.legend(loc="upper left", ncol=1, handlelength=1.0, borderaxespad=0.1)
    mark_truncated_y(ax)
    save(fig, "optimizer.pdf")
    return [(r["optimizer"], r["lr"], r["best_val_acc"], r["final_val_acc"]) for r in rows]


# ---------------------------------------------------------------- 6. latency
def fig_latency(ledger):
    e5 = by_id(ledger, "E5")
    algos = ["brute", "kd_tree", "ball_tree"]
    algo_lbl = {"brute": "Brute force", "kd_tree": "k-d tree", "ball_tree": "Ball tree"}
    variants = [("raw-784", "Raw pixels (784-D)", ORANGE), ("pca-64", "PCA-64", BLUE)]
    q = {(r["variant"], r["algorithm"]): r["query_s_median"] for r in e5["rows"]}
    x = np.arange(len(algos))
    w = 0.36
    fig, ax = plt.subplots(figsize=(W, H))
    for i, (v, lbl, c) in enumerate(variants):
        vals = [q[(v, a)] for a in algos]
        bars = ax.bar(x + (i - 0.5) * (w + 0.02), vals, w, color=c, label=lbl)
        for b, val in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, val * 1.12,
                    f"{val:.2f}" if val < 10 else f"{val:.0f}",
                    ha="center", va="bottom", color=INK)
    ax.set_yscale("log")
    ax.set_ylim(0.1, 400)
    ax.set_yticks([0.1, 1, 10, 100])
    ax.set_yticklabels(["0.1", "1", "10", "100"])
    ax.set_xticks(x)
    ax.set_xticklabels([algo_lbl[a] for a in algos])
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", which="minor", visible=False)
    ax.set_ylabel("Query time for 6,000 images (s, log)")
    ax.legend(loc="upper left", ncol=2, handlelength=1.0, columnspacing=1.0,
              borderaxespad=0.1)
    save(fig, "latency.pdf")
    return {f"{v}/{a}": q[(v, a)] for v, _, _ in variants for a in algos}


# ---------------------------------------------------------------- 7. confusion
def fig_confusion(ledger):
    cm = np.array(E25_CM)
    e25 = by_id(ledger, "E25")
    assert cm.shape == (10, 10)
    assert (cm.sum(1) == 1000).all(), cm.sum(1)
    assert cm.trace() == 8951, cm.trace()
    assert abs(cm.trace() / N_TEST - e25["test_acc"]) < 1e-9
    prec = cm.diagonal() / cm.sum(0)
    assert abs(prec.mean() - e25["precision_macro"]) < 5e-4, prec.mean()
    pct = cm / cm.sum(1, keepdims=True) * 100
    recall = pct.diagonal()

    cmap = LinearSegmentedColormap.from_list("seqblue", SEQ_BLUE)
    fig, ax = plt.subplots(figsize=(2.75, 3.0))
    ax.grid(False)
    ax.imshow(pct, cmap=cmap, vmin=0, vmax=100)
    for i in range(10):
        for j in range(10):
            v = pct[i, j]
            if round(v) == 0:
                continue
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if v >= 50 else INK)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(CLASS_NAMES)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 10), minor=True)
    ax.set_yticks(np.arange(-0.5, 10), minor=True)
    ax.grid(which="minor", color="white", lw=0.8)
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    # Right-hand column: exact per-class recall (one decimal) instead of a
    # colour bar; the cells carry rounded row percentages.
    for i, r in enumerate(recall):
        ax.text(10.0, i, f"{r:.1f}", ha="left", va="center", color=INK,
                fontweight="bold" if r == recall.min() else "normal")
    ax.text(10.0, -0.85, "Recall\n(%)", ha="left", va="bottom", color=INK2)
    ax.set_xlim(-0.5, 9.5)
    save(fig, "confusion_matrix.pdf")
    return {n: float(r) for n, r in zip(CLASS_NAMES, recall)}, cm


# ---------------------------------------------------------------- 8. seeds
def fig_seeds(ledger):
    with open(SEEDS) as f:
        seeds = json.load(f)
    e21 = by_id(ledger, "E21")
    for s in e21["per_seed"]:   # consistency check between the two logs
        assert abs(seeds[str(s["seed"])]["test_acc"] - s["test_acc"]) < 1e-9
    order = ["42", "7", "123"]
    labels = [f"Seed {s}" for s in order] + ["3-seed\nensemble"]
    test = [seeds[s]["test_acc"] * 100 for s in order] + [e21["test_acc"] * 100]
    val = [seeds[s]["val_acc"] * 100 for s in order] + [e21["val_acc"] * 100]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(W, H))
    ax.axvline(2.5, color=AXIS, lw=0.6)
    ax.errorbar(x - 0.09, val, yerr=binom_se(np.array(val) / 100, N_VAL) * 100,
                fmt="o", color=BLUE, ms=4, elinewidth=0.8, label="Validation (n = 6,000)")
    ax.errorbar(x + 0.09, test, yerr=binom_se(np.array(test) / 100, N_TEST) * 100,
                fmt="s", color=ORANGE, ms=4, elinewidth=0.8, label="Test (n = 10,000)")
    for xi, v in zip(x, test):
        ax.annotate(f"{v:.2f}", (xi + 0.09, v), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK)
    for xi, v in zip(x, val):
        ax.annotate(f"{v:.2f}", (xi - 0.09, v), xytext=(-6, 0), textcoords="offset points",
                    va="center", ha="right", color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="x", visible=False)
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(86.3, 90.6)
    ax.set_ylabel("Accuracy (%)")
    ax.legend(loc="upper left", handlelength=1.0, borderaxespad=0.1)
    mark_truncated_y(ax)
    save(fig, "seeds.pdf")
    t = np.array(test[:3])
    return {"test": dict(zip(labels, test)), "val": dict(zip(labels, val)),
            "seed_test_mean": float(t.mean()), "seed_test_sd": float(t.std(ddof=1))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", default=None,
                    help="directory with the Fashion-MNIST IDX .gz files "
                         "(default: <repo>/data, as in src/data.py)")
    ap.add_argument("--recompute-pca", action="store_true",
                    help="refit PCA even if figures/paper/pca_variance.json exists")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    with open(RESULTS) as f:
        ledger = json.load(f)

    nums = pca_variance_numbers(args.data_dir, args.recompute_pca)
    v64 = fig_pca_variance(nums, ledger)
    print(f"[pca] variance at d=64: {v64:.2f} %; 90 % reached at d={nums['dim_for_90pct']}")
    print("[E2] (d, val):", [(d, round(a * 100, 2)) for d, a in fig_pca_sweep(ledger)])
    print("[E4] (k, val):", [(k, round(a * 100, 2)) for k, a in fig_bias_variance(ledger)])
    print("[E8] (tau, val):", [(t, round(a * 100, 2)) for t, a in fig_tau_sweep(ledger)])
    for o, lr, b, fv in fig_optimizer(ledger):
        print(f"[E9] {o:8s} lr={lr:g} best={b * 100:.2f} final={fv * 100:.2f}")
    for k, v in fig_latency(ledger).items():
        print(f"[E5] {k:18s} {v:.3f} s")
    recall, cm = fig_confusion(ledger)
    print(f"[E25] trace={cm.trace()} acc={cm.trace() / N_TEST:.4f}")
    for n, r in recall.items():
        print(f"[E25] recall {n:12s} {r:.1f} %")
    s = fig_seeds(ledger)
    print("[seeds]", json.dumps(s, indent=None))


if __name__ == "__main__":
    main()
