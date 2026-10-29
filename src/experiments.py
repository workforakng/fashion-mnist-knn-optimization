"""Experiment ladder runner.

Usage:
    .venv/bin/python -m src.experiments --scale quick --stages E1,E2
    .venv/bin/python -m src.experiments --scale full --all

Every stage appends one JSON entry to results/results.json (the ledger) and
saves its figures under figures/. Stages already present in the ledger are
skipped unless --force.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from . import classical, evaluate
from .augment import Augmenter
from .data import CLASS_NAMES, load_raw, load_splits
from .softknn import (SoftKNN, predict_probs, predict_probs_tta, load_softknn,
                      embed_chunked)
from .train import train_softknn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_PATH = os.path.join(REPO_ROOT, "results", "results.json")
ARTIFACT_DIR = os.path.join(REPO_ROOT, "artifacts")

SCALES = {
    "tiny": dict(train=4000, val=1500, test=2000, optuna_trials=3, epochs=2,
                 pca_dims=[16, 32], ks=list(range(1, 12, 2)), embed_dim=32),
    "quick": dict(train=12000, val=3000, test=3000, optuna_trials=8, epochs=4,
                  pca_dims=[16, 32, 64, 96, 128], ks=list(range(1, 22, 2)), embed_dim=64),
    "full": dict(train=None, val=6000, test=None, optuna_trials=40, epochs=20,
                 pca_dims=[10, 20, 32, 48, 64, 80, 96, 128], ks=list(range(1, 26, 2)),
                 embed_dim=128),
}


class Ctx:
    """Shared data + PCA cache for one runner session."""

    def __init__(self, scale: str):
        cfg = SCALES[scale]
        self.scale, self.cfg, self.seed = scale, cfg, 42
        xa, ya, xv, yv, xte, yte = load_splits(val_size=cfg["val"], seed=self.seed)
        if cfg["train"] is not None:
            xa, _, ya, _ = train_test_split(xa, ya, train_size=cfg["train"],
                                            stratify=ya, random_state=self.seed)
        if cfg["test"] is not None:
            xte, _, yte, _ = train_test_split(xte, yte, train_size=cfg["test"],
                                              stratify=yte, random_state=self.seed)
        self.xa, self.ya, self.xv, self.yv, self.xte, self.yte = xa, ya, xv, yv, xte, yte
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._pca_cache = {}
        print(f"[ctx] scale={scale} fit={len(xa)} val={len(xv)} test={len(xte)} "
              f"device={self.device}")

    def pca_view(self, dim):
        """(pca, fit_proj, val_proj, test_proj) cached per dim (int or variance float)."""
        if dim not in self._pca_cache:
            pca = classical.fit_pca(self.xa, dim)
            self._pca_cache[dim] = (pca, pca.transform(self.xa),
                                    pca.transform(self.xv), pca.transform(self.xte))
        return self._pca_cache[dim]


# ---------------------------------------------------------------- ledger ----

def load_ledger() -> list:
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return []


def save_ledger(entries: list) -> None:
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(entries, f, indent=2)


# ---------------------------------------------------------------- stages ----

def e0_eda(ctx):
    x_train, y_train, x_test, _ = load_raw()
    counts = np.bincount(y_train, minlength=10)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(10), counts, color="steelblue")
    ax.set_xticks(range(10), CLASS_NAMES, rotation=30, ha="right")
    ax.set_ylabel("images")
    ax.set_title("Fashion-MNIST class distribution (60k train)")
    for i, c in enumerate(counts):
        ax.text(i, c + 100, str(c), ha="center", fontsize=8)
    evaluate.save_fig(fig, "01_class_distribution.png")

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(10, 8, figsize=(8, 10.5))
    for c in range(10):
        idxs = rng.choice(np.where(y_train == c)[0], size=8, replace=False)
        for j, ix in enumerate(idxs):
            axes[c, j].imshow(x_train[ix].reshape(28, 28), cmap="gray_r")
            axes[c, j].axis("off")
        axes[c, 0].set_ylabel(CLASS_NAMES[c], rotation=0, ha="right", va="center", fontsize=7)
    fig.suptitle("Fashion-MNIST samples (8 per class)")
    fig.tight_layout()
    evaluate.save_fig(fig, "02_samples.png")
    return {"name": "eda", "train_counts": counts.tolist()}


def e1_baseline(ctx):
    clf = classical.make_knn(k=5, metric="euclidean", weights="uniform")
    val_acc, fit_s, q_s, _ = classical.eval_knn(clf, ctx.xa, ctx.ya, ctx.xv, ctx.yv)
    test_pred = clf.predict(ctx.xte)
    test_metrics = evaluate.full_metrics(ctx.yte, test_pred)
    print(f"[E1] baseline val={val_acc:.4f} test={test_metrics['accuracy']:.4f} "
          f"fit={fit_s:.1f}s query={q_s:.1f}s")
    return {"name": "baseline_knn_raw784_k5", "config": {"k": 5, "metric": "euclidean",
            "weights": "uniform"}, "val_acc": float(val_acc),
            "test_acc": test_metrics["accuracy"], "fit_s": fit_s, "query_s": q_s,
            "test_f1_macro": test_metrics["f1_macro"]}


def e2_pca_sweep(ctx):
    rows = []
    for d in ctx.cfg["pca_dims"]:
        pca, za, zv, _ = ctx.pca_view(d)
        clf = classical.make_knn(k=5)
        acc, _, _, _ = classical.eval_knn(clf, za, ctx.ya, zv, ctx.yv)
        rows.append({"dim": int(pca.n_components_), "val_acc": float(acc)})
        print(f"[E2] dim={pca.n_components_:3d} val_acc={acc:.4f}")

    pca90, za90, zv90, _ = ctx.pca_view(0.90)
    clf = classical.make_knn(k=5)
    acc90, _, _, _ = classical.eval_knn(clf, za90, ctx.ya, zv90, ctx.yv)
    rows.append({"dim": int(pca90.n_components_), "val_acc": float(acc90),
                 "note": "90% variance"})
    print(f"[E2] dim={pca90.n_components_} (90% var) val_acc={acc90:.4f}")

    best = max(rows, key=lambda r: r["val_acc"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot([r["dim"] for r in rows], [r["val_acc"] for r in rows], "o-")
    ax.axvline(best["dim"], color="crimson", ls="--", label=f"best dim={best['dim']}")
    ax.set_xlabel("# PCA components"); ax.set_ylabel("validation accuracy")
    ax.set_title("PCA dimension sweep (k=5, Euclidean)")
    ax.legend(); ax.grid(alpha=0.3)
    evaluate.save_fig(fig, "04_pca_sweep.png")

    pca_full = ctx.pca_view(ctx.cfg["pca_dims"][-1])[0]
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(range(1, len(cum) + 1), cum, "-")
    ax.axhline(0.90, color="crimson", ls="--", label="90% variance")
    ax.set_xlabel("# components"); ax.set_ylabel("cumulative explained variance")
    ax.set_title("PCA explained variance"); ax.legend(); ax.grid(alpha=0.3)
    evaluate.save_fig(fig, "03_pca_variance.png")

    ctx.best_pca_dim = best["dim"]
    return {"name": "pca_sweep", "rows": rows, "best_dim": best["dim"],
            "var90_dim": int(pca90.n_components_), "best_val_acc": best["val_acc"]}


def e3_optuna(ctx):
    dim = getattr(ctx, "best_pca_dim", 64)
    _, za, zv, zte = ctx.pca_view(dim)
    study, best = classical.optuna_search(za, ctx.ya, zv, ctx.yv,
                                          n_trials=ctx.cfg["optuna_trials"],
                                          seed=ctx.seed)
    clf = classical.make_knn(**best)
    val_acc, fit_s, q_s, _ = classical.eval_knn(clf, za, ctx.ya, zv, ctx.yv)
    test_pred = clf.predict(zte)
    test_metrics = evaluate.full_metrics(ctx.yte, test_pred)
    print(f"[E3] best={best} val={val_acc:.4f} test={test_metrics['accuracy']:.4f}")

    trials = study.trials_dataframe()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(range(len(trials)), trials["value"], "o-", ms=3)
    ax.axhline(study.best_value, color="crimson", ls="--",
               label=f"best={study.best_value:.4f}")
    ax.set_xlabel("trial"); ax.set_ylabel("val accuracy")
    ax.set_title(f"Optuna TPE search ({len(trials)} trials, PCA-{dim})")
    ax.legend(); ax.grid(alpha=0.3)
    evaluate.save_fig(fig, "07_optuna_history.png")

    ctx.best_knn_params, ctx.best_pca_dim = best, dim
    return {"name": "optuna_knn", "pca_dim": dim, "best_params": best,
            "val_acc": float(val_acc), "test_acc": test_metrics["accuracy"],
            "test_f1_macro": test_metrics["f1_macro"], "n_trials": len(trials)}


def e4_bias_variance(ctx):
    dim = getattr(ctx, "best_pca_dim", 64)
    _, za, zv, _ = ctx.pca_view(dim)
    weights = getattr(ctx, "best_knn_params", {}).get("weights", "uniform")
    ks, tr, va = classical.bias_variance_curve(
        za, ctx.ya, zv, ctx.yv, ctx.cfg["ks"], weights=weights)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ks, tr, "o-", label="train accuracy")
    ax.plot(ks, va, "s-", label="validation accuracy")
    ax.set_xlabel("k (neighbors)"); ax.set_ylabel("accuracy")
    ax.set_title(f"Bias-variance tradeoff (PCA-{dim}, weights={weights})")
    ax.legend(); ax.grid(alpha=0.3)
    evaluate.save_fig(fig, "06_bias_variance.png")
    best_k = ks[int(np.argmax(va))]
    return {"name": "bias_variance", "ks": ks, "train_accs": tr, "val_accs": va,
            "best_k": best_k, "best_val_acc": float(max(va))}


def e5_latency(ctx):
    variants = [{"name": "raw-784", "xa": ctx.xa, "xq": ctx.xv},
                {"name": f"pca-{getattr(ctx, 'best_pca_dim', 64)}",
                 "xa": ctx.pca_view(getattr(ctx, "best_pca_dim", 64))[1],
                 "xq": ctx.pca_view(getattr(ctx, "best_pca_dim", 64))[2]}]
    rows = classical.latency_benchmark(ctx.xa, ctx.ya, ctx.xv, ctx.yv, variants)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.25
    algos = ["brute", "kd_tree", "ball_tree"]
    for i, algo in enumerate(algos):
        pts = [(r["variant"], r["query_s_median"]) for r in rows
               if r["algorithm"] == algo and "query_s_median" in r]
        if pts:
            ax.bar([j + i * width for j in range(len(pts))],
                   [p[1] for p in pts], width, label=algo)
    ax.set_xticks([j + width for j in range(len(variants))], [v["name"] for v in variants])
    ax.set_ylabel("query latency, 1k val (s, median of 3)")
    ax.set_title("KNN query latency: algorithm x feature space")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    evaluate.save_fig(fig, "05_latency.png")
    return {"name": "latency_benchmark", "rows": rows}


def _softknn_run(ctx, tag, augment=None, epochs=None, embed_dim=None, tau=0.5,
                 k=7, optimizer="adam", lr=1e-3, bank_copies=0, seed=42):
    pca, _, _, _ = ctx.pca_view(getattr(ctx, "best_pca_dim", 64))
    cfg = {"embed_dim": embed_dim or ctx.cfg["embed_dim"], "k": k, "tau": tau,
           "lr": lr, "optimizer": optimizer, "epochs": epochs or ctx.cfg["epochs"],
           "batch_size": 256, "patience": 4, "augment": augment,
           "augmented_bank_copies": bank_copies,
           "pca_components": pca.components_.copy(), "seed": seed}
    model, history, best_val = train_softknn(ctx.xa, ctx.ya, ctx.xv, ctx.yv,
                                             cfg, device=ctx.device)
    entry_cfg = {kk: vv for kk, vv in cfg.items() if kk != "pca_components"}
    return model, history, best_val, entry_cfg


def _softknn_eval(ctx, model, use_tta=False):
    device = ctx.device
    k = getattr(ctx, "best_k", 7)
    tau = getattr(ctx, "best_tau", 0.5)
    with torch.no_grad():
        bank_z = model.embed(torch.as_tensor(ctx.xa, dtype=torch.float32,
                                             device=device))
        bank_y = torch.as_tensor(ctx.ya, dtype=torch.long, device=device)
    if use_tta:
        t_img = torch.as_tensor(ctx.xte.reshape(len(ctx.xte), 1, 28, 28),
                                dtype=torch.float32)
        test_probs = predict_probs_tta(model, bank_z, bank_y, t_img,
                                       k=k, tau=tau, device=device).numpy()
        v_img = torch.as_tensor(ctx.xv.reshape(len(ctx.xv), 1, 28, 28),
                                dtype=torch.float32)
        val_probs = predict_probs_tta(model, bank_z, bank_y, v_img,
                                      k=k, tau=tau, device=device).numpy()
    else:
        test_probs = predict_probs(model, bank_z, bank_y, ctx.xte, k=k,
                                   tau=tau, device=device).numpy()
        val_probs = predict_probs(model, bank_z, bank_y, ctx.xv, k=k,
                                  tau=tau, device=device).numpy()
    return val_probs, test_probs


def e6_softknn_base(ctx):
    model, history, best_val, ecfg = _softknn_run(ctx, "softknn_base", augment=None)
    val_probs, test_probs = _softknn_eval(ctx, model)
    val_acc = float((val_probs.argmax(1) == ctx.yv).mean())
    test_pred = test_probs.argmax(1)
    test_metrics = evaluate.full_metrics(ctx.yte, test_pred)
    print(f"[E6] softknn base val={val_acc:.4f} test={test_metrics['accuracy']:.4f}")
    ctx._e6_val = val_acc

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot([h["epoch"] for h in history], [h["train_loss"] for h in history], "o-",
             label="train loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("cross-entropy"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot([h["epoch"] for h in history], [h["train_acc"] for h in history], "o-",
             label="train acc")
    ax2.plot([h["epoch"] for h in history], [h["val_acc"] for h in history], "s-",
             label="val acc")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("accuracy"); ax2.legend(); ax2.grid(alpha=0.3)
    fig.suptitle("Soft-KNN (no augmentation) convergence")
    fig.tight_layout()
    evaluate.save_fig(fig, "11_softknn_curves.png")

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ARTIFACT_DIR, "softknn_base.pt"))
    return {"name": "softknn_base", "config": ecfg, "history": history,
            "val_acc": val_acc, "test_acc": test_metrics["accuracy"],
            "test_f1_macro": test_metrics["f1_macro"]}


def e7_softknn_aug(ctx):
    aug = {"hflip_p": 0.5, "shift_px": 2, "erase": True, "rotate_deg": 0.0}
    model, history, best_val, ecfg = _softknn_run(ctx, "softknn_aug", augment=aug)
    val_probs, test_probs = _softknn_eval(ctx, model)
    val_acc = float((val_probs.argmax(1) == ctx.yv).mean())
    test_pred = test_probs.argmax(1)
    test_metrics = evaluate.full_metrics(ctx.yte, test_pred)
    print(f"[E7] softknn+aug val={val_acc:.4f} test={test_metrics['accuracy']:.4f}")

    # augmented samples grid
    x_img = torch.as_tensor(ctx.xa[:8].reshape(8, 1, 28, 28), dtype=torch.float32)
    auger = Augmenter(**aug)
    torch.manual_seed(0)
    show = torch.cat([x_img, auger(x_img)], 0)
    fig, axes = plt.subplots(2, 8, figsize=(10, 2.8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(show[i, 0].numpy(), cmap="gray_r"); ax.axis("off")
    fig.suptitle("Top: originals | Bottom: augmented (hflip, shift, erase)")
    evaluate.save_fig(fig, "08_augmented.png")

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ARTIFACT_DIR, "softknn_aug.pt"))
    ctx.best_softknn = {"aug": val_acc >= getattr(ctx, "_e6_val", 0)}
    ctx._e7_val = val_acc
    return {"name": "softknn_aug", "config": ecfg, "history": history,
            "val_acc": val_acc, "test_acc": test_metrics["accuracy"],
            "test_f1_macro": test_metrics["f1_macro"]}


def e8_tau_sweep(ctx):
    taus = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    rows = []
    for tau in taus:
        model, _, best_val, _ = _softknn_run(
            ctx, f"tau{tau}", augment={"hflip_p": 0.5, "shift_px": 2, "erase": True},
            epochs=max(2, ctx.cfg["epochs"] // 2), tau=tau)
        val_probs, _ = _softknn_eval(ctx, model)
        p = np.clip(val_probs, 1e-9, None)
        entropy = float(-(p * np.log(p)).sum(1).mean() / np.log(10))
        val_acc = float((val_probs.argmax(1) == ctx.yv).mean())
        rows.append({"tau": tau, "val_acc": val_acc, "mean_entropy": entropy})
        print(f"[E8] tau={tau} val={val_acc:.4f} entropy={entropy:.3f}")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot([r["tau"] for r in rows], [r["val_acc"] for r in rows], "o-", label="val acc")
    ax2 = ax.twinx()
    ax2.plot([r["tau"] for r in rows], [r["mean_entropy"] for r in rows],
             "s--", color="darkorange", label="mean pred. entropy")
    ax.set_xlabel("temperature tau"); ax.set_ylabel("validation accuracy")
    ax2.set_ylabel("mean normalized entropy")
    ax.set_title("Temperature sweep: boundary sharpening")
    ax.grid(alpha=0.3)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2)
    evaluate.save_fig(fig, "10_tau_sweep.png")
    best = max(rows, key=lambda r: r["val_acc"])
    ctx.best_tau = best["tau"]
    return {"name": "tau_sweep", "rows": rows, "best_tau": best["tau"]}


def e9_optimizers(ctx):
    rows, curves = [], {}
    for opt in ["adam", "rmsprop", "sgd"]:
        model, history, best_val, _ = _softknn_run(
            ctx, f"opt_{opt}", augment={"hflip_p": 0.5, "shift_px": 2, "erase": True},
            epochs=max(3, ctx.cfg["epochs"] // 2), optimizer=opt,
            lr=0.1 if opt == "sgd" else 1e-3)
        curves[opt] = history
        val_probs, _ = _softknn_eval(ctx, model)
        val_acc = float((val_probs.argmax(1) == ctx.yv).mean())
        rows.append({"optimizer": opt, "lr": 0.1 if opt == "sgd" else 1e-3,
                     "best_val_acc": best_val, "final_val_acc": val_acc})
        print(f"[E9] {opt} best_val={best_val:.4f}")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for opt, hist in curves.items():
        ax1.plot([h["epoch"] for h in hist], [h["train_loss"] for h in hist], "o-",
                 label=opt)
        ax2.plot([h["epoch"] for h in hist], [h["val_acc"] for h in hist], "s-",
                 label=opt)
    ax1.set_xlabel("epoch"); ax1.set_ylabel("train loss"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.set_xlabel("epoch"); ax2.set_ylabel("val acc"); ax2.legend(); ax2.grid(alpha=0.3)
    fig.suptitle("Optimizer comparison (Soft-KNN)")
    fig.tight_layout()
    evaluate.save_fig(fig, "09_optimizer_compare.png")
    best = max(rows, key=lambda r: r["best_val_acc"])
    ctx.best_optimizer = best["optimizer"]
    return {"name": "optimizer_comparison", "rows": rows, "best": best["optimizer"]}


def _best_softknn_tag(ctx):
    """Artifact filename of the highest-val Soft-KNN across the ledger."""
    best = ("softknn_base.pt", -1.0)
    for eid, fname in (("E6", "softknn_base.pt"), ("E7", "softknn_aug.pt"),
                       ("E13", "softknn_shift.pt"), ("E14", "softknn_bankaug.pt")):
        for e in ctx.ledger:
            if e["id"] == eid and e.get("val_acc", 0) > best[1]:
                best = (fname, e["val_acc"])
    return best[0]


def e10_tta(ctx):
    """TTA on the best Soft-KNN artifact from the ledger (no retraining)."""
    tag = _best_softknn_tag(ctx)
    state = torch.load(os.path.join(ARTIFACT_DIR, tag), map_location=ctx.device)
    model = SoftKNN(embed_dim=state["proj.lin.weight"].shape[0]).to(ctx.device)
    model.load_state_dict(state)
    val_probs_plain, test_probs_plain = _softknn_eval(ctx, model, use_tta=False)
    val_probs_tta, test_probs_tta = _softknn_eval(ctx, model, use_tta=True)
    val_plain = float((val_probs_plain.argmax(1) == ctx.yv).mean())
    val_tta = float((val_probs_tta.argmax(1) == ctx.yv).mean())
    test_plain = float((test_probs_plain.argmax(1) == ctx.yte).mean())
    test_tta = float((test_probs_tta.argmax(1) == ctx.yte).mean())
    print(f"[E10] plain val={val_plain:.4f}/test={test_plain:.4f} -> "
          f"TTA val={val_tta:.4f}/test={test_tta:.4f}")
    ctx.tta_test_probs, ctx.tta_val_probs = test_probs_tta, val_probs_tta
    return {"name": "tta", "source_model": tag, "val_plain": val_plain,
            "val_tta": val_tta, "test_plain": test_plain, "test_tta": test_tta}


def e11_ensemble(ctx):
    """Late fusion: TTA Soft-KNN probs + tuned classical KNN probs (PCA space)."""
    dim = getattr(ctx, "best_pca_dim", 64)
    _, za, zv, zte = ctx.pca_view(dim)
    params = getattr(ctx, "best_knn_params", {"k": 5, "metric": "euclidean",
                                              "weights": "distance"})
    clf = classical.make_knn(**params)
    clf.fit(za, ctx.ya)
    knn_val = clf.predict_proba(zv)
    knn_test = clf.predict_proba(zte)

    soft_val = getattr(ctx, "tta_val_probs", None)
    soft_test = getattr(ctx, "tta_test_probs", None)
    if soft_val is None:
        raise RuntimeError("run E10 before E11")

    best_w, best_val = 0.5, -1.0
    for w in np.linspace(0, 1, 11):
        acc = float(((w * soft_val + (1 - w) * knn_val).argmax(1) == ctx.yv).mean())
        if acc > best_val:
            best_w, best_val = float(w), acc
    fused_test = best_w * soft_test + (1 - best_w) * knn_test
    test_pred = fused_test.argmax(1)
    test_metrics = evaluate.full_metrics(ctx.yte, test_pred)
    print(f"[E11] fusion w_soft={best_w:.1f} val={best_val:.4f} "
          f"test={test_metrics['accuracy']:.4f}")
    return {"name": "ensemble_fusion", "fusion_weight_softknn": best_w,
            "knn_params": params, "pca_dim": dim, "val_acc": best_val,
            "test_acc": test_metrics["accuracy"],
            "test_f1_macro": test_metrics["f1_macro"],
            "precision_macro": test_metrics["precision_macro"],
            "recall_macro": test_metrics["recall_macro"]}


def e12_inference_grid(ctx):
    """Free wins: sweep inference-time (k, tau) on the frozen best Soft-KNN.

    Also produces the proper boundary-sharpening study: one fixed metric W,
    tau varied at inference only -> accuracy + prediction entropy response.
    """
    tag = _best_softknn_tag(ctx)
    state = torch.load(os.path.join(ARTIFACT_DIR, tag), map_location=ctx.device)
    model = SoftKNN(embed_dim=state["proj.lin.weight"].shape[0]).to(ctx.device)
    model.load_state_dict(state)
    with torch.no_grad():
        bank_z = model.embed(torch.as_tensor(ctx.xa, dtype=torch.float32,
                                             device=ctx.device))
        bank_y = torch.as_tensor(ctx.ya, dtype=torch.long, device=ctx.device)

    ks, taus = [3, 5, 7, 9, 11], [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    grid, ent_by_tau = [], {}
    for k in ks:
        for tau in taus:
            probs = predict_probs(model, bank_z, bank_y, ctx.xv, k=k, tau=tau,
                                  device=ctx.device).numpy()
            acc = float((probs.argmax(1) == ctx.yv).mean())
            grid.append({"k": k, "tau": tau, "val_acc": acc})
    for tau in taus:
        probs = predict_probs(model, bank_z, bank_y, ctx.xv, k=7, tau=tau,
                              device=ctx.device).numpy()
        p = np.clip(probs, 1e-9, None)
        ent_by_tau[tau] = float(-(p * np.log(p)).sum(1).mean() / np.log(10))

    best = max(grid, key=lambda r: r["val_acc"])
    print(f"[E12] best inference (k={best['k']}, tau={best['tau']}) "
          f"val={best['val_acc']:.4f}")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for k in ks:
        pts = [(r["tau"], r["val_acc"]) for r in grid if r["k"] == k]
        ax1.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=f"k={k}")
    ax1.set_xlabel("inference tau"); ax1.set_ylabel("val accuracy")
    ax1.set_title("Inference grid (frozen W)"); ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax2.plot(taus, [ent_by_tau[t] for t in taus], "s-", color="darkorange")
    ax2.set_xlabel("inference tau"); ax2.set_ylabel("mean normalized entropy")
    ax2.set_title("Boundary sharpening: entropy vs tau")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    evaluate.save_fig(fig, "14_inference_grid.png")

    test_probs = predict_probs(model, bank_z, bank_y, ctx.xte, k=best["k"],
                               tau=best["tau"], device=ctx.device).numpy()
    test_acc = float((test_probs.argmax(1) == ctx.yte).mean())
    ctx.best_k, ctx.best_tau = best["k"], best["tau"]
    ctx.soft_test_probs, ctx.soft_val_probs = test_probs, None
    return {"name": "inference_grid", "source": tag, "grid_best": best,
            "entropy_by_tau": ent_by_tau, "val_acc": best["val_acc"],
            "test_acc": test_acc}


def e13_shift_aug(ctx):
    """Mild augmentation: shift-only queries (flip/erase proved harmful in E7)."""
    aug = {"hflip_p": 0.0, "shift_px": 2, "erase": False}
    model, history, best_val, ecfg = _softknn_run(ctx, "shift_aug", augment=aug)
    val_probs, test_probs = _softknn_eval(ctx, model)
    val_acc = float((val_probs.argmax(1) == ctx.yv).mean())
    test_pred = test_probs.argmax(1)
    test_metrics = evaluate.full_metrics(ctx.yte, test_pred)
    print(f"[E13] shift-only aug val={val_acc:.4f} test={test_metrics['accuracy']:.4f}")
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ARTIFACT_DIR, "softknn_shift.pt"))
    if val_acc > getattr(ctx, "_e6_val", 0):
        ctx._e7_val = val_acc
    return {"name": "softknn_shift_aug", "config": ecfg, "history": history,
            "val_acc": val_acc, "test_acc": test_metrics["accuracy"],
            "test_f1_macro": test_metrics["f1_macro"]}


def e14_bank_aug(ctx):
    """Novelty N4 done right: clean queries, bank enriched with shifted copies."""
    aug = {"hflip_p": 0.0, "shift_px": 2, "erase": False}
    cfg = {"embed_dim": ctx.cfg["embed_dim"], "k": 7, "tau": 0.5, "lr": 1e-3,
           "optimizer": "adam", "epochs": ctx.cfg["epochs"], "batch_size": 256,
           "patience": 4, "augment": aug, "augmented_bank_copies": 1,
           "augment_queries": False,
           "pca_components": ctx.pca_view(getattr(ctx, "best_pca_dim", 64))[0].components_.copy(),
           "seed": 42}
    model, history, best_val = train_softknn(ctx.xa, ctx.ya, ctx.xv, ctx.yv,
                                             cfg, device=ctx.device)
    val_probs, test_probs = _softknn_eval(ctx, model)
    val_acc = float((val_probs.argmax(1) == ctx.yv).mean())
    test_pred = test_probs.argmax(1)
    test_metrics = evaluate.full_metrics(ctx.yte, test_pred)
    print(f"[E14] bank-aug val={val_acc:.4f} test={test_metrics['accuracy']:.4f}")
    torch.save(model.state_dict(), os.path.join(ARTIFACT_DIR, "softknn_bankaug.pt"))
    if val_acc > getattr(ctx, "_e6_val", 0):
        ctx._e7_val = val_acc
    return {"name": "softknn_bank_aug", "config": {kk: vv for kk, vv in cfg.items() if kk != "pca_components"},
            "history": history, "val_acc": val_acc,
            "test_acc": test_metrics["accuracy"],
            "test_f1_macro": test_metrics["f1_macro"]}


def e15_seed_ensemble(ctx):
    """3-seed ensemble of the best Soft-KNN config + final late fusion."""
    use_aug = getattr(ctx, "_e7_val", 0) >= getattr(ctx, "_e6_val", 0)
    tag = ("softknn_shift.pt" if os.path.exists(os.path.join(ARTIFACT_DIR, "softknn_shift.pt"))
           and getattr(ctx, "_e7_val", 0) > getattr(ctx, "_e6_val", 0)
           else "softknn_base.pt")
    state = torch.load(os.path.join(ARTIFACT_DIR, tag), map_location=ctx.device)
    base_cfg = {"embed_dim": ctx.cfg["embed_dim"], "k": 7, "tau": 0.5, "lr": 1e-3,
                "optimizer": "adam", "epochs": ctx.cfg["epochs"], "batch_size": 256,
                "patience": 4, "augment": None, "augmented_bank_copies": 0,
                "pca_components": ctx.pca_view(getattr(ctx, "best_pca_dim", 64))[0].components_.copy()}
    seed_probs_val, seed_probs_test = [], []
    for seed in (42, 7, 123):
        cfg = dict(base_cfg, seed=seed)
        model, _hist, _bv = train_softknn(ctx.xa, ctx.ya, ctx.xv, ctx.yv,
                                          cfg, device=ctx.device,
                                          verbose=(seed == 42))
        with torch.no_grad():
            bank_z = model.embed(torch.as_tensor(ctx.xa, dtype=torch.float32,
                                                 device=ctx.device))
            bank_y = torch.as_tensor(ctx.ya, dtype=torch.long, device=ctx.device)
        pv = predict_probs(model, bank_z, bank_y, ctx.xv,
                           k=getattr(ctx, "best_k", 7),
                           tau=getattr(ctx, "best_tau", 0.5),
                           device=ctx.device).numpy()
        pt = predict_probs(model, bank_z, bank_y, ctx.xte,
                           k=getattr(ctx, "best_k", 7),
                           tau=getattr(ctx, "best_tau", 0.5),
                           device=ctx.device).numpy()
        seed_probs_val.append(pv)
        seed_probs_test.append(pt)
        print(f"[E15] seed {seed}: val={(pv.argmax(1) == ctx.yv).mean():.4f}")
    ens_val = np.mean(seed_probs_val, axis=0)
    ens_test = np.mean(seed_probs_test, axis=0)
    ens_val_acc = float((ens_val.argmax(1) == ctx.yv).mean())
    ens_test_acc = float((ens_test.argmax(1) == ctx.yte).mean())
    print(f"[E15] seed-ensemble val={ens_val_acc:.4f} test={ens_test_acc:.4f}")

    dim = getattr(ctx, "best_pca_dim", 64)
    _, za, zv, zte = ctx.pca_view(dim)
    params = getattr(ctx, "best_knn_params", {"k": 5, "metric": "manhattan",
                                              "weights": "distance"})
    clf = classical.make_knn(**params)
    clf.fit(za, ctx.ya)
    knn_val, knn_test = clf.predict_proba(zv), clf.predict_proba(zte)
    best_w, best_val = 0.5, -1.0
    for w in np.linspace(0, 1, 11):
        acc = float(((w * ens_val + (1 - w) * knn_val).argmax(1) == ctx.yv).mean())
        if acc > best_val:
            best_w, best_val = float(w), acc
    fused = best_w * ens_test + (1 - best_w) * knn_test
    test_metrics = evaluate.full_metrics(ctx.yte, fused.argmax(1))
    print(f"[E15] fusion w={best_w:.1f} val={best_val:.4f} "
          f"test={test_metrics['accuracy']:.4f}")
    evaluate.plot_confusion_matrix(ctx.yte, fused.argmax(1))
    return {"name": "seed_ensemble_fusion", "source": tag, "seeds": [42, 7, 123],
            "ens_val_acc": ens_val_acc, "ens_test_acc": ens_test_acc,
            "fusion_weight_softknn": best_w, "val_acc": best_val,
            "test_acc": test_metrics["accuracy"],
            "test_f1_macro": test_metrics["f1_macro"],
            "precision_macro": test_metrics["precision_macro"],
            "recall_macro": test_metrics["recall_macro"]}


def e16_mlp_metric(ctx):
    """Nonlinear metric: 784 -> 256 (ReLU) -> 128, PCA warm-started first layer."""
    pca256, _, _, _ = ctx.pca_view(256)
    cfg = {"embed_dim": 128, "hidden_dim": 256, "k": 7, "tau": 0.5, "lr": 3e-4,
           "optimizer": "adam", "epochs": max(24, ctx.cfg["epochs"]),
           "batch_size": 256, "patience": 6, "augment": None,
           "augmented_bank_copies": 0, "bank_refresh_steps": 107, "clip_grad": 1.0,
           "pca_components": pca256.components_.copy(), "seed": 42}
    model, history, best_val = train_softknn(ctx.xa, ctx.ya, ctx.xv, ctx.yv,
                                             cfg, device=ctx.device)
    with torch.no_grad():
        bank_z = model.embed(torch.as_tensor(ctx.xa, dtype=torch.float32,
                                             device=ctx.device))
        bank_y = torch.as_tensor(ctx.ya, dtype=torch.long, device=ctx.device)
    grid = []
    for k in (5, 7, 9):
        for tau in (0.1, 0.3, 0.5):
            probs = predict_probs(model, bank_z, bank_y, ctx.xv, k=k, tau=tau,
                                  device=ctx.device).numpy()
            acc = float((probs.argmax(1) == ctx.yv).mean())
            grid.append({"k": k, "tau": tau, "val_acc": acc})
    best = max(grid, key=lambda r: r["val_acc"])
    test_probs = predict_probs(model, bank_z, bank_y, ctx.xte, k=best["k"],
                               tau=best["tau"], device=ctx.device).numpy()
    test_acc = float((test_probs.argmax(1) == ctx.yte).mean())
    print(f"[E16] mlp metric best grid={best} test={test_acc:.4f}")
    torch.save(model.state_dict(), os.path.join(ARTIFACT_DIR, "softknn_mlp.pt"))
    ctx.mlp_val_probs = predict_probs(model, bank_z, bank_y, ctx.xv,
                                      k=best["k"], tau=best["tau"],
                                      device=ctx.device).numpy()
    ctx.mlp_test_probs = test_probs
    ctx.mlp_grid_best = best
    return {"name": "mlp_metric", "config": {kk: vv for kk, vv in cfg.items() if kk != "pca_components"},
            "grid": grid, "grid_best": best, "val_acc": best["val_acc"],
            "test_acc": test_acc, "best_train_val": best_val}


def e17_champion_fusion(ctx):
    """Final model: val-weighted fusion of every strong probability source."""
    sources_val, sources_test, names = [], [], []

    def add(pv, pt, name):
        sources_val.append(pv)
        sources_test.append(pt)
        names.append(name)

    tag = _best_softknn_tag(ctx)
    model = load_softknn(os.path.join(ARTIFACT_DIR, tag), ctx.device)
    with torch.no_grad():
        bank_z = model.embed(torch.as_tensor(ctx.xa, dtype=torch.float32,
                                             device=ctx.device))
        bank_y = torch.as_tensor(ctx.ya, dtype=torch.long, device=ctx.device)
    k0, t0 = 5, 0.1
    for e in ctx.ledger:
        if e["id"] == "E12" and e.get("grid_best"):
            k0, t0 = e["grid_best"]["k"], e["grid_best"]["tau"]
    add(predict_probs(model, bank_z, bank_y, ctx.xv, k=k0, tau=t0,
                      device=ctx.device).numpy(),
        predict_probs(model, bank_z, bank_y, ctx.xte, k=k0, tau=t0,
                      device=ctx.device).numpy(), f"softknn[{tag}]")

    if (os.path.exists(os.path.join(ARTIFACT_DIR, "softknn_bankaug.pt"))
            and tag != "softknn_bankaug.pt"):
        m2 = load_softknn(os.path.join(ARTIFACT_DIR, "softknn_bankaug.pt"), ctx.device)
        with torch.no_grad():
            bz2 = m2.embed(torch.as_tensor(ctx.xa, dtype=torch.float32,
                                           device=ctx.device))
        add(predict_probs(m2, bz2, bank_y, ctx.xv, k=k0, tau=t0,
                          device=ctx.device).numpy(),
            predict_probs(m2, bz2, bank_y, ctx.xte, k=k0, tau=t0,
                          device=ctx.device).numpy(), "softknn[bankaug]")
    mlp_entry = next((e for e in ctx.ledger
                      if e["id"] == "E16" and e.get("grid_best")), None)
    if mlp_entry and os.path.exists(os.path.join(ARTIFACT_DIR, "softknn_mlp.pt")):
        m3 = load_softknn(os.path.join(ARTIFACT_DIR, "softknn_mlp.pt"), ctx.device)
        with torch.no_grad():
            bz3 = m3.embed(torch.as_tensor(ctx.xa, dtype=torch.float32,
                                           device=ctx.device))
        k1, t1 = mlp_entry["grid_best"]["k"], mlp_entry["grid_best"]["tau"]
        add(predict_probs(m3, bz3, bank_y, ctx.xv, k=k1, tau=t1,
                          device=ctx.device).numpy(),
            predict_probs(m3, bz3, bank_y, ctx.xte, k=k1, tau=t1,
                          device=ctx.device).numpy(), "softknn[mlp]")

    dim = getattr(ctx, "best_pca_dim", 64)
    _, za, zv, zte = ctx.pca_view(dim)
    params = getattr(ctx, "best_knn_params", {"k": 5, "metric": "manhattan",
                                              "weights": "distance"})
    clf = classical.make_knn(**params)
    clf.fit(za, ctx.ya)
    add(clf.predict_proba(zv), clf.predict_proba(zte), "knn_classical")

    best_w, best_val = None, -1.0
    n = len(sources_val)
    for combo in _weight_simplex(n):
        acc = float((sum(wi * p for wi, p in
                         zip(combo, sources_val)).argmax(1) == ctx.yv).mean())
        if acc > best_val:
            best_w, best_val = combo, acc
    fused_test = sum(wi * p for wi, p in zip(best_w, sources_test))
    test_metrics = evaluate.full_metrics(ctx.yte, fused_test.argmax(1))
    print(f"[E17] sources={names} weights={best_w} val={best_val:.4f} "
          f"test={test_metrics['accuracy']:.4f}")
    evaluate.plot_confusion_matrix(ctx.yte, fused_test.argmax(1))
    return {"name": "champion_fusion", "sources": names,
            "weights": [round(w, 2) for w in best_w], "val_acc": best_val,
            "test_acc": test_metrics["accuracy"],
            "test_f1_macro": test_metrics["f1_macro"],
            "precision_macro": test_metrics["precision_macro"],
            "recall_macro": test_metrics["recall_macro"]}


def _weight_simplex(n: int, step: float = 0.1):
    """All weight vectors over n sources summing to 1.0 on a fixed grid."""
    import itertools
    steps = int(round(1.0 / step))
    for parts in itertools.product(range(steps + 1), repeat=n):
        if sum(parts) == steps:
            yield tuple(p * step for p in parts)


def e19_conv_metric(ctx):
    """Conv metric: learned 2-block CNN embedding, classified by Soft-KNN."""
    cfg = {"embed_dim": 128, "metric": "conv", "k": 7, "tau": 0.5, "lr": 3e-4,
           "optimizer": "adam", "epochs": max(30, ctx.cfg["epochs"]),
           "batch_size": 256, "patience": 8, "augment": None,
           "augmented_bank_copies": 0, "bank_refresh_steps": 107, "clip_grad": 1.0,
           "seed": 42}
    model, history, best_val = train_softknn(ctx.xa, ctx.ya, ctx.xv, ctx.yv,
                                             cfg, device=ctx.device)
    with torch.no_grad():
        bank_z = embed_chunked(model, torch.as_tensor(ctx.xa, dtype=torch.float32,
                                                      device=ctx.device))
        bank_y = torch.as_tensor(ctx.ya, dtype=torch.long, device=ctx.device)
    grid = []
    for k in (5, 7, 9):
        for tau in (0.1, 0.3, 0.5):
            probs = predict_probs(model, bank_z, bank_y, ctx.xv, k=k, tau=tau,
                                  device=ctx.device).numpy()
            acc = float((probs.argmax(1) == ctx.yv).mean())
            grid.append({"k": k, "tau": tau, "val_acc": acc})
    best = max(grid, key=lambda r: r["val_acc"])
    test_probs = predict_probs(model, bank_z, bank_y, ctx.xte, k=best["k"],
                               tau=best["tau"], device=ctx.device).numpy()
    test_acc = float((test_probs.argmax(1) == ctx.yte).mean())
    print(f"[E19] conv metric best grid={best} test={test_acc:.4f}")
    torch.save(model.state_dict(), os.path.join(ARTIFACT_DIR, "softknn_conv.pt"))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot([h["epoch"] for h in history], [h["train_loss"] for h in history],
               marker="o")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("train loss")
    ax[1].plot([h["epoch"] for h in history], [h["val_acc"] for h in history],
               marker="o", color="tab:orange")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("val acc")
    fig.suptitle("E19: conv-metric Soft-KNN training curves")
    evaluate.save_fig(fig, "13_conv_curves.png")
    return {"name": "conv_metric", "config": dict(cfg),
            "grid": grid, "grid_best": best, "val_acc": best["val_acc"],
            "test_acc": test_acc, "best_train_val": best_val,
            "history": history}


def e20_conv_shift(ctx):
    """Conv metric + shift-only augmented queries (conv features are
    shift-tolerant, so augmentation should help where it hurt the linear
    metric)."""
    cfg = {"embed_dim": 128, "metric": "conv", "k": 7, "tau": 0.5, "lr": 3e-4,
           "optimizer": "adam", "epochs": max(40, ctx.cfg["epochs"]),
           "batch_size": 256, "patience": 8,
           "augment": {"hflip_p": 0.0, "shift_px": 2, "erase": False,
                       "rotate_deg": 0.0},
           "augmented_bank_copies": 0, "bank_refresh_steps": 107, "clip_grad": 1.0,
           "seed": 42}
    model, history, best_val = train_softknn(ctx.xa, ctx.ya, ctx.xv, ctx.yv,
                                             cfg, device=ctx.device)
    with torch.no_grad():
        bank_z = embed_chunked(model, torch.as_tensor(ctx.xa, dtype=torch.float32,
                                                      device=ctx.device))
        bank_y = torch.as_tensor(ctx.ya, dtype=torch.long, device=ctx.device)
    grid = []
    for k in (5, 7, 9):
        for tau in (0.1, 0.3, 0.5):
            probs = predict_probs(model, bank_z, bank_y, ctx.xv, k=k, tau=tau,
                                  device=ctx.device).numpy()
            acc = float((probs.argmax(1) == ctx.yv).mean())
            grid.append({"k": k, "tau": tau, "val_acc": acc})
    best = max(grid, key=lambda r: r["val_acc"])
    test_probs = predict_probs(model, bank_z, bank_y, ctx.xte, k=best["k"],
                               tau=best["tau"], device=ctx.device).numpy()
    test_acc = float((test_probs.argmax(1) == ctx.yte).mean())
    print(f"[E20] conv+shift best grid={best} test={test_acc:.4f}")
    torch.save(model.state_dict(), os.path.join(ARTIFACT_DIR, "softknn_conv_shift.pt"))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot([h["epoch"] for h in history], [h["train_loss"] for h in history],
               marker="o")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("train loss")
    ax[1].plot([h["epoch"] for h in history], [h["val_acc"] for h in history],
               marker="o", color="tab:orange")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("val acc")
    fig.suptitle("E20: conv metric + shift-augmented queries")
    evaluate.save_fig(fig, "15_conv_shift_curves.png")
    return {"name": "conv_shift_metric", "config": dict(cfg),
            "grid": grid, "grid_best": best, "val_acc": best["val_acc"],
            "test_acc": test_acc, "best_train_val": best_val,
            "history": history}


def e21_conv_ensemble(ctx):
    """Conv-metric seed ensemble: 3 seeds x 50 epochs, probability average."""
    seeds = [42, 7, 123]
    seed_probs_val, seed_probs_test, per_seed = [], [], []
    for seed in seeds:
        artifact = os.path.join(ARTIFACT_DIR, f"softknn_conv_s{seed}.pt")
        if os.path.exists(artifact):
            model = load_softknn(artifact, ctx.device)
            history, best_val = [], None
        else:
            cfg = {"embed_dim": 128, "metric": "conv", "k": 7, "tau": 0.5,
                   "lr": 3e-4, "optimizer": "adam", "epochs": 50,
                   "batch_size": 256, "patience": 10, "augment": None,
                   "augmented_bank_copies": 0, "bank_refresh_steps": 107,
                   "clip_grad": 1.0, "seed": seed}
            model, history, best_val = train_softknn(ctx.xa, ctx.ya, ctx.xv,
                                                     ctx.yv, cfg,
                                                     device=ctx.device)
            torch.save(model.state_dict(), artifact)
        with torch.no_grad():
            bank_z = embed_chunked(model, torch.as_tensor(ctx.xa,
                                                          dtype=torch.float32,
                                                          device=ctx.device))
            bank_y = torch.as_tensor(ctx.ya, dtype=torch.long, device=ctx.device)
        pv = predict_probs(model, bank_z, bank_y, ctx.xv, k=9, tau=0.1,
                           device=ctx.device).numpy()
        pt = predict_probs(model, bank_z, bank_y, ctx.xte, k=9, tau=0.1,
                           device=ctx.device).numpy()
        seed_probs_val.append(pv)
        seed_probs_test.append(pt)
        torch.save(model.state_dict(),
                   os.path.join(ARTIFACT_DIR, f"softknn_conv_s{seed}.pt"))
        v_acc = float((pv.argmax(1) == ctx.yv).mean())
        t_acc = float((pt.argmax(1) == ctx.yte).mean())
        per_seed.append({"seed": seed, "val_acc": v_acc, "test_acc": t_acc,
                         "best_train_val": best_val, "epochs_run": len(history)})
        print(f"[E21] conv seed {seed}: val={v_acc:.4f} test={t_acc:.4f}")
    ens_val = np.mean(seed_probs_val, axis=0)
    ens_test = np.mean(seed_probs_test, axis=0)
    val_acc = float((ens_val.argmax(1) == ctx.yv).mean())
    test_acc = float((ens_test.argmax(1) == ctx.yte).mean())
    print(f"[E21] conv ensemble (3 seeds): val={val_acc:.4f} test={test_acc:.4f}")
    return {"name": "conv_ensemble", "seeds": seeds, "per_seed": per_seed,
            "val_acc": val_acc, "test_acc": test_acc}


def _collect_sources(ctx, tta: bool = False):
    """(val_probs, test_probs, names) for every strong artifact source.

    With tta=True every Soft-KNN source is smoothed by query-side
    test-time augmentation (six deterministic views); the classical
    k-NN source is left unchanged (it operates on PCA features of raw
    pixels and has no learned embedding to augment).
    """
    sources_val, sources_test, names = [], [], []
    xa_t = torch.as_tensor(ctx.xa, dtype=torch.float32, device=ctx.device)
    ya_t = torch.as_tensor(ctx.ya, dtype=torch.long, device=ctx.device)
    if tta:
        v_img = torch.as_tensor(ctx.xv.reshape(len(ctx.xv), 1, 28, 28),
                                dtype=torch.float32)
        t_img = torch.as_tensor(ctx.xte.reshape(len(ctx.xte), 1, 28, 28),
                                dtype=torch.float32)

    def _probs(m, bz, k, tau):
        if tta:
            pv = predict_probs_tta(m, bz, ya_t, v_img, k=k, tau=tau,
                                   device=ctx.device).numpy()
            pt = predict_probs_tta(m, bz, ya_t, t_img, k=k, tau=tau,
                                   device=ctx.device).numpy()
        else:
            pv = predict_probs(m, bz, ya_t, ctx.xv, k=k, tau=tau,
                               device=ctx.device).numpy()
            pt = predict_probs(m, bz, ya_t, ctx.xte, k=k, tau=tau,
                               device=ctx.device).numpy()
        return pv, pt

    for seed in (42, 7, 123, 2024, 31415):
        path = os.path.join(ARTIFACT_DIR, f"softknn_conv_s{seed}.pt")
        if os.path.exists(path):
            m = load_softknn(path, ctx.device)
            with torch.no_grad():
                bz = embed_chunked(m, xa_t)
            pv, pt = _probs(m, bz, 9, 0.1)
            sources_val.append(pv)
            sources_test.append(pt)
            names.append(f"conv_s{seed}")

    path = os.path.join(ARTIFACT_DIR, "softknn_conv_distill_k5.pt")
    if os.path.exists(path):
        m = load_softknn(path, ctx.device)
        with torch.no_grad():
            bz = embed_chunked(m, xa_t)
        pv, pt = _probs(m, bz, 9, 0.1)
        sources_val.append(pv)
        sources_test.append(pt)
        names.append("conv_distill_k5")

    path = os.path.join(ARTIFACT_DIR, "softknn_bankaug.pt")
    if os.path.exists(path):
        m = load_softknn(path, ctx.device)
        with torch.no_grad():
            bz = embed_chunked(m, xa_t)
        pv, pt = _probs(m, bz, 5, 0.1)
        sources_val.append(pv)
        sources_test.append(pt)
        names.append("bankaug_linear")

    mlp_entry = next((e for e in ctx.ledger
                      if e["id"] == "E16" and e.get("grid_best")), None)
    path = os.path.join(ARTIFACT_DIR, "softknn_mlp.pt")
    if mlp_entry and os.path.exists(path):
        m = load_softknn(path, ctx.device)
        with torch.no_grad():
            bz = embed_chunked(m, xa_t)
        k1, t1 = mlp_entry["grid_best"]["k"], mlp_entry["grid_best"]["tau"]
        pv, pt = _probs(m, bz, k1, t1)
        sources_val.append(pv)
        sources_test.append(pt)
        names.append("mlp_metric")

    dim = getattr(ctx, "best_pca_dim", 64)
    _, za, zv, zte = ctx.pca_view(dim)
    params = getattr(ctx, "best_knn_params", {"k": 5, "metric": "manhattan",
                                              "weights": "distance"})
    clf = classical.make_knn(**params)
    clf.fit(za, ctx.ya)
    sources_val.append(clf.predict_proba(zv))
    sources_test.append(clf.predict_proba(zte))
    names.append("knn_classical")
    return sources_val, sources_test, names


def _fusion_eval(ctx, sources_val, sources_test, names, name):
    """Val-weighted simplex fusion of probability sources -> test metrics."""
    best_w, best_val = None, -1.0
    for combo in _weight_simplex(len(sources_val)):
        acc = float((sum(wi * p for wi, p in
                         zip(combo, sources_val)).argmax(1) == ctx.yv).mean())
        if acc > best_val:
            best_w, best_val = combo, acc
    fused_test = sum(wi * p for wi, p in zip(best_w, sources_test))
    test_metrics = evaluate.full_metrics(ctx.yte, fused_test.argmax(1))
    print(f"[{name}] sources={names} weights={best_w} val={best_val:.4f} "
          f"test={test_metrics['accuracy']:.4f}")
    evaluate.plot_confusion_matrix(ctx.yte, fused_test.argmax(1))
    return {"name": name, "sources": names,
            "weights": [round(w, 2) for w in best_w], "val_acc": best_val,
            "test_acc": test_metrics["accuracy"],
            "test_f1_macro": test_metrics["f1_macro"],
            "precision_macro": test_metrics["precision_macro"],
            "recall_macro": test_metrics["recall_macro"]}


def e22_final_fusion(ctx):
    """Final champion: ledger-driven fusion of all strong probability sources."""
    sv, st, names = _collect_sources(ctx)
    return _fusion_eval(ctx, sv, st, names, "E22")


def e23_distill(ctx):
    """Self-distillation: conv-seed ensemble teacher -> fresh conv student."""
    teacher, n_te = None, 0
    xa_t = torch.as_tensor(ctx.xa, dtype=torch.float32, device=ctx.device)
    ya_t = torch.as_tensor(ctx.ya, dtype=torch.long, device=ctx.device)
    for seed in (42, 7, 123):
        path = os.path.join(ARTIFACT_DIR, f"softknn_conv_s{seed}.pt")
        if os.path.exists(path):
            m = load_softknn(path, ctx.device)
            with torch.no_grad():
                bz = embed_chunked(m, xa_t)
            p = predict_probs(m, bz, ya_t, ctx.xa, k=9, tau=1.0,
                              device=ctx.device).numpy()
            teacher = p if teacher is None else teacher + p
            n_te += 1
    if not n_te:
        print("[E23] no teacher artifacts found; skipping")
        return {"name": "distill", "skipped": True}
    distill_path = os.path.join(ARTIFACT_DIR, "softknn_conv_distill.pt")
    if os.path.exists(distill_path):
        model = load_softknn(distill_path, ctx.device)
        with torch.no_grad():
            bank_z = embed_chunked(model, xa_t)
        grid = []
        for k in (7, 9):
            for tau in (0.1, 0.3):
                probs = predict_probs(model, bank_z, ya_t, ctx.xv, k=k, tau=tau,
                                      device=ctx.device).numpy()
                acc = float((probs.argmax(1) == ctx.yv).mean())
                grid.append({"k": k, "tau": tau, "val_acc": acc})
        best = max(grid, key=lambda r: r["val_acc"])
        test_probs = predict_probs(model, bank_z, ya_t, ctx.xte, k=best["k"],
                                   tau=best["tau"], device=ctx.device).numpy()
        test_acc = float((test_probs.argmax(1) == ctx.yte).mean())
        print(f"[E23] reusing existing distill artefact; grid_best={best} "
              f"test={test_acc:.4f} (no retraining)")
        return {"name": "distill", "reused_artifact": True,
                "teacher_models": n_te, "grid": grid, "grid_best": best,
                "val_acc": best["val_acc"], "test_acc": test_acc}
    teacher = teacher / n_te
    cfg = {"embed_dim": 128, "metric": "conv", "k": 7, "tau": 0.5, "lr": 3e-4,
           "optimizer": "adam", "epochs": 35, "batch_size": 256, "patience": 10,
           "augment": None, "augmented_bank_copies": 0, "bank_refresh_steps": 107,
           "clip_grad": 1.0, "seed": 100, "amp": True,
           "distill_soft_targets": teacher, "distill_alpha": 0.5, "distill_T": 1.0}
    model, history, best_val = train_softknn(ctx.xa, ctx.ya, ctx.xv, ctx.yv,
                                             cfg, device=ctx.device)
    with torch.no_grad():
        bank_z = embed_chunked(model, xa_t)
    grid = []
    for k in (7, 9):
        for tau in (0.1, 0.3):
            probs = predict_probs(model, bank_z, ya_t, ctx.xv, k=k, tau=tau,
                                  device=ctx.device).numpy()
            acc = float((probs.argmax(1) == ctx.yv).mean())
            grid.append({"k": k, "tau": tau, "val_acc": acc})
    best = max(grid, key=lambda r: r["val_acc"])
    test_probs = predict_probs(model, bank_z, ya_t, ctx.xte, k=best["k"],
                               tau=best["tau"], device=ctx.device).numpy()
    test_acc = float((test_probs.argmax(1) == ctx.yte).mean())
    print(f"[E23] distill student best grid={best} test={test_acc:.4f}")
    torch.save(model.state_dict(),
               os.path.join(ARTIFACT_DIR, "softknn_conv_distill.pt"))
    return {"name": "distill", "config": {k: v for k, v in cfg.items()
                                          if k != "distill_soft_targets"},
            "teacher_models": n_te, "grid": grid, "grid_best": best,
            "val_acc": best["val_acc"], "test_acc": test_acc,
            "best_train_val": best_val, "history": history}


def e24_fusion_distill(ctx):
    """Final fusion v2: all sources + the distilled student."""
    sv, st, names = _collect_sources(ctx)
    path = os.path.join(ARTIFACT_DIR, "softknn_conv_distill.pt")
    if os.path.exists(path):
        m = load_softknn(path, ctx.device)
        xa_t = torch.as_tensor(ctx.xa, dtype=torch.float32, device=ctx.device)
        ya_t = torch.as_tensor(ctx.ya, dtype=torch.long, device=ctx.device)
        with torch.no_grad():
            bz = embed_chunked(m, xa_t)
        sv.append(predict_probs(m, bz, ya_t, ctx.xv, k=9, tau=0.1,
                                device=ctx.device).numpy())
        st.append(predict_probs(m, bz, ya_t, ctx.xte, k=9, tau=0.1,
                                device=ctx.device).numpy())
        names.append("conv_distill")
    return _fusion_eval(ctx, sv, st, names, "E24")


def e26_distill_gen2(ctx):
    """Second-generation self-distillation: teacher = weighted source fusion.

    The teacher is the champion fusion (E24 validation-optimised weights)
    evaluated on the training bank, i.e. a strictly stronger target than the
    plain conv-seed mean used by E23. A fresh conv student (seed 200) is
    trained on the fused soft targets and saved as its own artefact so it can
    join later fusions as an additional source.
    """
    e24 = next((e for e in ctx.ledger if e.get("id") == "E24"), None)
    if not e24 or "weights" not in e24:
        print("[E26] E24 fusion entry missing from ledger; skipping")
        return {"name": "distill_gen2", "skipped": True}
    weights = {n: w for n, w in zip(e24["sources"], e24["weights"])}

    xa_t = torch.as_tensor(ctx.xa, dtype=torch.float32, device=ctx.device)
    ya_t = torch.as_tensor(ctx.ya, dtype=torch.long, device=ctx.device)
    probs_by_name = {}

    def _bank_probs_model(path, k, tau, name):
        if not os.path.exists(path):
            return
        m = load_softknn(path, ctx.device)
        with torch.no_grad():
            bz = embed_chunked(m, xa_t)
        probs_by_name[name] = predict_probs(m, bz, ya_t, ctx.xa, k=k, tau=tau,
                                            device=ctx.device).numpy()

    for seed in (42, 7, 123):
        _bank_probs_model(os.path.join(ARTIFACT_DIR, f"softknn_conv_s{seed}.pt"),
                          9, 0.1, f"conv_s{seed}")
    _bank_probs_model(os.path.join(ARTIFACT_DIR, "softknn_bankaug.pt"),
                      5, 0.1, "bankaug_linear")
    mlp_entry = next((e for e in ctx.ledger
                      if e["id"] == "E16" and e.get("grid_best")), None)
    if mlp_entry:
        _bank_probs_model(os.path.join(ARTIFACT_DIR, "softknn_mlp.pt"),
                          mlp_entry["grid_best"]["k"],
                          mlp_entry["grid_best"]["tau"], "mlp_metric")
    _bank_probs_model(os.path.join(ARTIFACT_DIR, "softknn_conv_distill.pt"),
                      9, 0.1, "conv_distill")

    dim = getattr(ctx, "best_pca_dim", 64)
    za = ctx.pca_view(dim)[1]
    params = getattr(ctx, "best_knn_params", {"k": 5, "metric": "manhattan",
                                              "weights": "distance"})
    clf = classical.make_knn(**params)
    clf.fit(za, ctx.ya)
    probs_by_name["knn_classical"] = clf.predict_proba(za)

    teacher, used = None, []
    for name, w in weights.items():
        if name in probs_by_name and w > 0:
            term = w * probs_by_name[name]
            teacher = term if teacher is None else teacher + term
            used.append((name, w))
    if teacher is None:
        print("[E26] no weighted teacher components available; skipping")
        return {"name": "distill_gen2", "skipped": True}
    print(f"[E26] gen-2 teacher = fusion of {used} on train bank")

    cfg = {"embed_dim": 128, "metric": "conv", "k": 7, "tau": 0.5, "lr": 3e-4,
           "optimizer": "adam", "epochs": 35, "batch_size": 256, "patience": 10,
           "augment": None, "augmented_bank_copies": 0, "bank_refresh_steps": 107,
           "clip_grad": 1.0, "seed": 200, "amp": True,
           "distill_soft_targets": teacher, "distill_alpha": 0.5, "distill_T": 1.0}
    model, history, best_val = train_softknn(ctx.xa, ctx.ya, ctx.xv, ctx.yv,
                                             cfg, device=ctx.device)
    with torch.no_grad():
        bank_z = embed_chunked(model, xa_t)
    grid = []
    for k in (7, 9):
        for tau in (0.1, 0.3):
            probs = predict_probs(model, bank_z, ya_t, ctx.xv, k=k, tau=tau,
                                  device=ctx.device).numpy()
            acc = float((probs.argmax(1) == ctx.yv).mean())
            grid.append({"k": k, "tau": tau, "val_acc": acc})
    best = max(grid, key=lambda r: r["val_acc"])
    test_probs = predict_probs(model, bank_z, ya_t, ctx.xte, k=best["k"],
                               tau=best["tau"], device=ctx.device).numpy()
    test_acc = float((test_probs.argmax(1) == ctx.yte).mean())
    out_path = os.path.join(ARTIFACT_DIR, "softknn_conv_distill_g2.pt")
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"[E26] gen-2 student best grid={best} test={test_acc:.4f} "
          f"saved -> {os.path.basename(out_path)}")
    return {"name": "distill_gen2", "config": {k: v for k, v in cfg.items()
                                               if k != "distill_soft_targets"},
            "teacher_weights": {n: w for n, w in used}, "grid": grid,
            "grid_best": best, "val_acc": best["val_acc"], "test_acc": test_acc,
            "best_train_val": best_val, "history": history}


def e25_tta_fusion(ctx):
    """Champion fusion with query-side TTA smoothing on every Soft-KNN source.

    Each neural-metric source predicts over six deterministic translations
    of the query images (identity, +/-1 px horizontal, +/-1 px vertical)
    and the probabilities are averaged before the simplex weight search.
    The distilled student is included when its artifact exists.
    """
    sv, st, names = _collect_sources(ctx, tta=True)
    path = os.path.join(ARTIFACT_DIR, "softknn_conv_distill.pt")
    if os.path.exists(path):
        m = load_softknn(path, ctx.device)
        xa_t = torch.as_tensor(ctx.xa, dtype=torch.float32, device=ctx.device)
        ya_t = torch.as_tensor(ctx.ya, dtype=torch.long, device=ctx.device)
        with torch.no_grad():
            bz = embed_chunked(m, xa_t)
        v_img = torch.as_tensor(ctx.xv.reshape(len(ctx.xv), 1, 28, 28),
                                dtype=torch.float32)
        t_img = torch.as_tensor(ctx.xte.reshape(len(ctx.xte), 1, 28, 28),
                                dtype=torch.float32)
        sv.append(predict_probs_tta(m, bz, ya_t, v_img, k=9, tau=0.1,
                                    device=ctx.device).numpy())
        st.append(predict_probs_tta(m, bz, ya_t, t_img, k=9, tau=0.1,
                                    device=ctx.device).numpy())
        names.append("conv_distill")
    return _fusion_eval(ctx, sv, st, names, "E25")


def e27_grand_fusion(ctx):
    """Definitive fusion: every artefact source with query-side TTA.

    Sources: all conv-seed models (local + Kaggle-trained), the bank-augmented
    linear metric, the MLP metric, classical k-NN, and every distillation
    generation present (E23 student, five-teacher student, gen-2 student).
    Simplex weights are re-optimised on TTA-smoothed validation probabilities.
    """
    sv, st, names = _collect_sources(ctx, tta=True)
    xa_t = torch.as_tensor(ctx.xa, dtype=torch.float32, device=ctx.device)
    ya_t = torch.as_tensor(ctx.ya, dtype=torch.long, device=ctx.device)
    v_img = torch.as_tensor(ctx.xv.reshape(len(ctx.xv), 1, 28, 28),
                            dtype=torch.float32)
    t_img = torch.as_tensor(ctx.xte.reshape(len(ctx.xte), 1, 28, 28),
                            dtype=torch.float32)
    for art, name in (("softknn_conv_distill.pt", "conv_distill"),
                      ("softknn_conv_distill_g2.pt", "conv_distill_g2")):
        path = os.path.join(ARTIFACT_DIR, art)
        if not os.path.exists(path):
            continue
        m = load_softknn(path, ctx.device)
        with torch.no_grad():
            bz = embed_chunked(m, xa_t)
        sv.append(predict_probs_tta(m, bz, ya_t, v_img, k=9, tau=0.1,
                                    device=ctx.device).numpy())
        st.append(predict_probs_tta(m, bz, ya_t, t_img, k=9, tau=0.1,
                                    device=ctx.device).numpy())
        names.append(name)
    return _fusion_eval(ctx, sv, st, names, "E27")


STAGES = {
    "E0": e0_eda, "E1": e1_baseline, "E2": e2_pca_sweep, "E3": e3_optuna,
    "E4": e4_bias_variance, "E5": e5_latency, "E6": e6_softknn_base,
    "E7": e7_softknn_aug, "E8": e8_tau_sweep, "E9": e9_optimizers,
    "E10": e10_tta, "E11": e11_ensemble, "E12": e12_inference_grid,
    "E13": e13_shift_aug, "E14": e14_bank_aug, "E15": e15_seed_ensemble,
    "E16": e16_mlp_metric, "E17": e17_champion_fusion, "E19": e19_conv_metric,
    "E20": e20_conv_shift, "E21": e21_conv_ensemble, "E22": e22_final_fusion,
    "E23": e23_distill, "E24": e24_fusion_distill, "E25": e25_tta_fusion,
    "E26": e26_distill_gen2, "E27": e27_grand_fusion,
}


def run_stage(stage_id, ctx, ledger, force=False):
    if not force and any(e["id"] == stage_id for e in ledger):
        print(f"[skip] {stage_id} already in ledger")
        return
    t0 = time.perf_counter()
    entry = STAGES[stage_id](ctx)
    entry.update({"id": stage_id, "scale": ctx.scale,
                  "wall_s": round(time.perf_counter() - t0, 1),
                  "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")})
    ledger.append(entry)
    save_ledger(ledger)
    print(f"[done] {stage_id} in {entry['wall_s']}s -> {entry.get('name')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="quick", choices=list(SCALES))
    ap.add_argument("--stages", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    ctx = Ctx(args.scale)
    ledger = load_ledger()
    ctx.ledger = ledger
    stages = list(STAGES) if args.all else [
        s.strip() for s in args.stages.split(",") if s.strip()]
    for sid in stages:
        run_stage(sid, ctx, ledger, force=args.force)
    print("=" * 62)
    print("LEDGER SUMMARY (last 3 entries)")
    for e in ledger[-3:]:
        val = e.get("val_acc", float("nan"))
        test = e.get("test_acc", float("nan"))
        extra = {k: v for k, v in e.items()
                 if k not in ("stage", "val_acc", "test_acc", "history",
                              "epochs", "per_epoch")}
        print(f"  [{e.get('stage', '?')}] val={val*100:.2f}%  test={test*100:.2f}%"
              f"  {extra if extra else ''}")
    print("=" * 62)


if __name__ == "__main__":
    main()
