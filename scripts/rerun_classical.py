"""Logged, reproducible re-run of the classical-KNN numbers quoted in docs/paper.tex.

Produces results/classical_reruns.json with:
  1. raw 784-D k=5 uniform KNN for euclidean / manhattan / cosine
     (val acc, test acc, macro P/R/F1 on test, query latency);
  2. a PCA sweep (PCA fit on the 54k fit split only) with k=5 euclidean uniform,
     val/test acc, cumulative explained variance per d and the d reaching 90 %;
  3. the Optuna winner logged in results/results.json (E3) re-evaluated;
  4. a fixed latency protocol (recorded in the JSON) and the raw-784 -> PCA-64
     speed-up ratio.

Protocol is identical to src/experiments.py at scale "full": src.data.load_splits
(54k fit / 6k stratified val, seed 42, official 10k test, pixels in [0,1]),
src.classical.make_knn / fit_pca (PCA svd_solver="full").

Usage (from repo root):
    python scripts/rerun_classical.py --data-dir /path/to/idx_gz_dir
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import statistics
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402
import sklearn  # noqa: E402
from sklearn.metrics import (accuracy_score, confusion_matrix,  # noqa: E402
                             precision_recall_fscore_support)

from src import classical, data  # noqa: E402

SEED = 42
K = 5
PCA_DIMS = [10, 16, 20, 32, 48, 64, 80, 96, 128]  # 16 added to match paper row E04
N_REPEATS = 3


def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def thread_info() -> list:
    try:
        from threadpoolctl import threadpool_info
        return [{k: d.get(k) for k in ("user_api", "internal_api", "num_threads", "version")}
                for d in threadpool_info()]
    except Exception as exc:  # pragma: no cover
        return [{"error": str(exc)}]


def metrics(y_true, y_pred) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro",
                                                  zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(p), "recall_macro": float(r), "f1_macro": float(f1),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()}


def timed_predict(clf, xq, repeats: int = N_REPEATS):
    """One untimed-for-reporting warm-up predict, then `repeats` timed predicts.
    Returns (predictions, median_s, all_timings_s, warmup_s)."""
    t0 = time.perf_counter()
    pred = clf.predict(xq)
    warm = time.perf_counter() - t0
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        p2 = clf.predict(xq)
        ts.append(time.perf_counter() - t0)
        assert np.array_equal(p2, pred), "non-deterministic predictions"
    return pred, float(statistics.median(ts)), [float(t) for t in ts], float(warm)


def evaluate_config(name, xa, ya, xv, yv, xte, yte, time_test: bool, **knn):
    clf = classical.make_knn(k=knn.get("k", K), metric=knn.get("metric", "euclidean"),
                             weights=knn.get("weights", "uniform"), algorithm="brute",
                             n_jobs=N_JOBS)
    clf.fit(xa, ya)
    val_acc = float(accuracy_score(yv, clf.predict(xv)))
    if time_test:
        pred, med, ts, warm = timed_predict(clf, xte)
        lat = {"query_s_median": med, "query_s_repeats": ts, "warmup_s": warm,
               "per_query_ms": med / len(xte) * 1e3}
    else:
        pred, lat = clf.predict(xte), None
    m = metrics(yte, pred)
    row = {"name": name, "config": {"k": knn.get("k", K),
                                    "metric": knn.get("metric", "euclidean"),
                                    "weights": knn.get("weights", "uniform"),
                                    "algorithm": "brute", "n_jobs": N_JOBS},
           "val_acc": val_acc, "test_acc": m["accuracy"],
           "test_precision_macro": m["precision_macro"],
           "test_recall_macro": m["recall_macro"], "test_f1_macro": m["f1_macro"],
           "test_confusion_matrix": m["confusion_matrix"], "latency": lat}
    print(f"[{name}] val={val_acc:.4f} test={m['accuracy']:.4f} "
          f"P={m['precision_macro']:.4f} R={m['recall_macro']:.4f} F1={m['f1_macro']:.4f}"
          + (f" lat={lat['query_s_median']:.3f}s" if lat else ""), flush=True)
    return row


def main():
    global N_JOBS
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", default=data.DATA_DIR,
                    help="directory holding the four Fashion-MNIST *.gz IDX files")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results",
                                                  "classical_reruns.json"))
    ap.add_argument("--ledger", default=os.path.join(REPO_ROOT, "results", "results.json"))
    ap.add_argument("--n-jobs", type=int, default=4)
    args = ap.parse_args()
    N_JOBS = args.n_jobs
    np.random.seed(SEED)

    data.DATA_DIR = os.path.abspath(args.data_dir)  # load_raw reads this module global
    started = dt.datetime.now(dt.timezone.utc)
    xa, ya, xv, yv, xte, yte = data.load_splits(val_size=6000, seed=SEED)
    print(f"[data] fit={len(xa)} val={len(xv)} test={len(xte)} "
          f"range=[{xa.min():.1f},{xa.max():.1f}]", flush=True)

    out = {
        "description": "Logged re-run of classical KNN numbers (scripts/rerun_classical.py)",
        "started_utc": started.isoformat(timespec="seconds"),
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__,
            "sklearn": sklearn.__version__, "platform": platform.platform(),
            "cpu_model": cpu_model(), "os_cpu_count": os.cpu_count(),
            "threadpools": thread_info(),
        },
        "data": {"loader": "src.data.load_splits(val_size=6000, seed=42)",
                 "n_fit": int(len(xa)), "n_val": int(len(xv)), "n_test": int(len(xte)),
                 "pixel_scale": "[0,1] float32", "seed": SEED},
        "latency_protocol": {
            "query_set": "official test set", "n_query_points": int(len(xte)),
            "n_reference_points": int(len(xa)), "algorithm": "brute",
            "n_jobs": N_JOBS, "k": K,
            "timed_call": "KNeighborsClassifier.predict(X_test) after fit (fit excluded)",
            "warmup": "1 untimed predict before timing",
            "repeats": N_REPEATS, "statistic": "median of repeats",
            "timer": "time.perf_counter", "pca_transform_time": "excluded (features "
            "pre-projected)", "cpu_model": cpu_model(),
            "threads": thread_info(),
            "loadavg_1_5_15_at_start": list(os.getloadavg()),
            "note": "latency is wall-clock and sensitive to other processes on the "
                    "host; compare loadavg at start/end against os_cpu_count",
        },
    }

    # 1. raw 784-D metrics ------------------------------------------------------
    out["raw784_metrics"] = [
        evaluate_config(f"raw784_{m}_k5_uniform", xa, ya, xv, yv, xte, yte,
                        time_test=True, metric=m)
        for m in ("euclidean", "manhattan", "cosine")]

    # 2. PCA sweep --------------------------------------------------------------
    pca_full = classical.fit_pca(xa, 784)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    var_rows = {}
    for thr in (0.80, 0.85, 0.86, 0.90, 0.95, 0.99):
        var_rows[f"{thr:.2f}"] = int(np.searchsorted(cum, thr) + 1)
    sweep = []
    pca_feats = {}
    for d in PCA_DIMS:
        pca = classical.fit_pca(xa, d)
        za, zv, zt = pca.transform(xa), pca.transform(xv), pca.transform(xte)
        pca_feats[d] = (za, zv, zt)
        row = evaluate_config(f"pca{d}_euclidean_k5_uniform", za, ya, zv, yv, zt, yte,
                              time_test=True, metric="euclidean")
        row["dim"] = d
        row["cum_explained_variance"] = float(pca.explained_variance_ratio_.sum())
        sweep.append(row)
    out["pca_sweep"] = {
        "pca": "sklearn PCA(svd_solver='full') fit on the 54k fit split only",
        "rows": sweep,
        "cum_explained_variance_by_dim": {str(d): float(cum[d - 1])
                                          for d in sorted(set(PCA_DIMS + [84]))},
        "dims_for_variance_threshold": var_rows,
        "dim_reaching_90pct": var_rows["0.90"],
        "best_val_dim": max(sweep, key=lambda r: r["val_acc"])["dim"],
    }
    print(f"[pca] cum var @64={cum[63]:.4f}  d(90%)={var_rows['0.90']}", flush=True)

    # 3. Optuna winner (E3 in ledger) ------------------------------------------
    with open(args.ledger) as f:
        ledger = json.load(f)
    e3 = next(e for e in ledger if e.get("id") == "E3")
    bp, d3 = dict(e3["best_params"]), int(e3["pca_dim"])
    if d3 not in pca_feats:
        pca = classical.fit_pca(xa, d3)
        pca_feats[d3] = (pca.transform(xa), pca.transform(xv), pca.transform(xte))
    za, zv, zt = pca_feats[d3]
    row = evaluate_config(f"optuna_E3_pca{d3}", za, ya, zv, yv, zt, yte, time_test=True,
                          k=bp["k"], metric=bp["metric"], weights=bp["weights"])
    row["dim"] = d3
    row["ledger_E3"] = {"best_params": bp, "pca_dim": d3, "val_acc": e3["val_acc"],
                        "test_acc": e3["test_acc"], "test_f1_macro": e3["test_f1_macro"],
                        "timestamp": e3.get("timestamp")}
    row["matches_ledger_test"] = abs(row["test_acc"] - e3["test_acc"]) < 1e-9
    row["test_acc_equals_0.8643"] = round(row["test_acc"], 4) == 0.8643
    out["optuna_winner"] = row

    # 4. speed-up ---------------------------------------------------------------
    raw = out["raw784_metrics"][0]["latency"]["query_s_median"]
    p64 = next(r for r in sweep if r["dim"] == 64)["latency"]["query_s_median"]
    out["speedup"] = {
        "raw784_euclidean_query_s": raw, "pca64_euclidean_query_s": p64,
        "ratio_raw_over_pca64_euclidean": raw / p64,
        "optuna_winner_query_s": row["latency"]["query_s_median"],
        "ratio_raw_over_optuna_winner": raw / row["latency"]["query_s_median"],
    }
    print(f"[speedup] raw/pca64 = {raw / p64:.2f}x", flush=True)

    out["latency_protocol"]["loadavg_1_5_15_at_end"] = list(os.getloadavg())
    out["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] wrote {args.out}")


N_JOBS = 4

if __name__ == "__main__":
    main()
