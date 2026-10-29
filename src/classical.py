"""Classical KNN experiments: baseline, PCA sweep, latency benchmark, Optuna HPO, bias-variance."""
from __future__ import annotations

import os
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


def make_knn(k: int = 5, metric: str = "euclidean", weights: str = "uniform",
             p: float = 2.0, algorithm: str = "auto",
             n_jobs: int = None) -> KNeighborsClassifier:
    if n_jobs is None:
        n_jobs = int(os.environ.get("KNN_NJOBS", "4"))  # thermal throttle knob
    params = dict(n_neighbors=k, weights=weights, metric=metric,
                  algorithm=algorithm, n_jobs=n_jobs)
    if metric == "minkowski":
        params["p"] = p
    return KNeighborsClassifier(**params)


def eval_knn(clf, xa, ya, xv, yv):
    """Fit + predict with timing. Returns (accuracy, fit_s, query_s, predictions)."""
    t0 = time.perf_counter()
    clf.fit(xa, ya)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = clf.predict(xv)
    query_s = time.perf_counter() - t0
    return accuracy_score(yv, pred), fit_s, query_s, pred


def fit_pca(xa, n_components):
    """n_components: int (exact dims) or float in (0,1) for variance fraction."""
    pca = PCA(n_components=n_components, svd_solver="full")
    pca.fit(xa)
    return pca


def pca_sweep(xa, ya, xv, yv, dims, k: int = 5):
    """Validation accuracy vs number of PCA components."""
    rows = []
    for d in dims:
        pca = fit_pca(xa, d)
        za, zv = pca.transform(xa), pca.transform(xv)
        acc, fit_s, q_s, _ = eval_knn(make_knn(k=k), za, ya, zv, yv)
        rows.append({"dim": int(pca.n_components_), "val_acc": float(acc),
                     "fit_s": fit_s, "query_s": q_s})
        print(f"[pca_sweep] dim={pca.n_components_:3d} val_acc={acc:.4f}")
    return rows


def latency_benchmark(xa, ya, xq, yq, variants, algos=("brute", "kd_tree", "ball_tree"),
                      repeats: int = 3, k: int = 5):
    """Median fit/query latency per (feature variant x algorithm).

    variants: list of {"name": str, "xa": array, "xq": array}.
    """
    rows = []
    for var in variants:
        for algo in algos:
            fit_ts, q_ts, accs = [], [], []
            error = None
            # Tree backends degenerate in high dimensions; 1 repeat is enough
            # to demonstrate that (brute keeps `repeats` for a stable median).
            reps = repeats if algo == "brute" else 1
            for _ in range(reps):
                clf = make_knn(k=k, algorithm=algo)
                try:
                    acc, fit_s, q_s, _ = eval_knn(clf, var["xa"], ya, var["xq"], yq)
                except Exception as exc:  # tree backends unsupported for some metric/dims
                    error = str(exc)[:100]
                    break
                fit_ts.append(fit_s)
                q_ts.append(q_s)
                accs.append(acc)
            if error is not None:
                rows.append({"variant": var["name"], "algorithm": algo, "error": error})
                print(f"[latency] {var['name']:>8s} {algo:>10s} ERROR {error}")
                continue
            row = {"variant": var["name"], "algorithm": algo,
                   "fit_s_median": float(np.median(fit_ts)),
                   "query_s_median": float(np.median(q_ts)),
                   "val_acc": float(np.mean(accs))}
            rows.append(row)
            print(f"[latency] {var['name']:>8s} {algo:>10s} "
                  f"fit={row['fit_s_median']:.2f}s query={row['query_s_median']:.3f}s "
                  f"acc={row['val_acc']:.4f}")
    return rows


def optuna_search(xa, ya, xv, yv, n_trials: int = 40, seed: int = 42, pca_dim=None):
    """TPE search over (k, weights, metric, minkowski p) on PCA features."""
    import optuna
    from optuna.samplers import TPESampler

    if pca_dim is not None:
        pca = fit_pca(xa, pca_dim)
        xa_s, xv_s = pca.transform(xa), pca.transform(xv)
    else:
        xa_s, xv_s = xa, xv

    def objective(trial):
        k = trial.suggest_int("k", 3, 15, step=2)
        weights = trial.suggest_categorical("weights", ["uniform", "distance"])
        metric = trial.suggest_categorical(
            "metric", ["euclidean", "manhattan", "chebyshev", "minkowski"])
        p = trial.suggest_float("p", 1.0, 3.0) if metric == "minkowski" else 2.0
        clf = make_knn(k=k, metric=metric, weights=weights, p=p)
        acc, _, _, _ = eval_knn(clf, xa_s, ya, xv_s, yv)
        return acc

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))

    def _log_trial(study, trial):
        val = f"{trial.value:.4f}" if trial.value is not None else "failed"
        print(f"[optuna] trial {trial.number:02d}/{n_trials}: val_acc={val}")

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False,
                   callbacks=[_log_trial])
    return study, dict(study.best_params)


def bias_variance_curve(xa, ya, xv, yv, ks, train_query_size: int = 10000,
                        seed: int = 42, **knn_kwargs):
    """Train vs validation accuracy across k. Train accuracy is measured on a
    fixed stratified-free random subsample of the fit set (k=1 still scores
    exactly 1.0 there because every query point is its own nearest neighbor)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(xa), size=min(train_query_size, len(xa)), replace=False)
    xa_q, ya_q = xa[idx], ya[idx]
    train_accs, val_accs = [], []
    for k in ks:
        clf = make_knn(k=k, **knn_kwargs)
        clf.fit(xa, ya)
        tr = accuracy_score(ya_q, clf.predict(xa_q))
        va = accuracy_score(yv, clf.predict(xv))
        train_accs.append(float(tr))
        val_accs.append(float(va))
        print(f"[bias_var] k={k:2d} train={tr:.4f} val={va:.4f}")
    return list(ks), train_accs, val_accs
