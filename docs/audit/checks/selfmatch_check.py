"""Audit check: (1) self-match during Soft-KNN training; (2) teacher self-match;
(3) distance-weighted sklearn KNN predict_proba on its own fit set."""
import sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, sys.argv[1])
from sklearn.model_selection import train_test_split
from src.data import load_splits
from src.classical import fit_pca, make_knn
from src.softknn import SoftKNN, embed_chunked, predict_probs, per_class_topk_logits
from src.train import train_softknn

xa, ya, xv, yv, xte, yte = load_splits(val_size=6000, seed=42)
xa, _, ya, _ = train_test_split(xa, ya, train_size=12000, stratify=ya, random_state=42)
pca = fit_pca(xa, 64)
cfg = dict(embed_dim=64, k=7, tau=0.5, lr=1e-3, optimizer="adam", epochs=4, batch_size=256,
           patience=4, augment=None, augmented_bank_copies=0,
           pca_components=pca.components_.copy(), seed=42)
model, hist, bv = train_softknn(xa, ya, xv, yv, cfg, device="cpu", verbose=False)
xa_t = torch.as_tensor(xa); ya_t = torch.as_tensor(ya, dtype=torch.long)
bz = embed_chunked(model, xa_t)
# (1) how often is the query itself among its own-class top-k, and what is its distance
with torch.no_grad():
    zq = model.embed(xa_t[:2000])
    d = torch.cdist(zq, bz)
    self_d = d[torch.arange(2000), torch.arange(2000)]
    print("self distance (fresh bank) max:", float(self_d.max()))
    # second-nearest same-class distance for comparison
    same = d.clone(); same[:, ya_t != ya_t[:2000, None].T[0][0]] = 0  # placeholder
# (2) teacher probs on train (bank includes self) vs on val (no self)
for tau in (1.0, 0.1):
    pt = predict_probs(model, bz, ya_t, xa, k=9, tau=tau, device="cpu").numpy()
    pv = predict_probs(model, bz, ya_t, xv, k=9, tau=tau, device="cpu").numpy()
    # leave-one-out teacher on train
    with torch.no_grad():
        zq = embed_chunked(model, xa_t)
        outs = []
        for i in range(0, len(zq), 1024):
            dd = torch.cdist(zq[i:i+1024], bz)
            idx = torch.arange(i, min(i+1024, len(zq)))
            dd[torch.arange(len(idx)), idx] = float("inf")
            outs.append(F.softmax(per_class_topk_logits(dd, ya_t, 9, 10, tau), 1))
        ploo = torch.cat(outs).numpy()
    def s(p, y): return f"acc={(p.argmax(1)==y).mean():.4f} p_true={p[np.arange(len(y)), y].mean():.4f}"
    print(f"tau={tau} teacher on TRAIN (self in bank): {s(pt, ya)} | train leave-one-out: {s(ploo, ya)} | VAL: {s(pv, yv)}")
# (3) distance-weighted sklearn KNN on own fit set
za = pca.transform(xa)
clf = make_knn(k=5, metric="manhattan", weights="distance", n_jobs=1).fit(za, ya)
pp = clf.predict_proba(za)
print("sklearn distance-weighted predict_proba on fit set: acc=%.4f, frac one-hot=%.4f" % (
    (pp.argmax(1) == ya).mean(), (pp.max(1) == 1.0).mean()))
