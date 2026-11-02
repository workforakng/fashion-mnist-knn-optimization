import sys, time
sys.path.insert(0, sys.argv[1])
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from src.data import load_splits
from src.classical import fit_pca
xa, ya, xv, yv, xte, yte = load_splits(val_size=6000, seed=42)
for m in ("euclidean", "cosine", "manhattan"):
    t=time.perf_counter(); clf = KNeighborsClassifier(5, metric=m, n_jobs=8).fit(xa, ya); p = clf.predict(xte)
    print(m, "test=%.4f f1=%.4f t=%.1fs" % ((p == yte).mean(), f1_score(yte, p, average="macro"), time.perf_counter()-t), flush=True)
pca = fit_pca(xa, 64); import numpy as np
print("PCA64 cum var", pca.explained_variance_ratio_.sum())
za, zv, zt = pca.transform(xa), pca.transform(xv), pca.transform(xte)
for d in (16, 32, 64, 128):
    pc = fit_pca(xa, d); a, v, t2 = pc.transform(xa), pc.transform(xv), pc.transform(xte)
    c = KNeighborsClassifier(5, n_jobs=8).fit(a, ya)
    print("pca", d, "val=%.4f test=%.4f" % ((c.predict(v)==yv).mean(), (c.predict(t2)==yte).mean()), flush=True)
for k,w,m in ((7,"distance","euclidean"),(5,"distance","manhattan")):
    c = KNeighborsClassifier(k, weights=w, metric=m, n_jobs=8).fit(za, ya)
    print("pca64", k, w, m, "val=%.4f test=%.4f" % ((c.predict(zv)==yv).mean(), (c.predict(zt)==yte).mean()), flush=True)
