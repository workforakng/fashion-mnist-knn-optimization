"""Tiny-scale smoke test of the whole harness.

Run after env setup:
    .venv/bin/python scripts/smoke.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from src.augment import (Augmenter, random_erasing, random_hflip, random_rotate,  # noqa: E402
                         shift, tta_views)
from src.classical import eval_knn, fit_pca, make_knn  # noqa: E402
from src.data import load_splits  # noqa: E402
from src.evaluate import full_metrics  # noqa: E402
from src.softknn import (SoftKNN, mask_self_matches, per_class_topk_logits,  # noqa: E402
                         predict_probs, predict_probs_tta)
from src.train import train_softknn  # noqa: E402


def main():
    print("== data ==")
    xa, ya, xv, yv, xte, yte = load_splits(val_size=1000, seed=42)
    xa, ya = xa[:3000], ya[:3000]
    xv, yv = xv[:800], yv[:800]
    xte, yte = xte[:800], yte[:800]
    print("shapes", xa.shape, xv.shape, xte.shape, "range", xa.min(), xa.max())

    print("== augment ==")
    x = torch.as_tensor(xa[:16].reshape(16, 1, 28, 28))
    for fn, arg in [(random_hflip, {}), (random_erasing, {}),
                    (random_rotate, {"max_deg": 10.0})]:
        out = fn(x, **arg)
        assert out.shape == x.shape and out.min() >= 0 and out.max() <= 1.0001, fn
    test = torch.zeros(1, 1, 4, 4)
    test[0, 0, 1, 1] = 1.0
    moved = shift(test, 1, 0)[0, 0]
    assert moved[1, 2] == 1.0 and moved.sum() == 1.0, "dx=1 must move pixel right"
    moved = shift(test, 0, 1)[0, 0]
    assert moved[2, 1] == 1.0, "dy=1 must move pixel down"
    out = Augmenter()(x)
    assert out.shape == x.shape
    views = tta_views(x)
    assert len(views) == 6 and all(v.shape == x.shape for v in views)
    print("augment OK")

    print("== classical ==")
    pca = fit_pca(xa, 32)
    za, zv = pca.transform(xa), pca.transform(xv)
    clf = make_knn(k=5)
    acc, fit_s, q_s, _ = eval_knn(clf, za, ya, zv, yv)
    print(f"knn pca32 val_acc={acc:.3f} fit={fit_s:.2f}s query={q_s:.2f}s")

    print("== softknn logits correctness ==")
    dist = torch.tensor([[0.0, 1.0, 2.0, 3.0], [5.0, 4.0, 3.0, 2.0]])
    bank_y = torch.tensor([0, 0, 1, 1])
    logits = per_class_topk_logits(dist, bank_y, k=1, n_classes=2, tau=0.5)
    expected = torch.tensor([[0.0, -4.0], [-8.0, -4.0]])
    assert torch.allclose(logits, expected), logits
    print("per_class_topk_logits OK")

    print("== exclude_self (leave-one-out) on random data ==")
    g = torch.Generator().manual_seed(0)
    xr = torch.rand(300, 784, generator=g)
    yr = torch.arange(300) % 10
    # mask helper: the query's own column becomes +inf, nothing else changes
    d = torch.rand(4, 6, generator=g)
    dm = mask_self_matches(d, torch.tensor([0, 1, 2, 3]))
    assert torch.isinf(dm[torch.arange(4), torch.arange(4)]).all()
    assert torch.isfinite(dm).sum() == 4 * 6 - 4
    m_r = SoftKNN(embed_dim=16).eval()
    with torch.no_grad():
        bz = m_r.embed(xr)
        # k=1: with self kept, every bank point's nearest same-class anchor is
        # itself (distance 0) -> train accuracy is trivially 100 %.
        p_self = predict_probs(m_r, bz, yr, xr, k=1, tau=0.5, device="cpu")
        p_loo = predict_probs(m_r, bz, yr, xr, k=1, tau=0.5, device="cpu",
                              exclude_self=True)
    acc_self = (p_self.argmax(1) == yr).float().mean().item()
    acc_loo = (p_loo.argmax(1) == yr).float().mean().item()
    assert acc_self == 1.0, acc_self
    assert acc_loo < 0.5, acc_loo  # random labels: LOO is ~chance
    assert torch.allclose(p_loo.sum(1), torch.ones(300))
    # training path, including augmented bank copies (bank = 2 x 300 anchors)
    cfg_r = dict(embed_dim=16, k=3, tau=0.5, lr=1e-3, optimizer="adam", epochs=1,
                 batch_size=64, patience=1,
                 augment={"hflip_p": 0.5, "shift_px": 1, "erase": False},
                 augmented_bank_copies=1, seed=0, exclude_self=True)
    _m, hist_r, _bv = train_softknn(xr.numpy(), yr.numpy(), xr[:50].numpy(),
                                    yr[:50].numpy(), cfg_r, device="cpu",
                                    verbose=False)
    assert len(hist_r) == 1 and hist_r[0]["train_loss"] == hist_r[0]["train_loss"]
    print(f"exclude_self OK (k=1 random-label train acc: self={acc_self:.2f} "
          f"loo={acc_loo:.2f})")

    print("== train softknn ==")
    cfg = dict(embed_dim=32, k=5, tau=0.5, lr=1e-3, optimizer="adam", epochs=2,
               batch_size=128, patience=2,
               augment={"hflip_p": 0.5, "shift_px": 2, "erase": True},
               augmented_bank_copies=1,
               pca_components=pca.components_.copy(), seed=42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    model, history, best_val = train_softknn(xa, ya, xv, yv, cfg, device=device)
    assert len(history) >= 1 and best_val > 0.3, history
    print(f"softknn best_val={best_val:.3f} epochs_run={len(history)}")

    print("== predict + TTA ==")
    with torch.no_grad():
        bank_z = model.embed(torch.as_tensor(xa, dtype=torch.float32, device=device))
        bank_y = torch.as_tensor(ya, dtype=torch.long, device=device)
    probs = predict_probs(model, bank_z, bank_y, xte, k=5, tau=0.5,
                          device=device).numpy()
    tta = predict_probs_tta(model, bank_z, bank_y,
                            torch.as_tensor(xte.reshape(len(xte), 1, 28, 28)),
                            k=5, tau=0.5, device=device).numpy()
    m = full_metrics(yte, probs.argmax(1))
    print(f"probs {probs.shape} acc={m['accuracy']:.3f} f1={m['f1_macro']:.3f} "
          f"tta_acc={(tta.argmax(1) == yte).mean():.3f}")
    assert probs.shape == (len(xte), 10)

    print("SMOKE_OK")


if __name__ == "__main__":
    main()
