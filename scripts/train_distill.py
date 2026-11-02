"""Self-distillation: conv-seed ensemble teacher -> fresh conv student.

Teacher = mean probability of the trained conv seed artifacts on the train set
(soft labels via tau=1.0). Student = fresh conv-metric Soft-KNN trained with
loss = (1-alpha)*CE + alpha*KL(teacher || student)   (Hinton direction; this is
what F.kl_div(log_softmax(student), teacher) computes).

Self-match caveat: the teacher is queried on the very images that form its
memory bank, so by default each image matches itself at distance ~0 and the
soft targets are biased toward the hard labels. --exclude-self masks each
image's own bank entry (leave-one-out) for the teacher targets and for the
student's training queries. Off by default to reproduce the logged runs
(results/kaggle_distill*.json).
"""
import argparse
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_splits
from src.softknn import embed_chunked, load_softknn, predict_probs
from src.train import train_softknn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(REPO_ROOT, "artifacts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--exclude-self", action="store_true",
                    help="leave-one-out teacher targets and training queries")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    xa, ya, xv, yv, xte, yte = load_splits(val_size=6000, seed=42)
    xa_t = torch.as_tensor(xa, dtype=torch.float32, device=device)
    ya_t = torch.as_tensor(ya, dtype=torch.long, device=device)

    teacher, n_te = None, 0
    seed_paths = sorted(glob.glob(os.path.join(ARTIFACT_DIR, "softknn_conv_s[0-9]*.pt")))
    for path in seed_paths:
        m = load_softknn(path, device)
        with torch.no_grad():
            bz = embed_chunked(m, xa_t)
        p = predict_probs(m, bz, ya_t, xa, k=9, tau=1.0, device=device,
                          exclude_self=args.exclude_self).numpy()
        teacher = p if teacher is None else teacher + p
        n_te += 1
    if not n_te:
        raise SystemExit("no teacher artifacts found")
    print(f"distill teachers: {n_te} from {[os.path.basename(p) for p in seed_paths]}")
    teacher = teacher / n_te

    cfg = {"embed_dim": 128, "metric": "conv", "k": 7, "tau": 0.5, "lr": 3e-4,
           "optimizer": "adam", "epochs": args.epochs, "batch_size": 256,
           "patience": 10, "augment": None, "augmented_bank_copies": 0,
           "bank_refresh_steps": 107, "clip_grad": 1.0, "seed": 100, "amp": True,
           "distill_soft_targets": teacher, "distill_alpha": args.alpha,
           "distill_T": 1.0, "exclude_self": args.exclude_self}
    model, history, best_val = train_softknn(xa, ya, xv, yv, cfg, device=device)
    with torch.no_grad():
        bank_z = embed_chunked(model, xa_t)
    best = None
    for k in (7, 9):
        for tau in (0.1, 0.3):
            pv = predict_probs(model, bank_z, ya_t, xv, k=k, tau=tau,
                               device=device).numpy()
            acc = float((pv.argmax(1) == yv).mean())
            if best is None or acc > best[2]:
                best = (k, tau, acc)
    k, tau, val_acc = best
    pt = predict_probs(model, bank_z, ya_t, xte, k=k, tau=tau,
                       device=device).numpy()
    test_acc = float((pt.argmax(1) == yte).mean())
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    torch.save(model.state_dict(),
               os.path.join(ARTIFACT_DIR, "softknn_conv_distill.pt"))
    out = {"val_acc": val_acc, "test_acc": test_acc, "k": k, "tau": tau,
           "teacher_models": n_te, "epochs_run": len(history),
           "best_train_val": best_val}
    os.makedirs(os.path.join(REPO_ROOT, "results"), exist_ok=True)
    with open(os.path.join(REPO_ROOT, "results", "distill.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("[distill]", json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
