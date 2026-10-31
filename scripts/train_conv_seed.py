"""Standalone conv-metric Soft-KNN trainer for one seed (parallel runs).

Trains one conv-metric Soft-KNN, evaluates at the E19 grid optimum (k=9, tau=0.1),
saves the artifact, and appends metrics to results/conv_seeds.json.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_splits
from src.softknn import embed_chunked, predict_probs
from src.train import train_softknn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(REPO_ROOT, "artifacts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--exclude-self", action="store_true",
                    help="leave-one-out: mask each training query's own bank "
                         "entry (off by default, as in the logged runs)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    xa, ya, xv, yv, xte, yte = load_splits(val_size=6000, seed=42)
    cfg = {"embed_dim": 128, "metric": "conv", "k": 7, "tau": 0.5, "lr": 3e-4,
           "optimizer": "adam", "epochs": args.epochs, "batch_size": 256,
           "patience": 10, "augment": None, "augmented_bank_copies": 0,
           "bank_refresh_steps": 107, "clip_grad": 1.0, "seed": args.seed,
           "amp": True, "exclude_self": args.exclude_self}
    model, history, best_val = train_softknn(xa, ya, xv, yv, cfg, device=device)
    with torch.no_grad():
        bank_z = embed_chunked(model, torch.as_tensor(xa, dtype=torch.float32,
                                                      device=device))
        bank_y = torch.as_tensor(ya, dtype=torch.long, device=device)
    pv = predict_probs(model, bank_z, bank_y, xv, k=9, tau=0.1,
                       device=device).numpy()
    pt = predict_probs(model, bank_z, bank_y, xte, k=9, tau=0.1,
                       device=device).numpy()
    val_acc = float((pv.argmax(1) == yv).mean())
    test_acc = float((pt.argmax(1) == yte).mean())
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    torch.save(model.state_dict(),
               os.path.join(ARTIFACT_DIR, f"softknn_conv_s{args.seed}.pt"))
    sidecar = os.path.join(REPO_ROOT, "results", "conv_seeds.json")
    os.makedirs(os.path.dirname(sidecar), exist_ok=True)
    data = {}
    if os.path.exists(sidecar):
        with open(sidecar) as f:
            data = json.load(f)
    data[str(args.seed)] = {"val_acc": val_acc, "test_acc": test_acc,
                            "best_train_val": best_val,
                            "epochs_run": len(history)}
    with open(sidecar, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[conv-seed {args.seed}] val={val_acc:.4f} test={test_acc:.4f} "
          f"epochs={len(history)}", flush=True)


if __name__ == "__main__":
    main()
