"""Training loop for Soft-KNN: on-the-fly augmentation, memory-bank refresh,
early stopping, history logging."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn.functional as F

from .augment import Augmenter
from .softknn import SoftKNN, embed_chunked


def make_optimizer(name: str, params, lr: float):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr)
    if name in ("sgd", "sgd_momentum"):
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    raise ValueError(f"unknown optimizer: {name}")


@torch.no_grad()
def predict_probs_chunked(model, x_img, bank_z, bank_y, k, tau, device, batch_size=1024):
    model.eval()
    probs = []
    for i in range(0, len(x_img), batch_size):
        xb = x_img[i:i + batch_size].to(device)
        logits = model.logits_from_z(model.embed(xb.flatten(1)), bank_z, bank_y, k, tau)
        probs.append(F.softmax(logits, dim=1).cpu())
    return torch.cat(probs)


def train_softknn(x_fit, y_fit, x_val, y_val, cfg: dict, device=None, verbose=True):
    """Train a SoftKNN model.

    cfg keys: embed_dim, k, tau, lr, optimizer, epochs, batch_size, patience,
              augment (dict for Augmenter, or None), augmented_bank_copies (int),
              pca_components (ndarray or None), seed.

    Returns (model, history, best_val_acc).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    n = len(x_fit)
    x_img = torch.as_tensor(x_fit.reshape(n, 1, 28, 28), dtype=torch.float32).to(device)
    y_t = torch.as_tensor(y_fit, dtype=torch.long).to(device)
    x_val_img = torch.as_tensor(
        x_val.reshape(len(x_val), 1, 28, 28), dtype=torch.float32)

    model = SoftKNN(metric=cfg.get("metric", "linear"),
                    pca_components=cfg.get("pca_components"),
                    embed_dim=cfg["embed_dim"],
                    hidden_dim=cfg.get("hidden_dim", 0)).to(device)
    opt = make_optimizer(cfg["optimizer"], model.parameters(), cfg["lr"])
    augmenter = Augmenter(**cfg["augment"]) if cfg.get("augment") else None

    # Memory bank: originals (+ optional fixed augmented copies, novelty N4).
    bank_x, bank_y = x_img, y_t
    n_copies = int(cfg.get("augmented_bank_copies", 0))
    if n_copies > 0 and augmenter is not None:
        torch.manual_seed(cfg["seed"] + 1)
        extra = [augmenter(x_img) for _ in range(n_copies)]
        bank_x = torch.cat([bank_x] + extra, dim=0)
        bank_y = torch.cat([bank_y] + [y_t] * n_copies, dim=0)
        if verbose:
            print(f"[softknn] augmented memory bank: {bank_x.shape[0]} anchors")

    history = []
    best_val, best_state, best_epoch = -1.0, None, -1
    bsz = cfg["batch_size"]
    steps_per_epoch = (n + bsz - 1) // bsz
    bank_refresh = int(cfg.get("bank_refresh_steps", 0)) or steps_per_epoch
    soft_targets = cfg.get("distill_soft_targets")
    if soft_targets is not None:
        soft_targets = torch.as_tensor(soft_targets, dtype=torch.float32,
                                       device=device)
    use_amp = bool(cfg.get("amp", False)) and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(cfg["epochs"]):
        model.train()
        t0 = time.perf_counter()
        bank_z = embed_chunked(model, bank_x)

        perm = torch.randperm(n, device=device)
        run_loss, run_correct, run_count = 0.0, 0, 0
        for step, i in enumerate(range(0, n, bsz)):
            if step and step % bank_refresh == 0:
                bank_z = embed_chunked(model, bank_x)
            idx = perm[i:i + bsz]
            xb, yb = x_img[idx], y_t[idx]
            if augmenter is not None and cfg.get("augment_queries", True):
                xb = augmenter(xb)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                zq = model.embed(xb.flatten(1))
            zq = zq.float()
            logits = model.logits_from_z(zq, bank_z, bank_y, cfg["k"], cfg["tau"])
            loss = F.cross_entropy(logits, yb)
            alpha = float(cfg.get("distill_alpha", 0.0))
            if alpha > 0 and soft_targets is not None:
                T = float(cfg.get("distill_T", 1.0))
                s_log = F.log_softmax(logits / T, dim=1)
                loss = (1.0 - alpha) * loss + alpha * (T ** 2) * \
                    F.kl_div(s_log, soft_targets[idx], reduction="batchmean")
            opt.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                if cfg.get("clip_grad"):
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   cfg["clip_grad"])
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if cfg.get("clip_grad"):
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   cfg["clip_grad"])
                opt.step()
            run_loss += loss.item() * len(idx)
            run_correct += (logits.argmax(1) == yb).sum().item()
            run_count += len(idx)

        train_loss = run_loss / run_count
        train_acc = run_correct / run_count
        val_probs = predict_probs_chunked(model, x_val_img, bank_z, bank_y,
                                          cfg["k"], cfg["tau"], device)
        val_acc = float((val_probs.argmax(1).numpy() == y_val).mean())
        dt = time.perf_counter() - t0
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "train_acc": train_acc, "val_acc": val_acc, "epoch_s": dt})
        if verbose:
            print(f"[softknn] epoch {epoch:02d} loss={train_loss:.4f} "
                  f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} ({dt:.1f}s)")

        if val_acc > best_val:
            best_val, best_epoch = val_acc, epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= cfg["patience"]:
            if verbose:
                print(f"[softknn] early stop at epoch {epoch} (best epoch {best_epoch})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_val
