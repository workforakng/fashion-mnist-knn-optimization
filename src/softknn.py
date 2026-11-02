"""Differentiable Soft-KNN: learnable metric projection + temperature-scaled
softmax over negative per-class distances.

Forward pass (all differentiable w.r.t. the projection matrix W):
    z      = L2normalize(W x)
    d_c    = mean of the k smallest distances from z to bank points of class c
    logits = -d / tau            (smaller distance -> larger logit)
    p      = softmax(logits)     -> trained with cross-entropy

The memory bank of training embeddings is refreshed periodically (once per
epoch by default, or every `bank_refresh_steps` steps) and kept detached
between refreshes. This is a periodically refreshed memory bank in the sense
of Wu et al. (2018); there is no momentum encoder (it is not MoCo). Gradients
flow only through the query embedding; the top-k selection itself has no
gradient.

Self-match caveat: when the queries are the bank's own training images, each
query finds itself at distance ~0. `logits_from_z(..., self_idx=...)` and
`predict_probs(..., exclude_self=True)` mask those entries (leave-one-out, as
in NCA). Masking is opt-in; the default keeps the behaviour that produced the
logged results.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .augment import tta_views


class MetricProjector(nn.Module):
    """z = L2-normalized projection of the input (learnable metric).

    hidden_dim > 0 adds a nonlinear layer: 784 -> hidden (ReLU) -> embed.
    """

    def __init__(self, in_dim: int = 784, embed_dim: int = 128, pca_components=None,
                 hidden_dim: int = 0):
        super().__init__()
        self.lin = nn.Linear(in_dim, hidden_dim if hidden_dim > 0 else embed_dim,
                             bias=False)
        self.out = nn.Linear(hidden_dim, embed_dim, bias=False) if hidden_dim > 0 else None
        if pca_components is not None:
            with torch.no_grad():
                comp = torch.as_tensor(pca_components, dtype=torch.float32)
                n_avail = min(comp.shape[0], self.lin.weight.shape[0])
                self.lin.weight[:n_avail].copy_(comp[:n_avail])
                if n_avail < self.lin.weight.shape[0]:
                    nn.init.orthogonal_(self.lin.weight[n_avail:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.lin(x)
        if self.out is not None:
            z = self.out(F.relu(z))
        return F.normalize(z, p=2, dim=-1)


def mask_self_matches(dist: torch.Tensor, self_idx: torch.Tensor) -> torch.Tensor:
    """Leave-one-out mask: return a copy of dist (B,N) with dist[i, self_idx[i, j]]
    set to +inf, so a query can never pick its own bank entry (or its own
    augmented copies) as a neighbour. self_idx: (B,) or (B, m) long tensor."""
    if self_idx.dim() == 1:
        self_idx = self_idx.unsqueeze(1)
    return dist.scatter(1, self_idx.to(dist.device), float("inf"))


def per_class_topk_logits(dist: torch.Tensor, bank_y: torch.Tensor,
                          k: int, n_classes: int, tau: float) -> torch.Tensor:
    """dist (B,N) -> logits (B,C) = -mean(top-k smallest distances per class) / tau."""
    b = dist.shape[0]
    out = torch.empty(b, n_classes, device=dist.device, dtype=dist.dtype)
    for c in range(n_classes):
        dc = dist[:, bank_y == c]
        kk = min(k, dc.shape[1])
        top = torch.topk(dc, kk, dim=1, largest=False).values
        out[:, c] = -top.mean(dim=1)
    return out / tau


class ConvMetric(nn.Module):
    """Convolutional metric: 2 conv blocks -> linear head -> L2-normalized z."""

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 14x14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 7x7
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, embed_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        img = x.view(-1, 1, 28, 28)
        return F.normalize(self.features(img), p=2, dim=-1)


class SoftKNN(nn.Module):
    def __init__(self, in_dim: int = 784, embed_dim: int = 128, n_classes: int = 10,
                 pca_components=None, hidden_dim: int = 0, metric: str = "linear"):
        super().__init__()
        if metric == "conv":
            self.proj = ConvMetric(embed_dim)
        else:
            self.proj = MetricProjector(in_dim, embed_dim, pca_components, hidden_dim)
        self.n_classes = n_classes

    def embed(self, x_flat: torch.Tensor) -> torch.Tensor:
        return self.proj(x_flat)

    def logits_from_z(self, zq, bank_z, bank_y, k: int, tau: float,
                      self_idx=None) -> torch.Tensor:
        """self_idx (optional): bank indices of each query's own entries; they
        are excluded from the neighbour search (leave-one-out)."""
        dist = torch.cdist(zq, bank_z)
        if self_idx is not None:
            dist = mask_self_matches(dist, self_idx)
        return per_class_topk_logits(dist, bank_y, k, self.n_classes, tau)

    def forward(self, xq_flat, bank_x_flat, bank_y, k: int, tau: float) -> torch.Tensor:
        zq = self.embed(xq_flat)
        zb = self.embed(bank_x_flat)
        return self.logits_from_z(zq, zb, bank_y, k, tau)


@torch.no_grad()
def embed_chunked(model, x, chunk: int = 8192):
    """Embed a large tensor in chunks to bound conv activation memory."""
    outs = []
    for i in range(0, x.shape[0], chunk):
        outs.append(model.embed(x[i:i + chunk].flatten(1)))
    return torch.cat(outs, dim=0)


def load_softknn(path: str, device: str) -> SoftKNN:
    """Rebuild a SoftKNN (linear, MLP, or conv metric) from a saved state_dict."""
    state = torch.load(path, map_location=device)
    if "proj.features.0.weight" in state:
        lin_idx = max(int(k.split(".")[2]) for k in state
                      if k.startswith("proj.features.") and k.endswith(".weight")
                      and state[k].dim() == 2)
        model = SoftKNN(embed_dim=state[f"proj.features.{lin_idx}.weight"].shape[0],
                        metric="conv").to(device)
    elif "proj.out.weight" in state:
        hidden = state["proj.lin.weight"].shape[0]
        embed = state["proj.out.weight"].shape[0]
        model = SoftKNN(embed_dim=embed, hidden_dim=hidden).to(device)
    else:
        model = SoftKNN(embed_dim=state["proj.lin.weight"].shape[0]).to(device)
    model.load_state_dict(state)
    return model


@torch.no_grad()
def predict_probs(model, bank_z, bank_y, x_flat, k: int, tau: float,
                  batch_size: int = 1024, device: str = "cuda",
                  exclude_self: bool = False) -> torch.Tensor:
    """Softmax probabilities for flat feature matrix x_flat (numpy or tensor).

    exclude_self=True is for querying the bank's own images (e.g. distillation
    teacher targets): row i of x_flat must be bank entry i, and that entry is
    masked out (leave-one-out). Default False = original behaviour, where a
    training image matches itself at distance ~0.
    """
    if exclude_self and len(x_flat) > len(bank_z):
        raise ValueError("exclude_self requires x_flat rows to be bank entries")
    model.eval()
    probs = []
    for i in range(0, len(x_flat), batch_size):
        xb = torch.as_tensor(x_flat[i:i + batch_size], dtype=torch.float32, device=device)
        self_idx = (torch.arange(i, i + len(xb), device=bank_z.device)
                    if exclude_self else None)
        logits = model.logits_from_z(model.embed(xb), bank_z, bank_y, k, tau,
                                     self_idx=self_idx)
        probs.append(F.softmax(logits, dim=1).cpu())
    return torch.cat(probs)


@torch.no_grad()
def predict_probs_tta(model, bank_z, bank_y, x_img, k: int, tau: float,
                      batch_size: int = 1024, device: str = "cuda") -> torch.Tensor:
    """TTA probabilities: average softmax over 6 deterministic views.

    x_img: (M, 1, 28, 28) tensor on CPU.
    """
    model.eval()
    m = len(x_img)
    n_views = 6
    probs = torch.zeros(m, model.n_classes)
    for i in range(0, m, batch_size):
        xb = x_img[i:i + batch_size].to(device)
        acc = None
        for v in tta_views(xb):
            logits = model.logits_from_z(model.embed(v.flatten(1)), bank_z, bank_y, k, tau)
            p = F.softmax(logits, dim=1).cpu()
            acc = p if acc is None else acc + p
        probs[i:i + batch_size] = acc / n_views
    return probs
