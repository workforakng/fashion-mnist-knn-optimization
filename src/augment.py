"""Pure-PyTorch batch image augmentation (no torchvision dependency).

All ops work on (B, 1, 28, 28) float tensors in [0, 1] on any device and are
fully vectorized (no per-sample GPU syncs).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def random_hflip(x: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    if p <= 0:
        return x
    mask = torch.rand(x.shape[0], device=x.device) < p
    if mask.any():
        x = x.clone()
        x[mask] = x[mask].flip(-1)
    return x


def shift(x: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """Deterministic integer shift with zero padding (dx: columns, dy: rows)."""
    b, _, h, w = x.shape
    padx, pady = abs(dx), abs(dy)
    xp = F.pad(x, (padx, padx, pady, pady))
    rows = (torch.arange(h, device=x.device).view(1, h, 1) - dy + pady).expand(b, h, w)
    cols = (torch.arange(w, device=x.device).view(1, 1, w) - dx + padx).expand(b, h, w)
    bidx = torch.arange(b, device=x.device).view(b, 1, 1).expand(b, h, w)
    zero = torch.zeros_like(bidx)
    out = xp[bidx, zero, rows, cols]  # (b, h, w)
    return out.unsqueeze(1)


def random_shift(x: torch.Tensor, max_px: int = 2) -> torch.Tensor:
    b = x.shape[0]
    dxs = torch.randint(-max_px, max_px + 1, (b,))
    dys = torch.randint(-max_px, max_px + 1, (b,))
    out = x.clone()
    for dx in range(-max_px, max_px + 1):
        for dy in range(-max_px, max_px + 1):
            mask = (dxs == dx) & (dys == dy)
            if mask.any():
                out[mask] = shift(x[mask], dx, dy)
    return out


def random_erasing(x: torch.Tensor, min_px: int = 2, max_px: int = 6) -> torch.Tensor:
    """Vectorized random erasing: per-sample zero rectangle."""
    b, _, h, w = x.shape
    dev = x.device
    eh = torch.randint(min_px, max_px + 1, (b,), device=dev)
    ew = torch.randint(min_px, max_px + 1, (b,), device=dev)
    y0 = (torch.rand(b, device=dev) * (h - eh).float()).floor().long().clamp_(min=0)
    x0 = (torch.rand(b, device=dev) * (w - ew).float()).floor().long().clamp_(min=0)
    rr = torch.arange(h, device=dev).view(1, h, 1)
    cc = torch.arange(w, device=dev).view(1, 1, w)
    in_row = (rr >= y0.view(b, 1, 1)) & (rr < (y0 + eh).view(b, 1, 1))
    in_col = (cc >= x0.view(b, 1, 1)) & (cc < (x0 + ew).view(b, 1, 1))
    mask = (in_row & in_col).unsqueeze(1)
    return x.masked_fill(mask, 0.0)


def random_rotate(x: torch.Tensor, max_deg: float = 10.0) -> torch.Tensor:
    if max_deg <= 0:
        return x
    b = x.shape[0]
    ang = (torch.rand(b, device=x.device) * 2 - 1) * math.radians(max_deg)
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.zeros(b, 2, 3, device=x.device, dtype=x.dtype)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class Augmenter:
    """Configurable on-the-fly augmentation pipeline."""

    def __init__(self, hflip_p: float = 0.5, shift_px: int = 2,
                 erase: bool = True, rotate_deg: float = 0.0):
        self.hflip_p = hflip_p
        self.shift_px = shift_px
        self.erase = erase
        self.rotate_deg = rotate_deg

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = random_hflip(x, self.hflip_p)
        if self.shift_px > 0:
            x = random_shift(x, self.shift_px)
        if self.rotate_deg > 0:
            x = random_rotate(x, self.rotate_deg)
        if self.erase:
            x = random_erasing(x)
        return x

    def __repr__(self) -> str:
        return (f"Augmenter(hflip_p={self.hflip_p}, shift_px={self.shift_px}, "
                f"rotate_deg={self.rotate_deg}, erase={self.erase})")


def tta_views(x: torch.Tensor) -> list[torch.Tensor]:
    """Deterministic TTA views: original, hflip, 4 single-axis 1px shifts."""
    views = [x, torch.flip(x, [-1])]
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        views.append(shift(x, dx, dy))
    return views
