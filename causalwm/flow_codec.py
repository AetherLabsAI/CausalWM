"""Optical flow <-> image codec (``flow_hsv_logsat_v1``), pure torch.

HSV encoding with log-compressed saturation, implemented in torch (no OpenCV
dependency):

    hue        = (atan2(v, u) + pi) / (2 pi)               in [0, 1)
    saturation = clamp(log1p(|f|) / log1p(clip_px), 0, 1)
    value      = 1
    image      = HSV->RGB * 2 - 1                           in [-1, 1]

Units: ``u, v`` are displacements in OUTPUT pixels (after resizing) between consecutive sampled frames, on the source frame's grid.
``clip_px`` is a fixed constant stored in the checkpoint metadata (no per-clip
normalization, so magnitudes stay comparable across clips). Zero flow decodes to white (S=0), which is the
frame-0 sentinel.
"""

from __future__ import annotations

import math

import torch

FLOW_CODEC = "flow_hsv_logsat_v1"
DEFAULT_CLIP_PX = 64.0


def hsv_to_rgb(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """h in [0,1), s,v in [0,1] -> rgb (..., 3) in [0,1] (standard cone conversion)."""
    h6 = (h % 1.0) * 6.0
    i = torch.floor(h6)
    f = h6 - i
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    i = i.long() % 6
    r = torch.where(i == 0, v, torch.where(i == 1, q, torch.where(i == 2, p, torch.where(i == 3, p, torch.where(i == 4, t, v)))))
    g = torch.where(i == 0, t, torch.where(i == 1, v, torch.where(i == 2, v, torch.where(i == 3, q, torch.where(i == 4, p, p)))))
    b = torch.where(i == 0, p, torch.where(i == 1, p, torch.where(i == 2, t, torch.where(i == 3, v, torch.where(i == 4, v, q)))))
    return torch.stack([r, g, b], dim=-1)


def rgb_to_hsv(rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """rgb (..., 3) in [0,1] -> (h in [0,1), s, v)."""
    r, g, b = rgb.unbind(-1)
    maxc = torch.maximum(torch.maximum(r, g), b)
    minc = torch.minimum(torch.minimum(r, g), b)
    v = maxc
    delta = maxc - minc
    s = torch.where(maxc > 0, delta / maxc.clamp(min=1e-12), torch.zeros_like(maxc))
    safe = delta.clamp(min=1e-12)
    rc = (maxc - r) / safe
    gc = (maxc - g) / safe
    bc = (maxc - b) / safe
    h = torch.where(r == maxc, bc - gc, torch.where(g == maxc, 2.0 + rc - bc, 4.0 + gc - rc))
    h = (h / 6.0) % 1.0
    h = torch.where(delta > 0, h, torch.zeros_like(h))
    return h, s, v


def encode_flow(flow_uv: torch.Tensor, clip_px: float = DEFAULT_CLIP_PX) -> torch.Tensor:
    """``(F, 2, H, W)`` px -> ``(F, 3, H, W)`` image in [-1, 1]."""
    if flow_uv.ndim != 4 or flow_uv.shape[1] != 2:
        raise ValueError(f"expected (F,2,H,W) flow, got {tuple(flow_uv.shape)}")
    u, v = flow_uv[:, 0].float(), flow_uv[:, 1].float()
    mag = torch.sqrt(u * u + v * v)
    hue = (torch.atan2(v, u) + math.pi) / (2 * math.pi)
    sat = (torch.log1p(mag) / math.log1p(clip_px)).clamp(0.0, 1.0)
    rgb = hsv_to_rgb(hue, sat, torch.ones_like(sat))  # (F,H,W,3)
    return rgb.permute(0, 3, 1, 2).contiguous() * 2.0 - 1.0


def decode_flow(image: torch.Tensor, clip_px: float = DEFAULT_CLIP_PX) -> torch.Tensor:
    """``(F, 3, H, W)`` image in [-1, 1] -> ``(F, 2, H, W)`` px."""
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(f"expected (F,3,H,W) image, got {tuple(image.shape)}")
    rgb = ((image.float() + 1.0) / 2.0).clamp(0.0, 1.0).permute(0, 2, 3, 1)
    h, s, _ = rgb_to_hsv(rgb)
    ang = h * 2 * math.pi - math.pi
    mag = torch.expm1(s * math.log1p(clip_px))
    return torch.stack([mag * torch.cos(ang), mag * torch.sin(ang)], dim=1)


def zero_flow_image(frames: int, height: int, width: int, device=None) -> torch.Tensor:
    """Frame-0 sentinel (white = zero displacement) as an image batch."""
    return torch.ones(frames, 3, height, width, device=device)
