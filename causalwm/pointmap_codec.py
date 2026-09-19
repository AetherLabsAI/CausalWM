"""XYZ pointmap image codec (signed-log), decode side.

The pointmap stream is a three-channel image in [-1, 1]: X/Y use a signed-log
companding with range +/-16, Z a log companding with range 0..32. Units are
relative to the first frame's median depth, NOT metres.
"""
from __future__ import annotations

import math

import torch


def decode_pointmap_log(image: torch.Tensor) -> torch.Tensor:
    """``(F, 3, H, W)`` image in [-1, 1] -> camera-frame XYZ ``(F, 3, H, W)``."""
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("expected image (F,3,H,W)")
    x = image.float().clamp(-1, 1)
    xy = x[:, :2].sign() * (x[:, :2].abs() * math.log1p(16)).expm1()
    z = ((x[:, 2:3] + 1) * .5 * math.log1p(32)).expm1()
    return torch.cat([xy, z], 1)
