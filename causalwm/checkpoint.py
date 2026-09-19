"""Fail-closed CausalWM checkpoint validation and loading.

Tensor shape alone is never evidence of compatibility: a checkpoint must carry
the stream-registry metadata and the matching stream parameters.
"""
from __future__ import annotations

from collections.abc import Collection, Mapping
import math

import torch

from causalwm.layout import NUM_STREAMS, registry_metadata

STREAM_KEYS = (
    "icl_stream_embeddings", "icl_stream_gates",
    # per-stream input/output heads + modality AdaLN rows
    "icl_stream_patchify_proj", "icl_stream_proj_out", "icl_stream_scale_shift_table", "icl_modality_adaln",
)
# keys whose leading dim is one row per aux stream
_PER_STREAM_ROW_KEYS = ("icl_stream_embeddings", "icl_stream_gates", "icl_stream_scale_shift_table")


def validate_full_scene_checkpoint(
    state: Mapping[str, torch.Tensor], metadata: Mapping[str, str] | None, *,
    expected_stream_keys: Collection[str] | None = None,
) -> None:
    metadata = metadata or {}
    # Filter keys first: a lazy Mapping.items() would read the entire 22B model.
    stream_tensors = {key: state[key] for key in state if any(name in key for name in STREAM_KEYS)}
    for key, expected in registry_metadata().items():
        if metadata.get(key) != expected:
            raise ValueError(f"checkpoint requires {key}={expected!r}; refusing an incompatible checkpoint")
    if not stream_tensors or "icl_stream_embeddings" not in stream_tensors:
        raise ValueError("checkpoint is missing its stream embeddings")
    if expected_stream_keys is not None and set(stream_tensors) != set(expected_stream_keys):
        missing = sorted(set(expected_stream_keys) - set(stream_tensors))
        extra = sorted(set(stream_tensors) - set(expected_stream_keys))
        raise ValueError(f"checkpoint stream/gate keys mismatch: missing={missing[:8]}, extra={extra[:8]}")
    try:
        flow_clip_px = float(metadata["flow_clip_px"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("checkpoint requires finite positive flow_clip_px") from exc
    if not math.isfinite(flow_clip_px) or flow_clip_px <= 0:
        raise ValueError("checkpoint requires finite positive flow_clip_px")
    for key, tensor in stream_tensors.items():
        if any(name in key for name in _PER_STREAM_ROW_KEYS):
            if tensor.ndim < 1 or tensor.shape[0] != NUM_STREAMS:
                raise ValueError(f"checkpoint {key} must have {NUM_STREAMS} stream rows")
        elif "icl_modality_adaln" in key:
            if tensor.ndim != 2 or tensor.shape[0] != NUM_STREAMS + 1:
                raise ValueError(f"checkpoint {key} must have {NUM_STREAMS + 1} rows (video + streams)")
    for head in ("icl_stream_patchify_proj", "icl_stream_proj_out"):
        idx = {int(key.split(f"{head}.")[1].split(".")[0]) for key in stream_tensors if f"{head}." in key}
        if idx and idx != set(range(NUM_STREAMS)):
            raise ValueError(f"checkpoint {head} must have exactly heads {list(range(NUM_STREAMS))}, got {sorted(idx)}")
    heads_in_ckpt = any("icl_stream_proj_out" in key for key in stream_tensors)
    tagged_heads = metadata.get("robot_cot_modality_heads") == "1"
    if heads_in_ckpt != tagged_heads:
        raise ValueError(
            f"checkpoint arch mismatch: proj_out heads present={heads_in_ckpt} but metadata robot_cot_modality_heads={metadata.get('robot_cot_modality_heads')!r}"
        )
    # The AdaLN rows need their own presence <-> tag check; a checkpoint that carries
    # icl_modality_adaln but is tagged adaln=0 would make the sampler build a model without the rows.
    adaln_in_ckpt = any("icl_modality_adaln" in key for key in stream_tensors)
    tagged_adaln = metadata.get("robot_cot_modality_adaln") == "1"
    if adaln_in_ckpt != tagged_adaln:
        raise ValueError(
            f"checkpoint arch mismatch: modality AdaLN present={adaln_in_ckpt} but metadata robot_cot_modality_adaln={metadata.get('robot_cot_modality_adaln')!r}"
        )


def load_full_scene_state(model: torch.nn.Module, state: dict[str, torch.Tensor],
                          metadata: Mapping[str, str] | None = None) -> dict[str, list[str]]:
    """Validate, then load ``state`` into ``model``; every parameter must be present."""
    stream_keys = [key for key in model.state_dict() if any(name in key for name in STREAM_KEYS)]
    validate_full_scene_checkpoint(state, metadata, expected_stream_keys=stream_keys)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={list(missing)[:8]}, unexpected={list(unexpected)[:8]}")
    return {"missing": [], "unexpected": []}
