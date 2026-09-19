# Modified by Aether AI for CausalWM (2026). Original file: Lightricks LTX-2 (`ltx-core`).
# Distributed under the LTX-2 Community License Agreement; see LICENSE and NOTICE.

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Modality:
    """
    Input data for a single modality (video or audio) in the transformer.
    Bundles the latent tokens, timestep embeddings, positional information,
    and text conditioning context for processing by the diffusion transformer.
    Attributes:
        latent: Patchified latent tokens, shape ``(B, T, D)`` where *B* is
            the batch size, *T* is the total number of tokens (noisy +
            conditioning), and *D* is the input dimension.
        timesteps: Per-token timestep embeddings, shape ``(B, T)``.
        positions: Positional coordinates, shape ``(B, 3, T)`` for video
            (time, height, width) or ``(B, 1, T)`` for audio.
        context: Text conditioning embeddings from the prompt encoder.
        enabled: Whether this modality is active in the current forward pass.
        context_mask: Optional mask for the text context tokens.
        attention_mask: Optional 2-D self-attention mask, shape ``(B, T, T)``.
            Values in ``[0, 1]`` where ``1`` = full attention and ``0`` = no
            attention. ``None`` means unrestricted (full) attention between
            all tokens. Built incrementally by conditioning items; see
            :class:`~ltx_core.conditioning.types.attention_strength_wrapper.ConditioningItemAttentionStrengthWrapper`.
        action: Optional per-latent-frame action conditioning, shape
            ``(B, T_lat, action_in)`` where *T_lat* is the number of latent
            frames and ``action_in`` is the per-frame action feature width
            (e.g. ``latent_action_dim * num_action_per_latent_frame``). Fed to
            the transformer's action embedder and added to the AdaLN timestep
            embedding (action conditioning). ``None``
            disables action conditioning and is bit-equivalent to the base
            (text-only) model.
    """

    latent: (
        torch.Tensor
    )  # Shape: (B, T, D) where B is the batch size, T is the number of tokens, and D is input dimension
    sigma: torch.Tensor  # Shape: (B,). Current sigma value, used for cross-attention timestep calculation.
    timesteps: torch.Tensor  # Shape: (B, T) where T is the number of timesteps
    positions: (
        torch.Tensor
    )  # Shape: (B, 3, T) for video, where 3 is the number of dimensions and T is the number of tokens
    context: torch.Tensor
    enabled: bool = True
    context_mask: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None
    action: torch.Tensor | None = None  # Shape: (B, T_lat, action_in). None disables action conditioning.
    # (n_video, n_aux1, ...) segment lengths when auxiliary streams
    # are concatenated after the video tokens. Consumed by the args preprocessor
    # so the frame-major action broadcast maps action frames to each segment's own
    # spatial grid (a naive repeat_interleave over the full concat would silently
    # mis-assign frames). None = no aux streams, bit-equivalent to base.
    icl_stream_slices: tuple[int, ...] | None = None

    def split(self, sizes: list[int]) -> list[Modality]:
        """Split along the batch dimension into chunks of the given sizes."""
        n = len(sizes)
        split_fields: dict[str, list[torch.Tensor | None] | list[bool]] = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if isinstance(value, torch.Tensor):
                split_fields[f.name] = list(value.split(sizes, dim=0))
            elif value is None or isinstance(value, (bool, tuple)):
                # bools and per-sequence tuples (icl_stream_slices) are batch-invariant
                split_fields[f.name] = [value] * n
            else:
                raise TypeError(f"Cannot split field {f.name!r}: unsupported type {type(value)}")
        return [Modality(**{name: parts[i] for name, parts in split_fields.items()}) for i in range(n)]
