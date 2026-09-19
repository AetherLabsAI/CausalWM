# Modified by Aether AI for CausalWM (2026). Original file: Lightricks LTX-2 (`ltx-core`).
# Distributed under the LTX-2 Community License Agreement; see LICENSE and NOTICE.

from dataclasses import dataclass, replace

import torch
from torch.utils._pytree import register_dataclass

from ltx_core.model.transformer.adaln import ActionEmbedder, ActionInputProjection, AdaLayerNormSingle
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.rope import (
    LTXRopeType,
    generate_freq_grid_np,
    generate_freq_grid_pytorch,
    precompute_freqs_cis,
)


@dataclass(frozen=True)
class TransformerArgs:
    x: torch.Tensor
    context: torch.Tensor
    context_mask: torch.Tensor
    timesteps: torch.Tensor
    embedded_timestep: torch.Tensor
    positional_embeddings: torch.Tensor
    cross_positional_embeddings: torch.Tensor | None
    cross_scale_shift_timestep: torch.Tensor | None
    cross_gate_timestep: torch.Tensor | None
    enabled: bool
    prompt_timestep: torch.Tensor | None = None
    self_attention_mask: torch.Tensor | None = (
        None  # Additive log-space self-attention bias (B, 1, T, T), None = full attention
    )


try:
    register_dataclass(TransformerArgs)
except ValueError as exc:
    if "already registered" not in str(exc):
        raise

class TransformerArgsPreprocessor:
    def __init__(  # noqa: PLR0913
        self,
        patchify_proj: torch.nn.Linear,
        adaln: AdaLayerNormSingle,
        inner_dim: int,
        max_pos: list[int],
        num_attention_heads: int,
        use_middle_indices_grid: bool,
        timestep_scale_multiplier: int,
        double_precision_rope: bool,
        positional_embedding_theta: float,
        rope_type: LTXRopeType,
        caption_projection: torch.nn.Module | None = None,
        prompt_adaln: AdaLayerNormSingle | None = None,
        action_embedder: ActionEmbedder | None = None,
        action_input_proj: ActionInputProjection | None = None,
    ) -> None:
        self.patchify_proj = patchify_proj
        self.adaln = adaln
        self.inner_dim = inner_dim
        self.max_pos = max_pos
        self.num_attention_heads = num_attention_heads
        self.use_middle_indices_grid = use_middle_indices_grid
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.double_precision_rope = double_precision_rope
        self.positional_embedding_theta = positional_embedding_theta
        self.rope_type = rope_type
        self.caption_projection = caption_projection
        self.prompt_adaln = prompt_adaln
        # Optional action embedder. None → action conditioning off.
        self.action_embedder = action_embedder
        # Optional raw-action → latent-action projection, applied per pair
        # before the embedder. None → action arrives already at latent width.
        self.action_input_proj = action_input_proj

    def _prepare_timestep(
        self,
        timestep: torch.Tensor,
        adaln: AdaLayerNormSingle,
        batch_size: int,
        hidden_dtype: torch.dtype,
        action_embed: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare timestep embeddings.

        ``action_embed``, if given, is a per-token action embedding of
        shape ``(B, T, D)`` already broadcast to match the flattened timestep
        layout; it is flattened to ``(B*T, D)`` and added inside the AdaLN
        before the modulation linear.
        """
        timestep_scaled = timestep * self.timestep_scale_multiplier
        action_embed_flat = None
        if action_embed is not None:
            action_embed_flat = action_embed.reshape(-1, action_embed.shape[-1])
        timestep, embedded_timestep = adaln(
            timestep_scaled.flatten(),
            hidden_dtype=hidden_dtype,
            action_embed=action_embed_flat,
        )
        # Second dimension is 1 or number of tokens (if timestep_per_token)
        timestep = timestep.view(batch_size, -1, timestep.shape[-1])
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.shape[-1])

        return timestep, embedded_timestep

    def _prepare_action(
        self,
        action: torch.Tensor,
        num_tokens: int,
        hidden_dtype: torch.dtype,
        stream_slices: tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Embed per-latent-frame action (dual-path) and broadcast to per-token.

        ``action``: ``(B, T_lat, action_in)`` per-latent-frame action features.
        Returns ``(action_main, action_mod)``:
          * ``action_main`` ``(B, num_tokens, D)`` — main path, added to the
            timestep embedding inside AdaLN.
          * ``action_mod`` ``(B, num_tokens, modulation_dim)`` or ``None`` —
            direct path, added straight to the per-token
            AdaLN modulation (``TransformerArgs.timesteps``).
        Each frame's embedding is repeated across its ``num_tokens // T_lat``
        spatial tokens (frame-major, matches patchifier ``b c (f h w) -> b (f h w) c``).

        ``stream_slices``: when auxiliary streams are concatenated
        after the video tokens, the sequence is a chain of frame-major segments
        with DIFFERENT spatial grids ([video f0..fT][aux1 f0..fT][aux2 f0..fT]);
        one global repeat_interleave would silently mis-assign frames (the total
        token count usually still divides T_lat, so nothing would error). Each
        segment instead gets its own frame-aligned broadcast — aux tokens receive
        the SAME per-frame action modulation as the video frame they mirror
        (action is a global conditioning signal, like the caption context).

        Both paths force the first (conditioning) latent frame to strictly zero
        AFTER the embedder (not by zeroing the raw input — a zero raw vector does
        NOT survive the MLP biases).
        """
        b, t_lat, _ = action.shape
        if stream_slices is not None:
            if sum(stream_slices) != num_tokens:
                raise ValueError(
                    f"stream_slices {stream_slices} do not sum to num_tokens ({num_tokens})"
                )
            bad = [seg for seg in stream_slices if seg % t_lat != 0]
            if bad:
                raise ValueError(
                    f"stream segment lengths {bad} not divisible by T_lat ({t_lat}) "
                    "for frame-major action broadcast"
                )
        elif num_tokens % t_lat != 0:
            raise ValueError(
                f"num_tokens ({num_tokens}) must be divisible by T_lat ({t_lat}) "
                "for frame-major action broadcast"
            )
        tokens_per_frame = num_tokens // t_lat
        action = action.to(hidden_dtype)
        # Project the raw per-pair action to latent width before the
        # embedder. ``action`` arrives as (B, T_lat, npf * action_in); reshape to
        # (B, T_lat, npf, action_in), project the last dim (raw → latent), then
        # re-flatten to (B, T_lat, npf * latent) so the embedder input width is
        # unchanged. No-op (skipped) when action_input_proj is None.
        if self.action_input_proj is not None:
            b, t_, w = action.shape
            raw = self.action_input_proj.action_in
            npf = w // raw
            action = self.action_input_proj(action.view(b, t_, npf, raw)).reshape(b, t_, -1)
        action_main, action_mod = self.action_embedder(action)

        # Force the first (conditioning) latent frame to strictly zero, AFTER the
        # MLP, on every path. Masking (not in-place) to stay autograd-safe.
        def _zero_first_then_broadcast(emb: torch.Tensor) -> torch.Tensor:
            if t_lat > 1:
                frame_mask = emb.new_ones(1, t_lat, 1)
                frame_mask[:, 0, :] = 0.0
                emb = emb * frame_mask
            if stream_slices is not None:
                # per-segment frame-aligned broadcast (see docstring)
                return torch.cat(
                    [emb.repeat_interleave(seg // t_lat, dim=1) for seg in stream_slices],
                    dim=1,
                )  # (B, num_tokens, *)
            return emb.repeat_interleave(tokens_per_frame, dim=1)  # (B, num_tokens, *)

        action_main = _zero_first_then_broadcast(action_main)
        action_mod = _zero_first_then_broadcast(action_mod) if action_mod is not None else None
        return action_main, action_mod

    def _prepare_context(
        self,
        context: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Prepare context for transformer blocks."""
        if self.caption_projection is not None:
            context = self.caption_projection(context)
        batch_size = x.shape[0]
        return context.view(batch_size, -1, x.shape[-1])

    def _prepare_attention_mask(self, attention_mask: torch.Tensor | None, x_dtype: torch.dtype) -> torch.Tensor | None:
        """Prepare attention mask."""
        if attention_mask is None or torch.is_floating_point(attention_mask):
            return attention_mask

        return (attention_mask - 1).to(x_dtype).reshape(
            (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
        ) * torch.finfo(x_dtype).max

    def _prepare_self_attention_mask(
        self, attention_mask: torch.Tensor | None, x_dtype: torch.dtype
    ) -> torch.Tensor | None:
        """Prepare self-attention mask by converting [0,1] values to additive log-space bias.
        Input shape: (B, T, T) with values in [0, 1].
        Output shape: (B, 1, T, T) with 0.0 for full attention and a large negative value
        for masked positions.
        Positions with attention_mask <= 0 are fully masked (mapped to the dtype's minimum
        representable value). Strictly positive entries are converted via log-space for
        smooth attenuation, with small values clamped for numerical stability.
        Returns None if input is None (no masking).
        """
        if attention_mask is None:
            return None

        # Convert [0, 1] attention mask to additive log-space bias:
        #   1.0 -> log(1.0) = 0.0  (no bias, full attention)
        #   0.0 -> finfo.min        (fully masked)
        finfo = torch.finfo(x_dtype)
        eps = finfo.tiny

        bias = torch.full_like(attention_mask, finfo.min, dtype=x_dtype)
        positive = attention_mask > 0
        if positive.any():
            bias[positive] = torch.log(attention_mask[positive].clamp(min=eps)).to(x_dtype)

        return bias.unsqueeze(1)  # (B, 1, T, T) for head broadcast

    def _prepare_positional_embeddings(
        self,
        positions: torch.Tensor,
        inner_dim: int,
        max_pos: list[int],
        use_middle_indices_grid: bool,
        num_attention_heads: int,
        x_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Prepare positional embeddings."""
        freq_grid_generator = generate_freq_grid_np if self.double_precision_rope else generate_freq_grid_pytorch
        pe = precompute_freqs_cis(
            positions,
            dim=inner_dim,
            out_dtype=x_dtype,
            theta=self.positional_embedding_theta,
            max_pos=max_pos,
            use_middle_indices_grid=use_middle_indices_grid,
            num_attention_heads=num_attention_heads,
            rope_type=self.rope_type,
            freq_grid_generator=freq_grid_generator,
        )
        return pe

    def prepare(
        self,
        modality: Modality,
        cross_modality: Modality | None = None,  # noqa: ARG002
        extra_timestep_embed: torch.Tensor | None = None,
    ) -> TransformerArgs:
        """``extra_timestep_embed`` (modality AdaLN): optional ``(B, T, D)``
        per-token embedding added to the timestep embedding BEFORE the modulation
        linear (same injection point as the action main path), so every
        block's scale/shift/gate and the output modulation know the token's
        modality. None keeps the base numerics exactly."""
        x = self.patchify_proj(modality.latent)
        batch_size = x.shape[0]
        # Dual-path action conditioning, if present.
        #   action_main → added to timestep embedding inside AdaLN (main path).
        #   action_mod  → added directly to the AdaLN modulation tensor below
        #                 (direct residual onto every block's scale/shift/gate).
        action_main = None
        action_mod = None
        if self.action_embedder is not None and modality.action is not None:
            action_main, action_mod = self._prepare_action(
                modality.action,
                num_tokens=x.shape[1],
                hidden_dtype=modality.latent.dtype,
                stream_slices=modality.icl_stream_slices,
            )
        if extra_timestep_embed is not None:
            if extra_timestep_embed.shape[:2] != (batch_size, x.shape[1]):
                raise ValueError(
                    f"extra_timestep_embed {tuple(extra_timestep_embed.shape)} must be (B={batch_size}, T={x.shape[1]}, D)"
                )
            action_main = extra_timestep_embed if action_main is None else action_main + extra_timestep_embed
        timestep, embedded_timestep = self._prepare_timestep(
            modality.timesteps, self.adaln, batch_size, modality.latent.dtype, action_embed=action_main
        )
        if action_mod is not None:
            # Direct residual onto the per-token AdaLN modulation. `timestep` is
            # (B, T, coeff*D) — the same tensor get_ada_values() reshapes into
            # per-block scale/shift/gate.
            timestep = timestep + action_mod.to(timestep.dtype)
        prompt_timestep = None
        if self.prompt_adaln is not None:
            prompt_timestep, _ = self._prepare_timestep(
                modality.sigma, self.prompt_adaln, batch_size, modality.latent.dtype
            )
        context = self._prepare_context(modality.context, x)
        attention_mask = self._prepare_attention_mask(modality.context_mask, modality.latent.dtype)
        pe = self._prepare_positional_embeddings(
            positions=modality.positions,
            inner_dim=self.inner_dim,
            max_pos=self.max_pos,
            use_middle_indices_grid=self.use_middle_indices_grid,
            num_attention_heads=self.num_attention_heads,
            x_dtype=modality.latent.dtype,
        )
        self_attention_mask = self._prepare_self_attention_mask(modality.attention_mask, modality.latent.dtype)
        return TransformerArgs(
            x=x,
            context=context,
            context_mask=attention_mask,
            timesteps=timestep,
            embedded_timestep=embedded_timestep,
            positional_embeddings=pe,
            cross_positional_embeddings=None,
            cross_scale_shift_timestep=None,
            cross_gate_timestep=None,
            enabled=modality.enabled,
            prompt_timestep=prompt_timestep,
            self_attention_mask=self_attention_mask,
        )


class MultiModalTransformerArgsPreprocessor:
    def __init__(  # noqa: PLR0913
        self,
        patchify_proj: torch.nn.Linear,
        adaln: AdaLayerNormSingle,
        cross_scale_shift_adaln: AdaLayerNormSingle,
        cross_gate_adaln: AdaLayerNormSingle,
        inner_dim: int,
        max_pos: list[int],
        num_attention_heads: int,
        cross_pe_max_pos: int,
        use_middle_indices_grid: bool,
        audio_cross_attention_dim: int,
        timestep_scale_multiplier: int,
        double_precision_rope: bool,
        positional_embedding_theta: float,
        rope_type: LTXRopeType,
        av_ca_timestep_scale_multiplier: int,
        caption_projection: torch.nn.Module | None = None,
        prompt_adaln: AdaLayerNormSingle | None = None,
        action_embedder: ActionEmbedder | None = None,
        action_input_proj: ActionInputProjection | None = None,
    ) -> None:
        self.simple_preprocessor = TransformerArgsPreprocessor(
            patchify_proj=patchify_proj,
            adaln=adaln,
            inner_dim=inner_dim,
            max_pos=max_pos,
            num_attention_heads=num_attention_heads,
            use_middle_indices_grid=use_middle_indices_grid,
            timestep_scale_multiplier=timestep_scale_multiplier,
            double_precision_rope=double_precision_rope,
            positional_embedding_theta=positional_embedding_theta,
            rope_type=rope_type,
            caption_projection=caption_projection,
            prompt_adaln=prompt_adaln,
            action_embedder=action_embedder,  # action injected on the video modality
            action_input_proj=action_input_proj,  # raw->latent projection
        )
        self.cross_scale_shift_adaln = cross_scale_shift_adaln
        self.cross_gate_adaln = cross_gate_adaln
        self.cross_pe_max_pos = cross_pe_max_pos
        self.audio_cross_attention_dim = audio_cross_attention_dim
        self.av_ca_timestep_scale_multiplier = av_ca_timestep_scale_multiplier

    def prepare(
        self,
        modality: Modality,
        cross_modality: Modality | None = None,
        extra_timestep_embed: torch.Tensor | None = None,
    ) -> TransformerArgs:
        transformer_args = self.simple_preprocessor.prepare(modality, extra_timestep_embed=extra_timestep_embed)
        if cross_modality is None:
            return transformer_args

        if cross_modality.sigma.numel() > 1:
            if cross_modality.sigma.shape[0] != modality.timesteps.shape[0]:
                raise ValueError("Cross modality sigma must have the same batch size as the modality")
            if cross_modality.sigma.ndim != 1:
                raise ValueError("Cross modality sigma must be a 1D tensor")

        cross_timestep = cross_modality.sigma.view(
            modality.timesteps.shape[0], 1, *[1] * len(modality.timesteps.shape[2:])
        )

        cross_pe = self.simple_preprocessor._prepare_positional_embeddings(
            positions=modality.positions[:, 0:1, :],
            inner_dim=self.audio_cross_attention_dim,
            max_pos=[self.cross_pe_max_pos],
            use_middle_indices_grid=True,
            num_attention_heads=self.simple_preprocessor.num_attention_heads,
            x_dtype=modality.latent.dtype,
        )

        cross_scale_shift_timestep, cross_gate_timestep = self._prepare_cross_attention_timestep(
            timestep=cross_timestep,
            timestep_scale_multiplier=self.simple_preprocessor.timestep_scale_multiplier,
            batch_size=transformer_args.x.shape[0],
            hidden_dtype=modality.latent.dtype,
        )

        return replace(
            transformer_args,
            cross_positional_embeddings=cross_pe,
            cross_scale_shift_timestep=cross_scale_shift_timestep,
            cross_gate_timestep=cross_gate_timestep,
        )

    def _prepare_cross_attention_timestep(
        self,
        timestep: torch.Tensor | None,
        timestep_scale_multiplier: int,
        batch_size: int,
        hidden_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare cross attention timestep embeddings."""
        timestep = timestep * timestep_scale_multiplier

        av_ca_factor = self.av_ca_timestep_scale_multiplier / timestep_scale_multiplier

        scale_shift_timestep, _ = self.cross_scale_shift_adaln(
            timestep.flatten(),
            hidden_dtype=hidden_dtype,
        )
        scale_shift_timestep = scale_shift_timestep.view(batch_size, -1, scale_shift_timestep.shape[-1])
        gate_noise_timestep, _ = self.cross_gate_adaln(
            timestep.flatten() * av_ca_factor,
            hidden_dtype=hidden_dtype,
        )
        gate_noise_timestep = gate_noise_timestep.view(batch_size, -1, gate_noise_timestep.shape[-1])

        return scale_shift_timestep, gate_noise_timestep
