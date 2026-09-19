# Modified by Aether AI for CausalWM (2026). Original file: Lightricks LTX-2 (`ltx-core`).
# Distributed under the LTX-2 Community License Agreement; see LICENSE and NOTICE.

from dataclasses import replace
from enum import Enum

import copy

import torch

from ltx_core.guidance.perturbations import BatchedPerturbationConfig
from ltx_core.model.transformer.adaln import (
    ActionEmbedder,
    ActionInputProjection,
    AdaLayerNormSingle,
    adaln_embedding_coefficient,
)
from ltx_core.model.transformer.attention import AttentionCallable, AttentionFunction
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.model.transformer.transformer import BasicAVTransformerBlock, TransformerConfig
from ltx_core.model.transformer.transformer_args import (
    MultiModalTransformerArgsPreprocessor,
    TransformerArgs,
    TransformerArgsPreprocessor,
)
from ltx_core.utils import to_denoised


class LTXModelType(Enum):
    AudioVideo = "ltx av model"
    VideoOnly = "ltx video only model"
    AudioOnly = "ltx audio only model"

    def is_video_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.VideoOnly)

    def is_audio_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.AudioOnly)


class LTXModel(torch.nn.Module):
    """
    LTX model transformer implementation.
    This class implements the transformer blocks for the LTX model.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        model_type: LTXModelType = LTXModelType.AudioVideo,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        in_channels: int = 128,
        out_channels: int = 128,
        num_layers: int = 48,
        cross_attention_dim: int = 4096,
        norm_eps: float = 1e-06,
        attention_type: AttentionFunction | AttentionCallable = AttentionFunction.DEFAULT,
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: list[int] | None = None,
        timestep_scale_multiplier: int = 1000,
        use_middle_indices_grid: bool = True,
        audio_num_attention_heads: int = 32,
        audio_attention_head_dim: int = 64,
        audio_in_channels: int = 128,
        audio_out_channels: int = 128,
        audio_cross_attention_dim: int = 2048,
        audio_positional_embedding_max_pos: list[int] | None = None,
        av_ca_timestep_scale_multiplier: int = 1,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        double_precision_rope: bool = False,
        apply_gated_attention: bool = False,
        caption_projection: torch.nn.Module | None = None,
        audio_caption_projection: torch.nn.Module | None = None,
        cross_attention_adaln: bool = False,
        action_conditioning: bool = False,
        latent_action_dim: int = 32,
        num_action_per_latent_frame: int = 8,
        action_feature_dim: int | None = None,
    ):
        super().__init__()
        self._enable_gradient_checkpointing = False
        self.cross_attention_adaln = cross_attention_adaln
        # Optional action conditioning (video modality only). Off by default
        # → bit-equivalent to the base model and loads base checkpoints cleanly.
        self.action_conditioning = action_conditioning
        self.latent_action_dim = latent_action_dim
        self.num_action_per_latent_frame = num_action_per_latent_frame
        # Per-latent-frame action vector width. The embedder input is
        # action_feature_dim * num_action_per_latent_frame. Defaults to latent_action_dim.
        self.action_feature_dim = action_feature_dim if action_feature_dim is not None else latent_action_dim
        self.use_middle_indices_grid = use_middle_indices_grid
        self.rope_type = rope_type
        self.double_precision_rope = double_precision_rope
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.positional_embedding_theta = positional_embedding_theta
        self.model_type = model_type
        cross_pe_max_pos = None
        if model_type.is_video_enabled():
            if positional_embedding_max_pos is None:
                positional_embedding_max_pos = [20, 2048, 2048]
            self.positional_embedding_max_pos = positional_embedding_max_pos
            self.num_attention_heads = num_attention_heads
            self.inner_dim = num_attention_heads * attention_head_dim
            self._init_video(
                in_channels=in_channels,
                out_channels=out_channels,
                norm_eps=norm_eps,
                caption_projection=caption_projection,
            )

        if model_type.is_audio_enabled():
            if audio_positional_embedding_max_pos is None:
                audio_positional_embedding_max_pos = [20]
            self.audio_positional_embedding_max_pos = audio_positional_embedding_max_pos
            self.audio_num_attention_heads = audio_num_attention_heads
            self.audio_inner_dim = self.audio_num_attention_heads * audio_attention_head_dim
            self._init_audio(
                in_channels=audio_in_channels,
                out_channels=audio_out_channels,
                norm_eps=norm_eps,
                caption_projection=audio_caption_projection,
            )

        if model_type.is_video_enabled() and model_type.is_audio_enabled():
            cross_pe_max_pos = max(self.positional_embedding_max_pos[0], self.audio_positional_embedding_max_pos[0])
            self.av_ca_timestep_scale_multiplier = av_ca_timestep_scale_multiplier
            self.audio_cross_attention_dim = audio_cross_attention_dim
            self._init_audio_video(num_scale_shift_values=4)

        self._init_preprocessors(cross_pe_max_pos)
        # Initialize transformer blocks
        self._init_transformer_blocks(
            num_layers=num_layers,
            attention_head_dim=attention_head_dim if model_type.is_video_enabled() else 0,
            cross_attention_dim=cross_attention_dim,
            audio_attention_head_dim=audio_attention_head_dim if model_type.is_audio_enabled() else 0,
            audio_cross_attention_dim=audio_cross_attention_dim,
            norm_eps=norm_eps,
            attention_type=attention_type,
            apply_gated_attention=apply_gated_attention,
        )

    @property
    def _adaln_embedding_coefficient(self) -> int:
        return adaln_embedding_coefficient(self.cross_attention_adaln)

    def _init_video(
        self,
        in_channels: int,
        out_channels: int,
        norm_eps: float,
        caption_projection: torch.nn.Module | None = None,
    ) -> None:
        """Initialize video-specific components."""
        # Video input components
        self.patchify_proj = torch.nn.Linear(in_channels, self.inner_dim, bias=True)
        if caption_projection is not None:
            self.caption_projection = caption_projection

        self.adaln_single = AdaLayerNormSingle(self.inner_dim, embedding_coefficient=self._adaln_embedding_coefficient)

        self.prompt_adaln_single = (
            AdaLayerNormSingle(self.inner_dim, embedding_coefficient=2) if self.cross_attention_adaln else None
        )

        # Dual-path action embedder (video modality). Input width = per-latent-frame
        # action feature size (action_feature_dim * num_action_per_latent_frame).
        # head_main → D (added to the timestep embedding); head_mod → coeff*D (added
        # directly to the AdaLN modulation).
        self.action_embedder = (
            ActionEmbedder(
                action_in=self.action_feature_dim * self.num_action_per_latent_frame,
                embedding_dim=self.inner_dim,
                modulation_dim=self._adaln_embedding_coefficient * self.inner_dim,
            )
            if self.action_conditioning
            else None
        )
        # Optional raw-action → latent-action projection.
        # Built post-hoc via ``enable_action_input_projection`` (the checkpoint
        # configurator does not know about it), so it starts as None here and is
        # wired into the video preprocessor when enabled.
        self.action_input_proj = None

        # Video output components
        self.scale_shift_table = torch.nn.Parameter(torch.empty(2, self.inner_dim))
        self.norm_out = torch.nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=norm_eps)
        self.proj_out = torch.nn.Linear(self.inner_dim, out_channels)

    def _init_audio(
        self,
        in_channels: int,
        out_channels: int,
        norm_eps: float,
        caption_projection: torch.nn.Module | None = None,
    ) -> None:
        """Initialize audio-specific components."""

        # Audio input components
        self.audio_patchify_proj = torch.nn.Linear(in_channels, self.audio_inner_dim, bias=True)
        if caption_projection is not None:
            self.audio_caption_projection = caption_projection

        self.audio_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=self._adaln_embedding_coefficient,
        )

        self.audio_prompt_adaln_single = (
            AdaLayerNormSingle(self.audio_inner_dim, embedding_coefficient=2) if self.cross_attention_adaln else None
        )

        # Audio output components
        self.audio_scale_shift_table = torch.nn.Parameter(torch.empty(2, self.audio_inner_dim))
        self.audio_norm_out = torch.nn.LayerNorm(self.audio_inner_dim, elementwise_affine=False, eps=norm_eps)
        self.audio_proj_out = torch.nn.Linear(self.audio_inner_dim, out_channels)

    def _init_audio_video(
        self,
        num_scale_shift_values: int,
    ) -> None:
        """Initialize audio-video cross-attention components."""
        self.av_ca_video_scale_shift_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            embedding_coefficient=num_scale_shift_values,
        )

        self.av_ca_audio_scale_shift_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=num_scale_shift_values,
        )

        self.av_ca_a2v_gate_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            embedding_coefficient=1,
        )

        self.av_ca_v2a_gate_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=1,
        )

    def _init_preprocessors(
        self,
        cross_pe_max_pos: int | None = None,
    ) -> None:
        """Initialize preprocessors for LTX."""

        if self.model_type.is_video_enabled() and self.model_type.is_audio_enabled():
            self.video_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                patchify_proj=self.patchify_proj,
                adaln=self.adaln_single,
                cross_scale_shift_adaln=self.av_ca_video_scale_shift_adaln_single,
                cross_gate_adaln=self.av_ca_a2v_gate_adaln_single,
                inner_dim=self.inner_dim,
                max_pos=self.positional_embedding_max_pos,
                num_attention_heads=self.num_attention_heads,
                cross_pe_max_pos=cross_pe_max_pos,
                use_middle_indices_grid=self.use_middle_indices_grid,
                audio_cross_attention_dim=self.audio_cross_attention_dim,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
                caption_projection=getattr(self, "caption_projection", None),
                prompt_adaln=getattr(self, "prompt_adaln_single", None),
                action_embedder=getattr(self, "action_embedder", None),
                action_input_proj=getattr(self, "action_input_proj", None),
            )
            self.audio_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                cross_scale_shift_adaln=self.av_ca_audio_scale_shift_adaln_single,
                cross_gate_adaln=self.av_ca_v2a_gate_adaln_single,
                inner_dim=self.audio_inner_dim,
                max_pos=self.audio_positional_embedding_max_pos,
                num_attention_heads=self.audio_num_attention_heads,
                cross_pe_max_pos=cross_pe_max_pos,
                use_middle_indices_grid=self.use_middle_indices_grid,
                audio_cross_attention_dim=self.audio_cross_attention_dim,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
                caption_projection=getattr(self, "audio_caption_projection", None),
                prompt_adaln=getattr(self, "audio_prompt_adaln_single", None),
            )
        elif self.model_type.is_video_enabled():
            self.video_args_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.patchify_proj,
                adaln=self.adaln_single,
                inner_dim=self.inner_dim,
                max_pos=self.positional_embedding_max_pos,
                num_attention_heads=self.num_attention_heads,
                use_middle_indices_grid=self.use_middle_indices_grid,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                caption_projection=getattr(self, "caption_projection", None),
                prompt_adaln=getattr(self, "prompt_adaln_single", None),
                action_embedder=getattr(self, "action_embedder", None),
                action_input_proj=getattr(self, "action_input_proj", None),
            )
        elif self.model_type.is_audio_enabled():
            self.audio_args_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                inner_dim=self.audio_inner_dim,
                max_pos=self.audio_positional_embedding_max_pos,
                num_attention_heads=self.audio_num_attention_heads,
                use_middle_indices_grid=self.use_middle_indices_grid,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                caption_projection=getattr(self, "audio_caption_projection", None),
                prompt_adaln=getattr(self, "audio_prompt_adaln_single", None),
            )

    def _init_transformer_blocks(
        self,
        num_layers: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        audio_attention_head_dim: int,
        audio_cross_attention_dim: int,
        norm_eps: float,
        attention_type: AttentionFunction | AttentionCallable,
        apply_gated_attention: bool,
    ) -> None:
        """Initialize transformer blocks for LTX."""
        video_config = (
            TransformerConfig(
                dim=self.inner_dim,
                heads=self.num_attention_heads,
                d_head=attention_head_dim,
                context_dim=cross_attention_dim,
                apply_gated_attention=apply_gated_attention,
                cross_attention_adaln=self.cross_attention_adaln,
            )
            if self.model_type.is_video_enabled()
            else None
        )
        audio_config = (
            TransformerConfig(
                dim=self.audio_inner_dim,
                heads=self.audio_num_attention_heads,
                d_head=audio_attention_head_dim,
                context_dim=audio_cross_attention_dim,
                apply_gated_attention=apply_gated_attention,
                cross_attention_adaln=self.cross_attention_adaln,
            )
            if self.model_type.is_audio_enabled()
            else None
        )
        self.transformer_blocks = torch.nn.ModuleList(
            [
                BasicAVTransformerBlock(
                    idx=idx,
                    video=video_config,
                    audio=audio_config,
                    rope_type=self.rope_type,
                    norm_eps=norm_eps,
                    attention_function=attention_type,
                )
                for idx in range(num_layers)
            ]
        )

    def set_gradient_checkpointing(self, enable: bool) -> None:
        """Enable or disable gradient checkpointing for transformer blocks.
        Gradient checkpointing trades compute for memory by recomputing activations
        during the backward pass instead of storing them. This can significantly
        reduce memory usage at the cost of ~20-30% slower training.
        Args:
            enable: Whether to enable gradient checkpointing
        """
        self._enable_gradient_checkpointing = enable

    def _process_transformer_blocks(
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
        perturbations: BatchedPerturbationConfig,
        stream_slices: tuple[int, ...] | None = None,
        stream_self_only: bool = False,
        stream_ids: tuple[int, ...] | None = None,
        stream_obs_len: int | None = None,
        stream_stages: tuple[int, ...] | None = None,
        invalid_key_mask: torch.Tensor | None = None,
    ) -> tuple[TransformerArgs, TransformerArgs]:
        """Process transformer blocks for LTXAV."""

        # Process transformer blocks
        for block in self.transformer_blocks:
            if self._enable_gradient_checkpointing and self.training:
                # Use gradient checkpointing to save memory during training.
                # With use_reentrant=False, we can pass dataclasses directly -
                # PyTorch will track all tensor leaves in the computation graph.
                video, audio = torch.utils.checkpoint.checkpoint(
                    block,
                    video,
                    audio,
                    perturbations,
                    stream_slices,
                    stream_self_only,
                    stream_ids,
                    stream_obs_len,
                    stream_stages,
                    invalid_key_mask,
                    use_reentrant=False,
                )
            else:
                video, audio = block(
                    video=video,
                    audio=audio,
                    perturbations=perturbations,
                    stream_slices=stream_slices,
                    stream_self_only=stream_self_only,
                    stream_ids=stream_ids,
                    stream_obs_len=stream_obs_len,
                    stream_stages=stream_stages,
                    invalid_key_mask=invalid_key_mask,
                )

        return video, audio

    def _process_output(
        self,
        scale_shift_table: torch.Tensor,
        norm_out: torch.nn.LayerNorm,
        proj_out: torch.nn.Linear,
        x: torch.Tensor,
        embedded_timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Process output for LTXV."""
        # Apply scale-shift modulation
        scale_shift_values = (
            scale_shift_table[None, None].to(device=x.device, dtype=x.dtype) + embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

        x = norm_out(x)
        x = x * (1 + scale) + shift
        x = proj_out(x)
        return x

    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: BatchedPerturbationConfig,
        icl_stream_slices: tuple[int, ...] | None = None,
        icl_aux_self_only: bool = False,
        icl_stream_ids: tuple[int, ...] | None = None,
        icl_stream_obs_len: int | None = None,
        icl_stream_stages: tuple[int, ...] | None = None,
        icl_invalid_key_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for LTX models.

        ``icl_stream_stages`` / ``icl_invalid_key_mask``: stage id per aux stream
        for the stage-ordered strict attention, and an optional ``(B, T)`` bool key
        mask of invalid tokens. Both default to None.

        ``icl_stream_slices`` — optional ``(n_video, n_aux1, n_aux2, ...)`` segment
        lengths marking in-context auxiliary streams (e.g. optical-flow and pointmap
        videos) concatenated after the video tokens in ``video.latent``.
        Requires ``enable_icl_aux_streams``. Effects: per-stream learnable
        embeddings are added to the aux segments, and self-attention becomes
        asymmetric (video queries see video kv only — bitwise-identical to the
        no-aux forward; each aux stream sees video + itself). The prediction for
        aux tokens comes from the shared video output head.

        Returns:
            Processed output tensors
        """
        if not self.model_type.is_video_enabled() and video is not None:
            raise ValueError("Video is not enabled for this model")
        if not self.model_type.is_audio_enabled() and audio is not None:
            raise ValueError("Audio is not enabled for this model")

        modality_embed = None
        if icl_stream_slices is not None:
            embs = getattr(self, "icl_stream_embeddings", None)
            if embs is None:
                raise ValueError("icl_stream_slices given but enable_icl_aux_streams was not called")
            if audio is not None:
                raise ValueError("ICL aux streams are video-only (audio must be None)")
            if video is None:
                raise ValueError("ICL aux streams require the video modality")
            n_aux = len(icl_stream_slices) - 1
            if icl_stream_ids is None:
                icl_stream_ids = tuple(range(n_aux))
            if len(icl_stream_ids) != n_aux:
                raise ValueError(
                    f"{n_aux} aux slices vs {len(icl_stream_ids)} stream ids"
                )
            if n_aux and max(icl_stream_ids) >= embs.shape[0]:
                raise ValueError(
                    f"stream id {max(icl_stream_ids)} out of range for {embs.shape[0]} enabled embeddings"
                )
            if sum(icl_stream_slices) != video.latent.shape[1]:
                raise ValueError(
                    f"stream slices {icl_stream_slices} do not sum to seq len {video.latent.shape[1]}"
                )
            # Modality AdaLN: per-token identity (row 0 = video, row 1+sid = aux
            # stream) added to the timestep embedding of EVERY token, so the identity reaches
            # all 48 blocks' modulation and the output modulation. The tag follows the token's
            # stream, not the stage being generated.
            mod_table = getattr(self, "icl_modality_adaln", None)
            if mod_table is not None:
                rows = [0] * icl_stream_slices[0]
                for sid, ln in zip(icl_stream_ids, icl_stream_slices[1:]):
                    rows += [1 + sid] * ln
                index = torch.tensor(rows, device=video.latent.device)
                modality_embed = mod_table[index].to(dtype=video.latent.dtype).unsqueeze(0).expand(
                    video.latent.shape[0], -1, -1
                )

        video_args = (
            self.video_args_preprocessor.prepare(video, audio, extra_timestep_embed=modality_embed)
            if video is not None
            else None
        )
        audio_args = self.audio_args_preprocessor.prepare(audio, video) if audio is not None else None

        if icl_stream_slices is not None:
            x = video_args.x
            in_heads = getattr(self, "icl_stream_patchify_proj", None)
            segs, off = [x[:, : icl_stream_slices[0]]], icl_stream_slices[0]
            for sid, ln in zip(icl_stream_ids, icl_stream_slices[1:]):
                if in_heads is not None:
                    # independent input projection per aux stream
                    seg = in_heads[sid](video.latent[:, off : off + ln]).to(dtype=x.dtype)
                else:
                    seg = x[:, off : off + ln]
                segs.append(seg + embs[sid].to(dtype=x.dtype))
                off += ln
            video_args = replace(video_args, x=torch.cat(segs, dim=1))

        # Process transformer blocks
        video_out, audio_out = self._process_transformer_blocks(
            video=video_args,
            audio=audio_args,
            perturbations=perturbations,
            stream_slices=icl_stream_slices,
            stream_self_only=icl_aux_self_only,
            stream_ids=icl_stream_ids,
            stream_obs_len=icl_stream_obs_len,
            stream_stages=icl_stream_stages,
            invalid_key_mask=icl_invalid_key_mask,
        )

        # Process output
        out_heads = getattr(self, "icl_stream_proj_out", None)
        if video_out is not None and icl_stream_slices is not None and out_heads is not None:
            # independent output head (scale/shift table + proj_out) per aux stream; the
            # video segment keeps the original head. norm_out has no affine params.
            x_out, emb_t = video_out.x, video_out.embedded_timestep
            n_video = icl_stream_slices[0]
            per_token_emb = emb_t.shape[1] != 1
            pieces = [
                self._process_output(
                    self.scale_shift_table, self.norm_out, self.proj_out,
                    x_out[:, :n_video], emb_t[:, :n_video] if per_token_emb else emb_t,
                )
            ]
            off = n_video
            for sid, ln in zip(icl_stream_ids, icl_stream_slices[1:]):
                pieces.append(
                    self._process_output(
                        self.icl_stream_scale_shift_table[sid], self.norm_out, out_heads[sid],
                        x_out[:, off : off + ln], emb_t[:, off : off + ln] if per_token_emb else emb_t,
                    )
                )
                off += ln
            vx = torch.cat(pieces, dim=1)
        else:
            vx = (
                self._process_output(
                    self.scale_shift_table, self.norm_out, self.proj_out, video_out.x, video_out.embedded_timestep
                )
                if video_out is not None
                else None
            )
        ax = (
            self._process_output(
                self.audio_scale_shift_table,
                self.audio_norm_out,
                self.audio_proj_out,
                audio_out.x,
                audio_out.embedded_timestep,
            )
            if audio_out is not None
            else None
        )
        return vx, ax

    def enable_icl_aux_streams(
        self,
        num_streams: int = 2,
        gated_video_cross: bool = False,
        chain: bool = False,
        modality_heads: bool = False,
        modality_adaln: bool = False,
        modality_adaln_init_std: float = 0.02,
    ) -> torch.nn.Parameter:
        """Enable in-context auxiliary streams for staged CoT generation.

        Creates one embedding per aux stream, added to that stream's token
        segment right after the patchify projection. Stream order in
        ``icl_stream_slices[1:]`` must follow generation order, e.g.
        ``(flow, pointmap)``.

        When ``gated_video_cross`` is enabled, future RGB queries add a
        per-block tanh-gated cross-attention term from each stream. ``chain``
        lets each later stream attend all earlier streams.
        """
        if not self.model_type.is_video_enabled():
            raise ValueError("ICL aux streams require the video modality")
        self.icl_stream_embeddings = torch.nn.Parameter(torch.zeros(num_streams, self.inner_dim))
        for block in self.transformer_blocks:
            if gated_video_cross:
                block.attn1.icl_stream_gates = torch.nn.Parameter(torch.zeros(num_streams))
            block.attn1.icl_stream_chain = chain
        if modality_heads:
            # Independent input projection + output head (scale/shift table + proj_out) per
            # aux stream, with the same shapes as the video ones (separate parameter objects).
            self.icl_stream_patchify_proj = torch.nn.ModuleList(
                [copy.deepcopy(self.patchify_proj) for _ in range(num_streams)]
            )
            self.icl_stream_proj_out = torch.nn.ModuleList([copy.deepcopy(self.proj_out) for _ in range(num_streams)])
            self.icl_stream_scale_shift_table = torch.nn.Parameter(
                self.scale_shift_table.detach().clone().unsqueeze(0).repeat(num_streams, 1, 1)
            )
        if modality_adaln:
            # Modality AdaLN table: row 0 = video, rows 1..n = aux streams.
            table = torch.zeros(num_streams + 1, self.inner_dim)
            generator = torch.Generator(device="cpu").manual_seed(20260909)
            table[1:] = torch.randn(num_streams, self.inner_dim, generator=generator) * modality_adaln_init_std
            self.icl_modality_adaln = torch.nn.Parameter(table)
        return self.icl_stream_embeddings

    def icl_modality_heads_enabled(self) -> bool:
        return getattr(self, "icl_stream_proj_out", None) is not None

    def icl_modality_adaln_enabled(self) -> bool:
        return getattr(self, "icl_modality_adaln", None) is not None

    @torch.no_grad()
    def init_icl_modality_heads_from_video(self) -> None:
        """Copy the video input projection / output head into every aux-stream head."""
        if not self.icl_modality_heads_enabled():
            return
        for head in self.icl_stream_patchify_proj:
            head.weight.copy_(self.patchify_proj.weight)
            head.bias.copy_(self.patchify_proj.bias)
        for head in self.icl_stream_proj_out:
            head.weight.copy_(self.proj_out.weight)
            head.bias.copy_(self.proj_out.bias)
        for i in range(self.icl_stream_scale_shift_table.shape[0]):
            self.icl_stream_scale_shift_table[i].copy_(self.scale_shift_table)

    @torch.no_grad()
    def icl_modality_scale_stats(self, embedded_timestep: torch.Tensor) -> dict[str, float]:
        """RMS of the modality AdaLN rows and of the given timestep embedding."""
        table = getattr(self, "icl_modality_adaln", None)
        if table is None:
            return {}
        emb_rms = float(embedded_timestep.detach().float().pow(2).mean().sqrt())
        return {
            "timestep_embed_rms": emb_rms,
            **{f"modality_row{i}_rms": float(table[i].float().pow(2).mean().sqrt()) for i in range(table.shape[0])},
        }

    def icl_gate_values(self) -> torch.Tensor | None:
        """Return tanh gate values as ``(num_blocks, num_streams)``."""
        values = [
            torch.tanh(block.attn1.icl_stream_gates.detach().float())
            for block in self.transformer_blocks
            if getattr(block.attn1, "icl_stream_gates", None) is not None
        ]
        return torch.stack(values) if values else None

    def enable_action_conditioning(
        self,
        latent_action_dim: int = 32,
        num_action_per_latent_frame: int = 8,
        action_feature_dim: int | None = None,
    ) -> "ActionEmbedder":
        """Attach an action embedder to an already-built model.

        Used when the model is constructed by the checkpoint configurator (which
        doesn't know about action conditioning) and action support is added
        post-hoc. Builds the embedder, registers it on the module, and wires it
        into the video args preprocessor's action path. Re-enabling rebuilds the
        embedder. Video modality must be enabled.

        ``action_feature_dim`` is the per-latent-frame action vector width; the
        embedder input is ``action_feature_dim * num_action_per_latent_frame``.
        Defaults to ``latent_action_dim``. It must match the width of the action
        features fed through ``Modality.action``.

        Returns the new embedder (so the caller can place it on the right device).
        """
        if not self.model_type.is_video_enabled():
            raise ValueError("action conditioning requires the video modality")
        self.action_conditioning = True
        self.latent_action_dim = latent_action_dim
        self.num_action_per_latent_frame = num_action_per_latent_frame
        self.action_feature_dim = action_feature_dim if action_feature_dim is not None else latent_action_dim
        self.action_embedder = ActionEmbedder(
            action_in=self.action_feature_dim * num_action_per_latent_frame,
            embedding_dim=self.inner_dim,
            modulation_dim=self._adaln_embedding_coefficient * self.inner_dim,
        )
        # Wire into the video preprocessor. MultiModal wraps a simple_preprocessor.
        prep = self.video_args_preprocessor
        if isinstance(prep, MultiModalTransformerArgsPreprocessor):
            prep.simple_preprocessor.action_embedder = self.action_embedder
        else:
            prep.action_embedder = self.action_embedder
        return self.action_embedder

    def enable_action_input_projection(
        self,
        raw_action_dim: int = 22,
        latent_action_dim: int = 32,
        num_action_per_latent_frame: int = 8,
        net_spec: dict | None = None,
    ) -> tuple["ActionEmbedder", "ActionInputProjection"]:
        """Attach raw-action conditioning to a built model.

        Two-stage action path:
          1. an :class:`ActionInputProjection` maps the raw per-pair
             robot action (``raw_action_dim``) to the latent action width
             (``latent_action_dim``);
          2. the standard :class:`ActionEmbedder` (input width
             ``latent_action_dim * num_action_per_latent_frame``) then injects it.

        ``Modality.action`` is therefore the raw action,
        shaped ``(B, T_lat, num_action_per_latent_frame * raw_action_dim)``; the
        preprocessor projects per pair before the embedder. Video modality only.
        """
        embedder = self.enable_action_conditioning(
            latent_action_dim=latent_action_dim,
            num_action_per_latent_frame=num_action_per_latent_frame,
            action_feature_dim=latent_action_dim,
        )
        self.action_input_proj = ActionInputProjection(
            action_in=raw_action_dim,
            out_dim=latent_action_dim,
            net_spec=net_spec,
        )
        prep = self.video_args_preprocessor
        if isinstance(prep, MultiModalTransformerArgsPreprocessor):
            prep.simple_preprocessor.action_input_proj = self.action_input_proj
        else:
            prep.action_input_proj = self.action_input_proj
        return embedder, self.action_input_proj


class LegacyX0Model(torch.nn.Module):
    """
    Legacy X0 model implementation.
    Returns fully denoised output based on the velocities produced by the base model.
    """

    def __init__(self, velocity_model: LTXModel):
        super().__init__()
        self.velocity_model = velocity_model

    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: BatchedPerturbationConfig,
        sigma: float,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Denoise the video and audio according to the sigma.
        Returns:
            Denoised video and audio
        """
        vx, ax = self.velocity_model(video, audio, perturbations)
        denoised_video = to_denoised(video.latent, vx, sigma) if vx is not None else None
        denoised_audio = to_denoised(audio.latent, ax, sigma) if ax is not None else None
        return denoised_video, denoised_audio


class X0Model(torch.nn.Module):
    """
    X0 model implementation.
    Returns fully denoised outputs based on the velocities produced by the base model.
    Applies scaled denoising to the video and audio according to the timesteps = sigma * denoising_mask.
    """

    def __init__(self, velocity_model: LTXModel):
        super().__init__()
        self.velocity_model = velocity_model

    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: BatchedPerturbationConfig,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Denoise the video and audio according to the sigma.
        Returns:
            Denoised video and audio
        """
        vx, ax = self.velocity_model(video, audio, perturbations)
        denoised_video = to_denoised(video.latent, vx, video.timesteps) if vx is not None else None
        denoised_audio = to_denoised(audio.latent, ax, audio.timesteps) if ax is not None else None
        return denoised_video, denoised_audio
