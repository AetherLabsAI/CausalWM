"""Deployable full-scene flow -> pointmap -> RGB sampling, with no geometry input.

The packed order is ``[video | flow | pointmap]``; generation order is flow,
pointmap, video. Only the observed RGB latent frame is required. Pointmap frame
zero is always generated, never replaced by an independently inferred pointmap.
The required flow-zero latent must be the VAE encoding of a fixed zero-flow
image. No ground-truth flow, pointmap or future frame is accepted.

Strict attention makes observation tokens read only observation tokens and
auxiliary streams read only observation plus same/earlier auxiliary stages.
This blocks indirect future-RGB and pointmap-to-flow leakage across layers.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Callable

import torch

from ltx_core.guidance.perturbations import (
    BatchedPerturbationConfig,
    Perturbation,
    PerturbationConfig,
    PerturbationType,
)
from ltx_core.model.transformer.modality import Modality

Context = tuple[torch.Tensor, torch.Tensor | None]
STAGE_ORDER = ("flow", "pointmap", "video")
LATENT_CHANNELS = 128


@dataclass(frozen=True, slots=True)
class FullSceneInputs:
    """Exactly one RGB latent frame and a required deterministic flow sentinel.

    Both tensors are patchified ``(1, H_lat * W_lat, 128)``. No pointmap,
    mask, future RGB or future auxiliary tensor is accepted.
    ``flow0`` must be computed from a constant zero-flow image in isolation.
    """

    rgb0: torch.Tensor
    flow0: torch.Tensor


def _validated_schedule(values: tuple[float, ...], name: str) -> tuple[float, ...]:
    if not isinstance(values, (tuple, list)) or len(values) < 2:
        raise ValueError(f"{name} schedule needs at least two values")
    if any(isinstance(value, bool) or not isinstance(value, (float, int)) for value in values):
        raise ValueError(f"{name} schedule must contain real scalar numbers")
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} schedule must be finite")
    # LTX2Scheduler.execute() returns sigma[0] = 0.99999994 (fp32 of the shifted
    # schedule), never an exact 1.0. Snap a sub-1e-5 deviation to the pure-noise
    # endpoint instead of rejecting the real production scheduler.
    if abs(result[0] - 1.0) <= 1e-5 and result[0] != 1.0:
        result = (1.0,) + result[1:]
    if result[0] != 1.0 or result[-1] != 0.0 or any(a <= b for a, b in zip(result, result[1:])):
        raise ValueError(f"{name} schedule must strictly decrease from 1 to 0")
    return result


@dataclass(frozen=True, slots=True)
class FullSceneSchedules:
    flow: tuple[float, ...]
    pointmap: tuple[float, ...]
    video: tuple[float, ...]

    def __post_init__(self) -> None:
        for name in STAGE_ORDER:
            object.__setattr__(self, name, _validated_schedule(getattr(self, name), name))


@dataclass(frozen=True, slots=True)
class FullSceneSequence:
    """Packed-sequence metadata: segment order, lengths and stage ids."""

    grid: tuple[int, int, int]

    def __post_init__(self) -> None:
        if len(self.grid) != 3 or any(type(n) is not int or n <= 0 for n in self.grid):
            raise ValueError("grid must contain three positive integers (latent frames, height, width)")

    @property
    def names(self) -> tuple[str, ...]:
        return ("video", "flow", "pointmap")

    @property
    def spatial(self) -> int:
        return self.grid[1] * self.grid[2]

    @property
    def seg(self) -> int:
        return self.grid[0] * self.spatial

    @property
    def total(self) -> int:
        return 3 * self.seg

    @property
    def slices(self) -> tuple[int, ...]:
        return (self.seg, self.seg, self.seg)

    @property
    def stream_ids(self) -> tuple[int, ...]:
        return (0, 1)

    @property
    def stream_stages(self) -> tuple[int, ...]:
        return (0, 1)

    def segment(self, name: str) -> slice:
        start = self.names.index(name) * self.seg
        return slice(start, start + self.seg)

    def frame0(self, name: str) -> slice:
        start = self.names.index(name) * self.seg
        return slice(start, start + self.spatial)


@dataclass(frozen=True, slots=True)
class FullSceneResult:
    latents: torch.Tensor
    token_sigmas: torch.Tensor
    seq: FullSceneSequence
    provenance: dict[str, object]


def stage_generator(seed: int, stage: str, device: torch.device | str) -> torch.Generator:
    """Independent noise streams; schedule changes never consume another stage's RNG."""
    if stage not in STAGE_ORDER:
        raise ValueError(f"unknown full-scene stage {stage!r}")
    digest = hashlib.sha256(f"full_scene_cot_v1|{seed}|{stage}".encode()).digest()
    generator = torch.Generator(device=device)
    generator.manual_seed(int.from_bytes(digest[:8], "little") % (2**62))
    return generator


def _check_frame(tensor: torch.Tensor, name: str, spatial: int) -> None:
    if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != (1, spatial, LATENT_CHANNELS):
        raise ValueError(f"{name} must contain exactly one latent frame: (1, {spatial}, {LATENT_CHANNELS})")
    if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain finite floating-point latents")


def _prepare_context(context: Context, device: torch.device, dtype: torch.dtype, name: str) -> Context:
    if not isinstance(context, tuple) or len(context) != 2:
        raise ValueError(f"{name} must be (embeddings, attention_mask)")
    embeds, mask = context
    if not isinstance(embeds, torch.Tensor) or embeds.ndim != 3 or embeds.shape[0] != 1:
        raise ValueError(f"{name} embeddings must have shape (1, text_tokens, channels)")
    if min(embeds.shape[1:]) <= 0 or not embeds.is_floating_point() or not bool(torch.isfinite(embeds).all()):
        raise ValueError(f"{name} embeddings must be nonempty and finite floating point")
    if mask is not None:
        if not isinstance(mask, torch.Tensor) or tuple(mask.shape) not in ((1, embeds.shape[1]), (1, 1, 1, embeds.shape[1])):
            raise ValueError(f"{name} attention mask must be (1, text_tokens) or (1, 1, 1, text_tokens)")
        if mask.is_floating_point():
            if bool(torch.isnan(mask).any()) or bool((mask > 0).any()) or not bool((mask == 0).any()):
                raise ValueError(f"{name} floating mask must be additive (zero visible, nonpositive masked)")
            mask = mask.to(device=device, dtype=dtype).reshape(1, 1, 1, -1)
        else:
            if not bool(((mask == 0) | (mask == 1)).all()) or not bool((mask == 1).any()):
                raise ValueError(f"{name} binary mask needs only 0/1 and at least one visible token")
            mask = (mask.to(device=device, dtype=dtype) - 1).reshape(1, 1, 1, -1) * torch.finfo(dtype).max
    return embeds.to(device=device, dtype=dtype), mask


def _velocity(
    model: torch.nn.Module,
    seq: FullSceneSequence,
    latents: torch.Tensor,
    token_sigmas: torch.Tensor,
    batch_sigma: float,
    context: Context,
    positions: torch.Tensor,
    dtype: torch.dtype,
    perturbations: BatchedPerturbationConfig | None = None,
) -> torch.Tensor:
    embeds, mask = context
    modality = Modality(
        enabled=True,
        latent=latents.to(dtype),
        sigma=torch.full((1,), batch_sigma, device=latents.device, dtype=dtype),
        timesteps=token_sigmas.unsqueeze(0).to(dtype),
        positions=positions,
        context=embeds,
        context_mask=mask,
        action=None,
        icl_stream_slices=seq.slices,
    )
    output, _ = model(
        video=modality,
        audio=None,
        perturbations=perturbations if perturbations is not None else BatchedPerturbationConfig.empty(1),
        icl_stream_slices=seq.slices,
        icl_stream_ids=seq.stream_ids,
        icl_stream_obs_len=seq.spatial,
        icl_stream_stages=seq.stream_stages,
    )
    if output.shape != latents.shape or not bool(torch.isfinite(output).all()):
        raise RuntimeError("full-scene model returned invalid velocity shape or nonfinite values")
    return output.float()


@torch.inference_mode()
def run_full_scene_cot(
    model: torch.nn.Module,
    grid: tuple[int, int, int],
    inputs: FullSceneInputs,
    context: Context,
    positions_fn: Callable[[FullSceneSequence], torch.Tensor],
    schedules: FullSceneSchedules,
    *,
    seed: int,
    device: torch.device | str,
    dtype: torch.dtype,
    guidance: float = 1.0,
    negative_context: Context | None = None,
    stg_scale: float = 0.0,
    stg_blocks: tuple[int, ...] | None = (29,),
    stg_stages: tuple[str, ...] = ("video",),
) -> FullSceneResult:
    """Generate full flow/pointmap and future RGB from RGB0 plus text context.

    Every pointmap token, including frame zero, starts at sigma=1 and is
    updated in the pointmap stage. Observed RGB is held bitwise unchanged at
    every Euler step; completed auxiliary latents are frozen downstream.
    ``model`` must be in eval mode with exactly two enabled auxiliary streams
    in the full-scene checkpoint semantics (flow, pointmap).
    """
    if not isinstance(inputs, FullSceneInputs) or not isinstance(schedules, FullSceneSchedules):
        raise TypeError("use FullSceneInputs and FullSceneSchedules, not plain dictionaries")
    if model.training:
        raise ValueError("full-scene sampling requires model.eval()")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("dtype must be float32, float16 or bfloat16")
    if isinstance(guidance, bool) or not isinstance(guidance, (int, float)) or not math.isfinite(guidance) or guidance < 0:
        raise ValueError("guidance must be a finite nonnegative scalar")
    if guidance != 1.0 and negative_context is None:
        raise ValueError("guidance != 1 requires negative_context; CFG must not silently be skipped")
    if isinstance(stg_scale, bool) or not isinstance(stg_scale, (int, float)) or not math.isfinite(stg_scale) or stg_scale < 0:
        raise ValueError("stg_scale must be a finite nonnegative scalar")
    stg_stages = tuple(stg_stages)
    if any(stage not in STAGE_ORDER for stage in stg_stages):
        raise ValueError(f"stg_stages must be a subset of {STAGE_ORDER}")
    if stg_blocks is not None and (len(stg_blocks) == 0 or any(type(b) is not int or b < 0 for b in stg_blocks)):
        raise ValueError("stg_blocks must be None (all blocks) or a non-empty tuple of nonnegative ints")
    # STG (spatiotemporal guidance, LTX default 1.0 @ block 29): the perturbed pass runs the listed blocks'
    # video self-attention as a value passthrough; the delta is applied in velocity space, where it is the
    # same linear correction as on x0 (x0 = x - sigma * v).
    stg_perturbations = None
    if stg_scale != 0.0:
        stg_perturbations = BatchedPerturbationConfig(perturbations=[PerturbationConfig(perturbations=[
            Perturbation(type=PerturbationType.SKIP_VIDEO_SELF_ATTN, blocks=None if stg_blocks is None else list(stg_blocks)),
        ])])
    seq = FullSceneSequence(grid)
    _check_frame(inputs.rgb0, "rgb0", seq.spatial)
    _check_frame(inputs.flow0, "flow0", seq.spatial)
    sampling_device = torch.device(device)
    context = _prepare_context(context, sampling_device, dtype, "context")
    if negative_context is not None:
        negative_context = _prepare_context(negative_context, sampling_device, dtype, "negative_context")
        if negative_context[0].shape[-1] != context[0].shape[-1]:
            raise ValueError("positive and negative context feature widths must match")
    positions = positions_fn(seq)
    if not isinstance(positions, torch.Tensor) or tuple(positions.shape) != (1, 3, seq.total, 2):
        raise ValueError(f"positions must have shape (1, 3, {seq.total}, 2)")
    if not positions.is_floating_point() or not bool(torch.isfinite(positions).all()):
        raise ValueError("positions must be finite floating point")
    positions = positions.to(device=sampling_device)

    latents = torch.empty((1, seq.total, LATENT_CHANNELS), device=sampling_device, dtype=torch.float32)
    for stage in STAGE_ORDER:
        latents[:, seq.segment(stage)] = torch.randn(
            (1, seq.seg, LATENT_CHANNELS), generator=stage_generator(seed, stage, sampling_device), device=sampling_device,
        )
    token_sigmas = torch.ones(seq.total, device=sampling_device)
    latents[:, seq.frame0("video")] = inputs.rgb0.to(latents)
    token_sigmas[seq.frame0("video")] = 0.0
    latents[:, seq.frame0("flow")] = inputs.flow0.to(latents)
    token_sigmas[seq.frame0("flow")] = 0.0

    for stage in STAGE_ORDER:
        schedule = getattr(schedules, stage)
        active = torch.zeros(seq.total, device=sampling_device, dtype=torch.bool)
        active[seq.segment(stage)] = True
        active &= token_sigmas > 0
        for sigma, next_sigma in zip(schedule, schedule[1:]):
            # Auxiliary stages keep the observation-only prompt sigma; the final
            # video stage follows its denoising schedule.
            batch_sigma = sigma if stage == "video" else 1.0
            conditioned = _velocity(model, seq, latents, token_sigmas, batch_sigma, context, positions, dtype)
            velocity = conditioned
            if guidance != 1.0:
                unconditioned = _velocity(
                    model, seq, latents, token_sigmas, batch_sigma, negative_context, positions, dtype,
                )
                velocity = unconditioned + guidance * (conditioned - unconditioned)
            if stg_perturbations is not None and stage in stg_stages:
                perturbed = _velocity(
                    model, seq, latents, token_sigmas, batch_sigma, context, positions, dtype, stg_perturbations,
                )
                velocity = velocity + stg_scale * (conditioned - perturbed)
            # Do not even perform arithmetic on observation/frozen tokens.
            latents[:, active] += (next_sigma - sigma) * velocity[:, active]
            token_sigmas[active] = next_sigma
            if not bool(torch.isfinite(latents[:, active]).all()):
                raise RuntimeError(f"nonfinite latents after {stage} denoising step")

    return FullSceneResult(
        latents=latents,
        token_sigmas=token_sigmas,
        seq=seq,
        provenance={
            "mode": "self_full_scene_cot_v1",
            "seed": seed,
            "stage_order": STAGE_ORDER,
            "stream_names": seq.names,
            "attention": "strict_stage_ordered",
            "external_inputs": ("rgb0", "caption"),
            "pointmap_frame0": "generated",
            "flow_frame0": "deterministic_zero_flow_sentinel",
            "flow_frame0_key": "visible_deterministic_sentinel",
            "guidance": float(guidance),
            "stg": {"scale": float(stg_scale), "blocks": None if stg_blocks is None else list(stg_blocks),
                    "stages": list(stg_stages) if stg_scale != 0.0 else [], "skip": "video_self_attn"},
            "schedules": {stage: getattr(schedules, stage) for stage in STAGE_ORDER},
        },
    )
