#!/usr/bin/env python3
"""CausalWM inference: one RGB image + a caption -> optical flow -> pointmap -> future RGB video.

The only inputs are a single first-frame image and a text prompt. No video, depth,
pointmap, segmentation or geometry estimate is read. A released CausalWM checkpoint is
required; the base LTX-2.3 checkpoint supplies the architecture config, the video VAE
and the text-embedding connector, and Gemma-3-12B is the text encoder.

  python inference.py --image first_frame.png --prompt "The robot arm picks up the red block." \
      --checkpoint CausalWMv1.safetensors --base-ckpt ltx-2.3-22b-dev.safetensors \
      --text-encoder-dir gemma-3-12b --out-dir outputs/demo
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Run from a plain checkout without `pip install -e`.
REPO_ROOT = Path(__file__).resolve().parent
for subdirectory in ("packages/ltx-core/src", "."):
    sys.path.insert(0, str(REPO_ROOT / subdirectory))

if TYPE_CHECKING:
    import torch

    from ltx_core.components.patchifiers import VideoLatentPatchifier
    from causalwm.sampler import Context, FullSceneSequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    parser.add_argument("--image", required=True, type=Path, help="Single local RGB observation image; not a video.")
    parser.add_argument("--prompt", required=True, help="Instruction/caption used verbatim, without enhancement.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Released CausalWM transformer weights (.safetensors).")
    parser.add_argument("--base-ckpt", required=True, type=Path, help="Base LTX-2.3 22B checkpoint: architecture config, video VAE, text connector.")
    parser.add_argument("--text-encoder-dir", required=True, type=Path, help="Local Gemma-3-12B directory.")
    parser.add_argument("--out-dir", required=True, type=Path, help="New or empty directory; existing artifacts are not overwritten.")
    parser.add_argument("--frames", type=int, default=121)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--fps", type=int, default=16, help="Integer output FPS and temporal positional-encoding rate.")
    parser.add_argument("--steps-flow", type=int, default=4, help="Denoising steps of the optical-flow stage.")
    parser.add_argument("--steps-pointmap", type=int, default=4, help="Denoising steps of the pointmap stage.")
    parser.add_argument("--steps-video", type=int, default=4, help="Denoising steps of the RGB stage.")
    parser.add_argument("--guidance", type=float, default=1.0, help="Classifier-free guidance scale; 1 disables CFG.")
    parser.add_argument("--negative-prompt", default="", help="Used only when guidance != 1; no hidden negative prompt.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-raw", action="store_true",
                        help="Also save lossless arrays: rgb_uint8.npz, flow_uv.npz, pointmap_xyz.npz and generated latents.")
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    if args.frames < 1 or args.frames % 8 != 1:
        raise ValueError("frames must be positive and equal to 8k+1")
    if min(args.height, args.width) <= 0 or args.height % 32 or args.width % 32:
        raise ValueError("height and width must be positive multiples of 32")
    if args.fps <= 0:
        raise ValueError("fps must be a positive integer")
    # LTX2Scheduler's default stretched schedule is undefined at a single step.
    if min(args.steps_flow, args.steps_pointmap, args.steps_video) < 2:
        raise ValueError("each LTX2Scheduler stage needs at least 2 denoising steps")
    if not math.isfinite(args.guidance) or args.guidance < 0:
        raise ValueError("guidance must be finite and nonnegative")
    if not args.prompt.strip():
        raise ValueError("prompt must be a nonempty instruction/caption")
    if args.guidance == 1.0 and args.negative_prompt:
        raise ValueError("negative-prompt would be ignored at guidance=1; remove it or enable CFG")


def read_deploy_checkpoint_metadata(checkpoint: Path) -> dict[str, str]:
    """Read only header/small stream tensors before loading any large component.

    Sampling never accepts an untagged base checkpoint or a checkpoint with a
    different stream registry.
    """
    from safetensors import safe_open

    from causalwm.checkpoint import STREAM_KEYS, validate_full_scene_checkpoint
    from causalwm.layout import registry_metadata

    if not checkpoint.is_file() or checkpoint.suffix != ".safetensors":
        raise ValueError("checkpoint must be an existing .safetensors file")
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        for key, expected in registry_metadata().items():
            if metadata.get(key) != expected:
                raise ValueError(f"deployment requires {key}={expected!r}; refusing a base or incompatible checkpoint")
        try:
            flow_clip_px = float(metadata["flow_clip_px"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("checkpoint requires flow_clip_px metadata") from error
        if not math.isfinite(flow_clip_px) or flow_clip_px <= 0:
            raise ValueError("checkpoint flow_clip_px must be finite and positive")
        streams = {key: handle.get_tensor(key) for key in handle.keys() if any(part in key for part in STREAM_KEYS)}
        validate_full_scene_checkpoint(streams, metadata)
        if not any("icl_stream_gates" in key for key in streams):
            raise ValueError("full-scene deployment checkpoint is missing its RGB-to-CoT gate parameters")
    return metadata


def load_rgb_observation(image: Path, height: int, width: int) -> tuple[torch.Tensor, dict[str, Any]]:
    """Decode one image, then resize the full view (no crop) with uint8 bicubic interpolation."""
    import numpy as np
    import torch
    from PIL import Image, ImageOps
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.functional import resize

    with Image.open(image) as source:
        if getattr(source, "n_frames", 1) != 1:
            raise ValueError("--image must contain one observation, not an animated/multiframe file")
        rgb = ImageOps.exif_transpose(source).convert("RGB")
        array = np.array(rgb, dtype=np.uint8, copy=True)
    src_height, src_width = array.shape[:2]
    observation = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    if (src_height, src_width) != (height, width):
        observation = resize(observation, [height, width], interpolation=InterpolationMode.BICUBIC, antialias=True)
    return observation.clamp(0, 255).to(torch.uint8).contiguous(), {
        "mode": "resize_full_frame_bicubic_v1", "source_hw": [src_height, src_width],
        "output_hw": [height, width], "crop": None, "exif_orientation_applied": True,
        "scale_xy": [width / src_width, height / src_height],
    }


def _identity(path: Path) -> dict[str, int]:
    """Record file size without disclosing local paths, names, or timestamps."""
    return {"size_bytes": path.stat().st_size}


def _public_checkpoint_metadata(metadata: dict[str, str]) -> dict[str, str]:
    """Export only validated runtime conventions and recognized public identifiers."""
    from causalwm.layout import registry_metadata

    public = registry_metadata()
    if any(metadata.get(key) != value for key, value in public.items()):
        raise ValueError("checkpoint stream metadata does not match the public registry")
    flow_clip_px = float(metadata["flow_clip_px"])
    if not math.isfinite(flow_clip_px) or flow_clip_px <= 0:
        raise ValueError("checkpoint requires finite positive flow_clip_px")
    public["flow_clip_px"] = str(flow_clip_px)
    for key in ("robot_cot_modality_heads", "robot_cot_modality_adaln"):
        if metadata.get(key) in ("0", "1"):
            public[key] = metadata[key]
    recognized = {
        "model_version": "CausalWMv1",
        "robot_cot_arch": "modality_heads_v1",
    }
    public.update({key: value for key, value in recognized.items() if metadata.get(key) == value})
    return public


def configure_text_padding_mask(embeddings: Any) -> None:
    """Use the explicit padding-mask mode of the text connector, not the base's register default."""
    if getattr(embeddings, "video_connector", None) is None:
        raise ValueError("full-scene inference requires a video text connector")
    embeddings.video_connector.apply_padding_mask = True
    if getattr(embeddings, "audio_connector", None) is not None:
        embeddings.audio_connector.apply_padding_mask = True


def _load_components(args: argparse.Namespace, metadata: dict[str, str]) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from safetensors.torch import load_file

    from causalwm.model_loader import load_embeddings_processor, load_model
    from causalwm.checkpoint import load_full_scene_state
    from causalwm.layout import NUM_STREAMS

    components = load_model(
        checkpoint_path=args.base_ckpt, text_encoder_path=args.text_encoder_dir, device="cpu", dtype=torch.bfloat16,
        with_video_vae_encoder=True, with_video_vae_decoder=True, with_audio_vae_decoder=False,
        with_vocoder=False, with_text_encoder=True,
    )
    embeddings = load_embeddings_processor(checkpoint_path=args.base_ckpt, device="cpu", dtype=torch.bfloat16)
    configure_text_padding_mask(embeddings)
    from causalwm.layout import arch_from_metadata  # noqa: PLC0415

    heads, adaln = arch_from_metadata(metadata)
    components.transformer.enable_icl_aux_streams(
        NUM_STREAMS, gated_video_cross=True, chain=True, modality_heads=heads, modality_adaln=adaln,
    )
    state = load_file(str(args.checkpoint), device="cpu")
    connector_state = {key.removeprefix("_connector_video."): value for key, value in state.items() if key.startswith("_connector_video.")}
    feature_state = {key.removeprefix("_feature_extractor."): value for key, value in state.items() if key.startswith("_feature_extractor.")}
    transformer_state = {key: value for key, value in state.items() if not key.startswith(("_connector_video.", "_feature_extractor."))}
    info = load_full_scene_state(components.transformer, transformer_state, metadata)
    # Load text-connector / feature-extractor weights if the checkpoint carries them.
    if connector_state:
        embeddings.video_connector.load_state_dict(connector_state, strict=True)
    if feature_state:
        if embeddings.feature_extractor is None:
            raise ValueError("checkpoint contains a text feature extractor but the base does not")
        embeddings.feature_extractor.load_state_dict(feature_state, strict=True)
    for model in (components.transformer, components.video_vae_encoder, components.video_vae_decoder, components.text_encoder, embeddings):
        model.eval().requires_grad_(False)
    return components, embeddings, {
        **info, "loaded_connector_tensors": len(connector_state), "loaded_feature_extractor_tensors": len(feature_state),
        "modality_heads": heads, "modality_adaln": adaln,
    }


def encode_text(text_encoder: Any, embeddings: Any, prompt: str) -> Context:
    from ltx_core.text_encoders.gemma import convert_to_additive_mask

    if not getattr(embeddings.video_connector, "apply_padding_mask", False):
        raise ValueError("full-scene inference requires text connector apply_padding_mask=True")
    hidden, attention_mask = text_encoder.encode(prompt)
    video_features, audio_features = embeddings.feature_extractor(hidden, attention_mask, "left")
    context, _, context_mask = embeddings.create_embeddings(
        video_features, audio_features, convert_to_additive_mask(attention_mask, video_features.dtype),
    )
    return context, context_mask


def encode_observation(encoder: Any, patchifier: VideoLatentPatchifier, image: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Only accepts a single frame; there is no full-video encoding entrypoint."""
    import torch

    if image.ndim != 4 or tuple(image.shape[:2]) != (1, 3):
        raise ValueError("observation encoder requires exactly one RGB/constant-flow image")
    value = image.permute(1, 0, 2, 3).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    latent = encoder(value)
    latent = latent[0] if isinstance(latent, (tuple, list)) else latent
    if latent.ndim != 5 or latent.shape[2] != 1:
        raise ValueError("single observation produced more than one latent frame")
    return patchifier.patchify(latent).float()


def decode_tokens(decoder: Any, patchifier: VideoLatentPatchifier, tokens: torch.Tensor,
                  grid: tuple[int, int, int], seed: int, device: torch.device) -> torch.Tensor:
    import torch

    from ltx_core.types import VideoLatentShape

    frames, height, width = grid
    latent = patchifier.unpatchify(tokens, VideoLatentShape(batch=1, channels=128, frames=frames, height=height, width=width))
    decoded = decoder(latent.to(device=device, dtype=torch.bfloat16), generator=torch.Generator(device=device).manual_seed(seed))
    decoded = decoded[0] if isinstance(decoded, (tuple, list)) else decoded
    return decoded[0].float().clamp(-1, 1).permute(1, 0, 2, 3).cpu()


def positions_for(seq: FullSceneSequence, fps: int, device: torch.device) -> torch.Tensor:
    import torch

    from ltx_core.components.patchifiers import VideoLatentPatchifier, get_pixel_coords
    from ltx_core.types import SpatioTemporalScaleFactors, VideoLatentShape

    frames, height, width = seq.grid
    latent_coords = VideoLatentPatchifier(patch_size=1).get_patch_grid_bounds(
        output_shape=VideoLatentShape(frames=frames, height=height, width=width, batch=1, channels=128), device="cpu",
    )
    one = get_pixel_coords(
        latent_coords=latent_coords, scale_factors=SpatioTemporalScaleFactors.default(), causal_fix=True,
    ).to(torch.float32)
    one[:, 0, ...] = one[:, 0, ...] / float(fps)  # temporal axis in seconds
    # Every stream shares the RGB stream's spatio-temporal positions.
    return torch.cat([one] * len(seq.names), dim=2).to(device)


def restore_observation_pixels(decoded_rgb: torch.Tensor, observation_u8: torch.Tensor) -> torch.Tensor:
    """Return lossless uint8 RGB with exact observed frame zero, not VAE reconstruction."""
    import torch

    if decoded_rgb.ndim != 4 or observation_u8.shape != decoded_rgb[:1].shape or observation_u8.dtype != torch.uint8:
        raise ValueError("RGB observation must be a single uint8 frame matching the decoded video")
    rgb_uint8 = ((decoded_rgb + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)
    rgb_uint8[:1] = observation_u8
    return rgb_uint8


def generate(args: argparse.Namespace, metadata: dict[str, str]) -> None:
    import numpy as np
    import torch
    from PIL import Image
    from safetensors.torch import save_file

    from ltx_core.components.patchifiers import VideoLatentPatchifier
    from ltx_core.components.schedulers import LTX2Scheduler
    from causalwm import logger
    from causalwm.flow_codec import decode_flow, zero_flow_image
    from causalwm.layout import registry_metadata
    from causalwm.sampler import FullSceneInputs, FullSceneSchedules, run_full_scene_cot
    from causalwm.pointmap_codec import decode_pointmap_log
    from causalwm.video_utils import save_video

    started = time.monotonic()
    if not args.image.is_file() or not args.base_ckpt.is_file() or not args.text_encoder_dir.is_dir():
        raise ValueError("image/base-ckpt files and text-encoder-dir must exist locally")
    if args.out_dir.exists() and (not args.out_dir.is_dir() or any(args.out_dir.iterdir())):
        raise ValueError("out-dir must be new or empty; existing artifacts will not be overwritten")
    observation_u8, transform = load_rgb_observation(args.image, args.height, args.width)
    device = torch.device(args.device)
    grid = ((args.frames - 1) // 8 + 1, args.height // 32, args.width // 32)
    components, embeddings, checkpoint_info = _load_components(args, metadata)
    patchifier = VideoLatentPatchifier(patch_size=1)
    scheduler = LTX2Scheduler()
    schedules = FullSceneSchedules(
        flow=tuple(scheduler.execute(steps=args.steps_flow).float().tolist()),
        pointmap=tuple(scheduler.execute(steps=args.steps_pointmap).float().tolist()),
        video=tuple(scheduler.execute(steps=args.steps_video).float().tolist()),
    )
    with torch.inference_mode():
        components.text_encoder.to(device)
        embeddings.to(device)
        context = encode_text(components.text_encoder, embeddings, args.prompt)
        negative_context = encode_text(components.text_encoder, embeddings, args.negative_prompt) if args.guidance != 1 else None
        components.text_encoder.to("cpu")
        embeddings.to("cpu")
        components.video_vae_encoder.to(device)
        rgb0 = encode_observation(components.video_vae_encoder, patchifier, observation_u8.float() / 127.5 - 1, device)
        flow0 = encode_observation(components.video_vae_encoder, patchifier, zero_flow_image(1, args.height, args.width), device)
        components.video_vae_encoder.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        components.transformer.to(device=device, dtype=torch.bfloat16)
        logger.info("Full-scene self sampling: flow -> complete pointmap -> future RGB; no P0 input.")
        result = run_full_scene_cot(
            components.transformer, grid, FullSceneInputs(rgb0, flow0), context,
            lambda seq: positions_for(seq, args.fps, device), schedules,
            seed=args.seed, device=device, dtype=torch.bfloat16, guidance=args.guidance, negative_context=negative_context,
        )
        components.transformer.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        components.video_vae_decoder.to(device)
        images = {
            name: decode_tokens(components.video_vae_decoder, patchifier, result.latents[:, result.seq.segment(name)], grid, args.seed, device)
            for name in result.seq.names
        }
    # Decoded tensors are inference-mode tensors; the in-place RGB0/flow0 restores below need normal tensors.
    images = {name: value.clone() for name, value in images.items()}
    if any(tuple(value.shape) != (args.frames, 3, args.height, args.width) for value in images.values()):
        raise RuntimeError("VAE decoded an unexpected video shape")
    if not all(bool(torch.isfinite(value).all()) for value in images.values()):
        raise RuntimeError("nonfinite VAE output")

    # A fixed latent observation does not guarantee exact decoded pixels.
    # Restore RGB0 in the lossless numeric result BEFORE lossy MP4 encoding.
    rgb_uint8 = restore_observation_pixels(images["video"], observation_u8)
    images["video"][:1] = observation_u8.float() / 127.5 - 1
    images["flow"][:1] = 1.0  # Exact t0 no-pair sentinel; P0 is NEVER replaced.
    flow_uv = decode_flow(images["flow"], clip_px=float(metadata["flow_clip_px"]))
    flow_uv[0] = 0
    pointmap_xyz = decode_pointmap_log(images["pointmap"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_video(rgb_uint8, args.out_dir / "rgb.mp4", fps=args.fps, video_format="FCHW")
    save_video((images["flow"] + 1) / 2, args.out_dir / "flow.mp4", fps=args.fps, video_format="FCHW")
    save_video((images["pointmap"] + 1) / 2, args.out_dir / "pointmap_xyz_codec.mp4", fps=args.fps, video_format="FCHW")
    panels = torch.cat([rgb_uint8.float() / 255, (images["flow"] + 1) / 2, (images["pointmap"] + 1) / 2], dim=-1)
    save_video(panels, args.out_dir / "diagnostic_rgb_flow_pointmap.mp4", fps=args.fps, video_format="FCHW")
    for frame in sorted({0, args.frames // 2, args.frames - 1}):
        panel = (panels[frame].permute(1, 2, 0) * 255).round().clamp(0, 255).to(torch.uint8).numpy()
        Image.fromarray(panel).save(args.out_dir / f"diagnostic_frame_{frame:03d}.png")
    Image.fromarray(observation_u8[0].permute(1, 2, 0).numpy()).save(args.out_dir / "rgb0_resized.png")
    if args.save_raw:
        np.savez_compressed(args.out_dir / "rgb_uint8.npz", rgb=rgb_uint8.numpy())
        np.savez_compressed(args.out_dir / "flow_uv.npz", uv=flow_uv.numpy(), pair_valid=np.arange(args.frames) > 0)
        np.savez_compressed(args.out_dir / "pointmap_xyz.npz", xyz=pointmap_xyz.numpy())
        save_file(
            {name: result.latents[:, result.seq.segment(name)].detach().cpu().contiguous() for name in result.seq.names},
            str(args.out_dir / "generated_latents.safetensors"),
            metadata={**registry_metadata(), "artifact_kind": "generated_latents_not_model_weights"},
        )
    provenance = {
        **result.provenance, "checkpoint": {**_identity(args.checkpoint), "metadata": _public_checkpoint_metadata(metadata), "load": checkpoint_info},
        "base_checkpoint": _identity(args.base_ckpt), "text_encoder": "Gemma-3-12B",
        "image": {**_identity(args.image), "sha256": hashlib.sha256(args.image.read_bytes()).hexdigest()},
        "prompt": args.prompt, "negative_prompt": args.negative_prompt if args.guidance != 1 else None,
        "transform": transform, "frames": args.frames, "height": args.height, "width": args.width,
        "fps": args.fps, "latent_grid": grid, "dtype": "bfloat16", "device": str(device),
        "text_connector_apply_padding_mask": True,
        "scheduler": "LTX2Scheduler defaults", "registry": registry_metadata(),
        "flow_clip_px": float(metadata["flow_clip_px"]), "flow_units": "output pixels between consecutive generated frames",
        "flow_indexing": "uv[t] = frame(t-1)->frame(t) on the source-frame grid; t0 is invalid zero sentinel",
        "pointmap_units": "relative camera_t XYZ, normalized by frame-0 median depth; NOT metres",
        "pointmap_validity": "generated prediction; no validity mask",
        "rgb0_pixels": "exact resized observation in rgb_uint8.npz and pre-encode tensor; MP4 is lossy",
        "flow0_pixels": "restored deterministic zero-flow sentinel after VAE decode",
        "pointmap0_pixels": "generated; no post-decode replacement",
        "diagnostic_columns": ["RGB", "full-scene optical flow", "signed-log XYZ pointmap codec"],
        "seconds": time.monotonic() - started,
    }
    (args.out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n")
    logger.info("Saved full-scene generated diagnostics to %s (pointmap units are not metres).", args.out_dir)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_arguments(args)
        metadata = read_deploy_checkpoint_metadata(args.checkpoint)
        generate(args, metadata)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
