# CausalWMv1 inference

The interface takes one RGB image and a text instruction and generates optical
flow, XYZ pointmaps, and future RGB video. See the [README](../README.md#installation)
for installation and model assets.

## Inputs and options

Run `python inference.py --help` from the repository root for the complete CLI.

| Option | Default | Meaning |
| --- | --- | --- |
| `--image` | Required | One local observation image; animated images are rejected. |
| `--prompt` | Required | Nonempty instruction, used without prompt enhancement. |
| `--checkpoint` | Required | CausalWMv1 `.safetensors` checkpoint. |
| `--base-ckpt` | Required | Local LTX-2.3-22B base checkpoint. |
| `--text-encoder-dir` | Required | Local Gemma-3-12B directory. |
| `--out-dir` | Required | New or empty output directory. |
| `--frames` | `121` | Positive frame count of the form `8k + 1`. |
| `--height`, `--width` | `480`, `640` | Positive multiples of 32. |
| `--fps` | `16` | Output frame rate and temporal positional-encoding rate. |
| `--steps-flow`, `--steps-pointmap`, `--steps-video` | `4` each | At least 2 denoising steps per stage with the current scheduler. |
| `--guidance` | `1.0` | Classifier-free guidance scale; 1 disables CFG. |
| `--negative-prompt` | Empty | Used when guidance differs from 1. |
| `--seed` | `42` | Random seed. |
| `--device` | `cuda` | PyTorch device. |
| `--save-raw` | Off | Also export lossless arrays and generated latents. |

The whole input image is resized with bicubic interpolation and no crop. Prompts
are truncated to 128 tokens. Flow frame zero is a fixed zero-motion sentinel;
pointmap frame zero is generated. Neither a depth map nor a future frame is
accepted as an input.

## Outputs

The default output contains RGB, flow, and pointmap MP4s, a combined diagnostic
video, diagnostic PNGs, and the resized observation. MP4 encoding is lossy.
The combined diagnostic uses **RGB | flow | pointmap** column order.

With `--save-raw`, the following files are added:

| File | Contents |
| --- | --- |
| `rgb_uint8.npz` | `rgb`: RGB frames in `(F, 3, H, W)`, uint8. Frame zero equals the resized input. |
| `flow_uv.npz` | `uv`: optical flow in `(F, 2, H, W)`, plus `pair_valid`. |
| `pointmap_xyz.npz` | `xyz`: relative camera-frame XYZ in `(F, 3, H, W)`. |
| `generated_latents.safetensors` | Generated `video`, `flow`, and `pointmap` latents. |

`uv[t]` represents displacement from frame `t-1` to frame `t`, in output pixels
on the source-frame grid. Index zero is an invalid zero sentinel. Pointmap XYZ
uses the first frame's median depth as its scale, so coordinates are **relative,
not metres**. Pointmaps are generated predictions without a validity mask.

`provenance.json` records the supplied prompts, settings, input image hash, file
sizes, and validated runtime metadata. Local paths, filenames, file timestamps,
and arbitrary checkpoint metadata are omitted. Prompts and generated content
remain part of the output; review them before sharing an output directory.
Generated latents are output artifacts and cannot substitute for model weights.

## Environment and resources

The tested stack is Python 3.11, PyTorch 2.9.1 with CUDA 12.8, torchvision 0.24.1,
and transformers 4.57.6. Transformers 5.x is outside the supported range of this
release's Gemma loader.

Inference has been tested on a single NVIDIA H200. Components are loaded in CPU
memory and moved to the GPU by stage. Host memory must accommodate the model
components and checkpoint loading; the checkpoint's file size alone is not a
minimum RAM specification. A validated minimum VRAM requirement for smaller GPUs
has not been established.

## Checkpoint compatibility

The loader checks the flow/pointmap registry, coordinate convention, modality
heads, flow codec scale, and the complete parameter keys and shapes. A compatible
CausalWM checkpoint is required. Base LTX-2 weights supply the VAE, backbone
configuration, and text connector; they do not provide CausalWM's CoT parameters.

Use the model card's recommended settings for the checkpoint you download.
