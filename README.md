<p align="center">
  <img src="docs/assets/aetherai-mark.svg" alt="Aether AI" width="88" height="88">
</p>

<h1 align="center">
  <img src="docs/assets/causalwm-wordmark.svg" alt="CausalWMv1" width="480">
</h1>

<h2 align="center">
  <img src="docs/assets/causalwm-subtitle.svg" alt="Causal Chain-of-Thought Reasoning for Embodied World Model" width="100%">
</h2>

<p align="center">
  <a href="https://openreview.net/pdf?id=3pf4d0EEqm"><img src="https://img.shields.io/badge/Technical_Report-OpenReview-b31b1b?style=flat" alt="Technical Report: OpenReview"></a>
  <a href="https://aetherlabsai.github.io/CausalWM/"><img src="https://img.shields.io/badge/Website-CausalWM-6554c0?style=flat" alt="Website: CausalWM"></a>
  <a href="https://huggingface.co/AetherLabs-AI/CausalWM"><img src="https://img.shields.io/badge/Weights-Hugging_Face-FFD21E?style=flat&amp;logo=huggingface&amp;logoColor=FFD21E" alt="CausalWMv1 on Hugging Face"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-LTX--2_Community-blue?style=flat" alt="License: LTX-2 Community"></a>
</p>

<p align="center">
  <a href="#news">News</a> · <a href="#demos">Demos</a> · <a href="#method">Method</a> · <a href="#results">Results</a> · <a href="#weights">Weights</a> · <a href="#installation">Quickstart</a> · <a href="#documentation">Documentation</a> · <a href="#citation">Citation</a>
</p>

<a id="news"></a>

## 📰 News & Highlights

- **September 2026** · CausalWMv1 TI2V CoT inference code and model weights. The technical report is now available; click the **Technical Report** badge above to read it. The current released checkpoint is tailored for **PAI-Bench-G**. Additional checkpoints will be released soon.

<a id="overview"></a>

## 🧭 Overview

**Release version: CausalWMv1.** This repository contains the first version of
CausalWM’s TI2V CoT inference interface and accompanying model release.

**CausalWM reasons through motion and geometry before predicting future video.**
Given an observed scene and an instruction, it generates an explicit causal chain
of thought: optical flow describes how the scene moves, pointmaps describe its
geometry, and these intermediate predictions guide the future RGB video.

One shared diffusion Transformer carries out the entire chain. Stage-causal
attention lets each prediction use the observation and preceding reasoning steps,
connecting the initial scene to its visual consequences.

```text
First-frame image + instruction
              ↓
         Optical flow → XYZ pointmaps → Future RGB video
```

The paper studies language- and action-conditioned world modeling, in-context
feature guidance, and generation with few denoising steps. This repository provides
the **TI2V CoT inference interface**, including generated motion,
geometry, and RGB streams.

<a id="demos"></a>

## 🎬 Demos

[![CausalWM animated demo: optical flow, XYZ pointmap visualization, and future RGB video.](docs/assets/demo-flow-pointmap-rgb.gif)](docs/assets/demo-flow-pointmap-rgb.mp4)

*A dual-arm manipulation example. Left to right: optical flow, pointmap
visualization, and RGB. The preview loops automatically.
[Watch the full-resolution video](docs/assets/demo-flow-pointmap-rgb.mp4), or explore
more cases on the [project website](https://aetherlabsai.github.io/CausalWM/).*

### Generated examples

Left to right: **Wan2.2 / LingBot / CausalWM**.

#### Retrieve a bottle from the refrigerator

[![Retrieve a bottle from the refrigerator — Wan2.2 / LingBot / CausalWM](docs/assets/cases/single_arm_0099-comparison.gif)](docs/assets/cases/single_arm_0099-comparison.mp4)

<details>
<summary>Try this input</summary>

```bash
python inference.py \
  --image examples/inputs/single_arm_0099.jpg \
  --prompt "The robotic gripper retrieves the deep red glass bottle from the refrigerator." \
  --checkpoint /path/to/CausalWMv1.safetensors \
  --base-ckpt /path/to/ltx-2.3-22b-dev.safetensors \
  --text-encoder-dir /path/to/gemma-3-12b \
  --seed 81942 \
  --out-dir outputs/single_arm_0099
```

</details>

#### Close the file cabinet drawer

[![Close the file cabinet drawer — Wan2.2 / LingBot / CausalWM](docs/assets/cases/single_arm_0090-comparison.gif)](docs/assets/cases/single_arm_0090-comparison.mp4)

<details>
<summary>Try this input</summary>

```bash
python inference.py \
  --image examples/inputs/single_arm_0090.jpg \
  --prompt "The robotic gripper closes the file cabinet drawer." \
  --checkpoint /path/to/CausalWMv1.safetensors \
  --base-ckpt /path/to/ltx-2.3-22b-dev.safetensors \
  --text-encoder-dir /path/to/gemma-3-12b \
  --seed 81554 \
  --out-dir outputs/single_arm_0090
```

</details>

<a id="framework"></a>

## 🧩 Framework

[![CausalWM framework: visual context, multimodal conditioning, and stage-causal prediction of motion, geometry, and future RGB.](docs/assets/framework.png)](docs/assets/framework.png)

*The paper's framework figure. The released TI2V interface follows the
flow → pointmap → RGB chain. Click the figure for full resolution.*

<a id="method"></a>

## 🔄 Method

Three generation stages share the same backbone:

| Stage | Prediction | Available context |
| --- | --- | --- |
| **1 · Motion** | Optical flow | First-frame observation and instruction |
| **2 · Geometry** | Camera-frame XYZ pointmaps | Observation, instruction, and generated flow |
| **3 · Future video** | Future RGB frames | Observation, instruction, generated flow, and generated pointmaps |

Flow and pointmaps are encoded as images and pass through the same video VAE as
RGB. Per-stream projections and modality conditioning distinguish the three
streams, while causal attention controls which stages can read one another.

Inference takes **one RGB image and a text instruction**. Motion and geometry
are generated by the model; a flow or depth estimator is not required at inference
time. The default schedule uses **4 denoising steps per stage** with guidance 1.0.

<a id="results"></a>

## 📊 Results

The manuscript reports the following benchmark results:

| Benchmark and setting | Metric | CausalWM |
| --- | --- | ---: |
| TriWorldBench · action-conditioned, multi-view | TWB-Score ↑ | **66.04** |
| PAI-Bench · language-conditioned, robot domain | RO score ↑ | **89.9** |

TriWorldBench uses the leaderboard snapshot of **September 11, 2026**. PAI-Bench
uses **Qwen3-VL-235B-A22B-Instruct** as the judge. These values correspond to the
manuscript's benchmark configurations; see the model card for the inference settings
of the released checkpoint.

See the [project website](https://aetherlabsai.github.io/CausalWM/) for comparisons,
the denoising-step study, and qualitative examples.

<a id="weights"></a>

## 📦 Model weights

**Hugging Face repository: [AetherLabs-AI/CausalWM](https://huggingface.co/AetherLabs-AI/CausalWM)**

Checkpoint: **`CausalWMv1.safetensors`**.

The current released checkpoint is tailored for **PAI-Bench-G**. Additional checkpoints will be released soon.

Inference uses three local assets:

| Asset | Purpose | Source |
| --- | --- | --- |
| **CausalWMv1** checkpoint (`CausalWMv1.safetensors`) | CoT Transformer weights | [Hugging Face](https://huggingface.co/AetherLabs-AI/CausalWM) |
| `ltx-2.3-22b-dev.safetensors` | Backbone configuration, video VAE, and text connector | [Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) |
| Gemma-3-12B directory | Text encoder | [google/gemma-3-12b-it-qat-q4_0-unquantized](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized) |

Download the base assets under their respective terms and pass their local paths
to the inference command. See the [CausalWMv1 model card](https://huggingface.co/AetherLabs-AI/CausalWM)
for checkpoint details and recommended settings.

<a id="installation"></a>

## ⚙️ Installation

Use **Python 3.10+**, a CUDA GPU, and
[uv](https://docs.astral.sh/uv/getting-started/installation/). Python 3.11 is the
tested version.

```bash
git clone https://github.com/AetherLabsAI/CausalWM.git
cd CausalWM
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e packages/ltx-core -e .
```

The tested environment uses `torch 2.9.1+cu128`, `torchvision 0.24.1`, and
`transformers 4.57.6`. Dependency ranges are specified in `pyproject.toml`.

<a id="inference"></a>

## 🚀 Quickstart

Run from the repository root:

```bash
python inference.py \
  --image /path/to/first_frame.png \
  --prompt "The robot arm picks up the red block and places it in the box." \
  --checkpoint /path/to/CausalWMv1.safetensors \
  --base-ckpt /path/to/ltx-2.3-22b-dev.safetensors \
  --text-encoder-dir /path/to/gemma-3-12b \
  --out-dir outputs/demo
```

The default output is **121 frames at 640 × 480 and 16 FPS**, using four denoising
steps for each stage. The full input frame is resized to the output dimensions.
Components move to the GPU in sequence; inference has been tested on a single
NVIDIA H200.

```text
outputs/demo/
  rgb.mp4                           Future RGB video
  flow.mp4                          Generated optical flow visualization
  pointmap_xyz_codec.mp4             Generated XYZ pointmap visualization
  diagnostic_rgb_flow_pointmap.mp4   Side-by-side RGB | flow | pointmap
  diagnostic_frame_*.png            First, middle, and last diagnostic frames
  rgb0_resized.png                   Resized conditioning image
  provenance.json                   Inputs, settings, and checkpoint metadata
```

Add `--save-raw` to export lossless arrays and generated latents. Pointmaps use
camera-frame coordinates normalized by the first frame's median depth, giving
a relative scale. See the [inference guide](docs/INFERENCE.md) for all options
and output conventions.

<a id="repository-layout"></a>

## 🗂️ Repository layout

```text
inference.py           TI2V inference entrypoint
causalwm/              Causal sampler, stream codecs, and checkpoint loading
packages/ltx-core/     LTX-2 core with the multi-stream causal Transformer
docs/                  Inference guide, figures, and case study videos
examples/              Case study initial frames and captions
```

<a id="documentation"></a>

## 📖 Documentation

- [Inference guide](docs/INFERENCE.md): arguments, output formats, and resource requirements.
- [Project website](https://aetherlabsai.github.io/CausalWM/): method, results, and demos.
- [Hugging Face repository](https://huggingface.co/AetherLabs-AI/CausalWM): CausalWMv1 weights and model card.
- [NOTICE](NOTICE): upstream attribution and modified components.

<a id="citation"></a>

## 📝 Citation

If CausalWM is useful for your research, please cite our technical report as follows.
The citation will be updated once the arXiv version is available.

```bibtex
@misc{xu2026causalwm,
  title        = {CausalWM: Causal Chain-of-Thought Reasoning for Embodied World Model},
  author       = {Xu, Ziming and Liang, Shuang and Han, Ruobing and Xi, Ziqiao and
                  Rao, Mingxing and Zhou, Kun and Zhang, Zijun and Yan, Yuchen and
                  Wei, Yufan and Huang, Junbo and Shao, Yifei and Nan, Fang and Huang, Biwei},
  year         = {2026},
  howpublished = {OpenReview},
  url          = {https://openreview.net/forum?id=3pf4d0EEqm},
  note         = {Preprint}
}
```

<a id="license"></a>

## ⚖️ License & acknowledgments

CausalWM builds on [LTX-2](https://github.com/Lightricks/LTX-2) and is distributed
under the [LTX-2 Community License Agreement](LICENSE). See [NOTICE](NOTICE) for
upstream attribution. The Gemma-3-12B text encoder is subject to the Gemma Terms
of Use.
