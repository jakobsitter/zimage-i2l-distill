# Environment Setup

This file documents the exact software stack required to run the teacher pair
generation pipeline. These versions were arrived at through trial and error —
the notes below explain *why* each version matters so you don't have to
rediscover it.

---

## Hardware

| Component | Spec |
|-----------|------|
| GPU | NVIDIA GeForce RTX 4090 (24 GB VRAM) |
| CUDA driver | 580.126.20 |
| CUDA toolkit | 12.4 |

The two encoder models together need **~16 GB of VRAM** at peak (SigLIP2 at
2.2 GB and DINOv3-7B at 13.4 GB, loaded sequentially). A 24 GB card is the
practical minimum. A 16 GB card will OOM on DINOv3 even alone.

---

## Python

```
Python 3.11.10
```

---

## PyTorch stack

Install with the **cu118** index URL even if your CUDA toolkit is newer.
The cu124 wheels bundle an older `nvidia-nccl-cu12` package (NCCL 2.14.3)
that is missing the `ncclCommRegister` symbol, causing an `ImportError` at
`import torch`. The cu118 wheels bring their own `nvidia-nccl-cu11` (2.20.5)
which works correctly.

```bash
pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
```

| Package | Version |
|---------|---------|
| torch | 2.4.1+cu118 |
| torchvision | 0.19.1+cu118 |

**Why not torch 2.0.x?** `transformers >= 4.33` disables PyTorch entirely
when it detects `torch < 2.1`, so the model classes are unavailable.

**Why not torch 2.1.x cu118?** Works, but transformers 4.57 uses
`torch.utils._pytree.register_pytree_node` which was renamed in 2.1.
Torch 2.4 restores the expected API.

---

## transformers

```bash
pip install "transformers>=4.46,<5.0"
```

Resolved to **4.57.6** at time of writing.

| Package | Version |
|---------|---------|
| transformers | 4.57.6 |
| tokenizers | 0.22.2 |

**Why not transformers 4.30.x (the original pinned version)?**
`DiffSynth-Studio` imports `is_deepspeed_zero3_enabled` from
`transformers.integrations`, which was added in 4.33. The old pin caused
an `ImportError` at startup.

**Breaking changes in 4.57 vs the DiffSynth vendor code:**

1. `SiglipVisionModel` now wraps its internals under `self.vision_model`
   (previously they were flat on `self`). The checkpoint keys have *no*
   `vision_model.` prefix, so a state-dict converter is required, and the
   `forward()` method had to be updated to route through `self.vision_model`.

2. `DINOv3ViTImageProcessor` was replaced by `DINOv3ViTImageProcessorFast`.

3. `DINOv3ViTModel` no longer has a `self.model` wrapper around the layer
   list; the layers are at `self.layer` directly.

All three are patched in the vendored `DiffSynth-Studio` fork
(`vendor/DiffSynth-Studio`, branch `fix/teacher-pair-generation-pipeline`).

---

## Other Python packages

```bash
pip install accelerate==1.13.0 peft==0.19.1 safetensors==0.7.0 \
    modelscope==1.36.3 imageio[ffmpeg] einops sentencepiece protobuf \
    ftfy pandas datasets pillow
```

| Package | Version |
|---------|---------|
| accelerate | 1.13.0 |
| peft | 0.19.1 |
| safetensors | 0.7.0 |
| modelscope | 1.36.3 |
| imageio | 2.37.3 |
| einops | 0.8.2 |
| sentencepiece | 0.2.1 |
| protobuf | 7.34.1 |
| ftfy | 6.3.1 |
| pandas | 3.0.2 |
| datasets | 4.8.5 |
| pillow | 12.2.0 |

---

## Installing this repo

```bash
git clone --recurse-submodules https://github.com/jakobsitter/zimage-i2l-distill
cd zimage-i2l-distill
pip install -e .
```

The vendored `DiffSynth-Studio` submodule must be initialised (`--recurse-submodules`
or `git submodule update --init`). The submodule points to the patched fork at
`jakobsitter/DiffSynth-Studio` on branch `fix/teacher-pair-generation-pipeline` —
do **not** use the upstream `modelscope/DiffSynth-Studio` main branch, it is
incompatible with transformers 4.57.

```bash
pip install -e vendor/DiffSynth-Studio
```

---

## Model weights

The CLI downloads weights automatically from ModelScope on first run. They
land in `./models/DiffSynth-Studio/General-Image-Encoders/`:

| Model | Size on disk | VRAM (bfloat16) |
|-------|-------------|-----------------|
| SigLIP2-G384 (`SigLIP2-G384/model.safetensors`) | 4.3 GB | 2.2 GB |
| DINOv3-7B (`DINOv3-7B/model.safetensors`) | 13.4 GB | 13.4 GB |

To cache them at a custom location set the env vars before running:

```bash
export ZIMAGE_SIGLIP2_PATH=/path/to/SigLIP2-G384/model.safetensors
export ZIMAGE_DINOV3_PATH=/path/to/DINOv3-7B/model.safetensors
```

---

## GPU memory strategy

Both models cannot reside on the GPU simultaneously (2.2 + 13.4 = 15.6 GB
weights alone; activation memory during forward pushes this to ~16–18 GB,
but fragmentation and the CUDA allocator overhead can push total VRAM usage
past 22 GB and OOM a 24 GB card).

The pipeline works around this by:

1. Loading both models to **CPU RAM** at startup.
2. Moving each encoder to GPU only during its own forward pass, then
   immediately moving it back to CPU and calling `torch.cuda.empty_cache()`.
3. Wrapping all forward passes in `torch.no_grad()` — without this, PyTorch
   builds an autograd graph that roughly doubles activation memory. For a
   13 GB model that is enough to OOM by itself.

---

## Quick sanity check

```bash
python3 -c "
import torch
print('torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')
import transformers
print('transformers:', transformers.__version__)
from diffsynth.models.siglip2_image_encoder import Siglip2ImageEncoder
from diffsynth.models.dinov3_image_encoder import DINOv3ImageEncoder
print('DiffSynth encoder imports: OK')
"
```
