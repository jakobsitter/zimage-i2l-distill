#!/bin/bash
# Cloud training script for zimage-i2l-distill student on a 5090 (32GB VRAM).
#
# === SETUP (run once) ===
#   1. Clone:  git clone <repo-url> && cd zimage-i2l-distill
#   2. Upload dataset (both parts):
#        rsync -avz data/teacher_pairs user@cloud:~/zimage-i2l-distill/data/teacher_pairs/
#        rsync -avz dataset_master_clean/ user@cloud:~/zimage-i2l-distill/dataset_master_clean/
#      Or upload the archive and extract:
#        scp dataset_master_clean.tar.gz user@cloud:~/zimage-i2l-distill/
#        ssh user@cloud "cd ~/zimage-i2l-distill && tar -xzf dataset_master_clean.tar.gz"
#   3. Set HF token (needed for live-teacher model downloads):
#        export HF_TOKEN="hf_..."
#   4. Run:    bash scripts/cloud_train.sh
#
# === DOWNLOAD RESULTS ===
#   scp user@cloud:~/zimage-i2l-distill/checkpoints/student_dual-dinov3l-siglip2l.pt .
#
set -euo pipefail

CKPT="checkpoints/student_dual-dinov3l-siglip2l.pt"
DATA_DIR="data/teacher_pairs"
OUTPUT_DIR="checkpoints"
MIN_GPU_MB=24000
MIN_DISK_GB=10

# ---- Preflight ----
echo "=== Preflight checks ==="

# Dataset (embedding manifests + teacher_embeddings)
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Dataset not found at $DATA_DIR"
    echo "  Upload it first: rsync -avz data/teacher_pairs user@cloud:~/zimage-i2l-distill/data/teacher_pairs/"
    exit 1
fi
sample_count=$(find "$DATA_DIR" -maxdepth 1 -type d -exec test -f "{}/manifest.json" \; -print | wc -l)
echo "  Dataset: $DATA_DIR ($sample_count samples with manifest.json)"

# Reference images (dataset_master_clean/)
if [ ! -d "dataset_master_clean" ]; then
    echo "ERROR: Reference images not found at dataset_master_clean/"
    echo "  Upload it: rsync -avz dataset_master_clean/ user@cloud:~/zimage-i2l-distill/dataset_master_clean/"
    echo "  Or extract from archive on the server: tar -xzf dataset_master_clean.tar.gz"
    exit 1
fi
ref_img_count=$(find -L dataset_master_clean -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) | wc -l)
echo "  Reference images: dataset_master_clean/ ($ref_img_count files)"

# Disk space
available_kb=$(df -k --output=avail . | tail -1)
available_gb=$((available_kb / 1024 / 1024))
echo "  Disk free: ${available_gb}G"
if [ "$available_gb" -lt "$MIN_DISK_GB" ]; then
    echo "ERROR: Less than ${MIN_DISK_GB}G free (have ${available_gb}G). Need space for checkpoints and model downloads."
    exit 1
fi

# HF token (warn only — Phase 1+2 don't need it, Phase 3 --live-teacher does)
if [ -z "${HF_TOKEN:-}" ]; then
    echo "  WARNING: HF_TOKEN not set. Phase 3 --live-teacher will fail to download models."
    echo "    Set it: export HF_TOKEN=\"hf_...\""
fi

# ---- Install ----
echo ""
echo "=== Installing dependencies ==="
python -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel

# cu128 = CUDA 12.8, required for RTX 5090 Blackwell (SM 12.0).
# NOTE: ENVIRONMENT.md uses cu118 but that is for the RTX 4090 (Ada Lovelace,
# SM 8.9). CUDA 11.8 has no Blackwell kernels — the 5090 won't work with it.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install "transformers>=4.46,<5.0" safetensors huggingface_hub modelscope pillow

# DiffSynth-Studio vendored fork (needed for live-teacher Phase 3)
git submodule update --init vendor/DiffSynth-Studio
pip install -e vendor/DiffSynth-Studio
pip install -e .

# ---- Verify GPU ----
echo ""
echo "=== GPU check ==="
python -c "import torch; assert torch.cuda.is_available(), 'GPU not found'; print(f'GPU: {torch.cuda.get_device_name(0)}')"
nvidia-smi --query-gpu=memory.total --format=csv,noheader
gpu_mem_mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "  GPU memory: ${gpu_mem_mb}MB (need >= ${MIN_GPU_MB}MB)"
if [ "$gpu_mem_mb" -lt "$MIN_GPU_MB" ]; then
    echo "ERROR: GPU VRAM too small for full dual-dinov3l-siglip2l training."
    exit 1
fi

# ---- Phase 1: Head-only (feature cache, trains fast) ----
echo ""
echo "============================================"
echo " PHASE 1: Head-only (feature cached)"
echo "============================================"
python -m zimage_distill.train \
  --backbone dual-dinov3l-siglip2l \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --phase1-epochs 30 --phase2-epochs 0 --phase3-epochs 0 \
  --lr 1e-3 \
  --val-fraction 0.1

if [ ! -f "$CKPT" ]; then
    echo "ERROR: Phase 1 did not produce $CKPT"
    exit 1
fi
echo "Phase 1 complete: $CKPT"

# ---- Phase 2: Full model fine-tuning ----
echo ""
echo "============================================"
echo " PHASE 2: Full model fine-tuning"
echo "============================================"
python -m zimage_distill.train \
  --backbone dual-dinov3l-siglip2l \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --phase1-epochs 0 --phase2-epochs 30 --phase3-epochs 0 \
  --lr 1e-3 \
  --val-fraction 0.1 \
  --clip-loss \
  --resume-checkpoint "$CKPT"

echo "Phase 2 complete: $CKPT"

# ---- Phase 3: Augmentation polish with live teacher ----
echo ""
echo "============================================"
echo " PHASE 3: Live-teacher augmented polish"
echo "============================================"

# Live-teacher needs HF_TOKEN to download DINOv3-7B + SigLIP2-G384
if [ -z "${HF_TOKEN:-}" ]; then
    echo "SKIPPING Phase 3: HF_TOKEN not set (needed for live teacher model download)."
    echo "  Re-run with: HF_TOKEN=\"hf_...\" bash scripts/cloud_train.sh"
else
    python -m zimage_distill.train \
      --backbone dual-dinov3l-siglip2l \
      --data-dir "$DATA_DIR" \
      --output-dir "$OUTPUT_DIR" \
      --phase1-epochs 0 --phase2-epochs 0 --phase3-epochs 10 \
      --lr 1e-3 \
      --val-fraction 0.1 \
      --augment \
      --clip-loss \
      --live-teacher \
      --resume-checkpoint "$CKPT"
    echo "Phase 3 complete: $CKPT"
fi

echo ""
echo "============================================"
echo " DONE"
echo "============================================"
echo "Checkpoint: $CKPT"
echo ""
echo "Download:"
echo "  scp user@cloud:~/zimage-i2l-distill/$CKPT ."
