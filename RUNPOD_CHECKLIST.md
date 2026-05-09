# RunPod Training Checklist

1. Build or select a CUDA-enabled RunPod image with Python 3.11 and PyTorch.
2. Mount the repo so the working directory contains `data/teacher_pairs/` and `checkpoints/`.
3. Copy the teacher pair folders into `data/teacher_pairs/`.
4. Install the repo requirements and the vendored DiffSynth-Studio submodule.
5. Run `scripts/runpod_train.sh`.
6. Download `checkpoints/student.pt` and `checkpoints/student_manifest.json`.
7. Use the local inference CLI with the downloaded student checkpoint and your decoder checkpoint to emit a LoRA `.safetensors` file.
8. Copy the LoRA file into your local Z-Image workflow.
