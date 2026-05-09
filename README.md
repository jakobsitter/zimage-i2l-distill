# zimage-i2l-distill

Standalone repository for zimage image-to-LoRA distillation work.

## Local CUDA inference

The local inference path loads only the student checkpoint and the existing Z-Image i2L decoder checkpoint, then writes a standard LoRA `.safetensors` file.

```bash
CUDA_VISIBLE_DEVICES=0 zimage-distill-infer \
  --student-checkpoint checkpoints/student.pt \
  --decoder-checkpoint /path/to/Z-Image-i2L/model.safetensors \
  --reference-image /path/to/reference_1.jpg \
  --reference-image /path/to/reference_2.jpg \
  --reference-image /path/to/reference_3.jpg \
  --reference-image /path/to/reference_4.jpg \
  --output outputs/lora.safetensors
```
