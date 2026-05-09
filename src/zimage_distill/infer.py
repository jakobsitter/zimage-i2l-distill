from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from safetensors.torch import save_file
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize, to_tensor

from .i2l_adapter import I2LDecoderAdapter
from .train import load_checkpoint

IMAGE_SIZE = 224


def load_reference_images(image_paths: Sequence[Path | str]) -> torch.Tensor:
    images: list[torch.Tensor] = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            resized = resize(rgb_image, [IMAGE_SIZE, IMAGE_SIZE], interpolation=InterpolationMode.BILINEAR, antialias=True)
            images.append(to_tensor(resized))
    return torch.stack(images)


def emit_lora(student_embedding: torch.Tensor, decoder: I2LDecoderAdapter, out_path: Path | str) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lora = decoder.predict_lora(student_embedding)
    save_file({name: tensor.detach().cpu() for name, tensor in lora.items()}, str(out_path))
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local zimage image-to-LoRA inference path")
    parser.add_argument("--student-checkpoint", required=True, type=Path)
    parser.add_argument("--decoder-checkpoint", required=True, type=Path)
    parser.add_argument("--reference-image", action="append", dest="reference_images", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(args.device)

    student = load_checkpoint(args.student_checkpoint).to(device)
    student.eval()
    decoder = I2LDecoderAdapter(args.decoder_checkpoint, device=device)

    reference_images = load_reference_images(args.reference_images).to(device)
    with torch.no_grad():
        student_embedding = student(reference_images)

    emit_lora(student_embedding, decoder, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
