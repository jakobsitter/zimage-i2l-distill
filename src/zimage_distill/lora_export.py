"""Export LoRA weights from reference images + student encoder + i2l decoder.

Usage:
  python -m zimage_distill.lora_export \
    --reference-image ref1.png --reference-image ref2.png \
    --student-checkpoint checkpoints/student_dual-dinov3s-siglip2b.pt \
    --output output/style.lora.safetensors \
    --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from .infer import load_reference_images
from .train import load_checkpoint
from .i2l_adapter import I2LDecoderAdapter


def export_lora(
    reference_images: list[Path],
    student_checkpoint: Path,
    decoder_checkpoint: Path | None,
    strengths: list[float] | None,
    output: Path,
    device: str = "cuda",
    backbone: str = "dual-dinov3b-siglip2l",
) -> None:
    student = load_checkpoint(student_checkpoint).to(device).eval()
    images = load_reference_images(reference_images).to(device)

    with torch.no_grad():
        embedding = student(images)  # [num_images, 5632]

    # Blend embeddings by strengths
    if strengths is not None:
        if len(strengths) == 1 and embedding.shape[0] > 1:
            strengths = strengths * embedding.shape[0]
        strengths_t = torch.tensor(strengths, device=embedding.device, dtype=embedding.dtype)
        strengths_t = strengths_t / strengths_t.sum()
        embedding = (embedding * strengths_t[:, None]).sum(dim=0)
    elif embedding.ndim == 2 and embedding.shape[0] > 1:
        embedding = embedding.mean(dim=0)

    decoder = I2LDecoderAdapter(checkpoint_path=decoder_checkpoint, device=device)
    lora = decoder.predict_lora(embedding)

    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.dumps({
        "source_images": [str(p) for p in reference_images],
        "student_checkpoint": str(student_checkpoint),
    })
    save_file({k: v.cpu() for k, v in lora.items()}, str(output), metadata={"info": metadata})
    print(f"Exported {len(lora)} LoRA tensors to {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export LoRA weights from reference images.")
    parser.add_argument("--reference-image", type=Path, required=True, action="append", dest="reference_images")
    parser.add_argument("--strength", type=float, action="append", default=None, dest="strengths",
                        help="Per-image strength (repeatable). Default: equal weight.")
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--decoder-checkpoint", type=Path, default=None,
                        help="I2L decoder checkpoint. Auto-downloads if omitted.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--backbone", default="dual-dinov3b-siglip2l",
                        help="Student backbone. Default: dual-dinov3b-siglip2l")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    export_lora(
        reference_images=args.reference_images,
        student_checkpoint=args.student_checkpoint,
        decoder_checkpoint=args.decoder_checkpoint,
        strengths=args.strengths,
        output=args.output,
        device=args.device,
        backbone=args.backbone,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
