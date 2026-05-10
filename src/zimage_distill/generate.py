from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file

from diffsynth.core import ModelConfig
from diffsynth.pipelines.z_image import ZImagePipeline

from .i2l_adapter import I2LDecoderAdapter, resolve_decoder_checkpoint
from .infer import load_reference_images
from .train import load_checkpoint

DEFAULT_NEGATIVE_PROMPT = (
    "泛黄，发绿，模糊，低分辨率，低质量图像，扭曲的肢体，诡异的外观，丑陋，AI感，"
    "噪点，网格感，JPEG压缩条纹，异常的肢体，水印，乱码，意义不明的字符"
)

Z_IMAGE_MODELS = [
    ("Tongyi-MAI/Z-Image",       "transformer/*.safetensors"),
    ("Tongyi-MAI/Z-Image-Turbo", "text_encoder/*.safetensors"),
    ("Tongyi-MAI/Z-Image-Turbo", "vae/diffusion_pytorch_model.safetensors"),
]
Z_IMAGE_TOKENIZER = ("Tongyi-MAI/Z-Image-Turbo", "tokenizer/")


def load_generation_pipeline(device: str) -> ZImagePipeline:
    print("Loading Z-Image generation pipeline (auto-downloads on first run)...")
    model_configs = [
        ModelConfig(model_id=mid, origin_file_pattern=pat)
        for mid, pat in Z_IMAGE_MODELS
    ]
    tok_mid, tok_pat = Z_IMAGE_TOKENIZER
    return ZImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=model_configs,
        tokenizer_config=ModelConfig(model_id=tok_mid, origin_file_pattern=tok_pat),
    )


def lora_from_checkpoint(
    student_checkpoint: Path,
    decoder_checkpoint: Path | None,
    reference_images: list[Path],
    device: str,
) -> dict[str, torch.Tensor]:
    student = load_checkpoint(student_checkpoint).to(device).eval()
    images = load_reference_images(reference_images).to(device)
    with torch.no_grad():
        embedding = student(images)

    decoder = I2LDecoderAdapter(checkpoint_path=decoder_checkpoint, device=device)
    return decoder.predict_lora(embedding)


def lora_from_file(lora_path: Path) -> dict[str, torch.Tensor]:
    return load_file(str(lora_path), device="cpu")


def generate_image(
    pipe: ZImagePipeline,
    lora: dict[str, torch.Tensor],
    prompt: str,
    negative_prompt: str,
    seed: int,
    cfg_scale: float,
    num_inference_steps: int,
    sigma_shift: float,
) -> Image.Image:
    return pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        cfg_scale=cfg_scale,
        num_inference_steps=num_inference_steps,
        positive_only_lora=lora,
        sigma_shift=sigma_shift,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zimage-distill-generate",
        description="Generate an image using a distilled student encoder and Z-Image.",
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--reference-image", action="append", dest="reference_images", type=Path,
        help="Reference image path (repeatable). Runs student + i2L to produce the LoRA.",
    )
    src.add_argument(
        "--lora", type=Path,
        help="Pre-computed LoRA safetensors file. Skips student and i2L steps.",
    )
    src.add_argument(
        "--no-lora", action="store_true",
        help="Generate with no LoRA at all. Useful for checking the base pipeline.",
    )

    parser.add_argument("--prompt", required=True, help="Text prompt for generation.")
    parser.add_argument("--output", required=True, type=Path, help="Output image path.")
    parser.add_argument("--student-checkpoint", type=Path, default=None,
                        help="Student checkpoint. Required when using --reference-image.")
    parser.add_argument("--decoder-checkpoint", type=Path, default=None,
                        help="i2L decoder checkpoint. Auto-downloads if omitted.")
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--sigma-shift", type=float, default=8.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.reference_images and args.student_checkpoint is None:
        build_parser().error("--student-checkpoint is required when using --reference-image")

    if args.no_lora:
        print("Generating with no LoRA (base pipeline check).")
        lora = None
    elif args.lora:
        print(f"Loading LoRA from {args.lora}")
        lora = lora_from_file(args.lora)
    else:
        print("Running student encoder + i2L decoder...")
        lora = lora_from_checkpoint(
            args.student_checkpoint,
            args.decoder_checkpoint,
            args.reference_images,
            args.device,
        )

    pipe = load_generation_pipeline(args.device)

    print(f"Generating: {args.prompt!r}")
    image = generate_image(
        pipe, lora,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        cfg_scale=args.cfg_scale,
        num_inference_steps=args.steps,
        sigma_shift=args.sigma_shift,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(out))
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
