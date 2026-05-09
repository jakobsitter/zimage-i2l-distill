from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from diffsynth.core import ModelConfig
from diffsynth.core.data.operators import ImageCropAndResize
from diffsynth.diffusion.base_pipeline import BasePipeline

from .paths import data_dir, repo_root

MANIFEST_FILENAME = "manifest.json"
EMBEDDING_FILENAME = "teacher_embedding.safetensors"
EMBEDDING_KEY = "teacher_embedding"
DEFAULT_OUTPUT_DIR = data_dir() / "teacher_pairs"
DEFAULT_MODEL_ROOT = Path(os.environ.get("ZIMAGE_MODEL_ROOT", "/workspace/z-image-models"))
DEFAULT_SIGLIP2_PATH = Path(
    os.environ.get(
        "ZIMAGE_SIGLIP2_PATH",
        str(DEFAULT_MODEL_ROOT / "SigLIP2-G384" / "model.safetensors"),
    )
)
DEFAULT_DINOV3_PATH = Path(
    os.environ.get(
        "ZIMAGE_DINOV3_PATH",
        str(DEFAULT_MODEL_ROOT / "DINOv3-7B" / "model.safetensors"),
    )
)


@dataclass(frozen=True)
class TeacherSample:
    image_paths: list[Path]
    teacher_embedding: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TeacherEncoderBundle:
    siglip2_image_encoder: Any
    dinov3_image_encoder: Any
    device: str | torch.device
    torch_dtype: torch.dtype
    preprocess: Callable[[Image.Image], Image.Image] = field(
        default_factory=lambda: ImageCropAndResize(height=1024, width=1024)
    )

    def encode_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        embeddings: list[torch.Tensor] = []
        for image in images:
            image = self.preprocess(image)
            siglip_embedding = self.siglip2_image_encoder(
                image,
                torch_dtype=self.torch_dtype,
                device=self.device,
            )
            dino_embedding = self.dinov3_image_encoder(
                image,
                torch_dtype=self.torch_dtype,
                device=self.device,
            )
            embeddings.append(
                torch.cat([siglip_embedding, dino_embedding], dim=-1)
                .squeeze(0)
                .to(dtype=self.torch_dtype)
            )
        return torch.stack(embeddings)


def save_teacher_sample(sample: TeacherSample, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    embedding_path = out_dir / EMBEDDING_FILENAME
    save_file({EMBEDDING_KEY: sample.teacher_embedding.detach().cpu()}, str(embedding_path))

    manifest = {
        "version": 1,
        "image_paths": [str(path) for path in sample.image_paths],
        "metadata": sample.metadata,
        "teacher_embedding_file": EMBEDDING_FILENAME,
        "teacher_embedding_key": EMBEDDING_KEY,
    }
    (out_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_teacher_sample(out_dir: Path) -> TeacherSample:
    manifest_path = out_dir / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    embedding_path = out_dir / manifest["teacher_embedding_file"]
    embedding_key = manifest.get("teacher_embedding_key", EMBEDDING_KEY)
    tensor = load_file(str(embedding_path), device="cpu")[embedding_key]

    return TeacherSample(
        image_paths=[Path(path) for path in manifest["image_paths"]],
        teacher_embedding=tensor,
        metadata=manifest["metadata"],
    )


def build_teacher_model_configs(siglip2_path: Path, dinov3_path: Path) -> list[ModelConfig]:
    return [
        ModelConfig(path=str(siglip2_path)),
        ModelConfig(path=str(dinov3_path)),
    ]


def load_teacher_encoder_bundle(
    siglip2_path: Path,
    dinov3_path: Path,
    *,
    device: str | torch.device,
    torch_dtype: torch.dtype,
) -> TeacherEncoderBundle:
    loader = BasePipeline(device=device, torch_dtype=torch_dtype)
    model_pool = loader.download_and_load_models(build_teacher_model_configs(siglip2_path, dinov3_path))

    siglip2_image_encoder = model_pool.fetch_model("siglip2_image_encoder")
    dinov3_image_encoder = model_pool.fetch_model("dinov3_image_encoder")
    if siglip2_image_encoder is None or dinov3_image_encoder is None:
        raise RuntimeError("Failed to load the teacher encoders")

    return TeacherEncoderBundle(
        siglip2_image_encoder=siglip2_image_encoder,
        dinov3_image_encoder=dinov3_image_encoder,
        device=device,
        torch_dtype=torch_dtype,
    )


def load_reference_images(image_paths: Sequence[Path]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            images.append(image.convert("RGB"))
    return images


def encode_teacher_embedding(images: Sequence[Image.Image], bundle: TeacherEncoderBundle) -> torch.Tensor:
    if not images:
        raise ValueError("At least one reference image is required")
    return bundle.encode_images(images).mean(dim=0)


def build_teacher_sample(
    image_paths: Sequence[Path],
    bundle: TeacherEncoderBundle,
    metadata: dict[str, Any],
) -> TeacherSample:
    images = load_reference_images(image_paths)
    teacher_embedding = encode_teacher_embedding(images, bundle)
    return TeacherSample(
        image_paths=[Path(path) for path in image_paths],
        teacher_embedding=teacher_embedding,
        metadata=metadata,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zimage-distill-pairs",
        description="Generate teacher pairs for zimage distillation.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root(),
        help="Repository root used to resolve relative paths.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where teacher pairs are written.",
    )
    parser.add_argument(
        "--sample-name",
        type=str,
        default="sample_0001",
        help="Folder name for the teacher sample.",
    )
    parser.add_argument(
        "--image-path",
        action="append",
        default=[],
        type=Path,
        help="Reference image path. May be repeated.",
    )
    parser.add_argument(
        "--metadata-json",
        type=str,
        default="{}",
        help="JSON metadata to store alongside the sample.",
    )
    parser.add_argument(
        "--siglip2-path",
        type=Path,
        default=DEFAULT_SIGLIP2_PATH,
        help="Path to the SigLIP2-G384 weights.",
    )
    parser.add_argument(
        "--dinov3-path",
        type=Path,
        default=DEFAULT_DINOV3_PATH,
        help="Path to the DINOv3-7B weights.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device used for the teacher encoders.",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Torch dtype used for the teacher encoders.",
    )
    return parser


def _parse_torch_dtype(name: str) -> torch.dtype:
    return getattr(torch, name)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.image_path:
        parser.error("at least one --image-path is required")

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = args.repo_root / output_dir

    metadata = json.loads(args.metadata_json)
    bundle = load_teacher_encoder_bundle(
        args.siglip2_path,
        args.dinov3_path,
        device=args.device,
        torch_dtype=_parse_torch_dtype(args.torch_dtype),
    )
    sample = build_teacher_sample(args.image_path, bundle, metadata)
    save_teacher_sample(sample, output_dir / args.sample_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
