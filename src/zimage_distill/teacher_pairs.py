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
ENCODER_HF_REPO = "DiffSynth-Studio/General-Image-Encoders"


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
        preprocessed = [self.preprocess(image) for image in images]

        self.siglip2_image_encoder.to(self.device)
        with torch.no_grad():
            siglip_embeddings = [
                self.siglip2_image_encoder(img, torch_dtype=self.torch_dtype, device=self.device)
                for img in preprocessed
            ]
        self.siglip2_image_encoder.to("cpu")
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()

        self.dinov3_image_encoder.to(self.device)
        with torch.no_grad():
            dino_embeddings = [
                self.dinov3_image_encoder(img, torch_dtype=self.torch_dtype, device=self.device)
                for img in preprocessed
            ]
        self.dinov3_image_encoder.to("cpu")
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()

        return torch.stack([
            torch.cat([s, d], dim=-1).squeeze(0).to(dtype=self.torch_dtype)
            for s, d in zip(siglip_embeddings, dino_embeddings)
        ])


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


def _model_config_for(local_path: Path, hf_pattern: str) -> ModelConfig:
    if local_path.exists():
        return ModelConfig(path=str(local_path))
    return ModelConfig(model_id=ENCODER_HF_REPO, origin_file_pattern=hf_pattern)


def build_teacher_model_configs(siglip2_path: Path, dinov3_path: Path) -> list[ModelConfig]:
    return [
        _model_config_for(siglip2_path, "SigLIP2-G384/model.safetensors"),
        _model_config_for(dinov3_path, "DINOv3-7B/model.safetensors"),
    ]


def load_teacher_encoder_bundle(
    siglip2_path: Path,
    dinov3_path: Path,
    *,
    device: str | torch.device,
    torch_dtype: torch.dtype,
) -> TeacherEncoderBundle:
    # Load both models onto CPU so they don't both occupy GPU at once.
    # encode_images moves each model to `device` only during its forward pass.
    loader = BasePipeline(device="cpu", torch_dtype=torch_dtype)
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
        "--refs-root",
        type=Path,
        default=None,
        help=(
            "Root directory containing subject sub-folders. Each sub-folder "
            "is treated as one sample; its images become the reference images "
            "and the folder name becomes the sample name. Mutually exclusive "
            "with --image-path / --sample-name."
        ),
    )
    parser.add_argument(
        "--sample-name",
        type=str,
        default="sample_0001",
        help="Folder name for the teacher sample (single-sample mode).",
    )
    parser.add_argument(
        "--image-path",
        action="append",
        default=[],
        type=Path,
        help="Reference image path. May be repeated (single-sample mode).",
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


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _collect_subject_folders(refs_root: Path) -> list[Path]:
    return sorted(p for p in refs_root.iterdir() if p.is_dir())


def _images_in_folder(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.refs_root and args.image_path:
        parser.error("--refs-root and --image-path are mutually exclusive")

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = args.repo_root / output_dir

    metadata = json.loads(args.metadata_json)
    torch_dtype = _parse_torch_dtype(args.torch_dtype)

    bundle = load_teacher_encoder_bundle(
        args.siglip2_path,
        args.dinov3_path,
        device=args.device,
        torch_dtype=torch_dtype,
    )

    if args.refs_root:
        refs_root = args.refs_root
        if not refs_root.is_absolute():
            refs_root = args.repo_root / refs_root
        subject_folders = _collect_subject_folders(refs_root)
        if not subject_folders:
            parser.error(f"no sub-folders found under --refs-root {refs_root}")
        for folder in subject_folders:
            image_paths = _images_in_folder(folder)
            if not image_paths:
                print(f"skipping {folder.name}: no images found")
                continue
            print(f"processing {folder.name} ({len(image_paths)} images)…")
            sample = build_teacher_sample(image_paths, bundle, {**metadata, "subject": folder.name})
            save_teacher_sample(sample, output_dir / folder.name)
    else:
        if not args.image_path:
            parser.error("at least one --image-path is required (or use --refs-root)")
        sample = build_teacher_sample(args.image_path, bundle, metadata)
        save_teacher_sample(sample, output_dir / args.sample_name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
