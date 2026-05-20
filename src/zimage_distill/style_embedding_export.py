from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize, to_tensor

from .style_embed import StyleEmbedding, fuse_style_vectors
from .style_encoder import StyleEncoder
from .teacher_pairs import load_reference_images

_IMAGE_SIZE = 224


def _image_to_tensor(image: Image.Image) -> torch.Tensor:
    resized = resize(
        image.convert("RGB"),
        [_IMAGE_SIZE, _IMAGE_SIZE],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    return to_tensor(resized)


def _encode_reference_image(encoder: StyleEncoder, image: Image.Image, device: str) -> torch.Tensor:
    image_tensor = _image_to_tensor(image).unsqueeze(0).to(device)
    embedding = encoder(image_tensor)
    return embedding.squeeze(0) if embedding.ndim == 2 else embedding


def build_style_embedding(
    reference_image_paths: Sequence[Path],
    *,
    backbone: str,
    embedding_dim: int,
    device: str,
    encoder: StyleEncoder | None = None,
) -> StyleEmbedding:
    if not reference_image_paths:
        raise ValueError("at least one --reference-image is required")

    reference_images = load_reference_images(reference_image_paths)
    style_encoder = StyleEncoder(backbone=backbone, embedding_dim=embedding_dim) if encoder is None else encoder
    vectors = torch.stack([_encode_reference_image(style_encoder, image, device) for image in reference_images])
    source_images = [str(path) for path in reference_image_paths]

    if len(reference_image_paths) == 1:
        return StyleEmbedding(vector=vectors[0], source_images=source_images, mode="single", metadata={})

    return StyleEmbedding(
        vector=fuse_style_vectors(vectors),
        vectors=vectors,
        source_images=source_images,
        mode="mixed",
        metadata={},
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zimage-distill-style-embedding-export",
        description="Export a style embedding from one or more reference images.",
    )
    parser.add_argument(
        "--reference-image",
        action="append",
        dest="reference_images",
        required=True,
        type=Path,
        help="Reference image path. Repeat for multiple inputs.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output style embedding path.")
    parser.add_argument("--backbone", default="mobilenet_v3_small", help="Style encoder backbone name.")
    parser.add_argument("--embedding-dim", type=int, default=256, help="Style embedding width.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    encoder = StyleEncoder(backbone=args.backbone, embedding_dim=args.embedding_dim).to(args.device)
    encoder.eval()
    embedding = build_style_embedding(
        args.reference_images,
        backbone=args.backbone,
        embedding_dim=args.embedding_dim,
        device=args.device,
        encoder=encoder,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    embedding.save(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
