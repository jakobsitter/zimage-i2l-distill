from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from .paths import data_dir, repo_root

MANIFEST_FILENAME = "manifest.json"
EMBEDDING_FILENAME = "teacher_embedding.safetensors"
EMBEDDING_KEY = "teacher_embedding"
DEFAULT_OUTPUT_DIR = data_dir() / "teacher_pairs"


@dataclass(frozen=True)
class TeacherSample:
    image_paths: list[Path]
    teacher_embedding: torch.Tensor
    metadata: dict[str, Any]


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zimage-distill-pairs",
        description=(
            "Generate teacher pairs for zimage distillation. The teacher adapter "
            "is intentionally minimal here and is wired in a later step."
        ),
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
        "--image-path",
        action="append",
        default=[],
        type=Path,
        help="Image path to include in a teacher sample. May be repeated.",
    )
    parser.add_argument(
        "--metadata-json",
        type=str,
        default="{}",
        help="JSON metadata to store alongside the sample.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = args.repo_root / output_dir

    metadata = json.loads(args.metadata_json)
    sample = TeacherSample(
        image_paths=list(args.image_path),
        teacher_embedding=torch.empty(0, dtype=torch.float32),
        metadata=metadata,
    )
    save_teacher_sample(sample, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
