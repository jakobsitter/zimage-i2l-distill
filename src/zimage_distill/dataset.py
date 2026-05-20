from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from safetensors.torch import load_file
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize, to_tensor
from torchvision.transforms import InterpolationMode

from .paths import repo_root

MANIFEST_FILENAME = "manifest.json"
EMBEDDING_FILENAME = "teacher_embedding.safetensors"
EMBEDDING_KEY = "teacher_embedding"
IMAGE_SIZE = 256


class TeacherEmbeddingDataset(Dataset[dict[str, Any]]):
    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        if (self.root_dir / MANIFEST_FILENAME).is_file():
            self.sample_dirs = [self.root_dir]
        else:
            self.sample_dirs = sorted(
                sample_dir
                for sample_dir in self.root_dir.iterdir()
                if sample_dir.is_dir() and (sample_dir / MANIFEST_FILENAME).is_file()
            )

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_dir = self.sample_dirs[index]
        manifest = json.loads((sample_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))

        images = [self._load_image(sample_dir, Path(image_path)) for image_path in manifest["image_paths"]]
        embedding_path = sample_dir / manifest.get("teacher_embedding_file", EMBEDDING_FILENAME)
        embedding_key = manifest.get("teacher_embedding_key", EMBEDDING_KEY)
        target = load_file(str(embedding_path), device="cpu")[embedding_key]

        return {
            "images": torch.stack(images),
            "target": target,
            "metadata": manifest["metadata"],
        }

    def _load_image(self, sample_dir: Path, image_path: Path) -> torch.Tensor:
        resolved_path = self._resolve_image_path(sample_dir, image_path)
        with Image.open(resolved_path) as image:
            rgb_image = image.convert("RGB")
            resized = resize(rgb_image, [IMAGE_SIZE, IMAGE_SIZE], interpolation=InterpolationMode.BILINEAR, antialias=True)
            return to_tensor(resized)

    def _resolve_image_path(self, sample_dir: Path, image_path: Path) -> Path:
        if not image_path.is_absolute():
            return sample_dir / image_path
        if image_path.is_file():
            return image_path

        parts = image_path.parts
        if "dataset_master_clean" in parts:
            suffix = Path(*parts[parts.index("dataset_master_clean") + 1 :])
            candidate = repo_root() / "dataset_master_clean" / suffix
            if candidate.is_file():
                return candidate
        return image_path
