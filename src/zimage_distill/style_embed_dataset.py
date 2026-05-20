from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Iterable

from .dataset import MANIFEST_FILENAME, TeacherEmbeddingDataset


IndexEntry = tuple[int, tuple[int, ...]]


def _image_count_for_sample(sample_dir: Path) -> int:
    manifest = json.loads((sample_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    return len(manifest["image_paths"])


def _build_index(
    sample_dirs: list[Path],
    image_groups_for_count: Callable[[int], Iterable[tuple[int, ...]]],
) -> list[IndexEntry]:
    index: list[IndexEntry] = []
    for sample_index, sample_dir in enumerate(sample_dirs):
        index.extend((sample_index, image_indices) for image_indices in image_groups_for_count(_image_count_for_sample(sample_dir)))
    return index


class SingleStyleDataset(TeacherEmbeddingDataset):
    def __init__(self, root_dir: Path):
        super().__init__(root_dir)
        self._index = _build_index(
            self.sample_dirs,
            lambda image_count: ((image_index,) for image_index in range(image_count)),
        )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_index, image_indices = self._index[index]
        item = super().__getitem__(sample_index)
        return {
            **item,
            "images": item["images"][list(image_indices)],
            "mode": "single",
        }


class FusedStyleDataset(TeacherEmbeddingDataset):
    def __init__(self, root_dir: Path, refs_per_item: int = 3):
        if refs_per_item not in {2, 3}:
            raise ValueError("refs_per_item must be 2 or 3")
        super().__init__(root_dir)
        self.refs_per_item = refs_per_item
        self._index = _build_index(
            self.sample_dirs,
            lambda image_count: combinations(range(image_count), self.refs_per_item),
        )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_index, image_indices = self._index[index]
        item = super().__getitem__(sample_index)
        return {
            **item,
            "images": item["images"][list(image_indices)],
            "mode": "fused",
        }
