from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from zimage_distill.style_embed import StyleEmbedding


class _FakeStyleEncoder:
    def __init__(self, backbone: str = "mobilenet_v3_small", embedding_dim: int = 256) -> None:
        self.backbone_name = backbone
        self.embedding_dim = embedding_dim
        self._call_count = 0

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[0] != 1:
            raise AssertionError(f"unexpected batch size: {images.shape[0]}")
        outputs = [torch.tensor([1.0, 2.0], dtype=torch.float32), torch.tensor([5.0, 6.0], dtype=torch.float32)]
        output = outputs[self._call_count]
        self._call_count += 1
        return output


@pytest.mark.parametrize(
    "reference_count, expected_vector, expected_vectors, expected_mode",
    [
        (1, torch.tensor([1.0, 2.0], dtype=torch.float32), None, "single"),
        (
            2,
            torch.tensor([3.0, 4.0], dtype=torch.float32),
            torch.tensor([[1.0, 2.0], [5.0, 6.0]], dtype=torch.float32),
            "mixed",
        ),
    ],
)
def test_style_embedding_export_main_saves_single_and_mixed_embeddings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reference_count: int,
    expected_vector: torch.Tensor,
    expected_vectors: torch.Tensor | None,
    expected_mode: str,
):
    from zimage_distill import style_embedding_export as export_module

    references = []
    for index in range(reference_count):
        path = tmp_path / f"ref_{index}.png"
        Image.new("RGB", (16, 16), color=(index * 32, 0, 0)).save(path)
        references.append(path)

    captured_paths: list[Path] = []

    def fake_load_reference_images(image_paths):
        captured_paths.extend(image_paths)
        return [Image.new("RGB", (16, 16), color=(index * 32, 0, 0)) for index, _ in enumerate(image_paths)]

    monkeypatch.setattr(export_module, "load_reference_images", fake_load_reference_images)
    monkeypatch.setattr(export_module, "StyleEncoder", _FakeStyleEncoder)

    output = tmp_path / "style_embedding.safetensors"
    argv = []
    for path in references:
        argv.extend(["--reference-image", str(path)])
    argv.extend(["--output", str(output), "--backbone", "mobilenet_v3_small", "--embedding-dim", "2"])

    exit_code = export_module.main(argv)

    assert exit_code == 0
    assert captured_paths == references

    loaded = StyleEmbedding.load(output)
    assert torch.equal(loaded.vector, expected_vector)
    if expected_vectors is None:
        assert loaded.vectors is None
    else:
        assert torch.equal(loaded.vectors, expected_vectors)
    assert loaded.mode == expected_mode
    assert loaded.source_images == [str(path) for path in references]
