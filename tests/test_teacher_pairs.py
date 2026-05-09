from pathlib import Path

import torch
from PIL import Image

from zimage_distill.teacher_pairs import (
    TeacherEncoderBundle,
    TeacherSample,
    build_teacher_sample,
    encode_teacher_embedding,
    load_teacher_sample,
    save_teacher_sample,
)


class FakeEncoder:
    def __init__(self, outputs: list[list[float]]):
        self._outputs = [torch.tensor(row, dtype=torch.float32).unsqueeze(0) for row in outputs]
        self._calls = 0

    def __call__(self, image, torch_dtype, device):
        output = self._outputs[self._calls]
        self._calls += 1
        return output.to(dtype=torch_dtype, device=device)


def test_encode_teacher_embedding_averages_per_image_outputs():
    bundle = TeacherEncoderBundle(
        siglip2_image_encoder=FakeEncoder([[1.0, 2.0], [5.0, 6.0]]),
        dinov3_image_encoder=FakeEncoder([[10.0, 20.0], [30.0, 40.0]]),
        device="cpu",
        torch_dtype=torch.float32,
        preprocess=lambda image: image,
    )

    embedding = encode_teacher_embedding([object(), object()], bundle)

    assert torch.equal(embedding, torch.tensor([3.0, 4.0, 20.0, 30.0]))


def test_teacher_sample_roundtrip_preserves_paths_tensor_and_metadata(tmp_path: Path):
    image_a = tmp_path / "source_a.png"
    image_b = tmp_path / "source_b.png"
    Image.new("RGB", (8, 8), (255, 0, 0)).save(image_a)
    Image.new("RGB", (8, 8), (0, 255, 0)).save(image_b)

    bundle = TeacherEncoderBundle(
        siglip2_image_encoder=FakeEncoder([[1.0, 2.0], [5.0, 6.0]]),
        dinov3_image_encoder=FakeEncoder([[10.0, 20.0], [30.0, 40.0]]),
        device="cpu",
        torch_dtype=torch.float32,
    )

    sample = build_teacher_sample(
        [image_a, image_b],
        bundle,
        {"source": "runpod", "batch": 3, "flags": ["siglip2", "dinov3"]},
    )

    out_dir = tmp_path / "teacher_pair"
    save_teacher_sample(sample, out_dir)

    loaded = load_teacher_sample(out_dir)

    assert loaded.image_paths == [image_a, image_b]
    assert torch.equal(loaded.teacher_embedding, torch.tensor([3.0, 4.0, 20.0, 30.0]))
    assert loaded.metadata == {"source": "runpod", "batch": 3, "flags": ["siglip2", "dinov3"]}
    assert (out_dir / "teacher_embedding.safetensors").exists()
    assert (out_dir / "manifest.json").exists()
