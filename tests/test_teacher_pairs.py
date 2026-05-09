from pathlib import Path

import torch

from zimage_distill.teacher_pairs import (
    TeacherSample,
    load_teacher_sample,
    save_teacher_sample,
)


def test_teacher_sample_roundtrip_preserves_paths_tensor_and_metadata(tmp_path):
    sample = TeacherSample(
        image_paths=[Path("/images/source_a.png"), Path("/images/source_b.png")],
        teacher_embedding=torch.arange(12, dtype=torch.float32),
        metadata={"source": "runpod", "batch": 3, "flags": ["siglip2", "dinov3"]},
    )

    out_dir = tmp_path / "teacher_pair"
    save_teacher_sample(sample, out_dir)

    loaded = load_teacher_sample(out_dir)

    assert loaded.image_paths == sample.image_paths
    assert torch.equal(loaded.teacher_embedding, sample.teacher_embedding)
    assert loaded.metadata == sample.metadata
    assert (out_dir / "teacher_embedding.safetensors").exists()
    assert (out_dir / "manifest.json").exists()
