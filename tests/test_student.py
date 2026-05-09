from pathlib import Path

import torch
from PIL import Image

from zimage_distill.dataset import TeacherEmbeddingDataset
from zimage_distill.student import StudentImageEncoder
from zimage_distill.teacher_pairs import TeacherSample, save_teacher_sample


def _make_teacher_sample(sample_dir: Path) -> None:
    refs_dir = sample_dir / "refs"
    refs_dir.mkdir(parents=True)

    image_paths = []
    for index, color in enumerate([(255, 0, 0), (0, 255, 0)], start=1):
        image_path = refs_dir / f"ref_{index}.png"
        Image.new("RGB", (32 + index, 40 + index), color).save(image_path)
        image_paths.append(image_path)

    save_teacher_sample(
        TeacherSample(
            image_paths=image_paths,
            teacher_embedding=torch.arange(5632, dtype=torch.float32),
            metadata={"source": "runpod", "batch": 3},
        ),
        sample_dir,
    )


def test_teacher_embedding_dataset_loads_images_target_and_metadata_and_student_outputs_teacher_dim(
    tmp_path: Path,
):
    samples_root = tmp_path / "teacher_pairs"
    sample_dir = samples_root / "sample_0001"
    _make_teacher_sample(sample_dir)

    dataset = TeacherEmbeddingDataset(samples_root)
    item = dataset[0]

    assert len(dataset) == 1
    assert set(item) == {"images", "target", "metadata"}
    assert item["images"].shape == (2, 3, 224, 224)
    assert item["images"].dtype == torch.float32
    assert torch.all(item["images"] >= 0)
    assert torch.all(item["images"] <= 1)
    assert torch.equal(item["target"], torch.arange(5632, dtype=torch.float32))
    assert item["metadata"] == {"source": "runpod", "batch": 3}

    student = StudentImageEncoder()
    output = student(item["images"])

    assert output.shape == (5632,)
