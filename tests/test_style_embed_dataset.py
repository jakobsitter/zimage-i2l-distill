from itertools import combinations
from pathlib import Path

import pytest
import torch
from PIL import Image

from zimage_distill.teacher_pairs import TeacherSample, save_teacher_sample



def _make_sample(root: Path, name: str, *, color_offset: int = 0, target_offset: int = 0) -> None:
    refs = []
    refs_dir = root / name / "refs"
    refs_dir.mkdir(parents=True)
    for i in range(4):
        path = refs_dir / f"ref_{i}.png"
        Image.new("RGB", (32, 32), (color_offset + i * 10, 0, 0)).save(path)
        refs.append(path)

    save_teacher_sample(
        TeacherSample(
            image_paths=refs,
            teacher_embedding=torch.arange(target_offset, target_offset + 8, dtype=torch.float32),
            metadata={"subject": name},
        ),
        root / name,
    )



def _image_red_values(images: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(image[0, 0, 0].item() * 255) for image in images)



def test_single_style_dataset_exposes_each_image_as_its_own_example(tmp_path: Path):
    from zimage_distill.style_embed_dataset import SingleStyleDataset

    _make_sample(tmp_path, "sample_0001")
    _make_sample(tmp_path, "sample_0002", color_offset=40, target_offset=100)

    dataset = SingleStyleDataset(tmp_path)

    assert len(dataset) == 8

    expected_red_values = [0, 10, 20, 30, 40, 50, 60, 70]
    expected_targets = [
        torch.arange(8, dtype=torch.float32),
        torch.arange(8, dtype=torch.float32),
        torch.arange(8, dtype=torch.float32),
        torch.arange(8, dtype=torch.float32),
        torch.arange(100, 108, dtype=torch.float32),
        torch.arange(100, 108, dtype=torch.float32),
        torch.arange(100, 108, dtype=torch.float32),
        torch.arange(100, 108, dtype=torch.float32),
    ]

    for index, (expected_red, expected_target) in enumerate(zip(expected_red_values, expected_targets, strict=True)):
        item = dataset[index]
        assert item["images"].shape == (1, 3, 224, 224)
        assert _image_red_values(item["images"]) == (expected_red,)
        assert torch.equal(item["target"], expected_target)
        assert item["mode"] == "single"



def test_fused_style_dataset_enumerates_all_three_reference_combinations(tmp_path: Path):
    from zimage_distill.style_embed_dataset import FusedStyleDataset

    _make_sample(tmp_path, "sample_0001")

    dataset = FusedStyleDataset(tmp_path, refs_per_item=3)

    assert len(dataset) == 4
    assert {
        _image_red_values(dataset[index]["images"])
        for index in range(len(dataset))
    } == set(combinations((0, 10, 20, 30), 3))
    assert all(dataset[index]["images"].shape == (3, 3, 224, 224) for index in range(len(dataset)))
    assert all(dataset[index]["mode"] == "fused" for index in range(len(dataset)))



def test_fused_style_dataset_enumerates_all_two_reference_combinations(tmp_path: Path):
    from zimage_distill.style_embed_dataset import FusedStyleDataset

    _make_sample(tmp_path, "sample_0001")

    dataset = FusedStyleDataset(tmp_path, refs_per_item=2)

    assert len(dataset) == 6
    assert {
        _image_red_values(dataset[index]["images"])
        for index in range(len(dataset))
    } == set(combinations((0, 10, 20, 30), 2))
    assert all(dataset[index]["images"].shape == (2, 3, 224, 224) for index in range(len(dataset)))
    assert all(dataset[index]["mode"] == "fused" for index in range(len(dataset)))


@pytest.mark.parametrize(
    ("dataset_class_name", "dataset_kwargs"),
    [
        ("SingleStyleDataset", {}),
        ("FusedStyleDataset", {"refs_per_item": 3}),
    ],
)
def test_style_datasets_build_indices_from_manifests_without_loading_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dataset_class_name: str,
    dataset_kwargs: dict[str, int],
):
    import zimage_distill.style_embed_dataset as style_embed_dataset

    _make_sample(tmp_path, "sample_0001")
    _make_sample(tmp_path, "sample_0002", color_offset=40, target_offset=100)

    getitem_calls: list[int] = []

    def fail_on_sample_load(self, index: int):
        getitem_calls.append(index)
        raise AssertionError("constructor should not materialize samples")

    monkeypatch.setattr(style_embed_dataset.TeacherEmbeddingDataset, "__getitem__", fail_on_sample_load)

    dataset_class = getattr(style_embed_dataset, dataset_class_name)
    dataset = dataset_class(tmp_path, **dataset_kwargs)

    assert getitem_calls == []
    assert len(dataset) > 0



def test_fused_style_dataset_keeps_combinations_within_each_sample(tmp_path: Path):
    from zimage_distill.style_embed_dataset import FusedStyleDataset

    _make_sample(tmp_path, "sample_0001")
    _make_sample(tmp_path, "sample_0002", color_offset=40, target_offset=100)

    dataset = FusedStyleDataset(tmp_path, refs_per_item=3)

    expected = [
        ("sample_0001", torch.arange(8, dtype=torch.float32), image_values)
        for image_values in combinations((0, 10, 20, 30), 3)
    ] + [
        ("sample_0002", torch.arange(100, 108, dtype=torch.float32), image_values)
        for image_values in combinations((40, 50, 60, 70), 3)
    ]

    assert len(dataset) == len(expected)
    actual = [
        (item["metadata"]["subject"], item["target"], _image_red_values(item["images"]))
        for item in (dataset[index] for index in range(len(dataset)))
    ]

    assert [subject for subject, _, _ in actual] == [subject for subject, _, _ in expected]
    assert [image_values for _, _, image_values in actual] == [image_values for _, _, image_values in expected]
    assert all(torch.equal(target, expected_target) for (_, target, _), (_, expected_target, _) in zip(actual, expected, strict=True))


@pytest.mark.parametrize("refs_per_item", [1, 4])
def test_fused_style_dataset_rejects_ref_counts_outside_bootstrap_range(refs_per_item: int, tmp_path: Path):
    from zimage_distill.style_embed_dataset import FusedStyleDataset

    _make_sample(tmp_path, "sample_0001")

    with pytest.raises(ValueError, match="refs_per_item must be 2 or 3"):
        FusedStyleDataset(tmp_path, refs_per_item=refs_per_item)



def test_teacher_embedding_dataset_resolves_repo_local_dataset_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from zimage_distill import dataset as dataset_mod

    teacher_root = tmp_path / "teacher_pairs"
    sample_dir = teacher_root / "sample_0001"
    refs_dir = tmp_path / "dataset_master_clean" / "sample_0001"
    refs_dir.mkdir(parents=True)
    sample_dir.mkdir(parents=True)

    image_path = refs_dir / "ref.png"
    Image.new("RGB", (32, 32), (255, 0, 0)).save(image_path)

    save_teacher_sample(
        TeacherSample(
            image_paths=[Path("/workspace/zimage-i2l-distill/../dataset_master_clean/sample_0001/ref.png")],
            teacher_embedding=torch.arange(8, dtype=torch.float32),
            metadata={"subject": "sample_0001"},
        ),
        sample_dir,
    )

    monkeypatch.setattr(dataset_mod, "repo_root", lambda: tmp_path)
    dataset = dataset_mod.TeacherEmbeddingDataset(teacher_root)

    item = dataset[0]
    assert item["images"].shape == (1, 3, 224, 224)
    assert item["metadata"] == {"subject": "sample_0001"}
    assert torch.equal(item["target"], torch.arange(8, dtype=torch.float32))
