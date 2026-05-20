from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image

from zimage_distill.student import StudentImageEncoder
from zimage_distill.teacher_pairs import TeacherSample, save_teacher_sample


class _StubBackbone(torch.nn.Module):
    def __init__(self, out_dim: int = 6) -> None:
        super().__init__()
        self.out_dim = out_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pooled = images.mean(dim=(-1, -2))
        repeats = (self.out_dim + pooled.shape[-1] - 1) // pooled.shape[-1]
        expanded = pooled.repeat(1, repeats)
        return expanded[:, : self.out_dim]


def _make_teacher_pairs_root(root: Path) -> Path:
    sample_dir = root / "sample_0001"
    refs_dir = sample_dir / "refs"
    refs_dir.mkdir(parents=True)

    image_paths = []
    for index, color in enumerate([(255, 0, 0), (0, 255, 0)], start=1):
        image_path = refs_dir / f"ref_{index}.png"
        Image.new("RGB", (32, 32), color).save(image_path)
        image_paths.append(image_path)

    save_teacher_sample(
        TeacherSample(
            image_paths=image_paths,
            teacher_embedding=torch.arange(5632, dtype=torch.float32),
            metadata={"source": "runpod", "batch": 3},
        ),
        sample_dir,
    )
    return root


def test_train_step_runs_one_mse_update():
    from zimage_distill.train import train_step

    torch.manual_seed(0)
    model = StudentImageEncoder(embedding_dim=8)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    images = torch.rand(2, 3, 32, 32)
    target = torch.zeros(8)

    before = [parameter.detach().clone() for parameter in model.parameters()]
    loss = train_step(model, optimizer, images, target)
    after = list(model.parameters())

    assert isinstance(loss, float)
    assert loss >= 0.0
    assert any(not torch.equal(previous, current) for previous, current in zip(before, after))


def test_checkpoint_roundtrip_restores_student_in_eval_mode(tmp_path: Path):
    from zimage_distill.train import load_checkpoint, save_checkpoint

    torch.manual_seed(1)
    model = StudentImageEncoder(embedding_dim=16)
    for parameter in model.parameters():
        parameter.data.uniform_(-0.5, 0.5)

    checkpoint_path = tmp_path / "student.pt"
    save_checkpoint(model, checkpoint_path)
    loaded = load_checkpoint(checkpoint_path)

    assert loaded.training is False
    assert loaded.head[-1].out_features == 16
    assert loaded.backbone_name == model.backbone_name
    for expected, actual in zip(model.state_dict().values(), loaded.state_dict().values()):
        assert torch.equal(expected, actual)


def test_checkpoint_loader_accepts_legacy_linear_head_state_dict(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from zimage_distill import student as student_module
    from zimage_distill.train import load_checkpoint

    monkeypatch.setattr(student_module, "_build_backbone", lambda name: _StubBackbone())

    model = StudentImageEncoder(embedding_dim=16)
    legacy_state = {name: tensor.clone() for name, tensor in model.state_dict().items() if not name.startswith("head.")}
    legacy_state["head.weight"] = torch.randn(16, model.encoder.out_dim)
    legacy_state["head.bias"] = torch.randn(16)

    checkpoint_path = tmp_path / "legacy_student.pt"
    torch.save(
        {
            "state_dict": legacy_state,
            "backbone_name": model.backbone_name,
            "embedding_dim": 16,
        },
        checkpoint_path,
    )

    loaded = load_checkpoint(checkpoint_path)
    images = torch.rand(2, 3, 32, 32)
    embedding = loaded(images)

    assert loaded.training is False
    assert isinstance(loaded.head, torch.nn.Linear)
    assert loaded.head.out_features == 16
    assert embedding.shape == (16,)


def test_cli_writes_checkpoint_and_manifest_with_hard_coded_paths(monkeypatch, tmp_path: Path):
    from zimage_distill import train as train_module

    data_root = tmp_path / "data"
    checkpoints_root = tmp_path / "checkpoints"
    teacher_pairs_root = _make_teacher_pairs_root(data_root / "teacher_pairs")

    class TinyStudent(torch.nn.Module):
        def __init__(self, backbone: str = "mobilenet_v3_small", embedding_dim: int = 5632):
            super().__init__()
            self.backbone_name = backbone
            self.embedding_dim = embedding_dim
            self.head = torch.nn.Linear(1, embedding_dim)
            self.register_buffer("_dummy", torch.ones(1, 1))

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return self.head(self._dummy).squeeze(0)

    monkeypatch.setattr(train_module, "data_dir", lambda: data_root)
    monkeypatch.setattr(train_module, "checkpoints_dir", lambda: checkpoints_root)
    monkeypatch.setattr(train_module, "StudentImageEncoder", TinyStudent)

    exit_code = train_module.main([])

    checkpoint_path = checkpoints_root / "student_mobilenet_v3_small.pt"
    manifest_path = checkpoints_root / "student_mobilenet_v3_small_manifest.json"

    assert exit_code == 0
    assert checkpoint_path.exists()
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest == {
        "backbone_name": "mobilenet_v3_small",
        "embedding_dim": 5632,
        "training_data_path": str(teacher_pairs_root),
        "epochs": 10,
        "lr": 0.001,
    }
