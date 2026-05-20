from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from zimage_distill.teacher_pairs import TeacherSample, save_teacher_sample


class TinyFrozenStyleEncoder(torch.nn.Module):
    last_instance: "TinyFrozenStyleEncoder | None" = None

    def __init__(self, backbone: str = "dual-dinov3s-siglip2b", embedding_dim: int = 4) -> None:
        super().__init__()
        TinyFrozenStyleEncoder.last_instance = self
        self.backbone_name = backbone
        self.embedding_dim = embedding_dim
        self.weight = torch.nn.Parameter(torch.linspace(0.1, 0.4, steps=embedding_dim, dtype=torch.float32))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.weight


class TinyTrainableAdapter(torch.nn.Module):
    last_instance: "TinyTrainableAdapter | None" = None

    def __init__(self, embedding_dim: int, hidden_size: int, num_style_tokens: int = 4) -> None:
        super().__init__()
        TinyTrainableAdapter.last_instance = self
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_style_tokens = num_style_tokens
        self.projection = torch.nn.Linear(embedding_dim, hidden_size * num_style_tokens)
        self.initial_projection_weight = self.projection.weight.detach().clone()

    def forward(self, vector: torch.Tensor) -> torch.Tensor:
        projected = self.projection(vector)
        if projected.ndim == 1:
            return projected.view(self.num_style_tokens, self.hidden_size)
        return projected.view(projected.shape[0], self.num_style_tokens, self.hidden_size)


def _make_teacher_pairs_root(root: Path, num_refs: int = 2) -> Path:
    sample_dir = root / "sample_0001"
    refs_dir = sample_dir / "refs"
    refs_dir.mkdir(parents=True)

    image_paths = []
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for index, color in enumerate(colors[:num_refs], start=1):
        image_path = refs_dir / f"ref_{index}.png"
        Image.new("RGB", (32, 32), color).save(image_path)
        image_paths.append(image_path)

    save_teacher_sample(
        TeacherSample(
            image_paths=image_paths,
            teacher_embedding=torch.arange(5632, dtype=torch.float32),
            metadata={"source": "test"},
        ),
        sample_dir,
    )
    return root


def test_train_style_bridge_token_diversity_loss_penalizes_similar_tokens():
    from zimage_distill.train_style_bridge import token_diversity_loss

    similar_tokens = torch.tensor(
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
        dtype=torch.float32,
    )
    diverse_tokens = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, -1.0, 0.0]],
        dtype=torch.float32,
    )

    assert token_diversity_loss(similar_tokens) > token_diversity_loss(diverse_tokens)
    assert token_diversity_loss(diverse_tokens).item() >= 0.0


def test_train_style_bridge_keeps_runtime_entry_points_importable():
    from zimage_distill import direct_generate, generate
    from zimage_distill.comfyui_style_payload_node import NODE_CLASS_MAPPINGS

    assert hasattr(generate, "main")
    assert hasattr(direct_generate, "main")
    assert "ZImageLoadStyleTokenPayload" in NODE_CLASS_MAPPINGS
    assert "ZImageApplyStyleTokenPayload" in NODE_CLASS_MAPPINGS


def test_train_style_bridge_trims_teacher_samples_to_supported_reference_count(monkeypatch, tmp_path: Path):
    from zimage_distill import train_style_bridge as mod

    class CountingStyleEncoder(TinyFrozenStyleEncoder):
        calls = 0

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            assert images.shape[0] == 1
            CountingStyleEncoder.calls += 1
            return super().forward(images)

    monkeypatch.setattr(mod, "StyleEncoder", CountingStyleEncoder)
    monkeypatch.setattr(mod, "StyleTokenAdapter", TinyTrainableAdapter)

    data_root = _make_teacher_pairs_root(tmp_path / "teacher_pairs", num_refs=4)
    output_dir = tmp_path / "checkpoints"

    exit_code = mod.main(
        [
            "--data-dir",
            str(data_root),
            "--output-dir",
            str(output_dir),
            "--epochs",
            "1",
            "--embedding-dim",
            "4",
            "--hidden-size",
            "2560",
            "--num-style-tokens",
            "4",
            "--device",
            "cpu",
        ]
    )

    assert exit_code == 0
    assert CountingStyleEncoder.calls == 3


def test_train_style_bridge_accepts_custom_bridge_shape(monkeypatch, tmp_path: Path):
    from zimage_distill import train_style_bridge as mod

    monkeypatch.setattr(mod, "StyleEncoder", TinyFrozenStyleEncoder)
    monkeypatch.setattr(mod, "StyleTokenAdapter", TinyTrainableAdapter)

    data_root = _make_teacher_pairs_root(tmp_path / "teacher_pairs")
    output_dir = tmp_path / "checkpoints"

    exit_code = mod.main(
        [
            "--data-dir",
            str(data_root),
            "--output-dir",
            str(output_dir),
            "--epochs",
            "1",
            "--embedding-dim",
            "4",
            "--hidden-size",
            "1408",
            "--num-style-tokens",
            "1",
            "--device",
            "cpu",
        ]
    )

    checkpoint = torch.load(output_dir / "style_bridge.pt", map_location="cpu", weights_only=False)

    assert exit_code == 0
    assert checkpoint["hidden_size"] == 1408
    assert checkpoint["num_style_tokens"] == 1
    assert checkpoint["adapter_kind"] == "ip_adapter"
    assert checkpoint["supports_multi_image"] is True


def test_train_style_bridge_writes_an_ip_adapter_checkpoint_by_default(monkeypatch, tmp_path: Path):
    from zimage_distill import train_style_bridge as mod

    monkeypatch.setattr(mod, "StyleEncoder", TinyFrozenStyleEncoder)
    monkeypatch.setattr(mod, "StyleTokenAdapter", TinyTrainableAdapter)

    data_root = _make_teacher_pairs_root(tmp_path / "teacher_pairs")
    output_dir = tmp_path / "checkpoints"

    exit_code = mod.main(
        [
            "--data-dir",
            str(data_root),
            "--output-dir",
            str(output_dir),
            "--epochs",
            "1",
            "--embedding-dim",
            "4",
            "--hidden-size",
            "2560",
            "--device",
            "cpu",
        ]
    )

    checkpoint_path = output_dir / "style_bridge.pt"
    encoder = TinyFrozenStyleEncoder.last_instance
    adapter = TinyTrainableAdapter.last_instance

    assert exit_code == 0
    assert checkpoint_path.exists()
    assert encoder is not None
    assert adapter is not None
    assert encoder.backbone_name == "dual-dinov3s-siglip2b"
    assert encoder.training is False
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    assert any(parameter.requires_grad for parameter in adapter.parameters())

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["supervision"] == "teacher_embedding"
    assert checkpoint["backbone"] == "dual-dinov3s-siglip2b"
    assert checkpoint["embedding_dim"] == 4
    assert checkpoint["hidden_size"] == 2560
    assert checkpoint["adapter_kind"] == "ip_adapter"
    assert adapter.hidden_size == 2560
    assert adapter.num_style_tokens == 4
    assert checkpoint["num_style_tokens"] == 4
    assert checkpoint["training_data_path"] == str(data_root)
    assert checkpoint["epochs"] == 1
    assert checkpoint["lr"] == 1e-3
    assert checkpoint["supports_multi_image"] is True
    assert "style_encoder_state_dict" not in checkpoint
    assert set(checkpoint["style_token_adapter_state_dict"]) == {"projection.weight", "projection.bias"}
    assert set(checkpoint["image_gate_state_dict"]) == {"projection.weight", "projection.bias"}
    assert checkpoint["style_token_adapter_state_dict"]["projection.weight"].shape == (2560 * 4, 4)
    assert not torch.equal(adapter.projection.weight.detach(), adapter.initial_projection_weight)
