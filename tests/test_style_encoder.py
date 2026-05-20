import pytest
import torch
from torch import nn

from zimage_distill.style_encoder import StyleEncoder


class _StubBackbone(nn.Module):
    def __init__(self, out_dim: int = 6) -> None:
        super().__init__()
        self.out_dim = out_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pooled = images.mean(dim=(-1, -2))
        repeats = (self.out_dim + pooled.shape[-1] - 1) // pooled.shape[-1]
        expanded = pooled.repeat(1, repeats)
        return expanded[:, : self.out_dim]


@pytest.fixture
def stub_backbone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("zimage_distill.student._build_backbone", lambda name: _StubBackbone())


@pytest.mark.usefixtures("stub_backbone")
def test_style_encoder_exposes_supported_backbones_and_preserves_choice() -> None:
    assert "mobilenet_v3_small" in StyleEncoder.SUPPORTED_BACKBONES
    assert "dinov3-vits16" in StyleEncoder.SUPPORTED_BACKBONES

    current_encoder = StyleEncoder(backbone="mobilenet_v3_small", embedding_dim=8)
    stronger_encoder = StyleEncoder(backbone="dinov3-vits16", embedding_dim=8)

    assert current_encoder.backbone_name == "mobilenet_v3_small"
    assert stronger_encoder.backbone_name == "dinov3-vits16"


@pytest.mark.usefixtures("stub_backbone")
def test_style_encoder_returns_normalized_embedding_for_single_image_input() -> None:
    encoder = StyleEncoder(backbone="mobilenet_v3_small", embedding_dim=8)

    images = torch.linspace(0.0, 1.0, steps=3 * 16 * 16, dtype=torch.float32).reshape(1, 3, 16, 16)

    embedding = encoder(images)

    assert embedding.shape == (8,)
    assert torch.isclose(torch.linalg.vector_norm(embedding), torch.tensor(1.0), atol=1e-5)


@pytest.mark.usefixtures("stub_backbone")
def test_style_encoder_aggregates_two_image_input_before_normalization() -> None:
    encoder = StyleEncoder(backbone="mobilenet_v3_small", embedding_dim=6)
    encoder.encoder = _StubBackbone(out_dim=6)
    encoder.head = torch.nn.Identity()

    images = torch.stack(
        [
            torch.tensor(
                [
                    [[1.0, 1.0], [1.0, 1.0]],
                    [[0.0, 0.0], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 0.0]],
                ],
                dtype=torch.float32,
            ),
            torch.tensor(
                [
                    [[0.0, 0.0], [0.0, 0.0]],
                    [[1.0, 1.0], [1.0, 1.0]],
                    [[0.0, 0.0], [0.0, 0.0]],
                ],
                dtype=torch.float32,
            ),
        ],
        dim=0,
    )

    embedding = encoder(images)
    expected = torch.tensor([0.5, 0.5, 0.0, 0.5, 0.5, 0.0], dtype=torch.float32)
    expected = expected / torch.linalg.vector_norm(expected)

    assert torch.allclose(embedding, expected)
    assert torch.isclose(torch.linalg.vector_norm(embedding), torch.tensor(1.0), atol=1e-5)


@pytest.mark.usefixtures("stub_backbone")
def test_style_encoder_returns_normalized_embedding_for_three_image_input() -> None:
    encoder = StyleEncoder(backbone="mobilenet_v3_small", embedding_dim=12)

    images = torch.stack(
        [
            torch.zeros(3, 8, 8, dtype=torch.float32),
            torch.full((3, 8, 8), 0.5, dtype=torch.float32),
            torch.ones(3, 8, 8, dtype=torch.float32),
        ],
        dim=0,
    )

    embedding = encoder(images)

    assert embedding.shape == (12,)
    assert torch.isclose(torch.linalg.vector_norm(embedding), torch.tensor(1.0), atol=1e-5)


@pytest.mark.usefixtures("stub_backbone")
def test_style_encoder_rejects_inputs_with_unsupported_reference_count() -> None:
    encoder = StyleEncoder(backbone="mobilenet_v3_small", embedding_dim=8)

    images = torch.rand(4, 3, 8, 8)

    with pytest.raises(ValueError, match="1, 2, or 3 reference images"):
        encoder(images)
