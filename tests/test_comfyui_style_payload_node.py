from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from zimage_distill.style_bridge import StyleImageGate
from zimage_distill.style_embed import StyleEmbedding
from zimage_distill.style_token_adapter import StyleTokenAdapter


def test_comfyui_node_module_exports_expected_mappings():
    from zimage_distill.comfyui_style_payload_node import (
        NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS,
        ApplyStyleTokenPayloadNode,
        LoadStyleTokenPayloadNode,
    )

    assert NODE_CLASS_MAPPINGS["ZImageLoadStyleTokenPayload"] is LoadStyleTokenPayloadNode
    assert NODE_DISPLAY_NAME_MAPPINGS["ZImageLoadStyleTokenPayload"] == "ZImage Load Style Token Payload"
    assert NODE_CLASS_MAPPINGS["ZImageApplyStyleTokenPayload"] is ApplyStyleTokenPayloadNode
    assert NODE_DISPLAY_NAME_MAPPINGS["ZImageApplyStyleTokenPayload"] == "ZImage Apply Style Token Payload"


def test_comfyui_node_loads_embedding_and_bridge_checkpoint_and_returns_style_token_payload(tmp_path: Path):
    from zimage_distill.comfyui_style_payload_node import LoadStyleTokenPayloadNode

    style_path = tmp_path / "style_embedding.safetensors"
    embedding = StyleEmbedding(
        vector=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
        source_images=["ref.png"],
        mode="single",
        metadata={"tag": "demo"},
    )
    embedding.save(style_path)

    bridge_path = _write_bridge_checkpoint(tmp_path, embedding_dim=3, hidden_size=5, num_style_tokens=2)

    node = LoadStyleTokenPayloadNode()
    (payload,) = node.load_style_token_payload(str(style_path), str(bridge_path))

    assert payload["type"] == "zimage_style_token_payload"
    assert payload["style_embedding_path"] == str(style_path)
    assert payload["bridge_checkpoint_path"] == str(bridge_path)
    assert payload["bridge_embedding_dim"] == 3
    assert payload["bridge_hidden_size"] == 5
    assert payload["bridge_num_style_tokens"] == 2
    assert payload["bridge_adapter_kind"] == "style_token_bridge"
    assert payload["source_images"] == ["ref.png"]
    assert payload["mode"] == "single"
    assert payload["metadata"] == {"tag": "demo"}
    assert torch.equal(payload["style_embedding_vector"], embedding.vector)
    assert payload["style_tokens"].shape == (2, 5)


def test_comfyui_node_projects_style_tokens_to_requested_shape(tmp_path: Path):
    from zimage_distill.comfyui_style_payload_node import LoadStyleTokenPayloadNode

    style_path = tmp_path / "style_embedding.safetensors"
    StyleEmbedding(
        vector=torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32),
        source_images=[],
        mode="single",
        metadata={},
    ).save(style_path)
    bridge_path = _write_bridge_checkpoint(tmp_path, embedding_dim=4, hidden_size=7, num_style_tokens=3)

    node = LoadStyleTokenPayloadNode()
    (payload,) = node.load_style_token_payload(str(style_path), str(bridge_path))

    assert payload["style_tokens"].shape == (3, 7)


def test_comfyui_node_loads_multi_image_specs_with_per_image_strengths_and_gate(tmp_path: Path, monkeypatch):
    from zimage_distill import comfyui_style_payload_node as mod

    style_a = tmp_path / "style_a.safetensors"
    style_b = tmp_path / "style_b.safetensors"
    StyleEmbedding(vector=torch.tensor([1.0, 2.0], dtype=torch.float32), source_images=["a.png"], mode="single", metadata={}).save(style_a)
    StyleEmbedding(vector=torch.tensor([5.0, 6.0], dtype=torch.float32), source_images=["b.png"], mode="single", metadata={}).save(style_b)

    class FakeAdapter:
        def __call__(self, vectors: torch.Tensor) -> torch.Tensor:
            return torch.stack([vectors, vectors + 10.0], dim=1)

    class FakeGate:
        def __call__(self, vectors: torch.Tensor) -> torch.Tensor:
            return torch.zeros(vectors.shape[0], dtype=vectors.dtype)

    monkeypatch.setattr(
        mod,
        "load_style_bridge_checkpoint",
        lambda bridge_path: SimpleNamespace(adapter=FakeAdapter(), gate=FakeGate(), checkpoint={"supports_multi_image": True}),
    )

    style_spec = json.dumps(
        {
            "embeddings": [
                {"path": str(style_a), "strength": 1.0},
                {"path": str(style_b), "strength": 3.0},
            ]
        }
    )
    payload = mod.load_style_token_payload(style_spec, "ignored-bridge.pt")

    assert payload["style_tokens"].shape == (2, 2)
    assert torch.allclose(payload["style_token_weights"], torch.tensor([0.25, 0.75], dtype=torch.float32))
    assert torch.allclose(payload["style_tokens"], torch.tensor([[4.0, 5.0], [14.0, 15.0]], dtype=torch.float32))
    assert payload["style_token_blocks"].shape == (2, 2, 2)
    assert payload["style_embedding_vectors"].shape == (2, 2)


def test_comfyui_node_loads_ip_adapter_payload_with_per_image_strengths(tmp_path: Path):
    from zimage_distill.comfyui_style_payload_node import LoadStyleTokenPayloadNode

    style_a = tmp_path / "style_a.safetensors"
    style_b = tmp_path / "style_b.safetensors"
    StyleEmbedding(vector=torch.tensor([1.0, 2.0], dtype=torch.float32), source_images=["a.png"], mode="single", metadata={}).save(style_a)
    StyleEmbedding(vector=torch.tensor([5.0, 6.0], dtype=torch.float32), source_images=["b.png"], mode="single", metadata={}).save(style_b)

    bridge_path = _write_bridge_checkpoint(tmp_path, embedding_dim=2, hidden_size=2, num_style_tokens=2, supports_multi_image=True, adapter_kind="ip_adapter")
    style_spec = json.dumps(
        {
            "embeddings": [
                {"path": str(style_a), "strength": 1.0},
                {"path": str(style_b), "strength": 3.0},
            ]
        }
    )

    node = LoadStyleTokenPayloadNode()
    (payload,) = node.load_style_token_payload(style_spec, str(bridge_path))

    assert payload["bridge_adapter_kind"] == "ip_adapter"
    assert len(payload["ipadapter_kwargs_list"]) == 30
    assert torch.allclose(payload["ipadapter_kwargs_list"][0]["ip_hidden_states"], payload["style_tokens"])
    assert payload["ipadapter_kwargs_list"][0]["scale"] == 1.0


def test_comfyui_node_applies_style_tokens_to_zimage_conditioning_and_preserves_metadata(tmp_path: Path):
    from zimage_distill.comfyui_style_payload_node import ApplyStyleTokenPayloadNode, LoadStyleTokenPayloadNode

    style_path = tmp_path / "style_embedding.safetensors"
    StyleEmbedding(
        vector=torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32),
        source_images=["ref.png"],
        mode="single",
        metadata={"tag": "demo"},
    ).save(style_path)
    bridge_path = _write_bridge_checkpoint(tmp_path, embedding_dim=4, hidden_size=5, num_style_tokens=2)

    load_node = LoadStyleTokenPayloadNode()
    (payload,) = load_node.load_style_token_payload(str(style_path), str(bridge_path))

    pooled_output = torch.zeros(1, 5)
    conditioning = [[torch.ones(1, 4, 5, dtype=torch.float32), {"pooled_output": pooled_output}]]
    apply_node = ApplyStyleTokenPayloadNode()
    (updated,) = apply_node.apply_style_token_payload(conditioning, payload)

    assert updated[0][0].shape == (1, 6, 5)
    assert torch.equal(updated[0][0][:, :4], conditioning[0][0])
    assert torch.equal(updated[0][0][:, 4:], payload["style_tokens"].to(dtype=torch.float32).unsqueeze(0))
    assert torch.equal(updated[0][1]["pooled_output"], pooled_output)
    assert updated[0][1]["zimage_style_token_payload"] == {
        "type": "zimage_style_token_payload",
        "style_embedding_path": str(style_path),
        "bridge_checkpoint_path": str(bridge_path),
        "bridge_embedding_dim": 4,
        "bridge_hidden_size": 5,
        "bridge_num_style_tokens": 2,
        "source_images": ["ref.png"],
        "mode": "single",
        "metadata": {"tag": "demo"},
    }
    assert "zimage_style_token_payload" not in conditioning[0][1]


@pytest.mark.parametrize(
    ("path_builder", "match"),
    [
        (lambda tmp_path: tmp_path / "missing.safetensors", "does not exist"),
        (lambda tmp_path: _write_invalid_embedding_file(tmp_path), "missing style embedding metadata"),
    ],
)
def test_comfyui_node_rejects_missing_or_invalid_embedding_files(tmp_path: Path, path_builder, match: str):
    from zimage_distill.comfyui_style_payload_node import ApplyStyleTokenPayloadNode, LoadStyleTokenPayloadNode

    bridge_path = _write_bridge_checkpoint(tmp_path, embedding_dim=4, hidden_size=5, num_style_tokens=2)
    load_node = LoadStyleTokenPayloadNode()

    with pytest.raises(ValueError, match=match):
        load_node.load_style_token_payload(str(path_builder(tmp_path)), str(bridge_path))

    apply_node = ApplyStyleTokenPayloadNode()
    with pytest.raises(ValueError, match="style_token_payload must include style_tokens"):
        apply_node.apply_style_token_payload([[torch.ones(1, 1, 5), {}]], {})


@pytest.mark.parametrize(
    ("path_builder", "match"),
    [
        (lambda tmp_path: tmp_path / "missing-bridge.pt", "does not exist"),
        (lambda tmp_path: _write_invalid_bridge_checkpoint(tmp_path), "bridge checkpoint is missing"),
    ],
)
def test_comfyui_node_rejects_missing_or_invalid_bridge_files(tmp_path: Path, path_builder, match: str):
    from zimage_distill.comfyui_style_payload_node import LoadStyleTokenPayloadNode

    style_path = tmp_path / "style_embedding.safetensors"
    StyleEmbedding(
        vector=torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32),
        source_images=[],
        mode="single",
        metadata={},
    ).save(style_path)

    node = LoadStyleTokenPayloadNode()
    with pytest.raises(ValueError, match=match):
        node.load_style_token_payload(str(style_path), str(path_builder(tmp_path)))



def _write_invalid_embedding_file(tmp_path: Path) -> Path:
    path = tmp_path / "invalid_embedding.safetensors"
    save_file({"vector": torch.tensor([1.0, 2.0], dtype=torch.float32)}, str(path))
    return path



def _write_bridge_checkpoint(
    tmp_path: Path,
    *,
    embedding_dim: int,
    hidden_size: int,
    num_style_tokens: int,
    supports_multi_image: bool = False,
    adapter_kind: str = "style_token_bridge",
) -> Path:
    adapter = StyleTokenAdapter(embedding_dim=embedding_dim, hidden_size=hidden_size, num_style_tokens=num_style_tokens)
    path = tmp_path / f"bridge_{embedding_dim}_{hidden_size}_{num_style_tokens}.pt"
    checkpoint = {
        "supervision": "teacher_embedding",
        "backbone": "dual-dinov3s-siglip2b",
        "embedding_dim": embedding_dim,
        "hidden_size": hidden_size,
        "num_style_tokens": num_style_tokens,
        "training_data_path": str(tmp_path),
        "epochs": 1,
        "lr": 1e-3,
        "adapter_kind": adapter_kind,
        "style_token_adapter_state_dict": adapter.state_dict(),
    }
    if supports_multi_image:
        gate = StyleImageGate(embedding_dim=embedding_dim)
        checkpoint["supports_multi_image"] = True
        checkpoint["image_gate_state_dict"] = gate.state_dict()
    torch.save(checkpoint, path)
    return path



def _write_invalid_bridge_checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "invalid_bridge.pt"
    torch.save({"embedding_dim": 4}, path)
    return path
