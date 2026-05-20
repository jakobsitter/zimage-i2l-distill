from __future__ import annotations

from types import SimpleNamespace

import torch
from PIL import Image

from zimage_distill.style_embed import StyleEmbedding


def test_style_token_injection_unit_prepends_tokens_to_positive_prompt_embeddings():
    from zimage_distill.direct_generate import StyleTokenInjectionUnit

    unit = StyleTokenInjectionUnit(torch.tensor([[10.0, 11.0], [12.0, 13.0]], dtype=torch.float32))
    inputs_shared = {}
    inputs_posi = {"prompt_embeds": [torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)]}
    inputs_nega = {"prompt_embeds": [torch.tensor([[5.0, 6.0]], dtype=torch.float32)]}

    _, updated_posi, updated_nega = unit.process(object(), inputs_shared, inputs_posi, inputs_nega)

    assert torch.equal(
        updated_posi["prompt_embeds"][0],
        torch.tensor([[10.0, 11.0], [12.0, 13.0], [1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
    )
    assert torch.equal(updated_nega["prompt_embeds"][0], torch.tensor([[5.0, 6.0]], dtype=torch.float32))


def test_direct_generate_main_loads_multiple_embeddings_applies_strengths_and_saves_output(tmp_path, monkeypatch):
    from zimage_distill import direct_generate

    style_a = tmp_path / "style_a.safetensors"
    style_b = tmp_path / "style_b.safetensors"
    StyleEmbedding(vector=torch.tensor([1.0, 2.0], dtype=torch.float32), source_images=["a.png"], mode="single", metadata={}).save(style_a)
    StyleEmbedding(vector=torch.tensor([5.0, 6.0], dtype=torch.float32), source_images=["b.png"], mode="single", metadata={}).save(style_b)

    captured: dict[str, object] = {}

    class FakeAdapter:
        def __init__(self, embedding_dim: int, hidden_size: int, num_style_tokens: int = 4) -> None:
            captured["bridge_dims"] = (embedding_dim, hidden_size, num_style_tokens)

        def to(self, device):
            captured["bridge_device"] = device
            return self

        def __call__(self, vectors: torch.Tensor) -> torch.Tensor:
            captured["bridge_vectors"] = vectors.clone()
            return torch.stack([vectors, vectors + 10.0], dim=1)

    class FakeGate:
        def __call__(self, vectors: torch.Tensor) -> torch.Tensor:
            return torch.zeros(vectors.shape[0], dtype=vectors.dtype)

    class FakePipe:
        def __init__(self) -> None:
            self.device = torch.device("cpu")
            self.dit = SimpleNamespace(layers=[object(), object()])
            self.text_encoder = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(hidden_size=2)))
            self.units = []

        def __call__(self, **kwargs):
            captured["pipe_kwargs"] = kwargs
            assert "ipadapter_kwargs_list" in kwargs
            assert len(kwargs["ipadapter_kwargs_list"]) == 2
            first_payload = kwargs["ipadapter_kwargs_list"][0]
            assert torch.allclose(first_payload["ip_hidden_states"].to(dtype=torch.float32), torch.tensor([[4.0, 5.0], [14.0, 15.0]], dtype=torch.float32))
            return Image.new("RGB", (4, 4), color=(12, 34, 56))

    monkeypatch.setattr(
        direct_generate,
        "load_style_bridge_checkpoint",
        lambda bridge_path: SimpleNamespace(adapter=FakeAdapter(2, 2, 2), gate=FakeGate(), checkpoint={"supports_multi_image": True, "adapter_kind": "ip_adapter"}),
    )
    monkeypatch.setattr(direct_generate, "load_generation_pipeline", lambda device: FakePipe())

    output = tmp_path / "out.png"
    exit_code = direct_generate.main(
        [
            "--style-embedding",
            str(style_a),
            "--style-embedding",
            str(style_b),
            "--style-strength",
            "1.0",
            "--style-strength",
            "3.0",
            "--bridge-checkpoint",
            str(tmp_path / "bridge.pt"),
            "--prompt",
            "a style prompt",
            "--output",
            str(output),
            "--device",
            "cpu",
            "--seed",
            "7",
            "--cfg-scale",
            "1.5",
            "--steps",
            "9",
            "--sigma-shift",
            "3.0",
            "--negative-prompt",
            "not this",
        ]
    )

    assert exit_code == 0
    assert output.exists()
    assert captured["bridge_dims"] == (2, 2, 2)
    assert torch.equal(captured["bridge_vectors"], torch.tensor([[1.0, 2.0], [5.0, 6.0]], dtype=torch.float32))
    assert captured["pipe_kwargs"]["prompt"] == "a style prompt"
    assert captured["pipe_kwargs"]["negative_prompt"] == "not this"
    assert captured["pipe_kwargs"]["seed"] == 7
    assert captured["pipe_kwargs"]["cfg_scale"] == 1.5
    assert captured["pipe_kwargs"]["num_inference_steps"] == 9
    assert captured["pipe_kwargs"]["sigma_shift"] == 3.0
    assert len(captured["pipe_kwargs"]["ipadapter_kwargs_list"]) == 2
    for payload in captured["pipe_kwargs"]["ipadapter_kwargs_list"]:
        assert payload["scale"] == 1.0
        assert torch.allclose(
            payload["ip_hidden_states"].to(dtype=torch.float32),
            torch.tensor([[4.0, 5.0], [14.0, 15.0]], dtype=torch.float32),
        )
