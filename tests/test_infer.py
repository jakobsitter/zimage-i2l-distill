from pathlib import Path

import torch
from safetensors.torch import load_file


def test_emit_lora_writes_nonempty_safetensors_file(tmp_path: Path):
    from zimage_distill.infer import emit_lora

    class FakeDecoder:
        def predict_lora(self, embedding: torch.Tensor) -> dict[str, torch.Tensor]:
            assert torch.equal(embedding, torch.tensor([1.0, 2.0, 3.0]))
            return {
                "layers.0.attention.to_q.lora_A.default.weight": torch.ones(2, 3),
                "layers.0.attention.to_q.lora_B.default.weight": torch.zeros(4, 2),
            }

    out_path = tmp_path / "lora.safetensors"

    emit_lora(torch.tensor([1.0, 2.0, 3.0]), FakeDecoder(), out_path)

    tensors = load_file(out_path)
    assert tensors
    assert set(tensors) == {
        "layers.0.attention.to_q.lora_A.default.weight",
        "layers.0.attention.to_q.lora_B.default.weight",
    }
    assert torch.equal(tensors["layers.0.attention.to_q.lora_A.default.weight"], torch.ones(2, 3))
    assert torch.equal(tensors["layers.0.attention.to_q.lora_B.default.weight"], torch.zeros(4, 2))
