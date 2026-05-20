from __future__ import annotations

import torch


def test_load_generation_pipeline_uses_low_vram_text_to_image_models(monkeypatch):
    from zimage_distill import generate as generate_module

    captured: dict[str, object] = {}

    class FakePipeline:
        @staticmethod
        def from_pretrained(**kwargs):
            captured.update(kwargs)
            return object()

    monkeypatch.setattr(generate_module, "ZImagePipeline", FakePipeline)

    pipe = generate_module.load_generation_pipeline("cpu")

    assert pipe is not None
    assert captured["torch_dtype"] == torch.bfloat16
    assert captured["device"] == torch.device("cpu")
    assert captured["vram_limit"] is None

    model_configs = captured["model_configs"]
    assert [config.path for config in model_configs] == [
        "/home/jakob/comfy/ComfyUI/models/diffusion_models/z_image_turbo_bf16.safetensors",
        "/home/jakob/comfy/ComfyUI/models/text_encoders/qwen_3_4b.safetensors",
        "/home/jakob/comfy/ComfyUI/models/vae/z_ae.safetensors",
    ]
    for config in model_configs:
        assert config.offload_device == "cpu"
        assert config.onload_device == "cpu"
        assert config.preparing_device == torch.device("cpu")
        assert config.computation_device == torch.device("cpu")

    tokenizer_config = captured["tokenizer_config"]
    assert tokenizer_config.model_id == "Tongyi-MAI/Z-Image-Turbo"
    assert tokenizer_config.origin_file_pattern == "tokenizer/"
