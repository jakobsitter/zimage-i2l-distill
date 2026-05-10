from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file

from diffsynth.core import ModelConfig
from diffsynth.models.z_image_image2lora import ZImageImage2LoRAModel

I2L_MODEL_ID = "DiffSynth-Studio/Z-Image-i2L"
I2L_FILE_PATTERN = "model.safetensors"


def resolve_decoder_checkpoint(path: Path | str | None) -> Path:
    if path is not None:
        return Path(path)
    cfg = ModelConfig(model_id=I2L_MODEL_ID, origin_file_pattern=I2L_FILE_PATTERN)
    cfg.download_if_necessary()
    return Path(cfg.path)


def _detect_compress_dim(state_dict: dict) -> int:
    # proj_a.proj_in.weight has shape [compress_dim, 5632]
    for key, value in state_dict.items():
        if "proj_a.proj_in.weight" in key:
            return value.shape[0]
    return 64  # default fallback


class I2LDecoderAdapter:
    def __init__(self, checkpoint_path: Path | str | None = None, device: torch.device | str = "cpu"):
        self.device = torch.device(device)
        resolved = resolve_decoder_checkpoint(checkpoint_path)
        print(f"Loading i2L decoder from: {resolved}")
        state_dict = load_file(str(resolved), device="cpu")
        compress_dim = _detect_compress_dim(state_dict)
        print(f"  compress_dim={compress_dim}")
        self.decoder = ZImageImage2LoRAModel(compress_dim=compress_dim)
        self.decoder.load_state_dict(state_dict)
        self.decoder.to(self.device)
        self.decoder.eval()

    def predict_lora(self, embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            embedding = embedding.to(self.device)
            if embedding.dim() == 2 and embedding.shape[0] == 1:
                embedding = embedding.squeeze(0)
            return self.decoder(embedding)
