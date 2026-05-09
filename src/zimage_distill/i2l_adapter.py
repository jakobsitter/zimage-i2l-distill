from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file

from diffsynth.models.z_image_image2lora import ZImageImage2LoRAModel


class I2LDecoderAdapter:
    def __init__(self, checkpoint_path: Path | str, device: torch.device | str = "cpu"):
        self.device = torch.device(device)
        self.decoder = ZImageImage2LoRAModel()
        state_dict = load_file(str(checkpoint_path), device="cpu")
        self.decoder.load_state_dict(state_dict)
        self.decoder.to(self.device)
        self.decoder.eval()

    def predict_lora(self, embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            embedding = embedding.to(self.device)
            if embedding.dim() == 2 and embedding.shape[0] == 1:
                embedding = embedding.squeeze(0)
            return self.decoder(embedding)
