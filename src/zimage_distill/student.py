from __future__ import annotations

import torch
from torch import nn
from torchvision.models import mobilenet_v3_small

TEACHER_EMBEDDING_DIM = 5632
MOBILENET_SMALL_FEATURES = 576


class StudentImageEncoder(nn.Module):
    backbone_name = "mobilenet_v3_small"

    def __init__(self, embedding_dim: int = TEACHER_EMBEDDING_DIM):
        super().__init__()
        backbone = mobilenet_v3_small(weights=None)
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.head = nn.Linear(MOBILENET_SMALL_FEATURES, embedding_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() == 4:
            return self._encode_images(images).mean(dim=0)
        if images.dim() == 5:
            batch_size, num_refs = images.shape[:2]
            per_image = self._encode_images(images.reshape(batch_size * num_refs, *images.shape[2:]))
            pooled = per_image.reshape(batch_size, num_refs, -1).mean(dim=1)
            return pooled.squeeze(0) if batch_size == 1 else pooled
        raise ValueError("Expected images with shape [num_refs, 3, H, W] or [batch, num_refs, 3, H, W]")

    def _encode_images(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(images))
