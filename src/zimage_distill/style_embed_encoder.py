from __future__ import annotations

import torch
from torch.nn import functional as F

from .student import BACKBONE_CONFIGS, StudentImageEncoder


class StyleEncoder(StudentImageEncoder):
    """Encode 1-3 reference images into one normalized style vector.

    The output is a single embedding for downstream mixing, and the inherited
    StudentImageEncoder metadata/mode behavior stays unchanged.
    """

    SUPPORTED_BACKBONES = tuple(BACKBONE_CONFIGS)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self._validate_num_refs(self._num_refs(images))
        embedding = super().forward(images)
        return F.normalize(embedding, dim=-1, eps=1e-6)

    @staticmethod
    def _num_refs(images: torch.Tensor) -> int:
        if images.dim() == 4:
            return images.shape[0]
        if images.dim() == 5:
            return images.shape[1]
        raise ValueError("Expected images with shape [num_refs, 3, H, W] or [batch, num_refs, 3, H, W]")

    @staticmethod
    def _validate_num_refs(num_refs: int) -> None:
        if num_refs not in {1, 2, 3}:
            raise ValueError("StyleEncoder expects 1, 2, or 3 reference images")
