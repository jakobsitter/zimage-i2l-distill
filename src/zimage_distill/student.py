from __future__ import annotations

import torch
from torch import nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

TEACHER_EMBEDDING_DIM = 5632

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]
_SIGLIP_MEAN   = [0.5, 0.5, 0.5]
_SIGLIP_STD    = [0.5, 0.5, 0.5]


# ---------------------------------------------------------------------------
# Backbone modules
# ---------------------------------------------------------------------------

class _MobileNetBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        net = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        net.classifier = nn.Identity()
        self.net = net
        self.out_dim = 576
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean.to(x)) / self.std.to(x)
        return self.net(x)


class _HFViTBackbone(nn.Module):
    def __init__(self, model_id: str, norm_mean: list[float], norm_std: list[float], freeze: bool = False) -> None:
        super().__init__()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(model_id)
        if freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)
        self.out_dim: int = self.model.config.hidden_size
        self.register_buffer("mean", torch.tensor(norm_mean).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor(norm_std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean.to(x)) / self.std.to(x)
        return _pool(self.model(pixel_values=x))


class _DualHFViTBackbone(nn.Module):
    def __init__(
        self,
        model_id_a: str,
        model_id_b: str,
        norm_mean_a: list[float],
        norm_std_a: list[float],
        norm_mean_b: list[float],
        norm_std_b: list[float],
        freeze: bool = True,
    ) -> None:
        super().__init__()
        from transformers import AutoModel
        self.model_a = AutoModel.from_pretrained(model_id_a)
        self.model_b = AutoModel.from_pretrained(model_id_b)
        if freeze:
            for p in list(self.model_a.parameters()) + list(self.model_b.parameters()):
                p.requires_grad_(False)
        self.out_dim: int = self.model_a.config.hidden_size + self.model_b.config.hidden_size
        self.register_buffer("mean_a", torch.tensor(norm_mean_a).view(1, 3, 1, 1))
        self.register_buffer("std_a",  torch.tensor(norm_std_a).view(1, 3, 1, 1))
        self.register_buffer("mean_b", torch.tensor(norm_mean_b).view(1, 3, 1, 1))
        self.register_buffer("std_b",  torch.tensor(norm_std_b).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xa = (x - self.mean_a.to(x)) / self.std_a.to(x)
        xb = (x - self.mean_b.to(x)) / self.std_b.to(x)
        return torch.cat([_pool(self.model_a(pixel_values=xa)),
                          _pool(self.model_b(pixel_values=xb))], dim=-1)


def _pool(outputs) -> torch.Tensor:
    if getattr(outputs, "pooler_output", None) is not None:
        return outputs.pooler_output
    return outputs.last_hidden_state[:, 0]


# ---------------------------------------------------------------------------
# Backbone registry
# ---------------------------------------------------------------------------

BACKBONE_CONFIGS: dict[str, dict] = {
    "mobilenet_v3_small": {
        "cls": _MobileNetBackbone,
        "kwargs": {},
    },
    "dinov3-vitb16": {
        "cls": _HFViTBackbone,
        "kwargs": {
            "model_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
            "norm_mean": _IMAGENET_MEAN,
            "norm_std":  _IMAGENET_STD,
            "freeze": False,
        },
    },
    "dinov3-vits16": {
        "cls": _HFViTBackbone,
        "kwargs": {
            "model_id": "facebook/dinov3-vits16-pretrain-lvd1689m",
            "norm_mean": _IMAGENET_MEAN,
            "norm_std":  _IMAGENET_STD,
            "freeze": False,
        },
    },
    "dual-dinov3s-siglip2b": {
        "cls": _DualHFViTBackbone,
        "kwargs": {
            "model_id_a":   "facebook/dinov3-vits16-pretrain-lvd1689m",
            "model_id_b":   "google/siglip2-base-patch16-224",
            "norm_mean_a":  _IMAGENET_MEAN,
            "norm_std_a":   _IMAGENET_STD,
            "norm_mean_b":  _SIGLIP_MEAN,
            "norm_std_b":   _SIGLIP_STD,
            "freeze": True,
        },
    },
}


def _build_backbone(name: str) -> nn.Module:
    if name not in BACKBONE_CONFIGS:
        raise ValueError(f"Unknown backbone '{name}'. Available: {list(BACKBONE_CONFIGS)}")
    cfg = BACKBONE_CONFIGS[name]
    return cfg["cls"](**cfg["kwargs"])


# ---------------------------------------------------------------------------
# Student model
# ---------------------------------------------------------------------------

class StudentImageEncoder(nn.Module):
    def __init__(self, backbone: str = "mobilenet_v3_small", embedding_dim: int = TEACHER_EMBEDDING_DIM) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.encoder = _build_backbone(backbone)
        self.head = nn.Linear(self.encoder.out_dim, embedding_dim)

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
        return self.head(self.encoder(images))
