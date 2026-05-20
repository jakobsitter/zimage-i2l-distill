from __future__ import annotations

import torch
from torch import nn


class StyleTokenAdapter(nn.Module):
    def __init__(self, embedding_dim: int, hidden_size: int, num_style_tokens: int = 4) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if num_style_tokens <= 0:
            raise ValueError("num_style_tokens must be positive")

        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_style_tokens = num_style_tokens
        self.projection = nn.Linear(embedding_dim, num_style_tokens * hidden_size)

    def forward(self, vector: torch.Tensor) -> torch.Tensor:
        if vector.ndim not in {1, 2}:
            raise ValueError("style vector must have shape [embedding_dim] or [batch, embedding_dim]")
        if vector.shape[-1] != self.embedding_dim:
            raise ValueError(f"style vector must have last dimension {self.embedding_dim}")

        projected = self.projection(vector.to(device=self.projection.weight.device, dtype=self.projection.weight.dtype))
        if vector.ndim == 1:
            return projected.view(self.num_style_tokens, self.hidden_size)
        return projected.view(vector.shape[0], self.num_style_tokens, self.hidden_size)
