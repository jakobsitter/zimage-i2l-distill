from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .style_embed import StyleEmbedding
from .style_token_adapter import StyleTokenAdapter


@dataclass(frozen=True)
class LoadedStyleBridge:
    adapter: StyleTokenAdapter
    gate: nn.Module | None
    checkpoint: dict[str, Any]


@dataclass(frozen=True)
class ResolvedStyleTokens:
    component_vectors: torch.Tensor
    per_component_tokens: torch.Tensor
    per_component_weights: torch.Tensor
    combined_tokens: torch.Tensor


@dataclass(frozen=True)
class ResolvedStyleAdapter:
    component_vectors: torch.Tensor
    per_component_tokens: torch.Tensor
    per_component_weights: torch.Tensor
    combined_tokens: torch.Tensor
    ipadapter_kwargs_list: list[dict[str, Any]]


class StyleImageGate(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.embedding_dim = embedding_dim
        self.projection = nn.Linear(embedding_dim, 1)

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        if vectors.ndim not in {1, 2}:
            raise ValueError("style vectors must have shape [embedding_dim] or [batch, embedding_dim]")
        if vectors.shape[-1] != self.embedding_dim:
            raise ValueError(f"style vectors must have last dimension {self.embedding_dim}")
        projected = self.projection(vectors.to(device=self.projection.weight.device, dtype=self.projection.weight.dtype))
        return projected.squeeze(-1)


def parse_style_embedding_spec(style_embedding_spec: str) -> list[tuple[Path, float]]:
    spec = style_embedding_spec.strip()
    if not spec:
        raise ValueError("style embedding spec must not be empty")
    if spec.startswith("[") or spec.startswith("{"):
        parsed = json.loads(spec)
        if isinstance(parsed, dict):
            parsed = parsed.get("embeddings", parsed.get("items"))
        if not isinstance(parsed, list):
            raise ValueError("style embedding spec must be a list or an object with embeddings/items")
        resolved: list[tuple[Path, float]] = []
        for item in parsed:
            if isinstance(item, str):
                resolved.append((Path(item).expanduser(), 1.0))
                continue
            if not isinstance(item, dict) or "path" not in item:
                raise ValueError("style embedding spec entries must be paths or objects with a path")
            resolved.append((Path(str(item["path"])).expanduser(), float(item.get("strength", 1.0))))
        return resolved
    return [(Path(spec).expanduser(), 1.0)]


def load_weighted_style_embeddings(weighted_paths: list[tuple[Path, float]]) -> list[tuple[StyleEmbedding, float]]:
    if not weighted_paths:
        raise ValueError("weighted_paths must not be empty")
    return [(StyleEmbedding.load(path), strength) for path, strength in weighted_paths]


def load_style_bridge_checkpoint(bridge_checkpoint_path: str | Path) -> LoadedStyleBridge:
    bridge_path = Path(bridge_checkpoint_path).expanduser()
    if not bridge_path.is_file():
        raise ValueError(f"bridge checkpoint file does not exist: {bridge_path}")

    checkpoint = torch.load(bridge_path, map_location="cpu", weights_only=False)
    for field in ("embedding_dim", "hidden_size", "num_style_tokens", "style_token_adapter_state_dict"):
        if field not in checkpoint:
            raise ValueError(f"bridge checkpoint is missing '{field}'")

    adapter = StyleTokenAdapter(
        embedding_dim=int(checkpoint["embedding_dim"]),
        hidden_size=int(checkpoint["hidden_size"]),
        num_style_tokens=int(checkpoint["num_style_tokens"]),
    )
    adapter.load_state_dict(checkpoint["style_token_adapter_state_dict"])

    gate: nn.Module | None = None
    if checkpoint.get("supports_multi_image") and "image_gate_state_dict" in checkpoint:
        gate = StyleImageGate(adapter.embedding_dim)
        gate.load_state_dict(checkpoint["image_gate_state_dict"])

    return LoadedStyleBridge(adapter=adapter, gate=gate, checkpoint=checkpoint)


def iter_style_vectors(embedding: StyleEmbedding) -> torch.Tensor:
    if embedding.vectors is not None:
        if embedding.vectors.ndim != 2:
            raise ValueError("style embedding vectors must have shape [num_refs, dim]")
        if embedding.vectors.shape[-1] != embedding.vector.shape[-1]:
            raise ValueError("style embedding vectors must match the aggregate vector width")
        return embedding.vectors
    return embedding.vector.unsqueeze(0)


def resolve_style_tokens(
    weighted_embeddings: list[tuple[StyleEmbedding, float]],
    bridge: LoadedStyleBridge,
) -> ResolvedStyleTokens:
    if not weighted_embeddings:
        raise ValueError("weighted_embeddings must not be empty")

    component_vectors: list[torch.Tensor] = []
    component_strengths: list[torch.Tensor] = []
    for embedding, strength in weighted_embeddings:
        vectors = iter_style_vectors(embedding)
        component_vectors.append(vectors)
        component_strengths.append(vectors.new_full((vectors.shape[0],), float(strength)))

    vectors = torch.cat(component_vectors, dim=0)
    strengths = torch.cat(component_strengths, dim=0)
    per_component_tokens = bridge.adapter(vectors)

    if bridge.gate is None:
        gate_logits = vectors.new_zeros(vectors.shape[0])
    else:
        gate_logits = bridge.gate(vectors)

    weights = torch.softmax(gate_logits + torch.log(strengths.clamp_min(1e-6)), dim=0)
    combined_tokens = (weights[:, None, None] * per_component_tokens).sum(dim=0)
    return ResolvedStyleTokens(
        component_vectors=vectors,
        per_component_tokens=per_component_tokens,
        per_component_weights=weights,
        combined_tokens=combined_tokens,
    )


def resolve_style_adapter(
    weighted_embeddings: list[tuple[StyleEmbedding, float]],
    bridge: LoadedStyleBridge,
    *,
    num_layers: int,
) -> ResolvedStyleAdapter:
    resolved = resolve_style_tokens(weighted_embeddings, bridge)
    ipadapter_kwargs_list = [
        {"ip_hidden_states": resolved.combined_tokens, "scale": 1.0}
        for _ in range(num_layers)
    ]
    return ResolvedStyleAdapter(
        component_vectors=resolved.component_vectors,
        per_component_tokens=resolved.per_component_tokens,
        per_component_weights=resolved.per_component_weights,
        combined_tokens=resolved.combined_tokens,
        ipadapter_kwargs_list=ipadapter_kwargs_list,
    )
