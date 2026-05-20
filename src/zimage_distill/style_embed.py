from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

_VECTOR_KEY = "vector"
_VECTORS_KEY = "vectors"
_METADATA_KEY = "style_embedding"
_REQUIRED_PAYLOAD_FIELDS = ("source_images", "mode", "metadata")


@dataclass
class StyleEmbedding:
    vector: torch.Tensor
    vectors: torch.Tensor | None = None
    source_images: list[str] = field(default_factory=list)
    mode: str = "single"
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        payload = _build_payload(self.source_images, self.mode, self.metadata)
        tensors = {_VECTOR_KEY: self.vector.detach().cpu()}
        if self.vectors is not None:
            _validate_vectors(self.vector, self.vectors)
            tensors[_VECTORS_KEY] = self.vectors.detach().cpu()
        save_file(tensors, str(path), metadata={_METADATA_KEY: json.dumps(payload)})

    @classmethod
    def load(cls, path: Path | str) -> "StyleEmbedding":
        path = Path(path)
        tensors = load_file(str(path), device="cpu")
        if _VECTOR_KEY not in tensors:
            raise ValueError(f"style embedding file is missing '{_VECTOR_KEY}' tensor")

        vectors = tensors.get(_VECTORS_KEY)
        if vectors is not None:
            _validate_vectors(tensors[_VECTOR_KEY], vectors)

        with safe_open(str(path), framework="pt", device="cpu") as handle:
            payload = _load_payload(handle.metadata())

        return cls(
            vector=tensors[_VECTOR_KEY],
            vectors=vectors,
            source_images=payload["source_images"],
            mode=payload["mode"],
            metadata=payload["metadata"],
        )



def _validate_vectors(vector: torch.Tensor, vectors: torch.Tensor) -> None:
    if vectors.ndim != 2:
        raise ValueError("style embedding vectors must have shape [num_refs, dim]")
    if vector.ndim != 1:
        raise ValueError("style embedding vector must have shape [dim]")
    if vectors.shape[-1] != vector.shape[-1]:
        raise ValueError("style embedding vectors must match the aggregate vector width")



def _build_payload(source_images: list[str], mode: str, metadata: dict[str, JsonValue]) -> dict[str, JsonValue]:
    if not isinstance(source_images, list) or not all(isinstance(image, str) for image in source_images):
        raise ValueError("source_images must be a list of strings")
    if not isinstance(mode, str):
        raise ValueError("mode must be a string")
    _validate_json_object("metadata", metadata)
    return {
        "source_images": source_images,
        "mode": mode,
        "metadata": metadata,
    }



def _load_payload(file_metadata: dict[str, str] | None) -> dict[str, JsonValue]:
    if not file_metadata or _METADATA_KEY not in file_metadata:
        raise ValueError("style embedding file is missing style embedding metadata")

    raw_payload = file_metadata[_METADATA_KEY]
    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        raise ValueError("style embedding metadata contains invalid JSON") from exc

    if not isinstance(payload, dict):
        raise ValueError("style embedding metadata must be a JSON object")

    missing_fields = [field for field in _REQUIRED_PAYLOAD_FIELDS if field not in payload]
    if missing_fields:
        missing = ", ".join(repr(field) for field in missing_fields)
        raise ValueError(f"style embedding metadata is missing required field(s): {missing}")

    return _build_payload(payload["source_images"], payload["mode"], payload["metadata"])



def _validate_json_object(field_name: str, value: object) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    _validate_json_value(field_name, value)



def _validate_json_value(field_name: str, value: object) -> None:
    if value is None or isinstance(value, bool | int | float | str):
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_value(field_name, item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{field_name} must be JSON-serializable with string keys")
            _validate_json_value(field_name, item)
        return
    raise ValueError(f"{field_name} must be JSON-serializable")



def mix_style_embeddings(weighted_embeddings: list[tuple[StyleEmbedding, float]]) -> StyleEmbedding:
    if not weighted_embeddings:
        raise ValueError("weighted_embeddings must not be empty")

    total_weight = sum(weight for _, weight in weighted_embeddings)
    if total_weight == 0:
        raise ValueError("total weight must not be zero")

    expected_shape = weighted_embeddings[0][0].vector.shape
    if any(embedding.vector.shape != expected_shape for embedding, _ in weighted_embeddings[1:]):
        raise ValueError("all style embedding vectors must have the same shape")

    mixed_vector = sum(embedding.vector * weight for embedding, weight in weighted_embeddings) / total_weight
    component_vectors = torch.cat(
        [embedding.vectors if embedding.vectors is not None else embedding.vector.unsqueeze(0) for embedding, _ in weighted_embeddings],
        dim=0,
    )
    return StyleEmbedding(
        vector=mixed_vector,
        vectors=component_vectors,
        source_images=[image for embedding, _ in weighted_embeddings for image in embedding.source_images],
        mode="mixed",
        metadata={
            "weights": [weight for _, weight in weighted_embeddings],
            "modes": [embedding.mode for embedding, _ in weighted_embeddings],
        },
    )



def fuse_style_vectors(vectors: torch.Tensor) -> torch.Tensor:
    if vectors.ndim != 2:
        raise ValueError("vectors must have shape [num_refs, dim]")
    if vectors.shape[0] == 0:
        raise ValueError("vectors must include at least one reference row")
    return vectors.mean(dim=0)
