import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from zimage_distill.style_embed import (
    StyleEmbedding,
    fuse_style_vectors,
    mix_style_embeddings,
)


def test_style_embedding_save_and_load_roundtrip(tmp_path: Path):
    path = tmp_path / "style_embedding.safetensors"
    embedding = StyleEmbedding(
        vector=torch.tensor([1.5, -2.0, 3.25], dtype=torch.float32),
        vectors=torch.tensor([[1.0, -1.0, 2.0], [2.0, -3.0, 4.0]], dtype=torch.float32),
        source_images=["ref_a.png", "ref_b.png"],
        mode="teacher_fused",
        metadata={"step": 12, "tags": ["siglip2", "dinov3"]},
    )

    embedding.save(path)
    loaded = StyleEmbedding.load(path)

    assert torch.equal(loaded.vector, embedding.vector)
    assert torch.equal(loaded.vectors, embedding.vectors)
    assert loaded.source_images == ["ref_a.png", "ref_b.png"]
    assert loaded.mode == "teacher_fused"
    assert loaded.metadata == {"step": 12, "tags": ["siglip2", "dinov3"]}



def test_mix_style_embeddings_returns_normalized_weighted_average_and_combined_sources():
    embedding_a = StyleEmbedding(
        vector=torch.tensor([1.0, 3.0], dtype=torch.float32),
        source_images=["a.png"],
        mode="single",
        metadata={"name": "a"},
    )
    embedding_b = StyleEmbedding(
        vector=torch.tensor([5.0, 7.0], dtype=torch.float32),
        source_images=["b.png", "c.png"],
        mode="single",
        metadata={"name": "b"},
    )

    mixed = mix_style_embeddings([(embedding_a, 1.0), (embedding_b, 3.0)])

    assert torch.allclose(mixed.vector, torch.tensor([4.0, 6.0], dtype=torch.float32))
    assert torch.equal(
        mixed.vectors,
        torch.tensor([[1.0, 3.0], [5.0, 7.0]], dtype=torch.float32),
    )
    assert mixed.source_images == ["a.png", "b.png", "c.png"]
    assert mixed.mode == "mixed"
    assert mixed.metadata == {"weights": [1.0, 3.0], "modes": ["single", "single"]}


@pytest.mark.parametrize(
    "weighted_embeddings, match",
    [
        ([], "must not be empty"),
        (
            [
                (StyleEmbedding(vector=torch.tensor([1.0, 2.0], dtype=torch.float32)), 0.0),
                (StyleEmbedding(vector=torch.tensor([3.0, 4.0], dtype=torch.float32)), 0.0),
            ],
            "must not be zero",
        ),
    ],
)
def test_mix_style_embeddings_rejects_empty_input_and_zero_total_weight(weighted_embeddings, match: str):
    with pytest.raises(ValueError, match=match):
        mix_style_embeddings(weighted_embeddings)



def test_mix_style_embeddings_rejects_mismatched_vector_shapes():
    embedding_a = StyleEmbedding(vector=torch.tensor([1.0, 2.0], dtype=torch.float32))
    embedding_b = StyleEmbedding(vector=torch.tensor([[3.0, 4.0]], dtype=torch.float32))

    with pytest.raises(ValueError, match="same shape"):
        mix_style_embeddings([(embedding_a, 1.0), (embedding_b, 1.0)])



def test_style_embedding_save_rejects_non_json_serializable_metadata(tmp_path: Path):
    path = tmp_path / "invalid_metadata.safetensors"
    embedding = StyleEmbedding(
        vector=torch.tensor([1.0, 2.0], dtype=torch.float32),
        metadata={"bad": object()},
    )

    with pytest.raises(ValueError, match="JSON-serializable"):
        embedding.save(path)



def test_style_embedding_load_rejects_missing_style_embedding_metadata(tmp_path: Path):
    path = tmp_path / "missing_metadata.safetensors"
    save_file({"vector": torch.tensor([1.0, 2.0], dtype=torch.float32)}, str(path))

    with pytest.raises(ValueError, match="missing style embedding metadata"):
        StyleEmbedding.load(path)



def test_style_embedding_load_rejects_malformed_style_embedding_metadata(tmp_path: Path):
    path = tmp_path / "malformed_metadata.safetensors"
    save_file(
        {"vector": torch.tensor([1.0, 2.0], dtype=torch.float32)},
        str(path),
        metadata={"style_embedding": "not-json"},
    )

    with pytest.raises(ValueError, match="invalid JSON"):
        StyleEmbedding.load(path)



def test_style_embedding_load_rejects_invalid_metadata_payload_shape(tmp_path: Path):
    path = tmp_path / "invalid_payload.safetensors"
    save_file(
        {"vector": torch.tensor([1.0, 2.0], dtype=torch.float32)},
        str(path),
        metadata={"style_embedding": json.dumps(["not", "a", "mapping"])},
    )

    with pytest.raises(ValueError, match="must be a JSON object"):
        StyleEmbedding.load(path)



def test_style_embedding_load_rejects_invalid_metadata_fields(tmp_path: Path):
    path = tmp_path / "invalid_fields.safetensors"
    save_file(
        {"vector": torch.tensor([1.0, 2.0], dtype=torch.float32)},
        str(path),
        metadata={
            "style_embedding": json.dumps(
                {
                    "source_images": "ref.png",
                    "mode": 7,
                    "metadata": ["bad"],
                }
            )
        },
    )

    with pytest.raises(ValueError, match="source_images"):
        StyleEmbedding.load(path)


def test_fuse_style_vectors_averages_2d_tensor():
    vectors = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=torch.float32,
    )

    fused = fuse_style_vectors(vectors)

    assert torch.equal(fused, torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32))


@pytest.mark.parametrize(
    "bad_vectors",
    [
        torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
        torch.ones(2, 3, 4, dtype=torch.float32),
    ],
)
def test_fuse_style_vectors_rejects_non_2d_inputs(bad_vectors: torch.Tensor):
    with pytest.raises(ValueError, match=r"\[num_refs, dim\]"):
        fuse_style_vectors(bad_vectors)
