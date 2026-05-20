from __future__ import annotations

import pytest
import torch


def test_style_token_adapter_projects_single_vector_to_expected_token_shape():
    from zimage_distill.style_token_adapter import StyleTokenAdapter

    adapter = StyleTokenAdapter(embedding_dim=3, hidden_size=5, num_style_tokens=4)

    tokens = adapter(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))

    assert tokens.shape == (4, 5)


def test_style_token_adapter_projects_batch_to_expected_token_shape():
    from zimage_distill.style_token_adapter import StyleTokenAdapter

    adapter = StyleTokenAdapter(embedding_dim=3, hidden_size=5, num_style_tokens=2)

    tokens = adapter(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32))

    assert tokens.shape == (2, 2, 5)


@pytest.mark.parametrize(
    "bad_vector, match",
    [
        (torch.tensor([[[1.0, 2.0]]], dtype=torch.float32), r"shape \[embedding_dim\] or \[batch, embedding_dim\]"),
        (torch.tensor([1.0, 2.0], dtype=torch.float32), "last dimension 3"),
    ],
)
def test_style_token_adapter_rejects_invalid_vectors(bad_vector: torch.Tensor, match: str):
    from zimage_distill.style_token_adapter import StyleTokenAdapter

    adapter = StyleTokenAdapter(embedding_dim=3, hidden_size=5, num_style_tokens=2)

    with pytest.raises(ValueError, match=match):
        adapter(bad_vector)
