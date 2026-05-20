from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

DIVERSITY_LOSS_WEIGHT = 0.05

from .dataset import TeacherEmbeddingDataset
from .style_bridge import StyleImageGate
from .style_embed_encoder import StyleEncoder
from .style_token_adapter import StyleTokenAdapter

DEFAULT_BACKBONE = "dual-dinov3s-siglip2b"
DEFAULT_EPOCHS = 1
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_NUM_STYLE_TOKENS = 4
DEFAULT_HIDDEN_SIZE = 3840
DEFAULT_CHECKPOINT_NAME = "style_bridge.pt"


def project_teacher_target(target: torch.Tensor, *, hidden_size: int, num_style_tokens: int) -> torch.Tensor:
    """Tile the teacher embedding to fill ``hidden_size * num_style_tokens``.

    Uses repetition instead of linear interpolation so that every value in
    the target is an *exact* teacher value — no blurring.  Each token gets a
    different 3840-dim slice of the (repeated) teacher, naturally producing a
    diverse target for the bridge to match.
    """
    if target.ndim != 1:
        raise ValueError("teacher target must have shape [target_dim]")
    out_dim = hidden_size * num_style_tokens
    repeats = (out_dim + target.shape[0] - 1) // target.shape[0]
    tiled = target.to(dtype=torch.float32, device="cpu").repeat(repeats)[:out_dim]
    return tiled.view(num_style_tokens, hidden_size)


def token_diversity_loss(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 2:
        raise ValueError("tokens must have shape [num_style_tokens, hidden_size]")
    if tokens.shape[0] < 2:
        return tokens.new_tensor(0.0)

    normalized_tokens = F.normalize(tokens, dim=-1, eps=1e-6)
    similarity_matrix = normalized_tokens @ normalized_tokens.T
    pairwise_penalties = similarity_matrix.triu(diagonal=1).pow(2)
    pair_count = tokens.shape[0] * (tokens.shape[0] - 1) // 2
    return pairwise_penalties.sum() / pair_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zimage-distill-train-style-bridge",
        description="Train a frozen style encoder bridge on teacher embedding targets.",
    )
    parser.add_argument("--data-dir", type=Path, required=True, help="Path to a teacher-pairs dataset root.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where the bridge checkpoint is written.")
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE, help=f"Style encoder backbone. Default: {DEFAULT_BACKBONE}")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help=f"Training epochs. Default: {DEFAULT_EPOCHS}")
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE, help=f"Learning rate. Default: {DEFAULT_LEARNING_RATE}")
    parser.add_argument("--embedding-dim", type=int, required=True, help="Style encoder embedding size.")
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=DEFAULT_HIDDEN_SIZE,
        help=f"Per-token hidden size. Default: {DEFAULT_HIDDEN_SIZE}",
    )
    parser.add_argument(
        "--num-style-tokens",
        type=int,
        default=DEFAULT_NUM_STYLE_TOKENS,
        help=f"Number of learned style tokens. Default: {DEFAULT_NUM_STYLE_TOKENS}",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device. Defaults to cuda when available, else cpu.",
    )
    return parser

def _device_from_arg(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    dataset = TeacherEmbeddingDataset(args.data_dir)
    if len(dataset) == 0:
        raise ValueError(f"No teacher samples found under {args.data_dir}")

    first_sample = dataset[0]
    target_dim = int(first_sample["target"].numel())
    if target_dim != 5632:
        raise ValueError(f"teacher target must have 5632 values, got {target_dim}")
    hidden_size = args.hidden_size
    device = _device_from_arg(args.device)

    encoder = StyleEncoder(backbone=args.backbone, embedding_dim=args.embedding_dim).to(device)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    adapter = StyleTokenAdapter(args.embedding_dim, hidden_size, args.num_style_tokens).to(device)
    gate = StyleImageGate(args.embedding_dim).to(device)
    optimizer = torch.optim.Adam(list(adapter.parameters()) + list(gate.parameters()), lr=args.lr)

    print(
        f"device={device} samples={len(dataset)} epochs={args.epochs} embedding_dim={args.embedding_dim} "
        f"hidden_size={hidden_size} num_style_tokens={args.num_style_tokens} target_dim={target_dim}"
    )

    for epoch in range(args.epochs):
        total_loss = 0.0
        for sample in dataset:
            optimizer.zero_grad()
            images = sample["images"]
            if images.shape[0] > 3:
                images = images[:3]

            component_vectors = []
            for image in images:
                with torch.no_grad():
                    component_vectors.append(encoder(image.unsqueeze(0).to(device)))
            style_vectors = torch.stack(component_vectors, dim=0)
            predicted_tokens = adapter(style_vectors)
            strengths = torch.rand(style_vectors.shape[0], device=device, dtype=style_vectors.dtype) + 0.5
            gate_logits = gate(style_vectors)
            weights = torch.softmax(gate_logits + torch.log(strengths.clamp_min(1e-6)), dim=0)
            combined_tokens = (weights[:, None, None] * predicted_tokens).sum(dim=0)
            supervised_target = project_teacher_target(
                sample["target"].to(device=device, dtype=torch.float32).reshape(-1),
                hidden_size=hidden_size,
                num_style_tokens=args.num_style_tokens,
            ).to(device=device, dtype=combined_tokens.dtype)
            loss = F.mse_loss(combined_tokens, supervised_target) + DIVERSITY_LOSS_WEIGHT * token_diversity_loss(
                combined_tokens
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        print(f"epoch {epoch + 1}/{args.epochs} done avg_loss {total_loss / len(dataset):.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / DEFAULT_CHECKPOINT_NAME
    torch.save(
        {
            "supervision": "teacher_embedding",
            "backbone": args.backbone,
            "embedding_dim": args.embedding_dim,
            "hidden_size": hidden_size,
            "num_style_tokens": args.num_style_tokens,
            "training_data_path": str(args.data_dir),
            "epochs": args.epochs,
            "lr": args.lr,
            "adapter_kind": "ip_adapter",
            "supports_multi_image": True,
            "style_token_adapter_state_dict": adapter.state_dict(),
            "image_gate_state_dict": gate.state_dict(),
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint: {checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
