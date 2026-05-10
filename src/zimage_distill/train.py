from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset

from .paths import checkpoints_dir, data_dir
from .student import BACKBONE_CONFIGS, StudentImageEncoder, TEACHER_EMBEDDING_DIM
from .dataset import TeacherEmbeddingDataset

DEFAULT_BACKBONE = "mobilenet_v3_small"
DEFAULT_EPOCHS = 10
DEFAULT_LEARNING_RATE = 1e-3
MANIFEST_FILENAME = "student_manifest.json"


def checkpoint_path_for(backbone: str, output_dir: Path) -> Path:
    return output_dir / f"student_{backbone}.pt"


def distillation_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.to(prediction)
    cos_loss = (1 - torch.nn.functional.cosine_similarity(prediction, target, dim=-1)).mean()
    mse_loss = torch.nn.functional.mse_loss(prediction, target)
    return cos_loss + 0.1 * mse_loss


def train_step(model: nn.Module, optimizer: torch.optim.Optimizer, images: torch.Tensor, target: torch.Tensor) -> float:
    model.train()
    device = next(model.parameters()).device
    images = images.to(device)
    target = target.to(device)
    optimizer.zero_grad()
    prediction = model(images)
    loss = distillation_loss(prediction, target)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def save_checkpoint(model: StudentImageEncoder, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "backbone_name": model.backbone_name,
            "embedding_dim": model.head.out_features,
        },
        path,
    )


def load_checkpoint(path: Path) -> StudentImageEncoder:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    model = StudentImageEncoder(
        backbone=checkpoint["backbone_name"],
        embedding_dim=int(checkpoint["embedding_dim"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def _is_frozen_backbone(backbone: str) -> bool:
    cfg = BACKBONE_CONFIGS.get(backbone, {})
    return cfg.get("kwargs", {}).get("freeze", False)


def _cache_backbone_features(model: StudentImageEncoder, dataset: Dataset, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute frozen backbone features once so epochs only train the head."""
    print("Pre-computing backbone features (one-time)...")
    model.eval()
    all_features, all_targets = [], []
    with torch.no_grad():
        for i, sample in enumerate(dataset):
            images = sample["images"].to(device)
            features = model.encoder(images.unsqueeze(0) if images.dim() == 3 else images)
            all_features.append(features.mean(dim=0).cpu())
            all_targets.append(sample["target"].cpu())
            if (i + 1) % 100 == 0:
                print(f"  cached {i + 1}/{len(dataset)}")
    print(f"Feature cache ready: {len(all_features)} samples")
    return torch.stack(all_features), torch.stack(all_targets)


def _train_model(
    dataset: Dataset,
    backbone: str,
    epochs: int,
    lr: float,
) -> StudentImageEncoder:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"backbone={backbone}  device={device}  samples={len(dataset)}  epochs={epochs}  lr={lr}")

    model = StudentImageEncoder(backbone=backbone).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"params: {trainable:,} trainable / {total:,} total")

    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=lr
    )

    # For frozen backbones, pre-compute features once and train only the head.
    if _is_frozen_backbone(backbone):
        features, targets = _cache_backbone_features(model, dataset, device)
        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(len(features))
            total_loss = 0.0
            for i, idx in enumerate(perm):
                feat = features[idx].to(device)
                target = targets[idx].to(device)
                optimizer.zero_grad()
                prediction = model.head(feat)
                loss = distillation_loss(prediction, target)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
            print(f"epoch {epoch + 1}/{epochs} done  avg_loss {total_loss / len(features):.4f}")
    else:
        for epoch in range(epochs):
            total_loss = 0.0
            for i, sample in enumerate(dataset):
                loss = train_step(model, optimizer, sample["images"], sample["target"])
                total_loss += loss
                if (i + 1) % 100 == 0:
                    print(f"  epoch {epoch + 1}/{epochs}  step {i + 1}/{len(dataset)}  avg_loss {total_loss / (i + 1):.4f}")
            print(f"epoch {epoch + 1}/{epochs} done  avg_loss {total_loss / len(dataset):.4f}")

    return model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="zimage-distill-train", description="Train a student encoder.")
    parser.add_argument(
        "--backbone",
        choices=list(BACKBONE_CONFIGS),
        default=DEFAULT_BACKBONE,
        help=f"Backbone architecture. Default: {DEFAULT_BACKBONE}",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Training epochs. Default: {DEFAULT_EPOCHS}",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f"Learning rate. Default: {DEFAULT_LEARNING_RATE}",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Path to teacher_pairs directory. Defaults to repo data/teacher_pairs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write checkpoints. Defaults to repo checkpoints/.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    training_data_path = args.data_dir or (data_dir() / "teacher_pairs")
    output_dir = args.output_dir or checkpoints_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {training_data_path}")
    dataset = TeacherEmbeddingDataset(training_data_path)

    model = _train_model(dataset, backbone=args.backbone, epochs=args.epochs, lr=args.lr)

    ckpt_path = checkpoint_path_for(args.backbone, output_dir)
    save_checkpoint(model, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")

    manifest = {
        "backbone_name": args.backbone,
        "embedding_dim": model.head.out_features,
        "training_data_path": str(training_data_path),
        "epochs": args.epochs,
        "lr": args.lr,
    }
    manifest_path = output_dir / f"student_{args.backbone}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Saved manifest:    {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
