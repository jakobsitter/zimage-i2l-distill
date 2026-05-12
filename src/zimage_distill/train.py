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

from torch.utils.data import random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms import RandomResizedCrop, ColorJitter, RandomHorizontalFlip, Compose

DEFAULT_BACKBONE = "mobilenet_v3_small"
DEFAULT_EPOCHS = 10
DEFAULT_LEARNING_RATE = 1e-3


def _build_augmentation():
    """Build augmentation transform pipeline."""
    return Compose([
        RandomResizedCrop(224, scale=(0.5, 1.0), antialias=True),
        RandomHorizontalFlip(p=0.5),
    ])


def _apply_augmentation(images):
    """Apply augmentation to a batch of images [N, C, H, W]."""
    transform = _build_augmentation()
    augmented = []
    for img in images:
        aug = transform(img)
        augmented.append(aug)
    return torch.stack(augmented)


def _split_dataset(dataset, val_fraction=0.1):
    """Split dataset into train and validation sets."""
    val_size = max(1, int(len(dataset) * val_fraction))
    train_size = len(dataset) - val_size
    return random_split(dataset, [train_size, val_size])


def _set_backbone_trainable(model, trainable):
    for p in model.encoder.parameters():
        p.requires_grad_(trainable)


def _validate_head(model, features, targets, device):
    model.head.eval()
    total = 0.0
    with torch.no_grad():
        for i in range(len(features)):
            pred = model.head(features[i].to(device))
            total += float(distillation_loss(pred, targets[i].to(device)))
    return total / len(features)


def _validate_full(model, dataset, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for sample in dataset:
            images = sample["images"].to(device)
            target = sample["target"].to(device)
            pred = model(images)
            total += float(distillation_loss(pred, target))
    return total / len(dataset)


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


def _embedding_dim_for_head(head: nn.Module) -> int:
    if isinstance(head, nn.Sequential):
        last = head[-1]
        if isinstance(last, nn.Linear):
            return last.out_features
    if isinstance(head, nn.Linear):
        return head.out_features
    raise TypeError(f"Unsupported head type: {type(head)!r}")


def _is_legacy_linear_head_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
    return "head.weight" in state_dict and "head.bias" in state_dict and "head.0.weight" not in state_dict


def save_checkpoint(model: StudentImageEncoder, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "backbone_name": model.backbone_name,
            "embedding_dim": _embedding_dim_for_head(model.head),
        },
        path,
    )


def load_checkpoint(path: Path) -> StudentImageEncoder:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    model = StudentImageEncoder(
        backbone=checkpoint["backbone_name"],
        embedding_dim=int(checkpoint["embedding_dim"]),
    )
    state_dict = checkpoint["state_dict"]
    if _is_legacy_linear_head_state_dict(state_dict):
        model.head = nn.Linear(model.encoder.out_dim, int(checkpoint["embedding_dim"]))
    model.load_state_dict(state_dict, strict=False)
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
    dataset,
    backbone,
    epochs,
    lr,
    *,
    val_fraction=0.1,
    augment=False,
    phase1_epochs=30,
    phase2_epochs=20,
    phase3_epochs=10,
):
    from torch.utils.data import DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"backbone={backbone}  device={device}  samples={len(dataset)}  epochs={epochs}  lr={lr}")

    train_ds, val_ds = _split_dataset(dataset, val_fraction)
    print(f"train={len(train_ds)}  val={len(val_ds)}")

    model = StudentImageEncoder(backbone=backbone).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"params: {trainable:,} trainable / {total_params:,} total")

    best_val_loss = float("inf")

    # --- Phase 1: Head-only with feature caching ---
    if phase1_epochs > 0:
        print("\n=== Phase 1: Head-only training (frozen backbones) ===")
        features, targets = _cache_backbone_features(model, train_ds, device)
        val_features, val_targets = _cache_backbone_features(model, val_ds, device)

        optimizer = torch.optim.Adam(model.head.parameters(), lr=lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=phase1_epochs)

        for epoch in range(phase1_epochs):
            model.head.train()
            perm = torch.randperm(len(features))
            total_loss = 0.0
            for idx in perm:
                feat = features[idx].to(device)
                target = targets[idx].to(device)
                optimizer.zero_grad()
                loss = distillation_loss(model.head(feat), target)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
            scheduler.step()
            val_loss = _validate_head(model, val_features, val_targets, device)
            print(f"  epoch {epoch+1}/{phase1_epochs}  train_loss={total_loss/len(features):.4f}  val_loss={val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss

    # --- Phase 2: Full model fine-tuning ---
    if phase2_epochs > 0:
        print("\n=== Phase 2: Full model fine-tuning ===")
        _set_backbone_trainable(model, True)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"trainable params: {trainable:,}")

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr * 0.1, weight_decay=1e-4
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=phase2_epochs)
        loader = DataLoader(train_ds, batch_size=2, shuffle=True)

        for epoch in range(phase2_epochs):
            model.train()
            total_loss = 0.0
            for batch in loader:
                images = batch["images"].to(device)
                target = batch["target"].to(device)
                optimizer.zero_grad()
                prediction = model(images)
                loss = distillation_loss(prediction, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += float(loss.item())
            scheduler.step()
            val_loss = _validate_full(model, val_ds, device)
            print(f"  epoch {epoch+1}/{phase2_epochs}  train_loss={total_loss/len(loader):.4f}  val_loss={val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, Path("checkpoints") / checkpoint_path_for(backbone, Path("checkpoints")))

    # --- Phase 3: Heavy augmentation polish ---
    if phase3_epochs > 0:
        print("\n=== Phase 3: Heavy augmentation ===")
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr * 0.01, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=phase3_epochs)
        loader = DataLoader(train_ds, batch_size=2, shuffle=True)

        for epoch in range(phase3_epochs):
            model.train()
            total_loss = 0.0
            for batch in loader:
                images = batch["images"].to(device)
                target = batch["target"].to(device)
                if augment and images.dim() >= 4:
                    images = _apply_augmentation(images)
                optimizer.zero_grad()
                prediction = model(images)
                loss = distillation_loss(prediction, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += float(loss.item())
            scheduler.step()
            val_loss = _validate_full(model, val_ds, device)
            print(f"  epoch {epoch+1}/{phase3_epochs}  train_loss={total_loss/len(loader):.4f}  val_loss={val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, Path("checkpoints") / checkpoint_path_for(backbone, Path("checkpoints")))

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
    parser.add_argument("--val-fraction", type=float, default=0.1,
                        help="Fraction of data for validation. Default: 0.1")
    parser.add_argument("--augment", action="store_true", default=False,
                        help="Enable data augmentation.")
    parser.add_argument("--phase1-epochs", type=int, default=30,
                        help="Phase 1 (head-only) epochs. Default: 30")
    parser.add_argument("--phase2-epochs", type=int, default=20,
                        help="Phase 2 (full model) epochs. Default: 20")
    parser.add_argument("--phase3-epochs", type=int, default=10,
                        help="Phase 3 (augmented) epochs. Default: 10")
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

    trained = _train_model(
        dataset, args.backbone, args.epochs, args.lr,
        val_fraction=args.val_fraction,
        augment=args.augment,
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
        phase3_epochs=args.phase3_epochs,
    )
    model = trained

    ckpt_path = checkpoint_path_for(args.backbone, output_dir)
    save_checkpoint(model, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")

    manifest = {
        "backbone_name": args.backbone,
        "embedding_dim": _embedding_dim_for_head(model.head),
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
