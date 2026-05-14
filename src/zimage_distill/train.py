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
        RandomResizedCrop(256, scale=(0.5, 1.0), antialias=True),
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


def contrastive_loss(predictions: torch.Tensor, targets: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """InfoNCE loss encouraging each prediction to be closest to its own target.

    Returns 0 for batch_size < 2 (no meaningful negatives).
    """
    if predictions.shape[0] < 2:
        return predictions.new_tensor(0.0)
    targets = targets.to(predictions)
    pred_norm = torch.nn.functional.normalize(predictions, dim=-1)
    target_norm = torch.nn.functional.normalize(targets, dim=-1)
    logits = (pred_norm @ target_norm.T) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return torch.nn.functional.cross_entropy(logits, labels)


_clip_model_cache: dict[str, object] = {}


def _get_clip_model(device: str = "cpu") -> object:
    """Load CLIP ViT-L/14 once and cache it."""
    if "clip" not in _clip_model_cache:
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        for p in model.parameters():
            p.requires_grad_(False)
        _clip_model_cache["clip"] = (model, processor)
    return _clip_model_cache["clip"]


def clip_embed_images(images: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    """Get CLIP image embeddings for a batch of images [B, 3, H, W]."""
    clip_model, processor = _get_clip_model(device)
    # Convert tensor images back to PIL for CLIP processor
    from PIL import Image
    from torchvision.transforms.functional import to_pil_image
    pil_images = [to_pil_image(img.cpu()) for img in images]
    inputs = processor(images=pil_images, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = clip_model.get_image_features(**inputs)  # [B, 768]
    return outputs


def clip_perceptual_loss(predictions: torch.Tensor, clip_targets: torch.Tensor) -> torch.Tensor:
    """Cosine distance between projected student embeddings and CLIP image features.

    predictions: [B, 5632] — student embedding
    clip_targets: [B, 768] — CLIP image features of reference images
    """
    clip_targets = clip_targets.to(predictions)
    return (1 - torch.nn.functional.cosine_similarity(predictions, clip_targets, dim=-1)).mean()


_teacher_bundle_cache: object | None = None


def _get_teacher_bundle(device: str = "cuda", torch_dtype: torch.dtype = torch.bfloat16) -> object:
    """Load the real teacher encoders (SigLIP2-G384 + DINOv3-7B) for live supervision.

    Models are staged one at a time on GPU via CPU offloading, so they fit in 32GB.
    """
    global _teacher_bundle_cache
    if _teacher_bundle_cache is None:
        from .teacher_pairs import load_teacher_encoder_bundle, build_teacher_model_configs
        from diffsynth.diffusion.base_pipeline import BasePipeline
        from pathlib import Path
        loader = BasePipeline(device="cpu", torch_dtype=torch_dtype)
        import os as _os
        _os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", "/root/autodl-tmp/models")
        configs = build_teacher_model_configs(
            siglip2_path=Path("/nonexistent"),  # force download
            dinov3_path=Path("/nonexistent"),
        )
        for cfg in configs:
            cfg.download_source = "huggingface"  # HF CDN is faster than ModelScope
        model_pool = loader.download_and_load_models(configs)
        siglip = model_pool.fetch_model("siglip2_image_encoder")
        dino = model_pool.fetch_model("dinov3_image_encoder")
        from .teacher_pairs import TeacherEncoderBundle, ImageCropAndResize
        _teacher_bundle_cache = TeacherEncoderBundle(
            siglip2_image_encoder=siglip,
            dinov3_image_encoder=dino,
            device=device,
            torch_dtype=torch_dtype,
            preprocess=ImageCropAndResize(height=1024, width=1024),
        )
    return _teacher_bundle_cache


def _compute_teacher_embedding(images: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    """Run the real teacher encoder to get a fresh teacher embedding."""
    bundle = _get_teacher_bundle(device)
    from PIL import Image
    from torchvision.transforms.functional import to_pil_image
    pil_images = [to_pil_image(img.cpu()) for img in images]
    return bundle.encode_images(pil_images).mean(dim=0)  # [5632]


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
            all_features.append(features.mean(dim=0).cpu().float())
            all_targets.append(sample["target"].cpu().float())
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
    output_dir: Path,
    val_fraction=0.1,
    augment=False,
    phase1_epochs=30,
    phase2_epochs=20,
    phase3_epochs=10,
    head_hidden_dim=None,
    head_depth=None,
    clip_loss=False,
    resume_checkpoint=None,
    live_teacher=False,
):
    from torch.utils.data import DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"backbone={backbone}  device={device}  samples={len(dataset)}  epochs={epochs}  lr={lr}")

    train_ds, val_ds = _split_dataset(dataset, val_fraction)
    print(f"train={len(train_ds)}  val={len(val_ds)}")

    if resume_checkpoint is not None:
        print(f"Resuming from {resume_checkpoint}")
        model = load_checkpoint(resume_checkpoint).to(device)
        if clip_loss and model.clip_proj is None:
            model.clip_proj = nn.Linear(5632, 768).to(device)
    else:
        clip_proj_dim = 768 if clip_loss else 0
        model = StudentImageEncoder(backbone=backbone, head_hidden_dim=head_hidden_dim, head_depth=head_depth, clip_proj_dim=clip_proj_dim).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"params: {trainable:,} trainable / {total_params:,} total")

    use_clip = getattr(model, "clip_proj", None) is not None

    best_val_loss = float("inf")

    # --- Phase 1: Head-only with feature caching ---
    if phase1_epochs > 0:
        print("\n=== Phase 1: Head-only training (frozen backbones) ===")
        features, targets = _cache_backbone_features(model, train_ds, device)
        val_features, val_targets = _cache_backbone_features(model, val_ds, device)
        contrastive_batch = 32  # minibatch size for contrastive loss

        optimizer = torch.optim.Adam(model.head.parameters(), lr=lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=phase1_epochs)

        for epoch in range(phase1_epochs):
            model.head.train()
            perm = torch.randperm(len(features))
            total_loss = 0.0
            n = len(features)
            for start in range(0, n, contrastive_batch):
                batch_idx = perm[start:start + contrastive_batch]
                feats_batch = features[batch_idx].to(device)
                targs_batch = targets[batch_idx].to(device)
                preds_batch = model.head(feats_batch)
                optimizer.zero_grad()
                # Per-sample distillation + batch-level contrastive
                loss = distillation_loss(preds_batch, targs_batch) + 0.1 * contrastive_loss(preds_batch, targs_batch)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * len(batch_idx)
            total_loss /= n
            scheduler.step()
            val_loss = _validate_head(model, val_features, val_targets, device)
            print(f"  epoch {epoch+1}/{phase1_epochs}  train_loss={total_loss:.4f}  val_loss={val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss

        ckpt = checkpoint_path_for(backbone, output_dir)
        save_checkpoint(model, ckpt)
        print(f"Phase 1 checkpoint saved: {ckpt}")

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
        loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)

        clip_step_interval = max(1, len(loader) // 20)  # CLIP loss every ~5% of batches
        for epoch in range(phase2_epochs):
            model.train()
            total_loss = 0.0
            for step, batch in enumerate(loader):
                images = batch["images"].to(device)
                target = batch["target"].to(device).squeeze(0)
                optimizer.zero_grad()
                prediction = model(images)
                loss = distillation_loss(prediction, target)
                # Add CLIP perceptual loss periodically (expensive: runs CLIP on reference images)
                if use_clip and step % clip_step_interval == 0:
                    clip_feats = clip_embed_images(images.squeeze(0) if images.dim() > 4 else images, device)
                    clip_pred = model.clip_proj(prediction.unsqueeze(0) if prediction.dim() == 1 else prediction)
                    loss = loss + 0.1 * clip_perceptual_loss(clip_pred, clip_feats)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += float(loss.item())
            scheduler.step()
            val_loss = _validate_full(model, val_ds, device)
            print(f"  epoch {epoch+1}/{phase2_epochs}  train_loss={total_loss/len(loader):.4f}  val_loss={val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, checkpoint_path_for(backbone, output_dir))

        ckpt = checkpoint_path_for(backbone, output_dir)
        save_checkpoint(model, ckpt)
        print(f"Phase 2 checkpoint saved: {ckpt}")

    # --- Phase 3: Heavy augmentation polish ---
    if phase3_epochs > 0:
        print("\n=== Phase 3: Heavy augmentation ===")
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr * 0.01, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=phase3_epochs)
        loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)

        for epoch in range(phase3_epochs):
            model.train()
            total_loss = 0.0
            for step, batch in enumerate(loader):
                images = batch["images"].to(device)
                target = batch["target"].to(device).squeeze(0)
                if augment and images.dim() >= 4:
                    images = _apply_augmentation(images)
                    if live_teacher:
                        target = _compute_teacher_embedding(images.squeeze(0) if images.dim() > 4 else images, device)
                optimizer.zero_grad()
                prediction = model(images)
                loss = distillation_loss(prediction, target)
                # CLIP loss on live-teacher batches too
                if use_clip and live_teacher:
                    clip_feats = clip_embed_images(images.squeeze(0) if images.dim() > 4 else images, device)
                    clip_pred = model.clip_proj(prediction.unsqueeze(0) if prediction.dim() == 1 else prediction)
                    loss = loss + 0.1 * clip_perceptual_loss(clip_pred, clip_feats)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += float(loss.item())
                if (step + 1) % 50 == 0:
                    print(f"    step {step+1}/{len(loader)}  loss={total_loss/(step+1):.4f}")
            scheduler.step()
            val_loss = _validate_full(model, val_ds, device)
            print(f"  epoch {epoch+1}/{phase3_epochs}  train_loss={total_loss/len(loader):.4f}  val_loss={val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, checkpoint_path_for(backbone, output_dir))

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
    parser.add_argument("--head-hidden-dim", type=int, default=None,
                        help="Override head hidden dim. Default: from backbone config")
    parser.add_argument("--head-depth", type=int, default=None,
                        help="Override head depth. Default: from backbone config")
    parser.add_argument("--clip-loss", action="store_true", default=False,
                        help="Enable CLIP perceptual loss (requires CLIP model, uses more VRAM).")
    parser.add_argument("--resume-checkpoint", type=Path, default=None,
                        help="Resume training from a saved checkpoint.")
    parser.add_argument("--live-teacher", action="store_true", default=False,
                        help="Use the real teacher encoder (DINOv3-7B + SigLIP2-G384) for live supervision on augmented images.")
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
        output_dir=output_dir,
        val_fraction=args.val_fraction,
        augment=args.augment,
        phase1_epochs=args.phase1_epochs,
        phase2_epochs=args.phase2_epochs,
        phase3_epochs=args.phase3_epochs,
        head_hidden_dim=args.head_hidden_dim,
        head_depth=args.head_depth,
        clip_loss=args.clip_loss,
        resume_checkpoint=args.resume_checkpoint,
        live_teacher=args.live_teacher,
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
