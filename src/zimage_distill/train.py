from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset

from .paths import checkpoints_dir, data_dir
from .student import StudentImageEncoder
from .dataset import TeacherEmbeddingDataset

BACKBONE_NAME = StudentImageEncoder.backbone_name
DEFAULT_EPOCHS = 2
DEFAULT_LEARNING_RATE = 1e-3
CHECKPOINT_FILENAME = "student.pt"
MANIFEST_FILENAME = "student_manifest.json"


def train_step(model: nn.Module, optimizer: torch.optim.Optimizer, images: torch.Tensor, target: torch.Tensor) -> float:
    model.train()
    device = next(model.parameters()).device
    images = images.to(device)
    target = target.to(device)

    optimizer.zero_grad()
    prediction = model(images)
    loss = torch.nn.functional.mse_loss(prediction, target)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def save_checkpoint(model: StudentImageEncoder, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "backbone_name": getattr(model, "backbone_name", BACKBONE_NAME),
            "embedding_dim": model.head.out_features,
        },
        path,
    )


def load_checkpoint(path: Path) -> StudentImageEncoder:
    checkpoint = torch.load(Path(path), map_location="cpu")
    if checkpoint["backbone_name"] != BACKBONE_NAME:
        raise ValueError(f"Unsupported backbone: {checkpoint['backbone_name']}")

    model = StudentImageEncoder(embedding_dim=int(checkpoint["embedding_dim"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def _train_model(dataset: Dataset[dict[str, torch.Tensor]]) -> StudentImageEncoder:
    model = StudentImageEncoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=DEFAULT_LEARNING_RATE)

    for _ in range(DEFAULT_EPOCHS):
        for sample in dataset:
            train_step(model, optimizer, sample["images"], sample["target"])

    return model


def main(argv: list[str] | None = None) -> int:
    _ = argv
    training_data_path = data_dir() / "teacher_pairs"
    output_dir = checkpoints_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = TeacherEmbeddingDataset(training_data_path)
    model = _train_model(dataset)

    checkpoint_path = output_dir / CHECKPOINT_FILENAME
    save_checkpoint(model, checkpoint_path)

    manifest = {
        "backbone_name": BACKBONE_NAME,
        "embedding_dim": model.head.out_features,
        "training_data_path": str(training_data_path),
    }
    (output_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
