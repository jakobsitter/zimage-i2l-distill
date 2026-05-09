from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return repo_root() / "data"


def checkpoints_dir() -> Path:
    return repo_root() / "checkpoints"
