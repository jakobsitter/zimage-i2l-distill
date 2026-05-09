from pathlib import Path

from zimage_distill import checkpoints_dir, data_dir, repo_root


def test_repo_root_points_at_repository_root():
    assert repo_root() == Path(__file__).resolve().parents[1]


def test_data_dir_is_under_repository_root():
    assert data_dir() == repo_root() / "data"


def test_checkpoints_dir_is_under_repository_root():
    assert checkpoints_dir() == repo_root() / "checkpoints"
