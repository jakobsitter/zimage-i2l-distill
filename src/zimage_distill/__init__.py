from pathlib import Path
import sys


_vendor_path = Path(__file__).resolve().parents[2] / "vendor" / "DiffSynth-Studio"
if _vendor_path.is_dir():
    vendor_str = str(_vendor_path)
    if vendor_str not in sys.path:
        sys.path.insert(0, vendor_str)

from .paths import checkpoints_dir, data_dir, repo_root

__all__ = ["checkpoints_dir", "data_dir", "repo_root"]
