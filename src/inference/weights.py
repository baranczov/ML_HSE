from __future__ import annotations

import hashlib
import os
from pathlib import Path


def resolve_weights_path(path: str | None = None) -> Path:
    """
    Return an absolute Path to the model checkpoint.

    Priority:
      1. ``path`` argument (if given)
      2. MODEL_PATH environment variable
      3. notebooks/models/distilled_resnet18.pth relative to repo root
    """
    if path:
        return Path(path).resolve()
    env = os.environ.get("MODEL_PATH")
    if env:
        return Path(env).resolve()
    # Default: distilled student from ГП_5
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "notebooks" / "models" / "distilled_resnet18.pth"


def model_version(weights_path: Path, length: int = 8) -> str:
    """SHA-256 prefix of the weights file — stable identifier for the loaded checkpoint."""
    h = hashlib.sha256()
    with open(weights_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:length]
