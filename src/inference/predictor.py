"""
AgePredictor — single entry point for inference in both the CLI and the web service.

Wraps CachedAgePredictor (two-layer cache from src.compression.caching) so that
repeated images skip the forward pass entirely.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from src.compression.caching import CachedAgePredictor
from src.models.resent_model import create_model, get_device
from src.inference.transforms import build_val_transform
from src.inference.weights import resolve_weights_path, model_version as _model_version

# UTKFace age bins — must match src/data/preprocessing.py create_age_bins
_AGE_BINS  = [0, 3, 13, 20, 30, 40, 50, 60, 70, 80, 117]
_AGE_LABELS = ["0-2", "3-12", "13-19", "20-29", "30-39",
               "40-49", "50-59", "60-69", "70-79", "80+"]


def age_to_bin(age: float) -> str:
    """Map a predicted age (float) to the matching UTKFace age-bin label."""
    for i in range(len(_AGE_BINS) - 1):
        if _AGE_BINS[i] <= age < _AGE_BINS[i + 1]:
            return _AGE_LABELS[i]
    return _AGE_LABELS[-1]


class AgePredictor:
    """
    Production inference wrapper.

    Usage:
        predictor = AgePredictor()           # loads distilled_resnet18.pth
        result = predictor.predict_bytes(raw_bytes)
        # → {"age": 34.2, "age_bin": "30-39", "cached": False, "model_version": "ab12cd34"}
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        backbone: str = "resnet18",
        img_size: int = 224,
        cache_size: int = 10_000,
        device: Optional[torch.device] = None,
    ):
        self.weights_path = resolve_weights_path(str(weights_path) if weights_path else None)
        self.backbone = backbone
        self.img_size = img_size
        self._version = _model_version(self.weights_path)

        self.device = device or get_device()
        self._transform = build_val_transform(img_size)

        model = create_model(backbone)
        state = torch.load(self.weights_path, map_location=self.device)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()

        self._cached = CachedAgePredictor(
            model, device=self.device, max_cache_size=cache_size
        )

    def warmup(self) -> None:
        """Run one dummy forward pass to JIT-compile and warm up the model."""
        dummy = torch.zeros(1, 3, self.img_size, self.img_size, device=self.device)
        with torch.inference_mode():
            self._cached._backbone(dummy)

    def predict_pil(self, image: Image.Image) -> dict:
        """Predict age for a PIL image. Returns result dict."""
        before_hits = self._cached.pred_cache.hits
        age = self._cached.predict_single(image, self._transform)
        age = float(np.clip(age, 0, 116))
        was_cached = self._cached.pred_cache.hits > before_hits
        return {
            "age": round(age, 2),
            "age_bin": age_to_bin(age),
            "cached": was_cached,
            "model_version": self._version,
        }

    def predict_bytes(self, raw: bytes) -> dict:
        """Predict age from raw image bytes (JPEG/PNG/etc.)."""
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        return self.predict_pil(image)

    @property
    def cache_stats(self) -> dict:
        pc = self._cached.pred_cache
        ec = self._cached.emb_cache
        return {
            "prediction_hits": pc.hits,
            "prediction_misses": pc.misses,
            "prediction_hit_rate": round(pc.hit_rate(), 4),
            "embedding_hits": ec.hits,
            "embedding_misses": ec.misses,
            "embedding_hit_rate": round(ec.hit_rate(), 4),
        }
