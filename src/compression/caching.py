"""
Two-layer inference cache for the age-prediction model.

PredictionCache  — maps image_hash → float age (skips entire forward pass).
EmbeddingCache   — maps image_hash → backbone embedding tensor (skips the heavy
                   backbone; only the small fc head re-runs on cache hits).

CachedAgePredictor wraps a model with both caches and exposes predict() with
hit/miss statistics for benchmarking.
"""
import hashlib
import io
from collections import OrderedDict
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def image_hash(source: Union[str, Image.Image, torch.Tensor]) -> str:
    """
    Compute a hex-digest SHA-256 hash of an image.

    Accepts:
      - str / Path: file path — reads raw bytes.
      - PIL.Image:  encodes to PNG bytes.
      - torch.Tensor: treats underlying bytes (for pre-processed tensors).
    """
    h = hashlib.sha256()

    if isinstance(source, str):
        with open(source, "rb") as f:
            h.update(f.read())
    elif isinstance(source, Image.Image):
        buf = io.BytesIO()
        source.save(buf, format="PNG")
        h.update(buf.getvalue())
    elif isinstance(source, torch.Tensor):
        h.update(source.detach().cpu().numpy().tobytes())
    else:
        raise TypeError(f"Unsupported source type: {type(source)}")

    return h.hexdigest()


# ---------------------------------------------------------------------------
# LRU dict (no extra deps)
# ---------------------------------------------------------------------------

class _LRUCache:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self._cache: OrderedDict = OrderedDict()

    def get(self, key):
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def set(self, key, value):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def __contains__(self, key):
        return key in self._cache

    def __len__(self):
        return len(self._cache)

    def clear(self):
        self._cache.clear()


# ---------------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------------

class PredictionCache:
    """Maps image_hash → predicted age (float)."""

    def __init__(self, max_size: int = 10_000):
        self._cache = _LRUCache(max_size)
        self.hits = 0
        self.misses = 0

    def get(self, key: str):
        val = self._cache.get(key)
        if val is not None:
            self.hits += 1
            return val
        self.misses += 1
        return None

    def set(self, key: str, age: float):
        self._cache.set(key, age)

    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def reset_stats(self):
        self.hits = 0
        self.misses = 0

    def clear(self):
        self._cache.clear()
        self.reset_stats()


class EmbeddingCache:
    """Maps image_hash → backbone embedding tensor (CPU, detached)."""

    def __init__(self, max_size: int = 10_000):
        self._cache = _LRUCache(max_size)
        self.hits = 0
        self.misses = 0

    def get(self, key: str):
        val = self._cache.get(key)
        if val is not None:
            self.hits += 1
            return val
        self.misses += 1
        return None

    def set(self, key: str, embedding: torch.Tensor):
        self._cache.set(key, embedding.detach().cpu())

    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def reset_stats(self):
        self.hits = 0
        self.misses = 0

    def clear(self):
        self._cache.clear()
        self.reset_stats()


# ---------------------------------------------------------------------------
# Cached predictor
# ---------------------------------------------------------------------------

class CachedAgePredictor:
    """
    Wraps a ResNet age-regression model with a two-layer cache.

    Lookup order for each image:
      1. PredictionCache hit  → return cached age (no GPU work).
      2. EmbeddingCache hit   → run only fc head on cached embedding.
      3. Full forward pass    → run backbone + head; populate both caches.

    Usage:
        predictor = CachedAgePredictor(model, device=device)
        age = predictor.predict_single(pil_image, transform)

        # Batch (list of PIL images + keys)
        ages = predictor.predict_batch(images, keys, transform)

        predictor.print_stats()
    """

    def __init__(self, model, device=None, max_cache_size: int = 10_000):
        self.model = model
        self.device = device or torch.device("cpu")
        model.eval()  # Ensure Dropout/BN are in inference mode
        self.pred_cache = PredictionCache(max_size=max_cache_size)
        self.emb_cache = EmbeddingCache(max_size=max_cache_size)

        # Split backbone and head for embedding-level caching
        self._backbone = nn.Sequential(
            model.conv1, model.bn1, model.relu, model.maxpool,
            model.layer1, model.layer2, model.layer3, model.layer4,
            model.avgpool,
        )
        self._head = model.fc

    def _get_embedding(self, x_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat = self._backbone(x_tensor.to(self.device))
            return torch.flatten(feat, 1)

    def predict_single(self, image: Image.Image, transform, key: str = None) -> float:
        """
        Predict age for a single PIL image.

        Args:
            image:     PIL.Image
            transform: torchvision val transform
            key:       optional pre-computed hash (avoids re-hashing)
        """
        if key is None:
            key = image_hash(image)

        # Layer 1: prediction cache
        cached_age = self.pred_cache.get(key)
        if cached_age is not None:
            return cached_age

        x = transform(image).unsqueeze(0)  # (1, C, H, W)

        # Layer 2: embedding cache
        cached_emb = self.emb_cache.get(key)
        if cached_emb is not None:
            emb = cached_emb.to(self.device)
        else:
            emb = self._get_embedding(x)
            self.emb_cache.set(key, emb)

        with torch.no_grad():
            age = float(np.clip(self._head(emb).squeeze().item(), 0, 116))

        self.pred_cache.set(key, age)
        return age

    def predict_batch(self, images, transform, keys=None):
        """
        Predict ages for a list of PIL images.

        Args:
            images:    list of PIL.Image
            transform: torchvision val transform
            keys:      optional list of pre-computed hashes

        Returns:
            list of float ages (same order as images)
        """
        if keys is None:
            keys = [image_hash(img) for img in images]
        return [self.predict_single(img, transform, key)
                for img, key in zip(images, keys)]

    def print_stats(self):
        print(f"PredictionCache: hits={self.pred_cache.hits}, "
              f"misses={self.pred_cache.misses}, "
              f"hit_rate={self.pred_cache.hit_rate():.1%}")
        print(f"EmbeddingCache:  hits={self.emb_cache.hits}, "
              f"misses={self.emb_cache.misses}, "
              f"hit_rate={self.emb_cache.hit_rate():.1%}")

    def reset_stats(self):
        self.pred_cache.reset_stats()
        self.emb_cache.reset_stats()

    def clear_caches(self):
        self.pred_cache.clear()
        self.emb_cache.clear()
