"""
Singleton inference service.

The model is loaded once during app lifespan and shared across all requests.
Heavy forward passes are offloaded to a thread pool so the async event loop
stays unblocked; a semaphore caps simultaneous GPU/CPU inference calls.
"""
import asyncio
from typing import Optional

from service.core.config import settings
from service.core.errors import ModelNotReadyError, InvalidImageError
from src.inference.predictor import AgePredictor

_predictor: Optional[AgePredictor] = None
_semaphore: Optional[asyncio.Semaphore] = None


def init_predictor() -> None:
    global _predictor, _semaphore
    _predictor = AgePredictor(
        weights_path=settings.model_path or None,
        backbone=settings.backbone_name,
        img_size=settings.image_size,
        cache_size=settings.cache_size,
    )
    _predictor.warmup()
    _semaphore = asyncio.Semaphore(settings.max_concurrency)


def get_predictor() -> AgePredictor:
    if _predictor is None:
        raise ModelNotReadyError()
    return _predictor


def is_ready() -> bool:
    return _predictor is not None


async def predict_bytes(raw: bytes) -> dict:
    """Async wrapper: runs blocking inference in a thread pool under a semaphore."""
    predictor = get_predictor()
    assert _semaphore is not None
    async with _semaphore:
        try:
            result = await asyncio.to_thread(predictor.predict_bytes, raw)
        except Exception as exc:
            raise InvalidImageError(f"Inference failed: {exc}") from exc
    return result
