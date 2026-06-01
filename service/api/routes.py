from fastapi import APIRouter, File, UploadFile, HTTPException

from service.core.config import settings
from service.core.errors import ImageTooLargeError, InvalidImageError
from service.api.schemas import (
    PredictResponse,
    BatchPredictResponse,
    MetaResponse,
    HealthResponse,
    ReadyResponse,
)
from service.services import inference_service

router = APIRouter()

_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


def _validate_upload(file: UploadFile) -> None:
    if file.content_type and file.content_type not in _ALLOWED_CONTENT_TYPES:
        raise InvalidImageError(
            f"Unsupported content type '{file.content_type}'. "
            f"Allowed: {', '.join(sorted(_ALLOWED_CONTENT_TYPES))}"
        )


async def _read_validated(file: UploadFile) -> bytes:
    _validate_upload(file)
    raw = await file.read()
    if not raw:
        raise InvalidImageError("Uploaded file is empty")
    if len(raw) > settings.max_upload_bytes:
        raise ImageTooLargeError(settings.max_upload_bytes)
    return raw


# ── Liveness / Readiness ──────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    return {"status": "ok"}


@router.get("/ready", response_model=ReadyResponse, tags=["system"])
async def ready():
    if not inference_service.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded")
    predictor = inference_service.get_predictor()
    return {"ready": True, "model_version": predictor._version}


# ── Meta ──────────────────────────────────────────────────────────────────────

@router.get("/v1/meta", response_model=MetaResponse, tags=["inference"])
async def meta():
    predictor = inference_service.get_predictor()
    return {
        "backbone": predictor.backbone,
        "model_version": predictor._version,
        "age_range": [0, 116],
        "image_size": predictor.img_size,
        "max_upload_bytes": settings.max_upload_bytes,
        "max_batch_size": settings.max_batch_size,
        "cache_size": settings.cache_size,
        "cache_stats": predictor.cache_stats,
    }


# ── Inference ─────────────────────────────────────────────────────────────────

@router.post("/v1/predict", response_model=PredictResponse, tags=["inference"])
async def predict(file: UploadFile = File(..., description="Face image (JPEG/PNG)")):
    raw = await _read_validated(file)
    result = await inference_service.predict_bytes(raw)
    return result


@router.post("/v1/predict_batch", response_model=BatchPredictResponse, tags=["inference"])
async def predict_batch(files: list[UploadFile] = File(..., description="Up to N face images")):
    if len(files) > settings.max_batch_size:
        raise InvalidImageError(
            f"Too many files: {len(files)} > max_batch_size={settings.max_batch_size}"
        )
    import asyncio as _asyncio
    raws = [await _read_validated(f) for f in files]
    results = await _asyncio.gather(*[inference_service.predict_bytes(r) for r in raws])
    return {"results": list(results), "count": len(results)}
