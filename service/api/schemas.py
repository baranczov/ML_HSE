from typing import Optional

from pydantic import BaseModel


class PredictResponse(BaseModel):
    age: float
    age_bin: str
    cached: bool
    model_version: str


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]
    count: int


class MetaResponse(BaseModel):
    backbone: str
    model_version: str
    age_range: list[int]
    image_size: int
    max_upload_bytes: int
    max_batch_size: int
    cache_size: int
    cache_stats: dict


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    ready: bool
    model_version: Optional[str] = None
