"""
Service tests using FastAPI TestClient.

Tests that require the model checkpoint are skipped automatically when
notebooks/models/distilled_resnet18.pth is not present, so CI without
large binary files still passes.
"""
import io
import os
import pytest
from pathlib import Path

# Locate the default checkpoint before importing the app so we can decide
# whether to skip model-dependent tests.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CKPT = _REPO_ROOT / "notebooks" / "models" / "distilled_resnet18.pth"
_MODEL_AVAILABLE = _DEFAULT_CKPT.exists()

_skip_no_model = pytest.mark.skipif(
    not _MODEL_AVAILABLE,
    reason="Model checkpoint not found; skipping model-dependent tests",
)


@pytest.fixture(scope="module")
def client():
    """Create a TestClient with the full app (model loaded once)."""
    from fastapi.testclient import TestClient
    from service.main import app
    with TestClient(app) as c:
        yield c


def _make_png_bytes(width: int = 64, height: int = 64) -> bytes:
    """Return a minimal valid PNG image as bytes (solid grey square)."""
    from PIL import Image
    img = Image.new("RGB", (width, height), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── System endpoints ──────────────────────────────────────────────────────────

@_skip_no_model
def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


@_skip_no_model
def test_ready(client):
    res = client.get("/ready")
    assert res.status_code == 200
    data = res.json()
    assert data["ready"] is True
    assert "model_version" in data


@_skip_no_model
def test_meta(client):
    res = client.get("/v1/meta")
    assert res.status_code == 200
    data = res.json()
    for key in ("backbone", "model_version", "age_range", "image_size",
                "max_upload_bytes", "cache_stats"):
        assert key in data, f"Missing key: {key}"
    assert data["age_range"] == [0, 116]


# ── Single predict ────────────────────────────────────────────────────────────

@_skip_no_model
def test_predict_synthetic(client):
    """Predict on a solid-grey synthetic image — checks API contract."""
    raw = _make_png_bytes()
    res = client.post("/v1/predict", files={"file": ("test.png", raw, "image/png")})
    assert res.status_code == 200
    data = res.json()
    assert "age" in data
    assert 0 <= data["age"] <= 116
    assert "age_bin" in data
    assert isinstance(data["cached"], bool)
    assert "model_version" in data


@_skip_no_model
def test_predict_cache_hit(client):
    """Second request with the same image should return cached=True."""
    raw = _make_png_bytes(80, 80)
    files = {"file": ("cached_test.png", raw, "image/png")}
    client.post("/v1/predict", files=files)          # warm up cache
    files = {"file": ("cached_test.png", raw, "image/png")}
    res2 = client.post("/v1/predict", files=files)
    assert res2.status_code == 200
    assert res2.json()["cached"] is True


# ── Batch predict ─────────────────────────────────────────────────────────────

@_skip_no_model
def test_predict_batch(client):
    raws = [_make_png_bytes(64 + i, 64 + i) for i in range(3)]
    files = [("files", (f"img{i}.png", r, "image/png")) for i, r in enumerate(raws)]
    res = client.post("/v1/predict_batch", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["count"] == 3
    assert len(data["results"]) == 3
    for r in data["results"]:
        assert 0 <= r["age"] <= 116


# ── Validation ────────────────────────────────────────────────────────────────

@_skip_no_model
def test_predict_empty_file(client):
    res = client.post("/v1/predict", files={"file": ("empty.png", b"", "image/png")})
    assert res.status_code in (400, 422)


@_skip_no_model
def test_predict_corrupt_file(client):
    res = client.post("/v1/predict", files={"file": ("bad.png", b"not an image", "image/png")})
    assert res.status_code in (400, 422)


@_skip_no_model
def test_predict_oversized_file(client):
    big = b"x" * (6 * 1024 * 1024)  # 6 MB > default 5 MB limit
    res = client.post("/v1/predict", files={"file": ("big.jpg", big, "image/jpeg")})
    assert res.status_code in (400, 413, 422)
