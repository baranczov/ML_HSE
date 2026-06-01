from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from service.core.config import settings
from service.core.errors import ServiceError, service_error_handler
from service.api.routes import router
from service.services.inference_service import init_predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model once at startup; warmup runs inside init_predictor
    init_predictor()
    yield
    # Nothing to clean up


app = FastAPI(
    title="Age Prediction Service",
    description="Predicts age from a face image using a compressed ResNet18 model.",
    version="1.0.0",
    debug=settings.debug,
    lifespan=lifespan,
)

app.add_exception_handler(ServiceError, service_error_handler)
app.include_router(router)

# Serve browser UI from web/static/ at "/"
_static_dir = Path(__file__).resolve().parents[1] / "web" / "static"
if _static_dir.exists():
    app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="static")
