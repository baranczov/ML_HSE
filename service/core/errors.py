from fastapi import Request
from fastapi.responses import JSONResponse


class ServiceError(Exception):
    def __init__(self, message: str, code: str, status: int = 400):
        self.message = message
        self.code = code
        self.status = status
        super().__init__(message)


class ImageTooLargeError(ServiceError):
    def __init__(self, max_bytes: int):
        super().__init__(
            f"Image exceeds maximum allowed size of {max_bytes // 1024 // 1024} MB",
            code="IMAGE_TOO_LARGE",
            status=413,
        )


class InvalidImageError(ServiceError):
    def __init__(self, detail: str = "Cannot decode image"):
        super().__init__(detail, code="INVALID_IMAGE", status=422)


class ModelNotReadyError(ServiceError):
    def __init__(self):
        super().__init__("Model is not loaded yet", code="MODEL_NOT_READY", status=503)


def _error_body(exc: ServiceError) -> dict:
    return {"message": exc.message, "code": exc.code, "status": exc.status}


async def service_error_handler(request: Request, exc: ServiceError) -> JSONResponse:
    return JSONResponse(status_code=exc.status, content=_error_body(exc))
