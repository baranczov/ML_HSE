FROM python:3.11-slim

WORKDIR /app

# Install CPU-only PyTorch (smaller image) + service deps
RUN pip install --no-cache-dir \
    torch==2.0.1+cpu torchvision==0.15.2+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY service/requirements.txt service/requirements.txt
RUN pip install --no-cache-dir \
    fastapi>=0.111.0 \
    "uvicorn[standard]>=0.29.0" \
    "pydantic-settings>=2.2.0" \
    "python-multipart>=0.0.9" \
    pillow>=10.0.0 \
    numpy>=1.24.3 \
    psutil>=5.9.0

# Copy source
COPY src/ src/
COPY service/ service/
COPY web/ web/
COPY notebooks/models/distilled_resnet18.pth notebooks/models/distilled_resnet18.pth

# Ensure src is importable
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000"]
