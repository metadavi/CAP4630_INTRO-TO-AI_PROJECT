# ── Stage 1: build dependencies ────────────────────────────────────────────
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# Install PyTorch CPU-only first (avoids pulling the 2 GB GPU build)
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime image ──────────────────────────────────────────────────
FROM python:3.11-slim

# OpenCV runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages \
                    /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code and trained models (includes models/plate_crnn.pth)
COPY . .

# Cloud Run injects PORT; default to 8080 for local Docker testing
ENV PORT=8080

EXPOSE 8080

# Use gunicorn + eventlet for production WebSocket support
CMD exec gunicorn \
    --worker-class eventlet \
    --workers 1 \
    --threads 4 \
    --bind "0.0.0.0:${PORT}" \
    --timeout 0 \
    --log-level info \
    "server:app"
