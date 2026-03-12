# ─────────────────────────────────────────────────────────────
# Face Mask Detection API — Dockerfile
# ─────────────────────────────────────────────────────────────
# Multi-stage build: keeps the final image lean.
# The model (158MB) is downloaded from Google Drive during build.
# ─────────────────────────────────────────────────────────────


# ── Stage 1: Builder ─────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ─────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed Python packages from builder stage
COPY --from=builder /install /usr/local

# Install minimal runtime dependencies (needed for OpenCV)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Copy application code
COPY app.py .

# Install gdown to download large model files
RUN pip install --no-cache-dir gdown

# Download model from Google Drive
RUN gdown https://drive.google.com/uc?id=1Mfjzif_-ylB-NxC6VEHr2k-lNcQ8tZox -O mask_detector.pth

# Environment variables
ENV MODEL_PATH="mask_detector.pth"
ENV CONFIDENCE_THRESHOLD=0.5
ENV PORT=8000

# Expose API port
EXPOSE 8000

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run API
CMD ["python", "app.py"]
