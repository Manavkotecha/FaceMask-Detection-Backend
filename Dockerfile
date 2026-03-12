# ─────────────────────────────────────────────────────────────
# Face Mask Detection API — Dockerfile
# ─────────────────────────────────────────────────────────────
# Multi-stage build: keeps the final image lean.
# Supports CPU out-of-the-box.  For GPU, swap the base image
# to an NVIDIA CUDA image (see comments below).
# ─────────────────────────────────────────────────────────────

# ── Stage 1: Builder ────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build-time OS deps (none needed for CPU-only torch)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install Python deps into a prefix so we can copy them cleanly
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Install minimal runtime OS deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Copy application code and model weights
COPY app.py .
COPY "mask_detector High acccu.pth" .

# Environment
ENV MODEL_PATH="mask_detector High acccu.pth"
ENV CONFIDENCE_THRESHOLD=0.5
ENV PORT=8000

# Expose the API port
EXPOSE 8000

# Health-check (container orchestrators like Docker Compose / K8s)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the server
CMD ["python", "app.py"]
