# ==============================================================================
# ScamShield Server - Production Dockerfile (CUDA 12 + Python 3.10 + Faster-Whisper + Local Nemotron)
# Resolves Windows DLL loading issues (cuBLAS, cuDNN, CTranslate2) by running on Linux.
# ==============================================================================

FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

# Avoid interactive dialogs during apt installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install system dependencies: Python 3.10, ffmpeg (audio), libsndfile, curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    libsndfile1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Symlink python -> python3
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 acceleration
RUN pip3 install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Python requirements
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# Create directories for persistent storage (calls, transcripts, recordings) and whisper/nemotron cache
RUN mkdir -p /app/data/recordings /app/test_samples /root/.cache/huggingface

# Copy server application, models, and test samples
COPY . /app/

# Environment variables with sensible defaults (can be overridden in docker-compose.yml)
ENV DETECTION_MODE=all
ENV STT_PROVIDER=whisper
ENV WHISPER_MODEL=medium
ENV OLLAMA_HOST=host.docker.internal
ENV OLLAMA_MODEL=phi3.5
ENV DATA_DIR=/app/data

# Expose port for FastAPI, WebSockets, and Web UI Dashboard
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/api/status || exit 1

# Start Uvicorn server without repetitive access logs
CMD ["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--ws-ping-interval", "20", "--no-access-log"]

