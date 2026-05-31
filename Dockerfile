# NotePay AI Flask API — Dockerfile untuk Render
# Platform: CPU-only (Render tidak punya GPU di free/standard tier)
# Python: 3.10 (sama dengan venv-tf lokal di WSL2)

FROM python:3.10-slim-bullseye

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    # OpenCV butuh libGL
    libgl1-mesa-glx \
    libglib2.0-0 \
    # Untuk OpenMP (dipakai PyTorch/ultralytics)
    libgomp1 \
    # Agar git bisa clone repo kecil jika dibutuhkan
    git \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Working directory
# ---------------------------------------------------------------------------
WORKDIR /app

# ---------------------------------------------------------------------------
# Install Python dependencies (layer terpisah agar cache efisien)
# ---------------------------------------------------------------------------
COPY requirements-render.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-render.txt

# ---------------------------------------------------------------------------
# Copy source code
# ---------------------------------------------------------------------------
# Hanya copy folder yang dibutuhkan server
COPY ai/ ./ai/
COPY api/ ./api/

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
ENV PYTHONUNBUFFERED=1
# Paksa CPU — Render tidak punya GPU
ENV NOTEPAY_DEVICE=cpu
# TF: matikan warning yang tidak perlu
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_ENABLE_ONEDNN_OPTS=0

# Port default Flask
EXPOSE 5000

# ---------------------------------------------------------------------------
# Health check — Render pakai ini untuk pastikan container sehat
# ---------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

# ---------------------------------------------------------------------------
# Run server
# ---------------------------------------------------------------------------
# --workers=1     : 1 worker cukup (model disimpan di memori, multi-worker = OOM)
# --timeout=120   : inference bisa 10-30 detik, beri waktu cukup
# --preload       : load model 1x sebelum fork worker (hemat RAM)
CMD ["gunicorn", \
     "--workers=1", \
     "--bind=0.0.0.0:5000", \
     "--timeout=120", \
     "--preload", \
     "--access-logfile=-", \
     "--error-logfile=-", \
     "api.app:create_app()"]
