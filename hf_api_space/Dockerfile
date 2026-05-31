# NotePay AI API — HuggingFace Spaces Docker
# HF Spaces wajib expose port 7860

FROM python:3.10-slim-bullseye

# System deps untuk OpenCV + PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies (layer terpisah agar re-build cepat saat hanya code yang berubah)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY app.py .
COPY ai/ ./ai/

# Environment
ENV PYTHONUNBUFFERED=1
ENV NOTEPAY_DEVICE=cpu
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_ENABLE_ONEDNN_OPTS=0

# HF Spaces WAJIB port 7860
EXPOSE 7860

# --preload: load model 1x sebelum gunicorn fork (hemat RAM, cepat)
# --timeout 180: beri waktu untuk download model saat cold start
CMD ["gunicorn", \
     "--workers=1", \
     "--bind=0.0.0.0:7860", \
     "--timeout=180", \
     "--preload", \
     "--access-logfile=-", \
     "--error-logfile=-", \
     "app:create_app()"]
