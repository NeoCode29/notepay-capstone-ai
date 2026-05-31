"""
NotePay FastAPI Server — OCR Struk + Klasifikasi Pengeluaran

Endpoint:
    GET  /health          → status model + device info
    POST /predict         → upload foto struk → JSON hasil OCR + kategori
    POST /classify        → klasifikasi teks item saja (tanpa OCR)

Jalankan di WSL2 (venv-tf aktif):
    source ~/venv-tf/bin/activate
    pip install fastapi uvicorn python-multipart

    # Development
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

    # Production
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1

    # Paksa CPU (tanpa GPU)
    NOTEPAY_DEVICE=cpu uvicorn api.main:app --host 0.0.0.0 --port 8000

Akses dari Windows (browser/Postman):
    http://localhost:8000/docs     ← Swagger UI
    http://localhost:8000/health
"""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai.model_loader import get_models, model_info, unload_models
from ai.inference import run, format_result

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Lifespan — load model saat startup, unload saat shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Startup: memuat model...")
    try:
        get_models()
        logger.info("Startup: semua model siap.")
    except Exception as e:
        logger.error(f"Startup: gagal load model — {e}")
        # Tetap jalan agar /health bisa diakses dan diagnosa kesalahan
    yield
    logger.info("Shutdown: unload model...")
    unload_models()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="NotePay OCR API",
    description="OCR struk belanja otomatis — deteksi teks + klasifikasi pengeluaran",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ganti ke domain frontend spesifik saat production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class ClassifyRequest(BaseModel):
    items: list[str]

class ClassifyResponse(BaseModel):
    results: list[dict]

class HealthResponse(BaseModel):
    status: str
    model_info: dict


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    """Cek status server dan model."""
    info = model_info()
    status = "ok" if info["loaded"] else "degraded"
    return {"status": status, "model_info": info}


@app.post("/predict", tags=["ocr"])
async def predict(
    file: UploadFile = File(..., description="Foto struk belanja (jpg/png/webp)"),
    conf: float = 0.25,
):
    """
    Upload foto struk → ekstrak teks + klasifikasi kategori pengeluaran.

    Returns JSON:
    ```json
    {
      "nama_toko": "INDOMARET",
      "tanggal_waktu": "12/05/2025 14:30",
      "tanggal_parsed": "2025-05-12 14:30",
      "total_belanja": "Rp 125.000",
      "total_parsed": 125000.0,
      "line_item": ["Indomie Goreng", "Aqua 600ml"],
      "line_item_classified": [
        {"text": "Indomie Goreng", "category": "Makanan & Minuman", "confidence": 0.98}
      ],
      "kategori_summary": {"Makanan & Minuman": 2},
      "elapsed_ms": 412
    }
    ```
    """
    # Validasi tipe file
    if file.content_type not in ("image/jpeg", "image/png", "image/webp", "image/jpg"):
        raise HTTPException(
            status_code=415,
            detail=f"Tipe file tidak didukung: {file.content_type}. Gunakan jpg/png/webp.",
        )

    # Baca bytes gambar
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="File kosong.")
    if len(contents) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File terlalu besar (maks 20 MB).")

    # Decode ke numpy array (cv2)
    try:
        import cv2
        arr = np.frombuffer(contents, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Gambar tidak bisa di-decode")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Gambar tidak valid: {e}")

    # Load model (sudah di-cache dari startup)
    try:
        yolo, crnn, classifier, torch_device = get_models()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model belum siap: {e}")

    # Simpan gambar ke temp file lalu jalankan pipeline
    # (inference.run() butuh path, bukan numpy array langsung)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
        cv2.imwrite(tmp_path, image)

    try:
        t0 = time.time()
        output, detections, elapsed = run(
            tmp_path, yolo, crnn,
            classifier=classifier,
            conf_threshold=conf,
            torch_device=torch_device,
        )
        result = format_result(output)
        result["elapsed_ms"] = round((time.time() - t0) * 1000)
        result["detections_count"] = len(detections)
    except Exception as e:
        logger.exception("Error saat inference")
        raise HTTPException(status_code=500, detail=f"Inference gagal: {e}")
    finally:
        os.unlink(tmp_path)

    return JSONResponse(content=result)


@app.post("/classify", response_model=ClassifyResponse, tags=["ocr"])
def classify_text(body: ClassifyRequest):
    """
    Klasifikasikan list teks item ke kategori pengeluaran (tanpa OCR).
    Berguna untuk klasifikasi ulang atau testing.

    Request body:
    ```json
    {"items": ["Indomie Goreng", "Sabun Lifebuoy", "Paracetamol"]}
    ```
    """
    if not body.items:
        return {"results": []}
    if len(body.items) > 100:
        raise HTTPException(status_code=400, detail="Maks 100 item per request.")

    try:
        _, _, classifier, _ = get_models()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model belum siap: {e}")

    from ai.inference import classify_items
    results = classify_items(classifier, body.items)
    return {"results": results}


# ---------------------------------------------------------------------------
# Dev runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
