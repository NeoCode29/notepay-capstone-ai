"""
NotePay Flask API Server — OCR Struk + Klasifikasi Pengeluaran

Endpoint:
    GET  /health          → status model + device info
    POST /predict         → upload foto struk → JSON hasil OCR + kategori
    POST /classify        → klasifikasi teks item (JSON body, tanpa gambar)

Jalankan di WSL2 (venv-tf aktif):
    source ~/venv-tf/bin/activate
    pip install -r requirements-api.txt

    # Development (auto-reload)
    python api/app.py

    # Production (gunakan gunicorn)
    gunicorn -w 1 -b 0.0.0.0:5000 "api.app:create_app()"

    # Paksa CPU
    NOTEPAY_DEVICE=cpu python api/app.py

Akses dari Windows/Next.js:
    http://localhost:5000/health
    http://localhost:5000/predict   (POST, multipart/form-data, field: file)
    http://localhost:5000/classify  (POST, application/json)
"""

import logging
import os
import sys
import time
import tempfile

import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai.model_loader import get_models, model_info, unload_models
from ai.inference import run, format_result, classify_items

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ALLOWED_MIME = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)  # izinkan semua origin — batasi ke domain frontend saat production

    # ---------------------------------------------------------------------------
    # Load model saat pertama kali app dibuat
    # ---------------------------------------------------------------------------
    logger.info("Memuat model AI...")
    try:
        get_models()
        logger.info("Semua model siap.")
    except Exception as e:
        logger.error(f"Gagal load model: {e}")
        # Server tetap jalan — /health bisa diakses untuk diagnosa

    # ---------------------------------------------------------------------------
    # Helper
    # ---------------------------------------------------------------------------

    def _error(message: str, code: int):
        return jsonify({"error": message}), code

    # ---------------------------------------------------------------------------
    # GET /health
    # ---------------------------------------------------------------------------

    @app.get("/health")
    def health():
        """Cek status server dan model."""
        info   = model_info()
        status = "ok" if info["loaded"] else "degraded"
        return jsonify({"status": status, "model_info": info})

    # ---------------------------------------------------------------------------
    # POST /predict
    # ---------------------------------------------------------------------------

    @app.post("/predict")
    def predict():
        """
        Upload foto struk → JSON lengkap hasil OCR + klasifikasi.

        Request  : multipart/form-data
          - file : foto struk (jpg/png/webp)
          - conf : float, confidence threshold YOLO (opsional, default 0.25)

        Response :
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
          "elapsed_ms": 412,
          "detections_count": 3
        }
        """
        # Validasi file ada
        if "file" not in request.files:
            return _error("Field 'file' tidak ditemukan di request.", 400)

        file = request.files["file"]
        if file.filename == "":
            return _error("Tidak ada file yang dipilih.", 400)

        # Validasi MIME type
        mime = file.content_type or ""
        if mime not in ALLOWED_MIME:
            return _error(
                f"Tipe file tidak didukung: '{mime}'. Gunakan jpg/png/webp.", 415
            )

        # Baca bytes
        contents = file.read()
        if len(contents) == 0:
            return _error("File kosong.", 400)
        if len(contents) > MAX_FILE_SIZE:
            return _error("File terlalu besar (maks 20 MB).", 413)

        # Decode gambar
        arr   = np.frombuffer(contents, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            return _error("Gambar tidak bisa di-decode. Pastikan file tidak rusak.", 422)

        # Confidence threshold (opsional dari form-data atau query string)
        try:
            conf = float(request.form.get("conf", request.args.get("conf", 0.25)))
            conf = max(0.05, min(0.95, conf))
        except ValueError:
            conf = 0.25

        # Ambil model
        try:
            yolo, crnn, classifier, torch_device = get_models()
        except Exception as e:
            return _error(f"Model belum siap: {e}", 503)

        # Simpan ke temp file → jalankan pipeline
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
                cv2.imwrite(tmp_path, image)

            t0 = time.time()
            output, detections, elapsed = run(
                tmp_path, yolo, crnn,
                classifier=classifier,
                conf_threshold=conf,
                torch_device=torch_device,
            )
            result = format_result(output)
            result["elapsed_ms"]       = round((time.time() - t0) * 1000)
            result["detections_count"] = len(detections)

        except Exception as e:
            logger.exception("Error saat inference")
            return _error(f"Inference gagal: {e}", 500)

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        return jsonify(result)

    # ---------------------------------------------------------------------------
    # POST /classify
    # ---------------------------------------------------------------------------

    @app.post("/classify")
    def classify():
        """
        Klasifikasikan list teks item ke kategori pengeluaran (tanpa OCR).

        Request  : application/json
          {"items": ["Indomie Goreng", "Sabun Lifebuoy"]}

        Response :
          {"results": [{"text": "...", "category": "...", "confidence": 0.98}]}
        """
        body = request.get_json(silent=True)
        if not body or "items" not in body:
            return _error("Body JSON harus punya field 'items' (list string).", 400)

        items = body["items"]
        if not isinstance(items, list):
            return _error("'items' harus berupa array string.", 400)
        if len(items) == 0:
            return jsonify({"results": []})
        if len(items) > 100:
            return _error("Maks 100 item per request.", 400)
        if not all(isinstance(i, str) for i in items):
            return _error("Semua elemen 'items' harus string.", 400)

        try:
            _, _, classifier, _ = get_models()
        except Exception as e:
            return _error(f"Model belum siap: {e}", 503)

        results = classify_items(classifier, items)
        return jsonify({"results": results})

    # ---------------------------------------------------------------------------
    # 404 handler
    # ---------------------------------------------------------------------------

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Endpoint tidak ditemukan.", "available": [
            "GET /health",
            "POST /predict",
            "POST /classify",
        ]}), 404

    return app


# ---------------------------------------------------------------------------
# Entry point dev
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    logger.info(f"Server berjalan di http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
