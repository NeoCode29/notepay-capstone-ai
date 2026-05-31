"""
NotePay AI API — HuggingFace Spaces (Docker)

Endpoint:
    GET  /health   → status model
    POST /predict  → foto struk → JSON OCR + kategori  (streaming, bypass 60s proxy timeout)
    POST /classify → klasifikasi teks item

Port: 7860 (wajib di HF Spaces)
URL : https://NeoCode77-notepay-api.hf.space
"""

import json
import logging
import os
import tempfile
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, stream_with_context
from flask_cors import CORS

from ai.model_loader import get_models, model_info, unload_models
from ai.inference import run, format_result, classify_items

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ALLOWED_MIME  = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    logger.info("Memuat model AI...")
    try:
        get_models()
        logger.info("Semua model siap.")
    except Exception as e:
        logger.error(f"Gagal load model saat startup: {e}")

    def _err(msg: str, code: int):
        return jsonify({"error": msg}), code

    # ------------------------------------------------------------------
    # GET /health
    # ------------------------------------------------------------------
    @app.get("/health")
    def health():
        info   = model_info()
        status = "ok" if info["loaded"] else "degraded"
        return jsonify({"status": status, "model_info": info})

    # ------------------------------------------------------------------
    # POST /predict  — streaming response agar tidak kena proxy timeout
    # ------------------------------------------------------------------
    @app.post("/predict")
    def predict():
        """
        Upload foto struk → JSON hasil OCR + klasifikasi.

        Menggunakan streaming response: server kirim heartbeat (spasi)
        setiap 3 detik selama inference berjalan agar koneksi tidak
        diputus oleh proxy HF Spaces (hard limit 60 detik).

        Client menerima JSON normal di akhir stream.

        Form-data:
          file : gambar struk (jpg/png/webp, maks 20 MB)
          conf : confidence threshold YOLO (opsional, default 0.25)
        """
        # --- Validasi input ---
        if "file" not in request.files:
            return _err("Field 'file' tidak ditemukan.", 400)

        file = request.files["file"]
        if not file.filename:
            return _err("Tidak ada file yang dipilih.", 400)
        if (file.content_type or "") not in ALLOWED_MIME:
            return _err(f"Tipe file tidak didukung: '{file.content_type}'.", 415)

        contents = file.read()
        if len(contents) == 0:
            return _err("File kosong.", 400)
        if len(contents) > MAX_FILE_SIZE:
            return _err("File terlalu besar (maks 20 MB).", 413)

        arr   = np.frombuffer(contents, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            return _err("Gambar tidak bisa di-decode.", 422)

        try:
            conf = float(request.form.get("conf", request.args.get("conf", 0.25)))
            conf = max(0.05, min(0.95, conf))
        except ValueError:
            conf = 0.25

        try:
            yolo, crnn, classifier, torch_device = get_models()
        except Exception as e:
            return _err(f"Model belum siap: {e}", 503)

        # --- Jalankan inference di thread terpisah ---
        result_box = [None]   # hasil inference
        error_box  = [None]   # error jika ada
        done_event = threading.Event()

        def _inference():
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    tmp_path = tmp.name
                    cv2.imwrite(tmp_path, image)

                t0 = time.time()
                output, detections, _ = run(
                    tmp_path, yolo, crnn,
                    classifier=classifier,
                    conf_threshold=conf,
                    torch_device=torch_device,
                )
                result = format_result(output)
                result["elapsed_ms"]       = round((time.time() - t0) * 1000)
                result["detections_count"] = len(detections)
                result_box[0] = result

            except Exception as e:
                logger.exception("Error saat inference")
                error_box[0] = str(e)
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                done_event.set()

        thread = threading.Thread(target=_inference, daemon=True)
        thread.start()

        # --- Generator: kirim heartbeat sampai inference selesai ---
        def _generate():
            # Kirim spasi setiap 3 detik agar proxy tidak timeout
            while not done_event.wait(timeout=3):
                yield b" "

            if error_box[0]:
                yield json.dumps({"error": error_box[0]}).encode("utf-8")
            else:
                yield json.dumps(result_box[0], ensure_ascii=False).encode("utf-8")

        return Response(
            stream_with_context(_generate()),
            content_type="application/json; charset=utf-8",
        )

    # ------------------------------------------------------------------
    # POST /classify
    # ------------------------------------------------------------------
    @app.post("/classify")
    def classify():
        """
        Klasifikasikan list teks item tanpa gambar.
        Body JSON: {"items": ["Indomie Goreng", "Sabun Lifebuoy"]}
        """
        body = request.get_json(silent=True)
        if not body or "items" not in body:
            return _err("Body JSON harus punya field 'items'.", 400)

        items = body["items"]
        if not isinstance(items, list) or not all(isinstance(i, str) for i in items):
            return _err("'items' harus array of string.", 400)
        if len(items) == 0:
            return jsonify({"results": []})
        if len(items) > 100:
            return _err("Maks 100 item per request.", 400)

        try:
            _, _, classifier, _ = get_models()
        except Exception as e:
            return _err(f"Model belum siap: {e}", 503)

        return jsonify({"results": classify_items(classifier, items)})

    # ------------------------------------------------------------------
    # 404
    # ------------------------------------------------------------------
    @app.errorhandler(404)
    def not_found(_):
        return jsonify({"error": "Endpoint tidak ditemukan.", "endpoints": [
            "GET /health", "POST /predict", "POST /classify"
        ]}), 404

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=7860, debug=False)
