"""
Konfigurasi terpusat untuk semua fase training OCR pipeline.
Sesuaikan path dan hyperparameter di sini sebelum menjalankan script lain.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input data
YOLO_DATASET_DIR = os.path.join(BASE_DIR, "data", "yolo_dataset")  # hasil download Roboflow
YOLO_DATA_YAML   = os.path.join(YOLO_DATASET_DIR, "data.yaml")

OCR_DATASET_DIR  = os.path.join(BASE_DIR, "data", "ocr_dataset")   # hasil fase3
OCR_IMAGES_DIR   = os.path.join(OCR_DATASET_DIR, "images")
OCR_LABELS_CSV   = os.path.join(OCR_DATASET_DIR, "labels.csv")

# Output model
MODELS_DIR       = os.path.join(BASE_DIR, "models")
YOLO_MODEL_DIR   = os.path.join(MODELS_DIR, "yolo")
CRNN_MODEL_DIR   = os.path.join(MODELS_DIR, "crnn")

YOLO_BEST_PT     = os.path.join(YOLO_MODEL_DIR, "best_yolo_v1.pt")
CRNN_KERAS_PATH  = os.path.join(CRNN_MODEL_DIR, "crnn_model.keras")
CRNN_INFER_PATH  = os.path.join(CRNN_MODEL_DIR, "inference_model.keras")

# TensorBoard logs
LOGS_DIR         = os.path.join(BASE_DIR, "logs", "crnn")

# ---------------------------------------------------------------------------
# YOLOv8 OBB — Fase 2
# ---------------------------------------------------------------------------
YOLO_BASE_MODEL  = "yolov8n-obb.pt"   # ganti ke yolov8s-obb.pt untuk akurasi lebih tinggi
YOLO_EPOCHS      = 100
YOLO_IMG_SIZE    = 640
YOLO_BATCH       = 16
YOLO_PATIENCE    = 20                  # early stopping
YOLO_CLASSES     = ["line_item", "nama_toko", "tanggal_waktu", "total_belanja"]
YOLO_RESUME      = False               # set True untuk lanjut dari last.pt

# ---------------------------------------------------------------------------
# OCR Dataset — Fase 3
# ---------------------------------------------------------------------------
CROP_HEIGHT = 32    # tinggi gambar potongan setelah deskew (px)
CROP_WIDTH  = 512   # lebar gambar potongan — 512 agar line_item panjang tidak menciut

# ---------------------------------------------------------------------------
# CRNN + CTC — Fase 4
# ---------------------------------------------------------------------------

# Karakter yang dikenali model. Indeks 0 dipakai sebagai token blank CTC.
# Urutan penting: jangan diubah setelah training dimulai.
BLANK_TOKEN = "[BLANK]"
CHARACTERS  = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,/-()%")

# char → int  (0 = blank, 1..N = karakter)
CHAR_TO_IDX = {ch: i + 1 for i, ch in enumerate(CHARACTERS)}
IDX_TO_CHAR = {i + 1: ch for i, ch in enumerate(CHARACTERS)}
IDX_TO_CHAR[0] = BLANK_TOKEN

NUM_CLASSES  = len(CHARACTERS) + 1   # +1 untuk blank
MAX_LABEL_LEN = 32                    # panjang label maksimum yang didukung

CRNN_EPOCHS     = 60
CRNN_BATCH_SIZE = 32
CRNN_LR         = 1e-3
VAL_SPLIT       = 0.15
