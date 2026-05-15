"""
Fase 2 — Training YOLOv8 OBB untuk deteksi region struk.

Jalankan di Windows dengan venv-yolo aktif:
    venv-yolo\\Scripts\\activate
    python ai/fase2_train_yolo.py

Hasil training tersimpan di:
    models/yolo/best.pt   ← dipakai di fase3
    runs/obb/train*/      ← log TensorBoard & grafik
"""

import os
import shutil

from ultralytics import YOLO

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai.config import (
    YOLO_BASE_MODEL, YOLO_DATA_YAML, YOLO_EPOCHS,
    YOLO_IMG_SIZE, YOLO_BATCH, YOLO_PATIENCE,
    YOLO_MODEL_DIR, YOLO_BEST_PT,
)


def train():
    if not os.path.exists(YOLO_DATA_YAML):
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {YOLO_DATA_YAML}\n"
            "Download dataset dari Roboflow terlebih dahulu ke folder data/yolo_dataset/."
        )

    os.makedirs(YOLO_MODEL_DIR, exist_ok=True)

    model = YOLO(YOLO_BASE_MODEL)

    results = model.train(
        data=YOLO_DATA_YAML,
        epochs=YOLO_EPOCHS,
        imgsz=YOLO_IMG_SIZE,
        batch=YOLO_BATCH,
        patience=YOLO_PATIENCE,
        device=0,              # GPU 0; ganti ke "cpu" jika tidak ada GPU
        workers=4,
        project="runs/obb",
        name="train",
        exist_ok=True,
        verbose=True,
    )

    # Salin best.pt ke folder models/ agar mudah dirujuk fase berikutnya
    src = os.path.join(results.save_dir, "weights", "best.pt")
    shutil.copy2(src, YOLO_BEST_PT)
    print(f"\nModel terbaik disalin ke: {YOLO_BEST_PT}")

    # Validasi akhir
    metrics = model.val()
    print(f"\nmAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    train()
