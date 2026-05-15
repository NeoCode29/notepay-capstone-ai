"""
Fase 3 — Penyiapan dataset OCR dari hasil inferensi YOLOv8.

Pipeline:
  1. Jalankan YOLO best.pt pada semua foto struk mentah
  2. Ekstrak setiap region OBB dan luruskan dengan perspective transform
  3. Auto-label dengan EasyOCR sebagai label sementara
  4. Simpan pasangan (gambar_crop, label_teks) ke data/ocr_dataset/

Jalankan di Windows dengan venv-yolo aktif:
    venv-yolo\\Scripts\\activate
    pip install easyocr   # hanya perlu sekali
    python ai/fase3_prepare_dataset.py --raw-dir data/raw_receipts

Setelah selesai, periksa data/ocr_dataset/labels.csv secara manual
untuk mengoreksi label yang salah sebelum masuk ke fase4.
"""

import argparse
import csv
import os

import cv2
import numpy as np
from ultralytics import YOLO

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai.config import (
    YOLO_BEST_PT, OCR_IMAGES_DIR, OCR_LABELS_CSV,
    CROP_HEIGHT, CROP_WIDTH,
)


def obb_to_quad(obb_result):
    """Konversi koordinat OBB (xywhr) ke 4 titik sudut dalam urutan TL-TR-BR-BL."""
    points = obb_result.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2)
    return points  # shape: (N, 4, 2)


def deskew_crop(image, quad, out_h=CROP_HEIGHT, out_w=CROP_WIDTH):
    """Perspective transform quad menjadi gambar lurus berukuran out_h x out_w."""
    src = quad.astype(np.float32)
    dst = np.array([
        [0,     0    ],
        [out_w, 0    ],
        [out_w, out_h],
        [0,     out_h],
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (out_w, out_h))


def auto_label(crops, reader):
    """Baca teks dari list gambar crop menggunakan EasyOCR."""
    labels = []
    for crop in crops:
        result = reader.readtext(crop, detail=0, paragraph=True)
        text = " ".join(result).strip()
        labels.append(text)
    return labels


def process_image(image_path, yolo_model, ocr_reader, class_filter=None):
    """
    Deteksi region struk pada satu foto, ekstrak crop, auto-label.

    Returns: list of (crop_image, label, class_name)
    class_filter: list nama kelas yang di-crop, None = semua kelas
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"  [SKIP] Tidak bisa membaca: {image_path}")
        return []

    results = yolo_model(image, verbose=False)
    if not results or results[0].obb is None:
        return []

    obb = results[0].obb
    quads = obb_to_quad(obb)
    class_ids = obb.cls.cpu().numpy().astype(int)

    from ai.config import YOLO_CLASSES
    pairs = []
    for quad, cls_id in zip(quads, class_ids):
        class_name = YOLO_CLASSES[cls_id] if cls_id < len(YOLO_CLASSES) else str(cls_id)
        if class_filter and class_name not in class_filter:
            continue
        crop = deskew_crop(image, quad)
        pairs.append((crop, class_name))

    if not pairs:
        return []

    crops_only = [p[0] for p in pairs]
    labels = auto_label(crops_only, ocr_reader)

    return [(crop, label, class_name) for (crop, class_name), label in zip(pairs, labels)]


def main(raw_dir, class_filter=None):
    if not os.path.exists(YOLO_BEST_PT):
        raise FileNotFoundError(
            f"Model YOLO tidak ditemukan: {YOLO_BEST_PT}\n"
            "Jalankan fase2_train_yolo.py terlebih dahulu."
        )

    # EasyOCR hanya untuk auto-labeling awal, bukan fitur utama produk
    import easyocr
    print("Memuat EasyOCR (model pertama kali akan diunduh)...")
    reader = easyocr.Reader(["id", "en"], gpu=True)

    print(f"Memuat YOLOv8: {YOLO_BEST_PT}")
    yolo = YOLO(YOLO_BEST_PT)

    os.makedirs(OCR_IMAGES_DIR, exist_ok=True)

    image_paths = [
        os.path.join(raw_dir, f)
        for f in os.listdir(raw_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]
    print(f"Memproses {len(image_paths)} foto struk...")

    rows = []
    crop_idx = 0

    for img_path in image_paths:
        print(f"  {os.path.basename(img_path)}", end=" ")
        pairs = process_image(img_path, yolo, reader, class_filter)
        print(f"→ {len(pairs)} crop")

        for crop, label, class_name in pairs:
            filename = f"crop_{crop_idx:05d}.jpg"
            save_path = os.path.join(OCR_IMAGES_DIR, filename)
            cv2.imwrite(save_path, crop)
            rows.append({"filename": filename, "label": label, "class": class_name, "verified": "0"})
            crop_idx += 1

    with open(OCR_LABELS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label", "class", "verified"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSelesai. {crop_idx} crop disimpan.")
    print(f"CSV label: {OCR_LABELS_CSV}")
    print("\nLangkah selanjutnya:")
    print("  Buka labels.csv, periksa kolom 'label', ubah 'verified' ke '1' jika sudah benar.")
    print("  Jalankan fase4_train_crnn.py setelah verifikasi selesai.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir", required=True,
        help="Folder berisi foto struk mentah (JPG/PNG)"
    )
    parser.add_argument(
        "--classes", nargs="*", default=None,
        help="Filter kelas yang di-crop, mis: line_item total_belanja"
    )
    args = parser.parse_args()
    main(args.raw_dir, args.classes)
