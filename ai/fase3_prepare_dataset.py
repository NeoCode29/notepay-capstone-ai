"""
Fase 3 — Penyiapan dataset OCR dari hasil inferensi YOLOv8.

Ada dua mode sumber:

  MODE 1 — raw (default)
    Jalankan YOLO inference pada foto struk mentah, lalu auto-label dengan EasyOCR.
    Cocok jika hanya punya foto mentah tanpa anotasi.

    python ai/fase3_prepare_dataset.py --raw-dir data/raw_receipts

  MODE 2 — annotated
    Koordinat crop dari file anotasi Roboflow (.txt), label teks tetap dari EasyOCR.
    Tidak perlu YOLO model (best.pt) — lebih cepat karena skip inference deteksi.

    python ai/fase3_prepare_dataset.py --annotated data/yolo_dataset
    python ai/fase3_prepare_dataset.py --annotated data/yolo_dataset --splits train valid
    python ai/fase3_prepare_dataset.py --annotated data/yolo_dataset --splits test --classes line_item

Kedua mode dapat digabung (jalankan berurutan) — crop_idx dilanjutkan otomatis
dari entri yang sudah ada di labels.csv agar tidak overwrite.

Setelah selesai, periksa data/ocr_dataset/labels.csv secara manual
untuk mengoreksi label yang salah sebelum masuk ke fase4.
"""

import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai.config import (
    YOLO_BEST_PT, OCR_DATASET_DIR, OCR_LABELS_CSV,
    CROP_HEIGHT, CROP_WIDTH, YOLO_CLASSES,
)


def obb_to_quad(obb_result):
    """Konversi koordinat OBB (xywhr) ke 4 titik sudut."""
    points = obb_result.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2)
    return points  # shape: (N, 4, 2)


def sort_quad_points(pts):
    """
    Urutkan 4 titik OBB menjadi urutan TL-TR-BR-BL yang konsisten.
    YOLO tidak menjamin urutan titik, sehingga perlu diurutkan manual.
    """
    pts = np.array(pts, dtype=np.float32)
    # Urutkan by y → ambil 2 teratas dan 2 terbawah
    sorted_y = pts[np.argsort(pts[:, 1])]
    top = sorted_y[:2][np.argsort(sorted_y[:2, 0])]   # TL, TR
    bot = sorted_y[2:][np.argsort(sorted_y[2:, 0])]   # BL, BR
    return np.array([top[0], top[1], bot[1], bot[0]], dtype=np.float32)  # TL TR BR BL


def pad_quad(pts, pad_ratio=0.08):
    """
    Perluas quad ke luar sebesar pad_ratio dari ukurannya.
    Mencegah teks terpotong saat OBB YOLO sedikit terlalu ketat.
    """
    center = pts.mean(axis=0)
    return (pts + (pts - center) * pad_ratio).astype(np.float32)


def deskew_crop(image, quad, target_h=CROP_HEIGHT * 2):
    """
    Perspective transform quad → gambar lurus dengan aspect ratio natural.
    Lebar output menyesuaikan lebar asli region agar teks tidak menciut.
    Tinggi selalu target_h; lebar minimum CROP_WIDTH.
    """
    pts = pad_quad(sort_quad_points(quad))

    # Hitung dimensi natural region OBB
    w = float(np.linalg.norm(pts[1] - pts[0]))   # lebar (sisi atas)
    h = float(np.linalg.norm(pts[3] - pts[0]))   # tinggi (sisi kiri)

    # Jika portrait (h > w), rotasi urutan titik 90° agar landscape
    if h > w:
        pts = np.array([pts[3], pts[0], pts[1], pts[2]], dtype=np.float32)
        w, h = h, w

    # Lebar output proporsional dengan aspect ratio natural region
    out_h = target_h
    out_w = max(CROP_WIDTH, int(target_h * (w / max(h, 1))))

    dst = np.array([
        [0,     0    ],
        [out_w, 0    ],
        [out_w, out_h],
        [0,     out_h],
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(image, M, (out_w, out_h))


def enhance_crop(crop, out_h=CROP_HEIGHT, out_w=CROP_WIDTH):
    """
    Tingkatkan kualitas crop thermal receipt sebelum masuk CRNN:
    1. Grayscale
    2. Gaussian blur — haluskan noise sebelum threshold
    3. CLAHE — perbaiki kontras lokal
    4. Otsu threshold — otomatis cari nilai terbaik untuk gambar ini
    5. Resize ke ukuran input CRNN (128×32)
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop.copy()

    # Haluskan noise dulu — kunci agar threshold tidak berantakan
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
    gray = clahe.apply(gray)

    # Otsu otomatis cari threshold global terbaik — lebih stabil dari adaptive untuk crop kecil
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Resize proporsional ke tinggi out_h, lalu pad kanan agar lebar = out_w.
    # Padding putih (255) aman untuk CTC — model memprediksi blank di area kosong.
    h, w = binary.shape[:2]
    new_w = max(1, int(w * out_h / h))
    resized = cv2.resize(binary, (new_w, out_h), interpolation=cv2.INTER_AREA)

    if new_w > out_w:
        # Teks lebih lebar dari out_w — compress horizontal sedikit, jangan potong
        return cv2.resize(binary, (out_w, out_h), interpolation=cv2.INTER_AREA)
    pad = np.full((out_h, out_w - new_w), 255, dtype=np.uint8)
    return np.hstack([resized, pad])


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

    pairs = []
    for quad, cls_id in zip(quads, class_ids):
        class_name = YOLO_CLASSES[cls_id] if cls_id < len(YOLO_CLASSES) else str(cls_id)
        if class_filter and class_name not in class_filter:
            continue
        crop = enhance_crop(deskew_crop(image, quad))
        pairs.append((crop, class_name))

    if not pairs:
        return []

    crops_only = [p[0] for p in pairs]
    labels = auto_label(crops_only, ocr_reader)

    return [(crop, label, class_name) for (crop, class_name), label in zip(pairs, labels)]


def _next_crop_idx():
    """
    Baca labels.csv yang sudah ada dan kembalikan indeks crop berikutnya.
    Sehingga mode raw dan annotated bisa dijalankan bergantian tanpa overwrite.
    """
    if not os.path.exists(OCR_LABELS_CSV):
        return 0
    with open(OCR_LABELS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return 0
    indices = []
    for r in rows:
        stem = Path(r["filepath"]).stem  # "crop_00042"
        if stem.startswith("crop_"):
            try:
                indices.append(int(stem.split("_")[1]))
            except ValueError:
                pass
    return max(indices) + 1 if indices else len(rows)


def _append_to_csv(rows):
    """Tambahkan baris ke labels.csv (buat baru jika belum ada)."""
    file_exists = os.path.exists(OCR_LABELS_CSV)
    with open(OCR_LABELS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filepath", "label", "class"],
            quoting=csv.QUOTE_ALL,
        )
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def process_from_label_file(image_path, label_path, class_filter=None):
    """
    Baca anotasi OBB dari file .txt Roboflow/YOLO dan crop setiap region.

    Format baris label: class_id x1 y1 x2 y2 x3 y3 x4 y4  (koordinat dinormalisasi)

    Returns: list of (crop_image, class_name)  — tanpa label teks
    """
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  [SKIP] Tidak bisa membaca: {image_path}")
        return []

    h_img, w_img = image.shape[:2]

    if not os.path.exists(label_path):
        return []  # gambar tanpa anotasi

    results = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 9:
                continue

            cls_id = int(parts[0])
            class_name = YOLO_CLASSES[cls_id] if cls_id < len(YOLO_CLASSES) else str(cls_id)
            if class_filter and class_name not in class_filter:
                continue

            # Denormalisasi 4 titik sudut OBB
            coords = list(map(float, parts[1:]))
            quad = np.array([
                [coords[0] * w_img, coords[1] * h_img],
                [coords[2] * w_img, coords[3] * h_img],
                [coords[4] * w_img, coords[5] * h_img],
                [coords[6] * w_img, coords[7] * h_img],
            ], dtype=np.float32)

            crop = enhance_crop(deskew_crop(image, quad))
            results.append((crop, class_name))

    return results


def main_from_annotations(dataset_dir, splits=None, class_filter=None):
    """
    Mode annotated: koordinat crop dari anotasi Roboflow (.txt), label teks dari EasyOCR.
    Tidak perlu YOLO model — lebih cepat dan tidak butuh best.pt.
    """
    if splits is None:
        splits = ["train", "valid", "test"]

    import easyocr
    print("Memuat EasyOCR (model pertama kali akan diunduh)...")
    reader = easyocr.Reader(["id", "en"], gpu=True)

    crop_idx = _next_crop_idx()
    rows = []
    total = 0

    for split in splits:
        img_dir   = Path(dataset_dir) / split / "images"
        label_dir = Path(dataset_dir) / split / "labels"

        if not img_dir.exists():
            print(f"  [SKIP] Folder tidak ditemukan: {img_dir}")
            continue

        image_files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        print(f"\nSplit '{split}': {len(image_files)} gambar")

        for img_path in image_files:
            label_path = label_dir / (img_path.stem + ".txt")
            pairs = process_from_label_file(img_path, label_path, class_filter)
            if not pairs:
                continue

            crops_only = [p[0] for p in pairs]
            labels = auto_label(crops_only, reader)

            print(f"  {img_path.name} → {len(pairs)} crop")
            for (crop, class_name), label in zip(pairs, labels):
                class_dir = os.path.join(OCR_DATASET_DIR, class_name)
                os.makedirs(class_dir, exist_ok=True)
                filename = f"crop_{crop_idx:05d}.jpg"
                cv2.imwrite(os.path.join(class_dir, filename), crop)
                rows.append({
                    "filepath": f"{class_name}/{filename}",
                    "label":    label,
                    "class":    class_name,
                })
                crop_idx += 1
                total += 1

    _append_to_csv(rows)

    print(f"\nSelesai. {total} crop baru disimpan ke {OCR_DATASET_DIR}/<class>/")
    print(f"CSV label: {OCR_LABELS_CSV}")
    print("\nLangkah selanjutnya:")
    print("  Buka labels.csv, periksa kolom 'label', lalu jalankan fase4_train_crnn.py.")


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

    image_paths = [
        os.path.join(raw_dir, f)
        for f in os.listdir(raw_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]
    print(f"Memproses {len(image_paths)} foto struk...")

    rows = []
    crop_idx = _next_crop_idx()

    for img_path in image_paths:
        print(f"  {os.path.basename(img_path)}", end=" ")
        pairs = process_image(img_path, yolo, reader, class_filter)
        print(f"→ {len(pairs)} crop")

        for crop, label, class_name in pairs:
            class_dir = os.path.join(OCR_DATASET_DIR, class_name)
            os.makedirs(class_dir, exist_ok=True)
            filename = f"crop_{crop_idx:05d}.jpg"
            cv2.imwrite(os.path.join(class_dir, filename), crop)
            rows.append({
                "filepath": f"{class_name}/{filename}",
                "label":    label,
                "class":    class_name,
            })
            crop_idx += 1

    _append_to_csv(rows)

    print(f"\nSelesai. {len(rows)} crop baru disimpan ke {OCR_DATASET_DIR}/<class>/")
    print(f"CSV label: {OCR_LABELS_CSV}")
    print("\nLangkah selanjutnya:")
    print("  Buka labels.csv, periksa kolom 'label', lalu jalankan fase4_train_crnn.py.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fase 3 — Siapkan dataset OCR dari foto mentah atau anotasi Roboflow.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh penggunaan:
  # Mode raw — YOLO inference + EasyOCR auto-label
  python ai/fase3_prepare_dataset.py --raw-dir data/raw_receipts

  # Mode annotated — baca anotasi Roboflow langsung, tanpa OCR
  python ai/fase3_prepare_dataset.py --annotated data/yolo_dataset
  python ai/fase3_prepare_dataset.py --annotated data/yolo_dataset --splits train valid
  python ai/fase3_prepare_dataset.py --annotated data/yolo_dataset --splits test --classes line_item
        """,
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--raw-dir",
        help="Mode raw: folder foto struk mentah (JPG/PNG)",
    )
    src.add_argument(
        "--annotated",
        metavar="DATASET_DIR",
        help="Mode annotated: root folder yolo_dataset yang berisi train/valid/test",
    )

    parser.add_argument(
        "--splits", nargs="+", default=["train", "valid", "test"],
        metavar="SPLIT",
        help="Split yang diproses untuk mode annotated (default: train valid test)",
    )
    parser.add_argument(
        "--classes", nargs="*", default=None,
        help="Filter kelas yang di-crop, mis: line_item total_belanja",
    )
    args = parser.parse_args()

    if args.annotated:
        main_from_annotations(args.annotated, args.splits, args.classes)
    else:
        main(args.raw_dir, args.classes)
