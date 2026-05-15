"""
Testing model YOLOv8-OBB hasil training.

Ada dua mode:
  1. Metrik formal  — evaluasi mAP/precision/recall pada test split dataset
  2. Visual         — tampilkan hasil deteksi pada foto struk dengan bounding box

Jalankan di Windows dengan venv-yolo aktif:
    venv-yolo\\Scripts\\activate

    # Mode 1: metrik pada test set
    python ai/test_yolo.py --mode metrics

    # Mode 2: visualisasi pada satu gambar
    python ai/test_yolo.py --mode visual --image data/raw_receipts/struk.jpg

    # Mode 2: visualisasi pada semua gambar di folder
    python ai/test_yolo.py --mode visual --image data/raw_receipts/

    # Keduanya sekaligus
    python ai/test_yolo.py --mode all --image data/raw_receipts/struk.jpg
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai.config import (
    YOLO_BEST_PT, YOLO_DATA_YAML, YOLO_IMG_SIZE,
    YOLO_CLASSES, BASE_DIR,
)

# Warna per kelas (BGR)
CLASS_COLORS = {
    "nama_toko":     (255, 100,  50),
    "line_item":     ( 50, 200,  50),
    "tanggal_waktu": ( 50, 150, 255),
    "total_belanja": ( 50,  50, 255),
}
DEFAULT_COLOR = (200, 200, 200)


# ---------------------------------------------------------------------------
# Mode 1: Metrik formal
# ---------------------------------------------------------------------------

def run_metrics(model):
    if not os.path.exists(YOLO_DATA_YAML):
        print(f"ERROR: data.yaml tidak ditemukan di {YOLO_DATA_YAML}")
        print("Pastikan dataset sudah di-download dengan download_dataset.py")
        return

    print("Mengevaluasi model pada test split...")
    print(f"Dataset : {YOLO_DATA_YAML}")
    print(f"Model   : {YOLO_BEST_PT}")
    print()

    metrics = model.val(
        data=YOLO_DATA_YAML,
        split="test",
        imgsz=YOLO_IMG_SIZE,
        verbose=False,
    )

    # Metrik keseluruhan
    print("=" * 50)
    print("HASIL EVALUASI — TEST SET")
    print("=" * 50)
    print(f"  mAP50      : {metrics.box.map50:.4f}  ({metrics.box.map50*100:.1f}%)")
    print(f"  mAP50-95   : {metrics.box.map:.4f}  ({metrics.box.map*100:.1f}%)")
    print(f"  Precision  : {metrics.box.mp:.4f}")
    print(f"  Recall     : {metrics.box.mr:.4f}")
    print()

    # Metrik per kelas
    print("Per kelas:")
    print(f"  {'Kelas':<18} {'Precision':>10} {'Recall':>10} {'mAP50':>10}")
    print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*10}")
    for i, name in enumerate(metrics.names.values()):
        p  = metrics.box.p[i]  if i < len(metrics.box.p)  else 0
        r  = metrics.box.r[i]  if i < len(metrics.box.r)  else 0
        ap = metrics.box.ap50[i] if i < len(metrics.box.ap50) else 0
        print(f"  {name:<18} {p:>10.4f} {r:>10.4f} {ap:>10.4f}")

    print()
    _print_verdict(metrics.box.map50)

    return metrics


def _print_verdict(map50):
    print("=" * 50)
    if map50 >= 0.90:
        print("STATUS: SANGAT BAIK (mAP50 ≥ 90%)")
    elif map50 >= 0.85:
        print("STATUS: BAIK — memenuhi target capstone (mAP50 ≥ 85%)")
    elif map50 >= 0.70:
        print("STATUS: CUKUP — perlu tambahan data atau training lebih lama")
    else:
        print("STATUS: KURANG — pertimbangkan: lebih banyak data, yolov8s-obb, atau epochs lebih banyak")
    print("=" * 50)


# ---------------------------------------------------------------------------
# Mode 2: Visualisasi
# ---------------------------------------------------------------------------

def run_visual(model, image_path, save_dir=None):
    path = Path(image_path)

    if path.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        images = [p for p in path.iterdir() if p.suffix.lower() in exts]
        if not images:
            print(f"Tidak ada gambar di folder: {image_path}")
            return
        print(f"Memproses {len(images)} gambar dari {image_path}")
        for img in sorted(images):
            _detect_and_draw(model, str(img), save_dir)
    else:
        if not path.exists():
            print(f"ERROR: Gambar tidak ditemukan: {image_path}")
            return
        _detect_and_draw(model, str(path), save_dir)


def _detect_and_draw(model, image_path, save_dir):
    image = cv2.imread(image_path)
    if image is None:
        print(f"  [SKIP] Tidak bisa membaca: {image_path}")
        return

    results = model(image, verbose=False)
    result  = results[0]

    annotated = image.copy()
    detections = []

    if result.obb is not None and len(result.obb):
        quads     = result.obb.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2)
        class_ids = result.obb.cls.cpu().numpy().astype(int)
        confs     = result.obb.conf.cpu().numpy()

        for quad, cls_id, conf in zip(quads, class_ids, confs):
            name  = YOLO_CLASSES[cls_id] if cls_id < len(YOLO_CLASSES) else str(cls_id)
            color = CLASS_COLORS.get(name, DEFAULT_COLOR)
            pts   = quad.astype(np.int32).reshape((-1, 1, 2))

            cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=2)

            # Label di sudut kiri atas kotak
            label    = f"{name} {conf:.2f}"
            tx, ty   = int(quad[0][0]), int(quad[0][1]) - 6
            cv2.putText(annotated, label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            detections.append((name, conf))

    # Ringkasan di sudut gambar
    h, w = annotated.shape[:2]
    _draw_legend(annotated, detections)

    # Simpan atau tampilkan
    filename = Path(image_path).stem
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"{filename}_detected.jpg")
        cv2.imwrite(out_path, annotated)
        print(f"  Tersimpan: {out_path}")
    else:
        out_path = os.path.join(BASE_DIR, "runs", "test_visual", f"{filename}_detected.jpg")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, annotated)
        print(f"  [{filename}] {len(detections)} deteksi → {out_path}")

    for name, conf in detections:
        print(f"    • {name:<18} conf={conf:.3f}")


def _draw_legend(image, detections):
    """Tampilkan ringkasan deteksi di pojok kiri bawah gambar."""
    h = image.shape[0]
    count = {}
    for name, _ in detections:
        count[name] = count.get(name, 0) + 1

    y = h - 10
    for name, n in reversed(sorted(count.items())):
        color = CLASS_COLORS.get(name, DEFAULT_COLOR)
        text  = f"{name}: {n}"
        cv2.putText(image, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        y -= 18


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Testing model YOLOv8-OBB NotePay")
    parser.add_argument(
        "--mode", choices=["metrics", "visual", "all"], default="metrics",
        help="metrics = evaluasi mAP | visual = gambar dengan bounding box | all = keduanya"
    )
    parser.add_argument(
        "--image", default=None,
        help="Path gambar atau folder (diperlukan untuk mode visual/all)"
    )
    parser.add_argument(
        "--model", default=YOLO_BEST_PT,
        help=f"Path model .pt (default: {YOLO_BEST_PT})"
    )
    parser.add_argument(
        "--save-dir", default=None,
        help="Folder untuk menyimpan hasil visualisasi (default: runs/test_visual/)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: Model tidak ditemukan: {args.model}")
        print("Jalankan fase2_train_yolo.py terlebih dahulu.")
        sys.exit(1)

    from ultralytics import YOLO
    print(f"Memuat model: {args.model}")
    model = YOLO(args.model)

    if args.mode in ("metrics", "all"):
        run_metrics(model)

    if args.mode in ("visual", "all"):
        if not args.image:
            print("ERROR: --image diperlukan untuk mode visual/all")
            sys.exit(1)
        print()
        run_visual(model, args.image, args.save_dir)


if __name__ == "__main__":
    main()
