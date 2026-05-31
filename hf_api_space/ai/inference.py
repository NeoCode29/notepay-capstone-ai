"""
Inference end-to-end: foto struk → teks terstruktur + kategori pengeluaran (JSON).

Pipeline:
    Foto
      → [Stage 1] YOLO OBB  — deteksi 4 region (nama_toko, line_item,
      |                        tanggal_waktu, total_belanja)
      → [Stage 2] CRNN+CTC  — text recognition per crop (deskew dulu)
      → [Stage 3] Classifier — kategori tiap line_item
      → JSON terstruktur

Memenuhi syarat capstone "simple inference code".

Setup (install ultralytics di venv-tf jika belum):
    pip install ultralytics

Jalankan di WSL2 dengan venv-tf aktif:
    source ~/venv-tf/bin/activate

    # Satu gambar
    python ai/inference.py --image foto_struk.jpg

    # Satu gambar + simpan hasil visual
    python ai/inference.py --image foto_struk.jpg --save-visual

    # Tanpa classifier (debug OCR saja)
    python ai/inference.py --image foto_struk.jpg --no-classify

    # Semua gambar di folder
    python ai/inference.py --image data/raw_receipts/

    # Output JSON ke file
    python ai/inference.py --image foto_struk.jpg --output hasil.json
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# GPU aktif secara default. Set CUDA_VISIBLE_DEVICES=-1 untuk paksa CPU.
# Contoh: CUDA_VISIBLE_DEVICES=-1 python ai/inference.py --image foto.jpg

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai.config import (
    YOLO_BEST_PT, CRNN_INFER_PATH,
    CROP_HEIGHT, CROP_WIDTH,
    IDX_TO_CHAR, YOLO_CLASSES, BASE_DIR,
    CLASSIFIER_MODEL_PATH, EXPENSE_CATEGORIES,
)

def _setup_gpu():
    """
    Konfigurasi GPU untuk inference dual-framework (PyTorch + TensorFlow).

    Strategi: PyTorch (YOLO) pakai GPU dengan memory growth.
    TF (CRNN) juga GPU tapi dibatasi agar tidak OOM bersamaan.
    Keduanya diinisialisasi sebelum model di-load supaya CUDA context
    hanya dibuat sekali dan tidak konflik.
    """
    use_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "") != "-1"

    if not use_gpu:
        # Mode CPU eksplisit — backward compat dengan env lama
        return "cpu", "cpu"

    # TensorFlow: memory growth (ambil sesuai kebutuhan, tidak pre-alokasi semua)
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.optimizer.set_experimental_options({"layout_optimizer": False})
        tf_device = "cuda" if gpus else "cpu"
    except Exception:
        tf_device = "cpu"

    # PyTorch: cek ketersediaan CUDA
    try:
        import torch
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        torch_device = "cpu"

    return torch_device, tf_device

# Warna per kelas untuk visualisasi (BGR)
CLASS_COLORS = {
    "nama_toko":     (255, 100,  50),
    "line_item":     ( 50, 200,  50),
    "tanggal_waktu": ( 50, 150, 255),
    "total_belanja": (  0,  50, 255),
}
DEFAULT_COLOR = (200, 200, 200)


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

def load_models(yolo_path=None, crnn_path=None, classifier_path=None):
    from ultralytics import YOLO
    import keras

    yolo_path       = yolo_path       or YOLO_BEST_PT
    crnn_path       = crnn_path       or CRNN_INFER_PATH
    classifier_path = classifier_path or CLASSIFIER_MODEL_PATH

    if not os.path.exists(yolo_path):
        raise FileNotFoundError(f"Model YOLO tidak ditemukan: {yolo_path}")
    if not os.path.exists(crnn_path):
        raise FileNotFoundError(f"Model CRNN tidak ditemukan: {crnn_path}")

    torch_device, tf_device = _setup_gpu()
    print(f"  Device YOLO : {torch_device}  |  Device CRNN : {tf_device}")

    print(f"  YOLO : {yolo_path}")
    yolo = YOLO(yolo_path)
    # Warm-up satu frame kosong — CUDA context PyTorch terbentuk lebih dulu
    # sebelum TF di-load, sehingga keduanya tidak rebutan inisialisasi cuDNN.
    yolo(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False, device=torch_device)

    print(f"  CRNN : {crnn_path}")
    # Keras 3.x tidak bisa infer output_shape Lambda (cast float16→float32) saat from_config.
    # Patch sementara: cast tidak mengubah shape, jadi return input_shape aman.
    try:
        from keras.src.layers.core.lambda_layer import Lambda as _KLambda
        _orig_cos = _KLambda.compute_output_shape
        def _patched_cos(self, input_shape):
            try:
                return _orig_cos(self, input_shape)
            except NotImplementedError:
                return input_shape
        _KLambda.compute_output_shape = _patched_cos
    except ImportError:
        pass
    crnn = keras.models.load_model(crnn_path, compile=False, safe_mode=False)

    # Classifier opsional — tidak error jika belum di-train
    classifier = None
    if os.path.exists(classifier_path):
        print(f"  Classifier : {classifier_path}")
        classifier = keras.models.load_model(classifier_path, compile=False)
    else:
        print(f"  Classifier : tidak ditemukan (jalankan fase5_train_classifier.py)")

    return yolo, crnn, classifier, torch_device


def classify_items(classifier, items: list[str]) -> list[dict]:
    """
    Klasifikasikan list teks item struk ke kategori pengeluaran.
    Kembalikan list dict {text, category, confidence}.
    Jika classifier None, kembalikan Lain-lain untuk semua item.
    """
    if not items:
        return []

    if classifier is None:
        return [{"text": t, "category": "Lain-lain", "confidence": 0.0} for t in items]

    import numpy as np
    import tensorflow as tf
    arr   = tf.constant([[item] for item in items])
    preds = classifier(arr, training=False).numpy()
    results = []
    for text, pred in zip(items, preds):
        idx  = int(np.argmax(pred))
        conf = float(pred[idx])
        results.append({
            "text"      : text,
            "category"  : EXPENSE_CATEGORIES[idx],
            "confidence": round(conf, 4),
        })
    return results


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

def parse_amount(text: str) -> float | None:
    """
    Ekstrak nilai rupiah dari teks OCR.
    Contoh: "Rp 125.000" → 125000.0 | "1.250.000" → 1250000.0 | "abc" → None
    """
    if not text:
        return None
    # Hapus prefix non-angka (Rp, IDR, spasi, dll)
    cleaned = re.sub(r"[^\d.,]", "", text)
    if not cleaned:
        return None

    # Format Indonesia: titik = pemisah ribuan, koma = desimal
    # Deteksi apakah ada koma di akhir sebagai desimal
    if re.search(r",\d{1,2}$", cleaned):
        # Ada desimal dengan koma: "125.000,50"
        cleaned = cleaned.replace(".", "").replace(",", ".")
    else:
        # Titik hanya sebagai ribuan: "125.000" → "125000"
        cleaned = cleaned.replace(".", "").replace(",", "")

    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_datetime(text: str) -> str | None:
    """
    Parse tanggal/waktu dari teks OCR ke format ISO (YYYY-MM-DD HH:MM).
    Mendukung format umum struk Indonesia.
    Kembalikan None jika tidak bisa di-parse.
    """
    if not text:
        return None

    patterns = [
        # DD/MM/YYYY HH:MM:SS  atau  DD/MM/YYYY HH:MM
        (r"(\d{2})[/\-.](\d{2})[/\-.](\d{4})\s+(\d{2}:\d{2})(?::\d{2})?",
         lambda m: f"{m[3]}-{m[2]}-{m[1]} {m[4]}"),
        # YYYY-MM-DD HH:MM
        (r"(\d{4})[/\-.](\d{2})[/\-.](\d{2})\s+(\d{2}:\d{2})",
         lambda m: f"{m[1]}-{m[2]}-{m[3]} {m[4]}"),
        # DD/MM/YYYY saja
        (r"(\d{2})[/\-.](\d{2})[/\-.](\d{4})",
         lambda m: f"{m[3]}-{m[2]}-{m[1]}"),
        # Tanggal dengan nama bulan Indonesia singkat: 12 MEI 2025
        (r"(\d{1,2})\s+(JAN|FEB|MAR|APR|MEI|JUN|JUL|AGU|SEP|OKT|NOV|DES)\s+(\d{4})",
         lambda m: f"{m[3]}-{_BULAN_IDX.get(m[2], '01'):02d}-{int(m[1]):02d}"),
    ]
    _BULAN_IDX = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MEI": 5, "JUN": 6,
        "JUL": 7, "AGU": 8, "SEP": 9, "OKT": 10, "NOV": 11, "DES": 12,
    }

    upper = text.upper()
    for pattern, formatter in patterns:
        m = re.search(pattern, upper)
        if m:
            try:
                return formatter(m.groups())
            except Exception:
                continue
    return None


def summarize_categories(classified_items: list[dict]) -> dict:
    """
    Hitung jumlah item per kategori dari hasil klasifikasi.
    Kembalikan dict {kategori: count} diurutkan descending.
    """
    summary: dict[str, int] = {}
    for item in classified_items:
        cat = item.get("category", "Lain-lain")
        summary[cat] = summary.get(cat, 0) + 1
    return dict(sorted(summary.items(), key=lambda x: x[1], reverse=True))


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------

def _order_quad(pts):
    """
    Urutkan 4 titik OBB → [top-left, top-right, bottom-right, bottom-left].
    Diperlukan agar perspective transform menghasilkan gambar tegak (tidak terbalik).
    """
    pts  = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)

    s        = pts.sum(axis=1)
    rect[0]  = pts[np.argmin(s)]   # top-left  = x+y terkecil
    rect[2]  = pts[np.argmax(s)]   # bot-right = x+y terbesar

    diff     = np.diff(pts, axis=1)
    rect[1]  = pts[np.argmin(diff)]  # top-right = y-x terkecil
    rect[3]  = pts[np.argmax(diff)]  # bot-left  = y-x terbesar

    return rect


def deskew_crop(image, quad, out_h=CROP_HEIGHT, out_w=CROP_WIDTH):
    """Perspective transform OBB quad → gambar lurus (horizontal)."""
    src = _order_quad(quad)
    dst = np.array([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]], dtype=np.float32)
    M   = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (out_w, out_h))


def preprocess_crop(crop):
    """
    Crop BGR/gray → tensor (1, H, W, 1) float32 [0,1].
    Preprocessing sama dengan fase3 enhance_crop:
      - grayscale
      - binarisasi adaptif (background putih, teks hitam)
      - resize proporsional + pad kanan dengan putih
      - normalize [0,1]
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop

    # Hapus border hitam (dari warpPerspective) sebelum binarisasi
    # Crop ke bounding rect area non-hitam
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    coords  = cv2.findNonZero(mask)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        gray = gray[y:y+h, x:x+w]

    # Binarisasi adaptif — teks hitam, background putih
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )

    # Resize proporsional ke CROP_HEIGHT, lalu pad kanan
    h, w   = binary.shape
    new_w  = max(1, int(w * CROP_HEIGHT / h))
    resized = cv2.resize(binary, (new_w, CROP_HEIGHT), interpolation=cv2.INTER_AREA)

    if new_w >= CROP_WIDTH:
        out = cv2.resize(binary, (CROP_WIDTH, CROP_HEIGHT), interpolation=cv2.INTER_AREA)
    else:
        pad = np.full((CROP_HEIGHT, CROP_WIDTH - new_w), 255, dtype=np.uint8)
        out = np.hstack([resized, pad])

    return out.astype(np.float32)[np.newaxis, :, :, np.newaxis] / 255.0


def ctc_decode(logits):
    """Greedy CTC decode: argmax → hapus blank (0) dan consecutive duplicate."""
    indices = np.argmax(logits, axis=-1)
    prev, chars = -1, []
    for idx in indices:
        if idx != prev:
            if idx != 0:
                chars.append(IDX_TO_CHAR.get(int(idx), ""))
            prev = idx
    return "".join(chars)


# ---------------------------------------------------------------------------
# Pipeline utama
# ---------------------------------------------------------------------------

def run(image_path, yolo, crnn, classifier=None, conf_threshold=0.25,
        save_crops_dir=None, torch_device="cpu"):
    """
    Jalankan pipeline end-to-end pada satu gambar.
    Kembalikan dict hasil dan list deteksi (untuk visualisasi).
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")

    t0 = time.time()

    # ---- Stage 1: YOLO deteksi region ----
    results   = yolo(image, verbose=False, device=torch_device)
    obb       = results[0].obb
    if obb is None or len(obb) == 0:
        print("  [!] Tidak ada region struk terdeteksi.")
        return {}, [], time.time() - t0

    quads     = obb.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2)
    class_ids = obb.cls.cpu().numpy().astype(int)
    confs     = obb.conf.cpu().numpy()

    # ---- Stage 2: CRNN text recognition per region ----
    output     = {}
    detections = []   # untuk visualisasi

    for quad, cls_id, conf in zip(quads, class_ids, confs):
        if conf < conf_threshold:
            continue

        class_name = YOLO_CLASSES[cls_id] if cls_id < len(YOLO_CLASSES) else f"class_{cls_id}"
        crop       = deskew_crop(image, quad)
        tensor     = preprocess_crop(crop)
        logits     = crnn(tensor, training=False).numpy()[0]
        text       = ctc_decode(logits)

        if save_crops_dir:
            os.makedirs(save_crops_dir, exist_ok=True)
            crop_name = f"{class_name}_{len(output.get(class_name, []))}.jpg"
            cv2.imwrite(os.path.join(save_crops_dir, crop_name), crop)

        if class_name not in output:
            output[class_name] = []
        output[class_name].append(text)

        detections.append({
            "class": class_name,
            "conf" : float(conf),
            "text" : text,
            "quad" : quad,
        })

    elapsed = time.time() - t0

    # ---- Stage 3: Classify line items ----
    raw_items = output.get("line_item", [])
    output["line_item_classified"] = classify_items(classifier, raw_items)

    return output, detections, elapsed


# ---------------------------------------------------------------------------
# Visualisasi
# ---------------------------------------------------------------------------

def draw_results(image, detections):
    """Gambar bounding box OBB + teks OCR di atas gambar."""
    annotated = image.copy()

    for det in detections:
        color = CLASS_COLORS.get(det["class"], DEFAULT_COLOR)
        pts   = det["quad"].astype(np.int32).reshape((-1, 1, 2))

        cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=2)

        # Label: class + conf + teks OCR
        tx, ty    = int(det["quad"][0][0]), int(det["quad"][0][1]) - 6
        label_top = f"{det['class']} {det['conf']:.2f}"
        label_ocr = f"\"{det['text']}\""

        cv2.putText(annotated, label_top, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        cv2.putText(annotated, label_ocr, (tx, ty - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    return annotated


def save_visual(image, detections, image_path, save_dir=None):
    annotated  = draw_results(image, detections)
    stem       = Path(image_path).stem
    out_dir    = save_dir or os.path.join(BASE_DIR, "runs", "inference")
    os.makedirs(out_dir, exist_ok=True)
    out_path   = os.path.join(out_dir, f"{stem}_result.jpg")
    cv2.imwrite(out_path, annotated)
    return out_path


# ---------------------------------------------------------------------------
# Format output
# ---------------------------------------------------------------------------

def format_result(output):
    """Buat dict terstruktur dari output pipeline."""
    tgl_raw   = output.get("tanggal_waktu", [""])[0]
    total_raw = output.get("total_belanja", [""])[0]
    classified = output.get("line_item_classified", [])

    return {
        "nama_toko"          : output.get("nama_toko", [""])[0],
        "tanggal_waktu"      : tgl_raw,
        "tanggal_parsed"     : parse_datetime(tgl_raw),
        "total_belanja"      : total_raw,
        "total_parsed"       : parse_amount(total_raw),
        "line_item"          : output.get("line_item", []),
        "line_item_classified": classified,
        "kategori_summary"   : summarize_categories(classified),
    }


def print_result(result, detections, elapsed):
    print()
    print("=" * 60)
    print("HASIL EKSTRAKSI STRUK")
    print("=" * 60)
    print(f"  Nama toko     : {result['nama_toko']}")

    tgl = result['tanggal_waktu']
    tgl_parsed = result.get('tanggal_parsed')
    print(f"  Tanggal/Waktu : {tgl}" + (f"  → {tgl_parsed}" if tgl_parsed else ""))

    total_raw    = result['total_belanja']
    total_parsed = result.get('total_parsed')
    if total_parsed is not None:
        total_fmt = f"Rp {total_parsed:,.0f}".replace(",", ".")
        print(f"  Total belanja : {total_raw}  → {total_fmt}")
    else:
        print(f"  Total belanja : {total_raw}")

    classified = result.get("line_item_classified", [])
    print(f"\n  Item ({len(result['line_item'])}):")
    if classified:
        for it in classified:
            print(f"    • {it['text']:<40} [{it['category']}  {it['confidence']*100:.0f}%]")
    else:
        for item in result["line_item"]:
            print(f"    • {item}")

    summary = result.get("kategori_summary", {})
    if summary:
        print(f"\n  Ringkasan kategori:")
        for cat, count in summary.items():
            print(f"    {cat:<30} {count} item")

    print()
    print(f"  Deteksi       : {len(detections)} region")
    print(f"  Waktu proses  : {elapsed*1000:.0f} ms")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OCR struk belanja end-to-end — NotePay")
    parser.add_argument("--image", required=True,
                        help="Path foto struk atau folder berisi foto")
    parser.add_argument("--save-visual", action="store_true",
                        help="Simpan gambar hasil anotasi YOLO + teks OCR")
    parser.add_argument("--output", default=None,
                        help="Simpan hasil JSON ke file (opsional)")
    parser.add_argument("--save-crops", action="store_true",
                        help="Simpan gambar crop tiap region ke runs/inference/crops/ untuk debug")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold YOLO (default: 0.25)")
    parser.add_argument("--yolo-model", default=None,
                        help=f"Path model YOLO .pt (default: {YOLO_BEST_PT})")
    parser.add_argument("--crnn-model", default=None,
                        help=f"Path model CRNN .keras (default: {CRNN_INFER_PATH})")
    parser.add_argument("--classifier-model", default=None,
                        help=f"Path model classifier .keras (default: {CLASSIFIER_MODEL_PATH})")
    parser.add_argument("--no-classify", action="store_true",
                        help="Skip klasifikasi kategori (debug OCR saja)")
    args = parser.parse_args()

    print("Memuat model...")
    yolo, crnn, classifier, torch_device = load_models(
        args.yolo_model, args.crnn_model, args.classifier_model
    )
    if args.no_classify:
        classifier = None
        print("  [!] Klasifikasi dinonaktifkan (--no-classify)")
    print("Model siap.\n")

    path = Path(args.image)
    if path.is_dir():
        exts   = {".jpg", ".jpeg", ".png", ".webp"}
        images = sorted([p for p in path.iterdir() if p.suffix.lower() in exts])
        if not images:
            print(f"Tidak ada gambar di folder: {args.image}")
            sys.exit(1)
        print(f"Memproses {len(images)} gambar dari {args.image}\n")
    else:
        images = [path]

    all_results = {}

    for img_path in images:
        print(f"[{img_path.name}]")
        image = cv2.imread(str(img_path))

        crops_dir = os.path.join(BASE_DIR, "runs", "inference", "crops", img_path.stem) if args.save_crops else None
        output, detections, elapsed = run(
            img_path, yolo, crnn,
            classifier=classifier,
            conf_threshold=args.conf,
            save_crops_dir=crops_dir,
            torch_device=torch_device,
        )
        result = format_result(output)

        print_result(result, detections, elapsed)
        all_results[str(img_path)] = result

        if args.save_visual and image is not None:
            out_path = save_visual(image, detections, img_path)
            print(f"  Visual tersimpan: {out_path}\n")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"JSON tersimpan: {args.output}")


if __name__ == "__main__":
    main()
