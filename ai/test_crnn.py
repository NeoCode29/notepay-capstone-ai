"""
Testing model CRNN hasil training.

Ada dua mode:
  1. Metrik formal  — hitung CER pada val/test split dataset
  2. Visual         — prediksi teks dari satu gambar crop atau folder

Jalankan di WSL2 dengan venv-tf aktif:
    source ~/venv-tf/bin/activate

    # Mode 1: CER pada val set (default)
    python ai/test_crnn.py --mode metrics

    # Mode 1: CER pada test set
    python ai/test_crnn.py --mode metrics --split test

    # Mode 2: prediksi satu gambar
    python ai/test_crnn.py --mode visual --image data/ocr_dataset/line_item/crop_00001.jpg

    # Mode 2: prediksi semua gambar di folder
    python ai/test_crnn.py --mode visual --image data/ocr_dataset/line_item/

    # Keduanya sekaligus
    python ai/test_crnn.py --mode all --image data/ocr_dataset/line_item/crop_00001.jpg

    # Pakai dataset augment (train/val/test terpisah)
    python ai/test_crnn.py --mode metrics --dataset data/ocr_dataset_augment
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai.config import (
    CRNN_INFER_PATH, OCR_DATASET_DIR,
    CROP_HEIGHT, CROP_WIDTH,
    CHAR_TO_IDX, IDX_TO_CHAR,
    MAX_LABEL_LEN, VAL_SPLIT,
)


# ---------------------------------------------------------------------------
# Preprocessing & Decode
# ---------------------------------------------------------------------------

def preprocess_image_cv(img_path):
    """Load gambar → grayscale 32×CROP_WIDTH, normalisasi [0,1]."""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (CROP_WIDTH, CROP_HEIGHT), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img[np.newaxis, :, :, np.newaxis]   # (1, H, W, 1)


def ctc_greedy_decode(logits_batch):
    """Greedy CTC decode: argmax → hapus blank dan consecutive duplicate."""
    texts = []
    for logits in logits_batch:
        indices = np.argmax(logits, axis=-1)
        prev, chars = -1, []
        for idx in indices:
            if idx != prev:
                if idx != 0:
                    chars.append(IDX_TO_CHAR.get(int(idx), "?"))
                prev = idx
        texts.append("".join(chars))
    return texts


def predict(model, img_path):
    """Prediksi teks dari satu gambar. Return string hasil atau None jika gagal."""
    img = preprocess_image_cv(img_path)
    if img is None:
        return None
    logits = model(img, training=False).numpy()
    return ctc_greedy_decode(logits)[0]


# ---------------------------------------------------------------------------
# Helpers dataset
# ---------------------------------------------------------------------------

def _edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if s1[i-1] == s2[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]


def _load_split(dataset_dir, split):
    """Kembalikan (paths, labels) untuk split tertentu."""
    import pandas as pd

    # Coba struktur train/val/test
    split_csv = os.path.join(dataset_dir, split, "labels.csv")
    if os.path.exists(split_csv):
        df    = pd.read_csv(split_csv, encoding="utf-8")
        base  = os.path.join(dataset_dir, split)
        paths  = [os.path.join(base, fp) for fp in df["filepath"].tolist()]
        labels = df["label"].fillna("").tolist()
        return paths, labels

    # Fallback: flat labels.csv + split otomatis
    flat_csv = os.path.join(dataset_dir, "labels.csv")
    if not os.path.exists(flat_csv):
        raise FileNotFoundError(f"labels.csv tidak ditemukan di: {dataset_dir}")

    df    = pd.read_csv(flat_csv, encoding="utf-8")
    paths  = [os.path.join(dataset_dir, fp) for fp in df["filepath"].tolist()]
    labels = df["label"].fillna("").tolist()

    n     = len(paths)
    n_val = int(n * VAL_SPLIT)

    if split == "val":
        return paths[-n_val:], labels[-n_val:]
    elif split == "test":
        print("  [INFO] Struktur flat tidak punya test split — memakai val split.")
        return paths[-n_val:], labels[-n_val:]
    else:  # train
        return paths[:-n_val], labels[:-n_val]


# ---------------------------------------------------------------------------
# Mode 1: Metrik CER
# ---------------------------------------------------------------------------

def run_metrics(model, dataset_dir, split="val"):
    print(f"Dataset : {dataset_dir}")
    print(f"Split   : {split}")
    print(f"Model   : {CRNN_INFER_PATH}")
    print()

    paths, true_labels = _load_split(dataset_dir, split)
    print(f"Jumlah sampel: {len(paths)}")
    print("Menghitung CER...\n")

    all_preds   = []
    total_chars  = 0
    total_errors = 0
    skipped      = 0

    for i, (img_path, gt) in enumerate(zip(paths, true_labels)):
        pred = predict(model, img_path)
        if pred is None:
            skipped += 1
            continue

        all_preds.append(pred)
        total_chars  += max(len(gt), 1)
        total_errors += _edit_distance(pred, gt)

        if (i + 1) % 200 == 0:
            running_cer = total_errors / total_chars
            print(f"  {i+1}/{len(paths)} — CER sementara: {running_cer*100:.2f}%")

    cer = total_errors / total_chars if total_chars > 0 else 1.0
    accuracy = sum(p == g for p, g in zip(all_preds, true_labels)) / max(len(all_preds), 1)

    print()
    print("=" * 55)
    print("HASIL EVALUASI — CRNN")
    print("=" * 55)
    print(f"  Sampel diproses : {len(all_preds)}")
    print(f"  Dilewati        : {skipped}")
    print(f"  CER             : {cer:.4f}  ({cer*100:.2f}%)")
    print(f"  Exact match     : {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print()
    _print_verdict(cer)

    # Tampilkan contoh prediksi salah terbesar
    errors = [
        (gt, pred, _edit_distance(pred, gt))
        for gt, pred in zip(true_labels[:len(all_preds)], all_preds)
        if pred != gt
    ]
    errors.sort(key=lambda x: -x[2])
    print("\nContoh prediksi terburuk (edit distance terbesar):")
    for gt, pred, dist in errors[:5]:
        print(f"  GT  : '{gt}'")
        print(f"  Pred: '{pred}'  (dist={dist})")
        print()

    return cer


def _print_verdict(cer):
    print("=" * 55)
    if cer <= 0.10:
        print("STATUS: SANGAT BAIK (CER ≤ 10%)")
    elif cer <= 0.15:
        print("STATUS: BAIK — memenuhi target capstone (CER ≤ 15%)")
    elif cer <= 0.30:
        print("STATUS: CUKUP — perlu lebih banyak data atau training lebih lama")
    else:
        print("STATUS: KURANG — pertimbangkan augmentasi atau data lebih banyak")
    print("=" * 55)


# ---------------------------------------------------------------------------
# Mode 2: Visual
# ---------------------------------------------------------------------------

def run_visual(model, image_path):
    path = Path(image_path)

    if path.is_dir():
        exts   = {".jpg", ".jpeg", ".png", ".bmp"}
        images = sorted([p for p in path.iterdir() if p.suffix.lower() in exts])
        if not images:
            print(f"Tidak ada gambar di folder: {image_path}")
            return
        print(f"Memproses {len(images)} gambar dari {image_path}\n")
        for img_path in images:
            _predict_and_print(model, img_path)
    else:
        if not path.exists():
            print(f"ERROR: Gambar tidak ditemukan: {image_path}")
            return
        _predict_and_print(model, path)


def _predict_and_print(model, img_path):
    pred = predict(model, img_path)
    if pred is None:
        print(f"  [SKIP] Tidak bisa membaca: {img_path}")
        return
    print(f"  {Path(img_path).name:<30}  →  '{pred}'")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Testing model CRNN NotePay")
    parser.add_argument(
        "--mode", choices=["metrics", "visual", "all"], default="metrics",
        help="metrics = hitung CER | visual = prediksi gambar | all = keduanya"
    )
    parser.add_argument(
        "--split", choices=["train", "val", "test"], default="val",
        help="Split yang dievaluasi untuk mode metrics (default: val)"
    )
    parser.add_argument(
        "--dataset", default=OCR_DATASET_DIR,
        help=f"Folder dataset (default: {OCR_DATASET_DIR})"
    )
    parser.add_argument(
        "--image", default=None,
        help="Path gambar atau folder untuk mode visual/all"
    )
    parser.add_argument(
        "--model", default=CRNN_INFER_PATH,
        help=f"Path model .keras (default: {CRNN_INFER_PATH})"
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: Model tidak ditemukan: {args.model}")
        print("Jalankan fase4_train_crnn.py terlebih dahulu.")
        sys.exit(1)

    import tensorflow as tf
    import keras

    print(f"Memuat model: {args.model}")
    model = keras.models.load_model(
        args.model,
        compile=False,
    )
    print("Model berhasil dimuat.\n")

    if args.mode in ("metrics", "all"):
        run_metrics(model, args.dataset, split=args.split)

    if args.mode in ("visual", "all"):
        if not args.image:
            print("\nERROR: --image diperlukan untuk mode visual/all")
            sys.exit(1)
        print()
        run_visual(model, args.image)


if __name__ == "__main__":
    main()
