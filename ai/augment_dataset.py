"""
Augmentasi dataset OCR dengan split train/val/test.

Input : folder raw (struktur seperti ocr_dataset — flat per kelas)
Output: folder baru dengan struktur train/val/test, augmentasi hanya di train.

Jalankan di WSL2 dengan venv-tf aktif:
    python ai/augment_dataset.py
    python ai/augment_dataset.py --src data/ocr_dataset --dst data/ocr_dataset_augment

Output folder:
    ocr_dataset_augment/
    ├── train/
    │   ├── line_item/crop_00001.jpg
    │   ├── line_item/crop_00001_aug1.jpg
    │   ├── line_item/crop_00001_aug2.jpg
    │   └── labels.csv
    ├── val/
    │   ├── line_item/crop_00005.jpg
    │   └── labels.csv
    └── test/
        ├── line_item/crop_00010.jpg
        └── labels.csv
"""

import os
import sys
import csv
import random
import shutil
import argparse
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai.config import OCR_DATASET_DIR

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Augmentasi
# ---------------------------------------------------------------------------

def _random_rotation(img, max_deg=3.0):
    h, w = img.shape[:2]
    angle = random.uniform(-max_deg, max_deg)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def _random_perspective(img, strength=0.015):
    h, w = img.shape[:2]
    d = max(1, int(min(h, w) * strength))
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [random.randint(0, d), random.randint(0, d)],
        [w - random.randint(0, d), random.randint(0, d)],
        [w - random.randint(0, d), h - random.randint(0, d)],
        [random.randint(0, d), h - random.randint(0, d)],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h),
                               borderMode=cv2.BORDER_REPLICATE)


def _random_brightness_contrast(img, b_range=40, c_range=0.3):
    beta  = random.randint(-b_range, b_range)
    alpha = 1.0 + random.uniform(-c_range, c_range)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def _random_blur(img):
    if random.random() < 0.5:
        return cv2.GaussianBlur(img, (3, 3), sigmaX=random.uniform(0.5, 1.2))
    return img


def _add_noise(img, stddev=8):
    noise = np.random.normal(0, stddev, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def augment(img):
    img = _random_rotation(img)
    img = _random_perspective(img)
    img = _random_brightness_contrast(img)
    img = _random_blur(img)
    img = _add_noise(img)
    return img


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_labels_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filepath", "label", "class"],
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        writer.writerows(rows)


def _save_image(img, dst_dir, rel_path):
    full_path = os.path.join(dst_dir, rel_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    cv2.imwrite(full_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=OCR_DATASET_DIR,
                        help="Folder input (berisi labels.csv + subfolder kelas)")
    parser.add_argument("--dst", default=os.path.join(
                            os.path.dirname(OCR_DATASET_DIR), "ocr_dataset_augment"),
                        help="Folder output")
    args = parser.parse_args()

    src_dir = args.src
    dst_dir = args.dst
    src_csv = os.path.join(src_dir, "labels.csv")

    if not os.path.exists(src_csv):
        print(f"ERROR: labels.csv tidak ditemukan di {src_dir}")
        sys.exit(1)

    if os.path.exists(dst_dir):
        print(f"Folder {dst_dir} sudah ada. Hapus dulu? (y/n) ", end="")
        if input().strip().lower() != "y":
            print("Dibatalkan.")
            sys.exit(0)
        shutil.rmtree(dst_dir)

    # Baca semua baris dari labels.csv
    with open(src_csv, encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    # Filter baris yang gambarnya benar-benar ada
    valid_rows = []
    for row in all_rows:
        img_path = os.path.join(src_dir, row["filepath"])
        if os.path.exists(img_path):
            valid_rows.append(row)

    print(f"Total valid  : {len(valid_rows)} sampel")
    print(f"Dilewati     : {len(all_rows) - len(valid_rows)} sampel (gambar tidak ditemukan)")

    # Shuffle deterministik lalu split
    random.seed(RANDOM_SEED)
    random.shuffle(valid_rows)

    n        = len(valid_rows)
    n_train  = int(n * TRAIN_RATIO)
    n_val    = int(n * VAL_RATIO)

    splits = {
        "train": valid_rows[:n_train],
        "val"  : valid_rows[n_train:n_train + n_val],
        "test" : valid_rows[n_train + n_val:],
    }

    print(f"Split        : train={len(splits['train'])}  "
          f"val={len(splits['val'])}  test={len(splits['test'])}")
    print(f"Output       : {dst_dir}")
    print()

    for split_name, rows in splits.items():
        split_dir  = os.path.join(dst_dir, split_name)
        split_rows = []

        for i, row in enumerate(rows):
            src_img_path = os.path.join(src_dir, row["filepath"])
            img = cv2.imread(src_img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Salin gambar asli
            rel_path = row["filepath"].replace("\\", "/")
            _save_image(img, split_dir, rel_path)
            split_rows.append({
                "filepath": rel_path,
                "label"   : row["label"],
                "class"   : row["class"],
            })

            # Augmentasi hanya untuk train
            if split_name == "train":
                dirname  = os.path.dirname(rel_path)
                basename = os.path.splitext(os.path.basename(rel_path))[0]
                for aug_idx in (1, 2):
                    aug_img  = augment(img)
                    aug_rel  = f"{dirname}/{basename}_aug{aug_idx}.jpg"
                    _save_image(aug_img, split_dir, aug_rel)
                    split_rows.append({
                        "filepath": aug_rel,
                        "label"   : row["label"],
                        "class"   : row["class"],
                    })

            if (i + 1) % 300 == 0:
                print(f"  [{split_name}] {i+1}/{len(rows)} diproses...")

        csv_path = os.path.join(split_dir, "labels.csv")
        _write_labels_csv(csv_path, split_rows)

        aug_info = f" (termasuk {len(split_rows) - len(rows)} augmentasi)" if split_name == "train" else ""
        print(f"  [{split_name}] {len(split_rows)} sampel{aug_info} → {csv_path}")

    train_total = len(splits["train"]) * 3
    total_all   = train_total + len(splits["val"]) + len(splits["test"])
    print(f"\nTotal dataset augment: {total_all} sampel")
    print("Selesai.")


if __name__ == "__main__":
    main()
