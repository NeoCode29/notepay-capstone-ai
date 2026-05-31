"""
Upload semua model NotePay ke Hugging Face Hub (satu repo).

Struktur repo HF: NeoCode77/notepay-models
    yolo/best.pt
    crnn/inference_model.keras
    classifier/classifier_model.keras

Jalankan di Windows (venv-yolo) setelah semua model selesai di-training:
    venv-yolo\\Scripts\\activate
    pip install huggingface_hub
    huggingface-cli login     ← butuh token WRITE dari hf.co/settings/tokens
    python ai/upload_to_hub.py

Untuk upload model tertentu saja:
    python ai/upload_to_hub.py --only yolo
    python ai/upload_to_hub.py --only crnn
    python ai/upload_to_hub.py --only classifier
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai.config import (
    YOLO_BEST_PT, CRNN_INFER_PATH, CLASSIFIER_MODEL_PATH, EXPENSE_CATEGORIES,
)

REPO_ID = "NeoCode77/notepay-models"

MODEL_FILES = {
    "yolo": {
        "local_path" : YOLO_BEST_PT,
        "repo_path"  : "yolo/best.pt",
        "commit_msg" : "upload: YOLOv8n-OBB receipt region detector",
    },
    "crnn": {
        "local_path" : CRNN_INFER_PATH,
        "repo_path"  : "crnn/inference_model.keras",
        "commit_msg" : "upload: CRNN+CTC text recognition model",
    },
    "classifier": {
        "local_path" : CLASSIFIER_MODEL_PATH,
        "repo_path"  : "classifier/classifier_model.keras",
        "commit_msg" : "upload: expense category classifier",
    },
}


def _make_readme() -> str:
    cat_rows = "\n".join(
        f"| {i} | {cat} |" for i, cat in enumerate(EXPENSE_CATEGORIES)
    )
    return f"""---
license: mit
tags:
  - ocr
  - receipt
  - object-detection
  - yolov8
  - tensorflow
  - keras
  - text-recognition
  - expense-classification
language:
  - id
---

# NotePay — Receipt OCR Models

Model AI untuk pipeline OCR struk belanja otomatis.
Bagian dari project **NotePay** (Coding Camp 2026 — DBS Foundation).

## Pipeline

```
Foto Struk
  → [1] YOLOv8n-OBB    : deteksi 4 region (nama_toko, line_item, tanggal_waktu, total_belanja)
  → [2] CRNN + CTC      : text recognition per crop (TensorFlow/Keras)
  → [3] Classifier       : klasifikasi kategori pengeluaran tiap line item
  → JSON terstruktur
```

## Model Files

| File | Deskripsi |
|---|---|
| `yolo/best.pt` | YOLOv8n-OBB — deteksi region struk |
| `crnn/inference_model.keras` | CRNN+CTC — baca teks dari crop |
| `classifier/classifier_model.keras` | Text classifier — kategori pengeluaran |

## Expense Categories (Classifier)

| ID | Kategori |
|---|---|
{cat_rows}

## Usage

```python
from huggingface_hub import hf_hub_download

# Download semua model
yolo_path       = hf_hub_download("{REPO_ID}", "yolo/best.pt")
crnn_path       = hf_hub_download("{REPO_ID}", "crnn/inference_model.keras")
classifier_path = hf_hub_download("{REPO_ID}", "classifier/classifier_model.keras")

# Load
from ultralytics import YOLO
import keras

yolo       = YOLO(yolo_path)
crnn       = keras.models.load_model(crnn_path, compile=False, safe_mode=False)
classifier = keras.models.load_model(classifier_path, compile=False)
```

Atau gunakan `ai/model_loader.py` dari repo ini yang sudah handle caching & GPU setup.
"""


def upload(only: str | None = None):
    try:
        from huggingface_hub import HfApi, whoami
    except ImportError:
        print("Install dulu: pip install huggingface_hub")
        sys.exit(1)

    username = whoami()["name"]
    print(f"Username HF : {username}")
    print(f"Repo target : {REPO_ID}\n")

    api = HfApi()
    api.create_repo(
        repo_id=REPO_ID,
        repo_type="model",
        exist_ok=True,
        private=False,
    )

    # Upload model files
    targets = {only: MODEL_FILES[only]} if only else MODEL_FILES
    all_ok  = True

    for name, spec in targets.items():
        local = spec["local_path"]
        if not os.path.exists(local):
            print(f"  [SKIP] {name}: file tidak ditemukan → {local}")
            print(f"         Jalankan training fase yang sesuai terlebih dahulu.\n")
            all_ok = False
            continue

        size_mb = os.path.getsize(local) / 1_048_576
        print(f"  Uploading {name} ({size_mb:.1f} MB) → {spec['repo_path']} ...")
        api.upload_file(
            path_or_fileobj=local,
            path_in_repo=spec["repo_path"],
            repo_id=REPO_ID,
            repo_type="model",
            commit_message=spec["commit_msg"],
        )
        print(f"  [OK] {name}\n")

    # Upload README (model card) — selalu diupdate
    if not only:
        print("  Uploading README.md (model card) ...")
        readme_bytes = _make_readme().encode("utf-8")
        api.upload_file(
            path_or_fileobj=readme_bytes,
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="model",
            commit_message="docs: update model card",
        )
        print("  [OK] README.md\n")

    print("=" * 55)
    if all_ok:
        print(f"Semua model berhasil diupload!")
    else:
        print(f"Upload selesai (beberapa model diskip karena file tidak ada).")
    print(f"Repo : https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload model NotePay ke HF Hub")
    parser.add_argument(
        "--only",
        choices=["yolo", "crnn", "classifier"],
        default=None,
        help="Upload hanya satu model tertentu",
    )
    args = parser.parse_args()
    upload(only=args.only)
