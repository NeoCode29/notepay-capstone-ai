"""
Upload model YOLO ke Hugging Face Hub.

Jalankan di Windows dengan venv-yolo aktif:
    venv-yolo\\Scripts\\activate
    pip install huggingface_hub
    huggingface-cli login
    python ai/upload_to_hub.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai.config import YOLO_BEST_PT

HF_USERNAME = "NeoCode77"


def upload():
    try:
        from huggingface_hub import HfApi, whoami
    except ImportError:
        print("Install dulu: pip install huggingface_hub")
        sys.exit(1)

    username = HF_USERNAME or whoami()["name"]
    repo_id = f"{username}/notepay-yolo-receipt"

    if not os.path.exists(YOLO_BEST_PT):
        print(f"Model tidak ditemukan: {YOLO_BEST_PT}")
        print("Pastikan training selesai dan best.pt ada di folder models/yolo/")
        sys.exit(1)

    print(f"Username HF  : {username}")
    print(f"Repo target  : {repo_id}")
    print(f"Mengupload   : {YOLO_BEST_PT}")

    api = HfApi()

    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        exist_ok=True,
        private=False,
    )

    api.upload_file(
        path_or_fileobj=YOLO_BEST_PT,
        path_in_repo="best.pt",
        repo_id=repo_id,
        repo_type="model",
        commit_message="upload: YOLOv8s-OBB trained on receipt dataset",
    )

    readme = f"""---
license: mit
tags:
  - object-detection
  - yolov8
  - obb
  - receipt
  - ocr
---

# NotePay — YOLOv8s OBB Receipt Region Detector

Model YOLOv8s dengan Oriented Bounding Box (OBB) untuk mendeteksi 4 region pada struk belanja.

## Classes

| ID | Label |
|---|---|
| 0 | nama_toko |
| 1 | line_item |
| 2 | tanggal_waktu |
| 3 | total_belanja |

## Usage

```python
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(repo_id="{repo_id}", filename="best.pt")
model = YOLO(model_path)

results = model("struk.jpg", conf=0.3)
results[0].show()
```

## Training

- **Base model:** YOLOv8s-OBB
- **Dataset:** Receipt images dengan anotasi OBB
- **Input size:** 640×640
- **Optimizer:** AdamW, lr=0.001
"""

    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="docs: tambah model card",
    )

    print(f"\nBerhasil! Model tersedia di:")
    print(f"https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    upload()
