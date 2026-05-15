"""
Inference end-to-end: foto struk → teks terstruktur.

Memenuhi syarat capstone "simple inference code".

Contoh penggunaan:
    python ai/inference.py --image foto_struk.jpg
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai.config import (
    YOLO_BEST_PT, CRNN_INFER_PATH,
    CROP_HEIGHT, CROP_WIDTH,
    IDX_TO_CHAR, YOLO_CLASSES,
)


def load_models():
    from ultralytics import YOLO
    import keras

    yolo  = YOLO(YOLO_BEST_PT)
    crnn  = keras.models.load_model(CRNN_INFER_PATH)
    return yolo, crnn


def deskew_crop(image, quad, out_h=CROP_HEIGHT, out_w=CROP_WIDTH):
    src = quad.astype(np.float32)
    dst = np.array([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]], dtype=np.float32)
    M   = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (out_w, out_h))


def preprocess_crop(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    gray = cv2.resize(gray, (CROP_WIDTH, CROP_HEIGHT))
    return gray.astype(np.float32) / 255.0


def ctc_decode(logits):
    """Greedy CTC decode pada satu sequence logits (time, classes)."""
    indices = np.argmax(logits, axis=-1)
    prev, chars = -1, []
    for idx in indices:
        if idx != prev:
            if idx != 0:
                chars.append(IDX_TO_CHAR.get(int(idx), ""))
            prev = idx
    return "".join(chars)


def run(image_path, yolo, crnn):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")

    # Stage 1: deteksi region
    results = yolo(image, verbose=False)
    obb     = results[0].obb
    if obb is None or len(obb) == 0:
        print("Tidak ada region struk terdeteksi.")
        return {}

    quads     = obb.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2)
    class_ids = obb.cls.cpu().numpy().astype(int)

    # Stage 2: text recognition per region
    output = {}
    for quad, cls_id in zip(quads, class_ids):
        class_name = YOLO_CLASSES[cls_id] if cls_id < len(YOLO_CLASSES) else f"class_{cls_id}"
        crop       = deskew_crop(image, quad)
        tensor     = preprocess_crop(crop)[np.newaxis, :, :, np.newaxis]  # (1, H, W, 1)
        logits     = crnn(tensor, training=False).numpy()[0]               # (time, classes)
        text       = ctc_decode(logits)

        if class_name not in output:
            output[class_name] = []
        output[class_name].append(text)

    return output


def main():
    parser = argparse.ArgumentParser(description="OCR struk belanja end-to-end")
    parser.add_argument("--image", required=True, help="Path foto struk")
    args = parser.parse_args()

    print("Memuat model...")
    yolo, crnn = load_models()

    print(f"Memproses: {args.image}")
    result = run(args.image, yolo, crnn)

    print("\n=== Hasil Ekstraksi ===")
    for region, texts in result.items():
        for t in texts:
            print(f"  [{region}]  {t}")


if __name__ == "__main__":
    main()
