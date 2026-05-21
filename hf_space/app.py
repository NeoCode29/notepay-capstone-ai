"""
Gradio Space — NotePay Receipt Region Detector
"""

# Patch bug gradio_client: get_type() crash saat schema berupa boolean
import gradio_client.utils as _gcu
_orig = _gcu._json_schema_to_python_type

def _safe_json_schema_to_python_type(schema, defs=None):
    if not isinstance(schema, dict):
        return "any"
    return _orig(schema, defs)

_gcu._json_schema_to_python_type = _safe_json_schema_to_python_type

import numpy as np
import gradio as gr
from PIL import Image
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

HF_REPO_ID = "NeoCode77/notepay-yolo-receipt"
LABELS = ["nama_toko", "line_item", "tanggal_waktu", "total_belanja"]

model_path = hf_hub_download(repo_id=HF_REPO_ID, filename="best.pt")
model = YOLO(model_path)


def predict(image: np.ndarray, confidence: float):
    if image is None:
        return None, "Tidak ada gambar."

    results = model(image, conf=confidence)[0]
    annotated = results.plot(line_width=2)
    output_img = Image.fromarray(annotated)

    if results.obb is None or len(results.obb) == 0:
        return output_img, "Tidak ada region yang terdeteksi. Coba turunkan confidence threshold."

    seen = {}
    for box in results.obb:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = LABELS[cls_id] if cls_id < len(LABELS) else f"class_{cls_id}"
        seen[label] = max(seen.get(label, 0), conf)

    icons = {
        "nama_toko":     "Nama Toko",
        "line_item":     "Item Belanja",
        "tanggal_waktu": "Tanggal/Waktu",
        "total_belanja": "Total Belanja",
    }
    lines = [f"{icons.get(k, k)}: {v:.1%}" for k, v in sorted(seen.items(), key=lambda x: -x[1])]

    return output_img, "\n".join(lines)


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="numpy", label="Upload Foto Struk"),
        gr.Slider(minimum=0.1, maximum=0.9, value=0.3, step=0.05, label="Confidence Threshold"),
    ],
    outputs=[
        gr.Image(label="Hasil Deteksi"),
        gr.Textbox(label="Region Terdeteksi", lines=6),
    ],
    title="NotePay — Deteksi Region Struk Belanja",
    description="Upload foto struk belanja. Model akan mendeteksi 4 region: **nama toko**, **item belanja**, **tanggal/waktu**, dan **total belanja**.",
    allow_flagging="never",
)

demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
