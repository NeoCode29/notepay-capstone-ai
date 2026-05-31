# Patch bug gradio_client: schema berupa bool → crash saat generate API
import gradio_client.utils as _gcu
_orig_schema = _gcu._json_schema_to_python_type

def _safe_schema(schema, defs=None):
    if not isinstance(schema, dict):
        return "any"
    return _orig_schema(schema, defs)

_gcu._json_schema_to_python_type = _safe_schema

# ---------------------------------------------------------------------------
import re
import json
import numpy as np
import cv2
import gradio as gr
from PIL import Image
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------------------
# Konstanta (inline dari config.py — Space tidak bisa import modul lokal)
# ---------------------------------------------------------------------------

HF_REPO_ID = "NeoCode77/notepay-models"

YOLO_CLASSES = ["line_item", "nama_toko", "tanggal_waktu", "total_belanja"]

CHARACTERS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,/-()%")
IDX_TO_CHAR = {i + 1: ch for i, ch in enumerate(CHARACTERS)}
IDX_TO_CHAR[0] = ""   # blank CTC

CROP_HEIGHT = 32
CROP_WIDTH  = 512

EXPENSE_CATEGORIES = [
    "Makanan & Minuman",
    "Kebersihan & Perawatan",
    "Rumah Tangga",
    "Kesehatan & Farmasi",
    "Elektronik & Pulsa",
    "Pakaian & Aksesori",
    "Lain-lain",
]

CLASS_COLORS = {
    "nama_toko":     (255, 100,  50),
    "line_item":     ( 50, 200,  50),
    "tanggal_waktu": ( 50, 150, 255),
    "total_belanja": (  0,  50, 255),
}

# ---------------------------------------------------------------------------
# Load model (sekali saat startup)
# ---------------------------------------------------------------------------

print("Mendownload model YOLO...")
_yolo_path = hf_hub_download(repo_id=HF_REPO_ID, filename="yolo/best.pt")

print("Mendownload model CRNN...")
_crnn_path = hf_hub_download(repo_id=HF_REPO_ID, filename="crnn/inference_model.keras")

print("Mendownload model Classifier...")
_clf_path  = hf_hub_download(repo_id=HF_REPO_ID, filename="classifier/classifier_model.keras")

# TF config: memory growth agar tidak OOM di CPU
import tensorflow as tf
import keras

tf.get_logger().setLevel("ERROR")

print("Loading YOLO...")
from ultralytics import YOLO as _YOLO
yolo_model = _YOLO(_yolo_path)

print("Loading CRNN...")
# Patch Keras 3.x: Lambda layer compute_output_shape kadang raise NotImplementedError
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

crnn_model = keras.models.load_model(_crnn_path, compile=False, safe_mode=False)

print("Loading Classifier...")
clf_model = keras.models.load_model(_clf_path, compile=False)

print("Semua model siap!")


# ---------------------------------------------------------------------------
# Fungsi image processing (inline dari inference.py)
# ---------------------------------------------------------------------------

def _order_quad(pts):
    pts  = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)
    s       = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff    = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def deskew_crop(image, quad, out_h=CROP_HEIGHT, out_w=CROP_WIDTH):
    src = _order_quad(quad)
    dst = np.array([[0,0],[out_w,0],[out_w,out_h],[0,out_h]], dtype=np.float32)
    M   = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (out_w, out_h))


def preprocess_crop(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    _, mask  = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    coords   = cv2.findNonZero(mask)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        gray = gray[y:y+h, x:x+w]
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
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
    indices = np.argmax(logits, axis=-1)
    prev, chars = -1, []
    for idx in indices:
        if idx != prev:
            if idx != 0:
                chars.append(IDX_TO_CHAR.get(int(idx), ""))
            prev = idx
    return "".join(chars)


def parse_amount(text):
    if not text:
        return None
    cleaned = re.sub(r"[^\d.,]", "", text)
    if not cleaned:
        return None
    if re.search(r",\d{1,2}$", cleaned):
        cleaned = cleaned.replace(".", "").replace(",", ".")
    else:
        cleaned = cleaned.replace(".", "").replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_datetime(text):
    if not text:
        return None
    _BULAN = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MEI":5,"JUN":6,
              "JUL":7,"AGU":8,"SEP":9,"OKT":10,"NOV":11,"DES":12}
    patterns = [
        (r"(\d{2})[/\-.](\d{2})[/\-.](\d{4})\s+(\d{2}:\d{2})(?::\d{2})?",
         lambda m: f"{m[2]}-{m[1]}-{m[0]} {m[3]}"),
        (r"(\d{4})[/\-.](\d{2})[/\-.](\d{2})\s+(\d{2}:\d{2})",
         lambda m: f"{m[0]}-{m[1]}-{m[2]} {m[3]}"),
        (r"(\d{2})[/\-.](\d{2})[/\-.](\d{4})",
         lambda m: f"{m[2]}-{m[1]}-{m[0]}"),
    ]
    upper = text.upper()
    for pattern, fmt in patterns:
        m = re.search(pattern, upper)
        if m:
            try:
                return fmt(m.groups())
            except Exception:
                continue
    return None


def classify_items(items):
    if not items:
        return []
    arr   = tf.constant([[item] for item in items])
    preds = clf_model(arr, training=False).numpy()
    return [
        {
            "text"      : text,
            "category"  : EXPENSE_CATEGORIES[int(np.argmax(pred))],
            "confidence": round(float(np.max(pred)), 3),
        }
        for text, pred in zip(items, preds)
    ]


def draw_results(image, detections):
    annotated = image.copy()
    for det in detections:
        color = CLASS_COLORS.get(det["class"], (200, 200, 200))
        pts   = det["quad"].astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=2)
        tx, ty = int(det["quad"][0][0]), int(det["quad"][0][1]) - 8
        cv2.putText(annotated, f"{det['class']} {det['conf']:.0%}",
                    (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        if det.get("text"):
            cv2.putText(annotated, f"\"{det['text'][:30]}\"",
                        (tx, ty - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (255, 255, 255), 1, cv2.LINE_AA)
    return annotated


# ---------------------------------------------------------------------------
# Pipeline utama
# ---------------------------------------------------------------------------

def predict(image_pil, confidence):
    if image_pil is None:
        return None, "Tidak ada gambar.", "{}"

    # PIL → BGR numpy (OpenCV format)
    image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    # Stage 1: YOLO
    results   = yolo_model(image, conf=confidence, verbose=False)[0]
    obb       = results.obb

    if obb is None or len(obb) == 0:
        annotated_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return annotated_pil, "⚠️ Tidak ada region terdeteksi.\nCoba turunkan confidence threshold.", "{}"

    quads     = obb.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2)
    class_ids = obb.cls.cpu().numpy().astype(int)
    confs     = obb.conf.cpu().numpy()

    # Stage 2: CRNN text recognition
    output     = {}
    detections = []

    for quad, cls_id, conf in zip(quads, class_ids, confs):
        cls_name = YOLO_CLASSES[cls_id] if cls_id < len(YOLO_CLASSES) else f"class_{cls_id}"
        crop     = deskew_crop(image, quad)
        tensor   = preprocess_crop(crop)
        logits   = crnn_model(tensor, training=False).numpy()[0]
        text     = ctc_decode(logits)

        output.setdefault(cls_name, []).append(text)
        detections.append({"class": cls_name, "conf": float(conf), "text": text, "quad": quad})

    # Stage 3: Classify line items
    raw_items  = output.get("line_item", [])
    classified = classify_items(raw_items)

    # Buat annotated image
    annotated     = draw_results(image, detections)
    annotated_pil = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

    # Structured result
    total_raw  = output.get("total_belanja", [""])[0]
    tgl_raw    = output.get("tanggal_waktu", [""])[0]
    total_num  = parse_amount(total_raw)
    summary    = {}
    for it in classified:
        cat = it["category"]
        summary[cat] = summary.get(cat, 0) + 1

    result = {
        "nama_toko"    : output.get("nama_toko", [""])[0],
        "tanggal_waktu": tgl_raw,
        "tanggal_parsed": parse_datetime(tgl_raw),
        "total_belanja": total_raw,
        "total_parsed" : total_num,
        "line_item"    : raw_items,
        "line_item_classified": classified,
        "kategori_summary": dict(sorted(summary.items(), key=lambda x: -x[1])),
    }

    # Format teks ringkas untuk tab "Hasil"
    total_fmt = f"Rp {total_num:,.0f}".replace(",", ".") if total_num else total_raw
    lines = [
        f"🏪  Nama Toko     : {result['nama_toko'] or '-'}",
        f"📅  Tanggal/Waktu : {result['tanggal_parsed'] or tgl_raw or '-'}",
        f"💰  Total Belanja : {total_fmt or '-'}",
        "",
        f"🛒  Item Belanja ({len(classified)}):",
    ]
    for it in classified:
        bar = "█" * int(it["confidence"] * 10)
        lines.append(f"   • {it['text'][:35]:<36} [{it['category']}  {it['confidence']*100:.0f}%]")

    if summary:
        lines += ["", "📊  Ringkasan Kategori:"]
        for cat, count in summary.items():
            lines.append(f"   {cat:<30} {count} item")

    return annotated_pil, "\n".join(lines), json.dumps(result, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

EXAMPLES = []   # isi dengan path contoh gambar jika ada

with gr.Blocks(title="NotePay — OCR Struk Belanja", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
# 🧾 NotePay — OCR Struk Belanja Otomatis
Pipeline AI lengkap: **Deteksi Region** (YOLOv8-OBB) → **Baca Teks** (CRNN+CTC) → **Klasifikasi Pengeluaran**
""")

    with gr.Row():
        with gr.Column(scale=1):
            inp_image = gr.Image(type="pil", label="Upload Foto Struk")
            inp_conf  = gr.Slider(0.1, 0.9, value=0.25, step=0.05,
                                  label="Confidence Threshold YOLO")
            btn = gr.Button("🔍 Proses Struk", variant="primary")

        with gr.Column(scale=1):
            out_image = gr.Image(type="pil", label="Hasil Deteksi Region")

    with gr.Row():
        with gr.Tabs():
            with gr.Tab("📋 Hasil Ekstraksi"):
                out_text = gr.Textbox(label="", lines=18, show_copy_button=True)
            with gr.Tab("{ } JSON"):
                out_json = gr.Code(language="json", label="Raw JSON", lines=30)

    btn.click(fn=predict, inputs=[inp_image, inp_conf],
              outputs=[out_image, out_text, out_json])
    inp_image.change(fn=predict, inputs=[inp_image, inp_conf],
                     outputs=[out_image, out_text, out_json])

    gr.Markdown("""
---
**Pipeline:** YOLO (deteksi 4 region) → CRNN+CTC (baca teks per crop) → Text Classifier (7 kategori pengeluaran)
**Model:** [`NeoCode77/notepay-models`](https://huggingface.co/NeoCode77/notepay-models)
**Project:** Coding Camp 2026 — DBS Foundation
""")

demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
