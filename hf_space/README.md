---
title: NotePay OCR Struk Belanja
emoji: 🧾
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
license: mit
python_version: "3.11"
---

# NotePay — OCR Struk Belanja Otomatis

Demo interaktif pipeline AI lengkap untuk mengekstrak dan mengklasifikasikan pengeluaran dari foto struk belanja fisik (thermal receipt).

## Pipeline

```
Foto Struk
  → [1] YOLOv8n-OBB    : deteksi 4 region (nama_toko, line_item, tanggal_waktu, total_belanja)
  → [2] CRNN + CTC      : baca teks dari setiap crop (TensorFlow/Keras)
  → [3] Text Classifier : klasifikasi kategori pengeluaran tiap item
  → JSON terstruktur
```

## Model

Semua model dihosting di [`NeoCode77/notepay-models`](https://huggingface.co/NeoCode77/notepay-models):

| File | Deskripsi |
|---|---|
| `yolo/best.pt` | YOLOv8n-OBB — deteksi region |
| `crnn/inference_model.keras` | CRNN+CTC — text recognition |
| `classifier/classifier_model.keras` | Kategori pengeluaran (7 kelas) |

## Kategori Pengeluaran

Makanan & Minuman · Kebersihan & Perawatan · Rumah Tangga · Kesehatan & Farmasi · Elektronik & Pulsa · Pakaian & Aksesori · Lain-lain

## Project

**NotePay** — Catatan Keuangan Otomatis dari Foto Struk
Coding Camp 2026 powered by DBS Foundation — Team CC26-PSU410
