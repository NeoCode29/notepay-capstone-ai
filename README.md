# NotePay — AI OCR Pipeline

Modul AI untuk ekstraksi teks dari foto struk belanja secara otomatis.
Pipeline terdiri dari dua model: **YOLOv8-OBB** untuk deteksi region dan **CRNN + CTC** untuk text recognition.

---

## Alur Pipeline

```
Foto Struk
    │
    ▼
[Fase 2] YOLOv8-OBB          → deteksi region: nama_toko, line_item,
    │     (PyTorch)              tanggal_waktu, total_belanja
    ▼
[Fase 3] Crop + Auto-Label   → potong & luruskan tiap region (OpenCV)
    │     (EasyOCR)             auto-label teks → labels.csv
    ▼
     verifikasi manual
    │
    ▼
[Fase 4] CRNN + CTC          → training text recognition
    │     (TensorFlow)          Custom Layer + Custom Callback
    ▼
[Inference] End-to-End       → foto struk → JSON teks terstruktur
```

---

## Struktur File

```
ai/
├── config.py                ← semua konstanta & path (ubah di sini saja)
├── download_dataset.py      ← download dataset dari Roboflow
├── fase2_train_yolo.py      ← training YOLOv8-OBB
├── fase3_prepare_dataset.py ← ekstrak crop + auto-label → labels.csv
├── fase4_train_crnn.py      ← training CRNN TensorFlow
└── inference.py             ← test pipeline end-to-end
```

---

## Prasyarat

Environment GPU harus sudah di-setup sesuai `../SETUP-ENV.md`.

Cek cepat sebelum mulai:

```powershell
# Windows — venv-yolo (untuk Fase 2 & 3)
venv-yolo\Scripts\activate
python -c "import torch; print('GPU:', torch.cuda.is_available())"
# Output yang diharapkan: GPU: True
```

```powershell
# WSL2 — venv-tf (untuk Fase 4)
wsl -d Ubuntu-22.04 -- bash -c "source ~/venv-tf/bin/activate && python -c \"import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))\""
# Output yang diharapkan: [PhysicalDevice(name='/physical_device:GPU:0', ...)]
```

---

## Fase 2 — Training YOLOv8 OBB

**Environment:** Windows, `venv-yolo`

### Langkah 1 — Konfigurasi Roboflow

Salin `.env.example` menjadi `.env` di root project, lalu isi nilainya:

```
ROBOFLOW_API_KEY=xxxxxxxxxxxxxxxx
ROBOFLOW_WORKSPACE=nama-workspace
ROBOFLOW_PROJECT=nama-project
ROBOFLOW_VERSION=1
```

Nilai workspace, project, dan version bisa dilihat di URL Roboflow:
`app.roboflow.com/<workspace>/<project>/<version>`

### Langkah 2 — Download dataset

```powershell
venv-yolo\Scripts\activate
python ai/download_dataset.py
```

Output yang diharapkan:

```
Menghubungkan ke Roboflow...
  Workspace : nama-workspace
  Project   : nama-project
  Version   : 1
  Output    : data/yolo_dataset

Mengunduh dataset (format: yolov8obb)...

Selesai. Dataset tersimpan di: data/yolo_dataset
  train : 312 gambar
  valid :  87 gambar
  test  :  44 gambar
```

Struktur folder hasil download:

```
data/yolo_dataset/
├── data.yaml
├── train/  images/ + labels/
├── valid/  images/ + labels/
└── test/   images/ + labels/
```

### Langkah 3 — Jalankan training

```powershell
python ai/fase2_train_yolo.py
```

Training berjalan ~1–3 jam. Progress tampil langsung di terminal.

### Langkah 4 — Pantau TensorBoard (opsional)

```powershell
# Buka terminal baru
venv-yolo\Scripts\activate
tensorboard --logdir runs/obb/train
# Buka browser: http://localhost:6006
```

### Output

```
models/yolo/best.pt     ← dipakai di Fase 3
runs/obb/train/
├── weights/best.pt
├── results.png
└── confusion_matrix.png
```

### Target Performa

| Metrik | Target |
|---|---|
| mAP50 | ≥ 0.85 |
| mAP50-95 | ≥ 0.60 |

Jika mAP50 masih rendah:
- Tambah data anotasi untuk kelas yang skornya rendah
- Ganti `YOLO_BASE_MODEL = "yolov8s-obb.pt"` di `config.py`
- Naikkan `YOLO_EPOCHS = 150` di `config.py`

---

## Fase 3 — Persiapan Dataset OCR

**Environment:** Windows, `venv-yolo`

### Langkah 1 — Install EasyOCR (sekali saja)

```powershell
venv-yolo\Scripts\activate
pip install easyocr
```

### Langkah 2 — Siapkan foto struk mentah

Kumpulkan semua foto struk (JPG/PNG) ke folder:

```
data/raw_receipts/
├── struk_001.jpg
├── struk_002.jpg
└── ...
```

### Langkah 3 — Ekstrak crop dan auto-label

```powershell
# Ekstrak semua kelas
python ai/fase3_prepare_dataset.py --raw-dir data/raw_receipts

# Atau filter kelas tertentu saja
python ai/fase3_prepare_dataset.py --raw-dir data/raw_receipts --classes line_item total_belanja
```

Script melakukan:
1. Inferensi YOLO pada setiap foto
2. Perspective transform (deskew) setiap region OBB
3. Auto-label teks dengan EasyOCR
4. Simpan ke `data/ocr_dataset/`

### Langkah 4 — Verifikasi labels.csv

> Langkah ini wajib dilakukan sebelum Fase 4.

Buka `data/ocr_dataset/labels.csv` di Excel atau VS Code:

```
filename,        label,             class,         verified
crop_00001.jpg,  INDOMARET,         nama_toko,     0
crop_00002.jpg,  Aqua 600ml,        line_item,     0
crop_00003.jpg,  Rp 4.000,          line_item,     0
crop_00004.jpg,  Total Rp 47.500,   total_belanja, 0
```

| Kolom `verified` | Artinya |
|---|---|
| `0` | Belum diperiksa (default) |
| `1` | Label benar, siap training |
| `-1` | Label salah / dilewati |

Minimal **500 baris** dengan `verified = 1` sebelum lanjut ke Fase 4.

---

## Fase 4 — Training CRNN TensorFlow

**Environment:** WSL2 Ubuntu 22.04, `venv-tf`

### Langkah 1 — Masuk ke WSL2

```powershell
wsl -d Ubuntu-22.04
```

### Langkah 2 — Aktifkan environment dan navigasi

```bash
source ~/venv-tf/bin/activate
cd "/mnt/c/Users/kresna/Documents/SIB Dicoding/Capstone Project"
```

> **Tips performa:** Jika dataset besar (> 5.000 gambar), copy ke filesystem WSL2
> agar I/O lebih cepat:
> ```bash
> cp -r data/ocr_dataset ~/ocr_dataset
> ```
> Lalu ubah `OCR_DATASET_DIR` di `config.py` ke `/root/ocr_dataset`.

### Langkah 3 — Jalankan training

```bash
python ai/fase4_train_crnn.py
```

Contoh output per epoch:

```
Epoch 5/60
 157/157 ━━━━━━━━━━━━━━━━━━━━ 23s loss: 8.4521 - val_loss: 9.1203

  [Epoch 5] Contoh prediksi:
    [✗] GT: 'Aqua 600ml'   →  Pred: 'Aua 60ml'
    [✓] GT: 'Rp 4.000'     →  Pred: 'Rp 4.000'
    [✗] GT: 'INDOMARET'    →  Pred: 'INDMARET'
    [✓] GT: 'Total 47500'  →  Pred: 'Total 47500'
```

### Langkah 4 — Pantau TensorBoard (opsional)

```bash
# Buka terminal WSL2 baru
source ~/venv-tf/bin/activate
cd "/mnt/c/Users/kresna/Documents/SIB Dicoding/Capstone Project"
tensorboard --logdir logs/crnn --host 0.0.0.0
# Buka di browser Windows: http://localhost:6006
```

### Output

```
models/crnn/
├── crnn_model.keras        ← model training (dengan CTCLayer)
├── inference_model.keras   ← model inference (input gambar → logits)
└── ckpt_best.keras         ← checkpoint terbaik selama training
logs/crnn/                  ← TensorBoard logs
```

### Target Performa

| Metrik | Target Side Quest | Realistis (dataset kecil) |
|---|---|---|
| CER (Character Error Rate) | ≤ 15% | ≤ 25% |
| Akurasi kata | ≥ 85% | ≥ 70% |

Jika CER masih tinggi setelah 60 epoch:
- Tambah data terverifikasi di `labels.csv`
- Kurangi `CRNN_BATCH_SIZE` ke 16 jika VRAM tidak cukup
- Coba `CRNN_LR = 3e-3` di `config.py`

---

## Inference — Test End-to-End

Setelah kedua model tersedia (`best.pt` + `inference_model.keras`):

```bash
# Dari WSL2
source ~/venv-tf/bin/activate
pip install ultralytics   # sekali saja di venv-tf
cd "/mnt/c/Users/kresna/Documents/SIB Dicoding/Capstone Project"

python ai/inference.py --image data/raw_receipts/struk_test.jpg
```

Contoh output:

```
=== Hasil Ekstraksi ===
  [nama_toko]     INDOMARET
  [tanggal_waktu] 12/05/2026 14:30
  [line_item]     Aqua 600ml      Rp 4.000
  [line_item]     Teh Botol Sosro Rp 5.000
  [line_item]     Indomie Goreng  Rp 3.500
  [total_belanja] Total Rp 47.500
```

---

## Konfigurasi (`config.py`)

Semua hyperparameter dipusatkan di `ai/config.py`. Ubah di sini, tidak perlu edit script lain.

| Variabel | Default | Keterangan |
|---|---|---|
| `YOLO_BASE_MODEL` | `yolov8n-obb.pt` | Ganti ke `yolov8s-obb.pt` untuk akurasi lebih tinggi |
| `YOLO_EPOCHS` | `100` | |
| `CROP_HEIGHT / WIDTH` | `32 / 128` | Ukuran input CRNN — jangan diubah setelah training |
| `NUM_CLASSES` | `74` | 73 karakter + 1 blank CTC |
| `CRNN_EPOCHS` | `60` | |
| `CRNN_BATCH_SIZE` | `32` | Kurangi ke `16` jika GPU OOM |
| `CRNN_LR` | `1e-3` | |

---

## Troubleshooting

| Error | Solusi |
|---|---|
| `data.yaml tidak ditemukan` | Jalankan `download_dataset.py` dulu |
| `best.pt tidak ditemukan` | Fase 2 belum selesai — cek `runs/obb/train/weights/` |
| `labels.csv tidak ditemukan` | Jalankan `fase3_prepare_dataset.py` dulu |
| YOLO training < 5 it/s | GPU tidak aktif — cek `torch.cuda.is_available()` di `venv-yolo` |
| TF training OOM | Kurangi `CRNN_BATCH_SIZE` di `config.py` |
| CTC loss = `inf` setelah epoch 5 | Cek label kosong atau karakter di luar `CHARACTERS` |
| WSL2 GPU tidak terbaca | Jalankan ulang `source ~/venv-tf/bin/activate` dan cek `echo $LD_LIBRARY_PATH` |
| `.env` tidak terbaca | Pastikan file bernama `.env` (bukan `.env.txt`) dan ada di root project |
