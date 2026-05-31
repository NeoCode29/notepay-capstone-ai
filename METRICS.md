# Metode Pengukuran Akurasi — NotePay AI Pipeline

Dokumen ini menjelaskan metode evaluasi yang digunakan untuk mengukur performa setiap komponen model AI pada project NotePay.

---

## 1. YOLO — Deteksi Region Struk

**File:** `ai/test_yolo.py`
**Metode:** mAP (mean Average Precision)

### Cara Kerja

YOLO mendeteksi 4 region pada foto struk menggunakan Oriented Bounding Box (OBB). Akurasi diukur dengan membandingkan kotak prediksi terhadap kotak ground truth menggunakan **IoU (Intersection over Union)**:

```
IoU = Luas irisan(prediksi ∩ ground truth) / Luas gabungan(prediksi ∪ ground truth)
```

Prediksi dianggap benar (True Positive) jika IoU ≥ threshold yang ditentukan.

### Metrik yang Dihasilkan

| Metrik | Keterangan |
|---|---|
| **mAP@0.5** | Rata-rata AP semua kelas pada IoU threshold 0.5 — metrik utama |
| **mAP@0.5:0.95** | Rata-rata AP pada IoU 0.5–0.95 (step 0.05) — lebih ketat |
| **Precision** | Dari semua region yang dideteksi, berapa yang benar |
| **Recall** | Dari semua region yang seharusnya terdeteksi, berapa yang ditemukan |
| **AP per kelas** | AP individual untuk `nama_toko`, `line_item`, `tanggal_waktu`, `total_belanja` |

### Cara Menjalankan

```powershell
venv-yolo\Scripts\activate
python ai/test_yolo.py --mode metrics
```

### Kriteria Kelulusan

| mAP@0.5 | Status |
|---|---|
| ≥ 90% | Sangat Baik |
| ≥ 85% | Baik — memenuhi target capstone |
| ≥ 70% | Cukup |
| < 70% | Kurang |

---

## 2. CRNN — Text Recognition

**File:** `ai/fase4_train_crnn.py`, `ai/test_crnn.py`
**Metode:** CER (Character Error Rate) berbasis Levenshtein Distance

### Cara Kerja

Setelah CRNN memprediksi teks dari crop gambar, hasilnya dibandingkan dengan label ground truth menggunakan **edit distance** (jarak Levenshtein) — jumlah minimum operasi substitusi, insersi, dan penghapusan karakter yang dibutuhkan untuk mengubah prediksi menjadi ground truth.

```
CER = edit_distance(prediksi, ground_truth) / jumlah_karakter_ground_truth
```

Implementasi menggunakan dynamic programming O(m×n) tanpa library eksternal (lihat `_edit_distance()` di `fase4_train_crnn.py`).

### Contoh Perhitungan

```
Ground truth : "INDOMARET"   (9 karakter)
Prediksi     : "IND0MARET"   (1 substitusi: 'O' → '0')
CER          : 1 / 9 = 11.1%
```

### Metrik yang Dihasilkan

| Metrik | Keterangan |
|---|---|
| **CER** | Character Error Rate — metrik utama OCR |
| **Exact Match Accuracy** | Persentase prediksi yang 100% identik dengan ground truth |
| **Contoh terburuk** | 5 sampel dengan edit distance terbesar, untuk analisis kegagalan |

### Cara Menjalankan

```bash
# Di WSL2 dengan venv-tf aktif
source ~/venv-tf/bin/activate
cd "/mnt/c/Users/kresna/Documents/SIB Dicoding/Capstone Project"

# Evaluasi pada val set
python ai/test_crnn.py --mode metrics

# Evaluasi pada test set
python ai/test_crnn.py --mode metrics --split test

# Prediksi visual satu gambar
python ai/test_crnn.py --mode visual --image data/ocr_dataset/line_item/crop_00001.jpg
```

### Kriteria Kelulusan

| CER | Status |
|---|---|
| ≤ 10% | Sangat Baik |
| ≤ 15% | Baik — memenuhi target capstone |
| ≤ 30% | Cukup |
| > 30% | Kurang |

---

## 3. Klasifikasi Pengeluaran

**File:** `ai/fase5_train_classifier.py`
**Metode:** Classification Report (scikit-learn) + Confusion Matrix

### Cara Kerja

Model menerima teks nama item struk dan memprediksi kategori pengeluaran. Evaluasi menggunakan metrik klasifikasi multi-kelas standar:

```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1-Score  = 2 × (Precision × Recall) / (Precision + Recall)
```

- **Macro average** — rata-rata tiap kelas tanpa memperhatikan jumlah sampel
- **Weighted average** — rata-rata berbobot jumlah sampel tiap kelas (lebih adil jika data tidak seimbang)

### Kategori yang Diklasifikasikan

1. Makanan & Minuman
2. Kebersihan & Perawatan
3. Rumah Tangga
4. Kesehatan & Farmasi
5. Elektronik & Pulsa
6. Pakaian & Aksesori
7. Lain-lain

### Metrik yang Dihasilkan

| Metrik | Keterangan |
|---|---|
| **Accuracy** | Dipantau selama training via Keras `metrics=["accuracy"]` |
| **Precision per kelas** | Ketepatan prediksi untuk setiap kategori |
| **Recall per kelas** | Kelengkapan deteksi untuk setiap kategori |
| **F1-Score per kelas** | Harmonic mean precision dan recall |
| **Weighted F1** | Metrik utama evaluasi akhir |
| **Confusion Matrix** | Tabel kesalahan antar kelas |

### Cara Menjalankan

```bash
# Di WSL2 dengan venv-tf aktif
python ai/fase5_train_classifier.py
```

Classification report dan confusion matrix dicetak otomatis setelah training selesai.

### Kriteria Kelulusan

| Weighted F1 | Status |
|---|---|
| ≥ 90% | Sangat Baik |
| ≥ 80% | Baik — memenuhi target capstone |
| ≥ 70% | Cukup |
| < 70% | Kurang |

---

## Ringkasan

| Komponen | Metode Pengukuran | Metrik Utama | Target Capstone |
|---|---|---|---|
| YOLO (deteksi) | mAP via IoU | mAP@0.5 | ≥ 85% |
| CRNN (OCR) | CER via Levenshtein Distance | CER | ≤ 15% |
| Klasifikasi | Classification Report | Weighted F1-Score | ≥ 80% |
