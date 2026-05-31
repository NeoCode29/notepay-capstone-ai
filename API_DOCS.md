# NotePay AI API — Dokumentasi

**Base URL (Production):**
```
https://NeoCode77-notepay-api.hf.space
```

**Base URL (Local Development):**
```
http://localhost:5000
```

---

## Daftar Endpoint

| Method | Path | Fungsi |
|---|---|---|
| `GET` | `/health` | Cek status server & model |
| `POST` | `/predict` | Upload foto struk → JSON hasil OCR |
| `POST` | `/classify` | Klasifikasi teks item pengeluaran |

---

## GET /health

Cek apakah server dan semua model sudah siap.

### Response

```json
{
  "status": "ok",
  "model_info": {
    "hf_repo": "NeoCode77/notepay-models",
    "loaded": true,
    "torch_device": "cpu",
    "expense_categories": [
      "Makanan & Minuman",
      "Kebersihan & Perawatan",
      "Rumah Tangga",
      "Kesehatan & Farmasi",
      "Elektronik & Pulsa",
      "Pakaian & Aksesori",
      "Lain-lain"
    ],
    "models": {
      "yolo":       { "cached": true, "loaded": true },
      "crnn":       { "cached": true, "loaded": true },
      "classifier": { "cached": true, "loaded": true }
    }
  }
}
```

| Field | Tipe | Keterangan |
|---|---|---|
| `status` | `string` | `"ok"` atau `"degraded"` (model belum siap) |
| `model_info.loaded` | `boolean` | `true` jika semua model sudah di-load |

### Contoh

```bash
curl https://NeoCode77-notepay-api.hf.space/health
```

---

## POST /predict

Upload foto struk belanja dan dapatkan hasil ekstraksi teks + klasifikasi pengeluaran.

### Request

- **Content-Type:** `multipart/form-data`

| Field | Tipe | Wajib | Keterangan |
|---|---|---|---|
| `file` | `File` | ✅ | Foto struk (jpg / png / webp, maks 20 MB) |
| `conf` | `float` | ❌ | Confidence threshold YOLO (0.05–0.95, default `0.25`) |

### Response `200 OK`

```json
{
  "nama_toko": "INDOMARET",
  "tanggal_waktu": "12/05/2025 14:30",
  "tanggal_parsed": "2025-05-12 14:30",
  "total_belanja": "Rp 125.000",
  "total_parsed": 125000.0,
  "line_item": [
    "Indomie Goreng",
    "Aqua 600ml",
    "Sabun Lifebuoy"
  ],
  "line_item_classified": [
    {
      "text": "Indomie Goreng",
      "category": "Makanan & Minuman",
      "confidence": 0.9823
    },
    {
      "text": "Aqua 600ml",
      "category": "Makanan & Minuman",
      "confidence": 0.9761
    },
    {
      "text": "Sabun Lifebuoy",
      "category": "Kebersihan & Perawatan",
      "confidence": 0.9914
    }
  ],
  "kategori_summary": {
    "Makanan & Minuman": 2,
    "Kebersihan & Perawatan": 1
  },
  "elapsed_ms": 1243,
  "detections_count": 4
}
```

| Field | Tipe | Keterangan |
|---|---|---|
| `nama_toko` | `string` | Nama toko dari OCR |
| `tanggal_waktu` | `string` | Tanggal/waktu raw dari OCR |
| `tanggal_parsed` | `string \| null` | Tanggal terformat `YYYY-MM-DD HH:MM`, `null` jika gagal parse |
| `total_belanja` | `string` | Total belanja raw dari OCR |
| `total_parsed` | `number \| null` | Total dalam angka (Rupiah), `null` jika gagal parse |
| `line_item` | `string[]` | List teks item mentah dari OCR |
| `line_item_classified` | `object[]` | List item + kategori + confidence |
| `kategori_summary` | `object` | Ringkasan jumlah item per kategori |
| `elapsed_ms` | `number` | Waktu proses total (ms) |
| `detections_count` | `number` | Jumlah region yang terdeteksi YOLO |

### Contoh

```bash
# cURL
curl -X POST https://NeoCode77-notepay-api.hf.space/predict \
  -F "file=@struk.jpg" \
  -F "conf=0.25"
```

```js
// Next.js / JavaScript
const formData = new FormData()
formData.append('file', file)          // File object dari <input type="file">
formData.append('conf', '0.25')

const res = await fetch('https://NeoCode77-notepay-api.hf.space/predict', {
  method: 'POST',
  body: formData,
})
const data = await res.json()
console.log(data.total_parsed)         // 125000
console.log(data.kategori_summary)     // { "Makanan & Minuman": 2, ... }
```

```python
# Python
import requests

with open('struk.jpg', 'rb') as f:
    res = requests.post(
        'https://NeoCode77-notepay-api.hf.space/predict',
        files={'file': ('struk.jpg', f, 'image/jpeg')},
        data={'conf': 0.25},
    )
print(res.json())
```

---

## POST /classify

Klasifikasikan list teks item ke kategori pengeluaran **tanpa memerlukan gambar**. Berguna untuk re-klasifikasi atau testing.

### Request

- **Content-Type:** `application/json`

```json
{
  "items": [
    "Indomie Goreng",
    "Sabun Lifebuoy",
    "Paracetamol 500mg",
    "Pulsa Telkomsel 50.000"
  ]
}
```

| Field | Tipe | Wajib | Keterangan |
|---|---|---|---|
| `items` | `string[]` | ✅ | List teks item (maks 100 item) |

### Response `200 OK`

```json
{
  "results": [
    {
      "text": "Indomie Goreng",
      "category": "Makanan & Minuman",
      "confidence": 0.9823
    },
    {
      "text": "Sabun Lifebuoy",
      "category": "Kebersihan & Perawatan",
      "confidence": 0.9914
    },
    {
      "text": "Paracetamol 500mg",
      "category": "Kesehatan & Farmasi",
      "confidence": 0.9756
    },
    {
      "text": "Pulsa Telkomsel 50.000",
      "category": "Elektronik & Pulsa",
      "confidence": 0.9988
    }
  ]
}
```

### Contoh

```bash
curl -X POST https://NeoCode77-notepay-api.hf.space/classify \
  -H "Content-Type: application/json" \
  -d '{"items": ["Indomie Goreng", "Sabun Lifebuoy"]}'
```

```js
// Next.js / JavaScript
const res = await fetch('https://NeoCode77-notepay-api.hf.space/classify', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    items: ['Indomie Goreng', 'Sabun Lifebuoy', 'Paracetamol'],
  }),
})
const { results } = await res.json()
```

---

## Kategori Pengeluaran

| ID | Kategori | Contoh Item |
|---|---|---|
| 0 | Makanan & Minuman | Indomie, Aqua, Beras, Kopi, Snack |
| 1 | Kebersihan & Perawatan | Sabun, Shampoo, Deterjen, Pembalut |
| 2 | Rumah Tangga | Sabun piring, Pembersih lantai, Baterai, Lampu |
| 3 | Kesehatan & Farmasi | Paracetamol, Vitamin, Betadine, Masker |
| 4 | Elektronik & Pulsa | Pulsa, Token Listrik, Kuota, Kabel charger |
| 5 | Pakaian & Aksesori | Kaos kaki, Celana dalam, Sendal |
| 6 | Lain-lain | Rokok, Administrasi, ATK |

---

## Error Responses

Semua error dikembalikan dalam format:

```json
{
  "error": "Pesan error yang menjelaskan masalah."
}
```

| Status Code | Kondisi |
|---|---|
| `400` | Request tidak valid (field hilang, data salah) |
| `413` | File terlalu besar (> 20 MB) |
| `415` | Tipe file tidak didukung (bukan jpg/png/webp) |
| `422` | Gambar tidak bisa di-decode / rusak |
| `500` | Error saat inference |
| `503` | Model belum siap (masih loading) |

---

## Integrasi Next.js (API Route)

Contoh API route di Next.js yang mem-proxy ke AI API:

```ts
// app/api/scan/route.ts
import { NextRequest, NextResponse } from 'next/server'

const AI_URL = process.env.NOTEPAY_AI_URL ?? 'https://NeoCode77-notepay-api.hf.space'

export async function POST(req: NextRequest) {
  const formData = await req.formData()
  const file = formData.get('file') as File

  if (!file) {
    return NextResponse.json({ error: 'File tidak ditemukan' }, { status: 400 })
  }

  // Forward ke AI API
  const aiForm = new FormData()
  aiForm.append('file', file)
  aiForm.append('conf', formData.get('conf')?.toString() ?? '0.25')

  const aiRes = await fetch(`${AI_URL}/predict`, {
    method: 'POST',
    body: aiForm,
  })

  if (!aiRes.ok) {
    const err = await aiRes.json()
    return NextResponse.json(err, { status: aiRes.status })
  }

  const result = await aiRes.json()

  // Simpan ke Supabase (opsional)
  // await supabase.from('transactions').insert({ ... })

  return NextResponse.json(result)
}
```

```ts
// .env.local
NOTEPAY_AI_URL=https://NeoCode77-notepay-api.hf.space
```

---

## Catatan Teknis

| Item | Detail |
|---|---|
| **Pipeline** | YOLO OBB → CRNN + CTC → Text Classifier |
| **Model repo** | `NeoCode77/notepay-models` (HuggingFace Hub) |
| **Framework** | Flask 3.x + Gunicorn |
| **Runtime** | Python 3.10, CPU-only |
| **Waktu cold start** | ~60–120 detik (download + load model pertama kali) |
| **Waktu inference** | ~1–5 detik per gambar |
| **Max file size** | 20 MB |
| **Format gambar** | jpg, png, webp |
