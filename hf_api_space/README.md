---
title: NotePay AI API
emoji: 🧾
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
license: mit
---

# NotePay AI API

REST API untuk pipeline OCR struk belanja — bisa dipanggil langsung dari Next.js atau backend manapun.

## Endpoints

| Method | Path | Fungsi |
|---|---|---|
| `GET` | `/health` | Status server & model |
| `POST` | `/predict` | Upload foto struk → JSON |
| `POST` | `/classify` | Klasifikasi teks item |

## POST /predict

```bash
curl -X POST https://NeoCode77-notepay-api.hf.space/predict \
  -F "file=@struk.jpg" \
  -F "conf=0.25"
```

Response:
```json
{
  "nama_toko": "INDOMARET",
  "tanggal_parsed": "2025-05-12 14:30",
  "total_parsed": 125000.0,
  "line_item_classified": [
    {"text": "Indomie Goreng", "category": "Makanan & Minuman", "confidence": 0.98}
  ],
  "kategori_summary": {"Makanan & Minuman": 3}
}
```

## POST /classify

```bash
curl -X POST https://NeoCode77-notepay-api.hf.space/classify \
  -H "Content-Type: application/json" \
  -d '{"items": ["Indomie Goreng", "Sabun Lifebuoy"]}'
```

## Model

Semua model diambil dari [`NeoCode77/notepay-models`](https://huggingface.co/NeoCode77/notepay-models) saat startup.
