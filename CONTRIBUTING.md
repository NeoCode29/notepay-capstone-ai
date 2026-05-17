# Contributing Guide — NotePay AI

## Branching Strategy

```
main  ←  production-ready, TIDAK boleh push langsung
  └── dev  ←  integrasi semua fitur
        ├── feat/ocr-pipeline
        ├── fix/crnn-inference
        └── chore/setup-fastapi
```

- Semua pekerjaan harian → buat branch baru dari `dev`
- Merge ke `dev` via Pull Request, minimal 1 reviewer
- Merge ke `main` hanya saat siap release

## Nama Branch

Format: `<type>/<deskripsi-singkat>`

| Type | Kapan dipakai |
|---|---|
| `feat/` | Fitur baru |
| `fix/` | Bug fix |
| `chore/` | Setup, config, dependencies |
| `docs/` | Dokumentasi |
| `refactor/` | Refactor tanpa ubah fungsi |

Contoh:
```
feat/fastapi-ocr-endpoint
fix/yolo-inference-crash
chore/setup-environment
```

## Format Commit Message

Format: `<type>(<scope>): <deskripsi singkat>`

Scope yang digunakan di repo ini: `yolo`, `crnn`, `inference`, `fastapi`, `config`

| Type | Kapan dipakai |
|---|---|
| `feat` | Fitur baru |
| `fix` | Bug fix |
| `chore` | Config, deps, tooling |
| `docs` | Perubahan dokumentasi |
| `refactor` | Refactor kode |
| `test` | Tambah/ubah test |

Contoh:
```
feat(crnn): tambah CTCDecodeCallback untuk monitoring akurasi per epoch
fix(inference): perbaiki crash saat gambar grayscale < 32px
chore(fastapi): setup struktur awal endpoint OCR
refactor(config): pindahkan semua hyperparameter ke config.py
docs(readme): update instruksi setup environment WSL2
```

## Alur Kerja Harian

```bash
# 1. Mulai dari dev terbaru
git checkout dev
git pull origin dev

# 2. Buat branch baru
git checkout -b feat/nama-fitur

# 3. Stage file spesifik (hindari git add .)
git add ai/fase4_train_crnn.py

# 4. Commit
git commit -m "feat(crnn): deskripsi singkat"

# 5. Push
git push origin feat/nama-fitur

# 6. Buat Pull Request ke dev via GitHub
```

## Aturan Pull Request

- Title PR ikuti format commit: `feat(scope): deskripsi`
- Minimal **1 reviewer** sebelum merge
- Hapus branch setelah PR di-merge
- Jangan merge PR milik sendiri

### Template Deskripsi PR

```
## Apa yang berubah?
- Deskripsi perubahan

## Cara test
1. Langkah pengujian

## Catatan
- Hal yang perlu diperhatikan reviewer
```

## File yang JANGAN di-commit

```
.env
models/          ← upload ke Google Drive
data/            ← terlalu besar
*.pt             ← model weights
*.keras          ← model weights
venv-yolo/
__pycache__/
*.pyc
logs/
```
