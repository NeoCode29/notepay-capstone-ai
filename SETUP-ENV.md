# Setup Environment GPU Lokal — NotePay Capstone

Setup ini mengkonfigurasi dua environment terpisah untuk training model AI di laptop Windows dengan GPU NVIDIA,
menggunakan WSL2 untuk TensorFlow dan Windows native untuk YOLOv8.

## Spesifikasi yang Diuji

| Komponen | Detail |
|---|---|
| OS | Windows 11 |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM) |
| Driver NVIDIA | 591.74 |
| CUDA (driver level) | 13.1 |
| Python (Windows) | 3.11.9 |
| Python (WSL2) | 3.10.12 |
| WSL2 Distro | Ubuntu 22.04 |

## Kenapa Dua Environment?

TensorFlow **tidak mendukung GPU secara native di Windows** untuk Python 3.11+.
Solusinya adalah dua environment:

| Environment | Platform | Digunakan untuk |
|---|---|---|
| `venv-yolo` (Python 3.11) | Windows | YOLOv8 training (Fase 1–3) |
| `venv-tf` (Python 3.10) | WSL2 Ubuntu 22.04 | TensorFlow CRNN training (Fase 4) |

---

## Prasyarat

1. **Driver NVIDIA** terinstall dan `nvidia-smi` bisa dijalankan di PowerShell/CMD.
2. **Python 3.11** terinstall di Windows (cek: `py -3.11 --version`).
3. **WSL2** aktif dengan Ubuntu 22.04. Cek dengan:
   ```powershell
   wsl --status
   ```
   Jika belum ada Ubuntu 22.04:
   ```powershell
   wsl --install -d Ubuntu-22.04
   ```
4. GPU harus terlihat di dalam WSL2:
   ```bash
   wsl -d Ubuntu-22.04 -- nvidia-smi
   ```
   Jika GPU tidak muncul, update driver NVIDIA ke versi terbaru dari situs resmi NVIDIA.

---

## Bagian 1: Windows — Environment YOLOv8

### 1.1 Buat Virtual Environment

Buka PowerShell di folder project, lalu:

```powershell
py -3.11 -m venv venv-yolo
```

### 1.2 Aktifkan dan Install PyTorch + CUDA

PyTorch menyertakan CUDA runtime-nya sendiri — **tidak perlu install CUDA Toolkit terpisah**.

```powershell
venv-yolo\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

> Ganti `cu124` dengan versi CUDA yang sesuai jika perlu. Lihat opsi di https://pytorch.org/get-started/locally/
> Driver NVIDIA 591.74 mendukung CUDA hingga 13.1, sehingga `cu124` kompatibel.

### 1.3 Verifikasi GPU

```powershell
python -c "import torch; print(torch.__version__); print('GPU:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Output yang diharapkan:
```
2.6.0+cu124
GPU: True
NVIDIA GeForce RTX 3050 Laptop GPU
```

### 1.4 Install Library YOLOv8 dan Pendukung

```powershell
pip install ultralytics opencv-python roboflow matplotlib pandas scikit-learn
```

### 1.5 Simpan Requirements

```powershell
pip freeze > requirements-yolo.txt
```

---

## Bagian 2: WSL2 — Environment TensorFlow GPU

Buka terminal WSL2:

```powershell
wsl -d Ubuntu-22.04
```

### 2.1 Buat Virtual Environment

```bash
python3 -m venv ~/venv-tf
```

### 2.2 Install TensorFlow dengan CUDA Bundled

`tensorflow[and-cuda]` secara otomatis mengunduh CUDA, cuDNN, dan library pendukung via pip.
**Tidak perlu install CUDA Toolkit di Ubuntu.**

```bash
source ~/venv-tf/bin/activate
pip install --upgrade pip
pip install 'tensorflow[and-cuda]' opencv-python-headless matplotlib pandas scikit-learn
```

### 2.3 Fix LD_LIBRARY_PATH (Langkah Kritis)

Library CUDA yang diunduh via pip disimpan di dalam folder venv, bukan di path sistem.
TensorFlow tidak bisa menemukannya tanpa langkah ini.

Jalankan script Python berikut **satu kali** untuk mempatch file `activate`:

```bash
python3 << 'EOF'
base = "/root/venv-tf/lib/python3.10/site-packages/nvidia"
subdirs = [
    "cublas/lib", "cuda_runtime/lib", "cudnn/lib", "cufft/lib",
    "curand/lib", "cusolver/lib", "cusparse/lib", "nvjitlink/lib", "nccl/lib"
]
ld_path = ":".join(f"{base}/{s}" for s in subdirs)

with open("/root/venv-tf/bin/activate", "r") as f:
    content = f.read()

if "# CUDA libraries for TensorFlow GPU" not in content:
    content += f"""
# CUDA libraries for TensorFlow GPU
export LD_LIBRARY_PATH="{ld_path}"
"""
    with open("/root/venv-tf/bin/activate", "w") as f:
        f.write(content)
    print("Patched successfully.")
else:
    print("Already patched, skipping.")
EOF
```

> **Catatan path:** Script di atas mengasumsikan venv ada di `/root/venv-tf` (user root).
> Jika menggunakan user non-root (misal `/home/username/venv-tf`), sesuaikan variabel `base` dan
> path di `open(...)`.

### 2.4 Verifikasi TensorFlow GPU

```bash
source ~/venv-tf/bin/activate
python -c "import tensorflow as tf; print('TF:', tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU'))"
```

Output yang diharapkan:
```
TF: 2.21.0
GPU: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

### 2.5 Simpan Requirements

```bash
pip freeze > /mnt/c/Users/<username>/Documents/.../requirements-tf.txt
```

---

## Cara Penggunaan Sehari-hari

### Menjalankan Training YOLOv8 (Windows)

```powershell
# Di PowerShell, dari folder project
venv-yolo\Scripts\activate
python fase2_train_yolo.py
```

### Menjalankan Training CRNN TF (WSL2)

```powershell
# Opsi 1: dari PowerShell Windows
wsl -d Ubuntu-22.04 -- bash -c "source ~/venv-tf/bin/activate && python /mnt/c/Users/kresna/Documents/SIB\ Dicoding/Capstone\ Project/fase4_train_crnn.py"

# Opsi 2: masuk ke WSL2 dulu
wsl -d Ubuntu-22.04
source ~/venv-tf/bin/activate
cd /mnt/c/Users/kresna/Documents/SIB\ Dicoding/Capstone\ Project/
python fase4_train_crnn.py
```

> **Tips performa:** Untuk I/O file yang lebih cepat saat training di WSL2, copy dataset ke dalam
> filesystem WSL2 (misal `~/data/`) alih-alih mengakses dari `/mnt/c/`. Path Windows via
> `/mnt/c/` lebih lambat karena melewati layer NTFS.

---

## Troubleshooting

### GPU tidak terdeteksi di PyTorch (Windows)

- Pastikan menginstall versi `cu124` atau `cu118`, bukan versi CPU.
- Cek: `python -c "import torch; print(torch.version.cuda)"` — harus menampilkan versi CUDA, bukan `None`.

### GPU tidak terdeteksi di TensorFlow (WSL2)

- Pastikan langkah **2.3 Fix LD_LIBRARY_PATH** sudah dijalankan.
- Cek apakah `LD_LIBRARY_PATH` sudah ter-set: `source ~/venv-tf/bin/activate && echo $LD_LIBRARY_PATH`
- Pastikan `nvidia-smi` berjalan di dalam WSL2 (GPU passthrough aktif).

### `nvidia-smi` tidak jalan di WSL2

- Update driver NVIDIA di Windows ke versi terbaru.
- Driver NVIDIA 470+ sudah include dukungan WSL2 GPU passthrough secara otomatis.

### Error `libcuda.so.1: cannot open shared object file`

Ini berbeda dari error library CUDA di venv. Jalankan:
```bash
ldconfig -p | grep libcuda
```
Jika tidak ada output, tambahkan path WSL2 NVIDIA ke ldconfig:
```bash
echo "/usr/lib/wsl/lib" | sudo tee /etc/ld.so.conf.d/nvidia-wsl.conf
sudo ldconfig
```

---

## Struktur File Setelah Setup

```
Capstone Project/
├── venv-yolo/              # Windows venv untuk YOLOv8 (tidak di-commit ke git)
├── requirements-yolo.txt   # Snapshot dependensi YOLOv8
├── requirements-tf.txt     # Snapshot dependensi TensorFlow (dari WSL2)
└── SETUP-ENV.md            # Dokumen ini
```

File `venv-yolo/` dan `~/venv-tf` (WSL2) tidak perlu di-commit ke git.
Cukup simpan kedua file `requirements-*.txt` agar orang lain bisa mereplikasi environment.
