"""
Download dataset YOLOv8 OBB dari Roboflow ke data/yolo_dataset/.

Cara pakai:
    1. Salin .env.example menjadi .env dan isi nilai-nilainya
    2. Aktifkan venv-yolo, lalu jalankan:
           python ai/download_dataset.py

Atau lewat argumen langsung (tanpa .env):
    python ai/download_dataset.py \\
        --api-key xxxxxxxx \\
        --workspace nama-workspace \\
        --project nama-project \\
        --version 1
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai.config import YOLO_DATASET_DIR


def load_env_file():
    """Baca file .env di root project jika ada."""
    env_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
    )
    if not os.path.exists(env_path):
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def parse_args():
    parser = argparse.ArgumentParser(description="Download dataset Roboflow YOLOv8 OBB")
    parser.add_argument("--api-key",   default=None, help="Roboflow API key")
    parser.add_argument("--workspace", default=None, help="Slug workspace Roboflow")
    parser.add_argument("--project",   default=None, help="Slug project Roboflow")
    parser.add_argument("--version",   default=None, type=int, help="Nomor versi dataset")
    parser.add_argument("--output",    default=YOLO_DATASET_DIR, help="Folder tujuan download")
    return parser.parse_args()


def resolve_config(args):
    """Gabungkan nilai dari argumen CLI dan .env, prioritaskan CLI."""
    api_key   = args.api_key   or os.environ.get("ROBOFLOW_API_KEY")
    workspace = args.workspace or os.environ.get("ROBOFLOW_WORKSPACE")
    project   = args.project   or os.environ.get("ROBOFLOW_PROJECT")
    version   = args.version   or int(os.environ.get("ROBOFLOW_VERSION", "1"))

    missing = [name for name, val in [
        ("ROBOFLOW_API_KEY",   api_key),
        ("ROBOFLOW_WORKSPACE", workspace),
        ("ROBOFLOW_PROJECT",   project),
    ] if not val]

    if missing:
        print("ERROR: Konfigurasi berikut belum diisi:")
        for m in missing:
            print(f"  - {m}")
        print("\nIsi file .env (salin dari .env.example) atau gunakan argumen CLI.")
        print("Contoh: python ai/download_dataset.py --api-key xxx --workspace yyy --project zzz")
        sys.exit(1)

    return api_key, workspace, project, version


def download(api_key, workspace, project_slug, version, output_dir):
    try:
        from roboflow import Roboflow
    except ImportError:
        print("ERROR: package 'roboflow' belum terinstall.")
        print("Jalankan: pip install roboflow")
        sys.exit(1)

    print(f"Menghubungkan ke Roboflow...")
    print(f"  Workspace : {workspace}")
    print(f"  Project   : {project_slug}")
    print(f"  Version   : {version}")
    print(f"  Output    : {output_dir}")
    print()

    rf      = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_slug)
    dataset = project.version(version)

    print("Mengunduh dataset (format: yolov8obb)...")
    dataset.download("yolov8obb", location=output_dir, overwrite=True)

    # Verifikasi hasil download
    yaml_path = os.path.join(output_dir, "data.yaml")
    if os.path.exists(yaml_path):
        print(f"\nSelesai. Dataset tersimpan di: {output_dir}")
        _print_summary(output_dir)
    else:
        print("\nWARNING: data.yaml tidak ditemukan. Periksa hasil download secara manual.")


def _print_summary(dataset_dir):
    """Tampilkan jumlah gambar per split."""
    for split in ["train", "valid", "test"]:
        img_dir = os.path.join(dataset_dir, split, "images")
        if os.path.isdir(img_dir):
            count = len([f for f in os.listdir(img_dir)
                         if f.lower().endswith((".jpg", ".jpeg", ".png"))])
            print(f"  {split:6s}: {count} gambar")


if __name__ == "__main__":
    load_env_file()
    args   = parse_args()
    config = resolve_config(args)
    download(*config, output_dir=args.output)
