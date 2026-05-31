"""
Deploy NotePay AI API ke HuggingFace Spaces (Docker).

Space baru: NeoCode77/notepay-api
URL publik : https://NeoCode77-notepay-api.hf.space

Jalankan di Windows (venv-yolo aktif):
    venv-yolo\\Scripts\\activate
    python ai/deploy_api_space.py
"""

import os
import sys

SPACE_ID  = "NeoCode77/notepay-api"
SPACE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hf_api_space")

# File yang di-upload ke Space (relatif terhadap SPACE_DIR)
UPLOAD_FILES = [
    "Dockerfile",
    "README.md",
    "requirements.txt",
    "app.py",
    "ai/__init__.py",
    "ai/config.py",
    "ai/inference.py",
    "ai/model_loader.py",
]


def deploy():
    try:
        from huggingface_hub import HfApi, whoami
    except ImportError:
        print("Install dulu: pip install huggingface_hub")
        sys.exit(1)

    username = whoami()["name"]
    print(f"Username HF  : {username}")
    print(f"Space target : {SPACE_ID}")
    print(f"Source dir   : {SPACE_DIR}\n")

    if not os.path.exists(SPACE_DIR):
        print(f"Folder tidak ditemukan: {SPACE_DIR}")
        sys.exit(1)

    api = HfApi()

    # Buat Space dengan sdk=docker
    print("Membuat Space (Docker)...")
    api.create_repo(
        repo_id=SPACE_ID,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
        private=False,
    )

    # Upload semua file
    ok, skip = 0, 0
    for filename in UPLOAD_FILES:
        filepath = os.path.join(SPACE_DIR, filename)

        if not os.path.exists(filepath):
            print(f"  [SKIP] {filename} — file tidak ditemukan")
            skip += 1
            continue

        print(f"  Uploading {filename} ...")
        api.upload_file(
            path_or_fileobj=filepath,
            path_in_repo=filename,
            repo_id=SPACE_ID,
            repo_type="space",
            commit_message=f"deploy: {filename}",
        )
        ok += 1

    print(f"\n{'='*55}")
    print(f"Upload selesai: {ok} file berhasil, {skip} diskip.")
    print(f"\nSpace URL : https://huggingface.co/spaces/{SPACE_ID}")
    print(f"API URL   : https://{SPACE_ID.replace('/', '-')}.hf.space")
    print(f"\nTunggu 5–10 menit untuk Docker build selesai.")
    print(f"Pantau log build di:")
    print(f"  https://huggingface.co/spaces/{SPACE_ID}/logs")
    print(f"\nSetelah live, test dengan:")
    print(f"  curl https://{SPACE_ID.replace('/', '-')}.hf.space/health")


if __name__ == "__main__":
    deploy()
