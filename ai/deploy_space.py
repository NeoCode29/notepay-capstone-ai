"""
Push Gradio Space ke Hugging Face Spaces.

Jalankan setelah upload_to_hub.py berhasil:
    venv-yolo\\Scripts\\activate
    python ai/deploy_space.py
"""

import os
import sys

HF_USERNAME = "NeoCode77"
SPACE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hf_space")


def deploy():
    try:
        from huggingface_hub import HfApi, whoami
    except ImportError:
        print("Install dulu: pip install huggingface_hub")
        sys.exit(1)

    username = HF_USERNAME or whoami()["name"]
    space_id = f"{username}/notepay-receipt-demo"

    if not os.path.exists(SPACE_DIR):
        print(f"Folder tidak ditemukan: {SPACE_DIR}")
        sys.exit(1)

    api = HfApi()

    print(f"Username HF  : {username}")
    print(f"Space target : {space_id}")

    api.create_repo(
        repo_id=space_id,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True,
        private=False,
    )

    files = ["app.py", "requirements.txt", "README.md"]
    for filename in files:
        filepath = os.path.join(SPACE_DIR, filename)
        if not os.path.exists(filepath):
            print(f"File tidak ditemukan, skip: {filename}")
            continue

        print(f"Uploading {filename} ...")
        api.upload_file(
            path_or_fileobj=filepath,
            path_in_repo=filename,
            repo_id=space_id,
            repo_type="space",
            commit_message=f"deploy: {filename}",
        )

    print(f"\nBerhasil! Space tersedia di:")
    print(f"https://huggingface.co/spaces/{space_id}")
    print("\nTunggu 2-3 menit untuk build selesai, lalu buka URL di atas.")


if __name__ == "__main__":
    deploy()
