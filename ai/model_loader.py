"""
Model loader untuk FastAPI server — download dari HF Hub, load sekali, simpan di memori.

Cara pakai di FastAPI:
    from ai.model_loader import get_models

    @app.on_event("startup")
    async def startup():
        get_models()   # pre-load saat startup

    @app.post("/predict")
    async def predict(file: UploadFile):
        yolo, crnn, classifier = get_models()
        # ... inference ...

Variabel environment yang bisa di-set:
    HF_REPO_ID          override repo HF (default: NeoCode77/notepay-models)
    MODEL_CACHE_DIR     override folder cache lokal (default: models/)
    NOTEPAY_DEVICE      "cpu" untuk paksa CPU (default: auto-detect GPU)
"""

import os
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai.config import (
    MODELS_DIR,
    YOLO_BEST_PT, CRNN_INFER_PATH, CLASSIFIER_MODEL_PATH,
    EXPENSE_CATEGORIES,
)

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------

HF_REPO_ID = os.environ.get("HF_REPO_ID", "NeoCode77/notepay-models")

# Path lokal cache — pakai MODELS_DIR dari config kecuali di-override
_CACHE_DIR = Path(os.environ.get("MODEL_CACHE_DIR", MODELS_DIR))

# Mapping nama model → path cache lokal & nama file di HF Hub
_MODEL_SPECS = {
    "yolo": {
        "local"   : Path(YOLO_BEST_PT),
        "hf_file" : "yolo/best.pt",
    },
    "crnn": {
        "local"   : Path(CRNN_INFER_PATH),
        "hf_file" : "crnn/inference_model.keras",
    },
    "classifier": {
        "local"   : Path(CLASSIFIER_MODEL_PATH),
        "hf_file" : "classifier/classifier_model.keras",
    },
}

# Singleton — diisi saat load_models() pertama kali dipanggil
_LOADED: dict = {}


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------

def _download_if_missing(name: str, spec: dict) -> Path:
    """
    Cek apakah model ada di local cache.
    Kalau tidak ada, download dari HF Hub ke path yang sama.
    """
    local: Path = spec["local"]

    if local.exists():
        logger.info(f"[model_loader] {name}: ditemukan di cache → {local}")
        return local

    logger.info(f"[model_loader] {name}: tidak ada lokal, download dari {HF_REPO_ID} ...")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub belum terinstall. Jalankan: pip install huggingface_hub"
        )

    local.parent.mkdir(parents=True, exist_ok=True)

    # hf_hub_download menyimpan ke cache HF (~/.cache/huggingface/hub/)
    # kita salin ke local path agar sesuai dengan config.py
    downloaded = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=spec["hf_file"],
        local_dir=str(local.parent),
        local_dir_use_symlinks=False,
    )

    # hf_hub_download dengan local_dir langsung simpan di sana,
    # tapi nama filenya bisa berbeda (nested subfolder dari hf_file).
    # Pastikan file ada di path yang diharapkan.
    downloaded_path = Path(downloaded)
    if downloaded_path != local:
        downloaded_path.rename(local)

    logger.info(f"[model_loader] {name}: download selesai → {local}")
    return local


# ---------------------------------------------------------------------------
# GPU setup (sama dengan inference.py agar tidak konflik CUDA context)
# ---------------------------------------------------------------------------

def _setup_devices() -> tuple[str, str]:
    force_cpu = os.environ.get("NOTEPAY_DEVICE", "").lower() == "cpu"

    if force_cpu:
        return "cpu", "cpu"

    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.optimizer.set_experimental_options({"layout_optimizer": False})
        tf_device = "cuda" if gpus else "cpu"
    except Exception:
        tf_device = "cpu"

    try:
        import torch
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        torch_device = "cpu"

    return torch_device, tf_device


# ---------------------------------------------------------------------------
# Load semua model
# ---------------------------------------------------------------------------

def load_models(force_reload: bool = False) -> tuple:
    """
    Download (jika perlu) dan load semua model ke memori.
    Simpan sebagai singleton — panggil berkali-kali aman, load hanya sekali.

    Returns:
        (yolo, crnn, classifier, torch_device)
        classifier bisa None jika model tidak tersedia.
    """
    global _LOADED

    if _LOADED and not force_reload:
        return (
            _LOADED["yolo"],
            _LOADED["crnn"],
            _LOADED["classifier"],
            _LOADED["torch_device"],
        )

    from ultralytics import YOLO
    import keras

    logger.info("[model_loader] Setup GPU ...")
    torch_device, _ = _setup_devices()
    logger.info(f"[model_loader] torch_device={torch_device}")

    # --- YOLO ---
    yolo_path = _download_if_missing("yolo", _MODEL_SPECS["yolo"])
    logger.info(f"[model_loader] Loading YOLO dari {yolo_path}")
    import numpy as np
    yolo = YOLO(str(yolo_path))
    yolo(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False, device=torch_device)
    logger.info("[model_loader] YOLO siap")

    # --- CRNN ---
    crnn_path = _download_if_missing("crnn", _MODEL_SPECS["crnn"])
    logger.info(f"[model_loader] Loading CRNN dari {crnn_path}")
    try:
        from keras.src.layers.core.lambda_layer import Lambda as _KLambda
        _orig = _KLambda.compute_output_shape
        def _patched(self, input_shape):
            try:
                return _orig(self, input_shape)
            except NotImplementedError:
                return input_shape
        _KLambda.compute_output_shape = _patched
    except ImportError:
        pass
    crnn = keras.models.load_model(str(crnn_path), compile=False, safe_mode=False)
    # Warmup CRNN: compile TF graph sekarang agar request pertama tidak lambat
    _dummy_img = np.zeros((1, 32, 512, 1), dtype=np.float32)
    crnn(_dummy_img, training=False)
    logger.info("[model_loader] CRNN siap + warmup selesai")

    # --- Classifier (opsional) ---
    classifier = None
    try:
        clf_path = _download_if_missing("classifier", _MODEL_SPECS["classifier"])
        logger.info(f"[model_loader] Loading classifier dari {clf_path}")
        classifier = keras.models.load_model(str(clf_path), compile=False)
        # Warmup Classifier: compile TF graph sekarang
        import tensorflow as tf
        classifier(tf.constant([["warmup"]]), training=False)
        logger.info("[model_loader] Classifier siap + warmup selesai")
    except Exception as e:
        logger.warning(f"[model_loader] Classifier tidak bisa di-load: {e}")

    _LOADED = {
        "yolo"        : yolo,
        "crnn"        : crnn,
        "classifier"  : classifier,
        "torch_device": torch_device,
    }

    logger.info("[model_loader] Semua model siap.")
    return yolo, crnn, classifier, torch_device


def get_models() -> tuple:
    """Shortcut — alias load_models() untuk dipakai di FastAPI dependency."""
    return load_models()


def unload_models():
    """Bebaskan memori GPU/CPU — panggil saat shutdown FastAPI jika perlu."""
    global _LOADED
    _LOADED.clear()
    try:
        import keras.backend as K
        K.clear_session()
    except Exception:
        pass
    logger.info("[model_loader] Semua model di-unload.")


# ---------------------------------------------------------------------------
# Info
# ---------------------------------------------------------------------------

def model_info() -> dict:
    """
    Kembalikan status model (sudah di-load atau belum, path, HF repo).
    Berguna untuk endpoint /health di FastAPI.
    """
    info: dict = {
        "hf_repo"   : HF_REPO_ID,
        "loaded"    : bool(_LOADED),
        "models"    : {},
    }
    for name, spec in _MODEL_SPECS.items():
        local: Path = spec["local"]
        info["models"][name] = {
            "local_path" : str(local),
            "cached"     : local.exists(),
            "hf_file"    : spec["hf_file"],
            "loaded"     : name in _LOADED,
        }
    if _LOADED:
        info["torch_device"] = _LOADED.get("torch_device", "unknown")
        info["expense_categories"] = EXPENSE_CATEGORIES
    return info


# ---------------------------------------------------------------------------
# CLI — test loader standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("Mengecek status model...\n")
    print(json.dumps(model_info(), indent=2, default=str))

    print("\nMemuat semua model...")
    yolo, crnn, classifier, torch_device = load_models()

    print("\nStatus setelah load:")
    print(json.dumps(model_info(), indent=2, default=str))

    print(f"\nDevice     : {torch_device}")
    print(f"YOLO       : {type(yolo).__name__}")
    print(f"CRNN       : {crnn.name}")
    print(f"Classifier : {classifier.name if classifier else 'tidak tersedia'}")
    print("\nSemua model siap dipakai oleh FastAPI server.")
