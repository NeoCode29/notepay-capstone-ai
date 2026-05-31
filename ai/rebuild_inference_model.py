"""
Rebuild inference_model.keras dari checkpoint training.

Digunakan ketika inference_model.keras tersimpan dengan Lambda layer lama
yang tidak kompatibel dengan Keras 3.x. Script ini:
  1. Load ckpt_best.keras (train model) dengan patch compute_output_shape
  2. Build model baru (arsitektur terbaru tanpa Lambda)
  3. Transfer bobot per layer berdasarkan nama
  4. Simpan inference_model.keras yang bersih

Jalankan di WSL2 dengan venv-tf atau venv-infer:
    python ai/rebuild_inference_model.py
    python ai/rebuild_inference_model.py --checkpoint models/crnn/ckpt_best.keras
"""

import argparse
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai.config import CRNN_MODEL_DIR, CRNN_KERAS_PATH, CRNN_INFER_PATH


def patch_lambda_shape():
    """Patch Keras 3.x Lambda.compute_output_shape agar tidak crash saat from_config."""
    try:
        from keras.src.layers.core.lambda_layer import Lambda as _KL
        _orig = _KL.compute_output_shape
        def _patched(self, input_shape):
            try:
                return _orig(self, input_shape)
            except NotImplementedError:
                return input_shape
        _KL.compute_output_shape = _patched
        return True
    except ImportError:
        return False


def load_train_model(path):
    import keras
    # Import CTCLayer agar dekorator @register_keras_serializable aktif
    from ai.fase4_train_crnn import CTCLayer  # noqa: F401
    patch_lambda_shape()
    print(f"  Loading: {path}")
    return keras.models.load_model(
        path, compile=False, safe_mode=False,
        custom_objects={"CTCLayer": CTCLayer},
    )


def rebuild_and_transfer(checkpoint_path, output_path):
    import keras

    # Load checkpoint (model lama, mungkin ada Lambda)
    print("Memuat checkpoint...")
    old_model = load_train_model(checkpoint_path)
    print(f"  Layers di checkpoint: {[l.name for l in old_model.layers]}")

    # Build model baru (arsitektur terbaru: Dense dtype=float32, tanpa Lambda)
    print("\nMembangun model baru...")
    from ai.fase4_train_crnn import build_crnn
    _, inference_model_new = build_crnn()

    # Transfer bobot per nama layer
    print("\nTransfer bobot...")
    transferred, skipped = 0, 0
    for layer in inference_model_new.layers:
        try:
            old_layer = old_model.get_layer(layer.name)
            weights = old_layer.get_weights()
            if weights:
                layer.set_weights(weights)
                transferred += 1
                print(f"  ✓ {layer.name} ({len(weights)} weight tensors)")
        except (ValueError, AttributeError):
            skipped += 1

    print(f"\nHasil: {transferred} layer ditransfer, {skipped} dilewati")

    # Simpan model baru
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    inference_model_new.save(output_path)
    print(f"Tersimpan: {output_path}")

    # Verifikasi
    print("\nVerifikasi load model baru...")
    test = keras.models.load_model(output_path, compile=False)
    print(f"  OK — input: {test.input_shape}  output: {test.output_shape}")


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild inference_model.keras dari checkpoint training."
    )
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(CRNN_MODEL_DIR, "ckpt_best.keras"),
        help="Path checkpoint training (default: models/crnn/ckpt_best.keras)",
    )
    parser.add_argument(
        "--output",
        default=CRNN_INFER_PATH,
        help=f"Path output inference model (default: {CRNN_INFER_PATH})",
    )
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: checkpoint tidak ditemukan: {args.checkpoint}")
        sys.exit(1)

    print("=" * 55)
    print("Rebuild inference_model.keras")
    print("=" * 55)
    rebuild_and_transfer(args.checkpoint, args.output)
    print("\nSelesai. Jalankan inference dengan:")
    print("  python ai/inference.py --image foto_struk.jpg")


if __name__ == "__main__":
    main()
