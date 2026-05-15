"""
Fase 4 — Training model CRNN + CTC Loss untuk text recognition.

Memenuhi syarat capstone:
  - TensorFlow Functional API                 ✓ (build_crnn)
  - Custom Layer                              ✓ (CTCLayer — menghitung CTC loss)
  - Custom Callback                           ✓ (CTCDecodeCallback — decode prediksi tiap epoch)
  - Export format .keras                      ✓ (crnn_model.keras + inference_model.keras)

Jalankan di WSL2 Ubuntu dengan venv-tf aktif:
    source ~/venv-tf/bin/activate
    python /mnt/c/.../ai/fase4_train_crnn.py

Output:
    models/crnn/crnn_model.keras       ← model training (dengan CTC layer, untuk fine-tune)
    models/crnn/inference_model.keras  ← model inference (input: gambar → output: logits)
    logs/crnn/                         ← TensorBoard logs
"""

import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
import keras

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai.config import (
    OCR_IMAGES_DIR, OCR_LABELS_CSV,
    CROP_HEIGHT, CROP_WIDTH,
    NUM_CLASSES, CHAR_TO_IDX, IDX_TO_CHAR,
    MAX_LABEL_LEN, CRNN_EPOCHS, CRNN_BATCH_SIZE,
    CRNN_LR, VAL_SPLIT,
    CRNN_MODEL_DIR, CRNN_KERAS_PATH, CRNN_INFER_PATH, LOGS_DIR,
)


# ---------------------------------------------------------------------------
# Custom Layer — CTC Loss (memenuhi syarat capstone)
# ---------------------------------------------------------------------------

class CTCLayer(keras.layers.Layer):
    """
    Layer khusus yang menghitung CTC loss dan menambahkannya ke model loss.
    Dengan pendekatan ini model bisa dilatih dengan label teks panjang bervariasi
    tanpa harus memotong gambar per karakter.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len   = tf.cast(tf.shape(y_pred)[0], dtype="int64")
        input_len   = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_len   = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_len * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_len * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(tf.reduce_mean(loss))
        return y_pred

    def get_config(self):
        return super().get_config()


# ---------------------------------------------------------------------------
# Arsitektur CRNN (CNN + Bi-LSTM) — TF Functional API
# ---------------------------------------------------------------------------

def build_crnn(img_h=CROP_HEIGHT, img_w=CROP_WIDTH, num_classes=NUM_CLASSES):
    """
    Returns (train_model, inference_model).

    train_model    : inputs=[image, label], output=CTC loss via add_loss()
    inference_model: inputs=[image],        output=softmax logits (batch, time, classes)

    Arsitektur CNN menghasilkan sequence dengan time_steps = img_w // 4 = 32.
    """
    # ---- Input ----
    image_input = keras.Input(shape=(img_h, img_w, 1), name="image")
    label_input = keras.Input(shape=(None,), name="label", dtype="int64")

    # ---- CNN ----
    # Setiap blok: Conv → BN → ReLU → MaxPool
    # Output akhir: (batch, 1, img_w//4, 512) → setelah reshape: (batch, img_w//4, 512)
    x = _cnn_block(image_input, filters=32,  pool=(2, 2))   # (16, 64, 32)
    x = _cnn_block(x,           filters=64,  pool=(2, 2))   # (8, 32, 64)
    x = _cnn_block(x,           filters=128, pool=(2, 1))   # (4, 32, 128)
    x = _cnn_block(x,           filters=256, pool=(2, 1))   # (2, 32, 256)
    x = _cnn_block(x,           filters=512, pool=(2, 1))   # (1, 32, 512)

    # Squeeze dimensi tinggi agar jadi sequence (batch, 32, 512)
    time_steps = img_w // 4   # = 32
    x = keras.layers.Reshape(target_shape=(time_steps, 512), name="reshape_to_seq")(x)

    # ---- Bi-LSTM ----
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(256, return_sequences=True, dropout=0.25),
        name="bilstm_1"
    )(x)
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(256, return_sequences=True, dropout=0.25),
        name="bilstm_2"
    )(x)

    # ---- Output ----
    logits = keras.layers.Dense(num_classes, activation="softmax", name="logits")(x)
    # shape: (batch, 32, num_classes)

    # ---- Inference model (tanpa CTC layer) ----
    inference_model = keras.Model(
        inputs=image_input, outputs=logits, name="crnn_inference"
    )

    # ---- Training model (dengan CTC layer) ----
    output = CTCLayer(name="ctc_loss")(label_input, logits)
    train_model = keras.Model(
        inputs=[image_input, label_input], outputs=output, name="crnn_train"
    )

    return train_model, inference_model


def _cnn_block(x, filters, pool):
    x = keras.layers.Conv2D(filters, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(pool_size=pool)(x)
    return x


# ---------------------------------------------------------------------------
# Custom Callback — CTC Decode (memenuhi syarat capstone)
# ---------------------------------------------------------------------------

class CTCDecodeCallback(keras.callbacks.Callback):
    """
    Setiap akhir epoch, decode prediksi pada beberapa sampel validasi dan
    cetak perbandingan label asli vs prediksi model.
    """

    def __init__(self, val_dataset, inference_model, n_samples=4):
        super().__init__()
        self.val_dataset    = val_dataset
        self.inference_model = inference_model
        self.n_samples      = n_samples

    def on_epoch_end(self, epoch, logs=None):
        for images, labels in self.val_dataset.take(1):
            preds = self.inference_model(images[:self.n_samples], training=False)
            decoded = _ctc_greedy_decode(preds.numpy())
            print(f"\n  [Epoch {epoch+1}] Contoh prediksi:")
            for i, (pred_text, true_label) in enumerate(zip(decoded, labels[:self.n_samples])):
                true_text = _decode_label(true_label.numpy())
                match = "✓" if pred_text == true_text else "✗"
                print(f"    [{match}] GT: '{true_text}'  →  Pred: '{pred_text}'")


def _ctc_greedy_decode(logits_batch):
    """Greedy CTC decode: ambil argmax tiap timestep, hapus blank dan repeat."""
    texts = []
    for logits in logits_batch:
        indices = np.argmax(logits, axis=-1)   # (time_steps,)
        # hapus consecutive duplicates, lalu hapus blank (index 0)
        prev = -1
        chars = []
        for idx in indices:
            if idx != prev:
                if idx != 0:
                    chars.append(IDX_TO_CHAR.get(int(idx), "?"))
                prev = idx
        texts.append("".join(chars))
    return texts


def _decode_label(label_indices):
    """Konversi array integer label kembali ke string."""
    return "".join(IDX_TO_CHAR.get(int(i), "") for i in label_indices if i > 0)


# ---------------------------------------------------------------------------
# Data pipeline — tf.data
# ---------------------------------------------------------------------------

def load_dataset():
    if not os.path.exists(OCR_LABELS_CSV):
        raise FileNotFoundError(
            f"CSV label tidak ditemukan: {OCR_LABELS_CSV}\n"
            "Jalankan fase3_prepare_dataset.py terlebih dahulu."
        )

    df = pd.read_csv(OCR_LABELS_CSV, encoding="utf-8")

    # Hanya pakai data yang sudah diverifikasi (verified == 1)
    # Saat pertama kali training, boleh pakai semua dengan verified != -1
    df_use = df[df["verified"] != -1].copy()
    print(f"Total sampel: {len(df_use)} (dari {len(df)} baris)")

    paths  = (OCR_IMAGES_DIR + os.sep + df_use["filename"]).tolist()
    labels = df_use["label"].fillna("").tolist()

    return paths, labels


def encode_label(text):
    """Konversi string teks ke array integer, padding ke MAX_LABEL_LEN."""
    indices = [CHAR_TO_IDX[c] for c in text if c in CHAR_TO_IDX]
    indices = indices[:MAX_LABEL_LEN]
    # padding dengan 0 sampai MAX_LABEL_LEN
    indices += [0] * (MAX_LABEL_LEN - len(indices))
    return np.array(indices, dtype=np.int64)


def preprocess_image(path):
    """Load, grayscale, resize, normalize ke [0,1], tambah channel dim."""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=1)
    img = tf.image.resize(img, [CROP_HEIGHT, CROP_WIDTH])
    img = tf.cast(img, tf.float32) / 255.0
    return img


def augment_image(image):
    """Augmentasi ringan on-the-fly untuk meningkatkan ketahanan model."""
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.clip_by_value(image, 0.0, 1.0)
    # Gaussian noise simulasi kamera buram
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.03)
    image = tf.clip_by_value(image + noise, 0.0, 1.0)
    return image


def make_tf_dataset(paths, labels, batch_size, augment=False, shuffle=False):
    img_ds   = tf.data.Dataset.from_tensor_slices(paths).map(
        preprocess_image, num_parallel_calls=tf.data.AUTOTUNE
    )
    label_ds = tf.data.Dataset.from_tensor_slices(
        np.stack([encode_label(l) for l in labels])
    )
    ds = tf.data.Dataset.zip((img_ds, label_ds))

    if augment:
        ds = ds.map(lambda img, lbl: (augment_image(img), lbl),
                    num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(paths), 2000))

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train():
    print("TF version:", tf.__version__)
    print("GPU:", tf.config.list_physical_devices("GPU"))

    os.makedirs(CRNN_MODEL_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    # Data
    paths, labels = load_dataset()
    split = int(len(paths) * (1 - VAL_SPLIT))
    train_ds = make_tf_dataset(paths[:split], labels[:split],
                               CRNN_BATCH_SIZE, augment=True, shuffle=True)
    val_ds   = make_tf_dataset(paths[split:], labels[split:],
                               CRNN_BATCH_SIZE, augment=False, shuffle=False)

    # Model
    train_model, inference_model = build_crnn()
    train_model.summary()

    train_model.compile(optimizer=keras.optimizers.Adam(CRNN_LR))

    # Callbacks
    callbacks = [
        CTCDecodeCallback(val_ds, inference_model, n_samples=4),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(CRNN_MODEL_DIR, "ckpt_best.keras"),
            save_best_only=True,
            monitor="val_loss",
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
        ),
        keras.callbacks.TensorBoard(log_dir=LOGS_DIR, histogram_freq=1),
    ]

    # Training
    train_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=CRNN_EPOCHS,
        callbacks=callbacks,
    )

    # Export — format .keras sesuai syarat capstone
    train_model.save(CRNN_KERAS_PATH)
    inference_model.save(CRNN_INFER_PATH)
    print(f"\nModel training   : {CRNN_KERAS_PATH}")
    print(f"Model inference  : {CRNN_INFER_PATH}")

    # Evaluasi akhir pada val set
    print("\nEvaluasi CER pada validation set...")
    evaluate_cer(inference_model, val_ds, paths[split:], labels[split:])


def evaluate_cer(model, val_ds, paths, true_labels):
    """Hitung Character Error Rate (CER) pada validation set."""
    all_preds = []
    for images, _ in val_ds:
        preds = model(images, training=False).numpy()
        all_preds.extend(_ctc_greedy_decode(preds))

    total_chars = 0
    total_errors = 0
    for pred, gt in zip(all_preds, true_labels):
        total_chars  += max(len(gt), 1)
        total_errors += _edit_distance(pred, gt)

    cer = total_errors / total_chars
    print(f"CER: {cer:.4f} ({cer*100:.2f}%)")
    return cer


def _edit_distance(s1, s2):
    """Levenshtein distance untuk menghitung CER."""
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if s1[i-1] == s2[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]


if __name__ == "__main__":
    train()
