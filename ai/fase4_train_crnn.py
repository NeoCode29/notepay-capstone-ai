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

import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
import keras

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai.config import (
    OCR_DATASET_DIR, OCR_LABELS_CSV,
    CROP_HEIGHT, CROP_WIDTH,
    NUM_CLASSES, CHAR_TO_IDX, IDX_TO_CHAR,
    MAX_LABEL_LEN, CRNN_EPOCHS, CRNN_BATCH_SIZE,
    CRNN_LR, VAL_SPLIT,
    CRNN_MODEL_DIR, CRNN_KERAS_PATH, CRNN_INFER_PATH, LOGS_DIR,
)


# ---------------------------------------------------------------------------
# Custom Layer — CTC Loss (memenuhi syarat capstone)
# ---------------------------------------------------------------------------

@keras.saving.register_keras_serializable(package="notepay")
class CTCLayer(keras.layers.Layer):
    """
    Layer khusus yang menghitung CTC loss dan menambahkannya ke model loss.
    Dengan pendekatan ini model bisa dilatih dengan label teks panjang bervariasi
    tanpa harus memotong gambar per karakter.
    """

    def call(self, y_true, y_pred):
        batch_len = tf.shape(y_pred)[0]
        input_len = tf.shape(y_pred)[1]

        input_length = tf.cast(tf.fill([batch_len], input_len), tf.int32)

        # Hitung panjang label ASLI (non-padding) per sampel.
        # Padding menggunakan 0 (= blank token CTC), jadi jangan ikutkan ke CTC loss.
        label_length = tf.cast(
            tf.reduce_sum(tf.cast(tf.not_equal(y_true, 0), tf.int32), axis=1),
            tf.int32,
        )

        # tf.nn.ctc_loss butuh log-probabilities, bukan softmax langsung
        log_probs = tf.math.log(tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0))

        loss = tf.nn.ctc_loss(
            labels=tf.cast(y_true, tf.int32),
            logits=log_probs,
            label_length=label_length,
            logit_length=input_length,
            logits_time_major=False,
            blank_index=0,
        )
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
    x = _cnn_block(image_input, filters=32,  pool=(2, 2), dropout=0.0)
    x = _cnn_block(x,           filters=64,  pool=(2, 2), dropout=0.1)
    x = _cnn_block(x,           filters=128, pool=(2, 1), dropout=0.1)
    x = _cnn_block(x,           filters=256, pool=(2, 1), dropout=0.2)
    x = _cnn_block(x,           filters=512, pool=(2, 1), dropout=0.2)

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
    # dtype='float32' memaksa Dense output tetap float32 meski global policy mixed_float16.
    # ctc_batch_cost membutuhkan float32 agar numerically stable.
    logits = keras.layers.Dense(
        num_classes, activation="softmax", name="logits", dtype="float32"
    )(x)
    # shape: (batch, 128, num_classes)

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


def _cnn_block(x, filters, pool, dropout=0.1):
    x = keras.layers.Conv2D(filters, 3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(pool_size=pool)(x)
    if dropout > 0:
        x = keras.layers.Dropout(dropout)(x)
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
        for batch in self.val_dataset.take(1):
            images = batch["image"][:self.n_samples]
            labels = batch["label"][:self.n_samples]
            preds = self.inference_model(images, training=False)
            decoded = _ctc_greedy_decode(preds.numpy())
            print(f"\n  [Epoch {epoch+1}] Contoh prediksi:")
            for pred_text, true_label in zip(decoded, labels):
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

def _load_split_from_csv(csv_path, base_dir):
    """Baca labels.csv dan kembalikan (paths, labels)."""
    df = pd.read_csv(
        csv_path,
        encoding="utf-8",
        quotechar='"',
        quoting=csv.QUOTE_ALL,
        on_bad_lines="warn",
    )
    paths  = [os.path.join(base_dir, fp) for fp in df["filepath"].tolist()]
    labels = df["label"].fillna("").tolist()
    return paths, labels


def load_dataset(dataset_dir=None):
    """
    Auto-detect struktur dataset:
      - Split    : dataset_dir/train/labels.csv + dataset_dir/val/labels.csv
      - Flat     : dataset_dir/labels.csv (split otomatis dengan VAL_SPLIT)
    """
    base = dataset_dir or OCR_DATASET_DIR

    train_csv = os.path.join(base, "train", "labels.csv")
    val_csv   = os.path.join(base, "val",   "labels.csv")

    if os.path.exists(train_csv) and os.path.exists(val_csv):
        # Struktur split (hasil augment_dataset.py)
        train_paths, train_labels = _load_split_from_csv(
            train_csv, os.path.join(base, "train"))
        val_paths, val_labels = _load_split_from_csv(
            val_csv, os.path.join(base, "val"))
        print(f"Dataset (split) : train={len(train_paths)}  val={len(val_paths)}")
        return train_paths, train_labels, val_paths, val_labels

    # Struktur flat — split manual
    flat_csv = os.path.join(base, "labels.csv")
    if not os.path.exists(flat_csv):
        raise FileNotFoundError(
            f"labels.csv tidak ditemukan di: {base}\n"
            "Jalankan fase3_prepare_dataset.py atau augment_dataset.py terlebih dahulu."
        )
    paths, labels = _load_split_from_csv(flat_csv, base)
    split = int(len(paths) * (1 - VAL_SPLIT))
    print(f"Dataset (flat)  : total={len(paths)}  "
          f"train={split}  val={len(paths)-split}")
    return paths[:split], labels[:split], paths[split:], labels[split:]


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

    ds = ds.batch(batch_size)

    # Keras 3 membaca (x, y) dari dataset — padahal model butuh KEDUANYA sebagai input.
    # Yield dict agar Keras 3 memetakan key ke nama InputLayer secara eksplisit.
    ds = ds.map(lambda img, lbl: {"image": img, "label": lbl},
                num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _configure_gpu(memory_limit_mb=3800):
    """Konfigurasi GPU: tumbuh dinamis hingga memory_limit_mb (default 3.8GB dari 4GB)."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_mb)]
            )
            print(f"GPU memory limit: {memory_limit_mb} MB")
        except RuntimeError as e:
            print(f"GPU config warning: {e}")

    # Disable layout optimizer — mencegah crash cuDNN 1002 (NHWC→NCHW conversion gagal
    # pada TF 2.21 + cuDNN 9.2 di WSL2 dengan GPU laptop tertentu).
    tf.config.optimizer.set_experimental_options({"layout_optimizer": False})


def train(dataset_dir=None):
    _configure_gpu(3800)

    # Mixed precision — float16 untuk forward pass, float32 untuk optimizer.
    # Hemat ~50% VRAM sehingga model 32×512 + BiLSTM muat di 4GB VRAM.
    keras.mixed_precision.set_global_policy("mixed_float16")
    print("Mixed precision:", keras.mixed_precision.global_policy().name)

    print("TF version:", tf.__version__)
    print("GPU:", tf.config.list_physical_devices("GPU"))

    os.makedirs(CRNN_MODEL_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    # Data
    train_paths, train_labels, val_paths, val_labels = load_dataset(dataset_dir)
    train_ds = make_tf_dataset(train_paths, train_labels,
                               CRNN_BATCH_SIZE, augment=True, shuffle=True)
    val_ds   = make_tf_dataset(val_paths, val_labels,
                               CRNN_BATCH_SIZE, augment=False, shuffle=False)

    # Model
    train_model, inference_model = build_crnn()
    train_model.summary()

    train_model.compile(
        optimizer=keras.optimizers.Adam(CRNN_LR, clipnorm=1.0)
    )

    # Callbacks
    callbacks = [
        keras.callbacks.TerminateOnNaN(),
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
    evaluate_cer(inference_model, val_ds, val_paths, val_labels)


def evaluate_cer(model, val_ds, paths, true_labels):
    """Hitung Character Error Rate (CER) pada validation set."""
    all_preds = []
    for batch in val_ds:
        preds = model(batch["image"], training=False).numpy()
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default=None,
        help="Path folder dataset. "
             "Jika ada train/labels.csv dan val/labels.csv maka pakai struktur split. "
             "Jika tidak ada, pakai labels.csv flat dengan VAL_SPLIT otomatis. "
             f"Default: {OCR_DATASET_DIR}"
    )
    args = parser.parse_args()
    train(dataset_dir=args.dataset)
