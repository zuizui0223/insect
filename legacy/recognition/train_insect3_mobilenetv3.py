# -*- coding: utf-8 -*-
r"""
train_insect3_mobilenetv3.py

Purpose:
- Train a 3-class insect classifier using already-downloaded iNaturalist images.
- Does NOT download images.
- Reads existing manifest CSV/JSON files from inat_insect3_dataset.
- Balances classes by downsampling to the smallest class or MAX_PER_CLASS.
- Uses MobileNetV3Small transfer learning.
- Saves Keras model, TFLite model, labels, training history, confusion matrix, and classification report.

Expected folder structure:
C:\Users\zuizui\mc\inat_insect3_dataset
  ├─ images
  │   ├─ lepidoptera
  │   ├─ hymenoptera
  │   └─ diptera
  ├─ all_manifest.csv  OR meta/*_manifest.json
  └─ model_runs

Run in PowerShell:
  cd C:\Users\zuizui\mc
  python train_insect3_mobilenetv3.py
"""

from __future__ import annotations

import json
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


# ============================================================
# User settings
# ============================================================

DATASET_DIR = Path("inat_insect3_dataset")
IMAGES_DIR = DATASET_DIR / "images"
META_DIR = DATASET_DIR / "meta"
RUNS_DIR = DATASET_DIR / "model_runs"

# Use 224 for MobileNetV3Small ImageNet pretrained weights.
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS_HEAD = 25
EPOCHS_FINE = 15
SEED = 42

# Class balance.
# None = use the smallest available class size.
# 20000 = cap each class at max 20000.
MAX_PER_CLASS = 20000

# Split ratios.
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# Fine-tuning.
DO_FINE_TUNE = True
FINE_TUNE_LAST_N_LAYERS = 80

# Training behavior.
AUTOTUNE = tf.data.AUTOTUNE

CLASS_NAMES = ["diptera", "hymenoptera", "lepidoptera"]
LABEL_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}


# ============================================================
# Utilities
# ============================================================

def set_reproducibility(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dirs() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)


def load_manifest() -> pd.DataFrame:
    """Load manifest from all_manifest.csv if present; otherwise combine meta JSON manifests."""
    csv_path = DATASET_DIR / "all_manifest.csv"

    if csv_path.exists():
        print(f"[load] {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print(f"[load] combine JSON manifests from {META_DIR}")
        rows = []
        for class_name in CLASS_NAMES:
            json_path = META_DIR / f"{class_name}_manifest.json"
            if not json_path.exists():
                print(f"[warn] missing {json_path}")
                continue
            with open(json_path, "r", encoding="utf-8") as f:
                rows.extend(json.load(f))
        df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("Manifest is empty. Check DATASET_DIR and meta/all_manifest.csv.")

    # Normalize required columns.
    if "label" not in df.columns and "class_name" in df.columns:
        df["label"] = df["class_name"]
    if "class_name" not in df.columns and "label" in df.columns:
        df["class_name"] = df["label"]

    if "relative_path" not in df.columns:
        if "filename" not in df.columns:
            raise RuntimeError("Manifest needs either relative_path or filename column.")
        df["relative_path"] = df.apply(
            lambda r: str(Path("images") / str(r["label"]) / str(r["filename"])).replace("\\", "/"),
            axis=1,
        )

    df = df[df["label"].isin(CLASS_NAMES)].copy()
    df["path"] = df["relative_path"].apply(lambda p: str(DATASET_DIR / Path(str(p))))
    df = df[df["path"].apply(lambda p: Path(p).exists())].copy()

    if df.empty:
        raise RuntimeError("No image files found from manifest paths.")

    print("[available counts]")
    print(df["label"].value_counts())
    return df


def balance_manifest(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["label"].value_counts()
    min_n = int(counts.min())
    if MAX_PER_CLASS is not None:
        min_n = min(min_n, int(MAX_PER_CLASS))

    print(f"[balance] use n per class = {min_n}")

    balanced = (
        df.groupby("label", group_keys=False)
        .apply(lambda x: x.sample(n=min_n, random_state=SEED))
        .reset_index(drop=True)
    )

    balanced = balanced.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    print("[balanced counts]")
    print(balanced["label"].value_counts())
    return balanced


def split_manifest(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["label_id"] = df["label"].map(LABEL_TO_ID).astype(int)

    train_df, temp_df = train_test_split(
        df,
        test_size=VAL_RATIO + TEST_RATIO,
        random_state=SEED,
        stratify=df["label_id"],
    )

    relative_test_ratio = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        random_state=SEED,
        stratify=temp_df["label_id"],
    )

    print("[counts]")
    print("train:")
    print(train_df["label"].value_counts())
    print("val:")
    print(val_df["label"].value_counts())
    print("test:")
    print(test_df["label"].value_counts())

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def decode_image(path: tf.Tensor, label: tf.Tensor, training: bool = False):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)  # 0..1
    image = tf.image.resize(image, IMAGE_SIZE, method="bilinear")

    if training:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.08)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        # Keep augmentation mild. Heavy rotation can hurt insect orientation cues.

    return image, label


def make_dataset(df: pd.DataFrame, training: bool) -> tf.data.Dataset:
    paths = df["path"].astype(str).to_numpy()
    labels = df["label_id"].astype(np.int32).to_numpy()

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=min(len(df), 10000), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(lambda p, y: decode_image(p, y, training=training), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


# ============================================================
# Model
# ============================================================

def build_model(num_classes: int) -> tf.keras.Model:
    """
    Correct MobileNetV3Small setup.

    decode_image() returns float images in 0..1.
    MobileNetV3Small with include_preprocessing=True expects 0..255
    and performs its own internal preprocessing.
    """
    base = tf.keras.applications.MobileNetV3Small(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        include_top=False,
        weights="imagenet",
        pooling=None,
        include_preprocessing=True,
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), name="image")

    # Convert 0..1 images back to 0..255 for MobileNetV3 internal preprocessing.
    x = tf.keras.layers.Lambda(lambda z: z * 255.0, name="scale_to_255")(inputs)

    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(0.30, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="insect3_mobilenetv3small_fixed")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_base_model(model: tf.keras.Model) -> tf.keras.Model:
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "MobileNetV3" in layer.name:
            return layer
    # Fallback by name search.
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer
    raise RuntimeError("Could not find base model for fine-tuning.")


def fine_tune_model(model: tf.keras.Model) -> tf.keras.Model:
    base = get_base_model(model)
    base.trainable = True

    # Freeze all except last N layers.
    if FINE_TUNE_LAST_N_LAYERS is not None:
        for layer in base.layers[:-FINE_TUNE_LAST_N_LAYERS]:
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ============================================================
# Evaluation and saving
# ============================================================

def predict_labels(model: tf.keras.Model, ds: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    y_true_list: List[np.ndarray] = []
    y_pred_list: List[np.ndarray] = []

    for images, labels in ds:
        probs = model.predict(images, verbose=0)
        preds = np.argmax(probs, axis=1)
        y_true_list.append(labels.numpy())
        y_pred_list.append(preds)

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    return y_true, y_pred


def save_text(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def export_tflite(model: tf.keras.Model, out_path: Path) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)


def save_run_files(
    run_dir: Path,
    model: tf.keras.Model,
    history_all: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_ds: tf.data.Dataset,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save manifests.
    train_df.to_csv(run_dir / "train_manifest_balanced.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(run_dir / "val_manifest_balanced.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(run_dir / "test_manifest_balanced.csv", index=False, encoding="utf-8-sig")

    # Save labels.
    save_text(run_dir / "labels.txt", "\n".join(CLASS_NAMES) + "\n")

    # Save history.
    with open(run_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history_all, f, ensure_ascii=False, indent=2)

    # Evaluate.
    print("[evaluate]")
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    y_true, y_pred = predict_labels(model, test_ds)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)

    print("\n[confusion matrix]")
    print(cm)
    print("\n[classification report]")
    print(report)

    np.savetxt(run_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    save_text(run_dir / "classification_report.txt", report)

    # Save model.
    keras_path = run_dir / "insect3_mobilenetv3small.keras"
    print(f"[save keras] {keras_path}")
    model.save(keras_path)

    # Save TFLite.
    tflite_path = run_dir / "insect3_mobilenetv3small.tflite"
    print(f"[export tflite] {tflite_path}")
    export_tflite(model, tflite_path)

    # Save settings.
    settings = {
        "dataset_dir": str(DATASET_DIR),
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs_head": EPOCHS_HEAD,
        "epochs_fine": EPOCHS_FINE,
        "do_fine_tune": DO_FINE_TUNE,
        "fine_tune_last_n_layers": FINE_TUNE_LAST_N_LAYERS,
        "max_per_class": MAX_PER_CLASS,
        "class_names": CLASS_NAMES,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
    }
    with open(run_dir / "settings.json", "w", encoding="utf-8") as f:
        json.dump(settings, f, ensure_ascii=False, indent=2)


# ============================================================
# Main
# ============================================================

def main() -> None:
    set_reproducibility(SEED)
    ensure_dirs()

    print(f"[start] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[dataset] {DATASET_DIR.resolve()}")
    print(f"[classes] {CLASS_NAMES}")
    print(f"[image size] {IMAGE_SIZE}")

    df = load_manifest()
    df = balance_manifest(df)
    train_df, val_df, test_df = split_manifest(df)

    train_ds = make_dataset(train_df, training=True)
    val_ds = make_dataset(val_df, training=False)
    test_ds = make_dataset(test_df, training=False)

    model = build_model(num_classes=len(CLASS_NAMES))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
            mode="max",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history_all = {}

    print("[train head]")
    hist1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        callbacks=callbacks,
        verbose=1,
    )
    history_all["head"] = hist1.history

    if DO_FINE_TUNE:
        print("[fine tune]")
        model = fine_tune_model(model)
        hist2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS_FINE,
            callbacks=callbacks,
            verbose=1,
        )
        history_all["fine"] = hist2.history

    run_dir = RUNS_DIR / f"mobilenetv3_run_{timestamp()}"
    save_run_files(run_dir, model, history_all, train_df, val_df, test_df, test_ds)

    print("\n学習完了")
    print(f"保存先: {run_dir.resolve()}")
    print(f"[end] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
