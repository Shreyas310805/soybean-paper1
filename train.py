"""
train.py  (Google Colab edition)
================================
Trains the four required transfer-learning models **sequentially**, clearing the
Keras session between runs so a T4 never holds more than one graph.

Colab hardening
---------------
* Each model is trained inside its own try/except. If DenseNet121 OOMs at 3 a.m.
  you still keep the three models that already finished, plus their tables.
* `--skip-existing` (recommended for every rerun) checks saved_models/ on Drive
  and resumes from the first model that has not finished.
* Per-epoch checkpoints go to local scratch, final models to Drive. Writing a
  30 MB checkpoint to a network mount every epoch stalls training.
* On an OOM the script prints the exact retry command with a smaller batch.

Usage
-----
    python train.py                              # all four models
    python train.py --skip-existing              # after a disconnect
    python train.py --models densenet121 --batch-size 16
    python train.py --fine-tune-epochs 5         # optional second stage
"""

from __future__ import annotations

import argparse
import csv
import gc
import inspect
import json
import random
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import (
    DenseNet121,
    EfficientNetB0,
    MobileNetV2,
    MobileNetV3Small,
)

import config
import data_loader

BASE_BUILDERS: Dict[str, Callable[..., keras.Model]] = {
    "mobilenetv2": MobileNetV2,
    "mobilenetv3_small": MobileNetV3Small,
    "efficientnetb0": EfficientNetB0,
    "densenet121": DenseNet121,
}

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_VAR = [0.229 ** 2, 0.224 ** 2, 0.225 ** 2]


# --------------------------------------------------------------------------- #
# Runtime setup
# --------------------------------------------------------------------------- #
def set_global_seeds(seed: int = config.RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        keras.utils.set_random_seed(seed)
    except AttributeError:
        pass


def configure_runtime(mixed_precision: bool) -> None:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    print(f"[train] TensorFlow {tf.__version__}")
    if gpus:
        try:
            name = tf.config.experimental.get_device_details(gpus[0]).get(
                "device_name", "unknown")
        except Exception:  # noqa: BLE001
            name = "unknown"
        print(f"[train] GPU: {name} ({len(gpus)} device(s)), memory growth on")
    else:
        print("[train] WARNING: no GPU visible. In Colab: "
              "Runtime > Change runtime type > T4 GPU.")

    if mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")
        print("[train] Mixed precision (float16) ENABLED - roughly 1.3-1.7x "
              "faster on a T4.")
        print("[train] NOTE: the saved model keeps a float16 policy. Retrain "
              "the winner in float32 before quantize.py so Table 4 compares "
              "a clean FP32 baseline against INT8.")


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #
def _build_backbone(slug: str):
    builder = BASE_BUILDERS[slug]
    kwargs = dict(include_top=False, weights="imagenet",
                  input_shape=config.INPUT_SHAPE)
    handles_scaling = False
    if "include_preprocessing" in inspect.signature(builder).parameters:
        kwargs["include_preprocessing"] = True
        handles_scaling = True
    return builder(**kwargs), handles_scaling


def _preprocessing_layers(slug: str, handles_scaling: bool) -> List[layers.Layer]:
    """Model-specific ImageNet normalisation as serialisable layers."""
    if handles_scaling:
        return []
    if slug == "efficientnetb0":
        return []
    if slug in {"mobilenetv2", "mobilenetv3_small"}:
        return [layers.Rescaling(1.0 / 127.5, offset=-1.0, name="preprocess")]
    return [
        layers.Rescaling(1.0 / 255.0, name="rescale"),
        layers.Normalization(mean=_IMAGENET_MEAN, variance=_IMAGENET_VAR,
                             name="imagenet_norm"),
    ]


def build_model(slug: str, num_classes: int) -> keras.Model:
    backbone, handles_scaling = _build_backbone(slug)
    backbone.trainable = False

    inputs = keras.Input(shape=config.INPUT_SHAPE, dtype="float32",
                         name="raw_image_0_255")
    x = inputs
    for layer in _preprocessing_layers(slug, handles_scaling):
        x = layer(x)
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(config.DROPOUT_RATE, name="dropout")(x)
    # dtype float32 keeps the softmax numerically safe under mixed precision.
    outputs = layers.Dense(num_classes, activation="softmax",
                           dtype="float32", name="predictions")(x)

    model = keras.Model(inputs, outputs, name=slug)
    model._backbone_name = backbone.name  # noqa: SLF001
    return model


def compile_model(model: keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )


def _callbacks(slug: str) -> List[keras.callbacks.Callback]:
    ckpt = config.CHECKPOINT_DIR / f"{slug}_best.keras"   # local scratch
    return [
        keras.callbacks.EarlyStopping(
            monitor=config.MONITOR_METRIC, mode=config.MONITOR_MODE,
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt), monitor=config.MONITOR_METRIC,
            mode=config.MONITOR_MODE, save_best_only=True, verbose=0),
        keras.callbacks.ReduceLROnPlateau(
            monitor=config.MONITOR_METRIC, mode=config.MONITOR_MODE,
            factor=config.REDUCE_LR_FACTOR, patience=config.REDUCE_LR_PATIENCE,
            min_lr=config.MIN_LR, verbose=1),
        keras.callbacks.CSVLogger(str(config.HISTORY_DIR / f"{slug}_log.csv")),
    ]


def _unfreeze_top(model: keras.Model, ratio: float) -> None:
    backbone = model.get_layer(model._backbone_name)  # noqa: SLF001
    backbone.trainable = True
    cutoff = int(len(backbone.layers) * (1.0 - ratio))
    for i, layer in enumerate(backbone.layers):
        layer.trainable = (i >= cutoff) and not isinstance(
            layer, layers.BatchNormalization)
    trainable = sum(1 for l in backbone.layers if l.trainable)
    print(f"[train] Fine-tuning: {trainable}/{len(backbone.layers)} backbone "
          f"layers unfrozen (BatchNorm kept frozen).")


def _merge_histories(first: dict, second: dict | None) -> dict:
    if not second:
        return first
    merged = {k: list(v) for k, v in first.items()}
    for k, v in second.items():
        merged.setdefault(k, []).extend(list(v))
    return merged


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_one(slug: str, args: argparse.Namespace) -> dict:
    print("\n" + "=" * 72)
    print(f"  Training {config.display_name(slug)}   "
          f"[{datetime.now():%H:%M:%S}]")
    print("=" * 72)

    keras.backend.clear_session()
    gc.collect()
    set_global_seeds(args.seed if args.seed is not None else config.RANDOM_SEED)

    train_ds = data_loader.make_dataset("train", batch_size=args.batch_size)
    val_ds = data_loader.make_dataset("val", batch_size=args.batch_size)
    class_names, _ = data_loader.load_manifest()

    model = build_model(slug, num_classes=len(class_names))
    compile_model(model, config.HEAD_LEARNING_RATE)
    if args.summary:
        model.summary()

    class_weight = data_loader.compute_class_weights() if args.class_weight else None
    if class_weight:
        print(f"[train] Class weights: {class_weight}")

    start = time.perf_counter()
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                        callbacks=_callbacks(slug), class_weight=class_weight,
                        verbose=1)
    hist = dict(history.history)
    epochs_run = len(hist["loss"])

    ft_hist = None
    if args.fine_tune_epochs > 0:
        _unfreeze_top(model, config.FINE_TUNE_UNFREEZE_RATIO)
        compile_model(model, config.FINE_TUNE_LEARNING_RATE)
        ft = model.fit(train_ds, validation_data=val_ds,
                       epochs=epochs_run + args.fine_tune_epochs,
                       initial_epoch=epochs_run, callbacks=_callbacks(slug),
                       class_weight=class_weight, verbose=1)
        ft_hist = dict(ft.history)

    elapsed = time.perf_counter() - start
    hist = _merge_histories(hist, ft_hist)

    # Final model -> Drive (persistent).
    out_path = config.SAVED_MODELS_DIR / f"{slug}.keras"
    model.save(out_path, include_optimizer=False)
    with open(config.HISTORY_DIR / f"{slug}_history.json", "w",
              encoding="utf-8") as fh:
        json.dump({k: [float(x) for x in v] for k, v in hist.items()}, fh, indent=2)

    best_idx = int(np.argmin(hist["val_loss"]))
    record = {
        "model": config.display_name(slug),
        "slug": slug,
        "epochs_run": len(hist["loss"]),
        "best_epoch": best_idx + 1,
        "best_val_loss": round(float(hist["val_loss"][best_idx]), 5),
        "best_val_accuracy": round(float(hist["val_accuracy"][best_idx]), 5),
        "train_time_sec": round(elapsed, 1),
        "total_params": int(model.count_params()),
        "batch_size": args.batch_size,
        "saved_to": str(out_path),
    }
    print(f"[train] Saved -> {out_path}  ({elapsed / 60:.1f} min, "
          f"best val_acc={record['best_val_accuracy']:.4f})")

    del model, train_ds, val_ds
    keras.backend.clear_session()
    gc.collect()
    return record


def _write_summary(records: List[dict]) -> None:
    """Merge with any previous run so resumed sessions keep earlier rows."""
    if not records:
        return
    existing: List[dict] = []
    if config.TRAINING_SUMMARY_CSV.exists():
        with open(config.TRAINING_SUMMARY_CSV, newline="", encoding="utf-8") as fh:
            existing = list(csv.DictReader(fh))

    fieldnames = list(records[0].keys())
    by_slug = {r["slug"]: {k: r.get(k, "") for k in fieldnames} for r in existing}
    for r in records:
        by_slug[r["slug"]] = {k: str(v) for k, v in r.items()}
    ordered = [by_slug[s] for s in config.MODEL_SLUGS if s in by_slug]

    with open(config.TRAINING_SUMMARY_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ordered)
    print(f"\n[train] Training summary -> {config.TRAINING_SUMMARY_CSV}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the four lightweight models.")
    parser.add_argument("--models", nargs="+", default=config.MODEL_SLUGS,
                        choices=config.MODEL_SLUGS)
    parser.add_argument("--epochs", type=int, default=config.MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--fine-tune-epochs", type=int,
                        default=config.FINE_TUNE_EPOCHS)
    parser.add_argument("--class-weight", action="store_true")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Resume: skip models already in saved_models/.")
    parser.add_argument("--mixed-precision", action="store_true",
                        default=config.MIXED_PRECISION)
    parser.add_argument("--seed", type=int, default=None,
                        help="Override config.RANDOM_SEED for this run.")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    config.ensure_dirs()
    config.describe()
    configure_runtime(args.mixed_precision)

    if not config.SPLIT_MANIFEST.exists():
        raise SystemExit("Split manifest missing. Run:\n"
                         "  python data_loader.py --audit --build")

    session_start = time.perf_counter()
    records: List[dict] = []
    failures: List[str] = []

    for slug in args.models:
        target = config.SAVED_MODELS_DIR / f"{slug}.keras"
        if args.skip_existing and target.exists():
            print(f"[train] Skipping {config.display_name(slug)} "
                  f"(already trained).")
            continue
        try:
            records.append(train_one(slug, args))
            _write_summary(records)          # checkpoint the table after each
        except tf.errors.ResourceExhaustedError:
            failures.append(slug)
            print(f"\n[train] OOM on {config.display_name(slug)}. Retry with:")
            print(f"    python train.py --models {slug} --batch-size "
                  f"{max(8, args.batch_size // 2)} --skip-existing\n")
            keras.backend.clear_session()
            gc.collect()
        except KeyboardInterrupt:
            print("\n[train] Interrupted. Finished models are already on Drive.")
            break
        except Exception:  # noqa: BLE001
            failures.append(slug)
            print(f"\n[train] {config.display_name(slug)} FAILED - continuing "
                  f"with the remaining models.")
            traceback.print_exc()
            keras.backend.clear_session()
            gc.collect()

    _write_summary(records)
    total_min = (time.perf_counter() - session_start) / 60
    print(f"\n[train] Session finished in {total_min:.1f} min. "
          f"Trained: {len(records)}  Failed: {len(failures)}")
    if failures:
        print(f"[train] Retry: python train.py --models "
              f"{' '.join(failures)} --skip-existing")
    trained = [s for s in config.MODEL_SLUGS
               if (config.SAVED_MODELS_DIR / f"{s}.keras").exists()]
    print(f"[train] Models on disk: {', '.join(trained) or 'none'}")
    print("\nNext step:  python evaluate.py")


if __name__ == "__main__":
    main()
