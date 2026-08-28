"""
train.py  (Kaggle / Colab)
==========================
Two-stage transfer learning: a frozen-backbone head stage, then a fine-tune
stage on the top `FINE_TUNE_UNFREEZE_RATIO` of the backbone under a cosine
schedule with linear warmup.

Loss and label format
---------------------
`CategoricalCrossentropy(label_smoothing=...)` needs one-hot targets, so the
training and validation datasets are built with `label_mode="one_hot"` and the
metric is `CategoricalAccuracy`. `evaluate.py` and `quantize.py` keep integer
labels; nothing outside this file sees one-hot targets.

Class weighting, when enabled, arrives as a third element in each dataset
element rather than through `model.fit(class_weight=...)`, which cannot be
combined with one-hot targets. It is verified against the intended weights
before a single step runs.

Stage 2
-------
The optimizer is rebuilt from scratch after unfreezing. Reusing the stage-1
Adam would carry stale moment estimates for the head and would have no slots
at all for the newly trainable backbone variables.

The cosine schedule is defined in *optimizer steps*, not epochs:
`warmup_steps + decay_steps == steps_per_epoch * fine_tune_epochs`. Both
numbers are printed at stage-2 start. Because a schedule object hides the
learning rate from Keras' own logging, an explicit callback records the
effective learning rate into `history.csv` every epoch.

`ReduceLROnPlateau` runs in stage 1 only. It mutates `optimizer.learning_rate`,
which is not a valid operation once that attribute is a schedule object.

Inference export
----------------
`model.keras` is the trained checkpoint and carries Adam state. Every reported
size and latency number must come from `model_inference.keras`, which is
rebuilt from the architecture config, given the trained weights, and never
compiled.

This is not cosmetic. `clone_model()` inherits the compile config, so the
obvious clone-and-save produces a file that still contains optimizer state and
is ~2.7x larger than it should be. The export is therefore verified rather than
assumed: the reloaded file is compared against the trained model on a real
batch and must agree to 1e-5, and it must come back with no optimizer attached.

Usage
-----
    python train.py --models mobilenetv2 --seed 42
    python train.py --models mobilenetv2 --seed 1337 --class-weight
    python train.py --models mobilenetv2 --unfreeze-ratio 0.5 --fine-tune-lr 5e-5
"""

from __future__ import annotations

import argparse
import csv
import gc
import inspect
import json
import math
import subprocess
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402
from tensorflow.keras import layers  # noqa: E402
from tensorflow.keras.applications import (  # noqa: E402
    DenseNet121,
    EfficientNetB0,
    MobileNetV2,
    MobileNetV3Small,
)

import config  # noqa: E402
import data_loader  # noqa: E402

BASE_BUILDERS: Dict[str, Callable[..., keras.Model]] = {
    "mobilenetv2": MobileNetV2,
    "mobilenetv3_small": MobileNetV3Small,
    "efficientnetb0": EfficientNetB0,
    "densenet121": DenseNet121,
}

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_VAR = [0.229 ** 2, 0.224 ** 2, 0.225 ** 2]


class NaNLossError(RuntimeError):
    """Raised when the loss goes non-finite during the first epoch."""


# --------------------------------------------------------------------------- #
# Runtime setup
# --------------------------------------------------------------------------- #
def set_global_seeds(seed: int = config.RANDOM_SEED) -> None:
    """
    Seed Python, NumPy and TensorFlow in one call.

    Op-level determinism is deliberately NOT enabled: it would suppress the
    GPU non-determinism the multi-seed protocol is designed to measure, and it
    is far too slow for a 40-epoch fine-tune.
    """
    keras.utils.set_random_seed(seed)


def gpu_details() -> dict:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return {"available": False, "name": None, "compute_capability": None,
                "count": 0}
    name, capability = "unknown", None
    try:
        details = tf.config.experimental.get_device_details(gpus[0])
        name = details.get("device_name", "unknown")
        raw = details.get("compute_capability")
        if raw:
            capability = float(f"{raw[0]}.{raw[1]}")
    except Exception:  # noqa: BLE001
        pass
    return {"available": True, "name": name, "compute_capability": capability,
            "count": len(gpus)}


def configure_runtime(request_mixed_precision: bool) -> bool:
    """
    Enable memory growth and decide whether mixed precision is actually worth
    it. Returns whether it was enabled.

    float16 pays off on tensor cores, which arrive at compute capability 7.0.
    A P100 is 6.0: there it buys close to nothing and adds overflow risk, so it
    is refused with a reason rather than silently accepted.
    """
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    info = gpu_details()
    print(f"[train] TensorFlow {tf.__version__} / Keras {keras.__version__}")
    if info["available"]:
        print(f"[train] GPU: {info['name']} "
              f"(compute {info['compute_capability']}, "
              f"{info['count']} device(s)), memory growth on")
    else:
        print("[train] WARNING: no GPU visible. On Kaggle: Settings > "
              "Accelerator > GPU P100.")

    if not request_mixed_precision:
        keras.mixed_precision.set_global_policy("float32")
        return False

    capability = info["compute_capability"]
    if not info["available"]:
        print("[train] Mixed precision requested but no GPU is visible; "
              "staying in float32.")
        keras.mixed_precision.set_global_policy("float32")
        return False
    if capability is not None and capability < config.MIXED_PRECISION_MIN_COMPUTE:
        print(f"[train] Mixed precision requested, but compute capability "
              f"{capability} < {config.MIXED_PRECISION_MIN_COMPUTE}: this GPU "
              f"has no tensor cores, so float16 would add overflow risk for "
              f"little speed. Staying in float32.")
        print("[train] Override with --mixed-precision on a T4 or newer.")
        keras.mixed_precision.set_global_policy("float32")
        return False

    keras.mixed_precision.set_global_policy("mixed_float16")
    print("[train] Mixed precision (float16) ENABLED. The output Dense stays "
          "float32; the first epoch is watched for non-finite loss.")
    return True


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
    """
    Raw 0-255 in, softmax out. Augmentation is NOT part of the graph: it lives
    in the tf.data pipeline so the exported model carries no random ops onto
    the Pi.
    """
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


def compile_model(model: keras.Model, learning_rate) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.CategoricalCrossentropy(
            label_smoothing=config.LABEL_SMOOTHING),
        metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")],
    )


# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #
class LearningRateLogger(keras.callbacks.Callback):
    """
    Record the effective learning rate into the epoch logs.

    Keras logs the optimizer's `learning_rate` attribute, which under a
    schedule is the schedule object rather than a number, so without this the
    cosine curve leaves no trace in history.csv.
    """

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        optimizer = self.model.optimizer
        rate = optimizer.learning_rate
        try:
            if isinstance(rate, keras.optimizers.schedules.LearningRateSchedule):
                value = float(keras.ops.convert_to_numpy(
                    rate(optimizer.iterations)))
            else:
                value = float(keras.ops.convert_to_numpy(rate))
        except Exception:  # noqa: BLE001
            value = float("nan")
        logs["lr"] = value


class NaNLossGuard(keras.callbacks.Callback):
    """
    Abort on a non-finite loss during the first epoch.

    Under mixed precision this almost always means float16 overflow, and it is
    worth failing in the first minute rather than after a 40-epoch run of NaNs.
    """

    def __init__(self, mixed_precision: bool):
        super().__init__()
        self.mixed_precision = mixed_precision
        self._epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch = epoch

    def _check(self, logs, where: str):
        loss = (logs or {}).get("loss")
        if loss is None or np.isfinite(loss):
            return
        raise NaNLossError(
            f"Loss became {loss} at {where}. "
            + ("Mixed precision (float16) is the usual cause."
               if self.mixed_precision else
               "Mixed precision is off, so this is not a float16 overflow.")
        )

    def on_train_batch_end(self, batch, logs=None):
        if self._epoch == 0:
            self._check(logs, f"epoch 1 batch {batch}")

    def on_epoch_end(self, epoch, logs=None):
        self._check(logs, f"end of epoch {epoch + 1}")


def _callbacks(run_dir: Path, slug: str, stage: str,
               mixed_precision: bool,
               use_plateau: bool) -> List[keras.callbacks.Callback]:
    ckpt = config.CHECKPOINT_DIR / f"{slug}_best.keras"
    stack: List[keras.callbacks.Callback] = [
        NaNLossGuard(mixed_precision),
        keras.callbacks.EarlyStopping(
            monitor=config.MONITOR_METRIC, mode=config.MONITOR_MODE,
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt), monitor=config.MONITOR_METRIC,
            mode=config.MONITOR_MODE, save_best_only=True, verbose=0),
    ]
    if use_plateau:
        stack.append(keras.callbacks.ReduceLROnPlateau(
            monitor=config.MONITOR_METRIC, mode=config.MONITOR_MODE,
            factor=config.REDUCE_LR_FACTOR, patience=config.REDUCE_LR_PATIENCE,
            min_lr=config.MIN_LR, verbose=1))
    stack.append(LearningRateLogger())
    stack.append(keras.callbacks.CSVLogger(
        str(run_dir / f"history_{stage}.csv")))
    return stack


# --------------------------------------------------------------------------- #
# Fine-tuning
# --------------------------------------------------------------------------- #
def _unfreeze_top(model: keras.Model, ratio: float) -> int:
    backbone = model.get_layer(model._backbone_name)  # noqa: SLF001
    backbone.trainable = True
    cutoff = int(len(backbone.layers) * (1.0 - ratio))
    for i, layer in enumerate(backbone.layers):
        layer.trainable = (i >= cutoff) and not isinstance(
            layer, layers.BatchNormalization)
    trainable = sum(1 for l in backbone.layers if l.trainable)
    print(f"[train] Fine-tuning: {trainable}/{len(backbone.layers)} backbone "
          f"layers unfrozen (BatchNorm kept frozen).")
    return trainable


def build_fine_tune_schedule(steps_per_epoch: int, fine_tune_epochs: int,
                             peak_lr: float, warmup_epochs: int) -> tuple:
    """
    Cosine decay with linear warmup, expressed in optimizer steps.

    Keras' CosineDecay treats warmup_steps and decay_steps as consecutive, so
    they are split such that their sum spans exactly the fine-tune run. Passing
    epoch counts here instead of step counts would collapse the learning rate
    to zero within the first epoch.
    """
    warmup_epochs = max(0, min(warmup_epochs, max(fine_tune_epochs - 1, 0)))
    warmup_steps = steps_per_epoch * warmup_epochs
    decay_steps = steps_per_epoch * max(fine_tune_epochs - warmup_epochs, 1)
    total_steps = warmup_steps + decay_steps

    schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=peak_lr / 100.0 if warmup_steps else peak_lr,
        decay_steps=decay_steps,
        warmup_target=peak_lr if warmup_steps else None,
        warmup_steps=warmup_steps,
        alpha=0.0)

    print(f"[train] Cosine schedule (in optimizer STEPS, not epochs):")
    print(f"[train]   steps_per_epoch   = {steps_per_epoch}")
    print(f"[train]   warmup_epochs     = {warmup_epochs}")
    print(f"[train]   warmup_steps      = {warmup_steps}")
    print(f"[train]   decay_steps       = {decay_steps}")
    print(f"[train]   total_steps       = {total_steps} "
          f"(= {steps_per_epoch} x {fine_tune_epochs} epochs)")
    print(f"[train]   peak_lr           = {peak_lr:g}")
    for probe in (0, warmup_steps, total_steps // 2, max(total_steps - 1, 0)):
        print(f"[train]   lr @ step {probe:<6} = "
              f"{float(keras.ops.convert_to_numpy(schedule(probe))):.3e}")
    return schedule, warmup_steps, decay_steps, total_steps


def _merge_histories(first: dict, second: dict | None) -> dict:
    if not second:
        return first
    merged = {k: list(v) for k, v in first.items()}
    for k, v in second.items():
        merged.setdefault(k, []).extend(list(v))
    return merged


# --------------------------------------------------------------------------- #
# Inference-only export
# --------------------------------------------------------------------------- #
def export_inference_model(model: keras.Model, out_path: Path) -> keras.Model:
    """
    Save an uncompiled copy carrying weights but no optimizer state.

    Rebuilt from the architecture config rather than with `clone_model`, which
    inherits the compile config and would write optimizer variables straight
    back into the file.
    """
    clean = model.__class__.from_config(model.get_config())
    clean.set_weights(model.get_weights())
    clean.save(out_path)
    return clean


def verify_inference_export(trained: keras.Model, out_path: Path,
                            batch: np.ndarray,
                            tolerance: float = 1e-5) -> dict:
    """
    Prove the exported file is the trained model.

    `set_weights` is easy to omit, and a randomly initialised clone saves and
    reloads perfectly happily, so equivalence is measured on a real batch
    rather than assumed. The reloaded model must also come back with no
    optimizer attached.
    """
    reloaded = keras.models.load_model(out_path, compile=False)

    expected = np.asarray(trained(batch, training=False))
    actual = np.asarray(reloaded(batch, training=False))
    max_diff = float(np.max(np.abs(expected - actual)))

    optimizer = getattr(reloaded, "optimizer", None)
    has_optimizer = optimizer is not None

    print(f"[train] Inference export check on {len(batch)} image(s): "
          f"max |diff| = {max_diff:.3e} (tolerance {tolerance:g})")
    print(f"[train] Reloaded model layers: "
          f"{[l.name for l in reloaded.layers]}")
    print(f"[train] Reloaded optimizer attached: {has_optimizer}")

    if max_diff >= tolerance:
        raise AssertionError(
            f"Exported model disagrees with the trained model "
            f"(max |diff| {max_diff:.3e} >= {tolerance:g}). "
            f"The export is not the model that was trained."
        )
    if has_optimizer:
        raise AssertionError(
            "Exported model came back with an optimizer attached, so it still "
            "carries optimizer state and its size is not a valid Table 3 "
            "number."
        )

    del reloaded
    return {"max_abs_diff": max_diff, "images_checked": int(len(batch))}


# --------------------------------------------------------------------------- #
# Run artefacts
# --------------------------------------------------------------------------- #
def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(config.PROJECT_ROOT),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _split_hash() -> str:
    try:
        return config.SPLIT_HASH_TXT.read_text(
            encoding="utf-8").splitlines()[0].strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def write_history_csv(history: dict, run_dir: Path) -> Path:
    path = run_dir / "history.csv"
    keys = sorted(history.keys())
    n = max(len(v) for v in history.values())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch"] + keys)
        for i in range(n):
            writer.writerow([i + 1] + [
                history[k][i] if i < len(history[k]) else "" for k in keys])
    return path


def write_config_snapshot(run_dir: Path, payload: dict) -> Path:
    path = run_dir / "config_snapshot.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return path


def plot_training_curves(history: dict, run_dir: Path, title: str) -> Path:
    epochs = range(1, len(history["loss"]) + 1)
    has_lr = "lr" in history
    fig, axes = plt.subplots(1, 3 if has_lr else 2,
                             figsize=(15 if has_lr else 10, 4.2))

    axes[0].plot(epochs, history["accuracy"], marker="o", ms=3, label="Train")
    axes[0].plot(epochs, history["val_accuracy"], marker="s", ms=3,
                 label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Accuracy", title=f"{title} - accuracy")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history["loss"], marker="o", ms=3, label="Train")
    axes[1].plot(epochs, history["val_loss"], marker="s", ms=3,
                 label="Validation")
    axes[1].set(xlabel="Epoch", ylabel="Loss", title=f"{title} - loss")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    if has_lr:
        axes[2].plot(epochs, history["lr"], marker=".", ms=3)
        axes[2].set(xlabel="Epoch", ylabel="Learning rate",
                    title=f"{title} - effective LR", yscale="log")
        axes[2].grid(alpha=0.3)

    fig.tight_layout()
    path = run_dir / "training_curves.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_one(slug: str, args: argparse.Namespace,
              mixed_precision: bool) -> dict:
    seed = args.seed if args.seed is not None else config.RANDOM_SEED
    run_name = args.run_name or f"{slug}_seed{seed}"
    run_dir = config.RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 72)
    print(f"  Training {config.display_name(slug)}  seed={seed}  "
          f"run={run_name}   [{datetime.now():%H:%M:%S}]")
    print("=" * 72)

    keras.backend.clear_session()
    gc.collect()
    set_global_seeds(seed)

    class_names, _ = data_loader.load_manifest()
    num_classes = len(class_names)

    class_weights = (data_loader.compute_class_weights()
                     if args.class_weight else None)
    if class_weights:
        print(f"[train] Class weighting ENABLED: "
              f"{ {k: round(v, 4) for k, v in sorted(class_weights.items())} }")
    else:
        print("[train] Class weighting disabled (unweighted is the headline "
              "configuration).")

    train_ds = data_loader.make_dataset(
        "train", batch_size=args.batch_size, label_mode="one_hot",
        class_weights=class_weights, seed=seed)
    val_ds = data_loader.make_dataset(
        "val", batch_size=args.batch_size, label_mode="one_hot", seed=seed)

    if class_weights:
        data_loader.assert_sample_weights(
            train_ds, class_weights, num_classes=num_classes,
            label_mode="one_hot", batches=args.weight_check_batches)

    n_train = data_loader.split_size("train")
    steps_per_epoch = math.ceil(n_train / args.batch_size)

    model = build_model(slug, num_classes=num_classes)
    compile_model(model, args.lr)
    if args.summary:
        model.summary()

    start = time.perf_counter()
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=args.head_epochs,
        callbacks=_callbacks(run_dir, slug, "head", mixed_precision,
                             use_plateau=True),
        verbose=1)
    hist = dict(history.history)
    epochs_run = len(hist["loss"])

    schedule_info = None
    ft_hist = None
    if args.fine_tune_epochs > 0:
        print("\n" + "-" * 72)
        print(f"  Stage 2: fine-tune  [{datetime.now():%H:%M:%S}]")
        print("-" * 72)
        unfrozen = _unfreeze_top(model, args.unfreeze_ratio)
        schedule, warmup_steps, decay_steps, total_steps = (
            build_fine_tune_schedule(steps_per_epoch, args.fine_tune_epochs,
                                     args.fine_tune_lr,
                                     config.FINE_TUNE_WARMUP_EPOCHS))
        compile_model(model, schedule)
        print("[train] Optimizer rebuilt from scratch for stage 2 (stage-1 "
              "Adam moments discarded, slots allocated for the newly "
              "trainable backbone variables).")
        schedule_info = {
            "steps_per_epoch": steps_per_epoch,
            "warmup_steps": warmup_steps,
            "decay_steps": decay_steps,
            "total_steps": total_steps,
            "warmup_epochs": config.FINE_TUNE_WARMUP_EPOCHS,
            "peak_lr": args.fine_tune_lr,
            "backbone_layers_unfrozen": unfrozen,
        }
        ft = model.fit(
            train_ds, validation_data=val_ds,
            epochs=epochs_run + args.fine_tune_epochs,
            initial_epoch=epochs_run,
            callbacks=_callbacks(run_dir, slug, "finetune", mixed_precision,
                                 use_plateau=False),
            verbose=1)
        ft_hist = dict(ft.history)

    elapsed = time.perf_counter() - start
    hist = _merge_histories(hist, ft_hist)

    trained_path = run_dir / "model.keras"
    model.save(trained_path)

    inference_path = run_dir / "model_inference.keras"
    export_inference_model(model, inference_path)

    probe = next(iter(data_loader.make_dataset(
        "test", batch_size=8, shuffle=False, augment=False, cache=False)))[0]
    export_check = verify_inference_export(model, inference_path,
                                           np.asarray(probe))

    trained_mb = round(trained_path.stat().st_size / (1024 ** 2), 3)
    inference_mb = round(inference_path.stat().st_size / (1024 ** 2), 3)
    print(f"[train] model.keras           : {trained_mb:>8.3f} MB  "
          f"(trained checkpoint, carries Adam optimizer state)")
    print(f"[train] model_inference.keras : {inference_mb:>8.3f} MB  "
          f"(REPORT THIS ONE - weights only, no optimizer)")

    best_idx = int(np.argmax(hist["val_accuracy"]))
    record = {
        "model": config.display_name(slug),
        "slug": slug,
        "run_name": run_name,
        "seed": seed,
        "epochs_run": len(hist["loss"]),
        "best_epoch": best_idx + 1,
        "best_val_accuracy": round(float(hist["val_accuracy"][best_idx]), 5),
        "best_val_loss": round(float(hist["val_loss"][best_idx]), 5),
        "train_time_sec": round(elapsed, 1),
        "total_params": int(model.count_params()),
        "batch_size": args.batch_size,
        "class_weight": bool(args.class_weight),
        "mixed_precision": mixed_precision,
        "model_size_mb": trained_mb,
        "inference_size_mb": inference_mb,
        "export_max_abs_diff": export_check["max_abs_diff"],
        "saved_to": str(inference_path),
    }

    write_history_csv(hist, run_dir)
    plot_training_curves(hist, run_dir, config.display_name(slug))
    write_config_snapshot(run_dir, {
        "run_name": run_name,
        "slug": slug,
        "seed": seed,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_sha(),
        "split_hash": _split_hash(),
        "class_names": class_names,
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "gpu": gpu_details(),
        "mixed_precision_requested": args.mixed_precision,
        "mixed_precision_active": mixed_precision,
        "batch_size": args.batch_size,
        "head_epochs": args.head_epochs,
        "fine_tune_epochs": args.fine_tune_epochs,
        "head_lr": args.lr,
        "fine_tune_lr": args.fine_tune_lr,
        "unfreeze_ratio": args.unfreeze_ratio,
        "label_smoothing": config.LABEL_SMOOTHING,
        "dropout_rate": config.DROPOUT_RATE,
        "monitor": config.MONITOR_METRIC,
        "monitor_mode": config.MONITOR_MODE,
        "early_stopping_patience": config.EARLY_STOPPING_PATIENCE,
        "class_weight_enabled": bool(args.class_weight),
        "class_weights": class_weights,
        "train_images": n_train,
        "steps_per_epoch": steps_per_epoch,
        "fine_tune_schedule": schedule_info,
        "augmentation": {
            "flip_mode": config.AUG_FLIP_MODE,
            "rotation": config.AUG_ROTATION_FACTOR,
            "zoom": config.AUG_ZOOM_FACTOR,
            "translation": config.AUG_TRANSLATION_FACTOR,
            "contrast": config.AUG_CONTRAST_FACTOR,
            "brightness": config.AUG_BRIGHTNESS_FACTOR,
            "value_range": list(config.AUG_VALUE_RANGE),
            "location": "tf.data pipeline, train split only",
        },
        "export_check": export_check,
        "result": record,
    })

    print(f"[train] Run artefacts -> {run_dir}")
    print(f"[train] Best val_accuracy={record['best_val_accuracy']:.4f} "
          f"at epoch {record['best_epoch']} "
          f"({elapsed / 60:.1f} min)")

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
    by_key = {r.get("run_name", r["slug"]): {k: r.get(k, "") for k in fieldnames}
              for r in existing}
    for r in records:
        by_key[r["run_name"]] = {k: str(v) for k, v in r.items()}

    with open(config.TRAINING_SUMMARY_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(by_key.values())
    print(f"[train] Training summary -> {config.TRAINING_SUMMARY_CSV}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train lightweight models on the frozen ASDID split.")
    parser.add_argument("--models", nargs="+", default=["mobilenetv2"],
                        choices=config.MODEL_SLUGS)
    parser.add_argument("--seed", type=int, default=None,
                        help="Weight init and shuffling only; never the split.")
    parser.add_argument("--run-name", default=None,
                        help="Artifact subfolder; defaults to <slug>_seed<N>.")
    parser.add_argument("--head-epochs", type=int, default=config.HEAD_EPOCHS)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Deprecated alias for --head-epochs.")
    parser.add_argument("--fine-tune-epochs", type=int,
                        default=config.FINE_TUNE_EPOCHS)
    parser.add_argument("--unfreeze-ratio", type=float,
                        default=config.FINE_TUNE_UNFREEZE_RATIO)
    parser.add_argument("--lr", type=float, default=config.HEAD_LEARNING_RATE)
    parser.add_argument("--fine-tune-lr", type=float,
                        default=config.FINE_TUNE_LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--class-weight", action="store_true",
                        help="Balanced sample weights; OFF by default.")
    parser.add_argument("--weight-check-batches", type=int, default=4)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction,
                        default=config.MIXED_PRECISION)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    if args.epochs is not None:
        args.head_epochs = args.epochs

    config.ensure_dirs()
    config.describe()
    mixed_precision = configure_runtime(args.mixed_precision)

    if not config.SPLIT_MANIFEST.exists():
        raise SystemExit("Split manifest missing. Run:\n"
                         "  python data_loader.py --audit --build")

    session_start = time.perf_counter()
    records: List[dict] = []
    failures: List[str] = []

    for slug in args.models:
        seed = args.seed if args.seed is not None else config.RANDOM_SEED
        run_name = args.run_name or f"{slug}_seed{seed}"
        if args.skip_existing and (config.RUNS_DIR / run_name /
                                   "model_inference.keras").exists():
            print(f"[train] Skipping {run_name} (already trained).")
            continue
        try:
            records.append(train_one(slug, args, mixed_precision))
            _write_summary(records)
        except NaNLossError as exc:
            print(f"\n[train] {exc}")
            if mixed_precision:
                print("[train] Disabling mixed precision and retrying this "
                      "model once in float32.")
                keras.backend.clear_session()
                gc.collect()
                mixed_precision = configure_runtime(False)
                try:
                    records.append(train_one(slug, args, mixed_precision))
                    _write_summary(records)
                    print("[train] NOTE: this run is float32. Set "
                          "MIXED_PRECISION = False in config.py so later runs "
                          "match it.")
                except Exception:  # noqa: BLE001
                    failures.append(slug)
                    traceback.print_exc()
            else:
                failures.append(slug)
                print("[train] Mixed precision was already off, so this is "
                      "not a float16 problem. Check the learning rate.")
        except tf.errors.ResourceExhaustedError:
            failures.append(slug)
            print(f"\n[train] OOM on {config.display_name(slug)}. Retry with:")
            print(f"    python train.py --models {slug} --batch-size "
                  f"{max(8, args.batch_size // 2)}\n")
            keras.backend.clear_session()
            gc.collect()
        except KeyboardInterrupt:
            print("\n[train] Interrupted. Finished runs are already on disk.")
            break
        except Exception:  # noqa: BLE001
            failures.append(slug)
            print(f"\n[train] {config.display_name(slug)} FAILED.")
            traceback.print_exc()
            keras.backend.clear_session()
            gc.collect()

    _write_summary(records)
    total_min = (time.perf_counter() - session_start) / 60
    print(f"\n[train] Session finished in {total_min:.1f} min. "
          f"Trained: {len(records)}  Failed: {len(failures)}")
    if failures:
        print(f"[train] Retry: python train.py --models {' '.join(failures)}")
    print("\nNext step:  python evaluate.py")


if __name__ == "__main__":
    main()
