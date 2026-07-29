"""
quantize.py
===========
Converts the selected best model to TensorFlow Lite and measures what
quantization costs, producing Table 4:

    FP32 (Keras)  ->  TFLite FP32  ->  TFLite INT8 (post-training)

The INT8 path uses a representative dataset drawn from the **validation** split,
never the test split - calibrating on test data would leak it and invalidate the
reported accuracy drop.

Because ImageNet normalisation is a layer inside the model, the TFLite files
accept raw 0-255 pixels: nothing has to be re-implemented on the Raspberry Pi.

Usage
-----
    python quantize.py                        # uses results/best_model.json
    python quantize.py --model mobilenetv2
    python quantize.py --threads 1 --limit 300
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, recall_score
from tensorflow import keras

import config
import data_loader

# TF moved the interpreter to a standalone package in recent releases.
try:  # pragma: no cover
    from ai_edge_litert.interpreter import Interpreter  # type: ignore
except ImportError:  # pragma: no cover
    Interpreter = tf.lite.Interpreter


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
def hide_gpu() -> None:
    try:
        tf.config.set_visible_devices([], "GPU")
        print("[quant] GPU hidden - TFLite benchmarking is CPU-only anyway.")
    except RuntimeError:
        pass


def resolve_model_slug(explicit: str | None) -> str:
    if explicit:
        return explicit
    if not config.BEST_MODEL_JSON.exists():
        raise SystemExit("results/best_model.json not found. Run evaluate.py "
                         "first, or pass --model <slug>.")
    with open(config.BEST_MODEL_JSON, encoding="utf-8") as fh:
        payload = json.load(fh)
    slug = payload["best_by_macro_f1"]["slug"]
    print(f"[quant] Selected best model from evaluate.py: "
          f"{config.display_name(slug)}")
    return slug


def file_size_mb(path: Path) -> float:
    return round(path.stat().st_size / (1024 ** 2), 3)


def export_saved_model(model: keras.Model, slug: str) -> Path:
    """Keras 3 needs `model.export()`; older tf.keras uses `tf.saved_model.save`."""
    out_dir = config.EXPORT_DIR / slug
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        model.export(str(out_dir))          # Keras 3
    except AttributeError:
        tf.saved_model.save(model, str(out_dir))   # tf.keras 2.x
    return out_dir


# --------------------------------------------------------------------------- #
# Conversion
# --------------------------------------------------------------------------- #
def _converter_for(model: keras.Model, slug: str) -> tf.lite.TFLiteConverter:
    """SavedModel route first: it is the reliable path under Keras 3."""
    try:
        return tf.lite.TFLiteConverter.from_saved_model(
            str(export_saved_model(model, slug)))
    except Exception as exc:  # noqa: BLE001
        print(f"[quant] SavedModel export failed ({exc}); "
              f"falling back to from_keras_model.")
        return tf.lite.TFLiteConverter.from_keras_model(model)


def convert_fp32(model: keras.Model, slug: str) -> Path:
    converter = _converter_for(model, slug)
    tflite_bytes = converter.convert()
    out = config.TFLITE_DIR / f"{slug}_fp32.tflite"
    out.write_bytes(tflite_bytes)
    print(f"[quant] TFLite FP32 -> {out} ({file_size_mb(out)} MB)")
    return out


def make_representative_dataset(n_samples: int = config.REPRESENTATIVE_SAMPLES):
    """
    Yields single, unaugmented VALIDATION images in the model's input domain
    (float32, 0-255), which is what the calibration pass needs to estimate
    activation ranges.
    """
    ds = data_loader.make_dataset("val", batch_size=1, shuffle=False,
                                  augment=False, cache=False)

    def generator():
        for i, (images, _) in enumerate(ds):
            if i >= n_samples:
                break
            yield [tf.cast(images, tf.float32).numpy()]

    return generator


def convert_int8(model: keras.Model, slug: str,
                 n_samples: int = config.REPRESENTATIVE_SAMPLES) -> Path:
    converter = _converter_for(model, slug)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = make_representative_dataset(n_samples)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    if config.INT8_INPUT_DTYPE == "uint8":
        # Full-integer I/O: exactly what a Pi Zero 2 W wants.
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.float32
    else:
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32

    print(f"[quant] Calibrating INT8 on {n_samples} validation images...")
    tflite_bytes = converter.convert()
    out = config.TFLITE_DIR / f"{slug}_int8.tflite"
    out.write_bytes(tflite_bytes)
    print(f"[quant] TFLite INT8 -> {out} ({file_size_mb(out)} MB)")
    return out


# --------------------------------------------------------------------------- #
# TFLite inference
# --------------------------------------------------------------------------- #
def _quantize_input(arr: np.ndarray, detail: dict) -> np.ndarray:
    dtype = detail["dtype"]
    if dtype in (np.uint8, np.int8):
        scale, zero_point = detail["quantization"]
        scale = scale or 1.0
        info = np.iinfo(dtype)
        arr = np.round(arr / scale + zero_point)
        return np.clip(arr, info.min, info.max).astype(dtype)
    return arr.astype(np.float32)


def _dequantize_output(arr: np.ndarray, detail: dict) -> np.ndarray:
    if detail["dtype"] in (np.uint8, np.int8):
        scale, zero_point = detail["quantization"]
        return (arr.astype(np.float32) - zero_point) * (scale or 1.0)
    return arr.astype(np.float32)


def run_tflite(tflite_path: Path,
               limit: int | None = None,
               threads: int = config.TFLITE_NUM_THREADS
               ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Run the whole test set through a .tflite file, timing `invoke()` only."""
    interpreter = Interpreter(model_path=str(tflite_path), num_threads=threads)
    interpreter.allocate_tensors()
    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]

    ds = data_loader.make_dataset("test", batch_size=1, shuffle=False,
                                  augment=False)
    y_true, y_pred, timings = [], [], []

    for i, (images, labels) in enumerate(ds):
        if limit is not None and i >= limit:
            break
        x = _quantize_input(images.numpy(), in_detail)
        interpreter.set_tensor(in_detail["index"], x)

        t0 = time.perf_counter()
        interpreter.invoke()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        probs = _dequantize_output(
            interpreter.get_tensor(out_detail["index"]), out_detail)
        y_true.append(int(labels.numpy()[0]))
        y_pred.append(int(np.argmax(probs, axis=-1)[0]))
        if i >= config.LATENCY_WARMUP_RUNS:      # discard warm-up invocations
            timings.append(elapsed_ms)

    arr = np.asarray(timings) if timings else np.asarray([float("nan")])
    stats = {
        "inference_ms_mean": round(float(np.nanmean(arr)), 3),
        "inference_ms_std": round(float(np.nanstd(arr)), 3),
        "inference_ms_p95": round(float(np.nanpercentile(arr, 95)), 3),
    }
    return np.asarray(y_true), np.asarray(y_pred), stats


def run_keras(model: keras.Model, limit: int | None = None
              ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Baseline FP32 pass under identical batch-1 CPU conditions."""
    ds = data_loader.make_dataset("test", batch_size=1, shuffle=False,
                                  augment=False)
    infer = tf.function(
        lambda x: model(x, training=False),
        input_signature=[tf.TensorSpec([1, *config.INPUT_SHAPE], tf.float32)],
    )
    y_true, y_pred, timings = [], [], []
    for i, (images, labels) in enumerate(ds):
        if limit is not None and i >= limit:
            break
        t0 = time.perf_counter()
        probs = infer(images).numpy()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        y_true.append(int(labels.numpy()[0]))
        y_pred.append(int(np.argmax(probs, axis=-1)[0]))
        if i >= config.LATENCY_WARMUP_RUNS:
            timings.append(elapsed_ms)

    arr = np.asarray(timings) if timings else np.asarray([float("nan")])
    return (np.asarray(y_true), np.asarray(y_pred), {
        "inference_ms_mean": round(float(np.nanmean(arr)), 3),
        "inference_ms_std": round(float(np.nanstd(arr)), 3),
        "inference_ms_p95": round(float(np.nanpercentile(arr, 95)), 3),
    })


def score(y_true: np.ndarray, y_pred: np.ndarray,
          class_names: List[str]) -> Dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 5),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro",
                                         zero_division=0)), 5),
        "per_class_recall": [
            round(float(v), 5) for v in recall_score(
                y_true, y_pred, average=None,
                labels=list(range(len(class_names))), zero_division=0)
        ],
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def write_table4(rows: List[dict], class_names: List[str], slug: str) -> None:
    baseline_acc = rows[0]["accuracy"]
    baseline_f1 = rows[0]["macro_f1"]
    baseline_size = rows[0]["size_mb"]

    with open(config.QUANTIZATION_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Model Format", "Size (MB)", "Compression",
                         "Inference Time (ms)", "Accuracy", "Macro-F1",
                         "Accuracy Drop (pp)", "Macro-F1 Drop (pp)"])
        for r in rows:
            writer.writerow([
                r["format"], r["size_mb"],
                f"{baseline_size / r['size_mb']:.2f}x" if r["size_mb"] else "-",
                r["inference_ms_mean"], r["accuracy"], r["macro_f1"],
                round((baseline_acc - r["accuracy"]) * 100, 3),
                round((baseline_f1 - r["macro_f1"]) * 100, 3),
            ])
    print(f"\n[quant] Table 4 -> {config.QUANTIZATION_CSV}")

    detail = {"model": config.display_name(slug), "class_names": class_names,
              "rows": rows}
    with open(config.RESULTS_DIR / f"quantization_{slug}.json", "w",
              encoding="utf-8") as fh:
        json.dump(detail, fh, indent=2)

    print("\n--- Table 4: Quantization Results ---")
    header = f"{'Format':<16}{'Size(MB)':>10}{'ms':>10}{'Acc':>10}{'MacroF1':>10}{'AccDrop(pp)':>14}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['format']:<16}{r['size_mb']:>10.3f}"
              f"{r['inference_ms_mean']:>10.2f}{r['accuracy']:>10.4f}"
              f"{r['macro_f1']:>10.4f}"
              f"{(baseline_acc - r['accuracy']) * 100:>14.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TFLite conversion and INT8 post-training quantization.")
    parser.add_argument("--model", default=None, choices=config.MODEL_SLUGS,
                        help="Override the model chosen by evaluate.py.")
    parser.add_argument("--samples", type=int,
                        default=config.REPRESENTATIVE_SAMPLES,
                        help="Representative (calibration) images from val.")
    parser.add_argument("--threads", type=int, default=config.TFLITE_NUM_THREADS)
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N test images (smoke test).")
    parser.add_argument("--skip-keras-baseline", action="store_true",
                        help="Reuse evaluate.py numbers instead of re-timing FP32.")
    args = parser.parse_args()

    config.ensure_dirs()
    hide_gpu()

    slug = resolve_model_slug(args.model)
    model_path = config.SAVED_MODELS_DIR / f"{slug}.keras"
    if not model_path.exists():
        raise SystemExit(f"{model_path} not found. Train it first.")

    class_names, _ = data_loader.load_manifest()
    model = keras.models.load_model(model_path, compile=False)

    fp32_path = convert_fp32(model, slug)
    int8_path = convert_int8(model, slug, args.samples)

    rows: List[dict] = []

    print("\n[quant] Benchmarking Keras FP32 baseline (batch = 1, CPU)...")
    if args.skip_keras_baseline and config.RESULTS_DIR.joinpath(
            "evaluation_full.json").exists():
        with open(config.RESULTS_DIR / "evaluation_full.json", encoding="utf-8") as fh:
            prev = {r["slug"]: r for r in json.load(fh)}[slug]
        rows.append({"format": "FP32 (Keras)", "size_mb": file_size_mb(model_path),
                     "accuracy": prev["accuracy"], "macro_f1": prev["macro_f1"],
                     "inference_ms_mean": prev["inference_ms_mean"],
                     "inference_ms_std": prev["inference_ms_std"]})
    else:
        y_true, y_pred, stats = run_keras(model, limit=args.limit)
        metrics = score(y_true, y_pred, class_names)
        rows.append({"format": "FP32 (Keras)", "size_mb": file_size_mb(model_path),
                     **{k: v for k, v in metrics.items() if k != "per_class_recall"},
                     "per_class_recall": metrics["per_class_recall"], **stats})

    del model
    keras.backend.clear_session()

    for label, path in [("TFLite FP32", fp32_path), ("TFLite INT8", int8_path)]:
        print(f"\n[quant] Benchmarking {label} ({args.threads} thread(s))...")
        y_true, y_pred, stats = run_tflite(path, limit=args.limit,
                                           threads=args.threads)
        metrics = score(y_true, y_pred, class_names)
        rows.append({"format": label, "size_mb": file_size_mb(path),
                     **{k: v for k, v in metrics.items() if k != "per_class_recall"},
                     "per_class_recall": metrics["per_class_recall"], **stats})

    write_table4(rows, class_names, slug)
    shutil.rmtree(config.EXPORT_DIR, ignore_errors=True)
    print("\n[quant] Done. Deployable artefacts are in tflite_models/.")


if __name__ == "__main__":
    main()
