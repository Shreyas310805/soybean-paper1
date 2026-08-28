"""
quantize.py
===========
Converts a run's inference model to TensorFlow Lite and measures what
quantization costs, producing Table IV:

    FP32 (Keras)  ->  TFLite FP32  ->  TFLite dynamic  ->  TFLite INT8

Source model
------------
Only `runs/<run_name>/model_inference.keras`, the same weights-only file
`evaluate.py` reports. The trained checkpoint carries optimizer state and is
never converted or measured.

Representative dataset
----------------------
Drawn from the TRAIN split with augmentation OFF and shuffling OFF, seeded at
42. This matters more than it looks: calibration estimates activation ranges,
and if the calibration images had been rotated, brightened or contrast-shifted
the ranges would be wrong for real inference. INT8 accuracy would then drop for
a reason that looks exactly like quantization damage but is a data bug. The
same preprocessing the model applies at training time still runs, because it
lives inside the model graph.

Test data is never used for calibration.

Input/output convention
-----------------------
Unchanged from the run that produced sensible numbers: INT8 takes uint8 input
and returns float32 output, which is what a Pi Zero 2 W wants. The convention
is recorded in every row of the output so the paper can state it rather than
leave a reader to guess.

Latency
-------
Shared with `evaluate.py` through `bench.measure_latency`, at batch size 1 with
the thread count pinned explicitly for the interpreter. TFLite does not default
to the same thread count TensorFlow uses, and Table IV compares them on
adjacent rows.

Usage
-----
    python quantize.py --run mobilenetv2_seed42
    python quantize.py --run mobilenetv2_seed42 --limit 200
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, recall_score
from tensorflow import keras

import bench
import config
import data_loader

try:  # pragma: no cover
    from ai_edge_litert.interpreter import Interpreter  # type: ignore
except ImportError:  # pragma: no cover
    Interpreter = tf.lite.Interpreter

INFERENCE_MODEL_NAME = "model_inference.keras"


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
def hide_gpu() -> None:
    try:
        tf.config.set_visible_devices([], "GPU")
        print("[quant] GPU hidden - TFLite benchmarking is CPU-only anyway.")
    except RuntimeError:
        pass


def resolve_run(explicit: str | None) -> Path:
    if explicit:
        run_dir = config.RUNS_DIR / explicit
        if not run_dir.is_dir():
            raise SystemExit(f"Run directory not found: {run_dir}")
        return run_dir

    candidates = sorted(
        d for d in config.RUNS_DIR.iterdir()
        if d.is_dir() and (d / INFERENCE_MODEL_NAME).exists()
    ) if config.RUNS_DIR.exists() else []
    if not candidates:
        raise SystemExit(
            f"No run under {config.RUNS_DIR} contains {INFERENCE_MODEL_NAME}. "
            f"Pass --run explicitly.")

    best, best_acc = None, -1.0
    for run_dir in candidates:
        path = run_dir / "evaluation.json"
        if not path.exists():
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                acc = json.load(fh)["headline"]["accuracy"]
        except (json.JSONDecodeError, KeyError):
            continue
        if acc > best_acc:
            best, best_acc = run_dir, acc

    chosen = best or candidates[0]
    reason = (f"highest test accuracy {best_acc:.5f}" if best
              else "first run on disk; run evaluate.py to pick by accuracy")
    print(f"[quant] Selected run '{chosen.name}' ({reason}).")
    return chosen


def inference_model_path(run_dir: Path) -> Path:
    path = run_dir / INFERENCE_MODEL_NAME
    if not path.exists():
        raise SystemExit(
            f"{run_dir.name}: {INFERENCE_MODEL_NAME} not found. The trained "
            f"checkpoint is not an acceptable substitute - it carries "
            f"optimizer state. Re-run training for this run.")
    return path


def file_size_mb(path: Path) -> float:
    return round(path.stat().st_size / (1024 ** 2), 3)


def export_saved_model(model: keras.Model, slug: str) -> Path:
    out_dir = config.EXPORT_DIR / slug
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        model.export(str(out_dir))
    except AttributeError:
        tf.saved_model.save(model, str(out_dir))
    return out_dir


# --------------------------------------------------------------------------- #
# Conversion
# --------------------------------------------------------------------------- #
def _converter_for(model: keras.Model, slug: str) -> tf.lite.TFLiteConverter:
    try:
        return tf.lite.TFLiteConverter.from_saved_model(
            str(export_saved_model(model, slug)))
    except Exception as exc:  # noqa: BLE001
        print(f"[quant] SavedModel export failed ({exc}); "
              f"falling back to from_keras_model.")
        return tf.lite.TFLiteConverter.from_keras_model(model)


def make_representative_dataset(n_samples: int = config.REPRESENTATIVE_SAMPLES):
    """
    Unaugmented, unshuffled TRAIN images in the model's input domain.

    augment=False is the load-bearing argument here. Calibrating on rotated or
    brightness-shifted images would estimate activation ranges the model never
    sees at inference, and the resulting accuracy loss would be indistinguish-
    able from genuine quantization damage.
    """
    ds = data_loader.make_dataset("train", batch_size=1, shuffle=False,
                                  augment=False, cache=False,
                                  seed=config.RANDOM_SEED)

    def generator():
        for i, (images, _) in enumerate(ds):
            if i >= n_samples:
                break
            yield [tf.cast(images, tf.float32).numpy()]

    return generator


def convert_fp32(model: keras.Model, slug: str, out_dir: Path) -> Path:
    tflite_bytes = _converter_for(model, slug).convert()
    out = out_dir / f"{slug}_fp32.tflite"
    out.write_bytes(tflite_bytes)
    print(f"[quant] TFLite FP32 -> {out} ({file_size_mb(out)} MB)")
    return out


def convert_dynamic(model: keras.Model, slug: str, out_dir: Path) -> Path:
    converter = _converter_for(model, slug)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_bytes = converter.convert()
    out = out_dir / f"{slug}_dynamic.tflite"
    out.write_bytes(tflite_bytes)
    print(f"[quant] TFLite dynamic-range -> {out} ({file_size_mb(out)} MB)")
    return out


def convert_int8(model: keras.Model, slug: str, out_dir: Path,
                 n_samples: int = config.REPRESENTATIVE_SAMPLES) -> Path:
    converter = _converter_for(model, slug)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = make_representative_dataset(n_samples)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    if config.INT8_INPUT_DTYPE == "uint8":
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.float32
    else:
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32

    print(f"[quant] Calibrating INT8 on {n_samples} UNAUGMENTED train "
          f"images (seed {config.RANDOM_SEED})...")
    tflite_bytes = converter.convert()
    out = out_dir / f"{slug}_int8.tflite"
    out.write_bytes(tflite_bytes)
    print(f"[quant] TFLite INT8 -> {out} ({file_size_mb(out)} MB)")
    return out


# --------------------------------------------------------------------------- #
# Inference
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


def io_convention(in_detail: dict, out_detail: dict) -> str:
    return (f"input={np.dtype(in_detail['dtype']).name}, "
            f"output={np.dtype(out_detail['dtype']).name}")


def score_tflite(tflite_path: Path, class_names: List[str],
                 limit: int | None, threads: int) -> Tuple[dict, str]:
    interpreter = Interpreter(model_path=str(tflite_path), num_threads=threads)
    interpreter.allocate_tensors()
    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]

    ds = data_loader.make_dataset("test", batch_size=1, shuffle=False,
                                  augment=False)
    y_true, y_pred = [], []
    for i, (images, labels) in enumerate(ds):
        if limit is not None and i >= limit:
            break
        interpreter.set_tensor(
            in_detail["index"], _quantize_input(images.numpy(), in_detail))
        interpreter.invoke()
        probs = _dequantize_output(
            interpreter.get_tensor(out_detail["index"]), out_detail)
        y_true.append(int(labels.numpy()[0]))
        y_pred.append(int(np.argmax(probs, axis=-1)[0]))

    return (score(np.asarray(y_true), np.asarray(y_pred), class_names),
            io_convention(in_detail, out_detail))


def bench_tflite(tflite_path: Path, samples: np.ndarray, label: str,
                 threads: int) -> dict:
    interpreter = Interpreter(model_path=str(tflite_path), num_threads=threads)
    interpreter.allocate_tensors()
    in_detail = interpreter.get_input_details()[0]
    index = in_detail["index"]

    prepared = [_quantize_input(img[None, ...], in_detail) for img in samples]

    def infer(sample):
        interpreter.set_tensor(index, sample)
        interpreter.invoke()

    return bench.measure_latency(infer, prepared, label=label, threads=threads)


def score(y_true: np.ndarray, y_pred: np.ndarray,
          class_names: List[str]) -> Dict[str, object]:
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
def write_table4(rows: List[dict], class_names: List[str],
                 run_dir: Path, machine: dict) -> None:
    baseline = rows[0]
    path = config.RESULTS_DIR / "table4.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Model Format", "Size (MB)", "Compression",
                         "Inference Time (ms)", "Accuracy", "Macro-F1",
                         "Accuracy Drop (pp)", "Macro-F1 Drop (pp)",
                         "Median (ms)", "p95 (ms)", "Threads",
                         "I/O Convention", "CPU"])
        for r in rows:
            writer.writerow([
                r["format"], r["size_mb"],
                f"{baseline['size_mb'] / r['size_mb']:.2f}x" if r["size_mb"] else "-",
                r.get("latency_ms_mean", ""), r["accuracy"], r["macro_f1"],
                round((baseline["accuracy"] - r["accuracy"]) * 100, 3),
                round((baseline["macro_f1"] - r["macro_f1"]) * 100, 3),
                r.get("latency_ms_median", ""), r.get("latency_ms_p95", ""),
                r.get("threads", ""), r.get("io_convention", ""),
                r.get("cpu_model", "")])
    print(f"\n[quant] Table IV -> {path}")

    with open(run_dir / "quantization.json", "w", encoding="utf-8") as fh:
        json.dump({"run": run_dir.name, "class_names": class_names,
                   "machine": machine, "rows": rows}, fh, indent=2,
                  default=str)

    print("\n--- Table IV: Quantization Results ---")
    header = (f"{'Format':<17}{'Size(MB)':>10}{'median ms':>11}"
              f"{'Acc':>9}{'MacroF1':>9}{'AccDrop(pp)':>13}  I/O")
    print(header)
    print("-" * (len(header) + 12))
    for r in rows:
        print(f"{r['format']:<17}{r['size_mb']:>10.3f}"
              f"{r.get('latency_ms_median', float('nan')):>11.3f}"
              f"{r['accuracy']:>9.4f}{r['macro_f1']:>9.4f}"
              f"{(baseline['accuracy'] - r['accuracy']) * 100:>13.3f}"
              f"  {r.get('io_convention', '')}")
    print(f"\n  threads={machine['threads']}  cpu={machine['cpu_model']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TFLite conversion and INT8 post-training quantization.")
    parser.add_argument("--run", default=None,
                        help="Run directory name under runs/.")
    parser.add_argument("--samples", type=int,
                        default=config.REPRESENTATIVE_SAMPLES)
    parser.add_argument("--threads", type=int, default=config.BENCH_NUM_THREADS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=None,
                        help="Score only the first N test images (smoke test).")
    args = parser.parse_args()

    config.ensure_dirs()
    hide_gpu()
    threads = bench.pin_threads(args.threads)
    machine = bench.describe_machine(threads)
    print(f"[quant] Machine: {json.dumps(machine)}")

    run_dir = resolve_run(args.run)
    model_path = inference_model_path(run_dir)
    slug = json.loads((run_dir / "config_snapshot.json").read_text(
        encoding="utf-8")).get("slug", run_dir.name) \
        if (run_dir / "config_snapshot.json").exists() else run_dir.name

    class_names, _ = data_loader.load_manifest()
    model = keras.models.load_model(model_path, compile=False)
    out_dir = config.TFLITE_DIR / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    fp32_path = convert_fp32(model, slug, out_dir)
    dyn_path = convert_dynamic(model, slug, out_dir)
    int8_path = convert_int8(model, slug, out_dir, args.samples)

    samples = next(iter(data_loader.make_dataset(
        "test", batch_size=32, shuffle=False, augment=False)))[0].numpy()

    print("\n[quant] Benchmarking Keras FP32 baseline (batch 1, CPU)...")
    ds = data_loader.make_dataset("test", batch_size=args.batch_size,
                                  shuffle=False, augment=False)
    y_true, y_prob = [], []
    for images, labels in ds:
        y_prob.append(model.predict_on_batch(images))
        y_true.append(labels.numpy())
    keras_metrics = score(np.concatenate(y_true),
                          np.concatenate(y_prob).argmax(axis=1), class_names)

    infer = tf.function(
        lambda x: model(x, training=False),
        input_signature=[tf.TensorSpec([1, *config.INPUT_SHAPE], tf.float32)])
    keras_latency = bench.measure_latency(
        infer, [tf.constant(img[None, ...]) for img in samples],
        label="FP32 (Keras)", threads=threads)

    rows: List[dict] = [{
        "format": "FP32 (Keras)", "size_mb": file_size_mb(model_path),
        "io_convention": "input=float32 (0-255), output=float32",
        **keras_metrics, **keras_latency,
    }]

    del model
    keras.backend.clear_session()

    for label, path in [("TFLite FP32", fp32_path),
                        ("TFLite Dynamic", dyn_path),
                        ("TFLite INT8", int8_path)]:
        print(f"\n[quant] Benchmarking {label} ({threads} thread(s))...")
        metrics, convention = score_tflite(path, class_names, args.limit,
                                           threads)
        latency = bench_tflite(path, samples, label, threads)
        rows.append({"format": label, "size_mb": file_size_mb(path),
                     "io_convention": convention, **metrics, **latency})

    write_table4(rows, class_names, run_dir, machine)
    shutil.rmtree(config.EXPORT_DIR, ignore_errors=True)
    print(f"\n[quant] Deployable artefacts -> {out_dir}")


if __name__ == "__main__":
    main()
