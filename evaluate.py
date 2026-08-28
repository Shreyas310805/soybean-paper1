"""
evaluate.py
===========
Evaluates a training run on the frozen, unseen test split and produces the
Table II and Table III rows plus the confusion matrix.

What it loads, and what it refuses to load
------------------------------------------
Only `runs/<run_name>/model_inference.keras`. If that file is absent the run
is skipped with a loud error and a non-zero exit.

It deliberately does NOT fall back to `model.keras`. That file carries Adam
optimizer state and is roughly 2.5x larger, and silently reporting its size
is what produced the wrong 23.343 MB model size in an earlier draft. A
missing export is a broken run, not something to paper over.

Latency
-------
All timings go through `bench.measure_latency` at batch size 1 with the thread
count pinned, so the Keras row here and the TFLite rows in `quantize.py` are
produced by one protocol on one machine. The CPU model string and thread count
are recorded in every output.

The GPU is hidden unless `--gpu` is passed. "CPU inference time" is only
meaningful if the forward pass actually runs on the CPU, and the Raspberry Pi
target makes the CPU number the one that matters.

Usage
-----
    python evaluate.py                          # every run in runs/
    python evaluate.py --runs mobilenetv2_seed42
    python evaluate.py --tta                    # adds a separate TTA row
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from tensorflow import keras  # noqa: E402

import bench  # noqa: E402
import config  # noqa: E402
import data_loader  # noqa: E402

INFERENCE_MODEL_NAME = "model_inference.keras"

TABLE2_COLUMNS = ["Model", "Accuracy", "Precision", "Recall", "Macro-F1"]
TABLE3_COLUMNS = ["Model", "Parameters", "Model Size (MB)",
                  "CPU Inference Time (ms)", "Std (ms)", "Macro-F1"]


class MissingInferenceModel(FileNotFoundError):
    """Raised when a run directory has no inference-only export."""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def hide_gpu() -> None:
    try:
        tf.config.set_visible_devices([], "GPU")
        print("[eval] GPU hidden - all timings are true CPU timings.")
    except RuntimeError:
        print("[eval] WARNING: GPU already initialised; latency may be skewed.")


def discover_runs(names: List[str] | None) -> List[Path]:
    if names:
        return [config.RUNS_DIR / n for n in names]
    if not config.RUNS_DIR.exists():
        return []
    return sorted(d for d in config.RUNS_DIR.iterdir() if d.is_dir())


def inference_model_path(run_dir: Path) -> Path:
    """
    Resolve the one model this script is allowed to evaluate.

    Falling back to model.keras here would reintroduce optimizer-state size
    inflation into Table III, so a missing export is fatal for that run.
    """
    path = run_dir / INFERENCE_MODEL_NAME
    if path.exists():
        return path

    trained = run_dir / "model.keras"
    detail = (f" A trained checkpoint ({trained.name}) IS present, but it "
              f"carries optimizer state and must never be used for reported "
              f"size or latency." if trained.exists() else "")
    raise MissingInferenceModel(
        f"{run_dir.name}: {INFERENCE_MODEL_NAME} not found.{detail} "
        f"Re-run training for this run; there is no safe fallback."
    )


def file_size_mb(path: Path) -> float:
    if path.is_dir():
        total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    else:
        total = path.stat().st_size
    return round(total / (1024 ** 2), 3)


def count_parameters(model: keras.Model) -> Dict[str, int]:
    trainable = int(sum(np.prod(v.shape) for v in model.trainable_weights))
    non_trainable = int(sum(np.prod(v.shape) for v in model.non_trainable_weights))
    return {
        "total_params": trainable + non_trainable,
        "trainable_params": trainable,
        "non_trainable_params": non_trainable,
    }


def run_metadata(run_dir: Path) -> dict:
    path = run_dir / "config_snapshot.json"
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError:
        return {}


# --------------------------------------------------------------------------- #
# Prediction and metrics
# --------------------------------------------------------------------------- #
def predict_test_set(model: keras.Model, test_ds: tf.data.Dataset):
    y_true, y_prob = [], []
    for images, labels in test_ds:
        y_prob.append(model.predict_on_batch(images))
        y_true.append(labels.numpy())
    return np.concatenate(y_true), np.concatenate(y_prob)


def predict_test_set_tta(model: keras.Model, test_ds: tf.data.Dataset):
    """Horizontal-flip TTA: mean of the original and mirrored probabilities."""
    y_true, y_prob = [], []
    for images, labels in test_ds:
        base = model.predict_on_batch(images)
        flipped = model.predict_on_batch(tf.image.flip_left_right(images))
        y_prob.append((np.asarray(base) + np.asarray(flipped)) / 2.0)
        y_true.append(labels.numpy())
    return np.concatenate(y_true), np.concatenate(y_prob)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                           class_names: List[str]) -> Dict[str, object]:
    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 5),
        "precision_macro": round(float(precision_score(
            y_true, y_pred, average="macro", zero_division=0)), 5),
        "recall_macro": round(float(recall_score(
            y_true, y_pred, average="macro", zero_division=0)), 5),
        "macro_f1": round(float(f1_score(
            y_true, y_pred, average="macro", zero_division=0)), 5),
        "weighted_f1": round(float(f1_score(
            y_true, y_pred, average="weighted", zero_division=0)), 5),
        "per_class": {
            name: {
                "precision": round(float(precision[i]), 5),
                "recall": round(float(recall[i]), 5),
                "f1": round(float(f1[i]), 5),
                "support": int(support[i]),
            }
            for i, name in enumerate(class_names)
        },
    }


# --------------------------------------------------------------------------- #
# Figures and tables
# --------------------------------------------------------------------------- #
def write_confusion_matrix(cm: np.ndarray, class_names: List[str],
                           run_dir: Path, title: str) -> None:
    with open(run_dir / "confusion_matrix.csv", "w", newline="",
              encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["true\\predicted"] + class_names)
        for name, row in zip(class_names, cm):
            writer.writerow([name] + [int(v) for v in row])

    fig, ax = plt.subplots(figsize=(1.35 * len(class_names) + 2.5,
                                    1.15 * len(class_names) + 2.0))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=np.arange(len(class_names)),
           yticks=np.arange(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           ylabel="True label", xlabel="Predicted label", title=title)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right",
             rotation_mode="anchor")
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(run_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_table_rows(records: List[dict]) -> None:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(config.RESULTS_DIR / "table2_row.csv", "w", newline="",
              encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(TABLE2_COLUMNS)
        for r in records:
            writer.writerow([r["model"], r["accuracy"], r["precision_macro"],
                             r["recall_macro"], r["macro_f1"]])
    print(f"[eval] Table II row(s) -> {config.RESULTS_DIR / 'table2_row.csv'}")

    with open(config.RESULTS_DIR / "table3_row.csv", "w", newline="",
              encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(TABLE3_COLUMNS)
        for r in records:
            writer.writerow([r["model"], r["total_params"], r["model_size_mb"],
                             r["latency_ms_mean"], r["latency_ms_std"],
                             r["macro_f1"]])
    print(f"[eval] Table III row(s) -> {config.RESULTS_DIR / 'table3_row.csv'}")

    extended = config.RESULTS_DIR / "table3_row_extended.csv"
    with open(extended, "w", newline="", encoding="utf-8") as fh:
        columns = (["Model", "run_name", "Parameters", "Model Size (MB)"]
                   + bench.latency_columns() + ["Macro-F1"])
        writer = csv.writer(fh)
        writer.writerow(columns)
        for r in records:
            writer.writerow(
                [r["model"], r["run_name"], r["total_params"],
                 r["model_size_mb"]]
                + [r.get(c, "") for c in bench.latency_columns()]
                + [r["macro_f1"]])
    print(f"[eval] Table III (median/p95/machine) -> {extended}")


def write_tta_rows(rows: List[dict]) -> None:
    if not rows:
        return
    path = config.RESULTS_DIR / "table2_row_tta.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Model (TTA)", "Augmentation", "Accuracy",
                         "Precision", "Recall", "Macro-F1",
                         "Accuracy Delta vs no-TTA (pp)"])
        for r in rows:
            writer.writerow([
                f"{r['model']} + TTA", "horizontal flip", r["accuracy"],
                r["precision_macro"], r["recall_macro"], r["macro_f1"],
                round((r["accuracy"] - r["baseline_accuracy"]) * 100, 3)])
    print(f"[eval] TTA rows (kept OUT of the headline table) -> {path}")


def write_per_class(records: List[dict], class_names: List[str]) -> None:
    path = config.RESULTS_DIR / "per_class_metrics.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Model", "run_name", "Class", "Precision", "Recall",
                         "F1", "Support"])
        for r in records:
            for name in class_names:
                stats = r["per_class"][name]
                writer.writerow([r["model"], r["run_name"], name,
                                 stats["precision"], stats["recall"],
                                 stats["f1"], stats["support"]])
    print(f"[eval] Per-class metrics -> {path}")


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate_run(run_dir: Path, test_ds: tf.data.Dataset,
                 latency_samples: np.ndarray, class_names: List[str],
                 skip_latency: bool, tta: bool, threads: int) -> tuple:
    model_path = inference_model_path(run_dir)
    meta = run_metadata(run_dir)
    display = meta.get("result", {}).get("model") or config.display_name(
        meta.get("slug", run_dir.name))

    print("\n" + "-" * 72)
    print(f"  Evaluating {display}   run={run_dir.name}")
    print(f"  Model: {model_path.name} ({file_size_mb(model_path)} MB)")
    print("-" * 72)

    keras.backend.clear_session()
    gc.collect()
    model = keras.models.load_model(model_path, compile=False)

    y_true, y_prob = predict_test_set(model, test_ds)
    y_pred = y_prob.argmax(axis=1)

    metrics = classification_metrics(y_true, y_pred, class_names)
    metrics.update(count_parameters(model))
    metrics["model"] = display
    metrics["run_name"] = run_dir.name
    metrics["slug"] = meta.get("slug", run_dir.name)
    metrics["seed"] = meta.get("seed")
    metrics["model_file"] = model_path.name
    metrics["model_size_mb"] = file_size_mb(model_path)
    metrics["test_images"] = int(len(y_true))

    if skip_latency:
        for key in bench.latency_columns():
            metrics.setdefault(key, "")
        metrics["latency_ms_mean"] = float("nan")
        metrics["latency_ms_std"] = float("nan")
    else:
        infer = tf.function(
            lambda x: model(x, training=False),
            input_signature=[tf.TensorSpec([1, *config.INPUT_SHAPE],
                                           tf.float32)])
        samples = [tf.constant(img[None, ...]) for img in latency_samples]
        metrics.update(bench.measure_latency(
            infer, samples, label=display, threads=threads))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    write_confusion_matrix(cm, class_names, run_dir,
                           f"Confusion Matrix - {display} ({run_dir.name})")

    report = classification_report(y_true, y_pred, target_names=class_names,
                                   digits=4, zero_division=0)
    (run_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    np.save(run_dir / "test_probabilities.npy", y_prob)
    np.save(run_dir / "test_labels.npy", y_true)
    print(report)

    tta_record = None
    if tta:
        print(f"[eval] Test-time augmentation (horizontal flip)...")
        y_true_t, y_prob_t = predict_test_set_tta(model, test_ds)
        tta_metrics = classification_metrics(
            y_true_t, y_prob_t.argmax(axis=1), class_names)
        tta_record = {
            "model": display,
            "run_name": run_dir.name,
            "baseline_accuracy": metrics["accuracy"],
            **{k: v for k, v in tta_metrics.items() if k != "per_class"},
        }
        print(f"[eval] TTA accuracy {tta_record['accuracy']:.5f} vs "
              f"{metrics['accuracy']:.5f} baseline "
              f"({(tta_record['accuracy'] - metrics['accuracy']) * 100:+.3f} pp)"
              f" - reported separately, not in Table II.")

    with open(run_dir / "evaluation.json", "w", encoding="utf-8") as fh:
        json.dump({"headline": metrics, "tta": tta_record}, fh, indent=2,
                  default=str)

    del model
    keras.backend.clear_session()
    gc.collect()
    return metrics, tta_record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate training runs on the frozen test split.")
    parser.add_argument("--runs", nargs="+", default=None,
                        help="Run directory names under runs/.")
    parser.add_argument("--gpu", action="store_true",
                        help="Keep the GPU visible (latency becomes invalid).")
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--tta", action="store_true",
                        help="Also report horizontal-flip TTA, separately.")
    parser.add_argument("--threads", type=int, default=config.BENCH_NUM_THREADS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    args = parser.parse_args()

    config.ensure_dirs()
    if not args.gpu:
        hide_gpu()
    threads = bench.pin_threads(args.threads)
    print(f"[eval] Machine: {json.dumps(bench.describe_machine(threads))}")

    run_dirs = discover_runs(args.runs)
    if not run_dirs:
        raise SystemExit(f"No run directories under {config.RUNS_DIR}. "
                         f"Run train.py first.")

    class_names, _ = data_loader.load_manifest()
    test_ds = data_loader.make_dataset("test", batch_size=args.batch_size,
                                       shuffle=False, augment=False)
    latency_samples = next(iter(data_loader.make_dataset(
        "test", batch_size=32, shuffle=False, augment=False)))[0].numpy()

    records: List[dict] = []
    tta_records: List[dict] = []
    missing: List[str] = []

    for run_dir in run_dirs:
        try:
            metrics, tta_record = evaluate_run(
                run_dir, test_ds, latency_samples, class_names,
                args.skip_latency or args.gpu, args.tta, threads)
        except MissingInferenceModel as exc:
            missing.append(str(exc))
            print(f"\n[eval] ERROR: {exc}")
            continue
        records.append(metrics)
        if tta_record:
            tta_records.append(tta_record)

    if records:
        write_table_rows(records)
        write_per_class(records, class_names)
        write_tta_rows(tta_records)
        with open(config.RESULTS_DIR / "evaluation_full.json", "w",
                  encoding="utf-8") as fh:
            json.dump(records, fh, indent=2, default=str)

        print("\n--- Table II ---")
        print(f"  {'Model':<28}{'Acc':>9}{'Prec':>9}{'Rec':>9}{'MacroF1':>10}")
        for r in records:
            print(f"  {r['model']:<28}{r['accuracy']:>9.4f}"
                  f"{r['precision_macro']:>9.4f}{r['recall_macro']:>9.4f}"
                  f"{r['macro_f1']:>10.4f}")

    if missing:
        raise SystemExit(
            f"\n[eval] {len(missing)} run(s) had no {INFERENCE_MODEL_NAME} "
            f"and were NOT evaluated. Nothing was substituted for them.")

    print("\nNext step:  python quantize.py")


if __name__ == "__main__":
    main()
