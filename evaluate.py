"""
evaluate.py
===========
Evaluates every model in `saved_models/` on the **fixed, unseen test split** and
produces Tables 2 and 3 plus Figures 2-5.

Reported metrics
----------------
Classification : accuracy, macro precision, macro recall, macro-F1,
                 per-class recall, confusion matrix
Hardware       : total/trainable parameters, on-disk size (MB),
                 average single-image CPU inference time (ms)

Why the GPU is hidden by default
--------------------------------
"Average CPU inference time" is only meaningful if the forward pass actually
runs on the CPU. Keeping a GPU visible and merely wrapping the call in
`tf.device('/CPU:0')` still forces a weight copy per call and gives misleading
numbers. This script therefore hides the GPU unless `--gpu` is passed, which
also makes the numbers comparable to the Raspberry Pi target.

Usage
-----
    python evaluate.py                       # all saved models, CPU
    python evaluate.py --models mobilenetv2
    python evaluate.py --gpu                 # faster metrics, latency invalid
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
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
    precision_score,
    recall_score,
)
from tensorflow import keras  # noqa: E402

import config  # noqa: E402
import data_loader  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def hide_gpu() -> None:
    try:
        tf.config.set_visible_devices([], "GPU")
        print("[eval] GPU hidden - all timings are true CPU timings.")
    except RuntimeError:
        print("[eval] WARNING: GPU already initialised; latency may be skewed.")


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


def measure_cpu_latency(model: keras.Model,
                        sample_images: np.ndarray,
                        warmup: int = config.LATENCY_WARMUP_RUNS,
                        runs: int = config.LATENCY_MEASURED_RUNS) -> Dict[str, float]:
    """Average single-image (batch = 1) forward-pass time in milliseconds."""
    infer = tf.function(
        lambda x: model(x, training=False),
        input_signature=[tf.TensorSpec([1, *config.INPUT_SHAPE], tf.float32)],
    )
    n = len(sample_images)
    for i in range(warmup):
        infer(tf.constant(sample_images[i % n][None, ...]))

    timings = []
    for i in range(runs):
        batch = tf.constant(sample_images[i % n][None, ...])
        t0 = time.perf_counter()
        infer(batch)
        timings.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(timings)
    return {
        "inference_ms_mean": round(float(arr.mean()), 3),
        "inference_ms_std": round(float(arr.std()), 3),
        "inference_ms_p95": round(float(np.percentile(arr, 95)), 3),
    }


def predict_test_set(model: keras.Model, test_ds: tf.data.Dataset):
    y_true, y_prob = [], []
    for images, labels in test_ds:
        y_prob.append(model.predict_on_batch(images))
        y_true.append(labels.numpy())
    return np.concatenate(y_true), np.concatenate(y_prob)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                           class_names: List[str]) -> Dict[str, object]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 5),
        "precision_macro": round(float(precision_score(
            y_true, y_pred, average="macro", zero_division=0)), 5),
        "recall_macro": round(float(recall_score(
            y_true, y_pred, average="macro", zero_division=0)), 5),
        "macro_f1": round(float(f1_score(
            y_true, y_pred, average="macro", zero_division=0)), 5),
        "per_class_recall": {
            name: round(float(v), 5) for name, v in zip(
                class_names,
                recall_score(y_true, y_pred, average=None,
                             labels=list(range(len(class_names))),
                             zero_division=0))
        },
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(cm: np.ndarray, class_names: List[str],
                          title: str, out_path: Path,
                          normalize: bool = False) -> None:
    data = cm.astype(float)
    if normalize:
        row_sums = data.sum(axis=1, keepdims=True)
        data = np.divide(data, row_sums, out=np.zeros_like(data),
                         where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(1.35 * len(class_names) + 2.5,
                                    1.15 * len(class_names) + 2.0))
    im = ax.imshow(data, interpolation="nearest", cmap="Blues",
                   vmin=0, vmax=data.max() if data.max() > 0 else 1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           ylabel="True label", xlabel="Predicted label", title=title)
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")

    thresh = data.max() / 2.0 if data.max() > 0 else 0.5
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            text = f"{data[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, text, ha="center", va="center", fontsize=9,
                    color="white" if data[i, j] > thresh else "black")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(slugs: List[str]) -> None:
    available = [(s, config.HISTORY_DIR / f"{s}_history.json") for s in slugs]
    available = [(s, p) for s, p in available if p.exists()]
    if not available:
        return

    fig, axes = plt.subplots(2, len(available),
                             figsize=(4.6 * len(available), 7.5), squeeze=False)
    for col, (slug, path) in enumerate(available):
        with open(path, encoding="utf-8") as fh:
            hist = json.load(fh)
        epochs = range(1, len(hist["loss"]) + 1)

        ax = axes[0][col]
        ax.plot(epochs, hist["accuracy"], marker="o", ms=3, label="Train")
        ax.plot(epochs, hist["val_accuracy"], marker="s", ms=3, label="Validation")
        ax.set_title(config.display_name(slug))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.grid(alpha=0.3)
        ax.legend()

        ax = axes[1][col]
        ax.plot(epochs, hist["loss"], marker="o", ms=3, label="Train")
        ax.plot(epochs, hist["val_loss"], marker="s", ms=3, label="Validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    out = config.FIGURES_DIR / "fig2_training_curves.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[eval] Figure 2 -> {out}")


def plot_tradeoff(records: List[dict], x_key: str, x_label: str,
                  filename: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for rec in records:
        ax.scatter(rec[x_key], rec["macro_f1"], s=90, zorder=3)
        ax.annotate(rec["model"], (rec[x_key], rec["macro_f1"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Macro-F1 (test set)")
    ax.set_title(title)
    ax.grid(alpha=0.3, zorder=0)
    fig.tight_layout()
    out = config.FIGURES_DIR / filename
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[eval] Figure -> {out}")


# --------------------------------------------------------------------------- #
# Main evaluation loop
# --------------------------------------------------------------------------- #
def evaluate_model(slug: str, test_ds: tf.data.Dataset,
                   sample_images: np.ndarray,
                   class_names: List[str],
                   skip_latency: bool) -> dict:
    model_path = config.SAVED_MODELS_DIR / f"{slug}.keras"
    print("\n" + "-" * 72)
    print(f"  Evaluating {config.display_name(slug)}")
    print("-" * 72)

    keras.backend.clear_session()
    gc.collect()
    model = keras.models.load_model(model_path, compile=False)

    y_true, y_prob = predict_test_set(model, test_ds)
    y_pred = y_prob.argmax(axis=1)

    metrics = classification_metrics(y_true, y_pred, class_names)
    metrics.update(count_parameters(model))
    metrics["model"] = config.display_name(slug)
    metrics["slug"] = slug
    metrics["model_size_mb"] = file_size_mb(model_path)
    metrics["test_images"] = int(len(y_true))

    if skip_latency:
        metrics.update({"inference_ms_mean": float("nan"),
                        "inference_ms_std": float("nan"),
                        "inference_ms_p95": float("nan")})
    else:
        metrics.update(measure_cpu_latency(model, sample_images))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    plot_confusion_matrix(cm, class_names,
                          f"Confusion Matrix - {config.display_name(slug)}",
                          config.FIGURES_DIR / f"cm_{slug}.png")
    plot_confusion_matrix(cm, class_names,
                          f"Normalised Confusion Matrix - {config.display_name(slug)}",
                          config.FIGURES_DIR / f"cm_{slug}_normalized.png",
                          normalize=True)
    np.savetxt(config.RESULTS_DIR / f"cm_{slug}.csv", cm, fmt="%d", delimiter=",")

    report = classification_report(y_true, y_pred, target_names=class_names,
                                   digits=4, zero_division=0)
    (config.RESULTS_DIR / f"classification_report_{slug}.txt").write_text(
        report, encoding="utf-8")
    np.save(config.RESULTS_DIR / f"probs_{slug}.npy", y_prob)

    print(report)
    print(f"  params={metrics['total_params']:,}  "
          f"size={metrics['model_size_mb']} MB  "
          f"cpu_latency={metrics['inference_ms_mean']} ms")

    del model
    keras.backend.clear_session()
    gc.collect()
    return metrics


def write_tables(records: List[dict], class_names: List[str]) -> None:
    with open(config.MODEL_PERFORMANCE_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Model", "Accuracy", "Precision", "Recall", "Macro-F1"])
        for r in records:
            writer.writerow([r["model"], r["accuracy"], r["precision_macro"],
                             r["recall_macro"], r["macro_f1"]])
    print(f"[eval] Table 2 -> {config.MODEL_PERFORMANCE_CSV}")

    with open(config.EFFICIENCY_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Model", "Parameters", "Model Size (MB)",
                         "CPU Inference Time (ms)", "Std (ms)", "Macro-F1"])
        for r in records:
            writer.writerow([r["model"], r["total_params"], r["model_size_mb"],
                             r["inference_ms_mean"], r["inference_ms_std"],
                             r["macro_f1"]])
    print(f"[eval] Table 3 -> {config.EFFICIENCY_CSV}")

    with open(config.PER_CLASS_RECALL_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Model"] + class_names)
        for r in records:
            writer.writerow([r["model"]] +
                            [r["per_class_recall"][c] for c in class_names])
    print(f"[eval] Per-class recall -> {config.PER_CLASS_RECALL_CSV}")

    with open(config.RESULTS_DIR / "evaluation_full.json", "w",
              encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)


def select_best(records: List[dict]) -> dict:
    """
    Primary selection: highest macro-F1.
    Also reports the most efficiency-friendly model whose macro-F1 is within
    1 percentage point of the top score - the deployment-oriented choice the
    paper's research question asks about.
    """
    best_f1 = max(records, key=lambda r: r["macro_f1"])
    tolerance = best_f1["macro_f1"] - 0.01
    contenders = [r for r in records if r["macro_f1"] >= tolerance]
    best_tradeoff = min(contenders,
                        key=lambda r: (r["model_size_mb"], r["total_params"]))

    payload = {
        "best_by_macro_f1": {"slug": best_f1["slug"], "model": best_f1["model"],
                             "macro_f1": best_f1["macro_f1"],
                             "accuracy": best_f1["accuracy"]},
        "best_accuracy_efficiency_tradeoff": {
            "slug": best_tradeoff["slug"], "model": best_tradeoff["model"],
            "macro_f1": best_tradeoff["macro_f1"],
            "model_size_mb": best_tradeoff["model_size_mb"],
            "inference_ms_mean": best_tradeoff["inference_ms_mean"],
            "rule": "smallest model within 1.0 pp macro-F1 of the top score",
        },
    }
    with open(config.BEST_MODEL_JSON, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n[eval] Best by macro-F1        : {best_f1['model']} "
          f"({best_f1['macro_f1']:.4f})")
    print(f"[eval] Best accuracy/efficiency: {best_tradeoff['model']} "
          f"({best_tradeoff['model_size_mb']} MB, "
          f"{best_tradeoff['inference_ms_mean']} ms)")
    print(f"[eval] Written -> {config.BEST_MODEL_JSON}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the trained models.")
    parser.add_argument("--models", nargs="+", default=None,
                        choices=config.MODEL_SLUGS)
    parser.add_argument("--gpu", action="store_true",
                        help="Keep the GPU visible (latency becomes invalid).")
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    args = parser.parse_args()

    config.ensure_dirs()
    if not args.gpu:
        hide_gpu()

    slugs = args.models or [s for s in config.MODEL_SLUGS
                            if (config.SAVED_MODELS_DIR / f"{s}.keras").exists()]
    if not slugs:
        raise SystemExit("No trained models found in saved_models/. Run train.py.")

    class_names, _ = data_loader.load_manifest()
    test_ds = data_loader.make_dataset("test", batch_size=args.batch_size,
                                       shuffle=False, augment=False)

    # Fixed pool of real test images for the latency benchmark.
    sample_images = next(iter(
        data_loader.make_dataset("test", batch_size=32, shuffle=False,
                                 augment=False)))[0].numpy()

    records = [evaluate_model(s, test_ds, sample_images, class_names,
                              args.skip_latency or args.gpu)
               for s in slugs]

    write_tables(records, class_names)
    plot_training_curves(slugs)
    if not (args.skip_latency or args.gpu):
        plot_tradeoff(records, "model_size_mb", "Model size (MB)",
                      "fig4_accuracy_vs_size.png",
                      "Accuracy-efficiency trade-off: macro-F1 vs model size")
        plot_tradeoff(records, "inference_ms_mean",
                      "Average CPU inference time (ms)",
                      "fig5_accuracy_vs_inference_time.png",
                      "Accuracy-efficiency trade-off: macro-F1 vs latency")
    select_best(records)
    print("\nNext step:  python quantize.py")


if __name__ == "__main__":
    main()
