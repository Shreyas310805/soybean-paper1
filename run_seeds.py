"""
run_seeds.py
============
Trains and evaluates MobileNetV2 at seeds 42 / 1337 / 2024 on the frozen split
and reports the spread.

What the spread does and does not mean
--------------------------------------
All three seeds read the identical partition. The seed varies weight
initialisation and batch shuffling only, so the spread measures optimisation
variance, not an accidental re-draw of an easier test set.

With n = 3, the standard deviation is a weak statistic and is reported for
completeness rather than as the headline. The three individual accuracies, the
min, the max and the range are what belong in the paper; sd uses ddof = 1
because these are a sample, not the population.

Crash behaviour
---------------
`seed_summary.csv` is rewritten after every seed finishes, not once at the end,
and each completed run drops a `done.json`. A rerun skips seeds that already
have one. A killed Kaggle session therefore costs the seed in flight and
nothing else.

Success bar
-----------
Mean test accuracy >= 0.95 with the worst seed >= 0.94. Landing between 0.93
and 0.95 is not quietly accepted: the script prints a diagnosis and a ranked
list of next levers, and dumps the highest-loss frogeye test images so the
weakest class can be looked at rather than guessed about.

Usage
-----
    python run_seeds.py
    python run_seeds.py --seeds 42 1337 --class-weight
    python run_seeds.py --resume
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np

import config

SUMMARY_COLUMNS = [
    "run_name", "model", "seed", "accuracy", "macro_f1", "precision_macro",
    "recall_macro", "weighted_f1", "best_val_accuracy", "epochs_run",
    "train_time_sec", "model_size_mb", "total_params", "class_weight",
    "mixed_precision", "status",
]

LEVERS = [
    ("Unfreeze more of the backbone (0.5, then all) at a lower LR",
     "+1.5 to +3.0 pp", "~1 GPU-hour per seed",
     "python run_seeds.py --unfreeze-ratio 0.5 --fine-tune-lr 5e-5",
     "The single biggest lever once data is plentiful. 0.35 was tuned on "
     "2712 training images; with ~6750 the backbone can take more."),
    ("Longer fine-tune schedule with cosine restarts",
     "+0.5 to +1.5 pp", "~1.5x current fine-tune time",
     "python run_seeds.py --fine-tune-epochs 60",
     "Cheap and low-risk, but the returns fade once val_accuracy plateaus. "
     "Check training_curves.png first: if the curve is already flat for the "
     "last 10 epochs this will buy nothing."),
    ("Resolution bump 224 -> 260",
     "+0.5 to +2.0 pp", "~1.35x latency AND ~1.35x training time",
     "edit IMG_HEIGHT/IMG_WIDTH in config.py, then rebuild the cache",
     "Costs inference latency on the Pi Zero 2 W, which is the deployment "
     "target this paper argues for. Weigh against Table III before using it; "
     "an accuracy gain bought with latency undercuts the paper's own thesis."),
    ("Frogeye-targeted analysis",
     "unknown until looked at", "minutes",
     "inspect results/frogeye_hardest.csv and the linked images",
     "Frogeye is the weak class. Look at the actual failures before "
     "proposing a data-side fix; the previous 'healthy' weakness turned out "
     "to be a frozen-backbone artefact and needed no data work at all."),
]


def log(message: str = "") -> None:
    print(message, flush=True)


def run_dir_for(slug: str, seed: int) -> Path:
    return config.RUNS_DIR / f"{slug}_seed{seed}"


def done_marker(run_dir: Path) -> Path:
    return run_dir / "done.json"


def is_done(run_dir: Path) -> bool:
    return (done_marker(run_dir).exists()
            and (run_dir / "model_inference.keras").exists()
            and (run_dir / "evaluation.json").exists())


def estimate_runtime(seeds: List[int], args: argparse.Namespace,
                     n_train: int) -> float:
    """
    Rough wall-clock estimate, printed before anything starts so the run can be
    checked against the 12 h session limit and the weekly GPU quota.

    Calibrated on the earlier P100 runs: roughly 0.9 s per epoch per 100
    training images for MobileNetV2 at 224x224, batch 32.
    """
    per_epoch = 0.9 * (n_train / 100.0)
    epochs = args.head_epochs + args.fine_tune_epochs
    train_sec = per_epoch * epochs
    eval_sec = 240.0
    total = (train_sec + eval_sec) * len(seeds)

    log("=" * 72)
    log(" Estimated runtime")
    log("=" * 72)
    log(f"  train images        : {n_train}")
    log(f"  epochs per seed     : {args.head_epochs} head + "
        f"{args.fine_tune_epochs} fine-tune = {epochs}")
    log(f"  est. per seed       : {timedelta(seconds=int(train_sec + eval_sec))}"
        f"  (train ~{timedelta(seconds=int(train_sec))})")
    log(f"  seeds               : {len(seeds)}  {seeds}")
    log(f"  EST. TOTAL          : {timedelta(seconds=int(total))}")
    log(f"  finishes around     : "
        f"{(datetime.now() + timedelta(seconds=total)):%H:%M on %d %b}")
    log("")
    if total > 12 * 3600:
        log("  WARNING: this exceeds Kaggle's 12 h session limit. Run fewer")
        log("           seeds per session; --resume picks up the rest.")
    else:
        log(f"  Fits inside one 12 h session "
            f"({total / 3600:.1f} h of the 12 h limit).")
    log(f"  Weekly GPU quota is ~30 h; this consumes ~{total / 3600:.1f} h.")
    log("=" * 72)
    log("")
    return total


# --------------------------------------------------------------------------- #
# Per-seed work
# --------------------------------------------------------------------------- #
def train_and_evaluate(slug: str, seed: int, args: argparse.Namespace,
                       mixed_precision: bool) -> dict:
    """Train one seed, evaluate it, and write its done marker."""
    import tensorflow as tf  # noqa: F401
    from tensorflow import keras

    import data_loader
    import evaluate
    import train as train_mod

    run_dir = run_dir_for(slug, seed)

    train_args = argparse.Namespace(
        models=[slug], seed=seed, run_name=run_dir.name,
        head_epochs=args.head_epochs, epochs=None,
        fine_tune_epochs=args.fine_tune_epochs,
        unfreeze_ratio=args.unfreeze_ratio, lr=args.lr,
        fine_tune_lr=args.fine_tune_lr, batch_size=args.batch_size,
        class_weight=args.class_weight,
        weight_check_batches=args.weight_check_batches,
        skip_existing=False, mixed_precision=mixed_precision, summary=False)

    record = train_mod.train_one(slug, train_args, mixed_precision)

    keras.backend.clear_session()
    gc.collect()

    class_names, _ = data_loader.load_manifest()
    test_ds = data_loader.make_dataset("test", batch_size=args.batch_size,
                                       shuffle=False, augment=False)
    latency_samples = next(iter(data_loader.make_dataset(
        "test", batch_size=32, shuffle=False, augment=False)))[0].numpy()

    metrics, _ = evaluate.evaluate_run(
        run_dir, test_ds, latency_samples, class_names,
        skip_latency=args.skip_latency, tta=False,
        threads=config.BENCH_NUM_THREADS)

    row = {
        "run_name": run_dir.name,
        "model": record["model"],
        "seed": seed,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "weighted_f1": metrics["weighted_f1"],
        "best_val_accuracy": record["best_val_accuracy"],
        "epochs_run": record["epochs_run"],
        "train_time_sec": record["train_time_sec"],
        "model_size_mb": metrics["model_size_mb"],
        "total_params": metrics["total_params"],
        "class_weight": record["class_weight"],
        "mixed_precision": record["mixed_precision"],
        "status": "ok",
    }

    with open(done_marker(run_dir), "w", encoding="utf-8") as fh:
        json.dump({"finished_at": datetime.now().isoformat(timespec="seconds"),
                   "row": row, "per_class": metrics["per_class"]}, fh, indent=2)

    del test_ds
    keras.backend.clear_session()
    gc.collect()
    return row


def load_done_row(run_dir: Path) -> dict | None:
    try:
        with open(done_marker(run_dir), encoding="utf-8") as fh:
            return json.load(fh)["row"]
    except (OSError, json.JSONDecodeError, KeyError):
        return None


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def write_summary(rows: List[dict]) -> Path:
    """Rewritten after every seed so a killed session keeps what finished."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_DIR / "seed_summary.csv"

    accuracies = [r["accuracy"] for r in rows if r["status"] == "ok"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS,
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

        if len(accuracies) >= 2:
            writer.writerow({})
            for name, fn in (("MEAN", statistics.mean),
                             ("STDEV (ddof=1)", lambda v: statistics.stdev(v)),
                             ("MIN", min), ("MAX", max),
                             ("RANGE", lambda v: max(v) - min(v))):
                stats = {"run_name": name}
                for key in ("accuracy", "macro_f1", "precision_macro",
                            "recall_macro", "weighted_f1"):
                    values = [r[key] for r in rows if r["status"] == "ok"]
                    stats[key] = round(float(fn(values)), 5)
                writer.writerow(stats)
    return path


def report(rows: List[dict], slug: str) -> bool:
    ok_rows = [r for r in rows if r["status"] == "ok"]
    accuracies = [r["accuracy"] for r in ok_rows]
    if not accuracies:
        log("\n[seeds] No seed completed; nothing to report.")
        return False

    log("")
    log("=" * 72)
    log(f" Multi-seed result - {slug}  (n = {len(accuracies)})")
    log("=" * 72)
    log(f"  {'seed':<8}{'accuracy':>11}{'macro-F1':>11}{'val_acc':>11}"
        f"{'epochs':>8}")
    log("  " + "-" * 47)
    for row in ok_rows:
        log(f"  {row['seed']:<8}{row['accuracy']:>11.4f}"
            f"{row['macro_f1']:>11.4f}{row['best_val_accuracy']:>11.4f}"
            f"{row['epochs_run']:>8}")
    log("  " + "-" * 47)

    log("")
    log(f"  individual accuracies : "
        f"{', '.join(f'{a:.4f}' for a in accuracies)}")
    log(f"  min / max / range     : {min(accuracies):.4f} / "
        f"{max(accuracies):.4f} / {max(accuracies) - min(accuracies):.4f}")
    log(f"  mean                  : {statistics.mean(accuracies):.4f}")
    if len(accuracies) >= 2:
        log(f"  stdev (ddof=1)        : {statistics.stdev(accuracies):.4f}"
            f"   <- weak at n={len(accuracies)}; cite the three values")
    log("")
    log("  For the paper, prefer the three individual accuracies over a")
    log("  mean +/- sd: with n=3 the sd is not a meaningful estimate.")

    mean_acc = statistics.mean(accuracies)
    worst = min(accuracies)
    passed = (mean_acc >= config.TARGET_MEAN_ACCURACY
              and worst >= config.TARGET_WORST_ACCURACY)

    log("")
    log(f"  BAR: mean >= {config.TARGET_MEAN_ACCURACY:.2f} "
        f"AND worst >= {config.TARGET_WORST_ACCURACY:.2f}")
    log(f"  GOT: mean {mean_acc:.4f}, worst {worst:.4f}  ->  "
        f"{'PASS' if passed else 'NOT MET'}")
    log("=" * 72)
    return passed


def diagnose(rows: List[dict], slug: str) -> None:
    ok_rows = [r for r in rows if r["status"] == "ok"]
    accuracies = [r["accuracy"] for r in ok_rows]
    mean_acc = statistics.mean(accuracies)

    log("")
    log("#" * 72)
    log(" DIAGNOSIS - the bar was not met, and this is not being accepted")
    log("#" * 72)

    if mean_acc < config.DIAGNOSE_FLOOR:
        log(f"\n  Mean {mean_acc:.4f} is below {config.DIAGNOSE_FLOOR:.2f}, "
            f"which is worse than the 484-per-class baseline (0.925) on 2.5x")
        log("  more data. That points at a pipeline regression, not a")
        log("  modelling ceiling. Check, in this order:")
        log("    1. splits/split_audit.json  - class counts and zero overlap")
        log("    2. runs/*/training_curves.png - is the LR curve sane, or did")
        log("       cosine collapse to zero early?")
        log("    3. config_snapshot.json     - is fine_tune_epochs actually")
        log("       non-zero, and did mixed precision stay off on a P100?")
        log("    4. augmentation strength    - 0.15/0.15/0.2/0.2 is a big jump")
        log("       from the 0.08/0.10 that produced 0.925.")
    else:
        log(f"\n  Mean {mean_acc:.4f} sits between {config.DIAGNOSE_FLOOR:.2f} "
            f"and {config.TARGET_MEAN_ACCURACY:.2f}. The pipeline is working;")
        log("  the configuration is short. Ranked levers:")

    log("")
    for i, (name, gain, cost, command, note) in enumerate(LEVERS, start=1):
        log(f"  {i}. {name}")
        log(f"     expected : {gain}")
        log(f"     cost     : {cost}")
        log(f"     try      : {command}")
        log(f"     note     : {note}")
        log("")

    worst_class = weakest_class(ok_rows)
    if worst_class:
        name, recall = worst_class
        log(f"  Weakest class across seeds: {name} (mean recall {recall:.3f}).")
    log("#" * 72)


def weakest_class(rows: List[dict]) -> tuple | None:
    totals: Dict[str, List[float]] = {}
    for row in rows:
        marker = done_marker(config.RUNS_DIR / row["run_name"])
        try:
            with open(marker, encoding="utf-8") as fh:
                per_class = json.load(fh)["per_class"]
        except (OSError, json.JSONDecodeError, KeyError):
            continue
        for name, stats in per_class.items():
            totals.setdefault(name, []).append(stats["recall"])
    if not totals:
        return None
    means = {k: statistics.mean(v) for k, v in totals.items()}
    name = min(means, key=means.get)
    return name, means[name]


def dump_hardest(run_name: str, class_name: str, top_n: int = 20) -> None:
    """
    Write the highest-loss test images for one class, so the failures can be
    looked at rather than speculated about.
    """
    run_dir = config.RUNS_DIR / run_name
    probs_path = run_dir / "test_probabilities.npy"
    labels_path = run_dir / "test_labels.npy"
    if not (probs_path.exists() and labels_path.exists()):
        log(f"[seeds] No stored predictions for {run_name}; "
            f"skipping the {class_name} dump.")
        return

    import data_loader

    class_names, buckets = data_loader.load_manifest()
    if class_name not in class_names:
        return
    index = class_names.index(class_name)

    probs = np.load(probs_path)
    labels = np.load(labels_path)
    paths = buckets["test"][0]

    mask = labels == index
    if not mask.any():
        return
    positions = np.flatnonzero(mask)
    true_prob = np.clip(probs[positions, index], 1e-12, 1.0)
    loss = -np.log(true_prob)
    order = positions[np.argsort(-loss)][:top_n]

    out = config.RESULTS_DIR / f"{class_name}_hardest.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["rank", "path", "true_class", "predicted_class",
                         "true_prob", "predicted_prob", "cross_entropy"])
        for rank, pos in enumerate(order, start=1):
            predicted = int(probs[pos].argmax())
            writer.writerow([
                rank, paths[pos], class_name, class_names[predicted],
                round(float(probs[pos, index]), 5),
                round(float(probs[pos, predicted]), 5),
                round(float(-np.log(max(probs[pos, index], 1e-12))), 5)])
    log(f"[seeds] {top_n} hardest '{class_name}' test images -> {out}")
    log(f"[seeds] Look at these before proposing any data-side fix.")


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-seed protocol on the frozen split.")
    parser.add_argument("--model", default="mobilenetv2",
                        choices=config.MODEL_SLUGS)
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=list(config.SEEDS))
    parser.add_argument("--head-epochs", type=int, default=config.HEAD_EPOCHS)
    parser.add_argument("--fine-tune-epochs", type=int,
                        default=config.FINE_TUNE_EPOCHS)
    parser.add_argument("--unfreeze-ratio", type=float,
                        default=config.FINE_TUNE_UNFREEZE_RATIO)
    parser.add_argument("--lr", type=float, default=config.HEAD_LEARNING_RATE)
    parser.add_argument("--fine-tune-lr", type=float,
                        default=config.FINE_TUNE_LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--class-weight", action="store_true")
    parser.add_argument("--weight-check-batches", type=int, default=4)
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--no-resume", action="store_true",
                        help="Retrain seeds that already have done.json.")
    args = parser.parse_args()

    config.ensure_dirs()
    config.describe()

    if not config.SPLIT_MANIFEST.exists():
        raise SystemExit("Split manifest missing. Run:\n"
                         "  python data_loader.py --audit --build")

    import data_loader
    import train as train_mod

    mixed_precision = train_mod.configure_runtime(config.MIXED_PRECISION)
    n_train = data_loader.split_size("train")
    estimate_runtime(args.seeds, args, n_train)

    rows: List[dict] = []
    session_start = time.perf_counter()

    for position, seed in enumerate(args.seeds, start=1):
        run_dir = run_dir_for(args.model, seed)
        log("")
        log("#" * 72)
        log(f"  SEED {seed}   ({position}/{len(args.seeds)})   "
            f"[{datetime.now():%H:%M:%S}]")
        log("#" * 72)

        if is_done(run_dir) and not args.no_resume:
            cached = load_done_row(run_dir)
            if cached:
                log(f"[seeds] {run_dir.name} already complete "
                    f"(accuracy {cached['accuracy']:.4f}); skipping. "
                    f"Use --no-resume to force a retrain.")
                rows.append(cached)
                write_summary(rows)
                continue

        try:
            rows.append(train_and_evaluate(args.model, seed, args,
                                           mixed_precision))
        except KeyboardInterrupt:
            log("\n[seeds] Interrupted. Completed seeds are already saved.")
            break
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            rows.append({"run_name": run_dir.name, "model": args.model,
                         "seed": seed, "accuracy": float("nan"),
                         "macro_f1": float("nan"), "status": f"failed: {exc}"})
            log(f"[seeds] Seed {seed} FAILED; continuing with the rest.")

        path = write_summary(rows)
        log(f"[seeds] Summary updated after seed {seed} -> {path}")

        from tensorflow import keras
        keras.backend.clear_session()
        gc.collect()
        log("[seeds] Keras session cleared and garbage collected.")

    path = write_summary(rows)
    log(f"\n[seeds] Final summary -> {path}")
    log(f"[seeds] Session took "
        f"{timedelta(seconds=int(time.perf_counter() - session_start))}.")

    passed = report(rows, args.model)
    ok_rows = [r for r in rows if r["status"] == "ok"]
    if not passed and ok_rows:
        diagnose(rows, args.model)
        weakest = weakest_class(ok_rows)
        best_run = max(ok_rows, key=lambda r: r["accuracy"])["run_name"]
        dump_hardest(best_run, weakest[0] if weakest else "frogeye")
        raise SystemExit(1)

    if passed:
        log("\n[seeds] Bar met. Next: python quantize.py --run "
            f"{max(ok_rows, key=lambda r: r['accuracy'])['run_name']}")


if __name__ == "__main__":
    main()
