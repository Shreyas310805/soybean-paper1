"""
data_loader.py
==============
Dataset auditing, a deterministic stratified 70/15/15 split, and performant
`tf.data.Dataset` objects.

Design decisions that matter downstream
---------------------------------------
1. The split is written **once** to `splits/split_manifest.csv` and every other
   script reads that file. The test set therefore never changes between
   `train.py`, `evaluate.py` and `quantize.py`, or between the four models.
2. The pipeline yields **raw float32 images in [0, 255]**. Model-specific
   ImageNet normalisation lives *inside* each Keras model (see `train.py`), so
   one dataset serves all four architectures and the exported TFLite file is
   self-contained.
3. Augmentation is applied **after batching, to the training split only**, and
   is identical for all four models.
4. `cache()` writes to disk by default; caching 224x224 float tensors in RAM is
   the classic Colab OOM trap.

CLI
---
    python data_loader.py --audit --build        # audit + write the split
    python data_loader.py --show                 # print Table 1 distribution
    python data_loader.py --clear-cache          # drop stale tf.data caches
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import tensorflow as tf

import config

AUTOTUNE = tf.data.AUTOTUNE

_SPLIT_DIR_NAMES = {
    "train", "training", "val", "valid", "validation", "test", "testing"
}


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def _class_directories(root: Path) -> Dict[str, List[Path]]:
    """
    Map class name -> list of directories holding that class's images.

    Handles both `<root>/<class>/` and `<root>/{train,valid,test}/<class>/`
    (the second layout is merged back together, because we build our own split).
    """
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root '{root}' does not exist. "
            f"Set SOY_DATA_ROOT or place the dataset under {config.DATA_ROOT}."
        )

    subdirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not subdirs:
        raise FileNotFoundError(f"No class sub-directories found under '{root}'.")

    looks_presplit = all(d.name.lower() in _SPLIT_DIR_NAMES for d in subdirs)
    mapping: Dict[str, List[Path]] = defaultdict(list)

    if looks_presplit:
        print("[data] Detected a pre-split dataset; merging it before "
              "re-splitting 70/15/15.")
        for split_dir in subdirs:
            for class_dir in sorted(d for d in split_dir.iterdir() if d.is_dir()):
                mapping[class_dir.name].append(class_dir)
    else:
        for class_dir in subdirs:
            mapping[class_dir.name].append(class_dir)

    return dict(sorted(mapping.items()))


def _list_images(directories: Sequence[Path]) -> List[Path]:
    files: List[Path] = []
    for d in directories:
        for p in sorted(d.rglob("*")):
            if p.is_file() and p.suffix.lower() in config.VALID_IMAGE_EXTENSIONS:
                files.append(p)
    return files


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _is_readable(path: Path) -> bool:
    try:
        from PIL import Image  # noqa: WPS433 (optional fast path)
        with Image.open(path) as im:
            im.verify()
        return True
    except ImportError:
        try:
            tf.io.decode_image(tf.io.read_file(str(path)),
                               channels=3, expand_animations=False)
            return True
        except Exception:  # noqa: BLE001
            return False
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# Audit + split
# --------------------------------------------------------------------------- #
def audit_dataset(root: Path | None = None) -> Tuple[Dict[str, List[Path]], List[dict]]:
    """
    Guideline section 2, steps 1-3: count images, drop unreadable files, drop
    duplicates. Returns (kept mapping, dropped records).
    """
    root = Path(root or config.DATA_ROOT)
    class_dirs = _class_directories(root)

    kept: Dict[str, List[Path]] = {}
    dropped: List[dict] = []
    seen_hashes: Dict[str, str] = {}

    for class_name, dirs in class_dirs.items():
        files = _list_images(dirs)
        good: List[Path] = []
        for path in files:
            if config.DROP_UNREADABLE and not _is_readable(path):
                dropped.append({"file": str(path), "class": class_name,
                                "reason": "unreadable"})
                continue
            if config.DROP_DUPLICATES:
                digest = _md5(path)
                if digest in seen_hashes:
                    dropped.append({"file": str(path), "class": class_name,
                                    "reason": f"duplicate_of:{seen_hashes[digest]}"})
                    continue
                seen_hashes[digest] = str(path)
            good.append(path)
        kept[class_name] = good
        print(f"[audit] {class_name:<28} found={len(files):>5}  kept={len(good):>5}")

    config.ensure_dirs()
    with open(config.AUDIT_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file", "class", "reason"])
        writer.writeheader()
        writer.writerows(dropped)

    print(f"[audit] Dropped {len(dropped)} file(s) -> {config.AUDIT_CSV}")
    return kept, dropped


def build_manifest(root: Path | None = None,
                   run_audit: bool = True,
                   force: bool = False) -> Path:
    """
    Write a deterministic, class-stratified 70/15/15 manifest.

    Stratification guarantees every class appears in all three splits, and the
    fixed seed guarantees the identical split on every machine.
    """
    config.ensure_dirs()
    if config.SPLIT_MANIFEST.exists() and not force:
        print(f"[split] Manifest already exists: {config.SPLIT_MANIFEST} "
              f"(use --force to rebuild)")
        return config.SPLIT_MANIFEST

    if run_audit:
        kept, _ = audit_dataset(root)
    else:
        kept = {c: _list_images(d)
                for c, d in _class_directories(Path(root or config.DATA_ROOT)).items()}

    class_names = sorted(kept.keys())
    rng = random.Random(config.RANDOM_SEED)
    rows: List[dict] = []

    for label, class_name in enumerate(class_names):
        files = sorted(str(p) for p in kept[class_name])
        rng.shuffle(files)
        n = len(files)
        n_train = int(round(n * config.TRAIN_SPLIT))
        n_val = int(round(n * config.VAL_SPLIT))
        n_train = min(n_train, max(n - 2, 0))          # keep >=1 for val/test
        n_val = min(n_val, max(n - n_train - 1, 0))

        assignment = (["train"] * n_train
                      + ["val"] * n_val
                      + ["test"] * (n - n_train - n_val))
        for path, split in zip(files, assignment):
            rows.append({"filepath": path, "class_name": class_name,
                         "label": label, "split": split})

    with open(config.SPLIT_MANIFEST, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["filepath", "class_name", "label", "split"])
        writer.writeheader()
        writer.writerows(rows)

    with open(config.CLASS_NAMES_JSON, "w", encoding="utf-8") as fh:
        json.dump(class_names, fh, indent=2)

    _write_distribution_table(rows, class_names)
    clear_cache()  # a new split invalidates any cached tensors
    print(f"[split] Wrote {len(rows)} rows -> {config.SPLIT_MANIFEST}")
    return config.SPLIT_MANIFEST


def _write_distribution_table(rows: List[dict], class_names: List[str]) -> None:
    counts = Counter((r["class_name"], r["split"]) for r in rows)
    with open(config.DISTRIBUTION_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Class", "Training Images", "Validation Images",
                         "Testing Images", "Total"])
        for name in class_names:
            tr = counts[(name, "train")]
            va = counts[(name, "val")]
            te = counts[(name, "test")]
            writer.writerow([name, tr, va, te, tr + va + te])
        writer.writerow([
            "TOTAL",
            sum(counts[(c, "train")] for c in class_names),
            sum(counts[(c, "val")] for c in class_names),
            sum(counts[(c, "test")] for c in class_names),
            len(rows),
        ])
    print(f"[split] Table 1 (dataset distribution) -> {config.DISTRIBUTION_CSV}")


def load_manifest() -> Tuple[List[str], Dict[str, Tuple[List[str], List[int]]]]:
    """Return (class_names, {split: (filepaths, labels)})."""
    if not config.SPLIT_MANIFEST.exists():
        raise FileNotFoundError(
            "Split manifest missing. Run: python data_loader.py --audit --build"
        )
    with open(config.CLASS_NAMES_JSON, encoding="utf-8") as fh:
        class_names = json.load(fh)

    buckets: Dict[str, Tuple[List[str], List[int]]] = {
        "train": ([], []), "val": ([], []), "test": ([], [])
    }
    with open(config.SPLIT_MANIFEST, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            paths, labels = buckets[row["split"]]
            paths.append(row["filepath"])
            labels.append(int(row["label"]))
    return class_names, buckets


# --------------------------------------------------------------------------- #
# tf.data pipeline
# --------------------------------------------------------------------------- #
def _decode(path: tf.Tensor, label: tf.Tensor):
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=config.IMG_CHANNELS,
                             expand_animations=False)
    img = tf.image.resize(img, config.IMG_SIZE, method="bilinear")
    # Cache as uint8: ~150 KB/image instead of ~600 KB as float32.
    img = tf.cast(tf.round(tf.clip_by_value(img, 0.0, 255.0)), tf.uint8)
    img.set_shape(config.INPUT_SHAPE)
    return img, label


def build_augmenter() -> tf.keras.Sequential:
    """Geometric augmentation only; identical across all four models."""
    layers = tf.keras.layers
    stack = []
    if config.AUG_HORIZONTAL_FLIP:
        stack.append(layers.RandomFlip("horizontal", seed=config.RANDOM_SEED))
    if config.AUG_ROTATION_FACTOR:
        stack.append(layers.RandomRotation(config.AUG_ROTATION_FACTOR,
                                           fill_mode="reflect",
                                           seed=config.RANDOM_SEED))
    if config.AUG_ZOOM_FACTOR:
        stack.append(layers.RandomZoom(config.AUG_ZOOM_FACTOR,
                                       fill_mode="reflect",
                                       seed=config.RANDOM_SEED))
    if config.AUG_TRANSLATION_FACTOR:
        stack.append(layers.RandomTranslation(config.AUG_TRANSLATION_FACTOR,
                                              config.AUG_TRANSLATION_FACTOR,
                                              fill_mode="reflect",
                                              seed=config.RANDOM_SEED))
    return tf.keras.Sequential(stack, name="augmentation")


def make_dataset(split: str,
                 batch_size: int | None = None,
                 shuffle: bool | None = None,
                 augment: bool | None = None,
                 cache: bool = True,
                 drop_remainder: bool = False) -> tf.data.Dataset:
    """
    Build a `tf.data.Dataset` yielding (float32 image in [0,255], int32 label).

    Defaults: the training split is shuffled and augmented; val/test are not.
    """
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown split '{split}'.")

    batch_size = config.BATCH_SIZE if batch_size is None else batch_size
    shuffle = (split == "train") if shuffle is None else shuffle
    augment = (split == "train") if augment is None else augment

    _, buckets = load_manifest()
    paths, labels = buckets[split]
    if not paths:
        raise ValueError(f"Split '{split}' is empty.")

    ds = tf.data.Dataset.from_tensor_slices(
        (tf.constant(paths), tf.constant(labels, dtype=tf.int32)))
    ds = ds.map(_decode, num_parallel_calls=AUTOTUNE)

    if cache:
        if config.CACHE_TO_DISK:
            config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
            ds = ds.cache(str(config.CACHE_DIR / f"{split}"))
        else:
            ds = ds.cache()

    if shuffle:
        ds = ds.shuffle(min(len(paths), config.SHUFFLE_BUFFER),
                        seed=config.RANDOM_SEED,
                        reshuffle_each_iteration=True)

    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y),
                num_parallel_calls=AUTOTUNE)

    if augment:
        augmenter = build_augmenter()
        ds = ds.map(lambda x, y: (augmenter(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)

    return ds.prefetch(AUTOTUNE)


def get_datasets(batch_size: int | None = None):
    """Convenience: (train_ds, val_ds, test_ds, class_names)."""
    class_names, _ = load_manifest()
    return (make_dataset("train", batch_size=batch_size),
            make_dataset("val", batch_size=batch_size),
            make_dataset("test", batch_size=batch_size),
            class_names)


def split_size(split: str) -> int:
    _, buckets = load_manifest()
    return len(buckets[split][0])


def get_labels(split: str) -> List[int]:
    """Ground-truth labels in manifest order (matches an unshuffled dataset)."""
    _, buckets = load_manifest()
    return buckets[split][1]


def compute_class_weights() -> Dict[int, float]:
    """Inverse-frequency weights for the training split (optional, `--class-weight`)."""
    labels = get_labels("train")
    counts = Counter(labels)
    total = len(labels)
    n_classes = len(counts)
    return {int(k): total / (n_classes * v) for k, v in counts.items()}


def clear_cache() -> None:
    if config.CACHE_DIR.exists():
        shutil.rmtree(config.CACHE_DIR, ignore_errors=True)
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("[data] tf.data cache cleared.")


def print_distribution() -> None:
    if not config.DISTRIBUTION_CSV.exists():
        raise FileNotFoundError("Run --build first.")
    with open(config.DISTRIBUTION_CSV, encoding="utf-8") as fh:
        for row in csv.reader(fh):
            print("  ".join(f"{cell:<22}" for cell in row))


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset audit and splitting.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--audit", action="store_true",
                        help="Check for unreadable and duplicate images.")
    parser.add_argument("--build", action="store_true",
                        help="Write the deterministic 70/15/15 manifest.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing manifest.")
    parser.add_argument("--show", action="store_true",
                        help="Print the Table 1 distribution.")
    parser.add_argument("--clear-cache", action="store_true")
    args = parser.parse_args()

    config.ensure_dirs()
    if args.clear_cache:
        clear_cache()
    if args.build:
        build_manifest(root=args.data_root, run_audit=args.audit, force=args.force)
    elif args.audit:
        audit_dataset(args.data_root)
    if args.show or args.build:
        print("\n--- Table 1: Dataset Distribution ---")
        print_distribution()
    if not any([args.audit, args.build, args.show, args.clear_cache]):
        parser.print_help()


if __name__ == "__main__":
    main()
