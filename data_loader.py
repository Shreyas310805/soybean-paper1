"""
data_loader.py
==============
Dataset auditing, a deterministic stratified 70/15/15 split, and performant
`tf.data.Dataset` objects.

The split is frozen, and that is a methodological claim
-------------------------------------------------------
The partition is computed once, from sorted filenames under seed 42, and
written to `splits/{train,val,test}.csv` plus `splits/split_manifest.csv`.
Every subsequent run reads those files rather than re-partitioning.

**The training seed varies weight initialisation and batch shuffling. It never
varies the data partition.** All three seeds in the multi-seed protocol
(42 / 1337 / 2024) train on byte-identical splits, so the reported spread
measures optimisation variance alone and not an accidental re-draw of an easier
test set. `splits/split_hash.txt` records a SHA-256 of the partition so two
notebooks can be proven to have used the same one.

Other decisions that matter downstream
--------------------------------------
1. Because the split is frozen, the test set never changes between `train.py`,
   `evaluate.py` and `quantize.py`, or between the four models.
2. The pipeline yields **raw float32 images in [0, 255]**. Model-specific
   ImageNet normalisation lives *inside* each Keras model (see `train.py`), so
   one dataset serves all four architectures and the exported TFLite file is
   self-contained.
3. Labels default to integers. `label_mode="one_hot"` is opt-in and used only
   by `train.py`, which needs one-hot targets for `CategoricalCrossentropy`
   label smoothing; `evaluate.py` and `quantize.py` keep integer labels so
   their sklearn paths are unaffected.
4. Passing `class_weights` makes the pipeline yield `(image, label, weight)`
   3-tuples instead of 2-tuples. This replaces `model.fit(class_weight=...)`,
   which cannot be combined with one-hot targets. Off unless asked for.
5. Augmentation is applied **after batching, to the training split only**.
6. `cache()` writes to disk by default; caching 224x224 tensors for ~9.6k
   images in RAM is the classic Kaggle/Colab OOM trap.

CLI
---
    python data_loader.py --audit --build        # audit + write the split
    python data_loader.py --audit                # verify an existing split
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

    subdirs = sorted([d for d in root.iterdir()
                      if d.is_dir() and not d.name.startswith((".", "_"))])
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
def _reuse_build_manifest(root: Path) -> Dict[str, List[Path]] | None:
    """
    Reuse `dataset_manifest.csv` from build_dataset.py instead of re-verifying.

    That script already decoded, EXIF-corrected and de-duplicated every file it
    wrote, so re-opening and re-hashing ~9.6k images here would spend minutes
    re-deriving a result we already have.

    The manifest records absolute build-time paths (`/kaggle/working/...`),
    which are stale the moment the dataset is remounted as a Kaggle Dataset, so
    entries are re-rooted onto the current DATA_ROOT by class and filename.
    Anything that does not line up falls back to the full audit.
    """
    manifest = root / "dataset_manifest.csv"
    if not manifest.exists():
        return None

    kept: Dict[str, List[Path]] = defaultdict(list)
    try:
        with open(manifest, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                class_name = row["class"]
                path = root / class_name / Path(row["path"]).name
                if not path.exists():
                    print(f"[audit] Manifest entry missing on disk ({path}); "
                          f"falling back to a full audit.")
                    return None
                kept[class_name].append(path)
    except (OSError, KeyError, csv.Error):
        return None

    if not kept:
        return None

    for class_name in kept:
        on_disk = len(_list_images([root / class_name]))
        if on_disk != len(kept[class_name]):
            print(f"[audit] {class_name}: manifest lists "
                  f"{len(kept[class_name])} but {on_disk} are on disk; "
                  f"falling back to a full audit.")
            return None

    total = sum(len(v) for v in kept.values())
    print(f"[audit] Reusing dataset_manifest.csv from build_dataset.py "
          f"({total} images across {len(kept)} classes, already verified "
          f"and de-duplicated).")
    return {k: sorted(v) for k, v in sorted(kept.items())}


def audit_dataset(root: Path | None = None) -> Tuple[Dict[str, List[Path]], List[dict]]:
    """
    Guideline section 2, steps 1-3: count images, drop unreadable files, drop
    duplicates. Returns (kept mapping, dropped records).
    """
    root = Path(root or config.DATA_ROOT)

    reused = _reuse_build_manifest(root)
    if reused is not None:
        config.ensure_dirs()
        with open(config.AUDIT_CSV, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["file", "class", "reason"])
            writer.writeheader()
        for class_name, files in reused.items():
            print(f"[audit] {class_name:<28} kept={len(files):>5}")
        return reused, []

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
    _write_split_csvs(rows)
    digest = write_split_hash(rows)
    clear_cache()  # a new split invalidates any cached tensors
    print(f"[split] Wrote {len(rows)} rows -> {config.SPLIT_MANIFEST}")
    print(f"[split] Split hash: {digest}")
    verify_splits(rows, class_names)
    return config.SPLIT_MANIFEST


def _write_split_csvs(rows: List[dict]) -> None:
    """One CSV per split, with (path, class, class_index)."""
    for split, path in config.SPLIT_CSVS.items():
        subset = [r for r in rows if r["split"] == split]
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["path", "class", "class_index"])
            for row in subset:
                writer.writerow([row["filepath"], row["class_name"],
                                 row["label"]])
        print(f"[split] {split:<5} {len(subset):>6} rows -> {path}")


def split_digest(rows: List[dict]) -> str:
    """
    SHA-256 over (split, class, filename) triples.

    Deliberately keyed on the basename rather than the full path: the same
    partition read from /kaggle/working in the build notebook and from
    /kaggle/input in the training notebook must produce the *same* hash, or
    the hash cannot serve as proof that two notebooks used one partition.
    """
    canonical = sorted(
        f"{r['split']}|{r['class_name']}|{Path(r['filepath']).name}"
        for r in rows
    )
    return hashlib.sha256("\n".join(canonical).encode("utf-8")).hexdigest()


def write_split_hash(rows: List[dict]) -> str:
    digest = split_digest(rows)
    with open(config.SPLIT_HASH_TXT, "w", encoding="utf-8") as fh:
        fh.write(f"{digest}\n")
        fh.write(f"rows={len(rows)}\n")
        fh.write(f"seed={config.RANDOM_SEED}\n")
    return digest


def verify_splits(rows: List[dict] | None = None,
                  class_names: List[str] | None = None) -> dict:
    """
    Per-class split counts, imbalance ratio, and a hard zero-overlap check.

    Overlap is tested on (class, filename) rather than the full path so that a
    remounted dataset cannot mask a genuine leak behind a path difference.
    """
    if rows is None:
        rows = _manifest_rows()
    if class_names is None:
        class_names = sorted({r["class_name"] for r in rows})

    by_split: Dict[str, set] = {s: set() for s in ("train", "val", "test")}
    counts: Dict[str, Counter] = {c: Counter() for c in class_names}
    for row in rows:
        key = f"{row['class_name']}/{Path(row['filepath']).name}"
        by_split[row["split"]].add(key)
        counts[row["class_name"]][row["split"]] += 1

    overlaps = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = by_split[left] & by_split[right]
        if shared:
            overlaps[f"{left}&{right}"] = sorted(shared)[:20]

    totals = {c: sum(counts[c].values()) for c in class_names}
    non_zero = [v for v in totals.values() if v]
    imbalance = (round(max(non_zero) / min(non_zero), 3) if non_zero else None)

    payload = {
        "split_hash": split_digest(rows),
        "seed": config.RANDOM_SEED,
        "total_images": len(rows),
        "class_totals": totals,
        "imbalance_ratio": imbalance,
        "per_class": {c: dict(counts[c]) for c in class_names},
        "split_totals": {s: len(v) for s, v in by_split.items()},
        "overlaps": overlaps,
    }

    config.ensure_dirs()
    with open(config.SPLIT_AUDIT_JSON, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    _print_split_audit(payload, class_names, counts)

    if overlaps:
        raise SystemExit(
            f"[split] FATAL: splits overlap ({', '.join(overlaps)}). "
            f"The test set is contaminated; do not report these numbers."
        )
    return payload


def _print_split_audit(payload: dict, class_names: List[str],
                       counts: Dict[str, Counter]) -> None:
    print("\n--- Split audit ---")
    print(f"  {'class':<26}{'train':>8}{'val':>7}{'test':>7}{'total':>8}")
    print("  " + "-" * 56)
    for name in class_names:
        c = counts[name]
        print(f"  {name:<26}{c['train']:>8}{c['val']:>7}{c['test']:>7}"
              f"{sum(c.values()):>8}")
    print("  " + "-" * 56)
    totals = payload["split_totals"]
    print(f"  {'TOTAL':<26}{totals['train']:>8}{totals['val']:>7}"
          f"{totals['test']:>7}{payload['total_images']:>8}")

    class_totals = payload["class_totals"]
    biggest = max(class_totals, key=class_totals.get)
    smallest = min(class_totals, key=class_totals.get)
    print(f"\n  imbalance ratio : {payload['imbalance_ratio']}x  "
          f"({biggest} {class_totals[biggest]} / "
          f"{smallest} {class_totals[smallest]})")
    print(f"  split hash      : {payload['split_hash']}")
    if payload["overlaps"]:
        print(f"  OVERLAP         : {payload['overlaps']}")
    else:
        print("  overlap         : none (train/val/test are disjoint)")


def _manifest_rows() -> List[dict]:
    if not config.SPLIT_MANIFEST.exists():
        raise FileNotFoundError(
            "Split manifest missing. Run: python data_loader.py --audit --build"
        )
    with open(config.SPLIT_MANIFEST, newline="", encoding="utf-8") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


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


def build_augmenter(seed: int | None = None) -> tf.keras.Sequential:
    """Geometric augmentation only; identical across all four models."""
    layers = tf.keras.layers
    seed = config.RANDOM_SEED if seed is None else seed
    stack = []
    if config.AUG_HORIZONTAL_FLIP:
        stack.append(layers.RandomFlip("horizontal", seed=seed))
    if config.AUG_ROTATION_FACTOR:
        stack.append(layers.RandomRotation(config.AUG_ROTATION_FACTOR,
                                           fill_mode="reflect", seed=seed))
    if config.AUG_ZOOM_FACTOR:
        stack.append(layers.RandomZoom(config.AUG_ZOOM_FACTOR,
                                       fill_mode="reflect", seed=seed))
    if config.AUG_TRANSLATION_FACTOR:
        stack.append(layers.RandomTranslation(config.AUG_TRANSLATION_FACTOR,
                                              config.AUG_TRANSLATION_FACTOR,
                                              fill_mode="reflect", seed=seed))
    return tf.keras.Sequential(stack, name="augmentation")


def make_dataset(split: str,
                 batch_size: int | None = None,
                 shuffle: bool | None = None,
                 augment: bool | None = None,
                 cache: bool = True,
                 drop_remainder: bool = False,
                 label_mode: str = "int",
                 class_weights: Dict[int, float] | None = None,
                 seed: int | None = None) -> tf.data.Dataset:
    """
    Build a `tf.data.Dataset` of float32 images in [0, 255].

    Element shape depends on the two opt-in arguments:
        default                       -> (image, int label)
        label_mode="one_hot"          -> (image, one-hot label)
        class_weights={...}           -> (image, label, sample weight)

    Both extras are applied in a single final transform *after* batching and
    augmentation, so every earlier stage keeps operating on plain 2-tuples.

    `seed` varies shuffling and augmentation only. It never touches the
    partition, which is read from the frozen manifest.
    """
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown split '{split}'.")
    if label_mode not in {"int", "one_hot"}:
        raise ValueError(f"Unknown label_mode '{label_mode}'.")

    batch_size = config.BATCH_SIZE if batch_size is None else batch_size
    shuffle = (split == "train") if shuffle is None else shuffle
    augment = (split == "train") if augment is None else augment
    seed = config.RANDOM_SEED if seed is None else seed

    class_names, buckets = load_manifest()
    paths, labels = buckets[split]
    if not paths:
        raise ValueError(f"Split '{split}' is empty.")
    num_classes = len(class_names)

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
                        seed=seed, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y),
                num_parallel_calls=AUTOTUNE)

    if augment:
        augmenter = build_augmenter(seed=seed)
        ds = ds.map(lambda x, y: (augmenter(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)

    if label_mode == "one_hot" or class_weights is not None:
        weight_table = None
        if class_weights is not None:
            missing = [i for i in range(num_classes) if i not in class_weights]
            if missing:
                raise ValueError(
                    f"class_weights is missing index/indices {missing}; "
                    f"expected one entry per class (0..{num_classes - 1})."
                )
            weight_table = tf.constant(
                [float(class_weights[i]) for i in range(num_classes)],
                dtype=tf.float32)

        def _finalize(images, targets):
            out = (tf.one_hot(targets, num_classes)
                   if label_mode == "one_hot" else targets)
            if weight_table is None:
                return images, out
            return images, out, tf.gather(weight_table, targets)

        ds = ds.map(_finalize, num_parallel_calls=AUTOTUNE)

    return ds.prefetch(AUTOTUNE)


def assert_sample_weights(dataset: tf.data.Dataset,
                          class_weights: Dict[int, float],
                          num_classes: int,
                          label_mode: str = "one_hot",
                          batches: int = 1) -> Dict[int, float]:
    """
    Verify that per-class sample weights carry the intended class weights.

    Replacing `model.fit(class_weight=...)` with a sample-weight column is the
    kind of change that fails silently: training simply proceeds with the wrong
    weighting and the loss curve looks plausible. So this pulls real batches
    and asserts, for every class actually present, that the mean sample weight
    equals the intended class weight to within 1e-6.

    Raises AssertionError on mismatch. Returns the observed means.
    """
    tolerance = 1e-6
    sums: Dict[int, float] = {}
    counts: Dict[int, int] = {}

    for element in dataset.take(batches):
        if len(element) != 3:
            raise AssertionError(
                f"Dataset yielded {len(element)}-tuples; expected 3-tuples "
                f"(image, label, sample_weight). Sample weighting is not "
                f"reaching the model."
            )
        _, targets, weights = element
        labels = (tf.argmax(targets, axis=-1) if label_mode == "one_hot"
                  else targets)
        labels = labels.numpy().astype(int)
        weights = weights.numpy().astype(float)
        for label, weight in zip(labels, weights):
            sums[label] = sums.get(label, 0.0) + weight
            counts[label] = counts.get(label, 0) + 1

    if not counts:
        raise AssertionError("No batches were produced; cannot verify weights.")

    observed: Dict[int, float] = {}
    failures: List[str] = []
    for label in sorted(counts):
        mean = sums[label] / counts[label]
        observed[label] = mean
        expected = float(class_weights[label])
        if abs(mean - expected) > tolerance:
            failures.append(
                f"class {label}: mean sample weight {float(mean):.8f} != "
                f"intended {expected:.8f} (delta {abs(mean - expected):.3e}, "
                f"n={counts[label]})"
            )

    if failures:
        raise AssertionError(
            "Sample-weight conversion is broken:\n  " + "\n  ".join(failures)
        )

    print(f"[data] Sample weights verified on {batches} batch(es): "
          f"{len(observed)}/{num_classes} classes present, all matching "
          f"their intended class weight to {tolerance:g}.")
    for label in sorted(observed):
        print(f"[data]   class {label}: weight {observed[label]:.6f}  "
              f"(n={counts[label]})")
    if len(observed) < num_classes:
        absent = sorted(set(range(num_classes)) - set(observed))
        print(f"[data]   NOTE: class(es) {absent} did not appear in the "
              f"sampled batch(es) and were not checked.")
    return observed


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
        build_manifest(root=args.data_root, run_audit=args.audit,
                       force=args.force)
    elif args.audit:
        if config.SPLIT_MANIFEST.exists():
            verify_splits()
        else:
            audit_dataset(args.data_root)
    if args.show or args.build:
        print("\n--- Table 1: Dataset Distribution ---")
        print_distribution()
    if not any([args.audit, args.build, args.show, args.clear_cache]):
        parser.print_help()


if __name__ == "__main__":
    main()
