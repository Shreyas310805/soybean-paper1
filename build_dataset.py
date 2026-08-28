"""
build_dataset.py
================
Builds the full ASDID dataset (all 8 classes, all images, imbalanced) from
Zenodo record 7304859 directly inside a Kaggle CPU notebook.

Source
------
Bevers, Schnaufer et al. (2022), "Soybean disease identification using original
field images and transfer learning with convolutional neural networks",
Computers and Electronics in Agriculture 203, 107449.
Zenodo record 7304859 / DOI 10.5061/dryad.41ns1rnj3.

The record ships one zip per class, ~43 GB in total, plus three `unused_*`
archives that were excluded from the authors' own analysis and are skipped here
so our class counts match the published ones.

Why this streams instead of extracting
--------------------------------------
A Kaggle session has ~20 GB on /kaggle/working and the full record is 43 GB, so
a naive download-then-extract cannot fit. This script processes exactly one
class at a time: download that class's zip to the scratch dir, walk its members
with `ZipFile.open()` and resize each one in memory, then delete the zip before
touching the next class. Peak disk is therefore one zip (8.4 GB worst case) plus
the growing 256x256 output (~600 MB for the whole dataset), which stays under
10 GB throughout.

Determinism
-----------
Everything is seeded at 42 and member order is sorted before any sampling, so a
capped smoke-test run selects the same images on every machine.

Hashing
-------
Duplicate detection compares the SHA-1 of the *source* member bytes, so an exact
duplicate is skipped before it is ever re-encoded. The `sha1` column in
`dataset_manifest.csv` is the digest of the *written* JPEG, matching the `path`
and `bytes` columns that describe the same output file.

Resume
------
Kaggle sessions die. Each finished class writes `.build_state.json` inside its
own folder recording the settings it was built with; a rerun skips any class
whose state file matches the current settings and whose image count still
agrees. `--force` rebuilds regardless.

CLI
---
    python build_dataset.py --list
    python build_dataset.py --cap-per-class 20
    python build_dataset.py
    python build_dataset.py --classes frogeye target_spot
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import shutil
import time
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import requests
from PIL import Image, ImageOps

ZENODO_RECORD_ID = "7304859"
ZENODO_API = "https://zenodo.org/api/records/{record_id}"

RANDOM_SEED = 42

CANONICAL_CLASSES: Tuple[str, ...] = (
    "bacterial_blight",
    "cercospora_leaf_blight",
    "downey_mildew",
    "frogeye",
    "healthy",
    "potassium_deficiency",
    "soybean_rust",
    "target_spot",
)

README_COUNTS: Dict[str, int] = {
    "bacterial_blight": 484,
    "cercospora_leaf_blight": 1598,
    "downey_mildew": 652,
    "frogeye": 1540,
    "healthy": 1632,
    "potassium_deficiency": 1034,
    "soybean_rust": 1627,
    "target_spot": 1081,
}

SKIP_ARCHIVE_PREFIXES = ("unused_",)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEFAULT_OUT_ROOT = Path("/kaggle/working/asdid_full")
DEFAULT_WORK_DIR = Path("/kaggle/temp")

TARGET_SIZE = 256
JPEG_QUALITY = 95
STATE_FILENAME = ".build_state.json"
MANIFEST_NAME = "dataset_manifest.csv"
AUDIT_NAME = "dataset_audit.json"

PROGRESS_EVERY = 100
DOWNLOAD_CHUNK = 1 << 20
DOWNLOAD_REPORT_BYTES = 200 * (1 << 20)
DOWNLOAD_ATTEMPTS = 4


def log(message: str) -> None:
    """Print immediately; Kaggle buffers stdout aggressively otherwise."""
    print(message, flush=True)


def _fmt_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return f"{seconds:.0f}s"
    minutes = seconds / 60.0
    if minutes < 90:
        return f"{minutes:.1f}m"
    return f"{minutes / 60.0:.1f}h"


def _fmt_gb(num_bytes: float) -> str:
    return f"{num_bytes / (1024 ** 3):.2f} GB"


# --------------------------------------------------------------------------- #
# Zenodo
# --------------------------------------------------------------------------- #
def fetch_record_files(record_id: str = ZENODO_RECORD_ID,
                       timeout: int = 60) -> List[dict]:
    """Enumerate the record's files through the Zenodo API."""
    url = ZENODO_API.format(record_id=record_id)
    log(f"[build] Querying {url}")
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    files = response.json().get("files", [])
    if not files:
        raise SystemExit(f"Zenodo record {record_id} returned no files.")
    log(f"[build] Record exposes {len(files)} file(s).")
    return files


def map_class_archives(files: Sequence[dict]) -> Dict[str, dict]:
    """
    Map each canonical class name to its archive entry.

    Raises rather than proceeding on a partial match: silently training on
    seven classes because one archive was renamed upstream is exactly the sort
    of failure that is invisible until the paper is written.
    """
    mapping: Dict[str, dict] = {}
    skipped: List[str] = []

    for entry in files:
        key = entry.get("key") or entry.get("filename") or ""
        if not key.lower().endswith(".zip"):
            continue
        stem = key[:-4]
        if stem.startswith(SKIP_ARCHIVE_PREFIXES):
            skipped.append(key)
            continue
        if stem in CANONICAL_CLASSES:
            mapping[stem] = {
                "key": key,
                "size": int(entry.get("size", 0)),
                "url": (entry.get("links") or {}).get("self", ""),
            }
        else:
            skipped.append(key)

    missing = [c for c in CANONICAL_CLASSES if c not in mapping]
    if missing:
        found = ", ".join(sorted(mapping)) or "none"
        raise SystemExit(
            f"Could not match {len(missing)} class archive(s): "
            f"{', '.join(missing)}.\nMatched: {found}\n"
            f"The Zenodo record layout has changed; update CANONICAL_CLASSES."
        )

    if skipped:
        log(f"[build] Skipping {len(skipped)} archive(s): {', '.join(skipped)}")
    return mapping


def print_archive_table(mapping: Dict[str, dict]) -> None:
    log("")
    log(f"  {'class':<26}{'archive':<32}{'size':>10}{'README':>9}")
    log("  " + "-" * 77)
    total = 0
    for name in CANONICAL_CLASSES:
        entry = mapping[name]
        total += entry["size"]
        log(f"  {name:<26}{entry['key']:<32}{_fmt_gb(entry['size']):>10}"
            f"{README_COUNTS[name]:>9}")
    log("  " + "-" * 77)
    log(f"  {'TOTAL':<26}{'':<32}{_fmt_gb(total):>10}"
        f"{sum(README_COUNTS.values()):>9}")
    log("")


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def download_archive(url: str, dest: Path, expected_size: int) -> Path:
    """
    Stream an archive to `dest`, resuming a partial file with a Range request.

    Zenodo drops long transfers often enough that a plain single-shot download
    of an 8 GB archive is not reliable on a Kaggle session.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and expected_size and dest.stat().st_size == expected_size:
        log(f"[build] Archive already complete: {dest.name} "
            f"({_fmt_gb(expected_size)})")
        return dest

    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if expected_size and have > expected_size:
            dest.unlink()
            have = 0

        headers = {"Range": f"bytes={have}-"} if have else {}
        mode = "ab" if have else "wb"
        if have:
            log(f"[build] Resuming {dest.name} at {_fmt_gb(have)} "
                f"(attempt {attempt}/{DOWNLOAD_ATTEMPTS})")
        else:
            log(f"[build] Downloading {dest.name} ({_fmt_gb(expected_size)}) "
                f"(attempt {attempt}/{DOWNLOAD_ATTEMPTS})")

        try:
            with requests.get(url, stream=True, headers=headers,
                              timeout=(30, 300)) as response:
                if have and response.status_code == 200:
                    have = 0
                    mode = "wb"
                response.raise_for_status()

                start = time.perf_counter()
                written = have
                next_report = written + DOWNLOAD_REPORT_BYTES
                with open(dest, mode) as handle:
                    for chunk in response.iter_content(DOWNLOAD_CHUNK):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        written += len(chunk)
                        if written >= next_report:
                            next_report = written + DOWNLOAD_REPORT_BYTES
                            elapsed = time.perf_counter() - start
                            rate = (written - have) / max(elapsed, 1e-6)
                            remaining = max(expected_size - written, 0)
                            log(f"[build]   {_fmt_gb(written)} / "
                                f"{_fmt_gb(expected_size)}  "
                                f"{rate / (1024 ** 2):.1f} MB/s  "
                                f"eta {_fmt_duration(remaining / max(rate, 1e-6))}")
        except (requests.RequestException, OSError) as exc:
            log(f"[build]   transfer failed: {exc}")
            if attempt == DOWNLOAD_ATTEMPTS:
                raise
            time.sleep(5 * attempt)
            continue

        size = dest.stat().st_size
        if not expected_size or size == expected_size:
            log(f"[build] Downloaded {dest.name} ({_fmt_gb(size)})")
            return dest
        log(f"[build]   size mismatch: got {_fmt_gb(size)}, "
            f"expected {_fmt_gb(expected_size)}")
        if attempt == DOWNLOAD_ATTEMPTS:
            raise SystemExit(f"{dest.name} incomplete after "
                             f"{DOWNLOAD_ATTEMPTS} attempts.")
        time.sleep(5 * attempt)

    raise SystemExit(f"Could not download {dest.name}.")


# --------------------------------------------------------------------------- #
# Archive walking
# --------------------------------------------------------------------------- #
def _image_members(archive: zipfile.ZipFile) -> List[zipfile.ZipInfo]:
    """Sorted image members, ignoring directories and macOS resource forks."""
    members = []
    for info in archive.infolist():
        if info.is_dir():
            continue
        name = info.filename
        if "__MACOSX" in name or Path(name).name.startswith("."):
            continue
        if Path(name).suffix.lower() not in IMAGE_SUFFIXES:
            continue
        members.append(info)
    return sorted(members, key=lambda i: i.filename)


def _read_state(class_dir: Path) -> dict | None:
    state_path = class_dir / STATE_FILENAME
    if not state_path.exists():
        return None
    try:
        with open(state_path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _class_is_current(class_dir: Path, cap: int | None,
                      size_px: int, quality: int) -> bool:
    state = _read_state(class_dir)
    if not state or not state.get("complete"):
        return False
    if (state.get("cap") != cap or state.get("size_px") != size_px
            or state.get("quality") != quality):
        return False
    on_disk = sum(1 for p in class_dir.iterdir()
                  if p.suffix.lower() == ".jpg")
    return on_disk == state.get("images")


def process_class(class_name: str,
                  archive_path: Path,
                  out_root: Path,
                  cap: int | None,
                  size_px: int,
                  quality: int,
                  seen_hashes: Dict[str, str]) -> dict:
    """
    Stream one archive into `<out_root>/<class_name>/`, resizing member by
    member. Returns the per-class audit record.
    """
    class_dir = out_root / class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    dropped: List[dict] = []
    added_hashes: Dict[str, str] = {}
    duplicates = 0

    with zipfile.ZipFile(archive_path) as archive:
        members = _image_members(archive)
        source_members = len(members)

        if cap is not None and cap < len(members):
            rng = random.Random(RANDOM_SEED)
            members = sorted(rng.sample(members, cap),
                             key=lambda i: i.filename)

        total = len(members)
        log(f"[build] {class_name}: {source_members} image member(s), "
            f"processing {total}")

        start = time.perf_counter()
        index = 0
        for position, info in enumerate(members, start=1):
            try:
                with archive.open(info) as handle:
                    payload = handle.read()
            except (zipfile.BadZipFile, OSError, RuntimeError) as exc:
                dropped.append({"member": info.filename, "class": class_name,
                                "reason": f"unreadable_member:{exc}"})
                continue

            digest = hashlib.sha1(payload).hexdigest()
            if digest in seen_hashes:
                duplicates += 1
                dropped.append({"member": info.filename, "class": class_name,
                                "reason": f"duplicate_of:{seen_hashes[digest]}"})
                continue

            try:
                with Image.open(io.BytesIO(payload)) as image:
                    image = ImageOps.exif_transpose(image)
                    image = image.convert("RGB")
                    image = image.resize((size_px, size_px),
                                         Image.Resampling.LANCZOS)
                    out_path = class_dir / f"{class_name}_{index:05d}.jpg"
                    image.save(out_path, format="JPEG", quality=quality)
            except Exception as exc:  # noqa: BLE001
                dropped.append({"member": info.filename, "class": class_name,
                                "reason": f"decode_failed:{type(exc).__name__}"})
                continue

            label = f"{class_name}/{info.filename}"
            seen_hashes[digest] = label
            added_hashes[digest] = label
            written = out_path.stat().st_size
            rows.append({
                "path": str(out_path),
                "class": class_name,
                "width": size_px,
                "height": size_px,
                "bytes": written,
                "sha1": hashlib.sha1(out_path.read_bytes()).hexdigest(),
            })
            index += 1

            if position % PROGRESS_EVERY == 0 or position == total:
                elapsed = time.perf_counter() - start
                rate = position / max(elapsed, 1e-6)
                remaining = (total - position) / max(rate, 1e-6)
                log(f"[build]   {class_name} {position}/{total}  "
                    f"kept={len(rows)}  {_fmt_duration(elapsed)} elapsed  "
                    f"eta {_fmt_duration(remaining)}")

    record = {
        "class": class_name,
        "source_members": source_members,
        "images": len(rows),
        "duplicates": duplicates,
        "dropped": len(dropped),
        "bytes": sum(r["bytes"] for r in rows),
        "cap": cap,
        "size_px": size_px,
        "quality": quality,
        "complete": True,
    }
    state = dict(record)
    state["dropped_files"] = dropped
    state["source_sha1"] = added_hashes
    with open(class_dir / STATE_FILENAME, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)

    return {"record": record, "rows": rows, "dropped": dropped}


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def write_manifest(rows: Iterable[dict], out_root: Path) -> Path:
    path = out_root / MANIFEST_NAME
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["path", "class", "width", "height",
                                "bytes", "sha1"])
        writer.writeheader()
        writer.writerows(rows)
    log(f"[build] Manifest -> {path}")
    return path


def write_audit(records: List[dict], dropped: List[dict],
                out_root: Path, elapsed: float) -> Path:
    counts = {r["class"]: r["images"] for r in records}
    non_zero = [v for v in counts.values() if v]
    mismatches = {
        name: {"built": counts[name], "readme": README_COUNTS[name]}
        for name in counts
        if name in README_COUNTS and counts[name] != README_COUNTS[name]
    }

    payload = {
        "record_id": ZENODO_RECORD_ID,
        "seed": RANDOM_SEED,
        "elapsed_sec": round(elapsed, 1),
        "total_images": sum(counts.values()),
        "total_bytes": sum(r["bytes"] for r in records),
        "per_class": records,
        "class_counts": counts,
        "imbalance_ratio": (round(max(non_zero) / min(non_zero), 3)
                            if non_zero else None),
        "readme_mismatches": mismatches,
        "duplicates_skipped": sum(r.get("duplicates", 0) for r in records),
        "dropped_total": sum(r.get("dropped", 0) for r in records),
        "dropped_detail_rows": len(dropped),
        "dropped_files": dropped,
    }
    path = out_root / AUDIT_NAME
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    log(f"[build] Audit -> {path}")
    return path


def print_summary(records: List[dict], dropped: List[dict],
                  elapsed: float) -> None:
    log("")
    log("=" * 72)
    log(" ASDID full dataset - build summary")
    log("=" * 72)
    log(f"  {'class':<26}{'built':>8}{'README':>9}{'dupes':>8}"
        f"{'dropped':>9}{'MB':>9}")
    log("  " + "-" * 67)

    total_images = 0
    total_bytes = 0
    for record in records:
        name = record["class"]
        total_images += record["images"]
        total_bytes += record["bytes"]
        expected = README_COUNTS.get(name, "-")
        flag = "" if expected in ("-", record["images"]) else "  <-- differs"
        log(f"  {name:<26}{record['images']:>8}{expected:>9}"
            f"{record['duplicates']:>8}{record['dropped']:>9}"
            f"{record['bytes'] / (1024 ** 2):>9.1f}{flag}")

    log("  " + "-" * 67)
    log(f"  {'TOTAL':<26}{total_images:>8}{sum(README_COUNTS.values()):>9}"
        f"{sum(r.get('duplicates', 0) for r in records):>8}"
        f"{sum(r.get('dropped', 0) for r in records):>9}"
        f"{total_bytes / (1024 ** 2):>9.1f}")

    counts = [r["images"] for r in records if r["images"]]
    if counts:
        log("")
        log(f"  imbalance ratio : {max(counts) / min(counts):.2f}x  "
            f"(max {max(counts)} / min {min(counts)})")
    log(f"  elapsed         : {_fmt_duration(elapsed)}")
    log(f"  on disk         : {total_bytes / (1024 ** 2):.1f} MB")
    log("=" * 72)


# --------------------------------------------------------------------------- #
def build(args: argparse.Namespace) -> None:
    random.seed(RANDOM_SEED)

    out_root = Path(args.out)
    work_dir = Path(args.work_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    mapping = map_class_archives(fetch_record_files(args.record_id))
    print_archive_table(mapping)
    if args.list:
        return

    targets = args.classes or list(CANONICAL_CLASSES)
    unknown = [c for c in targets if c not in CANONICAL_CLASSES]
    if unknown:
        raise SystemExit(f"Unknown class(es): {', '.join(unknown)}")

    session_start = time.perf_counter()
    records: List[dict] = []
    all_rows: List[dict] = []
    all_dropped: List[dict] = []
    seen_hashes: Dict[str, str] = {}

    for position, class_name in enumerate(targets, start=1):
        class_dir = out_root / class_name
        log("")
        log("-" * 72)
        log(f"  [{position}/{len(targets)}] {class_name}")
        log("-" * 72)

        if args.force and class_dir.exists():
            shutil.rmtree(class_dir)

        if _class_is_current(class_dir, args.cap_per_class,
                             args.size, args.quality):
            state = _read_state(class_dir) or {}
            log(f"[build] Already built ({state.get('images')} images) - "
                f"skipping. Use --force to rebuild.")
            restored_drops = state.get("dropped_files", [])
            restored_hashes = state.get("source_sha1", {})
            all_dropped.extend(restored_drops)
            seen_hashes.update(restored_hashes)
            records.append({k: v for k, v in state.items()
                            if k not in ("dropped_files", "source_sha1")})
            log(f"[build] Restored {len(restored_drops)} drop record(s) and "
                f"{len(restored_hashes)} source hash(es) from state.")
            for path in sorted(class_dir.glob("*.jpg")):
                all_rows.append({
                    "path": str(path),
                    "class": class_name,
                    "width": args.size,
                    "height": args.size,
                    "bytes": path.stat().st_size,
                    "sha1": hashlib.sha1(path.read_bytes()).hexdigest(),
                })
            continue

        entry = mapping[class_name]
        archive_path = work_dir / entry["key"]
        try:
            download_archive(entry["url"], archive_path, entry["size"])
            result = process_class(class_name, archive_path, out_root,
                                   args.cap_per_class, args.size,
                                   args.quality, seen_hashes)
        finally:
            if archive_path.exists() and not args.keep_archives:
                archive_path.unlink()
                log(f"[build] Removed {archive_path.name} from scratch.")

        records.append(result["record"])
        all_rows.extend(result["rows"])
        all_dropped.extend(result["dropped"])
        log(f"[build] {class_name}: kept {result['record']['images']}, "
            f"dropped {result['record']['dropped']}")

    elapsed = time.perf_counter() - session_start
    write_manifest(all_rows, out_root)
    write_audit(records, all_dropped, out_root, elapsed)
    print_summary(records, all_dropped, elapsed)

    if args.cap_per_class is not None:
        log("")
        log(f"[build] NOTE: --cap-per-class {args.cap_per_class} was set, so "
            f"this is a smoke test, not the paper dataset.")
    log("")
    log("[build] Next: click 'Save Version' before the session ends, then "
        "create the Kaggle Dataset from /kaggle/working/asdid_full "
        "(see KAGGLE_CELLS.md).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the full ASDID dataset from Zenodo record 7304859.")
    parser.add_argument("--out", default=str(DEFAULT_OUT_ROOT),
                        help="Output root holding the eight class folders.")
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR),
                        help="Scratch dir for archive downloads.")
    parser.add_argument("--record-id", default=ZENODO_RECORD_ID)
    parser.add_argument("--classes", nargs="+", default=None,
                        choices=list(CANONICAL_CLASSES),
                        help="Build a subset; useful across two sessions.")
    parser.add_argument("--cap-per-class", type=int, default=None,
                        help="Smoke test: keep at most N images per class.")
    parser.add_argument("--size", type=int, default=TARGET_SIZE)
    parser.add_argument("--quality", type=int, default=JPEG_QUALITY)
    parser.add_argument("--force", action="store_true",
                        help="Rebuild classes that are already complete.")
    parser.add_argument("--keep-archives", action="store_true",
                        help="Do not delete each zip after processing.")
    parser.add_argument("--list", action="store_true",
                        help="Print the archive mapping and exit.")
    args = parser.parse_args()

    build(args)


if __name__ == "__main__":
    main()
