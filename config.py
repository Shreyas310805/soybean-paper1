"""
config.py  (Kaggle / Colab)
===========================
Single source of truth for Paper 1:
"An Accuracy-Efficiency Study of Lightweight CNN Models for Soybean Leaf
Disease Classification".

Storage tiers
-------------
Both Kaggle and Colab hand you a disposable VM, so three tiers are kept apart:

  CODE_ROOT      the cloned repo             ephemeral   re-cloneable
  SCRATCH_ROOT   /kaggle/temp | /content/... ephemeral   tf.data cache, ckpts
  ARTIFACT_ROOT  /kaggle/working | Drive     keep-me     models, results, figures

On Kaggle, /kaggle/working survives ONLY if you click "Save Version" before the
session closes, and it is capped at 20 GB. The tf.data file cache and per-epoch
checkpoints therefore go to /kaggle/temp, which is scratch and does not count
against that cap.

Dataset discovery
-----------------
Kaggle's mount path for a user dataset is not stable across accounts and
dataset versions, so DATA_ROOT is discovered with a glob rather than hardcoded:
any `asdid_full` directory under /kaggle/input wins, and failing that, any
directory holding the eight expected class folders. `SOY_DATA_ROOT` overrides
the search entirely.

Overrides (set before running any script):
    export SOY_DATA_ROOT=/kaggle/input/asdid-full/asdid_full
    export SOY_ARTIFACT_ROOT=/kaggle/working/artifacts
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Environment detection
# --------------------------------------------------------------------------- #
def _in_colab() -> bool:
    if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU"):
        return True
    return Path("/content").is_dir() and Path("/usr/local/lib/python3").exists()


def _in_kaggle() -> bool:
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_URL_BASE"):
        return True
    return Path("/kaggle/working").is_dir()


IN_COLAB = _in_colab()
IN_KAGGLE = _in_kaggle()
DRIVE_MOUNT = Path("/content/drive/MyDrive")
DRIVE_MOUNTED = DRIVE_MOUNT.is_dir()

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
KAGGLE_TEMP = Path("/kaggle/temp")
DATASET_DIR_NAME = "asdid_full"

EXPECTED_CLASSES = (
    "bacterial_blight",
    "cercospora_leaf_blight",
    "downey_mildew",
    "frogeye",
    "healthy",
    "potassium_deficiency",
    "soybean_rust",
    "target_spot",
)

RANDOM_SEED = 42

# --------------------------------------------------------------------------- #
# Storage tiers
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent   # the cloned repo
CODE_ROOT = PROJECT_ROOT


def _looks_like_dataset_root(candidate: Path) -> bool:
    if not candidate.is_dir():
        return False
    present = sum(1 for name in EXPECTED_CLASSES if (candidate / name).is_dir())
    return present >= len(EXPECTED_CLASSES) - 2


def discover_dataset_root() -> Path | None:
    """
    Locate the mounted ASDID dataset without hardcoding Kaggle's mount shape.

    Tries an `asdid_full` directory at increasing depth first, then falls back
    to any directory that actually holds the expected class folders.
    """
    if not KAGGLE_INPUT.is_dir():
        return None

    for pattern in (DATASET_DIR_NAME,
                    f"*/{DATASET_DIR_NAME}",
                    f"*/*/{DATASET_DIR_NAME}",
                    f"*/*/*/{DATASET_DIR_NAME}"):
        for candidate in sorted(KAGGLE_INPUT.glob(pattern)):
            if candidate.is_dir():
                return candidate

    for pattern in ("*", "*/*"):
        for candidate in sorted(KAGGLE_INPUT.glob(pattern)):
            if _looks_like_dataset_root(candidate):
                return candidate
    return None


def _resolve_data_root() -> Path:
    env = os.environ.get("SOY_DATA_ROOT")
    if env:
        return Path(env)
    if IN_KAGGLE:
        discovered = discover_dataset_root()
        if discovered is not None:
            return discovered
        return KAGGLE_WORKING / DATASET_DIR_NAME
    if IN_COLAB:
        # Local SSD. Never read thousands of small images straight from Drive -
        # it is 10-50x slower than /content and will bottleneck every epoch.
        return Path("/content/data/raw")
    return PROJECT_ROOT / "data" / "raw"


def _resolve_artifact_root() -> Path:
    env = os.environ.get("SOY_ARTIFACT_ROOT")
    if env:
        return Path(env)
    if IN_KAGGLE:
        return KAGGLE_WORKING / "artifacts"
    if DRIVE_MOUNTED:
        return DRIVE_MOUNT / "soybean_paper1"
    return PROJECT_ROOT / "artifacts"


def _resolve_scratch_root() -> Path:
    if IN_KAGGLE:
        return KAGGLE_TEMP / "soybean_scratch"
    if IN_COLAB:
        return Path("/content/scratch")
    return PROJECT_ROOT / ".scratch"


DATA_ROOT = _resolve_data_root()
ARTIFACT_ROOT = _resolve_artifact_root()
SCRATCH_ROOT = _resolve_scratch_root()
PERSISTENT_ARTIFACTS = (DRIVE_MOUNTED or IN_KAGGLE
                        or bool(os.environ.get("SOY_ARTIFACT_ROOT")))

# --- persistent (Drive) ---------------------------------------------------- #
SPLITS_DIR = ARTIFACT_ROOT / "splits"
SAVED_MODELS_DIR = ARTIFACT_ROOT / "saved_models"
RUNS_DIR = ARTIFACT_ROOT / "runs"
RESULTS_DIR = ARTIFACT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
HISTORY_DIR = RESULTS_DIR / "history"
TFLITE_DIR = ARTIFACT_ROOT / "tflite_models"
LOGS_DIR = ARTIFACT_ROOT / "logs"

# --- ephemeral (local SSD) ------------------------------------------------- #
CHECKPOINT_DIR = SCRATCH_ROOT / "checkpoints"    # rewritten every epoch
CACHE_DIR = SCRATCH_ROOT / "tfdata_cache"        # never put this on Drive
EXPORT_DIR = SCRATCH_ROOT / "export"             # temp SavedModel for TFLite

# Split / audit artefacts
SPLIT_MANIFEST = SPLITS_DIR / "split_manifest.csv"
SPLIT_CSVS = {name: SPLITS_DIR / f"{name}.csv"
              for name in ("train", "val", "test")}
SPLIT_HASH_TXT = SPLITS_DIR / "split_hash.txt"
DISTRIBUTION_CSV = SPLITS_DIR / "dataset_distribution.csv"   # Table 1
AUDIT_CSV = SPLITS_DIR / "audit_report.csv"
SPLIT_AUDIT_JSON = SPLITS_DIR / "split_audit.json"
CLASS_NAMES_JSON = SPLITS_DIR / "class_names.json"

# Result artefacts
TRAINING_SUMMARY_CSV = RESULTS_DIR / "training_summary.csv"
MODEL_PERFORMANCE_CSV = RESULTS_DIR / "model_performance.csv"       # Table 2
EFFICIENCY_CSV = RESULTS_DIR / "efficiency_comparison.csv"          # Table 3
PER_CLASS_RECALL_CSV = RESULTS_DIR / "per_class_recall.csv"
QUANTIZATION_CSV = RESULTS_DIR / "quantization_results.csv"         # Table 4
BEST_MODEL_JSON = RESULTS_DIR / "best_model.json"

ALL_DIRS = [
    SPLITS_DIR, SAVED_MODELS_DIR, RUNS_DIR, RESULTS_DIR, FIGURES_DIR,
    HISTORY_DIR, TFLITE_DIR, LOGS_DIR, CHECKPOINT_DIR, CACHE_DIR, EXPORT_DIR,
]

# --------------------------------------------------------------------------- #
# Dataset preparation (guideline section 2)
# --------------------------------------------------------------------------- #
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
IMG_CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DROP_DUPLICATES = True
DROP_UNREADABLE = True

# --------------------------------------------------------------------------- #
# tf.data pipeline
# --------------------------------------------------------------------------- #
BATCH_SIZE = 32                 # comfortable for a 16 GB T4 at 224x224
SHUFFLE_BUFFER = 2048
CACHE_TO_DISK = True            # False = RAM cache; Colab gives ~12 GB, so this
                                # is only safe for small preprocessed datasets

AUG_FLIP_MODE = "horizontal_and_vertical"
AUG_ROTATION_FACTOR = 0.15
AUG_ZOOM_FACTOR = 0.15
AUG_TRANSLATION_FACTOR = 0.10
AUG_CONTRAST_FACTOR = 0.20
AUG_BRIGHTNESS_FACTOR = 0.20
AUG_VALUE_RANGE = (0.0, 255.0)

# --------------------------------------------------------------------------- #
# Training (guideline "Common Training Configuration")
# --------------------------------------------------------------------------- #
HEAD_EPOCHS = 10
FINE_TUNE_EPOCHS = 40
MAX_EPOCHS = HEAD_EPOCHS
OPTIMIZER = "adam"
HEAD_LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-4
FINE_TUNE_UNFREEZE_RATIO = 0.35
FINE_TUNE_WARMUP_EPOCHS = 2
DROPOUT_RATE = 0.2
LABEL_SMOOTHING = 0.05

MONITOR_METRIC = "val_accuracy"
MONITOR_MODE = "max"
EARLY_STOPPING_PATIENCE = 8
REDUCE_LR_PATIENCE = 2
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-6

MIXED_PRECISION = True
MIXED_PRECISION_MIN_COMPUTE = 7.0

# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
MODEL_SLUGS = [
    "mobilenetv2",
    "mobilenetv3_small",
    "efficientnetb0",
    "densenet121",
]

DISPLAY_NAMES = {
    "mobilenetv2": "MobileNetV2",
    "mobilenetv3_small": "MobileNetV3-Small",
    "efficientnetb0": "EfficientNet-B0",
    "densenet121": "DenseNet121",
}

# --------------------------------------------------------------------------- #
# Benchmarking / quantization
# --------------------------------------------------------------------------- #
LATENCY_WARMUP_RUNS = 50
LATENCY_MEASURED_RUNS = 200
TFLITE_NUM_THREADS = 1          # 1 thread ~ single-core edge behaviour
BENCH_NUM_THREADS = TFLITE_NUM_THREADS
REPRESENTATIVE_SAMPLES = 200
INT8_INPUT_DTYPE = "uint8"
INT8_OUTPUT_DTYPE = "float32"

SEEDS = (42, 1337, 2024)
TARGET_MEAN_ACCURACY = 0.95
TARGET_WORST_ACCURACY = 0.94
DIAGNOSE_FLOOR = 0.93


def ensure_dirs() -> None:
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def display_name(slug: str) -> str:
    return DISPLAY_NAMES.get(slug, slug)


def describe() -> None:
    print("=" * 68)
    print(" Paper 1 - resolved configuration")
    print("=" * 68)
    print(f"  Platform        : "
          f"{'Kaggle' if IN_KAGGLE else 'Colab' if IN_COLAB else 'local'}")
    if DRIVE_MOUNTED:
        print(f"  Drive mounted   : {DRIVE_MOUNTED}")
    print(f"  Code root       : {CODE_ROOT}")
    print(f"  Data root       : {DATA_ROOT}  (exists={DATA_ROOT.exists()})")
    print(f"  Artifacts       : {ARTIFACT_ROOT}")
    print(f"  Scratch (temp)  : {SCRATCH_ROOT}")
    print(f"  Cache to disk   : {CACHE_TO_DISK}  -> {CACHE_DIR}")
    print(f"  Input / batch   : {INPUT_SHAPE} / {BATCH_SIZE}")
    print(f"  Early stopping  : monitor={MONITOR_METRIC} ({MONITOR_MODE}), "
          f"patience={EARLY_STOPPING_PATIENCE}")
    print(f"  Models          : {', '.join(display_name(s) for s in MODEL_SLUGS)}")
    if not DATA_ROOT.exists():
        print("\n  WARNING: the dataset root does not exist. Attach the "
              "asdid-full")
        print("           dataset, or set SOY_DATA_ROOT explicitly.")
    if IN_KAGGLE:
        print("\n  REMINDER: /kaggle/working is discarded unless you click")
        print("            'Save Version' before the session ends.")
    elif not PERSISTENT_ARTIFACTS:
        print("\n  WARNING: artifacts are on ephemeral storage. Set")
        print("           SOY_ARTIFACT_ROOT or trained models will be lost.")
    print("=" * 68)


if __name__ == "__main__":
    ensure_dirs()
    describe()
