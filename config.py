"""
config.py  (Google Colab edition)
=================================
Single source of truth for Paper 1:
"An Accuracy-Efficiency Study of Lightweight CNN Models for Soybean Leaf
Disease Classification".

Colab-specific design
---------------------
A Colab VM is disposable. Everything under /content disappears when the runtime
recycles, which on a free T4 can happen mid-training. So three storage tiers are
kept apart:

  CODE_ROOT      /content/<repo>            ephemeral   git clone, re-cloneable
  SCRATCH_ROOT   /content/scratch           ephemeral   tf.data cache, ckpts
  ARTIFACT_ROOT  /content/drive/MyDrive/... PERSISTENT  models, results, figures

Trained models, tables and figures land on Drive, so a disconnect costs you the
current model and nothing else - relaunch with `python train.py --skip-existing`
and it picks up where it stopped.

The tf.data cache and per-epoch checkpoints deliberately stay on local disk:
Drive is a network mount, and writing a checkpoint to it every epoch stalls
training for seconds at a time.

Overrides (set before running any script):
    export SOY_DATA_ROOT=/content/data/raw
    export SOY_ARTIFACT_ROOT=/content/drive/MyDrive/soybean_paper1
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


IN_COLAB = _in_colab()
DRIVE_MOUNT = Path("/content/drive/MyDrive")
DRIVE_MOUNTED = DRIVE_MOUNT.is_dir()

RANDOM_SEED = 42

# --------------------------------------------------------------------------- #
# Storage tiers
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent   # the cloned repo
CODE_ROOT = PROJECT_ROOT


def _resolve_data_root() -> Path:
    env = os.environ.get("SOY_DATA_ROOT")
    if env:
        return Path(env)
    if IN_COLAB:
        # Local SSD. Never read thousands of small images straight from Drive -
        # it is 10-50x slower than /content and will bottleneck every epoch.
        return Path("/content/data/raw")
    return PROJECT_ROOT / "data" / "raw"


def _resolve_artifact_root() -> Path:
    env = os.environ.get("SOY_ARTIFACT_ROOT")
    if env:
        return Path(env)
    if DRIVE_MOUNTED:
        return DRIVE_MOUNT / "soybean_paper1"
    return PROJECT_ROOT / "artifacts"


def _resolve_scratch_root() -> Path:
    if IN_COLAB:
        return Path("/content/scratch")
    return PROJECT_ROOT / ".scratch"


DATA_ROOT = _resolve_data_root()
ARTIFACT_ROOT = _resolve_artifact_root()
SCRATCH_ROOT = _resolve_scratch_root()
PERSISTENT_ARTIFACTS = DRIVE_MOUNTED or bool(os.environ.get("SOY_ARTIFACT_ROOT"))

# --- persistent (Drive) ---------------------------------------------------- #
SPLITS_DIR = ARTIFACT_ROOT / "splits"
SAVED_MODELS_DIR = ARTIFACT_ROOT / "saved_models"
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
DISTRIBUTION_CSV = SPLITS_DIR / "dataset_distribution.csv"   # Table 1
AUDIT_CSV = SPLITS_DIR / "audit_report.csv"
CLASS_NAMES_JSON = SPLITS_DIR / "class_names.json"

# Result artefacts
TRAINING_SUMMARY_CSV = RESULTS_DIR / "training_summary.csv"
MODEL_PERFORMANCE_CSV = RESULTS_DIR / "model_performance.csv"       # Table 2
EFFICIENCY_CSV = RESULTS_DIR / "efficiency_comparison.csv"          # Table 3
PER_CLASS_RECALL_CSV = RESULTS_DIR / "per_class_recall.csv"
QUANTIZATION_CSV = RESULTS_DIR / "quantization_results.csv"         # Table 4
BEST_MODEL_JSON = RESULTS_DIR / "best_model.json"

ALL_DIRS = [
    SPLITS_DIR, SAVED_MODELS_DIR, RESULTS_DIR, FIGURES_DIR, HISTORY_DIR,
    TFLITE_DIR, LOGS_DIR, CHECKPOINT_DIR, CACHE_DIR, EXPORT_DIR,
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

AUG_HORIZONTAL_FLIP = True
AUG_ROTATION_FACTOR = 0.08
AUG_ZOOM_FACTOR = 0.10
AUG_TRANSLATION_FACTOR = 0.10

# --------------------------------------------------------------------------- #
# Training (guideline "Common Training Configuration")
# --------------------------------------------------------------------------- #
MAX_EPOCHS = 15
OPTIMIZER = "adam"
HEAD_LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5
FINE_TUNE_EPOCHS = 0
FINE_TUNE_UNFREEZE_RATIO = 0.25
DROPOUT_RATE = 0.2

MONITOR_METRIC = "val_loss"
MONITOR_MODE = "min"
EARLY_STOPPING_PATIENCE = 4
REDUCE_LR_PATIENCE = 2
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-6

MIXED_PRECISION = False         # opt-in; see the warning in train.py

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
LATENCY_WARMUP_RUNS = 10
LATENCY_MEASURED_RUNS = 100
TFLITE_NUM_THREADS = 1          # 1 thread ~ single-core edge behaviour
REPRESENTATIVE_SAMPLES = 200
INT8_INPUT_DTYPE = "uint8"


def ensure_dirs() -> None:
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def display_name(slug: str) -> str:
    return DISPLAY_NAMES.get(slug, slug)


def describe() -> None:
    print("=" * 68)
    print(" Paper 1 - resolved configuration")
    print("=" * 68)
    print(f"  Colab detected  : {IN_COLAB}")
    print(f"  Drive mounted   : {DRIVE_MOUNTED}")
    print(f"  Code root       : {CODE_ROOT}")
    print(f"  Data root       : {DATA_ROOT}  (exists={DATA_ROOT.exists()})")
    print(f"  Artifacts       : {ARTIFACT_ROOT}")
    print(f"  Scratch (temp)  : {SCRATCH_ROOT}")
    print(f"  Input / batch   : {INPUT_SHAPE} / {BATCH_SIZE}")
    print(f"  Max epochs      : {MAX_EPOCHS}  (early stop patience "
          f"{EARLY_STOPPING_PATIENCE})")
    print(f"  Models          : {', '.join(display_name(s) for s in MODEL_SLUGS)}")
    if not PERSISTENT_ARTIFACTS:
        print("\n  WARNING: artifacts are on ephemeral storage. Mount Drive")
        print("           (python colab_setup.py --mount-drive) or trained")
        print("           models will be lost when the runtime recycles.")
    print("=" * 68)


if __name__ == "__main__":
    ensure_dirs()
    describe()
