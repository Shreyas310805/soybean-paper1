# Kaggle cells

Copy-paste cells for the two notebooks, in order. Every cell says what it does
and roughly how long it takes.

---

## Hard Kaggle constraints these cells are designed around

| Constraint | Consequence |
|---|---|
| `/kaggle/working` is **discarded** unless you click **Save Version** | A closed session wipes a finished run. There is a loud checklist at the end of each notebook. |
| `/kaggle/working` is capped at **20 GB** | The ~39 GiB record is never fully on disk. Archives stream through `/kaggle/temp`, which is scratch and does not count against the cap. |
| GPU sessions: **12 h** each, **~30 h/week** quota | `run_seeds.py` prints a runtime estimate before it starts and checks both limits. Notebook 1 must run on **CPU** so a download does not burn GPU quota. |
| **Internet must be ON** (sidebar toggle) | Without it the `git clone` and the Zenodo download both fail. |
| No interactive terminal | Everything runs as `!python script.py` from a cell. |

---

# Notebook 1 — dataset build

> ## SETTINGS FOR THIS NOTEBOOK
> **Accelerator: `None` (CPU)** — Settings → Accelerator → None
> **Internet: `ON`** — Settings → Internet → On
>
> **Do not select a GPU here.** This notebook only downloads and resizes
> images. Running it on a GPU burns your 30 h/week quota on work that never
> touches the GPU.

---

**Clone the repo. Must be the `full-dataset` branch — `main` is the old
frozen-backbone code and would run to completion looking perfectly fine while
meaning nothing.** (~5 s)

```python
!rm -rf /kaggle/working/soybean-paper1
!git clone -b full-dataset https://github.com/Shreyas310805/soybean-paper1.git /kaggle/working/soybean-paper1
```

---

**Confirm which code is actually about to execute. Halts if the branch is
wrong.** (~1 s)

```python
import subprocess

REPO = "/kaggle/working/soybean-paper1"
g = lambda *a: subprocess.check_output(["git", "-C", REPO, *a], text=True).strip()

branch, sha, subject = g("branch", "--show-current"), g("rev-parse", "HEAD"), g("log", "-1", "--format=%s")
print("=" * 72)
print(f"  BRANCH  : {branch}")
print(f"  HEAD    : {sha}")
print(f"  SUBJECT : {subject}")
print("=" * 72)
assert branch == "full-dataset", (
    f"WRONG BRANCH: {branch!r}. You are about to run the OLD frozen-backbone "
    f"code and the results will be meaningless. Re-run the clone cell.")
print("OK - this is the full-dataset code.")
```

---

**Check the libraries. Kaggle ships all of these; this prints versions rather
than installing, so a surprise version shows up now and not mid-run.** (~10 s)

```python
%cd /kaggle/working/soybean-paper1
import PIL, requests, sklearn, matplotlib, tensorflow as tf
print("tensorflow ", tf.__version__)
print("pillow     ", PIL.__version__)
print("requests   ", requests.__version__)
print("sklearn    ", sklearn.__version__)
print("matplotlib ", matplotlib.__version__)
```

---

## RUN THIS FIRST — smoke test

**Everything below the smoke test costs 1–3 hours. These two cells cost about
fifteen minutes and exercise every code path first.**

**Smoke 1 of 2: hit the Zenodo API and confirm all eight class archives resolve.
No download.** (~10 s)

```python
!python build_dataset.py --list
```

Expect eight classes, three `unused_*` skipped, and a total of 9,648 README
images. If a class fails to match, this raises rather than silently building a
seven-class dataset.

---

**Smoke 2 of 2: the whole chain on two real classes, 20 images each — download,
resume, stream, resize, audit, split, train both stages, export, evaluate,
quantize.**

**A note on why this is not `--cap-per-class 20` alone:** the cap limits how
many images are *kept*, but members can only be read after the archive is on
disk, so a capped run across all eight classes still downloads all ~39 GiB.
Pairing it with `--classes` on the two smallest archives (2.65 + 2.73 GiB) is
what makes this minutes instead of hours. (~10–15 min, mostly download)

```python
import os
os.environ["SOY_DATA_ROOT"] = "/kaggle/working/smoke/asdid_full"
os.environ["SOY_ARTIFACT_ROOT"] = "/kaggle/working/smoke/artifacts"

!python build_dataset.py --classes bacterial_blight downey_mildew --cap-per-class 20 --out /kaggle/working/smoke/asdid_full
!python data_loader.py --audit --build --force
!python train.py --models mobilenetv2 --seed 42 --head-epochs 1 --fine-tune-epochs 1 --batch-size 8 --no-mixed-precision
!python evaluate.py --runs mobilenetv2_seed42 --skip-latency
!python quantize.py --run mobilenetv2_seed42 --samples 16 --limit 12
```

What to look for before continuing:

- `Split hash:` printed, and `overlap : none (train/val/test are disjoint)`
- `warmup_steps` and `decay_steps` printed as **step counts**, not `1`
- `max |diff| = 0.00e+00` on the inference export check
- `model.keras` clearly **larger** than `model_inference.keras`
- `Calibrating INT8 on 16 UNAUGMENTED train images`
- `table4.csv` written with an `input=uint8, output=float32` row

Then clear the smoke environment before the real build:

```python
import os, shutil
shutil.rmtree("/kaggle/working/smoke", ignore_errors=True)
for k in ("SOY_DATA_ROOT", "SOY_ARTIFACT_ROOT"):
    os.environ.pop(k, None)
print("Smoke artefacts removed.")
```

---

## The real build

**Download and resize all eight classes. One archive at a time; each is deleted
before the next starts, so peak disk stays near 8 GiB.** (~1.5–3 h, download
bound)

```python
!python build_dataset.py
```

Progress prints every 100 images with an ETA. If the session dies, re-run the
same cell — completed classes are skipped via their `.build_state.json`.

---

**Read the audit: per-class counts against the published ones, duplicates,
unreadable files, imbalance ratio, total size.** (~2 s)

```python
import json
audit = json.load(open("/kaggle/working/asdid_full/dataset_audit.json"))
print("total images   :", audit["total_images"], "(expected 9648)")
print("imbalance ratio:", audit["imbalance_ratio"], "(expected ~3.37)")
print("duplicates     :", audit["duplicates_skipped"])
print("dropped        :", audit["dropped_total"])
print("mismatches vs README:", json.dumps(audit["readme_mismatches"], indent=2))
for k, v in sorted(audit["class_counts"].items()):
    print(f"  {k:<26}{v:>6}")
```

---

**Confirm what is on disk before saving.** (~5 s)

```python
!du -sh /kaggle/working/asdid_full
!ls /kaggle/working/asdid_full
!df -h /kaggle/working | tail -1
```

---

## SAVE — do this before closing notebook 1

> # ⛔ STOP. `/kaggle/working` IS DELETED WHEN THIS SESSION CLOSES.
>
> A finished 3-hour build is gone the moment you close the tab without saving.
>
> **1. Save the notebook output**
> Top right → **Save Version** → **Save & Run All (Commit)** → wait for it to
> finish. "Quick Save" does *not* reliably persist `/kaggle/working`.
>
> **2. Turn the output into a reusable dataset**
> Open the completed version → **Output** tab → **New Dataset**
> - Title: `asdid-full`
> - Visibility: **Private**
> - Wait for "Your dataset is being created"
>
> **3. Exact paths that must be inside it**
> ```
> /kaggle/working/asdid_full/                     <- the 8 class folders
> /kaggle/working/asdid_full/dataset_manifest.csv
> /kaggle/working/asdid_full/dataset_audit.json
> ```
>
> **4. Confirm before you close the tab**
> The dataset appears under `Your Work → Datasets → asdid-full` and shows
> ~9,648 files. If it does not, the build is not saved.

In notebook 2 it mounts at roughly
`/kaggle/input/asdid-full/asdid_full/<class>/`. You do not need to hardcode
that — `config.discover_dataset_root()` globs for it, and falls back to any
directory holding the expected class folders.

---
---

# Notebook 2 — training

> ## SETTINGS FOR THIS NOTEBOOK
> **Accelerator: `GPU P100`** — Settings → Accelerator → GPU P100
> **Internet: `ON`** — Settings → Internet → On
>
> Also: **Add Data → Your Datasets → `asdid-full`** before running anything.

---

**Clone. Same branch check — `main` is the old code.** (~5 s)

```python
!rm -rf /kaggle/working/soybean-paper1
!git clone -b full-dataset https://github.com/Shreyas310805/soybean-paper1.git /kaggle/working/soybean-paper1
```

---

**Confirm which code is executing. Halts on the wrong branch.** (~1 s)

```python
import subprocess

REPO = "/kaggle/working/soybean-paper1"
g = lambda *a: subprocess.check_output(["git", "-C", REPO, *a], text=True).strip()

branch, sha, subject = g("branch", "--show-current"), g("rev-parse", "HEAD"), g("log", "-1", "--format=%s")
print("=" * 72)
print(f"  BRANCH  : {branch}")
print(f"  HEAD    : {sha}")
print(f"  SUBJECT : {subject}")
print("=" * 72)
assert branch == "full-dataset", (
    f"WRONG BRANCH: {branch!r}. This would train the OLD frozen-backbone "
    f"code and waste a GPU session. Re-run the clone cell.")
print("OK - this is the full-dataset code.")
```

Record that SHA next to your results. It is also written into every run's
`config_snapshot.json`.

---

**Point the scripts at the mounted dataset and at persistent output.** (~1 s)

```python
import os, glob
%cd /kaggle/working/soybean-paper1

found = glob.glob("/kaggle/input/**/asdid_full", recursive=True)
print("discovered dataset roots:", found)
assert found, "asdid-full is not attached. Add Data -> Your Datasets -> asdid-full."

os.environ["SOY_DATA_ROOT"] = found[0]
os.environ["SOY_ARTIFACT_ROOT"] = "/kaggle/working/artifacts"
print("SOY_DATA_ROOT     =", os.environ["SOY_DATA_ROOT"])
print("SOY_ARTIFACT_ROOT =", os.environ["SOY_ARTIFACT_ROOT"])
```

---

**Confirm the GPU is real and see the resolved configuration.** (~15 s)

```python
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
!python -c "import tensorflow as tf; print('TF', tf.__version__); print('GPUs', tf.config.list_physical_devices('GPU'))"
!python config.py
```

A P100 is compute capability 6.0 and has no tensor cores, so `train.py` will
report that it is **staying in float32** even though `MIXED_PRECISION = True`.
That is intended. On a T4 it enables float16 instead.

---

**Build the frozen split and audit it: per-class counts, imbalance ratio, and a
hard zero-overlap check.** (~2–4 min, hashes ~9.6k files unless the build
manifest is reused)

```python
!python data_loader.py --audit --build
```

Copy the printed **split hash** somewhere. It is keyed on
`(split, class, filename)`, not absolute paths, so it is comparable across
notebooks and proves two runs used the same partition.

---

**Train all three seeds and evaluate each. Prints a runtime estimate first —
check it against the 12 h session limit before letting it run.** (~2.5–4 h)

```python
!python run_seeds.py
```

- Writes `results/seed_summary.csv` after **every** seed, so a killed session
  keeps whatever finished.
- Re-running skips seeds that already have a `done.json`.
- Exits non-zero and prints a ranked diagnosis if mean < 0.95 or worst < 0.94.

---

**Optional: the weighted comparison run. Unweighted is the headline number;
this is the reported comparison.** (~2.5–4 h — only if you have quota)

```python
!python run_seeds.py --class-weight --seeds 42
```

---

**Full evaluation with latency, plus the separate TTA row.** (~10 min)

```python
!python evaluate.py --tta
```

---

**Quantize the best run and write Table IV.** (~15 min)

```python
import json, pathlib
best = json.load(open("/kaggle/working/artifacts/results/evaluation_full.json"))
best_run = max(best, key=lambda r: r["accuracy"])["run_name"]
print("best run:", best_run)
```

```python
!python quantize.py --run {best_run}
```

---

**Collect everything into one archive so a single download rescues the run.**
(~30 s)

```python
import shutil, os
from pathlib import Path

ART = Path("/kaggle/working/artifacts")
STAGE = Path("/kaggle/working/_bundle")
shutil.rmtree(STAGE, ignore_errors=True)
STAGE.mkdir(parents=True)

for name in ("runs", "results", "splits", "tflite_models"):
    src = ART / name
    if src.exists():
        shutil.copytree(src, STAGE / name)
        print(f"staged {name}")
    else:
        print(f"MISSING {name}")

out = shutil.make_archive("/kaggle/working/soybean_run_bundle", "zip", STAGE)
print("\narchive:", out, f"({os.path.getsize(out)/1024**2:.1f} MB)")
```

---

## SAVE — do this before closing notebook 2

> # ⛔ STOP. THIS IS WHERE A COMPLETED RUN GETS LOST.
>
> Closing the tab without saving deletes every trained model, every table and
> every figure. It has already happened once on this project.
>
> **1. Download the bundle right now**
> Right panel → **Output** → `soybean_run_bundle.zip` → download. Do this
> *before* Save Version; it is the copy that does not depend on Kaggle.
>
> **2. Then Save Version**
> Top right → **Save Version** → **Save & Run All (Commit)** → wait for
> completion.
>
> **3. Exact paths that must survive**
> ```
> /kaggle/working/soybean_run_bundle.zip                  <- download this
> /kaggle/working/artifacts/results/seed_summary.csv      <- the multi-seed result
> /kaggle/working/artifacts/results/table2_row.csv        <- Table II
> /kaggle/working/artifacts/results/table3_row.csv        <- Table III
> /kaggle/working/artifacts/results/table3_row_extended.csv
> /kaggle/working/artifacts/results/table4.csv            <- Table IV
> /kaggle/working/artifacts/results/per_class_metrics.csv
> /kaggle/working/artifacts/splits/split_hash.txt         <- proves the partition
> /kaggle/working/artifacts/splits/split_audit.json
> /kaggle/working/artifacts/runs/<run>/model_inference.keras
> /kaggle/working/artifacts/runs/<run>/config_snapshot.json
> /kaggle/working/artifacts/runs/<run>/confusion_matrix.png
> /kaggle/working/artifacts/runs/<run>/training_curves.png
> /kaggle/working/artifacts/tflite_models/<run>/*.tflite
> ```
>
> **4. Confirm**
> The zip is in your Downloads folder and opens. Only then close the tab.

---

## If the session dies mid-run

Nothing needs redoing from scratch:

| Died during | Recovery |
|---|---|
| `build_dataset.py` | Re-run the same cell. Completed classes are skipped. |
| `run_seeds.py` | Re-run it. Seeds with a `done.json` are skipped; `seed_summary.csv` already holds the finished ones. |
| Anything after training | Re-run just that script against the existing run directory. |

The one thing that cannot be recovered is a session closed without **Save
Version**.
