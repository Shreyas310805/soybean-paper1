# Soybean leaf disease classification — accuracy/efficiency study

An accuracy–efficiency study of lightweight CNNs for soybean leaf disease
classification, ending in TFLite FP32/INT8 quantization, targeting deployment
on a Raspberry Pi Zero 2 W.

Shreyas Tiwari, VIT Bhopal · co-author Riya Mathur

The pipeline runs end to end on Kaggle notebooks: one CPU notebook builds the
dataset, one GPU notebook trains and evaluates. See
[KAGGLE_CELLS.md](KAGGLE_CELLS.md) for the copy-paste cells.

---

## Dataset

Auburn Soybean Disease Image Dataset (ASDID) — Bevers et al. (2022),
*Computers and Electronics in Agriculture* 203, 107449.
Zenodo record [7304859](https://zenodo.org/records/7304859),
DOI `10.5061/dryad.41ns1rnj3`.

`build_dataset.py` downloads all eight per-class archives (~39 GiB), streams
each one member by member without ever extracting it, and writes 256×256 JPEGs.
The three `unused_*` archives are skipped so counts match the published ones.

| Class | Images | Train | Val | Test |
|---|---:|---:|---:|---:|
| bacterial_blight | 484 | 339 | 73 | 72 |
| cercospora_leaf_blight | 1598 | 1119 | 240 | 239 |
| downey_mildew | 652 | 456 | 98 | 98 |
| frogeye | 1540 | 1078 | 231 | 231 |
| healthy | 1632 | 1142 | 245 | 245 |
| potassium_deficiency | 1034 | 724 | 155 | 155 |
| soybean_rust | 1627 | 1139 | 244 | 244 |
| target_spot | 1081 | 757 | 162 | 162 |
| **Total** | **9648** | **6754** | **1448** | **1446** |

Imbalance ratio **3.37×** (healthy 1632 / bacterial_blight 484). The split is
stratified, so every class keeps its 70/15/15 proportion despite the imbalance.
Split columns are computed by `data_loader.py`; the exact per-class rows land in
`splits/dataset_distribution.csv`.

**The split is frozen: it is computed once under seed 42 and written to disk,
and the three training seeds (42 / 1337 / 2024) vary only weight initialisation
and batch shuffling — never the data partition.** `splits/split_hash.txt` holds
a SHA-256 of the partition, keyed on `(split, class, filename)` rather than
absolute paths so it is comparable across notebooks and machines.

---

## Commands, in order

### Notebook 1 — dataset build (CPU, internet ON)

```bash
git clone -b full-dataset https://github.com/Shreyas310805/soybean-paper1.git
cd soybean-paper1
```

```bash
python build_dataset.py --list
```

```bash
python build_dataset.py --classes bacterial_blight downey_mildew --cap-per-class 20
```

```bash
python build_dataset.py
```

Then save `/kaggle/working/asdid_full` as a private Kaggle Dataset named
`asdid-full`.

### Notebook 2 — training (GPU P100, internet ON)

```bash
export SOY_DATA_ROOT=$(ls -d /kaggle/input/*/asdid_full | head -1)
export SOY_ARTIFACT_ROOT=/kaggle/working/artifacts
```

```bash
python data_loader.py --audit --build
```

```bash
python run_seeds.py
```

```bash
python evaluate.py --tta
```

```bash
python quantize.py --run mobilenetv2_seed42
```

---

## Modules

| File | Role |
|---|---|
| `config.py` | All paths and hyperparameters. Discovers the Kaggle mount by glob; `SOY_DATA_ROOT` / `SOY_ARTIFACT_ROOT` override. |
| `build_dataset.py` | Zenodo → streamed resize → 256×256 JPEGs, manifest and audit. Resumable per class. |
| `data_loader.py` | Frozen stratified split, split hash, overlap audit, `tf.data` pipeline. |
| `train.py` | Two-stage transfer learning, cosine fine-tune, verified inference export. |
| `bench.py` | The one latency harness, shared by `evaluate.py` and `quantize.py`. |
| `evaluate.py` | Test-set metrics, confusion matrix, Table II and III rows. |
| `quantize.py` | TFLite FP32 / dynamic / INT8 and Table IV. |
| `run_seeds.py` | Multi-seed protocol and the pass/fail bar. |

---

## Things worth knowing before changing anything

**Report sizes from `model_inference.keras`, never `model.keras`.** Keras 3
ignores `include_optimizer=False`, and `clone_model()` inherits the compile
config, so the obvious ways to strip optimizer state do not work. `train.py`
rebuilds from the architecture config and never compiles the copy. Measured:
23.347 MB vs 9.212 MB for the same MobileNetV2 weights. `evaluate.py` refuses
to fall back to the checkpoint and exits non-zero instead.

**Augmentation lives in the `tf.data` pipeline, not in the model**, so no random
ops reach the TFLite graph being benchmarked on the Pi. It runs on raw 0–255
before the model's own normalisation, with an explicit `value_range` and a
terminating clip.

**The cosine schedule is defined in optimizer steps.** `warmup_steps +
decay_steps == steps_per_epoch × fine_tune_epochs`. Passing epoch counts would
drive the LR to zero inside the first epoch and look like a model that simply
stopped learning. Both numbers are printed at stage-2 start, and the effective
LR is logged into `history.csv` every epoch.

**Label smoothing needs one-hot targets**, so `train.py` uses
`CategoricalCrossentropy` + `CategoricalAccuracy`. `evaluate.py` and
`quantize.py` keep integer labels. Class weighting arrives as a sample-weight
column, verified against the intended weights to 1e-6 before training starts,
and is **off by default** — unweighted is the headline number.

**Early stopping monitors `val_accuracy`, not `val_loss`.** Label smoothing
raises the loss floor, so `val_loss` can climb while accuracy still improves.

**Mixed precision is refused below compute capability 7.0.** A P100 is 6.0 and
has no tensor cores; `train.py` says so and stays in float32.

**INT8 calibration uses the unaugmented train split.** Calibrating on augmented
images estimates activation ranges the model never sees at inference, and the
resulting accuracy loss is indistinguishable from genuine quantization damage.

**Latency is one protocol, one machine, one thread count** — batch size 1, 50
warmup, 200 timed, median/mean/p95, with the CPU model string and thread count
recorded in every row. TFLite does not default to TensorFlow's thread count, and
Table IV compares them on adjacent rows.

---

## Success bar

Mean test accuracy **≥ 0.95** with the worst seed **≥ 0.94**. `run_seeds.py`
exits non-zero below that and prints a ranked list of next levers rather than
accepting the result quietly.

With n = 3 the standard deviation is a weak statistic. The three individual
accuracies plus min/max/range are what belong in the paper; sd is reported at
`ddof=1` alongside, flagged for what it is.

---

## Repo contents

Code only. `asdid_full/`, `artifacts/`, `runs/`, `*.keras` and `*.tflite` are
git-ignored — the dataset and every trained model live on Kaggle, not here.
