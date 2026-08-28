"""
bench.py
========
One latency harness, shared by `evaluate.py` and `quantize.py`.

Why this is a module rather than a helper in each script
--------------------------------------------------------
Table IV puts Keras FP32 and TFLite INT8 on adjacent rows. Those numbers are
only comparable if they were produced by the same protocol on the same machine
with the same thread count, and TFLite's interpreter does not default to the
same thread count TensorFlow uses. So the protocol, the thread pinning and the
machine description all live in one place and both callers go through
`measure_latency`.

Protocol
--------
Batch size 1, `LATENCY_WARMUP_RUNS` untimed iterations, then
`LATENCY_MEASURED_RUNS` timed ones, reported as median / mean / p95 (plus std,
min and max). The median is the headline: a single scheduler hiccup skews a
mean over 200 samples far more than it skews the middle of the distribution.

Every result carries the CPU model string and the thread count, because the
older Colab and Kaggle numbers in this project were never comparable and there
was nothing recorded to prove it either way.
"""

from __future__ import annotations

import platform
import re
import subprocess
import time
from pathlib import Path
from typing import Callable, Dict, Sequence

import numpy as np

import config


def cpu_model() -> str:
    """Best available human-readable CPU name for the current machine."""
    try:
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            for line in cpuinfo.read_text(encoding="utf-8").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass

    if platform.system() == "Darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                stderr=subprocess.DEVNULL).decode().strip()
        except Exception:  # noqa: BLE001
            pass

    return platform.processor() or platform.machine() or "unknown"


def cpu_count_logical() -> int:
    try:
        import os
        return os.cpu_count() or 0
    except Exception:  # noqa: BLE001
        return 0


def pin_threads(threads: int = config.BENCH_NUM_THREADS) -> int:
    """
    Pin TensorFlow's intra/inter-op pools so Keras timings match the thread
    count handed to the TFLite interpreter.

    Must run before any op executes; TensorFlow refuses the change once the
    runtime is initialised, in which case the requested count is still returned
    but a warning is printed rather than silently reporting a false pin.
    """
    import tensorflow as tf

    try:
        tf.config.threading.set_intra_op_parallelism_threads(threads)
        tf.config.threading.set_inter_op_parallelism_threads(threads)
        print(f"[bench] TensorFlow threads pinned to {threads} "
              f"(intra and inter op).")
    except RuntimeError:
        print(f"[bench] WARNING: TensorFlow runtime already initialised; "
              f"could not pin threads to {threads}. Keras and TFLite latency "
              f"may not be comparable on this run.")
    return threads


def describe_machine(threads: int = config.BENCH_NUM_THREADS) -> Dict[str, object]:
    return {
        "cpu_model": cpu_model(),
        "cpu_count_logical": cpu_count_logical(),
        "threads": int(threads),
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


def measure_latency(infer: Callable[[np.ndarray], object],
                    samples: Sequence[np.ndarray],
                    label: str = "",
                    threads: int = config.BENCH_NUM_THREADS,
                    warmup: int = config.LATENCY_WARMUP_RUNS,
                    runs: int = config.LATENCY_MEASURED_RUNS
                    ) -> Dict[str, object]:
    """
    Time `infer` on single samples and return the latency record.

    `infer` takes one already-batched sample (leading dim 1) and runs exactly
    the operation being measured, nothing else: the caller is responsible for
    quantising inputs or building tensors outside the timed region.
    """
    if not len(samples):
        raise ValueError("measure_latency needs at least one sample.")

    n = len(samples)
    for i in range(warmup):
        infer(samples[i % n])

    timings = np.empty(runs, dtype=np.float64)
    for i in range(runs):
        sample = samples[i % n]
        start = time.perf_counter()
        infer(sample)
        timings[i] = (time.perf_counter() - start) * 1000.0

    record = {
        "latency_ms_median": round(float(np.median(timings)), 3),
        "latency_ms_mean": round(float(timings.mean()), 3),
        "latency_ms_p95": round(float(np.percentile(timings, 95)), 3),
        "latency_ms_std": round(float(timings.std(ddof=1)), 3),
        "latency_ms_min": round(float(timings.min()), 3),
        "latency_ms_max": round(float(timings.max()), 3),
        "latency_warmup_runs": int(warmup),
        "latency_measured_runs": int(runs),
    }
    record.update(describe_machine(threads))
    print_latency_row(label or "model", record)
    return record


def print_latency_row(label: str, record: Dict[str, object]) -> None:
    print(f"[bench] {label:<18} "
          f"median={record['latency_ms_median']:>8.3f} ms  "
          f"mean={record['latency_ms_mean']:>8.3f} ms  "
          f"p95={record['latency_ms_p95']:>8.3f} ms  "
          f"threads={record['threads']}  "
          f"cpu={record['cpu_model']}")


def latency_columns() -> list:
    """Column order used wherever a latency record is flattened into a CSV."""
    return [
        "latency_ms_median", "latency_ms_mean", "latency_ms_p95",
        "latency_ms_std", "latency_ms_min", "latency_ms_max",
        "latency_warmup_runs", "latency_measured_runs",
        "threads", "cpu_model", "cpu_count_logical", "platform", "python",
    ]
