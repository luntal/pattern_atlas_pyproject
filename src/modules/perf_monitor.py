"""Throughput / GPU power measurement for the training (what 'GPU util 100 %' does not tell).

GpuSampler polls nvidia-smi (power draw, utilisation, memory) in a background thread while a training epoch runs.
It never raises: without nvidia-smi / CUDA it simply reports no GPU values.
Set RBF_PERF_MONITOR=0 to switch the sampling off (the throughput numbers are always available).
"""
import os
import shutil
import subprocess
import threading
import time

PERF_MONITOR = str(os.environ.get("RBF_PERF_MONITOR", "1")).strip().lower() not in {"0", "false", "no", "off"}


def _query(indices: list[int]) -> list[dict]:
    exe = shutil.which("nvidia-smi")
    if exe is None or not indices:
        return []
    try:
        out = subprocess.run(
            [exe, "-i", ",".join(str(i) for i in indices),
             "--query-gpu=index,power.draw,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5.0,
        ).stdout
    except Exception:
        return []
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            rows.append({"index": int(parts[0]), "power_w": float(parts[1]), "util": float(parts[2]),
                         "mem_gb": float(parts[3]) / 1024.0})
        except ValueError:
            continue
    return rows


class GpuSampler:
    def __init__(self, indices, interval: float = 1.0):
        self.indices = [int(i) for i in (indices or [])]
        self.interval = float(interval)
        self._lock = threading.Lock()
        self._samples: dict[int, list[dict]] = {}
        self._stop = threading.Event()
        self._thread = None
        self.enabled = bool(PERF_MONITOR and self.indices and shutil.which("nvidia-smi"))

    def start(self):
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _loop(self):
        while not self._stop.is_set():
            for row in _query(self.indices):
                with self._lock:
                    self._samples.setdefault(row["index"], []).append(row)
            self._stop.wait(self.interval)

    def collect(self) -> list[dict]:
        """Per-GPU averages (peak memory) since the last collect(); resets the buffer."""
        with self._lock:
            samples, self._samples = self._samples, {}
        result = []
        for index in sorted(samples):
            rows = samples[index]
            n = max(len(rows), 1)
            result.append({
                "index": index,
                "power_w": sum(r["power_w"] for r in rows) / n,
                "util": sum(r["util"] for r in rows) / n,
                "mem_gb": max(r["mem_gb"] for r in rows),
                "n": len(rows),
            })
        return result


def summarize(gpus: list[dict]) -> dict:
    if not gpus:
        return {"power_w_total": None, "util_avg": None}
    return {
        "power_w_total": sum(g["power_w"] for g in gpus),
        "util_avg": sum(g["util"] for g in gpus) / len(gpus),
    }


def format_perf(samples_per_s: float, train_time: float, gpus: list[dict]) -> str:
    text = f"Throughput: {samples_per_s:,.0f} samples/s (train phase {train_time:.1f}s)"
    if gpus:
        parts = " | ".join(
            f"GPU{g['index']} {g['power_w']:.0f} W, util {g['util']:.0f}%, {g['mem_gb']:.1f} GB" for g in gpus)
        text += " | " + parts
    return text
