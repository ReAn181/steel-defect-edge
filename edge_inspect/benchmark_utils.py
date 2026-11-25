"""benchmark_utils.py - latency stats, RAM sampling, Jetson power via tegrastats. Python 3.8+."""
from __future__ import annotations

import re
import shutil
import statistics
import subprocess
import threading
import time
from typing import Dict, List, Optional

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


def summarize(ms: List[float]) -> dict:
    if not ms:
        return {}
    s = sorted(ms)
    q = lambda f: s[min(len(s) - 1, int(round(f * (len(s) - 1))))]
    return {"n": len(s), "mean": round(statistics.mean(s), 3), "p50": round(q(0.5), 3),
            "p95": round(q(0.95), 3), "min": round(s[0], 3), "max": round(s[-1], 3)}


class TegraMonitor:
    """Background `tegrastats` sampler: RAM used (MB) and board power VDD_IN (mW). No-op off Jetson."""
    RAM = re.compile(r"RAM (\d+)/(\d+)MB")
    PWR = re.compile(r"VDD_IN (\d+)mW(?:/(\d+)mW)?")

    def __init__(self, interval_ms: int = 500):
        self.interval_ms, self.proc, self.samples = interval_ms, None, []
        self.available = shutil.which("tegrastats") is not None

    def start(self):
        if not self.available:
            return self
        self.proc = subprocess.Popen(["tegrastats", "--interval", str(self.interval_ms)],
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        threading.Thread(target=self._read, daemon=True).start()
        return self

    def _read(self):
        for line in self.proc.stdout:
            ram, pwr = self.RAM.search(line), self.PWR.search(line)
            self.samples.append({"ram_mb": int(ram.group(1)) if ram else None,
                                 "power_mw": int(pwr.group(1)) if pwr else None})

    def stop(self) -> dict:
        if self.proc:
            self.proc.terminate()
        rams = [s["ram_mb"] for s in self.samples if s["ram_mb"] is not None]
        pw = [s["power_mw"] for s in self.samples if s["power_mw"] is not None]
        return {"tegrastats": self.available, "samples": len(self.samples),
                "ram_used_mb_peak": max(rams) if rams else None,
                "ram_used_mb_mean": round(statistics.mean(rams), 1) if rams else None,
                "power_mw_mean": round(statistics.mean(pw), 1) if pw else None,
                "power_mw_peak": max(pw) if pw else None}


def process_rss_mb() -> Optional[float]:
    return round(psutil.Process().memory_info().rss / 1e6, 1) if psutil else None
