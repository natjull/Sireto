"""Lightweight timing instrumentation for pipeline stages.

Usage:
    timer = PipelineTimer()
    with timer.stage("tfidf_fit"):
        vectorizer.fit_transform(texts)
    with timer.stage("feature_compute"):
        features = make_features(...)
    timer.log_summary()
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, Iterator, List

import numpy as np

_logger = logging.getLogger(__name__)


class PipelineTimer:
    """Collects per-stage timings and reports p50/p95/p99 statistics."""

    def __init__(self) -> None:
        self._timings: Dict[str, List[float]] = defaultdict(list)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        self._timings[name].append(elapsed)

    def record(self, name: str, elapsed: float) -> None:
        """Manually record a timing (e.g. from external measurement)."""
        self._timings[name].append(elapsed)

    def summary(self) -> Dict[str, Dict[str, float]]:
        result: Dict[str, Dict[str, float]] = {}
        for name, times in sorted(self._timings.items()):
            arr = np.array(times)
            result[name] = {
                "count": float(len(arr)),
                "total_s": float(arr.sum()),
                "mean_s": float(arr.mean()),
                "p50_s": float(np.percentile(arr, 50)),
                "p95_s": float(np.percentile(arr, 95)),
                "p99_s": float(np.percentile(arr, 99)),
            }
        return result

    def log_summary(self, prefix: str = "") -> None:
        tag = f"[{prefix}] " if prefix else ""
        for name, stats in self.summary().items():
            _logger.info(
                "%s%-28s  n=%-5d  total=%7.1fs  p50=%7.3fs  p95=%7.3fs",
                tag,
                name,
                int(stats["count"]),
                stats["total_s"],
                stats["p50_s"],
                stats["p95_s"],
            )

    def reset(self) -> None:
        self._timings.clear()


# Global timer instance for convenience (optional — callers can create their own)
_GLOBAL_TIMER: PipelineTimer | None = None


def get_global_timer() -> PipelineTimer:
    global _GLOBAL_TIMER
    if _GLOBAL_TIMER is None:
        _GLOBAL_TIMER = PipelineTimer()
    return _GLOBAL_TIMER


__all__ = ["PipelineTimer", "get_global_timer"]
