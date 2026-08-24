from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from xgb_matcher.retrieval_fold_fused_physical_benchmark_v12 import (
    PhysicalBenchmarkInvariantError,
    current_national_worker_guard_v12,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "config/retrieval_all_human_v12_sharded_execution_plan.json"
GIB = 1024**3


class _Sequence:
    def __init__(self, values: list[int]) -> None:
        self._values = iter(values)

    def __call__(self) -> int:
        return next(self._values)


def _memory(used: int):
    return lambda: SimpleNamespace(used=used)


def test_current_guard_authorizes_w3_only_under_ram_and_zero_swap() -> None:
    slept: list[float] = []
    times = _Sequence([1_000, 1_005])
    record = current_national_worker_guard_v12(
        3,
        PLAN,
        memory_reader=_memory(13 * GIB),
        swap_reader=_Sequence([100, 100]),
        sleeper=slept.append,
        clock=times,
    )
    assert record["status"] == "PASS_W3"
    assert record["selected_maximum_concurrent_workers"] == 3
    assert record["fixed_shard_count"] == 3
    assert record["memory_used_peak_bytes"] == 13 * GIB
    assert record["swapout_delta_bytes"] == 0
    assert record["swapout_page_size_bytes"] == 1
    assert record["observation_started_at_unix_seconds"] == 1_000
    assert record["observation_finished_at_unix_seconds"] == 1_005
    assert slept == [5.0]
    unsigned = dict(record)
    digest = unsigned.pop("record_sha256")
    import hashlib

    canonical = json.dumps(
        unsigned,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    assert digest == hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def test_current_guard_falls_back_to_w2_when_current_machine_ram_is_too_high() -> None:
    record = current_national_worker_guard_v12(
        3,
        PLAN,
        memory_reader=_memory(14 * GIB + 1),
        swap_reader=_Sequence([0, 0]),
        sleeper=lambda _: None,
    )
    assert record["status"] == "PASS_W2_FALLBACK"
    assert record["selected_maximum_concurrent_workers"] == 2
    assert record["selection_reason"] == "W3_MEMORY_LIMIT_EXCEEDED_FALLBACK_W2"


def test_current_guard_allows_w2_above_w3_ram_limit_but_never_swapout() -> None:
    record = current_national_worker_guard_v12(
        2,
        PLAN,
        memory_reader=_memory(16 * GIB),
        swap_reader=_Sequence([0, 0]),
        sleeper=lambda _: None,
    )
    assert record["status"] == "PASS_W2_FALLBACK"
    assert record["selected_maximum_concurrent_workers"] == 2
    swapped = current_national_worker_guard_v12(
        2,
        PLAN,
        memory_reader=_memory(16 * GIB),
        swap_reader=_Sequence([10, 11]),
        sleeper=lambda _: None,
    )
    assert swapped["selected_maximum_concurrent_workers"] == 2
    assert swapped["status"] == "PASS_W2_FALLBACK"


def test_current_guard_falls_back_to_w2_when_measurement_is_unavailable() -> None:
    def unavailable():
        raise OSError("fixture")

    record = current_national_worker_guard_v12(
        3,
        PLAN,
        memory_reader=unavailable,
        swap_reader=unavailable,
        sleeper=lambda _: None,
    )
    assert record["measurement_valid"] is False
    assert record["selected_maximum_concurrent_workers"] == 2
    assert record["selection_reason"] == "W3_MEASUREMENT_UNAVAILABLE_FALLBACK_W2"


@pytest.mark.parametrize("worker_count", [0, 1, 4])
def test_current_guard_rejects_unregistered_worker_counts(worker_count: int) -> None:
    with pytest.raises(PhysicalBenchmarkInvariantError, match="policy drifted"):
        current_national_worker_guard_v12(
            worker_count,
            PLAN,
            memory_reader=_memory(1),
            swap_reader=_Sequence([0, 0]),
            sleeper=lambda _: None,
        )
