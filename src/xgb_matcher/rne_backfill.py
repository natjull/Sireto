"""Resumable, partitioned backfill of the official INPI RNE differential API.

The low-level synchronizer seals one immutable interval.  This module plans a
complete interval, records every sealed partition in an append-only receipt
and can safely resume after interruption.  It deliberately delegates secret
handling and HTTPS validation to :mod:`official_source_sync`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, timedelta
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterable, Mapping

from .official_source_sync import RneSyncConfig, canonical_json, sha256_file, sync_rne


RNE_BACKFILL_SCHEMA_VERSION = "sireto-rne-backfill-v1"


@dataclass(frozen=True)
class RneBackfillPartition:
    from_exclusive: str
    to_inclusive: str

    @property
    def partition_id(self) -> str:
        return f"{self.from_exclusive}__{self.to_inclusive}"


@dataclass(frozen=True)
class RneBackfillConfig:
    sync: RneSyncConfig
    start_exclusive: str
    end_inclusive: str
    partition_days: int = 7

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RneBackfillConfig":
        sync_raw = raw.get("sync")
        if not isinstance(sync_raw, Mapping):
            raise ValueError("RNE backfill requires a sync object")
        sync = RneSyncConfig.from_dict(sync_raw)
        if sync.api is None:
            raise ValueError("RNE backfill requires the HTTPS API mode")
        start = str(raw.get("start_exclusive") or "")
        end = str(raw.get("end_inclusive") or "")
        try:
            start_day = date.fromisoformat(start)
            end_day = date.fromisoformat(end)
        except ValueError as exc:
            raise ValueError("backfill dates must use YYYY-MM-DD") from exc
        if start_day >= end_day:
            raise ValueError("start_exclusive must precede end_inclusive")
        partition_days = int(raw.get("partition_days", 7))
        if not 1 <= partition_days <= 31:
            raise ValueError("partition_days must be between 1 and 31")
        return cls(sync, start, end, partition_days)


def plan_rne_backfill(config: RneBackfillConfig) -> tuple[RneBackfillPartition, ...]:
    cursor = date.fromisoformat(config.start_exclusive)
    end = date.fromisoformat(config.end_inclusive)
    values: list[RneBackfillPartition] = []
    while cursor < end:
        upper = min(end, cursor + timedelta(days=config.partition_days))
        values.append(RneBackfillPartition(cursor.isoformat(), upper.isoformat()))
        cursor = upper
    return tuple(values)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_json(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        parent_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _load_receipt(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": RNE_BACKFILL_SCHEMA_VERSION, "partitions": []}
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != RNE_BACKFILL_SCHEMA_VERSION:
        raise ValueError("incompatible RNE backfill receipt")
    if not isinstance(value.get("partitions"), list):
        raise ValueError("malformed RNE backfill receipt")
    return value


def run_rne_backfill(
    config: RneBackfillConfig,
    *,
    output_root: Path,
    receipt_path: Path,
    sync_function: Callable[..., Path] = sync_rne,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Seal every missing partition and update a resumable receipt.

    A partition is considered complete only when its source manifest exists,
    has the expected interval and its manifest digest matches the receipt.
    """
    receipt = _load_receipt(receipt_path)
    completed = {
        str(item.get("partition_id")): item
        for item in receipt["partitions"]
        if isinstance(item, Mapping)
    }
    planned = plan_rne_backfill(config)
    for index, partition in enumerate(planned, start=1):
        existing = completed.get(partition.partition_id)
        if existing:
            manifest_path = Path(str(existing.get("manifest_path") or ""))
            if not manifest_path.is_file():
                raise ValueError(f"sealed RNE partition is missing: {manifest_path}")
            if sha256_file(manifest_path) != str(existing.get("manifest_sha256") or ""):
                raise ValueError(f"sealed RNE partition digest mismatch: {manifest_path}")
            continue
        if progress:
            progress(f"RNE {index}/{len(planned)} {partition.partition_id}")
        api = replace(
            config.sync.api,
            from_date=partition.from_exclusive,
            to_date=partition.to_inclusive,
            output_name=f"rne-formalites-{partition.partition_id}.jsonl",
        )
        output = sync_function(
            config=replace(config.sync, api=api), output_root=output_root
        )
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        provenance = manifest.get("provenance") or {}
        if (
            provenance.get("from_exclusive") != partition.from_exclusive
            or provenance.get("to_inclusive") != partition.to_inclusive
        ):
            raise ValueError("RNE sealed partition interval mismatch")
        completed[partition.partition_id] = {
            "partition_id": partition.partition_id,
            "from_exclusive": partition.from_exclusive,
            "to_inclusive": partition.to_inclusive,
            "records": int(provenance.get("records") or 0),
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": sha256_file(manifest_path),
            "build_id": str(manifest.get("build_id") or ""),
        }
        receipt = {
            "schema_version": RNE_BACKFILL_SCHEMA_VERSION,
            "start_exclusive": config.start_exclusive,
            "end_inclusive": config.end_inclusive,
            "partition_days": config.partition_days,
            "planned_partition_count": len(planned),
            "completed_partition_count": len(completed),
            "complete": len(completed) == len(planned),
            "partitions": [completed[key] for key in sorted(completed)],
        }
        receipt["logical_sha256"] = hashlib.sha256(
            canonical_json(receipt["partitions"])
        ).hexdigest()
        _atomic_write_json(receipt_path, receipt)
    return receipt_path


def iter_manifest_paths(receipt_path: Path) -> Iterable[Path]:
    receipt = _load_receipt(receipt_path)
    for item in receipt["partitions"]:
        yield Path(str(item["manifest_path"]))


__all__ = [
    "RNE_BACKFILL_SCHEMA_VERSION",
    "RneBackfillConfig",
    "RneBackfillPartition",
    "iter_manifest_paths",
    "plan_rne_backfill",
    "run_rne_backfill",
]
