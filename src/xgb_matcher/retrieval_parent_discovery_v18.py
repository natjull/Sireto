"""Truth-free primitives for the v18 SIRET-first parent discovery sprint.

The module deliberately stops at the raw SIREN union.  It never opens human
labels, never truncates a SIRET to manufacture a parent, and never trains a
model.  A retrieved establishment may expose its official SIREN field; that
field is used only as a grouping key.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq


GENERATION_SCHEMA_VERSION = "sireto-parent-discovery-v18-generation-1"
EVALUATION_SCHEMA_VERSION = "sireto-parent-discovery-v18-evaluation-1"
ALLOWED_FOLDS = frozenset({2, 3, 4})

SIGNAL_SCHEMA = pa.schema(
    [
        pa.field("query_id", pa.string(), nullable=False),
        pa.field("fold", pa.int8(), nullable=False),
        pa.field("channel", pa.string(), nullable=False),
        pa.field("channel_rank", pa.int32(), nullable=False),
        pa.field("siren", pa.string(), nullable=False),
        pa.field("source_siret", pa.string(), nullable=False),
        pa.field("score", pa.float64(), nullable=False),
        pa.field("seed_rank", pa.int32()),
        pa.field("relation_direction", pa.string(), nullable=False),
        pa.field("relation_type", pa.string(), nullable=False),
    ]
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_siren(value: Any) -> str:
    text = "".join(character for character in str(value or "") if character.isdigit())
    return text if len(text) == 9 else ""


def normalize_siret(value: Any) -> str:
    text = "".join(character for character in str(value or "") if character.isdigit())
    return text if len(text) == 14 else ""


def load_query_selection(path: Path, *, expected_count: int | None = None) -> tuple[tuple[str, int], ...]:
    table = pq.read_table(path)
    if set(table.column_names) != {"query_id", "fold"}:
        raise ValueError("v18 generator accepts only query_id and fold")
    rows = tuple((str(row["query_id"]), int(row["fold"])) for row in table.to_pylist())
    if not rows or len(rows) != len({query_id for query_id, _fold in rows}):
        raise ValueError("v18 query population must be non-empty and unique")
    if any(not query_id or fold not in ALLOWED_FOLDS for query_id, fold in rows):
        raise ValueError("v18 opened a forbidden query or fold")
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError("v18 query population drifted")
    return rows


@dataclass(frozen=True)
class SiretChannelSpec:
    name: str
    backend_channel: str
    limit: int

    def __post_init__(self) -> None:
        if not self.name or not self.backend_channel:
            raise ValueError("v18 SIRET channel identity is empty")
        if not 1 <= self.limit <= 1024:
            raise ValueError("v18 SIRET channel limit must be within 1..1024")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "SiretChannelSpec":
        return cls(
            name=str(raw["name"]),
            backend_channel=str(raw["backend_channel"]),
            limit=int(raw["limit"]),
        )


def collapse_siret_hits_to_parent_rows(
    *,
    query_id: str,
    fold: int,
    channel: str,
    hits: Sequence[Any],
) -> list[dict[str, Any]]:
    """Collapse ranked SIRET hits by their stored official SIREN.

    The first occurrence wins because backends already return a deterministic
    descending score order.  No parent is derived from ``siret[:9]``.
    """

    best: dict[str, tuple[str, float]] = {}
    for hit in hits:
        record = hit.record
        siren = normalize_siren(getattr(record, "siren", ""))
        siret = normalize_siret(getattr(record, "siret", ""))
        if not siren or not siret or siren in best:
            continue
        best[siren] = (siret, float(hit.score))
    ordered = sorted(best.items(), key=lambda item: (-item[1][1], item[0]))
    return [
        {
            "query_id": query_id,
            "fold": int(fold),
            "channel": channel,
            "channel_rank": rank,
            "siren": siren,
            "source_siret": siret,
            "score": score,
            "seed_rank": None,
            "relation_direction": "",
            "relation_type": "SIRET_NATIVE",
        }
        for rank, (siren, (siret, score)) in enumerate(ordered, start=1)
    ]


def validate_signal_rows(rows: Iterable[Mapping[str, Any]]) -> None:
    seen: set[tuple[str, str, str]] = set()
    expected_rank: dict[tuple[str, str], int] = {}
    count = 0
    for row in rows:
        count += 1
        query_id = str(row["query_id"])
        fold = int(row["fold"])
        channel = str(row["channel"])
        siren = normalize_siren(row["siren"])
        rank = int(row["channel_rank"])
        if not query_id or fold not in ALLOWED_FOLDS or not channel or not siren:
            raise ValueError("invalid v18 parent signal")
        key = (query_id, channel, siren)
        if key in seen:
            raise ValueError("duplicate query/channel/SIREN v18 signal")
        seen.add(key)
        group = (query_id, channel)
        wanted = expected_rank.get(group, 1)
        if rank != wanted:
            raise ValueError("v18 channel ranks are not contiguous")
        expected_rank[group] = wanted + 1
    if count == 0:
        raise ValueError("v18 emitted no parent signal")


def file_record(path: Path, *, parquet: bool = True) -> dict[str, Any]:
    record: dict[str, Any] = {
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if parquet:
        metadata = pq.ParquetFile(path).metadata
        record.update(rows=metadata.num_rows, row_groups=metadata.num_row_groups)
    return record


def write_manifest(output_dir: Path, payload: Mapping[str, Any], files: Sequence[str]) -> Path:
    root = Path(output_dir)
    records = {name: file_record(root / name) for name in files}
    manifest = {
        **dict(payload),
        "schema_version": GENERATION_SCHEMA_VERSION,
        "truth_columns_opened": False,
        "positive_injection": False,
        "fold0_opened": False,
        "fold1_opened": False,
        "model_trained": False,
        "files": records,
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


__all__ = [
    "ALLOWED_FOLDS",
    "EVALUATION_SCHEMA_VERSION",
    "GENERATION_SCHEMA_VERSION",
    "SIGNAL_SCHEMA",
    "SiretChannelSpec",
    "collapse_siret_hits_to_parent_rows",
    "file_record",
    "load_query_selection",
    "normalize_siren",
    "normalize_siret",
    "sha256_file",
    "validate_signal_rows",
    "write_manifest",
]
