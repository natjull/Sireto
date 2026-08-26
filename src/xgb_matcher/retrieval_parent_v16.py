"""Label-blind native-SIREN retrieval workbench for the v16 parent sprint.

Only documents whose indexed ``document_type`` is ``siren`` are queried.  In
particular, this module never derives a parent by truncating a retrieved SIRET.
Human truth is intentionally absent; evaluation lives in the sibling
``retrieval_parent_v16_evaluation`` module.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from .hierarchical_retrieval import RetrievalQuery, normalize_code


GENERATION_SCHEMA_VERSION = "sireto-parent-v16-native-workbench-generation-1"
QUERY_SCHEMA = pa.schema([pa.field("query_id", pa.string()), pa.field("fold", pa.int8())])
SIGNAL_SCHEMA = pa.schema(
    [
        pa.field("query_id", pa.string()),
        pa.field("fold", pa.int8()),
        pa.field("backend", pa.string()),
        pa.field("family", pa.string()),
        pa.field("channel", pa.string()),
        pa.field("channel_rank", pa.int16()),
        pa.field("siren", pa.string()),
        pa.field("score", pa.float64()),
    ]
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ParentChannelSpec:
    name: str
    backend: str
    family: str
    mode: str
    fields: tuple[str, ...]
    query_value: str
    limit: int
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.mode not in {"exact", "word", "char"}:
            raise ValueError(f"unsupported native-parent mode: {self.mode}")
        if self.query_value not in {"name", "address"}:
            raise ValueError(f"unsupported query value: {self.query_value}")
        if not self.name or not self.backend or not self.family or not self.fields:
            raise ValueError("parent channel identity and fields must be non-empty")
        if not 1 <= self.limit <= 512:
            raise ValueError("parent channel limit must be between 1 and 512")
        if self.weight <= 0:
            raise ValueError("parent channel weight must be positive")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ParentChannelSpec":
        return cls(
            name=str(raw["name"]),
            backend=str(raw["backend"]),
            family=str(raw["family"]),
            mode=str(raw["mode"]),
            fields=tuple(str(value) for value in raw["fields"]),
            query_value=str(raw.get("query_value", "name")),
            limit=int(raw["limit"]),
            weight=float(raw.get("weight", 1.0)),
        )


def load_query_selection(path: Path) -> tuple[tuple[str, int], ...]:
    table = pq.read_table(path)
    if set(table.column_names) != {"query_id", "fold"}:
        raise ValueError("generator query selection must contain only query_id and fold")
    rows = tuple((str(row["query_id"]), int(row["fold"])) for row in table.to_pylist())
    if not rows or len(rows) != len({query_id for query_id, _fold in rows}):
        raise ValueError("query selection must be non-empty and query_id-unique")
    if any(not query_id or fold not in {2, 3, 4} for query_id, fold in rows):
        raise ValueError("query selection opened a forbidden query or fold")
    return rows


def _documents_for_channel(
    backend: Any,
    query: RetrievalQuery,
    spec: ParentChannelSpec,
) -> list[tuple[float, Mapping[str, Any]]]:
    """Return native SIREN documents for one channel.

    A small explicit hook makes the semantics independently unit-testable.
    Production Tantivy backends use the same geo guard as the frozen sparse
    retriever while forcing ``document_type=siren``.
    """
    hook = getattr(backend, "search_native_parent_documents", None)
    text = getattr(query, spec.query_value)
    if hook is not None:
        return list(hook(query=query, spec=spec))
    if not text:
        return []
    if spec.mode in {"exact", "word"}:
        return list(
            backend._query(
                query,
                spec.fields,
                text,
                spec.limit,
                exact=spec.mode == "exact",
                document_type="siren",
            )
        )
    character_query = backend._character_query(spec.fields[0], text)
    parsed = backend._combined_query(query, character_query, document_type="siren")
    result = backend.searcher.search(parsed, limit=spec.limit)
    return [
        (float(score), backend.searcher.doc(address).to_dict())
        for score, address in result.hits
    ]


def search_native_parents(
    backend: Any,
    query: RetrievalQuery,
    spec: ParentChannelSpec,
) -> tuple[tuple[str, float], ...]:
    """Search and deduplicate native SIREN heads without inspecting SIRETs."""
    best: dict[str, float] = {}
    for score, document in _documents_for_channel(backend, query, spec):
        raw_siren = backend._first(document, "siren") if hasattr(backend, "_first") else document.get("siren", "")
        siren = normalize_code(raw_siren, 9)
        if len(siren) != 9 or not siren.isdigit():
            continue
        best[siren] = max(best.get(siren, float("-inf")), float(score))
    ordered = sorted(best.items(), key=lambda item: (-item[1], item[0]))
    return tuple(ordered[: spec.limit])


def generate_query_signals(
    *,
    query_id: str,
    fold: int,
    query: RetrievalQuery,
    backends: Mapping[str, Any],
    specs: Sequence[ParentChannelSpec],
    workers: int = 12,
) -> list[dict[str, Any]]:
    missing = sorted({spec.backend for spec in specs} - set(backends))
    if missing:
        raise ValueError(f"missing configured parent backends: {missing}")
    results: dict[str, tuple[tuple[str, float], ...]] = {}
    with ThreadPoolExecutor(max_workers=min(workers, len(specs))) as pool:
        futures = {
            pool.submit(search_native_parents, backends[spec.backend], query, spec): spec
            for spec in specs
        }
        for future in as_completed(futures):
            results[futures[future].name] = future.result()
    rows: list[dict[str, Any]] = []
    for spec in specs:
        for rank, (siren, score) in enumerate(results[spec.name], start=1):
            rows.append(
                {
                    "query_id": query_id,
                    "fold": fold,
                    "backend": spec.backend,
                    "family": spec.family,
                    "channel": spec.name,
                    "channel_rank": rank,
                    "siren": siren,
                    "score": score,
                }
            )
    return rows


def generate_batch_signals(
    *,
    queries: Sequence[tuple[str, int, RetrievalQuery]],
    backends: Mapping[str, Any],
    specs: Sequence[ParentChannelSpec],
    workers: int = 12,
) -> list[dict[str, Any]]:
    """Search a batch with one shared pool to avoid per-query tail stalls."""
    missing = sorted({spec.backend for spec in specs} - set(backends))
    if missing:
        raise ValueError(f"missing configured parent backends: {missing}")
    results: dict[tuple[str, str], tuple[tuple[str, float], ...]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(search_native_parents, backends[spec.backend], query, spec):
            (query_id, spec.name)
            for query_id, _fold, query in queries
            for spec in specs
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    rows: list[dict[str, Any]] = []
    for query_id, fold, _query in queries:
        for spec in specs:
            for rank, (siren, score) in enumerate(results[(query_id, spec.name)], start=1):
                rows.append(
                    {
                        "query_id": query_id,
                        "fold": fold,
                        "backend": spec.backend,
                        "family": spec.family,
                        "channel": spec.name,
                        "channel_rank": rank,
                        "siren": siren,
                        "score": score,
                    }
                )
    return rows


def fuse_parent_signals(
    rows: Iterable[Mapping[str, Any]],
    *,
    allowed_channels: set[str],
    channel_weights: Mapping[str, float],
    cap: int,
    rrf_k: int = 60,
) -> tuple[dict[str, Any], ...]:
    if not 1 <= cap <= 512 or rrf_k != 60:
        raise ValueError("v16 fusion requires cap<=512 and frozen RRF k=60")
    scores: dict[str, float] = {}
    support: dict[str, set[str]] = {}
    best_rank: dict[str, int] = {}
    for row in rows:
        channel = str(row["channel"])
        if channel not in allowed_channels:
            continue
        siren, rank = str(row["siren"]), int(row["channel_rank"])
        weight = float(channel_weights.get(channel, 1.0))
        scores[siren] = scores.get(siren, 0.0) + weight / (rrf_k + rank)
        support.setdefault(siren, set()).add(channel)
        best_rank[siren] = min(best_rank.get(siren, rank), rank)
    ordered = sorted(
        scores,
        key=lambda siren: (-scores[siren], -len(support[siren]), best_rank[siren], siren),
    )[:cap]
    return tuple(
        {
            "siren": siren,
            "parent_rank": rank,
            "rrf_score": scores[siren],
            "channel_support": len(support[siren]),
        }
        for rank, siren in enumerate(ordered, start=1)
    )


def write_generation(
    output_dir: Path,
    signal_rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> Path:
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    signals_path = output_dir / "parent_signals.parquet"
    pq.write_table(
        pa.Table.from_pylist(list(signal_rows), schema=SIGNAL_SCHEMA),
        signals_path,
        compression="zstd",
    )
    payload = {
        **dict(manifest),
        "schema_version": GENERATION_SCHEMA_VERSION,
        "native_parent_only": True,
        "siret_truncation": False,
        "files": {
            "parent_signals.parquet": {
                "rows": len(signal_rows),
                "size_bytes": signals_path.stat().st_size,
                "sha256": sha256_file(signals_path),
            }
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


__all__ = [
    "GENERATION_SCHEMA_VERSION",
    "ParentChannelSpec",
    "fuse_parent_signals",
    "generate_batch_signals",
    "generate_query_signals",
    "load_query_selection",
    "search_native_parents",
    "sha256_file",
    "write_generation",
]
