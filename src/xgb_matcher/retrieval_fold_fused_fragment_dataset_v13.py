"""Zero-copy canonical datasets over retrieval-v12 worker fragments.

The v12 physical executor writes one compact training matrix and one
label-blind held-out spool per worker.  Historically those files were copied
into three merged training Parquets and one merged held-out Parquet before fit
and replay.  This module makes the *ordered fragments themselves* the
canonical dataset instead:

* every source file and source manifest remains content addressed;
* a physical Merkle root seals the exact ordered fragment layout;
* per-query logical Merkle roots seal content independently of W1/W2/W3
  fragmentation;
* sequential readers reproduce the monolithic query/group order without a
  merged file;
* LambdaMART arrays are allocated once and filled directly from fragment row
  groups.

This is an execution/storage adapter only.  It does not alter candidates,
features, labels, sampling, nested OOF folds, model parameters or metrics.
Fold 0 and fold 1 are rejected at every public boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
import gc
import hashlib
import json
import math
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .retrieval_fold_fused_oof_v12 import (
    CompactTrainBatch,
    FittedSiteModel,
    FoldFusedInvariantError,
    MasterContext,
    TRAIN_FOLDS,
    TrainMatrixArtifact,
    sha256_file,
)
from .retrieval_fold_fused_relocatable_paths_v13 import (
    relocatable_record_path_v13,
)
from .retrieval_fold_fused_sharded_oof_v12 import (
    ShardBuildArtifactV12,
    ShardSpecV12,
    _canonical_compact_train_schema_v12,
    _feature_names_from_train_schema,
    _iter_heldout_query_tables,
    _ordered_build_artifacts_v12,
    _sequence_sha256,
    _writer_side_arrow_schema_v12,
    load_shard_build_manifest_v12,
)


SCHEMA_VERSION = "sireto-retrieval-v13-fragment-dataset-1"
ENGINE_ID = "FOLD_FUSED_ZERO_COPY_FRAGMENT_DATASET_V13_1"
_ALLOWED_FOLDS = frozenset(TRAIN_FOLDS)
_SHA256_CHARS = frozenset("0123456789abcdef")


def _canonical_json(value: Any) -> str:
    def compatible(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): compatible(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [compatible(val) for val in item]
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, np.generic):
            return compatible(item.item())
        if isinstance(item, float) and not math.isfinite(item):
            raise FoldFusedInvariantError("non-finite value cannot be sealed")
        return item

    return json.dumps(
        compatible(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(_SHA256_CHARS)
    )


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def _relative_or_absolute(path: Path, root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(root.resolve()))
    except ValueError:
        return str(resolved)


def _record_path_v13(
    path: Path,
    root: Path,
    relocation_root: Path | None,
) -> str:
    if relocation_root is None:
        return _relative_or_absolute(path, root)
    return relocatable_record_path_v13(
        path, record_root=root, run_root=relocation_root
    )


def _resolve_path(root: Path, value: Any) -> Path:
    path = Path(str(value or ""))
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def merkle_root_v13(records: Sequence[Mapping[str, Any]]) -> str:
    """Domain-separated binary Merkle root over ordered canonical records."""

    if not records:
        raise FoldFusedInvariantError("a canonical fragment dataset cannot be empty")
    level = [
        hashlib.sha256(b"\x00" + _canonical_json(record).encode("utf-8")).digest()
        for record in records
    ]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [
            hashlib.sha256(b"\x01" + level[index] + level[index + 1]).digest()
            for index in range(0, len(level), 2)
        ]
    return level[0].hex()


def logical_table_sha256_v13(table: pa.Table) -> str:
    """Hash one logical query/group independently of Parquet fragmentation."""

    # ``Table.slice`` may keep non-zero offsets into a larger row-group buffer.
    # Arrow IPC can encode those offsets differently even though the logical
    # rows are equal.  ``concat_arrays`` materialises each sliced chunk into a
    # zero-offset array without round-tripping values through Python.
    canonical = pa.Table.from_arrays(
        [pa.concat_arrays(list(column.chunks)) for column in table.columns],
        schema=table.schema.remove_metadata(),
    )
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, canonical.schema) as writer:
        writer.write_table(canonical, max_chunksize=max(1, canonical.num_rows))
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def _schema_sha256(schema: pa.Schema) -> str:
    canonical = schema.remove_metadata()
    return hashlib.sha256(canonical.serialize().to_pybytes()).hexdigest()


@dataclass(frozen=True)
class FragmentFileV13:
    kind: str
    shard_id: str
    shard_ordinal: int
    path: Path
    sha256: str
    size_bytes: int
    row_count: int
    query_count: int
    schema_sha256: str
    outer_fold: int | None = None

    def __post_init__(self) -> None:
        if (
            not self.kind
            or not self.shard_id
            or self.shard_ordinal < 0
            or not _valid_sha256(self.sha256)
            or not _valid_sha256(self.schema_sha256)
            or self.size_bytes <= 0
            or self.row_count < self.query_count
            or self.query_count <= 0
            or (
                self.outer_fold is not None
                and int(self.outer_fold) not in _ALLOWED_FOLDS
            )
        ):
            raise FoldFusedInvariantError("invalid v13 fragment file record")


@dataclass(frozen=True)
class TrainOuterDatasetV13:
    outer_fold: int
    fragments: tuple[FragmentFileV13, ...]
    row_count: int
    query_count: int
    fit_folds: tuple[int, ...]
    feature_names: tuple[str, ...]
    feature_schema_sha256: str
    parquet_schema_sha256: str
    group_sequence_sha256: str
    logical_merkle_root: str

    def __post_init__(self) -> None:
        if (
            self.outer_fold not in _ALLOWED_FOLDS
            or not self.fragments
            or any(value.outer_fold != self.outer_fold for value in self.fragments)
            or self.row_count != sum(value.row_count for value in self.fragments)
            or self.query_count != sum(value.query_count for value in self.fragments)
            or self.fit_folds
            != tuple(fold for fold in TRAIN_FOLDS if fold != self.outer_fold)
            or not self.feature_names
            or not all(
                _valid_sha256(value)
                for value in (
                    self.feature_schema_sha256,
                    self.parquet_schema_sha256,
                    self.group_sequence_sha256,
                    self.logical_merkle_root,
                )
            )
        ):
            raise FoldFusedInvariantError("invalid v13 outer training dataset")


@dataclass(frozen=True)
class BuildFragmentDatasetV13:
    manifest_path: Path
    manifest_sha256: str
    dataset_id: str
    lineage: Mapping[str, str]
    specs: tuple[ShardSpecV12, ...]
    build_manifest_sha256: tuple[str, ...]
    train: Mapping[int, TrainOuterDatasetV13]
    heldout_fragments: tuple[FragmentFileV13, ...]
    query_count: int
    heldout_row_count: int
    query_sequence_sha256: str
    heldout_schema_sha256: str
    heldout_parquet_schema_sha256: str
    heldout_logical_merkle_root: str
    physical_merkle_root: str

    def __post_init__(self) -> None:
        if (
            not _valid_sha256(self.manifest_sha256)
            or not _valid_sha256(self.dataset_id)
            or not _valid_sha256(self.query_sequence_sha256)
            or not _valid_sha256(self.heldout_schema_sha256)
            or not _valid_sha256(self.heldout_parquet_schema_sha256)
            or not _valid_sha256(self.heldout_logical_merkle_root)
            or not _valid_sha256(self.physical_merkle_root)
            or set(self.train) != set(TRAIN_FOLDS)
            or not self.specs
            or len(self.specs) != len(self.heldout_fragments)
            or len(self.specs) != len(self.build_manifest_sha256)
            or self.query_count != sum(spec.query_count for spec in self.specs)
            or self.heldout_row_count
            != sum(value.row_count for value in self.heldout_fragments)
            or set(int(value) for spec in self.specs for value in spec.folds)
            - _ALLOWED_FOLDS
        ):
            raise FoldFusedInvariantError("invalid v13 canonical build dataset")


@dataclass(frozen=True)
class FragmentedTrainArraysV13:
    outer_fold: int
    values: np.ndarray
    labels: np.ndarray
    qid: np.ndarray
    query_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    group_sequence_sha256: str
    logical_merkle_root: str

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        labels = np.asarray(self.labels)
        qid = np.asarray(self.qid)
        if (
            self.outer_fold not in _ALLOWED_FOLDS
            or values.dtype != np.float32
            or values.ndim != 2
            or labels.dtype != np.int8
            or qid.dtype != np.int64
            or len(values) != len(labels)
            or len(values) != len(qid)
            or values.shape[1] != len(self.feature_names)
            or not self.query_ids
            or len(set(self.query_ids)) != len(self.query_ids)
            or not np.isfinite(values).all()
        ):
            raise FoldFusedInvariantError("invalid v13 fragmented train arrays")


class BufferedCompactTrainingStoreV13:
    """Drop-in compact train sink with whole-query multi-query row groups."""

    def __init__(
        self,
        output_dir: Path,
        *,
        row_group_target: int = 65_536,
    ) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.row_group_target = int(row_group_target)
        if self.row_group_target < 128:
            raise FoldFusedInvariantError("train row-group target must fit one group")
        self.paths = {
            outer: self.output_dir / f"outer{outer}_site_train_matrix.parquet"
            for outer in TRAIN_FOLDS
        }
        self._writers: dict[int, pq.ParquetWriter] = {}
        self._buffers: dict[int, list[pa.Table]] = {
            outer: [] for outer in TRAIN_FOLDS
        }
        self._buffer_rows = {outer: 0 for outer in TRAIN_FOLDS}
        self._rows = {outer: 0 for outer in TRAIN_FOLDS}
        self._queries: dict[int, list[tuple[str, int]]] = {
            outer: [] for outer in TRAIN_FOLDS
        }
        self._seen = {outer: set() for outer in TRAIN_FOLDS}
        self._feature_names: tuple[str, ...] | None = None
        self._schema: pa.Schema | None = None

    def _table(self, batch: CompactTrainBatch) -> pa.Table:
        names = tuple(str(value) for value in batch.feature_names)
        if not names or len(names) != len(set(names)):
            raise FoldFusedInvariantError("buffered train feature schema invalid")
        if self._feature_names is None:
            self._feature_names = names
        elif names != self._feature_names:
            raise FoldFusedInvariantError("buffered train feature schema drift")
        width = len(names)
        flat = pa.array(batch.values.reshape(-1), type=pa.float32())
        matrix = pa.FixedSizeListArray.from_arrays(flat, width)
        count = len(batch.candidate_sirets)
        table = pa.table(
            {
                "query_id": pa.array([batch.query_id] * count, type=pa.string()),
                "scene_fold": pa.array([batch.scene_fold] * count, type=pa.int8()),
                "candidate_siret": pa.array(batch.candidate_sirets, type=pa.string()),
                "candidate_parent": pa.array(batch.candidate_parents, type=pa.string()),
                "site_label": pa.array(batch.labels, type=pa.int8()),
                "statistics_sha256": pa.array(
                    [batch.statistics_sha256] * count, type=pa.string()
                ),
                "feature_values": matrix,
            }
        )
        metadata = dict(table.schema.metadata or {})
        metadata[b"feature_names_json"] = _canonical_json(list(names)).encode("utf-8")
        metadata[b"feature_schema_sha256"] = _sha256_json(list(names)).encode("ascii")
        table = table.replace_schema_metadata(metadata)
        if self._schema is None:
            self._schema = table.schema
        elif not table.schema.equals(self._schema, check_metadata=True):
            raise FoldFusedInvariantError("buffered train physical schema drift")
        return table

    def _flush(self, outer: int) -> None:
        values = self._buffers[outer]
        if not values:
            return
        table = values[0] if len(values) == 1 else pa.concat_tables(values)
        writer = self._writers.get(outer)
        if writer is None:
            assert self._schema is not None
            writer = pq.ParquetWriter(
                self.paths[outer],
                self._schema,
                compression="zstd",
                compression_level=3,
                use_dictionary=True,
                write_statistics=True,
                data_page_version="1.0",
            )
            self._writers[outer] = writer
        writer.write_table(table, row_group_size=table.num_rows)
        values.clear()
        self._buffer_rows[outer] = 0

    def append(self, batch: CompactTrainBatch) -> None:
        outer = int(batch.outer_fold)
        if outer not in _ALLOWED_FOLDS or batch.query_id in self._seen[outer]:
            raise FoldFusedInvariantError("buffered train outer/query identity drift")
        table = self._table(batch)
        if (
            self._buffers[outer]
            and self._buffer_rows[outer] + table.num_rows > self.row_group_target
        ):
            self._flush(outer)
        self._buffers[outer].append(table)
        self._buffer_rows[outer] += table.num_rows
        self._rows[outer] += table.num_rows
        self._seen[outer].add(batch.query_id)
        self._queries[outer].append((batch.query_id, batch.scene_fold))

    def seal_outer(self, outer_fold: int) -> TrainMatrixArtifact:
        outer = int(outer_fold)
        if outer not in _ALLOWED_FOLDS or not self._queries[outer]:
            raise FoldFusedInvariantError("buffered outer train matrix is empty")
        self._flush(outer)
        writer = self._writers.pop(outer, None)
        if writer is None or self._feature_names is None:
            raise FoldFusedInvariantError("buffered outer writer was not opened")
        writer.close()
        fit_folds = tuple(sorted({fold for _query, fold in self._queries[outer]}))
        return TrainMatrixArtifact(
            outer_fold=outer,
            path=self.paths[outer],
            sha256=sha256_file(self.paths[outer]),
            row_count=self._rows[outer],
            query_count=len(self._queries[outer]),
            fit_folds=fit_folds,
            feature_names=self._feature_names,
            feature_schema_sha256=_sha256_json(list(self._feature_names)),
        )


def _train_query_tables(
    fragment: FragmentFileV13,
) -> Iterable[tuple[pa.Table, str, int]]:
    parquet = pq.ParquetFile(fragment.path)
    if parquet.metadata.num_rows != fragment.row_count:
        raise FoldFusedInvariantError("train fragment row count drift")
    seen: set[str] = set()
    for row_group in range(parquet.num_row_groups):
        row_group_table = parquet.read_row_group(row_group)
        query_values = row_group_table["query_id"].to_pylist()
        start = 0
        while start < len(query_values):
            query_id = str(query_values[start])
            stop = start + 1
            while stop < len(query_values) and str(query_values[stop]) == query_id:
                stop += 1
            if query_id in seen:
                raise FoldFusedInvariantError(
                    "train fragment query is split or duplicated"
                )
            seen.add(query_id)
            table = row_group_table.slice(start, stop - start)
            folds = {int(value.as_py()) for value in table["scene_fold"]}
            labels = np.asarray(table["site_label"], dtype=np.int8)
            statistics = {
                str(value)
                for value in table["statistics_sha256"].to_pylist()
            }
            if (
                len(folds) != 1
                or fragment.outer_fold in folds
                or not folds.issubset(_ALLOWED_FOLDS)
                or len(statistics) != 1
                or int(labels.sum()) != 1
                or not 2 <= table.num_rows <= 128
            ):
                raise FoldFusedInvariantError("train fragment group semantics drift")
            yield table, query_id, next(iter(folds))
            start = stop


def _ordered_strict_builds(
    artifacts: Sequence[ShardBuildArtifactV12],
) -> tuple[ShardBuildArtifactV12, ...]:
    # Reloading makes publication depend on sealed source manifests, not on a
    # potentially stale in-memory dataclass supplied by a caller.
    strict = tuple(
        load_shard_build_manifest_v12(value.manifest_path, verify_hashes=True)
        for value in artifacts
    )
    return _ordered_build_artifacts_v12(strict)


def _fragment_record(
    value: FragmentFileV13,
    root: Path,
    relocation_root: Path | None = None,
) -> dict[str, Any]:
    return {
        "kind": value.kind,
        "shard_id": value.shard_id,
        "shard_ordinal": value.shard_ordinal,
        "outer_fold": value.outer_fold,
        "path": _record_path_v13(value.path, root, relocation_root),
        "sha256": value.sha256,
        "size_bytes": value.size_bytes,
        "row_count": value.row_count,
        "query_count": value.query_count,
        "schema_sha256": value.schema_sha256,
    }


def _fragment_leaf_record(value: FragmentFileV13) -> dict[str, Any]:
    """Path-independent physical identity used by Merkle/content addresses."""

    return {
        "kind": value.kind,
        "shard_id": value.shard_id,
        "shard_ordinal": value.shard_ordinal,
        "outer_fold": value.outer_fold,
        "sha256": value.sha256,
        "size_bytes": value.size_bytes,
        "row_count": value.row_count,
        "query_count": value.query_count,
        "schema_sha256": value.schema_sha256,
    }


def _read_fragment_record(
    value: Mapping[str, Any], root: Path, *, verify_hashes: bool
) -> FragmentFileV13:
    path = _resolve_path(root, value.get("path"))
    record = FragmentFileV13(
        kind=str(value.get("kind") or ""),
        shard_id=str(value.get("shard_id") or ""),
        shard_ordinal=int(value.get("shard_ordinal", -1)),
        outer_fold=(
            int(value["outer_fold"])
            if value.get("outer_fold") is not None
            else None
        ),
        path=path,
        sha256=str(value.get("sha256") or ""),
        size_bytes=int(value.get("size_bytes", -1)),
        row_count=int(value.get("row_count", -1)),
        query_count=int(value.get("query_count", -1)),
        schema_sha256=str(value.get("schema_sha256") or ""),
    )
    if (
        not path.is_file()
        or path.stat().st_size != record.size_bytes
        or (verify_hashes and sha256_file(path) != record.sha256)
    ):
        raise FoldFusedInvariantError("v13 fragment file hash/size drift")
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != record.row_count:
        raise FoldFusedInvariantError("v13 fragment Parquet cardinality drift")
    return record


def _scan_train_outer(
    outer: int,
    fragments: Sequence[FragmentFileV13],
    global_positions: Mapping[str, int],
    global_folds: Mapping[str, int],
) -> tuple[
    tuple[str, ...],
    str,
    str,
    str,
    tuple[int, ...],
    tuple[Mapping[str, Any], ...],
]:
    feature_names: tuple[str, ...] | None = None
    schema_sha: str | None = None
    groups: list[tuple[str, int]] = []
    leaves: list[Mapping[str, Any]] = []
    previous_position = -1
    seen: set[str] = set()
    for fragment in fragments:
        parquet = pq.ParquetFile(fragment.path)
        names = _feature_names_from_train_schema(parquet.schema_arrow)
        canonical_schema = _canonical_compact_train_schema_v12(
            parquet.schema_arrow, names
        )
        current_schema_sha = _schema_sha256(canonical_schema)
        if feature_names is None:
            feature_names = names
            schema_sha = current_schema_sha
        elif names != feature_names or current_schema_sha != schema_sha:
            raise FoldFusedInvariantError("train fragment feature/schema drift")
        if current_schema_sha != fragment.schema_sha256:
            raise FoldFusedInvariantError("train fragment sealed schema drift")
        local_count = 0
        for table, query_id, fold in _train_query_tables(fragment):
            position = global_positions.get(query_id, -1)
            if (
                query_id in seen
                or position <= previous_position
                or global_folds.get(query_id) != fold
            ):
                raise FoldFusedInvariantError("train fragment logical order drift")
            seen.add(query_id)
            previous_position = position
            groups.append((query_id, fold))
            leaves.append(
                {
                    "kind": "train_group",
                    "outer_fold": outer,
                    "query_id": query_id,
                    "fold": fold,
                    "row_count": table.num_rows,
                    "table_sha256": logical_table_sha256_v13(table),
                }
            )
            local_count += 1
        if local_count != fragment.query_count:
            raise FoldFusedInvariantError("train fragment query count drift")
    if feature_names is None or schema_sha is None or not groups:
        raise FoldFusedInvariantError(f"outer{outer} fragmented train is empty")
    fit_folds = tuple(sorted({fold for _query_id, fold in groups}))
    expected = tuple(fold for fold in TRAIN_FOLDS if fold != outer)
    if fit_folds != expected:
        raise FoldFusedInvariantError(f"outer{outer} misses a fit fold")
    return (
        feature_names,
        schema_sha,
        _sequence_sha256(groups),
        merkle_root_v13(leaves),
        fit_folds,
        tuple(leaves),
    )


def _scan_heldout(
    fragments: Sequence[FragmentFileV13],
    specs: Sequence[ShardSpecV12],
) -> tuple[str, str, tuple[Mapping[str, Any], ...]]:
    schema_sha: str | None = None
    observed: list[tuple[str, int]] = []
    leaves: list[Mapping[str, Any]] = []
    for fragment, spec in zip(fragments, specs, strict=True):
        parquet = pq.ParquetFile(fragment.path)
        canonical_schema = _writer_side_arrow_schema_v12(parquet.schema_arrow)
        current_schema_sha = _schema_sha256(canonical_schema)
        if schema_sha is None:
            schema_sha = current_schema_sha
        elif current_schema_sha != schema_sha:
            raise FoldFusedInvariantError("heldout fragment schema drift")
        if current_schema_sha != fragment.schema_sha256:
            raise FoldFusedInvariantError("heldout fragment sealed schema drift")
        shard_sequence: list[tuple[str, int]] = []
        for table in _iter_heldout_query_tables(fragment.path):
            query_ids = {str(value) for value in table["query_id"].to_pylist()}
            folds = {int(value.as_py()) for value in table["fold"]}
            ranks = tuple(int(value.as_py()) for value in table["master_rank"])
            if (
                len(query_ids) != 1
                or len(folds) != 1
                or not folds.issubset(_ALLOWED_FOLDS)
                or ranks != tuple(range(1, table.num_rows + 1))
                or not 1 <= table.num_rows <= 2_000
            ):
                raise FoldFusedInvariantError("heldout fragment query semantics drift")
            current = (next(iter(query_ids)), next(iter(folds)))
            shard_sequence.append(current)
            observed.append(current)
            leaves.append(
                {
                    "kind": "heldout_query",
                    "query_id": current[0],
                    "fold": current[1],
                    "row_count": table.num_rows,
                    "table_sha256": logical_table_sha256_v13(table),
                }
            )
        if tuple(shard_sequence) != tuple(
            zip(spec.query_ids, spec.query_folds, strict=True)
        ):
            raise FoldFusedInvariantError("heldout fragment sequence drift")
    if schema_sha is None or not observed:
        raise FoldFusedInvariantError("heldout fragmented dataset is empty")
    return schema_sha, merkle_root_v13(leaves), tuple(leaves)


def seal_build_fragment_dataset_v13(
    artifacts: Sequence[ShardBuildArtifactV12],
    output_dir: Path,
    *,
    relocation_root: Path | None = None,
) -> BuildFragmentDatasetV13:
    """Publish an immutable manifest over existing worker files, without copy."""

    ordered = _ordered_strict_builds(artifacts)
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    lineage = dict(ordered[0].lineage)
    specs = tuple(value.spec for value in ordered)
    global_sequence = tuple(
        (query_id, fold)
        for spec in specs
        for query_id, fold in zip(spec.query_ids, spec.query_folds, strict=True)
    )
    global_positions = {
        query_id: index for index, (query_id, _fold) in enumerate(global_sequence)
    }
    global_folds = dict(global_sequence)

    heldout: list[FragmentFileV13] = []
    train_fragments: dict[int, list[FragmentFileV13]] = {
        outer: [] for outer in TRAIN_FOLDS
    }
    physical_leaves: list[Mapping[str, Any]] = []
    for artifact in ordered:
        physical_leaves.append(
            {
                "kind": "build_manifest",
                "shard_ordinal": artifact.spec.shard_ordinal,
                "spec_sha256": artifact.spec.digest,
                "sha256": artifact.manifest_sha256,
                "build_id": artifact.build_id,
            }
        )
        heldout_parquet = pq.ParquetFile(artifact.heldout_master.path)
        heldout_schema = _schema_sha256(
            _writer_side_arrow_schema_v12(heldout_parquet.schema_arrow)
        )
        held = FragmentFileV13(
            kind="heldout_master",
            shard_id=artifact.spec.shard_id,
            shard_ordinal=artifact.spec.shard_ordinal,
            path=artifact.heldout_master.path,
            sha256=artifact.heldout_master.sha256,
            size_bytes=artifact.heldout_master.size_bytes,
            row_count=artifact.heldout_master.row_count,
            query_count=artifact.heldout_master.query_count,
            schema_sha256=heldout_schema,
        )
        heldout.append(held)
        physical_leaves.append(_fragment_leaf_record(held))
        for outer in TRAIN_FOLDS:
            matrix = artifact.train_matrices.get(outer)
            if matrix is None:
                continue
            parquet = pq.ParquetFile(matrix.path)
            names = _feature_names_from_train_schema(parquet.schema_arrow)
            schema = _canonical_compact_train_schema_v12(
                parquet.schema_arrow, names
            )
            fragment = FragmentFileV13(
                kind="train_matrix",
                shard_id=artifact.spec.shard_id,
                shard_ordinal=artifact.spec.shard_ordinal,
                outer_fold=outer,
                path=matrix.path,
                sha256=matrix.sha256,
                size_bytes=matrix.path.stat().st_size,
                row_count=matrix.row_count,
                query_count=matrix.query_count,
                schema_sha256=_schema_sha256(schema),
            )
            train_fragments[outer].append(fragment)
            physical_leaves.append(_fragment_leaf_record(fragment))

    heldout_schema, heldout_logical_root, _heldout_leaves = _scan_heldout(
        heldout, specs
    )
    train: dict[int, TrainOuterDatasetV13] = {}
    train_manifest: dict[str, Any] = {}
    for outer in TRAIN_FOLDS:
        fragments = tuple(train_fragments[outer])
        (
            feature_names,
            parquet_schema_sha,
            group_sequence_sha,
            logical_root,
            fit_folds,
            _leaves,
        ) = _scan_train_outer(
            outer, fragments, global_positions, global_folds
        )
        feature_schema_sha = _sha256_json(list(feature_names))
        dataset = TrainOuterDatasetV13(
            outer_fold=outer,
            fragments=fragments,
            row_count=sum(value.row_count for value in fragments),
            query_count=sum(value.query_count for value in fragments),
            fit_folds=fit_folds,
            feature_names=feature_names,
            feature_schema_sha256=feature_schema_sha,
            parquet_schema_sha256=parquet_schema_sha,
            group_sequence_sha256=group_sequence_sha,
            logical_merkle_root=logical_root,
        )
        train[outer] = dataset
        train_manifest[str(outer)] = {
            "outer_fold": outer,
            "fragments": [
                _fragment_record(value, output, relocation_root)
                for value in dataset.fragments
            ],
            "row_count": dataset.row_count,
            "query_count": dataset.query_count,
            "fit_folds": list(dataset.fit_folds),
            "feature_names": list(dataset.feature_names),
            "feature_schema_sha256": dataset.feature_schema_sha256,
            "parquet_schema_sha256": dataset.parquet_schema_sha256,
            "group_sequence_sha256": dataset.group_sequence_sha256,
            "logical_merkle_root": dataset.logical_merkle_root,
        }

    physical_root = merkle_root_v13(physical_leaves)
    query_sequence_sha = _sequence_sha256(global_sequence)
    identity = {
        "engine_id": ENGINE_ID,
        "lineage": lineage,
        "ordered_spec_sha256": [value.digest for value in specs],
        "ordered_build_manifest_sha256": [
            value.manifest_sha256 for value in ordered
        ],
        "query_count": len(global_sequence),
        "heldout_row_count": sum(value.row_count for value in heldout),
        "query_sequence_sha256": query_sequence_sha,
        "heldout_schema_sha256": ordered[0].heldout_master.schema_sha256,
        "heldout_parquet_schema_sha256": heldout_schema,
        "heldout_logical_merkle_root": heldout_logical_root,
        "train_logical_merkle_root": {
            str(outer): train[outer].logical_merkle_root for outer in TRAIN_FOLDS
        },
        "physical_merkle_root": physical_root,
    }
    dataset_id = _sha256_json(identity)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "SEALED_CANONICAL_FRAGMENT_DATASET",
        "engine_id": ENGINE_ID,
        "dataset_id": dataset_id,
        "identity": identity,
        "folds": list(TRAIN_FOLDS),
        "fold0_opened": False,
        "fold1_opened": False,
        "positive_injection": False,
        "copy_materialized": False,
        "lineage": lineage,
        "specs": [value.as_manifest() for value in specs],
        "ordered_build_manifests": [
            {
                "path": _record_path_v13(
                    value.manifest_path, output, relocation_root
                ),
                "sha256": value.manifest_sha256,
                "build_id": value.build_id,
            }
            for value in ordered
        ],
        "query_count": len(global_sequence),
        "heldout_row_count": sum(value.row_count for value in heldout),
        "query_sequence_sha256": query_sequence_sha,
        "heldout": {
            "fragments": [
                _fragment_record(value, output, relocation_root)
                for value in heldout
            ],
            "schema_sha256": ordered[0].heldout_master.schema_sha256,
            "parquet_schema_sha256": heldout_schema,
            "logical_merkle_root": heldout_logical_root,
        },
        "train": train_manifest,
        "physical_merkle_root": physical_root,
    }
    manifest_path = output / "manifest.json"
    _atomic_json(manifest_path, manifest)
    return BuildFragmentDatasetV13(
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        dataset_id=dataset_id,
        lineage=lineage,
        specs=specs,
        build_manifest_sha256=tuple(
            value.manifest_sha256 for value in ordered
        ),
        train=train,
        heldout_fragments=tuple(heldout),
        query_count=len(global_sequence),
        heldout_row_count=sum(value.row_count for value in heldout),
        query_sequence_sha256=query_sequence_sha,
        heldout_schema_sha256=ordered[0].heldout_master.schema_sha256,
        heldout_parquet_schema_sha256=heldout_schema,
        heldout_logical_merkle_root=heldout_logical_root,
        physical_merkle_root=physical_root,
    )


def load_build_fragment_dataset_v13(
    path: Path,
    *,
    verify_hashes: bool = True,
    verify_logical: bool = True,
) -> BuildFragmentDatasetV13:
    """Strict loader; optional logical scan may be skipped after preflight."""

    manifest_path = Path(path).resolve()
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        raw.get("schema_version") != SCHEMA_VERSION
        or raw.get("status") != "SEALED_CANONICAL_FRAGMENT_DATASET"
        or raw.get("engine_id") != ENGINE_ID
        or raw.get("folds") != list(TRAIN_FOLDS)
        or raw.get("fold0_opened") is not False
        or raw.get("fold1_opened") is not False
        or raw.get("positive_injection") is not False
        or raw.get("copy_materialized") is not False
    ):
        raise FoldFusedInvariantError("invalid or unsafe v13 fragment manifest")
    identity = raw.get("identity")
    if (
        not isinstance(identity, Mapping)
        or raw.get("dataset_id") != _sha256_json(identity)
    ):
        raise FoldFusedInvariantError("v13 fragment dataset content address drift")
    root = manifest_path.parent
    specs = tuple(
        ShardSpecV12.from_manifest(value) for value in raw.get("specs") or ()
    )
    if not specs:
        raise FoldFusedInvariantError("v13 fragment dataset has no shards")
    sequence_stop = 0
    seen: set[str] = set()
    for ordinal, spec in enumerate(specs):
        if (
            spec.shard_ordinal != ordinal
            or spec.sequence_start != sequence_stop
            or seen.intersection(spec.query_ids)
        ):
            raise FoldFusedInvariantError("v13 fragment shard order/gap drift")
        seen.update(spec.query_ids)
        sequence_stop = spec.sequence_stop
    build_records = raw.get("ordered_build_manifests")
    if not isinstance(build_records, list) or len(build_records) != len(specs):
        raise FoldFusedInvariantError("v13 source build manifest set drift")
    build_hashes: list[str] = []
    strict_builds: list[ShardBuildArtifactV12] = []
    for record, spec in zip(build_records, specs, strict=True):
        if not isinstance(record, Mapping):
            raise FoldFusedInvariantError("invalid v13 source build record")
        source_path = _resolve_path(root, record.get("path"))
        digest = str(record.get("sha256") or "")
        if (
            not source_path.is_file()
            or not _valid_sha256(digest)
            or (verify_hashes and sha256_file(source_path) != digest)
        ):
            raise FoldFusedInvariantError("v13 source build manifest hash drift")
        loaded = load_shard_build_manifest_v12(
            source_path, verify_hashes=verify_hashes
        )
        if (
            loaded.spec != spec
            or loaded.manifest_sha256 != digest
            or loaded.build_id != record.get("build_id")
        ):
            raise FoldFusedInvariantError("v13 source build identity drift")
        strict_builds.append(loaded)
        build_hashes.append(digest)
    lineage = {str(key): str(value) for key, value in (raw.get("lineage") or {}).items()}
    if (
        not lineage
        or any(not _valid_sha256(value) for value in lineage.values())
        or any(dict(value.lineage) != lineage for value in strict_builds)
    ):
        raise FoldFusedInvariantError("v13 fragment lineage drift")

    heldout_raw = raw.get("heldout")
    train_raw = raw.get("train")
    if not isinstance(heldout_raw, Mapping) or not isinstance(train_raw, Mapping):
        raise FoldFusedInvariantError("v13 fragment dataset components are missing")
    heldout = tuple(
        _read_fragment_record(value, root, verify_hashes=verify_hashes)
        for value in heldout_raw.get("fragments") or ()
    )
    if len(heldout) != len(specs) or any(
        value.kind != "heldout_master"
        or value.shard_ordinal != index
        or value.outer_fold is not None
        for index, value in enumerate(heldout)
    ):
        raise FoldFusedInvariantError("v13 heldout fragment order drift")
    train: dict[int, TrainOuterDatasetV13] = {}
    for outer in TRAIN_FOLDS:
        record = train_raw.get(str(outer))
        if not isinstance(record, Mapping) or int(record.get("outer_fold", -1)) != outer:
            raise FoldFusedInvariantError("invalid v13 outer train manifest")
        fragments = tuple(
            _read_fragment_record(value, root, verify_hashes=verify_hashes)
            for value in record.get("fragments") or ()
        )
        if any(
            value.kind != "train_matrix" or value.outer_fold != outer
            for value in fragments
        ):
            raise FoldFusedInvariantError("v13 train fragment identity drift")
        dataset = TrainOuterDatasetV13(
            outer_fold=outer,
            fragments=fragments,
            row_count=int(record.get("row_count", -1)),
            query_count=int(record.get("query_count", -1)),
            fit_folds=tuple(int(value) for value in record.get("fit_folds") or ()),
            feature_names=tuple(str(value) for value in record.get("feature_names") or ()),
            feature_schema_sha256=str(record.get("feature_schema_sha256") or ""),
            parquet_schema_sha256=str(record.get("parquet_schema_sha256") or ""),
            group_sequence_sha256=str(record.get("group_sequence_sha256") or ""),
            logical_merkle_root=str(record.get("logical_merkle_root") or ""),
        )
        train[outer] = dataset

    # Recreate physical leaves in precisely the publisher's interleaving.
    physical_leaves = []
    by_outer_shard = {
        (outer, value.shard_ordinal): value
        for outer, dataset in train.items()
        for value in dataset.fragments
    }
    for build, held in zip(strict_builds, heldout, strict=True):
        physical_leaves.append(
            {
                "kind": "build_manifest",
                "shard_ordinal": build.spec.shard_ordinal,
                "spec_sha256": build.spec.digest,
                "sha256": build.manifest_sha256,
                "build_id": build.build_id,
            }
        )
        physical_leaves.append(_fragment_leaf_record(held))
        for outer in TRAIN_FOLDS:
            fragment = by_outer_shard.get((outer, build.spec.shard_ordinal))
            if fragment is not None:
                physical_leaves.append(_fragment_leaf_record(fragment))
    physical_root = merkle_root_v13(physical_leaves)
    if physical_root != raw.get("physical_merkle_root"):
        raise FoldFusedInvariantError("v13 physical Merkle root drift")

    global_sequence = tuple(
        (query_id, fold)
        for spec in specs
        for query_id, fold in zip(spec.query_ids, spec.query_folds, strict=True)
    )
    query_sequence_sha = _sequence_sha256(global_sequence)
    heldout_schema = str(heldout_raw.get("parquet_schema_sha256") or "")
    heldout_logical = str(heldout_raw.get("logical_merkle_root") or "")
    if verify_logical:
        observed_schema, observed_logical, _ = _scan_heldout(heldout, specs)
        if observed_schema != heldout_schema or observed_logical != heldout_logical:
            raise FoldFusedInvariantError("v13 heldout logical Merkle/schema drift")
        positions = {
            query_id: index
            for index, (query_id, _fold) in enumerate(global_sequence)
        }
        folds = dict(global_sequence)
        for outer in TRAIN_FOLDS:
            names, schema, groups, logical, fit_folds, _ = _scan_train_outer(
                outer, train[outer].fragments, positions, folds
            )
            expected = train[outer]
            if (
                names != expected.feature_names
                or schema != expected.parquet_schema_sha256
                or groups != expected.group_sequence_sha256
                or logical != expected.logical_merkle_root
                or fit_folds != expected.fit_folds
            ):
                raise FoldFusedInvariantError("v13 train logical Merkle/schema drift")

    expected_identity = {
        "engine_id": ENGINE_ID,
        "lineage": lineage,
        "ordered_spec_sha256": [value.digest for value in specs],
        "ordered_build_manifest_sha256": build_hashes,
        "query_count": len(global_sequence),
        "heldout_row_count": sum(value.row_count for value in heldout),
        "query_sequence_sha256": query_sequence_sha,
        "heldout_schema_sha256": str(heldout_raw.get("schema_sha256") or ""),
        "heldout_parquet_schema_sha256": heldout_schema,
        "heldout_logical_merkle_root": heldout_logical,
        "train_logical_merkle_root": {
            str(outer): train[outer].logical_merkle_root for outer in TRAIN_FOLDS
        },
        "physical_merkle_root": physical_root,
    }
    if (
        dict(identity) != expected_identity
        or int(raw.get("query_count", -1)) != len(global_sequence)
        or int(raw.get("heldout_row_count", -1))
        != sum(value.row_count for value in heldout)
        or raw.get("query_sequence_sha256") != query_sequence_sha
    ):
        raise FoldFusedInvariantError("v13 fragment dataset identity/count drift")
    return BuildFragmentDatasetV13(
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        dataset_id=str(raw["dataset_id"]),
        lineage=lineage,
        specs=specs,
        build_manifest_sha256=tuple(build_hashes),
        train=train,
        heldout_fragments=heldout,
        query_count=len(global_sequence),
        heldout_row_count=sum(value.row_count for value in heldout),
        query_sequence_sha256=query_sequence_sha,
        heldout_schema_sha256=str(heldout_raw.get("schema_sha256") or ""),
        heldout_parquet_schema_sha256=heldout_schema,
        heldout_logical_merkle_root=heldout_logical,
        physical_merkle_root=physical_root,
    )


def load_fragmented_train_arrays_v13(
    dataset: BuildFragmentDatasetV13,
    outer_fold: int,
) -> FragmentedTrainArraysV13:
    """Load one outer matrix directly from ordered fragments into RAM once."""

    outer = int(outer_fold)
    if outer not in _ALLOWED_FOLDS:
        raise FoldFusedInvariantError("fragmented fit refuses fold 0/1")
    descriptor = dataset.train[outer]
    width = len(descriptor.feature_names)
    values = np.empty((descriptor.row_count, width), dtype=np.float32)
    labels = np.empty(descriptor.row_count, dtype=np.int8)
    qid = np.empty(descriptor.row_count, dtype=np.int64)
    query_ids: list[str] = []
    row_start = 0
    group = 0
    groups: list[tuple[str, int]] = []
    leaves: list[Mapping[str, Any]] = []
    for fragment in descriptor.fragments:
        for table, query_id, fold in _train_query_tables(fragment):
            count = table.num_rows
            stop = row_start + count
            feature_column = table["feature_values"].combine_chunks()
            matrix = np.asarray(feature_column.values, dtype=np.float32).reshape(
                count, width
            )
            values[row_start:stop] = matrix
            labels[row_start:stop] = np.asarray(table["site_label"], dtype=np.int8)
            qid[row_start:stop] = group
            query_ids.append(query_id)
            groups.append((query_id, fold))
            leaves.append(
                {
                    "kind": "train_group",
                    "outer_fold": outer,
                    "query_id": query_id,
                    "fold": fold,
                    "row_count": count,
                    "table_sha256": logical_table_sha256_v13(table),
                }
            )
            row_start = stop
            group += 1
    if (
        row_start != descriptor.row_count
        or group != descriptor.query_count
        or _sequence_sha256(groups) != descriptor.group_sequence_sha256
        or merkle_root_v13(leaves) != descriptor.logical_merkle_root
    ):
        raise FoldFusedInvariantError("fragmented train materialization drift")
    return FragmentedTrainArraysV13(
        outer_fold=outer,
        values=np.ascontiguousarray(values, dtype=np.float32),
        labels=labels,
        qid=qid,
        query_ids=tuple(query_ids),
        feature_names=descriptor.feature_names,
        group_sequence_sha256=descriptor.group_sequence_sha256,
        logical_merkle_root=descriptor.logical_merkle_root,
    )


class XGBoostFragmentedSiteLambdaMARTBackendV13:
    """Frozen XGBRanker recipe fitted from fragment-backed arrays."""

    def __init__(self, *, params: Mapping[str, Any], output_dir: Path) -> None:
        self.params = dict(params)
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.recipe_sha256 = _sha256_json(self.params)

    def fit(
        self,
        dataset: BuildFragmentDatasetV13,
        outer_fold: int,
    ) -> FittedSiteModel:
        import xgboost as xgb

        arrays = load_fragmented_train_arrays_v13(dataset, outer_fold)
        model = xgb.XGBRanker(**self.params)
        model.fit(arrays.values, arrays.labels, qid=arrays.qid, verbose=False)
        path = self.output_dir / f"fold{outer_fold}_site_ranker.json"
        model.save_model(path)
        return FittedSiteModel(
            outer_fold=int(outer_fold),
            handle=model,
            path=path,
            sha256=sha256_file(path),
            recipe_sha256=self.recipe_sha256,
            feature_names=arrays.feature_names,
        )

    def score(self, model: FittedSiteModel, features: Any) -> np.ndarray:
        if tuple(features.feature_names) != tuple(model.feature_names):
            raise FoldFusedInvariantError("validation feature schema differs from fit")
        return np.asarray(model.handle.predict(features.values), dtype=np.float64)


def fit_site_models_sequential_from_fragments_v13(
    backend: Any,
    dataset: BuildFragmentDatasetV13,
) -> Mapping[int, FittedSiteModel]:
    """Fit outer 2/3/4 sequentially so only one large matrix is resident."""

    models: dict[int, FittedSiteModel] = {}
    for outer in TRAIN_FOLDS:
        model = backend.fit(dataset, outer)
        if (
            model.outer_fold != outer
            or tuple(model.feature_names) != dataset.train[outer].feature_names
            or not _valid_sha256(model.sha256)
            or not _valid_sha256(model.recipe_sha256)
        ):
            raise FoldFusedInvariantError("v13 sequential model identity drift")
        models[outer] = model
        # The next outer fit rereads its fragments and does not reuse the
        # previous XGBoost matrices.  Force reclamation between outers so the
        # three sequential fits cannot retain a hidden cumulative RSS tail.
        gc.collect()
    if len({value.recipe_sha256 for value in models.values()}) != 1:
        raise FoldFusedInvariantError("v13 site models mix frozen recipes")
    return models


def iter_heldout_query_tables_v13(
    dataset: BuildFragmentDatasetV13,
) -> Iterable[pa.Table]:
    """Yield held-out query tables in the exact logical monolithic order."""

    observed: list[tuple[str, int]] = []
    rows_seen = 0
    leaves: list[Mapping[str, Any]] = []
    for fragment, spec in zip(
        dataset.heldout_fragments, dataset.specs, strict=True
    ):
        shard: list[tuple[str, int]] = []
        for table in _iter_heldout_query_tables(fragment.path):
            query_ids = {str(value) for value in table["query_id"].to_pylist()}
            folds = {int(value.as_py()) for value in table["fold"]}
            ranks = tuple(int(value.as_py()) for value in table["master_rank"])
            if (
                len(query_ids) != 1
                or len(folds) != 1
                or not folds.issubset(_ALLOWED_FOLDS)
                or ranks != tuple(range(1, table.num_rows + 1))
                or not 1 <= table.num_rows <= 2_000
            ):
                raise FoldFusedInvariantError("v13 heldout query semantics drift")
            current = (next(iter(query_ids)), next(iter(folds)))
            shard.append(current)
            observed.append(current)
            rows_seen += table.num_rows
            leaves.append(
                {
                    "kind": "heldout_query",
                    "query_id": current[0],
                    "fold": current[1],
                    "row_count": table.num_rows,
                    "table_sha256": logical_table_sha256_v13(table),
                }
            )
            yield table
        if tuple(shard) != tuple(
            zip(spec.query_ids, spec.query_folds, strict=True)
        ):
            raise FoldFusedInvariantError("v13 heldout shard order drift")
    if (
        rows_seen != dataset.heldout_row_count
        or len(observed) != dataset.query_count
        or _sequence_sha256(observed) != dataset.query_sequence_sha256
        or merkle_root_v13(leaves) != dataset.heldout_logical_merkle_root
    ):
        raise FoldFusedInvariantError("v13 heldout dataset cardinality/content drift")


def iter_heldout_contexts_from_fragments_v13(
    dataset: BuildFragmentDatasetV13,
    *,
    payload_factory: Callable[[tuple[Mapping[str, Any], ...]], Any] | None = None,
) -> Iterable[MasterContext]:
    """Reconstruct exact v12 Master contexts without a merged heldout spool."""

    if payload_factory is None:
        from .retrieval_fold_fused_runtime_v12 import _MasterPayload

        payload_factory = _MasterPayload
    for table in iter_heldout_query_tables_v13(dataset):
        rows = tuple(table.to_pylist())
        query_id = str(rows[0]["query_id"])
        fold = int(rows[0]["fold"])
        yield MasterContext(
            query_id=query_id,
            scene_fold=fold,
            outer_fold=fold,
            candidate_sirets=tuple(str(row["siret"]) for row in rows),
            candidate_parents=tuple(str(row["siren"]) for row in rows),
            opaque=payload_factory(rows),
            metadata={
                "heldout_spool_schema_sha256": dataset.heldout_schema_sha256,
                "heldout_spool_query_sequence_sha256": dataset.query_sequence_sha256,
                "fragment_dataset_id": dataset.dataset_id,
            },
        )


def logical_train_view_from_parquet_v13(
    paths: Sequence[Path],
    outer_fold: int,
) -> Mapping[str, Any]:
    """Compute a layout-independent train digest for v12 or v13 Parquets."""

    outer = int(outer_fold)
    if outer not in _ALLOWED_FOLDS or not paths:
        raise FoldFusedInvariantError("logical train view requires outer 2/3/4")
    names: tuple[str, ...] | None = None
    schema_sha: str | None = None
    groups: list[tuple[str, int]] = []
    leaves: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    rows = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        current_names = _feature_names_from_train_schema(parquet.schema_arrow)
        current_schema = _schema_sha256(
            _canonical_compact_train_schema_v12(
                parquet.schema_arrow, current_names
            )
        )
        if names is None:
            names, schema_sha = current_names, current_schema
        elif names != current_names or schema_sha != current_schema:
            raise FoldFusedInvariantError("logical train comparison schema drift")
        for row_group in range(parquet.num_row_groups):
            row_group_table = parquet.read_row_group(row_group)
            query_values = row_group_table["query_id"].to_pylist()
            start = 0
            while start < len(query_values):
                query_id = str(query_values[start])
                stop = start + 1
                while stop < len(query_values) and str(query_values[stop]) == query_id:
                    stop += 1
                if query_id in seen:
                    raise FoldFusedInvariantError(
                        "logical train query split or duplicated"
                    )
                seen.add(query_id)
                table = row_group_table.slice(start, stop - start)
                folds = {int(value.as_py()) for value in table["scene_fold"]}
                labels = np.asarray(table["site_label"], dtype=np.int8)
                if (
                    len(folds) != 1
                    or outer in folds
                    or not folds.issubset(_ALLOWED_FOLDS)
                    or int(labels.sum()) != 1
                    or not 2 <= table.num_rows <= 128
                ):
                    raise FoldFusedInvariantError(
                        "logical train comparison group drift"
                    )
                fold = next(iter(folds))
                groups.append((query_id, fold))
                rows += table.num_rows
                leaves.append(
                    {
                        "kind": "train_group",
                        "outer_fold": outer,
                        "query_id": query_id,
                        "fold": fold,
                        "row_count": table.num_rows,
                        "table_sha256": logical_table_sha256_v13(table),
                    }
                )
                start = stop
    if names is None or schema_sha is None or not groups:
        raise FoldFusedInvariantError("logical train comparison is empty")
    return {
        "outer_fold": outer,
        "row_count": rows,
        "query_count": len(groups),
        "feature_names": list(names),
        "parquet_schema_sha256": schema_sha,
        "group_sequence_sha256": _sequence_sha256(groups),
        "logical_merkle_root": merkle_root_v13(leaves),
    }


def logical_heldout_view_from_parquet_v13(
    paths: Sequence[Path],
) -> Mapping[str, Any]:
    """Compute a layout-independent heldout digest across ordered files."""

    if not paths:
        raise FoldFusedInvariantError("logical heldout comparison is empty")
    sequence: list[tuple[str, int]] = []
    leaves: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    rows = 0
    schema_sha: str | None = None
    for path in paths:
        parquet = pq.ParquetFile(path)
        current_schema = _schema_sha256(
            _writer_side_arrow_schema_v12(parquet.schema_arrow)
        )
        if schema_sha is None:
            schema_sha = current_schema
        elif schema_sha != current_schema:
            raise FoldFusedInvariantError("logical heldout comparison schema drift")
        for table in _iter_heldout_query_tables(path):
            query_id = str(table["query_id"][0].as_py())
            fold = int(table["fold"][0].as_py())
            ranks = tuple(int(value.as_py()) for value in table["master_rank"])
            if (
                query_id in seen
                or fold not in _ALLOWED_FOLDS
                or ranks != tuple(range(1, table.num_rows + 1))
            ):
                raise FoldFusedInvariantError(
                    "logical heldout comparison query drift"
                )
            seen.add(query_id)
            sequence.append((query_id, fold))
            rows += table.num_rows
            leaves.append(
                {
                    "kind": "heldout_query",
                    "query_id": query_id,
                    "fold": fold,
                    "row_count": table.num_rows,
                    "table_sha256": logical_table_sha256_v13(table),
                }
            )
    assert schema_sha is not None
    return {
        "row_count": rows,
        "query_count": len(sequence),
        "parquet_schema_sha256": schema_sha,
        "query_sequence_sha256": _sequence_sha256(sequence),
        "logical_merkle_root": merkle_root_v13(leaves),
    }


def build_logical_parity_digests_v13(
    dataset: BuildFragmentDatasetV13,
) -> Mapping[str, Any]:
    """Stable bridge digests comparable with a monolithic W1 v12 run."""

    train = {
        str(outer): logical_train_view_from_parquet_v13(
            [value.path for value in dataset.train[outer].fragments], outer
        )
        for outer in TRAIN_FOLDS
    }
    heldout = logical_heldout_view_from_parquet_v13(
        [value.path for value in dataset.heldout_fragments]
    )
    return {
        "train_matrix": {
            key: value["logical_merkle_root"] for key, value in train.items()
        },
        "group_sequence": {
            key: value["group_sequence_sha256"] for key, value in train.items()
        },
        "heldout_spool": heldout["logical_merkle_root"],
        "query_sequence": heldout["query_sequence_sha256"],
    }


__all__ = [
    "BuildFragmentDatasetV13",
    "BufferedCompactTrainingStoreV13",
    "ENGINE_ID",
    "FragmentFileV13",
    "FragmentedTrainArraysV13",
    "SCHEMA_VERSION",
    "TrainOuterDatasetV13",
    "XGBoostFragmentedSiteLambdaMARTBackendV13",
    "fit_site_models_sequential_from_fragments_v13",
    "build_logical_parity_digests_v13",
    "iter_heldout_contexts_from_fragments_v13",
    "iter_heldout_query_tables_v13",
    "load_build_fragment_dataset_v13",
    "load_fragmented_train_arrays_v13",
    "logical_heldout_view_from_parquet_v13",
    "logical_train_view_from_parquet_v13",
    "logical_table_sha256_v13",
    "merkle_root_v13",
    "seal_build_fragment_dataset_v13",
]
