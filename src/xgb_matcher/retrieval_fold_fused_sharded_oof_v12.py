"""Exact query-sharded physical executor for retrieval v12.

This module is deliberately a *physical* layer over the frozen fold-fused
scientific implementation.  It partitions the immutable SceneIR query order
into contiguous weighted ranges, lets isolated processes build narrow train
matrices and label-blind held-out spools, and then merges those artifacts back
in the exact monolithic order before fitting.  Held-out replay can likewise be
performed per range and merged without changing candidates, features, models,
selection rules, labels or metrics.

No worker accepts an Arrow, DuckDB or XGBoost object from its parent process.
The optional process launcher exchanges JSON/pickle-safe job descriptions and
content-addressed manifest paths only.  Fold 0 and fold 1 are rejected at every
boundary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from importlib import import_module
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable, Iterable, Mapping, Sequence
import bisect
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .retrieval_fold_fused_heldout_spool_v12 import (
    DEFAULT_ROW_GROUP_TARGET,
    MANIFEST_VERSION as HELDOUT_MANIFEST_VERSION,
    ParquetHeldoutMasterSpoolV12,
)
from .retrieval_fold_fused_prefit_bridge_v12 import (
    PrefitBridgeArtifactV12,
    load_prefit_bridge_artifact_v12,
    merge_prefit_bridge_artifacts_v12,
    prefit_bridge_record_v12,
)
from .retrieval_fold_fused_oof_v12 import (
    ArtifactRecord,
    CompactTrainBatch,
    ExecutionCaps,
    FeatureBatch,
    FittedSiteModel,
    FoldFusedInvariantError,
    FullMasterStatistics,
    GateThresholds,
    HeldoutMasterArtifact,
    HumanLabel,
    MasterContext,
    ParquetCompactTrainingStore,
    ParquetOOFOutputStore,
    QueryOutcome,
    RankingSelection,
    SceneRecord,
    TRAIN_FOLDS,
    TrainMatrixArtifact,
    sha256_file,
)


SHARD_SPEC_SCHEMA_VERSION = "sireto-retrieval-v12-shard-spec-1"
SHARD_BUILD_SCHEMA_VERSION = "sireto-retrieval-v12-shard-build-manifest-1"
SHARD_REPLAY_SCHEMA_VERSION = "sireto-retrieval-v12-shard-replay-manifest-1"
SHARDED_RUN_SCHEMA_VERSION = "sireto-retrieval-v12-sharded-oof-manifest-1"
ENGINE_ID = "FOLD_FUSED_CROSSFIT_SHARDED_V12_1"
_ALLOWED_FOLDS = frozenset(TRAIN_FOLDS)
_SHA256 = frozenset("0123456789abcdef")


def _canonical_json(value: Any) -> str:
    def compatible(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): compatible(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [compatible(value) for value in item]
        if isinstance(item, (set, frozenset)):
            values = [compatible(value) for value in item]
            return sorted(values, key=repr)
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


def _sequence_sha256(values: Sequence[tuple[str, int]]) -> str:
    return _sha256_json([[query_id, int(fold)] for query_id, fold in values])


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(_SHA256)
    )


def _validate_lineage(lineage: Mapping[str, str]) -> dict[str, str]:
    output = {str(key): str(value) for key, value in lineage.items()}
    if not output or any(not _valid_sha256(value) for value in output.values()):
        raise FoldFusedInvariantError(
            "sharded execution lineage must contain lowercase SHA-256 values"
        )
    return output


def _schema_descriptor(schema: pa.Schema) -> list[dict[str, Any]]:
    return [
        {
            "name": field.name,
            "type": str(field.type),
            "nullable": bool(field.nullable),
        }
        for field in schema
    ]


def _relative_or_absolute(path: Path, root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(root.resolve()))
    except ValueError:
        return str(resolved)


def _resolve_record_path(root: Path, value: Any) -> Path:
    path = Path(str(value or ""))
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, pa.Table):
        raw = value.to_pylist()
    elif hasattr(value, "to_dict"):
        try:
            raw = value.to_dict("records")
        except TypeError:
            raw = value.to_dict()
    else:
        raw = list(value)
    if isinstance(raw, Mapping):
        keys = list(raw)
        lengths = {len(raw[key]) for key in keys}
        if len(lengths) != 1:
            raise FoldFusedInvariantError("scene index columns differ in length")
        raw = [
            {key: raw[key][index] for key in keys}
            for index in range(next(iter(lengths), 0))
        ]
    output = [dict(row) for row in raw]
    return output


@dataclass(frozen=True)
class ShardSpecV12:
    """One contiguous slice of the *selected* immutable SceneIR order."""

    shard_id: str
    shard_ordinal: int
    sequence_start: int
    sequence_stop: int
    query_ids: tuple[str, ...]
    query_folds: tuple[int, ...]
    source_ordinals: tuple[int, ...]
    source_ranges: tuple[tuple[int, int], ...]
    candidate_row_count: int
    query_sequence_sha256: str

    def __post_init__(self) -> None:
        count = self.sequence_stop - self.sequence_start
        if (
            not self.shard_id
            or self.shard_ordinal < 0
            or self.sequence_start < 0
            or count <= 0
            or count != len(self.query_ids)
            or len(self.query_ids) != len(self.query_folds)
            or len(self.query_ids) != len(self.source_ordinals)
            or len(set(self.query_ids)) != len(self.query_ids)
            or any(not query_id for query_id in self.query_ids)
            or not set(self.query_folds).issubset(_ALLOWED_FOLDS)
            or any(
                self.source_ordinals[index] >= self.source_ordinals[index + 1]
                for index in range(len(self.source_ordinals) - 1)
            )
            or self.candidate_row_count <= 0
            or not _valid_sha256(self.query_sequence_sha256)
        ):
            raise FoldFusedInvariantError("invalid contiguous v12 shard specification")
        if self.query_sequence_sha256 != _sequence_sha256(
            tuple(zip(self.query_ids, self.query_folds, strict=True))
        ):
            raise FoldFusedInvariantError("shard query-sequence digest drift")
        previous_stop = -1
        for start, stop in self.source_ranges:
            if start < 0 or stop <= start or start < previous_stop:
                raise FoldFusedInvariantError("shard source ranges overlap or are invalid")
            previous_stop = stop

    @property
    def folds(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.query_folds)))

    @property
    def query_count(self) -> int:
        return len(self.query_ids)

    @property
    def digest(self) -> str:
        return _sha256_json(self.as_manifest())

    def as_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": SHARD_SPEC_SCHEMA_VERSION,
            "shard_id": self.shard_id,
            "shard_ordinal": self.shard_ordinal,
            "sequence_start": self.sequence_start,
            "sequence_stop": self.sequence_stop,
            "query_ids": list(self.query_ids),
            "query_folds": list(self.query_folds),
            "source_ordinals": list(self.source_ordinals),
            "source_ranges": [list(value) for value in self.source_ranges],
            "candidate_row_count": self.candidate_row_count,
            "query_sequence_sha256": self.query_sequence_sha256,
            "fold0_opened": False,
            "fold1_opened": False,
        }

    @classmethod
    def from_manifest(cls, value: Mapping[str, Any]) -> "ShardSpecV12":
        if value.get("schema_version") != SHARD_SPEC_SCHEMA_VERSION:
            raise FoldFusedInvariantError("unsupported v12 shard specification")
        if value.get("fold0_opened") is not False or value.get("fold1_opened") is not False:
            raise FoldFusedInvariantError("shard specification attempted fold 0/1")
        return cls(
            shard_id=str(value.get("shard_id") or ""),
            shard_ordinal=int(value.get("shard_ordinal", -1)),
            sequence_start=int(value.get("sequence_start", -1)),
            sequence_stop=int(value.get("sequence_stop", -1)),
            query_ids=tuple(str(item) for item in value.get("query_ids") or ()),
            query_folds=tuple(int(item) for item in value.get("query_folds") or ()),
            source_ordinals=tuple(int(item) for item in value.get("source_ordinals") or ()),
            source_ranges=tuple(
                (int(item[0]), int(item[1]))
                for item in value.get("source_ranges") or ()
            ),
            candidate_row_count=int(value.get("candidate_row_count", 0)),
            query_sequence_sha256=str(value.get("query_sequence_sha256") or ""),
        )


def _coalesced_source_ranges(
    selected: Sequence[Mapping[str, Any]],
) -> tuple[tuple[int, int], ...]:
    output: list[tuple[int, int]] = []
    for row in selected:
        start = int(row["row_start"])
        stop = start + int(row["row_count"])
        if output and output[-1][1] == start:
            output[-1] = (output[-1][0], stop)
        else:
            output.append((start, stop))
    return tuple(output)


def plan_query_shards(
    scene_index: Any,
    query_ids: Sequence[str],
    workers: int,
) -> tuple[ShardSpecV12, ...]:
    """Partition selected queries into deterministic, row-weighted ranges.

    The order is always the global SceneIR order, never caller order.  For the
    national population every spec owns one contiguous source range.  A sparse
    benchmark selection can own several non-overlapping ranges, but every
    worker still sees a contiguous slice of the selected sequence.
    """

    rows = _records(scene_index)
    required = {"query_id", "fold", "row_start", "row_count"}
    if not rows or any(not required.issubset(row) for row in rows):
        raise FoldFusedInvariantError("scene index lacks sealed range columns")
    observed_ids: set[str] = set()
    previous_stop = 0
    normalised: list[dict[str, Any]] = []
    for ordinal, raw in enumerate(rows):
        query_id = str(raw["query_id"] or "")
        fold = int(raw["fold"])
        start = int(raw["row_start"])
        count = int(raw["row_count"])
        if (
            not query_id
            or query_id in observed_ids
            or fold not in _ALLOWED_FOLDS
            or start != previous_stop
            or not 1 <= count <= 2_000
        ):
            raise FoldFusedInvariantError(
                "scene index is duplicated, non-contiguous or opens fold 0/1"
            )
        observed_ids.add(query_id)
        normalised.append(
            {
                "query_id": query_id,
                "fold": fold,
                "row_start": start,
                "row_count": count,
                "source_ordinal": ordinal,
            }
        )
        previous_stop = start + count

    requested = tuple(str(value) for value in query_ids)
    wanted = set(requested)
    if (
        not requested
        or len(wanted) != len(requested)
        or any(not value for value in requested)
        or not wanted.issubset(observed_ids)
        or not isinstance(workers, int)
        or not 1 <= workers <= len(requested)
    ):
        raise FoldFusedInvariantError("invalid query population or worker count")
    selected = [row for row in normalised if row["query_id"] in wanted]
    if len(selected) != len(requested):
        raise FoldFusedInvariantError("shard planner lost a selected query")

    fold_boundaries = [
        index
        for index in range(1, len(selected))
        if int(selected[index - 1]["fold"]) != int(selected[index]["fold"])
    ]
    natural_three_fold_plan = bool(
        workers == 3
        and tuple(sorted({int(row["fold"]) for row in selected})) == TRAIN_FOLDS
        and len(fold_boundaries) == 2
    )
    if natural_three_fold_plan:
        # The preregistered national W3 layout is one exact contiguous range
        # per outer fold.  Preserve that scientific/operational boundary even
        # when the three candidate-row counts differ slightly.
        boundaries = [0, *fold_boundaries, len(selected)]
    else:
        weights = [int(row["row_count"]) for row in selected]
        prefix = [0]
        for value in weights:
            prefix.append(prefix[-1] + value)
        boundaries = [0]
        for split in range(1, workers):
            lower = boundaries[-1] + 1
            upper = len(selected) - (workers - split)
            target = prefix[-1] * split / workers
            position = bisect.bisect_left(prefix, target, lo=lower, hi=upper + 1)
            candidates = {max(lower, min(upper, position))}
            if position - 1 >= lower:
                candidates.add(position - 1)
            boundary = min(
                candidates,
                key=lambda value: (abs(prefix[value] - target), value),
            )
            boundaries.append(boundary)
        boundaries.append(len(selected))

    specs: list[ShardSpecV12] = []
    for shard_ordinal, (start, stop) in enumerate(
        zip(boundaries, boundaries[1:])
    ):
        values = selected[start:stop]
        sequence = tuple(
            (str(row["query_id"]), int(row["fold"])) for row in values
        )
        shard_id = (
            f"fold{sequence[0][1]}"
            if natural_three_fold_plan
            else f"shard-{shard_ordinal:03d}"
        )
        specs.append(
            ShardSpecV12(
                shard_id=shard_id,
                shard_ordinal=shard_ordinal,
                sequence_start=start,
                sequence_stop=stop,
                query_ids=tuple(value[0] for value in sequence),
                query_folds=tuple(value[1] for value in sequence),
                source_ordinals=tuple(
                    int(row["source_ordinal"]) for row in values
                ),
                source_ranges=_coalesced_source_ranges(values),
                candidate_row_count=sum(int(row["row_count"]) for row in values),
                query_sequence_sha256=_sequence_sha256(sequence),
            )
        )
    if (
        tuple(value for spec in specs for value in spec.query_ids)
        != tuple(str(row["query_id"]) for row in selected)
        or any(
            specs[index].sequence_stop != specs[index + 1].sequence_start
            for index in range(len(specs) - 1)
        )
    ):
        raise AssertionError("v12 shard planner failed exact coverage")
    return tuple(specs)


@dataclass(frozen=True)
class ShardBuildArtifactV12:
    spec: ShardSpecV12
    manifest_path: Path
    manifest_sha256: str
    build_id: str
    lineage: Mapping[str, str]
    train_matrices: Mapping[int, TrainMatrixArtifact]
    train_group_sequence_sha256: Mapping[int, str]
    heldout_master: HeldoutMasterArtifact
    prefit_bridge: PrefitBridgeArtifactV12 | None
    diagnostics: Mapping[str, Any]
    digests: Mapping[str, Any]


@dataclass(frozen=True)
class ShardReplayArtifactV12:
    spec: ShardSpecV12
    manifest_path: Path
    manifest_sha256: str
    build_manifest_sha256: str
    lineage: Mapping[str, str]
    outputs: Mapping[str, ArtifactRecord]
    outcomes: tuple[QueryOutcome, ...]
    digests: Mapping[str, Any]


@dataclass(frozen=True)
class MergedTrainingV12:
    matrices: Mapping[int, TrainMatrixArtifact]
    group_sequence_sha256: Mapping[int, str]
    logical_sha256: str


@dataclass(frozen=True)
class MergedHeldoutV12:
    artifact: HeldoutMasterArtifact
    logical_sha256: str


@dataclass(frozen=True)
class MergedReplayV12:
    outputs: Mapping[str, ArtifactRecord]
    outcomes: tuple[QueryOutcome, ...]
    query_sequence_sha256: str
    digests: Mapping[str, str]


@dataclass(frozen=True)
class WorkerJobV12:
    job_id: str
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.job_id:
            raise FoldFusedInvariantError("worker job_id must be non-empty")
        # Fails early for Path/Arrow/model objects accidentally shared through
        # the process boundary.  Entrypoints reconstruct heavy readers/models.
        _canonical_json(self.payload)


def _feature_names_sha256_v12(feature_names: Sequence[str]) -> str:
    names = tuple(str(value) for value in feature_names)
    if not names or len(names) != len(set(names)) or any(not value for value in names):
        raise FoldFusedInvariantError("site model feature schema is invalid")
    return hashlib.sha256(
        _canonical_json(list(names)).encode("utf-8")
    ).hexdigest()


def heldout_master_record_v12(
    artifact: HeldoutMasterArtifact,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    """Serialize every field needed for strict replay reconstruction."""

    base = Path(root).resolve() if root is not None else None
    path = Path(artifact.path).resolve()
    manifest_path = Path(artifact.manifest_path).resolve()
    return {
        "path": (
            _relative_or_absolute(path, base) if base is not None else str(path)
        ),
        "manifest_path": (
            _relative_or_absolute(manifest_path, base)
            if base is not None
            else str(manifest_path)
        ),
        "sha256": artifact.sha256,
        "manifest_sha256": artifact.manifest_sha256,
        "row_count": int(artifact.row_count),
        "query_count": int(artifact.query_count),
        "size_bytes": int(artifact.size_bytes),
        "maximum_bytes": int(artifact.maximum_bytes),
        "schema_sha256": artifact.schema_sha256,
        "query_sequence_sha256": artifact.query_sequence_sha256,
    }


def load_heldout_master_record_v12(
    record: Mapping[str, Any],
    *,
    root: Path | None = None,
    verify_hashes: bool = True,
) -> HeldoutMasterArtifact:
    """Fail-closed loader for a label-blind held-out spool record.

    The Parquet content address and its own manifest are verified here.  Replay
    subsequently recomputes the complete query sequence while consuming the
    rows, so this loader need not add a second full scan before replay.
    """

    base = Path(root).resolve() if root is not None else Path.cwd().resolve()
    path = _resolve_record_path(base, record.get("path"))
    manifest_path = _resolve_record_path(base, record.get("manifest_path"))
    sha256 = str(record.get("sha256") or "")
    manifest_sha256 = str(record.get("manifest_sha256") or "")
    row_count = int(record.get("row_count", -1))
    query_count = int(record.get("query_count", -1))
    size_bytes = int(record.get("size_bytes", -1))
    maximum_bytes = int(record.get("maximum_bytes", -1))
    schema_sha256 = str(record.get("schema_sha256") or "")
    sequence_sha256 = str(record.get("query_sequence_sha256") or "")
    if (
        not path.is_file()
        or not manifest_path.is_file()
        or not _valid_sha256(sha256)
        or not _valid_sha256(manifest_sha256)
        or not _valid_sha256(schema_sha256)
        or not _valid_sha256(sequence_sha256)
        or row_count < query_count
        or query_count <= 0
        or size_bytes <= 0
        or maximum_bytes < size_bytes
        or path.stat().st_size != size_bytes
        or (verify_hashes and sha256_file(path) != sha256)
        or (verify_hashes and sha256_file(manifest_path) != manifest_sha256)
    ):
        raise FoldFusedInvariantError("heldout replay record is incomplete or corrupt")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "schema_version": HELDOUT_MANIFEST_VERSION,
        "status": "SEALED_LABEL_BLIND",
        "folds": list(TRAIN_FOLDS),
        "fold0_opened": False,
        "fold1_opened": False,
        "positive_injection": False,
        "label_columns": [],
        "schema_sha256": schema_sha256,
        "query_count": query_count,
        "row_count": row_count,
        "query_sequence_sha256": sequence_sha256,
        "maximum_bytes": maximum_bytes,
    }
    artifact_record = raw.get("artifact")
    if (
        any(raw.get(key) != value for key, value in required.items())
        or not isinstance(artifact_record, Mapping)
        or _resolve_record_path(manifest_path.parent, artifact_record.get("path"))
        != path
        or artifact_record.get("sha256") != sha256
        or artifact_record.get("size_bytes") != size_bytes
    ):
        raise FoldFusedInvariantError("heldout replay manifest drift")
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    writer_schema = _writer_side_arrow_schema_v12(schema)
    metadata = schema.metadata or {}
    if (
        parquet.metadata.num_rows != row_count
        or metadata.get(b"schema_sha256", b"").decode("ascii", "strict")
        != schema_sha256
        or raw.get("projection") != schema.names
        or raw.get("schema")
        != _schema_descriptor(writer_schema.remove_metadata())
    ):
        raise FoldFusedInvariantError("heldout replay Parquet schema/cardinality drift")
    return HeldoutMasterArtifact(
        path=path,
        manifest_path=manifest_path,
        sha256=sha256,
        manifest_sha256=manifest_sha256,
        row_count=row_count,
        query_count=query_count,
        size_bytes=size_bytes,
        maximum_bytes=maximum_bytes,
        schema_sha256=schema_sha256,
        query_sequence_sha256=sequence_sha256,
    )


def fitted_site_model_records_v12(
    models: Mapping[int, FittedSiteModel],
) -> dict[str, dict[str, Any]]:
    """Serialize three fitted models with recipe and feature-schema seals."""

    if set(models) != set(TRAIN_FOLDS):
        raise FoldFusedInvariantError("site model record requires outers 2/3/4")
    output: dict[str, dict[str, Any]] = {}
    recipes: set[str] = set()
    feature_schemas: set[str] = set()
    for outer in TRAIN_FOLDS:
        model = models[outer]
        path = Path(model.path).resolve()
        feature_schema = _feature_names_sha256_v12(model.feature_names)
        if (
            model.outer_fold != outer
            or not path.is_file()
            or not _valid_sha256(model.sha256)
            or sha256_file(path) != model.sha256
            or not _valid_sha256(model.recipe_sha256)
        ):
            raise FoldFusedInvariantError("fitted site model identity/hash drift")
        recipes.add(model.recipe_sha256)
        feature_schemas.add(feature_schema)
        output[str(outer)] = {
            "outer_fold": outer,
            "path": str(path),
            "sha256": model.sha256,
            "size_bytes": path.stat().st_size,
            "recipe_sha256": model.recipe_sha256,
            "feature_names": list(model.feature_names),
            "feature_schema_sha256": feature_schema,
        }
    if len(recipes) != 1 or len(feature_schemas) != 1:
        raise FoldFusedInvariantError("site models do not share recipe/feature schema")
    return output


def load_fitted_site_model_records_v12(
    records: Mapping[str, Any],
    *,
    handle_loader: Callable[[Path, int], Any],
    root: Path | None = None,
    verify_hashes: bool = True,
) -> dict[int, FittedSiteModel]:
    """Strictly reconstruct three models without sharing a live XGBoost object."""

    if set(records) != {str(value) for value in TRAIN_FOLDS}:
        raise FoldFusedInvariantError("site replay needs exactly three outer models")
    base = Path(root).resolve() if root is not None else Path.cwd().resolve()
    output: dict[int, FittedSiteModel] = {}
    recipes: set[str] = set()
    schemas: set[str] = set()
    for outer in TRAIN_FOLDS:
        record = records[str(outer)]
        if not isinstance(record, Mapping) or int(record.get("outer_fold", -1)) != outer:
            raise FoldFusedInvariantError("invalid outer site model record")
        path = _resolve_record_path(base, record.get("path"))
        digest = str(record.get("sha256") or "")
        size_bytes = int(record.get("size_bytes", -1))
        recipe = str(record.get("recipe_sha256") or "")
        names = tuple(str(value) for value in record.get("feature_names") or ())
        schema = str(record.get("feature_schema_sha256") or "")
        if (
            not path.is_file()
            or not _valid_sha256(digest)
            or not _valid_sha256(recipe)
            or not _valid_sha256(schema)
            or size_bytes <= 0
            or path.stat().st_size != size_bytes
            or _feature_names_sha256_v12(names) != schema
            or (verify_hashes and sha256_file(path) != digest)
        ):
            raise FoldFusedInvariantError("site replay model content/schema drift")
        recipes.add(recipe)
        schemas.add(schema)
        output[outer] = FittedSiteModel(
            outer_fold=outer,
            handle=handle_loader(path, outer),
            path=path,
            sha256=digest,
            recipe_sha256=recipe,
            feature_names=names,
        )
    if len(recipes) != 1 or len(schemas) != 1:
        raise FoldFusedInvariantError("site replay models mix recipe/feature schema")
    return output


@dataclass(frozen=True)
class ShardedPhysicalInputsV12:
    scene_index: Any
    query_ids: tuple[str, ...]
    lineage: Mapping[str, str]
    build_entrypoint: str
    build_payload: Mapping[str, Any]
    replay_entrypoint: str
    replay_payload: Mapping[str, Any]
    model_backend: Any
    caps: ExecutionCaps = field(default_factory=ExecutionCaps)
    gates: GateThresholds = field(default_factory=GateThresholds)
    maximum_spool_bytes: int = 20 * 1024**3
    maximum_workers: int = 3
    execution_evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_lineage(self.lineage)
        if (
            not self.query_ids
            or len(self.query_ids) != len(set(self.query_ids))
            or not 1 <= self.maximum_workers <= 4
            or self.maximum_spool_bytes <= 0
        ):
            raise FoldFusedInvariantError("invalid sharded physical inputs")
        for entrypoint in (self.build_entrypoint, self.replay_entrypoint):
            _parse_entrypoint(entrypoint)
        _canonical_json(self.build_payload)
        _canonical_json(self.replay_payload)
        _canonical_json(self.execution_evidence)


@dataclass(frozen=True)
class ShardedRunArtifactV12:
    manifest_path: Path
    manifest_sha256: str
    query_count: int
    folds: tuple[int, ...]
    lineage: Mapping[str, str]
    io_bytes: Mapping[str, int]
    specs: tuple[ShardSpecV12, ...]
    build_artifacts: tuple[ShardBuildArtifactV12, ...]
    train: MergedTrainingV12
    heldout: MergedHeldoutV12
    prefit_bridge: PrefitBridgeArtifactV12 | None
    models: Mapping[int, FittedSiteModel]
    replay_artifacts: tuple[ShardReplayArtifactV12, ...]
    replay: MergedReplayV12
    gates: Mapping[str, Any]
    digests: Mapping[str, Any]


class SceneIRShardSourceV12:
    """Expose one :class:`ShardSpecV12` through the existing SceneSource API."""

    def __init__(self, cache: Any, spec: ShardSpecV12) -> None:
        if not callable(getattr(cache, "iter_selected_scenes", None)):
            raise TypeError("sharded source requires SceneIR selective ranges")
        self.cache = cache
        self.spec = spec

    def iter_scenes(self, *, folds: tuple[int, ...]) -> Iterable[SceneRecord]:
        if tuple(folds) != TRAIN_FOLDS:
            raise FoldFusedInvariantError("sharded SceneIR source refuses fold 0/1")
        observed: list[tuple[str, int]] = []
        for scene in self.cache.iter_selected_scenes(self.spec.query_ids):
            if int(scene.fold) not in _ALLOWED_FOLDS:
                raise FoldFusedInvariantError("SceneIR shard opened fold 0/1")
            observed.append((str(scene.query_id), int(scene.fold)))
            yield SceneRecord(
                query_id=str(scene.query_id),
                fold=int(scene.fold),
                payload=scene,
                metadata={
                    "scene_ir_row_start": int(scene.row_start),
                    "scene_ir_row_count": int(scene.row_count),
                    "shard_id": self.spec.shard_id,
                },
            )
        expected = tuple(
            zip(self.spec.query_ids, self.spec.query_folds, strict=True)
        )
        if tuple(observed) != expected:
            raise FoldFusedInvariantError(
                "SceneIR shard sequence differs from the sealed specification"
            )


def _validate_master_context_v12(
    scene: SceneRecord,
    outer: int,
    context: MasterContext,
    caps: ExecutionCaps,
) -> None:
    if (
        outer not in _ALLOWED_FOLDS
        or context.query_id != scene.query_id
        or context.scene_fold != scene.fold
        or context.outer_fold != outer
        or not context.candidate_sirets
        or len(context.candidate_sirets) > caps.master
    ):
        raise FoldFusedInvariantError("sharded Master identity/cap drift")


def _validate_statistics_v12(
    context: MasterContext,
    statistics: FullMasterStatistics,
) -> None:
    if (
        not statistics.complete
        or statistics.query_id != context.query_id
        or statistics.outer_fold != context.outer_fold
        or statistics.master_count != len(context.candidate_sirets)
        or statistics.candidate_order_sha256
        != context.candidate_order_sha256
        or not _valid_sha256(statistics.statistics_sha256)
    ):
        raise FoldFusedInvariantError(
            "sharded statistics were not sealed on the complete Master"
        )


def _validate_feature_batch_v12(
    context: MasterContext,
    statistics: FullMasterStatistics,
    features: FeatureBatch,
    indices: Sequence[int],
) -> None:
    expected = tuple(int(value) for value in indices)
    if (
        features.query_id != context.query_id
        or features.scene_fold != context.scene_fold
        or features.outer_fold != context.outer_fold
        or features.statistics_sha256 != statistics.statistics_sha256
        or features.candidate_indices != expected
        or features.candidate_sirets
        != tuple(context.candidate_sirets[index] for index in expected)
        or features.candidate_parents
        != tuple(context.candidate_parents[index] for index in expected)
    ):
        raise FoldFusedInvariantError("sharded feature batch identity/order drift")


def build_shard_v12(
    spec: ShardSpecV12,
    *,
    source: Any,
    labels: Any,
    contexts: Any,
    sampler: Any,
    admission_config: Any,
    output_dir: Path,
    lineage: Mapping[str, str],
    caps: ExecutionCaps = ExecutionCaps(),
    maximum_spool_bytes: int = 20 * 1024**3,
    prefit_bridge_sink: Any | None = None,
    train_store_factory: Callable[[Path], Any] = ParquetCompactTrainingStore,
) -> ShardBuildArtifactV12:
    """Build exact train contributions and heldout spool for one shard.

    Label access remains after all three label-blind contexts have been built
    and after the heldout context has been appended to its spool.
    """

    lineage_value = _validate_lineage(lineage)
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    # V12 keeps its historical one-query-per-row-group sink by default.  A
    # physical-only executor may inject a schema-compatible buffered sink;
    # generation, sampling, features, labels and group order remain here.
    train_store = train_store_factory(output / "train")
    if any(
        not callable(getattr(train_store, method, None))
        for method in ("append", "seal_outer")
    ):
        raise FoldFusedInvariantError("train store contract drifted")
    heldout_store = ParquetHeldoutMasterSpoolV12(
        output / "heldout",
        config=admission_config,
        maximum_bytes=int(maximum_spool_bytes),
    )
    diagnostics: dict[int, dict[str, int]] = {
        outer: {
            "queries_seen": 0,
            "groups_written": 0,
            "rows_written": 0,
            "missing_natural_positive": 0,
            "without_negative": 0,
            "operational_neutrals_excluded": 0,
        }
        for outer in TRAIN_FOLDS
    }
    observed: list[tuple[str, int]] = []
    observed_fit_folds: dict[int, set[int]] = {
        outer: set() for outer in TRAIN_FOLDS
    }
    train_groups: dict[int, list[tuple[str, int]]] = {
        outer: [] for outer in TRAIN_FOLDS
    }
    feature_names: tuple[str, ...] | None = None

    for scene in source.iter_scenes(folds=TRAIN_FOLDS):
        if scene.fold not in _ALLOWED_FOLDS:
            raise FoldFusedInvariantError("shard worker attempted fold 0/1")
        observed.append((scene.query_id, scene.fold))
        prepared: dict[int, tuple[MasterContext, FullMasterStatistics]] = {}
        bridge_prepared: dict[int, tuple[MasterContext, FullMasterStatistics]] = {}
        for outer in TRAIN_FOLDS:
            context = contexts.prepare(scene, outer_fold=outer)
            _validate_master_context_v12(scene, outer, context, caps)
            if outer == scene.fold:
                heldout_store.append(context)
            if outer != scene.fold or prefit_bridge_sink is not None:
                statistics = contexts.compute_full_master_statistics(context)
                _validate_statistics_v12(context, statistics)
                if outer != scene.fold:
                    prepared[outer] = (context, statistics)
                if prefit_bridge_sink is not None:
                    # Benchmark-only direct bridge.  The complete 319-feature
                    # view is the exact historical PREFIT projection and is
                    # sealed before the ledger is opened below.  With no sink
                    # (the national path), this branch has zero materialisation.
                    full_features = contexts.materialize(
                        context,
                        statistics,
                        candidate_indices=None,
                    )
                    full_indices = tuple(range(len(context.candidate_sirets)))
                    _validate_feature_batch_v12(
                        context, statistics, full_features, full_indices
                    )
                    prefit_bridge_sink.observe_label_blind(
                        context, statistics, full_features
                    )
                    bridge_prepared[outer] = (context, statistics)

        label: HumanLabel = labels.get(scene.query_id)
        if label.query_id != scene.query_id or label.fold != scene.fold:
            raise FoldFusedInvariantError("shard label identity/fold mismatch")
        for outer in TRAIN_FOLDS:
            if outer == scene.fold:
                if prefit_bridge_sink is not None:
                    context, statistics = bridge_prepared[outer]
                    plan = sampler.choose(
                        context,
                        statistics,
                        label,
                        maximum_rows=caps.fit_sample,
                    )
                    indices = tuple(int(value) for value in plan.candidate_indices)
                    if (
                        plan.statistics_sha256 != statistics.statistics_sha256
                        or len(indices) > caps.fit_sample
                        or len(indices) != len(set(indices))
                        or any(
                            index < 0 or index >= len(context.candidate_sirets)
                            for index in indices
                        )
                    ):
                        raise FoldFusedInvariantError(
                            "shard heldout audit sampler drift/injection"
                        )
                    prefit_bridge_sink.observe_fit(context, plan)
                continue
            context, statistics = prepared[outer]
            diag = diagnostics[outer]
            diag["queries_seen"] += 1
            plan = sampler.choose(
                context,
                statistics,
                label,
                maximum_rows=caps.fit_sample,
            )
            indices = tuple(int(value) for value in plan.candidate_indices)
            if (
                plan.statistics_sha256 != statistics.statistics_sha256
                or len(indices) > caps.fit_sample
                or len(indices) != len(set(indices))
                or any(index < 0 or index >= len(context.candidate_sirets) for index in indices)
            ):
                raise FoldFusedInvariantError("shard fit sampler drift/injection")
            if prefit_bridge_sink is not None:
                prefit_bridge_sink.observe_fit(context, plan)
            if not indices:
                diag["missing_natural_positive"] += 1
                continue
            features = contexts.materialize(
                context,
                statistics,
                candidate_indices=indices,
            )
            _validate_feature_batch_v12(context, statistics, features, indices)
            if feature_names is None:
                feature_names = features.feature_names
            elif features.feature_names != feature_names:
                raise FoldFusedInvariantError("site feature schema drift across shard")
            neutral = set(label.operational_sirets) - {label.human_siret}
            keep = [
                position
                for position, siret in enumerate(features.candidate_sirets)
                if siret not in neutral
            ]
            diag["operational_neutrals_excluded"] += len(features.candidate_sirets) - len(keep)
            retained = features.take(keep)
            target = np.asarray(
                [int(siret == label.human_siret) for siret in retained.candidate_sirets],
                dtype=np.int8,
            )
            if int(target.sum()) != 1:
                diag["missing_natural_positive"] += 1
                continue
            if len(target) < 2:
                diag["without_negative"] += 1
                continue
            train_store.append(
                CompactTrainBatch(
                    query_id=scene.query_id,
                    scene_fold=scene.fold,
                    outer_fold=outer,
                    candidate_sirets=retained.candidate_sirets,
                    candidate_parents=retained.candidate_parents,
                    feature_names=retained.feature_names,
                    values=retained.values,
                    labels=target,
                    statistics_sha256=statistics.statistics_sha256,
                )
            )
            observed_fit_folds[outer].add(scene.fold)
            train_groups[outer].append((scene.query_id, scene.fold))
            diag["groups_written"] += 1
            diag["rows_written"] += len(target)

    expected_sequence = tuple(
        zip(spec.query_ids, spec.query_folds, strict=True)
    )
    if tuple(observed) != expected_sequence:
        raise FoldFusedInvariantError("shard source population/order drift")
    heldout = heldout_store.seal()
    if (
        heldout.query_count != spec.query_count
        or heldout.query_sequence_sha256 != spec.query_sequence_sha256
    ):
        raise FoldFusedInvariantError("shard heldout sequence drift")
    prefit_bridge = (
        prefit_bridge_sink.seal(output / "prefit_bridge")
        if prefit_bridge_sink is not None
        else None
    )

    matrices: dict[int, TrainMatrixArtifact] = {}
    for outer in TRAIN_FOLDS:
        if not train_groups[outer]:
            continue
        sealed = train_store.seal_outer(outer)
        matrices[outer] = TrainMatrixArtifact(
            outer_fold=outer,
            path=sealed.path,
            sha256=sealed.sha256,
            row_count=sealed.row_count,
            query_count=sealed.query_count,
            fit_folds=tuple(sorted(observed_fit_folds[outer])),
            feature_names=sealed.feature_names,
            feature_schema_sha256=sealed.feature_schema_sha256,
        )
    if feature_names is None or not matrices:
        raise FoldFusedInvariantError("shard produced no trainable site rows")

    group_digests = {
        outer: _sequence_sha256(train_groups[outer])
        for outer in matrices
    }
    digests: dict[str, Any] = {
        "train_matrix": {
            str(outer): artifact.sha256 for outer, artifact in matrices.items()
        },
        "group_sequence": {
            str(outer): group_digests[outer] for outer in matrices
        },
        "heldout_spool": heldout.sha256,
    }
    if prefit_bridge is not None:
        digests["prefit_bridge"] = prefit_bridge.sha256
    identity = {
        "engine_id": ENGINE_ID,
        "spec_sha256": spec.digest,
        "lineage": lineage_value,
        "digests": digests,
        "query_count": spec.query_count,
    }
    build_id = _sha256_json(identity)
    manifest_path = output / "manifest.json"
    manifest = {
        "schema_version": SHARD_BUILD_SCHEMA_VERSION,
        "status": "SEALED_SHARD_BUILD",
        "build_id": build_id,
        "build_identity": identity,
        "engine_id": ENGINE_ID,
        "spec": spec.as_manifest(),
        "lineage": lineage_value,
        "folds": list(spec.folds),
        "fold0_opened": False,
        "fold1_opened": False,
        "positive_injection": False,
        "query_count": spec.query_count,
        "query_sequence_sha256": spec.query_sequence_sha256,
        "train_matrices": {
            str(outer): {
                "path": _relative_or_absolute(artifact.path, output),
                "sha256": artifact.sha256,
                "rows": artifact.row_count,
                "queries": artifact.query_count,
                "fit_folds": list(artifact.fit_folds),
                "feature_names": list(artifact.feature_names),
                "feature_schema_sha256": artifact.feature_schema_sha256,
                "group_sequence_sha256": group_digests[outer],
            }
            for outer, artifact in matrices.items()
        },
        "heldout_master": {
            "path": _relative_or_absolute(heldout.path, output),
            "manifest_path": _relative_or_absolute(heldout.manifest_path, output),
            "sha256": heldout.sha256,
            "manifest_sha256": heldout.manifest_sha256,
            "rows": heldout.row_count,
            "queries": heldout.query_count,
            "size_bytes": heldout.size_bytes,
            "maximum_bytes": heldout.maximum_bytes,
            "schema_sha256": heldout.schema_sha256,
            "query_sequence_sha256": heldout.query_sequence_sha256,
        },
        "diagnostics": {
            str(outer): dict(value) for outer, value in diagnostics.items()
        },
        "digests": digests,
    }
    if prefit_bridge is not None:
        manifest["prefit_bridge"] = prefit_bridge_record_v12(
            prefit_bridge, root=output
        )
    _atomic_json(manifest_path, manifest)
    return ShardBuildArtifactV12(
        spec=spec,
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        build_id=build_id,
        lineage=lineage_value,
        train_matrices=matrices,
        train_group_sequence_sha256=group_digests,
        heldout_master=heldout,
        prefit_bridge=prefit_bridge,
        diagnostics=manifest["diagnostics"],
        digests=digests,
    )


def load_shard_build_manifest_v12(
    path: Path,
    *,
    verify_hashes: bool = True,
) -> ShardBuildArtifactV12:
    manifest_path = Path(path).resolve()
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        raw.get("schema_version") != SHARD_BUILD_SCHEMA_VERSION
        or raw.get("status") != "SEALED_SHARD_BUILD"
        or raw.get("engine_id") != ENGINE_ID
        or raw.get("fold0_opened") is not False
        or raw.get("fold1_opened") is not False
        or raw.get("positive_injection") is not False
    ):
        raise FoldFusedInvariantError("invalid or unsafe shard-build manifest")
    identity = raw.get("build_identity")
    if (
        not isinstance(identity, Mapping)
        or raw.get("build_id") != _sha256_json(identity)
    ):
        raise FoldFusedInvariantError("shard-build content address drift")
    spec = ShardSpecV12.from_manifest(raw.get("spec") or {})
    if (
        raw.get("query_count") != spec.query_count
        or raw.get("query_sequence_sha256") != spec.query_sequence_sha256
        or raw.get("folds") != list(spec.folds)
        or identity.get("spec_sha256") != spec.digest
    ):
        raise FoldFusedInvariantError("shard-build specification drift")
    lineage = _validate_lineage(raw.get("lineage") or {})
    if identity.get("lineage") != lineage:
        raise FoldFusedInvariantError("shard-build lineage drift")
    root = manifest_path.parent
    matrices: dict[int, TrainMatrixArtifact] = {}
    group_digests: dict[int, str] = {}
    records = raw.get("train_matrices")
    if not isinstance(records, Mapping) or not records:
        raise FoldFusedInvariantError("shard-build train matrices are missing")
    for key, record in records.items():
        outer = int(key)
        if outer not in _ALLOWED_FOLDS or not isinstance(record, Mapping):
            raise FoldFusedInvariantError("invalid shard outer train record")
        artifact_path = _resolve_record_path(root, record.get("path"))
        digest = str(record.get("sha256") or "")
        if not artifact_path.is_file() or not _valid_sha256(digest):
            raise FoldFusedInvariantError("shard train artifact is missing")
        if verify_hashes and sha256_file(artifact_path) != digest:
            raise FoldFusedInvariantError("shard train artifact hash mismatch")
        names = tuple(str(value) for value in record.get("feature_names") or ())
        schema_digest = str(record.get("feature_schema_sha256") or "")
        if not names or not _valid_sha256(schema_digest):
            raise FoldFusedInvariantError("shard train feature schema is missing")
        group_digest = str(record.get("group_sequence_sha256") or "")
        if not _valid_sha256(group_digest):
            raise FoldFusedInvariantError("shard train group sequence is missing")
        group_digests[outer] = group_digest
        matrices[outer] = TrainMatrixArtifact(
            outer_fold=outer,
            path=artifact_path,
            sha256=digest,
            row_count=int(record.get("rows", -1)),
            query_count=int(record.get("queries", -1)),
            fit_folds=tuple(int(value) for value in record.get("fit_folds") or ()),
            feature_names=names,
            feature_schema_sha256=schema_digest,
        )
    held = raw.get("heldout_master")
    if not isinstance(held, Mapping):
        raise FoldFusedInvariantError("shard heldout record is missing")
    heldout = load_heldout_master_record_v12(
        {
            "path": held.get("path"),
            "manifest_path": held.get("manifest_path"),
            "sha256": held.get("sha256"),
            "manifest_sha256": held.get("manifest_sha256"),
            "row_count": held.get("rows"),
            "query_count": held.get("queries"),
            "size_bytes": held.get("size_bytes"),
            "maximum_bytes": held.get("maximum_bytes"),
            "schema_sha256": held.get("schema_sha256"),
            "query_sequence_sha256": held.get("query_sequence_sha256"),
        },
        root=root,
        verify_hashes=verify_hashes,
    )
    if (
        heldout.query_count != spec.query_count
        or heldout.query_sequence_sha256 != spec.query_sequence_sha256
        or heldout.size_bytes != heldout.path.stat().st_size
    ):
        raise FoldFusedInvariantError("shard heldout cardinality/size drift")
    bridge_raw = raw.get("prefit_bridge")
    if bridge_raw is not None and not isinstance(bridge_raw, Mapping):
        raise FoldFusedInvariantError("shard PREFIT bridge record drifted")
    prefit_bridge = (
        load_prefit_bridge_artifact_v12(
            bridge_raw,
            root=root,
            verify_hashes=verify_hashes,
        )
        if isinstance(bridge_raw, Mapping)
        else None
    )
    if prefit_bridge is not None and (
        prefit_bridge.query_count != spec.query_count
        or dict(prefit_bridge.lineage) != lineage
    ):
        raise FoldFusedInvariantError("shard PREFIT bridge population/lineage drift")
    digests = raw.get("digests")
    expected_digests = {
        "train_matrix": {
            str(outer): artifact.sha256
            for outer, artifact in matrices.items()
        },
        "group_sequence": {
            str(outer): group_digests[outer] for outer in matrices
        },
        "heldout_spool": heldout.sha256,
    }
    if prefit_bridge is not None:
        expected_digests["prefit_bridge"] = prefit_bridge.sha256
    if (
        not isinstance(digests, Mapping)
        or identity.get("digests") != digests
        or dict(digests) != expected_digests
    ):
        raise FoldFusedInvariantError("shard build digest set drift")
    return ShardBuildArtifactV12(
        spec=spec,
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        build_id=str(raw["build_id"]),
        lineage=lineage,
        train_matrices=matrices,
        train_group_sequence_sha256=group_digests,
        heldout_master=heldout,
        prefit_bridge=prefit_bridge,
        diagnostics=dict(raw.get("diagnostics") or {}),
        digests=dict(digests),
    )


def _ordered_build_artifacts_v12(
    artifacts: Sequence[ShardBuildArtifactV12],
) -> tuple[ShardBuildArtifactV12, ...]:
    ordered = tuple(sorted(artifacts, key=lambda value: value.spec.shard_ordinal))
    if not ordered:
        raise FoldFusedInvariantError("no shard-build artifacts were supplied")
    lineage = dict(ordered[0].lineage)
    sequence_stop = 0
    seen_queries: set[str] = set()
    for ordinal, artifact in enumerate(ordered):
        spec = artifact.spec
        if (
            spec.shard_ordinal != ordinal
            or spec.sequence_start != sequence_stop
            or dict(artifact.lineage) != lineage
            or artifact.manifest_sha256 != sha256_file(artifact.manifest_path)
            or seen_queries.intersection(spec.query_ids)
        ):
            raise FoldFusedInvariantError(
                "shard builds overlap, leave a gap or have mixed lineage"
            )
        seen_queries.update(spec.query_ids)
        sequence_stop = spec.sequence_stop
    return ordered


def _feature_names_from_train_schema(schema: pa.Schema) -> tuple[str, ...]:
    metadata = schema.metadata or {}
    encoded = metadata.get(b"feature_names_json")
    if encoded is None:
        raise FoldFusedInvariantError("train matrix lacks feature-name metadata")
    names = tuple(str(value) for value in json.loads(encoded.decode("utf-8")))
    if not names or len(names) != len(set(names)):
        raise FoldFusedInvariantError("train feature-name metadata is invalid")
    return names


def _canonical_compact_train_schema_v12(
    observed: pa.Schema,
    feature_names: Sequence[str],
) -> pa.Schema:
    """Restore the writer-side FixedSizeList child name lost by Parquet read.

    PyArrow reconstructs ``fixed_size_list<item>`` as
    ``fixed_size_list<element>``.  The two schemas compare equal, but writing
    the reconstructed spelling changes the embedded Arrow schema and therefore
    the file SHA by eight bytes.  Reinstating ``item`` reproduces the exact
    monolithic compact-matrix bytes.
    """

    expected_names = (
        "query_id",
        "scene_fold",
        "candidate_siret",
        "candidate_parent",
        "site_label",
        "statistics_sha256",
        "feature_values",
    )
    width = len(tuple(feature_names))
    if tuple(observed.names) != expected_names or width <= 0:
        raise FoldFusedInvariantError("compact train physical schema drift")
    fields = list(observed)
    fields[-1] = pa.field(
        "feature_values",
        pa.list_(pa.field("item", pa.float32()), width),
    )
    canonical = pa.schema(fields, metadata=observed.metadata)
    observed_values = observed.field("feature_values")
    if (
        tuple(observed)[:6] != tuple(canonical)[:6]
        or observed.metadata != canonical.metadata
        or not pa.types.is_fixed_size_list(observed_values.type)
        or observed_values.type.list_size != width
        or observed_values.type.value_type != pa.float32()
        or observed_values.nullable != canonical.field("feature_values").nullable
    ):
        raise FoldFusedInvariantError("compact train feature matrix type drift")
    return canonical


def merge_training_shards_v12(
    artifacts: Sequence[ShardBuildArtifactV12],
    output_dir: Path,
) -> MergedTrainingV12:
    """Stream shard train groups into the exact monolithic query order."""

    ordered = _ordered_build_artifacts_v12(artifacts)
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    global_sequence = tuple(
        (query_id, fold)
        for artifact in ordered
        for query_id, fold in zip(
            artifact.spec.query_ids,
            artifact.spec.query_folds,
            strict=True,
        )
    )
    global_positions = {
        query_id: index for index, (query_id, _fold) in enumerate(global_sequence)
    }
    matrices: dict[int, TrainMatrixArtifact] = {}
    group_digests: dict[int, str] = {}

    for outer in TRAIN_FOLDS:
        sources = [
            artifact.train_matrices[outer]
            for artifact in ordered
            if outer in artifact.train_matrices
        ]
        if not sources:
            raise FoldFusedInvariantError(f"outer{outer} has no shard train rows")
        path = output / f"outer{outer}_site_train_matrix.parquet"
        writer: pq.ParquetWriter | None = None
        expected_schema: pa.Schema | None = None
        feature_names: tuple[str, ...] | None = None
        observed_groups: list[tuple[str, int]] = []
        seen: set[str] = set()
        previous_position = -1
        rows_written = 0
        try:
            for source in sources:
                parquet = pq.ParquetFile(source.path)
                schema = parquet.schema_arrow
                names = _feature_names_from_train_schema(schema)
                canonical_schema = _canonical_compact_train_schema_v12(
                    schema, names
                )
                if expected_schema is None:
                    expected_schema = canonical_schema
                    feature_names = names
                    writer = pq.ParquetWriter(
                        path, expected_schema, compression="zstd"
                    )
                elif (
                    not canonical_schema.equals(
                        expected_schema, check_metadata=True
                    )
                    or names != feature_names
                ):
                    raise FoldFusedInvariantError(
                        "train shard physical/feature schema drift"
                    )
                for row_group in range(parquet.num_row_groups):
                    table = parquet.read_row_group(row_group)
                    query_ids = {str(value) for value in table["query_id"].to_pylist()}
                    folds = {int(value.as_py()) for value in table["scene_fold"]}
                    labels = np.asarray(table["site_label"], dtype=np.int8)
                    statistics = {str(value) for value in table["statistics_sha256"].to_pylist()}
                    if (
                        len(query_ids) != 1
                        or len(folds) != 1
                        or outer in folds
                        or not folds.issubset(_ALLOWED_FOLDS)
                        or len(statistics) != 1
                        or int(labels.sum()) != 1
                        or not 2 <= table.num_rows <= 128
                    ):
                        raise FoldFusedInvariantError(
                            "train shard group semantics drift"
                        )
                    query_id = next(iter(query_ids))
                    fold = next(iter(folds))
                    position = global_positions.get(query_id, -1)
                    if (
                        query_id in seen
                        or position <= previous_position
                        or global_sequence[position][1] != fold
                    ):
                        raise FoldFusedInvariantError(
                            "train shard group order/identity drift"
                        )
                    seen.add(query_id)
                    previous_position = position
                    observed_groups.append((query_id, fold))
                    assert writer is not None
                    canonical_table = pa.Table.from_arrays(
                        table.columns, schema=expected_schema
                    )
                    writer.write_table(
                        canonical_table, row_group_size=canonical_table.num_rows
                    )
                    rows_written += table.num_rows
        finally:
            if writer is not None:
                writer.close()
        if expected_schema is None or feature_names is None or not observed_groups:
            raise FoldFusedInvariantError(f"outer{outer} merged matrix is empty")
        fit_folds = tuple(sorted({fold for _query_id, fold in observed_groups}))
        expected_fit_folds = tuple(fold for fold in TRAIN_FOLDS if fold != outer)
        if fit_folds != expected_fit_folds:
            raise FoldFusedInvariantError(
                f"outer{outer} merged matrix lacks a frozen fit fold"
            )
        digest = sha256_file(path)
        feature_schema_sha256 = hashlib.sha256(
            _canonical_json(list(feature_names)).encode("utf-8")
        ).hexdigest()
        group_digest = _sequence_sha256(observed_groups)
        group_digests[outer] = group_digest
        matrices[outer] = TrainMatrixArtifact(
            outer_fold=outer,
            path=path,
            sha256=digest,
            row_count=rows_written,
            query_count=len(observed_groups),
            fit_folds=fit_folds,
            feature_names=feature_names,
            feature_schema_sha256=feature_schema_sha256,
        )
    logical = _sha256_json(
        {
            "train_matrix": {
                str(outer): matrices[outer].sha256 for outer in TRAIN_FOLDS
            },
            "group_sequence": {
                str(outer): group_digests[outer] for outer in TRAIN_FOLDS
            },
        }
    )
    return MergedTrainingV12(matrices, group_digests, logical)


def _iter_heldout_query_tables(path: Path) -> Iterable[pa.Table]:
    parquet = pq.ParquetFile(path)
    seen: set[str] = set()
    for row_group in range(parquet.num_row_groups):
        table = parquet.read_row_group(row_group)
        query_ids = table["query_id"].to_pylist()
        start = 0
        while start < len(query_ids):
            query_id = str(query_ids[start])
            end = start + 1
            while end < len(query_ids) and str(query_ids[end]) == query_id:
                end += 1
            if query_id in seen:
                raise FoldFusedInvariantError(
                    "heldout shard query is split or duplicated"
                )
            seen.add(query_id)
            yield table.slice(start, end - start)
            start = end


def _writer_side_arrow_type_v12(data_type: pa.DataType) -> pa.DataType:
    """Restore nested child names emitted by the original in-memory schema."""

    if pa.types.is_fixed_size_list(data_type):
        value = data_type.value_field
        return pa.list_(
            pa.field(
                "item",
                _writer_side_arrow_type_v12(value.type),
                nullable=value.nullable,
                metadata=value.metadata,
            ),
            data_type.list_size,
        )
    if pa.types.is_list(data_type):
        value = data_type.value_field
        return pa.list_(
            pa.field(
                "item",
                _writer_side_arrow_type_v12(value.type),
                nullable=value.nullable,
                metadata=value.metadata,
            )
        )
    if pa.types.is_large_list(data_type):
        value = data_type.value_field
        return pa.large_list(
            pa.field(
                "item",
                _writer_side_arrow_type_v12(value.type),
                nullable=value.nullable,
                metadata=value.metadata,
            )
        )
    if pa.types.is_struct(data_type):
        return pa.struct(
            [
                pa.field(
                    field.name,
                    _writer_side_arrow_type_v12(field.type),
                    nullable=field.nullable,
                    metadata=field.metadata,
                )
                for field in data_type
            ]
        )
    return data_type


def _writer_side_arrow_schema_v12(observed: pa.Schema) -> pa.Schema:
    return pa.schema(
        [
            pa.field(
                field.name,
                _writer_side_arrow_type_v12(field.type),
                nullable=field.nullable,
                metadata=field.metadata,
            )
            for field in observed
        ],
        metadata=observed.metadata,
    )


def merge_heldout_shards_v12(
    artifacts: Sequence[ShardBuildArtifactV12],
    output_dir: Path,
    maximum_bytes: int = 20 * 1024**3,
) -> MergedHeldoutV12:
    """Repack label-blind rows with the monolithic 16,384-row policy."""

    ordered = _ordered_build_artifacts_v12(artifacts)
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    path = output / "heldout_master.parquet"
    manifest_path = output / "manifest.json"
    expected_schema: pa.Schema | None = None
    writer: pq.ParquetWriter | None = None
    buffer: list[pa.Table] = []
    buffer_rows = 0
    row_count = 0
    sequence: list[tuple[str, int]] = []

    def flush() -> None:
        nonlocal buffer_rows
        if not buffer:
            return
        assert writer is not None
        table = buffer[0] if len(buffer) == 1 else pa.concat_tables(buffer)
        writer.write_table(table, row_group_size=table.num_rows)
        buffer.clear()
        buffer_rows = 0

    try:
        for shard in ordered:
            parquet = pq.ParquetFile(shard.heldout_master.path)
            schema = parquet.schema_arrow
            canonical_schema = _writer_side_arrow_schema_v12(schema)
            if expected_schema is None:
                expected_schema = canonical_schema
                writer = pq.ParquetWriter(
                    path,
                    expected_schema,
                    compression="zstd",
                    compression_level=3,
                    use_dictionary=True,
                    write_statistics=True,
                )
            elif not canonical_schema.equals(
                expected_schema, check_metadata=True
            ):
                raise FoldFusedInvariantError("heldout shard schema drift")
            observed_shard: list[tuple[str, int]] = []
            for table in _iter_heldout_query_tables(shard.heldout_master.path):
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
                    raise FoldFusedInvariantError(
                        "heldout shard query semantics drift"
                    )
                current = (next(iter(query_ids)), next(iter(folds)))
                observed_shard.append(current)
                sequence.append(current)
                if buffer and buffer_rows + table.num_rows > DEFAULT_ROW_GROUP_TARGET:
                    flush()
                buffer.append(
                    pa.Table.from_arrays(table.columns, schema=expected_schema)
                )
                buffer_rows += table.num_rows
                row_count += table.num_rows
            expected_shard = tuple(
                zip(shard.spec.query_ids, shard.spec.query_folds, strict=True)
            )
            if tuple(observed_shard) != expected_shard:
                raise FoldFusedInvariantError("heldout shard query sequence drift")
        flush()
    finally:
        if writer is not None:
            writer.close()
    if expected_schema is None or not sequence:
        raise FoldFusedInvariantError("merged heldout spool is empty")
    size_bytes = path.stat().st_size
    if size_bytes > int(maximum_bytes):
        raise FoldFusedInvariantError("merged heldout spool exceeds resource cap")
    schema_sha = str((expected_schema.metadata or {}).get(b"schema_sha256", b""), "ascii")
    if not _valid_sha256(schema_sha):
        raise FoldFusedInvariantError("heldout schema digest is missing")
    parquet_sha = sha256_file(path)
    sequence_sha = _sequence_sha256(sequence)
    manifest = {
        "schema_version": HELDOUT_MANIFEST_VERSION,
        "status": "SEALED_LABEL_BLIND",
        "folds": list(TRAIN_FOLDS),
        "fold0_opened": False,
        "fold1_opened": False,
        "positive_injection": False,
        "label_columns": [],
        "projection": expected_schema.names,
        "schema": _schema_descriptor(expected_schema.remove_metadata()),
        "schema_sha256": schema_sha,
        "query_count": len(sequence),
        "row_count": row_count,
        "query_sequence_sha256": sequence_sha,
        "maximum_bytes": int(maximum_bytes),
        "artifact": {
            "path": path.name,
            "sha256": parquet_sha,
            "size_bytes": size_bytes,
        },
        "physical_provenance": {
            "engine_id": ENGINE_ID,
            "ordered_shard_manifest_sha256": [
                artifact.manifest_sha256 for artifact in ordered
            ],
            "canonical_row_group_target": DEFAULT_ROW_GROUP_TARGET,
        },
    }
    _atomic_json(manifest_path, manifest)
    artifact = HeldoutMasterArtifact(
        path=path,
        manifest_path=manifest_path,
        sha256=parquet_sha,
        manifest_sha256=sha256_file(manifest_path),
        row_count=row_count,
        query_count=len(sequence),
        size_bytes=size_bytes,
        maximum_bytes=int(maximum_bytes),
        schema_sha256=schema_sha,
        query_sequence_sha256=sequence_sha,
    )
    # The physical content address is the parity surface.  A digest of only
    # counts/schema/order could miss changed candidate payloads.
    return MergedHeldoutV12(artifact, parquet_sha)


def fit_site_models_sequential_v12(
    backend: Any,
    artifacts: Mapping[int, TrainMatrixArtifact] | MergedTrainingV12,
) -> Mapping[int, FittedSiteModel]:
    """Fit outer 2, 3 and 4 in order; never overlap their large matrices."""

    matrices = artifacts.matrices if isinstance(artifacts, MergedTrainingV12) else artifacts
    if set(matrices) != set(TRAIN_FOLDS):
        raise FoldFusedInvariantError("three merged outer matrices are required")
    models: dict[int, FittedSiteModel] = {}
    for outer in TRAIN_FOLDS:
        model = backend.fit(matrices[outer])
        if (
            model.outer_fold != outer
            or model.feature_names != matrices[outer].feature_names
            or not _valid_sha256(model.sha256)
            or not _valid_sha256(model.recipe_sha256)
        ):
            raise FoldFusedInvariantError("sequential site model identity drift")
        models[outer] = model
    if len({model.recipe_sha256 for model in models.values()}) != 1:
        raise FoldFusedInvariantError("site models do not share one frozen recipe")
    return models


def _iter_heldout_contexts_v12(
    artifact: HeldoutMasterArtifact,
    *,
    payload_factory: Callable[[tuple[Mapping[str, Any], ...]], Any] | None = None,
) -> Iterable[MasterContext]:
    if payload_factory is None:
        from .retrieval_fold_fused_runtime_v12 import _MasterPayload

        payload_factory = _MasterPayload
    if (
        not artifact.path.is_file()
        or artifact.sha256 != sha256_file(artifact.path)
        or artifact.size_bytes != artifact.path.stat().st_size
    ):
        raise FoldFusedInvariantError("heldout replay artifact hash/size drift")
    parquet = pq.ParquetFile(artifact.path)
    sequence: list[tuple[str, int]] = []
    rows_seen = 0
    for table in _iter_heldout_query_tables(artifact.path):
        rows = tuple(table.to_pylist())
        query_ids = {str(row["query_id"]) for row in rows}
        folds = {int(row["fold"]) for row in rows}
        if (
            len(query_ids) != 1
            or len(folds) != 1
            or not folds.issubset(_ALLOWED_FOLDS)
            or tuple(int(row["master_rank"]) for row in rows)
            != tuple(range(1, len(rows) + 1))
        ):
            raise FoldFusedInvariantError("heldout replay query semantics drift")
        query_id = next(iter(query_ids))
        fold = next(iter(folds))
        sirets = tuple(str(row["siret"]) for row in rows)
        parents = tuple(str(row["siren"]) for row in rows)
        sequence.append((query_id, fold))
        rows_seen += len(rows)
        yield MasterContext(
            query_id=query_id,
            scene_fold=fold,
            outer_fold=fold,
            candidate_sirets=sirets,
            candidate_parents=parents,
            opaque=payload_factory(rows),
            metadata={
                "heldout_spool_schema_sha256": artifact.schema_sha256,
                "heldout_spool_query_sequence_sha256": artifact.query_sequence_sha256,
            },
        )
    if (
        rows_seen != artifact.row_count
        or len(sequence) != artifact.query_count
        or _sequence_sha256(sequence) != artifact.query_sequence_sha256
        or parquet.metadata.num_rows != artifact.row_count
    ):
        raise FoldFusedInvariantError("heldout replay cardinality/order drift")


def _rank_of_v12(indices: Sequence[int], truth_index: int | None) -> int | None:
    if truth_index is None:
        return None
    try:
        return list(indices).index(truth_index) + 1
    except ValueError:
        return None


def _query_outcome_v12(
    context: MasterContext,
    selection: RankingSelection,
    label: HumanLabel,
) -> QueryOutcome:
    try:
        truth_index = context.candidate_sirets.index(label.human_siret)
    except ValueError:
        truth_index = None
    master_rank = truth_index + 1 if truth_index is not None else None
    prerank_rank = _rank_of_v12(selection.prerank_indices, truth_index)
    product_rank = _rank_of_v12(selection.product_indices, truth_index)
    selected_sirets = {
        context.candidate_sirets[index] for index in selection.product_indices
    }
    operational = set(label.operational_sirets) | {label.human_siret}
    truth_parent_present = label.human_siret[:9] in set(context.candidate_parents)
    catalog = label.audit_metadata.get("official_catalog_present")
    if truth_index is None and catalog is False:
        loss = "OFFICIAL_CATALOG_ABSENT"
    elif truth_index is None and catalog is None:
        loss = "OFFICIAL_CATALOG_STATUS_UNKNOWN"
    elif truth_index is None and not truth_parent_present:
        loss = "PARENT_NOT_RETRIEVED"
    elif truth_index is None:
        loss = "SITE_NOT_EXPANDED"
    elif prerank_rank is None:
        loss = "EVICTED_MASTER_TO_500"
    elif product_rank is not None:
        loss = "HIT"
    else:
        loss = "EVICTED_500_TO_100"
    return QueryOutcome(
        query_id=context.query_id,
        fold=context.scene_fold,
        human_siret=label.human_siret,
        master_candidate_count=len(context.candidate_sirets),
        prerank_candidate_count=len(selection.prerank_indices),
        product_candidate_count=len(selection.product_indices),
        truth_master_rank=master_rank,
        truth_prerank_rank=prerank_rank,
        truth_product_rank=product_rank,
        master_hit_at_2000=truth_index is not None,
        prerank_hit_at_500=prerank_rank is not None,
        exact_hit_at_100=product_rank is not None,
        operational_hit_at_100=bool(selected_sirets.intersection(operational)),
        loss_reason=loss,
        audit_metadata=dict(label.audit_metadata),
    )


def _validate_selection_v12(
    context: MasterContext,
    selection: RankingSelection,
    caps: ExecutionCaps,
) -> None:
    prerank = selection.prerank_indices
    product = selection.product_indices
    if (
        len(prerank) > caps.prerank
        or len(product) > caps.product
        or len(prerank) != len(set(prerank))
        or len(product) != len(set(product))
        or any(
            index < 0 or index >= len(context.candidate_sirets)
            for index in (*prerank, *product)
        )
        or not set(product).issubset(prerank)
        or len(selection.product_reasons) != len(product)
    ):
        raise FoldFusedInvariantError("sharded selector escaped frozen caps/Master")


def replay_shard_v12(
    spec: ShardSpecV12,
    *,
    heldout: HeldoutMasterArtifact,
    labels: Any,
    contexts: Any,
    model_backend: Any,
    models: Mapping[int, FittedSiteModel],
    selector: Any,
    output_dir: Path,
    lineage: Mapping[str, str],
    build_manifest_sha256: str,
    caps: ExecutionCaps = ExecutionCaps(),
    payload_factory: Callable[[tuple[Mapping[str, Any], ...]], Any] | None = None,
    output_store_factory: Callable[[Path], Any] = ParquetOOFOutputStore,
) -> ShardReplayArtifactV12:
    """Replay one label-blind heldout shard with its exact outer models."""

    lineage_value = _validate_lineage(lineage)
    if not _valid_sha256(build_manifest_sha256) or set(models) != set(TRAIN_FOLDS):
        raise FoldFusedInvariantError("replay lacks sealed build/three site models")
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    # V12 keeps its historical Parquet sink by default.  A physical-only
    # executor may inject a schema-compatible sink (for example a compact,
    # buffered PreRank500 writer) without duplicating any generation,
    # scoring, selection or label-side evaluation logic.
    store = output_store_factory(output / "outputs")
    if any(
        not callable(getattr(store, method, None))
        for method in ("append_rankings", "append_outcome", "seal")
    ):
        raise FoldFusedInvariantError("replay output store contract drifted")
    outcomes: list[QueryOutcome] = []
    observed: list[tuple[str, int]] = []
    feature_names: tuple[str, ...] | None = None
    for context in _iter_heldout_contexts_v12(
        heldout, payload_factory=payload_factory
    ):
        if context.outer_fold != context.scene_fold:
            raise FoldFusedInvariantError("replay context is not held out")
        observed.append((context.query_id, context.scene_fold))
        statistics = contexts.compute_full_master_statistics(context)
        _validate_statistics_v12(context, statistics)
        features = contexts.materialize(
            context,
            statistics,
            candidate_indices=None,
        )
        indices = tuple(range(len(context.candidate_sirets)))
        _validate_feature_batch_v12(context, statistics, features, indices)
        if feature_names is None:
            feature_names = features.feature_names
        elif features.feature_names != feature_names:
            raise FoldFusedInvariantError("replay feature schema drift")
        model = models[context.scene_fold]
        scores = np.asarray(
            model_backend.score(model, features), dtype=np.float64
        )
        if scores.shape != (len(indices),) or not np.isfinite(scores).all():
            raise FoldFusedInvariantError("replay site scores are invalid")
        selection = selector.select(
            context,
            features,
            scores,
            prerank_cap=caps.prerank,
            product_cap=caps.product,
        )
        _validate_selection_v12(context, selection, caps)
        store.append_rankings(context, features, scores, selection)
        # Label-side evaluation remains after generation, feature computation,
        # model scoring and selection.
        label: HumanLabel = labels.get(context.query_id)
        if label.fold != context.scene_fold:
            raise FoldFusedInvariantError("replay label fold mismatch")
        outcome = _query_outcome_v12(context, selection, label)
        outcomes.append(outcome)
        store.append_outcome(outcome)
    expected = tuple(zip(spec.query_ids, spec.query_folds, strict=True))
    if tuple(observed) != expected:
        raise FoldFusedInvariantError("replay shard sequence differs from build")
    outputs = store.seal()
    digests = {
        "prerank500": outputs["top500"].sha256,
        "top100": outputs["top100"].sha256,
        "outcomes": outputs["outcomes"].sha256,
        "heldout_spool": heldout.sha256,
        "model_by_outer": {
            str(outer): models[outer].sha256 for outer in TRAIN_FOLDS
        },
    }
    replay_identity = {
        "engine_id": ENGINE_ID,
        "spec_sha256": spec.digest,
        "build_manifest_sha256": build_manifest_sha256,
        "lineage": lineage_value,
        "query_count": len(outcomes),
        "digests": digests,
    }
    manifest_path = output / "manifest.json"
    manifest = {
        "schema_version": SHARD_REPLAY_SCHEMA_VERSION,
        "status": "SEALED_SHARD_REPLAY",
        "replay_id": _sha256_json(replay_identity),
        "replay_identity": replay_identity,
        "engine_id": ENGINE_ID,
        "spec": spec.as_manifest(),
        "lineage": lineage_value,
        "build_manifest_sha256": build_manifest_sha256,
        "folds": list(spec.folds),
        "fold0_opened": False,
        "fold1_opened": False,
        "positive_injection": False,
        "query_count": len(outcomes),
        "query_sequence_sha256": _sequence_sha256(observed),
        "heldout_master": heldout_master_record_v12(heldout),
        "model_by_outer": fitted_site_model_records_v12(models),
        "outputs": {
            name: {
                "path": _relative_or_absolute(record.path, output),
                "sha256": record.sha256,
                "rows": record.rows,
            }
            for name, record in outputs.items()
        },
        "digests": digests,
    }
    _atomic_json(manifest_path, manifest)
    return ShardReplayArtifactV12(
        spec=spec,
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        build_manifest_sha256=build_manifest_sha256,
        lineage=lineage_value,
        outputs=outputs,
        outcomes=tuple(outcomes),
        digests=digests,
    )


def _outcome_from_row_v12(row: Mapping[str, Any]) -> QueryOutcome:
    metadata = row.get("audit_metadata_json")
    decoded = json.loads(str(metadata or "{}"))
    return QueryOutcome(
        query_id=str(row["query_id"]),
        fold=int(row["fold"]),
        human_siret=str(row["human_siret"]),
        master_candidate_count=int(row["master_candidate_count"]),
        prerank_candidate_count=int(row["prerank_candidate_count"]),
        product_candidate_count=int(row["product_candidate_count"]),
        truth_master_rank=(
            int(row["truth_master_rank"])
            if row.get("truth_master_rank") is not None
            else None
        ),
        truth_prerank_rank=(
            int(row["truth_prerank_rank"])
            if row.get("truth_prerank_rank") is not None
            else None
        ),
        truth_product_rank=(
            int(row["truth_product_rank"])
            if row.get("truth_product_rank") is not None
            else None
        ),
        master_hit_at_2000=bool(row["master_hit_at_2000"]),
        prerank_hit_at_500=bool(row["prerank_hit_at_500"]),
        exact_hit_at_100=bool(row["exact_hit_at_100"]),
        operational_hit_at_100=bool(row["operational_hit_at_100"]),
        loss_reason=str(row["loss_reason"]),
        audit_metadata=decoded,
    )


def load_shard_replay_manifest_v12(
    path: Path,
    *,
    verify_hashes: bool = True,
) -> ShardReplayArtifactV12:
    manifest_path = Path(path).resolve()
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        raw.get("schema_version") != SHARD_REPLAY_SCHEMA_VERSION
        or raw.get("status") != "SEALED_SHARD_REPLAY"
        or raw.get("engine_id") != ENGINE_ID
        or raw.get("fold0_opened") is not False
        or raw.get("fold1_opened") is not False
        or raw.get("positive_injection") is not False
    ):
        raise FoldFusedInvariantError("invalid or unsafe shard replay manifest")
    replay_identity = raw.get("replay_identity")
    if (
        not isinstance(replay_identity, Mapping)
        or raw.get("replay_id") != _sha256_json(replay_identity)
    ):
        raise FoldFusedInvariantError("shard replay content address drift")
    spec = ShardSpecV12.from_manifest(raw.get("spec") or {})
    if (
        raw.get("query_count") != spec.query_count
        or raw.get("query_sequence_sha256") != spec.query_sequence_sha256
        or raw.get("folds") != list(spec.folds)
    ):
        raise FoldFusedInvariantError("shard replay population drift")
    lineage = _validate_lineage(raw.get("lineage") or {})
    build_sha = str(raw.get("build_manifest_sha256") or "")
    if not _valid_sha256(build_sha):
        raise FoldFusedInvariantError("shard replay build lineage is missing")
    outputs_raw = raw.get("outputs")
    if not isinstance(outputs_raw, Mapping) or set(outputs_raw) != {
        "top500",
        "top100",
        "outcomes",
    }:
        raise FoldFusedInvariantError("shard replay outputs are incomplete")
    root = manifest_path.parent
    heldout_raw = raw.get("heldout_master")
    model_records = raw.get("model_by_outer")
    if not isinstance(heldout_raw, Mapping) or not isinstance(model_records, Mapping):
        raise FoldFusedInvariantError("shard replay inputs are not sealed")
    heldout = load_heldout_master_record_v12(
        heldout_raw,
        root=root,
        verify_hashes=verify_hashes,
    )
    models = load_fitted_site_model_records_v12(
        model_records,
        handle_loader=lambda _path, _outer: None,
        root=root,
        verify_hashes=verify_hashes,
    )
    if (
        heldout.query_count != spec.query_count
        or heldout.query_sequence_sha256 != spec.query_sequence_sha256
    ):
        raise FoldFusedInvariantError("shard replay heldout population drift")
    outputs: dict[str, ArtifactRecord] = {}
    for name, record in outputs_raw.items():
        artifact_path = _resolve_record_path(root, record.get("path"))
        digest = str(record.get("sha256") or "")
        if (
            not artifact_path.is_file()
            or not _valid_sha256(digest)
            or (verify_hashes and sha256_file(artifact_path) != digest)
        ):
            raise FoldFusedInvariantError("shard replay output hash mismatch")
        outputs[name] = ArtifactRecord(
            artifact_path, digest, int(record.get("rows", -1))
        )
    outcomes = tuple(
        _outcome_from_row_v12(row)
        for row in pq.read_table(outputs["outcomes"].path).to_pylist()
    )
    if (
        tuple((value.query_id, value.fold) for value in outcomes)
        != tuple(zip(spec.query_ids, spec.query_folds, strict=True))
    ):
        raise FoldFusedInvariantError("shard replay outcome sequence drift")
    digests = raw.get("digests")
    expected_digests = {
        "prerank500": outputs["top500"].sha256,
        "top100": outputs["top100"].sha256,
        "outcomes": outputs["outcomes"].sha256,
        "heldout_spool": heldout.sha256,
        "model_by_outer": {
            str(outer): models[outer].sha256 for outer in TRAIN_FOLDS
        },
    }
    if not isinstance(digests, Mapping) or dict(digests) != expected_digests:
        raise FoldFusedInvariantError("shard replay digest set is missing")
    expected_identity = {
        "engine_id": ENGINE_ID,
        "spec_sha256": spec.digest,
        "build_manifest_sha256": build_sha,
        "lineage": lineage,
        "query_count": spec.query_count,
        "digests": expected_digests,
    }
    if dict(replay_identity) != expected_identity:
        raise FoldFusedInvariantError("shard replay identity/input drift")
    return ShardReplayArtifactV12(
        spec=spec,
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        build_manifest_sha256=build_sha,
        lineage=lineage,
        outputs=outputs,
        outcomes=outcomes,
        digests=dict(digests),
    )


def _ordered_replay_artifacts_v12(
    artifacts: Sequence[ShardReplayArtifactV12],
) -> tuple[ShardReplayArtifactV12, ...]:
    ordered = tuple(sorted(artifacts, key=lambda value: value.spec.shard_ordinal))
    if not ordered:
        raise FoldFusedInvariantError("no shard replay artifacts were supplied")
    lineage = dict(ordered[0].lineage)
    sequence_stop = 0
    seen: set[str] = set()
    for ordinal, artifact in enumerate(ordered):
        if (
            artifact.spec.shard_ordinal != ordinal
            or artifact.spec.sequence_start != sequence_stop
            or dict(artifact.lineage) != lineage
            or artifact.manifest_sha256 != sha256_file(artifact.manifest_path)
            or seen.intersection(artifact.spec.query_ids)
        ):
            raise FoldFusedInvariantError("replay shards overlap/gap or mix lineage")
        seen.update(artifact.spec.query_ids)
        sequence_stop = artifact.spec.sequence_stop
    return ordered


def merge_replay_shards_v12(
    artifacts: Sequence[ShardReplayArtifactV12],
    output_dir: Path,
) -> MergedReplayV12:
    """Merge output row groups in immutable global query order."""

    ordered = _ordered_replay_artifacts_v12(artifacts)
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    expected_sequence = tuple(
        (query_id, fold)
        for artifact in ordered
        for query_id, fold in zip(
            artifact.spec.query_ids,
            artifact.spec.query_folds,
            strict=True,
        )
    )
    merged: dict[str, ArtifactRecord] = {}
    for name in ("top500", "top100", "outcomes"):
        path = output / {
            "top500": "oof_prerank500.parquet",
            "top100": "oof_top100.parquet",
            "outcomes": "oof_query_outcomes.parquet",
        }[name]
        writer: pq.ParquetWriter | None = None
        expected_schema: pa.Schema | None = None
        rows_written = 0
        observed_queries: list[tuple[str, int]] = []
        try:
            for artifact in ordered:
                parquet = pq.ParquetFile(artifact.outputs[name].path)
                schema = parquet.schema_arrow
                if expected_schema is None:
                    expected_schema = schema
                    writer = pq.ParquetWriter(path, schema, compression="zstd")
                elif not schema.equals(expected_schema, check_metadata=True):
                    raise FoldFusedInvariantError("replay output shard schema drift")
                for row_group in range(parquet.num_row_groups):
                    table = parquet.read_row_group(row_group)
                    query_ids = {str(value) for value in table["query_id"].to_pylist()}
                    folds = {int(value.as_py()) for value in table["fold"]}
                    if (
                        len(query_ids) != 1
                        or len(folds) != 1
                        or not folds.issubset(_ALLOWED_FOLDS)
                    ):
                        raise FoldFusedInvariantError("replay output group identity drift")
                    query_id = next(iter(query_ids))
                    fold = next(iter(folds))
                    if name in {"top500", "top100"}:
                        ranks = tuple(int(value.as_py()) for value in table["rank"])
                        cap = 500 if name == "top500" else 100
                        if (
                            not 1 <= table.num_rows <= cap
                            or ranks != tuple(range(1, table.num_rows + 1))
                        ):
                            raise FoldFusedInvariantError("replay ranking cap/order drift")
                    elif table.num_rows != 1:
                        raise FoldFusedInvariantError("replay emitted duplicate outcome")
                    observed_queries.append((query_id, fold))
                    assert writer is not None
                    writer.write_table(table, row_group_size=table.num_rows)
                    rows_written += table.num_rows
        finally:
            if writer is not None:
                writer.close()
        if expected_schema is None or tuple(observed_queries) != expected_sequence:
            raise FoldFusedInvariantError(
                f"merged replay {name} query sequence drift"
            )
        merged[name] = ArtifactRecord(path, sha256_file(path), rows_written)
    outcomes = tuple(
        _outcome_from_row_v12(row)
        for row in pq.read_table(merged["outcomes"].path).to_pylist()
    )
    if tuple((value.query_id, value.fold) for value in outcomes) != expected_sequence:
        raise FoldFusedInvariantError("merged replay outcome semantics drift")
    digests = {
        "prerank500": merged["top500"].sha256,
        "top100": merged["top100"].sha256,
        "outcomes": merged["outcomes"].sha256,
    }
    return MergedReplayV12(
        outputs=merged,
        outcomes=outcomes,
        query_sequence_sha256=_sequence_sha256(expected_sequence),
        digests=digests,
    )


def _metric_v12(successes: int, total: int) -> Mapping[str, Any]:
    if total <= 0:
        return {"successes": 0, "total": 0, "rate": None, "wilson_95": None}
    p = successes / total
    z = NormalDist().inv_cdf(0.975)
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    spread = z * math.sqrt(
        p * (1.0 - p) / total + z * z / (4.0 * total * total)
    ) / denominator
    return {
        "successes": int(successes),
        "total": int(total),
        "rate": p,
        "wilson_95": [max(0.0, center - spread), min(1.0, center + spread)],
    }


def aggregate_sharded_gates_v12(
    outcomes: Sequence[QueryOutcome],
    gates: GateThresholds = GateThresholds(),
) -> Mapping[str, Any]:
    """Reproduce the monolithic global metrics, without issuing a final GO."""

    values = tuple(outcomes)
    total = len(values)
    if total <= 0 or any(value.fold not in _ALLOWED_FOLDS for value in values):
        raise FoldFusedInvariantError("global gates require non-empty folds 2/3/4")
    master = _metric_v12(sum(value.master_hit_at_2000 for value in values), total)
    prerank = _metric_v12(sum(value.prerank_hit_at_500 for value in values), total)
    top100 = _metric_v12(sum(value.exact_hit_at_100 for value in values), total)
    operational = _metric_v12(
        sum(value.operational_hit_at_100 for value in values), total
    )
    per_fold = {
        str(fold): _metric_v12(
            sum(value.exact_hit_at_100 for value in values if value.fold == fold),
            sum(value.fold == fold for value in values),
        )
        for fold in TRAIN_FOLDS
    }
    recall_pass = bool(
        prerank["rate"] is not None
        and prerank["rate"] >= gates.prerank_recall_at_500
        and top100["rate"] is not None
        and top100["rate"] >= gates.top100_recall
        and top100["wilson_95"][0] >= gates.top100_wilson95_lower
        and all(
            value["rate"] is not None
            and value["rate"] >= gates.minimum_each_outer_fold_top100
            for value in per_fold.values()
        )
    )

    # These views are label-side audits only.  Every outcome is assigned to
    # exactly one value in every dimension; missing metadata therefore remains
    # visible as UNKNOWN instead of silently shrinking a segment denominator.
    segment_specs: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("active_closed", ("active_closed", "ground_truth_state", "gt_state")),
        ("crm_origin", ("crm_origin", "data_origin", "source_kind")),
        (
            "seen_unseen_siren",
            ("seen_unseen_siren", "siren_novelty", "unseen_siren"),
        ),
        (
            "known_new_site",
            ("known_new_site", "site_novelty", "new_site_known_siren"),
        ),
        ("megapolis", ("megapolis", "is_megapolis", "mega_base_pool")),
        ("qualification", ("qualification", "qualification_status")),
        ("label_kind", ("label_kind",)),
        ("qualification_v2", ("qualification_v2", "v2_label_kind")),
        ("qualification_v3", ("qualification_v3", "v3_label_kind")),
    )

    def normalized_segment_value(
        metadata: Mapping[str, Any],
        dimension: str,
        aliases: Sequence[str],
    ) -> str:
        value: Any = None
        source_key = ""
        for key in aliases:
            candidate = metadata.get(key)
            if candidate is None:
                continue
            if isinstance(candidate, float) and math.isnan(candidate):
                continue
            text = str(candidate).strip()
            if text.upper() in {"", "<NA>", "NA", "N/A", "NAN", "NONE", "NULL", "UNKNOWN"}:
                continue
            value = candidate
            source_key = key
            break
        if value is None:
            return "UNKNOWN"
        if isinstance(value, bool):
            if dimension == "seen_unseen_siren" and source_key == "unseen_siren":
                return "UNSEEN" if value else "SEEN"
            if dimension == "known_new_site" and source_key == "new_site_known_siren":
                return "NEW_SITE" if value else "KNOWN_SITE"
            if dimension == "megapolis":
                return "MEGAPOLIS" if value else "NON_MEGAPOLIS"
            return "TRUE" if value else "FALSE"
        return str(value).strip()

    def segment_metrics(part: Sequence[QueryOutcome]) -> Mapping[str, Any]:
        part_values = tuple(part)
        part_total = len(part_values)
        loss_counts: dict[str, int] = {}
        for value in part_values:
            reason = str(value.loss_reason).strip() or "UNKNOWN"
            loss_counts[reason] = loss_counts.get(reason, 0) + 1
        return {
            "query_count": part_total,
            "master_recall_at_2000": _metric_v12(
                sum(value.master_hit_at_2000 for value in part_values),
                part_total,
            ),
            "prerank_recall_at_500": _metric_v12(
                sum(value.prerank_hit_at_500 for value in part_values),
                part_total,
            ),
            "exact_recall_at_100": _metric_v12(
                sum(value.exact_hit_at_100 for value in part_values),
                part_total,
            ),
            "operational_recall_at_100": _metric_v12(
                sum(value.operational_hit_at_100 for value in part_values),
                part_total,
            ),
            "loss_counts": {
                reason: loss_counts[reason] for reason in sorted(loss_counts)
            },
        }

    segments: dict[str, Mapping[str, Any]] = {}
    for dimension, aliases in segment_specs:
        grouped: dict[str, list[QueryOutcome]] = {}
        for value in values:
            segment_value = normalized_segment_value(
                value.audit_metadata,
                dimension,
                aliases,
            )
            grouped.setdefault(segment_value, []).append(value)
        # UNKNOWN is present even when its count is zero, which makes metadata
        # completeness auditable without changing any scientific denominator.
        grouped.setdefault("UNKNOWN", [])
        segments[dimension] = {
            segment_value: segment_metrics(grouped[segment_value])
            for segment_value in sorted(grouped)
        }

    return {
        "denominator": "ALL_USABLE_HUMAN_SIRET_LABELS_FOLDS_2_3_4",
        "master_recall_at_2000": master,
        "master_headroom_target_met": bool(
            master["rate"] is not None
            and master["rate"] >= gates.master_recall_at_2000
        ),
        "master_decision_blocking": False,
        "prerank_recall_at_500": prerank,
        "exact_recall_at_100": top100,
        "operational_recall_at_100": operational,
        "exact_recall_at_100_by_outer_fold": per_fold,
        "segments": segments,
        "thresholds": asdict(gates),
        "caps_observed": {
            "maximum_master": max(value.master_candidate_count for value in values),
            "maximum_prerank": max(value.prerank_candidate_count for value in values),
            "maximum_product": max(value.product_candidate_count for value in values),
        },
        "recall_gates_pass": recall_pass,
        "postfit_parity_gate": "PENDING",
        "latency_gate": "NOT_EVALUATED",
        "gate_eligible": False,
        "decision": "NO_READY_DECISION",
    }


def _parse_entrypoint(value: str) -> tuple[str, str]:
    module, separator, function = str(value).partition(":")
    if not separator or not module or not function or function.startswith("_"):
        raise FoldFusedInvariantError(
            "worker entrypoint must be a public 'module:function'"
        )
    return module, function


def _invoke_worker_entrypoint_v12(
    entrypoint: str,
    job: WorkerJobV12,
) -> str:
    module_name, function_name = _parse_entrypoint(entrypoint)
    function = getattr(import_module(module_name), function_name, None)
    if not callable(function):
        raise FoldFusedInvariantError("worker entrypoint is not callable")
    result = function(dict(job.payload))
    if isinstance(result, Mapping):
        result = result.get("manifest_path")
    path = Path(str(result or "")).resolve()
    if not path.is_file() or path.name != "manifest.json":
        raise FoldFusedInvariantError(
            f"worker {job.job_id} did not publish a manifest"
        )
    return str(path)


def spawn_manifest_workers_v12(
    entrypoint: str,
    jobs: Sequence[WorkerJobV12],
    max_workers: int,
) -> tuple[Path, ...]:
    """Spawn isolated manifest workers with no live heavy Python objects.

    Direct subprocesses are used instead of ``multiprocessing`` queues and
    semaphores.  Besides working in restricted macOS environments, this keeps
    the process boundary auditably JSON-only: request file in, sealed manifest
    path out.
    """

    _parse_entrypoint(entrypoint)
    values = tuple(jobs)
    if (
        not values
        or len({job.job_id for job in values}) != len(values)
        or not 1 <= int(max_workers) <= 4
    ):
        raise FoldFusedInvariantError("invalid sharded worker job set")
    worker_limit = min(int(max_workers), len(values))
    results: list[Path] = []
    module_name = __spec__.name if __spec__ is not None else __name__
    with tempfile.TemporaryDirectory(prefix="sireto-v12-workers-") as temporary:
        exchange = Path(temporary)
        for batch_start in range(0, len(values), worker_limit):
            launched: list[
                tuple[subprocess.Popen[bytes], WorkerJobV12, Path, Path, Any]
            ] = []
            for offset, job in enumerate(
                values[batch_start : batch_start + worker_limit]
            ):
                ordinal = batch_start + offset
                request_path = exchange / f"request-{ordinal:04d}.json"
                response_path = exchange / f"response-{ordinal:04d}.json"
                log_path = exchange / f"worker-{ordinal:04d}.log"
                _atomic_json(
                    request_path,
                    {
                        "entrypoint": entrypoint,
                        "job_id": job.job_id,
                        "payload": dict(job.payload),
                    },
                )
                log_handle = log_path.open("wb")
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        module_name,
                        "--worker-job",
                        str(request_path),
                        str(response_path),
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    close_fds=True,
                )
                launched.append(
                    (process, job, response_path, log_path, log_handle)
                )
            failures: list[str] = []
            batch_results: list[Path] = []
            try:
                for process, job, response_path, log_path, log_handle in launched:
                    return_code = process.wait()
                    log_handle.close()
                    if return_code != 0 or not response_path.is_file():
                        tail = (
                            log_path.read_text(
                                encoding="utf-8", errors="replace"
                            )[-4_000:]
                            if log_path.is_file()
                            else ""
                        )
                        failures.append(
                            f"{job.job_id}: exit={return_code}; log_tail={tail!r}"
                        )
                        continue
                    response = json.loads(
                        response_path.read_text(encoding="utf-8")
                    )
                    if response.get("job_id") != job.job_id:
                        failures.append(f"{job.job_id}: response identity drift")
                        continue
                    manifest_path = Path(
                        str(response.get("manifest_path") or "")
                    ).resolve()
                    if (
                        not manifest_path.is_file()
                        or manifest_path.name != "manifest.json"
                    ):
                        failures.append(f"{job.job_id}: invalid manifest response")
                        continue
                    batch_results.append(manifest_path)
            except BaseException:
                for process, _job, _response, _log, log_handle in launched:
                    if process.poll() is None:
                        process.terminate()
                    if not log_handle.closed:
                        log_handle.close()
                for process, _job, _response, _log, _handle in launched:
                    try:
                        process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                raise
            if failures:
                raise FoldFusedInvariantError(
                    "sharded worker subprocess failed: " + " | ".join(failures)
                )
            results.extend(batch_results)
    if len(results) != len(values):
        raise AssertionError("sharded subprocess launcher lost a worker response")
    return tuple(results)


def _worker_subprocess_main_v12(request_path: Path, response_path: Path) -> None:
    raw = json.loads(Path(request_path).read_text(encoding="utf-8"))
    entrypoint = str(raw.get("entrypoint") or "")
    job = WorkerJobV12(
        job_id=str(raw.get("job_id") or ""),
        payload=dict(raw.get("payload") or {}),
    )
    manifest_path = _invoke_worker_entrypoint_v12(entrypoint, job)
    _atomic_json(
        Path(response_path),
        {"job_id": job.job_id, "manifest_path": manifest_path},
    )


def _model_records_v12(
    models: Mapping[int, FittedSiteModel],
) -> dict[str, dict[str, Any]]:
    return fitted_site_model_records_v12(models)


def _artifact_bytes_v12(
    build: Sequence[ShardBuildArtifactV12],
    train: MergedTrainingV12,
    heldout: MergedHeldoutV12,
    prefit_bridge: PrefitBridgeArtifactV12 | None,
    replay_shards: Sequence[ShardReplayArtifactV12],
    replay: MergedReplayV12,
) -> Mapping[str, int]:
    build_bytes = sum(
        artifact.heldout_master.size_bytes
        + sum(matrix.path.stat().st_size for matrix in artifact.train_matrices.values())
        + (
            artifact.prefit_bridge.size_bytes
            + artifact.prefit_bridge.manifest_path.stat().st_size
            if artifact.prefit_bridge is not None
            else 0
        )
        for artifact in build
    )
    merge_bytes = (
        heldout.artifact.size_bytes
        + sum(matrix.path.stat().st_size for matrix in train.matrices.values())
        + (
            prefit_bridge.size_bytes + prefit_bridge.manifest_path.stat().st_size
            if prefit_bridge is not None
            else 0
        )
    )
    replay_shard_bytes = sum(
        record.path.stat().st_size
        for artifact in replay_shards
        for record in artifact.outputs.values()
    )
    replay_merge_bytes = sum(
        record.path.stat().st_size for record in replay.outputs.values()
    )
    return {
        "worker_build_artifacts": int(build_bytes),
        "canonical_merged_build_artifacts": int(merge_bytes),
        "worker_replay_artifacts": int(replay_shard_bytes),
        "canonical_merged_replay_artifacts": int(replay_merge_bytes),
        "total_materialized": int(
            build_bytes + merge_bytes + replay_shard_bytes + replay_merge_bytes
        ),
    }


def run_sharded_physical_sample(
    inputs: ShardedPhysicalInputsV12,
    specs: Sequence[ShardSpecV12],
    output_dir: Path,
    miniature_fit: bool = True,
) -> ShardedRunArtifactV12:
    """Run the complete exact sharded DAG through importable worker hooks.

    The same function is used by the bounded W1/W2/W3 benchmark and by a full
    national execution.  ``miniature_fit=True`` is sealed in the manifest and
    can never be mistaken for an official OOF result.
    """

    lineage = _validate_lineage(inputs.lineage)
    ordered_specs = tuple(sorted(specs, key=lambda value: value.shard_ordinal))
    if not ordered_specs:
        raise FoldFusedInvariantError("sharded run requires at least one shard")
    # Shard topology and process concurrency are deliberately independent.
    # The frozen fallback W2 still executes the same three fold shards as W3;
    # it merely schedules at most two of them simultaneously.
    concurrent_workers = min(inputs.maximum_workers, len(ordered_specs))
    expected_specs = plan_query_shards(
        inputs.scene_index,
        inputs.query_ids,
        len(ordered_specs),
    )
    if tuple(spec.as_manifest() for spec in ordered_specs) != tuple(
        spec.as_manifest() for spec in expected_specs
    ):
        raise FoldFusedInvariantError("caller shard plan differs from deterministic plan")
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    build_jobs = tuple(
        WorkerJobV12(
            job_id=f"build-{spec.shard_id}",
            payload={
                "phase": "BUILD",
                "spec": spec.as_manifest(),
                "common": dict(inputs.build_payload),
                "lineage": lineage,
                "caps": asdict(inputs.caps),
                "maximum_spool_bytes": inputs.maximum_spool_bytes,
                "output_dir": str(output / "worker_build" / spec.shard_id),
                "miniature_fit": bool(miniature_fit),
                "execution_evidence": dict(inputs.execution_evidence),
            },
        )
        for spec in ordered_specs
    )
    build_manifests = spawn_manifest_workers_v12(
        inputs.build_entrypoint,
        build_jobs,
        max_workers=concurrent_workers,
    )
    build_artifacts = tuple(
        load_shard_build_manifest_v12(path) for path in build_manifests
    )
    if tuple(value.spec.digest for value in build_artifacts) != tuple(
        value.digest for value in ordered_specs
    ):
        raise FoldFusedInvariantError("worker build returned another shard plan")
    train = merge_training_shards_v12(build_artifacts, output / "train_matrices")
    heldout = merge_heldout_shards_v12(
        build_artifacts,
        output / "heldout_master_spool",
        inputs.maximum_spool_bytes,
    )
    bridge_presence = tuple(
        artifact.prefit_bridge is not None for artifact in build_artifacts
    )
    if any(bridge_presence) and not all(bridge_presence):
        raise FoldFusedInvariantError("PREFIT bridge is missing from one shard")
    global_query_sequence = tuple(
        (query_id, fold)
        for spec in ordered_specs
        for query_id, fold in zip(
            spec.query_ids, spec.query_folds, strict=True
        )
    )
    prefit_bridge = (
        merge_prefit_bridge_artifacts_v12(
            tuple(
                artifact.prefit_bridge
                for artifact in build_artifacts
                if artifact.prefit_bridge is not None
            ),
            expected_query_sequence=global_query_sequence,
            output_dir=output / "prefit_bridge",
        )
        if all(bridge_presence)
        else None
    )
    models = fit_site_models_sequential_v12(inputs.model_backend, train)

    replay_jobs = tuple(
        WorkerJobV12(
            job_id=f"replay-{spec.shard_id}",
            payload={
                "phase": "REPLAY",
                "spec": spec.as_manifest(),
                "common": dict(inputs.replay_payload),
                "lineage": lineage,
                "caps": asdict(inputs.caps),
                "build_manifest_path": str(build_artifacts[index].manifest_path),
                "build_manifest_sha256": build_artifacts[index].manifest_sha256,
                "heldout": heldout_master_record_v12(
                    build_artifacts[index].heldout_master
                ),
                "models": _model_records_v12(models),
                "output_dir": str(output / "worker_replay" / spec.shard_id),
                "miniature_fit": bool(miniature_fit),
                "execution_evidence": dict(inputs.execution_evidence),
            },
        )
        for index, spec in enumerate(ordered_specs)
    )
    replay_manifests = spawn_manifest_workers_v12(
        inputs.replay_entrypoint,
        replay_jobs,
        max_workers=concurrent_workers,
    )
    replay_artifacts = tuple(
        load_shard_replay_manifest_v12(path) for path in replay_manifests
    )
    if tuple(value.spec.digest for value in replay_artifacts) != tuple(
        value.digest for value in ordered_specs
    ):
        raise FoldFusedInvariantError("worker replay returned another shard plan")
    replay = merge_replay_shards_v12(replay_artifacts, output / "outputs")
    gates = aggregate_sharded_gates_v12(replay.outcomes, inputs.gates)
    digests: dict[str, Any] = {
        "train_matrix": {
            str(outer): train.matrices[outer].sha256 for outer in TRAIN_FOLDS
        },
        "group_sequence": {
            str(outer): train.group_sequence_sha256[outer]
            for outer in TRAIN_FOLDS
        },
        "heldout_spool": heldout.artifact.sha256,
        "model_by_outer": {
            str(outer): models[outer].sha256 for outer in TRAIN_FOLDS
        },
        "prerank500": replay.digests["prerank500"],
        "top100": replay.digests["top100"],
        "outcomes": replay.digests["outcomes"],
    }
    if prefit_bridge is not None:
        digests["prefit_bridge"] = prefit_bridge.sha256
    io_bytes = dict(
        _artifact_bytes_v12(
            build_artifacts,
            train,
            heldout,
            prefit_bridge,
            replay_artifacts,
            replay,
        )
    )
    # Include worker/run manifests and fitted model bytes in the actual output
    # tree, not just the seven large Parquet product families.
    io_bytes["total_materialized"] = sum(
        path.stat().st_size for path in output.rglob("*") if path.is_file()
    )
    final_manifest_reserve_bytes = 1024**2
    if (
        io_bytes["total_materialized"] + final_manifest_reserve_bytes
        > inputs.maximum_spool_bytes
    ):
        raise FoldFusedInvariantError(
            "sharded run materialized artifacts exceed the frozen temporary budget"
        )
    run_kind = (
        "PHYSICAL_BENCHMARK_MINIATURE"
        if miniature_fit
        else "NESTED_OUTER_OOF_FOLD_FUSED_SHARDED"
    )
    run_identity = {
        "engine_id": ENGINE_ID,
        "run_kind": run_kind,
        "query_count": len(inputs.query_ids),
        "query_sequence_sha256": replay.query_sequence_sha256,
        "lineage": lineage,
        "spec_sha256": [spec.digest for spec in ordered_specs],
        "shard_count": len(ordered_specs),
        "maximum_concurrent_workers": concurrent_workers,
        "execution_evidence": dict(inputs.execution_evidence),
        "digests": digests,
        "miniature_fit": bool(miniature_fit),
    }
    manifest_path = output / "manifest.json"
    manifest = {
        "schema_version": SHARDED_RUN_SCHEMA_VERSION,
        "engine_id": ENGINE_ID,
        "run_kind": run_kind,
        "status": "EXECUTED_NO_READY_CLAIM",
        "run_id": _sha256_json(run_identity),
        "run_identity": run_identity,
        "query_count": len(inputs.query_ids),
        "folds": list(TRAIN_FOLDS),
        "fold0_opened": False,
        "fold1_opened": False,
        "positive_injection": False,
        "logical_recipe_changes": False,
        "miniature_fit": bool(miniature_fit),
        "lineage": lineage,
        # ``worker_count`` is retained for the physical-benchmark schema and
        # means maximum simultaneous subprocesses, not number of shards.
        "worker_count": concurrent_workers,
        "shard_count": len(ordered_specs),
        "maximum_concurrent_workers": concurrent_workers,
        "execution_evidence": dict(inputs.execution_evidence),
        "specs": [spec.as_manifest() for spec in ordered_specs],
        "worker_build_manifest_sha256": [
            value.manifest_sha256 for value in build_artifacts
        ],
        "worker_replay_manifest_sha256": [
            value.manifest_sha256 for value in replay_artifacts
        ],
        "train_matrices": {
            str(outer): {
                "path": str(train.matrices[outer].path),
                "sha256": train.matrices[outer].sha256,
                "rows": train.matrices[outer].row_count,
                "queries": train.matrices[outer].query_count,
                "fit_folds": list(train.matrices[outer].fit_folds),
                "feature_names": list(train.matrices[outer].feature_names),
                "feature_schema_sha256": train.matrices[
                    outer
                ].feature_schema_sha256,
                "group_sequence_sha256": train.group_sequence_sha256[outer],
            }
            for outer in TRAIN_FOLDS
        },
        "heldout_master": heldout_master_record_v12(heldout.artifact),
        "site_models": _model_records_v12(models),
        "outputs": {
            name: {
                "path": str(record.path),
                "sha256": record.sha256,
                "rows": record.rows,
            }
            for name, record in replay.outputs.items()
        },
        "gates": gates,
        "digests": digests,
        "io_bytes": io_bytes,
        "temporary_budget": {
            "maximum_bytes": inputs.maximum_spool_bytes,
            "observed_materialized_bytes": io_bytes["total_materialized"],
            "final_manifest_reserve_bytes": final_manifest_reserve_bytes,
            "pass": True,
        },
    }
    if prefit_bridge is not None:
        manifest["prefit_bridge"] = prefit_bridge_record_v12(prefit_bridge)
    _atomic_json(manifest_path, manifest)
    return ShardedRunArtifactV12(
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        query_count=len(inputs.query_ids),
        folds=TRAIN_FOLDS,
        lineage=lineage,
        io_bytes=io_bytes,
        specs=ordered_specs,
        build_artifacts=build_artifacts,
        train=train,
        heldout=heldout,
        prefit_bridge=prefit_bridge,
        models=models,
        replay_artifacts=replay_artifacts,
        replay=replay,
        gates=gates,
        digests=digests,
    )


__all__ = [
    "ENGINE_ID",
    "MergedHeldoutV12",
    "MergedReplayV12",
    "MergedTrainingV12",
    "SceneIRShardSourceV12",
    "ShardBuildArtifactV12",
    "ShardReplayArtifactV12",
    "ShardSpecV12",
    "ShardedPhysicalInputsV12",
    "ShardedRunArtifactV12",
    "WorkerJobV12",
    "aggregate_sharded_gates_v12",
    "build_shard_v12",
    "fit_site_models_sequential_v12",
    "fitted_site_model_records_v12",
    "heldout_master_record_v12",
    "load_fitted_site_model_records_v12",
    "load_heldout_master_record_v12",
    "load_shard_build_manifest_v12",
    "load_shard_replay_manifest_v12",
    "merge_heldout_shards_v12",
    "merge_replay_shards_v12",
    "merge_training_shards_v12",
    "plan_query_shards",
    "replay_shard_v12",
    "run_sharded_physical_sample",
    "spawn_manifest_workers_v12",
]


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--worker-job":
        raise SystemExit(
            "usage: python -m xgb_matcher.retrieval_fold_fused_sharded_oof_v12 "
            "--worker-job REQUEST.json RESPONSE.json"
        )
    _worker_subprocess_main_v12(Path(sys.argv[2]), Path(sys.argv[3]))
