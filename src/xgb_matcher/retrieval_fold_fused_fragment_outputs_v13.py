"""Compact, buffered and zero-copy replay outputs for retrieval v13.

Worker output Parquets are retained as ordered canonical fragments.  A small
manifest seals their physical layout and their layout-independent logical
content, so consumers can stream the global monolithic order without creating
merged copies.  PreRank500 deliberately omits ``payload_json``; top100 keeps
the complete payload and official proof material.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import hashlib
import json
import math
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .retrieval_fold_fused_fragment_dataset_v13 import (
    BuildFragmentDatasetV13,
    iter_heldout_query_tables_v13,
    logical_table_sha256_v13,
    merkle_root_v13,
)
from .retrieval_fold_fused_oof_v12 import (
    ArtifactRecord,
    ExecutionCaps,
    FeatureBatch,
    FittedSiteModel,
    FoldFusedInvariantError,
    GateThresholds,
    HeldoutMasterArtifact,
    HumanLabel,
    MasterContext,
    QueryOutcome,
    RankingSelection,
    TRAIN_FOLDS,
    sha256_file,
)
from .retrieval_fold_fused_sharded_oof_v12 import (
    ShardSpecV12,
    _iter_heldout_contexts_v12,
    _outcome_from_row_v12,
    _query_outcome_v12,
    _sequence_sha256,
    _validate_feature_batch_v12,
    _validate_selection_v12,
    _validate_statistics_v12,
    aggregate_sharded_gates_v12,
    fitted_site_model_records_v12,
    replay_shard_v12,
)


FRAGMENT_SCHEMA_VERSION = "sireto-retrieval-v13-compact-replay-fragment-1"
DATASET_SCHEMA_VERSION = "sireto-retrieval-v13-compact-replay-dataset-1"
ENGINE_ID = "FOLD_FUSED_ZERO_COPY_COMPACT_REPLAY_V13_1"
DEFAULT_ROW_GROUP_TARGET = 16_384
_ALLOWED_FOLDS = frozenset(TRAIN_FOLDS)
_SHA256_CHARS = frozenset("0123456789abcdef")


TOP500_SCHEMA = pa.schema(
    [
        ("query_id", pa.string()),
        ("fold", pa.int8()),
        ("candidate_siret", pa.string()),
        ("candidate_parent", pa.string()),
        ("rank", pa.int16()),
        ("site_score", pa.float64()),
        ("selection_reason", pa.string()),
    ]
)
TOP100_SCHEMA = pa.schema(
    [
        ("query_id", pa.string()),
        ("fold", pa.int8()),
        ("candidate_siret", pa.string()),
        ("candidate_parent", pa.string()),
        ("rank", pa.int16()),
        ("site_score", pa.float64()),
        ("selection_reason", pa.string()),
        ("payload_json", pa.string()),
    ]
)
OUTCOME_SCHEMA = pa.schema(
    [
        ("query_id", pa.string()),
        ("fold", pa.int8()),
        ("human_siret", pa.string()),
        ("master_candidate_count", pa.int16()),
        ("prerank_candidate_count", pa.int16()),
        ("product_candidate_count", pa.int16()),
        ("truth_master_rank", pa.int16()),
        ("truth_prerank_rank", pa.int16()),
        ("truth_product_rank", pa.int16()),
        ("master_hit_at_2000", pa.bool_()),
        ("prerank_hit_at_500", pa.bool_()),
        ("exact_hit_at_100", pa.bool_()),
        ("operational_hit_at_100", pa.bool_()),
        ("loss_reason", pa.string()),
        ("audit_metadata_json", pa.string()),
    ]
)
OUTPUT_SCHEMAS = {
    "top500": TOP500_SCHEMA,
    "top100": TOP100_SCHEMA,
    "outcomes": OUTCOME_SCHEMA,
}
OUTPUT_CAPS = {"top500": 500, "top100": 100, "outcomes": 1}


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


def _schema_sha256(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.remove_metadata().serialize().to_pybytes()).hexdigest()


def feature_batch_logical_leaf_v13(
    context: MasterContext,
    features: FeatureBatch,
) -> Mapping[str, Any]:
    """Seal all 319 float32 values and their exact candidate row identities."""

    values = np.ascontiguousarray(features.values, dtype=np.float32)
    if (
        features.query_id != context.query_id
        or features.scene_fold != context.scene_fold
        or len(values) != len(context.candidate_sirets)
        or features.candidate_sirets != context.candidate_sirets
        or features.candidate_parents != context.candidate_parents
    ):
        raise FoldFusedInvariantError("feature digest requires the complete ordered Master")
    return {
        "kind": "site_feature_values",
        "query_id": context.query_id,
        "fold": context.scene_fold,
        "row_count": len(values),
        "feature_schema_sha256": _sha256_json(list(features.feature_names)),
        "candidate_order_sha256": _sha256_json(
            list(zip(features.candidate_sirets, features.candidate_parents, strict=True))
        ),
        "statistics_sha256": features.statistics_sha256,
        "float32_c_order_sha256": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
    }


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


def _resolve_path(root: Path, value: Any) -> Path:
    path = Path(str(value or ""))
    if not path.is_absolute():
        path = root / path
    return path.resolve()


@dataclass(frozen=True)
class CompactOutputFileV13:
    name: str
    shard_id: str
    shard_ordinal: int
    path: Path
    sha256: str
    size_bytes: int
    row_count: int
    query_count: int
    row_group_count: int
    schema_sha256: str
    query_sequence_sha256: str
    logical_merkle_root: str

    def __post_init__(self) -> None:
        if (
            self.name not in OUTPUT_SCHEMAS
            or not self.shard_id
            or self.shard_ordinal < 0
            or not all(
                _valid_sha256(value)
                for value in (
                    self.sha256,
                    self.schema_sha256,
                    self.query_sequence_sha256,
                    self.logical_merkle_root,
                )
            )
            or self.size_bytes <= 0
            or self.row_count < self.query_count
            or self.query_count <= 0
            or self.row_group_count <= 0
        ):
            raise FoldFusedInvariantError("invalid compact output fragment")


@dataclass(frozen=True)
class CompactReplayFragmentArtifactV13:
    spec: ShardSpecV12
    manifest_path: Path
    manifest_sha256: str
    fragment_id: str
    scientific_manifest_path: Path
    scientific_manifest_sha256: str
    build_manifest_sha256: str
    heldout_dataset_merkle_root: str
    lineage: Mapping[str, str]
    outputs: Mapping[str, CompactOutputFileV13]
    outcomes: tuple[QueryOutcome, ...]
    model_sha256: Mapping[int, str]
    feature_values_logical_leaves: tuple[Mapping[str, Any], ...]
    feature_values_logical_merkle_root: str

    def __post_init__(self) -> None:
        if (
            not _valid_sha256(self.manifest_sha256)
            or not _valid_sha256(self.fragment_id)
            or not _valid_sha256(self.scientific_manifest_sha256)
            or not _valid_sha256(self.build_manifest_sha256)
            or not _valid_sha256(self.heldout_dataset_merkle_root)
            or set(self.outputs) != set(OUTPUT_SCHEMAS)
            or set(self.model_sha256) != set(TRAIN_FOLDS)
            or len(self.outcomes) != self.spec.query_count
            or len(self.feature_values_logical_leaves) != self.spec.query_count
            or not _valid_sha256(self.feature_values_logical_merkle_root)
        ):
            raise FoldFusedInvariantError("invalid compact replay fragment artifact")


@dataclass(frozen=True)
class CompactReplayDatasetV13:
    manifest_path: Path
    manifest_sha256: str
    dataset_id: str
    lineage: Mapping[str, str]
    specs: tuple[ShardSpecV12, ...]
    outputs: Mapping[str, tuple[CompactOutputFileV13, ...]]
    query_count: int
    query_sequence_sha256: str
    logical_merkle_root: Mapping[str, str]
    physical_merkle_root: str
    model_sha256: Mapping[int, str]
    build_manifest_sha256: tuple[str, ...]
    heldout_dataset_merkle_root: str
    feature_values_logical_merkle_root: str

    def __post_init__(self) -> None:
        if (
            not _valid_sha256(self.manifest_sha256)
            or not _valid_sha256(self.dataset_id)
            or not _valid_sha256(self.query_sequence_sha256)
            or not _valid_sha256(self.physical_merkle_root)
            or not _valid_sha256(self.heldout_dataset_merkle_root)
            or not _valid_sha256(self.feature_values_logical_merkle_root)
            or set(self.outputs) != set(OUTPUT_SCHEMAS)
            or set(self.logical_merkle_root) != set(OUTPUT_SCHEMAS)
            or not all(_valid_sha256(value) for value in self.logical_merkle_root.values())
            or set(self.model_sha256) != set(TRAIN_FOLDS)
            or len(self.specs) != len(self.build_manifest_sha256)
            or self.query_count != sum(value.query_count for value in self.specs)
            or any(len(values) != len(self.specs) for values in self.outputs.values())
        ):
            raise FoldFusedInvariantError("invalid compact replay fragment dataset")


class BufferedCompactOOFOutputStoreV13:
    """Write many whole queries per row group while preserving query bounds."""

    def __init__(
        self,
        output_dir: Path,
        *,
        row_group_target: int = DEFAULT_ROW_GROUP_TARGET,
    ) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.row_group_target = int(row_group_target)
        if self.row_group_target < 500:
            raise FoldFusedInvariantError("output row-group target must fit one top500")
        self.paths = {
            "top500": self.output_dir / "oof_prerank500_compact.parquet",
            "top100": self.output_dir / "oof_top100.parquet",
            "outcomes": self.output_dir / "oof_query_outcomes.parquet",
        }
        self._writers = {
            name: pq.ParquetWriter(path, OUTPUT_SCHEMAS[name], compression="zstd")
            for name, path in self.paths.items()
        }
        self._buffers: dict[str, list[pa.Table]] = {
            name: [] for name in self.paths
        }
        self._buffer_rows = {name: 0 for name in self.paths}
        self._rows = {name: 0 for name in self.paths}
        self._queries = {name: 0 for name in self.paths}
        self._sequences: dict[str, list[tuple[str, int]]] = {
            name: [] for name in self.paths
        }
        self._leaves: dict[str, list[Mapping[str, Any]]] = {
            name: [] for name in self.paths
        }
        self._sealed = False
        self.feature_values_logical_leaves: list[Mapping[str, Any]] = []

    def _flush(self, name: str) -> None:
        values = self._buffers[name]
        if not values:
            return
        table = values[0] if len(values) == 1 else pa.concat_tables(values)
        self._writers[name].write_table(table, row_group_size=table.num_rows)
        values.clear()
        self._buffer_rows[name] = 0

    def _append_query_table(self, name: str, table: pa.Table) -> None:
        if self._sealed or not table.num_rows:
            raise FoldFusedInvariantError("cannot append empty/sealed output query")
        query_ids = {str(value) for value in table["query_id"].to_pylist()}
        folds = {int(value.as_py()) for value in table["fold"]}
        if len(query_ids) != 1 or len(folds) != 1 or not folds.issubset(_ALLOWED_FOLDS):
            raise FoldFusedInvariantError("compact output query identity drift")
        query_id = next(iter(query_ids))
        fold = next(iter(folds))
        cap = OUTPUT_CAPS[name]
        if not 1 <= table.num_rows <= cap:
            raise FoldFusedInvariantError("compact output query escaped cap")
        if name == "top500" and len(
            set(str(value) for value in table["candidate_siret"].to_pylist())
        ) != table.num_rows:
            raise FoldFusedInvariantError("compact PreRank500 natural join is not unique")
        if name in {"top500", "top100"} and tuple(
            int(value.as_py()) for value in table["rank"]
        ) != tuple(range(1, table.num_rows + 1)):
            raise FoldFusedInvariantError("compact output ranking order drift")
        if self._buffers[name] and self._buffer_rows[name] + table.num_rows > self.row_group_target:
            self._flush(name)
        canonical = pa.Table.from_arrays(table.columns, schema=OUTPUT_SCHEMAS[name])
        self._buffers[name].append(canonical)
        self._buffer_rows[name] += table.num_rows
        self._rows[name] += table.num_rows
        self._queries[name] += 1
        self._sequences[name].append((query_id, fold))
        self._leaves[name].append(
            {
                "kind": name,
                "query_id": query_id,
                "fold": fold,
                "row_count": table.num_rows,
                "table_sha256": logical_table_sha256_v13(canonical),
            }
        )

    def _ranking_table(
        self,
        context: MasterContext,
        features: FeatureBatch,
        scores: np.ndarray,
        indices: Sequence[int],
        reasons: Sequence[str],
        *,
        include_payload: bool,
    ) -> pa.Table:
        schema = TOP100_SCHEMA if include_payload else TOP500_SCHEMA
        records: list[dict[str, Any]] = []
        for rank, (index, reason) in enumerate(zip(indices, reasons, strict=True), start=1):
            row = {
                "query_id": context.query_id,
                "fold": context.scene_fold,
                "candidate_siret": features.candidate_sirets[index],
                "candidate_parent": features.candidate_parents[index],
                "rank": rank,
                "site_score": float(scores[index]),
                "selection_reason": str(reason),
            }
            if include_payload:
                payload = features.row_payloads[index] if features.row_payloads else {}
                row["payload_json"] = _canonical_json(payload)
            records.append(row)
        return pa.Table.from_pylist(records, schema=schema)

    def append_rankings(
        self,
        context: MasterContext,
        features: FeatureBatch,
        scores: np.ndarray,
        selection: RankingSelection,
    ) -> None:
        self.feature_values_logical_leaves.append(
            feature_batch_logical_leaf_v13(context, features)
        )
        top500 = self._ranking_table(
            context,
            features,
            scores,
            selection.prerank_indices,
            ("site_lambdamart_prerank",) * len(selection.prerank_indices),
            include_payload=False,
        )
        top100 = self._ranking_table(
            context,
            features,
            scores,
            selection.product_indices,
            selection.product_reasons,
            include_payload=True,
        )
        self._append_query_table("top500", top500)
        self._append_query_table("top100", top100)

    def append_outcome(self, outcome: QueryOutcome) -> None:
        row = {
            **{
                name: getattr(outcome, name)
                for name in outcome.__dataclass_fields__
                if name != "audit_metadata"
            },
            "audit_metadata_json": _canonical_json(outcome.audit_metadata),
        }
        self._append_query_table(
            "outcomes", pa.Table.from_pylist([row], schema=OUTCOME_SCHEMA)
        )

    def seal(
        self, *, shard_id: str, shard_ordinal: int
    ) -> Mapping[str, CompactOutputFileV13]:
        if self._sealed:
            raise FoldFusedInvariantError("compact output store was sealed twice")
        self._sealed = True
        for name in self.paths:
            self._flush(name)
            self._writers[name].close()
        output: dict[str, CompactOutputFileV13] = {}
        for name, path in self.paths.items():
            parquet = pq.ParquetFile(path)
            if (
                self._queries[name] <= 0
                or parquet.metadata.num_rows != self._rows[name]
                or not parquet.schema_arrow.equals(OUTPUT_SCHEMAS[name])
            ):
                raise FoldFusedInvariantError("compact output seal drift")
            output[name] = CompactOutputFileV13(
                name=name,
                shard_id=str(shard_id),
                shard_ordinal=int(shard_ordinal),
                path=path,
                sha256=sha256_file(path),
                size_bytes=path.stat().st_size,
                row_count=self._rows[name],
                query_count=self._queries[name],
                row_group_count=parquet.num_row_groups,
                schema_sha256=_schema_sha256(parquet.schema_arrow),
                query_sequence_sha256=_sequence_sha256(self._sequences[name]),
                logical_merkle_root=merkle_root_v13(self._leaves[name]),
            )
        return output


class CompactParquetOOFOutputStoreV13:
    """V12 replay-sink adapter backed by the buffered compact writer.

    ``replay_shard_v12`` consumes only the historical ``ArtifactRecord``
    surface.  The richer fragment records remain available on
    :attr:`compact_outputs` for v13 dataset publication.
    """

    def __init__(
        self,
        output_dir: Path,
        *,
        shard_id: str,
        shard_ordinal: int,
        row_group_target: int = DEFAULT_ROW_GROUP_TARGET,
    ) -> None:
        self.shard_id = str(shard_id)
        self.shard_ordinal = int(shard_ordinal)
        self.delegate = BufferedCompactOOFOutputStoreV13(
            output_dir,
            row_group_target=row_group_target,
        )
        self.compact_outputs: Mapping[str, CompactOutputFileV13] | None = None
        self.feature_values_logical_leaves: tuple[Mapping[str, Any], ...] = ()
        self.feature_values_logical_merkle_root: str | None = None

    def append_rankings(
        self,
        context: MasterContext,
        features: FeatureBatch,
        scores: np.ndarray,
        selection: RankingSelection,
    ) -> None:
        self.delegate.append_rankings(context, features, scores, selection)

    def append_outcome(self, outcome: QueryOutcome) -> None:
        self.delegate.append_outcome(outcome)

    def seal(self) -> Mapping[str, ArtifactRecord]:
        compact = self.delegate.seal(
            shard_id=self.shard_id, shard_ordinal=self.shard_ordinal
        )
        self.compact_outputs = compact
        self.feature_values_logical_leaves = tuple(
            self.delegate.feature_values_logical_leaves
        )
        self.feature_values_logical_merkle_root = merkle_root_v13(
            self.feature_values_logical_leaves
        )
        return {
            name: ArtifactRecord(value.path, value.sha256, value.row_count)
            for name, value in compact.items()
        }


def _output_record(value: CompactOutputFileV13, root: Path) -> dict[str, Any]:
    return {
        "name": value.name,
        "shard_id": value.shard_id,
        "shard_ordinal": value.shard_ordinal,
        "path": _relative_or_absolute(value.path, root),
        "sha256": value.sha256,
        "size_bytes": value.size_bytes,
        "row_count": value.row_count,
        "query_count": value.query_count,
        "row_group_count": value.row_group_count,
        "schema_sha256": value.schema_sha256,
        "query_sequence_sha256": value.query_sequence_sha256,
        "logical_merkle_root": value.logical_merkle_root,
    }


def _output_leaf_record(value: CompactOutputFileV13) -> dict[str, Any]:
    """Path-independent physical identity for Merkle/content addressing."""

    return {
        "name": value.name,
        "shard_id": value.shard_id,
        "shard_ordinal": value.shard_ordinal,
        "sha256": value.sha256,
        "size_bytes": value.size_bytes,
        "row_count": value.row_count,
        "query_count": value.query_count,
        "row_group_count": value.row_group_count,
        "schema_sha256": value.schema_sha256,
        "query_sequence_sha256": value.query_sequence_sha256,
        "logical_merkle_root": value.logical_merkle_root,
    }


def replay_fragment_v13(
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
    heldout_dataset_merkle_root: str,
    caps: ExecutionCaps = ExecutionCaps(),
    payload_factory: Any | None = None,
    row_group_target: int = DEFAULT_ROW_GROUP_TARGET,
) -> CompactReplayFragmentArtifactV13:
    """Run the frozen v12 replay with only a compact physical sink swapped."""

    lineage_value = {str(key): str(value) for key, value in lineage.items()}
    if (
        not lineage_value
        or any(not _valid_sha256(value) for value in lineage_value.values())
        or not _valid_sha256(build_manifest_sha256)
        or not _valid_sha256(heldout_dataset_merkle_root)
        or set(models) != set(TRAIN_FOLDS)
    ):
        raise FoldFusedInvariantError("compact replay lacks sealed lineage/models")
    output = Path(output_dir).resolve()
    store_box: dict[str, CompactParquetOOFOutputStoreV13] = {}

    def output_store_factory(path: Path) -> CompactParquetOOFOutputStoreV13:
        if store_box:
            raise FoldFusedInvariantError("compact replay created two output sinks")
        value = CompactParquetOOFOutputStoreV13(
            path,
            shard_id=spec.shard_id,
            shard_ordinal=spec.shard_ordinal,
            row_group_target=row_group_target,
        )
        store_box["value"] = value
        return value

    scientific = replay_shard_v12(
        spec,
        heldout=heldout,
        labels=labels,
        contexts=contexts,
        model_backend=model_backend,
        models=models,
        selector=selector,
        output_dir=output,
        lineage=lineage_value,
        build_manifest_sha256=build_manifest_sha256,
        caps=caps,
        payload_factory=payload_factory,
        output_store_factory=output_store_factory,
    )
    store = store_box.get("value")
    if (
        store is None
        or store.compact_outputs is None
        or store.feature_values_logical_merkle_root is None
    ):
        raise FoldFusedInvariantError("v12 replay did not seal compact outputs")
    outputs = store.compact_outputs
    outcomes = scientific.outcomes
    model_sha = {outer: models[outer].sha256 for outer in TRAIN_FOLDS}
    identity = {
        "engine_id": ENGINE_ID,
        "spec_sha256": spec.digest,
        "build_manifest_sha256": build_manifest_sha256,
        "heldout_dataset_merkle_root": heldout_dataset_merkle_root,
        "lineage": lineage_value,
        "query_count": len(outcomes),
        "model_sha256": {str(key): value for key, value in model_sha.items()},
        "output_sha256": {name: value.sha256 for name, value in outputs.items()},
        "output_logical_merkle_root": {
            name: value.logical_merkle_root for name, value in outputs.items()
        },
        "feature_values_logical_merkle_root": (
            store.feature_values_logical_merkle_root
        ),
    }
    fragment_id = _sha256_json(identity)
    manifest = {
        "schema_version": FRAGMENT_SCHEMA_VERSION,
        "status": "SEALED_COMPACT_REPLAY_FRAGMENT",
        "engine_id": ENGINE_ID,
        "fragment_id": fragment_id,
        "identity": identity,
        "spec": spec.as_manifest(),
        "folds": list(spec.folds),
        "fold0_opened": False,
        "fold1_opened": False,
        "positive_injection": False,
        "lineage": lineage_value,
        "build_manifest_sha256": build_manifest_sha256,
        "heldout_sha256": heldout.sha256,
        "heldout_dataset_merkle_root": heldout_dataset_merkle_root,
        "model_by_outer": fitted_site_model_records_v12(models),
        "query_count": len(outcomes),
        "query_sequence_sha256": spec.query_sequence_sha256,
        "row_group_target": int(row_group_target),
        "feature_values_logical_leaves": list(
            store.feature_values_logical_leaves
        ),
        "feature_values_logical_merkle_root": (
            store.feature_values_logical_merkle_root
        ),
        "scientific_replay_manifest": {
            "path": _relative_or_absolute(scientific.manifest_path, output),
            "sha256": scientific.manifest_sha256,
        },
        "outputs": {
            name: _output_record(value, output) for name, value in outputs.items()
        },
    }
    manifest_path = output / "compact_fragment_manifest.json"
    _atomic_json(manifest_path, manifest)
    return CompactReplayFragmentArtifactV13(
        spec=spec,
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        fragment_id=fragment_id,
        scientific_manifest_path=scientific.manifest_path,
        scientific_manifest_sha256=scientific.manifest_sha256,
        build_manifest_sha256=build_manifest_sha256,
        heldout_dataset_merkle_root=heldout_dataset_merkle_root,
        lineage=lineage_value,
        outputs=outputs,
        outcomes=tuple(outcomes),
        model_sha256=model_sha,
        feature_values_logical_leaves=store.feature_values_logical_leaves,
        feature_values_logical_merkle_root=(
            store.feature_values_logical_merkle_root
        ),
    )


def _iter_output_query_tables(
    path: Path,
    name: str,
) -> Iterable[pa.Table]:
    parquet = pq.ParquetFile(path)
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
                    "compact output query is split or duplicated"
                )
            seen.add(query_id)
            table = row_group_table.slice(start, stop - start)
            folds = {int(value.as_py()) for value in table["fold"]}
            if len(folds) != 1 or not folds.issubset(_ALLOWED_FOLDS):
                raise FoldFusedInvariantError("compact output fold drift")
            if not 1 <= table.num_rows <= OUTPUT_CAPS[name]:
                raise FoldFusedInvariantError("compact output cap drift")
            if name == "top500" and len(
                set(str(value) for value in table["candidate_siret"].to_pylist())
            ) != table.num_rows:
                raise FoldFusedInvariantError(
                    "compact PreRank500 natural join is not unique"
                )
            if name in {"top500", "top100"} and tuple(
                int(value.as_py()) for value in table["rank"]
            ) != tuple(range(1, table.num_rows + 1)):
                raise FoldFusedInvariantError("compact output rank drift")
            yield table
            start = stop


def _scan_output_file(
    record: CompactOutputFileV13,
) -> tuple[list[tuple[str, int]], list[Mapping[str, Any]]]:
    parquet = pq.ParquetFile(record.path)
    if (
        parquet.metadata.num_rows != record.row_count
        or parquet.num_row_groups != record.row_group_count
        or _schema_sha256(parquet.schema_arrow) != record.schema_sha256
        or not parquet.schema_arrow.equals(OUTPUT_SCHEMAS[record.name])
    ):
        raise FoldFusedInvariantError("compact output physical schema/count drift")
    sequence: list[tuple[str, int]] = []
    leaves: list[Mapping[str, Any]] = []
    for table in _iter_output_query_tables(record.path, record.name):
        query_id = str(table["query_id"][0].as_py())
        fold = int(table["fold"][0].as_py())
        sequence.append((query_id, fold))
        leaves.append(
            {
                "kind": record.name,
                "query_id": query_id,
                "fold": fold,
                "row_count": table.num_rows,
                "table_sha256": logical_table_sha256_v13(table),
            }
        )
    if (
        len(sequence) != record.query_count
        or _sequence_sha256(sequence) != record.query_sequence_sha256
        or merkle_root_v13(leaves) != record.logical_merkle_root
    ):
        raise FoldFusedInvariantError("compact output logical content drift")
    return sequence, leaves


def _load_output_record(
    value: Mapping[str, Any],
    root: Path,
    *,
    verify_hashes: bool,
    verify_logical: bool,
) -> CompactOutputFileV13:
    path = _resolve_path(root, value.get("path"))
    record = CompactOutputFileV13(
        name=str(value.get("name") or ""),
        shard_id=str(value.get("shard_id") or ""),
        shard_ordinal=int(value.get("shard_ordinal", -1)),
        path=path,
        sha256=str(value.get("sha256") or ""),
        size_bytes=int(value.get("size_bytes", -1)),
        row_count=int(value.get("row_count", -1)),
        query_count=int(value.get("query_count", -1)),
        row_group_count=int(value.get("row_group_count", -1)),
        schema_sha256=str(value.get("schema_sha256") or ""),
        query_sequence_sha256=str(value.get("query_sequence_sha256") or ""),
        logical_merkle_root=str(value.get("logical_merkle_root") or ""),
    )
    if (
        not path.is_file()
        or path.stat().st_size != record.size_bytes
        or (verify_hashes and sha256_file(path) != record.sha256)
    ):
        raise FoldFusedInvariantError("compact output hash/size drift")
    if verify_logical:
        _scan_output_file(record)
    return record


def load_compact_replay_fragment_v13(
    path: Path,
    *,
    verify_hashes: bool = True,
    verify_logical: bool = True,
) -> CompactReplayFragmentArtifactV13:
    manifest_path = Path(path).resolve()
    if manifest_path.is_dir():
        manifest_path = manifest_path / "compact_fragment_manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        raw.get("schema_version") != FRAGMENT_SCHEMA_VERSION
        or raw.get("status") != "SEALED_COMPACT_REPLAY_FRAGMENT"
        or raw.get("engine_id") != ENGINE_ID
        or raw.get("fold0_opened") is not False
        or raw.get("fold1_opened") is not False
        or raw.get("positive_injection") is not False
    ):
        raise FoldFusedInvariantError("invalid or unsafe compact replay fragment")
    identity = raw.get("identity")
    if (
        not isinstance(identity, Mapping)
        or raw.get("fragment_id") != _sha256_json(identity)
    ):
        raise FoldFusedInvariantError("compact replay fragment address drift")
    spec = ShardSpecV12.from_manifest(raw.get("spec") or {})
    lineage = {str(key): str(value) for key, value in (raw.get("lineage") or {}).items()}
    build_sha = str(raw.get("build_manifest_sha256") or "")
    heldout_dataset_merkle = str(
        raw.get("heldout_dataset_merkle_root") or ""
    )
    if (
        not lineage
        or any(not _valid_sha256(value) for value in lineage.values())
        or not _valid_sha256(build_sha)
        or not _valid_sha256(heldout_dataset_merkle)
        or raw.get("folds") != list(spec.folds)
        or raw.get("query_count") != spec.query_count
        or raw.get("query_sequence_sha256") != spec.query_sequence_sha256
    ):
        raise FoldFusedInvariantError("compact replay fragment population drift")
    outputs_raw = raw.get("outputs")
    if not isinstance(outputs_raw, Mapping) or set(outputs_raw) != set(OUTPUT_SCHEMAS):
        raise FoldFusedInvariantError("compact replay outputs incomplete")
    outputs = {
        name: _load_output_record(
            record,
            manifest_path.parent,
            verify_hashes=verify_hashes,
            verify_logical=verify_logical,
        )
        for name, record in outputs_raw.items()
    }
    expected_sequence = tuple(zip(spec.query_ids, spec.query_folds, strict=True))
    if any(
        value.query_sequence_sha256 != spec.query_sequence_sha256
        or value.query_count != spec.query_count
        or value.shard_id != spec.shard_id
        or value.shard_ordinal != spec.shard_ordinal
        for value in outputs.values()
    ):
        raise FoldFusedInvariantError("compact replay output/spec drift")
    outcomes = tuple(
        _outcome_from_row_v12(table.to_pylist()[0])
        for table in _iter_output_query_tables(outputs["outcomes"].path, "outcomes")
    )
    if tuple((value.query_id, value.fold) for value in outcomes) != expected_sequence:
        raise FoldFusedInvariantError("compact replay outcome sequence drift")
    model_records = raw.get("model_by_outer")
    if not isinstance(model_records, Mapping) or set(model_records) != {
        str(value) for value in TRAIN_FOLDS
    }:
        raise FoldFusedInvariantError("compact replay model records incomplete")
    model_sha = {
        outer: str(model_records[str(outer)].get("sha256") or "")
        for outer in TRAIN_FOLDS
    }
    if any(not _valid_sha256(value) for value in model_sha.values()):
        raise FoldFusedInvariantError("compact replay model hashes invalid")
    feature_leaves_raw = raw.get("feature_values_logical_leaves")
    feature_root = str(raw.get("feature_values_logical_merkle_root") or "")
    if (
        not isinstance(feature_leaves_raw, list)
        or len(feature_leaves_raw) != spec.query_count
        or not _valid_sha256(feature_root)
        or merkle_root_v13(feature_leaves_raw) != feature_root
        or tuple(
            (str(value.get("query_id") or ""), int(value.get("fold", -1)))
            for value in feature_leaves_raw
        )
        != expected_sequence
    ):
        raise FoldFusedInvariantError("compact replay feature-value digest drift")
    scientific_record = raw.get("scientific_replay_manifest")
    if not isinstance(scientific_record, Mapping):
        raise FoldFusedInvariantError("compact replay scientific manifest missing")
    scientific_path = _resolve_path(
        manifest_path.parent, scientific_record.get("path")
    )
    scientific_sha = str(scientific_record.get("sha256") or "")
    if (
        not scientific_path.is_file()
        or not _valid_sha256(scientific_sha)
        or (verify_hashes and sha256_file(scientific_path) != scientific_sha)
    ):
        raise FoldFusedInvariantError("compact replay scientific manifest drift")
    expected_identity = {
        "engine_id": ENGINE_ID,
        "spec_sha256": spec.digest,
        "build_manifest_sha256": build_sha,
        "heldout_dataset_merkle_root": heldout_dataset_merkle,
        "lineage": lineage,
        "query_count": spec.query_count,
        "model_sha256": {str(key): value for key, value in model_sha.items()},
        "output_sha256": {name: value.sha256 for name, value in outputs.items()},
        "output_logical_merkle_root": {
            name: value.logical_merkle_root for name, value in outputs.items()
        },
        "feature_values_logical_merkle_root": feature_root,
    }
    if dict(identity) != expected_identity:
        raise FoldFusedInvariantError("compact replay fragment identity drift")
    return CompactReplayFragmentArtifactV13(
        spec=spec,
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        fragment_id=str(raw["fragment_id"]),
        scientific_manifest_path=scientific_path,
        scientific_manifest_sha256=scientific_sha,
        build_manifest_sha256=build_sha,
        heldout_dataset_merkle_root=heldout_dataset_merkle,
        lineage=lineage,
        outputs=outputs,
        outcomes=outcomes,
        model_sha256=model_sha,
        feature_values_logical_leaves=tuple(dict(value) for value in feature_leaves_raw),
        feature_values_logical_merkle_root=feature_root,
    )


def _ordered_replays(
    artifacts: Sequence[CompactReplayFragmentArtifactV13],
) -> tuple[CompactReplayFragmentArtifactV13, ...]:
    strict = tuple(
        load_compact_replay_fragment_v13(value.manifest_path)
        for value in artifacts
    )
    ordered = tuple(sorted(strict, key=lambda value: value.spec.shard_ordinal))
    if not ordered:
        raise FoldFusedInvariantError("no compact replay fragments supplied")
    lineage = dict(ordered[0].lineage)
    model_sha = dict(ordered[0].model_sha256)
    heldout_merkle = ordered[0].heldout_dataset_merkle_root
    stop = 0
    seen: set[str] = set()
    for ordinal, value in enumerate(ordered):
        if (
            value.spec.shard_ordinal != ordinal
            or value.spec.sequence_start != stop
            or dict(value.lineage) != lineage
            or dict(value.model_sha256) != model_sha
            or value.heldout_dataset_merkle_root != heldout_merkle
            or seen.intersection(value.spec.query_ids)
        ):
            raise FoldFusedInvariantError("compact replay fragments overlap/gap/mix")
        seen.update(value.spec.query_ids)
        stop = value.spec.sequence_stop
    return ordered


def seal_compact_replay_dataset_v13(
    artifacts: Sequence[CompactReplayFragmentArtifactV13],
    output_dir: Path,
) -> CompactReplayDatasetV13:
    """Seal ordered worker outputs as one logical dataset, without merge."""

    ordered = _ordered_replays(artifacts)
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    specs = tuple(value.spec for value in ordered)
    lineage = dict(ordered[0].lineage)
    model_sha = dict(ordered[0].model_sha256)
    heldout_dataset_merkle = ordered[0].heldout_dataset_merkle_root
    global_sequence = tuple(
        (query_id, fold)
        for spec in specs
        for query_id, fold in zip(spec.query_ids, spec.query_folds, strict=True)
    )
    outputs = {
        name: tuple(value.outputs[name] for value in ordered)
        for name in OUTPUT_SCHEMAS
    }
    logical_roots: dict[str, str] = {}
    physical_leaves: list[Mapping[str, Any]] = []
    for artifact in ordered:
        for name in OUTPUT_SCHEMAS:
            physical_leaves.append(_output_leaf_record(artifact.outputs[name]))
    for name, fragments in outputs.items():
        sequence: list[tuple[str, int]] = []
        leaves: list[Mapping[str, Any]] = []
        for fragment in fragments:
            current_sequence, current_leaves = _scan_output_file(fragment)
            sequence.extend(current_sequence)
            leaves.extend(current_leaves)
        if tuple(sequence) != global_sequence:
            raise FoldFusedInvariantError(f"compact {name} global order drift")
        logical_roots[name] = merkle_root_v13(leaves)
    feature_leaves = tuple(
        value
        for artifact in ordered
        for value in artifact.feature_values_logical_leaves
    )
    feature_root = merkle_root_v13(feature_leaves)
    physical_root = merkle_root_v13(physical_leaves)
    identity = {
        "engine_id": ENGINE_ID,
        "lineage": lineage,
        "ordered_spec_sha256": [value.digest for value in specs],
        "ordered_fragment_id": [
            value.fragment_id for value in ordered
        ],
        "ordered_build_manifest_sha256": [
            value.build_manifest_sha256 for value in ordered
        ],
        "query_count": len(global_sequence),
        "query_sequence_sha256": _sequence_sha256(global_sequence),
        "model_sha256": {str(key): value for key, value in model_sha.items()},
        "heldout_dataset_merkle_root": heldout_dataset_merkle,
        "logical_merkle_root": logical_roots,
        "feature_values_logical_merkle_root": feature_root,
        "physical_merkle_root": physical_root,
    }
    dataset_id = _sha256_json(identity)
    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "status": "SEALED_CANONICAL_COMPACT_REPLAY_DATASET",
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
        "ordered_fragment_manifests": [
            {
                "path": _relative_or_absolute(value.manifest_path, output),
                "sha256": value.manifest_sha256,
                "fragment_id": value.fragment_id,
            }
            for value in ordered
        ],
        "ordered_build_manifest_sha256": [
            value.build_manifest_sha256 for value in ordered
        ],
        "query_count": len(global_sequence),
        "query_sequence_sha256": _sequence_sha256(global_sequence),
        "model_sha256": {str(key): value for key, value in model_sha.items()},
        "heldout_dataset_merkle_root": heldout_dataset_merkle,
        "outputs": {
            name: {
                "fragments": [_output_record(value, output) for value in fragments],
                "rows": sum(value.row_count for value in fragments),
                "fragment_count": len(fragments),
                "logical_merkle_root": logical_roots[name],
            }
            for name, fragments in outputs.items()
        },
        "feature_values_logical_merkle_root": feature_root,
        "physical_merkle_root": physical_root,
    }
    manifest_path = output / "manifest.json"
    _atomic_json(manifest_path, manifest)
    return CompactReplayDatasetV13(
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        dataset_id=dataset_id,
        lineage=lineage,
        specs=specs,
        outputs=outputs,
        query_count=len(global_sequence),
        query_sequence_sha256=_sequence_sha256(global_sequence),
        logical_merkle_root=logical_roots,
        physical_merkle_root=physical_root,
        model_sha256=model_sha,
        build_manifest_sha256=tuple(
            value.build_manifest_sha256 for value in ordered
        ),
        heldout_dataset_merkle_root=heldout_dataset_merkle,
        feature_values_logical_merkle_root=feature_root,
    )


def iter_compact_output_query_tables_v13(
    dataset: CompactReplayDatasetV13,
    name: str,
) -> Iterable[pa.Table]:
    """Stream one compact family in global monolithic query order."""

    if name not in OUTPUT_SCHEMAS:
        raise KeyError(name)
    observed: list[tuple[str, int]] = []
    leaves: list[Mapping[str, Any]] = []
    for fragment in dataset.outputs[name]:
        for table in _iter_output_query_tables(fragment.path, name):
            query_id = str(table["query_id"][0].as_py())
            fold = int(table["fold"][0].as_py())
            observed.append((query_id, fold))
            leaves.append(
                {
                    "kind": name,
                    "query_id": query_id,
                    "fold": fold,
                    "row_count": table.num_rows,
                    "table_sha256": logical_table_sha256_v13(table),
                }
            )
            yield table
    if (
        len(observed) != dataset.query_count
        or _sequence_sha256(observed) != dataset.query_sequence_sha256
        or merkle_root_v13(leaves) != dataset.logical_merkle_root[name]
    ):
        raise FoldFusedInvariantError("compact output dataset logical drift")


def iter_compact_outcomes_v13(
    dataset: CompactReplayDatasetV13,
) -> Iterable[QueryOutcome]:
    for table in iter_compact_output_query_tables_v13(dataset, "outcomes"):
        yield _outcome_from_row_v12(table.to_pylist()[0])


def aggregate_compact_replay_gates_v13(
    dataset: CompactReplayDatasetV13,
    gates: GateThresholds = GateThresholds(),
) -> Mapping[str, Any]:
    return aggregate_sharded_gates_v12(tuple(iter_compact_outcomes_v13(dataset)), gates)


def load_compact_replay_dataset_v13(
    path: Path,
    *,
    verify_hashes: bool = True,
    verify_logical: bool = True,
) -> CompactReplayDatasetV13:
    manifest_path = Path(path).resolve()
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        raw.get("schema_version") != DATASET_SCHEMA_VERSION
        or raw.get("status") != "SEALED_CANONICAL_COMPACT_REPLAY_DATASET"
        or raw.get("engine_id") != ENGINE_ID
        or raw.get("folds") != list(TRAIN_FOLDS)
        or raw.get("fold0_opened") is not False
        or raw.get("fold1_opened") is not False
        or raw.get("positive_injection") is not False
        or raw.get("copy_materialized") is not False
    ):
        raise FoldFusedInvariantError("invalid or unsafe compact replay dataset")
    identity = raw.get("identity")
    if (
        not isinstance(identity, Mapping)
        or raw.get("dataset_id") != _sha256_json(identity)
    ):
        raise FoldFusedInvariantError("compact replay dataset address drift")
    specs = tuple(
        ShardSpecV12.from_manifest(value) for value in raw.get("specs") or ()
    )
    fragment_records = raw.get("ordered_fragment_manifests")
    if not specs or not isinstance(fragment_records, list) or len(fragment_records) != len(specs):
        raise FoldFusedInvariantError("compact replay source fragments missing")
    root = manifest_path.parent
    fragments: list[CompactReplayFragmentArtifactV13] = []
    for ordinal, (record, spec) in enumerate(zip(fragment_records, specs, strict=True)):
        source = _resolve_path(root, record.get("path"))
        digest = str(record.get("sha256") or "")
        if (
            spec.shard_ordinal != ordinal
            or not source.is_file()
            or not _valid_sha256(digest)
            or (verify_hashes and sha256_file(source) != digest)
        ):
            raise FoldFusedInvariantError("compact replay source manifest drift")
        loaded = load_compact_replay_fragment_v13(
            source,
            verify_hashes=verify_hashes,
            verify_logical=verify_logical,
        )
        if (
            loaded.spec != spec
            or loaded.manifest_sha256 != digest
            or loaded.fragment_id != record.get("fragment_id")
        ):
            raise FoldFusedInvariantError("compact replay source identity drift")
        fragments.append(loaded)
    ordered = _ordered_replays(fragments)
    lineage = dict(ordered[0].lineage)
    model_sha = dict(ordered[0].model_sha256)
    heldout_dataset_merkle = ordered[0].heldout_dataset_merkle_root
    outputs = {
        name: tuple(value.outputs[name] for value in ordered)
        for name in OUTPUT_SCHEMAS
    }
    global_sequence = tuple(
        (query_id, fold)
        for spec in specs
        for query_id, fold in zip(spec.query_ids, spec.query_folds, strict=True)
    )
    logical_roots: dict[str, str] = {}
    physical_leaves: list[Mapping[str, Any]] = []
    for artifact in ordered:
        for name in OUTPUT_SCHEMAS:
            physical_leaves.append(_output_leaf_record(artifact.outputs[name]))
    for name, values in outputs.items():
        leaves: list[Mapping[str, Any]] = []
        sequence: list[tuple[str, int]] = []
        for value in values:
            part_sequence, part_leaves = _scan_output_file(value)
            sequence.extend(part_sequence)
            leaves.extend(part_leaves)
        if tuple(sequence) != global_sequence:
            raise FoldFusedInvariantError("compact replay dataset output order drift")
        logical_roots[name] = merkle_root_v13(leaves)
    feature_leaves = tuple(
        value
        for artifact in ordered
        for value in artifact.feature_values_logical_leaves
    )
    feature_root = merkle_root_v13(feature_leaves)
    physical_root = merkle_root_v13(physical_leaves)
    expected_identity = {
        "engine_id": ENGINE_ID,
        "lineage": lineage,
        "ordered_spec_sha256": [value.digest for value in specs],
        "ordered_fragment_id": [
            value.fragment_id for value in ordered
        ],
        "ordered_build_manifest_sha256": [
            value.build_manifest_sha256 for value in ordered
        ],
        "query_count": len(global_sequence),
        "query_sequence_sha256": _sequence_sha256(global_sequence),
        "model_sha256": {str(key): value for key, value in model_sha.items()},
        "heldout_dataset_merkle_root": heldout_dataset_merkle,
        "logical_merkle_root": logical_roots,
        "feature_values_logical_merkle_root": feature_root,
        "physical_merkle_root": physical_root,
    }
    if dict(identity) != expected_identity:
        raise FoldFusedInvariantError("compact replay dataset identity/Merkle drift")
    return CompactReplayDatasetV13(
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        dataset_id=str(raw["dataset_id"]),
        lineage=lineage,
        specs=specs,
        outputs=outputs,
        query_count=len(global_sequence),
        query_sequence_sha256=_sequence_sha256(global_sequence),
        logical_merkle_root=logical_roots,
        physical_merkle_root=physical_root,
        model_sha256=model_sha,
        build_manifest_sha256=tuple(
            value.build_manifest_sha256 for value in ordered
        ),
        heldout_dataset_merkle_root=heldout_dataset_merkle,
        feature_values_logical_merkle_root=feature_root,
    )


def logical_output_merkle_from_parquet_v13(
    paths: Sequence[Path],
    name: str,
) -> str:
    """Layout-independent digest for v12 merged or v13 fragment outputs.

    For v12 PreRank500 the payload column is intentionally projected away so
    its common fields can be compared directly with compact v13 top500.
    """

    if name not in OUTPUT_SCHEMAS:
        raise KeyError(name)
    comparison_schema = OUTPUT_SCHEMAS[name]
    leaves: list[Mapping[str, Any]] = []
    global_seen: set[str] = set()
    for path in paths:
        parquet = pq.ParquetFile(path)
        available = set(parquet.schema_arrow.names)
        columns = OUTPUT_SCHEMAS[name].names
        missing = set(columns) - available
        if missing:
            raise FoldFusedInvariantError(
                f"{name} comparison columns missing: {sorted(missing)}"
            )
        # Read projected query slices so v12 one-query RG and v13 buffered RG
        # share precisely the same Arrow schema and logical bytes.
        for row_group in range(parquet.num_row_groups):
            row_group_table = parquet.read_row_group(row_group, columns=columns)
            query_values = row_group_table["query_id"].to_pylist()
            start = 0
            while start < len(query_values):
                query_id = str(query_values[start])
                stop = start + 1
                while stop < len(query_values) and str(query_values[stop]) == query_id:
                    stop += 1
                if query_id in global_seen:
                    raise FoldFusedInvariantError("comparison output query split/duplicate")
                global_seen.add(query_id)
                source = row_group_table.slice(start, stop - start)
                table = pa.Table.from_arrays(
                    source.columns, schema=comparison_schema
                )
                fold = int(table["fold"][0].as_py())
                leaves.append(
                    {
                        "kind": name,
                        "query_id": query_id,
                        "fold": fold,
                        "row_count": table.num_rows,
                        "table_sha256": logical_table_sha256_v13(table),
                    }
                )
                start = stop
    return merkle_root_v13(leaves)


def rehydrate_top500_row_v13(
    compact_row: Mapping[str, Any],
    heldout_dataset: BuildFragmentDatasetV13,
) -> Mapping[str, Any]:
    """Resolve the frozen natural key ``(query_id, candidate_siret)``.

    The caller supplies the already verified canonical heldout dataset; its
    Merkle root is sealed once in the replay manifest rather than repeated on
    every PreRank500 row.
    """

    query_id = str(compact_row.get("query_id") or "")
    candidate_siret = str(compact_row.get("candidate_siret") or "")
    matches: list[Mapping[str, Any]] = []
    for table in iter_heldout_query_tables_v13(heldout_dataset):
        if str(table["query_id"][0].as_py()) != query_id:
            continue
        matches.extend(
            row
            for row in table.to_pylist()
            if str(row.get("siret") or "") == candidate_siret
        )
        break
    if (
        len(matches) != 1
        or str(matches[0].get("siren") or "")
        != str(compact_row.get("candidate_parent") or "")
        or int(matches[0].get("fold", -1)) != int(compact_row.get("fold", -1))
    ):
        raise FoldFusedInvariantError("compact PreRank500 rehydrated identity drift")
    return matches[0]


def model_set_logical_sha256_v13(
    models: Mapping[int, FittedSiteModel],
) -> str:
    if set(models) != set(TRAIN_FOLDS):
        raise FoldFusedInvariantError("model digest needs outer 2/3/4")
    return _sha256_json(
        [
            {
                "outer_fold": outer,
                "sha256": models[outer].sha256,
                "recipe_sha256": models[outer].recipe_sha256,
                "feature_names": list(models[outer].feature_names),
            }
            for outer in TRAIN_FOLDS
        ]
    )


__all__ = [
    "BufferedCompactOOFOutputStoreV13",
    "CompactOutputFileV13",
    "CompactParquetOOFOutputStoreV13",
    "CompactReplayDatasetV13",
    "CompactReplayFragmentArtifactV13",
    "DATASET_SCHEMA_VERSION",
    "DEFAULT_ROW_GROUP_TARGET",
    "ENGINE_ID",
    "FRAGMENT_SCHEMA_VERSION",
    "OUTCOME_SCHEMA",
    "TOP100_SCHEMA",
    "TOP500_SCHEMA",
    "aggregate_compact_replay_gates_v13",
    "iter_compact_outcomes_v13",
    "iter_compact_output_query_tables_v13",
    "load_compact_replay_dataset_v13",
    "load_compact_replay_fragment_v13",
    "logical_output_merkle_from_parquet_v13",
    "model_set_logical_sha256_v13",
    "replay_fragment_v13",
    "rehydrate_top500_row_v13",
    "seal_compact_replay_dataset_v13",
]
