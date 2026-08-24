"""Direct PREFIT-to-sharded parity bridge for the bounded v12 benchmark.

The historical PREFIT run already sealed the fused parent, Master, complete
site-feature and fit-sample projections for 512 queries x three outer
contexts.  This module observes those *same* projections inside the sharded
build worker and fails closed on any difference.  It is deliberately optional:
the national OOF path passes no sink and therefore performs no bridge feature
materialisation or sidecar I/O.

The merged bridge Parquet is canonical across W1/W2/W3.  Besides proving the
direct PREFIT link, its ``statistics_sha256`` column makes context-statistic
parity between worker topologies observable without inventing a second
execution engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import hashlib
import json
import os
import shutil
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

from .retrieval_fold_fused_oof_v12 import (
    FeatureBatch,
    FoldFusedInvariantError,
    FullMasterStatistics,
    MasterContext,
    SamplingPlan,
    TRAIN_FOLDS,
    sha256_file,
)


PREFIT_BRIDGE_ROW_SCHEMA_VERSION = (
    "sireto-retrieval-v12-fold-fused-prefit-bridge-row-1"
)
PREFIT_BRIDGE_MANIFEST_SCHEMA_VERSION = (
    "sireto-retrieval-v12-fold-fused-prefit-bridge-manifest-1"
)
PREFIT_BRIDGE_STATUS = "PASS_DIRECT_PREFIT_BRIDGE"
_SHA256 = frozenset("0123456789abcdef")
_REFERENCE_COLUMNS = (
    "query_id",
    "fold",
    "outer_fold",
    "fused_parent_sha256",
    "fused_master_sha256",
    "fused_features_sha256",
    "fused_fit_rows_sha256",
    "master_count",
    "fit_sample_count",
)
_BRIDGE_COLUMNS = (
    "schema_version",
    "query_id",
    "fold",
    "outer_fold",
    "fused_parent_sha256",
    "fused_master_sha256",
    "statistics_sha256",
    "fused_features_sha256",
    "master_count",
    "fused_fit_rows_sha256",
    "fit_sample_count",
)


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(_SHA256)
    )


def _schema_descriptor(schema: pa.Schema) -> list[dict[str, Any]]:
    return [
        {
            "name": field.name,
            "type": str(field.type),
            "nullable": bool(field.nullable),
        }
        for field in schema
    ]


def _bridge_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("schema_version", pa.string(), nullable=False),
            pa.field("query_id", pa.string(), nullable=False),
            pa.field("fold", pa.int8(), nullable=False),
            pa.field("outer_fold", pa.int8(), nullable=False),
            pa.field("fused_parent_sha256", pa.string(), nullable=False),
            pa.field("fused_master_sha256", pa.string(), nullable=False),
            pa.field("statistics_sha256", pa.string(), nullable=False),
            pa.field("fused_features_sha256", pa.string(), nullable=False),
            pa.field("master_count", pa.int32(), nullable=False),
            pa.field("fused_fit_rows_sha256", pa.string(), nullable=False),
            pa.field("fit_sample_count", pa.int16(), nullable=False),
        ]
    )


def _resolve_record_path(root: Path, value: Any) -> Path:
    path = Path(str(value or ""))
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _relative_or_absolute(path: Path, root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(Path(root).resolve()))
    except ValueError:
        return str(resolved)


def _reference_rows_v12(
    prefit_manifest_path: Path,
    *,
    expected_lineage: Mapping[str, str],
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, str]]:
    """Load the strict physical PREFIT evidence and its fused row projection."""

    # Local import keeps the national sink=None path free from the parity
    # harness and its pandas dependency.
    from .retrieval_fold_fused_runtime_v12 import load_prefit_parity_evidence_v12

    manifest_path = Path(prefit_manifest_path).resolve()
    raw, manifest_sha = load_prefit_parity_evidence_v12(
        manifest_path,
        expected_lineage=expected_lineage,
    )
    artifacts = raw.get("artifacts") or {}
    rows_record = artifacts.get("parity_rows") or {}
    reference_record = artifacts.get("reference_manifest") or {}
    rows_path = (manifest_path.parent / "parity_rows.parquet").resolve()
    reference_path = (manifest_path.parent / "reference_manifest.json").resolve()
    if (
        rows_record.get("path") != rows_path.name
        or reference_record.get("path") != reference_path.name
        or not rows_path.is_file()
        or not reference_path.is_file()
        or rows_record.get("sha256") != sha256_file(rows_path)
        or reference_record.get("sha256") != sha256_file(reference_path)
    ):
        raise FoldFusedInvariantError("PREFIT bridge reference artifacts drifted")
    table = pq.read_table(rows_path, columns=list(_REFERENCE_COLUMNS))
    if table.num_rows != 512 * len(TRAIN_FOLDS):
        raise FoldFusedInvariantError("PREFIT bridge reference cardinality drifted")
    output: dict[tuple[str, int], dict[str, Any]] = {}
    folds_by_query: dict[str, int] = {}
    for raw_row in table.to_pylist():
        row = dict(raw_row)
        query_id = str(row.get("query_id") or "")
        fold = int(row.get("fold", -1))
        outer = int(row.get("outer_fold", -1))
        key = (query_id, outer)
        digests = (
            row.get("fused_parent_sha256"),
            row.get("fused_master_sha256"),
            row.get("fused_features_sha256"),
            row.get("fused_fit_rows_sha256"),
        )
        master_count = int(row.get("master_count", -1))
        fit_count = int(row.get("fit_sample_count", -1))
        if (
            not query_id
            or fold not in TRAIN_FOLDS
            or outer not in TRAIN_FOLDS
            or key in output
            or any(not _valid_sha256(value) for value in digests)
            or not 1 <= master_count <= 2_000
            or not 0 <= fit_count <= 128
            or (
                query_id in folds_by_query
                and folds_by_query[query_id] != fold
            )
        ):
            raise FoldFusedInvariantError("PREFIT bridge reference row drifted")
        folds_by_query[query_id] = fold
        row.update(
            {
                "query_id": query_id,
                "fold": fold,
                "outer_fold": outer,
                "master_count": master_count,
                "fit_sample_count": fit_count,
            }
        )
        output[key] = row
    if (
        len(folds_by_query) != 512
        or any(
            {(query_id, outer) for outer in TRAIN_FOLDS}
            - set(output)
            for query_id in folds_by_query
        )
    ):
        raise FoldFusedInvariantError("PREFIT bridge reference coverage drifted")
    return output, {
        "prefit_manifest_sha256": manifest_sha,
        "reference_manifest_sha256": sha256_file(reference_path),
        "parity_rows_sha256": sha256_file(rows_path),
    }


def _observed_label_blind_row_v12(
    context: MasterContext,
    statistics: FullMasterStatistics,
    features: FeatureBatch,
) -> dict[str, Any]:
    # Import the exact historical digest functions rather than reimplementing
    # their float32/canonical-JSON semantics in a second engine.
    from .retrieval_fold_fused_parity_v12 import (
        _digest,
        _feature_digest,
        _master_projection,
        _parent_digest,
    )

    opaque = context.opaque
    rows = getattr(opaque, "raw_rows", None)
    if not isinstance(rows, (list, tuple)) or len(rows) != len(
        context.candidate_sirets
    ):
        raise FoldFusedInvariantError("PREFIT bridge cannot inspect complete Master rows")
    return {
        "schema_version": PREFIT_BRIDGE_ROW_SCHEMA_VERSION,
        "query_id": context.query_id,
        "fold": int(context.scene_fold),
        "outer_fold": int(context.outer_fold),
        "fused_parent_sha256": _parent_digest(rows),
        "fused_master_sha256": _digest(_master_projection(rows)),
        "statistics_sha256": statistics.statistics_sha256,
        "fused_features_sha256": _feature_digest(features),
        "master_count": len(context.candidate_sirets),
    }


def _fit_projection_v12(
    context: MasterContext,
    plan: SamplingPlan,
) -> tuple[str, int]:
    from .retrieval_fold_fused_parity_v12 import _digest

    sirets = tuple(
        context.candidate_sirets[int(index)] for index in plan.candidate_indices
    )
    return _digest(sirets), len(sirets)


@dataclass(frozen=True)
class PrefitBridgeArtifactV12:
    path: Path
    manifest_path: Path
    sha256: str
    manifest_sha256: str
    row_count: int
    query_count: int
    size_bytes: int
    schema_sha256: str
    query_sequence_sha256: str
    lineage: Mapping[str, str]
    prefit_manifest_sha256: str
    reference_manifest_sha256: str
    parity_rows_sha256: str


def prefit_bridge_record_v12(
    artifact: PrefitBridgeArtifactV12,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    base = Path(root).resolve() if root is not None else None
    return {
        "path": (
            _relative_or_absolute(artifact.path, base)
            if base is not None
            else str(artifact.path)
        ),
        "manifest_path": (
            _relative_or_absolute(artifact.manifest_path, base)
            if base is not None
            else str(artifact.manifest_path)
        ),
        "sha256": artifact.sha256,
        "manifest_sha256": artifact.manifest_sha256,
        "rows": int(artifact.row_count),
        "queries": int(artifact.query_count),
        "size_bytes": int(artifact.size_bytes),
        "schema_sha256": artifact.schema_sha256,
        "query_sequence_sha256": artifact.query_sequence_sha256,
        "lineage": dict(artifact.lineage),
        "prefit_manifest_sha256": artifact.prefit_manifest_sha256,
        "reference_manifest_sha256": artifact.reference_manifest_sha256,
        "parity_rows_sha256": artifact.parity_rows_sha256,
    }


def _write_bridge_artifact_v12(
    rows: Sequence[Mapping[str, Any]],
    *,
    query_sequence: Sequence[tuple[str, int]],
    lineage: Mapping[str, str],
    reference_hashes: Mapping[str, str],
    output_dir: Path,
) -> PrefitBridgeArtifactV12:
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    if not query_sequence:
        raise FoldFusedInvariantError("PREFIT bridge cannot seal an empty query set")
    lineage_value = {str(key): str(value) for key, value in lineage.items()}
    if not lineage_value or any(
        not _valid_sha256(value) for value in lineage_value.values()
    ):
        raise FoldFusedInvariantError("PREFIT bridge lineage is not SHA-256 sealed")
    hashes = {
        key: str(reference_hashes.get(key) or "")
        for key in (
            "prefit_manifest_sha256",
            "reference_manifest_sha256",
            "parity_rows_sha256",
        )
    }
    if any(not _valid_sha256(value) for value in hashes.values()):
        raise FoldFusedInvariantError("PREFIT bridge reference hashes are incomplete")
    expected_keys = tuple(
        (str(query_id), int(outer))
        for query_id, _fold in query_sequence
        for outer in TRAIN_FOLDS
    )
    by_key = {
        (str(row.get("query_id") or ""), int(row.get("outer_fold", -1))): dict(row)
        for row in rows
    }
    if len(by_key) != len(rows) or set(by_key) != set(expected_keys):
        raise FoldFusedInvariantError("PREFIT bridge row coverage/order drifted")
    ordered_rows = [by_key[key] for key in expected_keys]
    expected_folds = {str(query_id): int(fold) for query_id, fold in query_sequence}
    for row in ordered_rows:
        if (
            tuple(row) != _BRIDGE_COLUMNS
            or row["schema_version"] != PREFIT_BRIDGE_ROW_SCHEMA_VERSION
            or int(row["fold"]) != expected_folds[str(row["query_id"])]
            or int(row["outer_fold"]) not in TRAIN_FOLDS
            or any(
                not _valid_sha256(row[name])
                for name in (
                    "fused_parent_sha256",
                    "fused_master_sha256",
                    "statistics_sha256",
                    "fused_features_sha256",
                    "fused_fit_rows_sha256",
                )
            )
            or not 1 <= int(row["master_count"]) <= 2_000
            or not 0 <= int(row["fit_sample_count"]) <= 128
        ):
            raise FoldFusedInvariantError("PREFIT bridge observed row schema drifted")

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        schema = _bridge_schema()
        table = pa.Table.from_pylist(ordered_rows, schema=schema)
        rows_path = stage / "prefit_bridge_rows.parquet"
        pq.write_table(
            table,
            rows_path,
            compression="zstd",
            use_dictionary=False,
            write_statistics=True,
            row_group_size=table.num_rows,
        )
        schema_sha = _sha256_json(_schema_descriptor(schema))
        sequence_sha = _sha256_json(
            [
                [str(query_id), int(fold), int(outer)]
                for query_id, fold in query_sequence
                for outer in TRAIN_FOLDS
            ]
        )
        identity = {
            "artifact_sha256": sha256_file(rows_path),
            "row_count": table.num_rows,
            "query_count": len(query_sequence),
            "schema_sha256": schema_sha,
            "query_sequence_sha256": sequence_sha,
            "lineage": lineage_value,
            "reference": hashes,
        }
        manifest = {
            "schema_version": PREFIT_BRIDGE_MANIFEST_SCHEMA_VERSION,
            "status": PREFIT_BRIDGE_STATUS,
            "phase": "PHYSICAL_PARITY_BENCHMARK_512_ONLY",
            "bridge_id": _sha256_json(identity),
            "bridge_identity": identity,
            "folds": list(TRAIN_FOLDS),
            "fold0_opened": False,
            "fold1_opened": False,
            "positive_injection": False,
            "label_columns": [],
            "query_count": len(query_sequence),
            "context_count": table.num_rows,
            "schema": _schema_descriptor(schema),
            "projection": list(_BRIDGE_COLUMNS),
            "lineage": lineage_value,
            "reference": hashes,
            "checks": {
                "parent_matches_prefit_fused": True,
                "master_matches_prefit_fused": True,
                "full_site_features_match_prefit_fused": True,
                "fit_rows_match_prefit_fused": True,
            },
            "artifact": {
                "path": rows_path.name,
                "sha256": identity["artifact_sha256"],
                "rows": table.num_rows,
                "size_bytes": rows_path.stat().st_size,
            },
        }
        manifest_path = stage / "manifest.json"
        manifest_path.write_bytes(_canonical_json(manifest))
        os.rename(stage, output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return load_prefit_bridge_artifact_v12(
        {
            "path": "prefit_bridge_rows.parquet",
            "manifest_path": "manifest.json",
            "sha256": identity["artifact_sha256"],
            "manifest_sha256": sha256_file(output / "manifest.json"),
            "rows": len(ordered_rows),
            "queries": len(query_sequence),
            "size_bytes": (output / "prefit_bridge_rows.parquet").stat().st_size,
            "schema_sha256": schema_sha,
            "query_sequence_sha256": sequence_sha,
            "lineage": lineage_value,
            **hashes,
        },
        root=output,
        verify_hashes=True,
    )


class PrefitBridgeSinkV12:
    """One benchmark-build observer; methods never return candidate data."""

    def __init__(
        self,
        *,
        expected_rows: Mapping[tuple[str, int], Mapping[str, Any]],
        expected_query_sequence: Sequence[tuple[str, int]],
        lineage: Mapping[str, str],
        reference_hashes: Mapping[str, str],
    ) -> None:
        self._expected = {
            (str(query_id), int(outer)): dict(row)
            for (query_id, outer), row in expected_rows.items()
        }
        self._query_sequence = tuple(
            (str(query_id), int(fold)) for query_id, fold in expected_query_sequence
        )
        self._lineage = {str(key): str(value) for key, value in lineage.items()}
        self._reference_hashes = dict(reference_hashes)
        wanted = {
            (query_id, outer)
            for query_id, _fold in self._query_sequence
            for outer in TRAIN_FOLDS
        }
        if set(self._expected) != wanted:
            raise FoldFusedInvariantError("PREFIT bridge sink subset coverage drifted")
        self._observed: dict[tuple[str, int], dict[str, Any]] = {}

    @classmethod
    def from_prefit_manifest(
        cls,
        prefit_manifest_path: Path,
        *,
        expected_prefit_lineage: Mapping[str, str],
        bridge_lineage: Mapping[str, str],
        expected_query_ids: Sequence[str],
        expected_query_folds: Sequence[int],
        expected_prefit_manifest_sha256: str,
    ) -> "PrefitBridgeSinkV12":
        if len(expected_query_ids) != len(expected_query_folds):
            raise FoldFusedInvariantError("PREFIT bridge query/fold sequence drifted")
        reference, hashes = _reference_rows_v12(
            prefit_manifest_path,
            expected_lineage=expected_prefit_lineage,
        )
        if hashes["prefit_manifest_sha256"] != expected_prefit_manifest_sha256:
            raise FoldFusedInvariantError("PREFIT bridge manifest hash drifted")
        sequence = tuple(
            (str(query_id), int(fold))
            for query_id, fold in zip(
                expected_query_ids, expected_query_folds, strict=True
            )
        )
        subset: dict[tuple[str, int], Mapping[str, Any]] = {}
        for query_id, fold in sequence:
            if fold not in TRAIN_FOLDS:
                raise FoldFusedInvariantError("PREFIT bridge attempted fold 0/1")
            for outer in TRAIN_FOLDS:
                row = reference.get((query_id, outer))
                if row is None or int(row["fold"]) != fold:
                    raise FoldFusedInvariantError(
                        "PREFIT bridge sample differs from historical PREFIT"
                    )
                subset[(query_id, outer)] = row
        return cls(
            expected_rows=subset,
            expected_query_sequence=sequence,
            lineage=bridge_lineage,
            reference_hashes=hashes,
        )

    def observe_label_blind(
        self,
        context: MasterContext,
        statistics: FullMasterStatistics,
        features: FeatureBatch,
    ) -> None:
        key = (context.query_id, int(context.outer_fold))
        if key in self._observed or key not in self._expected:
            raise FoldFusedInvariantError("PREFIT bridge label-blind identity drifted")
        observed = _observed_label_blind_row_v12(context, statistics, features)
        expected = self._expected[key]
        comparisons = {
            "fused_parent_sha256": observed["fused_parent_sha256"],
            "fused_master_sha256": observed["fused_master_sha256"],
            "fused_features_sha256": observed["fused_features_sha256"],
            "master_count": observed["master_count"],
        }
        for name, value in comparisons.items():
            if value != expected[name]:
                raise FoldFusedInvariantError(
                    f"PREFIT bridge divergence {name} query={key[0]} outer={key[1]}"
                )
        self._observed[key] = observed

    def observe_fit(self, context: MasterContext, plan: SamplingPlan) -> None:
        key = (context.query_id, int(context.outer_fold))
        row = self._observed.get(key)
        expected = self._expected.get(key)
        if row is None or expected is None or "fused_fit_rows_sha256" in row:
            raise FoldFusedInvariantError("PREFIT bridge fit identity/order drifted")
        if plan.statistics_sha256 != row["statistics_sha256"]:
            raise FoldFusedInvariantError("PREFIT bridge fit statistics drifted")
        digest, count = _fit_projection_v12(context, plan)
        if (
            digest != expected["fused_fit_rows_sha256"]
            or count != expected["fit_sample_count"]
        ):
            raise FoldFusedInvariantError(
                f"PREFIT bridge divergence fit rows query={key[0]} outer={key[1]}"
            )
        row["fused_fit_rows_sha256"] = digest
        row["fit_sample_count"] = count

    def seal(self, output_dir: Path) -> PrefitBridgeArtifactV12:
        if set(self._observed) != set(self._expected) or any(
            "fused_fit_rows_sha256" not in row
            for row in self._observed.values()
        ):
            raise FoldFusedInvariantError("PREFIT bridge did not observe every context")
        artifact = _write_bridge_artifact_v12(
            tuple(self._observed.values()),
            query_sequence=self._query_sequence,
            lineage=self._lineage,
            reference_hashes=self._reference_hashes,
            output_dir=output_dir,
        )
        validate_prefit_bridge_against_reference_v12(
            artifact,
            prefit_manifest_path=None,
            expected_rows=self._expected,
        )
        return artifact


def load_prefit_bridge_artifact_v12(
    record: Mapping[str, Any],
    *,
    root: Path | None = None,
    verify_hashes: bool = True,
) -> PrefitBridgeArtifactV12:
    base = Path(root).resolve() if root is not None else Path.cwd().resolve()
    path = _resolve_record_path(base, record.get("path"))
    manifest_path = _resolve_record_path(base, record.get("manifest_path"))
    sha = str(record.get("sha256") or "")
    manifest_sha = str(record.get("manifest_sha256") or "")
    rows = int(record.get("rows", -1))
    queries = int(record.get("queries", -1))
    size_bytes = int(record.get("size_bytes", -1))
    schema_sha = str(record.get("schema_sha256") or "")
    sequence_sha = str(record.get("query_sequence_sha256") or "")
    lineage = {str(key): str(value) for key, value in dict(record.get("lineage") or {}).items()}
    hashes = {
        key: str(record.get(key) or "")
        for key in (
            "prefit_manifest_sha256",
            "reference_manifest_sha256",
            "parity_rows_sha256",
        )
    }
    if (
        not path.is_file()
        or not manifest_path.is_file()
        or path.name != "prefit_bridge_rows.parquet"
        or manifest_path.name != "manifest.json"
        or any(
            not _valid_sha256(value)
            for value in (
                sha,
                manifest_sha,
                schema_sha,
                sequence_sha,
                *hashes.values(),
                *lineage.values(),
            )
        )
        or not lineage
        or rows != queries * len(TRAIN_FOLDS)
        or queries <= 0
        or size_bytes <= 0
        or path.stat().st_size != size_bytes
        or (verify_hashes and sha256_file(path) != sha)
        or (verify_hashes and sha256_file(manifest_path) != manifest_sha)
    ):
        raise FoldFusedInvariantError("PREFIT bridge artifact record is corrupt")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = raw.get("bridge_identity")
    artifact_record = raw.get("artifact") or {}
    required = {
        "schema_version": PREFIT_BRIDGE_MANIFEST_SCHEMA_VERSION,
        "status": PREFIT_BRIDGE_STATUS,
        "phase": "PHYSICAL_PARITY_BENCHMARK_512_ONLY",
        "folds": list(TRAIN_FOLDS),
        "fold0_opened": False,
        "fold1_opened": False,
        "positive_injection": False,
        "label_columns": [],
        "query_count": queries,
        "context_count": rows,
        "projection": list(_BRIDGE_COLUMNS),
        "lineage": lineage,
        "reference": hashes,
        "checks": {
            "parent_matches_prefit_fused": True,
            "master_matches_prefit_fused": True,
            "full_site_features_match_prefit_fused": True,
            "fit_rows_match_prefit_fused": True,
        },
    }
    if (
        any(raw.get(key) != value for key, value in required.items())
        or not isinstance(identity, Mapping)
        or raw.get("bridge_id") != _sha256_json(identity)
        or identity
        != {
            "artifact_sha256": sha,
            "row_count": rows,
            "query_count": queries,
            "schema_sha256": schema_sha,
            "query_sequence_sha256": sequence_sha,
            "lineage": lineage,
            "reference": hashes,
        }
        or artifact_record
        != {
            "path": path.name,
            "sha256": sha,
            "rows": rows,
            "size_bytes": size_bytes,
        }
    ):
        raise FoldFusedInvariantError("PREFIT bridge manifest drifted")
    parquet = pq.ParquetFile(path)
    schema = _bridge_schema()
    table = parquet.read()
    if (
        parquet.metadata.num_rows != rows
        or not table.schema.equals(schema, check_metadata=True)
        or raw.get("schema") != _schema_descriptor(schema)
        or _sha256_json(_schema_descriptor(schema)) != schema_sha
        or table.column_names != list(_BRIDGE_COLUMNS)
    ):
        raise FoldFusedInvariantError("PREFIT bridge Parquet schema/cardinality drifted")
    seen: set[tuple[str, int]] = set()
    sequence: list[list[Any]] = []
    previous_query: str | None = None
    previous_fold: int | None = None
    expected_outer_index = 0
    for row in table.to_pylist():
        query_id = str(row["query_id"])
        fold = int(row["fold"])
        outer = int(row["outer_fold"])
        key = (query_id, outer)
        if (
            row["schema_version"] != PREFIT_BRIDGE_ROW_SCHEMA_VERSION
            or fold not in TRAIN_FOLDS
            or outer != TRAIN_FOLDS[expected_outer_index]
            or key in seen
            or any(
                not _valid_sha256(row[name])
                for name in (
                    "fused_parent_sha256",
                    "fused_master_sha256",
                    "statistics_sha256",
                    "fused_features_sha256",
                    "fused_fit_rows_sha256",
                )
            )
            or not 1 <= int(row["master_count"]) <= 2_000
            or not 0 <= int(row["fit_sample_count"]) <= 128
        ):
            raise FoldFusedInvariantError("PREFIT bridge row content/order drifted")
        if expected_outer_index == 0:
            if query_id == previous_query:
                raise FoldFusedInvariantError("PREFIT bridge query groups overlap")
            previous_query, previous_fold = query_id, fold
        elif query_id != previous_query or fold != previous_fold:
            raise FoldFusedInvariantError("PREFIT bridge query group split")
        seen.add(key)
        sequence.append([query_id, fold, outer])
        expected_outer_index = (expected_outer_index + 1) % len(TRAIN_FOLDS)
    if expected_outer_index != 0 or _sha256_json(sequence) != sequence_sha:
        raise FoldFusedInvariantError("PREFIT bridge sequence digest drifted")
    return PrefitBridgeArtifactV12(
        path=path,
        manifest_path=manifest_path,
        sha256=sha,
        manifest_sha256=manifest_sha,
        row_count=rows,
        query_count=queries,
        size_bytes=size_bytes,
        schema_sha256=schema_sha,
        query_sequence_sha256=sequence_sha,
        lineage=lineage,
        prefit_manifest_sha256=hashes["prefit_manifest_sha256"],
        reference_manifest_sha256=hashes["reference_manifest_sha256"],
        parity_rows_sha256=hashes["parity_rows_sha256"],
    )


def validate_prefit_bridge_against_reference_v12(
    artifact: PrefitBridgeArtifactV12,
    *,
    prefit_manifest_path: Path | None,
    expected_lineage: Mapping[str, str] | None = None,
    expected_rows: Mapping[tuple[str, int], Mapping[str, Any]] | None = None,
) -> None:
    """Recompute the direct join; authorization can call this independently."""

    if (prefit_manifest_path is None) == (expected_rows is None):
        raise FoldFusedInvariantError(
            "PREFIT bridge validation requires exactly one reference authority"
        )
    if prefit_manifest_path is not None:
        reference, hashes = _reference_rows_v12(
            prefit_manifest_path,
            expected_lineage=expected_lineage or artifact.lineage,
        )
        if hashes != {
            "prefit_manifest_sha256": artifact.prefit_manifest_sha256,
            "reference_manifest_sha256": artifact.reference_manifest_sha256,
            "parity_rows_sha256": artifact.parity_rows_sha256,
        }:
            raise FoldFusedInvariantError("PREFIT bridge reference hash binding drifted")
    else:
        reference = {
            (str(query_id), int(outer)): dict(row)
            for (query_id, outer), row in (expected_rows or {}).items()
        }
    table = pq.read_table(artifact.path)
    for row in table.to_pylist():
        key = (str(row["query_id"]), int(row["outer_fold"]))
        expected = reference.get(key)
        if expected is None or int(expected["fold"]) != int(row["fold"]):
            raise FoldFusedInvariantError("PREFIT bridge reference population drifted")
        for name in (
            "fused_parent_sha256",
            "fused_master_sha256",
            "fused_features_sha256",
            "master_count",
            "fused_fit_rows_sha256",
            "fit_sample_count",
        ):
            if row[name] != expected[name]:
                raise FoldFusedInvariantError(
                    f"PREFIT bridge direct comparison failed: {name}"
                )


def merge_prefit_bridge_artifacts_v12(
    artifacts: Sequence[PrefitBridgeArtifactV12],
    *,
    expected_query_sequence: Sequence[tuple[str, int]],
    output_dir: Path,
) -> PrefitBridgeArtifactV12:
    """Canonicalise one or three bridge shards into identical W1/W3 bytes."""

    values = tuple(artifacts)
    if not values:
        raise FoldFusedInvariantError("no PREFIT bridge shards supplied")
    first = values[0]
    rows: list[dict[str, Any]] = []
    for artifact in values:
        loaded = load_prefit_bridge_artifact_v12(
            prefit_bridge_record_v12(artifact), verify_hashes=True
        )
        if (
            dict(loaded.lineage) != dict(first.lineage)
            or loaded.prefit_manifest_sha256 != first.prefit_manifest_sha256
            or loaded.reference_manifest_sha256 != first.reference_manifest_sha256
            or loaded.parity_rows_sha256 != first.parity_rows_sha256
        ):
            raise FoldFusedInvariantError("PREFIT bridge shard authority drifted")
        rows.extend(pq.read_table(loaded.path).to_pylist())
    return _write_bridge_artifact_v12(
        rows,
        query_sequence=expected_query_sequence,
        lineage=first.lineage,
        reference_hashes={
            "prefit_manifest_sha256": first.prefit_manifest_sha256,
            "reference_manifest_sha256": first.reference_manifest_sha256,
            "parity_rows_sha256": first.parity_rows_sha256,
        },
        output_dir=output_dir,
    )


__all__ = [
    "PREFIT_BRIDGE_MANIFEST_SCHEMA_VERSION",
    "PREFIT_BRIDGE_ROW_SCHEMA_VERSION",
    "PrefitBridgeArtifactV12",
    "PrefitBridgeSinkV12",
    "load_prefit_bridge_artifact_v12",
    "merge_prefit_bridge_artifacts_v12",
    "prefit_bridge_record_v12",
    "validate_prefit_bridge_against_reference_v12",
]
