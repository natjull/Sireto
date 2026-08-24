"""External single-pass held-out Master spool for retrieval v12.

Only label-blind columns required to reconstruct the frozen 319 site features
and their official evidence payload are persisted.  The spool is written while
the two training contexts are sampled, sealed before model fitting, and replayed
after fit without reopening SceneIR, the national index, or official sidecars.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from .retrieval_fold_fused_oof_v12 import (
    FoldFusedInvariantError,
    HeldoutMasterArtifact,
    MasterContext,
    TRAIN_FOLDS,
    sha256_file,
)
from .retrieval_hierarchical_admission_v12 import (
    AdmissionConfigV12,
    CASCADE_PARENT_FEATURES,
    LABEL_COLUMNS,
)
from .retrieval_top100_cascade_v12 import is_forbidden_input_key


SCHEMA_VERSION = "sireto-retrieval-v12-heldout-master-spool-1"
MANIFEST_VERSION = "sireto-retrieval-v12-heldout-master-spool-manifest-1"
CALIBRATION_VERSION = "sireto-retrieval-v12-heldout-master-spool-calibration-1"
DEFAULT_MAXIMUM_BYTES = 20 * 1024**3
DEFAULT_ROW_GROUP_TARGET = 16_384

_STRING_LISTS = (
    "lanes",
    "selection_reasons",
    "channels",
    "sources",
    "official_evidence_ids",
    "official_evidence_sources",
    "candidate_names",
    "candidate_addresses",
)
_STRING_FIELDS = (
    "query_id",
    "siret",
    "siren",
    "selection_lane",
    "candidate_insee",
    "candidate_postcode",
    "candidate_number",
    "candidate_state",
    "temporal_valid_from",
    "temporal_valid_to",
    "crm_name",
    "crm_address",
    "crm_number",
    "crm_postcode",
    "crm_insee",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sequence_sha256(values: Sequence[tuple[str, int]]) -> str:
    return _sha256_json([[query_id, int(fold)] for query_id, fold in values])


def _schema_descriptor(schema: pa.Schema) -> list[dict[str, Any]]:
    return [
        {
            "name": field.name,
            "type": str(field.type),
            "nullable": bool(field.nullable),
        }
        for field in schema
    ]


def heldout_master_schema_v12(config: AdmissionConfigV12) -> pa.Schema:
    signal = pa.struct(
        [
            pa.field("source", pa.string()),
            pa.field("channel", pa.string()),
            pa.field("rank", pa.int64()),
            pa.field("score", pa.float64()),
            pa.field("parent_identifier", pa.string()),
            pa.field("parent_rank", pa.int64()),
            pa.field("parent_support_count", pa.int64()),
            pa.field("site_rank", pa.int64()),
            pa.field("relation_direction", pa.string()),
            pa.field("relation_date", pa.string()),
        ]
    )
    fields: list[pa.Field] = [
        pa.field("query_id", pa.string(), nullable=False),
        pa.field("fold", pa.int8(), nullable=False),
        pa.field("siret", pa.string(), nullable=False),
        pa.field("siren", pa.string(), nullable=False),
        pa.field("union_rank", pa.int32(), nullable=False),
        pa.field("master_rank", pa.int32(), nullable=False),
        pa.field("hydration_rank", pa.int32()),
        pa.field("parent_rank", pa.int32()),
        pa.field("site_rank", pa.int32()),
        pa.field("parent_score", pa.float64()),
        pa.field("parent_feature_names", pa.list_(pa.string()), nullable=False),
        pa.field(
            "parent_feature_values",
            pa.list_(pa.float64(), len(CASCADE_PARENT_FEATURES)),
            nullable=False,
        ),
        pa.field("site_score", pa.float64()),
        pa.field("selection_lane", pa.string()),
        pa.field("lanes", pa.list_(pa.string()), nullable=False),
        pa.field("selection_reasons", pa.list_(pa.string()), nullable=False),
        pa.field("channels", pa.list_(pa.string()), nullable=False),
        pa.field("sources", pa.list_(pa.string()), nullable=False),
        pa.field("signals", pa.list_(signal), nullable=False),
        pa.field("official_evidence_ids", pa.list_(pa.string()), nullable=False),
        pa.field("official_evidence_sources", pa.list_(pa.string()), nullable=False),
        pa.field("candidate_names", pa.list_(pa.string()), nullable=False),
        pa.field("candidate_addresses", pa.list_(pa.string()), nullable=False),
        pa.field("candidate_insee", pa.string()),
        pa.field("candidate_postcode", pa.string()),
        pa.field("candidate_number", pa.string()),
        pa.field("candidate_state", pa.string()),
        pa.field("candidate_is_headquarters", pa.bool_()),
        pa.field("temporal_valid_from", pa.string()),
        pa.field("temporal_valid_to", pa.string()),
        pa.field("crm_name", pa.string()),
        pa.field("crm_address", pa.string()),
        pa.field("crm_number", pa.string()),
        pa.field("crm_postcode", pa.string()),
        pa.field("crm_insee", pa.string()),
    ]
    for channel in config.channels:
        fields.append(pa.field(f"{channel}_rank", pa.float64()))
        fields.append(pa.field(f"{channel}_score", pa.float64()))
    schema = pa.schema(fields)
    descriptor = _schema_descriptor(schema)
    metadata = {
        b"schema_version": SCHEMA_VERSION.encode("ascii"),
        b"projection_json": _canonical_json(schema.names).encode("utf-8"),
        b"schema_sha256": _sha256_json(descriptor).encode("ascii"),
    }
    return schema.with_metadata(metadata)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return [value] if value else []
        value = decoded if isinstance(decoded, list) else [value]
    if not isinstance(value, (list, tuple, set)):
        return [str(value)] if str(value) else []
    return [str(item) for item in value if str(item)]


def _signals(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, (list, tuple)):
        return []
    output: list[dict[str, Any]] = []
    for raw in value:
        if not isinstance(raw, Mapping):
            continue
        relation_date = (
            raw.get("relation_date")
            or raw.get("event_date")
            or raw.get("date")
            or ""
        )
        output.append(
            {
                "source": str(raw.get("source") or ""),
                "channel": str(raw.get("channel") or ""),
                "rank": (
                    int(raw["rank"])
                    if raw.get("rank") not in (None, "")
                    else None
                ),
                "score": (
                    float(raw["score"])
                    if raw.get("score") not in (None, "")
                    else None
                ),
                "parent_identifier": str(raw.get("parent_identifier") or ""),
                "parent_rank": (
                    int(raw["parent_rank"])
                    if raw.get("parent_rank") not in (None, "")
                    else None
                ),
                "parent_support_count": (
                    int(raw["parent_support_count"])
                    if raw.get("parent_support_count") not in (None, "")
                    else None
                ),
                "site_rank": (
                    int(raw["site_rank"])
                    if raw.get("site_rank") not in (None, "")
                    else None
                ),
                "relation_direction": str(raw.get("relation_direction") or ""),
                "relation_date": str(relation_date),
            }
        )
    return output


def _project_row(
    raw: Mapping[str, Any], config: AdmissionConfigV12
) -> dict[str, Any]:
    forbidden = {
        str(key)
        for key in raw
        if str(key) in LABEL_COLUMNS or is_forbidden_input_key(str(key))
    }
    if forbidden:
        raise FoldFusedInvariantError(
            f"heldout spool received forbidden label columns: {sorted(forbidden)}"
        )
    required = {
        "query_id",
        "fold",
        "siret",
        "siren",
        "master_rank",
        "crm_name",
        "crm_address",
        "parent_feature_names",
        "parent_feature_values",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise FoldFusedInvariantError(
            f"heldout Master lacks spool projection columns: {missing}"
        )
    projected: dict[str, Any] = {
        "query_id": str(raw["query_id"]),
        "fold": int(raw["fold"]),
        "siret": str(raw["siret"]),
        "siren": str(raw["siren"]),
        "union_rank": int(raw.get("union_rank") or raw["master_rank"]),
        "master_rank": int(raw["master_rank"]),
        "hydration_rank": (
            int(raw["hydration_rank"])
            if raw.get("hydration_rank") not in (None, "")
            else None
        ),
        "parent_rank": (
            int(raw["parent_rank"])
            if raw.get("parent_rank") not in (None, "")
            else None
        ),
        "site_rank": (
            int(raw["site_rank"])
            if raw.get("site_rank") not in (None, "")
            else None
        ),
        "parent_score": (
            float(raw["parent_score"])
            if raw.get("parent_score") not in (None, "")
            else None
        ),
        "parent_feature_names": _string_list(raw.get("parent_feature_names")),
        "parent_feature_values": [
            float(value) for value in (raw.get("parent_feature_values") or [])
        ],
        "site_score": (
            float(raw["site_score"])
            if raw.get("site_score") not in (None, "")
            else None
        ),
        "signals": _signals(raw.get("signals")),
        "candidate_is_headquarters": bool(
            raw.get("candidate_is_headquarters", raw.get("candidate_is_siege", False))
        ),
    }
    for name in _STRING_LISTS:
        projected[name] = _string_list(raw.get(name))
    for name in _STRING_FIELDS:
        if name in projected:
            continue
        value = raw.get(name)
        projected[name] = "" if value is None else str(value)
    for channel in config.channels:
        for suffix in ("rank", "score"):
            name = f"{channel}_{suffix}"
            value = raw.get(name)
            projected[name] = (
                float(value) if value not in (None, "") else None
            )
    return projected


class ParquetHeldoutMasterSpoolV12:
    """Write/replay one held-out label-blind Master per query."""

    def __init__(
        self,
        output_dir: Path,
        *,
        config: AdmissionConfigV12,
        payload_factory: Callable[[tuple[Mapping[str, Any], ...]], Any] | None = None,
        maximum_bytes: int = DEFAULT_MAXIMUM_BYTES,
        row_group_target: int = DEFAULT_ROW_GROUP_TARGET,
    ) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.path = self.output_dir / "heldout_master.parquet"
        self.manifest_path = self.output_dir / "manifest.json"
        self.config = config
        if payload_factory is None:
            # Deferred import avoids a module cycle while keeping the public
            # spool runnable from the production CLI.
            from .retrieval_fold_fused_runtime_v12 import _MasterPayload

            payload_factory = _MasterPayload
        self.payload_factory = payload_factory
        self.maximum_bytes = int(maximum_bytes)
        self.row_group_target = int(row_group_target)
        if self.maximum_bytes <= 0 or self.row_group_target < 2_000:
            raise FoldFusedInvariantError("heldout spool resource bounds are invalid")
        self.schema = heldout_master_schema_v12(config)
        self.schema_sha256 = str(self.schema.metadata[b"schema_sha256"], "ascii")
        self._writer = pq.ParquetWriter(
            self.path,
            self.schema,
            compression="zstd",
            compression_level=3,
            use_dictionary=True,
            write_statistics=True,
        )
        self._buffer: list[pa.Table] = []
        self._buffer_rows = 0
        self._rows = 0
        self._sequence: list[tuple[str, int]] = []
        self._seen: set[str] = set()
        self._sealed: HeldoutMasterArtifact | None = None

    def _flush(self) -> None:
        if not self._buffer:
            return
        table = pa.concat_tables(self._buffer)
        self._writer.write_table(table, row_group_size=table.num_rows)
        self._buffer.clear()
        self._buffer_rows = 0

    def append(self, context: MasterContext) -> None:
        if self._sealed is not None:
            raise FoldFusedInvariantError("heldout spool is already sealed")
        if context.outer_fold != context.scene_fold or context.scene_fold not in TRAIN_FOLDS:
            raise FoldFusedInvariantError("spool accepts only heldout folds 2/3/4")
        if context.query_id in self._seen:
            raise FoldFusedInvariantError("heldout query was spooled twice")
        raw_rows = getattr(context.opaque, "raw_rows", None)
        if not isinstance(raw_rows, tuple) or len(raw_rows) != len(
            context.candidate_sirets
        ):
            raise FoldFusedInvariantError("heldout context lacks serializable Master rows")
        projected = [_project_row(row, self.config) for row in raw_rows]
        if tuple(row["siret"] for row in projected) != context.candidate_sirets:
            raise FoldFusedInvariantError("heldout spool candidate order drift")
        if tuple(row["siren"] for row in projected) != context.candidate_parents:
            raise FoldFusedInvariantError("heldout spool parent order drift")
        expected_ranks = tuple(range(1, len(projected) + 1))
        if tuple(row["master_rank"] for row in projected) != expected_ranks:
            raise FoldFusedInvariantError("heldout Master ranks are not contiguous")
        table = pa.Table.from_pylist(projected, schema=self.schema)
        if self._buffer and self._buffer_rows + table.num_rows > self.row_group_target:
            self._flush()
        self._buffer.append(table)
        self._buffer_rows += table.num_rows
        self._rows += table.num_rows
        self._sequence.append((context.query_id, context.scene_fold))
        self._seen.add(context.query_id)

    def seal(self) -> HeldoutMasterArtifact:
        if self._sealed is not None:
            return self._sealed
        if not self._sequence or self._rows < len(self._sequence):
            raise FoldFusedInvariantError("heldout spool is empty")
        self._flush()
        self._writer.close()
        size_bytes = self.path.stat().st_size
        if size_bytes > self.maximum_bytes:
            raise FoldFusedInvariantError(
                "heldout spool exceeds the frozen 20 GiB temporary budget"
            )
        parquet_sha = sha256_file(self.path)
        sequence_sha = _sequence_sha256(self._sequence)
        manifest = {
            "schema_version": MANIFEST_VERSION,
            "status": "SEALED_LABEL_BLIND",
            "folds": list(TRAIN_FOLDS),
            "fold0_opened": False,
            "fold1_opened": False,
            "positive_injection": False,
            "label_columns": [],
            "projection": self.schema.names,
            "schema": _schema_descriptor(self.schema.remove_metadata()),
            "schema_sha256": self.schema_sha256,
            "query_count": len(self._sequence),
            "row_count": self._rows,
            "query_sequence_sha256": sequence_sha,
            "maximum_bytes": self.maximum_bytes,
            "artifact": {
                "path": self.path.name,
                "sha256": parquet_sha,
                "size_bytes": size_bytes,
            },
        }
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.manifest_path)
        self._sealed = HeldoutMasterArtifact(
            path=self.path,
            manifest_path=self.manifest_path,
            sha256=parquet_sha,
            manifest_sha256=sha256_file(self.manifest_path),
            row_count=self._rows,
            query_count=len(self._sequence),
            size_bytes=size_bytes,
            maximum_bytes=self.maximum_bytes,
            schema_sha256=self.schema_sha256,
            query_sequence_sha256=sequence_sha,
        )
        return self._sealed

    def _validate_artifact(self, artifact: HeldoutMasterArtifact) -> None:
        if self._sealed != artifact:
            raise FoldFusedInvariantError("heldout spool artifact identity drift")
        manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
        required = {
            "schema_version": MANIFEST_VERSION,
            "status": "SEALED_LABEL_BLIND",
            "folds": list(TRAIN_FOLDS),
            "fold0_opened": False,
            "fold1_opened": False,
            "positive_injection": False,
            "label_columns": [],
            "schema_sha256": artifact.schema_sha256,
            "query_count": artifact.query_count,
            "row_count": artifact.row_count,
            "query_sequence_sha256": artifact.query_sequence_sha256,
            "maximum_bytes": artifact.maximum_bytes,
        }
        if any(manifest.get(key) != value for key, value in required.items()):
            raise FoldFusedInvariantError("heldout spool manifest drift")
        record = manifest.get("artifact") or {}
        if (
            record.get("path") != artifact.path.name
            or record.get("sha256") != artifact.sha256
            or record.get("size_bytes") != artifact.size_bytes
            or artifact.size_bytes > artifact.maximum_bytes
            or artifact.manifest_sha256 != sha256_file(artifact.manifest_path)
            or artifact.sha256 != sha256_file(artifact.path)
            or artifact.size_bytes != artifact.path.stat().st_size
        ):
            raise FoldFusedInvariantError("heldout spool bytes/hash drift")
        observed_schema = pq.ParquetFile(artifact.path).schema_arrow
        # Arrow may rewrite child-field metadata while round-tripping nested
        # list/struct types.  Validate the physical fields and the sealed
        # top-level metadata independently.
        if (
            not observed_schema.equals(self.schema, check_metadata=False)
            or observed_schema.metadata != self.schema.metadata
        ):
            raise FoldFusedInvariantError("heldout spool physical schema drift")

    def iter_contexts(
        self, artifact: HeldoutMasterArtifact
    ) -> Iterable[MasterContext]:
        self._validate_artifact(artifact)
        parquet = pq.ParquetFile(artifact.path)
        observed: list[tuple[str, int]] = []
        observed_rows = 0
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
                        "heldout spool query is non-contiguous or duplicated"
                    )
                rows = tuple(table.slice(start, end - start).to_pylist())
                folds = {int(row["fold"]) for row in rows}
                if len(folds) != 1 or next(iter(folds)) not in TRAIN_FOLDS:
                    raise FoldFusedInvariantError("heldout spool fold drift")
                fold = next(iter(folds))
                if tuple(int(row["master_rank"]) for row in rows) != tuple(
                    range(1, len(rows) + 1)
                ):
                    raise FoldFusedInvariantError("replayed Master rank drift")
                sirets = tuple(str(row["siret"]) for row in rows)
                parents = tuple(str(row["siren"]) for row in rows)
                context = MasterContext(
                    query_id=query_id,
                    scene_fold=fold,
                    outer_fold=fold,
                    candidate_sirets=sirets,
                    candidate_parents=parents,
                    opaque=self.payload_factory(rows),
                    metadata={
                        "heldout_spool_schema_sha256": artifact.schema_sha256,
                        "heldout_spool_query_sequence_sha256": artifact.query_sequence_sha256,
                    },
                )
                observed.append((query_id, fold))
                observed_rows += len(rows)
                seen.add(query_id)
                yield context
                start = end
        if (
            observed_rows != artifact.row_count
            or len(observed) != artifact.query_count
            or _sequence_sha256(observed) != artifact.query_sequence_sha256
        ):
            raise FoldFusedInvariantError("heldout spool replay cardinality/order drift")


def projected_national_spool_bytes(
    *, sample_bytes: int, sample_queries: int, national_queries: int
) -> int:
    if sample_bytes <= 0 or sample_queries <= 0 or national_queries <= 0:
        raise ValueError("spool projection inputs must be positive")
    return math.ceil(sample_bytes * national_queries / sample_queries)


def validate_spool_calibration_v12(
    parity_manifest_path: Path,
    calibration: Mapping[str, Any],
    *,
    expected_sample_queries: int,
    expected_national_queries: int,
    expected_maximum_bytes: int,
) -> Mapping[str, Any]:
    """Validate the non-circular 512-query size proof before national OOF.

    The calibration files are retained beside the PREFIT parity manifest, so
    the compressed-byte projection can be recomputed rather than trusted as a
    manifest-only claim.
    """

    expected = {
        "schema_version": CALIBRATION_VERSION,
        "status": "PASS",
        "sample_query_count": int(expected_sample_queries),
        "national_query_count": int(expected_national_queries),
        "maximum_bytes": int(expected_maximum_bytes),
        "budget_pass": True,
        "fold0_opened": False,
        "fold1_opened": False,
        "label_columns": [],
    }
    if any(calibration.get(key) != value for key, value in expected.items()):
        raise FoldFusedInvariantError("heldout spool calibration claims drifted")
    sample_bytes = int(calibration.get("actual_sample_size_bytes") or 0)
    projected = int(calibration.get("projected_national_size_bytes") or 0)
    if (
        sample_bytes <= 0
        or projected
        != projected_national_spool_bytes(
            sample_bytes=sample_bytes,
            sample_queries=expected_sample_queries,
            national_queries=expected_national_queries,
        )
        or projected > expected_maximum_bytes
    ):
        raise FoldFusedInvariantError("heldout spool calibration exceeds or misstates budget")

    root = Path(parity_manifest_path).resolve().parent
    artifact = calibration.get("artifact")
    manifest_record = calibration.get("artifact_manifest")
    if not isinstance(artifact, Mapping) or not isinstance(manifest_record, Mapping):
        raise FoldFusedInvariantError("heldout spool calibration artifacts are missing")
    relative_dir = Path("heldout_spool_calibration")
    parquet_path = (root / relative_dir / "heldout_master.parquet").resolve()
    manifest_path = (root / relative_dir / "manifest.json").resolve()
    if (
        parquet_path.parent != (root / relative_dir).resolve()
        or manifest_path.parent != (root / relative_dir).resolve()
        or artifact.get("path") != str(relative_dir / parquet_path.name)
        or manifest_record.get("path") != str(relative_dir / manifest_path.name)
        or not parquet_path.is_file()
        or not manifest_path.is_file()
        or artifact.get("sha256") != sha256_file(parquet_path)
        or manifest_record.get("sha256") != sha256_file(manifest_path)
        or int(artifact.get("size_bytes") or -1) != parquet_path.stat().st_size
        or parquet_path.stat().st_size != sample_bytes
    ):
        raise FoldFusedInvariantError("heldout spool calibration bytes/hash drifted")
    inner = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        inner.get("schema_version") != MANIFEST_VERSION
        or inner.get("status") != "SEALED_LABEL_BLIND"
        or inner.get("query_count") != expected_sample_queries
        or inner.get("row_count") != calibration.get("sample_row_count")
        or inner.get("maximum_bytes") != expected_maximum_bytes
        or inner.get("label_columns") != []
        or (inner.get("artifact") or {}).get("sha256") != artifact.get("sha256")
        or (inner.get("artifact") or {}).get("size_bytes") != sample_bytes
        or inner.get("schema_sha256") != calibration.get("schema_sha256")
    ):
        raise FoldFusedInvariantError("heldout spool calibration inner manifest drifted")
    return dict(calibration)


__all__ = [
    "CALIBRATION_VERSION",
    "DEFAULT_MAXIMUM_BYTES",
    "MANIFEST_VERSION",
    "ParquetHeldoutMasterSpoolV12",
    "SCHEMA_VERSION",
    "heldout_master_schema_v12",
    "projected_national_spool_bytes",
    "validate_spool_calibration_v12",
]
