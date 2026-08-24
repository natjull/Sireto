"""Compact, label-blind scene index for the fold-fused v12 executor.

The cache deliberately does **not** copy the national candidate parquet.  It
seals that immutable input and writes only a query -> contiguous row-range
index.  Consumers then stream Arrow slices from the source once and can fan a
single :class:`SceneIR` out to several nested-OOF contexts without reparsing
the source or materialising three Master datasets.

Human truth columns are removed at the scanner boundary.  Folds 0 and 1 are
rejected both while building and while reading the cache.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import bisect
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterator, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from .retrieval_top100_cascade_v12 import (
    is_forbidden_input_key,
    production_projection,
)


SCENE_IR_SCHEMA_VERSION = "sireto-retrieval-v12-scene-ir-1"
SCENE_IR_MANIFEST_VERSION = "sireto-retrieval-v12-scene-ir-manifest-1"
V11_MANIFEST_VERSION = "sireto-retrieval-all-human-v11-candidates-manifest-1"
V11_CANDIDATE_VERSION = "sireto-retrieval-all-human-v11-union-1"
V11_LABEL_BLIND_RUN_KINDS = frozenset(
    {"LABEL_BLIND_RETRIEVAL", "LABEL_BLIND_RETRIEVAL_COMBINED"}
)
ALLOWED_FOLDS = frozenset({2, 3, 4})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_INDEX_SCHEMA = pa.schema(
    [
        pa.field("query_id", pa.string(), nullable=False),
        pa.field("fold", pa.int8(), nullable=False),
        pa.field("row_start", pa.int64(), nullable=False),
        pa.field("row_count", pa.int32(), nullable=False),
    ]
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _schema_descriptor(schema: pa.Schema) -> list[dict[str, Any]]:
    return [
        {
            "name": field.name,
            "type": str(field.type),
            "nullable": bool(field.nullable),
        }
        for field in schema
    ]


def _outside_workspace(output_root: Path) -> None:
    """Fail closed when a heavy cache is accidentally aimed at the repo."""

    workspace = Path(__file__).resolve().parents[2]
    resolved = output_root.resolve()
    if resolved == workspace or workspace in resolved.parents:
        raise ValueError(
            "v12 SceneIR output must be outside the workspace (use the external volume)"
        )


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"v11 source manifest lacks valid {field}")
    return value


def _source_manifest_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".manifest.json")


def _strict_source_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return every v11 value on which production lineage depends."""

    fields = (
        "schema_version",
        "candidate_schema_version",
        "run_kind",
        "candidate_parquet_sha256",
        "source_input_sha256",
        "config_sha256",
        "official_config_sha256",
        "sidecar_manifest_sha256",
        "folds",
        "queries",
        "candidate_rows",
        "empty_queries",
        "maximum_union_candidates",
        "positive_injection",
        "human_labels_opened_by_retrieval",
        "source_shards",
    )
    return {field: manifest.get(field) for field in fields}


def _validate_source_manifest_contract(
    source: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Port of the strict real-v11 validation used by the v12 cascade."""

    if (
        manifest.get("schema_version") != V11_MANIFEST_VERSION
        or manifest.get("candidate_schema_version") != V11_CANDIDATE_VERSION
    ):
        raise ValueError("SceneIR requires the sealed v11 candidate contract")
    run_kind = manifest.get("run_kind")
    if run_kind not in V11_LABEL_BLIND_RUN_KINDS:
        raise ValueError("SceneIR source is not an official label-blind v11 run")
    if manifest.get("positive_injection") is not False:
        raise ValueError("SceneIR source does not prove zero positive injection")
    if manifest.get("human_labels_opened_by_retrieval") is not False:
        raise ValueError("SceneIR source does not prove label-blind retrieval")

    folds_raw = manifest.get("folds")
    if not isinstance(folds_raw, list) or any(
        not isinstance(value, int) for value in folds_raw
    ):
        raise ValueError("v11 source folds must be an explicit integer list")
    folds = set(folds_raw)
    if len(folds) != len(folds_raw) or not folds or not folds.issubset(ALLOWED_FOLDS):
        raise ValueError("SceneIR refuses any source declaring fold 0 or fold 1")

    candidate_sha = _require_sha256(
        manifest.get("candidate_parquet_sha256"),
        field="candidate_parquet_sha256",
    )
    for field in (
        "source_input_sha256",
        "config_sha256",
        "official_config_sha256",
        "sidecar_manifest_sha256",
    ):
        _require_sha256(manifest.get(field), field=field)
    if candidate_sha != sha256_file(source):
        raise ValueError("SceneIR source candidate hash mismatch")

    parquet_rows = int(pq.ParquetFile(source).metadata.num_rows)
    queries = manifest.get("queries")
    candidate_rows = manifest.get("candidate_rows")
    empty_queries = manifest.get("empty_queries")
    maximum = manifest.get("maximum_union_candidates")
    if not isinstance(queries, int) or queries <= 0:
        raise ValueError("v11 source manifest has an invalid query count")
    if not isinstance(candidate_rows, int) or candidate_rows != parquet_rows:
        raise ValueError("v11 source manifest candidate row count mismatch")
    if (
        not isinstance(empty_queries, int)
        or empty_queries < 0
        or empty_queries > queries
    ):
        raise ValueError("v11 source manifest has an invalid empty-query count")
    if not isinstance(maximum, int) or not 1 <= maximum <= 2_000:
        raise ValueError("v11 source manifest exceeds the 2,000-candidate cap")

    source_shards = manifest.get("source_shards")
    if run_kind == "LABEL_BLIND_RETRIEVAL_COMBINED":
        if not isinstance(source_shards, list) or len(source_shards) < 2:
            raise ValueError("combined v11 source lacks sealed source shards")
        shard_folds: set[int] = set()
        shard_queries = shard_rows = shard_empty = 0
        shard_maximum = 0
        stable_fields = (
            "schema_version",
            "candidate_schema_version",
            "source_input_sha256",
            "config_sha256",
            "official_config_sha256",
            "sidecar_manifest_sha256",
            "positive_injection",
            "human_labels_opened_by_retrieval",
        )
        seen_paths: set[Path] = set()
        for shard in source_shards:
            if not isinstance(shard, Mapping):
                raise ValueError("combined v11 source has an invalid shard entry")
            raw_shard_path = shard.get("path")
            if not isinstance(raw_shard_path, str) or not raw_shard_path:
                raise ValueError("combined v11 source shard lacks a path")
            shard_path = Path(raw_shard_path)
            if not shard_path.is_absolute() or shard_path in seen_paths:
                raise ValueError(
                    "combined v11 source shard paths must be unique/absolute"
                )
            seen_paths.add(shard_path)
            shard_manifest_path = _source_manifest_path(shard_path)
            if not shard_path.is_file() or not shard_manifest_path.is_file():
                raise FileNotFoundError(
                    f"combined v11 source shard lineage is missing: {shard_path}"
                )
            declared_shard_sha = _require_sha256(
                shard.get("sha256"), field="shard sha256"
            )
            declared_manifest_sha = _require_sha256(
                shard.get("manifest_sha256"), field="shard manifest_sha256"
            )
            if sha256_file(shard_manifest_path) != declared_manifest_sha:
                raise ValueError("combined v11 source shard manifest hash mismatch")
            if sha256_file(shard_path) != declared_shard_sha:
                raise ValueError("combined v11 source shard candidate hash mismatch")
            shard_manifest = json.loads(
                shard_manifest_path.read_text(encoding="utf-8")
            )
            if not isinstance(shard_manifest, Mapping):
                raise ValueError("combined v11 source shard manifest is not an object")
            if (
                shard_manifest.get("run_kind") != "LABEL_BLIND_RETRIEVAL"
                or shard_manifest.get("candidate_parquet_sha256")
                != declared_shard_sha
                or any(
                    shard_manifest.get(field) != manifest.get(field)
                    for field in stable_fields
                )
            ):
                raise ValueError("combined v11 source shard lineage mismatch")
            _require_sha256(
                shard_manifest.get("source_manifest_sha256"),
                field="shard source_manifest_sha256",
            )
            current_folds = shard_manifest.get("folds")
            if (
                not isinstance(current_folds, list)
                or len(current_folds) != 1
                or not isinstance(current_folds[0], int)
                or current_folds[0] not in ALLOWED_FOLDS
                or current_folds[0] in shard_folds
            ):
                raise ValueError(
                    "combined v11 source shards have invalid/overlapping folds"
                )
            shard_folds.add(current_folds[0])
            current_rows = shard_manifest.get("candidate_rows")
            current_queries = shard_manifest.get("queries")
            current_empty = shard_manifest.get("empty_queries")
            current_maximum = shard_manifest.get("maximum_union_candidates")
            if (
                not isinstance(current_rows, int)
                or current_rows
                != int(pq.ParquetFile(shard_path).metadata.num_rows)
                or not isinstance(current_queries, int)
                or current_queries <= 0
                or not isinstance(current_empty, int)
                or current_empty < 0
                or current_empty > current_queries
                or not isinstance(current_maximum, int)
                or not 1 <= current_maximum <= 2_000
            ):
                raise ValueError("combined v11 source shard counts/cap are invalid")
            shard_rows += current_rows
            shard_queries += current_queries
            shard_empty += current_empty
            shard_maximum = max(shard_maximum, current_maximum)
        if (
            shard_folds != folds
            or shard_rows != candidate_rows
            or shard_queries != queries
            or shard_empty != empty_queries
            or shard_maximum != maximum
        ):
            raise ValueError(
                "combined v11 source shard counts/folds do not reconcile"
            )
    elif source_shards is not None:
        raise ValueError("non-combined v11 source may not declare source shards")
    return dict(manifest)


def _read_source_manifest(
    source: Path, manifest_path: Path
) -> tuple[dict[str, Any], str, str]:
    if not source.is_file():
        raise FileNotFoundError(source)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("v11 source manifest must be a JSON object")
    validated = _validate_source_manifest_contract(source, manifest)
    return validated, validated["candidate_parquet_sha256"], sha256_file(
        manifest_path
    )


def _validate_existing(target: Path, build_id: str) -> "SceneIRBuild":
    manifest_path = target / "manifest.json"
    index_path = target / "scene_index.parquet"
    if not manifest_path.is_file() or not index_path.is_file():
        raise ValueError("existing SceneIR publication is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != SCENE_IR_MANIFEST_VERSION
        or manifest.get("build_id") != build_id
        or manifest.get("build_identity") is None
        or hashlib.sha256(
            _canonical_json(manifest["build_identity"])
        ).hexdigest()
        != build_id
        or target.name != build_id[:16]
    ):
        raise ValueError("SceneIR content-address collision")
    record = (manifest.get("files") or {}).get(index_path.name) or {}
    if (
        record.get("size_bytes") != index_path.stat().st_size
        or record.get("sha256") != sha256_file(index_path)
        or record.get("schema") != _schema_descriptor(pq.read_schema(index_path))
    ):
        raise ValueError("existing SceneIR index integrity mismatch")
    return SceneIRBuild(
        output_dir=target,
        manifest_path=manifest_path,
        build_id=build_id,
        query_count=int(manifest["query_count"]),
        candidate_rows=int(manifest["candidate_rows"]),
    )


@dataclass(frozen=True)
class SceneIRBuild:
    output_dir: Path
    manifest_path: Path
    build_id: str
    query_count: int
    candidate_rows: int


@dataclass(frozen=True)
class SceneIR:
    """One immutable label-blind query scene backed by Arrow columns."""

    query_id: str
    fold: int
    row_start: int
    rows: pa.Table

    @property
    def row_count(self) -> int:
        return self.rows.num_rows

    def as_mappings(self) -> list[dict[str, Any]]:
        """Compatibility bridge for the reference Python cascade.

        The fold-fused executor should consume Arrow columns directly.  This
        conversion exists only for parity tests and legacy adapters.
        """

        return self.rows.to_pylist()


def build_scene_ir_cache(
    *,
    source: Path,
    source_manifest: Path,
    output_root: Path,
    batch_size: int = 262_144,
) -> SceneIRBuild:
    """Seal a v11 source and publish its compact scene-offset index."""

    source = Path(source).resolve()
    source_manifest = Path(source_manifest).resolve()
    output_root = Path(output_root).resolve()
    _outside_workspace(output_root)
    if batch_size < 1:
        raise ValueError("SceneIR batch_size must be positive")

    manifest, source_sha, source_manifest_sha = _read_source_manifest(
        source, source_manifest
    )
    parquet = pq.ParquetFile(source)
    names = parquet.schema_arrow.names
    if "query_id" not in names:
        raise ValueError("SceneIR source lacks query_id")
    fold_columns = [name for name in ("fold", "oof_fold") if name in names]
    if not fold_columns:
        raise ValueError("SceneIR source lacks fold/oof_fold")

    strict_source = _strict_source_contract(manifest)
    identity = {
        "schema_version": SCENE_IR_SCHEMA_VERSION,
        "source_candidate_sha256": source_sha,
        "source_manifest_sha256": source_manifest_sha,
        "source_candidate_schema": manifest["candidate_schema_version"],
        "strict_source_manifest": strict_source,
        "folds": [2, 3, 4],
        "projection": "PRODUCTION_PROJECTION_AT_ARROW_SCANNER_BOUNDARY",
        "candidate_data_copied": False,
        "positive_injection": False,
        "human_labels_opened": False,
    }
    build_id = hashlib.sha256(_canonical_json(identity)).hexdigest()
    target = output_root / build_id[:16]
    if target.exists():
        return _validate_existing(target, build_id)

    output_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".scene-ir-v12-", dir=output_root))
    try:
        records: list[dict[str, Any]] = []
        current_id: str | None = None
        current_fold: int | None = None
        current_start = 0
        current_count = 0
        absolute_row = 0
        closed: set[str] = set()
        observed_folds: set[int] = set()

        scan_columns = ["query_id", *fold_columns]
        for batch in parquet.iter_batches(
            batch_size=batch_size, columns=scan_columns
        ):
            column_values = {
                name: batch.column(index).to_pylist()
                for index, name in enumerate(scan_columns)
            }
            for offset, query_raw in enumerate(column_values["query_id"]):
                query_id = str(query_raw or "")
                if not query_id:
                    raise ValueError("SceneIR source contains an empty query_id")
                fold_values = {
                    int(column_values[name][offset]) for name in fold_columns
                }
                if len(fold_values) != 1:
                    raise ValueError("SceneIR fold and oof_fold disagree")
                fold = fold_values.pop()
                if fold not in ALLOWED_FOLDS:
                    raise ValueError("SceneIR refuses fold 0/1")
                observed_folds.add(fold)
                if current_id is None:
                    current_id = query_id
                    current_fold = fold
                    current_start = absolute_row
                elif query_id != current_id:
                    records.append(
                        {
                            "query_id": current_id,
                            "fold": current_fold,
                            "row_start": current_start,
                            "row_count": current_count,
                        }
                    )
                    closed.add(current_id)
                    if query_id in closed:
                        raise ValueError("SceneIR source query groups are not contiguous")
                    current_id = query_id
                    current_fold = fold
                    current_start = absolute_row
                    current_count = 0
                elif fold != current_fold:
                    raise ValueError("SceneIR query group mixes folds")
                current_count += 1
                absolute_row += 1
        if current_id is not None:
            records.append(
                {
                    "query_id": current_id,
                    "fold": current_fold,
                    "row_start": current_start,
                    "row_count": current_count,
                }
            )
        if not records or observed_folds != ALLOWED_FOLDS:
            raise ValueError("SceneIR source must cover folds 2/3/4")
        if absolute_row != parquet.metadata.num_rows:
            raise AssertionError("SceneIR scan row count drift")
        if int(manifest.get("candidate_rows", -1)) != absolute_row:
            raise ValueError("SceneIR source manifest row count mismatch")
        expected_queries = int(manifest.get("queries", -1)) - int(
            manifest.get("empty_queries", 0)
        )
        if expected_queries != len(records):
            raise ValueError("SceneIR source manifest query count mismatch")
        maximum = max(int(record["row_count"]) for record in records)
        if maximum > 2_000 or maximum != int(
            manifest.get("maximum_union_candidates", -1)
        ):
            raise ValueError("SceneIR source candidate ceiling mismatch")

        index_path = stage / "scene_index.parquet"
        table = pa.Table.from_pylist(records, schema=_INDEX_SCHEMA)
        pq.write_table(
            table,
            index_path,
            compression="zstd",
            row_group_size=65_536,
        )
        file_record = {
            "size_bytes": index_path.stat().st_size,
            "sha256": sha256_file(index_path),
            "rows": len(records),
            "schema": _schema_descriptor(pq.read_schema(index_path)),
        }
        publication = {
            **identity,
            "schema_version": SCENE_IR_MANIFEST_VERSION,
            "scene_schema_version": SCENE_IR_SCHEMA_VERSION,
            "build_id": build_id,
            "build_identity": identity,
            "source_path": str(source),
            "source_manifest_path": str(source_manifest),
            "strict_source_manifest": strict_source,
            "query_count": len(records),
            "candidate_rows": absolute_row,
            "maximum_scene_rows": maximum,
            "fold_counts": {
                str(fold): sum(int(row["fold"]) == fold for row in records)
                for fold in sorted(ALLOWED_FOLDS)
            },
            "forbidden_columns_materialized": [],
            "files": {index_path.name: file_record},
        }
        (stage / "manifest.json").write_bytes(_canonical_json(publication))
        os.rename(stage, target)
        return SceneIRBuild(
            output_dir=target,
            manifest_path=target / "manifest.json",
            build_id=build_id,
            query_count=len(records),
            candidate_rows=absolute_row,
        )
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


class SceneIRCache:
    """Validated streaming reader for a published SceneIR index."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        source_override: Path | None = None,
        verify_source_hash: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir).resolve()
        manifest_path = self.cache_dir / "manifest.json"
        index_path = self.cache_dir / "scene_index.parquet"
        if not manifest_path.is_file() or not index_path.is_file():
            raise FileNotFoundError("SceneIR publication is incomplete")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != SCENE_IR_MANIFEST_VERSION:
            raise ValueError("unsupported SceneIR manifest")
        identity = self.manifest.get("build_identity")
        build_id = self.manifest.get("build_id")
        if (
            not isinstance(identity, Mapping)
            or not isinstance(build_id, str)
            or hashlib.sha256(_canonical_json(identity)).hexdigest() != build_id
            or self.cache_dir.name != build_id[:16]
        ):
            raise ValueError("SceneIR content address does not match its identity")
        if self.manifest.get("folds") != [2, 3, 4]:
            raise ValueError("SceneIR manifest opened a forbidden fold")
        record = (self.manifest.get("files") or {}).get(index_path.name) or {}
        if (
            record.get("size_bytes") != index_path.stat().st_size
            or record.get("sha256") != sha256_file(index_path)
            or record.get("schema") != _schema_descriptor(pq.read_schema(index_path))
        ):
            raise ValueError("SceneIR index integrity mismatch")
        self.index = pq.read_table(index_path)
        if self.index.schema != _INDEX_SCHEMA:
            raise ValueError("SceneIR index schema drift")
        source_raw = source_override or Path(str(self.manifest.get("source_path") or ""))
        self.source = Path(source_raw).resolve()
        if not self.source.is_file():
            raise FileNotFoundError(self.source)
        source_manifest_path = Path(
            str(self.manifest.get("source_manifest_path") or "")
        ).resolve()
        if not source_manifest_path.is_file():
            raise FileNotFoundError(source_manifest_path)
        if sha256_file(source_manifest_path) != self.manifest.get(
            "source_manifest_sha256"
        ):
            raise ValueError("SceneIR source manifest hash mismatch")
        source_manifest_value = json.loads(
            source_manifest_path.read_text(encoding="utf-8")
        )
        if not isinstance(source_manifest_value, Mapping):
            raise ValueError("SceneIR source manifest is not an object")
        if _strict_source_contract(source_manifest_value) != self.manifest.get(
            "strict_source_manifest"
        ):
            raise ValueError("SceneIR strict source contract drift")
        if verify_source_hash:
            validated = _validate_source_manifest_contract(
                self.source, source_manifest_value
            )
            if validated["candidate_parquet_sha256"] != self.manifest.get(
                "source_candidate_sha256"
            ):
                raise ValueError("SceneIR source hash mismatch")

    def iter_scenes(
        self,
        *,
        columns: Sequence[str] | None = None,
        batch_size: int = 262_144,
    ) -> Iterator[SceneIR]:
        parquet = pq.ParquetFile(self.source)
        available = parquet.schema_arrow.names
        safe = production_projection(available)
        if columns is None:
            selected = list(safe)
        else:
            requested = [str(value) for value in columns]
            forbidden = [name for name in requested if is_forbidden_input_key(name)]
            if forbidden:
                raise ValueError(
                    f"SceneIR refuses truth-side columns: {sorted(forbidden)}"
                )
            missing = sorted(set(requested).difference(available))
            if missing:
                raise ValueError(f"SceneIR source lacks requested columns: {missing}")
            selected = requested
        for required in ("query_id", "fold"):
            if required in available and required not in selected:
                selected.append(required)
        if "query_id" not in selected:
            raise ValueError("SceneIR projection requires query_id")
        if "fold" not in selected and "oof_fold" in available:
            selected.append("oof_fold")

        expected = self.index.to_pylist()
        expected_position = 0
        current_id: str | None = None
        current_fold: int | None = None
        current_start = 0
        current_parts: list[pa.Table] = []
        absolute_row = 0

        def emit() -> SceneIR:
            nonlocal expected_position
            assert current_id is not None and current_fold is not None
            table = (
                current_parts[0]
                if len(current_parts) == 1
                else pa.concat_tables(current_parts)
            )
            if expected_position >= len(expected):
                raise ValueError("SceneIR source has more scenes than its index")
            sealed = expected[expected_position]
            expected_position += 1
            if (
                sealed["query_id"] != current_id
                or int(sealed["fold"]) != current_fold
                or int(sealed["row_start"]) != current_start
                or int(sealed["row_count"]) != table.num_rows
            ):
                raise ValueError("SceneIR source no longer matches its sealed offsets")
            return SceneIR(current_id, current_fold, current_start, table)

        for batch in parquet.iter_batches(batch_size=batch_size, columns=selected):
            table = pa.Table.from_batches([batch])
            ids = table["query_id"].combine_chunks().to_pylist()
            start = 0
            while start < len(ids):
                query_id = str(ids[start] or "")
                end = start + 1
                while end < len(ids) and str(ids[end] or "") == query_id:
                    end += 1
                piece = table.slice(start, end - start)
                fold_name = "fold" if "fold" in piece.column_names else "oof_fold"
                folds = {int(value) for value in piece[fold_name].to_pylist()}
                if len(folds) != 1 or not folds.issubset(ALLOWED_FOLDS):
                    raise ValueError("SceneIR streamed scene mixes or opens folds")
                fold = folds.pop()
                if current_id is None:
                    current_id = query_id
                    current_fold = fold
                    current_start = absolute_row + start
                elif query_id != current_id:
                    yield emit()
                    current_id = query_id
                    current_fold = fold
                    current_start = absolute_row + start
                    current_parts = []
                elif fold != current_fold:
                    raise ValueError("SceneIR streamed query changes fold")
                current_parts.append(piece)
                start = end
            absolute_row += batch.num_rows
        if current_id is not None:
            yield emit()
        if expected_position != len(expected):
            raise ValueError("SceneIR source has fewer scenes than its index")

    def iter_selected_scenes(
        self,
        query_ids: Sequence[str],
        *,
        columns: Sequence[str] | None = None,
    ) -> Iterator[SceneIR]:
        """Read sealed query ranges without scanning the complete candidate file.

        This is used only by bounded parity replays.  National parent/site OOF
        still streams the source sequentially; a 512-query audit must not turn
        into another full 39M-row pass merely to discard nearly every scene.
        Selected ranges are visited in source order and each touched Parquet row
        group is decoded at most once.
        """

        requested = tuple(str(value) for value in query_ids)
        if not requested or len(requested) != len(set(requested)) or any(
            not value for value in requested
        ):
            raise ValueError("SceneIR selective query IDs must be non-empty and unique")
        index_rows = self.index.to_pylist()
        wanted = set(requested)
        selected_index = [row for row in index_rows if str(row["query_id"]) in wanted]
        if {str(row["query_id"]) for row in selected_index} != wanted:
            raise ValueError("SceneIR selective query population differs from its index")

        parquet = pq.ParquetFile(self.source)
        available = parquet.schema_arrow.names
        safe = production_projection(available)
        if columns is None:
            selected_columns = list(safe)
        else:
            selected_columns = [str(value) for value in columns]
            forbidden = [
                name for name in selected_columns if is_forbidden_input_key(name)
            ]
            if forbidden:
                raise ValueError(
                    f"SceneIR refuses truth-side columns: {sorted(forbidden)}"
                )
            missing = sorted(set(selected_columns).difference(available))
            if missing:
                raise ValueError(f"SceneIR source lacks requested columns: {missing}")
        for required in ("query_id", "fold"):
            if required in available and required not in selected_columns:
                selected_columns.append(required)
        if "query_id" not in selected_columns:
            raise ValueError("SceneIR projection requires query_id")
        if "fold" not in selected_columns and "oof_fold" in available:
            selected_columns.append("oof_fold")

        row_group_starts: list[int] = []
        absolute = 0
        for row_group in range(parquet.num_row_groups):
            row_group_starts.append(absolute)
            absolute += parquet.metadata.row_group(row_group).num_rows
        if absolute != int(self.manifest.get("candidate_rows", -1)):
            raise ValueError("SceneIR selective source row count drift")
        cached_group = -1
        cached_table: pa.Table | None = None
        for sealed in selected_index:
            position = int(sealed["row_start"])
            remaining = int(sealed["row_count"])
            parts: list[pa.Table] = []
            while remaining:
                row_group = bisect.bisect_right(row_group_starts, position) - 1
                if row_group < 0 or row_group >= parquet.num_row_groups:
                    raise ValueError("SceneIR selective row range escaped source")
                if row_group != cached_group:
                    cached_table = parquet.read_row_group(
                        row_group, columns=selected_columns
                    )
                    cached_group = row_group
                assert cached_table is not None
                local = position - row_group_starts[row_group]
                take = min(remaining, cached_table.num_rows - local)
                if take <= 0:
                    raise ValueError("SceneIR selective row-group offsets drifted")
                parts.append(cached_table.slice(local, take))
                position += take
                remaining -= take
            table = parts[0] if len(parts) == 1 else pa.concat_tables(parts)
            query_id = str(sealed["query_id"])
            fold_name = "fold" if "fold" in table.column_names else "oof_fold"
            if (
                table.num_rows != int(sealed["row_count"])
                or set(str(value or "") for value in table["query_id"].to_pylist())
                != {query_id}
                or {int(value) for value in table[fold_name].to_pylist()}
                != {int(sealed["fold"])}
                or int(sealed["fold"]) not in ALLOWED_FOLDS
            ):
                raise ValueError("SceneIR selective rows no longer match sealed index")
            yield SceneIR(
                query_id,
                int(sealed["fold"]),
                int(sealed["row_start"]),
                table,
            )


__all__ = [
    "ALLOWED_FOLDS",
    "SCENE_IR_MANIFEST_VERSION",
    "SCENE_IR_SCHEMA_VERSION",
    "SceneIR",
    "SceneIRBuild",
    "SceneIRCache",
    "build_scene_ir_cache",
]
