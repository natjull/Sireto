#!/usr/bin/env python3
"""Build a label-blind v11 union, then attach sealed labels for OOF only."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Iterator, Mapping

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xgb_matcher.official_evidence_retrieval import (  # noqa: E402
    OfficialEvidenceRetrievalConfig,
    OfficialEvidenceRetriever,
)
from src.xgb_matcher.retrieval_all_human_v11 import (  # noqa: E402
    AllHumanV11Retriever,
    RetrievalV11Config,
    V11_RETRIEVAL_SCHEMA_VERSION,
)
from src.xgb_matcher.retrieval_sidecars_v11 import RetrievalSidecarReader  # noqa: E402


MANIFEST_SCHEMA = "sireto-retrieval-all-human-v11-candidates-manifest-1"
LABEL_COLUMNS = frozenset(
    {
        "ground_truth_siret",
        "historical_ground_truth_siret",
        "human_ground_truth_siret",
        "human_ground_truth_siren",
        "acceptable_sirets_operational",
        "acceptable_sirets_operational_json",
    }
)
SAFE_QUERY_METADATA = (
    "query_id",
    "oof_fold",
    "split",
    "crm_name",
    "crm_address",
    "crm_city",
    "postcode",
    "insee",
    "reference_date",
    "ground_truth_state",
    "label_kind",
    "source_kind",
    "data_origin",
    "unseen_siren",
    "new_site_known_siren",
    "pool_size",
    "mega_base_pool",
)

_STRING_LIST_COLUMNS = frozenset(
    {
        "sources",
        "channels",
        "official_evidence_ids",
        "official_evidence_sources",
        "candidate_names",
        "candidate_addresses",
    }
)
_BOOLEAN_COLUMNS = frozenset(
    {
        "unseen_siren",
        "new_site_known_siren",
        "mega_base_pool",
        "candidate_is_siege",
        "union_truncated",
        "candidate_present",
    }
)
_INTEGER_COLUMNS = frozenset(
    {
        "oof_fold",
        "fold",
        "pool_size",
        "union_rank",
        "retrieval_rank",
        "raw_union_size",
    }
)
_SIGNAL_TYPE = pa.list_(
    pa.struct(
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
        ]
    )
)


def _candidate_schema(row: Mapping[str, Any]) -> pa.Schema:
    """Return a stable schema even when the first batch has empty channels.

    Arrow otherwise infers ``null`` for a channel absent from every row in the
    first write batch, making the Parquet writer reject a later numeric value.
    """
    fields: list[pa.Field] = []
    for name in row:
        if name == "signals":
            dtype = _SIGNAL_TYPE
        elif name in _STRING_LIST_COLUMNS:
            dtype = pa.list_(pa.string())
        elif name in _BOOLEAN_COLUMNS:
            dtype = pa.bool_()
        elif name in _INTEGER_COLUMNS or name.endswith("_rank"):
            dtype = pa.int64()
        elif name.endswith("_score") or name == "retrieval_latency_ms":
            dtype = pa.float64()
        else:
            dtype = pa.string()
        fields.append(pa.field(name, dtype))
    return pa.schema(fields)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".manifest.json")


def _write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _rows(path: Path, folds: set[int], batch_size: int) -> Iterator[dict[str, Any]]:
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
        for raw in batch.to_pylist():
            fold = int(raw.get("oof_fold", raw.get("fold", -1)))
            if fold not in folds:
                continue
            # This is the hard production boundary: truth-side columns never
            # enter RetrievalQuery, the official retriever or the sidecars.
            yield {key: value for key, value in raw.items() if key not in LABEL_COLUMNS}


def _query_metadata(
    row: Mapping[str, Any], *, normalized_query: Any
) -> dict[str, Any]:
    metadata = {
        key: row.get(key)
        for key in SAFE_QUERY_METADATA
        if key in row and key != "query_id"
    }
    metadata["fold"] = int(row.get("oof_fold", row.get("fold", -1)))
    metadata["crm_name"] = normalized_query.name
    metadata["crm_address"] = normalized_query.address
    metadata["crm_number"] = normalized_query.number
    metadata["crm_postcode"] = normalized_query.postcode
    metadata["crm_insee"] = normalized_query.insee
    return metadata


def retrieve(args: argparse.Namespace) -> Path:
    output = args.output.resolve()
    manifest_path = _manifest_path(output)
    if output.exists() or manifest_path.exists():
        raise FileExistsError(output if output.exists() else manifest_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)

    # Pin the exact bytes loaded by this process.  The manifest must never hash
    # a config that changed while a long retrieval run was in progress.
    loaded_config_sha256 = sha256_file(args.config)
    loaded_official_config_sha256 = sha256_file(args.official_config)
    v11_config = RetrievalV11Config.load(args.config)
    if not 1 <= int(args.search_workers) <= 16:
        raise ValueError("search_workers must be between 1 and 16")
    official_config = OfficialEvidenceRetrievalConfig.load(args.official_config)
    official_config = replace(
        official_config,
        enable_direct_parent_expansion=False,
        enable_consensus_parent_expansion=False,
        relation_seed_limit=v11_config.relation_seed_limit,
        search_workers=int(args.search_workers),
    )
    official = OfficialEvidenceRetriever.from_index_paths(
        args.base_index,
        args.overlay_index,
        official_config,
        temporal_overlay_index_path=args.temporal_overlay_index,
        legal_entity_role_index_path=args.legal_entity_role_index,
        sector_official_index_path=args.sector_official_index,
    )
    writer: pq.ParquetWriter | None = None
    buffer: list[dict[str, Any]] = []
    query_count = candidate_count = empty_count = max_candidates = 0
    folds = {int(value) for value in args.folds.split(",") if value.strip()}
    if not folds or not folds.issubset({0, 2, 3, 4}):
        raise ValueError("retrieve folds must be a non-empty subset of 0,2,3,4")

    try:
        with RetrievalSidecarReader(args.sidecars) as sidecars:
            retriever = AllHumanV11Retriever(official, sidecars, v11_config)
            for row in _rows(args.input, folds, args.batch_size):
                query_count += 1
                query_id = str(row.get("query_id") or query_count)
                result = retriever.retrieve(row)
                rows = result.candidate_rows(
                    query_id, _query_metadata(row, normalized_query=result.query)
                )
                if not rows:
                    empty_count += 1
                max_candidates = max(max_candidates, len(rows))
                candidate_count += len(rows)
                buffer.extend(rows)
                if len(buffer) < args.write_batch_rows:
                    continue
                schema = writer.schema if writer else _candidate_schema(buffer[0])
                table = pa.Table.from_pylist(buffer, schema=schema)
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
                writer.write_table(table)
                buffer.clear()
            if buffer:
                schema = writer.schema if writer else _candidate_schema(buffer[0])
                table = pa.Table.from_pylist(buffer, schema=schema)
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
                writer.write_table(table)
        if writer is None:
            raise ValueError("retrieval produced no candidate rows")
        writer.close()
        writer = None
        if max_candidates > v11_config.internal_cap:
            raise AssertionError("v11 candidate cap exceeded")
        os.replace(temporary, output)
        input_manifest = args.input.parent / "manifest.json"
        payload = {
            "schema_version": MANIFEST_SCHEMA,
            "candidate_schema_version": V11_RETRIEVAL_SCHEMA_VERSION,
            "run_kind": "LABEL_BLIND_RETRIEVAL",
            "candidate_parquet_sha256": sha256_file(output),
            "source_input_sha256": sha256_file(args.input),
            "source_manifest_sha256": (
                sha256_file(input_manifest) if input_manifest.is_file() else None
            ),
            "config_sha256": loaded_config_sha256,
            "official_config_sha256": loaded_official_config_sha256,
            "sidecar_manifest_sha256": sha256_file(args.sidecars / "manifest.json"),
            "folds": sorted(folds),
            "queries": query_count,
            "candidate_rows": candidate_count,
            "empty_queries": empty_count,
            "maximum_union_candidates": max_candidates,
            "positive_injection": False,
            "human_labels_opened_by_retrieval": False,
        }
        _write_manifest(manifest_path, payload)
        return output
    except Exception:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
        raise


def attach_labels(args: argparse.Namespace) -> Path:
    candidates = args.candidates.resolve()
    labels = args.labels.resolve()
    operational = args.operational_labels.resolve()
    output = args.output.resolve()
    manifest_path = _manifest_path(output)
    if output.exists() or manifest_path.exists():
        raise FileExistsError(output if output.exists() else manifest_path)
    source_manifest_path = _manifest_path(candidates)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("positive_injection") is not False:
        raise ValueError("candidate source does not prove zero positive injection")
    if source_manifest.get("human_labels_opened_by_retrieval") is not False:
        raise ValueError("candidate source is not label-blind")
    folds = tuple(int(value) for value in source_manifest.get("folds") or ())
    if not folds or not set(folds).issubset({0, 2, 3, 4}):
        raise ValueError("candidate source has invalid folds")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    connection = duckdb.connect()
    try:
        candidate_queries = int(
            connection.execute(
                f"SELECT count(DISTINCT query_id) FROM read_parquet('{_sql_path(candidates)}')"
            ).fetchone()[0]
        )
        fold_sql = ",".join(str(value) for value in folds)
        label_queries = int(
            connection.execute(
                f"SELECT count(*) FROM read_parquet('{_sql_path(labels)}') "
                f"WHERE oof_fold IN ({fold_sql})"
            ).fetchone()[0]
        )
        if candidate_queries != label_queries or candidate_queries != int(
            source_manifest.get("queries", -1)
        ):
            raise ValueError(
                "Every all-human label query must have a natural candidate union; "
                "empty queries cannot be silently removed from the denominator"
            )
        connection.execute(
            "COPY ("
            "SELECT c.*, l.human_ground_truth_siret, l.human_ground_truth_source, "
            "o.acceptable_sirets_operational_json "
            f"FROM read_parquet('{_sql_path(candidates)}') c "
            f"JOIN read_parquet('{_sql_path(labels)}') l USING(query_id) "
            f"LEFT JOIN read_parquet('{_sql_path(operational)}') o USING(query_id) "
            "ORDER BY c.query_id, c.union_rank"
            f") TO '{_sql_path(temporary)}' "
            "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 65536)"
        )
        counts = connection.execute(
            "SELECT count(*), count(DISTINCT query_id), max(n) FROM ("
            "SELECT query_id, count(*) OVER (PARTITION BY query_id) n "
            "FROM read_parquet(?) WHERE human_ground_truth_siret IS NOT NULL) t",
            [str(temporary)],
        ).fetchone()
    finally:
        connection.close()
    if int(counts[2] or 0) > 2_000:
        temporary.unlink(missing_ok=True)
        raise AssertionError("label attachment changed the v11 union cap")
    os.replace(temporary, output)
    _write_manifest(
        manifest_path,
        {
            "schema_version": MANIFEST_SCHEMA,
            "run_kind": "OOF_LABEL_SIDE_ATTACHMENT",
            "candidate_parquet_sha256": sha256_file(output),
            "source_candidate_sha256": sha256_file(candidates),
            "source_candidate_manifest_sha256": sha256_file(source_manifest_path),
            "exact_labels_sha256": sha256_file(labels),
            "operational_labels_sha256": sha256_file(operational),
            "candidate_rows": int(counts[0]),
            "queries": int(counts[1]),
            "maximum_union_candidates": int(counts[2] or 0),
            "positive_injection": False,
            "labels_attached_after_retrieval": True,
            "all_label_queries_have_natural_candidates": True,
        },
    )
    return output


def combine(args: argparse.Namespace) -> Path:
    inputs = tuple(path.resolve() for path in args.input)
    if len(inputs) < 2:
        raise ValueError("combine requires at least two candidate shards")
    manifests = [
        json.loads(_manifest_path(path).read_text(encoding="utf-8"))
        for path in inputs
    ]
    stable_keys = (
        "schema_version",
        "candidate_schema_version",
        "source_input_sha256",
        "config_sha256",
        "official_config_sha256",
        "sidecar_manifest_sha256",
        "positive_injection",
        "human_labels_opened_by_retrieval",
    )
    for key in stable_keys:
        if len({json.dumps(manifest.get(key), sort_keys=True) for manifest in manifests}) != 1:
            raise ValueError(f"candidate shards disagree on {key}")
    for path, manifest in zip(inputs, manifests):
        if manifest.get("candidate_parquet_sha256") != sha256_file(path):
            raise ValueError(f"candidate shard hash mismatch: {path}")
        if manifest.get("positive_injection") is not False:
            raise ValueError("candidate shard does not prove zero injection")
    folds = [int(value) for manifest in manifests for value in manifest.get("folds", [])]
    if len(folds) != len(set(folds)):
        raise ValueError("candidate shards have overlapping folds")
    output = args.output.resolve()
    manifest_path = _manifest_path(output)
    if output.exists() or manifest_path.exists():
        raise FileExistsError(output if output.exists() else manifest_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)

    # Do not ORDER BY the national union in DuckDB.  The three retrieval
    # shards are already deterministically ordered and have disjoint folds;
    # sorting ~40M wide rows caused a multi-gigabyte materialisation and an
    # OS OOM kill on the 24 GiB production Mac.  Validate the only global key
    # we need (query_id) from the narrow column, then append Arrow batches.
    query_ids: set[str] = set()
    observed_queries = 0
    observed_max_candidates = 0
    connection = duckdb.connect()
    try:
        for path, manifest in zip(inputs, manifests):
            rows = connection.execute(
                f"SELECT query_id, count(*) AS n "
                f"FROM read_parquet('{_sql_path(path)}') GROUP BY query_id"
            ).fetchall()
            shard_ids = {str(row[0]) for row in rows}
            if len(shard_ids) != len(rows):
                raise AssertionError("candidate shard contains duplicate query groups")
            overlap = query_ids.intersection(shard_ids)
            if overlap:
                raise ValueError("candidate shards contain overlapping query ids")
            query_ids.update(shard_ids)
            observed_queries += len(rows)
            observed_max_candidates = max(
                observed_max_candidates, max((int(row[1]) for row in rows), default=0)
            )
            if len(rows) != int(manifest.get("queries", -1)):
                raise ValueError(f"candidate shard query count mismatch: {path}")
    finally:
        connection.close()

    if observed_max_candidates > 2_000:
        raise AssertionError("combined v11 union exceeds 2,000 candidates")

    writer: pq.ParquetWriter | None = None
    candidate_rows = 0
    try:
        for path in inputs:
            parquet = pq.ParquetFile(path)
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary, parquet.schema_arrow, compression="zstd"
                )
            elif parquet.schema_arrow != writer.schema:
                raise ValueError(f"candidate shard schema mismatch: {path}")
            for batch in parquet.iter_batches(batch_size=65_536):
                table = pa.Table.from_batches([batch], schema=writer.schema)
                writer.write_table(table, row_group_size=65_536)
                candidate_rows += batch.num_rows
        if writer is None:
            raise ValueError("combine produced no candidate rows")
        writer.close()
        writer = None
    except Exception:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
        raise

    expected_rows = sum(
        int(item.get("candidate_rows", pq.ParquetFile(path).metadata.num_rows))
        for path, item in zip(inputs, manifests)
    )
    if candidate_rows != expected_rows:
        temporary.unlink(missing_ok=True)
        raise ValueError("combined candidate row count mismatch")
    os.replace(temporary, output)
    _write_manifest(
        manifest_path,
        {
            **{key: manifests[0].get(key) for key in stable_keys},
            "run_kind": "LABEL_BLIND_RETRIEVAL_COMBINED",
            "candidate_parquet_sha256": sha256_file(output),
            "source_shards": [
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "manifest_sha256": sha256_file(_manifest_path(path)),
                }
                for path in inputs
            ],
            "folds": sorted(folds),
            "queries": observed_queries,
            "candidate_rows": candidate_rows,
            "empty_queries": sum(int(item.get("empty_queries", 0)) for item in manifests),
            "maximum_union_candidates": observed_max_candidates,
        },
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    retrieval = subparsers.add_parser("retrieve")
    retrieval.add_argument("--input", type=Path, required=True)
    retrieval.add_argument("--base-index", type=Path, required=True)
    retrieval.add_argument("--overlay-index", type=Path, action="append", default=[])
    retrieval.add_argument("--temporal-overlay-index", type=Path)
    retrieval.add_argument("--legal-entity-role-index", type=Path)
    retrieval.add_argument("--sector-official-index", type=Path, action="append", default=[])
    retrieval.add_argument("--sidecars", type=Path, required=True)
    retrieval.add_argument("--output", type=Path, required=True)
    retrieval.add_argument("--config", type=Path, default=ROOT / "config/retrieval_all_human_v11.json")
    retrieval.add_argument("--official-config", type=Path, required=True)
    retrieval.add_argument("--folds", default="2,3,4")
    retrieval.add_argument("--batch-size", type=int, default=512)
    retrieval.add_argument("--write-batch-rows", type=int, default=8192)
    retrieval.add_argument("--search-workers", type=int, default=4)
    labels = subparsers.add_parser("attach-labels")
    labels.add_argument("--candidates", type=Path, required=True)
    labels.add_argument("--labels", type=Path, required=True)
    labels.add_argument("--operational-labels", type=Path, required=True)
    labels.add_argument("--output", type=Path, required=True)
    combined = subparsers.add_parser("combine")
    combined.add_argument("--input", type=Path, action="append", required=True)
    combined.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "retrieve":
        output = retrieve(args)
    elif args.command == "attach-labels":
        output = attach_labels(args)
    else:
        output = combine(args)
    print(output)


if __name__ == "__main__":
    main()
