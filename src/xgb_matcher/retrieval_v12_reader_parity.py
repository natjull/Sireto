"""Independent v11-reader versus v12-mmap parity evidence.

The fold-fused PREFIT harness exercises scalar and vectorized orchestration,
but both paths intentionally share the fast mmap site reader.  This module
closes that independence gap: it compares the mmap publication with the
historical DuckDB/Parquet reader on a deterministic, label-blind sample and
publishes row-level evidence whose bytes are verified again by the gate
loader.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence
import hashlib
import json
import os
import re
import shutil
import tempfile

import pandas as pd
import pyarrow.parquet as pq

from .retrieval_fold_fused_oof_v12 import FoldFusedInvariantError, sha256_file
from .retrieval_sidecars_v11 import RetrievalSidecarReader, SiteQueryResult
from .retrieval_v12_mmap_sidecar import MMapLocalSitesReader, SITE_COLUMNS


READER_PARITY_SCHEMA_VERSION = "sireto-retrieval-v12-mmap-reader-parity-1"
READER_PARITY_ROW_SCHEMA_VERSION = (
    "sireto-retrieval-v12-mmap-reader-parity-rows-1"
)
READER_PARITY_CASE_COUNT = 64
READER_PARITY_CHECKS = (
    "source_manifest_hash_bound",
    "source_local_sites_hash_bound",
    "source_and_mmap_row_count_identical",
    "sampled_source_rows_byte_identical",
    "historical_and_mmap_results_byte_identical",
    "historical_and_mmap_counts_identical",
    "historical_and_mmap_order_identical",
)
_POSTCODE_RE = re.compile(r"^\d{5}$")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _result_payload(result: SiteQueryResult) -> dict[str, Any]:
    return {
        "strategy": result.strategy,
        "matched_count": int(result.matched_count),
        "max_sites": int(result.max_sites),
        "capped": bool(result.capped),
        "sites": [asdict(site) for site in result.sites],
    }


def _batch_payload(
    sirens: Sequence[str], results: Mapping[str, SiteQueryResult]
) -> list[dict[str, Any]]:
    return [
        {"siren": siren, "result": _result_payload(results[siren])}
        for siren in sirens
    ]


def _payload_counts(payload: Sequence[Mapping[str, Any]]) -> list[list[Any]]:
    return [
        [str(item["siren"]), int(item["result"]["matched_count"])]
        for item in payload
    ]


def _payload_order(payload: Sequence[Mapping[str, Any]]) -> list[list[Any]]:
    return [
        [
            str(item["siren"]),
            [str(site["siret"]) for site in item["result"]["sites"]],
        ]
        for item in payload
    ]


def _mmap_row(reader: MMapLocalSitesReader, position: int) -> dict[str, Any]:
    table = reader._slice_global(int(position), 1)  # exact physical-order audit
    if table.num_rows != 1:
        raise FoldFusedInvariantError("mmap parity could not resolve sampled row")
    return table.select(list(SITE_COLUMNS)).to_pylist()[0]


def _sample_positions_and_rows(
    reader: MMapLocalSitesReader,
    *,
    row_count: int,
    case_count: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    if not 1 <= case_count <= row_count:
        raise FoldFusedInvariantError("reader parity sample cardinality is invalid")
    positions: list[int] = []
    rows: list[dict[str, Any]] = []
    occupied: set[int] = set()
    for index in range(case_count):
        start = ((2 * index + 1) * row_count) // (2 * case_count)
        position = start
        while True:
            if position not in occupied:
                row = _mmap_row(reader, position)
                insee = str(row.get("insee") or "")
                postcode = str(row.get("postcode") or "")
                if insee or _POSTCODE_RE.fullmatch(postcode) is not None:
                    occupied.add(position)
                    positions.append(position)
                    rows.append(row)
                    break
            position = (position + 1) % row_count
            if position == start:
                raise FoldFusedInvariantError(
                    "reader parity could not find enough geocoded official sites"
                )
    return positions, rows


def _source_rows_at_positions(
    local_sites_path: Path, positions: Sequence[int]
) -> list[dict[str, Any]]:
    parquet = pq.ParquetFile(local_sites_path)
    row_group_starts: list[int] = []
    total = 0
    for index in range(parquet.num_row_groups):
        row_group_starts.append(total)
        total += parquet.metadata.row_group(index).num_rows
    grouped: dict[int, list[tuple[int, int]]] = {}
    for output_index, position in enumerate(positions):
        # The sample is tiny (64); a linear metadata lookup is deterministic
        # and avoids reading the full 43.5M-row source again.
        row_group = max(
            index
            for index, start in enumerate(row_group_starts)
            if start <= int(position)
        )
        grouped.setdefault(row_group, []).append((output_index, int(position)))
    output: list[dict[str, Any] | None] = [None] * len(positions)
    for row_group, wanted in sorted(grouped.items()):
        table = parquet.read_row_group(row_group, columns=list(SITE_COLUMNS))
        start = row_group_starts[row_group]
        for output_index, position in wanted:
            output[output_index] = table.slice(position - start, 1).to_pylist()[0]
    if any(value is None for value in output):
        raise FoldFusedInvariantError("reader parity source sampling was incomplete")
    return [dict(value) for value in output if value is not None]


def run_mmap_reader_parity_v12(
    *,
    sidecar_dir: Path,
    mmap_dir: Path,
    output_dir: Path,
    case_count: int = READER_PARITY_CASE_COUNT,
) -> Path:
    """Publish independent, label-blind historical-reader parity evidence."""

    sidecar_dir = Path(sidecar_dir).resolve()
    mmap_dir = Path(mmap_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    official_manifest_path = sidecar_dir / "manifest.json"
    mmap_manifest_path = mmap_dir / "manifest.json"
    local_sites_path = sidecar_dir / "siren_local_sites.parquet"
    if not all(
        path.is_file()
        for path in (official_manifest_path, mmap_manifest_path, local_sites_path)
    ):
        raise FileNotFoundError("reader parity input publication is incomplete")
    official_manifest = json.loads(official_manifest_path.read_text(encoding="utf-8"))
    mmap_manifest = json.loads(mmap_manifest_path.read_text(encoding="utf-8"))
    local_record = (official_manifest.get("files") or {}).get(
        "siren_local_sites.parquet"
    ) or {}
    official_sha = sha256_file(official_manifest_path)
    local_sha = sha256_file(local_sites_path)
    mmap_sha = sha256_file(mmap_manifest_path)
    source_rows = pq.ParquetFile(local_sites_path).metadata.num_rows
    lineage_checks = {
        "source_manifest_hash_bound": (
            mmap_manifest.get("source_manifest_sha256") == official_sha
        ),
        "source_local_sites_hash_bound": (
            local_record.get("sha256") == local_sha
            and mmap_manifest.get("source_local_sites_sha256") == local_sha
        ),
        "source_and_mmap_row_count_identical": (
            int(mmap_manifest.get("site_rows", -1)) == source_rows
            and int(mmap_manifest.get("source_rows", -1)) == source_rows
        ),
    }
    if not all(lineage_checks.values()):
        raise FoldFusedInvariantError("mmap reader parity input lineage drifted")

    with RetrievalSidecarReader(sidecar_dir, verify_hashes=True) as historical, MMapLocalSitesReader(
        mmap_dir, verify_hashes=True
    ) as mmap_reader:
        positions, mmap_rows = _sample_positions_and_rows(
            mmap_reader,
            row_count=source_rows,
            case_count=int(case_count),
        )
        source_rows_sample = _source_rows_at_positions(local_sites_path, positions)
        records: list[dict[str, Any]] = []
        sampled_parity: list[bool] = []
        count_parity: list[bool] = []
        order_parity: list[bool] = []
        source_parity: list[bool] = []
        sampled_sirens = [str(row["siren"]) for row in mmap_rows]
        for index, (position, mmap_row, source_row) in enumerate(
            zip(positions, mmap_rows, source_rows_sample, strict=True)
        ):
            sirens = tuple(
                dict.fromkeys(
                    sampled_sirens[(index + offset) % len(sampled_sirens)]
                    for offset in range(min(8, len(sampled_sirens)))
                )
            )
            include_active = index % 3 != 2
            include_closed = index % 3 != 1
            max_sites = (1, 2, 8, 32)[index % 4]
            postcode = str(mmap_row.get("postcode") or "")
            if _POSTCODE_RE.fullmatch(postcode) is None:
                postcode = ""
            request = {
                "sirens": list(sirens),
                "insee": str(mmap_row.get("insee") or ""),
                "postcode": postcode,
                "address": str(mmap_row.get("current_address_normalized") or ""),
                "street_number": str(mmap_row.get("street_number") or ""),
                "street": str(mmap_row.get("street_normalized") or ""),
                "include_active": include_active,
                "include_closed": include_closed,
                "max_sites_per_siren": max_sites,
            }
            historical_result = _batch_payload(
                sirens, historical.sites_by_sirens_geo(**request)
            )
            mmap_result = _batch_payload(
                sirens, mmap_reader.sites_by_sirens_geo(**request)
            )
            historical_json = _canonical_json(historical_result)
            mmap_json = _canonical_json(mmap_result)
            source_json = _canonical_json(source_row)
            mmap_row_json = _canonical_json(mmap_row)
            result_equal = historical_json == mmap_json
            counts_equal = _payload_counts(historical_result) == _payload_counts(
                mmap_result
            )
            order_equal = _payload_order(historical_result) == _payload_order(
                mmap_result
            )
            row_equal = source_json == mmap_row_json
            sampled_parity.append(result_equal)
            count_parity.append(counts_equal)
            order_parity.append(order_equal)
            source_parity.append(row_equal)
            records.append(
                {
                    "schema_version": READER_PARITY_ROW_SCHEMA_VERSION,
                    "case_index": index,
                    "source_row_index": int(position),
                    "request_json": _canonical_json(request),
                    "source_row_json": source_json,
                    "mmap_row_json": mmap_row_json,
                    "source_row_sha256": _digest_text(source_json),
                    "mmap_row_sha256": _digest_text(mmap_row_json),
                    "historical_result_json": historical_json,
                    "mmap_result_json": mmap_json,
                    "historical_result_sha256": _digest_text(historical_json),
                    "mmap_result_sha256": _digest_text(mmap_json),
                    "source_row_byte_identical": row_equal,
                    "result_byte_identical": result_equal,
                    "count_identical": counts_equal,
                    "order_identical": order_equal,
                }
            )

    checks = {
        **lineage_checks,
        "sampled_source_rows_byte_identical": all(source_parity),
        "historical_and_mmap_results_byte_identical": all(sampled_parity),
        "historical_and_mmap_counts_identical": all(count_parity),
        "historical_and_mmap_order_identical": all(order_parity),
    }
    if not all(checks.values()):
        raise FoldFusedInvariantError("historical-reader/mmap parity divergence")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        rows_path = stage / "reader_parity_rows.parquet"
        pd.DataFrame(records).to_parquet(rows_path, index=False)
        manifest = {
            "schema_version": READER_PARITY_SCHEMA_VERSION,
            "status": "PASS",
            "case_count": int(case_count),
            "sample_rule": "MMAP_GLOBAL_EVEN_OFFSETS_WITH_GEO_THEN_SOURCE_ROW_CHECK",
            "human_labels_opened": False,
            "crm_data_opened": False,
            "fold0_opened": False,
            "fold1_opened": False,
            "checks": checks,
            "lineage": {
                "official_sidecar_manifest_sha256": official_sha,
                "source_local_sites_sha256": local_sha,
                "source_rows": source_rows,
                "mmap_sidecar_manifest_sha256": mmap_sha,
                "mmap_build_id": mmap_manifest.get("build_id"),
                "mmap_rows": int(mmap_manifest.get("site_rows", -1)),
            },
            "artifacts": {
                "reader_parity_rows": {
                    "path": rows_path.name,
                    "sha256": sha256_file(rows_path),
                    "rows": len(records),
                }
            },
        }
        manifest_path = stage / "manifest.json"
        manifest_path.write_text(_canonical_json(manifest) + "\n", encoding="utf-8")
        os.rename(stage, output_dir)
        return output_dir / "manifest.json"
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def load_mmap_reader_parity_evidence_v12(
    path: Path,
    *,
    expected_official_manifest_sha256: str,
    expected_mmap_manifest_sha256: str,
    expected_source_local_sites_sha256: str,
    expected_source_rows: int,
    expected_mmap_build_id: str,
    expected_case_count: int = READER_PARITY_CASE_COUNT,
) -> tuple[Mapping[str, Any], str]:
    """Load and independently recompute a sealed reader-parity artifact."""

    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version": READER_PARITY_SCHEMA_VERSION,
        "status": "PASS",
        "case_count": int(expected_case_count),
        "sample_rule": "MMAP_GLOBAL_EVEN_OFFSETS_WITH_GEO_THEN_SOURCE_ROW_CHECK",
        "human_labels_opened": False,
        "crm_data_opened": False,
        "fold0_opened": False,
        "fold1_opened": False,
    }
    if any(raw.get(key) != value for key, value in required.items()):
        raise FoldFusedInvariantError("mmap reader parity claims drifted")
    lineage = raw.get("lineage")
    if (
        not isinstance(lineage, Mapping)
        or lineage.get("official_sidecar_manifest_sha256")
        != expected_official_manifest_sha256
        or lineage.get("mmap_sidecar_manifest_sha256")
        != expected_mmap_manifest_sha256
        or lineage.get("source_local_sites_sha256")
        != expected_source_local_sites_sha256
        or lineage.get("source_rows") != int(expected_source_rows)
        or lineage.get("mmap_rows") != int(expected_source_rows)
        or lineage.get("mmap_build_id") != expected_mmap_build_id
    ):
        raise FoldFusedInvariantError("mmap reader parity lineage mismatch")
    artifacts = raw.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {"reader_parity_rows"}:
        raise FoldFusedInvariantError("mmap reader parity artifact set drifted")
    record = artifacts["reader_parity_rows"]
    if not isinstance(record, Mapping) or record.get("path") != "reader_parity_rows.parquet":
        raise FoldFusedInvariantError("mmap reader parity artifact record drifted")
    rows_path = (path.parent / "reader_parity_rows.parquet").resolve()
    if rows_path.parent != path.parent or not rows_path.is_file():
        raise FoldFusedInvariantError("mmap reader parity row artifact is missing")
    if record.get("sha256") != sha256_file(rows_path):
        raise FoldFusedInvariantError("mmap reader parity row hash mismatch")
    rows = pd.read_parquet(rows_path)
    if record.get("rows") != expected_case_count or len(rows) != expected_case_count:
        raise FoldFusedInvariantError("mmap reader parity row count mismatch")
    required_columns = {
        "schema_version",
        "case_index",
        "source_row_index",
        "request_json",
        "source_row_json",
        "mmap_row_json",
        "source_row_sha256",
        "mmap_row_sha256",
        "historical_result_json",
        "mmap_result_json",
        "historical_result_sha256",
        "mmap_result_sha256",
        "source_row_byte_identical",
        "result_byte_identical",
        "count_identical",
        "order_identical",
    }
    if set(rows.columns) != required_columns:
        raise FoldFusedInvariantError("mmap reader parity row schema drifted")
    if (
        not rows["schema_version"].eq(READER_PARITY_ROW_SCHEMA_VERSION).all()
        or rows["case_index"].astype(int).tolist() != list(range(expected_case_count))
        or rows["source_row_index"].duplicated().any()
    ):
        raise FoldFusedInvariantError("mmap reader parity row identity drifted")
    recomputed = {
        "source_manifest_hash_bound": (
            lineage["official_sidecar_manifest_sha256"]
            == expected_official_manifest_sha256
        ),
        "source_local_sites_hash_bound": (
            lineage["source_local_sites_sha256"]
            == expected_source_local_sites_sha256
        ),
        "source_and_mmap_row_count_identical": (
            int(lineage["source_rows"])
            == int(lineage["mmap_rows"])
            == int(expected_source_rows)
        ),
        "sampled_source_rows_byte_identical": True,
        "historical_and_mmap_results_byte_identical": True,
        "historical_and_mmap_counts_identical": True,
        "historical_and_mmap_order_identical": True,
    }
    for row in rows.itertuples(index=False):
        try:
            source = json.loads(row.source_row_json)
            mmap_row = json.loads(row.mmap_row_json)
            historical = json.loads(row.historical_result_json)
            mmap_result = json.loads(row.mmap_result_json)
            json.loads(row.request_json)
        except (TypeError, json.JSONDecodeError) as error:
            raise FoldFusedInvariantError("invalid reader parity row JSON") from error
        source_json = _canonical_json(source)
        mmap_row_json = _canonical_json(mmap_row)
        historical_json = _canonical_json(historical)
        mmap_json = _canonical_json(mmap_result)
        row_source_equal = source_json == mmap_row_json
        row_result_equal = historical_json == mmap_json
        row_count_equal = _payload_counts(historical) == _payload_counts(mmap_result)
        row_order_equal = _payload_order(historical) == _payload_order(mmap_result)
        digest_claims = (
            row.source_row_sha256 == _digest_text(source_json),
            row.mmap_row_sha256 == _digest_text(mmap_row_json),
            row.historical_result_sha256 == _digest_text(historical_json),
            row.mmap_result_sha256 == _digest_text(mmap_json),
        )
        boolean_claims = (
            row.source_row_byte_identical is True,
            row.result_byte_identical is True,
            row.count_identical is True,
            row.order_identical is True,
        )
        if not all(digest_claims) or not all(boolean_claims):
            raise FoldFusedInvariantError("mmap reader parity row claims drifted")
        recomputed["sampled_source_rows_byte_identical"] &= row_source_equal
        recomputed["historical_and_mmap_results_byte_identical"] &= row_result_equal
        recomputed["historical_and_mmap_counts_identical"] &= row_count_equal
        recomputed["historical_and_mmap_order_identical"] &= row_order_equal
    checks = raw.get("checks")
    if not isinstance(checks, Mapping) or set(checks) != set(READER_PARITY_CHECKS):
        raise FoldFusedInvariantError("mmap reader parity check schema drifted")
    for name in READER_PARITY_CHECKS:
        observed = recomputed[name]
        if checks.get(name) is not True or observed is not True:
            raise FoldFusedInvariantError(f"mmap reader parity check failed: {name}")
    return raw, sha256_file(path)


__all__ = [
    "READER_PARITY_CASE_COUNT",
    "READER_PARITY_CHECKS",
    "READER_PARITY_SCHEMA_VERSION",
    "load_mmap_reader_parity_evidence_v12",
    "run_mmap_reader_parity_v12",
]
