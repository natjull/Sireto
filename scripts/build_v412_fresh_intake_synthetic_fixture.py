#!/usr/bin/env python3
"""Build the preregistered V4.12 S0 synthetic intake fixture.

This producer is deliberately separate from the scanner.  It only writes
under an explicitly injected synthetic root and never reads CRM data.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import stat
from pathlib import Path
from typing import Any, Mapping

import pyarrow as pa
import pyarrow.parquet as pq


PLAN_DEFAULT = Path("config/v4_12_fresh_intake_synthetic_scanner_sealer_plan.json")
FORBIDDEN_COMPONENTS = {
    "data",
    "models",
    "reports",
    "challenges",
    "final_holdout_inputs",
    "fresh_holdout_intake",
    "fresh_holdout_evaluation_ledger",
    "registries",
}


class FixtureBuildError(RuntimeError):
    """Fail-closed fixture construction error."""


def _stop(message: str) -> None:
    raise FixtureBuildError(message)


def canonical_json_bytes(value: Any, *, final_lf: bool = True) -> bytes:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return raw + (b"\n" if final_lf else b"")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def opaque_digest(domain: str, payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        domain.encode("utf-8") + canonical_json_bytes(payload, final_lf=False)
    ).hexdigest()
    return "".join(chr(ord("a") + int(nibble, 16)) for nibble in digest)


def _load_plan(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    try:
        plan = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _stop(f"invalid plan: {exc}")
    if raw != canonical_json_bytes(plan):
        _stop("plan is not canonical JSON with one final LF")
    contract_path = Path(plan["contract"]["path"])
    if sha256_bytes(contract_path.read_bytes()) != plan["contract"]["sha256"]:
        _stop("contract pin mismatch")
    fixture_hash = sha256_bytes(
        canonical_json_bytes(plan["fixture"], final_lf=False)
    )
    if fixture_hash != plan["control_manifest"]["fixture_spec_sha256"]:
        _stop("fixture spec pin mismatch")
    return plan, raw


def _validate_root(root: Path) -> Path:
    if not root.is_absolute():
        _stop("synthetic root must be absolute")
    normalized = Path(os.path.abspath(os.fspath(root)))
    if any(part in FORBIDDEN_COMPONENTS for part in normalized.parts):
        _stop("synthetic root contains a forbidden real-data component")
    repository = Path(__file__).resolve().parents[1]
    try:
        normalized.relative_to(repository)
    except ValueError:
        pass
    else:
        _stop("synthetic root must be outside the repository")
    return normalized


def _mkdir_private(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=False)


def _prepare_empty_root(path: Path) -> None:
    if not path.exists():
        _mkdir_private(path)
        return
    info = path.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o700
        or any(path.iterdir())
    ):
        _stop("existing synthetic root must be empty, private 0700, and owned")


def _write_exclusive(path: Path, payload: bytes) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    fd = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            count = os.write(fd, view)
            if count <= 0:
                _stop(f"short write: {path}")
            view = view[count:]
        os.fchmod(fd, 0o600)
        os.fsync(fd)
    finally:
        os.close(fd)


def _arrow_type(name: str) -> pa.DataType:
    mapping = {
        "string": pa.string(),
        "int64": pa.int64(),
        "uint8": pa.uint8(),
    }
    try:
        return mapping[name]
    except KeyError:
        _stop(f"unsupported Arrow type: {name}")


def _empty_evidence_parquet(plan: Mapping[str, Any]) -> bytes:
    expected_version = plan["parquet_writer"]["pyarrow"]
    if pa.__version__ != expected_version:
        _stop(
            f"pyarrow version mismatch: {pa.__version__} != {expected_version}"
        )
    schema = pa.schema(
        [
            pa.field(name, _arrow_type(type_name), nullable=nullable)
            for name, type_name, nullable in plan["evidence_parquet"][
                "exact_schema"
            ]
        ]
    )
    table = pa.Table.from_arrays(
        [pa.array([], type=field.type) for field in schema],
        schema=schema,
    )
    output = io.BytesIO()
    writer = plan["parquet_writer"]
    pq.write_table(
        table,
        output,
        compression=writer["compression"],
        compression_level=writer["compression_level"],
        version=writer["format_version"],
        data_page_version=writer["data_page_version"],
        use_dictionary=writer["use_dictionary"],
        write_statistics=writer["write_statistics"],
        row_group_size=writer["row_group_size"],
        store_schema=writer["store_schema"],
    )
    payload = output.getvalue()
    metadata = pq.ParquetFile(io.BytesIO(payload)).metadata
    if metadata.num_rows != 0 or metadata.num_row_groups != 1:
        _stop("empty evidence Parquet physical shape mismatch")
    return payload


def _manifest_objects(
    plan: Mapping[str, Any],
    run_id: str,
    csv_bytes: bytes,
    evidence_bytes: bytes,
) -> dict[str, bytes]:
    fixture = plan["fixture"]
    provenance = fixture["common_provenance"]
    common = {
        "schema_version": (
            "sireto-v4.12-fresh-synthetic-collection-source-manifest-1"
        ),
        "synthetic_fixture": True,
        "synthetic_run_id": run_id,
        "collection_id": fixture["collection_id"],
        "export_snapshot_id": fixture["export_snapshot_id"],
        "reference_date": fixture["reference_date"],
        "population_name": fixture["population_name"],
        "population_definition": fixture["population_definition"],
        "period_start_utc": fixture["period_start_utc"],
        "period_end_utc": fixture["period_end_utc"],
        "export_cutoff_utc": fixture["export_cutoff_utc"],
        "portfolio_ids": [provenance["portfolio_id"]],
        "expected_batch_count": 1,
        "ordered_source_batch_ids": [provenance["source_batch_id"]],
        "expected_source_row_count": fixture["expected"]["source_row_count"],
        "inclusion_rule": "ALL_SOURCE_RECORDS_IN_FRAME",
        "producer_exclusions": [],
        "producer_manifest_id": fixture["producer_manifest_id"],
        "producer_created_at_utc": fixture["producer_created_at_utc"],
    }
    source = {
        "schema_version": "sireto-v4.12-fresh-synthetic-source-manifest-1",
        "synthetic_fixture": True,
        "synthetic_run_id": run_id,
        "collection_id": fixture["collection_id"],
        "export_snapshot_id": fixture["export_snapshot_id"],
        "source_batch_id": provenance["source_batch_id"],
        "source_system": provenance["source_system"],
        "portfolio_id": provenance["portfolio_id"],
        "reference_date": fixture["reference_date"],
        "period_start_utc": fixture["period_start_utc"],
        "period_end_utc": fixture["period_end_utc"],
        "export_cutoff_utc": fixture["export_cutoff_utc"],
        "source_filename": "crm_safe.csv",
        "source_format": "CSV",
        "source_row_count": fixture["expected"]["source_row_count"],
        "source_size_bytes": len(csv_bytes),
        "source_sha256": sha256_bytes(csv_bytes),
        "producer_manifest_id": fixture["producer_manifest_id"],
        "producer_created_at_utc": fixture["producer_created_at_utc"],
        "source_record_id_semantics": fixture["source_record_id_semantics"],
        "v411_service_id_equivalence_attested": fixture[
            "v411_service_id_equivalence_attested"
        ],
        "lineage_attestation_reference": fixture[
            "lineage_attestation_reference"
        ],
    }
    evidence = {
        "schema_version": (
            "sireto-v4.12-fresh-synthetic-evidence-source-manifest-1"
        ),
        "synthetic_fixture": True,
        "synthetic_run_id": run_id,
        "collection_id": fixture["collection_id"],
        "export_snapshot_id": fixture["export_snapshot_id"],
        "source_batch_id": provenance["source_batch_id"],
        "source_system": provenance["source_system"],
        "portfolio_id": provenance["portfolio_id"],
        "reference_date": fixture["reference_date"],
        "evidence_filename": "evidence_source.parquet",
        "evidence_row_count": 0,
        "evidence_size_bytes": len(evidence_bytes),
        "evidence_sha256": sha256_bytes(evidence_bytes),
        "authority_type": fixture["authority_type"],
        "producer_manifest_id": fixture["producer_manifest_id"],
        "producer_created_at_utc": fixture["producer_created_at_utc"],
    }
    return {
        "collection_source_manifest.json": canonical_json_bytes(common),
        "source_manifest.json": canonical_json_bytes(source),
        "crm_safe.csv": csv_bytes,
        "evidence_source_manifest.json": canonical_json_bytes(evidence),
        "evidence_source.parquet": evidence_bytes,
    }


def build_fixture(plan_path: Path, root: Path) -> dict[str, Any]:
    """Build one immutable synthetic inbox package and control manifest."""

    old_umask = os.umask(0o077)
    try:
        plan, plan_bytes = _load_plan(Path(plan_path))
        root = _validate_root(Path(root))
        fixture = plan["fixture"]
        plan_sha256 = sha256_bytes(plan_bytes)
        run_id = opaque_digest(
            plan["ids"]["run"]["domain"],
            {
                "fixture_spec_sha256": plan["control_manifest"][
                    "fixture_spec_sha256"
                ],
                "plan_sha256": plan_sha256,
            },
        )
        csv_bytes = fixture["csv"]["exact_utf8_text"].encode("utf-8")
        evidence_bytes = _empty_evidence_parquet(plan)
        payloads = _manifest_objects(
            plan, run_id, csv_bytes, evidence_bytes
        )

        package = root / "inbox" / run_id / "package"
        control_dir = root / "control" / run_id
        _prepare_empty_root(root)
        for path in (
            root / "inbox",
            root / "inbox" / run_id,
            package,
            root / "control",
            control_dir,
        ):
            _mkdir_private(path)
        for name in plan["input_package"]["exact_files"]:
            _write_exclusive(package / name, payloads[name])

        control = {
            "schema_version": plan["control_manifest"]["schema"],
            "synthetic_fixture": True,
            "fixture_spec_sha256": plan["control_manifest"][
                "fixture_spec_sha256"
            ],
            "synthetic_run_id": run_id,
            "logical_time_utc": fixture["logical_time_utc"],
            "batch_count": 1,
            "expected_source_row_count": fixture["expected"][
                "source_row_count"
            ],
            "producer_exclusions": [],
            "collection_source_manifest_sha256": sha256_bytes(
                payloads["collection_source_manifest.json"]
            ),
            "source_manifest_sha256": sha256_bytes(
                payloads["source_manifest.json"]
            ),
            "crm_safe_csv_sha256": sha256_bytes(payloads["crm_safe.csv"]),
            "evidence_source_manifest_sha256": sha256_bytes(
                payloads["evidence_source_manifest.json"]
            ),
            "evidence_source_parquet_sha256": sha256_bytes(
                payloads["evidence_source.parquet"]
            ),
        }
        control_bytes = canonical_json_bytes(control)
        control_path = control_dir / "fixture_control_manifest.json"
        _write_exclusive(control_path, control_bytes)
        for path in (
            package,
            package.parent,
            control_dir,
            control_dir.parent,
            root,
        ):
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        return {
            "synthetic_run_id": run_id,
            "package_path": str(package),
            "control_manifest_path": str(control_path),
            "fixture_control_manifest_sha256": sha256_bytes(control_bytes),
            "plan_sha256": plan_sha256,
        }
    finally:
        os.umask(old_umask)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN_DEFAULT)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = build_fixture(args.plan, args.root)
    print(canonical_json_bytes(result).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
