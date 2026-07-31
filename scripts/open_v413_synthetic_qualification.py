#!/usr/bin/env python3
"""One-shot V4.13 Gate 0B runner for synthetic collections only.

The runner binds the Gate 0A claim to its ledger and collection, durably
creates the payload-opening marker before either payload FD exists, reads each
payload through one O_NOFOLLOW FD, and publishes a terminal receipt only after
the separated qualification artifacts are durable.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from scripts import audit_v413_fresh_source_availability as gate0a
    from scripts.build_v413_fresh_qualification import (
        CRM_COLUMNS,
        MAPPING_COLUMNS,
        QualificationError,
        qualify_fixture_rows,
        write_fixture_outputs,
    )
    from scripts.audit_v413_synthetic_contamination import (
        audit_synthetic_contamination,
    )
    from scripts.seal_v413_fresh_splits import build_manifests, seal_manifests
    from scripts.validate_v413_fresh_artifacts import validate_artifacts
except ModuleNotFoundError:  # Direct ``python scripts/...py`` execution.
    import audit_v413_fresh_source_availability as gate0a
    from build_v413_fresh_qualification import (
        CRM_COLUMNS,
        MAPPING_COLUMNS,
        QualificationError,
        qualify_fixture_rows,
        write_fixture_outputs,
    )
    from audit_v413_synthetic_contamination import audit_synthetic_contamination
    from seal_v413_fresh_splits import build_manifests, seal_manifests
    from validate_v413_fresh_artifacts import validate_artifacts


MARKER_FILENAME = "payload_opening.json"
RECEIPT_FILENAME = "qualification_receipt.json"
MARKER_SCHEMA = "sireto-v4.13-payload-opening-synthetic-1"
RECEIPT_SCHEMA = "sireto-v4.13-qualification-receipt-synthetic-1"

class Gate0BStop(RuntimeError):
    """Fail-closed synthetic Gate 0B error."""


def _stop(reason: str) -> None:
    raise Gate0BStop(f"STOP_V413_SYNTHETIC_GATE_0B: {reason}")


def _canonical_json(value: Any) -> bytes:
    return gate0a.canonical_json(value)


def _tmp_path(path: Path, *, must_exist: bool, label: str) -> Path:
    temporary = Path(tempfile.gettempdir()).resolve()
    try:
        resolved = path.resolve(strict=must_exist)
    except OSError:
        _stop(f"{label}_RESOLVE")
    if resolved == temporary or temporary not in resolved.parents:
        _stop(f"{label}_OUTSIDE_OS_TMP")
    if path.is_symlink():
        _stop(f"{label}_SYMLINK")
    return resolved


def _read_fd(fd: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_uid,
        value.st_nlink,
        stat.S_IMODE(value.st_mode),
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_payload_once(
    collection_fd: int,
    filename: str,
    *,
    expected_size: int,
    expected_sha256: str,
    expected_uid: int,
    label: str,
) -> tuple[bytes, int, tuple[int, ...]]:
    try:
        fd = os.open(
            filename,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=collection_fd,
        )
    except OSError:
        _stop(f"{label}_OPEN")
    try:
        before = os.fstat(fd)
        gate0a._regular_metadata(before, expected_uid, label)
        raw = _read_fd(fd)
        after = os.fstat(fd)
        if _identity(before) != _identity(after):
            _stop(f"{label}_CHANGED_DURING_READ")
        if len(raw) != expected_size or before.st_size != expected_size:
            _stop(f"{label}_SIZE")
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            _stop(f"{label}_SHA256")
        return raw, fd, _identity(before)
    except Exception:
        os.close(fd)
        raise


def _csv_rows(raw: bytes, columns: Sequence[str], label: str) -> list[dict[str, Any]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        _stop(f"{label}_UTF8")
    try:
        reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
        if reader.fieldnames != list(columns):
            _stop(f"{label}_SCHEMA")
        rows: list[dict[str, Any]] = []
        for row in reader:
            if None in row or set(row) != set(columns):
                _stop(f"{label}_ROW_SCHEMA")
            rows.append(dict(row))
    except csv.Error:
        _stop(f"{label}_CSV")
    return rows


def _decode_crm(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    decoded: list[dict[str, Any]] = []
    nullable = {
        "source_group_id",
        "crm_name_raw",
        "crm_address_raw",
        "crm_postcode_raw",
        "crm_city_raw",
        "crm_insee_raw",
    }
    for row in rows:
        value: dict[str, Any] = dict(row)
        for field in nullable:
            if value[field] == "":
                value[field] = None
        if value["source_record_id_equivalence_attested"] == "true":
            value["source_record_id_equivalence_attested"] = True
        elif value["source_record_id_equivalence_attested"] == "false":
            value["source_record_id_equivalence_attested"] = False
        else:
            _stop("CRM_BOOLEAN")
        decoded.append(value)
    return decoded


def _decode_mapping(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    decoded: list[dict[str, Any]] = []
    nullable = {
        "authoritative_siret",
        "authoritative_siren",
        "valid_from",
        "valid_to",
    }
    for row in rows:
        value: dict[str, Any] = dict(row)
        for field in nullable:
            if value[field] == "":
                value[field] = None
        if value["matching_pipeline_used"] == "false":
            value["matching_pipeline_used"] = False
        elif value["matching_pipeline_used"] == "true":
            value["matching_pipeline_used"] = True
        else:
            _stop("MAPPING_BOOLEAN")
        decoded.append(value)
    return decoded


def _read_manifest(
    collection_fd: int,
    directory_name: str,
    bundle: gate0a.ControlBundle,
    expected_uid: int,
) -> tuple[dict[str, Any], bytes]:
    try:
        fd = os.open(
            "collection_manifest.json",
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=collection_fd,
        )
    except OSError:
        _stop("COLLECTION_MANIFEST_OPEN")
    try:
        before = os.fstat(fd)
        gate0a._regular_metadata(before, expected_uid, "COLLECTION_MANIFEST")
        raw = _read_fd(fd)
        after = os.fstat(fd)
        if _identity(before) != _identity(after):
            _stop("COLLECTION_MANIFEST_CHANGED")
    finally:
        os.close(fd)
    manifest = gate0a._parse_canonical_object(raw, "COLLECTION_MANIFEST")
    gate0a._validate_manifest(
        manifest,
        raw,
        directory_name=directory_name,
        plan=bundle.plan,
        bundle=bundle,
    )
    return manifest, raw


def _validate_claim_binding(
    claim: Mapping[str, Any],
    claim_raw: bytes,
    ledger: Mapping[str, Any],
    ledger_raw: bytes,
) -> Mapping[str, Any]:
    if hashlib.sha256(ledger_raw).hexdigest() != claim["availability_ledger_sha256"]:
        _stop("CLAIM_LEDGER_SHA256")
    if (
        claim["directory_name"] != ledger["selected_directory_name"]
        or claim["collection_manifest_sha256"] != ledger["selected_manifest_sha256"]
    ):
        _stop("CLAIM_LEDGER_SELECTION")
    match = gate0a.DIRECTORY_PATTERN.fullmatch(claim["directory_name"])
    if (
        match is None
        or int(match.group(1)) != claim["arrival_epoch_ns"]
        or match.group(2) != claim["collection_manifest_sha256"]
    ):
        _stop("CLAIM_DIRECTORY_BINDING")
    records = [
        row
        for row in ledger["observed_manifests"]
        if row["directory_name"] == claim["directory_name"]
        and row["manifest_sha256"] == claim["collection_manifest_sha256"]
    ]
    if len(records) != 1 or records[0]["collection_id"] != claim["collection_id"]:
        _stop("CLAIM_COLLECTION_BINDING")
    return {
        "claim_sha256": hashlib.sha256(claim_raw).hexdigest(),
        "ledger_sha256": hashlib.sha256(ledger_raw).hexdigest(),
    }


OUTPUT_FILES = {
    "queries": Path("queries/queries.csv"),
    "oracle": Path("oracle/oracle.csv"),
    "audit": Path("audit/qualification.json"),
    "contamination": Path("audit/contamination.json"),
    "private_split_rows": Path("audit/private_split_rows.jsonl"),
    "split_fit": Path("splits/fit/split_manifest.json"),
    "split_dev": Path("splits/dev/split_manifest.json"),
    "split_test": Path("splits/test/split_manifest.json"),
}

OUTPUT_DIRECTORIES = {
    Path("."): {"queries", "oracle", "audit", "splits"},
    Path("queries"): {"queries.csv"},
    Path("oracle"): {"oracle.csv"},
    Path("audit"): {
        "qualification.json",
        "contamination.json",
        "private_split_rows.jsonl",
    },
    Path("splits"): {"fit", "dev", "test"},
    Path("splits/fit"): {"split_manifest.json"},
    Path("splits/dev"): {"split_manifest.json"},
    Path("splits/test"): {"split_manifest.json"},
}


def _retain_valid_output_tree(
    output_root: Path,
    expected_uid: int,
) -> list[tuple[int, tuple[int, ...]]]:
    retained: list[tuple[int, tuple[int, ...]]] = []
    try:
        for suffix, expected_entries in OUTPUT_DIRECTORIES.items():
            fd = os.open(
                output_root / suffix,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            )
            retained_here = False
            try:
                metadata = os.fstat(fd)
                label = (
                    "OUTPUT_ROOT"
                    if suffix == Path(".")
                    else f"OUTPUT_{str(suffix).replace('/', '_').upper()}_DIR"
                )
                gate0a._directory_metadata(metadata, expected_uid, label)
                retained.append((fd, _identity(metadata)))
                retained_here = True
                if set(os.listdir(fd)) != expected_entries:
                    _stop(f"{label}_ENTRIES")
            except Exception:
                if not retained_here:
                    os.close(fd)
                raise
        return retained
    except Exception:
        for fd, _ in retained:
            os.close(fd)
        raise


def _artifact_hashes(output_root: Path, expected_uid: int) -> dict[str, str]:
    _retained_directories = _retain_valid_output_tree(output_root, expected_uid)
    try:
        hashes: dict[str, str] = {}
        for name, suffix in OUTPUT_FILES.items():
            path = output_root / suffix
            try:
                fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
            except OSError:
                _stop(f"OUTPUT_{name.upper()}_READ")
            try:
                metadata = os.fstat(fd)
                gate0a._regular_metadata(
                    metadata, expected_uid, f"OUTPUT_{name.upper()}"
                )
                raw = _read_fd(fd)
            finally:
                os.close(fd)
            hashes[name] = hashlib.sha256(raw).hexdigest()
        return hashes
    finally:
        for fd, _ in _retained_directories:
            os.close(fd)


def _fsync_outputs(output_root: Path) -> None:
    files = (
        output_root / "queries/queries.csv",
        output_root / "oracle/oracle.csv",
        output_root / "audit/qualification.json",
        output_root / "audit/contamination.json",
        output_root / "audit/private_split_rows.jsonl",
        output_root / "splits/fit/split_manifest.json",
        output_root / "splits/dev/split_manifest.json",
        output_root / "splits/test/split_manifest.json",
    )
    for path in files:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    for path in (
        output_root / "queries",
        output_root / "oracle",
        output_root / "audit",
        output_root / "splits/fit",
        output_root / "splits/dev",
        output_root / "splits/test",
        output_root / "splits",
        output_root,
        output_root.parent,
    ):
        fd = os.open(
            path,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _read_receipt(control_fd: int) -> dict[str, Any] | None:
    try:
        os.stat(RECEIPT_FILENAME, dir_fd=control_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError:
        _stop("RECEIPT_STAT")
    receipt, _ = gate0a._read_control(control_fd, RECEIPT_FILENAME, "RECEIPT")
    expected = {
        "schema_version",
        "stage",
        "synthetic_only",
        "terminal_state",
        "claim_sha256",
        "availability_ledger_sha256",
        "payload_opening_sha256",
        "collection_manifest_sha256",
        "crm_sha256",
        "mapping_sha256",
        "output_root",
        "output_hashes",
        "counts",
    }
    hash_fields = {
        "claim_sha256",
        "availability_ledger_sha256",
        "payload_opening_sha256",
        "collection_manifest_sha256",
        "crm_sha256",
        "mapping_sha256",
    }
    if (
        set(receipt) != expected
        or receipt["schema_version"] != RECEIPT_SCHEMA
        or receipt["stage"] != "GATE_0B"
        or receipt["synthetic_only"] is not True
        or receipt["terminal_state"] != "QUALIFICATION_SEALED"
        or any(
            not isinstance(receipt[field], str)
            or gate0a.HEX64.fullmatch(receipt[field]) is None
            for field in hash_fields
        )
        or not isinstance(receipt["output_root"], str)
        or not isinstance(receipt["output_hashes"], dict)
        or set(receipt["output_hashes"])
        != {
            "queries",
            "oracle",
            "audit",
            "contamination",
            "private_split_rows",
            "split_fit",
            "split_dev",
            "split_test",
        }
        or any(
            not isinstance(value, str) or gate0a.HEX64.fullmatch(value) is None
            for value in receipt["output_hashes"].values()
        )
        or not isinstance(receipt["counts"], dict)
    ):
        _stop("RECEIPT_POLICY")
    return receipt


def _paths_overlap(left: Path, right: Path) -> bool:
    return (
        left == right
        or left in right.parents
        or right in left.parents
    )


def _read_written_artifacts(
    raw_by_name: Mapping[str, bytes],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    query_raw = raw_by_name["queries"]
    oracle_raw = raw_by_name["oracle"]
    queries = _csv_rows(query_raw, [
        "query_id", "reference_date", "crm_name_raw", "crm_address_raw",
        "crm_postcode_raw", "crm_city_raw", "crm_insee_raw",
    ], "WRITTEN_QUERIES")
    oracle_rows = _csv_rows(oracle_raw, [
        "query_id", "label", "authoritative_siret", "authoritative_siren",
        "reason_code", "evidence_count", "evidence_payload_sha256s",
    ], "WRITTEN_ORACLE")
    oracle: list[dict[str, Any]] = []
    for row in oracle_rows:
        decoded = dict(row)
        decoded["authoritative_siret"] = decoded["authoritative_siret"] or None
        decoded["authoritative_siren"] = decoded["authoritative_siren"] or None
        try:
            decoded["evidence_count"] = int(decoded["evidence_count"])
            decoded["evidence_payload_sha256s"] = json.loads(
                decoded["evidence_payload_sha256s"]
            )
        except (ValueError, TypeError, json.JSONDecodeError):
            _stop("WRITTEN_ORACLE_ENCODING")
        oracle.append(decoded)
    split_rows: list[dict[str, Any]] = []
    for line in raw_by_name["private_split_rows"].decode("utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            _stop("WRITTEN_SPLIT_ROW")
        split_rows.append(value)
    return queries, oracle, split_rows


def _validate_final_outputs(
    output_root: Path,
    *,
    expected_counts: Mapping[str, Any],
    expected_contamination: Mapping[str, Any],
    expected_uid: int,
) -> tuple[dict[str, str], list[tuple[int, tuple[int, ...]]]]:
    relative = {
        "queries": Path("queries/queries.csv"),
        "oracle": Path("oracle/oracle.csv"),
        "audit": Path("audit/qualification.json"),
        "contamination": Path("audit/contamination.json"),
        "private_split_rows": Path("audit/private_split_rows.jsonl"),
        "split_fit": Path("splits/fit/split_manifest.json"),
        "split_dev": Path("splits/dev/split_manifest.json"),
        "split_test": Path("splits/test/split_manifest.json"),
    }
    raw_by_name: dict[str, bytes] = {}
    retained: list[tuple[int, tuple[int, ...]]] = []
    try:
        for name, suffix in relative.items():
            fd = os.open(
                output_root / suffix,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            )
            retained_here = False
            try:
                metadata = os.fstat(fd)
                gate0a._regular_metadata(
                    metadata, expected_uid, f"FINAL_{name.upper()}"
                )
                retained.append((fd, _identity(metadata)))
                retained_here = True
                raw_by_name[name] = _read_fd(fd)
            except Exception:
                if not retained_here:
                    os.close(fd)
                raise
    except Exception:
        for fd, _ in retained:
            os.close(fd)
        raise
    try:
        queries, oracle, split_rows = _read_written_artifacts(raw_by_name)
        validate_artifacts(queries, oracle)
        audit = json.loads(raw_by_name["audit"])
        expected_audit = {
            "schema_version": "sireto-v4.13-synthetic-qualification-audit-1",
            "synthetic_fixtures_only": True,
            "counts": dict(expected_counts),
        }
        if audit != expected_audit:
            _stop("FINAL_AUDIT_COUNTS")
        contamination = json.loads(raw_by_name["contamination"])
        if contamination != expected_contamination:
            _stop("FINAL_CONTAMINATION")
        expected_splits = build_manifests(split_rows)
        for split, expected in expected_splits.items():
            observed = json.loads(raw_by_name[f"split_{split}"])
            if observed != expected:
                _stop(f"FINAL_SPLIT_{split.upper()}")
        hashes = {
            name: hashlib.sha256(raw).hexdigest()
            for name, raw in raw_by_name.items()
        }
        return hashes, retained
    except Exception:
        for fd, _ in retained:
            os.close(fd)
        raise


def open_synthetic_qualification(
    *,
    inbox: Path,
    control_root: Path,
    output_root: Path,
    authority_catalog: Mapping[str, Any],
    contamination_keysets: Mapping[str, set[str]],
    synthetic_hmac_key: bytes,
    synthetic_only: bool,
    crash_stage: str | None = None,
    repository: Path = gate0a.REPOSITORY,
    plan_path: Path | None = None,
    lock_path: Path | None = None,
) -> dict[str, Any]:
    """Execute synthetic Gate 0B exactly once, or return its sealed receipt."""

    if synthetic_only is not True:
        _stop("REAL_PAYLOAD_OPEN_FORBIDDEN")
    inbox = _tmp_path(inbox, must_exist=True, label="INBOX")
    control_root = _tmp_path(control_root, must_exist=True, label="CONTROL_ROOT")
    output_root = _tmp_path(output_root, must_exist=False, label="OUTPUT_ROOT")
    for left, right in (
        (inbox, control_root),
        (inbox, output_root),
        (control_root, output_root),
    ):
        if _paths_overlap(left, right):
            _stop("ROOTS_NOT_PAIRWISE_DISJOINT")
    if crash_stage not in {None, "after_marker"}:
        _stop("CRASH_STAGE")

    bundle = gate0a._validate_bundle(
        repository,
        plan_path or repository / gate0a.PLAN_PATH.relative_to(gate0a.REPOSITORY),
        lock_path or repository / gate0a.LOCK_PATH.relative_to(gate0a.REPOSITORY),
    )
    expected_uid = bundle.lock["uid"]
    control_fd = gate0a._open_directory(control_root, expected_uid, "CONTROL_ROOT")
    retained_payload_fds: list[tuple[int, tuple[int, ...]]] = []
    retained_output_fds: list[tuple[int, tuple[int, ...]]] = []
    try:
        fcntl.flock(control_fd, fcntl.LOCK_EX)
        receipt = _read_receipt(control_fd)
        if receipt is not None:
            if Path(receipt["output_root"]) != output_root:
                _stop("RECEIPT_OUTPUT_ROOT")
            marker, marker_raw = gate0a._read_control(
                control_fd, MARKER_FILENAME, "PAYLOAD_MARKER"
            )
            claim = gate0a._existing_claim(control_fd, bundle)
            if claim is None:
                _stop("RECEIPT_CLAIM_MISSING")
            ledger, ledger_raw = gate0a._read_control(
                control_fd, gate0a.LEDGER_FILENAME, "AVAILABILITY_LEDGER"
            )
            gate0a._validate_ledger(ledger, bundle)
            _, claim_raw = gate0a._read_control(
                control_fd, gate0a.CLAIM_FILENAME, "CLAIM"
            )
            binding = _validate_claim_binding(
                claim, claim_raw, ledger, ledger_raw
            )
            if (
                marker.get("schema_version") != MARKER_SCHEMA
                or marker.get("stage") != "GATE_0B"
                or marker.get("synthetic_only") is not True
                or marker.get("state")
                != "OPENING_O_EXCL_BEFORE_FIRST_PAYLOAD_FD"
                or hashlib.sha256(marker_raw).hexdigest()
                != receipt["payload_opening_sha256"]
                or marker.get("claim_sha256") != receipt["claim_sha256"]
                or marker.get("availability_ledger_sha256")
                != receipt["availability_ledger_sha256"]
                or marker.get("collection_manifest_sha256")
                != receipt["collection_manifest_sha256"]
                or binding["claim_sha256"] != receipt["claim_sha256"]
                or binding["ledger_sha256"]
                != receipt["availability_ledger_sha256"]
            ):
                _stop("RECEIPT_MARKER_BINDING")
            if _artifact_hashes(output_root, expected_uid) != receipt["output_hashes"]:
                _stop("RECEIPT_OUTPUT_BINDING")
            audit = json.loads(
                (output_root / "audit/qualification.json").read_text(
                    encoding="utf-8"
                )
            )
            if audit.get("counts") != receipt["counts"]:
                _stop("RECEIPT_COUNTS_BINDING")
            return receipt

        try:
            os.stat(MARKER_FILENAME, dir_fd=control_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        except OSError:
            _stop("PAYLOAD_MARKER_STAT")
        else:
            _stop("INCOMPLETE_PRIOR_PAYLOAD_OPEN")
        if output_root.exists():
            _stop("OUTPUT_ROOT_PREEXISTS")

        claim = gate0a._existing_claim(control_fd, bundle)
        if claim is None:
            _stop("CLAIM_MISSING")
        ledger, ledger_raw = gate0a._read_control(
            control_fd, gate0a.LEDGER_FILENAME, "AVAILABILITY_LEDGER"
        )
        gate0a._validate_ledger(ledger, bundle)
        _, claim_raw = gate0a._read_control(
            control_fd, gate0a.CLAIM_FILENAME, "CLAIM"
        )
        binding = _validate_claim_binding(claim, claim_raw, ledger, ledger_raw)

        inbox_fd = gate0a._open_directory(inbox, expected_uid, "INBOX")
        try:
            collection_fd = os.open(
                claim["directory_name"],
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=inbox_fd,
            )
            try:
                gate0a._directory_metadata(
                    os.fstat(collection_fd), expected_uid, "COLLECTION"
                )
                manifest, manifest_raw = _read_manifest(
                    collection_fd,
                    claim["directory_name"],
                    bundle,
                    expected_uid,
                )
                if (
                    hashlib.sha256(manifest_raw).hexdigest()
                    != claim["collection_manifest_sha256"]
                    or manifest["collection_id"] != claim["collection_id"]
                ):
                    _stop("CLAIM_MANIFEST_BINDING")
                gate0a._inspect_collection_entries(
                    collection_fd, manifest, expected_uid
                )

                marker = {
                    "schema_version": MARKER_SCHEMA,
                    "stage": "GATE_0B",
                    "synthetic_only": True,
                    "claim_sha256": binding["claim_sha256"],
                    "availability_ledger_sha256": binding["ledger_sha256"],
                    "directory_name": claim["directory_name"],
                    "collection_manifest_sha256": claim[
                        "collection_manifest_sha256"
                    ],
                    "state": "OPENING_O_EXCL_BEFORE_FIRST_PAYLOAD_FD",
                }
                marker_raw = _canonical_json(marker)
                if not gate0a._write_exclusive(
                    control_fd, MARKER_FILENAME, marker_raw
                ):
                    _stop("PAYLOAD_MARKER_EXISTS")
                if crash_stage == "after_marker":
                    raise Gate0BStop("SYNTHETIC_CRASH_AFTER_MARKER")

                if manifest["crm_format"] != "CSV" or manifest["mapping_format"] != "CSV":
                    _stop("SYNTHETIC_GATE_0B_CSV_ONLY")
                crm_raw, crm_fd, crm_identity = _read_payload_once(
                    collection_fd,
                    manifest["crm_file"],
                    expected_size=manifest["crm_size_bytes"],
                    expected_sha256=manifest["crm_sha256"],
                    expected_uid=expected_uid,
                    label="CRM_PAYLOAD",
                )
                retained_payload_fds.append((crm_fd, crm_identity))
                mapping_raw, mapping_fd, mapping_identity = _read_payload_once(
                    collection_fd,
                    manifest["mapping_file"],
                    expected_size=manifest["mapping_size_bytes"],
                    expected_sha256=manifest["mapping_sha256"],
                    expected_uid=expected_uid,
                    label="MAPPING_PAYLOAD",
                )
                retained_payload_fds.append((mapping_fd, mapping_identity))
            finally:
                os.close(collection_fd)
        finally:
            os.close(inbox_fd)

        crm_rows = _decode_crm(_csv_rows(crm_raw, CRM_COLUMNS, "CRM"))
        mapping_rows = _decode_mapping(
            _csv_rows(mapping_raw, MAPPING_COLUMNS, "MAPPING")
        )
        if len(crm_rows) != manifest["crm_row_count"]:
            _stop("CRM_ROW_COUNT")
        if len(mapping_rows) != manifest["mapping_row_count"]:
            _stop("MAPPING_ROW_COUNT")
        try:
            result = qualify_fixture_rows(
                manifest=manifest,
                collection_manifest_sha256=claim["collection_manifest_sha256"],
                source_file_sha256=manifest["crm_sha256"],
                crm_rows=crm_rows,
                mapping_rows=mapping_rows,
                authority_catalog=authority_catalog,
                synthetic_fixtures_only=True,
            )
            contamination = audit_synthetic_contamination(
                crm_rows=crm_rows,
                split_rows=result["split_rows"],
                keysets=contamination_keysets,
                hmac_key=synthetic_hmac_key,
                synthetic_only=True,
            )
            write_fixture_outputs(result, output_root)
        except QualificationError as exc:
            _stop(f"QUALIFICATION:{exc}")
        contamination_path = output_root / "audit/contamination.json"
        contamination_path.write_bytes(_canonical_json(contamination))
        contamination_path.chmod(0o600)
        preseal_raw = {
            "queries": (output_root / "queries/queries.csv").read_bytes(),
            "oracle": (output_root / "oracle/oracle.csv").read_bytes(),
            "private_split_rows": (
                output_root / "audit/private_split_rows.jsonl"
            ).read_bytes(),
        }
        queries_written, oracle_written, split_rows_written = (
            _read_written_artifacts(preseal_raw)
        )
        validate_artifacts(queries_written, oracle_written)
        seal_manifests(split_rows_written, output_root / "splits")
        _fsync_outputs(output_root)

        inbox_fd = gate0a._open_directory(inbox, expected_uid, "INBOX")
        try:
            collection_fd = os.open(
                claim["directory_name"],
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=inbox_fd,
            )
            try:
                gate0a._inspect_collection_entries(
                    collection_fd, manifest, expected_uid
                )
            finally:
                os.close(collection_fd)
        finally:
            os.close(inbox_fd)
        retained_output_fds = _retain_valid_output_tree(
            output_root, expected_uid
        )
        final_output_hashes, retained_final_file_fds = _validate_final_outputs(
            output_root,
            expected_counts=result["counts"],
            expected_contamination=contamination,
            expected_uid=expected_uid,
        )
        retained_output_fds.extend(retained_final_file_fds)
        inbox_fd = gate0a._open_directory(inbox, expected_uid, "INBOX")
        try:
            collection_fd = os.open(
                claim["directory_name"],
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=inbox_fd,
            )
            try:
                final_manifest_raw, final_manifest_fd, final_manifest_identity = (
                    _read_payload_once(
                        collection_fd,
                        "collection_manifest.json",
                        expected_size=len(manifest_raw),
                        expected_sha256=claim["collection_manifest_sha256"],
                        expected_uid=expected_uid,
                        label="FINAL_COLLECTION_MANIFEST",
                    )
                )
                retained_payload_fds.append(
                    (final_manifest_fd, final_manifest_identity)
                )
                if final_manifest_raw != manifest_raw:
                    _stop("FINAL_COLLECTION_MANIFEST_CHANGED")
            finally:
                os.close(collection_fd)
        finally:
            os.close(inbox_fd)
        for retained_fd, expected_identity in (
            retained_payload_fds + retained_output_fds
        ):
            if _identity(os.fstat(retained_fd)) != expected_identity:
                _stop("RETAINED_FD_CHANGED_BEFORE_RECEIPT")

        receipt = {
            "schema_version": RECEIPT_SCHEMA,
            "stage": "GATE_0B",
            "synthetic_only": True,
            "terminal_state": "QUALIFICATION_SEALED",
            "claim_sha256": binding["claim_sha256"],
            "availability_ledger_sha256": binding["ledger_sha256"],
            "payload_opening_sha256": hashlib.sha256(marker_raw).hexdigest(),
            "collection_manifest_sha256": claim["collection_manifest_sha256"],
            "crm_sha256": manifest["crm_sha256"],
            "mapping_sha256": manifest["mapping_sha256"],
            "output_root": str(output_root),
            "output_hashes": final_output_hashes,
            "counts": result["counts"],
        }
        if not gate0a._write_exclusive(
            control_fd, RECEIPT_FILENAME, _canonical_json(receipt)
        ):
            _stop("RECEIPT_EXISTS")
        return receipt
    finally:
        for payload_fd, _ in retained_payload_fds:
            try:
                os.close(payload_fd)
            except OSError:
                pass
        for output_fd, _ in retained_output_fds:
            try:
                os.close(output_fd)
            except OSError:
                pass
        os.close(control_fd)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inbox", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--authority-catalog", type=Path, required=True)
    parser.add_argument("--contamination-keysets", type=Path, required=True)
    parser.add_argument("--synthetic-hmac-key-hex", required=True)
    parser.add_argument("--synthetic-only", action="store_true")
    args = parser.parse_args(argv)
    catalog_path = _tmp_path(
        args.authority_catalog, must_exist=True, label="AUTHORITY_CATALOG"
    )
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    keysets_path = _tmp_path(
        args.contamination_keysets,
        must_exist=True,
        label="CONTAMINATION_KEYSETS",
    )
    keysets_raw = json.loads(keysets_path.read_text(encoding="utf-8"))
    if not isinstance(keysets_raw, dict):
        _stop("CONTAMINATION_KEYSETS_SCHEMA")
    try:
        keysets = {key: set(values) for key, values in keysets_raw.items()}
        synthetic_key = bytes.fromhex(args.synthetic_hmac_key_hex)
    except (TypeError, ValueError):
        _stop("SYNTHETIC_CONTAMINATION_INPUT")
    result = open_synthetic_qualification(
        inbox=args.inbox,
        control_root=args.control_root,
        output_root=args.output_root,
        authority_catalog=catalog,
        contamination_keysets=keysets,
        synthetic_hmac_key=synthetic_key,
        synthetic_only=args.synthetic_only,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
