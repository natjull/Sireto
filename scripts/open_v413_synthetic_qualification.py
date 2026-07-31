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
from typing import Any, Callable, Mapping, Sequence

try:
    from scripts import audit_v413_fresh_source_availability as gate0a
    from scripts.build_v413_fresh_qualification import (
        CRM_COLUMNS,
        MAPPING_COLUMNS,
        QualificationError,
        qualify_fixture_rows,
        write_fixture_outputs,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...py`` execution.
    import audit_v413_fresh_source_availability as gate0a
    from build_v413_fresh_qualification import (
        CRM_COLUMNS,
        MAPPING_COLUMNS,
        QualificationError,
        qualify_fixture_rows,
        write_fixture_outputs,
    )


MARKER_FILENAME = "payload_opening.json"
RECEIPT_FILENAME = "qualification_receipt.json"
MARKER_SCHEMA = "sireto-v4.13-payload-opening-synthetic-1"
RECEIPT_SCHEMA = "sireto-v4.13-qualification-receipt-synthetic-1"

CrashHook = Callable[[str], None]


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
) -> bytes:
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
        return raw
    finally:
        os.close(fd)


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


def _artifact_hashes(output_root: Path) -> dict[str, str]:
    relative = {
        "queries": Path("queries/queries.csv"),
        "oracle": Path("oracle/oracle.csv"),
        "audit": Path("audit/qualification.json"),
        "private_split_rows": Path("audit/private_split_rows.jsonl"),
    }
    hashes: dict[str, str] = {}
    for name, suffix in relative.items():
        path = output_root / suffix
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        except OSError:
            _stop(f"OUTPUT_{name.upper()}_READ")
        try:
            metadata = os.fstat(fd)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                _stop(f"OUTPUT_{name.upper()}_TYPE")
            raw = _read_fd(fd)
        finally:
            os.close(fd)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    return hashes


def _fsync_outputs(output_root: Path) -> None:
    files = (
        output_root / "queries/queries.csv",
        output_root / "oracle/oracle.csv",
        output_root / "audit/qualification.json",
        output_root / "audit/private_split_rows.jsonl",
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
        != {"queries", "oracle", "audit", "private_split_rows"}
        or any(
            not isinstance(value, str) or gate0a.HEX64.fullmatch(value) is None
            for value in receipt["output_hashes"].values()
        )
        or not isinstance(receipt["counts"], dict)
    ):
        _stop("RECEIPT_POLICY")
    return receipt


def open_synthetic_qualification(
    *,
    inbox: Path,
    control_root: Path,
    output_root: Path,
    authority_catalog: Mapping[str, Any],
    synthetic_only: bool,
    crash_hook: CrashHook | None = None,
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
    hook = crash_hook or (lambda _stage: None)
    if not callable(hook):
        _stop("CRASH_HOOK")

    bundle = gate0a._validate_bundle(
        repository,
        plan_path or repository / gate0a.PLAN_PATH.relative_to(gate0a.REPOSITORY),
        lock_path or repository / gate0a.LOCK_PATH.relative_to(gate0a.REPOSITORY),
    )
    expected_uid = bundle.lock["uid"]
    control_fd = gate0a._open_directory(control_root, expected_uid, "CONTROL_ROOT")
    try:
        fcntl.flock(control_fd, fcntl.LOCK_EX)
        receipt = _read_receipt(control_fd)
        if receipt is not None:
            if Path(receipt["output_root"]) != output_root:
                _stop("RECEIPT_OUTPUT_ROOT")
            marker, marker_raw = gate0a._read_control(
                control_fd, MARKER_FILENAME, "PAYLOAD_MARKER"
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
            ):
                _stop("RECEIPT_MARKER_BINDING")
            if _artifact_hashes(output_root) != receipt["output_hashes"]:
                _stop("RECEIPT_OUTPUT_BINDING")
            return receipt

        try:
            os.stat(MARKER_FILENAME, dir_fd=control_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        except OSError:
            _stop("PAYLOAD_MARKER_STAT")
        else:
            _stop("INCOMPLETE_PRIOR_PAYLOAD_OPEN")

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
                hook("after_marker")

                if manifest["crm_format"] != "CSV" or manifest["mapping_format"] != "CSV":
                    _stop("SYNTHETIC_GATE_0B_CSV_ONLY")
                crm_raw = _read_payload_once(
                    collection_fd,
                    manifest["crm_file"],
                    expected_size=manifest["crm_size_bytes"],
                    expected_sha256=manifest["crm_sha256"],
                    expected_uid=expected_uid,
                    label="CRM_PAYLOAD",
                )
                hook("after_crm_payload")
                mapping_raw = _read_payload_once(
                    collection_fd,
                    manifest["mapping_file"],
                    expected_size=manifest["mapping_size_bytes"],
                    expected_sha256=manifest["mapping_sha256"],
                    expected_uid=expected_uid,
                    label="MAPPING_PAYLOAD",
                )
                hook("after_mapping_payload")
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
            write_fixture_outputs(result, output_root)
        except QualificationError as exc:
            _stop(f"QUALIFICATION:{exc}")
        _fsync_outputs(output_root)
        hook("after_outputs")

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
            "output_hashes": _artifact_hashes(output_root),
            "counts": result["counts"],
        }
        if not gate0a._write_exclusive(
            control_fd, RECEIPT_FILENAME, _canonical_json(receipt)
        ):
            _stop("RECEIPT_EXISTS")
        return receipt
    finally:
        os.close(control_fd)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inbox", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--authority-catalog", type=Path, required=True)
    parser.add_argument("--synthetic-only", action="store_true")
    args = parser.parse_args(argv)
    catalog_path = _tmp_path(
        args.authority_catalog, must_exist=True, label="AUTHORITY_CATALOG"
    )
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    result = open_synthetic_qualification(
        inbox=args.inbox,
        control_root=args.control_root,
        output_root=args.output_root,
        authority_catalog=catalog,
        synthetic_only=args.synthetic_only,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
