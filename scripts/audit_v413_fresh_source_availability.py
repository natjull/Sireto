#!/usr/bin/env python3
"""V4.13 Gate 0A availability auditor.

This module is deliberately synthetic-only until the V4.13 implementation
lock exists.  It validates the preregistration authorities, observes only
collection manifests and filesystem metadata, and creates the synthetic
availability ledger and claim with O_EXCL.  It never opens either payload.
"""

from __future__ import annotations

import hashlib
import fcntl
import json
import os
import re
import stat
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping


REPOSITORY = Path(__file__).resolve().parents[1]
PLAN_PATH = REPOSITORY / "config/v4_13_fresh_labels_minimal_plan.json"
LOCK_PATH = REPOSITORY / "config/v4_13_fresh_labels_preregistration_lock.json"

EXPECTED_PREREGISTRATION_COMMIT = "bf4ed261ba80e0242bda8b46884fcf484c6c8e1b"
EXPECTED_LOCK_SHA256 = (
    "b0648ae9ccdce25b9d0562328ce237a121503b95dcbc88f1e117c755186b6878"
)
EXPECTED_PLAN_SHA256 = (
    "ec1c8891edfbb8d7cfb56afaab8a689fddbe9041bc2d6b5345fb5e30f9febbfe"
)
EXPECTED_AUTHORITY_CATALOG_SHA256 = (
    "780e920ed529199dfbede007053328b46c2a929532d1b089161a10779568ea25"
)

DIRECTORY_PATTERN = re.compile(r"^([0-9]{20})_([0-9a-f]{64})$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SAFE_BASENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

LEDGER_FILENAME = "availability_ledger.json"
CLAIM_FILENAME = "collection.claim.json"

StabilityObserver = Callable[[float], float]


class AvailabilityStop(RuntimeError):
    """Closed failure for an invalid or unsafe Gate 0A state."""


@dataclass(frozen=True)
class ControlBundle:
    plan: dict[str, Any]
    plan_raw: bytes
    plan_sha256: str
    lock: dict[str, Any]
    lock_raw: bytes
    lock_sha256: str
    lock_created_at: datetime


@dataclass(frozen=True)
class ManifestObservation:
    directory_name: str
    arrival_epoch_ns: int
    manifest_sha256: str
    manifest_size_bytes: int
    manifest: dict[str, Any]
    stability: dict[str, Any]

    @property
    def selection_tuple(self) -> tuple[int, str]:
        return self.arrival_epoch_ns, self.manifest_sha256


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        ).encode("utf-8")
        + b"\n"
    )


def _stop(reason: str) -> None:
    raise AvailabilityStop(reason)


def _read_all(fd: int) -> bytes:
    os.lseek(fd, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _parse_canonical_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except Exception:
        _stop(f"{label}_JSON")
    if type(value) is not dict or canonical_json(value) != raw:
        _stop(f"{label}_NONCANONICAL")
    return value


def _read_canonical_file(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError:
        _stop(f"{label}_OPEN")
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            _stop(f"{label}_TYPE")
        raw = _read_all(fd)
        after = os.fstat(fd)
        if _stat_identity(before) != _stat_identity(after):
            _stop(f"{label}_CHANGED")
    finally:
        os.close(fd)
    return _parse_canonical_object(raw, label), raw


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if type(value) is not dict or set(value) != expected:
        _stop(f"{label}_SCHEMA")


def _parse_utc(value: Any, label: str) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        _stop(f"{label}_TIMESTAMP")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        _stop(f"{label}_TIMESTAMP")
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        _stop(f"{label}_TIMESTAMP")
    return parsed


def _parse_date(value: Any, label: str) -> date:
    if type(value) is not str:
        _stop(f"{label}_DATE")
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        _stop(f"{label}_DATE")
    if parsed.isoformat() != value:
        _stop(f"{label}_DATE")
    return parsed


def _regular_metadata(st: os.stat_result, expected_uid: int, label: str) -> None:
    if not stat.S_ISREG(st.st_mode):
        _stop(f"{label}_NOT_REGULAR")
    if st.st_nlink != 1:
        _stop(f"{label}_HARDLINK")
    if st.st_uid != expected_uid:
        _stop(f"{label}_UID")
    if stat.S_IMODE(st.st_mode) != 0o600:
        _stop(f"{label}_MODE")


def _directory_metadata(st: os.stat_result, expected_uid: int, label: str) -> None:
    if not stat.S_ISDIR(st.st_mode):
        _stop(f"{label}_NOT_DIRECTORY")
    if st.st_uid != expected_uid:
        _stop(f"{label}_UID")
    if stat.S_IMODE(st.st_mode) != 0o700:
        _stop(f"{label}_MODE")


def _stat_identity(st: os.stat_result) -> tuple[int, ...]:
    return (
        st.st_dev,
        st.st_ino,
        st.st_uid,
        st.st_nlink,
        stat.S_IMODE(st.st_mode),
        st.st_size,
        st.st_mtime_ns,
        st.st_ctime_ns,
    )


def _full_fsync(fd: int) -> None:
    os.fsync(fd)
    full = getattr(fcntl, "F_FULLFSYNC", None)
    if full is not None:
        try:
            fcntl.fcntl(fd, full)
        except OSError:
            _stop("FULLFSYNC")


def _open_directory(path: Path, expected_uid: int, label: str) -> int:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError:
        _stop(f"{label}_OPEN")
    try:
        _directory_metadata(os.fstat(fd), expected_uid, label)
    except Exception:
        os.close(fd)
        raise
    return fd


def _validate_bundle(
    repository: Path,
    plan_path: Path,
    lock_path: Path,
) -> ControlBundle:
    plan, plan_raw = _read_canonical_file(plan_path, "PLAN")
    lock, lock_raw = _read_canonical_file(lock_path, "PREREGISTRATION_LOCK")
    plan_sha = sha256_bytes(plan_raw)
    lock_sha = sha256_bytes(lock_raw)
    if plan_sha != EXPECTED_PLAN_SHA256:
        _stop("PLAN_SHA256")
    if lock_sha != EXPECTED_LOCK_SHA256:
        _stop("PREREGISTRATION_LOCK_SHA256")

    expected_lock_keys = {
        "schema_version",
        "git_commit",
        "created_at_utc",
        "status",
        "uid",
        "ssd_volume_uuid",
        "real_collection_open_authorized",
        "plan",
        "plan_schema",
        "contract",
        "authority_catalog",
        "tests",
        "independent_audits",
    }
    _exact_keys(lock, expected_lock_keys, "PREREGISTRATION_LOCK")
    if (
        lock["schema_version"]
        != "sireto-v4.13-fresh-labels-preregistration-lock-1"
        or lock["git_commit"] != EXPECTED_PREREGISTRATION_COMMIT
        or lock["status"] != "GO_V413_PREREGISTRATION_IMPLEMENT_SYNTHETIC_ONLY"
        or lock["real_collection_open_authorized"] is not False
        or type(lock["uid"]) is not int
        or lock["uid"] != os.getuid()
    ):
        _stop("PREREGISTRATION_LOCK_POLICY")
    audits = lock["independent_audits"]
    if (
        type(audits) is not list
        or len(audits) != 2
        or len({item.get("auditor") for item in audits if type(item) is dict}) != 2
        or any(
            type(item) is not dict
            or set(item) != {"auditor", "commit", "verdict"}
            or item["commit"] != EXPECTED_PREREGISTRATION_COMMIT
            or item["verdict"] != "GO_V413_PREREGISTRATION"
            for item in audits
        )
    ):
        _stop("PREREGISTRATION_AUDITS")

    for field, expected_path in (
        ("plan", "config/v4_13_fresh_labels_minimal_plan.json"),
        ("plan_schema", "config/v4_13_fresh_labels_minimal_plan.schema.json"),
        ("contract", "docs/v4_13_fresh_labels_minimal_contract.md"),
        ("authority_catalog", "config/v4_13_fresh_labels_authority_catalog.json"),
        ("tests", "tests/test_v413_fresh_labels_minimal_plan.py"),
    ):
        pin = lock[field]
        if type(pin) is not dict or pin.get("path") != expected_path:
            _stop(f"LOCK_{field.upper()}_PIN")
        pinned_path = repository / expected_path
        try:
            pinned_raw = pinned_path.read_bytes()
        except OSError:
            _stop(f"LOCK_{field.upper()}_READ")
        if sha256_bytes(pinned_raw) != pin.get("sha256"):
            _stop(f"LOCK_{field.upper()}_SHA256")

    if lock["plan"]["sha256"] != plan_sha:
        _stop("LOCK_PLAN_SHA256")
    schema, _ = _read_canonical_file(
        repository / lock["plan_schema"]["path"], "PLAN_SCHEMA"
    )
    if schema.get("target_sha256") != plan_sha:
        _stop("PLAN_SCHEMA_TARGET")

    catalog, catalog_raw = _read_canonical_file(
        repository / lock["authority_catalog"]["path"], "AUTHORITY_CATALOG"
    )
    if (
        sha256_bytes(catalog_raw) != EXPECTED_AUTHORITY_CATALOG_SHA256
        or catalog.get("allowlist") != []
        or catalog.get("real_collection_open_authorized") is not False
        or catalog.get("status") != "EMPTY_NO_REAL_AUTHORITY_REGISTERED"
    ):
        _stop("AUTHORITY_CATALOG_POLICY")

    if (
        plan.get("status") != "PREREGISTRATION_AMENDMENT_AWAITING_TWO_GO"
        or plan.get("implementation", {}).get("status")
        != "UNIMPLEMENTED_NOT_AUTHORIZED"
        or plan.get("authority_catalog", {}).get("real_collection_open_authorized")
        is not False
        or plan.get("source_protocol", {})
        .get("manifest_only_0a", {})
        .get("payload_open_forbidden")
        is not True
    ):
        _stop("PLAN_SYNTHETIC_POLICY")
    if plan["authority_catalog"].get("real_allowlist") != {
        "path": lock["authority_catalog"]["path"],
        "sha256": lock["authority_catalog"]["sha256"],
        "status": "EMPTY_NO_REAL_AUTHORITY_REGISTERED",
    }:
        _stop("PLAN_AUTHORITY_CATALOG_PIN")

    return ControlBundle(
        plan=plan,
        plan_raw=plan_raw,
        plan_sha256=plan_sha,
        lock=lock,
        lock_raw=lock_raw,
        lock_sha256=lock_sha,
        lock_created_at=_parse_utc(lock["created_at_utc"], "LOCK_CREATED"),
    )


def _validate_manifest(
    value: dict[str, Any],
    raw: bytes,
    *,
    directory_name: str,
    plan: dict[str, Any],
    bundle: ControlBundle,
) -> None:
    schema = plan["artifact_schemas"]["collection_manifest"]
    expected = set(schema["fields"])
    _exact_keys(value, expected, "COLLECTION_MANIFEST")
    suffix = DIRECTORY_PATTERN.fullmatch(directory_name)
    if suffix is None or suffix.group(2) != sha256_bytes(raw):
        _stop("COLLECTION_DIRECTORY_NAME")

    string_nonempty = {
        "collection_id",
        "export_id",
        "population_definition",
        "producer_id",
        "source_record_id_semantics",
    }
    for field in string_nonempty:
        if type(value[field]) is not str or not value[field]:
            _stop(f"MANIFEST_{field.upper()}")
    if (
        value["schema_version"] != "sireto-v4.13-collection-manifest-1"
        or value["authority_catalog_id"]
        != plan["authority_catalog"]["catalog_id"]
        or value["plan_git_commit"] != bundle.lock["git_commit"]
        or value["plan_sha256"] != bundle.plan_sha256
        or value["preregistration_lock_sha256"] != bundle.lock_sha256
        or value["population_is_exhaustive"] is not True
        or value["matching_based_exclusions"] is not False
        or type(value["population_exclusions"]) is not list
    ):
        _stop("COLLECTION_MANIFEST_POLICY")

    for field in ("crm_sha256", "mapping_sha256"):
        if type(value[field]) is not str or HEX64.fullmatch(value[field]) is None:
            _stop(f"MANIFEST_{field.upper()}")
    if HEX40.fullmatch(value["plan_git_commit"]) is None:
        _stop("MANIFEST_PLAN_COMMIT")
    for field in (
        "crm_row_count",
        "crm_size_bytes",
        "mapping_row_count",
        "mapping_size_bytes",
    ):
        if type(value[field]) is not int or value[field] <= 0:
            _stop(f"MANIFEST_{field.upper()}")

    pairs = (
        ("crm_file", "crm_format", "crm_source"),
        ("mapping_file", "mapping_format", "authoritative_mapping"),
    )
    for file_field, format_field, stem in pairs:
        name = value[file_field]
        fmt = value[format_field]
        if (
            type(name) is not str
            or SAFE_BASENAME.fullmatch(name) is None
            or name not in {f"{stem}.csv", f"{stem}.parquet"}
            or fmt not in {"CSV", "PARQUET"}
            or name.rsplit(".", 1)[1].upper() != fmt
        ):
            _stop(f"MANIFEST_{file_field.upper()}")

    period_start = _parse_utc(value["period_start_utc"], "PERIOD_START")
    period_end = _parse_utc(value["period_end_utc"], "PERIOD_END")
    cutoff = _parse_utc(value["export_cutoff_utc"], "EXPORT_CUTOFF")
    created = _parse_utc(value["created_at_utc"], "COLLECTION_CREATED")
    if not (
        bundle.lock_created_at < period_start <= period_end <= cutoff <= created
    ):
        _stop("MANIFEST_TEMPORAL_ORDER")
    reference = _parse_date(value["reference_date"], "REFERENCE")
    if not (period_start.date() <= reference <= cutoff.date()):
        _stop("MANIFEST_REFERENCE_DATE")


def _observe_manifest(
    directory_fd: int,
    directory_name: str,
    *,
    plan: dict[str, Any],
    bundle: ControlBundle,
    observer: StabilityObserver,
    minimum_seconds: float,
    expected_uid: int,
) -> ManifestObservation:
    try:
        manifest_fd = os.open(
            "collection_manifest.json",
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=directory_fd,
        )
    except OSError:
        _stop("COLLECTION_MANIFEST_OPEN")
    try:
        first_stat = os.fstat(manifest_fd)
        _regular_metadata(first_stat, expected_uid, "COLLECTION_MANIFEST")
        first_raw = _read_all(manifest_fd)
        first_value = _parse_canonical_object(first_raw, "COLLECTION_MANIFEST")
        _validate_manifest(
            first_value,
            first_raw,
            directory_name=directory_name,
            plan=plan,
            bundle=bundle,
        )
        elapsed = observer(minimum_seconds)
        if type(elapsed) not in {int, float} or elapsed < minimum_seconds:
            _stop("STABILITY_INTERVAL")
        second_stat = os.fstat(manifest_fd)
        second_raw = _read_all(manifest_fd)
        if (
            _stat_identity(first_stat) != _stat_identity(second_stat)
            or first_raw != second_raw
        ):
            _stop("MANIFEST_UNSTABLE")
    finally:
        os.close(manifest_fd)

    match = DIRECTORY_PATTERN.fullmatch(directory_name)
    assert match is not None
    return ManifestObservation(
        directory_name=directory_name,
        arrival_epoch_ns=int(match.group(1)),
        manifest_sha256=sha256_bytes(first_raw),
        manifest_size_bytes=len(first_raw),
        manifest=first_value,
        stability={
            "elapsed_seconds": float(elapsed),
            "identity": {
                "device": first_stat.st_dev,
                "inode": first_stat.st_ino,
                "uid": first_stat.st_uid,
                "nlink": first_stat.st_nlink,
                "mode": f"{stat.S_IMODE(first_stat.st_mode):04o}",
                "size": first_stat.st_size,
                "mtime_ns": first_stat.st_mtime_ns,
                "ctime_ns": first_stat.st_ctime_ns,
            },
            "sha256": sha256_bytes(first_raw),
        },
    )


def _inspect_collection_entries(
    directory_fd: int,
    manifest: Mapping[str, Any],
    expected_uid: int,
) -> None:
    try:
        names = sorted(os.listdir(directory_fd))
    except OSError:
        _stop("COLLECTION_ENUMERATION")
    expected = {
        "collection_manifest.json",
        manifest["crm_file"],
        manifest["mapping_file"],
    }
    if set(names) != expected or len(names) != 3:
        _stop("COLLECTION_EXTRA_OR_MISSING_ENTRY")
    for name in names:
        try:
            metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError:
            _stop("COLLECTION_ENTRY_STAT")
        _regular_metadata(metadata, expected_uid, f"COLLECTION_ENTRY_{name}")
        if name == manifest["crm_file"] and metadata.st_size != manifest["crm_size_bytes"]:
            _stop("CRM_DECLARED_SIZE")
        if (
            name == manifest["mapping_file"]
            and metadata.st_size != manifest["mapping_size_bytes"]
        ):
            _stop("MAPPING_DECLARED_SIZE")


def _enumerate_observations(
    inbox: Path,
    *,
    plan: dict[str, Any],
    bundle: ControlBundle,
    observer: StabilityObserver,
    minimum_seconds: float,
    expected_uid: int,
) -> list[ManifestObservation]:
    inbox_fd = _open_directory(inbox, expected_uid, "INBOX")
    try:
        names = sorted(os.listdir(inbox_fd))
        observations: list[ManifestObservation] = []
        for name in names:
            if DIRECTORY_PATTERN.fullmatch(name) is None:
                _stop("INBOX_CHILD_NAME")
            try:
                child_fd = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=inbox_fd,
                )
            except OSError:
                _stop("INBOX_CHILD_OPEN")
            try:
                _directory_metadata(os.fstat(child_fd), expected_uid, "COLLECTION")
                observation = _observe_manifest(
                    child_fd,
                    name,
                    plan=plan,
                    bundle=bundle,
                    observer=observer,
                    minimum_seconds=minimum_seconds,
                    expected_uid=expected_uid,
                )
                _inspect_collection_entries(child_fd, observation.manifest, expected_uid)
                observations.append(observation)
            finally:
                os.close(child_fd)
        return observations
    finally:
        os.close(inbox_fd)


def _write_exclusive(directory_fd: int, name: str, raw: bytes) -> bool:
    try:
        fd = os.open(
            name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_NOFOLLOW
            | os.O_CLOEXEC,
            0o600,
            dir_fd=directory_fd,
        )
    except FileExistsError:
        return False
    except OSError:
        _stop(f"CONTROL_CREATE_{name}")
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(fd, raw[offset:])
            if written <= 0:
                _stop(f"CONTROL_WRITE_{name}")
            offset += written
        _full_fsync(fd)
        _regular_metadata(os.fstat(fd), os.getuid(), f"CONTROL_{name}")
    finally:
        os.close(fd)
    _full_fsync(directory_fd)
    return True


def _read_control(directory_fd: int, name: str, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        fd = os.open(
            name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=directory_fd,
        )
    except OSError:
        _stop(f"{label}_OPEN")
    try:
        metadata = os.fstat(fd)
        _regular_metadata(metadata, os.getuid(), label)
        raw = _read_all(fd)
    finally:
        os.close(fd)
    return _parse_canonical_object(raw, label), raw


def _validate_ledger(ledger: dict[str, Any], bundle: ControlBundle) -> None:
    expected_keys = {
        "schema_version",
        "stage",
        "synthetic_only",
        "plan_sha256",
        "preregistration_lock_sha256",
        "selection_rule",
        "observed_manifests",
        "selected_directory_name",
        "selected_manifest_sha256",
    }
    _exact_keys(ledger, expected_keys, "AVAILABILITY_LEDGER")
    if (
        ledger["schema_version"]
        != "sireto-v4.13-availability-ledger-synthetic-1"
        or ledger["stage"] != "GATE_0A"
        or ledger["synthetic_only"] is not True
        or ledger["plan_sha256"] != bundle.plan_sha256
        or ledger["preregistration_lock_sha256"] != bundle.lock_sha256
        or ledger["selection_rule"]
        != "MINIMUM_ARRIVAL_EPOCH_NS_THEN_MANIFEST_SHA256"
        or type(ledger["observed_manifests"]) is not list
        or not ledger["observed_manifests"]
        or type(ledger["selected_directory_name"]) is not str
        or DIRECTORY_PATTERN.fullmatch(ledger["selected_directory_name"]) is None
        or type(ledger["selected_manifest_sha256"]) is not str
        or HEX64.fullmatch(ledger["selected_manifest_sha256"]) is None
    ):
        _stop("AVAILABILITY_LEDGER_POLICY")
    tuples: list[tuple[int, str]] = []
    selected_membership = 0
    for record in ledger["observed_manifests"]:
        _exact_keys(
            record,
            {
                "directory_name",
                "arrival_epoch_ns",
                "manifest_sha256",
                "manifest_size_bytes",
                "collection_id",
                "stability",
            },
            "AVAILABILITY_LEDGER_RECORD",
        )
        name_match = (
            DIRECTORY_PATTERN.fullmatch(record["directory_name"])
            if type(record["directory_name"]) is str
            else None
        )
        if (
            name_match is None
            or type(record["arrival_epoch_ns"]) is not int
            or int(name_match.group(1)) != record["arrival_epoch_ns"]
            or type(record["manifest_sha256"]) is not str
            or name_match.group(2) != record["manifest_sha256"]
            or type(record["manifest_size_bytes"]) is not int
            or record["manifest_size_bytes"] <= 0
            or type(record["collection_id"]) is not str
            or not record["collection_id"]
            or type(record["stability"]) is not dict
        ):
            _stop("AVAILABILITY_LEDGER_RECORD")
        item = record["arrival_epoch_ns"], record["manifest_sha256"]
        tuples.append(item)
        if (
            record["directory_name"] == ledger["selected_directory_name"]
            and record["manifest_sha256"] == ledger["selected_manifest_sha256"]
        ):
            selected_membership += 1
    if tuples != sorted(tuples) or len(tuples) != len(set(tuples)):
        _stop("AVAILABILITY_LEDGER_ORDER")
    if (
        selected_membership != 1
        or tuples[0]
        != (
            int(ledger["selected_directory_name"].split("_", 1)[0]),
            ledger["selected_manifest_sha256"],
        )
    ):
        _stop("AVAILABILITY_LEDGER_SELECTION")


def _existing_claim(
    control_fd: int,
    bundle: ControlBundle,
) -> dict[str, Any] | None:
    try:
        os.stat(CLAIM_FILENAME, dir_fd=control_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError:
        _stop("CLAIM_STAT")
    claim, _ = _read_control(control_fd, CLAIM_FILENAME, "CLAIM")
    expected_keys = {
        "schema_version",
        "stage",
        "synthetic_only",
        "plan_sha256",
        "preregistration_lock_sha256",
        "availability_ledger_sha256",
        "directory_name",
        "arrival_epoch_ns",
        "collection_manifest_sha256",
        "collection_id",
        "claim_state",
    }
    _exact_keys(claim, expected_keys, "CLAIM")
    if (
        claim["schema_version"] != "sireto-v4.13-availability-claim-synthetic-1"
        or claim["stage"] != "GATE_0A"
        or claim["synthetic_only"] is not True
        or claim["plan_sha256"] != bundle.plan_sha256
        or claim["preregistration_lock_sha256"] != bundle.lock_sha256
        or claim["claim_state"] != "MANIFEST_ONLY_NO_PAYLOAD_OPEN"
        or type(claim["arrival_epoch_ns"]) is not int
        or DIRECTORY_PATTERN.fullmatch(claim["directory_name"]) is None
        or HEX64.fullmatch(claim["collection_manifest_sha256"]) is None
    ):
        _stop("CLAIM_POLICY")
    return claim


def audit_synthetic_availability(
    *,
    inbox: Path,
    control_root: Path,
    stability_observer: StabilityObserver,
    minimum_stability_seconds: float = 60.0,
    repository: Path = REPOSITORY,
    plan_path: Path | None = None,
    lock_path: Path | None = None,
    synthetic_only: bool = False,
) -> dict[str, Any]:
    """Select and claim one stable synthetic collection without payload reads."""

    if synthetic_only is not True:
        _stop("REAL_COLLECTION_OPEN_FORBIDDEN")
    if (
        type(minimum_stability_seconds) not in {int, float}
        or minimum_stability_seconds < 60
    ):
        _stop("STABILITY_MINIMUM")
    if not callable(stability_observer):
        _stop("STABILITY_OBSERVER")

    plan_path = plan_path or repository / PLAN_PATH.relative_to(REPOSITORY)
    lock_path = lock_path or repository / LOCK_PATH.relative_to(REPOSITORY)
    bundle = _validate_bundle(repository, plan_path, lock_path)
    expected_uid = bundle.lock["uid"]
    control_fd = _open_directory(control_root, expected_uid, "CONTROL_ROOT")
    try:
        try:
            fcntl.flock(control_fd, fcntl.LOCK_EX)
        except OSError:
            _stop("CONTROL_LOCK")
        existing = _existing_claim(control_fd, bundle)
        if existing is not None:
            ledger, ledger_raw = _read_control(
                control_fd, LEDGER_FILENAME, "AVAILABILITY_LEDGER"
            )
            _validate_ledger(ledger, bundle)
            if (
                sha256_bytes(ledger_raw) != existing["availability_ledger_sha256"]
                or ledger.get("selected_directory_name")
                != existing["directory_name"]
                or ledger.get("selected_manifest_sha256")
                != existing["collection_manifest_sha256"]
            ):
                _stop("CLAIM_LEDGER_BINDING")
            return existing

        observations = _enumerate_observations(
            inbox,
            plan=bundle.plan,
            bundle=bundle,
            observer=stability_observer,
            minimum_seconds=float(minimum_stability_seconds),
            expected_uid=expected_uid,
        )
        if not observations:
            return {
                "schema_version": "sireto-v4.13-availability-result-synthetic-1",
                "stage": "GATE_0A",
                "synthetic_only": True,
                "verdict": "WAITING_FOR_NEW_SOURCE",
                "observed_manifest_count": 0,
            }

        selected = min(observations, key=lambda item: item.selection_tuple)
        ledger = {
            "schema_version": "sireto-v4.13-availability-ledger-synthetic-1",
            "stage": "GATE_0A",
            "synthetic_only": True,
            "plan_sha256": bundle.plan_sha256,
            "preregistration_lock_sha256": bundle.lock_sha256,
            "selection_rule": "MINIMUM_ARRIVAL_EPOCH_NS_THEN_MANIFEST_SHA256",
            "observed_manifests": [
                {
                    "directory_name": item.directory_name,
                    "arrival_epoch_ns": item.arrival_epoch_ns,
                    "manifest_sha256": item.manifest_sha256,
                    "manifest_size_bytes": item.manifest_size_bytes,
                    "collection_id": item.manifest["collection_id"],
                    "stability": item.stability,
                }
                for item in sorted(observations, key=lambda item: item.selection_tuple)
            ],
            "selected_directory_name": selected.directory_name,
            "selected_manifest_sha256": selected.manifest_sha256,
        }
        ledger_raw = canonical_json(ledger)
        if not _write_exclusive(control_fd, LEDGER_FILENAME, ledger_raw):
            persisted_ledger, persisted_raw = _read_control(
                control_fd, LEDGER_FILENAME, "AVAILABILITY_LEDGER"
            )
            _validate_ledger(persisted_ledger, bundle)
            ledger = persisted_ledger
            ledger_raw = persisted_raw

        selected_name = ledger.get("selected_directory_name")
        selected_hash = ledger.get("selected_manifest_sha256")
        matching = [
            item
            for item in observations
            if item.directory_name == selected_name
            and item.manifest_sha256 == selected_hash
        ]
        if len(matching) != 1:
            _stop("PERSISTED_LEDGER_SELECTION")
        selected = matching[0]
        claim = {
            "schema_version": "sireto-v4.13-availability-claim-synthetic-1",
            "stage": "GATE_0A",
            "synthetic_only": True,
            "plan_sha256": bundle.plan_sha256,
            "preregistration_lock_sha256": bundle.lock_sha256,
            "availability_ledger_sha256": sha256_bytes(ledger_raw),
            "directory_name": selected.directory_name,
            "arrival_epoch_ns": selected.arrival_epoch_ns,
            "collection_manifest_sha256": selected.manifest_sha256,
            "collection_id": selected.manifest["collection_id"],
            "claim_state": "MANIFEST_ONLY_NO_PAYLOAD_OPEN",
        }
        claim_raw = canonical_json(claim)
        if not _write_exclusive(control_fd, CLAIM_FILENAME, claim_raw):
            persisted = _existing_claim(control_fd, bundle)
            if persisted is None:
                _stop("CLAIM_RACE")
            return persisted
        return claim
    finally:
        os.close(control_fd)


def main() -> int:
    raise SystemExit(
        "V4.13 availability auditor is synthetic-only until an implementation "
        "lock and a non-empty double-audited authority catalog exist"
    )


if __name__ == "__main__":
    main()
