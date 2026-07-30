#!/usr/bin/env python3
"""Build the private V4.12 historical compatibility registry.

The module deliberately has two layers:

* pure projection/table functions, used by tests and independent auditors;
* a closed runner which reads only plan-pinned files through anchored file
  descriptors, writes an exclusive private staging tree, validates it, and
  promotes it without replacement.

It never opens a future CRM, a model, candidates, scores, or predictions.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import csv
import ctypes
from dataclasses import dataclass
import fcntl
import hashlib
import hmac
import io
import json
import os
from pathlib import Path
import platform
import plistlib
import stat
import subprocess
import sys
import unicodedata
from typing import Any, Iterable, Mapping, Protocol, Sequence

import pyarrow as pa
import pyarrow.parquet as pq


PLAN_SCHEMA = "sireto-v4.12-consumed-compatibility-registry-plan-2"
PROVENANCE_SCHEMA = "sireto-v4.12-consumed-compatibility-provenance-1"
INTEGRITY_SCHEMA = "sireto-v4.12-consumed-compatibility-integrity-1"
PAYLOAD_MANIFEST_SCHEMA = "sireto-v4.12-payload-manifest-1"
SEAL_SCHEMA = "sireto-v4.12-consumed-compatibility-seal-1"
RECEIPT_SCHEMA = "sireto-v4.12-consumed-compatibility-attempt-receipt-1"
EVENT_SCHEMA = "sireto-v4.12-consumed-compatibility-event-1"
EVENT_MANIFEST_SCHEMA = "sireto-v4.12-consumed-compatibility-events-manifest-1"
EXECUTION_LOCK_SCHEMA = (
    "sireto-v4.12-consumed-compatibility-execution-lock-1"
)

SERVICE_DOMAIN = b"SIRETO-V412-SERVICE-ID-LINEAGE\0"
SIRET_DOMAIN = b"SIRETO-V412-INPUT-SIRET-LINEAGE\0"
HEX64 = frozenset("0123456789abcdef")
KEYCHAIN_SERVICE = "com.sireto.v412.compatibility-hmac"
KEYCHAIN_ACCOUNT = "SIRETO"
HISTORICAL_CSV_BOM_POLICY = "REQUIRED_EXACTLY_ONE_AT_FILE_START"
UTF8_BOM = b"\xef\xbb\xbf"
CRM_COLUMNS = (
    "SITE",
    "CODE_POSTAL",
    "CODE_INSEE",
    "SERVICE ID",
    "COMMUNE",
    "SIRET",
    "SITE_CLI_ADRESSE",
    "SITE_CLI_COMMUNE",
)
PAYLOAD_FILES = (
    "compatibility_rows.parquet",
    "service_id_keyset.parquet",
    "input_siret_lineage_keyset.parquet",
    "siret_masked_keyset.parquet",
    "fuzzy_historical_observations.parquet",
    "fuzzy_historical_keyset.parquet",
    "provenance.json",
    "rejected_rows.parquet",
    "integrity.json",
)


class CompatibilityRegistryError(RuntimeError):
    """Closed failure carrying the contract verdict in its message."""


def _stop(message: str, verdict: str = "STOP_V412_COMPATIBILITY_REGISTRY") -> None:
    raise CompatibilityRegistryError(f"{verdict}: {message}")


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def canonical_json_without_lf(value: Any) -> bytes:
    return canonical_json(value)[:-1]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def is_hex64(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(HEX64)
    )


def canonical_text(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    return " ".join(text.strip().upper().split())


def normalize_siret(value: Any) -> str:
    digits = "".join(character for character in canonical_text(value) if character.isdigit())
    return digits if len(digits) == 14 else ""


def fuzzy_text(value: Any) -> str:
    if value is None:
        return ""
    decomposed = unicodedata.normalize("NFKD", str(value))
    without_marks = "".join(
        character
        for character in decomposed
        if unicodedata.category(character) != "Mn"
    )
    lowered = without_marks.lower()
    spaced = "".join(
        character if character.isalnum() else " " for character in lowered
    )
    return " ".join(spaced.split())


def _canonical_fingerprint(payload: Mapping[str, str]) -> str:
    return sha256_bytes(canonical_json_without_lf(dict(payload)))


def v411_row_fingerprint(row: Mapping[str, Any]) -> str:
    return _canonical_fingerprint(
        {column: canonical_text(row.get(column)) for column in CRM_COLUMNS}
    )


def siret_masked_fingerprint(row: Mapping[str, Any]) -> str:
    projection = {
        column: canonical_text("" if column == "SIRET" else row.get(column))
        for column in CRM_COLUMNS
    }
    return _canonical_fingerprint(projection)


def fuzzy_singletons(row: Mapping[str, Any]) -> list[tuple[int, str, str]]:
    cities = sorted(
        {
            value
            for value in (
                fuzzy_text(row.get("COMMUNE")),
                fuzzy_text(row.get("SITE_CLI_COMMUNE")),
            )
            if value
        }
    )
    if not cities:
        cities = [""]
    base = {
        "name": fuzzy_text(row.get("SITE")),
        "address": fuzzy_text(row.get("SITE_CLI_ADRESSE")),
        "postcode": fuzzy_text(row.get("CODE_POSTAL")),
        "insee": fuzzy_text(row.get("CODE_INSEE")),
    }
    return [
        (
            ordinal,
            city,
            _canonical_fingerprint({**base, "city": city}),
        )
        for ordinal, city in enumerate(cities)
    ]


def lineage_hmac(
    key: bytes | bytearray, domain: bytes, normalized_value: str
) -> str:
    if not normalized_value:
        _stop("empty HMAC value")
    return hmac.new(
        key,
        domain + normalized_value.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def service_lineage_hmac(key: bytes | bytearray, value: Any) -> str | None:
    normalized = canonical_text(value)
    return lineage_hmac(key, SERVICE_DOMAIN, normalized) if normalized else None


def input_siret_lineage_hmac(
    key: bytes | bytearray, value: Any
) -> str | None:
    normalized = normalize_siret(value)
    return lineage_hmac(key, SIRET_DOMAIN, normalized) if normalized else None


def _schema(fields: Sequence[tuple[str, pa.DataType, bool]]) -> pa.Schema:
    return pa.schema(
        [pa.field(name, data_type, nullable=nullable) for name, data_type, nullable in fields],
        metadata=None,
    )


COMPATIBILITY_SCHEMA = _schema(
    (
        ("source_row_number", pa.int64(), False),
        ("effective_consumed", pa.bool_(), False),
        ("consumption_reason", pa.string(), False),
        ("service_id_present", pa.bool_(), False),
        ("input_siret_present", pa.bool_(), False),
        ("v411_row_fingerprint_sha256", pa.string(), False),
        ("siret_masked_fingerprint_sha256", pa.string(), False),
        ("fuzzy_fingerprint_count", pa.uint8(), False),
        ("service_id_lineage_hmac_sha256", pa.string(), True),
        ("input_siret_lineage_hmac_sha256", pa.string(), True),
    )
)
SERVICE_KEYSET_SCHEMA = _schema(
    (
        ("service_id_lineage_hmac_sha256", pa.string(), False),
        ("row_count", pa.uint32(), False),
    )
)
INPUT_SIRET_KEYSET_SCHEMA = _schema(
    (
        ("input_siret_lineage_hmac_sha256", pa.string(), False),
        ("row_count", pa.uint32(), False),
    )
)
MASKED_KEYSET_SCHEMA = _schema(
    (
        ("siret_masked_fingerprint_sha256", pa.string(), False),
        ("row_count", pa.uint32(), False),
    )
)
FUZZY_OBSERVATIONS_SCHEMA = _schema(
    (
        ("source_row_number", pa.int64(), False),
        ("city_ordinal", pa.uint8(), False),
        ("fuzzy_historical_fingerprint_sha256", pa.string(), False),
    )
)
FUZZY_KEYSET_SCHEMA = _schema(
    (
        ("fuzzy_historical_fingerprint_sha256", pa.string(), False),
        ("row_count", pa.uint32(), False),
    )
)
REJECTED_SCHEMA = _schema(
    (
        ("source_row_number", pa.int64(), False),
        ("reason_code", pa.string(), False),
    )
)


TABLE_SCHEMAS = {
    "compatibility_rows.parquet": COMPATIBILITY_SCHEMA,
    "service_id_keyset.parquet": SERVICE_KEYSET_SCHEMA,
    "input_siret_lineage_keyset.parquet": INPUT_SIRET_KEYSET_SCHEMA,
    "siret_masked_keyset.parquet": MASKED_KEYSET_SCHEMA,
    "fuzzy_historical_observations.parquet": FUZZY_OBSERVATIONS_SCHEMA,
    "fuzzy_historical_keyset.parquet": FUZZY_KEYSET_SCHEMA,
    "rejected_rows.parquet": REJECTED_SCHEMA,
}


@dataclass(frozen=True)
class RegistryTables:
    compatibility_rows: pa.Table
    service_keyset: pa.Table
    input_siret_keyset: pa.Table
    masked_keyset: pa.Table
    fuzzy_observations: pa.Table
    fuzzy_keyset: pa.Table
    rejected_rows: pa.Table
    integrity: dict[str, Any]

    def parquet_payloads(self) -> dict[str, pa.Table]:
        return {
            "compatibility_rows.parquet": self.compatibility_rows,
            "service_id_keyset.parquet": self.service_keyset,
            "input_siret_lineage_keyset.parquet": self.input_siret_keyset,
            "siret_masked_keyset.parquet": self.masked_keyset,
            "fuzzy_historical_observations.parquet": self.fuzzy_observations,
            "fuzzy_historical_keyset.parquet": self.fuzzy_keyset,
            "rejected_rows.parquet": self.rejected_rows,
        }


def _counter_table(
    counter: Counter[str], key_name: str, schema: pa.Schema
) -> pa.Table:
    rows = [
        {key_name: key, "row_count": count}
        for key, count in sorted(counter.items())
    ]
    return pa.Table.from_pylist(rows, schema=schema)


def _table(rows: Sequence[Mapping[str, Any]], schema: pa.Schema) -> pa.Table:
    return pa.Table.from_pylist(list(rows), schema=schema).combine_chunks()


def build_registry_tables(
    historical_rows: Sequence[Mapping[str, Any]],
    *,
    hmac_key: bytes | bytearray,
    challenge_source_rows: Iterable[int] = (),
    expected_rows: int | None = None,
    expected_empty_service_ids: int | None = None,
) -> RegistryTables:
    if len(hmac_key) < 32:
        _stop("HMAC key must contain at least 256 bits")
    challenge_rows = set(challenge_source_rows)
    compatibility: list[dict[str, Any]] = []
    fuzzy_observations: list[dict[str, Any]] = []
    service_counts: Counter[str] = Counter()
    input_siret_counts: Counter[str] = Counter()
    masked_counts: Counter[str] = Counter()
    fuzzy_counts: Counter[str] = Counter()
    seen_row_numbers: set[int] = set()
    empty_services = 0

    for position, raw in enumerate(historical_rows, start=1):
        source_row_number = int(raw.get("source_row_number", position))
        if source_row_number in seen_row_numbers:
            _stop(f"duplicate source_row_number {source_row_number}")
        seen_row_numbers.add(source_row_number)
        row = {column: raw.get(column) for column in CRM_COLUMNS}
        service_norm = canonical_text(row["SERVICE ID"])
        siret_norm = normalize_siret(row["SIRET"])
        if not service_norm:
            empty_services += 1
        service_digest = (
            lineage_hmac(hmac_key, SERVICE_DOMAIN, service_norm)
            if service_norm
            else None
        )
        input_digest = (
            lineage_hmac(hmac_key, SIRET_DOMAIN, siret_norm)
            if siret_norm
            else None
        )
        masked = siret_masked_fingerprint(row)
        fuzzy = fuzzy_singletons(row)
        original = v411_row_fingerprint(row)
        if service_digest:
            service_counts[service_digest] += 1
        if input_digest:
            input_siret_counts[input_digest] += 1
        masked_counts[masked] += 1
        for ordinal, _city, digest in fuzzy:
            fuzzy_counts[digest] += 1
            fuzzy_observations.append(
                {
                    "source_row_number": source_row_number,
                    "city_ordinal": ordinal,
                    "fuzzy_historical_fingerprint_sha256": digest,
                }
            )
        compatibility.append(
            {
                "source_row_number": source_row_number,
                "effective_consumed": True,
                "consumption_reason": (
                    "V411_CHALLENGE_225"
                    if source_row_number in challenge_rows
                    else "V411_HISTORICAL_CONSUMED"
                ),
                "service_id_present": bool(service_norm),
                "input_siret_present": bool(siret_norm),
                "v411_row_fingerprint_sha256": original,
                "siret_masked_fingerprint_sha256": masked,
                "fuzzy_fingerprint_count": len(fuzzy),
                "service_id_lineage_hmac_sha256": service_digest,
                "input_siret_lineage_hmac_sha256": input_digest,
            }
        )

    compatibility.sort(key=lambda row: row["source_row_number"])
    fuzzy_observations.sort(
        key=lambda row: (
            row["source_row_number"],
            row["city_ordinal"],
            row["fuzzy_historical_fingerprint_sha256"],
        )
    )
    expected_sequence = list(range(1, len(historical_rows) + 1))
    if [row["source_row_number"] for row in compatibility] != expected_sequence:
        _stop("source row numbers must be the exact contiguous range")
    if expected_rows is not None and len(compatibility) != expected_rows:
        _stop(f"expected {expected_rows} rows, observed {len(compatibility)}")
    if (
        expected_empty_service_ids is not None
        and empty_services != expected_empty_service_ids
    ):
        _stop(
            "empty service count mismatch: "
            f"{empty_services} != {expected_empty_service_ids}"
        )
    if not challenge_rows.issubset(seen_row_numbers):
        _stop("challenge rows are not a subset of historical source rows")

    integrity = {
        "schema_version": INTEGRITY_SCHEMA,
        "build_id": "",
        "counts": {
            "compatibility_rows": len(compatibility),
            "challenge_rows": len(challenge_rows),
            "empty_service_ids": empty_services,
            "nonempty_service_ids": len(compatibility) - empty_services,
            "fuzzy_observations": len(fuzzy_observations),
            "rejected_rows": 0,
        },
        "multiplicity_sums": {
            "service_id": sum(service_counts.values()),
            "input_siret": sum(input_siret_counts.values()),
            "siret_masked": sum(masked_counts.values()),
            "fuzzy": sum(fuzzy_counts.values()),
        },
        "invariants": {
            "source_rows_contiguous": True,
            "all_effective_consumed": True,
            "zero_rejected_rows": True,
            "keysets_unique_sorted": True,
            "multiplicities_positive": True,
            "fuzzy_one_or_two_per_row": all(
                row["fuzzy_fingerprint_count"] in (1, 2)
                for row in compatibility
            ),
        },
    }
    return RegistryTables(
        compatibility_rows=_table(compatibility, COMPATIBILITY_SCHEMA),
        service_keyset=_counter_table(
            service_counts,
            "service_id_lineage_hmac_sha256",
            SERVICE_KEYSET_SCHEMA,
        ),
        input_siret_keyset=_counter_table(
            input_siret_counts,
            "input_siret_lineage_hmac_sha256",
            INPUT_SIRET_KEYSET_SCHEMA,
        ),
        masked_keyset=_counter_table(
            masked_counts,
            "siret_masked_fingerprint_sha256",
            MASKED_KEYSET_SCHEMA,
        ),
        fuzzy_observations=_table(
            fuzzy_observations, FUZZY_OBSERVATIONS_SCHEMA
        ),
        fuzzy_keyset=_counter_table(
            fuzzy_counts,
            "fuzzy_historical_fingerprint_sha256",
            FUZZY_KEYSET_SCHEMA,
        ),
        rejected_rows=_table([], REJECTED_SCHEMA),
        integrity=integrity,
    )


def _read_all_fd(fd: int) -> bytes:
    os.lseek(fd, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


def read_hmac_key_from_fd(
    fd: int,
    *,
    expected_sha256: str | None,
    require_regular: bool = True,
) -> bytearray:
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    if flags & os.O_ACCMODE != os.O_RDONLY:
        _stop("HMAC key descriptor is not read-only")
    fd_flags = fcntl.fcntl(fd, fcntl.F_GETFD)
    if not (fd_flags & fcntl.FD_CLOEXEC):
        _stop("HMAC key descriptor is not O_CLOEXEC")
    info_before = os.fstat(fd)
    if require_regular and (
        not stat.S_ISREG(info_before.st_mode)
        or info_before.st_nlink != 1
        or info_before.st_uid != os.getuid()
        or stat.S_IMODE(info_before.st_mode) != 0o600
    ):
        _stop(
            "HMAC key descriptor is not a mode-0600, current-UID, "
            "private regular single-link file"
        )
    key = _read_all_fd(fd)
    info_after = os.fstat(fd)
    identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_uid", "st_nlink")
    if any(getattr(info_before, item) != getattr(info_after, item) for item in identity):
        _stop("HMAC key descriptor changed during read")
    if len(key) < 32:
        _stop("HMAC key contains fewer than 256 bits")
    observed = sha256_bytes(key)
    if expected_sha256 is not None and observed != expected_sha256:
        _stop("HMAC key hash differs from execution lock")
    return bytearray(key)


@dataclass(frozen=True)
class FdHmacKeyProvider:
    """Test-only provider for unit tests which already own a private FD."""

    key_id: str
    descriptor: int
    expected_sha256: str

    def load(self, *, expected_key_id: str) -> bytearray:
        if self.key_id != expected_key_id:
            _stop("HMAC key ID differs from plan")
        return read_hmac_key_from_fd(
            self.descriptor,
            expected_sha256=self.expected_sha256,
        )


class HmacKeyProvider(Protocol):
    def load(self, *, expected_key_id: str) -> bytearray:
        """Return an already validated secret without persisting it."""


def _framework_constant(library: ctypes.CDLL, name: str) -> ctypes.c_void_p:
    try:
        value = ctypes.c_void_p.in_dll(library, name).value
    except ValueError:
        _stop(f"required macOS security constant unavailable: {name}")
    if value is None:
        _stop(f"required macOS security constant is null: {name}")
    return ctypes.c_void_p(value)


def _load_keychain_frameworks() -> tuple[ctypes.CDLL, ctypes.CDLL]:
    try:
        return (
            ctypes.CDLL(
                "/System/Library/Frameworks/Security.framework/Security"
            ),
            ctypes.CDLL(
                "/System/Library/Frameworks/"
                "CoreFoundation.framework/CoreFoundation"
            ),
        )
    except OSError as exc:
        _stop(f"cannot load macOS Keychain frameworks: {exc}")


def _copy_keychain_generic_password_no_ui() -> bytearray:
    """Read the pinned generic-password item with authentication UI disabled."""
    if platform.system() != "Darwin":
        _stop("Keychain HMAC retrieval requires macOS")
    security, core = _load_keychain_frameworks()

    core.CFDictionaryCreateMutable.argtypes = [
        ctypes.c_void_p,
        ctypes.c_long,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    core.CFDictionaryCreateMutable.restype = ctypes.c_void_p
    core.CFDictionarySetValue.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    core.CFDictionarySetValue.restype = None
    core.CFStringCreateWithCString.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_uint32,
    ]
    core.CFStringCreateWithCString.restype = ctypes.c_void_p
    core.CFGetTypeID.argtypes = [ctypes.c_void_p]
    core.CFGetTypeID.restype = ctypes.c_ulong
    core.CFDataGetTypeID.argtypes = []
    core.CFDataGetTypeID.restype = ctypes.c_ulong
    core.CFDataGetLength.argtypes = [ctypes.c_void_p]
    core.CFDataGetLength.restype = ctypes.c_long
    core.CFDataGetBytePtr.argtypes = [ctypes.c_void_p]
    core.CFDataGetBytePtr.restype = ctypes.POINTER(ctypes.c_uint8)
    core.CFRelease.argtypes = [ctypes.c_void_p]
    core.CFRelease.restype = None
    security.SecItemCopyMatching.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    security.SecItemCopyMatching.restype = ctypes.c_int32

    query_value = core.CFDictionaryCreateMutable(None, 0, None, None)
    if not query_value:
        _stop("cannot allocate Keychain query")
    query = ctypes.c_void_p(query_value)
    created: list[ctypes.c_void_p] = []
    result = ctypes.c_void_p()
    try:
        utf8 = 0x08000100
        for raw in (KEYCHAIN_SERVICE, KEYCHAIN_ACCOUNT):
            value = core.CFStringCreateWithCString(
                None,
                raw.encode("utf-8"),
                utf8,
            )
            if not value:
                _stop("cannot allocate pinned Keychain locator")
            created.append(ctypes.c_void_p(value))
        pairs = (
            (
                _framework_constant(security, "kSecClass"),
                _framework_constant(security, "kSecClassGenericPassword"),
            ),
            (
                _framework_constant(security, "kSecAttrService"),
                created[0],
            ),
            (
                _framework_constant(security, "kSecAttrAccount"),
                created[1],
            ),
            (
                _framework_constant(security, "kSecReturnData"),
                _framework_constant(core, "kCFBooleanTrue"),
            ),
            (
                _framework_constant(security, "kSecMatchLimit"),
                _framework_constant(security, "kSecMatchLimitOne"),
            ),
            (
                _framework_constant(security, "kSecUseAuthenticationUI"),
                _framework_constant(security, "kSecUseAuthenticationUIFail"),
            ),
        )
        for key, value in pairs:
            core.CFDictionarySetValue(
                query.value,
                key.value,
                value.value,
            )
        status = security.SecItemCopyMatching(
            query.value,
            ctypes.byref(result),
        )
        if status != 0:
            # Never ask Security.framework for a human-readable error: it can
            # include item metadata. Numeric OSStatus is sufficient to audit.
            _stop(f"Keychain HMAC unavailable without UI (OSStatus {status})")
        if not result.value or (
            core.CFGetTypeID(result.value) != core.CFDataGetTypeID()
        ):
            _stop("Keychain HMAC result is not data")
        length = core.CFDataGetLength(result.value)
        pointer = core.CFDataGetBytePtr(result.value)
        if length < 0 or (length and not pointer):
            _stop("Keychain HMAC data is invalid")
        return bytearray(pointer[:length])
    finally:
        if result.value:
            core.CFRelease(result.value)
        for value in created:
            core.CFRelease(value.value)
        core.CFRelease(query.value)


@dataclass(frozen=True)
class MacOSKeychainHmacKeyProvider:
    logical_key_id: str
    expected_sha256: str

    def load(self, *, expected_key_id: str) -> bytearray:
        if self.logical_key_id != expected_key_id:
            _stop("HMAC key ID differs between plan and execution lock")
        key = _copy_keychain_generic_password_no_ui()
        if not isinstance(key, bytearray):
            key = bytearray(key)
        if len(key) < 32:
            zeroize_secret(key)
            _stop("Keychain HMAC contains fewer than 256 bits")
        if sha256_bytes(key) != self.expected_sha256:
            zeroize_secret(key)
            _stop("Keychain HMAC hash differs from execution lock")
        return key


def zeroize_secret(secret: bytearray) -> None:
    """Best-effort overwrite of the owned Python buffer.

    CPython and Security.framework may transiently hold internal copies which
    Python cannot control. The builder owns only this mutable buffer, never
    serialises it, and overwrites it immediately after HMAC table projection.
    """
    secret[:] = b"\0" * len(secret)


def validate_provider_key(
    provider: HmacKeyProvider,
    *,
    plan_key_id: str,
    lock_key_id: str,
    lock_key_sha256: str,
) -> bytearray:
    """Revalidate an injected provider at the run boundary."""
    if lock_key_id != plan_key_id:
        _stop("HMAC key ID differs between plan and execution lock")
    supplied = provider.load(expected_key_id=plan_key_id)
    if not isinstance(supplied, (bytes, bytearray)):
        _stop("HMAC provider returned an invalid secret type")
    secret = supplied if isinstance(supplied, bytearray) else bytearray(supplied)
    if len(secret) < 32:
        zeroize_secret(secret)
        _stop("HMAC key contains fewer than 256 bits")
    if sha256_bytes(secret) != lock_key_sha256:
        zeroize_secret(secret)
        _stop("HMAC key hash differs from execution lock")
    return secret


def _openat_anchored(path: Path, *, directory: bool) -> tuple[Path, int]:
    absolute = path.absolute()
    if not absolute.is_absolute() or ".." in absolute.parts or absolute == Path("/"):
        _stop(f"unsafe anchored path: {path}")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    current = os.open("/", directory_flags)
    try:
        for index, component in enumerate(absolute.parts[1:]):
            final = index == len(absolute.parts[1:]) - 1
            flags = (
                directory_flags
                if not final or directory
                else os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            child = os.open(component, flags, dir_fd=current)
            os.close(current)
            current = child
        info = os.fstat(current)
        expected = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
        if not expected:
            _stop(f"anchored path has wrong type: {absolute}")
        return absolute, current
    except BaseException:
        os.close(current)
        raise


def _mkdirs_anchored(path: Path, *, final_mode: int = 0o700) -> None:
    absolute = path.absolute()
    if not absolute.is_absolute() or ".." in absolute.parts or absolute == Path("/"):
        _stop(f"unsafe directory path: {path}")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    current = os.open("/", flags)
    try:
        components = absolute.parts[1:]
        for index, component in enumerate(components):
            final = index == len(components) - 1
            try:
                child = os.open(component, flags, dir_fd=current)
            except FileNotFoundError:
                os.mkdir(component, final_mode if final else 0o700, dir_fd=current)
                child = os.open(component, flags, dir_fd=current)
            os.close(current)
            current = child
        info = os.fstat(current)
        if not stat.S_ISDIR(info.st_mode):
            _stop(f"created path is not a directory: {path}")
        if stat.S_IMODE(info.st_mode) != final_mode:
            _stop(f"private root mode mismatch: {path}")
    finally:
        os.close(current)


def _mkdir_exclusive_anchored(path: Path, *, mode: int = 0o700) -> None:
    _parent, parent_fd = _openat_anchored(path.parent, directory=True)
    try:
        os.mkdir(path.name, mode, dir_fd=parent_fd)
        child = os.open(
            path.name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        try:
            info = os.fstat(child)
            if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != mode:
                _stop(f"exclusive directory mode mismatch: {path}")
        finally:
            os.close(child)
    finally:
        os.close(parent_fd)


def _open_directory_at(parent_fd: int, name: str) -> int:
    if not name or name in {".", ".."} or "/" in name:
        _stop(f"unsafe relative directory name: {name}")
    fd = os.open(
        name,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=parent_fd,
    )
    info = os.fstat(fd)
    if not stat.S_ISDIR(info.st_mode):
        os.close(fd)
        _stop(f"relative path is not a directory: {name}")
    return fd


def _mkdir_exclusive_at(
    parent_fd: int, name: str, *, mode: int = 0o700
) -> int:
    if not name or name in {".", ".."} or "/" in name:
        _stop(f"unsafe relative directory name: {name}")
    os.mkdir(name, mode, dir_fd=parent_fd)
    fd = _open_directory_at(parent_fd, name)
    info = os.fstat(fd)
    if stat.S_IMODE(info.st_mode) != mode:
        os.close(fd)
        _stop(f"exclusive directory mode mismatch: {name}")
    return fd


@dataclass(frozen=True)
class FileSnapshot:
    data: bytes
    sha256: str
    size: int
    device: int
    inode: int
    uid: int
    nlink: int


@dataclass(frozen=True)
class IdentityPin:
    uid: int
    device: int
    volume_uuid: str


@dataclass(frozen=True)
class ExecutionLock:
    plan_path: Path
    plan_sha256: str
    builder_git_commit: str
    builder_source_sha256: str
    tests_path: Path
    tests_sha256: str
    hmac_key_id: str
    hmac_key_sha256: str
    attempt_id: str
    identity_pins: Mapping[str, IdentityPin]
    output_identity_pin: IdentityPin


@dataclass(frozen=True)
class TreeValidation:
    root_device: int
    root_inode: int
    publication: Mapping[str, str]


@dataclass(frozen=True)
class AttemptChainValidation:
    receipt: Mapping[str, Any]
    complete_events: Sequence[Mapping[str, Any]]
    orphan_event_paths: Sequence[Path]


class _DarwinFsid(ctypes.Structure):
    _fields_ = [("val", ctypes.c_int32 * 2)]


class _DarwinStatfs(ctypes.Structure):
    # Darwin 24 / macOS 15 struct statfs, from <sys/mount.h>.
    _fields_ = [
        ("f_bsize", ctypes.c_uint32),
        ("f_iosize", ctypes.c_int32),
        ("f_blocks", ctypes.c_uint64),
        ("f_bfree", ctypes.c_uint64),
        ("f_bavail", ctypes.c_uint64),
        ("f_files", ctypes.c_uint64),
        ("f_ffree", ctypes.c_uint64),
        ("f_fsid", _DarwinFsid),
        ("f_owner", ctypes.c_uint32),
        ("f_type", ctypes.c_uint32),
        ("f_flags", ctypes.c_uint32),
        ("f_fssubtype", ctypes.c_uint32),
        ("f_fstypename", ctypes.c_char * 16),
        ("f_mntonname", ctypes.c_char * 1024),
        ("f_mntfromname", ctypes.c_char * 1024),
        ("f_flags_ext", ctypes.c_uint32),
        ("f_reserved", ctypes.c_uint32 * 7),
    ]


def _darwin_fstatfs(fd: int) -> tuple[str, str]:
    try:
        libc = ctypes.CDLL(
            "/usr/lib/libSystem.B.dylib",
            use_errno=True,
        )
    except OSError as exc:
        _stop(f"cannot load macOS mount API: {exc}")
    function = libc.fstatfs
    function.argtypes = [ctypes.c_int, ctypes.POINTER(_DarwinStatfs)]
    function.restype = ctypes.c_int
    result = _DarwinStatfs()
    if function(fd, ctypes.byref(result)) != 0:
        error = ctypes.get_errno()
        _stop(f"cannot resolve mount from descriptor: errno {error}")

    def decode(value: bytes, field: str) -> str:
        try:
            decoded = value.split(b"\0", 1)[0].decode("utf-8", "strict")
        except UnicodeDecodeError:
            _stop(f"invalid UTF-8 in macOS {field}")
        if not decoded:
            _stop(f"empty macOS {field}")
        return decoded

    return (
        decode(bytes(result.f_mntfromname), "mounted device"),
        decode(bytes(result.f_mntonname), "mount point"),
    )


def _open_anchored_identity(path: Path) -> tuple[Path, int]:
    """Open a regular file or directory without following any path symlink."""
    absolute = path.absolute()
    if not absolute.is_absolute() or ".." in absolute.parts or absolute == Path("/"):
        _stop(f"unsafe anchored path: {path}")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    current = os.open("/", directory_flags)
    try:
        components = absolute.parts[1:]
        for index, component in enumerate(components):
            final = index == len(components) - 1
            flags = (
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | (0 if final else getattr(os, "O_DIRECTORY", 0))
            )
            child = os.open(component, flags, dir_fd=current)
            os.close(current)
            current = child
        info = os.fstat(current)
        if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
            _stop(f"anchored identity path has wrong type: {absolute}")
        return absolute, current
    except BaseException:
        os.close(current)
        raise


def macos_volume_uuid(
    path: Path,
    *,
    expected_device: int | None = None,
    expected_inode: int | None = None,
) -> str:
    if platform.system() != "Darwin":
        _stop("volume UUID verification requires macOS")
    absolute, fd = _open_anchored_identity(path)
    try:
        before = os.fstat(fd)
        if (
            expected_device is not None
            and before.st_dev != expected_device
        ) or (
            expected_inode is not None
            and before.st_ino != expected_inode
        ):
            _stop(f"path identity changed before volume lookup: {absolute}")
        mounted_device, mount_point = _darwin_fstatfs(fd)
        if (
            not mounted_device.startswith("/dev/")
            or mounted_device != os.path.normpath(mounted_device)
            or "\0" in mounted_device
        ):
            _stop(f"unexpected mounted device for {absolute}")
        result = subprocess.run(
            ["/usr/sbin/diskutil", "info", "-plist", mounted_device],
            check=False,
            capture_output=True,
            env={"LANG": "C", "LC_ALL": "C"},
            close_fds=True,
        )
        if result.returncode != 0 or result.stderr:
            _stop(f"cannot resolve volume UUID for {absolute}")
        try:
            payload = plistlib.loads(result.stdout)
        except (plistlib.InvalidFileException, ValueError, TypeError):
            _stop(f"invalid diskutil plist for {absolute}")
        if not isinstance(payload, dict):
            _stop(f"invalid diskutil payload for {absolute}")
        if payload.get("DeviceNode") != mounted_device:
            _stop(f"diskutil device mismatch for {absolute}")
        if payload.get("MountPoint") != mount_point:
            _stop(f"diskutil mount-point mismatch for {absolute}")
        value = payload.get("VolumeUUID")
        if not isinstance(value, str) or not value:
            _stop(f"volume UUID absent for {absolute}")
        after_device, after_mount_point = _darwin_fstatfs(fd)
        after = os.fstat(fd)
        identity_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_uid",
            "st_nlink",
        )
        if any(
            getattr(before, field) != getattr(after, field)
            for field in identity_fields
        ) or (mounted_device, mount_point) != (
            after_device,
            after_mount_point,
        ):
            _stop(f"path or mount changed during volume lookup: {absolute}")
        return value.upper()
    finally:
        os.close(fd)


def load_identity_pins(
    path: Path, *, expected_sha256: str
) -> tuple[dict[str, IdentityPin], IdentityPin]:
    snapshot = read_pinned_file(
        path,
        expected_sha256=expected_sha256,
        expected_size=None,
        expected_uid=os.getuid(),
    )
    try:
        payload = json.loads(snapshot.data)
    except json.JSONDecodeError as exc:
        _stop(f"identity-pin lock is invalid JSON: {exc}")
    if canonical_json(payload) != snapshot.data:
        _stop("identity-pin lock is not canonical")
    if payload.get("schema_version") != (
        "sireto-v4.12-consumed-compatibility-identity-pins-1"
    ):
        _stop("identity-pin lock schema mismatch")

    def parse(value: Mapping[str, Any]) -> IdentityPin:
        pin = IdentityPin(
            uid=int(value["uid"]),
            device=int(value["device"]),
            volume_uuid=str(value["volume_uuid"]).upper(),
        )
        if pin.uid < 0 or pin.device < 0 or not pin.volume_uuid:
            _stop("invalid identity pin")
        return pin

    files = {
        str(Path(file_path).absolute()): parse(value)
        for file_path, value in payload["files"].items()
    }
    return files, parse(payload["output_root"])


def _parse_identity_pin(value: Mapping[str, Any]) -> IdentityPin:
    if set(value) != {"uid", "device", "volume_uuid"}:
        _stop("identity pin fields mismatch")
    pin = IdentityPin(
        uid=int(value["uid"]),
        device=int(value["device"]),
        volume_uuid=str(value["volume_uuid"]).upper(),
    )
    if pin.uid < 0 or pin.device < 0 or not pin.volume_uuid:
        _stop("invalid identity pin")
    return pin


def load_execution_lock(
    path: Path, *, expected_sha256: str
) -> ExecutionLock:
    snapshot = read_pinned_file(
        path,
        expected_sha256=expected_sha256,
        expected_size=None,
        expected_uid=os.getuid(),
    )
    try:
        payload = json.loads(snapshot.data)
    except json.JSONDecodeError as exc:
        _stop(f"execution lock is invalid JSON: {exc}")
    if canonical_json(payload) != snapshot.data:
        _stop("execution lock is not canonical")
    exact_fields = {
        "schema_version",
        "plan",
        "builder",
        "tests",
        "hmac",
        "attempt_id",
        "identity",
    }
    if set(payload) != exact_fields:
        _stop("execution lock fields mismatch")
    if payload["schema_version"] != EXECUTION_LOCK_SCHEMA:
        _stop("execution lock schema mismatch")
    if set(payload["plan"]) != {"path", "sha256"}:
        _stop("execution lock plan fields mismatch")
    if set(payload["builder"]) != {
        "git_commit",
        "path",
        "source_sha256",
    }:
        _stop("execution lock builder fields mismatch")
    if set(payload["tests"]) != {"path", "sha256"}:
        _stop("execution lock tests fields mismatch")
    if set(payload["hmac"]) != {"key_id", "key_sha256"}:
        _stop("execution lock HMAC fields mismatch")
    if set(payload["identity"]) != {"files", "output_root"}:
        _stop("execution lock identity fields mismatch")
    builder_path = Path(payload["builder"]["path"]).absolute()
    if builder_path != Path(__file__).absolute():
        _stop("execution lock builder path mismatch")
    hash_fields = (
        payload["plan"]["sha256"],
        payload["builder"]["source_sha256"],
        payload["tests"]["sha256"],
        payload["hmac"]["key_sha256"],
    )
    if not all(is_hex64(value) for value in hash_fields):
        _stop("execution lock contains an invalid SHA-256")
    if (
        not payload["builder"]["git_commit"]
        or not payload["hmac"]["key_id"]
        or not payload["attempt_id"]
    ):
        _stop("execution lock contains an empty mandatory identifier")
    files = {
        str(Path(file_path).absolute()): _parse_identity_pin(value)
        for file_path, value in payload["identity"]["files"].items()
    }
    return ExecutionLock(
        plan_path=Path(payload["plan"]["path"]),
        plan_sha256=payload["plan"]["sha256"],
        builder_git_commit=payload["builder"]["git_commit"],
        builder_source_sha256=payload["builder"]["source_sha256"],
        tests_path=Path(payload["tests"]["path"]),
        tests_sha256=payload["tests"]["sha256"],
        hmac_key_id=payload["hmac"]["key_id"],
        hmac_key_sha256=payload["hmac"]["key_sha256"],
        attempt_id=payload["attempt_id"],
        identity_pins=files,
        output_identity_pin=_parse_identity_pin(
            payload["identity"]["output_root"]
        ),
    )


def verify_identity_pin(path: Path, snapshot: FileSnapshot, pin: IdentityPin) -> None:
    if snapshot.uid != pin.uid or snapshot.device != pin.device:
        _stop(f"UID/device pin mismatch: {path}")
    if macos_volume_uuid(
        path,
        expected_device=snapshot.device,
        expected_inode=snapshot.inode,
    ) != pin.volume_uuid:
        _stop(f"volume UUID pin mismatch: {path}")


def _read_private_regular(
    path: Path, *, expected_mode: int | None = 0o600
) -> bytes:
    _absolute, fd = _openat_anchored(path, directory=False)
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (
                expected_mode is not None
                and stat.S_IMODE(before.st_mode) != expected_mode
            )
        ):
            _stop(f"private file metadata mismatch: {path}")
        first = _read_all_fd(fd)
        second = _read_all_fd(fd)
        after = os.fstat(fd)
        identity = (
            "st_dev",
            "st_ino",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
            "st_uid",
            "st_nlink",
            "st_mode",
        )
        if (
            first != second
            or len(first) != before.st_size
            or any(
                getattr(before, item) != getattr(after, item)
                for item in identity
            )
        ):
            _stop(f"private file changed during read: {path}")
        return first
    finally:
        os.close(fd)


def _read_private_regular_at(
    directory_fd: int,
    name: str,
    *,
    expected_mode: int | None = 0o600,
) -> bytes:
    if not name or name in {".", ".."} or "/" in name:
        _stop(f"unsafe relative file name: {name}")
    fd = os.open(
        name,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=directory_fd,
    )
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (
                expected_mode is not None
                and stat.S_IMODE(before.st_mode) != expected_mode
            )
        ):
            _stop(f"private relative file metadata mismatch: {name}")
        first = _read_all_fd(fd)
        second = _read_all_fd(fd)
        after = os.fstat(fd)
        identity = (
            "st_dev",
            "st_ino",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
            "st_uid",
            "st_nlink",
            "st_mode",
        )
        if (
            first != second
            or len(first) != before.st_size
            or any(
                getattr(before, item) != getattr(after, item)
                for item in identity
            )
        ):
            _stop(f"private relative file changed during read: {name}")
        return first
    finally:
        os.close(fd)


def read_pinned_file(
    path: Path,
    *,
    expected_sha256: str,
    expected_size: int | None,
    expected_uid: int | None = None,
    expected_device: int | None = None,
) -> FileSnapshot:
    _absolute, fd = _openat_anchored(path, directory=False)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            _stop(f"input is not a regular single-link file: {path}")
        if expected_uid is not None and before.st_uid != expected_uid:
            _stop(f"input UID drift: {path}")
        if expected_device is not None and before.st_dev != expected_device:
            _stop(f"input device drift: {path}")
        first = _read_all_fd(fd)
        first_hash = sha256_bytes(first)
        second = _read_all_fd(fd)
        after = os.fstat(fd)
        identity = (
            "st_dev",
            "st_ino",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
            "st_uid",
            "st_nlink",
        )
        if any(getattr(before, item) != getattr(after, item) for item in identity):
            _stop(f"input changed on same descriptor: {path}")
        if first != second:
            _stop(f"input bytes changed on same descriptor: {path}")
        if len(first) != before.st_size:
            _stop(f"short or overlong input read: {path}")
        if expected_size is not None and len(first) != expected_size:
            _stop(f"input size drift: {path}")
        if first_hash != expected_sha256:
            _stop(f"input hash drift: {path}", verdict="STOP_INPUT_DRIFT")
        return FileSnapshot(
            data=first,
            sha256=first_hash,
            size=len(first),
            device=after.st_dev,
            inode=after.st_ino,
            uid=after.st_uid,
            nlink=after.st_nlink,
        )
    finally:
        os.close(fd)


def parse_historical_csv(
    data: bytes,
    *,
    leading_utf8_bom: str,
) -> list[dict[str, str]]:
    if leading_utf8_bom != HISTORICAL_CSV_BOM_POLICY:
        _stop("historical CSV BOM policy is absent or unsupported")
    if not data.startswith(UTF8_BOM):
        _stop(
            "historical CSV required leading UTF-8 BOM is absent",
            "STOP_INPUT_DRIFT",
        )
    payload = data[len(UTF8_BOM) :]
    if payload.startswith(UTF8_BOM):
        _stop(
            "historical CSV contains more than one leading UTF-8 BOM",
            "STOP_INPUT_DRIFT",
        )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        _stop(f"historical CSV is not UTF-8: {exc}", "STOP_INPUT_DRIFT")
    reader = csv.DictReader(io.StringIO(text, newline=""), delimiter=";")
    if tuple(reader.fieldnames or ()) != CRM_COLUMNS:
        _stop("historical CSV columns drift", "STOP_INPUT_DRIFT")
    rows: list[dict[str, str]] = []
    for row in reader:
        if None in row or set(row) != set(CRM_COLUMNS):
            _stop("historical CSV row shape drift", "STOP_INPUT_DRIFT")
        rows.append({column: row[column] for column in CRM_COLUMNS})
    return rows


def _schema_digest(schema: pa.Schema) -> str:
    return sha256_bytes(schema.serialize().to_pybytes())


def _load_pinned_parquet(
    snapshot: FileSnapshot,
    *,
    expected_rows: int,
    expected_row_groups: int,
    expected_schema_sha256: str,
) -> pa.Table:
    parquet = pq.ParquetFile(pa.BufferReader(snapshot.data))
    if parquet.metadata.num_rows != expected_rows:
        _stop("Parquet row count drift", "STOP_INPUT_DRIFT")
    if parquet.metadata.num_row_groups != expected_row_groups:
        _stop("Parquet row-group drift", "STOP_INPUT_DRIFT")
    if _schema_digest(parquet.schema_arrow) != expected_schema_sha256:
        _stop("Parquet Arrow schema drift", "STOP_INPUT_DRIFT")
    return parquet.read()


def validate_plan(plan: Mapping[str, Any], raw_bytes: bytes) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA:
        _stop("plan schema mismatch")
    if canonical_json(plan) != raw_bytes:
        _stop("plan is not canonical JSON")
    build = plan.get("build", {})
    if build.get("execute_now") is not False:
        _stop("plan unexpectedly authorizes immediate execution")
    if build.get("status") != "PREREGISTERED_DO_NOT_EXECUTE":
        _stop("plan status mismatch")
    historical_raw = plan.get("inputs", {}).get("historical_raw", {})
    if (
        historical_raw.get("leading_utf8_bom")
        != HISTORICAL_CSV_BOM_POLICY
    ):
        _stop("historical CSV BOM policy missing or mismatched in plan")
    if tuple(plan["outputs"]["payload_files_exact"]) != PAYLOAD_FILES:
        _stop("payload list mismatch")
    for filename, schema in TABLE_SCHEMAS.items():
        declared = plan["outputs"]["files"][filename]["arrow_schema"]
        observed = [
            {"name": field.name, "nullable": field.nullable, "type": str(field.type)}
            for field in schema
        ]
        if declared != observed:
            _stop(f"output Arrow schema mismatch: {filename}")
    writer = plan["writer"]
    expected_writer = {
        "format_version": "2.6",
        "compression": "zstd",
        "compression_level": 9,
        "use_dictionary": False,
        "write_statistics": True,
        "data_page_version": "1.0",
        "row_group_size": 65536,
        "store_schema": True,
    }
    if any(writer.get(key) != value for key, value in expected_writer.items()):
        _stop("writer options drift")


def validate_runtime(plan: Mapping[str, Any]) -> None:
    expected = plan["runtime"]
    observed = {
        "architecture": platform.machine(),
        "os": "macOS" if platform.system() == "Darwin" else platform.system(),
        "pandas_serialization_allowed": False,
        "pyarrow": pa.__version__,
        "python": ".".join(str(item) for item in sys.version_info[:3]),
    }
    if observed != expected:
        _stop(f"runtime drift: expected {expected}, observed {observed}")


def validate_audited_git_state(
    *,
    repo_root: Path,
    audited_commit: str,
    artifact_hashes: Mapping[Path, str],
) -> str:
    if (
        len(audited_commit) != 40
        or any(character not in "0123456789abcdef" for character in audited_commit)
    ):
        _stop("audited Git commit must be a full lowercase SHA-1")
    resolved_root = repo_root.resolve()

    def git(
        arguments: Sequence[str], *, text: bool = False
    ) -> subprocess.CompletedProcess[Any]:
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("GIT_")
        }
        environment.update(
            {
                "GIT_NO_REPLACE_OBJECTS": "1",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": "/dev/null",
                "LC_ALL": "C",
                "LANG": "C",
            }
        )
        return subprocess.run(
            ["/usr/bin/git", "--no-replace-objects", *arguments],
            cwd=resolved_root,
            check=False,
            capture_output=True,
            text=text,
            env=environment,
        )

    top_level = git(["rev-parse", "--show-toplevel"], text=True)
    if (
        top_level.returncode != 0
        or Path(top_level.stdout.strip()).resolve() != resolved_root
    ):
        _stop("Git top-level differs from audited repository root")
    audited_type = git(["cat-file", "-t", audited_commit], text=True)
    if (
        audited_type.returncode != 0
        or audited_type.stdout.strip() != "commit"
    ):
        _stop("audited Git object is not a commit")
    head_result = git(["rev-parse", "--verify", "HEAD"], text=True)
    if head_result.returncode != 0:
        _stop("cannot resolve current Git HEAD")
    current_head = head_result.stdout.strip()
    if (
        len(current_head) != 40
        or any(
            character not in "0123456789abcdef"
            for character in current_head
        )
    ):
        _stop("current Git HEAD is not a full lowercase SHA-1")
    head_type = git(["cat-file", "-t", current_head], text=True)
    if head_type.returncode != 0 or head_type.stdout.strip() != "commit":
        _stop("current Git HEAD object is not a commit")
    ancestor_result = git(
        ["merge-base", "--is-ancestor", audited_commit, current_head]
    )
    if ancestor_result.returncode != 0:
        _stop("audited builder commit is not an ancestor of HEAD")
    for artifact_path, expected_sha256 in artifact_hashes.items():
        if not is_hex64(expected_sha256):
            _stop("invalid audited artifact hash")
        try:
            relative = artifact_path.resolve().relative_to(resolved_root)
        except ValueError:
            _stop(f"audited artifact is outside repository: {artifact_path}")
        blob_result = git(
            [
                "cat-file",
                "blob",
                f"{audited_commit}:{relative.as_posix()}",
            ]
        )
        if blob_result.returncode != 0:
            _stop(f"audited Git blob is unavailable: {relative}")
        if sha256_bytes(blob_result.stdout) != expected_sha256:
            _stop(f"audited Git blob hash mismatch: {relative}")
    return current_head


def load_plan(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = _read_private_regular(path, expected_mode=None)
    try:
        plan = json.loads(raw)
    except json.JSONDecodeError as exc:
        _stop(f"plan JSON invalid: {exc}")
    validate_plan(plan, raw)
    contract_path = Path(plan["contract"]["path"])
    if (
        sha256_bytes(_read_private_regular(contract_path, expected_mode=None))
        != plan["contract"]["sha256"]
    ):
        _stop("contract hash differs from plan pin")
    return plan, raw


def validate_golden_vectors(plan: Mapping[str, Any]) -> None:
    golden = plan["golden_vectors"]
    test_key = bytes.fromhex(golden["hmac_test_key"]["key_hex"])
    if sha256_bytes(test_key) != golden["hmac_test_key"]["key_sha256"]:
        _stop("golden HMAC key hash mismatch")
    if not golden["hmac_test_key"].get("production_use_forbidden"):
        _stop("golden HMAC key is not marked test-only")
    for vector in golden["vectors"]:
        expected = vector.get("expected", {})
        historical = vector.get("historical")
        if historical is not None:
            observed_fuzzy = {
                (city, digest)
                for _ordinal, city, digest in fuzzy_singletons(historical)
            }
            expected_fuzzy = {
                (item["city"], item["sha256"])
                for item in expected.get("fuzzy_singletons", [])
            }
            if expected_fuzzy and observed_fuzzy != expected_fuzzy:
                _stop(f"golden fuzzy mismatch: {vector['id']}")
            if "siret_masked_fingerprint_sha256" in expected and (
                siret_masked_fingerprint(historical)
                != expected["siret_masked_fingerprint_sha256"]
            ):
                _stop(f"golden masked mismatch: {vector['id']}")
        if "service_id_lineage_hmac_sha256" in expected:
            observed = service_lineage_hmac(test_key, historical["SERVICE ID"])
            if observed != expected["service_id_lineage_hmac_sha256"]:
                _stop(f"golden service HMAC mismatch: {vector['id']}")
        if "input_siret_lineage_hmac_sha256" in expected:
            value = (historical or {}).get("SIRET") or vector.get(
                "input_siret"
            )
            if "normalized_input_siret" in expected and (
                normalize_siret(value) != expected["normalized_input_siret"]
            ):
                _stop(f"golden SIRET normalization mismatch: {vector['id']}")
            observed = input_siret_lineage_hmac(test_key, value)
            if observed != expected["input_siret_lineage_hmac_sha256"]:
                _stop(f"golden SIRET HMAC mismatch: {vector['id']}")
        if "accepted_by_python_str_isdigit" in expected:
            value = vector["input_siret"]
            if (
                all(character.isdigit() for character in value)
                != expected["accepted_by_python_str_isdigit"]
                or all(character.isdecimal() for character in value)
                != expected["accepted_by_python_str_isdecimal"]
            ):
                _stop(f"golden Unicode predicate mismatch: {vector['id']}")


def _sync_fd(fd: int, *, require_fullfsync: bool) -> None:
    os.fsync(fd)
    full = getattr(fcntl, "F_FULLFSYNC", None)
    if full is None:
        if require_fullfsync:
            _stop("F_FULLFSYNC unavailable")
        return
    try:
        fcntl.fcntl(fd, full)
    except OSError:
        if require_fullfsync:
            _stop("F_FULLFSYNC failed")


def _write_exclusive(
    path: Path,
    payload: bytes,
    *,
    require_fullfsync: bool,
) -> None:
    _parent, parent_fd = _openat_anchored(path.parent, directory=True)
    try:
        fd = os.open(
            path.name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=parent_fd,
        )
        try:
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    _stop(f"short write: {path}")
                view = view[written:]
            os.fchmod(fd, 0o600)
            _sync_fd(fd, require_fullfsync=require_fullfsync)
        finally:
            os.close(fd)
    finally:
        os.close(parent_fd)


def _write_exclusive_at(
    directory_fd: int,
    name: str,
    payload: bytes,
    *,
    require_fullfsync: bool,
) -> None:
    if not name or name in {".", ".."} or "/" in name:
        _stop(f"unsafe relative file name: {name}")
    fd = os.open(
        name,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        0o600,
        dir_fd=directory_fd,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                _stop(f"short relative write: {name}")
            view = view[written:]
        os.fchmod(fd, 0o600)
        _sync_fd(fd, require_fullfsync=require_fullfsync)
    finally:
        os.close(fd)


def _write_parquet_exclusive(
    path: Path,
    table: pa.Table,
    *,
    require_fullfsync: bool,
) -> None:
    _parent, parent_fd = _openat_anchored(path.parent, directory=True)
    try:
        fd = os.open(
            path.name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=parent_fd,
        )
        file_object = os.fdopen(fd, "wb", closefd=False)
        try:
            table = table.combine_chunks()
            pq.write_table(
                table,
                file_object,
                version="2.6",
                compression="zstd",
                compression_level=9,
                use_dictionary=False,
                write_statistics=True,
                data_page_version="1.0",
                row_group_size=65536,
                store_schema=True,
            )
            file_object.flush()
            os.fchmod(fd, 0o600)
            _sync_fd(fd, require_fullfsync=require_fullfsync)
        finally:
            file_object.close()
            os.close(fd)
    finally:
        os.close(parent_fd)


def _write_parquet_exclusive_at(
    directory_fd: int,
    name: str,
    table: pa.Table,
    *,
    require_fullfsync: bool,
) -> None:
    if not name or name in {".", ".."} or "/" in name:
        _stop(f"unsafe relative file name: {name}")
    fd = os.open(
        name,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        0o600,
        dir_fd=directory_fd,
    )
    file_object = os.fdopen(fd, "wb", closefd=False)
    try:
        pq.write_table(
            table.combine_chunks(),
            file_object,
            version="2.6",
            compression="zstd",
            compression_level=9,
            use_dictionary=False,
            write_statistics=True,
            data_page_version="1.0",
            row_group_size=65536,
            store_schema=True,
        )
        file_object.flush()
        os.fchmod(fd, 0o600)
        _sync_fd(fd, require_fullfsync=require_fullfsync)
    finally:
        file_object.close()
        os.close(fd)


def _sync_directory(path: Path, *, require_fullfsync: bool) -> None:
    _absolute, fd = _openat_anchored(path, directory=True)
    try:
        os.fchmod(fd, 0o700)
        _sync_fd(fd, require_fullfsync=require_fullfsync)
    finally:
        os.close(fd)


def _payload_records(root: Path) -> list[dict[str, Any]]:
    records = []
    for filename in PAYLOAD_FILES:
        path = root / filename
        data = _read_private_regular(path)
        records.append(
            {
                "relative_path": filename,
                "size_bytes": len(data),
                "sha256": sha256_bytes(data),
            }
        )
    return records


def _payload_records_at(root_fd: int) -> list[dict[str, Any]]:
    records = []
    for filename in PAYLOAD_FILES:
        data = _read_private_regular_at(root_fd, filename)
        records.append(
            {
                "relative_path": filename,
                "size_bytes": len(data),
                "sha256": sha256_bytes(data),
            }
        )
    return records


def logical_payload_head(records: Sequence[Mapping[str, Any]]) -> str:
    return sha256_bytes(canonical_json_without_lf(list(records)))


def compare_complete_trees(first: Path, second: Path) -> None:
    filenames = PAYLOAD_FILES + ("payload_manifest.json", "seal.json")
    for filename in filenames:
        if _read_private_regular(first / filename) != _read_private_regular(
            second / filename
        ):
            _stop(f"independent build differs byte-for-byte: {filename}")


def compare_complete_trees_at(
    parent_fd: int, first_name: str, second_name: str
) -> None:
    first_fd = _open_directory_at(parent_fd, first_name)
    second_fd = _open_directory_at(parent_fd, second_name)
    try:
        filenames = PAYLOAD_FILES + ("payload_manifest.json", "seal.json")
        for filename in filenames:
            if _read_private_regular_at(
                first_fd, filename
            ) != _read_private_regular_at(second_fd, filename):
                _stop(
                    f"independent build differs byte-for-byte: {filename}"
                )
    finally:
        os.close(first_fd)
        os.close(second_fd)


def build_specification(
    plan: Mapping[str, Any],
    *,
    builder_git_commit: str,
    builder_source_sha256: str,
    tests_sha256: str,
    hmac_key_sha256: str,
) -> dict[str, Any]:
    for name, value in (
        ("builder_git_commit", builder_git_commit),
        ("builder_source_sha256", builder_source_sha256),
        ("tests_sha256", tests_sha256),
        ("hmac_key_sha256", hmac_key_sha256),
    ):
        if name != "builder_git_commit" and not is_hex64(value):
            _stop(f"invalid build pin {name}")
        if name == "builder_git_commit" and not value:
            _stop("empty builder Git commit")
    return {
        "schema_version": PLAN_SCHEMA,
        "contract": plan["contract"],
        "plan_sha256": sha256_bytes(canonical_json(plan)),
        "inputs": plan["inputs"],
        "fingerprints": plan["fingerprints"],
        "lineage": plan["lineage"],
        "hmac_key_id": plan["hmac_lineage"]["key_id"],
        "hmac_key_sha256": hmac_key_sha256,
        "golden_vectors_sha256": sha256_bytes(
            canonical_json(plan["golden_vectors"])
        ),
        "runtime": plan["runtime"],
        "writer": plan["writer"],
        "outputs": plan["outputs"],
        "builder_git_commit": builder_git_commit,
        "builder_source_sha256": builder_source_sha256,
        "tests_sha256": tests_sha256,
    }


def build_id_for_spec(build_spec: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json(build_spec))


def _expected_provenance(
    plan: Mapping[str, Any],
    build_spec: Mapping[str, Any],
    build_id: str,
) -> dict[str, Any]:
    return {
        "schema_version": PROVENANCE_SCHEMA,
        "build_id": build_id,
        "contract_sha256": plan["contract"]["sha256"],
        "build_spec_sha256": build_id,
        "input_pins": plan["inputs"],
        "runtime": plan["runtime"],
        "writer": plan["writer"],
        "builder_commit": build_spec["builder_git_commit"],
        "builder_source_hashes": {
            "builder": build_spec["builder_source_sha256"]
        },
        "tests_sha256": build_spec["tests_sha256"],
        "hmac_key_id": build_spec["hmac_key_id"],
        "hmac_key_sha256": build_spec["hmac_key_sha256"],
    }


def write_payload_tree(
    root: Path,
    tables: RegistryTables,
    *,
    plan: Mapping[str, Any],
    build_spec: Mapping[str, Any],
    require_fullfsync: bool,
    parent_fd: int | None = None,
) -> dict[str, Any]:
    if parent_fd is None:
        _mkdir_exclusive_anchored(root, mode=0o700)
        _root_path, root_fd = _openat_anchored(root, directory=True)
    else:
        root_fd = _mkdir_exclusive_at(parent_fd, root.name, mode=0o700)
    build_id = build_id_for_spec(build_spec)
    try:
        integrity = json.loads(json.dumps(tables.integrity))
        integrity["build_id"] = build_id
        provenance = _expected_provenance(plan, build_spec, build_id)
        for filename, table in tables.parquet_payloads().items():
            if table.schema != TABLE_SCHEMAS[filename]:
                _stop(f"table schema mismatch before write: {filename}")
            _write_parquet_exclusive_at(
                root_fd,
                filename,
                table,
                require_fullfsync=require_fullfsync,
            )
        _write_exclusive_at(
            root_fd,
            "provenance.json",
            canonical_json(provenance),
            require_fullfsync=require_fullfsync,
        )
        _write_exclusive_at(
            root_fd,
            "integrity.json",
            canonical_json(integrity),
            require_fullfsync=require_fullfsync,
        )
        records = _payload_records_at(root_fd)
        payload_manifest = {
            "schema_version": PAYLOAD_MANIFEST_SCHEMA,
            "payload_files": records,
        }
        manifest_bytes = canonical_json(payload_manifest)
        _write_exclusive_at(
            root_fd,
            "payload_manifest.json",
            manifest_bytes,
            require_fullfsync=require_fullfsync,
        )
        seal = {
            "schema_version": SEAL_SCHEMA,
            "build_id": build_id,
            "build_spec_sha256": build_id,
            "payload_manifest_size_bytes": len(manifest_bytes),
            "payload_manifest_sha256": sha256_bytes(manifest_bytes),
            "payload_logical_head_sha256": logical_payload_head(records),
        }
        seal_bytes = canonical_json(seal)
        _write_exclusive_at(
            root_fd,
            "seal.json",
            seal_bytes,
            require_fullfsync=require_fullfsync,
        )
        os.fchmod(root_fd, 0o700)
        _sync_fd(root_fd, require_fullfsync=require_fullfsync)
        _validate_complete_tree_fd(
            root_fd,
            expected_build_id=build_id,
            plan=plan,
            build_spec=build_spec,
        )
        return {
            "build_id": build_id,
            "payload_manifest_sha256": sha256_bytes(manifest_bytes),
            "seal_sha256": sha256_bytes(seal_bytes),
            "payload_logical_head_sha256": seal[
                "payload_logical_head_sha256"
            ],
        }
    finally:
        os.close(root_fd)


def _validate_complete_tree_fd(
    root_fd: int,
    *,
    expected_build_id: str,
    plan: Mapping[str, Any],
    build_spec: Mapping[str, Any],
) -> TreeValidation:
    if build_id_for_spec(build_spec) != expected_build_id:
        _stop("expected build specification does not derive build ID")
    if build_specification(
        plan,
        builder_git_commit=build_spec["builder_git_commit"],
        builder_source_sha256=build_spec["builder_source_sha256"],
        tests_sha256=build_spec["tests_sha256"],
        hmac_key_sha256=build_spec["hmac_key_sha256"],
    ) != build_spec:
        _stop("build specification differs from plan and execution pins")
    allowed = set(PAYLOAD_FILES) | {"payload_manifest.json", "seal.json"}
    try:
        root_info = os.fstat(root_fd)
        if stat.S_IMODE(root_info.st_mode) != 0o700:
            _stop("tree root mode mismatch")
        actual = set(os.listdir(root_fd))
        if actual != allowed:
            _stop(
                "tree mismatch: "
                f"expected {sorted(allowed)}, observed {sorted(actual)}"
            )
        records = _payload_records_at(root_fd)
        manifest_bytes = _read_private_regular_at(
            root_fd, "payload_manifest.json"
        )
        manifest = json.loads(manifest_bytes)
        if canonical_json(manifest) != manifest_bytes:
            _stop("payload manifest is not canonical")
        if manifest != {
            "schema_version": PAYLOAD_MANIFEST_SCHEMA,
            "payload_files": records,
        }:
            _stop("payload manifest does not close exact payload")
        seal_bytes = _read_private_regular_at(root_fd, "seal.json")
        seal = json.loads(seal_bytes)
        if canonical_json(seal) != seal_bytes:
            _stop("seal is not canonical")
        expected_seal = {
            "schema_version": SEAL_SCHEMA,
            "build_id": expected_build_id,
            "build_spec_sha256": expected_build_id,
            "payload_manifest_size_bytes": len(manifest_bytes),
            "payload_manifest_sha256": sha256_bytes(manifest_bytes),
            "payload_logical_head_sha256": logical_payload_head(records),
        }
        if seal != expected_seal:
            _stop("seal mismatch")

        tables: dict[str, pa.Table] = {}
        for filename, schema in TABLE_SCHEMAS.items():
            parquet = pq.ParquetFile(
                pa.BufferReader(_read_private_regular_at(root_fd, filename))
            )
            if parquet.schema_arrow != schema:
                _stop(f"written schema mismatch: {filename}")
            file_contract = plan["outputs"]["files"][filename]
            expected_groups = file_contract["expected_row_groups"]
            if parquet.metadata.num_row_groups != expected_groups:
                _stop(f"written row-group mismatch: {filename}")
            if "rows" in file_contract and (
                parquet.metadata.num_rows != file_contract["rows"]
            ):
                _stop(f"written row count mismatch: {filename}")
            if "row_bounds" in file_contract:
                lower, upper = file_contract["row_bounds"]
                if not lower <= parquet.metadata.num_rows <= upper:
                    _stop(f"written row bounds mismatch: {filename}")
            tables[filename] = parquet.read()

        compatibility = tables["compatibility_rows.parquet"].to_pylist()
        fuzzy_observations = tables[
            "fuzzy_historical_observations.parquet"
        ].to_pylist()
        expected_rows = plan["invariants"]["compatibility_rows"]
        if (
            [row["source_row_number"] for row in compatibility]
            != list(range(1, expected_rows + 1))
            or not all(row["effective_consumed"] for row in compatibility)
        ):
            _stop("compatibility population invariant mismatch")
        challenge_count = sum(
            row["consumption_reason"] == "V411_CHALLENGE_225"
            for row in compatibility
        )
        if challenge_count != plan["invariants"]["challenge_rows"]:
            _stop("challenge population invariant mismatch")
        if any(
            row["consumption_reason"]
            not in {"V411_CHALLENGE_225", "V411_HISTORICAL_CONSUMED"}
            for row in compatibility
        ):
            _stop("unknown consumption reason")
        for row in compatibility:
            for name in (
                "v411_row_fingerprint_sha256",
                "siret_masked_fingerprint_sha256",
            ):
                if not is_hex64(row[name]):
                    _stop(f"invalid compatibility digest: {name}")
            for name in (
                "service_id_lineage_hmac_sha256",
                "input_siret_lineage_hmac_sha256",
            ):
                if row[name] is not None and not is_hex64(row[name]):
                    _stop(f"invalid compatibility HMAC: {name}")
            if row["service_id_present"] != (
                row["service_id_lineage_hmac_sha256"] is not None
            ) or row["input_siret_present"] != (
                row["input_siret_lineage_hmac_sha256"] is not None
            ):
                _stop("presence flags and HMAC values disagree")
        observed_fuzzy_counts = Counter(
            row["source_row_number"] for row in fuzzy_observations
        )
        fuzzy_sort_keys = [
            (
                row["source_row_number"],
                row["city_ordinal"],
                row["fuzzy_historical_fingerprint_sha256"],
            )
            for row in fuzzy_observations
        ]
        if (
            fuzzy_sort_keys != sorted(fuzzy_sort_keys)
            or any(
                row["city_ordinal"] not in (0, 1)
                or not is_hex64(
                    row["fuzzy_historical_fingerprint_sha256"]
                )
                for row in fuzzy_observations
            )
            or any(
                observed_fuzzy_counts[row["source_row_number"]]
                != row["fuzzy_fingerprint_count"]
                for row in compatibility
            )
        ):
            _stop("fuzzy observation count differs from compatibility rows")

        keyset_specs = (
            (
                "service_id_keyset.parquet",
                "service_id_lineage_hmac_sha256",
                Counter(
                    row["service_id_lineage_hmac_sha256"]
                    for row in compatibility
                    if row["service_id_lineage_hmac_sha256"] is not None
                ),
            ),
            (
                "input_siret_lineage_keyset.parquet",
                "input_siret_lineage_hmac_sha256",
                Counter(
                    row["input_siret_lineage_hmac_sha256"]
                    for row in compatibility
                    if row["input_siret_lineage_hmac_sha256"] is not None
                ),
            ),
            (
                "siret_masked_keyset.parquet",
                "siret_masked_fingerprint_sha256",
                Counter(
                    row["siret_masked_fingerprint_sha256"]
                    for row in compatibility
                ),
            ),
            (
                "fuzzy_historical_keyset.parquet",
                "fuzzy_historical_fingerprint_sha256",
                Counter(
                    row["fuzzy_historical_fingerprint_sha256"]
                    for row in fuzzy_observations
                ),
            ),
        )
        multiplicities: dict[str, int] = {}
        for filename, key_name, expected_counter in keyset_specs:
            rows = tables[filename].to_pylist()
            keys = [row[key_name] for row in rows]
            if (
                keys != sorted(set(keys))
                or not all(is_hex64(key) for key in keys)
                or not all(row["row_count"] > 0 for row in rows)
            ):
                _stop(f"keyset invariant mismatch: {filename}")
            observed_counter = Counter(
                {row[key_name]: row["row_count"] for row in rows}
            )
            if observed_counter != expected_counter:
                _stop(f"keyset exact content mismatch: {filename}")
            observed_sum = sum(row["row_count"] for row in rows)
            if observed_sum != sum(expected_counter.values()):
                _stop(f"keyset multiplicity mismatch: {filename}")
            multiplicities[filename] = observed_sum

        integrity_bytes = _read_private_regular_at(root_fd, "integrity.json")
        provenance_bytes = _read_private_regular_at(
            root_fd, "provenance.json"
        )
        integrity = json.loads(integrity_bytes)
        provenance = json.loads(provenance_bytes)
        if canonical_json(integrity) != integrity_bytes:
            _stop("integrity file is not canonical")
        if canonical_json(provenance) != provenance_bytes:
            _stop("provenance file is not canonical")
        if provenance != _expected_provenance(
            plan, build_spec, expected_build_id
        ):
            _stop("provenance differs from plan and execution pins")
        expected_counts = {
            "compatibility_rows": len(compatibility),
            "challenge_rows": challenge_count,
            "empty_service_ids": sum(
                not row["service_id_present"] for row in compatibility
            ),
            "nonempty_service_ids": sum(
                row["service_id_present"] for row in compatibility
            ),
            "fuzzy_observations": len(fuzzy_observations),
            "rejected_rows": tables["rejected_rows.parquet"].num_rows,
        }
        if (
            expected_counts["empty_service_ids"]
            != plan["invariants"]["expected_empty_service_id"]
            or expected_counts["nonempty_service_ids"]
            != plan["invariants"]["expected_nonempty_service_id"]
            or expected_counts["rejected_rows"]
            != plan["invariants"]["expected_rejected_rows"]
        ):
            _stop("payload population counts differ from plan invariants")
        expected_multiplicity_sums = {
            "service_id": multiplicities["service_id_keyset.parquet"],
            "input_siret": multiplicities[
                "input_siret_lineage_keyset.parquet"
            ],
            "siret_masked": multiplicities[
                "siret_masked_keyset.parquet"
            ],
            "fuzzy": multiplicities["fuzzy_historical_keyset.parquet"],
        }
        if (
            integrity.get("schema_version") != INTEGRITY_SCHEMA
            or integrity.get("build_id") != expected_build_id
            or integrity.get("counts") != expected_counts
            or integrity.get("multiplicity_sums")
            != expected_multiplicity_sums
            or set(integrity.get("invariants", {}))
            != {
                "source_rows_contiguous",
                "all_effective_consumed",
                "zero_rejected_rows",
                "keysets_unique_sorted",
                "multiplicities_positive",
                "fuzzy_one_or_two_per_row",
            }
            or not all(integrity["invariants"].values())
        ):
            _stop("integrity invariants do not match payload")
        publication = {
            "build_id": expected_build_id,
            "payload_manifest_sha256": sha256_bytes(manifest_bytes),
            "seal_sha256": sha256_bytes(seal_bytes),
            "payload_logical_head_sha256": seal[
                "payload_logical_head_sha256"
            ],
        }
        return TreeValidation(
            root_device=root_info.st_dev,
            root_inode=root_info.st_ino,
            publication=publication,
        )
    finally:
        pass


def validate_complete_tree(
    root: Path,
    *,
    expected_build_id: str,
    plan: Mapping[str, Any],
    build_spec: Mapping[str, Any],
    parent_fd: int | None = None,
) -> TreeValidation:
    if parent_fd is None:
        _absolute, root_fd = _openat_anchored(root, directory=True)
    else:
        root_fd = _open_directory_at(parent_fd, root.name)
    try:
        return _validate_complete_tree_fd(
            root_fd,
            expected_build_id=expected_build_id,
            plan=plan,
            build_spec=build_spec,
        )
    finally:
        os.close(root_fd)


def create_attempt_receipt(
    attempts_root: Path,
    *,
    attempt_id: str,
    plan_sha256: str,
    input_pins_sha256: str,
    require_fullfsync: bool,
) -> Path:
    _mkdirs_anchored(attempts_root, final_mode=0o700)
    attempt = attempts_root / attempt_id
    _mkdir_exclusive_anchored(attempt, mode=0o700)
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "attempt_id": attempt_id,
        "plan_sha256": plan_sha256,
        "input_pins_sha256": input_pins_sha256,
    }
    _write_exclusive(
        attempt / "receipt.json",
        canonical_json(receipt),
        require_fullfsync=require_fullfsync,
    )
    _mkdir_exclusive_anchored(attempt / "events", mode=0o700)
    _mkdir_exclusive_anchored(attempt / "events_manifests", mode=0o700)
    _sync_directory(attempt, require_fullfsync=require_fullfsync)
    return attempt


def append_attempt_event(
    attempt_root: Path,
    *,
    event_type: str,
    fields: Mapping[str, Any],
    require_fullfsync: bool,
) -> str:
    events = attempt_root / "events"
    existing = sorted(events.iterdir())
    state = validate_attempt_chain(attempt_root)
    if state.orphan_event_paths:
        _stop("orphan event is preserved and may never be applied")
    sequence = len(existing) + 1
    previous = sha256_bytes(
        _read_private_regular(attempt_root / "receipt.json")
    )
    if existing:
        previous = existing[-1].stem.split("-", 1)[1]
    event = {
        "schema_version": EVENT_SCHEMA,
        "sequence": sequence,
        "event_type": event_type,
        "previous_event_sha256": previous,
        "fields": dict(fields),
    }
    event_bytes = canonical_json(event)
    event_hash = sha256_bytes(event_bytes)
    event_name = f"{sequence:08d}-{event_hash}.json"
    _write_exclusive(
        events / event_name,
        event_bytes,
        require_fullfsync=require_fullfsync,
    )
    records = []
    for child in sorted(events.iterdir()):
        data = _read_private_regular(child)
        records.append(
            {
                "relative_path": f"events/{child.name}",
                "size_bytes": len(data),
                "sha256": sha256_bytes(data),
            }
        )
    manifest = {
        "schema_version": EVENT_MANIFEST_SCHEMA,
        "generation": sequence,
        "event_count": sequence,
        "head_event_sha256": event_hash,
        "ordered_event_records": records,
    }
    manifest_bytes = canonical_json(manifest)
    manifest_hash = sha256_bytes(manifest_bytes)
    _write_exclusive(
        attempt_root
        / "events_manifests"
        / f"{sequence:08d}-{manifest_hash}.json",
        manifest_bytes,
        require_fullfsync=require_fullfsync,
    )
    _sync_directory(events, require_fullfsync=require_fullfsync)
    _sync_directory(
        attempt_root / "events_manifests",
        require_fullfsync=require_fullfsync,
    )
    return event_hash


def validate_attempt_chain(attempt_root: Path) -> AttemptChainValidation:
    receipt = attempt_root / "receipt.json"
    if not receipt.is_file():
        _stop("attempt receipt missing")
    receipt_bytes = _read_private_regular(receipt)
    receipt_payload = json.loads(receipt_bytes)
    if (
        canonical_json(receipt_payload) != receipt_bytes
        or receipt_payload.get("schema_version") != RECEIPT_SCHEMA
    ):
        _stop("attempt receipt is not canonical or has wrong schema")
    events = sorted((attempt_root / "events").iterdir())
    manifests = sorted((attempt_root / "events_manifests").iterdir())
    if len(manifests) > len(events):
        _stop("event generation count exceeds event count")
    previous = sha256_bytes(receipt_bytes)
    event_hashes: list[str] = []
    event_payloads: list[Mapping[str, Any]] = []
    for sequence, child in enumerate(events, start=1):
        data = _read_private_regular(child)
        digest = sha256_bytes(data)
        if child.name != f"{sequence:08d}-{digest}.json":
            _stop("event filename/hash mismatch")
        event = json.loads(data)
        if canonical_json(event) != data or (
            event["sequence"] != sequence
            or event["previous_event_sha256"] != previous
        ):
            _stop("event chain mismatch")
        previous = digest
        event_hashes.append(digest)
        event_payloads.append(event)
    for generation, child in enumerate(manifests, start=1):
        data = _read_private_regular(child)
        digest = sha256_bytes(data)
        if child.name != f"{generation:08d}-{digest}.json":
            _stop("event manifest filename/hash mismatch")
        manifest = json.loads(data)
        if (
            canonical_json(manifest) != data
            or manifest["generation"] != generation
            or manifest["event_count"] != generation
            or manifest["head_event_sha256"] != event_hashes[generation - 1]
            or len(manifest["ordered_event_records"]) != generation
        ):
            _stop("event manifest generation mismatch")
        for sequence, record in enumerate(
            manifest["ordered_event_records"], start=1
        ):
            expected_event = events[sequence - 1]
            if record["relative_path"] != f"events/{expected_event.name}":
                _stop("event manifest order/path mismatch")
            event_data = _read_private_regular(expected_event)
            if (
                len(event_data) != record["size_bytes"]
                or sha256_bytes(event_data) != record["sha256"]
            ):
                _stop("event manifest record mismatch")
    return AttemptChainValidation(
        receipt=receipt_payload,
        complete_events=tuple(event_payloads[: len(manifests)]),
        orphan_event_paths=tuple(events[len(manifests) :]),
    )


def _rename_exclusive(
    source: Path,
    destination: Path,
    *,
    expected_root_device: int,
    expected_root_inode: int,
) -> None:
    if source.parent.absolute() == destination.parent.absolute():
        _parent_path, parent_fd = _openat_anchored(
            source.parent, directory=True
        )
        try:
            _rename_exclusive_at(
                parent_fd,
                source.name,
                destination.name,
                expected_root_device=expected_root_device,
                expected_root_inode=expected_root_inode,
            )
        finally:
            os.close(parent_fd)
        return
    source_parent_path, source_parent_fd = _openat_anchored(
        source.parent, directory=True
    )
    destination_parent_path, destination_parent_fd = _openat_anchored(
        destination.parent, directory=True
    )
    _source_path, source_fd = _openat_anchored(source, directory=True)
    source_identity = os.fstat(source_fd)
    try:
        if (
            source_identity.st_dev,
            source_identity.st_ino,
        ) != (expected_root_device, expected_root_inode):
            _stop("staging tree identity changed after validation")
        if os.fstat(source_parent_fd).st_dev != os.fstat(destination_parent_fd).st_dev:
            _stop("promotion crosses filesystem")
        libc = ctypes.CDLL(None, use_errno=True)
        rename = getattr(libc, "renameatx_np", None)
        if rename is None:
            _stop("renameatx_np unavailable")
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(
            source_parent_fd,
            os.fsencode(source.name),
            destination_parent_fd,
            os.fsencode(destination.name),
            0x00000004,  # RENAME_EXCL
        )
        if result != 0:
            error = ctypes.get_errno()
            _stop(
                f"exclusive promotion failed: {os.strerror(error)} "
                f"({source_parent_path} -> {destination_parent_path})"
            )
        _destination_path, destination_fd = _openat_anchored(
            destination, directory=True
        )
        try:
            destination_identity = os.fstat(destination_fd)
            if (
                destination_identity.st_dev,
                destination_identity.st_ino,
            ) != (source_identity.st_dev, source_identity.st_ino):
                _stop("promotion changed tree identity")
        finally:
            os.close(destination_fd)
        _sync_fd(destination_parent_fd, require_fullfsync=True)
    finally:
        os.close(source_fd)
        os.close(source_parent_fd)
        os.close(destination_parent_fd)


def _rename_exclusive_at(
    parent_fd: int,
    source_name: str,
    destination_name: str,
    *,
    expected_root_device: int,
    expected_root_inode: int,
) -> None:
    source_fd = _open_directory_at(parent_fd, source_name)
    try:
        source_identity = os.fstat(source_fd)
        if (
            source_identity.st_dev,
            source_identity.st_ino,
        ) != (expected_root_device, expected_root_inode):
            _stop("staging tree identity changed after validation")
        libc = ctypes.CDLL(None, use_errno=True)
        rename = getattr(libc, "renameatx_np", None)
        if rename is None:
            _stop("renameatx_np unavailable")
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(
            parent_fd,
            os.fsencode(source_name),
            parent_fd,
            os.fsencode(destination_name),
            0x00000004,
        )
        if result != 0:
            error = ctypes.get_errno()
            _stop(f"exclusive promotion failed: {os.strerror(error)}")
        destination_fd = _open_directory_at(parent_fd, destination_name)
        try:
            destination_identity = os.fstat(destination_fd)
            if (
                destination_identity.st_dev,
                destination_identity.st_ino,
            ) != (source_identity.st_dev, source_identity.st_ino):
                _stop("promotion changed tree identity")
        finally:
            os.close(destination_fd)
        _sync_fd(parent_fd, require_fullfsync=True)
    finally:
        os.close(source_fd)


def recover_validated_tree(
    staging: Path,
    destination: Path,
    *,
    expected_build_id: str,
    plan: Mapping[str, Any],
    build_spec: Mapping[str, Any],
    attempt_root: Path,
    expected_attempt_id: str,
    expected_plan_sha256: str,
    expected_input_pins_sha256: str,
    output_root_fd: int | None = None,
) -> None:
    chain = validate_attempt_chain(attempt_root)
    expected_receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "attempt_id": expected_attempt_id,
        "plan_sha256": expected_plan_sha256,
        "input_pins_sha256": expected_input_pins_sha256,
    }
    if chain.receipt != expected_receipt:
        _stop("attempt receipt differs from execution lock")
    validation = validate_complete_tree(
        staging,
        expected_build_id=expected_build_id,
        plan=plan,
        build_spec=build_spec,
        parent_fd=output_root_fd,
    )
    if not chain.complete_events:
        _stop("no complete event generation authorizes recovery")
    latest = chain.complete_events[-1]
    if (
        latest.get("event_type") != "TREE_VALIDATED"
        or latest.get("fields") != validation.publication
    ):
        _stop("latest complete generation does not authorize this tree")
    if output_root_fd is None:
        _rename_exclusive(
            staging,
            destination,
            expected_root_device=validation.root_device,
            expected_root_inode=validation.root_inode,
        )
    else:
        _rename_exclusive_at(
            output_root_fd,
            staging.name,
            destination.name,
            expected_root_device=validation.root_device,
            expected_root_inode=validation.root_inode,
        )


def recover_existing_attempt(
    *,
    output_root: Path,
    output_root_fd: int,
    attempt_root: Path,
    attempt_id: str,
    plan: Mapping[str, Any],
    plan_sha256: str,
    build_spec: Mapping[str, Any],
    input_pins_sha256: str,
    require_fullfsync: bool,
) -> Path:
    build_id = build_id_for_spec(build_spec)
    chain = validate_attempt_chain(attempt_root)
    expected_receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "attempt_id": attempt_id,
        "plan_sha256": plan_sha256,
        "input_pins_sha256": input_pins_sha256,
    }
    if chain.receipt != expected_receipt:
        _stop("existing attempt receipt differs from execution lock")
    if not chain.complete_events:
        _stop("existing attempt has no complete recoverable generation")
    destination = output_root / build_id
    staging = output_root / f".tmp-{build_id}-{attempt_id}-primary"
    latest = chain.complete_events[-1]

    def validate_named_tree(name: str) -> TreeValidation:
        return validate_complete_tree(
            output_root / name,
            expected_build_id=build_id,
            plan=plan,
            build_spec=build_spec,
            parent_fd=output_root_fd,
        )

    try:
        destination_validation = validate_named_tree(build_id)
    except FileNotFoundError:
        destination_validation = None
    if destination_validation is not None:
        if latest["event_type"] == "TREE_PROMOTED":
            expected_fields = {
                **destination_validation.publication,
                "destination": str(destination),
            }
            if latest["fields"] != expected_fields:
                _stop("promoted event differs from validated destination")
        elif latest["event_type"] != "TREE_VALIDATED":
            _stop("destination exists without a recoverable closed event")
        elif latest["fields"] != destination_validation.publication:
            _stop("validated destination differs from tree event")
        return destination

    if latest["event_type"] != "TREE_VALIDATED":
        _stop("existing attempt is not at a recoverable tree generation")
    recover_validated_tree(
        staging,
        destination,
        expected_build_id=build_id,
        plan=plan,
        build_spec=build_spec,
        attempt_root=attempt_root,
        expected_attempt_id=attempt_id,
        expected_plan_sha256=plan_sha256,
        expected_input_pins_sha256=input_pins_sha256,
        output_root_fd=output_root_fd,
    )
    if not chain.orphan_event_paths:
        append_attempt_event(
            attempt_root,
            event_type="TREE_PROMOTED",
            fields={**latest["fields"], "destination": str(destination)},
            require_fullfsync=require_fullfsync,
        )
        validate_attempt_chain(attempt_root)
    return destination


def _validate_v411_parity(
    historical_rows: Sequence[Mapping[str, Any]],
    source_registry: pa.Table,
    consumed: pa.Table,
    unseen: pa.Table,
) -> set[int]:
    if source_registry.num_rows != len(historical_rows):
        _stop("CSV/source registry row count mismatch", "STOP_INPUT_DRIFT")
    source_rows = source_registry.to_pylist()
    for position, (raw, registered) in enumerate(
        zip(historical_rows, source_rows, strict=True), start=1
    ):
        if registered["source_row_number"] != position:
            _stop("source registry numbering drift", "STOP_INPUT_DRIFT")
        for column in CRM_COLUMNS:
            if (registered[column] or "") != (raw[column] or ""):
                _stop(f"raw field drift at row {position}", "STOP_INPUT_DRIFT")
        if registered["service_id_norm"] != canonical_text(raw["SERVICE ID"]):
            _stop(f"service normalization drift at row {position}")
        if registered["input_siret_norm"] != normalize_siret(raw["SIRET"]):
            _stop(f"SIRET normalization drift at row {position}")
        if registered["row_fingerprint_sha256"] != v411_row_fingerprint(raw):
            _stop(f"V4.11 fingerprint drift at row {position}")
    consumed_rows = {
        int(value)
        for value in consumed.column("source_row_number").to_pylist()
    }
    unseen_rows = {
        int(value) for value in unseen.column("source_row_number").to_pylist()
    }
    if consumed_rows & unseen_rows or consumed_rows | unseen_rows != set(
        range(1, len(historical_rows) + 1)
    ):
        _stop("consumed/unseen partition mismatch")
    return unseen_rows


def run_build(
    *,
    execution_lock: ExecutionLock,
    key_provider: HmacKeyProvider | None = None,
    require_fullfsync: bool = True,
) -> Path:
    def locked_snapshot(
        source_path: Path,
        *,
        expected_sha256: str,
        expected_size: int | None,
    ) -> FileSnapshot:
        pin = execution_lock.identity_pins.get(str(source_path.absolute()))
        if pin is None:
            _stop(f"identity pin missing: {source_path}")
        snapshot = read_pinned_file(
            source_path,
            expected_sha256=expected_sha256,
            expected_size=expected_size,
            expected_uid=pin.uid,
            expected_device=pin.device,
        )
        verify_identity_pin(source_path, snapshot, pin)
        return snapshot

    plan_snapshot = locked_snapshot(
        execution_lock.plan_path,
        expected_sha256=execution_lock.plan_sha256,
        expected_size=None,
    )
    plan, plan_bytes = load_plan(execution_lock.plan_path)
    if plan_bytes != plan_snapshot.data:
        _stop("plan changed between execution-lock validation and parsing")
    locked_snapshot(
        Path(plan["contract"]["path"]),
        expected_sha256=plan["contract"]["sha256"],
        expected_size=None,
    )
    validate_runtime(plan)
    locked_snapshot(
        Path(__file__).absolute(),
        expected_sha256=execution_lock.builder_source_sha256,
        expected_size=None,
    )
    locked_snapshot(
        execution_lock.tests_path,
        expected_sha256=execution_lock.tests_sha256,
        expected_size=None,
    )
    validate_audited_git_state(
        repo_root=Path(__file__).resolve().parent.parent,
        audited_commit=execution_lock.builder_git_commit,
        artifact_hashes={
            Path(__file__).absolute(): execution_lock.builder_source_sha256,
            execution_lock.tests_path: execution_lock.tests_sha256,
        },
    )
    validate_golden_vectors(plan)
    build_spec = build_specification(
        plan,
        builder_git_commit=execution_lock.builder_git_commit,
        builder_source_sha256=execution_lock.builder_source_sha256,
        tests_sha256=execution_lock.tests_sha256,
        hmac_key_sha256=execution_lock.hmac_key_sha256,
    )
    build_id = build_id_for_spec(build_spec)
    output_root = Path(plan["build"]["output_root"])
    _mkdirs_anchored(output_root, final_mode=0o700)
    _output_path, output_fd = _openat_anchored(output_root, directory=True)
    output_stat = os.fstat(output_fd)
    output_snapshot = FileSnapshot(
        data=b"",
        sha256=sha256_bytes(b""),
        size=0,
        device=output_stat.st_dev,
        inode=output_stat.st_ino,
        uid=output_stat.st_uid,
        nlink=output_stat.st_nlink,
    )
    verify_identity_pin(
        output_root, output_snapshot, execution_lock.output_identity_pin
    )
    attempts_root = Path(plan["outputs"]["attempts_root"])
    _mkdirs_anchored(attempts_root, final_mode=0o700)
    attempt_root = attempts_root / execution_lock.attempt_id
    plan_sha256 = sha256_bytes(plan_bytes)
    input_pins_sha256 = sha256_bytes(canonical_json(plan["inputs"]))
    if attempt_root.exists():
        try:
            return recover_existing_attempt(
                output_root=output_root,
                output_root_fd=output_fd,
                attempt_root=attempt_root,
                attempt_id=execution_lock.attempt_id,
                plan=plan,
                plan_sha256=plan_sha256,
                build_spec=build_spec,
                input_pins_sha256=input_pins_sha256,
                require_fullfsync=require_fullfsync,
            )
        finally:
            os.close(output_fd)
    provider = key_provider or MacOSKeychainHmacKeyProvider(
        logical_key_id=execution_lock.hmac_key_id,
        expected_sha256=execution_lock.hmac_key_sha256,
    )
    # Recovery above is deliberately keyless and source-free. Only a fresh
    # attempt crosses this boundary, before receipt creation or source reads.
    expected_key_id = plan["hmac_lineage"]["key_id"]
    key = validate_provider_key(
        provider,
        plan_key_id=expected_key_id,
        lock_key_id=execution_lock.hmac_key_id,
        lock_key_sha256=execution_lock.hmac_key_sha256,
    )

    def project_fresh_attempt(
        secret: bytearray,
    ) -> tuple[Path, RegistryTables]:
        fresh_attempt_root = create_attempt_receipt(
            attempts_root,
            attempt_id=execution_lock.attempt_id,
            plan_sha256=plan_sha256,
            input_pins_sha256=input_pins_sha256,
            require_fullfsync=require_fullfsync,
        )
        append_attempt_event(
            fresh_attempt_root,
            event_type="ATTEMPT_RECEIPTED",
            fields={"plan_sha256": sha256_bytes(plan_bytes)},
            require_fullfsync=require_fullfsync,
        )

        def pinned(spec: Mapping[str, Any]) -> FileSnapshot:
            source_path = Path(spec["path"])
            return locked_snapshot(
                source_path,
                expected_sha256=spec["sha256"],
                expected_size=spec.get("size_bytes"),
            )

        raw_spec = plan["inputs"]["historical_raw"]
        historical_rows = parse_historical_csv(
            pinned(raw_spec).data,
            leading_utf8_bom=raw_spec.get("leading_utf8_bom", ""),
        )
        registry_spec = plan["inputs"]["v411_registry"]
        schema_hash = registry_spec["arrow_schema"]["ipc_serialized_sha256"]
        loaded = {}
        for name in ("source_registry", "consumed", "unseen"):
            spec = registry_spec[name]
            loaded[name] = _load_pinned_parquet(
                pinned(spec),
                expected_rows=spec["rows"],
                expected_row_groups=spec["row_groups"],
                expected_schema_sha256=schema_hash,
            )
        pinned(registry_spec["manifest"])
        for spec in plan["inputs"]["challenge_225"].values():
            challenge_spec = {
                "path": spec["manifest_path"],
                "sha256": spec["manifest_sha256"],
                "size_bytes": spec["size_bytes"],
            }
            snapshot = pinned(challenge_spec)
            json.loads(snapshot.data)
        challenge_rows = _validate_v411_parity(
            historical_rows,
            loaded["source_registry"],
            loaded["consumed"],
            loaded["unseen"],
        )
        append_attempt_event(
            fresh_attempt_root,
            event_type="INPUTS_VALIDATED",
            fields={"source_rows": len(historical_rows)},
            require_fullfsync=require_fullfsync,
        )
        projected = build_registry_tables(
            historical_rows,
            hmac_key=secret,
            challenge_source_rows=challenge_rows,
            expected_rows=plan["invariants"]["compatibility_rows"],
            expected_empty_service_ids=plan["invariants"][
                "expected_empty_service_id"
            ],
        )
        return fresh_attempt_root, projected

    try:
        attempt_root, tables = project_fresh_attempt(key)
    finally:
        zeroize_secret(key)
    staging = output_root / (
        f".tmp-{build_id}-{execution_lock.attempt_id}-primary"
    )
    reproduction = output_root / (
        f".tmp-{build_id}-{execution_lock.attempt_id}-reproduction"
    )
    publication = write_payload_tree(
        staging,
        tables,
        plan=plan,
        build_spec=build_spec,
        require_fullfsync=require_fullfsync,
        parent_fd=output_fd,
    )
    reproduced_publication = write_payload_tree(
        reproduction,
        tables,
        plan=plan,
        build_spec=build_spec,
        require_fullfsync=require_fullfsync,
        parent_fd=output_fd,
    )
    compare_complete_trees_at(
        output_fd, staging.name, reproduction.name
    )
    if reproduced_publication != publication:
        _stop("independent build publication differs")
    append_attempt_event(
        attempt_root,
        event_type="BYTE_REPRODUCIBILITY_VALIDATED",
        fields=publication,
        require_fullfsync=require_fullfsync,
    )
    append_attempt_event(
        attempt_root,
        event_type="TREE_VALIDATED",
        fields=publication,
        require_fullfsync=require_fullfsync,
    )
    destination = output_root / build_id
    recover_validated_tree(
        staging,
        destination,
        expected_build_id=build_id,
        plan=plan,
        build_spec=build_spec,
        attempt_root=attempt_root,
        expected_attempt_id=execution_lock.attempt_id,
        expected_plan_sha256=sha256_bytes(plan_bytes),
        expected_input_pins_sha256=sha256_bytes(
            canonical_json(plan["inputs"])
        ),
        output_root_fd=output_fd,
    )
    append_attempt_event(
        attempt_root,
        event_type="TREE_PROMOTED",
        fields={**publication, "destination": str(destination)},
        require_fullfsync=require_fullfsync,
    )
    validate_attempt_chain(attempt_root)
    os.close(output_fd)
    return destination


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument("--execution-lock-sha256", required=True)
    return parser.parse_args(argv)


@contextmanager
def private_umask() -> Iterable[None]:
    previous = os.umask(0o077)
    try:
        yield
    finally:
        os.umask(previous)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    execution_lock = load_execution_lock(
        args.execution_lock,
        expected_sha256=args.execution_lock_sha256,
    )
    with private_umask():
        destination = run_build(
            execution_lock=execution_lock,
        )
    sys.stdout.write(str(destination) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
