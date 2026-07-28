"""Fail-closed, read-only stores for the V4.12 Gate A probe.

This module deliberately has no dependency on historical SIRETO modules.  It
contains the frozen name normalization needed to validate the existing TF-IDF
cache, the three strict adapters, and the sandboxed child probe CLI.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import errno
import gc
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import pickle
import re
import resource
import socket
import stat
import sys
import time
from typing import Any, BinaryIO, Iterator

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


STOP_PARTITION = "STOP_V412_STRICT_PARTITION"
STOP_TFIDF = "STOP_V412_STRICT_TFIDF"
STOP_LOOKUP = "STOP_V412_STRICT_LOOKUP"
STOP_PROBE = "STOP_V412_STRICT_STORES"

RUN_SPEC_SCHEMA = "sireto-v4.12-strict-stores-run-spec-1"
LOOKUP_DESCRIPTOR_SCHEMA = "sireto-v4.12-strict-lookup-descriptor-1"
PROBE_SCHEMA = "sireto-v4.12-strict-stores-probe-1"
SIDECAR_SCHEMA = "sireto-tfidf-cache-integrity-1"
CACHE_NAMESPACE = (
    "296c7891107249a073c00d93c7310c55a652243de4bcfa7165d09dbfc3349a82"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PARTITION_KEY_RE = re.compile(r"^(?:[0-9]{5}_|_[0-9]{5})$")
_SIRET_RE = re.compile(r"^[0-9]{14}$")
_INSEE_RELATIVE_RE = re.compile(
    r"^insee/insee=([0-9]{5})/([^/]+\.parquet)$"
)
_POSTCODE_RELATIVE_RE = re.compile(
    r"^cp/postcode=([0-9]{5})/([^/]+\.parquet)$"
)
_TFIDF_PUNCT_RE = re.compile(r"[^\w\s]")

_RUN_SPEC_KEYS = {
    "schema_version",
    "safe_input_build_id",
    "query_count",
    "routing_payload_sha256",
    "partition_records",
    "cache_records",
    "lookup_descriptor_sha256",
    "allowed_read_files",
    "staging_dir",
    "tmp_dir",
    "max_rss_bytes",
    "declarations",
}
_PARTITION_RECORD_KEYS = {"relative_path", "size_bytes", "sha256"}
_CACHE_RECORD_KEYS = {
    "partition_key",
    "pickle_relative_path",
    "pickle_size_bytes",
    "pickle_sha256",
    "sidecar_relative_path",
    "sidecar_size_bytes",
    "sidecar_sha256",
}
_ALLOWED_FILE_KEYS = {
    "role",
    "partition_key",
    "absolute_path",
    "size_bytes",
    "sha256",
}
_LOOKUP_DESCRIPTOR_KEYS = {
    "schema_version",
    "database_sha256",
    "database_size_bytes",
    "table_name",
    "columns",
    "column_types",
    "index_name",
    "index_unique",
    "row_count",
    "max_sirets_per_call",
    "read_only",
}
_DECLARATIONS = {
    "labels_opened": False,
    "oracle_opened": False,
    "historical_candidates_opened": False,
    "models_opened": False,
    "network_used": False,
    "writes_outside_staging": False,
    "cache_rebuild_attempted": False,
}

_LOOKUP_TABLE = "candidate_details"
_LOOKUP_INDEX = "candidate_details_siret_uidx"
_LOOKUP_COLUMNS = [
    "siret",
    "candidate_state",
    "enseigne1",
    "enseigne2",
    "enseigne3",
    "denomination_usuelle",
    "activity_code",
]

_NAME_FILTER_COLUMNS = (
    "denomination",
    "denomination_usuelle_ul",
    "enseigne1",
    "enseigne2",
    "enseigne3",
    "denomination_ul",
    "sigle_ul",
    "nom_ul",
    "prenom_usuel_ul",
)
_LEGAL_STOPWORDS = {
    "SAS",
    "SASU",
    "SARL",
    "EURL",
    "SCI",
    "SCIC",
    "SCOP",
    "SA",
    "SNC",
    "SELARL",
    "SELAS",
    "SELASU",
}
_PERSON_UL_CODES = {
    "1000",
    "1100",
    "1200",
    "1300",
    "1400",
    "1500",
    "1600",
    "2110",
}
_ACCENT_REPLACEMENTS = {
    "É": "E",
    "È": "E",
    "Ê": "E",
    "Ë": "E",
    "à": "a",
    "â": "a",
    "ä": "a",
    "á": "a",
    "é": "e",
    "è": "e",
    "ê": "e",
    "ë": "e",
    "î": "i",
    "ï": "i",
    "í": "i",
    "ô": "o",
    "ö": "o",
    "ó": "o",
    "û": "u",
    "ü": "u",
    "ú": "u",
    "ù": "u",
    "ç": "c",
    "À": "A",
    "Â": "A",
    "Ä": "A",
    "Á": "A",
    "Î": "I",
    "Ï": "I",
    "Í": "I",
    "Ô": "O",
    "Ö": "O",
    "Ó": "O",
    "Û": "U",
    "Ü": "U",
    "Ú": "U",
    "Ù": "U",
    "Ç": "C",
}

_COMMON_PREFIX_FIELDS = [
    ("siret", pa.string()),
    ("siren", pa.string()),
    ("denomination", pa.string()),
    ("enseigne1", pa.string()),
    ("enseigne2", pa.string()),
    ("enseigne3", pa.string()),
    ("etablissementSiege", pa.bool_()),
    ("is_siege", pa.bool_()),
    ("numeroVoie", pa.string()),
    ("typeVoie", pa.string()),
    ("libelleVoie", pa.string()),
    ("complementAdresse", pa.string()),
]
_COMMON_SUFFIX_FIELDS = [
    ("cj_ul", pa.string()),
    ("etat_admin", pa.string()),
    ("last_treatment_date", pa.timestamp("us")),
    ("sigle_ul", pa.string()),
    ("denomination_ul", pa.string()),
    ("denomination_usuelle_ul", pa.string()),
    ("nom_ul", pa.string()),
    ("prenom_usuel_ul", pa.string()),
    ("pm_dirigeant_names", pa.string()),
]
_INSEE_FIELDS = (
    _COMMON_PREFIX_FIELDS
    + [("postcode", pa.string()), ("city", pa.string())]
    + _COMMON_SUFFIX_FIELDS
)
_POSTCODE_FIELDS = (
    _COMMON_PREFIX_FIELDS
    + [("city", pa.string()), ("insee", pa.string())]
    + _COMMON_SUFFIX_FIELDS
)


class StrictStoreError(RuntimeError):
    """Base class for a fail-closed V4.12 store error."""


class StrictPartitionError(StrictStoreError):
    """A partition could not be proven to match its frozen record."""


class StrictTfidfError(StrictStoreError):
    """A TF-IDF artifact could not be proven compatible."""


class StrictLookupError(StrictStoreError):
    """The frozen lookup or a lookup request violated its contract."""


class StrictProbeError(StrictStoreError):
    """The child probe contract or sandbox check failed."""


def _fail(error_type: type[StrictStoreError], code: str, message: str) -> None:
    raise error_type(f"{code}: {message}")


def _pairs_without_duplicates(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json_bytes_strict(
    payload: bytes,
    *,
    error_type: type[StrictStoreError],
    code: str,
    label: str,
) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_pairs_without_duplicates,
        )
    except Exception as exc:
        _fail(error_type, code, f"invalid JSON for {label}: {exc}")
    if type(value) is not dict:
        _fail(error_type, code, f"{label} must be a JSON object")
    return value


def load_json_strict(path: str | Path) -> dict[str, Any]:
    """Read JSON while rejecting duplicate keys.

    This convenience entry point is intended for trusted local configuration
    files. Data-store reads use the verified-file primitive below.
    """

    supplied = Path(path)
    try:
        payload = supplied.read_bytes()
    except Exception as exc:
        _fail(StrictProbeError, STOP_PROBE, f"cannot read JSON: {exc}")
    return _load_json_bytes_strict(
        payload,
        error_type=StrictProbeError,
        code=STOP_PROBE,
        label=supplied.name,
    )


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
    except Exception as exc:
        _fail(StrictProbeError, STOP_PROBE, f"non-canonical JSON value: {exc}")


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    error_type: type[StrictStoreError],
    code: str,
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        _fail(
            error_type,
            code,
            f"{label} keyset mismatch; missing={missing}, extra={extra}",
        )


def _require_plain_int(
    value: Any,
    *,
    error_type: type[StrictStoreError],
    code: str,
    label: str,
    minimum: int = 0,
) -> int:
    if type(value) is not int or value < minimum:
        _fail(error_type, code, f"{label} must be an integer >= {minimum}")
    return value


def _require_sha256(
    value: Any,
    *,
    error_type: type[StrictStoreError],
    code: str,
    label: str,
) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        _fail(error_type, code, f"{label} is not a lowercase SHA-256")
    return value


def _safe_key(partition_key: str) -> str:
    return (
        partition_key.replace("|", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


def _validate_partition_key(
    value: Any,
    *,
    error_type: type[StrictStoreError],
    code: str,
) -> str:
    if type(value) is not str or _PARTITION_KEY_RE.fullmatch(value) is None:
        _fail(error_type, code, "invalid partition key")
    return value


def _path_chain(
    path: Path,
    *,
    error_type: type[StrictStoreError],
    code: str,
) -> list[Path]:
    if not path.is_absolute():
        _fail(error_type, code, "data path is not absolute")
    parts = path.parts
    chain = [Path(parts[0])]
    current = chain[0]
    for component in parts[1:]:
        current = current / component
        chain.append(current)
    return chain


def _snapshot_path_chain(
    path: Path,
    *,
    error_type: type[StrictStoreError],
    code: str,
    final_kind: str = "file",
) -> tuple[tuple[str, int, int, int], ...]:
    snapshots: list[tuple[str, int, int, int]] = []
    chain = _path_chain(path, error_type=error_type, code=code)
    for index, component in enumerate(chain):
        try:
            value = os.lstat(component)
        except Exception as exc:
            _fail(error_type, code, f"lstat failed for allowed path: {exc}")
        if stat.S_ISLNK(value.st_mode):
            _fail(error_type, code, "symlink in allowed path")
        if index < len(chain) - 1 and not stat.S_ISDIR(value.st_mode):
            _fail(error_type, code, "non-directory ancestor in allowed path")
        snapshots.append(
            (
                str(component),
                int(value.st_dev),
                int(value.st_ino),
                int(value.st_mode),
            )
        )
    final_mode = os.lstat(path).st_mode
    if final_kind == "file" and not stat.S_ISREG(final_mode):
        _fail(error_type, code, "allowed data path is not a regular file")
    if final_kind == "directory" and not stat.S_ISDIR(final_mode):
        _fail(error_type, code, "private path is not a directory")
    if final_kind not in {"file", "directory"}:
        _fail(error_type, code, "invalid internal final path kind")
    try:
        resolved = path.resolve(strict=True)
    except Exception as exc:
        _fail(error_type, code, f"cannot resolve allowed path: {exc}")
    if resolved != path:
        _fail(error_type, code, "allowed path is not canonical")
    return tuple(snapshots)


def _ensure_private_directory(root: Path, name: str) -> Path:
    if name not in {"output", "tmp"}:
        _fail(StrictProbeError, STOP_PROBE, "invalid private directory name")
    root_before = _snapshot_path_chain(
        root,
        error_type=StrictProbeError,
        code=STOP_PROBE,
        final_kind="directory",
    )
    path = root / name
    try:
        os.mkdir(path, 0o700)
    except FileExistsError:
        pass
    except Exception as exc:
        _fail(StrictProbeError, STOP_PROBE, f"cannot create private directory: {exc}")
    _snapshot_path_chain(
        path,
        error_type=StrictProbeError,
        code=STOP_PROBE,
        final_kind="directory",
    )
    root_after = _snapshot_path_chain(
        root,
        error_type=StrictProbeError,
        code=STOP_PROBE,
        final_kind="directory",
    )
    # Creating the direct child legitimately changes root metadata, but never
    # its device, inode, or mode; those are the only values stored here.
    if root_after != root_before:
        _fail(StrictProbeError, STOP_PROBE, "private run root changed")
    return path


def _read_regular_file_without_expected_hash(
    path: Path,
    *,
    error_type: type[StrictStoreError],
    code: str,
) -> bytes:
    """Read a private run file after lstat/no-symlink and TOCTOU checks."""

    chain_before = _snapshot_path_chain(
        path,
        error_type=error_type,
        code=code,
    )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except Exception as exc:
        _fail(error_type, code, f"cannot open private run file: {exc}")
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            _fail(error_type, code, "private run file is not regular")
        chunks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            chunks.append(block)
        after = os.fstat(descriptor)
        if _file_identity(before) != _file_identity(after):
            _fail(error_type, code, "private run file changed while open")
    except StrictStoreError:
        raise
    except Exception as exc:
        _fail(error_type, code, f"private run file read failed: {exc}")
    finally:
        os.close(descriptor)
    chain_after = _snapshot_path_chain(
        path,
        error_type=error_type,
        code=code,
    )
    if chain_after != chain_before:
        _fail(error_type, code, "private run path changed during access")
    return b"".join(chunks)


def _file_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


@contextmanager
def _verified_file(
    path: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    error_type: type[StrictStoreError],
    code: str,
) -> Iterator[BinaryIO]:
    chain_before = _snapshot_path_chain(
        path,
        error_type=error_type,
        code=code,
    )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except Exception as exc:
        _fail(error_type, code, f"open failed for allowed file: {exc}")
    handle: BinaryIO | None = None
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            _fail(error_type, code, "opened object is not a regular file")
        if int(before.st_size) != expected_size:
            _fail(error_type, code, "allowed file size mismatch")
        digest = hashlib.sha256()
        while block := os.read(descriptor, 8 * 1024 * 1024):
            digest.update(block)
        if digest.hexdigest() != expected_sha256:
            _fail(error_type, code, "allowed file SHA-256 mismatch")
        os.lseek(descriptor, 0, os.SEEK_SET)
        handle = os.fdopen(descriptor, "rb", closefd=True)
        descriptor = -1
        yield handle
        after = os.fstat(handle.fileno())
        if _file_identity(after) != _file_identity(before):
            _fail(error_type, code, "file changed while open")
    except StrictStoreError:
        raise
    except Exception as exc:
        _fail(error_type, code, f"verified file operation failed: {exc}")
    finally:
        if handle is not None:
            handle.close()
        elif descriptor >= 0:
            os.close(descriptor)
    chain_after = _snapshot_path_chain(
        path,
        error_type=error_type,
        code=code,
    )
    if chain_after != chain_before:
        _fail(error_type, code, "allowed path changed during access")


def _normalize_text(value: Any) -> str:
    if value is None or (
        isinstance(value, float) and math.isnan(value)
    ):
        return ""
    text = str(value)
    if text.strip() in ("[ND]", "[nd]", "ND", "nan", "NaN", "None"):
        return ""
    text = text.upper().replace("-", " ")
    for old, new in _ACCENT_REPLACEMENTS.items():
        text = text.replace(old, new)
    return " ".join(text.split())


def _normalize_name(value: Any, *, max_len: int = 100) -> str:
    base = _normalize_text(value)
    if not base:
        return ""
    tokens = [token for token in base.split() if token not in _LEGAL_STOPWORDS]
    cleaned = " ".join(tokens) or base
    if len(cleaned) <= max_len:
        return cleaned
    cutoff = cleaned[: max_len + 1]
    if " " in cutoff:
        cutoff = cutoff.rsplit(" ", 1)[0]
    return cutoff


def _candidate_name_values(row: Mapping[str, Any]) -> list[str]:
    output: list[str] = []

    def add(value: Any) -> None:
        normalized = _normalize_name(value)
        if (
            normalized
            and not normalized.isdigit()
            and len(normalized) > 2
        ):
            output.append(normalized)

    add(row.get("enseigne1"))
    add(row.get("denomination"))
    add(row.get("enseigne2"))
    add(row.get("enseigne3"))

    pm_names = row.get("pm_dirigeant_names") or []
    if isinstance(pm_names, str):
        pm_names = [
            item.strip()
            for item in pm_names.split("|")
            if item.strip()
        ]
    for item in pm_names:
        add(item)

    add(row.get("sigle_ul"))
    add(row.get("denomination_usuelle_ul"))
    add(row.get("denomination_ul"))

    person_name = None
    if row.get("prenom_usuel_ul") or row.get("nom_ul"):
        person_name = " ".join(
            filter(
                None,
                [row.get("prenom_usuel_ul"), row.get("nom_ul")],
            )
        )
    if row.get("cj_ul") in _PERSON_UL_CODES and person_name:
        add(person_name)
    return output


def _candidate_tfidf_text(row: Mapping[str, Any]) -> str:
    seen: set[str] = set()
    parts: list[str] = []
    for value in _candidate_name_values(row):
        if value and value not in seen:
            seen.add(value)
            parts.append(value)
    return " ".join(parts)


def _normalize_text_for_tfidf(value: Any) -> str:
    base = _normalize_text(value)
    if not base:
        return ""
    base = _TFIDF_PUNCT_RE.sub(" ", base)
    tokens = " ".join(base.split()).split()
    acronyms: list[str] = []
    buffer: list[str] = []
    for token in tokens:
        if len(token) == 1 and token.isalpha():
            buffer.append(token)
        else:
            if len(buffer) >= 2:
                acronyms.append("".join(buffer))
            buffer = []
    if len(buffer) >= 2:
        acronyms.append("".join(buffer))
    tokens.extend(acronyms)

    output: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if not token:
            continue
        if token not in seen:
            output.append(token)
            seen.add(token)
        singular = None
        if token.isalpha() and len(token) >= 5:
            if token.endswith("AUX") and len(token) >= 6:
                singular = token[:-3] + "AL"
            elif token.endswith("S") and not token.endswith("SS"):
                singular = token[:-1]
            elif token.endswith("X"):
                singular = token[:-1]
        if singular and singular not in seen:
            output.append(singular)
            seen.add(singular)
    return " ".join(output)


def _build_aligned_pool(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    filtered: list[tuple[str, dict[str, Any]]] = []
    for raw in rows:
        row = dict(raw)
        if not any(row.get(column) for column in _NAME_FILTER_COLUMNS):
            continue
        if row.get("etat_admin") == "F":
            continue
        siret = str(row.get("siret") or "").strip()
        if not siret:
            continue
        filtered.append((siret, row))
    deduplicated: dict[str, dict[str, Any]] = {}
    for siret, row in filtered:
        deduplicated[siret] = row
    return list(deduplicated.values())


def _expected_tfidf_names(
    aligned_pool: Sequence[Mapping[str, Any]],
) -> list[str]:
    return [
        _normalize_text_for_tfidf(_candidate_tfidf_text(row) or "")
        for row in aligned_pool
    ]


def _schema_signature(
    fields: Sequence[tuple[str, pa.DataType]],
) -> tuple[tuple[str, str], ...]:
    return tuple((name, str(data_type)) for name, data_type in fields)


def _partition_key_from_relative_path(
    relative_path: str,
) -> tuple[str, str]:
    match = _INSEE_RELATIVE_RE.fullmatch(relative_path)
    if match is not None:
        return f"{match.group(1)}_", "insee"
    match = _POSTCODE_RELATIVE_RE.fullmatch(relative_path)
    if match is not None:
        return f"_{match.group(1)}", "postcode"
    _fail(
        StrictPartitionError,
        STOP_PARTITION,
        "non-canonical partition relative path",
    )


def _validate_allowed_file_record(
    raw: Mapping[str, Any],
    *,
    expected_role: str | None = None,
    error_type: type[StrictStoreError] = StrictProbeError,
    code: str = STOP_PROBE,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        _fail(error_type, code, "allowed file record must be an object")
    _require_exact_keys(
        raw,
        _ALLOWED_FILE_KEYS,
        error_type=error_type,
        code=code,
        label="allowed file record",
    )
    role = raw["role"]
    allowed_roles = {
        "partition",
        "cache_pickle",
        "cache_sidecar",
        "lookup_database",
    }
    if (
        type(role) is not str
        or role not in allowed_roles
        or (expected_role is not None and role != expected_role)
    ):
        _fail(error_type, code, "allowed file role mismatch")
    partition_key = raw["partition_key"]
    if type(partition_key) is not str:
        _fail(error_type, code, "allowed file partition_key must be a string")
    if role == "lookup_database":
        if partition_key != "":
            _fail(error_type, code, "lookup partition_key must be empty")
    elif _PARTITION_KEY_RE.fullmatch(partition_key) is None:
        _fail(error_type, code, "allowed file partition_key is invalid")
    absolute_path = raw["absolute_path"]
    if type(absolute_path) is not str:
        _fail(error_type, code, "allowed file path must be a string")
    path = Path(absolute_path)
    size = _require_plain_int(
        raw["size_bytes"],
        error_type=error_type,
        code=code,
        label="allowed file size",
    )
    sha256 = _require_sha256(
        raw["sha256"],
        error_type=error_type,
        code=code,
        label="allowed file hash",
    )
    return {
        "role": role,
        "partition_key": partition_key,
        "absolute_path": path,
        "size_bytes": size,
        "sha256": sha256,
    }


class StrictPartitionStore:
    """Read only the exact partition files sealed into a run specification."""

    def __init__(
        self,
        partition_records: Sequence[Mapping[str, Any]],
        allowed_read_files: Sequence[Mapping[str, Any]],
        *,
        max_cache_entries: int = 5,
    ) -> None:
        if type(max_cache_entries) is not int or not 1 <= max_cache_entries <= 5:
            _fail(
                StrictPartitionError,
                STOP_PARTITION,
                "partition cache limit must be between one and five",
            )
        allowed: list[dict[str, Any]] = []
        for item in allowed_read_files:
            if not isinstance(item, Mapping):
                _fail(
                    StrictPartitionError,
                    STOP_PARTITION,
                    "allowed file record must be an object",
                )
            if item.get("role") != "partition":
                continue
            allowed.append(
                _validate_allowed_file_record(
                    item,
                    expected_role="partition",
                    error_type=StrictPartitionError,
                    code=STOP_PARTITION,
                )
            )
        allowed_by_key: dict[str, dict[str, Any]] = {}
        for item in allowed:
            key = _validate_partition_key(
                item["partition_key"],
                error_type=StrictPartitionError,
                code=STOP_PARTITION,
            )
            if key in allowed_by_key:
                _fail(
                    StrictPartitionError,
                    STOP_PARTITION,
                    "duplicate allowed partition key",
                )
            allowed_by_key[key] = item

        records_by_key: dict[str, dict[str, Any]] = {}
        for raw in partition_records:
            if not isinstance(raw, Mapping):
                _fail(
                    StrictPartitionError,
                    STOP_PARTITION,
                    "partition record must be an object",
                )
            _require_exact_keys(
                raw,
                _PARTITION_RECORD_KEYS,
                error_type=StrictPartitionError,
                code=STOP_PARTITION,
                label="partition record",
            )
            relative = raw["relative_path"]
            if type(relative) is not str:
                _fail(
                    StrictPartitionError,
                    STOP_PARTITION,
                    "partition relative path must be a string",
                )
            key, kind = _partition_key_from_relative_path(relative)
            if key in records_by_key:
                _fail(
                    StrictPartitionError,
                    STOP_PARTITION,
                    "duplicate partition record key",
                )
            size = _require_plain_int(
                raw["size_bytes"],
                error_type=StrictPartitionError,
                code=STOP_PARTITION,
                label="partition size",
            )
            sha256 = _require_sha256(
                raw["sha256"],
                error_type=StrictPartitionError,
                code=STOP_PARTITION,
                label="partition hash",
            )
            allowed_item = allowed_by_key.get(key)
            if allowed_item is None:
                _fail(
                    StrictPartitionError,
                    STOP_PARTITION,
                    "partition absent from allowed files",
                )
            if (
                allowed_item["size_bytes"] != size
                or allowed_item["sha256"] != sha256
                or not str(allowed_item["absolute_path"]).endswith(
                    f"/{relative}"
                )
            ):
                _fail(
                    StrictPartitionError,
                    STOP_PARTITION,
                    "partition record and allowed file disagree",
                )
            records_by_key[key] = {
                "partition_key": key,
                "kind": kind,
                "relative_path": relative,
                "absolute_path": allowed_item["absolute_path"],
                "size_bytes": size,
                "sha256": sha256,
            }
        if set(records_by_key) != set(allowed_by_key):
            _fail(
                StrictPartitionError,
                STOP_PARTITION,
                "partition records and allow-list are not bijective",
            )
        self._records = records_by_key
        self._max_cache_entries = max_cache_entries
        self._cache: OrderedDict[str, tuple[dict[str, Any], ...]] = OrderedDict()

    @property
    def partition_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._records, key=lambda value: value.encode("utf-8")))

    def _read_partition(self, key: str) -> tuple[dict[str, Any], ...]:
        record = self._records[key]
        try:
            with _verified_file(
                record["absolute_path"],
                expected_size=record["size_bytes"],
                expected_sha256=record["sha256"],
                error_type=StrictPartitionError,
                code=STOP_PARTITION,
            ) as handle:
                parquet_file = pq.ParquetFile(handle)
                schema = parquet_file.schema_arrow
                expected_fields = (
                    _INSEE_FIELDS
                    if record["kind"] == "insee"
                    else _POSTCODE_FIELDS
                )
                if tuple(
                    (field.name, str(field.type)) for field in schema
                ) != _schema_signature(expected_fields):
                    _fail(
                        StrictPartitionError,
                        STOP_PARTITION,
                        "partition schema or column order mismatch",
                    )
                table = parquet_file.read()
        except StrictStoreError:
            raise
        except Exception as exc:
            _fail(
                StrictPartitionError,
                STOP_PARTITION,
                f"Parquet read failed: {exc}",
            )
        rows = table.to_pylist()
        injected_name = "insee" if record["kind"] == "insee" else "postcode"
        injected_value = key[:-1] if record["kind"] == "insee" else key[1:]
        segment = (
            _INSEE_RELATIVE_RE.fullmatch(record["relative_path"])
            if record["kind"] == "insee"
            else _POSTCODE_RELATIVE_RE.fullmatch(record["relative_path"])
        )
        if segment is None or segment.group(1) != injected_value:
            _fail(
                StrictPartitionError,
                STOP_PARTITION,
                "partition key does not match its Hive segment",
            )
        for row in rows:
            if injected_name in row:
                _fail(
                    StrictPartitionError,
                    STOP_PARTITION,
                    "Hive field unexpectedly present in physical rows",
                )
            row[injected_name] = injected_value
        return tuple(rows)

    def load(self, partition_key: str) -> list[dict[str, Any]]:
        key = _validate_partition_key(
            partition_key,
            error_type=StrictPartitionError,
            code=STOP_PARTITION,
        )
        if key not in self._records:
            _fail(
                StrictPartitionError,
                STOP_PARTITION,
                "partition key is not in the frozen subset",
            )
        if key in self._cache:
            self._cache.move_to_end(key)
            cached = self._cache[key]
        else:
            cached = self._read_partition(key)
            self._cache[key] = cached
            while len(self._cache) > self._max_cache_entries:
                self._cache.popitem(last=False)
        return [dict(row) for row in cached]

    def load_with_status(
        self,
        partition_key: str,
    ) -> tuple[str, list[dict[str, Any]]]:
        rows = self.load(partition_key)
        return ("VALID_EMPTY" if not rows else "VALID_ROWS", rows)

    def release(self, partition_key: str) -> None:
        self._cache.pop(partition_key, None)


def _matrix_buffer_bytes(matrix: Any) -> int:
    if matrix is None:
        return 0
    total = 0
    for name in ("data", "indices", "indptr", "row", "col"):
        value = getattr(matrix, name, None)
        total += int(getattr(value, "nbytes", 0))
    return total


def _estimate_tfidf_bytes(artifacts: tuple[Any, ...]) -> int:
    name_vec, name_mat, names, char_vec, char_mat, addr_vec, addr_mat = artifacts
    total = sum(
        _matrix_buffer_bytes(matrix)
        for matrix in (name_mat, char_mat, addr_mat)
    )
    total += sum(len(value.encode("utf-8")) for value in names)
    for vectorizer in (name_vec, char_vec, addr_vec):
        if vectorizer is None:
            continue
        total += sum(
            len(str(key).encode("utf-8")) + 8
            for key in getattr(vectorizer, "vocabulary_", {})
        )
        idf = getattr(vectorizer, "idf_", None)
        total += int(getattr(idf, "nbytes", 0))
    return total


class StrictVerifiedTfidfCache:
    """Verified pickle/sidecar cache with no write or rebuild API."""

    def __init__(
        self,
        cache_records: Sequence[Mapping[str, Any]],
        allowed_read_files: Sequence[Mapping[str, Any]],
        *,
        namespace: str,
        sidecar_schema_version: str = SIDECAR_SCHEMA,
        max_cache_entries: int = 20,
        max_cache_bytes: int = 1 << 30,
    ) -> None:
        if (
            type(namespace) is not str
            or _SHA256_RE.fullmatch(namespace) is None
        ):
            _fail(StrictTfidfError, STOP_TFIDF, "invalid cache namespace")
        if sidecar_schema_version != SIDECAR_SCHEMA:
            _fail(StrictTfidfError, STOP_TFIDF, "invalid sidecar schema")
        if type(max_cache_entries) is not int or not 1 <= max_cache_entries <= 20:
            _fail(StrictTfidfError, STOP_TFIDF, "invalid cache entry limit")
        if type(max_cache_bytes) is not int or not 1 <= max_cache_bytes <= 1 << 30:
            _fail(StrictTfidfError, STOP_TFIDF, "invalid cache byte limit")

        allowed_by_role_key: dict[tuple[str, str], dict[str, Any]] = {}
        for raw in allowed_read_files:
            if not isinstance(raw, Mapping):
                _fail(
                    StrictTfidfError,
                    STOP_TFIDF,
                    "allowed file record must be an object",
                )
            if raw.get("role") not in {"cache_pickle", "cache_sidecar"}:
                continue
            item = _validate_allowed_file_record(
                raw,
                error_type=StrictTfidfError,
                code=STOP_TFIDF,
            )
            key = _validate_partition_key(
                item["partition_key"],
                error_type=StrictTfidfError,
                code=STOP_TFIDF,
            )
            role_key = (item["role"], key)
            if role_key in allowed_by_role_key:
                _fail(StrictTfidfError, STOP_TFIDF, "duplicate cache allow record")
            allowed_by_role_key[role_key] = item

        records: dict[str, dict[str, Any]] = {}
        seen_paths: set[str] = set()
        for raw in cache_records:
            if not isinstance(raw, Mapping):
                _fail(
                    StrictTfidfError,
                    STOP_TFIDF,
                    "cache record must be an object",
                )
            _require_exact_keys(
                raw,
                _CACHE_RECORD_KEYS,
                error_type=StrictTfidfError,
                code=STOP_TFIDF,
                label="cache record",
            )
            key = _validate_partition_key(
                raw["partition_key"],
                error_type=StrictTfidfError,
                code=STOP_TFIDF,
            )
            if key in records:
                _fail(StrictTfidfError, STOP_TFIDF, "duplicate cache key")
            safe = _safe_key(key)
            if safe != key:
                _fail(StrictTfidfError, STOP_TFIDF, "safe_key collision")
            pickle_relative = raw["pickle_relative_path"]
            sidecar_relative = raw["sidecar_relative_path"]
            if (
                pickle_relative != f"{safe}.pkl"
                or sidecar_relative != f"{safe}.pkl.sha256.json"
            ):
                _fail(StrictTfidfError, STOP_TFIDF, "non-canonical cache path")
            if pickle_relative in seen_paths or sidecar_relative in seen_paths:
                _fail(StrictTfidfError, STOP_TFIDF, "cache path collision")
            seen_paths.update((pickle_relative, sidecar_relative))
            pickle_size = _require_plain_int(
                raw["pickle_size_bytes"],
                error_type=StrictTfidfError,
                code=STOP_TFIDF,
                label="pickle size",
            )
            sidecar_size = _require_plain_int(
                raw["sidecar_size_bytes"],
                error_type=StrictTfidfError,
                code=STOP_TFIDF,
                label="sidecar size",
            )
            pickle_sha = _require_sha256(
                raw["pickle_sha256"],
                error_type=StrictTfidfError,
                code=STOP_TFIDF,
                label="pickle hash",
            )
            sidecar_sha = _require_sha256(
                raw["sidecar_sha256"],
                error_type=StrictTfidfError,
                code=STOP_TFIDF,
                label="sidecar hash",
            )
            pickle_allowed = allowed_by_role_key.get(("cache_pickle", key))
            sidecar_allowed = allowed_by_role_key.get(("cache_sidecar", key))
            if pickle_allowed is None or sidecar_allowed is None:
                _fail(StrictTfidfError, STOP_TFIDF, "cache file not allowed")
            if (
                pickle_allowed["size_bytes"] != pickle_size
                or pickle_allowed["sha256"] != pickle_sha
                or sidecar_allowed["size_bytes"] != sidecar_size
                or sidecar_allowed["sha256"] != sidecar_sha
                or pickle_allowed["absolute_path"].name != pickle_relative
                or sidecar_allowed["absolute_path"].name != sidecar_relative
            ):
                _fail(StrictTfidfError, STOP_TFIDF, "cache records disagree")
            records[key] = {
                "partition_key": key,
                "pickle_path": pickle_allowed["absolute_path"],
                "pickle_size": pickle_size,
                "pickle_sha": pickle_sha,
                "sidecar_path": sidecar_allowed["absolute_path"],
                "sidecar_size": sidecar_size,
                "sidecar_sha": sidecar_sha,
            }
        expected_role_keys = {
            (role, key)
            for key in records
            for role in ("cache_pickle", "cache_sidecar")
        }
        if set(allowed_by_role_key) != expected_role_keys:
            _fail(StrictTfidfError, STOP_TFIDF, "cache allow-list is not bijective")
        self._records = records
        self._namespace = namespace
        self._sidecar_schema_version = sidecar_schema_version
        self._max_cache_entries = max_cache_entries
        self._max_cache_bytes = max_cache_bytes
        self._cache: OrderedDict[str, tuple[tuple[Any, ...], int]] = OrderedDict()
        self._cached_bytes = 0

    @property
    def partition_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._records, key=lambda value: value.encode("utf-8")))

    def _validate_artifacts(
        self,
        artifacts: Any,
        aligned_pool: Sequence[Mapping[str, Any]],
    ) -> tuple[Any, ...]:
        if type(artifacts) is not tuple or len(artifacts) != 7:
            _fail(StrictTfidfError, STOP_TFIDF, "pickle is not a tuple of seven")
        (
            name_vec,
            name_mat,
            names,
            char_vec,
            char_mat,
            addr_vec,
            addr_mat,
        ) = artifacts
        if type(names) is not list or any(type(value) is not str for value in names):
            _fail(StrictTfidfError, STOP_TFIDF, "names must be list[str]")
        expected_rows = len(aligned_pool)
        if len(names) != expected_rows:
            _fail(StrictTfidfError, STOP_TFIDF, "names row count mismatch")
        for label, vectorizer, matrix in (
            ("name", name_vec, name_mat),
            ("char", char_vec, char_mat),
            ("address", addr_vec, addr_mat),
        ):
            if (vectorizer is None) != (matrix is None):
                _fail(StrictTfidfError, STOP_TFIDF, f"{label} pair mismatch")
            if vectorizer is None:
                continue
            if type(vectorizer) is not TfidfVectorizer:
                _fail(StrictTfidfError, STOP_TFIDF, f"{label} vectorizer type mismatch")
            if not sparse.issparse(matrix) or len(matrix.shape) != 2:
                _fail(StrictTfidfError, STOP_TFIDF, f"{label} matrix type mismatch")
            if int(matrix.shape[0]) != expected_rows:
                _fail(StrictTfidfError, STOP_TFIDF, f"{label} matrix row mismatch")
            vocabulary = getattr(vectorizer, "vocabulary_", None)
            if type(vocabulary) is not dict:
                _fail(StrictTfidfError, STOP_TFIDF, f"{label} vocabulary mismatch")
            if int(matrix.shape[1]) != len(vocabulary):
                _fail(StrictTfidfError, STOP_TFIDF, f"{label} matrix column mismatch")
            vocabulary_indices = list(vocabulary.values())
            if (
                any(type(index) is not int for index in vocabulary_indices)
                or set(vocabulary_indices) != set(range(int(matrix.shape[1])))
            ):
                _fail(
                    StrictTfidfError,
                    STOP_TFIDF,
                    f"{label} vocabulary indices mismatch",
                )
            idf = getattr(vectorizer, "idf_", None)
            if (
                idf is None
                or getattr(idf, "ndim", None) != 1
                or int(getattr(idf, "shape", (0,))[0]) != int(matrix.shape[1])
                or not all(math.isfinite(float(value)) for value in idf)
            ):
                _fail(StrictTfidfError, STOP_TFIDF, f"{label} IDF mismatch")
            matrix_indices = getattr(matrix, "indices", None)
            if matrix_indices is not None and int(getattr(matrix_indices, "size", 0)):
                if (
                    int(matrix_indices.min()) < 0
                    or int(matrix_indices.max()) >= int(matrix.shape[1])
                ):
                    _fail(
                        StrictTfidfError,
                        STOP_TFIDF,
                        f"{label} sparse index out of bounds",
                    )
            matrix_columns = getattr(matrix, "col", None)
            if matrix_columns is not None and int(getattr(matrix_columns, "size", 0)):
                if (
                    int(matrix_columns.min()) < 0
                    or int(matrix_columns.max()) >= int(matrix.shape[1])
                ):
                    _fail(
                        StrictTfidfError,
                        STOP_TFIDF,
                        f"{label} sparse column out of bounds",
                    )
        if names != _expected_tfidf_names(aligned_pool):
            _fail(StrictTfidfError, STOP_TFIDF, "TF-IDF names parity mismatch")
        return artifacts

    def _load(
        self,
        key: str,
        aligned_pool: Sequence[Mapping[str, Any]],
    ) -> tuple[Any, ...]:
        record = self._records[key]
        with _verified_file(
            record["sidecar_path"],
            expected_size=record["sidecar_size"],
            expected_sha256=record["sidecar_sha"],
            error_type=StrictTfidfError,
            code=STOP_TFIDF,
        ) as handle:
            sidecar = _load_json_bytes_strict(
                handle.read(),
                error_type=StrictTfidfError,
                code=STOP_TFIDF,
                label="TF-IDF sidecar",
            )
        _require_exact_keys(
            sidecar,
            {"schema_version", "config_hash", "partition_key", "size_bytes", "sha256"},
            error_type=StrictTfidfError,
            code=STOP_TFIDF,
            label="TF-IDF sidecar",
        )
        if (
            type(sidecar["schema_version"]) is not str
            or type(sidecar["config_hash"]) is not str
            or type(sidecar["partition_key"]) is not str
            or type(sidecar["size_bytes"]) is not int
            or type(sidecar["sha256"]) is not str
            or sidecar["schema_version"] != self._sidecar_schema_version
            or sidecar["config_hash"] != self._namespace
            or sidecar["partition_key"] != key
            or sidecar["size_bytes"] != record["pickle_size"]
            or sidecar["sha256"] != record["pickle_sha"]
        ):
            _fail(StrictTfidfError, STOP_TFIDF, "TF-IDF sidecar seal mismatch")
        try:
            with _verified_file(
                record["pickle_path"],
                expected_size=record["pickle_size"],
                expected_sha256=record["pickle_sha"],
                error_type=StrictTfidfError,
                code=STOP_TFIDF,
            ) as handle:
                artifacts = pickle.load(handle)
        except StrictStoreError:
            raise
        except Exception as exc:
            _fail(StrictTfidfError, STOP_TFIDF, f"pickle load failed: {exc}")
        return self._validate_artifacts(artifacts, aligned_pool)

    def get(
        self,
        partition_key: str,
        aligned_pool: Sequence[Mapping[str, Any]],
    ) -> tuple[Any, ...]:
        key = _validate_partition_key(
            partition_key,
            error_type=StrictTfidfError,
            code=STOP_TFIDF,
        )
        if key not in self._records:
            _fail(StrictTfidfError, STOP_TFIDF, "cache miss")
        if key in self._cache:
            self._cache.move_to_end(key)
            artifacts = self._cache[key][0]
            return self._validate_artifacts(artifacts, aligned_pool)
        artifacts = self._load(key, aligned_pool)
        estimated = _estimate_tfidf_bytes(artifacts)
        if estimated <= self._max_cache_bytes:
            while self._cache and (
                len(self._cache) >= self._max_cache_entries
                or self._cached_bytes + estimated > self._max_cache_bytes
            ):
                _, (_, removed_size) = self._cache.popitem(last=False)
                self._cached_bytes -= removed_size
            self._cache[key] = (artifacts, estimated)
            self._cached_bytes += estimated
        return artifacts

    def release(self, partition_key: str) -> None:
        cached = self._cache.pop(partition_key, None)
        if cached is not None:
            self._cached_bytes -= cached[1]


def _validate_requested_sirets(
    sirets: Sequence[str],
    *,
    maximum: int,
) -> list[str]:
    if isinstance(sirets, (str, bytes, bytearray)) or not isinstance(
        sirets,
        Sequence,
    ):
        _fail(StrictLookupError, STOP_LOOKUP, "expected a sequence of strings")
    if len(sirets) > maximum:
        _fail(StrictLookupError, STOP_LOOKUP, "too many SIRET requested")
    unique: list[str] = []
    seen: set[str] = set()
    for value in sirets:
        if type(value) is not str or _SIRET_RE.fullmatch(value) is None:
            _fail(StrictLookupError, STOP_LOOKUP, "invalid canonical SIRET")
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


class StrictSnapshotLookup:
    """Bounded, read-only access to the frozen lookup DuckDB."""

    def __init__(
        self,
        descriptor: Mapping[str, Any],
        allowed_read_files: Sequence[Mapping[str, Any]],
    ) -> None:
        if not isinstance(descriptor, Mapping):
            _fail(StrictLookupError, STOP_LOOKUP, "lookup descriptor must be an object")
        _require_exact_keys(
            descriptor,
            _LOOKUP_DESCRIPTOR_KEYS,
            error_type=StrictLookupError,
            code=STOP_LOOKUP,
            label="lookup descriptor",
        )
        if descriptor["schema_version"] != LOOKUP_DESCRIPTOR_SCHEMA:
            _fail(StrictLookupError, STOP_LOOKUP, "lookup descriptor version mismatch")
        expected_size = _require_plain_int(
            descriptor["database_size_bytes"],
            error_type=StrictLookupError,
            code=STOP_LOOKUP,
            label="lookup database size",
        )
        expected_sha = _require_sha256(
            descriptor["database_sha256"],
            error_type=StrictLookupError,
            code=STOP_LOOKUP,
            label="lookup database hash",
        )
        if (
            descriptor["table_name"] != _LOOKUP_TABLE
            or descriptor["columns"] != _LOOKUP_COLUMNS
            or descriptor["column_types"] != ["VARCHAR"] * len(_LOOKUP_COLUMNS)
            or descriptor["index_name"] != _LOOKUP_INDEX
            or descriptor["index_unique"] is not True
            or descriptor["read_only"] is not True
        ):
            _fail(StrictLookupError, STOP_LOOKUP, "lookup descriptor values changed")
        row_count = _require_plain_int(
            descriptor["row_count"],
            error_type=StrictLookupError,
            code=STOP_LOOKUP,
            label="lookup row count",
        )
        maximum = _require_plain_int(
            descriptor["max_sirets_per_call"],
            error_type=StrictLookupError,
            code=STOP_LOOKUP,
            label="lookup maximum",
            minimum=1,
        )
        if maximum != 100:
            _fail(StrictLookupError, STOP_LOOKUP, "lookup maximum must be 100")
        allowed: list[dict[str, Any]] = []
        for item in allowed_read_files:
            if not isinstance(item, Mapping):
                _fail(
                    StrictLookupError,
                    STOP_LOOKUP,
                    "allowed file record must be an object",
                )
            if item.get("role") != "lookup_database":
                continue
            allowed.append(
                _validate_allowed_file_record(
                    item,
                    expected_role="lookup_database",
                    error_type=StrictLookupError,
                    code=STOP_LOOKUP,
                )
            )
        if len(allowed) != 1:
            _fail(StrictLookupError, STOP_LOOKUP, "expected one lookup database")
        item = allowed[0]
        if (
            item["partition_key"] != ""
            or item["size_bytes"] != expected_size
            or item["sha256"] != expected_sha
        ):
            _fail(StrictLookupError, STOP_LOOKUP, "lookup allow record mismatch")
        self._path = item["absolute_path"]
        self._expected_size = expected_size
        self._expected_sha = expected_sha
        self._expected_rows = row_count
        self._maximum = maximum
        self._closed = True
        self.__database_fd = -1
        self._assert_auxiliary_absent()
        chain_before = _snapshot_path_chain(
            self._path,
            error_type=StrictLookupError,
            code=STOP_LOOKUP,
        )
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            self.__database_fd = os.open(self._path, flags)
        except Exception as exc:
            _fail(StrictLookupError, STOP_LOOKUP, f"lookup FD open failed: {exc}")
        try:
            self._database_identity = _file_identity(
                os.fstat(self.__database_fd)
            )
            if (
                not stat.S_ISREG(self._database_identity[2])
                or self._database_identity[3] != self._expected_size
            ):
                _fail(StrictLookupError, STOP_LOOKUP, "lookup FD identity mismatch")
            self._verify_open_database_hash()
            if (
                _snapshot_path_chain(
                    self._path,
                    error_type=StrictLookupError,
                    code=STOP_LOOKUP,
                )
                != chain_before
            ):
                _fail(StrictLookupError, STOP_LOOKUP, "lookup path changed before connect")
            # Opening /dev/fd/N makes DuckDB consume the exact O_NOFOLLOW file
            # descriptor that was hashed above. It cannot race by reopening the
            # original pathname on another inode.
            duckdb_path = f"/dev/fd/{self.__database_fd}"
            try:
                self.__connection = duckdb.connect(
                    duckdb_path,
                    read_only=True,
                    config={
                        "enable_external_access": "false",
                        "autoinstall_known_extensions": "false",
                        "autoload_known_extensions": "false",
                    },
                )
            except Exception as exc:
                _fail(StrictLookupError, STOP_LOOKUP, f"lookup open failed: {exc}")
            self._inspect()
            self._path_chain_identity = _snapshot_path_chain(
                self._path,
                error_type=StrictLookupError,
                code=STOP_LOOKUP,
            )
            if self._path_chain_identity != chain_before:
                _fail(StrictLookupError, STOP_LOOKUP, "lookup path changed during connect")
            self._assert_auxiliary_absent()
            self._closed = False
        except BaseException:
            if hasattr(self, "_StrictSnapshotLookup__connection"):
                self.__connection.close()
            if self.__database_fd >= 0:
                os.close(self.__database_fd)
                self.__database_fd = -1
            raise

    def _assert_auxiliary_absent(self) -> None:
        for suffix in (".wal", ".tmp"):
            auxiliary = self._path.with_suffix(self._path.suffix + suffix)
            try:
                os.lstat(auxiliary)
            except FileNotFoundError:
                continue
            except Exception as exc:
                _fail(
                    StrictLookupError,
                    STOP_LOOKUP,
                    f"cannot inspect lookup auxiliary file: {exc}",
                )
            _fail(
                StrictLookupError,
                STOP_LOOKUP,
                f"lookup auxiliary file exists: {suffix}",
            )

    def _verify_open_database_hash(self) -> None:
        before = os.fstat(self.__database_fd)
        if _file_identity(before) != self._database_identity:
            _fail(StrictLookupError, STOP_LOOKUP, "lookup FD identity changed")
        try:
            os.lseek(self.__database_fd, 0, os.SEEK_SET)
            digest = hashlib.sha256()
            while block := os.read(self.__database_fd, 8 * 1024 * 1024):
                digest.update(block)
            os.lseek(self.__database_fd, 0, os.SEEK_SET)
        except Exception as exc:
            _fail(StrictLookupError, STOP_LOOKUP, f"lookup FD hash failed: {exc}")
        after = os.fstat(self.__database_fd)
        if (
            _file_identity(after) != self._database_identity
            or digest.hexdigest() != self._expected_sha
        ):
            _fail(StrictLookupError, STOP_LOOKUP, "lookup FD reseal mismatch")

    def _assert_unchanged(self) -> None:
        if (
            self.__database_fd < 0
            or _file_identity(os.fstat(self.__database_fd))
            != self._database_identity
        ):
            _fail(StrictLookupError, STOP_LOOKUP, "lookup FD changed")
        current = _snapshot_path_chain(
            self._path,
            error_type=StrictLookupError,
            code=STOP_LOOKUP,
        )
        if current != self._path_chain_identity:
            _fail(StrictLookupError, STOP_LOOKUP, "lookup path changed")
        self._assert_auxiliary_absent()

    def _inspect(self) -> None:
        tables = self.__connection.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main' AND table_type = 'BASE TABLE'
            ORDER BY table_name
            """
        ).fetchall()
        if tables != [(_LOOKUP_TABLE,)]:
            _fail(StrictLookupError, STOP_LOOKUP, "lookup table set changed")
        columns = self.__connection.execute(
            """
            SELECT column_name, data_type, ordinal_position
            FROM information_schema.columns
            WHERE table_schema = 'main' AND table_name = ?
            ORDER BY ordinal_position
            """,
            [_LOOKUP_TABLE],
        ).fetchall()
        if [str(row[0]) for row in columns] != _LOOKUP_COLUMNS:
            _fail(StrictLookupError, STOP_LOOKUP, "lookup column order changed")
        if any(str(row[1]).upper() != "VARCHAR" for row in columns):
            _fail(StrictLookupError, STOP_LOOKUP, "lookup column type changed")
        indexes = self.__connection.execute(
            """
            SELECT index_name, is_unique, expressions
            FROM duckdb_indexes()
            WHERE schema_name = 'main' AND table_name = ?
            ORDER BY index_name
            """,
            [_LOOKUP_TABLE],
        ).fetchall()
        if len(indexes) != 1:
            _fail(StrictLookupError, STOP_LOOKUP, "lookup index set changed")
        index_name, unique, expressions = indexes[0]
        if (
            str(index_name) != _LOOKUP_INDEX
            or unique is not True
            or "[siret]" not in str(expressions).replace('"', "")
        ):
            _fail(StrictLookupError, STOP_LOOKUP, "lookup unique index changed")
        count = self.__connection.execute(
            f"SELECT COUNT(*) FROM {_LOOKUP_TABLE}"
        ).fetchone()
        if count is None or int(count[0]) != self._expected_rows:
            _fail(StrictLookupError, STOP_LOOKUP, "lookup row count changed")

    def get_candidate_scene_details(
        self,
        sirets: Sequence[str],
    ) -> dict[str, dict[str, str | None]]:
        if self._closed:
            _fail(StrictLookupError, STOP_LOOKUP, "lookup is closed")
        requested = _validate_requested_sirets(sirets, maximum=self._maximum)
        if not requested:
            return {}
        self._assert_unchanged()
        try:
            rows = self.__connection.execute(
                f"""
                SELECT {", ".join(_LOOKUP_COLUMNS)}
                FROM {_LOOKUP_TABLE}
                WHERE siret IN (SELECT unnest(?))
                ORDER BY siret
                """,
                [requested],
            ).fetchall()
        except Exception as exc:
            _fail(StrictLookupError, STOP_LOOKUP, f"lookup query failed: {exc}")
        self._assert_unchanged()
        requested_set = set(requested)
        result: dict[str, dict[str, str | None]] = {}
        for row in rows:
            siret = row[0]
            if type(siret) is not str or siret not in requested_set:
                _fail(StrictLookupError, STOP_LOOKUP, "lookup returned extra SIRET")
            if siret in result:
                _fail(StrictLookupError, STOP_LOOKUP, "lookup returned duplicate SIRET")
            values = row[1:]
            if any(type(value) is not str and value is not None for value in values):
                _fail(StrictLookupError, STOP_LOOKUP, "lookup value type changed")
            result[siret] = dict(zip(_LOOKUP_COLUMNS[1:], values, strict=True))
        return result

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            failure: BaseException | None = None
            try:
                self.__connection.close()
                self._verify_open_database_hash()
                self._assert_unchanged()
            except BaseException as exc:
                failure = exc
            finally:
                if self.__database_fd >= 0:
                    os.close(self.__database_fd)
                    self.__database_fd = -1
            if failure is not None:
                raise failure

    def __enter__(self) -> "StrictSnapshotLookup":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _validate_declarations(value: Any) -> dict[str, bool]:
    if type(value) is not dict or value != _DECLARATIONS:
        _fail(StrictProbeError, STOP_PROBE, "declarations changed")
    return dict(_DECLARATIONS)


def _validate_run_spec(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(StrictProbeError, STOP_PROBE, "run spec must be an object")
    _require_exact_keys(
        value,
        _RUN_SPEC_KEYS,
        error_type=StrictProbeError,
        code=STOP_PROBE,
        label="run spec",
    )
    if value["schema_version"] != RUN_SPEC_SCHEMA:
        _fail(StrictProbeError, STOP_PROBE, "run spec version mismatch")
    if (
        type(value["safe_input_build_id"]) is not str
        or _SHA256_RE.fullmatch(value["safe_input_build_id"]) is None
    ):
        _fail(StrictProbeError, STOP_PROBE, "invalid safe input build ID")
    _require_plain_int(
        value["query_count"],
        error_type=StrictProbeError,
        code=STOP_PROBE,
        label="query count",
    )
    _require_sha256(
        value["routing_payload_sha256"],
        error_type=StrictProbeError,
        code=STOP_PROBE,
        label="routing payload hash",
    )
    _require_sha256(
        value["lookup_descriptor_sha256"],
        error_type=StrictProbeError,
        code=STOP_PROBE,
        label="lookup descriptor hash",
    )
    if value["staging_dir"] != "output" or value["tmp_dir"] != "tmp":
        _fail(StrictProbeError, STOP_PROBE, "non-canonical work directories")
    _require_plain_int(
        value["max_rss_bytes"],
        error_type=StrictProbeError,
        code=STOP_PROBE,
        label="RSS limit",
        minimum=1,
    )
    if type(value["partition_records"]) is not list:
        _fail(StrictProbeError, STOP_PROBE, "partition_records must be a list")
    if type(value["cache_records"]) is not list:
        _fail(StrictProbeError, STOP_PROBE, "cache_records must be a list")
    if type(value["allowed_read_files"]) is not list:
        _fail(StrictProbeError, STOP_PROBE, "allowed_read_files must be a list")
    _validate_declarations(value["declarations"])
    if any(
        not isinstance(item, Mapping)
        for item in value["partition_records"] + value["cache_records"]
    ):
        _fail(StrictProbeError, STOP_PROBE, "data records must be objects")
    for item in value["partition_records"]:
        _require_exact_keys(
            item,
            _PARTITION_RECORD_KEYS,
            error_type=StrictProbeError,
            code=STOP_PROBE,
            label="partition record",
        )
        if type(item["relative_path"]) is not str:
            _fail(StrictProbeError, STOP_PROBE, "invalid partition relative path")
    for item in value["cache_records"]:
        _require_exact_keys(
            item,
            _CACHE_RECORD_KEYS,
            error_type=StrictProbeError,
            code=STOP_PROBE,
            label="cache record",
        )
        if any(
            type(item[field]) is not str
            for field in (
                "partition_key",
                "pickle_relative_path",
                "sidecar_relative_path",
            )
        ):
            _fail(StrictProbeError, STOP_PROBE, "invalid cache record path")
    if value["partition_records"] != sorted(
        value["partition_records"],
        key=lambda item: item["relative_path"].encode("utf-8"),
    ):
        _fail(StrictProbeError, STOP_PROBE, "partition records are not sorted")
    if value["cache_records"] != sorted(
        value["cache_records"],
        key=lambda item: (
            item["partition_key"].encode("utf-8"),
            item["pickle_relative_path"].encode("utf-8"),
            item["sidecar_relative_path"].encode("utf-8"),
        ),
    ):
        _fail(StrictProbeError, STOP_PROBE, "cache records are not sorted")

    allowed = [
        _validate_allowed_file_record(item)
        for item in value["allowed_read_files"]
    ]
    allowed_order = sorted(
        allowed,
        key=lambda item: (
            item["role"].encode("utf-8"),
            str(item["absolute_path"]).encode("utf-8"),
        ),
    )
    if allowed != allowed_order:
        _fail(StrictProbeError, STOP_PROBE, "allowed files are not canonically sorted")
    if len(allowed) != 1_945:
        _fail(StrictProbeError, STOP_PROBE, "allowed file count changed")
    paths = [str(item["absolute_path"]) for item in allowed]
    if len(set(paths)) != len(paths):
        _fail(StrictProbeError, STOP_PROBE, "allowed file path is not unique")
    role_counts = {
        role: sum(item["role"] == role for item in allowed)
        for role in ("partition", "cache_pickle", "cache_sidecar", "lookup_database")
    }
    if role_counts != {
        "partition": 648,
        "cache_pickle": 648,
        "cache_sidecar": 648,
        "lookup_database": 1,
    }:
        _fail(StrictProbeError, STOP_PROBE, "allowed file role counts changed")
    if len(value["partition_records"]) != 648 or len(value["cache_records"]) != 648:
        _fail(StrictProbeError, STOP_PROBE, "frozen subset cardinality changed")
    return dict(value)


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _require_denied_open(path: str, label: str) -> bool:
    try:
        with open(path, "rb"):
            pass
    except OSError as exc:
        if exc.errno == errno.EPERM:
            return True
        _fail(
            StrictProbeError,
            STOP_PROBE,
            f"{label} denied with unexpected errno {exc.errno}",
        )
    _fail(StrictProbeError, STOP_PROBE, f"{label} was readable")


def _require_network_denied() -> bool:
    for address in (("127.0.0.1", 9), ("1.1.1.1", 53)):
        connection: socket.socket | None = None
        try:
            connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            connection.settimeout(0.05)
            connection.connect(address)
        except OSError as exc:
            if exc.errno != errno.EPERM:
                _fail(
                    StrictProbeError,
                    STOP_PROBE,
                    f"network denied with unexpected errno {exc.errno}",
                )
        else:
            _fail(StrictProbeError, STOP_PROBE, "network connection succeeded")
        finally:
            if connection is not None:
                connection.close()
    return True


def _require_write_denied(path: Path) -> bool:
    try:
        with path.open("xb"):
            pass
    except OSError as exc:
        if exc.errno == errno.EPERM:
            return True
        _fail(
            StrictProbeError,
            STOP_PROBE,
            f"write denied with unexpected errno {exc.errno}",
        )
    else:
        try:
            path.unlink()
        finally:
            _fail(StrictProbeError, STOP_PROBE, "write outside staging succeeded")


def _sample_add(
    heap: list[tuple[int, tuple[int, ...], bytes, str]],
    selected: set[str],
    siret: Any,
    *,
    maximum: int,
) -> None:
    if type(siret) is not str or _SIRET_RE.fullmatch(siret) is None:
        return
    if siret in selected:
        return
    digest = hashlib.sha256(
        f"v412-store-lookup:{siret}".encode("utf-8")
    ).digest()
    reverse = (
        -int.from_bytes(digest, "big"),
        tuple(-ord(character) for character in siret),
        digest,
        siret,
    )
    if len(heap) < maximum:
        heapq.heappush(heap, reverse)
        selected.add(siret)
        return
    largest_digest, largest_siret = heap[0][2], heap[0][3]
    if (digest, siret) < (largest_digest, largest_siret):
        removed = heapq.heapreplace(heap, reverse)
        selected.remove(removed[3])
        selected.add(siret)


def run_child_probe(
    *,
    run_spec_path: Path,
    lookup_descriptor_path: Path,
    run_root: Path,
    forbidden_oracle: str,
    forbidden_audit: str,
    build_id: str,
) -> dict[str, Any]:
    """Execute the label-free Gate A probe and write ``store_probe.json``."""

    started = time.perf_counter_ns()
    def read_inherited(path: Path, label: str) -> bytes:
        match = re.fullmatch(r"/dev/fd/([0-9]+)", str(path))
        if match is None:
            _fail(StrictProbeError, STOP_PROBE, f"{label} must be an inherited FD")
        descriptor_fd = int(match.group(1))
        before = os.fstat(descriptor_fd)
        if not stat.S_ISREG(before.st_mode):
            _fail(StrictProbeError, STOP_PROBE, f"{label} FD is not regular")
        chunks = []
        offset = 0
        while True:
            block = os.pread(descriptor_fd, 1024 * 1024, offset)
            if not block:
                break
            chunks.append(block)
            offset += len(block)
        after = os.fstat(descriptor_fd)
        identity = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns")
        if any(getattr(before, key) != getattr(after, key) for key in identity):
            _fail(StrictProbeError, STOP_PROBE, f"{label} FD changed while reading")
        return b"".join(chunks)

    run_spec_bytes = read_inherited(run_spec_path, "run spec")
    run_spec_raw = _load_json_bytes_strict(
        run_spec_bytes,
        error_type=StrictProbeError,
        code=STOP_PROBE,
        label="run_spec.json",
    )
    run_spec = _validate_run_spec(run_spec_raw)
    descriptor_bytes = read_inherited(lookup_descriptor_path, "lookup descriptor")
    if hashlib.sha256(descriptor_bytes).hexdigest() != run_spec["lookup_descriptor_sha256"]:
        _fail(StrictProbeError, STOP_PROBE, "lookup descriptor hash mismatch")
    descriptor = _load_json_bytes_strict(
        descriptor_bytes,
        error_type=StrictProbeError,
        code=STOP_PROBE,
        label="lookup_descriptor.json",
    )
    if _SHA256_RE.fullmatch(build_id) is None:
        _fail(StrictProbeError, STOP_PROBE, "invalid certification build ID")

    _snapshot_path_chain(
        run_root,
        error_type=StrictProbeError,
        code=STOP_PROBE,
        final_kind="directory",
    )
    if Path.cwd() != run_root:
        _fail(StrictProbeError, STOP_PROBE, "probe cwd is not RUN_ROOT")
    output_dir = _ensure_private_directory(
        run_root,
        run_spec["staging_dir"],
    )
    tmp_dir = _ensure_private_directory(run_root, run_spec["tmp_dir"])
    if os.environ.get("TMPDIR") != str(tmp_dir):
        _fail(StrictProbeError, STOP_PROBE, "TMPDIR is not the private run tmp")

    allowed = run_spec["allowed_read_files"]
    sandbox_checks = {
        "allowed_read": False,
        "oracle_denied": False,
        "oracle_audit_denied": False,
        "network_denied": False,
        "write_denied": False,
    }
    first_allowed = _validate_allowed_file_record(allowed[0])
    try:
        with _verified_file(
            first_allowed["absolute_path"],
            expected_size=first_allowed["size_bytes"],
            expected_sha256=first_allowed["sha256"],
            error_type=StrictProbeError,
            code=STOP_PROBE,
        ) as handle:
            handle.read(1)
    except Exception as exc:
        _fail(StrictProbeError, STOP_PROBE, f"allowed read failed: {exc}")
    sandbox_checks["allowed_read"] = True
    sandbox_checks["oracle_denied"] = _require_denied_open(
        forbidden_oracle,
        "oracle",
    )
    sandbox_checks["oracle_audit_denied"] = _require_denied_open(
        forbidden_audit,
        "oracle audit",
    )
    sandbox_checks["network_denied"] = _require_network_denied()
    sandbox_checks["write_denied"] = _require_write_denied(
        run_root / "write-denied-sentinel"
    )

    partition_store = StrictPartitionStore(
        run_spec["partition_records"],
        allowed,
        max_cache_entries=1,
    )
    cache_store = StrictVerifiedTfidfCache(
        run_spec["cache_records"],
        allowed,
        namespace=CACHE_NAMESPACE,
        max_cache_entries=1,
    )
    if partition_store.partition_keys != cache_store.partition_keys:
        _fail(StrictProbeError, STOP_PROBE, "partition/cache key sets differ")
    partition_duration = 0
    cache_duration = 0
    raw_count = 0
    aligned_count = 0
    sample_heap: list[tuple[int, tuple[int, ...], bytes, str]] = []
    sample_selected: set[str] = set()
    for key in partition_store.partition_keys:
        stage = time.perf_counter_ns()
        status, rows = partition_store.load_with_status(key)
        partition_duration += time.perf_counter_ns() - stage
        if status not in {"VALID_ROWS", "VALID_EMPTY"}:
            _fail(StrictProbeError, STOP_PROBE, "invalid partition status")
        raw_count += len(rows)
        for row in rows:
            _sample_add(
                sample_heap,
                sample_selected,
                row.get("siret"),
                maximum=10_000,
            )
        aligned = _build_aligned_pool(rows)
        aligned_count += len(aligned)
        stage = time.perf_counter_ns()
        cache_store.get(key, aligned)
        cache_duration += time.perf_counter_ns() - stage
        cache_store.release(key)
        partition_store.release(key)
        del rows, aligned
        gc.collect()
        if _peak_rss_bytes() > run_spec["max_rss_bytes"]:
            _fail(StrictProbeError, STOP_PROBE, "RSS limit exceeded")
    lookup_started = time.perf_counter_ns()
    sample = sorted(
        ((item[2], item[3]) for item in sample_heap),
        key=lambda item: (item[0], item[1]),
    )
    requested_sample = [siret for _, siret in sample]
    missing = 0
    extra = 0
    with StrictSnapshotLookup(descriptor, allowed) as lookup:
        for offset in range(0, len(requested_sample), 100):
            batch = requested_sample[offset : offset + 100]
            result = lookup.get_candidate_scene_details(batch)
            missing += len(set(batch) - set(result))
            extra += len(set(result) - set(batch))
    lookup_duration = time.perf_counter_ns() - lookup_started
    if missing or extra:
        _fail(StrictProbeError, STOP_PROBE, "lookup sample mismatch")

    peak_rss = _peak_rss_bytes()
    if peak_rss > run_spec["max_rss_bytes"]:
        _fail(StrictProbeError, STOP_PROBE, "RSS limit exceeded")
    total_duration = time.perf_counter_ns() - started
    probe = {
        "schema_version": PROBE_SCHEMA,
        "build_id": build_id,
        "query_count": run_spec["query_count"],
        "distinct_key_count": len(partition_store.partition_keys),
        "partition_verified_count": len(partition_store.partition_keys),
        "partition_raw_row_count": raw_count,
        "cache_verified_count": len(cache_store.partition_keys),
        "aligned_pool_row_count": aligned_count,
        "cache_miss_count": 0,
        "rebuild_count": 0,
        "write_count": 0,
        "lookup_sample_count": len(requested_sample),
        "lookup_missing_count": missing,
        "lookup_extra_count": extra,
        "sandbox_checks": sandbox_checks,
        "peak_rss_bytes": peak_rss,
        "durations_ns": {
            "partitions": partition_duration,
            "cache": cache_duration,
            "lookup": lookup_duration,
            "total": total_duration,
        },
        "declarations": dict(_DECLARATIONS),
    }
    output_path = output_dir / "store_probe.json"
    if output_path.exists():
        _fail(StrictProbeError, STOP_PROBE, "store_probe.json already exists")
    payload = _canonical_json_bytes(probe)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor_fd = os.open(output_path, flags, 0o444)
    try:
        with os.fdopen(descriptor_fd, "wb", closefd=True) as handle:
            descriptor_fd = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor_fd >= 0:
            os.close(descriptor_fd)
    return probe


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the V4.12 strict-store probe")
    parser.add_argument("--run-spec", required=True, type=Path)
    parser.add_argument("--lookup-descriptor", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--forbidden-oracle", required=True)
    parser.add_argument("--forbidden-audit", required=True)
    parser.add_argument(
        "--build-id",
        required=True,
        help="Certification build ID.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        run_child_probe(
            run_spec_path=args.run_spec,
            lookup_descriptor_path=args.lookup_descriptor,
            run_root=args.run_root,
            forbidden_oracle=args.forbidden_oracle,
            forbidden_audit=args.forbidden_audit,
            build_id=args.build_id,
        )
    except StrictStoreError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except BaseException as exc:
        print(f"{STOP_PROBE}: unexpected failure: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
