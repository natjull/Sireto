#!/usr/bin/env python3
"""Fail-closed, offline-only skeleton for the V4.12 R30 review runtime.

This module implements the part of the execution contract that can be proved
without touching the network: canonical identifiers, the primary hash-chained
journal, an irreversibly closed broker, and two sequential worker boundaries.
It deliberately contains no network client and no switch that can enable one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import select
import stat
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


CONTRACT_COMMIT = "0fbdd80afcc618c33886d7b38badf010b1aba400"
SCHEMA_VERSION = "sireto-v4.12-r30-access-event-1"
ZERO_HASH = "0" * 64
HEX64 = re.compile(r"^[0-9a-f]{64}$")
UINT8_MAX = 255
UINT16_MAX = 65535
UINT64_MAX = 2**64 - 1

EVENT_FIELDS = (
    "schema_version",
    "event_ordinal",
    "event_kind",
    "attempt_id",
    "parent_intent_ordinal",
    "phase",
    "operation",
    "target_kind",
    "target_canonical",
    "query_id",
    "query_ordinal",
    "result_rank",
    "outcome",
    "error_type",
    "http_status",
    "byte_count",
    "content_sha256",
    "previous_event_sha256",
    "event_sha256",
)
PHASES = {
    "PREFLIGHT",
    "IDENTITY_DISCOVERY",
    "IDENTITY_SEAL",
    "COMPARISON",
    "PUBLICATION",
}
ACTION_OPERATIONS = {
    "OPEN_LOCAL",
    "WRITE_LOCAL",
    "SEARCH_REQUEST",
    "PAGE_REQUEST",
    "DNS_RESOLUTION",
    "SIRENE_LOOKUP",
}
NETWORK_OPERATIONS = {"SEARCH_REQUEST", "PAGE_REQUEST", "DNS_RESOLUTION"}
TARGET_KINDS = {"PATH", "URL", "HOSTNAME", "SIRET", "STATE", "NONE"}
OUTCOMES = {
    "PLANNED",
    "SUCCESS",
    "DENIED",
    "NETWORK_ERROR",
    "TIMEOUT",
    "HTTP_ERROR",
    "PARSE_ERROR",
    "IO_ERROR",
    "STOP_INTEGRITY",
    "NONE",
}
ERROR_TYPES = {
    "DNS",
    "PRIVATE_ADDRESS",
    "NETWORK",
    "CONNECT_TIMEOUT",
    "READ_TIMEOUT",
    "TLS",
    "HTTP_STATUS",
    "REDIRECT_FORBIDDEN",
    "TOO_LARGE",
    "CONTENT_ENCODING",
    "UNSUPPORTED_MIME",
    "MALFORMED_RESPONSE",
    "PARSE",
    "IO_INTEGRITY",
}
TRANSITIONS = (
    ("PREFLIGHT", "IDENTITY_NETWORK_OPEN"),
    ("IDENTITY_SEAL", "IDENTITY_SEALED_NETWORK_REVOKED"),
)
RESULT_ERROR_TYPES = {
    "NETWORK_ERROR": frozenset({"DNS", "PRIVATE_ADDRESS", "NETWORK", "TLS"}),
    "TIMEOUT": frozenset({"CONNECT_TIMEOUT", "READ_TIMEOUT"}),
    "PARSE_ERROR": frozenset({"PARSE"}),
    "IO_ERROR": frozenset({"IO_INTEGRITY"}),
    "STOP_INTEGRITY": frozenset({"IO_INTEGRITY"}),
}
MAX_WORKER_MESSAGE_BYTES = 64 * 1024
WORKER_TIMEOUT_SECONDS = 15.0
NATIVE_BUILD_SCHEMA = "sireto-v4.12-r30-native-worker-build-2"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
NATIVE_WORKER_PATH = PROJECT_ROOT / "build/v412_review_native_worker_r31"
NATIVE_WORKER_SOURCE = PROJECT_ROOT / "scripts/native/v412_review_worker.c"
NATIVE_WORKER_BUILDER = PROJECT_ROOT / "scripts/build_v412_review_native_worker.py"
NATIVE_WORKER_CLANG = Path("/Library/Developer/CommandLineTools/usr/bin/clang")
NATIVE_WORKER_LD = Path("/Library/Developer/CommandLineTools/usr/bin/ld")
NATIVE_WORKER_CODESIGN = Path("/usr/bin/codesign")
NATIVE_WORKER_SDK_SETTINGS = Path(
    "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/SDKSettings.json"
)
SANDBOX_EXEC_PATH = Path("/usr/bin/sandbox-exec")
NATIVE_WORKER_BASENAME = "v412_review_native_worker_r31"
NATIVE_WORKER_ARCH = "arm64"
NATIVE_WORKER_BUILD_FLAGS = (
    "-isysroot", "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk",
    "-arch", "arm64", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
    "-Wno-deprecated-declarations", "-Wl,-no_adhoc_codesign",
)
NATIVE_WORKER_SIGN_FLAGS = (
    "--force", "--sign", "-", "--identifier", "com.sireto.v412.review-worker",
)
PINNED_NATIVE_HASHES = {
    "artifact_sha256": "5f8be425e4aaf9d02ce5e71c162a13f63f7d14b629ce5b7960eb5197aef705f2",
    "builder_sha256": "64b326608355fd0d05bda5de1809db656070a1997e79caee787d3096298102cf",
    "clang_sha256": "49fdba60aca4c2eabc48ab9ee6d5f0659a840ecc2dd5bbca0554d6fa4d59601d",
    "codesign_sha256": "214d455584d19abc0d74d02b9cbc7d3da6bdcb0596c235e6156dd9ed2f4e1ba7",
    "ld_sha256": "be72c25252c843298882c07fab77c323f52843ad7037269c2e2ac4f4a6e6ee90",
    "sandbox_exec_sha256": "8290e4be7387a0df83cd1559e86afd880464f269450573d012795761fe298f16",
    "sdk_settings_sha256": "f231d03c93d59ebfffd410e941e50db8fcef6ec19ebe715967ab07ac44a51fc8",
    "source_sha256": "ac813189c426e7db6271b7649dee0f4f5c82f4b400bc53626378dc332a81c1ce",
}
PINNED_ROOT_ENTRIES = frozenset(
    {
        ".VolumeIcon.icns", ".file", ".nofollow", ".resolve", ".vol",
        "Applications", "Library", "System", "Users", "Volumes", "bin",
        "cores", "dev", "etc", "home", "opt", "private", "sbin", "tmp",
        "usr", "var",
    }
)
PINNED_USR_ENTRIES = frozenset(
    {"X11", "X11R6", "bin", "lib", "libexec", "local", "sbin", "share", "standalone"}
)
DENIED_ROOT_READ_PREFIXES = (
    "/Applications", "/Library", "/System/Volumes", "/Users", "/Volumes", "/bin", "/cores",
    "/dev", "/etc", "/home", "/opt", "/private", "/sbin", "/tmp",
    "/var", "/workspace", "/usr/bin", "/usr/libexec", "/usr/local",
    "/usr/sbin", "/usr/standalone", "/usr/X11", "/usr/X11R6",
)


class IntegrityStop(RuntimeError):
    """A contract invariant failed; no action may be retried implicitly."""


class OfflineNetworkDenied(IntegrityStop):
    """The synthetic broker has no network capability."""


def canonical_json(value: Any) -> bytes:
    """Return the exact no-LF canonical JSON encoding pinned by the contract."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _domain_id(domain: str, projection: Sequence[Any]) -> str:
    return hashlib.sha256(domain.encode("utf-8") + canonical_json(list(projection))).hexdigest()


def search_attempt_id(query_id: str, query_ordinal: int, search_query: str) -> str:
    return _domain_id(
        "SIRETO-V412-R30-SEARCH\0", [query_id, query_ordinal, search_query]
    )


def page_attempt_id(
    query_id: str,
    query_ordinal: int,
    result_rank: int,
    resolved_url: str,
    query_open_slot: int,
    dossier_open_ordinal: int,
) -> str:
    return _domain_id(
        "SIRETO-V412-R30-PAGE\0",
        [
            query_id,
            query_ordinal,
            result_rank,
            resolved_url,
            query_open_slot,
            dossier_open_ordinal,
        ],
    )


def dns_attempt_id(parent_attempt_id: str, normalized_hostname: str) -> str:
    return _domain_id(
        "SIRETO-V412-R30-DNS\0", [parent_attempt_id, normalized_hostname, 443]
    )


def occurrence_id(
    query_id: str,
    page_id: str,
    extracted_text_sha256: str,
    identifier_type: str,
    identifier_value: str,
    text_byte_start: int,
    text_byte_end: int,
) -> str:
    return _domain_id(
        "SIRETO-V412-R30-OCCURRENCE\0",
        [
            query_id,
            page_id,
            extracted_text_sha256,
            identifier_type,
            identifier_value,
            text_byte_start,
            text_byte_end,
        ],
    )


def error_id(
    query_id: str,
    stage: str,
    query_ordinal: int,
    result_rank: int | None,
    page_id: str | None,
    error_type: str,
) -> str:
    return _domain_id(
        "SIRETO-V412-R30-ERROR\0",
        [query_id, stage, query_ordinal, result_rank, page_id, error_type],
    )


def _uint(value: Any, maximum: int) -> bool:
    return type(value) is int and 0 <= value <= maximum


def _event_hash(record_without_hash: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        b"SIRETO-V412-R30-EVENT\0" + canonical_json(record_without_hash)
    ).hexdigest()


def _attempt_id(record: Mapping[str, Any]) -> str:
    projection = [
        record["event_ordinal"],
        record["phase"],
        record["operation"],
        record["target_kind"],
        record["target_canonical"],
        record["query_id"],
        record["query_ordinal"],
        record["result_rank"],
    ]
    return _domain_id("SIRETO-V412-R30-ATTEMPT\0", projection)


_DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def _write_exclusive_at(directory_fd: int, name: str, payload: bytes) -> None:
    if not name or name in {".", ".."} or "/" in name:
        raise IntegrityStop("unsafe exclusive-write basename")
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
        0o600,
        dir_fd=directory_fd,
    )
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise IntegrityStop("short exclusive write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.fsync(directory_fd)


def _open_directory_chain(path: Path) -> tuple[Path, list[tuple[int, int | None, str | None, tuple[int, int]]]]:
    raw = os.fspath(path)
    if not os.path.isabs(raw) or any(part in {"", ".", ".."} for part in Path(raw).parts[1:]):
        raise IntegrityStop("journal path is not canonical absolute syntax")
    absolute = Path(raw)
    if not absolute.is_absolute() or any(part in {"", ".", ".."} for part in absolute.parts[1:]):
        raise IntegrityStop("journal path is not canonical absolute syntax")
    retained: list[tuple[int, int | None, str | None, tuple[int, int]]] = []
    current_fd = os.open("/", _DIRECTORY_FLAGS | _NOFOLLOW)
    root_info = os.fstat(current_fd)
    retained.append((current_fd, None, None, (root_info.st_dev, root_info.st_ino)))
    try:
        for index, component in enumerate(absolute.parts[1:]):
            parent_fd = current_fd
            try:
                child_fd = os.open(component, _DIRECTORY_FLAGS | _NOFOLLOW, dir_fd=parent_fd)
            except FileNotFoundError:
                if index != len(absolute.parts[1:]) - 1:
                    raise IntegrityStop("journal parent directory does not exist")
                os.mkdir(component, 0o700, dir_fd=parent_fd)
                os.fsync(parent_fd)
                child_fd = os.open(component, _DIRECTORY_FLAGS | _NOFOLLOW, dir_fd=parent_fd)
            except OSError as exc:
                raise IntegrityStop("journal path contains a symlink or unsafe component") from exc
            info = os.fstat(child_fd)
            if not stat.S_ISDIR(info.st_mode):
                os.close(child_fd)
                raise IntegrityStop("journal path component is not a directory")
            retained.append((child_fd, parent_fd, component, (info.st_dev, info.st_ino)))
            current_fd = child_fd
    except Exception:
        for descriptor, _, _, _ in reversed(retained):
            os.close(descriptor)
        raise
    return absolute, retained


def _read_fd_all(descriptor: int, maximum: int = MAX_WORKER_MESSAGE_BYTES) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = os.read(descriptor, min(8192, maximum + 1 - total))
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
        total += len(chunk)
        if total > maximum:
            raise IntegrityStop("worker message exceeds closed size limit")


def _read_fd_all_before(descriptor: int, deadline: float, maximum: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([descriptor], [], [], remaining)[0]:
            raise IntegrityStop("sandbox worker output timeout")
        chunk = os.read(descriptor, min(8192, maximum + 1 - total))
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)
        total += len(chunk)
        if total > maximum:
            raise IntegrityStop("worker message exceeds closed size limit")


def _write_fd_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise IntegrityStop("worker pipe short write")
        offset += written


def _write_fd_all_before(descriptor: int, payload: bytes, deadline: float) -> None:
    offset = 0
    while offset < len(payload):
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([], [descriptor], [], remaining)[1]:
            raise IntegrityStop("sandbox worker input timeout")
        try:
            written = os.write(descriptor, payload[offset:])
        except BrokenPipeError as exc:
            raise IntegrityStop("sandbox worker closed its input") from exc
        if written <= 0:
            raise IntegrityStop("worker pipe short write")
        offset += written


@dataclass(frozen=True)
class Intent:
    event_ordinal: int
    attempt_id: str
    phase: str
    operation: str
    target_kind: str
    target_canonical: str
    query_id: str | None
    query_ordinal: int | None
    result_rank: int | None


class AccessJournal:
    """Append-only O_EXCL records; JSONL is only an exported projection."""

    def __init__(self, root: Path):
        self.root, self._directory_chain = _open_directory_chain(Path(root))
        self._root_fd = self._directory_chain[-1][0]
        self.events_dir = self.root / "journal_events"
        try:
            os.mkdir("journal_events", 0o700, dir_fd=self._root_fd)
        except FileExistsError as exc:
            raise IntegrityStop("journal_events already exists; runs are never resumed") from exc
        os.fsync(self._root_fd)
        self._events_fd = os.open(
            "journal_events", _DIRECTORY_FLAGS | _NOFOLLOW, dir_fd=self._root_fd
        )
        event_info = os.fstat(self._events_fd)
        self._events_identity = (event_info.st_dev, event_info.st_ino)
        self._lock = threading.Lock()
        self._records: list[dict[str, Any]] = []
        self._pending: dict[str, Intent] = {}
        self._transition_count = 0
        self._closed = False
        self._append(
            event_kind="GENESIS",
            attempt_id=None,
            parent_intent_ordinal=None,
            phase="PREFLIGHT",
            operation="GENESIS",
            target_kind="NONE",
            target_canonical=None,
            query_id=None,
            query_ordinal=None,
            result_rank=None,
            outcome="NONE",
            error_type=None,
            http_status=None,
            byte_count=None,
            content_sha256=None,
        )

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(record) for record in self._records)

    @property
    def state(self) -> str:
        return "GENESIS" if self._transition_count == 0 else TRANSITIONS[self._transition_count - 1][1]

    def _verify_directory_chain(self) -> None:
        if self._closed:
            raise IntegrityStop("journal descriptors are closed")
        for descriptor, parent_fd, component, identity in self._directory_chain:
            current = os.fstat(descriptor)
            if (current.st_dev, current.st_ino) != identity or not stat.S_ISDIR(current.st_mode):
                raise IntegrityStop("retained journal directory identity changed")
            if parent_fd is not None and component is not None:
                try:
                    linked = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
                except OSError as exc:
                    raise IntegrityStop("journal ancestor link was removed or substituted") from exc
                if not stat.S_ISDIR(linked.st_mode) or (linked.st_dev, linked.st_ino) != identity:
                    raise IntegrityStop("journal ancestor link was substituted")
        try:
            linked_events = os.stat("journal_events", dir_fd=self._root_fd, follow_symlinks=False)
        except OSError as exc:
            raise IntegrityStop("journal_events link was removed or substituted") from exc
        if not stat.S_ISDIR(linked_events.st_mode) or (
            linked_events.st_dev,
            linked_events.st_ino,
        ) != self._events_identity:
            raise IntegrityStop("journal_events link was substituted")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        events_fd = getattr(self, "_events_fd", -1)
        if events_fd >= 0:
            os.close(events_fd)
            self._events_fd = -1
        for descriptor, _, _, _ in reversed(getattr(self, "_directory_chain", [])):
            os.close(descriptor)
        self._directory_chain = []

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _append(self, **fields: Any) -> dict[str, Any]:
        with self._lock:
            self._verify_directory_chain()
            ordinal = len(self._records)
            record = {
                "schema_version": SCHEMA_VERSION,
                "event_ordinal": ordinal,
                **fields,
                "previous_event_sha256": (
                    ZERO_HASH if ordinal == 0 else self._records[-1]["event_sha256"]
                ),
            }
            _validate_event(record, prior=self._records)
            record["event_sha256"] = _event_hash(record)
            _validate_event(record, prior=self._records)
            _write_exclusive_at(
                self._events_fd,
                f"{ordinal:020d}.json",
                canonical_json(record) + b"\n",
            )
            self._records.append(record)
            return dict(record)

    def intent(
        self,
        *,
        phase: str,
        operation: str,
        target_kind: str,
        target_canonical: str,
        query_id: str | None = None,
        query_ordinal: int | None = None,
        result_rank: int | None = None,
    ) -> Intent:
        seed = {
            "event_ordinal": len(self._records),
            "phase": phase,
            "operation": operation,
            "target_kind": target_kind,
            "target_canonical": target_canonical,
            "query_id": query_id,
            "query_ordinal": query_ordinal,
            "result_rank": result_rank,
        }
        attempt = _attempt_id(seed)
        record = self._append(
            event_kind="INTENT",
            attempt_id=attempt,
            parent_intent_ordinal=None,
            phase=phase,
            operation=operation,
            target_kind=target_kind,
            target_canonical=target_canonical,
            query_id=query_id,
            query_ordinal=query_ordinal,
            result_rank=result_rank,
            outcome="PLANNED",
            error_type=None,
            http_status=None,
            byte_count=None,
            content_sha256=None,
        )
        handle = Intent(
            record["event_ordinal"],
            attempt,
            phase,
            operation,
            target_kind,
            target_canonical,
            query_id,
            query_ordinal,
            result_rank,
        )
        self._pending[attempt] = handle
        return handle

    def result(
        self,
        intent: Intent,
        *,
        outcome: str,
        error_type: str | None = None,
        http_status: int | None = None,
        byte_count: int | None = None,
        content_sha256: str | None = None,
    ) -> dict[str, Any]:
        if self._pending.get(intent.attempt_id) != intent:
            raise IntegrityStop("RESULT has no unique pending INTENT")
        record = self._append(
            event_kind="RESULT",
            attempt_id=intent.attempt_id,
            parent_intent_ordinal=intent.event_ordinal,
            phase=intent.phase,
            operation=intent.operation,
            target_kind=intent.target_kind,
            target_canonical=intent.target_canonical,
            query_id=intent.query_id,
            query_ordinal=intent.query_ordinal,
            result_rank=intent.result_rank,
            outcome=outcome,
            error_type=error_type,
            http_status=http_status,
            byte_count=byte_count,
            content_sha256=content_sha256,
        )
        del self._pending[intent.attempt_id]
        return record

    def state_transition(self, *, phase: str, target_state: str) -> dict[str, Any]:
        if self._transition_count >= len(TRANSITIONS) or (
            phase,
            target_state,
        ) != TRANSITIONS[self._transition_count]:
            raise IntegrityStop("invalid or non-monotone state transition")
        if target_state == "IDENTITY_SEALED_NETWORK_REVOKED" and any(
            intent.operation in NETWORK_OPERATIONS for intent in self._pending.values()
        ):
            raise IntegrityStop("network INTENT is still pending at revocation")
        record = self._append(
            event_kind="STATE_TRANSITION",
            attempt_id=None,
            parent_intent_ordinal=None,
            phase=phase,
            operation="STATE_TRANSITION",
            target_kind="STATE",
            target_canonical=target_state,
            query_id=None,
            query_ordinal=None,
            result_rank=None,
            outcome="SUCCESS",
            error_type=None,
            http_status=None,
            byte_count=None,
            content_sha256=None,
        )
        self._transition_count += 1
        return record

    def verify_complete(self) -> str:
        self._verify_directory_chain()
        loaded: list[dict[str, Any]] = []
        names = sorted(os.listdir(self._events_fd))
        for expected, name in enumerate(names):
            if name != f"{expected:020d}.json":
                raise IntegrityStop("non-contiguous or unsafe journal record")
            descriptor = os.open(
                name, os.O_RDONLY | _NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                dir_fd=self._events_fd,
            )
            try:
                info = os.fstat(descriptor)
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise IntegrityStop("journal record is not a mono-link regular file")
                chunks: list[bytes] = []
                while True:
                    chunk = os.read(descriptor, 8192)
                    if not chunk:
                        break
                    chunks.append(chunk)
                raw = b"".join(chunks)
            finally:
                os.close(descriptor)
            try:
                value = json.loads(raw.decode("utf-8"))
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise IntegrityStop("invalid journal JSON") from exc
            if type(value) is not dict or raw != canonical_json(value) + b"\n":
                raise IntegrityStop("journal record is not canonical JSON plus LF")
            _validate_event(value, prior=loaded)
            loaded.append(value)
        if not loaded or loaded[0]["event_kind"] != "GENESIS":
            raise IntegrityStop("missing genesis")
        pending: dict[str, int] = {}
        for record in loaded:
            if record["event_kind"] == "INTENT":
                pending[record["attempt_id"]] = record["event_ordinal"]
            elif record["event_kind"] == "RESULT":
                if pending.pop(record["attempt_id"], None) != record["parent_intent_ordinal"]:
                    raise IntegrityStop("RESULT does not close its INTENT")
        if pending:
            raise IntegrityStop("terminal INTENT without RESULT")
        if loaded != self._records:
            raise IntegrityStop("journal memory/disk divergence")
        transition_count = sum(record["event_kind"] == "STATE_TRANSITION" for record in loaded)
        if transition_count != self._transition_count:
            raise IntegrityStop("journal transition state diverges from records")
        return loaded[-1]["event_sha256"]

    def project_jsonl(self, destination: Path) -> str:
        self.verify_complete()
        destination = Path(os.path.abspath(os.fspath(destination)))
        if destination.parent != self.root or destination.name != "access_journal.jsonl":
            raise IntegrityStop("journal projection destination is not the closed path")
        payload = b"".join(canonical_json(record) + b"\n" for record in self._records)
        _write_exclusive_at(self._root_fd, destination.name, payload)
        return hashlib.sha256(payload).hexdigest()


def _validate_event(record: Mapping[str, Any], *, prior: Sequence[Mapping[str, Any]]) -> None:
    expected_keys = set(EVENT_FIELDS)
    if "event_sha256" not in record:
        expected_keys.remove("event_sha256")
    if type(record) is not dict or set(record) != expected_keys:
        raise IntegrityStop("event fields differ from the closed schema")
    if record["schema_version"] != SCHEMA_VERSION:
        raise IntegrityStop("event schema mismatch")
    scalar_strings = ("event_kind", "phase", "operation", "target_kind", "outcome")
    if any(type(record[key]) is not str for key in scalar_strings):
        raise IntegrityStop("event enum fields must be strings")
    if record["target_canonical"] is not None and type(record["target_canonical"]) is not str:
        raise IntegrityStop("target canonical must be string or null")
    if record["query_id"] is not None and type(record["query_id"]) is not str:
        raise IntegrityStop("query id must be string or null")
    if record["attempt_id"] is not None and (
        type(record["attempt_id"]) is not str or not HEX64.fullmatch(record["attempt_id"])
    ):
        raise IntegrityStop("attempt id must be 64 lowercase hex or null")
    if record["parent_intent_ordinal"] is not None and not _uint(
        record["parent_intent_ordinal"], UINT64_MAX
    ):
        raise IntegrityStop("parent intent ordinal must be uint64 or null")
    if record["error_type"] is not None and type(record["error_type"]) is not str:
        raise IntegrityStop("error type must be string or null")
    if record["http_status"] is not None and not _uint(record["http_status"], UINT16_MAX):
        raise IntegrityStop("http status must be uint16 or null")
    if record["byte_count"] is not None and not _uint(record["byte_count"], UINT64_MAX):
        raise IntegrityStop("byte count must be uint64 or null")
    if record["content_sha256"] is not None and (
        type(record["content_sha256"]) is not str
        or not HEX64.fullmatch(record["content_sha256"])
    ):
        raise IntegrityStop("content hash must be 64 lowercase hex or null")
    ordinal = record["event_ordinal"]
    if not _uint(ordinal, UINT64_MAX) or ordinal != len(prior):
        raise IntegrityStop("event ordinal is not contiguous uint64")
    expected_previous = ZERO_HASH if ordinal == 0 else prior[-1]["event_sha256"]
    if record["previous_event_sha256"] != expected_previous:
        raise IntegrityStop("journal chain is broken")
    if "event_sha256" in record:
        if not isinstance(record["event_sha256"], str) or not HEX64.fullmatch(record["event_sha256"]):
            raise IntegrityStop("invalid event hash")
        unsigned = {key: value for key, value in record.items() if key != "event_sha256"}
        if record["event_sha256"] != _event_hash(unsigned):
            raise IntegrityStop("event hash mismatch")
    if record["phase"] not in PHASES or record["target_kind"] not in TARGET_KINDS:
        raise IntegrityStop("invalid event enum")
    if record["outcome"] not in OUTCOMES:
        raise IntegrityStop("invalid event outcome")
    if record["query_ordinal"] is not None and not _uint(record["query_ordinal"], UINT8_MAX):
        raise IntegrityStop("query ordinal is not uint8")
    if record["result_rank"] is not None and not _uint(record["result_rank"], UINT8_MAX):
        raise IntegrityStop("result rank is not uint8")
    kind = record["event_kind"]
    result_fields = ("error_type", "http_status", "byte_count", "content_sha256")
    if kind == "GENESIS":
        if ordinal != 0 or any(
            record[key] is not None
            for key in (
                "attempt_id", "parent_intent_ordinal", "target_canonical",
                "query_id", "query_ordinal", "result_rank", *result_fields,
            )
        ) or (record["phase"], record["operation"], record["target_kind"], record["outcome"]) != (
            "PREFLIGHT", "GENESIS", "NONE", "NONE"
        ):
            raise IntegrityStop("invalid GENESIS matrix")
        return
    if kind == "INTENT":
        if record["operation"] not in ACTION_OPERATIONS or record["target_canonical"] is None:
            raise IntegrityStop("invalid INTENT action or target")
        required_target = {
            "OPEN_LOCAL": "PATH",
            "WRITE_LOCAL": "PATH",
            "SEARCH_REQUEST": "URL",
            "PAGE_REQUEST": "URL",
            "DNS_RESOLUTION": "HOSTNAME",
            "SIRENE_LOOKUP": "SIRET",
        }[record["operation"]]
        if record["target_kind"] != required_target:
            raise IntegrityStop("INTENT target kind does not match its operation")
        if record["parent_intent_ordinal"] is not None or record["outcome"] != "PLANNED":
            raise IntegrityStop("invalid INTENT matrix")
        if any(record[key] is not None for key in result_fields):
            raise IntegrityStop("INTENT contains result fields")
        if record["attempt_id"] != _attempt_id(record):
            raise IntegrityStop("INTENT attempt id mismatch")
        return
    if kind == "STATE_TRANSITION":
        prior_transitions = [
            (item["phase"], item["target_canonical"])
            for item in prior
            if item["event_kind"] == "STATE_TRANSITION"
        ]
        expected_transition = (
            TRANSITIONS[len(prior_transitions)]
            if len(prior_transitions) < len(TRANSITIONS)
            else None
        )
        if any(
            record[key] is not None
            for key in (
                "attempt_id", "parent_intent_ordinal", "query_id", "query_ordinal",
                "result_rank", *result_fields,
            )
        ) or (record["operation"], record["target_kind"], record["outcome"]) != (
            "STATE_TRANSITION", "STATE", "SUCCESS"
        ) or (record["phase"], record["target_canonical"]) != expected_transition:
            raise IntegrityStop("invalid STATE_TRANSITION matrix")
        return
    if kind != "RESULT":
        raise IntegrityStop("invalid event kind")
    parent_ordinal = record["parent_intent_ordinal"]
    if not _uint(parent_ordinal, UINT64_MAX) or parent_ordinal >= len(prior):
        raise IntegrityStop("RESULT parent is invalid")
    parent = prior[parent_ordinal]
    comparable = (
        "attempt_id", "phase", "operation", "target_kind", "target_canonical",
        "query_id", "query_ordinal", "result_rank",
    )
    if parent["event_kind"] != "INTENT" or any(record[key] != parent[key] for key in comparable):
        raise IntegrityStop("RESULT differs from its INTENT")
    outcome = record["outcome"]
    if outcome in {"PLANNED", "NONE"}:
        raise IntegrityStop("RESULT is not terminal")
    error_type = record["error_type"]
    if error_type is not None and error_type not in ERROR_TYPES:
        raise IntegrityStop("invalid collection error type")
    if outcome == "SUCCESS":
        if error_type is not None or record["http_status"] is not None:
            raise IntegrityStop("SUCCESS contains error fields")
        paired = (record["byte_count"] is None, record["content_sha256"] is None)
        if paired[0] != paired[1]:
            raise IntegrityStop("SUCCESS payload fields are not paired")
        if record["byte_count"] is not None and (
            not _uint(record["byte_count"], UINT64_MAX)
            or not isinstance(record["content_sha256"], str)
            or not HEX64.fullmatch(record["content_sha256"])
        ):
            raise IntegrityStop("invalid SUCCESS payload")
    elif outcome == "HTTP_ERROR":
        if not _uint(record["http_status"], UINT16_MAX) or error_type is not None:
            raise IntegrityStop("HTTP_ERROR requires uint16 status")
    elif outcome == "DENIED":
        if error_type != "IO_INTEGRITY" or record["http_status"] is not None:
            raise IntegrityStop("DENIED requires IO_INTEGRITY")
    elif outcome in RESULT_ERROR_TYPES:
        if error_type not in RESULT_ERROR_TYPES[outcome] or record["http_status"] is not None:
            raise IntegrityStop(f"{outcome} requires error_type")
    if outcome != "SUCCESS" and (record["byte_count"] is not None or record["content_sha256"] is not None):
        raise IntegrityStop("non-SUCCESS result contains payload fields")


class OfflineBroker:
    """Closed broker facade: it can record requests but cannot execute them."""

    def __init__(self, journal: AccessJournal):
        if journal.state != "IDENTITY_NETWORK_OPEN":
            raise IntegrityStop("broker requires the exact network-open transition")
        self._journal = journal
        self._state = "IDENTITY_NETWORK_OPEN"

    @property
    def state(self) -> str:
        return self._state

    def revoke(self) -> None:
        if self._state != "IDENTITY_NETWORK_OPEN":
            raise IntegrityStop("network revocation is irreversible")
        self._state = "IDENTITY_SEALED_NETWORK_REVOKED"
        self._journal.state_transition(phase="IDENTITY_SEAL", target_state=self._state)

    def assert_revoked(self) -> None:
        if self._state != "IDENTITY_SEALED_NETWORK_REVOKED":
            raise IntegrityStop("comparison requires prior network revocation")

    def _deny(
        self,
        *,
        operation: str,
        target_kind: str,
        target: str,
        query_id: str,
        query_ordinal: int,
        result_rank: int | None = None,
    ) -> None:
        if operation not in NETWORK_OPERATIONS:
            raise IntegrityStop("broker operation is not a network operation")
        intent = self._journal.intent(
            phase="IDENTITY_DISCOVERY",
            operation=operation,
            target_kind=target_kind,
            target_canonical=target,
            query_id=query_id,
            query_ordinal=query_ordinal,
            result_rank=result_rank,
        )
        outcome = "STOP_INTEGRITY" if self._state == "IDENTITY_SEALED_NETWORK_REVOKED" else "DENIED"
        self._journal.result(intent, outcome=outcome, error_type="IO_INTEGRITY")
        raise OfflineNetworkDenied(f"{operation} is unavailable in offline synthetic mode")

    def search_request(self, url: str, *, query_id: str, query_ordinal: int) -> None:
        self._deny(
            operation="SEARCH_REQUEST", target_kind="URL", target=url,
            query_id=query_id, query_ordinal=query_ordinal,
        )

    def page_request(
        self, url: str, *, query_id: str, query_ordinal: int, result_rank: int
    ) -> None:
        self._deny(
            operation="PAGE_REQUEST", target_kind="URL", target=url,
            query_id=query_id, query_ordinal=query_ordinal, result_rank=result_rank,
        )

    def dns_resolution(
        self, hostname: str, *, query_id: str, query_ordinal: int
    ) -> None:
        self._deny(
            operation="DNS_RESOLUTION", target_kind="HOSTNAME", target=hostname,
            query_id=query_id, query_ordinal=query_ordinal,
        )




@dataclass(frozen=True)
class IdentityInput:
    query_id: str
    crm_name: str
    crm_address: str
    crm_postcode: str

    def __post_init__(self) -> None:
        if any(
            type(value) is not str
            for value in (self.query_id, self.crm_name, self.crm_address, self.crm_postcode)
        ) or not self.query_id:
            raise IntegrityStop("invalid identity input types")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "IdentityInput":
        fields = {"query_id", "crm_name", "crm_address", "crm_postcode"}
        if type(value) is not dict or set(value) != fields or any(type(value[key]) is not str for key in fields):
            raise IntegrityStop("identity input fields differ from the closed allowlist")
        return cls(**value)


@dataclass(frozen=True)
class IdentityArtifact:
    query_id: str
    payload_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.query_id) is not str
            or not self.query_id
            or type(self.payload_sha256) is not str
            or not HEX64.fullmatch(self.payload_sha256)
        ):
            raise IntegrityStop("invalid identity artifact")


class IdentityDiscoveryWorker:
    """Synthetic identity pass with no candidate-bearing input type."""

    def run(self, value: IdentityInput) -> IdentityArtifact:
        if type(value) is not IdentityInput:
            raise IntegrityStop("identity worker received an untrusted input type")
        projection = {
            "crm_address": value.crm_address,
            "crm_name": value.crm_name,
            "crm_postcode": value.crm_postcode,
            "query_id": value.query_id,
        }
        return IdentityArtifact(value.query_id, hashlib.sha256(canonical_json(projection)).hexdigest())


@dataclass(frozen=True)
class ComparisonInput:
    query_id: str
    candidate_sirets: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.query_id) is not str or not self.query_id or type(self.candidate_sirets) is not tuple:
            raise IntegrityStop("invalid comparison input types")
        if len(self.candidate_sirets) > 100 or any(
            type(item) is not str
            or len(item) != 14
            or not item.isascii()
            or not item.isdigit()
            for item in self.candidate_sirets
        ):
            raise IntegrityStop("comparison candidates violate the ASCII SIRET/budget contract")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ComparisonInput":
        if type(value) is not dict or set(value) != {"query_id", "candidate_sirets"}:
            raise IntegrityStop("comparison input fields differ from the closed allowlist")
        candidates = value["candidate_sirets"]
        if type(value["query_id"]) is not str or type(candidates) not in {list, tuple}:
            raise IntegrityStop("invalid comparison input types")
        return cls(value["query_id"], tuple(candidates))


@dataclass(frozen=True)
class ComparisonArtifact:
    query_id: str
    identity_payload_sha256: str
    candidate_count: int
    candidates_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.query_id) is not str
            or not self.query_id
            or type(self.identity_payload_sha256) is not str
            or not HEX64.fullmatch(self.identity_payload_sha256)
            or not _uint(self.candidate_count, 100)
            or type(self.candidates_sha256) is not str
            or not HEX64.fullmatch(self.candidates_sha256)
        ):
            raise IntegrityStop("invalid comparison artifact")


class FrozenCandidateComparisonWorker:
    """Second worker boundary; it cannot start while the broker is open."""

    def run(
        self,
        identity: IdentityArtifact,
        value: ComparisonInput,
        *,
        broker: OfflineBroker,
    ) -> ComparisonArtifact:
        if type(identity) is not IdentityArtifact or type(value) is not ComparisonInput:
            raise IntegrityStop("comparison worker received an untrusted boundary type")
        if type(broker) is not OfflineBroker:
            raise IntegrityStop("comparison worker received an untrusted broker")
        broker.assert_revoked()
        if identity.query_id != value.query_id:
            raise IntegrityStop("identity/comparison query mismatch")
        return ComparisonArtifact(
            query_id=value.query_id,
            identity_payload_sha256=identity.payload_sha256,
            candidate_count=len(value.candidate_sirets),
            candidates_sha256=hashlib.sha256(canonical_json(list(value.candidate_sirets))).hexdigest(),
        )


WORKER_SCHEMA = "sireto-v4.12-r30-offline-worker-message-1"


def _sha256_fd(descriptor: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    size = os.fstat(descriptor).st_size
    while offset < size:
        chunk = os.pread(descriptor, min(1024 * 1024, size - offset), offset)
        if not chunk:
            raise IntegrityStop("authenticated artifact changed during hashing")
        digest.update(chunk)
        offset += len(chunk)
    return digest.hexdigest()


def _open_authenticated_regular(path: Path) -> int:
    descriptor = os.open(path, os.O_RDONLY | _NOFOLLOW | getattr(os, "O_CLOEXEC", 0))
    info = os.fstat(descriptor)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or info.st_uid not in {0, os.getuid()}
        or info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    ):
        os.close(descriptor)
        raise IntegrityStop("artifact is not an authenticated mono-link regular file")
    return descriptor


def _open_native_worker(path: Path) -> int:
    if path.name != NATIVE_WORKER_BASENAME or os.uname().machine != NATIVE_WORKER_ARCH:
        raise IntegrityStop("native worker basename or architecture is not pinned")
    receipt_path = path.with_suffix(path.suffix + ".json")
    receipt_fd = -1
    try:
        receipt_fd = _open_authenticated_regular(receipt_path)
        if stat.S_IMODE(os.fstat(receipt_fd).st_mode) != 0o600:
            raise IntegrityStop("native worker receipt mode is not pinned")
        receipt_raw = _read_fd_all(receipt_fd)
        receipt = json.loads(receipt_raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IntegrityStop("native worker build receipt is unavailable") from exc
    finally:
        if receipt_fd >= 0:
            os.close(receipt_fd)
    expected_receipt: dict[str, Any] = {
        "architecture": NATIVE_WORKER_ARCH,
        "basename": NATIVE_WORKER_BASENAME,
        "build_flags": list(NATIVE_WORKER_BUILD_FLAGS),
        "sign_flags": list(NATIVE_WORKER_SIGN_FLAGS),
        **PINNED_NATIVE_HASHES,
        "schema_version": NATIVE_BUILD_SCHEMA,
    }
    if (
        type(receipt) is not dict
        or receipt != expected_receipt
        or receipt_raw != canonical_json(receipt) + b"\n"
    ):
        raise IntegrityStop("native worker build receipt is not canonical")
    pinned_inputs = (
        (NATIVE_WORKER_BUILDER, "builder_sha256"),
        (NATIVE_WORKER_SOURCE, "source_sha256"),
        (NATIVE_WORKER_CLANG, "clang_sha256"),
        (NATIVE_WORKER_CODESIGN, "codesign_sha256"),
        (NATIVE_WORKER_LD, "ld_sha256"),
        (NATIVE_WORKER_SDK_SETTINGS, "sdk_settings_sha256"),
        (SANDBOX_EXEC_PATH, "sandbox_exec_sha256"),
    )
    input_fds: list[tuple[int, str]] = []
    worker_fd = -1
    try:
        for input_path, hash_name in pinned_inputs:
            input_fds.append((_open_authenticated_regular(input_path), hash_name))
        worker_fd = _open_authenticated_regular(path)
        if stat.S_IMODE(os.fstat(worker_fd).st_mode) != 0o500:
            raise IntegrityStop("native worker executable mode is not pinned")
        if any(
            _sha256_fd(descriptor) != PINNED_NATIVE_HASHES[hash_name]
            for descriptor, hash_name in input_fds
        ) or _sha256_fd(worker_fd) != PINNED_NATIVE_HASHES["artifact_sha256"]:
            raise IntegrityStop("native worker build receipt authentication failed")
    except Exception:
        if worker_fd >= 0:
            os.close(worker_fd)
        raise
    finally:
        for descriptor, _ in input_fds:
            os.close(descriptor)
    return worker_fd


def _assert_closed_runtime_roots() -> None:
    if set(os.listdir("/")) != PINNED_ROOT_ENTRIES:
        raise IntegrityStop("filesystem root differs from the pinned closed inventory")
    if set(os.listdir("/usr")) != PINNED_USR_ENTRIES:
        raise IntegrityStop("/usr differs from the pinned closed inventory")
    for path in (Path("/System"), Path("/usr/lib"), Path("/usr/share")):
        descriptor = os.open(path, _DIRECTORY_FLAGS | _NOFOLLOW)
        try:
            info = os.fstat(descriptor)
            if info.st_uid != 0 or info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
                raise IntegrityStop("runtime read root ownership or mode is unsafe")
        finally:
            os.close(descriptor)


def _sandbox_profile(worker_path: Path) -> str:
    worker_literal = json.dumps(os.fspath(worker_path))
    denied_subpaths = "".join(
        f"(deny file-read-data (subpath {json.dumps(path)}))"
        for path in DENIED_ROOT_READ_PREFIXES
    )
    return "".join(
        (
            "(version 1)",
            "(deny default)",
            "(deny network*)",
            "(deny process-fork)",
            "(deny process-exec)",
            f"(allow process-exec (literal {worker_literal}))",
            # dyld requires a pathless file-read-data grant on current macOS.
            "(allow file-read-data)",
            "(deny file-read-data (regex #\"^/\\.[^/]+(?:/|$)\"))",
            denied_subpaths,
            f"(allow file-read* (subpath \"/System\") (subpath \"/usr/lib\") "
            f"(subpath \"/usr/share\") (literal {worker_literal}))",
            "(deny file-write* (regex #\"^/\"))",
            "(allow file-ioctl)",
            "(allow sysctl-read)",
            "(allow signal (target self))",
        )
    )


def _run_sandboxed_worker(
    role: str,
    payload: Mapping[str, Any],
    *,
    broker: OfflineBroker | None = None,
    native_worker: Path | None = None,
) -> dict[str, Any]:
    if role not in {"IDENTITY_DISCOVERY", "FROZEN_CANDIDATE_COMPARISON", "CAPABILITY_PROBE"}:
        raise IntegrityStop("unknown sandbox worker role")
    raw = canonical_json(payload)
    if len(raw) > MAX_WORKER_MESSAGE_BYTES:
        raise IntegrityStop("worker input exceeds closed size limit")
    if role == "FROZEN_CANDIDATE_COMPARISON":
        if type(broker) is not OfflineBroker:
            raise IntegrityStop("comparison spawn requires the live broker capability")
        broker.assert_revoked()
        identity = payload.get("identity_artifact")
        item = payload.get("comparison_input")
        if (
            type(identity) is not dict
            or type(item) is not dict
            or payload.get("schema_version") != WORKER_SCHEMA
            or payload.get("network_state") != broker.state
            or identity.get("query_id") != item.get("query_id")
            or not isinstance(identity.get("payload_sha256"), str)
            or not HEX64.fullmatch(identity["payload_sha256"])
        ):
            raise IntegrityStop("comparison native envelope is invalid")
        candidates = ComparisonInput.from_mapping(item).candidate_sirets
        native_input = (
            bytes.fromhex(identity["payload_sha256"])
            + len(candidates).to_bytes(4, "big")
            + canonical_json(list(candidates))
        )
    elif broker is not None:
        raise IntegrityStop("broker capability supplied to wrong worker")
    elif role == "IDENTITY_DISCOVERY":
        if payload.get("schema_version") != WORKER_SCHEMA:
            raise IntegrityStop("identity native envelope is invalid")
        item = IdentityInput.from_mapping(payload.get("identity_input"))
        native_input = canonical_json(
            {
                "crm_address": item.crm_address,
                "crm_name": item.crm_name,
                "crm_postcode": item.crm_postcode,
                "query_id": item.query_id,
            }
        )
    else:
        if (
            set(payload) != {"schema_version", "forbidden_path"}
            or payload.get("schema_version") != WORKER_SCHEMA
            or type(payload.get("forbidden_path")) is not str
        ):
            raise IntegrityStop("capability probe envelope is invalid")
        native_input = payload["forbidden_path"].encode("utf-8")
    if len(native_input) > MAX_WORKER_MESSAGE_BYTES:
        raise IntegrityStop("native worker input exceeds closed size limit")

    _assert_closed_runtime_roots()
    worker_fd = _open_native_worker(
        NATIVE_WORKER_PATH if native_worker is None else Path(native_worker)
    )
    sandbox_fd = _open_authenticated_regular(SANDBOX_EXEC_PATH)
    if _sha256_fd(sandbox_fd) != PINNED_NATIVE_HASHES["sandbox_exec_sha256"]:
        os.close(worker_fd)
        os.close(sandbox_fd)
        raise IntegrityStop("sandbox-exec hash differs from the pinned trust anchor")
    descriptors = [worker_fd, sandbox_fd]
    staging: Path | None = None
    worker_copy: Path | None = None
    process: subprocess.Popen[bytes] | None = None
    native_output: bytes | None = None
    try:
        staging = Path(tempfile.mkdtemp(prefix="sireto-v412-native-", dir="/private/tmp"))
        worker_copy = staging / NATIVE_WORKER_BASENAME
        copy_fd = os.open(
            worker_copy,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW,
            0o500,
        )
        descriptors.append(copy_fd)
        worker_size = os.fstat(worker_fd).st_size
        offset = 0
        while offset < worker_size:
            chunk = os.pread(worker_fd, min(1024 * 1024, worker_size - offset), offset)
            if not chunk:
                raise IntegrityStop("native worker changed while staging")
            _write_fd_all(copy_fd, chunk)
            offset += len(chunk)
        os.fsync(copy_fd)
        os.close(copy_fd)
        descriptors.remove(copy_fd)
        staged_worker_fd = _open_authenticated_regular(worker_copy)
        descriptors.append(staged_worker_fd)
        if _sha256_fd(staged_worker_fd) != PINNED_NATIVE_HASHES["artifact_sha256"]:
            raise IntegrityStop("staged native worker differs from the pinned trust anchor")
        staging_fd = os.open(staging, _DIRECTORY_FLAGS | _NOFOLLOW)
        descriptors.append(staging_fd)
        os.fsync(staging_fd)

        pipes: list[tuple[int, int]] = []
        for _ in range(4):
            pair = os.pipe()
            pipes.append(pair)
            descriptors.extend(pair)
        (input_read, input_write), (output_read, output_write), (
            ready_read,
            ready_write,
        ), (gate_read, gate_write) = pipes
        command = [
            os.fspath(SANDBOX_EXEC_PATH),
            "-p",
            _sandbox_profile(worker_copy),
            os.fspath(worker_copy),
            role,
            str(input_read),
            str(output_write),
            str(ready_write),
            str(gate_read),
        ]
        process = subprocess.Popen(
            command,
            close_fds=True,
            pass_fds=(input_read, output_write, ready_write, gate_read),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for descriptor in (input_read, output_write, ready_write, gate_read):
            os.close(descriptor)
            descriptors.remove(descriptor)
        deadline = time.monotonic() + WORKER_TIMEOUT_SECONDS
        if not select.select([ready_read], [], [], WORKER_TIMEOUT_SECONDS)[0] or os.read(ready_read, 1) != b"R":
            raise IntegrityStop(f"native worker {role} READY timeout or failure")
        linked = os.stat(worker_copy, follow_symlinks=False)
        copied = os.fstat(staged_worker_fd)
        if (
            (linked.st_dev, linked.st_ino) != (copied.st_dev, copied.st_ino)
            or linked.st_size != copied.st_size
            or _sha256_fd(staged_worker_fd) != PINNED_NATIVE_HASHES["artifact_sha256"]
        ):
            raise IntegrityStop("staged native worker changed across spawn")
        sandbox_linked = os.stat(SANDBOX_EXEC_PATH, follow_symlinks=False)
        sandbox_open = os.fstat(sandbox_fd)
        if (sandbox_linked.st_dev, sandbox_linked.st_ino) != (sandbox_open.st_dev, sandbox_open.st_ino):
            raise IntegrityStop("sandbox-exec changed across spawn")
        os.unlink(worker_copy)
        os.fsync(staging_fd)
        _write_fd_all_before(gate_write, b"G", deadline)
        os.close(gate_write)
        descriptors.remove(gate_write)
        _write_fd_all_before(input_write, native_input, deadline)
        os.close(input_write)
        descriptors.remove(input_write)
        native_output = _read_fd_all_before(output_read, deadline, 68)
        try:
            process.wait(timeout=max(0.001, deadline - time.monotonic()))
        except subprocess.TimeoutExpired as exc:
            raise IntegrityStop(f"native worker {role} timeout") from exc
        if process.returncode != 0:
            raise IntegrityStop(f"native worker {role} failed rc={process.returncode}")
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass
        if worker_copy is not None:
            try:
                os.unlink(worker_copy)
            except FileNotFoundError:
                pass
        if staging is not None:
            try:
                os.rmdir(staging)
            except FileNotFoundError:
                pass

    if native_output is None or process is None:
        raise IntegrityStop("native worker did not produce a closed result")
    if role == "IDENTITY_DISCOVERY" and len(native_output) == 32:
        artifact = {"query_id": item.query_id, "payload_sha256": native_output.hex()}
    elif role == "FROZEN_CANDIDATE_COMPARISON" and len(native_output) == 68:
        artifact = {
            "query_id": item["query_id"],
            "identity_payload_sha256": native_output[:32].hex(),
            "candidate_count": int.from_bytes(native_output[32:36], "big"),
            "candidates_sha256": native_output[36:].hex(),
        }
    elif role == "CAPABILITY_PROBE" and len(native_output) == 1:
        mask = native_output[0]
        artifact = {
            "network": bool(mask & 1), "fork": bool(mask & 2),
            "exec": bool(mask & 4), "local_read": bool(mask & 8),
            "local_write": bool(mask & 16), "reexec": bool(mask & 32),
        }
    else:
        raise IntegrityStop("native worker response shape mismatch")
    return {"schema_version": WORKER_SCHEMA, "role": role, "pid": process.pid, "artifact": artifact}


@dataclass(frozen=True)
class SyntheticRunResult:
    comparison: ComparisonArtifact
    journal_head_sha256: str
    journal_jsonl_sha256: str
    identity_worker_pid: int
    comparison_worker_pid: int


class SyntheticOfflineLauncher:
    """Sequential launcher for the proof run; never performs external I/O."""

    def run(
        self,
        *,
        output_root: Path,
        identity_input: IdentityInput,
        comparison_input: ComparisonInput,
    ) -> SyntheticRunResult:
        if type(identity_input) is not IdentityInput or type(comparison_input) is not ComparisonInput:
            raise IntegrityStop("launcher inputs did not cross validated boundaries")
        if identity_input.query_id != comparison_input.query_id:
            raise IntegrityStop("launcher query mismatch")
        journal = AccessJournal(Path(output_root))
        journal.state_transition(
            phase="PREFLIGHT", target_state="IDENTITY_NETWORK_OPEN"
        )
        broker = OfflineBroker(journal)
        identity_response = _run_sandboxed_worker(
            "IDENTITY_DISCOVERY",
            {
                "identity_input": {
                    "crm_address": identity_input.crm_address,
                    "crm_name": identity_input.crm_name,
                    "crm_postcode": identity_input.crm_postcode,
                    "query_id": identity_input.query_id,
                },
                "schema_version": WORKER_SCHEMA,
            },
        )
        identity_payload = identity_response["artifact"]
        if set(identity_payload) != {"query_id", "payload_sha256"}:
            raise IntegrityStop("identity worker artifact schema mismatch")
        identity = IdentityArtifact(**identity_payload)
        broker.revoke()
        comparison_response = _run_sandboxed_worker(
            "FROZEN_CANDIDATE_COMPARISON",
            {
                "comparison_input": {
                    "candidate_sirets": list(comparison_input.candidate_sirets),
                    "query_id": comparison_input.query_id,
                },
                "identity_artifact": {
                    "payload_sha256": identity.payload_sha256,
                    "query_id": identity.query_id,
                },
                "network_state": broker.state,
                "schema_version": WORKER_SCHEMA,
            },
            broker=broker,
        )
        comparison_payload = comparison_response["artifact"]
        if set(comparison_payload) != {
            "query_id", "identity_payload_sha256", "candidate_count", "candidates_sha256"
        }:
            raise IntegrityStop("comparison worker artifact schema mismatch")
        comparison = ComparisonArtifact(**comparison_payload)
        if comparison.identity_payload_sha256 != identity.payload_sha256:
            raise IntegrityStop("comparison did not bind the sealed identity artifact")
        if identity_response["pid"] == comparison_response["pid"]:
            raise IntegrityStop("workers did not execute in distinct processes")
        head = journal.verify_complete()
        projection_hash = journal.project_jsonl(Path(output_root) / "access_journal.jsonl")
        journal.close()
        return SyntheticRunResult(
            comparison,
            head,
            projection_hash,
            identity_response["pid"],
            comparison_response["pid"],
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-offline", action="store_true", required=True)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args(argv)
    result = SyntheticOfflineLauncher().run(
        output_root=args.output_root,
        identity_input=IdentityInput("synthetic-q1", "Société Exemple", "1 rue Exemple", "75001"),
        comparison_input=ComparisonInput("synthetic-q1", ("55210055400013",)),
    )
    print(canonical_json({
        "contract_commit": CONTRACT_COMMIT,
        "journal_head_sha256": result.journal_head_sha256,
        "journal_jsonl_sha256": result.journal_jsonl_sha256,
        "mode": "SYNTHETIC_OFFLINE_ONLY",
    }).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
