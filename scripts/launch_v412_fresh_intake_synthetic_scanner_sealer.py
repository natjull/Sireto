#!/usr/bin/env python3
"""Launch the single authoritative V4.12 synthetic S0 worker.

This program deliberately has no public arguments.  Its sole trust anchor is
the launch-authorization manifest committed at ``AUTHORIZATION_RELATIVE_PATH``.
Every failure is fail-closed and, after the pre-spawn claim exists, a run is
never replayed.
"""

from __future__ import annotations

import datetime as dt
import ctypes
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import plistlib
import re
import selectors
import socket
import stat
import struct
import subprocess
import sys
import time
import uuid
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUTHORIZATION_RELATIVE_PATH = Path(
    "config/v4_12_fresh_s0_launch_authorization.json"
)
AUTHORIZATION_PATH = REPOSITORY_ROOT / AUTHORIZATION_RELATIVE_PATH
PLAN_RELATIVE_PATH = Path("config/v4_12_fresh_s0_authoritative_run_plan.json")
PLAN_PATH = REPOSITORY_ROOT / PLAN_RELATIVE_PATH
SANDBOX_EXEC_PATH = Path("/usr/bin/sandbox-exec")

HEX64 = re.compile(r"^[0-9a-f]{64}$")
COMMIT40 = re.compile(r"^[0-9a-f]{40}$")
OPAQUE_ID = re.compile(r"^[a-p]{64}$")
RFC3339 = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")
MAX_CAPTURE_BYTES = 1_048_576

AUTHORIZATION_FIELDS = (
    "schema_version",
    "implementation_commit",
    "authoritative_plan_sha256",
    "authoritative_contract_sha256",
    "execution_lock_absolute_path",
    "execution_lock_sha256",
    "synthetic_run_id",
    "attempt_id",
    "authorization_status",
)
LOCK_FIELDS = (
    "schema_version",
    "purpose",
    "status",
    "implementation_commit",
    "implementation_blobs",
    "core",
    "runtime",
    "r2_smoke",
    "read_fds",
    "paths",
    "sandbox",
    "policy",
    "synthetic_run_id",
    "attempt_id",
    "logical_time_utc",
)
CLAIM_FIELDS = (
    "schema_version",
    "implementation_commit",
    "authorization_manifest_sha256",
    "execution_lock_sha256",
    "synthetic_run_id",
    "attempt_id",
    "claimed_at_utc",
    "claim_status",
)
IDENTITY_FIELDS = (
    "device",
    "inode",
    "size_bytes",
    "mtime_ns",
    "ctime_ns",
    "uid",
    "volume_uuid",
    "link_count",
    "mode",
)
OBSERVATION_FIELDS = (
    "role",
    "identity",
    "size_bytes",
    "sha256",
    "read_to_eof",
    "position_restored",
)
OUTPUT_AUTHORITY_FIELDS = (
    "sealed_input_payload_manifest_sha256",
    "sealed_input_seal_sha256",
    "terminal_tree_kind",
    "terminal_tree_payload_manifest_sha256",
    "terminal_tree_seal_sha256",
    "journal_generation",
    "journal_generation_manifest_sha256",
    "journal_head_event_sha256",
)
STABILITY_FIELDS = (
    "same_worker_process",
    "same_five_payload_fds",
    "monotonic_elapsed_seconds",
)
RECEIPT_FIELDS = (
    "schema_version",
    "phase",
    "reason_code",
    "authorization_manifest_sha256",
    "execution_lock_path",
    "execution_lock_sha256",
    "implementation_commit",
    "implementation_blob_hashes",
    "synthetic_run_id",
    "attempt_id",
    "claim_sha256",
    "lease_path",
    "lease_held_for_spawn",
    "runtime",
    "sandbox_profile_sha256",
    "effective_sandbox_profile_sha256",
    "parent_before_observations",
    "worker_receipt",
    "parent_after_observations",
    "stability",
    "canaries",
    "output_authority",
    "macos_limitations",
    "terminal_result",
    "verdict",
    "started_at_utc",
    "finished_at_utc",
)
WORKER_PAYLOAD_ROLES = (
    "CONTROL_MANIFEST",
    "COLLECTION_MANIFEST",
    "SOURCE_MANIFEST",
    "CRM_SAFE_CSV",
    "EVIDENCE_MANIFEST",
    "EVIDENCE_PARQUET",
)
LOCK_INPUT_ROLES = WORKER_PAYLOAD_ROLES + (
    "HOST_PYTHON_FRAMEWORK",
    "PRIVATE_RUNTIME_MANIFEST",
    "CANARY_MANIFEST",
)
PARENT_RETAINED_ROLES = (
    "EXECUTION_LOCK",
    "AUTHORIZATION",
    "WORKER",
    "SANDBOX_PROFILE",
    "HOST_PYTHON_FRAMEWORK",
    "PRIVATE_RUNTIME_MANIFEST",
    *WORKER_PAYLOAD_ROLES,
    "WORKER_SPEC",
    "CANARY_MANIFEST",
)
WRITE_DIRECTORY_ROLES = ("SEALED", "SCAN", "QUARANTINE", "AUDIT", "TMP")


class LauncherStop(RuntimeError):
    """A closed launch failure with a contract phase and reason."""

    def __init__(self, phase: str, reason_code: str, detail: str):
        super().__init__(f"STOP [{phase}/{reason_code}] {detail}")
        self.phase = phase
        self.reason_code = reason_code
        self.detail = detail


def _stop(phase: str, reason: str, detail: str) -> None:
    raise LauncherStop(phase, reason, detail)


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def canonical_json(value: Any, *, final_lf: bool = True) -> bytes:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return encoded + (b"\n" if final_lf else b"")


def decode_canonical_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8", "strict"),
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite number: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        _stop("AUTHORIZATION", "AUTHORIZATION_INVALID", f"{label}: {exc}")
    if type(value) is not dict or raw != canonical_json(value):
        _stop(
            "AUTHORIZATION",
            "AUTHORIZATION_INVALID",
            f"{label} is not canonical JSON",
        )
    return value


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _exact_keys(value: Mapping[str, Any], fields: Sequence[str], label: str) -> None:
    if type(value) is not dict or set(value) != set(fields):
        _stop(
            "AUTHORIZATION",
            "AUTHORIZATION_INVALID",
            f"{label} fields differ from contract",
        )


def _is_uint(value: Any) -> bool:
    return type(value) is int and value >= 0


def _require_hex(value: Any, label: str, *, commit: bool = False) -> str:
    pattern = COMMIT40 if commit else HEX64
    if type(value) is not str or pattern.fullmatch(value) is None:
        _stop("AUTHORIZATION", "AUTHORIZATION_INVALID", f"invalid {label}")
    return value


def _require_id(value: Any, label: str) -> str:
    if type(value) is not str or OPAQUE_ID.fullmatch(value) is None:
        _stop("AUTHORIZATION", "AUTHORIZATION_INVALID", f"invalid {label}")
    return value


def _require_rfc3339(value: Any, label: str) -> str:
    if type(value) is not str or RFC3339.fullmatch(value) is None:
        _stop("AUTHORIZATION", "AUTHORIZATION_INVALID", f"invalid {label}")
    try:
        dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        _stop("AUTHORIZATION", "AUTHORIZATION_INVALID", f"invalid {label}")
    return value


def _now_utc() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _normalized_absolute(value: Any, label: str) -> Path:
    if type(value) is not str or "\0" in value:
        _stop("AUTHORIZATION", "AUTHORIZATION_INVALID", f"invalid {label}")
    path = Path(value)
    if (
        not path.is_absolute()
        or value != os.path.normpath(value)
        or any(part in {"", ".", ".."} for part in path.parts[1:])
    ):
        _stop("AUTHORIZATION", "AUTHORIZATION_INVALID", f"unsafe {label}")
    return path


def _repo_relative(value: Any, label: str) -> Path:
    if type(value) is not str or not value or "\0" in value:
        _stop("AUTHORIZATION", "IMPLEMENTATION_INVALID", f"invalid {label}")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        _stop("AUTHORIZATION", "IMPLEMENTATION_INVALID", f"unsafe {label}")
    return path


def _git_loose_object(object_id: str) -> tuple[str, bytes]:
    if COMMIT40.fullmatch(object_id) is None:
        _stop("AUTHORIZATION", "IMPLEMENTATION_INVALID", "invalid Git object id")
    path = (
        REPOSITORY_ROOT
        / ".git"
        / "objects"
        / object_id[:2]
        / object_id[2:]
    )
    payload = _read_anchored_path(path, f"Git object {object_id}")
    try:
        decoded = zlib.decompress(payload)
    except zlib.error:
        _stop("AUTHORIZATION", "IMPLEMENTATION_INVALID", "invalid Git object")
    header, separator, body = decoded.partition(b"\0")
    if not separator:
        _stop("AUTHORIZATION", "IMPLEMENTATION_INVALID", "Git object header")
    try:
        kind_raw, size_raw = header.split(b" ", 1)
        size = int(size_raw)
        kind = kind_raw.decode("ascii", "strict")
    except (ValueError, UnicodeError):
        _stop("AUTHORIZATION", "IMPLEMENTATION_INVALID", "Git object metadata")
    if size != len(body) or hashlib.sha1(decoded).hexdigest() != object_id:
        _stop("AUTHORIZATION", "IMPLEMENTATION_INVALID", "Git object integrity")
    return kind, body


def _git_head() -> str:
    raw = _read_anchored_path(REPOSITORY_ROOT / ".git" / "HEAD", "Git HEAD")
    try:
        line = raw.decode("ascii", "strict").rstrip("\n")
    except UnicodeError:
        _stop("AUTHORIZATION", "IMPLEMENTATION_INVALID", "invalid Git HEAD")
    if line.startswith("ref: "):
        relative = _repo_relative(line[5:], "Git HEAD reference")
        if relative.parts[:2] != ("refs", "heads"):
            _stop("AUTHORIZATION", "IMPLEMENTATION_INVALID", "HEAD is not a branch")
        raw = _read_anchored_path(REPOSITORY_ROOT / ".git" / relative, "Git branch")
        try:
            line = raw.decode("ascii", "strict").strip()
        except UnicodeError:
            _stop("AUTHORIZATION", "IMPLEMENTATION_INVALID", "invalid Git branch")
    if COMMIT40.fullmatch(line) is None:
        _stop("AUTHORIZATION", "IMPLEMENTATION_INVALID", "invalid Git HEAD id")
    return line


def _git_commit(object_id: str) -> bytes:
    kind, body = _git_loose_object(object_id)
    if kind != "commit":
        _stop("AUTHORIZATION", "IMPLEMENTATION_INVALID", "Git object is not commit")
    return body


def _git_blob(revision: str, path: Path) -> bytes:
    commit_id = _git_head() if revision == "HEAD" else revision
    commit = _git_commit(commit_id)
    first = commit.splitlines()[0]
    if not first.startswith(b"tree ") or len(first) != 45:
        _stop("AUTHORIZATION", "IMPLEMENTATION_INVALID", "Git commit tree")
    current = first[5:].decode("ascii")
    relative = _repo_relative(path.as_posix(), "Git blob path")
    selected_mode = 0
    for index, component in enumerate(relative.parts):
        kind, tree = _git_loose_object(current)
        if kind != "tree":
            _stop("AUTHORIZATION", "IMPLEMENTATION_INVALID", "Git non-tree prefix")
        cursor = 0
        found: tuple[int, str] | None = None
        while cursor < len(tree):
            space = tree.find(b" ", cursor)
            nul = tree.find(b"\0", space + 1)
            if space < 0 or nul < 0 or nul + 21 > len(tree):
                _stop("AUTHORIZATION", "IMPLEMENTATION_INVALID", "malformed Git tree")
            mode = int(tree[cursor:space], 8)
            name = tree[space + 1 : nul].decode("utf-8", "strict")
            child_id = tree[nul + 1 : nul + 21].hex()
            cursor = nul + 21
            if name == component:
                if found is not None:
                    _stop("AUTHORIZATION", "IMPLEMENTATION_INVALID", "duplicate Git entry")
                found = (mode, child_id)
        if found is None:
            _stop("AUTHORIZATION", "IMPLEMENTATION_INVALID", f"Git path absent: {path}")
        selected_mode, current = found
        if index < len(relative.parts) - 1 and selected_mode != 0o40000:
            _stop("AUTHORIZATION", "IMPLEMENTATION_INVALID", "Git prefix not tree")
    kind, blob = _git_loose_object(current)
    if kind != "blob" or selected_mode not in {0o100644, 0o100755}:
        _stop("AUTHORIZATION", "IMPLEMENTATION_INVALID", "Git target not blob")
    return blob


def _git_is_ancestor(ancestor: str, descendant: str) -> bool:
    pending = [descendant]
    visited: set[str] = set()
    while pending:
        current = pending.pop()
        if current == ancestor:
            return True
        if current in visited:
            continue
        visited.add(current)
        commit = _git_commit(current)
        for line in commit.splitlines()[1:]:
            if line.startswith(b"parent "):
                parent = line[7:].decode("ascii", "strict")
                if COMMIT40.fullmatch(parent) is None:
                    _stop("AUTHORIZATION", "IMPLEMENTATION_INVALID", "invalid Git parent")
                pending.append(parent)
            elif not line:
                break
    return False


def _read_regular_fd(
    fd: int,
    label: str,
    *,
    expected_uid: int | None = None,
) -> tuple[bytes, os.stat_result]:
    before = os.fstat(fd)
    owner_uid = os.getuid() if expected_uid is None else expected_uid
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_uid != owner_uid
        or stat.S_IMODE(before.st_mode) & 0o022
    ):
        _stop("PRESPAWN", "FD_INVALID", f"unsafe regular file: {label}")
    position = os.lseek(fd, 0, os.SEEK_CUR)
    os.lseek(fd, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    os.lseek(fd, position, os.SEEK_SET)
    after = os.fstat(fd)
    fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_uid",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(getattr(before, field) != getattr(after, field) for field in fields):
        _stop("PRESPAWN", "FD_INVALID", f"identity drift while reading {label}")
    raw = b"".join(chunks)
    if len(raw) != before.st_size:
        _stop("PRESPAWN", "FD_INVALID", f"short read: {label}")
    return raw, before


def _open_anchored(path: Path, *, directory: bool = False, writable: bool = False) -> int:
    if not path.is_absolute() or path == Path("/") or ".." in path.parts:
        _stop("PRESPAWN", "FD_INVALID", f"unsafe anchored path: {path}")
    dir_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    current = os.open("/", dir_flags)
    try:
        components = path.parts[1:]
        for index, component in enumerate(components):
            final = index == len(components) - 1
            flags = (
                (os.O_RDWR if final and writable else os.O_RDONLY)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | (getattr(os, "O_DIRECTORY", 0) if (not final or directory) else 0)
            )
            child = os.open(component, flags, dir_fd=current)
            os.close(current)
            current = child
        info = os.fstat(current)
        expected = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
        if not expected:
            _stop("PRESPAWN", "FD_INVALID", f"wrong object type: {path}")
        return current
    except BaseException:
        os.close(current)
        raise


def _read_anchored_path(path: Path, label: str) -> bytes:
    fd = _open_anchored(path)
    try:
        raw, _ = _read_regular_fd(fd, label)
        return raw
    finally:
        os.close(fd)


def _anchored_entry_exists(path: Path) -> bool:
    parent_fd = _open_anchored(path.parent, directory=True)
    try:
        try:
            os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        return True
    finally:
        os.close(parent_fd)


def _full_sync(fd: int) -> None:
    os.fsync(fd)
    full = getattr(fcntl, "F_FULLFSYNC", None)
    if platform.system() == "Darwin":
        if full is None:
            _stop("RECEIPT", "RECEIPT_CONFLICT", "F_FULLFSYNC unavailable")
        try:
            fcntl.fcntl(fd, full)
        except OSError as exc:
            _stop("RECEIPT", "RECEIPT_CONFLICT", f"F_FULLFSYNC failed: {exc}")


def _write_exclusive(path: Path, value: Mapping[str, Any], *, mode: int = 0o400) -> bytes:
    raw = canonical_json(value)
    parent_fd = _open_anchored(path.parent, directory=True)
    fd = -1
    previous_umask = os.umask(0o077)
    try:
        fd = os.open(
            path.name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            mode,
            dir_fd=parent_fd,
        )
        if os.write(fd, raw) != len(raw):
            _stop("RECEIPT", "RECEIPT_CONFLICT", f"short write: {path}")
        os.fchmod(fd, mode)
        _full_sync(fd)
        _full_sync(parent_fd)
        return raw
    except FileExistsError:
        _stop("RECEIPT", "RECEIPT_CONFLICT", f"immutable file exists: {path}")
    finally:
        os.umask(previous_umask)
        if fd >= 0:
            os.close(fd)
        os.close(parent_fd)


@dataclass
class OpenAuthority:
    role: str
    path: Path
    fd: int
    expected_sha256: str
    expected_size: int
    expected_identity: Mapping[str, Any] | None


@dataclass
class WorkerExecution:
    pid: int
    exit_code: int | None
    signal_number: int | None
    stdout: bytes
    stderr: bytes
    ready: Mapping[str, Any] | None
    result: Mapping[str, Any] | None


class _AttrList(ctypes.Structure):
    _fields_ = [
        ("bitmapcount", ctypes.c_ushort),
        ("reserved", ctypes.c_uint16),
        ("commonattr", ctypes.c_uint32),
        ("volattr", ctypes.c_uint32),
        ("dirattr", ctypes.c_uint32),
        ("fileattr", ctypes.c_uint32),
        ("forkattr", ctypes.c_uint32),
    ]


class VolumeUUIDResolver:
    def for_fd(self, fd: int) -> str:
        if platform.system() != "Darwin":
            _stop("PRESPAWN", "RUNTIME_INVALID", "volume UUID requires macOS")
        before = os.fstat(fd)
        attributes = _AttrList(
            5, 0, 0, 0x80000000 | 0x00040000, 0, 0, 0
        )
        buffer = ctypes.create_string_buffer(4 + 16)
        libc = ctypes.CDLL(None, use_errno=True)
        function = getattr(libc, "fgetattrlist", None)
        if function is None:
            _stop("PRESPAWN", "RUNTIME_INVALID", "fgetattrlist unavailable")
        function.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(_AttrList),
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_ulong,
        ]
        function.restype = ctypes.c_int
        if function(
            fd,
            ctypes.byref(attributes),
            ctypes.byref(buffer),
            ctypes.sizeof(buffer),
            0,
        ):
            _stop("PRESPAWN", "RUNTIME_INVALID", "volume UUID lookup failed")
        if struct.unpack_from("=I", buffer.raw, 0)[0] < 20:
            _stop("PRESPAWN", "RUNTIME_INVALID", "short volume UUID")
        value = str(uuid.UUID(bytes=buffer.raw[4:20])).lower()
        after = os.fstat(fd)
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            _stop("PRESPAWN", "RUNTIME_INVALID", "volume anchor drift")
        # APFS firmlinks can expose the System and Data volumes with the same
        # st_dev while fgetattrlist correctly returns distinct volume UUIDs.
        # Caching by st_dev would therefore substitute one trust boundary for
        # another depending on lookup order.
        return value


def _identity(fd: int, volume_uuid: str) -> dict[str, Any]:
    info = os.fstat(fd)
    return {
        "device": info.st_dev,
        "inode": info.st_ino,
        "size_bytes": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
        "uid": info.st_uid,
        "volume_uuid": volume_uuid,
        "link_count": info.st_nlink,
        "mode": f"{stat.S_IMODE(info.st_mode):04o}",
    }


def _validate_identity(actual: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    if type(expected) is not dict or tuple(expected.keys()) != IDENTITY_FIELDS:
        # Canonical JSON ordering is lexical; logical field order cannot be
        # recovered, so accept exact key sets but never missing/extra keys.
        if type(expected) is not dict or set(expected) != set(IDENTITY_FIELDS):
            _stop("PRESPAWN", "FD_INVALID", f"invalid identity schema: {label}")
    if dict(actual) != dict(expected):
        _stop("PRESPAWN", "FD_INVALID", f"identity mismatch: {label}")


def observe(authority: OpenAuthority, resolver: VolumeUUIDResolver) -> dict[str, Any]:
    position = os.lseek(authority.fd, 0, os.SEEK_CUR)
    raw, info = _read_regular_fd(authority.fd, authority.role)
    restored = os.lseek(authority.fd, 0, os.SEEK_CUR) == position
    digest = sha256_bytes(raw)
    volume_uuid = resolver.for_fd(authority.fd)
    identity = _identity(authority.fd, volume_uuid)
    if (
        digest != authority.expected_sha256
        or info.st_size != authority.expected_size
        or not restored
    ):
        _stop("PRESPAWN", "FD_INVALID", f"authority mismatch: {authority.role}")
    if authority.expected_identity is not None:
        _validate_identity(identity, authority.expected_identity, authority.role)
    return {
        "role": authority.role,
        "identity": identity,
        "size_bytes": info.st_size,
        "sha256": digest,
        "read_to_eof": True,
        "position_restored": restored,
    }


def _load_plan() -> tuple[dict[str, Any], bytes]:
    raw = _git_blob("HEAD", PLAN_RELATIVE_PATH)
    disk = _read_anchored_path(PLAN_PATH, "authoritative plan")
    if raw != disk:
        _stop("AUTHORIZATION", "IMPLEMENTATION_INVALID", "plan differs from HEAD")
    plan = decode_canonical_json(raw, "authoritative plan")
    if (
        plan.get("status")
        != "PREREGISTERED_R2B_DO_NOT_IMPLEMENT_UNTIL_TWO_INDEPENDENT_AUDITS"
    ):
        _stop("AUTHORIZATION", "IMPLEMENTATION_INVALID", "unexpected plan status")
    return plan, raw


def _load_authorization(plan: Mapping[str, Any]) -> tuple[dict[str, Any], bytes]:
    fd = _open_anchored(AUTHORIZATION_PATH)
    try:
        raw, _ = _read_regular_fd(fd, "authorization manifest")
    finally:
        os.close(fd)
    if raw != _git_blob("HEAD", AUTHORIZATION_RELATIVE_PATH):
        _stop(
            "AUTHORIZATION",
            "AUTHORIZATION_INVALID",
            "authorization differs from HEAD blob",
        )
    value = decode_canonical_json(raw, "launch authorization")
    _exact_keys(value, AUTHORIZATION_FIELDS, "launch authorization")
    expected = plan["authorization"]
    if (
        value["schema_version"] != expected["schema_version"]
        or value["authorization_status"] != "AUTHORIZED_SYNTHETIC_S0"
    ):
        _stop("AUTHORIZATION", "AUTHORIZATION_INVALID", "authorization constants")
    _require_hex(value["implementation_commit"], "implementation_commit", commit=True)
    _require_hex(value["authoritative_plan_sha256"], "plan hash")
    _require_hex(value["authoritative_contract_sha256"], "contract hash")
    _require_hex(value["execution_lock_sha256"], "lock hash")
    _require_id(value["synthetic_run_id"], "synthetic_run_id")
    _require_id(value["attempt_id"], "attempt_id")
    _normalized_absolute(value["execution_lock_absolute_path"], "execution lock path")
    if (
        value["authoritative_plan_sha256"]
        != sha256_bytes(_read_anchored_path(PLAN_PATH, "authoritative plan"))
        or value["authoritative_contract_sha256"] != plan["contract"]["sha256"]
    ):
        _stop("AUTHORIZATION", "AUTHORIZATION_INVALID", "authoritative pins mismatch")
    return value, raw


def _load_lock(
    authorization: Mapping[str, Any], plan: Mapping[str, Any]
) -> tuple[dict[str, Any], bytes, int]:
    path = _normalized_absolute(
        authorization["execution_lock_absolute_path"], "execution lock path"
    )
    fd = _open_anchored(path)
    raw, _ = _read_regular_fd(fd, "execution lock")
    if sha256_bytes(raw) != authorization["execution_lock_sha256"]:
        os.close(fd)
        _stop("AUTHORIZATION", "LOCK_INVALID", "execution lock hash mismatch")
    lock = decode_canonical_json(raw, "execution lock")
    if set(lock) != set(LOCK_FIELDS):
        os.close(fd)
        _stop("AUTHORIZATION", "LOCK_INVALID", "execution lock fields")
    if (
        lock["schema_version"] != plan["execution_lock"]["schema_version"]
        or lock["purpose"]
        != "SIRETO_V412_FRESH_SYNTHETIC_S0_R2_AUTHORITATIVE_RUN"
        or lock["status"] != "SEALED_AUTHORITY_READY_TO_AUTHORIZE"
        or lock["implementation_commit"] != authorization["implementation_commit"]
        or lock["synthetic_run_id"] != authorization["synthetic_run_id"]
        or lock["attempt_id"] != authorization["attempt_id"]
        or lock["core"] != plan["core"]
    ):
        os.close(fd)
        _stop("AUTHORIZATION", "LOCK_INVALID", "execution lock constants mismatch")
    _require_rfc3339(lock["logical_time_utc"], "logical time")
    if lock["policy"] != plan["lock_values"]["policy"]:
        os.close(fd)
        _stop("AUTHORIZATION", "LOCK_INVALID", "execution policy mismatch")
    return lock, raw, fd


def _validate_implementation(
    lock: Mapping[str, Any],
    authorization: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, str]:
    commit = authorization["implementation_commit"]
    head = _git_head()
    _git_commit(commit)
    if not _git_is_ancestor(commit, head):
        _stop("AUTHORIZATION", "IMPLEMENTATION_INVALID", "commit order invalid")
    records = lock["implementation_blobs"]
    expected_roles = tuple(plan["execution_lock"]["implementation_blob_roles"])
    if (
        type(records) is not list
        or tuple(record.get("role") for record in records) != expected_roles
    ):
        _stop("AUTHORIZATION", "IMPLEMENTATION_INVALID", "implementation blob roles")
    hashes: dict[str, str] = {}
    for record in records:
        if type(record) is not dict or set(record) != {
            "role", "path", "size_bytes", "sha256", "mode"
        }:
            _stop("AUTHORIZATION", "IMPLEMENTATION_INVALID", "source blob schema")
        path = _repo_relative(record["path"], f"{record.get('role')} path")
        blob = _git_blob(commit, path)
        head_blob = _git_blob(head, path)
        if (
            not _is_uint(record["size_bytes"])
            or len(blob) != record["size_bytes"]
            or sha256_bytes(blob) != record["sha256"]
            or not HEX64.fullmatch(record["sha256"])
            or head_blob != blob
        ):
            _stop("AUTHORIZATION", "IMPLEMENTATION_INVALID", f"blob mismatch: {path}")
        disk = _read_anchored_path(REPOSITORY_ROOT / path, f"closed source {path}")
        if disk != blob:
            _stop("AUTHORIZATION", "IMPLEMENTATION_INVALID", f"dirty closed path: {path}")
        hashes[record["role"]] = record["sha256"]
    for pin in plan["core"]["pins"].values():
        path = _repo_relative(pin["path"], "core pin path")
        blob = _git_blob(plan["core"]["git_commit"], path)
        if (
            sha256_bytes(blob) != pin["sha256"]
            or _read_anchored_path(REPOSITORY_ROOT / path, f"core source {path}")
            != blob
        ):
            _stop("AUTHORIZATION", "IMPLEMENTATION_INVALID", f"core pin mismatch: {path}")
    return hashes


def _validate_runtime(lock: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    runtime = lock["runtime"]
    if type(runtime) is not dict or set(runtime) != {
        "system", "python_version", "pyarrow_version", "sandbox_exec", "private_runtime_manifest"
    }:
        _stop("PRESPAWN", "RUNTIME_INVALID", "runtime schema")
    system = runtime["system"]
    system_version_fd = _open_anchored(
        Path("/System/Library/CoreServices/SystemVersion.plist")
    )
    try:
        system_version_raw, _ = _read_regular_fd(
            system_version_fd,
            "macOS system version",
            expected_uid=0,
        )
    finally:
        os.close(system_version_fd)
    try:
        system_version = plistlib.loads(system_version_raw)
    except (plistlib.InvalidFileException, ValueError, TypeError):
        _stop("PRESPAWN", "RUNTIME_INVALID", "invalid macOS version authority")
    if (
        type(system) is not dict
        or set(system) != {"name", "product_version", "build_version", "kernel_release", "machine", "uid", "volumes"}
        or system["name"] != "macOS"
        or system["machine"] != "arm64"
        or system["uid"] != os.getuid()
        or platform.system() != "Darwin"
        or platform.machine() != "arm64"
        or system["kernel_release"] != platform.release()
        or not isinstance(system_version, dict)
        or system["product_version"] != system_version.get("ProductVersion")
        or system["build_version"] != system_version.get("ProductBuildVersion")
        or runtime["python_version"] != platform.python_version()
        or runtime["pyarrow_version"] != importlib.metadata.version("pyarrow")
    ):
        _stop("PRESPAWN", "RUNTIME_INVALID", "system runtime mismatch")
    sandbox = runtime["sandbox_exec"]
    if (
        type(sandbox) is not dict
        or sandbox.get("role") != "SANDBOX_EXEC"
        or sandbox.get("path") != os.fspath(SANDBOX_EXEC_PATH)
    ):
        _stop("PRESPAWN", "RUNTIME_INVALID", "sandbox-exec record")
    fd = _open_anchored(SANDBOX_EXEC_PATH)
    try:
        raw, info = _read_regular_fd(fd, "sandbox-exec", expected_uid=0)
    finally:
        os.close(fd)
    if (
        sha256_bytes(raw) != sandbox.get("sha256")
        or info.st_size != sandbox.get("size_bytes")
        or info.st_uid != 0
        or stat.S_IMODE(info.st_mode) & 0o022
    ):
        _stop("PRESPAWN", "RUNTIME_INVALID", "sandbox-exec identity")
    manifest = runtime["private_runtime_manifest"]
    schema = plan["schema_definitions"]["private_runtime_manifest"]
    if type(manifest) is not dict or set(manifest) != set(schema["exact_fields"]):
        _stop("PRESPAWN", "RUNTIME_INVALID", "private runtime manifest schema")
    records = manifest["records"]
    if (
        type(records) is not list
        or not records
        or manifest["record_count"] != len(records)
        or manifest["implementation_commit"] != lock["implementation_commit"]
    ):
        _stop("PRESPAWN", "RUNTIME_INVALID", "private runtime manifest values")
    projection = {
        "schema_version": manifest["schema_version"],
        "implementation_commit": manifest["implementation_commit"],
        "record_count": manifest["record_count"],
        "records": records,
    }
    if sha256_bytes(canonical_json(projection, final_lf=False)) != manifest["dependency_closure_sha256"]:
        _stop("PRESPAWN", "RUNTIME_INVALID", "private runtime closure hash")
    relative_paths: list[str] = []
    for record in records:
        if type(record) is not dict or set(record) != {
            "role", "source_path", "private_relative_path", "size_bytes", "sha256", "mode"
        }:
            _stop("PRESPAWN", "RUNTIME_INVALID", "private runtime record schema")
        relative = _repo_relative(record["private_relative_path"], "private runtime path")
        relative_paths.append(relative.as_posix())
        _require_hex(record["sha256"], "private runtime hash")
        if record["mode"] not in {"0400", "0500"} or not _is_uint(record["size_bytes"]):
            _stop("PRESPAWN", "RUNTIME_INVALID", "private runtime record values")
    if relative_paths != sorted(relative_paths, key=lambda item: item.encode("utf-8")) or len(set(relative_paths)) != len(relative_paths):
        _stop("PRESPAWN", "RUNTIME_INVALID", "private runtime order/uniqueness")


def _substitute_paths(lock: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Path]:
    run_id = lock["synthetic_run_id"]
    attempt_id = lock["attempt_id"]
    allowed_root = Path(plan["paths"]["allowed_root"])
    replacements = {
        "<allowed_root>": os.fspath(allowed_root),
        "<synthetic_run_id>": run_id,
        "<attempt_id>": attempt_id,
    }
    expected: dict[str, str] = {}
    for key, template in plan["lock_values"]["paths"].items():
        value = template
        for token, replacement in replacements.items():
            value = value.replace(token, replacement)
        expected[key] = value
    if lock["paths"] != expected:
        _stop("AUTHORIZATION", "LOCK_INVALID", "lock paths mismatch")
    return {key: _normalized_absolute(value, f"path {key}") for key, value in expected.items()}


def _input_records(lock: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    records = lock["read_fds"]
    if (
        type(records) is not list
        or tuple(record.get("role") for record in records) != LOCK_INPUT_ROLES
    ):
        _stop("AUTHORIZATION", "LOCK_INVALID", "lock input role order")
    result: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if type(record) is not dict or set(record) != {
            "role", "absolute_path", "identity", "volume_uuid", "size_bytes", "sha256"
        }:
            _stop("AUTHORIZATION", "LOCK_INVALID", "lock input schema")
        if record["volume_uuid"] != record["identity"].get("volume_uuid"):
            _stop("AUTHORIZATION", "LOCK_INVALID", "lock input volume mismatch")
        result[record["role"]] = record
    return result


def _validate_lock_sandbox(
    lock: Mapping[str, Any],
    plan: Mapping[str, Any],
    implementation_hashes: Mapping[str, str],
) -> None:
    profile_record = _private_record(lock, "EFFECTIVE_SANDBOX_PROFILE")
    expected = dict(plan["lock_values"]["sandbox"])
    expected["template_profile_sha256"] = implementation_hashes["SANDBOX_PROFILE"]
    expected["effective_profile_sha256"] = profile_record["sha256"]
    if lock["sandbox"] != expected:
        _stop("AUTHORIZATION", "LOCK_INVALID", "sandbox authority mismatch")


def _validate_canary_manifest(
    raw: bytes, lock: Mapping[str, Any], plan: Mapping[str, Any]
) -> None:
    value = decode_canonical_json(raw, "canary manifest")
    fields = set(plan["schema_definitions"]["canary_manifest"]["exact_fields"])
    records = value.get("ordered_records") if type(value) is dict else None
    codes = plan["canary_matrix"]["runtime_codes_exact_order"]
    if (
        type(value) is not dict
        or set(value) != fields
        or value["schema_version"] != "sireto-v4.12-fresh-s0-canary-manifest-1"
        or value["synthetic_run_id"] != lock["synthetic_run_id"]
        or type(records) is not list
        or value["record_count"] != len(records)
        or [record.get("code") for record in records if type(record) is dict] != codes
        or value["records_sha256"]
        != sha256_bytes(canonical_json(records, final_lf=False))
    ):
        _stop("PRESPAWN", "SANDBOX_EXPECTATION_FAILED", "canary manifest mismatch")
    exact = {
        "code",
        "kind",
        "absolute_path_or_capability",
        "identity",
        "size_bytes",
        "sha256",
    }
    allowed_root = plan["paths"]["allowed_root"]
    target_templates = plan["canary_matrix"]["synthetic_target_by_runtime_code"]
    for record in records:
        if (
            type(record) is not dict
            or set(record) != exact
            or record["kind"]
            not in {
                "EXISTING_FILE",
                "EXISTING_DIRECTORY",
                "EXPECTED_ABSENT",
                "NETWORK_CAPABILITY",
            }
        ):
            _stop("PRESPAWN", "SANDBOX_EXPECTATION_FAILED", "canary record schema")
        expected_target = target_templates[record["code"]].replace(
            "<allowed_root>", allowed_root
        ).replace("<synthetic_run_id>", lock["synthetic_run_id"])
        if record["absolute_path_or_capability"] != expected_target:
            _stop("PRESPAWN", "SANDBOX_EXPECTATION_FAILED", "canary target mismatch")
        kind = record["kind"]
        nullable = (record["identity"], record["size_bytes"], record["sha256"])
        if kind in {"EXPECTED_ABSENT", "NETWORK_CAPABILITY"}:
            if nullable != (None, None, None):
                _stop(
                    "PRESPAWN",
                    "SANDBOX_EXPECTATION_FAILED",
                    "capability/absence canary carried file authority",
                )
        elif kind == "EXISTING_DIRECTORY":
            if nullable != (None, None, None):
                _stop(
                    "PRESPAWN",
                    "SANDBOX_EXPECTATION_FAILED",
                    "directory canary authority invalid",
                )
            directory_path = Path(expected_target)
            directory_fd: int | None = None
            try:
                directory_fd = _open_anchored(directory_path, directory=True)
                directory_info = os.fstat(directory_fd)
                run_volume = lock["runtime"]["system"]["volumes"]["run"]
                if (
                    type(run_volume) is not dict
                    or set(run_volume) != {"device", "volume_uuid"}
                    or directory_info.st_uid != os.getuid()
                    or directory_info.st_dev != run_volume["device"]
                    or stat.S_IMODE(directory_info.st_mode) & 0o022
                ):
                    raise ValueError("unsafe directory canary")
            except (KeyError, OSError, LauncherStop, ValueError):
                _stop(
                    "PRESPAWN",
                    "SANDBOX_EXPECTATION_FAILED",
                    "directory canary target invalid",
                )
            finally:
                if directory_fd is not None:
                    os.close(directory_fd)
        elif (
            type(record["identity"]) is not dict
            or set(record["identity"]) != set(IDENTITY_FIELDS)
            or not _is_uint(record["size_bytes"])
            or type(record["sha256"]) is not str
            or HEX64.fullmatch(record["sha256"]) is None
        ):
            _stop(
                "PRESPAWN",
                "SANDBOX_EXPECTATION_FAILED",
                "file canary authority invalid",
            )


def _open_authority(
    role: str,
    path: Path,
    expected_hash: str,
    expected_size: int,
    expected_identity: Mapping[str, Any] | None,
) -> OpenAuthority:
    return OpenAuthority(
        role=role,
        path=path,
        fd=_open_anchored(path),
        expected_sha256=expected_hash,
        expected_size=expected_size,
        expected_identity=expected_identity,
    )


def _private_record(
    lock: Mapping[str, Any], role: str
) -> Mapping[str, Any]:
    matches = [
        record
        for record in lock["runtime"]["private_runtime_manifest"]["records"]
        if record["role"] == role
    ]
    if len(matches) != 1:
        _stop("PRESPAWN", "RUNTIME_INVALID", f"private role cardinality: {role}")
    return matches[0]


def _runtime_path(lock: Mapping[str, Any], plan: Mapping[str, Any], record: Mapping[str, Any]) -> Path:
    root = Path(plan["paths"]["allowed_root"]) / "runtime" / lock["synthetic_run_id"]
    relative = _repo_relative(record["private_relative_path"], "private record path")
    return root / relative


def _r2b_runtime_boundary(plan: Mapping[str, Any]) -> Mapping[str, Any]:
    boundary = plan["r2_successor"]["runtime_boundary_amendment"]
    if (
        type(boundary) is not dict
        or boundary.get("status") != "PREREGISTERED_R2B_RUNTIME_BOUNDARY"
        or boundary.get("dyld_environment_forbidden")
        != ["DYLD_LIBRARY_PATH", "DYLD_FRAMEWORK_PATH", "DYLD_ROOT_PATH"]
    ):
        _stop("PRESPAWN", "RUNTIME_INVALID", "R2-B runtime boundary")
    return boundary


def _private_python_path(
    lock: Mapping[str, Any], plan: Mapping[str, Any]
) -> Path:
    boundary = _r2b_runtime_boundary(plan)
    expected = boundary["private_python_helper"]
    record = _private_record(lock, "PYTHON_EXECUTABLE")
    source = expected["source_path"]
    if (
        type(expected) is not dict
        or record["source_path"] != source
        or record["sha256"] != expected["sha256"]
        or source == boundary["private_python_stub_forbidden"]
        or record["private_relative_path"]
        != f"rootfs/{source.removeprefix('/')}"
    ):
        _stop("PRESPAWN", "RUNTIME_INVALID", "private Python.app helper authority")
    return _runtime_path(lock, plan, record)


def _host_python_framework(
    lock: Mapping[str, Any], plan: Mapping[str, Any]
) -> Mapping[str, Any]:
    expected = _r2b_runtime_boundary(plan)["host_python_framework"]
    records = [
        record
        for record in lock["read_fds"]
        if record.get("role") == "HOST_PYTHON_FRAMEWORK"
    ]
    if (
        type(expected) is not dict
        or len(records) != 1
        or records[0].get("absolute_path") != expected.get("path")
        or records[0].get("sha256") != expected.get("sha256")
        or expected.get("retained_parent_authority") is not True
        or expected.get("sandbox_read_rule")
        != "LITERAL_ONLY_NO_OPT_OR_CELLAR_SUBPATH"
    ):
        _stop("PRESPAWN", "RUNTIME_INVALID", "host Python framework authority")
    return records[0]


def _macho_dylib_load_names(raw: bytes) -> tuple[str, ...]:
    # R2-B pins a thin little-endian arm64 MH_EXECUTE.  Parsing it in-process
    # keeps the authoritative launcher at exactly one child process.
    if len(raw) < 32:
        _stop("PRESPAWN", "RUNTIME_INVALID", "short Mach-O helper")
    try:
        (
            magic,
            cpu_type,
            _cpu_subtype,
            file_type,
            command_count,
            command_bytes,
            _flags,
            _reserved,
        ) = struct.unpack_from("<IiiIIIII", raw, 0)
    except struct.error:
        _stop("PRESPAWN", "RUNTIME_INVALID", "invalid Mach-O header")
    if (
        magic != 0xFEEDFACF
        or cpu_type != 0x0100000C
        or file_type != 2
        or command_count > 4096
        or command_bytes > len(raw) - 32
    ):
        _stop("PRESPAWN", "RUNTIME_INVALID", "unexpected Mach-O helper")
    cursor = 32
    end = cursor + command_bytes
    names: list[str] = []
    dylib_commands = {0xC, 0x18, 0x1F, 0x23}
    for _index in range(command_count):
        if cursor + 8 > end:
            _stop("PRESPAWN", "RUNTIME_INVALID", "truncated Mach-O command")
        command, command_size = struct.unpack_from("<II", raw, cursor)
        if command_size < 8 or command_size % 4 or cursor + command_size > end:
            _stop("PRESPAWN", "RUNTIME_INVALID", "invalid Mach-O command size")
        if (command & 0x7FFFFFFF) in dylib_commands:
            if command_size < 24:
                _stop("PRESPAWN", "RUNTIME_INVALID", "short dylib command")
            name_offset = struct.unpack_from("<I", raw, cursor + 8)[0]
            if name_offset < 24 or name_offset >= command_size:
                _stop("PRESPAWN", "RUNTIME_INVALID", "invalid dylib name offset")
            payload = raw[cursor + name_offset : cursor + command_size]
            nul = payload.find(b"\0")
            if nul < 1:
                _stop("PRESPAWN", "RUNTIME_INVALID", "invalid dylib name")
            try:
                name = payload[:nul].decode("utf-8", "strict")
            except UnicodeDecodeError:
                _stop("PRESPAWN", "RUNTIME_INVALID", "non-UTF8 dylib name")
            names.append(name)
        cursor += command_size
    if cursor != end:
        _stop("PRESPAWN", "RUNTIME_INVALID", "Mach-O command size mismatch")
    return tuple(names)


def _validate_python_helper_install_name(
    lock: Mapping[str, Any], plan: Mapping[str, Any], *, phase: str
) -> None:
    python_path = _private_python_path(lock, plan)
    fd = _open_anchored(python_path)
    try:
        raw, _ = _read_regular_fd(fd, "private Python.app helper")
    finally:
        os.close(fd)
    expected = _r2b_runtime_boundary(plan)["host_python_framework"]["path"]
    names = _macho_dylib_load_names(raw)
    non_system = tuple(
        name
        for name in names
        if not name.startswith("/System/") and not name.startswith("/usr/lib/")
    )
    if non_system != (expected,):
        _stop(phase, "RUNTIME_INVALID", "Python helper install-name authority")


def _verified_profile_text(authority: OpenAuthority) -> str:
    raw, _ = _read_regular_fd(authority.fd, "effective sandbox profile")
    if (
        len(raw) != authority.expected_size
        or sha256_bytes(raw) != authority.expected_sha256
        or not raw.endswith(b"\n")
        or raw.endswith(b"\n\n")
        or b"\r" in raw
        or b"\0" in raw
        or b"@@" in raw
    ):
        _stop("PRESPAWN", "RUNTIME_INVALID", "effective sandbox profile bytes")
    try:
        return raw.decode("utf-8", "strict")
    except UnicodeDecodeError:
        _stop("PRESPAWN", "RUNTIME_INVALID", "effective sandbox profile UTF-8")


def _enumerate_anchored_tree(root: Path) -> list[str]:
    root_fd = _open_anchored(root, directory=True)
    root_info = os.fstat(root_fd)
    if (
        root_info.st_uid != os.getuid()
        or stat.S_IMODE(root_info.st_mode) != 0o700
    ):
        os.close(root_fd)
        _stop("PRESPAWN", "RUNTIME_INVALID", "unsafe private runtime root")
    stack: list[tuple[int, tuple[str, ...]]] = [(root_fd, ())]
    files: list[str] = []
    try:
        while stack:
            current, prefix = stack.pop()
            try:
                names = sorted(os.listdir(current), key=lambda name: os.fsencode(name))
                for name in names:
                    if (
                        type(name) is not str
                        or not name
                        or name in {".", ".."}
                        or "/" in name
                        or "\0" in name
                    ):
                        _stop("PRESPAWN", "RUNTIME_INVALID", "unsafe runtime entry")
                    info = os.stat(name, dir_fd=current, follow_symlinks=False)
                    relative = (*prefix, name)
                    if stat.S_ISDIR(info.st_mode):
                        if (
                            info.st_uid != os.getuid()
                            or info.st_dev != root_info.st_dev
                            or stat.S_IMODE(info.st_mode) != 0o700
                        ):
                            _stop(
                                "PRESPAWN",
                                "RUNTIME_INVALID",
                                "unsafe private runtime directory",
                            )
                        child = os.open(
                            name,
                            os.O_RDONLY
                            | getattr(os, "O_DIRECTORY", 0)
                            | getattr(os, "O_NOFOLLOW", 0)
                            | getattr(os, "O_CLOEXEC", 0),
                            dir_fd=current,
                        )
                        stack.append((child, relative))
                    elif stat.S_ISREG(info.st_mode):
                        if (
                            info.st_nlink != 1
                            or info.st_uid != os.getuid()
                            or info.st_dev != root_info.st_dev
                            or stat.S_IMODE(info.st_mode) & 0o022
                        ):
                            _stop(
                                "PRESPAWN",
                                "RUNTIME_INVALID",
                                "hardlinked private runtime file",
                            )
                        files.append("/".join(relative))
                    else:
                        _stop(
                            "PRESPAWN",
                            "RUNTIME_INVALID",
                            "non-regular private runtime entry",
                        )
            finally:
                os.close(current)
    except BaseException:
        for fd, _prefix in stack:
            try:
                os.close(fd)
            except OSError:
                pass
        raise
    return sorted(files, key=lambda item: item.encode("utf-8"))


def _validate_private_runtime_tree(lock: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    root = Path(plan["paths"]["allowed_root"]) / "runtime" / lock["synthetic_run_id"]
    expected: dict[str, Mapping[str, Any]] = {
        record["private_relative_path"]: record
        for record in lock["runtime"]["private_runtime_manifest"]["records"]
    }
    manifest_inputs = [
        Path(record["absolute_path"])
        for record in lock["read_fds"]
        if record["role"] == "PRIVATE_RUNTIME_MANIFEST"
    ]
    if len(manifest_inputs) != 1:
        _stop("PRESPAWN", "RUNTIME_INVALID", "private manifest input cardinality")
    try:
        manifest_relative = manifest_inputs[0].relative_to(root).as_posix()
    except ValueError:
        _stop("PRESPAWN", "RUNTIME_INVALID", "private manifest is outside runtime root")
    if manifest_relative in expected:
        _stop("PRESPAWN", "RUNTIME_INVALID", "runtime manifest recursively inventories itself")
    actual = [
        relative
        for relative in _enumerate_anchored_tree(root)
        if relative != manifest_relative
    ]
    if actual != list(expected):
        _stop("PRESPAWN", "RUNTIME_INVALID", "private runtime missing files")
    for relative, record in expected.items():
        path = root / relative
        fd = _open_anchored(path)
        try:
            raw, info = _read_regular_fd(fd, f"private runtime {relative}")
        finally:
            os.close(fd)
        if (
            sha256_bytes(raw) != record["sha256"]
            or len(raw) != record["size_bytes"]
            or f"{stat.S_IMODE(info.st_mode):04o}" != record["mode"]
        ):
            _stop("PRESPAWN", "RUNTIME_INVALID", f"private runtime mismatch: {relative}")


def _revalidate_private_runtime_critical(
    lock: Mapping[str, Any],
    plan: Mapping[str, Any],
    authorities: Sequence[OpenAuthority],
    *,
    phase: str,
) -> None:
    _validate_private_runtime_tree(lock, plan)
    retained_by_role = {authority.role: authority for authority in authorities}
    for role, private_role in (
        ("WORKER", "WORKER"),
        ("SANDBOX_PROFILE", "EFFECTIVE_SANDBOX_PROFILE"),
    ):
        authority = retained_by_role.get(role)
        if authority is None:
            _stop(phase, "RUNTIME_INVALID", f"retained runtime FD absent: {role}")
        reopened = _open_anchored(authority.path)
        try:
            raw, reopened_info = _read_regular_fd(reopened, f"reopened {role}")
            held_info = os.fstat(authority.fd)
        finally:
            os.close(reopened)
        identity_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_uid",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if (
            any(
                getattr(reopened_info, field) != getattr(held_info, field)
                for field in identity_fields
            )
            or sha256_bytes(raw) != authority.expected_sha256
            or len(raw) != authority.expected_size
        ):
            _stop(phase, "RUNTIME_INVALID", f"path/FD drift: {role}")
    python_record = _private_record(lock, "PYTHON_EXECUTABLE")
    python_path = _runtime_path(lock, plan, python_record)
    reopened = _open_anchored(python_path)
    try:
        raw, info = _read_regular_fd(reopened, "private Python executable")
    finally:
        os.close(reopened)
    if (
        sha256_bytes(raw) != python_record["sha256"]
        or len(raw) != python_record["size_bytes"]
        or f"{stat.S_IMODE(info.st_mode):04o}" != python_record["mode"]
    ):
        _stop(phase, "RUNTIME_INVALID", "private Python path drift")
    host = retained_by_role.get("HOST_PYTHON_FRAMEWORK")
    expected_host = _host_python_framework(lock, plan)
    if host is None:
        _stop(phase, "RUNTIME_INVALID", "retained host framework FD absent")
    reopened = _open_anchored(Path(expected_host["absolute_path"]))
    try:
        host_raw, reopened_info = _read_regular_fd(
            reopened, "reopened host Python framework"
        )
        held_info = os.fstat(host.fd)
    finally:
        os.close(reopened)
    identity_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_uid",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if (
        any(
            getattr(reopened_info, field) != getattr(held_info, field)
            for field in identity_fields
        )
        or sha256_bytes(host_raw) != expected_host["sha256"]
        or len(host_raw) != expected_host["size_bytes"]
    ):
        _stop(phase, "RUNTIME_INVALID", "host Python framework path/FD drift")
    _validate_python_helper_install_name(lock, plan, phase=phase)


def _validate_precreated_directory(path: Path, *, allowed_root: Path) -> int:
    try:
        path.relative_to(allowed_root)
    except ValueError:
        _stop("PRESPAWN", "FD_INVALID", f"directory escapes allowed root: {path}")
    fd = _open_anchored(path, directory=True)
    info = os.fstat(fd)
    root_fd = _open_anchored(allowed_root, directory=True)
    try:
        root_info = os.fstat(root_fd)
    finally:
        os.close(root_fd)
    if (
        info.st_uid != os.getuid()
        or info.st_dev != root_info.st_dev
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        os.close(fd)
        _stop("PRESPAWN", "FD_INVALID", f"unsafe precreated directory: {path}")
    return fd


def _validate_volume_pins(
    lock: Mapping[str, Any],
    plan: Mapping[str, Any],
    resolver: VolumeUUIDResolver,
) -> None:
    volumes = lock["runtime"]["system"]["volumes"]
    if type(volumes) is not dict or set(volumes) != {
        "repository",
        "run",
        "system_tcb",
    }:
        _stop("PRESPAWN", "RUNTIME_INVALID", "runtime volume map")
    targets = {
        "repository": REPOSITORY_ROOT,
        "run": Path(plan["paths"]["allowed_root"]),
    }
    for role, path in targets.items():
        fd = _open_anchored(path, directory=True)
        try:
            info = os.fstat(fd)
            expected = volumes[role]
            if (
                type(expected) is not dict
                or set(expected) != {"device", "volume_uuid"}
                or expected["device"] != info.st_dev
                or expected["volume_uuid"] != resolver.for_fd(fd)
            ):
                _stop("PRESPAWN", "RUNTIME_INVALID", f"volume mismatch: {role}")
        finally:
            os.close(fd)
    root_fd = os.open(
        "/",
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        info = os.fstat(root_fd)
        expected = volumes["system_tcb"]
        if (
            type(expected) is not dict
            or set(expected) != {"device", "volume_uuid"}
            or expected["device"] != info.st_dev
            or expected["volume_uuid"] != resolver.for_fd(root_fd)
        ):
            _stop("PRESPAWN", "RUNTIME_INVALID", "volume mismatch: system_tcb")
    finally:
        os.close(root_fd)


def _claim_value(
    lock: Mapping[str, Any], auth_raw: bytes, lock_raw: bytes
) -> dict[str, Any]:
    return {
        "schema_version": "sireto-v4.12-fresh-s0-pre-spawn-claim-1",
        "implementation_commit": lock["implementation_commit"],
        "authorization_manifest_sha256": sha256_bytes(auth_raw),
        "execution_lock_sha256": sha256_bytes(lock_raw),
        "synthetic_run_id": lock["synthetic_run_id"],
        "attempt_id": lock["attempt_id"],
        "claimed_at_utc": _now_utc(),
        "claim_status": "CLAIMED_PRE_SPAWN",
    }


def _validate_existing_claim(
    path: Path, lock: Mapping[str, Any], auth_raw: bytes, lock_raw: bytes
) -> bytes:
    fd = _open_anchored(path)
    try:
        raw, _ = _read_regular_fd(fd, "claim")
    finally:
        os.close(fd)
    value = decode_canonical_json(raw, "claim")
    if (
        set(value) != set(CLAIM_FIELDS)
        or value.get("schema_version") != "sireto-v4.12-fresh-s0-pre-spawn-claim-1"
        or value.get("implementation_commit") != lock["implementation_commit"]
        or value.get("authorization_manifest_sha256") != sha256_bytes(auth_raw)
        or value.get("execution_lock_sha256") != sha256_bytes(lock_raw)
        or value.get("synthetic_run_id") != lock["synthetic_run_id"]
        or value.get("attempt_id") != lock["attempt_id"]
        or value.get("claim_status") != "CLAIMED_PRE_SPAWN"
    ):
        _stop("CLAIM", "CLAIM_CONFLICT", "existing claim invalid")
    _require_rfc3339(value.get("claimed_at_utc"), "claim timestamp")
    return raw


def _validate_observation_array(
    value: Any, *, allow_prefix: bool
) -> None:
    if type(value) is not list:
        _stop("RECEIPT", "RECEIPT_CONFLICT", "receipt observations not array")
    roles = [record.get("role") for record in value if type(record) is dict]
    expected = list(PARENT_RETAINED_ROLES)
    if (
        (allow_prefix and roles != expected[: len(roles)])
        or (not allow_prefix and roles != expected)
    ):
        _stop("RECEIPT", "RECEIPT_CONFLICT", "receipt observation roles")
    for record in value:
        if (
            type(record) is not dict
            or set(record) != set(OBSERVATION_FIELDS)
            or type(record["identity"]) is not dict
            or set(record["identity"]) != set(IDENTITY_FIELDS)
            or not _is_uint(record["size_bytes"])
            or type(record["sha256"]) is not str
            or HEX64.fullmatch(record["sha256"]) is None
            or record["read_to_eof"] is not True
            or record["position_restored"] is not True
        ):
            _stop("RECEIPT", "RECEIPT_CONFLICT", "receipt observation schema")


def _validate_existing_receipt(
    path: Path,
    *,
    lock: Mapping[str, Any],
    auth_raw: bytes,
    lock_raw: bytes,
    claim_raw: bytes,
    execution_lock_path: str,
    plan: Mapping[str, Any],
    implementation_hashes: Mapping[str, str],
) -> bytes:
    fd = _open_anchored(path)
    try:
        raw, _ = _read_regular_fd(fd, "existing launch receipt")
    finally:
        os.close(fd)
    value = decode_canonical_json(raw, "existing launch receipt")
    if (
        set(value) != set(RECEIPT_FIELDS)
        or value["schema_version"]
        != "sireto-v4.12-fresh-s0-authoritative-launch-receipt-2"
        or value["authorization_manifest_sha256"] != sha256_bytes(auth_raw)
        or value["execution_lock_path"] != execution_lock_path
        or value["execution_lock_sha256"] != sha256_bytes(lock_raw)
        or value["implementation_commit"] != lock["implementation_commit"]
        or value["implementation_blob_hashes"] != dict(implementation_hashes)
        or value["synthetic_run_id"] != lock["synthetic_run_id"]
        or value["attempt_id"] != lock["attempt_id"]
        or value["claim_sha256"] != sha256_bytes(claim_raw)
        or value["runtime"] != lock["runtime"]
        or value["macos_limitations"]
        != plan["macos_runtime_boundary"]["acknowledged_limitations"]
        or value["phase"] not in plan["enum_definitions"]["launch_phases"]
        or value["reason_code"] not in plan["enum_definitions"]["reason_codes"]
        or value["terminal_result"]
        not in plan["enum_definitions"]["scanner_terminal_results"]
        or value["verdict"] not in {"INGESTED", "QUARANTINED", "STOP"}
        or value["lease_path"]
        != os.fspath(
            Path(lock["paths"]["audit_parent"])
            / "leases"
            / f"{lock['attempt_id']}.lease"
        )
        or type(value["lease_held_for_spawn"]) is not bool
        or value["sandbox_profile_sha256"]
        != implementation_hashes["SANDBOX_PROFILE"]
        or value["effective_sandbox_profile_sha256"]
        != _private_record(lock, "EFFECTIVE_SANDBOX_PROFILE")["sha256"]
    ):
        _stop("RECEIPT", "RECEIPT_CONFLICT", "existing receipt incomplete")
    _require_rfc3339(value["started_at_utc"], "receipt start")
    _require_rfc3339(value["finished_at_utc"], "receipt finish")
    _validate_observation_array(
        value["parent_before_observations"], allow_prefix=value["verdict"] == "STOP"
    )
    if value["parent_after_observations"] is not None:
        _validate_observation_array(
            value["parent_after_observations"], allow_prefix=False
        )
    _validate_stability(value["stability"])
    _validate_output_authority(
        value["output_authority"], allow_null=value["verdict"] == "STOP"
    )
    worker_receipt = value["worker_receipt"]
    if worker_receipt is not None and (
        type(worker_receipt) is not dict
        or set(worker_receipt)
        != {
            "pid",
            "exit_code",
            "signal",
            "stdout_size_bytes",
            "stdout_sha256",
            "stderr_size_bytes",
            "stderr_sha256",
            "control_result",
        }
        or type(worker_receipt["pid"]) is not int
        or worker_receipt["pid"] <= 0
        or not _is_uint(worker_receipt["stdout_size_bytes"])
        or not _is_uint(worker_receipt["stderr_size_bytes"])
        or type(worker_receipt["stdout_sha256"]) is not str
        or HEX64.fullmatch(worker_receipt["stdout_sha256"]) is None
        or type(worker_receipt["stderr_sha256"]) is not str
        or HEX64.fullmatch(worker_receipt["stderr_sha256"]) is None
        or (
            worker_receipt["exit_code"] is not None
            and type(worker_receipt["exit_code"]) is not int
        )
        or (
            worker_receipt["signal"] is not None
            and (
                type(worker_receipt["signal"]) is not int
                or worker_receipt["signal"] <= 0
            )
        )
        or (
            worker_receipt["exit_code"] is not None
            and worker_receipt["signal"] is not None
        )
    ):
        _stop("RECEIPT", "RECEIPT_CONFLICT", "worker receipt schema")
    if value["verdict"] in {"INGESTED", "QUARANTINED"}:
        if (
            value["reason_code"] != "OK"
            or value["phase"] != "RECEIPT"
            or not _success_stability(value["stability"])
            or value["parent_after_observations"] is None
            or value["parent_before_observations"]
            != value["parent_after_observations"]
            or not value["canaries"]
            or worker_receipt is None
            or worker_receipt["control_result"] is None
            or value["lease_held_for_spawn"] is not True
        ):
            _stop("RECEIPT", "RECEIPT_CONFLICT", "successful receipt invariants")
        _validate_canaries(
            value["canaries"], plan, run_id=lock["synthetic_run_id"]
        )
        _validate_terminal(worker_receipt["control_result"], lock)
        expected_terminal = {
            "INGESTED": "INGESTED_SYNTHETIC_SCANNER_SEALER_V412",
            "QUARANTINED": "QUARANTINED_SYNTHETIC_SCANNER_SEALER_V412",
        }[value["verdict"]]
        if (
            value["terminal_result"] != expected_terminal
            or worker_receipt["exit_code"] != 0
            or worker_receipt["signal"] is not None
            or worker_receipt["control_result"]["terminal_result"]
            != expected_terminal
            or worker_receipt["control_result"]["output_authority"]
            != value["output_authority"]
            or worker_receipt["control_result"]["stability"]
            != value["stability"]
        ):
            _stop("RECEIPT", "RECEIPT_CONFLICT", "receipt result consistency")
        lock_paths = {
            key: Path(path_value) for key, path_value in lock["paths"].items()
        }
        if (
            _validate_output_paths(
                worker_receipt["control_result"], lock, lock_paths
            )
            != value["output_authority"]
        ):
            _stop("RECEIPT", "RECEIPT_CONFLICT", "idempotent output authority")
        audit_fd = _open_anchored(lock_paths["audit_worker"], directory=True)
        try:
            if _read_canary_proof(audit_fd, lock, plan) != value["canaries"]:
                _stop(
                    "RECEIPT",
                    "RECEIPT_CONFLICT",
                    "idempotent canary authority",
                )
        finally:
            os.close(audit_fd)
    elif (
        value["terminal_result"] != "STOP_SYNTHETIC_SCANNER_SEALER_V412"
        or value["reason_code"] == "OK"
    ):
        _stop("RECEIPT", "RECEIPT_CONFLICT", "STOP receipt invariants")
    else:
        control_result = (
            worker_receipt["control_result"]
            if worker_receipt is not None
            else None
        )
        has_output = any(
            item is not None for item in value["output_authority"].values()
        )
        if worker_receipt is not None and worker_receipt["control_result"] is not None:
            _validate_terminal(control_result, lock)
            if control_result["stability"] != value["stability"]:
                _stop(
                    "RECEIPT",
                    "RECEIPT_CONFLICT",
                    "STOP stability differs from observed control",
                )
        if value["reason_code"] == "STABILITY_FAILED":
            if (
                value["phase"] != "RECEIPT"
                or control_result is None
                or control_result["message_type"] != "RESULT"
                or worker_receipt["exit_code"] != 0
                or worker_receipt["signal"] is not None
                or not has_output
                or control_result["output_authority"]
                != value["output_authority"]
                or _success_stability(value["stability"])
            ):
                _stop(
                    "RECEIPT",
                    "RECEIPT_CONFLICT",
                    "stability downgrade consistency",
                )
        elif value["reason_code"] == "WORKER_CONTROLLED_STOP":
            if (
                control_result is None
                or control_result["message_type"] != "STOP"
                or worker_receipt["exit_code"] != 2
                or worker_receipt["signal"] is not None
                or has_output
            ):
                _stop(
                    "RECEIPT",
                    "RECEIPT_CONFLICT",
                    "controlled STOP result consistency",
                )
        elif has_output and (
            control_result is None
            or control_result["output_authority"] != value["output_authority"]
        ):
            _stop(
                "RECEIPT",
                "RECEIPT_CONFLICT",
                "parent STOP output lacks matching worker claim",
            )
        if has_output:
            lock_paths = {
                key: Path(path_value)
                for key, path_value in lock["paths"].items()
            }
            if (
                control_result is None
                or _validate_output_paths(control_result, lock, lock_paths)
                != value["output_authority"]
            ):
                _stop(
                    "RECEIPT",
                    "RECEIPT_CONFLICT",
                    "STOP output authority invalid",
                )
        if not value["canaries"]:
            audit_fd = _open_anchored(
                Path(lock["paths"]["audit_worker"]), directory=True
            )
            try:
                _require_absent_canary_proof(audit_fd)
            finally:
                os.close(audit_fd)
            return raw
        _validate_canaries(
            value["canaries"], plan, run_id=lock["synthetic_run_id"]
        )
        audit_fd = _open_anchored(
            Path(lock["paths"]["audit_worker"]), directory=True
        )
        try:
            if _read_canary_proof(audit_fd, lock, plan) != value["canaries"]:
                _stop("RECEIPT", "RECEIPT_CONFLICT", "STOP canary authority")
        finally:
            os.close(audit_fd)
    return raw


def _invalid_recovery_receipt(error: LauncherStop) -> None:
    _stop(
        "CLAIM",
        "STOP_NO_RERUN",
        f"claim has invalid terminal receipt: {error.detail}",
    )


def _claim_without_receipt() -> None:
    _stop("CLAIM", "STOP_NO_RERUN", "claim exists without complete receipt")


def _worker_spec(
    lock: Mapping[str, Any],
    lock_raw: bytes,
    payloads: Sequence[OpenAuthority],
    write_fds: Mapping[str, int],
    resolver: VolumeUUIDResolver,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for authority in payloads:
        observation = observe(authority, resolver)
        records.append(
            {
                "role": authority.role,
                "fd_number": authority.fd,
                "identity": observation["identity"],
                "size_bytes": observation["size_bytes"],
                "sha256": observation["sha256"],
                "access": "READ_ONLY",
            }
        )
    return {
        "schema_version": "sireto-v4.12-fresh-s0-worker-spec-1",
        "implementation_commit": lock["implementation_commit"],
        "execution_lock_sha256": sha256_bytes(lock_raw),
        "synthetic_run_id": lock["synthetic_run_id"],
        "attempt_id": lock["attempt_id"],
        "logical_time_utc": lock["logical_time_utc"],
        "minimum_stability_seconds": 60,
        "payload_fds": records,
        "write_directory_fds": {role: write_fds[role] for role in WRITE_DIRECTORY_ROLES},
        "control_protocol": "CANONICAL_LENGTH_PREFIXED_JSON_V1",
    }


def _message_hash(value: Mapping[str, Any]) -> str:
    projection = dict(value)
    projection.pop("message_sha256", None)
    return sha256_bytes(canonical_json(projection, final_lf=False))


def _validate_stability(value: Any) -> None:
    if type(value) is not dict or set(value) != set(STABILITY_FIELDS):
        _stop("WORKER", "WORKER_CONTROL_INVALID", "stability schema")
    for key in ("same_worker_process", "same_five_payload_fds"):
        if value[key] is not None and type(value[key]) is not bool:
            _stop("WORKER", "WORKER_CONTROL_INVALID", f"invalid {key}")
    elapsed = value["monotonic_elapsed_seconds"]
    if elapsed is not None:
        if type(elapsed) is not str:
            _stop("WORKER", "WORKER_CONTROL_INVALID", "invalid elapsed time")
        try:
            if float(elapsed) < 0 or not re.fullmatch(r"(0|[1-9][0-9]*)(\.[0-9]+)?", elapsed):
                raise ValueError
        except ValueError:
            _stop("WORKER", "WORKER_CONTROL_INVALID", "invalid elapsed time")


def _validate_output_authority(value: Any, *, allow_null: bool) -> None:
    if type(value) is not dict or set(value) != set(OUTPUT_AUTHORITY_FIELDS):
        _stop("WORKER", "WORKER_CONTROL_INVALID", "output authority schema")
    for key in OUTPUT_AUTHORITY_FIELDS:
        item = value[key]
        if item is None:
            if not allow_null:
                _stop("WORKER", "OUTPUT_AUTHORITY_INVALID", f"null output authority: {key}")
            continue
        if key == "terminal_tree_kind":
            if item not in {"SCAN_OUTPUT", "BATCH_QUARANTINE"}:
                _stop("WORKER", "OUTPUT_AUTHORITY_INVALID", "terminal tree kind")
        elif key == "journal_generation":
            if not _is_uint(item):
                _stop("WORKER", "OUTPUT_AUTHORITY_INVALID", "journal generation")
        else:
            _require_hex(item, key)


def _validate_ready(value: Mapping[str, Any], lock: Mapping[str, Any], pid: int) -> None:
    fields = {
        "schema_version", "message_type", "synthetic_run_id", "attempt_id",
        "worker_pid", "payload_fd_roles", "message_sha256",
    }
    if (
        type(value) is not dict
        or set(value) != fields
        or value.get("schema_version") != "sireto-v4.12-fresh-s0-control-ready-1"
        or value.get("message_type") != "READY"
        or value.get("synthetic_run_id") != lock["synthetic_run_id"]
        or value.get("attempt_id") != lock["attempt_id"]
        or value.get("worker_pid") != pid
        or value.get("payload_fd_roles") != list(WORKER_PAYLOAD_ROLES)
        or value.get("message_sha256") != _message_hash(value)
    ):
        _stop("WORKER", "WORKER_READY_INVALID", "invalid READY message")


def _validate_terminal(value: Mapping[str, Any], lock: Mapping[str, Any]) -> None:
    fields = {
        "schema_version", "message_type", "synthetic_run_id", "attempt_id",
        "phase", "reason_code", "terminal_result", "stability",
        "output_authority", "message_sha256",
    }
    if (
        type(value) is not dict
        or set(value) != fields
        or value.get("schema_version") != "sireto-v4.12-fresh-s0-control-result-1"
        or value.get("message_type") not in {"RESULT", "STOP"}
        or value.get("synthetic_run_id") != lock["synthetic_run_id"]
        or value.get("attempt_id") != lock["attempt_id"]
        or value.get("phase") != "WORKER"
        or value.get("message_sha256") != _message_hash(value)
    ):
        _stop("WORKER", "WORKER_CONTROL_INVALID", "invalid terminal control")
    _validate_stability(value["stability"])
    success = value["message_type"] == "RESULT"
    expected_results = {
        "INGESTED_SYNTHETIC_SCANNER_SEALER_V412",
        "QUARANTINED_SYNTHETIC_SCANNER_SEALER_V412",
    }
    if success:
        if value["reason_code"] != "OK" or value["terminal_result"] not in expected_results:
            _stop("WORKER", "WORKER_CONTROL_INVALID", "inconsistent RESULT")
    elif (
        value["reason_code"] != "WORKER_CONTROLLED_STOP"
        or value["terminal_result"] != "STOP_SYNTHETIC_SCANNER_SEALER_V412"
    ):
        _stop("WORKER", "WORKER_CONTROL_INVALID", "inconsistent STOP")
    _validate_output_authority(value["output_authority"], allow_null=not success)


def _parse_frames(buffer: bytearray, *, eof: bool = False) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    while len(buffer) >= 4:
        length = struct.unpack(">I", buffer[:4])[0]
        if length == 0 or length > 65_536:
            _stop("WORKER", "WORKER_CONTROL_INVALID", "invalid frame length")
        if len(buffer) < 4 + length:
            break
        raw = bytes(buffer[4 : 4 + length])
        del buffer[: 4 + length]
        try:
            value = json.loads(
                raw.decode("utf-8", "strict"),
                object_pairs_hook=_no_duplicate_keys,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"non-finite number: {token}")
                ),
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            _stop("WORKER", "WORKER_CONTROL_INVALID", f"invalid control JSON: {exc}")
        if type(value) is not dict or raw != canonical_json(value, final_lf=False):
            _stop("WORKER", "WORKER_CONTROL_INVALID", "non-canonical control frame")
        frames.append(value)
    if eof and buffer:
        _stop("WORKER", "WORKER_CONTROL_INVALID", "partial terminal frame")
    return frames


def _execute_worker(
    command: Sequence[str],
    environment: Mapping[str, str],
    pass_fds: Sequence[int],
    control_parent: socket.socket,
    control_child_fd: int,
    lock: Mapping[str, Any],
) -> WorkerExecution:
    process = subprocess.Popen(
        list(command),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        close_fds=True,
        pass_fds=tuple(pass_fds),
        env=dict(environment),
        cwd="/",
    )
    if process.stdout is None or process.stderr is None:
        os.close(control_child_fd)
        process.kill()
        process.wait()
        _stop("WORKER", "WORKER_CONTROL_INVALID", "worker pipes unavailable")
    os.close(control_child_fd)
    control_parent.setblocking(False)
    os.set_blocking(process.stdout.fileno(), False)
    os.set_blocking(process.stderr.fileno(), False)
    selector = selectors.DefaultSelector()
    selector.register(control_parent, selectors.EVENT_READ, "control")
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    buffers = {"control": bytearray(), "stdout": bytearray(), "stderr": bytearray()}
    frames: list[dict[str, Any]] = []
    ready_deadline = time.monotonic() + 10
    terminal_deadline: float | None = None
    control_eof = False
    try:
        while selector.get_map():
            deadline = ready_deadline if not frames else terminal_deadline
            if len(frames) == 1 and terminal_deadline is None:
                terminal_deadline = time.monotonic() + 180
                deadline = terminal_deadline
            timeout = max(0.0, (deadline or time.monotonic()) - time.monotonic())
            events = selector.select(timeout=min(timeout, 1.0))
            if not events:
                if time.monotonic() >= (deadline or 0):
                    process.kill()
                    _stop(
                        "WORKER",
                        "WORKER_READY_INVALID" if not frames else "WORKER_CONTROL_INVALID",
                        "worker control timeout",
                    )
                continue
            for key, _ in events:
                name = key.data
                try:
                    chunk = os.read(key.fileobj.fileno(), 65536)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(key.fileobj)
                    if name == "control":
                        control_eof = True
                        frames.extend(_parse_frames(buffers["control"], eof=True))
                    continue
                buffers[name].extend(chunk)
                if name == "control":
                    frames.extend(_parse_frames(buffers["control"]))
                    if len(frames) > 2:
                        process.kill()
                        _stop("WORKER", "WORKER_CONTROL_INVALID", "extra control frame")
                    if len(frames) == 1:
                        _validate_ready(frames[0], lock, process.pid)
                elif len(buffers[name]) > MAX_CAPTURE_BYTES:
                    process.kill()
                    _stop("WORKER", "WORKER_CONTROL_INVALID", f"{name} capture exceeded")
            if process.poll() is not None and control_eof:
                # Continue draining stdout/stderr until their pipe EOF.
                if all(key.data == "control" for key in selector.get_map().values()):
                    break
        return_code = process.wait(timeout=5)
        if len(frames) != 2 or not control_eof:
            _stop("WORKER", "WORKER_CONTROL_INVALID", "control sequence incomplete")
        _validate_ready(frames[0], lock, process.pid)
        _validate_terminal(frames[1], lock)
    except LauncherStop as exc:
        if process.poll() is None:
            process.kill()
        return_code = process.wait()
        if return_code < 0:
            exit_code: int | None = None
            signal_number: int | None = -return_code
        else:
            exit_code = return_code
            signal_number = None
        exc.worker_execution = WorkerExecution(  # type: ignore[attr-defined]
            pid=process.pid,
            exit_code=exit_code,
            signal_number=signal_number,
            stdout=bytes(buffers["stdout"]),
            stderr=bytes(buffers["stderr"]),
            ready=frames[0] if frames else None,
            result=None,
        )
        raise
    finally:
        selector.close()
        if process.poll() is None:
            process.kill()
            process.wait()
    if return_code < 0:
        exit_code = None
        signal_number = -return_code
    else:
        exit_code = return_code
        signal_number = None
    return WorkerExecution(
        pid=process.pid,
        exit_code=exit_code,
        signal_number=signal_number,
        stdout=bytes(buffers["stdout"]),
        stderr=bytes(buffers["stderr"]),
        ready=frames[0],
        result=frames[1],
    )


def _worker_environment(
    lock: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, str]:
    root = Path(plan["paths"]["allowed_root"]) / "runtime" / lock["synthetic_run_id"]
    # Exact concrete paths are stored in the private records.  The sealer also
    # stores the derived environment under runtime.layout when present; reject
    # any unknown source rather than accepting environment overrides.
    runtime_layout = lock["runtime"].get("layout")
    if runtime_layout is not None:
        _stop("PRESPAWN", "RUNTIME_INVALID", "runtime object has extra layout")
    _private_python_path(lock, plan)
    pythonhome = (
        root
        / "rootfs/opt/homebrew/Cellar/python@3.14/3.14.3_1/"
        "Frameworks/Python.framework/Versions/3.14"
    )
    encodings_records = [
        record
        for record in lock["runtime"]["private_runtime_manifest"]["records"]
        if record["role"] == "PYTHON_STDLIB"
        and record["private_relative_path"]
        == (
            "rootfs/opt/homebrew/Cellar/python@3.14/3.14.3_1/"
            "Frameworks/Python.framework/Versions/3.14/lib/python3.14/"
            "encodings/__init__.py"
        )
    ]
    if len(encodings_records) != 1:
        _stop("PRESPAWN", "RUNTIME_INVALID", "private encodings authority absent")
    pyarrow_records = [
        record
        for record in lock["runtime"]["private_runtime_manifest"]["records"]
        if record["role"] == "PYARROW"
    ]
    if not pyarrow_records:
        _stop("PRESPAWN", "RUNTIME_INVALID", "PyArrow runtime records absent")
    init_records = [
        record
        for record in pyarrow_records
        if Path(record["private_relative_path"]).name == "__init__.py"
        and Path(record["private_relative_path"]).parent.name == "pyarrow"
    ]
    if len(init_records) != 1:
        _stop("PRESPAWN", "RUNTIME_INVALID", "PyArrow package root ambiguous")
    private_pyarrow = _runtime_path(lock, plan, init_records[0]).parent
    private_site = private_pyarrow.parent
    app_root = root / "app"
    environment = {
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHOME": os.fspath(pythonhome),
        "PYTHONPATH": f"{private_site}:{app_root}",
        "TMPDIR": os.fspath(Path(plan["paths"]["allowed_root"]) / "tmp" / lock["synthetic_run_id"]),
    }
    if (
        list(environment) != plan["launcher"]["environment_exact_keys"]
        or any(
            key in environment
            for key in _r2b_runtime_boundary(plan)["dyld_environment_forbidden"]
        )
    ):
        _stop("PRESPAWN", "RUNTIME_INVALID", "R2-B worker environment")
    return environment


def _private_import_assertion(
    lock: Mapping[str, Any], plan: Mapping[str, Any]
) -> str:
    root = (
        Path(plan["paths"]["allowed_root"])
        / "runtime"
        / lock["synthetic_run_id"]
    )
    root_literal = json.dumps(os.fspath(root), ensure_ascii=True)
    return (
        "import encodings,os,pyarrow;"
        f"r=os.path.realpath({root_literal});"
        "e=os.path.realpath(encodings.__file__);"
        "p=os.path.realpath(pyarrow.__file__);"
        "assert pyarrow.__version__=='23.0.1';"
        "assert os.path.isfile(e) and e!=r and os.path.commonpath((r,e))==r;"
        "assert os.path.isfile(p) and p!=r and os.path.commonpath((r,p))==r"
    )


def _validate_r2_smoke(
    lock: Mapping[str, Any],
    plan: Mapping[str, Any],
    profile_authority: OpenAuthority,
) -> None:
    smoke = lock["r2_smoke"]
    schema = plan["schema_definitions"]["r2_smoke_attestation"]
    fields = schema["exact_fields"]
    if type(smoke) is not dict or set(smoke) != set(fields):
        _stop("PRESPAWN", "LOCK_INVALID", "R2-B smoke schema")
    profile_text = _verified_profile_text(profile_authority)
    python_record = _private_record(lock, "PYTHON_EXECUTABLE")
    python_path = _private_python_path(lock, plan)
    environment = _worker_environment(lock, plan)
    argv = [
        os.fspath(SANDBOX_EXEC_PATH),
        "-p",
        profile_text,
        os.fspath(python_path),
        "-c",
        _private_import_assertion(lock, plan),
    ]
    required = plan["r2_successor"]["smoke_attestation"]["required_result"]
    expected = {
        "schema_version": "sireto-v4.12-fresh-s0-r2-smoke-attestation-2",
        "implementation_commit": lock["implementation_commit"],
        "synthetic_run_id": lock["synthetic_run_id"],
        "attempt_id": lock["attempt_id"],
        "python_sha256": python_record["sha256"],
        "profile_sha256": profile_authority.expected_sha256,
        "environment_sha256": sha256_bytes(
            canonical_json(environment, final_lf=False)
        ),
        "argv_sha256": sha256_bytes(canonical_json(argv, final_lf=False)),
        "pass_fds": [],
        "exit_code": required["exit_code"],
        "signal": required["signal"],
        "stdout_size_bytes": required["stdout_size_bytes"],
        "stdout_sha256": required["stdout_sha256"],
        "stderr_size_bytes": required["stderr_size_bytes"],
        "stderr_sha256": required["stderr_sha256"],
        "five_output_directories_empty_before": required[
            "five_output_directories_empty_before"
        ],
        "five_output_directories_empty_after": required[
            "five_output_directories_empty_after"
        ],
    }
    if any(smoke.get(key) != value for key, value in expected.items()):
        _stop("PRESPAWN", "LOCK_INVALID", "R2-B smoke authority mismatch")
    projection = {key: smoke[key] for key in fields if key != "smoke_sha256"}
    if smoke["smoke_sha256"] != sha256_bytes(
        canonical_json(projection, final_lf=False)
    ):
        _stop("PRESPAWN", "LOCK_INVALID", "R2-B smoke hash")


def _validate_canaries(
    value: Any, plan: Mapping[str, Any], *, run_id: str
) -> None:
    expected_codes = plan["canary_matrix"]["runtime_codes_exact_order"]
    if (
        type(value) is not list
        or any(type(item) is not dict for item in value)
        or [item["code"] for item in value] != expected_codes
    ):
        _stop("POSTWORKER", "SANDBOX_EXPECTATION_FAILED", "canary order")
    allowed_operations = {"OPEN_READ", "ENUMERATE_PARENT", "OPEN_NETWORK", "WRITE"}
    for record in value:
        code = record.get("code") if type(record) is dict else None
        if code == "DENY_NETWORK":
            expected_operation = "OPEN_NETWORK"
        elif code == "DENY_PARENT_ENUMERATION":
            expected_operation = "ENUMERATE_PARENT"
        elif code == "DENY_WRITE_PARENT_AUDIT":
            expected_operation = "WRITE"
        else:
            expected_operation = "OPEN_READ"
        target_template = (
            plan["canary_matrix"]["synthetic_target_by_runtime_code"].get(code)
            if type(code) is str
            else None
        )
        expected_target = (
            target_template.replace(
                "<allowed_root>", plan["paths"]["allowed_root"]
            ).replace("<synthetic_run_id>", run_id)
            if type(target_template) is str
            else None
        )
        target_exact = record.get("synthetic_target") == expected_target
        if (
            type(record) is not dict
            or set(record) != {"code", "operation", "synthetic_target", "result", "errno"}
            or record["operation"] not in allowed_operations
            or record["operation"] != expected_operation
            or not target_exact
            or record["result"] != "DENIED"
            or type(record["errno"]) is not int
            or record["errno"] not in plan["canary_matrix"]["errno_allowed_by_runtime_code"][record["code"]]
        ):
            _stop("POSTWORKER", "SANDBOX_EXPECTATION_FAILED", "canary result")


def _read_canary_proof(
    audit_fd: int,
    lock: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    try:
        fd = os.open(
            "canaries.json",
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=audit_fd,
        )
    except OSError as exc:
        _stop("POSTWORKER", "SANDBOX_EXPECTATION_FAILED", f"canary proof absent: {exc}")
    try:
        raw, _ = _read_regular_fd(fd, "worker canary proof")
    finally:
        os.close(fd)
    value = decode_canonical_json(raw, "worker canary proof")
    fields = {
        "schema_version",
        "synthetic_run_id",
        "attempt_id",
        "ordered_records",
        "record_count",
        "records_sha256",
    }
    records = value.get("ordered_records") if type(value) is dict else None
    if (
        type(value) is not dict
        or set(value) != fields
        or value["schema_version"] != "sireto-v4.12-fresh-s0-canary-proof-1"
        or value["synthetic_run_id"] != lock["synthetic_run_id"]
        or value["attempt_id"] != lock["attempt_id"]
        or type(records) is not list
        or value["record_count"] != len(records)
        or value["records_sha256"]
        != sha256_bytes(canonical_json(records, final_lf=False))
    ):
        _stop("POSTWORKER", "SANDBOX_EXPECTATION_FAILED", "canary proof schema")
    _validate_canaries(records, plan, run_id=lock["synthetic_run_id"])
    return records


def _read_stop_canaries(
    audit_fd: int,
    lock: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    try:
        return _read_canary_proof(audit_fd, lock, plan)
    except LauncherStop as failure:
        try:
            os.stat(
                "canaries.json",
                dir_fd=audit_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return []
        except OSError:
            _stop(
                "POSTWORKER",
                "SANDBOX_EXPECTATION_FAILED",
                "canary proof existence is indeterminate",
            )
        _stop(
            "POSTWORKER",
            "SANDBOX_EXPECTATION_FAILED",
            f"existing canary proof invalid: {failure.detail}",
        )


def _require_absent_canary_proof(audit_fd: int) -> None:
    try:
        os.stat(
            "canaries.json",
            dir_fd=audit_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    except OSError:
        _stop(
            "RECEIPT",
            "RECEIPT_CONFLICT",
            "empty canary proof absence indeterminate",
        )
    _stop(
        "RECEIPT",
        "RECEIPT_CONFLICT",
        "empty canaries conflict with existing proof",
    )


def _validate_sealed_tree(
    path: Path,
    *,
    expected_package_kind: str,
    run_id: str,
) -> tuple[str, str]:
    root_fd = _open_anchored(path, directory=True)

    def identity(info: os.stat_result) -> tuple[int, ...]:
        return (
            info.st_dev,
            info.st_ino,
            info.st_mode,
            info.st_uid,
            info.st_nlink,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )

    def snapshot() -> tuple[tuple[int, ...], dict[str, tuple[int, ...]]]:
        root_info = os.fstat(root_fd)
        if (
            not stat.S_ISDIR(root_info.st_mode)
            or root_info.st_uid != os.getuid()
            or stat.S_IMODE(root_info.st_mode) != 0o700
        ):
            _stop("POSTWORKER", "OUTPUT_AUTHORITY_INVALID", "unsafe tree root")
        records: dict[str, tuple[int, ...]] = {}
        for name in sorted(os.listdir(root_fd), key=lambda item: os.fsencode(item)):
            if (
                type(name) is not str
                or not name
                or name in {".", ".."}
                or "/" in name
                or "\0" in name
            ):
                _stop("POSTWORKER", "OUTPUT_AUTHORITY_INVALID", "unsafe tree entry")
            info = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or info.st_uid != os.getuid()
                or info.st_dev != root_info.st_dev
                or stat.S_IMODE(info.st_mode) != 0o600
            ):
                _stop("POSTWORKER", "OUTPUT_AUTHORITY_INVALID", "unsafe tree payload")
            records[name] = identity(info)
        return identity(root_info), records

    def read_name(name: str, label: str) -> bytes:
        if not name or "/" in name or name in {".", ".."}:
            _stop("POSTWORKER", "OUTPUT_AUTHORITY_INVALID", "unsafe tree filename")
        fd = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=root_fd,
        )
        try:
            raw, _ = _read_regular_fd(fd, label)
            return raw
        finally:
            os.close(fd)

    try:
        before = snapshot()
        names = sorted(before[1], key=lambda item: item.encode("utf-8"))
        if "payload_manifest.json" not in names or "seal.json" not in names:
            _stop("POSTWORKER", "OUTPUT_AUTHORITY_INVALID", "tree authority files absent")
        manifest_raw = read_name("payload_manifest.json", "tree payload manifest")
        seal_raw = read_name("seal.json", "tree seal")
        manifest = decode_canonical_json(manifest_raw, "tree payload manifest")
        seal = decode_canonical_json(seal_raw, "tree seal")
        manifest_fields = {
            "schema_version",
            "package_kind",
            "synthetic_run_id",
            "collection_id",
            "source_batch_id",
            "logical_time_utc",
            "ordered_payload_records",
            "payload_count",
            "payload_tree_sha256",
        }
        seal_fields = {
            "schema_version",
            "package_kind",
            "synthetic_run_id",
            "collection_id",
            "source_batch_id",
            "logical_time_utc",
            "payload_manifest_size_bytes",
            "payload_manifest_sha256",
            "payload_tree_sha256",
        }
        records = manifest.get("ordered_payload_records")
        if (
            set(manifest) != manifest_fields
            or set(seal) != seal_fields
            or manifest["schema_version"]
            != "sireto-v4.12-fresh-synthetic-payload-manifest-1"
            or seal["schema_version"] != "sireto-v4.12-fresh-synthetic-seal-1"
            or manifest["package_kind"] != expected_package_kind
            or seal["package_kind"] != expected_package_kind
            or manifest["synthetic_run_id"] != run_id
            or seal["synthetic_run_id"] != run_id
            or type(records) is not list
            or manifest["payload_count"] != len(records)
        ):
            _stop("POSTWORKER", "OUTPUT_AUTHORITY_INVALID", "tree manifest schema")
        payload_names: list[str] = []
        for record in records:
            if (
                type(record) is not dict
                or set(record) != {"relative_path", "size_bytes", "sha256"}
                or type(record["relative_path"]) is not str
                or not record["relative_path"]
                or "/" in record["relative_path"]
                or record["relative_path"]
                in {".", "..", "payload_manifest.json", "seal.json"}
                or not _is_uint(record["size_bytes"])
                or type(record["sha256"]) is not str
                or HEX64.fullmatch(record["sha256"]) is None
            ):
                _stop("POSTWORKER", "OUTPUT_AUTHORITY_INVALID", "tree payload record")
            payload_names.append(record["relative_path"])
            payload = read_name(record["relative_path"], "sealed tree payload")
            if len(payload) != record["size_bytes"] or sha256_bytes(payload) != record["sha256"]:
                _stop("POSTWORKER", "OUTPUT_AUTHORITY_INVALID", "tree payload mismatch")
        if (
            len(set(payload_names)) != len(payload_names)
            or names
            != sorted(
                [*payload_names, "payload_manifest.json", "seal.json"],
                key=lambda item: item.encode("utf-8"),
            )
            or manifest["payload_tree_sha256"]
            != sha256_bytes(canonical_json(records, final_lf=False))
        ):
            _stop("POSTWORKER", "OUTPUT_AUTHORITY_INVALID", "tree exact-set mismatch")
        shared = (
            "collection_id",
            "source_batch_id",
            "logical_time_utc",
            "payload_tree_sha256",
        )
        if (
            any(seal[key] != manifest[key] for key in shared)
            or seal["payload_manifest_size_bytes"] != len(manifest_raw)
            or seal["payload_manifest_sha256"] != sha256_bytes(manifest_raw)
        ):
            _stop("POSTWORKER", "OUTPUT_AUTHORITY_INVALID", "tree seal mismatch")
        after = snapshot()
        if after != before:
            _stop("POSTWORKER", "OUTPUT_AUTHORITY_INVALID", "tree drift during validation")
        return sha256_bytes(manifest_raw), sha256_bytes(seal_raw)
    finally:
        os.close(root_fd)


def _validate_output_paths(
    result: Mapping[str, Any], lock: Mapping[str, Any], paths: Mapping[str, Path]
) -> dict[str, Any]:
    """Revalidate every terminal hash from parent-owned paths.

    The worker result is not authoritative by itself.  The conventional core
    filenames are looked up only below exact output roots already fixed by the
    execution lock.
    """
    authority = dict(result["output_authority"])
    if result["message_type"] == "STOP":
        if any(value is not None for value in authority.values()):
            _stop("POSTWORKER", "OUTPUT_AUTHORITY_INVALID", "STOP claimed output authority")
        return authority
    sealed_hashes = _validate_sealed_tree(
        paths["sealed"] / "input",
        expected_package_kind="SEALED_INPUT",
        run_id=lock["synthetic_run_id"],
    )
    if sealed_hashes != (
        authority["sealed_input_payload_manifest_sha256"],
        authority["sealed_input_seal_sha256"],
    ):
        _stop("POSTWORKER", "OUTPUT_AUTHORITY_INVALID", "sealed input authority")
    is_scan = authority["terminal_tree_kind"] == "SCAN_OUTPUT"
    terminal_hashes = _validate_sealed_tree(
        (paths["scan"] / "output") if is_scan else (paths["quarantine"] / "batch"),
        expected_package_kind=(
            "SCAN_OUTPUT" if is_scan else "BATCH_QUARANTINE"
        ),
        run_id=lock["synthetic_run_id"],
    )
    if terminal_hashes != (
        authority["terminal_tree_payload_manifest_sha256"],
        authority["terminal_tree_seal_sha256"],
    ):
        _stop("POSTWORKER", "OUTPUT_AUTHORITY_INVALID", "terminal tree authority")
    generation = authority["journal_generation"]
    if generation != 3:
        _stop("POSTWORKER", "OUTPUT_AUTHORITY_INVALID", "unexpected journal generation")
    generation_path = (
        paths["audit_worker"]
        / "events_manifests"
        / f"{generation:08d}-{authority['journal_generation_manifest_sha256']}.json"
    )
    event_path = (
        paths["audit_worker"]
        / "events"
        / f"{generation:08d}-{authority['journal_head_event_sha256']}.json"
    )
    generation_raw = _read_anchored_path(
        generation_path, "journal generation manifest"
    )
    event_raw = _read_anchored_path(event_path, "journal head event")
    if (
        sha256_bytes(generation_raw)
        != authority["journal_generation_manifest_sha256"]
        or sha256_bytes(event_raw) != authority["journal_head_event_sha256"]
    ):
        _stop("POSTWORKER", "OUTPUT_AUTHORITY_INVALID", "journal hash mismatch")
    generation_value = decode_canonical_json(
        generation_raw, "journal generation manifest"
    )
    records = generation_value.get("ordered_event_records")
    if (
        generation_value.get("generation") != generation
        or generation_value.get("event_count") != generation
        or generation_value.get("head_event_sha256")
        != authority["journal_head_event_sha256"]
        or type(records) is not list
        or len(records) != generation
        or type(records[-1]) is not dict
        or records[-1].get("relative_path") != f"events/{event_path.name}"
        or records[-1].get("sha256") != authority["journal_head_event_sha256"]
        or records[-1].get("size_bytes") != len(event_raw)
    ):
        _stop("POSTWORKER", "OUTPUT_AUTHORITY_INVALID", "journal generation mismatch")
    return authority


def _result_row(execution: WorkerExecution) -> tuple[str, str, str]:
    result = execution.result
    if result is None:
        return "STOP", "WORKER_CRASHED", "STOP_SYNTHETIC_SCANNER_SEALER_V412"
    terminal = result["terminal_result"]
    if (
        result["message_type"] == "RESULT"
        and terminal == "INGESTED_SYNTHETIC_SCANNER_SEALER_V412"
        and execution.exit_code == 0
        and execution.signal_number is None
    ):
        return "INGESTED", "OK", terminal
    if (
        result["message_type"] == "RESULT"
        and terminal == "QUARANTINED_SYNTHETIC_SCANNER_SEALER_V412"
        and execution.exit_code == 0
        and execution.signal_number is None
    ):
        return "QUARANTINED", "OK", terminal
    if (
        result["message_type"] == "STOP"
        and terminal == "STOP_SYNTHETIC_SCANNER_SEALER_V412"
        and execution.exit_code == 2
        and execution.signal_number is None
    ):
        return "STOP", "WORKER_CONTROLLED_STOP", terminal
    return "STOP", "WORKER_EXIT_INVALID", "STOP_SYNTHETIC_SCANNER_SEALER_V412"


def _success_stability(stability: Mapping[str, Any]) -> bool:
    try:
        elapsed = float(stability["monotonic_elapsed_seconds"])
    except (TypeError, ValueError):
        return False
    return (
        stability["same_worker_process"] is True
        and stability["same_five_payload_fds"] is True
        and elapsed >= 60.0
    )


def _select_receipt_outcome(
    execution: WorkerExecution,
) -> tuple[str, str, str, dict[str, Any]]:
    verdict, reason, terminal = _result_row(execution)
    if execution.result is None:
        _stop("WORKER", "WORKER_CONTROL_INVALID", "terminal result absent")
    stability = dict(execution.result["stability"])
    if verdict in {"INGESTED", "QUARANTINED"} and not _success_stability(
        stability
    ):
        return (
            "STOP",
            "STABILITY_FAILED",
            "STOP_SYNTHETIC_SCANNER_SEALER_V412",
            stability,
        )
    return verdict, reason, terminal, stability


def run_authoritative_launch() -> dict[str, Any]:
    started = _now_utc()
    plan, _ = _load_plan()
    authorization, auth_raw = _load_authorization(plan)
    lock, lock_raw, lock_fd = _load_lock(authorization, plan)
    authorities: list[OpenAuthority] = []
    write_fds: dict[str, int] = {}
    lease_fd = -1
    control_parent: socket.socket | None = None
    control_child: socket.socket | None = None
    claim_raw: bytes | None = None
    receipt_path: Path | None = None
    lease_path: Path | None = None
    implementation_hashes: dict[str, str] = {}
    before: list[dict[str, Any]] = []
    after: list[dict[str, Any]] | None = None
    execution: WorkerExecution | None = None
    profile_record: Mapping[str, Any] | None = None
    canaries: list[dict[str, Any]] = []
    validated_output_authority: dict[str, Any] | None = None
    try:
        implementation_hashes = _validate_implementation(lock, authorization, plan)
        paths = _substitute_paths(lock, plan)
        _validate_runtime(lock, plan)
        _validate_lock_sandbox(lock, plan, implementation_hashes)
        _validate_private_runtime_tree(lock, plan)
        allowed_root = Path(plan["paths"]["allowed_root"])
        parent_root = paths["audit_parent"]
        for directory in (
            parent_root,
            parent_root / "leases",
            parent_root / "claims",
            parent_root / "launch_receipts",
            parent_root / "spec",
        ):
            directory_fd = _validate_precreated_directory(
                directory, allowed_root=allowed_root
            )
            os.close(directory_fd)
        lease_path = parent_root / "leases" / f"{lock['attempt_id']}.lease"
        lease_parent_fd = _open_anchored(lease_path.parent, directory=True)
        previous_umask = os.umask(0o077)
        try:
            lease_fd = os.open(
                lease_path.name,
                os.O_CREAT
                | os.O_RDWR
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=lease_parent_fd,
            )
        finally:
            os.umask(previous_umask)
            os.close(lease_parent_fd)
        lease_info = os.fstat(lease_fd)
        allowed_root_fd = _open_anchored(allowed_root, directory=True)
        try:
            root_info = os.fstat(allowed_root_fd)
        finally:
            os.close(allowed_root_fd)
        if (
            not stat.S_ISREG(lease_info.st_mode)
            or lease_info.st_nlink != 1
            or lease_info.st_uid != os.getuid()
            or lease_info.st_dev != root_info.st_dev
            or stat.S_IMODE(lease_info.st_mode) & 0o077
        ):
            _stop("LEASE", "LEASE_CONFLICT", "unsafe lease")
        try:
            fcntl.flock(lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _stop("LEASE", "LEASE_CONFLICT", "lease already held")
        claim_path = parent_root / "claims" / f"{lock['attempt_id']}.json"
        receipt_path = parent_root / "launch_receipts" / f"{lock['attempt_id']}.json"
        claim_exists = _anchored_entry_exists(claim_path)
        receipt_exists = _anchored_entry_exists(receipt_path)
        if claim_exists:
            claim_raw = _validate_existing_claim(claim_path, lock, auth_raw, lock_raw)
            if receipt_exists:
                try:
                    raw = _validate_existing_receipt(
                        receipt_path,
                        lock=lock,
                        auth_raw=auth_raw,
                        lock_raw=lock_raw,
                        claim_raw=claim_raw,
                        execution_lock_path=authorization[
                            "execution_lock_absolute_path"
                        ],
                        plan=plan,
                        implementation_hashes=implementation_hashes,
                    )
                except LauncherStop as invalid_receipt:
                    _invalid_recovery_receipt(invalid_receipt)
                return decode_canonical_json(raw, "idempotent launch receipt")
            _claim_without_receipt()
        if receipt_exists:
            _stop("CLAIM", "CLAIM_CONFLICT", "receipt exists without claim")

        resolver = VolumeUUIDResolver()
        _validate_volume_pins(lock, plan, resolver)
        authorities.append(
            OpenAuthority(
                "EXECUTION_LOCK",
                Path(authorization["execution_lock_absolute_path"]),
                lock_fd,
                sha256_bytes(lock_raw),
                len(lock_raw),
                None,
            )
        )
        auth_fd = _open_anchored(AUTHORIZATION_PATH)
        authorities.append(
            OpenAuthority(
                "AUTHORIZATION",
                AUTHORIZATION_PATH,
                auth_fd,
                sha256_bytes(auth_raw),
                len(auth_raw),
                None,
            )
        )
        input_records = _input_records(lock)
        worker_record = _private_record(lock, "WORKER")
        profile_record = _private_record(lock, "EFFECTIVE_SANDBOX_PROFILE")
        for role, record in (("WORKER", worker_record), ("SANDBOX_PROFILE", profile_record)):
            authorities.append(
                _open_authority(
                    role,
                    _runtime_path(lock, plan, record),
                    record["sha256"],
                    record["size_bytes"],
                    None,
                )
            )
        host_framework_record = _host_python_framework(lock, plan)
        authorities.append(
            _open_authority(
                "HOST_PYTHON_FRAMEWORK",
                Path(host_framework_record["absolute_path"]),
                host_framework_record["sha256"],
                host_framework_record["size_bytes"],
                host_framework_record["identity"],
            )
        )
        for role in ("PRIVATE_RUNTIME_MANIFEST", *WORKER_PAYLOAD_ROLES):
            record = input_records[role]
            authorities.append(
                _open_authority(
                    role,
                    Path(record["absolute_path"]),
                    record["sha256"],
                    record["size_bytes"],
                    record["identity"],
                )
            )
        runtime_manifest_authority = next(
            item for item in authorities if item.role == "PRIVATE_RUNTIME_MANIFEST"
        )
        runtime_manifest_raw, _ = _read_regular_fd(
            runtime_manifest_authority.fd, "private runtime manifest input"
        )
        if (
            decode_canonical_json(
                runtime_manifest_raw, "private runtime manifest input"
            )
            != lock["runtime"]["private_runtime_manifest"]
        ):
            _stop(
                "PRESPAWN",
                "RUNTIME_INVALID",
                "lock embeds a different private runtime manifest",
            )
        payloads = [
            authority for authority in authorities if authority.role in WORKER_PAYLOAD_ROLES
        ]
        write_paths = {
            "SEALED": paths["sealed"],
            "SCAN": paths["scan"],
            "QUARANTINE": paths["quarantine"],
            "AUDIT": paths["audit_worker"],
            "TMP": paths["tmp"],
        }
        for role, path in write_paths.items():
            write_fds[role] = _validate_precreated_directory(
                path, allowed_root=allowed_root
            )
            if os.listdir(write_fds[role]):
                _stop("PRESPAWN", "FD_INVALID", f"pre-spawn output not empty: {role}")
        host_framework_authority = next(
            item
            for item in authorities
            if item.role == "HOST_PYTHON_FRAMEWORK"
        )
        observe(host_framework_authority, resolver)
        _validate_python_helper_install_name(lock, plan, phase="PRESPAWN")
        profile_authority = next(
            item for item in authorities if item.role == "SANDBOX_PROFILE"
        )
        _validate_r2_smoke(lock, plan, profile_authority)
        worker_spec_value = _worker_spec(lock, lock_raw, payloads, write_fds, resolver)
        claim_raw = _write_exclusive(
            claim_path, _claim_value(lock, auth_raw, lock_raw), mode=0o400
        )
        spec_path = paths["worker_spec"]
        spec_raw = _write_exclusive(spec_path, worker_spec_value, mode=0o400)
        spec_fd = _open_anchored(spec_path)
        authorities.append(
            OpenAuthority(
                "WORKER_SPEC",
                spec_path,
                spec_fd,
                sha256_bytes(spec_raw),
                len(spec_raw),
                None,
            )
        )
        canary = input_records["CANARY_MANIFEST"]
        authorities.append(
            _open_authority(
                "CANARY_MANIFEST",
                Path(canary["absolute_path"]),
                canary["sha256"],
                canary["size_bytes"],
                canary["identity"],
            )
        )
        canary_authority = authorities[-1]
        canary_raw, _ = _read_regular_fd(
            canary_authority.fd, "canary manifest input"
        )
        _validate_canary_manifest(canary_raw, lock, plan)
        if tuple(authority.role for authority in authorities) != PARENT_RETAINED_ROLES:
            _stop("PRESPAWN", "FD_INVALID", "parent retained role order")
        before = [observe(authority, resolver) for authority in authorities]
        control_parent, control_child = socket.socketpair(
            socket.AF_UNIX, socket.SOCK_STREAM
        )
        worker_authority = next(item for item in authorities if item.role == "WORKER")
        profile_authority = next(item for item in authorities if item.role == "SANDBOX_PROFILE")
        python_path = _private_python_path(lock, plan)
        child_fd = control_child.fileno()
        pass_fds = [
            *(authority.fd for authority in payloads),
            *(write_fds[role] for role in WRITE_DIRECTORY_ROLES),
            spec_fd,
            child_fd,
        ]
        if len(pass_fds) != len(set(pass_fds)):
            _stop("PRESPAWN", "FD_INVALID", "duplicate passed descriptor")
        _revalidate_private_runtime_critical(
            lock, plan, authorities, phase="PRESPAWN"
        )
        profile_text = _verified_profile_text(profile_authority)
        command = [
            os.fspath(SANDBOX_EXEC_PATH),
            "-p",
            profile_text,
            os.fspath(python_path),
            os.fspath(worker_authority.path),
            "--worker-spec-fd",
            str(spec_fd),
            "--worker-control-fd",
            str(child_fd),
        ]
        detached_child_fd = control_child.detach()
        control_child = None
        execution = _execute_worker(
            command,
            _worker_environment(lock, plan),
            pass_fds,
            control_parent,
            detached_child_fd,
            lock,
        )
        _revalidate_private_runtime_critical(
            lock, plan, authorities, phase="POSTWORKER"
        )
        after = [observe(authority, resolver) for authority in authorities]
        if before != after:
            _stop("POSTWORKER", "PARENT_OBSERVATION_DRIFT", "authority drift")
        verdict, reason, terminal, stability = _select_receipt_outcome(
            execution
        )
        result = execution.result
        if result is None:
            _stop("WORKER", "WORKER_CONTROL_INVALID", "terminal result absent")
        canaries = _read_canary_proof(
            write_fds["AUDIT"], lock, plan
        )
        output_authority = _validate_output_paths(result, lock, paths)
        validated_output_authority = dict(output_authority)
        worker_receipt = {
            "pid": execution.pid,
            "exit_code": execution.exit_code,
            "signal": execution.signal_number,
            "stdout_size_bytes": len(execution.stdout),
            "stdout_sha256": sha256_bytes(execution.stdout),
            "stderr_size_bytes": len(execution.stderr),
            "stderr_sha256": sha256_bytes(execution.stderr),
            "control_result": result,
        }
        receipt = {
            "schema_version": "sireto-v4.12-fresh-s0-authoritative-launch-receipt-2",
            "phase": "RECEIPT",
            "reason_code": reason,
            "authorization_manifest_sha256": sha256_bytes(auth_raw),
            "execution_lock_path": authorization["execution_lock_absolute_path"],
            "execution_lock_sha256": sha256_bytes(lock_raw),
            "implementation_commit": lock["implementation_commit"],
            "implementation_blob_hashes": implementation_hashes,
            "synthetic_run_id": lock["synthetic_run_id"],
            "attempt_id": lock["attempt_id"],
            "claim_sha256": sha256_bytes(claim_raw),
            "lease_path": os.fspath(lease_path),
            "lease_held_for_spawn": True,
            "runtime": lock["runtime"],
            "sandbox_profile_sha256": implementation_hashes["SANDBOX_PROFILE"],
            "effective_sandbox_profile_sha256": profile_record["sha256"],
            "parent_before_observations": before,
            "worker_receipt": worker_receipt,
            "parent_after_observations": after,
            "stability": stability,
            "canaries": canaries,
            "output_authority": output_authority,
            "macos_limitations": plan["macos_runtime_boundary"]["acknowledged_limitations"],
            "terminal_result": terminal,
            "verdict": verdict,
            "started_at_utc": started,
            "finished_at_utc": _now_utc(),
        }
        if set(receipt) != set(RECEIPT_FIELDS):
            _stop("RECEIPT", "RECEIPT_CONFLICT", "internal receipt schema")
        _write_exclusive(receipt_path, receipt, mode=0o400)
        return receipt
    except LauncherStop as failure:
        if (
            claim_raw is not None
            and receipt_path is not None
            and lease_path is not None
            and not _anchored_entry_exists(receipt_path)
        ):
            partial_execution = execution or getattr(
                failure, "worker_execution", None
            )
            if partial_execution is not None:
                observed_control = partial_execution.result
                worker_receipt: dict[str, Any] | None = {
                    "pid": partial_execution.pid,
                    "exit_code": partial_execution.exit_code,
                    "signal": partial_execution.signal_number,
                    "stdout_size_bytes": len(partial_execution.stdout),
                    "stdout_sha256": sha256_bytes(partial_execution.stdout),
                    "stderr_size_bytes": len(partial_execution.stderr),
                    "stderr_sha256": sha256_bytes(partial_execution.stderr),
                    "control_result": observed_control,
                }
            else:
                observed_control = None
                worker_receipt = None
            if write_fds.get("AUDIT") is not None:
                # An existing malformed/partial proof is an unrecoverable
                # post-claim state.  Never summarize it into a receipt:
                # propagate, keep the claim, and force STOP_NO_RERUN later.
                canaries = _read_stop_canaries(
                    write_fds["AUDIT"], lock, plan
                )
            if authorities and before:
                try:
                    after = [
                        observe(authority, resolver) for authority in authorities
                    ]
                except (LauncherStop, OSError):
                    after = None
            observed_stability = (
                dict(observed_control["stability"])
                if type(observed_control) is dict
                and type(observed_control.get("stability")) is dict
                else {
                    "same_worker_process": None,
                    "same_five_payload_fds": None,
                    "monotonic_elapsed_seconds": None,
                }
            )
            honest_output_authority = (
                dict(validated_output_authority)
                if validated_output_authority is not None
                else {key: None for key in OUTPUT_AUTHORITY_FIELDS}
            )
            stop_receipt = {
                "schema_version": "sireto-v4.12-fresh-s0-authoritative-launch-receipt-2",
                "phase": failure.phase,
                "reason_code": failure.reason_code,
                "authorization_manifest_sha256": sha256_bytes(auth_raw),
                "execution_lock_path": authorization["execution_lock_absolute_path"],
                "execution_lock_sha256": sha256_bytes(lock_raw),
                "implementation_commit": lock["implementation_commit"],
                "implementation_blob_hashes": implementation_hashes,
                "synthetic_run_id": lock["synthetic_run_id"],
                "attempt_id": lock["attempt_id"],
                "claim_sha256": sha256_bytes(claim_raw),
                "lease_path": os.fspath(lease_path),
                "lease_held_for_spawn": partial_execution is not None,
                "runtime": lock["runtime"],
                "sandbox_profile_sha256": implementation_hashes[
                    "SANDBOX_PROFILE"
                ],
                "effective_sandbox_profile_sha256": profile_record["sha256"],  # type: ignore[index]
                "parent_before_observations": before,
                "worker_receipt": worker_receipt,
                "parent_after_observations": after,
                "stability": observed_stability,
                "canaries": canaries,
                "output_authority": honest_output_authority,
                "macos_limitations": plan["macos_runtime_boundary"][
                    "acknowledged_limitations"
                ],
                "terminal_result": "STOP_SYNTHETIC_SCANNER_SEALER_V412",
                "verdict": "STOP",
                "started_at_utc": started,
                "finished_at_utc": _now_utc(),
            }
            _write_exclusive(receipt_path, stop_receipt, mode=0o400)
            return stop_receipt
        raise
    finally:
        if control_child is not None:
            control_child.close()
        if control_parent is not None:
            control_parent.close()
        for fd in write_fds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        seen: set[int] = set()
        for authority in authorities:
            if authority.fd in seen:
                continue
            seen.add(authority.fd)
            try:
                os.close(authority.fd)
            except OSError:
                pass
        if lock_fd not in seen:
            try:
                os.close(lock_fd)
            except OSError:
                pass
        if lease_fd >= 0:
            try:
                fcntl.flock(lease_fd, fcntl.LOCK_UN)
            finally:
                os.close(lease_fd)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments:
        print(
            "STOP [AUTHORIZATION/AUTHORIZATION_INVALID] launcher accepts no arguments",
            file=sys.stderr,
        )
        return 2
    try:
        receipt = run_authoritative_launch()
    except LauncherStop as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(canonical_json(receipt).decode("utf-8"), end="")
    return 0 if receipt["verdict"] in {"INGESTED", "QUARANTINED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
