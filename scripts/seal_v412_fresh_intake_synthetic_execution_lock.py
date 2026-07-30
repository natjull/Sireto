#!/usr/bin/env python3
"""Build the V4.12 synthetic S0 private runtime and immutable execution lock.

This program is a pre-run authority builder.  It never imports or launches the
launcher/worker and never scans an input row.  Its only permitted child process
is the pinned ``/usr/bin/otool`` used to close Mach-O dependencies.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
import platform
import plistlib
import re
import stat
import struct
import subprocess
import sys
import sysconfig
import uuid
import zlib
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pyarrow as pa


REPOSITORY = Path(__file__).resolve().parents[1]
PLAN_PATH = REPOSITORY / "config/v4_12_fresh_s0_authoritative_run_plan.json"
CORE_PLAN_PATH = (
    REPOSITORY
    / "config/v4_12_fresh_intake_synthetic_scanner_sealer_plan.json"
)
EXPECTED_PLAN_SHA256 = (
    "f73d855b9d6c76f6175cae5e04f2bd2bc61a19a5d78d356ebe99d3d6289f8596"
)
EXPECTED_CONTRACT_SHA256 = (
    "b969a8d552ba060e5b7e24bd1e295abbaf025f1dfbfb7e5683bd5853b689b5df"
)
ALLOWED_ROOT = Path(
    "/Volumes/CATNAT_DATA/SIRETO_RECALL100/fresh_holdout_intake_synthetic"
)
SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")
OTOOL = Path("/usr/bin/otool")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
OPAQUE_ID = re.compile(r"^[a-p]{64}$")
MACHO_MAGICS = {
    b"\xfe\xed\xfa\xce",
    b"\xce\xfa\xed\xfe",
    b"\xfe\xed\xfa\xcf",
    b"\xcf\xfa\xed\xfe",
    b"\xca\xfe\xba\xbe",
    b"\xbe\xba\xfe\xca",
    b"\xca\xfe\xba\xbf",
    b"\xbf\xba\xfe\xca",
}
IMPLEMENTATION_ROLE_PATHS = {
    "LOCK_SEALER": "scripts/seal_v412_fresh_intake_synthetic_execution_lock.py",
    "LAUNCHER": "scripts/launch_v412_fresh_intake_synthetic_scanner_sealer.py",
    "WORKER": "scripts/run_v412_fresh_s0_worker.py",
    "SANDBOX_PROFILE": (
        "config/v4_12_fresh_intake_synthetic_scanner_sealer.sb"
    ),
    "IMPLEMENTATION_TESTS": "tests/test_v412_fresh_s0_authoritative_run.py",
    "AUTHORITATIVE_PLAN": (
        "config/v4_12_fresh_s0_authoritative_run_plan.json"
    ),
    "AUTHORITATIVE_CONTRACT": (
        "docs/v4_12_fresh_s0_authoritative_run_contract.md"
    ),
}
PAYLOAD_ROLE_NAMES = {
    "CONTROL_MANIFEST": "fixture_control_manifest.json",
    "COLLECTION_MANIFEST": "collection_source_manifest.json",
    "SOURCE_MANIFEST": "source_manifest.json",
    "CRM_SAFE_CSV": "crm_safe.csv",
    "EVIDENCE_MANIFEST": "evidence_source_manifest.json",
    "EVIDENCE_PARQUET": "evidence_source.parquet",
}


class LockSealError(RuntimeError):
    """A fail-closed S0 execution-lock construction error."""


def _stop(message: str) -> None:
    if not message.startswith("STOP"):
        message = f"STOP {message}"
    raise LockSealError(message)


def canonical_json(value: Any, *, final_lf: bool = True) -> bytes:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        _stop(f"cannot encode canonical JSON: {exc}")
    return payload + (b"\n" if final_lf else b"")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_canonical_json(
    payload: bytes, label: str, *, final_lf: bool = True
) -> dict[str, Any]:
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"forbidden JSON constant: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        _stop(f"invalid {label}: {exc}")
    if not isinstance(value, dict):
        _stop(f"{label} must be an object")
    if payload != canonical_json(value, final_lf=final_lf):
        _stop(f"{label} is not canonical JSON")
    return value


def _open_anchored(
    path: Path,
    final_flags: int,
    *,
    directory: bool = False,
) -> int:
    """Open an absolute path without following any component symlink."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    if not absolute.is_absolute() or any(
        part in {"", ".", ".."} for part in absolute.parts[1:]
    ):
        _stop(f"invalid anchored path: {path}")
    current = os.open(
        "/",
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        components = absolute.parts[1:]
        if not components:
            if not directory:
                _stop("root cannot be opened as a regular file")
            return os.dup(current)
        for component in components[:-1]:
            child = os.open(
                component,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=current,
            )
            info = os.fstat(child)
            if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
                os.close(child)
                _stop(f"unsafe anchored ancestor: {absolute}")
            os.close(current)
            current = child
        flags = (
            final_flags
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        if directory:
            flags |= getattr(os, "O_DIRECTORY", 0)
        result = os.open(components[-1], flags, dir_fd=current)
        return result
    except OSError as exc:
        _stop(f"anchored open failed for {absolute}: {exc}")
    finally:
        os.close(current)


def _open_parent_anchored(path: Path) -> tuple[int, str]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    if absolute == Path("/"):
        _stop("root has no writable parent")
    return _open_anchored(
        absolute.parent, os.O_RDONLY, directory=True
    ), absolute.name


def _read_regular(
    path: Path,
    *,
    maximum: int = 1 << 31,
    allowed_uids: frozenset[int] | None = None,
    require_single_link: bool = True,
) -> tuple[bytes, os.stat_result]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    descriptor = _open_anchored(absolute, os.O_RDONLY)
    try:
        before = os.fstat(descriptor)
        accepted_uids = (
            frozenset({os.getuid()}) if allowed_uids is None else allowed_uids
        )
        if (
            not stat.S_ISREG(before.st_mode)
            or (require_single_link and before.st_nlink != 1)
            or before.st_uid not in accepted_uids
            or before.st_size < 0
            or before.st_size > maximum
        ):
            _stop(f"unsafe regular file identity: {absolute}")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                _stop(f"short read: {absolute}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            _stop(f"file grew while reading: {absolute}")
        after = os.fstat(descriptor)
        fields = (
            "st_dev",
            "st_ino",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
            "st_mode",
            "st_uid",
            "st_nlink",
        )
        if any(getattr(before, key) != getattr(after, key) for key in fields):
            _stop(f"file changed while reading: {absolute}")
        return b"".join(chunks), before
    finally:
        os.close(descriptor)


def _full_fsync(descriptor: int) -> None:
    os.fsync(descriptor)
    full = getattr(__import__("fcntl"), "F_FULLFSYNC", None)
    if full is None:
        _stop("F_FULLFSYNC unavailable")
    try:
        __import__("fcntl").fcntl(descriptor, full)
    except OSError as exc:
        _stop(f"F_FULLFSYNC failed: {exc}")


def _sync_directory(path: Path) -> None:
    descriptor = _open_anchored(path, os.O_RDONLY, directory=True)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mkdir_exclusive(path: Path, mode: int = 0o700) -> None:
    parent_fd, name = _open_parent_anchored(path)
    try:
        os.mkdir(name, mode=mode, dir_fd=parent_fd)
    except OSError as exc:
        _stop(f"cannot create directory {path}: {exc}")
    finally:
        os.close(parent_fd)
    descriptor = _open_anchored(path, os.O_RDONLY, directory=True)
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(info.st_mode)
            or stat.S_ISLNK(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != mode
        ):
            _stop(f"created path is not a private directory: {path}")
    finally:
        os.close(descriptor)


def _ensure_private_directory(path: Path) -> None:
    if os.path.lexists(path):
        descriptor = _open_anchored(path, os.O_RDONLY, directory=True)
        try:
            info = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        if (
            not stat.S_ISDIR(info.st_mode)
            or stat.S_ISLNK(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o700
        ):
            _stop(f"unsafe existing directory: {path}")
        return
    parent = path.parent
    if parent != path and not os.path.lexists(parent):
        _ensure_private_directory(parent)
    _mkdir_exclusive(path)
    _sync_directory(parent)


def _write_exclusive(path: Path, payload: bytes, mode: int) -> None:
    if mode not in {0o400, 0o500, 0o600}:
        _stop(f"unsupported sealed mode: {mode:o}")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    parent_fd, name = _open_parent_anchored(path)
    try:
        try:
            descriptor = os.open(name, flags, mode, dir_fd=parent_fd)
        except OSError as exc:
            _stop(f"cannot create {path}: {exc}")
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    _stop(f"short write: {path}")
                view = view[written:]
            os.fchmod(descriptor, mode)
            _full_fsync(descriptor)
            info = os.fstat(descriptor)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or info.st_size != len(payload)
                or stat.S_IMODE(info.st_mode) != mode
            ):
                _stop(f"sealed output identity mismatch: {path}")
        finally:
            os.close(descriptor)
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


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


def volume_uuid(path: Path) -> str:
    """Return the Darwin volume UUID without invoking an external program."""

    if platform.system() != "Darwin":
        _stop("volume UUID verification requires macOS")
    descriptor = _open_anchored(
        path,
        os.O_RDONLY,
        directory=path == Path("/"),
    )
    try:
        before = os.fstat(descriptor)
        attributes = _AttrList(
            5, 0, 0, 0x80000000 | 0x00040000, 0, 0, 0
        )
        buffer = ctypes.create_string_buffer(4 + 16)
        libc = ctypes.CDLL(None, use_errno=True)
        function = getattr(libc, "fgetattrlist", None)
        if function is None:
            _stop("fgetattrlist unavailable")
        function.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(_AttrList),
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_ulong,
        ]
        function.restype = ctypes.c_int
        if function(
            descriptor,
            ctypes.byref(attributes),
            ctypes.byref(buffer),
            ctypes.sizeof(buffer),
            0,
        ):
            error = ctypes.get_errno()
            _stop(f"cannot read volume UUID: errno={error}")
        length = struct.unpack_from("=I", buffer.raw, 0)[0]
        if length < 20:
            _stop("short volume UUID attribute")
        value = str(uuid.UUID(bytes=buffer.raw[4:20])).lower()
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            _stop("volume anchor changed")
        return value
    finally:
        os.close(descriptor)


def _mode_string(mode: int) -> str:
    return f"{stat.S_IMODE(mode):04o}"


def fd_identity(path: Path, info: os.stat_result | None = None) -> dict[str, Any]:
    if info is None:
        info = path.stat(follow_symlinks=False)
    return {
        "device": info.st_dev,
        "inode": info.st_ino,
        "size_bytes": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
        "uid": info.st_uid,
        "volume_uuid": volume_uuid(path),
        "link_count": info.st_nlink,
        "mode": _mode_string(info.st_mode),
    }


def _load_plans() -> tuple[dict[str, Any], dict[str, Any]]:
    plan_bytes, _ = _read_regular(PLAN_PATH)
    if sha256_bytes(plan_bytes) != EXPECTED_PLAN_SHA256:
        _stop("authoritative plan hash mismatch")
    plan = parse_canonical_json(plan_bytes, "authoritative plan")
    contract = REPOSITORY / plan["contract"]["path"]
    contract_bytes, _ = _read_regular(contract)
    if (
        sha256_bytes(contract_bytes) != EXPECTED_CONTRACT_SHA256
        or plan["contract"]["sha256"] != EXPECTED_CONTRACT_SHA256
    ):
        _stop("authoritative contract hash mismatch")
    core_bytes, _ = _read_regular(CORE_PLAN_PATH)
    core_plan = parse_canonical_json(core_bytes, "core plan")
    for pin in plan["core"]["pins"].values():
        raw, _ = _read_regular(REPOSITORY / pin["path"])
        if sha256_bytes(raw) != pin["sha256"]:
            _stop(f"core pin mismatch: {pin['path']}")
    return plan, core_plan


def opaque_digest(domain: str, payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        domain.encode("utf-8") + canonical_json(payload, final_lf=False)
    ).hexdigest()
    return "".join(chr(ord("a") + int(char, 16)) for char in digest)


def _git_object_loose(repository: Path, object_id: str) -> tuple[str, bytes]:
    if not HEX40.fullmatch(object_id):
        _stop("invalid Git object id")
    object_path = repository / ".git" / "objects" / object_id[:2] / object_id[2:]
    payload, _ = _read_regular(object_path, maximum=1 << 30)
    try:
        decoded = zlib.decompress(payload)
    except zlib.error as exc:
        _stop(f"cannot decompress loose Git object {object_id}: {exc}")
    header, separator, body = decoded.partition(b"\0")
    if not separator:
        _stop(f"invalid loose Git object {object_id}")
    kind_raw, size_raw = header.split(b" ", 1)
    if int(size_raw) != len(body):
        _stop(f"Git object size mismatch: {object_id}")
    if hashlib.sha1(decoded).hexdigest() != object_id:
        _stop(f"Git object digest mismatch: {object_id}")
    return kind_raw.decode("ascii"), body


def _git_blob_at_commit(
    repository: Path, commit_id: str, relative_path: str
) -> tuple[bytes, int]:
    kind, commit = _git_object_loose(repository, commit_id)
    if kind != "commit":
        _stop("implementation_commit is not a commit")
    first = commit.splitlines()[0]
    if not first.startswith(b"tree ") or len(first) != 45:
        _stop("commit tree header invalid")
    current = first[5:].decode("ascii")
    parts = Path(relative_path).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        _stop(f"invalid repository path: {relative_path}")
    selected_mode = 0
    for index, component in enumerate(parts):
        kind, tree = _git_object_loose(repository, current)
        if kind != "tree":
            _stop(f"Git path is not a tree: {relative_path}")
        cursor = 0
        found: tuple[int, str] | None = None
        while cursor < len(tree):
            space = tree.find(b" ", cursor)
            nul = tree.find(b"\0", space + 1)
            if space < 0 or nul < 0 or nul + 21 > len(tree):
                _stop("malformed Git tree")
            mode = int(tree[cursor:space], 8)
            name = tree[space + 1 : nul].decode("utf-8", "strict")
            object_id = tree[nul + 1 : nul + 21].hex()
            cursor = nul + 21
            if name == component:
                if found is not None:
                    _stop("duplicate Git tree entry")
                found = (mode, object_id)
        if found is None:
            _stop(f"path absent from implementation commit: {relative_path}")
        selected_mode, current = found
        if index < len(parts) - 1 and selected_mode != 0o40000:
            _stop(f"non-directory Git prefix: {relative_path}")
    kind, blob = _git_object_loose(repository, current)
    if kind != "blob" or selected_mode not in {0o100644, 0o100755}:
        _stop(f"implementation path is not a regular blob: {relative_path}")
    return blob, selected_mode


def implementation_blobs(
    plan: Mapping[str, Any], implementation_commit: str
) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    if not HEX40.fullmatch(implementation_commit):
        _stop("implementation commit must be 40 lowercase hex characters")
    records: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    expected_roles = plan["execution_lock"]["implementation_blob_roles"]
    if expected_roles != list(IMPLEMENTATION_ROLE_PATHS):
        _stop("implementation role order differs from source contract")
    for role in expected_roles:
        relative = IMPLEMENTATION_ROLE_PATHS[role]
        blob, git_mode = _git_blob_at_commit(
            REPOSITORY, implementation_commit, relative
        )
        current, info = _read_regular(REPOSITORY / relative)
        if current != blob:
            _stop(f"working source differs from implementation blob: {relative}")
        executable = git_mode == 0o100755
        if executable != bool(stat.S_IMODE(info.st_mode) & 0o111):
            _stop(f"working source executable mode differs from Git: {relative}")
        records.append(
            {
                "role": role,
                "path": relative,
                "size_bytes": len(blob),
                "sha256": sha256_bytes(blob),
                "mode": "0755" if executable else "0644",
            }
        )
        payloads[role] = blob
    return records, payloads


def _validate_fixture(
    plan: Mapping[str, Any], core_plan: Mapping[str, Any], root: Path
) -> tuple[str, str, str, list[tuple[str, Path, bytes, os.stat_result]]]:
    core_plan_bytes, _ = _read_regular(CORE_PLAN_PATH)
    run_id = opaque_digest(
        core_plan["ids"]["run"]["domain"],
        {
            "fixture_spec_sha256": core_plan["control_manifest"][
                "fixture_spec_sha256"
            ],
            "plan_sha256": sha256_bytes(core_plan_bytes),
        },
    )
    if not OPAQUE_ID.fullmatch(run_id):
        _stop("derived synthetic run id invalid")
    package = root / "inbox" / run_id / "package"
    control_path = (
        root / "control" / run_id / "fixture_control_manifest.json"
    )
    package_fd = _open_anchored(package, os.O_RDONLY, directory=True)
    control_fd = _open_anchored(
        control_path.parent, os.O_RDONLY, directory=True
    )
    try:
        with os.scandir(package_fd) as iterator:
            package_entries = list(iterator)
        package_names = {
            entry.name
            for entry in package_entries
            if entry.is_file(follow_symlinks=False)
        }
        non_files = [
            entry.name
            for entry in package_entries
            if not entry.is_file(follow_symlinks=False)
        ]
        with os.scandir(control_fd) as iterator:
            control_entries = list(iterator)
        control_names = {
            entry.name
            for entry in control_entries
            if entry.is_file(follow_symlinks=False)
        }
    except OSError as exc:
        _stop(f"cannot enumerate synthetic fixture package: {exc}")
    finally:
        os.close(package_fd)
        os.close(control_fd)
    if (
        package_names != set(PAYLOAD_ROLE_NAMES.values()) - {
            "fixture_control_manifest.json"
        }
        or non_files
        or len(control_entries) != 1
        or control_names != {"fixture_control_manifest.json"}
    ):
        _stop("synthetic fixture package has missing or extra entries")
    control_bytes, control_info = _read_regular(control_path)
    control = parse_canonical_json(control_bytes, "fixture control manifest")
    if list(control) != sorted(control):
        # Canonical JSON sorts encoded keys; object semantics are checked below.
        pass
    if set(control) != set(core_plan["control_manifest"]["exact_fields"]):
        _stop("fixture control field set mismatch")
    required = core_plan["control_manifest"]["required"]
    if (
        control["schema_version"] != core_plan["control_manifest"]["schema"]
        or control["synthetic_run_id"] != run_id
        or control["logical_time_utc"] != core_plan["fixture"]["logical_time_utc"]
        or control["fixture_spec_sha256"]
        != core_plan["control_manifest"]["fixture_spec_sha256"]
        or any(control[key] != value for key, value in required.items())
    ):
        _stop("fixture control invariant mismatch")
    payloads: list[tuple[str, Path, bytes, os.stat_result]] = [
        ("CONTROL_MANIFEST", control_path, control_bytes, control_info)
    ]
    role_to_control = {
        "COLLECTION_MANIFEST": "collection_source_manifest_sha256",
        "SOURCE_MANIFEST": "source_manifest_sha256",
        "CRM_SAFE_CSV": "crm_safe_csv_sha256",
        "EVIDENCE_MANIFEST": "evidence_source_manifest_sha256",
        "EVIDENCE_PARQUET": "evidence_source_parquet_sha256",
    }
    for role in plan["fd_protocol"]["worker_payload_roles_exact_order"][1:]:
        path = package / PAYLOAD_ROLE_NAMES[role]
        raw, info = _read_regular(path)
        if sha256_bytes(raw) != control[role_to_control[role]]:
            _stop(f"fixture payload hash mismatch: {role}")
        if role.endswith("MANIFEST"):
            parsed = parse_canonical_json(raw, f"{role} payload")
            if parsed.get("synthetic_run_id") != run_id:
                _stop(f"fixture payload run mismatch: {role}")
        payloads.append((role, path, raw, info))
    control_sha = sha256_bytes(control_bytes)
    attempt_id = opaque_digest(
        core_plan["ids"]["attempt"]["domain"],
        {
            "synthetic_run_id": run_id,
            "fixture_control_manifest_sha256": control_sha,
            "logical_time_utc": control["logical_time_utc"],
        },
    )
    if not OPAQUE_ID.fullmatch(attempt_id):
        _stop("derived attempt id invalid")
    return run_id, attempt_id, control["logical_time_utc"], payloads


def _is_macho(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(4) in MACHO_MAGICS
    except OSError:
        return False


def _run_otool(arguments: Sequence[str]) -> str:
    _raw, info = _read_regular(
        OTOOL, allowed_uids=frozenset({0}), require_single_link=False
    )
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != 0
        or stat.S_IMODE(info.st_mode) & 0o022
    ):
        _stop("pinned /usr/bin/otool identity is unsafe")
    result = subprocess.run(
        [str(OTOOL), *arguments],
        check=False,
        capture_output=True,
        env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
        close_fds=True,
    )
    if result.returncode or result.stderr:
        _stop(f"otool failed for {' '.join(arguments)}")
    try:
        return result.stdout.decode("utf-8")
    except UnicodeDecodeError:
        _stop("otool output is not UTF-8")


def _macho_dependencies(path: Path) -> list[str]:
    lines = _run_otool(["-L", str(path)]).splitlines()
    if not lines:
        _stop(f"empty otool -L output: {path}")
    dependencies: list[str] = []
    for line in lines[1:]:
        value = line.strip().split(" (", 1)[0]
        if value:
            dependencies.append(value)
    return dependencies


def _macho_rpaths(path: Path) -> list[str]:
    lines = _run_otool(["-l", str(path)]).splitlines()
    result: list[str] = []
    for index, line in enumerate(lines):
        if line.strip() == "cmd LC_RPATH":
            for candidate in lines[index + 1 : index + 8]:
                stripped = candidate.strip()
                if stripped.startswith("path "):
                    result.append(stripped[5:].split(" (", 1)[0])
                    break
    return result


def _expand_install_name(
    value: str, loader: Path, executable: Path, rpaths: Sequence[str]
) -> Path | None:
    if value.startswith("/System/") or value.startswith("/usr/lib/"):
        return None
    replacements = {
        "@loader_path": str(loader.parent),
        "@executable_path": str(executable.parent),
    }
    for marker, replacement in replacements.items():
        if value == marker or value.startswith(marker + "/"):
            return Path(replacement + value[len(marker) :]).resolve()
    if value.startswith("@rpath/"):
        tail = value[len("@rpath/") :]
        candidates: list[Path] = []
        for rpath in rpaths:
            base = rpath
            for marker, replacement in replacements.items():
                if base == marker or base.startswith(marker + "/"):
                    base = replacement + base[len(marker) :]
            if base.startswith("@"):
                _stop(f"unresolved nested rpath for {loader}: {base}")
            candidate = (Path(base) / tail).resolve()
            if candidate.is_file():
                candidates.append(candidate)
        unique = list(dict.fromkeys(candidates))
        if len(unique) != 1:
            _stop(f"ambiguous or unresolved @rpath dependency: {value}")
        return unique[0]
    if value.startswith("@"):
        _stop(f"unsupported Mach-O install name: {value}")
    if value.startswith("/"):
        return Path(value).resolve()
    _stop(f"relative Mach-O install name forbidden: {value}")


def _iter_source_files(root: Path) -> Iterable[Path]:
    pending = [Path(os.path.abspath(os.fspath(root)))]
    excluded = {"__pycache__", "test", "tests", "site-packages"}
    while pending:
        current = pending.pop()
        descriptor = _open_anchored(current, os.O_RDONLY, directory=True)
        try:
            with os.scandir(descriptor) as iterator:
                entries = sorted(
                    list(iterator), key=lambda entry: entry.name.encode("utf-8")
                )
        finally:
            os.close(descriptor)
        directories: list[Path] = []
        for entry in entries:
            path = current / entry.name
            if entry.name in excluded:
                continue
            if entry.is_dir(follow_symlinks=False):
                directories.append(path)
                continue
            if entry.name.endswith((".pyc", ".pyo")):
                continue
            if entry.is_symlink():
                target = path.resolve(strict=True)
                payload, info = _read_regular(target)
                del payload
                if not stat.S_ISREG(info.st_mode):
                    _stop(f"source symlink is not a regular file: {path}")
                # Preserve the logical relative name while copying the
                # resolved target bytes; the private tree itself has no links.
                yield path
            elif entry.is_file(follow_symlinks=False):
                yield path
            else:
                _stop(f"unsupported runtime source entry: {path}")
        pending.extend(reversed(directories))


def _seatbelt_literal(value: str) -> str:
    if "\n" in value or "\r" in value or "\0" in value:
        _stop("sandbox path contains forbidden control characters")
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _render_profile(
    template: bytes,
    plan: Mapping[str, Any],
    root: Path,
    run_id: str,
    runtime_root: Path,
) -> bytes:
    try:
        text = template.decode("utf-8")
    except UnicodeDecodeError:
        _stop("sandbox template is not UTF-8")
    replacements = {
        "@@PRIVATE_RUNTIME_ROOT@@": str(runtime_root),
        "@@SEALED_RUN_ROOT@@": str(root / "sealed" / run_id),
        "@@SCAN_RUN_ROOT@@": str(root / "scan" / run_id),
        "@@QUARANTINE_RUN_ROOT@@": str(root / "quarantine" / run_id),
        "@@WORKER_AUDIT_ROOT@@": str(root / "audit" / run_id / "worker"),
        "@@TMP_RUN_ROOT@@": str(root / "tmp" / run_id),
    }
    expected = plan["sandbox_profile_derivation"]["placeholder_order"]
    if list(replacements) != expected:
        _stop("sandbox placeholder order mismatch")
    for marker in expected:
        if text.count(marker) < 1:
            _stop(f"sandbox placeholder absent: {marker}")
        text = text.replace(marker, _seatbelt_literal(replacements[marker]))
    if "@@" in text:
        _stop("unknown or residual sandbox placeholder")
    return text.rstrip("\r\n").encode("utf-8") + b"\n"


def _copy_runtime_file(
    source: Path,
    destination: Path,
    role: str,
    runtime_root: Path,
    records: dict[str, dict[str, Any]],
    *,
    mode: int | None = None,
) -> None:
    source = source.resolve()
    payload, source_info = _read_regular(source)
    relative = destination.relative_to(runtime_root).as_posix()
    if relative in records:
        if records[relative]["sha256"] != sha256_bytes(payload):
            _stop(f"runtime destination collision: {relative}")
        return
    selected_mode = mode
    if selected_mode is None:
        selected_mode = 0o500 if (
            stat.S_IMODE(source_info.st_mode) & 0o111 or _is_macho(source)
        ) else 0o400
    _ensure_private_directory(destination.parent)
    _write_exclusive(destination, payload, selected_mode)
    records[relative] = {
        "role": role,
        "source_path": str(source),
        "private_relative_path": relative,
        "size_bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "mode": f"{selected_mode:04o}",
    }


def _build_private_runtime(
    plan: Mapping[str, Any],
    implementation_commit: str,
    implementation_payloads: Mapping[str, bytes],
    root: Path,
    run_id: str,
) -> tuple[dict[str, Any], Path, str]:
    runtime_root = root / "runtime" / run_id
    if runtime_root.exists():
        _stop("private runtime already exists")
    _ensure_private_directory(runtime_root.parent)
    _mkdir_exclusive(runtime_root)
    rootfs = runtime_root / "rootfs"
    app = runtime_root / "app"
    profile_dir = runtime_root / "profile"
    for path in (rootfs, app, profile_dir):
        _mkdir_exclusive(path)
    records: dict[str, dict[str, Any]] = {}
    executable = Path(sys.executable).resolve()
    if (
        executable
        != Path(
            "/opt/homebrew/Cellar/python@3.14/3.14.3_1/Frameworks/"
            "Python.framework/Versions/3.14/bin/python3.14"
        )
        or platform.machine() != "arm64"
        or platform.python_version() != "3.14.3"
        or pa.__version__ != "23.0.1"
    ):
        _stop("pinned Python/PyArrow runtime differs from this Mac")

    def rootfs_destination(source_path: Path) -> Path:
        resolved = source_path.resolve()
        if not resolved.is_absolute():
            _stop("runtime source is not absolute")
        return rootfs.joinpath(*resolved.parts[1:])

    _copy_runtime_file(
        executable,
        rootfs_destination(executable),
        "PYTHON_EXECUTABLE",
        runtime_root,
        records,
        mode=0o500,
    )
    stdlib_logical = Path(sysconfig.get_path("stdlib"))
    stdlib_source = stdlib_logical.resolve()
    for source in _iter_source_files(stdlib_source):
        try:
            relative = source.relative_to(stdlib_source)
        except ValueError:
            relative = Path(source.name)
        _copy_runtime_file(
            source,
            # The launcher derives PYTHONHOME from the real Python executable.
            # Preserve the real Cellar prefix, not Homebrew's ``opt`` symlink.
            rootfs.joinpath(*stdlib_source.parts[1:]) / relative,
            "PYTHON_STDLIB",
            runtime_root,
            records,
        )
    spec = importlib.util.find_spec("pyarrow")
    if (
        spec is None
        or spec.submodule_search_locations is None
        or len(spec.submodule_search_locations) != 1
    ):
        _stop("cannot resolve pinned PyArrow package")
    pyarrow_logical = Path(next(iter(spec.submodule_search_locations)))
    pyarrow_source = pyarrow_logical.resolve()
    for source in _iter_source_files(pyarrow_source):
        try:
            relative = source.relative_to(pyarrow_source)
        except ValueError:
            relative = Path(source.name)
        _copy_runtime_file(
            source,
            rootfs.joinpath(*pyarrow_logical.parts[1:]) / relative,
            "PYARROW",
            runtime_root,
            records,
        )

    repo_sources = {
        "WORKER": IMPLEMENTATION_ROLE_PATHS["WORKER"],
        "CORE_SCANNER": plan["core"]["pins"]["scanner"]["path"],
        "CORE_FIXTURE_BUILDER": plan["core"]["pins"]["fixture_builder"]["path"],
        "CORE_PLAN": plan["core"]["pins"]["plan"]["path"],
        "CORE_CONTRACT": plan["core"]["pins"]["contract"]["path"],
    }
    for role, relative in repo_sources.items():
        if role == "WORKER":
            payload = implementation_payloads["WORKER"]
            source = REPOSITORY / relative
        else:
            payload, _ = _read_regular(REPOSITORY / relative)
            source = REPOSITORY / relative
        destination = app / relative
        _ensure_private_directory(destination.parent)
        _write_exclusive(destination, payload, 0o500 if role == "WORKER" else 0o400)
        records[destination.relative_to(runtime_root).as_posix()] = {
            "role": role,
            "source_path": str(source.resolve()),
            "private_relative_path": destination.relative_to(runtime_root).as_posix(),
            "size_bytes": len(payload),
            "sha256": sha256_bytes(payload),
            "mode": "0500" if role == "WORKER" else "0400",
        }

    # Close every non-system dependency reachable from copied Mach-O files.
    pending = [
        runtime_root / relative
        for relative in sorted(records, key=lambda value: value.encode("utf-8"))
        if _is_macho(runtime_root / relative)
    ]
    visited: set[str] = set()
    framework_source: Path | None = None
    while pending:
        private_loader = pending.pop(0)
        loader_record = records[private_loader.relative_to(runtime_root).as_posix()]
        source_loader = Path(loader_record["source_path"])
        if str(source_loader) in visited:
            continue
        visited.add(str(source_loader))
        rpaths = _macho_rpaths(source_loader)
        for install_name in _macho_dependencies(source_loader):
            dependency = _expand_install_name(
                install_name, source_loader, executable, rpaths
            )
            if dependency is None:
                continue
            if not dependency.is_file():
                _stop(f"Mach-O dependency is absent: {dependency}")
            if (
                framework_source is None
                and "Python.framework/" in str(dependency)
            ):
                role = "PYTHON_FRAMEWORK"
                framework_source = dependency
            else:
                role = "MACHO_DEPENDENCY"
            destination = rootfs_destination(dependency)
            before = set(records)
            _copy_runtime_file(
                dependency, destination, role, runtime_root, records
            )
            relative = destination.relative_to(runtime_root).as_posix()
            if relative not in before and _is_macho(destination):
                pending.append(destination)
    if framework_source is None:
        _stop("Python framework dependency was not closed")

    profile_template = implementation_payloads["SANDBOX_PROFILE"]
    effective = _render_profile(
        profile_template, plan, root, run_id, runtime_root
    )
    effective_path = profile_dir / "effective.sb"
    _write_exclusive(effective_path, effective, 0o400)
    effective_relative = effective_path.relative_to(runtime_root).as_posix()
    records[effective_relative] = {
        "role": "EFFECTIVE_SANDBOX_PROFILE",
        "source_path": str(effective_path),
        "private_relative_path": effective_relative,
        "size_bytes": len(effective),
        "sha256": sha256_bytes(effective),
        "mode": "0400",
    }
    ordered = [
        records[key]
        for key in sorted(records, key=lambda value: value.encode("utf-8"))
    ]
    projection = {
        "schema_version": (
            "sireto-v4.12-fresh-s0-private-runtime-manifest-1"
        ),
        "implementation_commit": implementation_commit,
        "record_count": len(ordered),
        "records": ordered,
    }
    manifest = {
        **projection,
        "dependency_closure_sha256": sha256_bytes(
            canonical_json(projection, final_lf=False)
        ),
    }
    manifest_path = runtime_root / "private_runtime_manifest.json"
    _write_exclusive(manifest_path, canonical_json(manifest), 0o400)
    return manifest, manifest_path, sha256_bytes(effective)


def _create_canaries(
    plan: Mapping[str, Any], root: Path, run_id: str
) -> tuple[dict[str, Any], Path]:
    forbidden = root / "canaries" / "forbidden"
    if os.path.lexists(forbidden):
        _stop("canary root already exists")
    _ensure_private_directory(forbidden.parent)
    _mkdir_exclusive(forbidden)
    targets = plan["canary_matrix"]["synthetic_target_by_runtime_code"]
    records: list[dict[str, Any]] = []
    for code in plan["canary_matrix"]["runtime_codes_exact_order"]:
        rendered = (
            targets[code]
            .replace("<allowed_root>", str(root))
            .replace("<synthetic_run_id>", run_id)
        )
        if code == "DENY_NETWORK":
            record = {
                "code": code,
                "kind": "NETWORK_CAPABILITY",
                "absolute_path_or_capability": rendered,
                "identity": None,
                "size_bytes": None,
                "sha256": None,
            }
        elif code == "DENY_WRITE_PARENT_AUDIT":
            path = Path(rendered)
            if os.path.lexists(path):
                _stop("expected-absent write canary already exists")
            record = {
                "code": code,
                "kind": "EXPECTED_ABSENT",
                "absolute_path_or_capability": rendered,
                "identity": None,
                "size_bytes": None,
                "sha256": None,
            }
        elif code == "DENY_PARENT_ENUMERATION":
            record = {
                "code": code,
                "kind": "EXISTING_DIRECTORY",
                "absolute_path_or_capability": rendered,
                # Directory link counts are not mono-link authority records.
                "identity": None,
                "size_bytes": None,
                "sha256": None,
            }
        else:
            path = Path(rendered)
            _ensure_private_directory(path.parent)
            payload = canonical_json(
                {
                    "schema_version": "sireto-v4.12-fresh-s0-canary-1",
                    "synthetic_run_id": run_id,
                    "code": code,
                }
            )
            _write_exclusive(path, payload, 0o400)
            raw, info = _read_regular(path)
            record = {
                "code": code,
                "kind": "EXISTING_FILE",
                "absolute_path_or_capability": rendered,
                "identity": fd_identity(path, info),
                "size_bytes": len(raw),
                "sha256": sha256_bytes(raw),
            }
        records.append(record)
    manifest = {
        "schema_version": "sireto-v4.12-fresh-s0-canary-manifest-1",
        "synthetic_run_id": run_id,
        "ordered_records": records,
        "record_count": len(records),
        "records_sha256": sha256_bytes(
            canonical_json(records, final_lf=False)
        ),
    }
    manifest_path = root / "control" / run_id / "canary_manifest.json"
    _write_exclusive(manifest_path, canonical_json(manifest), 0o400)
    return manifest, manifest_path


def _substitute_lock_paths(
    plan: Mapping[str, Any], root: Path, run_id: str, attempt_id: str
) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, template in plan["lock_values"]["paths"].items():
        value = (
            template.replace("<allowed_root>", str(root))
            .replace("<synthetic_run_id>", run_id)
            .replace("<attempt_id>", attempt_id)
        )
        if "<" in value or ">" in value or not Path(value).is_absolute():
            _stop(f"unresolved lock path: {key}")
        result[key] = value
    return result


def _prepare_run_directories(
    root: Path, run_id: str, attempt_id: str
) -> None:
    paths = [
        root / "sealed" / run_id,
        root / "scan" / run_id,
        root / "quarantine" / run_id,
        root / "audit" / run_id / "worker",
        root / "tmp" / run_id,
        root / "audit" / run_id / "parent" / "spec",
        root / "audit" / run_id / "parent" / "claims",
        root / "audit" / run_id / "parent" / "leases",
        root / "audit" / run_id / "parent" / "launch_receipts",
    ]
    for path in paths:
        _ensure_private_directory(path)
        descriptor = _open_anchored(path, os.O_RDONLY, directory=True)
        try:
            with os.scandir(descriptor) as iterator:
                if next(iterator, None) is not None:
                    _stop(f"pre-run directory is not empty: {path}")
        finally:
            os.close(descriptor)
    forbidden = (
        root
        / "audit"
        / run_id
        / "parent"
        / "claims"
        / f"{attempt_id}.json"
    )
    if os.path.lexists(forbidden):
        _stop("claim exists before authorization")


def _input_record(
    role: str, path: Path
) -> dict[str, Any]:
    raw, info = _read_regular(path)
    identity = fd_identity(path, info)
    return {
        "role": role,
        "absolute_path": str(path),
        "identity": identity,
        "volume_uuid": identity["volume_uuid"],
        "size_bytes": len(raw),
        "sha256": sha256_bytes(raw),
    }


def _system_runtime(root: Path) -> dict[str, Any]:
    system_plist, _ = _read_regular(
        Path("/System/Library/CoreServices/SystemVersion.plist"),
        allowed_uids=frozenset({0}),
        require_single_link=False,
    )
    try:
        system = plistlib.loads(system_plist)
    except (plistlib.InvalidFileException, ValueError):
        _stop("cannot parse macOS SystemVersion.plist")
    if not isinstance(system, dict):
        _stop("invalid macOS system version payload")

    def directory_device(path: Path) -> int:
        descriptor = _open_anchored(path, os.O_RDONLY, directory=True)
        try:
            return os.fstat(descriptor).st_dev
        finally:
            os.close(descriptor)

    return {
        "name": "macOS",
        "product_version": str(system.get("ProductVersion", "")),
        "build_version": str(system.get("ProductBuildVersion", "")),
        "kernel_release": platform.release(),
        "machine": platform.machine(),
        "uid": os.getuid(),
        "volumes": {
            "repository": {
                "device": directory_device(REPOSITORY),
                "volume_uuid": volume_uuid(REPOSITORY),
            },
            "run": {
                "device": directory_device(root),
                "volume_uuid": volume_uuid(root),
            },
            "system_tcb": {
                "device": os.stat("/").st_dev,
                "volume_uuid": volume_uuid(Path("/")),
            },
        },
    }


def seal_execution_lock(implementation_commit: str) -> dict[str, Any]:
    """Construct all pre-lock S0 authorities and seal one immutable lock."""

    old_umask = os.umask(0o077)
    try:
        plan, core_plan = _load_plans()
        root = ALLOWED_ROOT
        root_fd = _open_anchored(root, os.O_RDONLY, directory=True)
        try:
            root_info = os.fstat(root_fd)
        finally:
            os.close(root_fd)
        if (
            not stat.S_ISDIR(root_info.st_mode)
            or stat.S_ISLNK(root_info.st_mode)
            or root_info.st_uid != os.getuid()
            or stat.S_IMODE(root_info.st_mode) != 0o700
        ):
            _stop("authoritative synthetic root is absent or unsafe")
        implementation, implementation_payloads = implementation_blobs(
            plan, implementation_commit
        )
        run_id, attempt_id, logical_time, fixture_inputs = _validate_fixture(
            plan, core_plan, root
        )
        _prepare_run_directories(root, run_id, attempt_id)
        private_manifest, private_manifest_path, effective_profile_sha = (
            _build_private_runtime(
                plan,
                implementation_commit,
                implementation_payloads,
                root,
                run_id,
            )
        )
        _canary_manifest, canary_manifest_path = _create_canaries(
            plan, root, run_id
        )
        input_paths = {
            role: path for role, path, _raw, _info in fixture_inputs
        }
        input_paths["PRIVATE_RUNTIME_MANIFEST"] = private_manifest_path
        input_paths["CANARY_MANIFEST"] = canary_manifest_path
        read_inputs = [
            _input_record(role, input_paths[role])
            for role in plan["fd_protocol"]["lock_input_roles_exact_order"]
        ]
        sandbox_exec_raw, sandbox_exec_info = _read_regular(
            SANDBOX_EXEC, allowed_uids=frozenset({0})
        )
        if stat.S_IMODE(sandbox_exec_info.st_mode) & 0o022:
            _stop("sandbox-exec is group/other writable")
        sandbox_exec_record = {
            "role": "SANDBOX_EXEC",
            "path": str(SANDBOX_EXEC),
            "size_bytes": len(sandbox_exec_raw),
            "sha256": sha256_bytes(sandbox_exec_raw),
            "mode": _mode_string(sandbox_exec_info.st_mode),
        }
        template_sha = next(
            record["sha256"]
            for record in implementation
            if record["role"] == "SANDBOX_PROFILE"
        )
        sandbox = dict(plan["lock_values"]["sandbox"])
        sandbox["template_profile_sha256"] = template_sha
        sandbox["effective_profile_sha256"] = effective_profile_sha
        if any(
            isinstance(value, str) and value.startswith("DERIVE_FROM_")
            for value in sandbox.values()
        ):
            _stop("sandbox derivation token remained in execution lock")
        lock = {
            "schema_version": plan["execution_lock"]["schema_version"],
            "purpose": "SIRETO_V412_FRESH_SYNTHETIC_S0_AUTHORITATIVE_RUN",
            "status": plan["execution_lock"]["status"],
            "implementation_commit": implementation_commit,
            "implementation_blobs": implementation,
            "core": plan["core"],
            "runtime": {
                "system": _system_runtime(root),
                "python_version": platform.python_version(),
                "pyarrow_version": pa.__version__,
                "sandbox_exec": sandbox_exec_record,
                "private_runtime_manifest": private_manifest,
            },
            "read_fds": read_inputs,
            "paths": _substitute_lock_paths(
                plan, root, run_id, attempt_id
            ),
            "sandbox": sandbox,
            "policy": plan["lock_values"]["policy"],
            "synthetic_run_id": run_id,
            "attempt_id": attempt_id,
            "logical_time_utc": logical_time,
        }
        if set(lock) != set(plan["execution_lock"]["exact_fields"]):
            _stop("execution lock field set mismatch")
        lock_bytes = canonical_json(lock)
        if b"UNIMPLEMENTED" in lock_bytes or b"DERIVE_FROM_" in lock_bytes:
            _stop("execution lock contains a forbidden sentinel")
        lock_path = root / "control" / run_id / "execution_lock.json"
        _write_exclusive(lock_path, lock_bytes, 0o400)
        return {
            "schema_version": (
                "sireto-v4.12-fresh-s0-lock-sealer-result-1"
            ),
            "implementation_commit": implementation_commit,
            "synthetic_run_id": run_id,
            "attempt_id": attempt_id,
            "execution_lock_absolute_path": str(lock_path),
            "execution_lock_sha256": sha256_bytes(lock_bytes),
            "private_runtime_manifest_sha256": sha256_file(
                private_manifest_path
            ),
            "canary_manifest_sha256": sha256_file(canary_manifest_path),
            "status": "SEALED_AUTHORITY_READY_TO_AUTHORIZE",
        }
    finally:
        os.umask(old_umask)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--implementation-commit",
        required=True,
        help="exact implementation commit, 40 lowercase hexadecimal digits",
    )
    arguments = parser.parse_args()
    try:
        result = seal_execution_lock(arguments.implementation_commit)
    except LockSealError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(canonical_json(result).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
