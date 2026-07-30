#!/usr/bin/env python3
"""FD-only worker for the authoritative V4.12 synthetic S0 run.

The launcher is the sole owner of paths and process creation.  This worker
accepts one sealed specification FD and one already-connected local control
socket.  Every synthetic input and writable tree is supplied as an inherited
FD; no CRM input is resolved by path.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import re
import socket
import stat
import struct
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from scripts import run_v412_fresh_intake_synthetic_scanner_sealer as core
except ModuleNotFoundError:
    import run_v412_fresh_intake_synthetic_scanner_sealer as core


ALLOWED_ROOT = Path(
    "/Volumes/CATNAT_DATA/SIRETO_RECALL100/fresh_holdout_intake_synthetic"
)
CORE_PLAN_SHA256 = (
    "e8a55a999035183363c0bf7711280b09553a305434173286e41c696ea3e4772f"
)
CORE_CONTRACT_SHA256 = (
    "ad8eed1bf5d8d8a280ea8b212d3d308eb5c8b048efb3ebefb567956b3eb60ca8"
)
CORE_TESTS_SHA256 = (
    "b43309bbccbc37fced14c1b731956bad372c35c09d243b23df1d8efb9a6f72e1"
)
WORKER_SPEC_SCHEMA = "sireto-v4.12-fresh-s0-worker-spec-1"
CONTROL_READY_SCHEMA = "sireto-v4.12-fresh-s0-control-ready-1"
CONTROL_RESULT_SCHEMA = "sireto-v4.12-fresh-s0-control-result-1"
CANARY_REPORT_SCHEMA = "sireto-v4.12-fresh-s0-canary-proof-1"
CONTROL_PROTOCOL = "CANONICAL_LENGTH_PREFIXED_JSON_V1"
MAX_SPEC_BYTES = 1024 * 1024
MAX_FRAME_BYTES = 65536
PAYLOAD_ROLES = (
    "CONTROL_MANIFEST",
    "COLLECTION_MANIFEST",
    "SOURCE_MANIFEST",
    "CRM_SAFE_CSV",
    "EVIDENCE_MANIFEST",
    "EVIDENCE_PARQUET",
)
SOURCE_PAYLOAD_ROLES = PAYLOAD_ROLES[1:]
PAYLOAD_NAMES = {
    "COLLECTION_MANIFEST": "collection_source_manifest.json",
    "SOURCE_MANIFEST": "source_manifest.json",
    "CRM_SAFE_CSV": "crm_safe.csv",
    "EVIDENCE_MANIFEST": "evidence_source_manifest.json",
    "EVIDENCE_PARQUET": "evidence_source.parquet",
}
WRITE_ROLES = ("SEALED", "SCAN", "QUARANTINE", "AUDIT", "TMP")
ID_RE = re.compile(r"^[a-p]{64}$")
HEX_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
DECIMAL_RE = re.compile(r"^(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$")

# The core plan pins the referenced V4.12 opaque-ID plan.  S0 carries only
# these three closed domains into the private runtime, avoiding any path lookup
# outside the copied application bundle.
OPAQUE_SPEC = {
    "domains": {
        "batch": "SIRETO-V412-FRESH-BATCH-ID\x00",
        "query": "SIRETO-V412-FRESH-QUERY-ID\x00",
        "stratum": "SIRETO-V412-FRESH-STRATUM-ID\x00",
    }
}


class WorkerStop(RuntimeError):
    """Controlled, fail-closed worker termination."""


def _stop(message: str) -> None:
    raise WorkerStop(message)


def _canonical_json(value: Any, *, final_lf: bool = False) -> bytes:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return raw + (b"\n" if final_lf else b"")


def _strict_json_object(raw: bytes, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result

    def invalid_constant(value: str) -> None:
        raise ValueError(f"invalid JSON constant {value}")

    try:
        decoded = raw.decode("utf-8", errors="strict")
        value = json.loads(
            decoded,
            object_pairs_hook=pairs,
            parse_constant=invalid_constant,
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        _stop(f"{label} is not strict JSON: {exc}")
    if type(value) is not dict or raw != _canonical_json(value, final_lf=True):
        _stop(f"{label} is not canonical JSON with one final LF")
    return value


def _read_regular_fd(fd: int, *, maximum: int | None = None) -> bytes:
    try:
        info = os.fstat(fd)
    except OSError as exc:
        _stop(f"FD {fd} cannot be inspected: {exc}")
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        _stop(f"FD {fd} is not a mono-link regular file")
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o022:
        _stop(f"FD {fd} has unsafe ownership or mode")
    if info.st_size < 0 or (maximum is not None and info.st_size > maximum):
        _stop(f"FD {fd} exceeds its size limit")
    chunks: list[bytes] = []
    offset = 0
    while offset < info.st_size:
        chunk = os.pread(fd, min(1024 * 1024, info.st_size - offset), offset)
        if not chunk:
            _stop(f"FD {fd} ended before its pinned size")
        chunks.append(chunk)
        offset += len(chunk)
    if os.pread(fd, 1, info.st_size):
        _stop(f"FD {fd} contains bytes after its pinned EOF")
    raw = b"".join(chunks)
    if len(raw) != info.st_size:
        _stop(f"FD {fd} short read")
    return raw


def _mode_string(mode: int) -> str:
    return f"{stat.S_IMODE(mode):04o}"


def _validate_identity(identity: Any, info: os.stat_result) -> None:
    exact = {
        "device",
        "inode",
        "size_bytes",
        "mtime_ns",
        "ctime_ns",
        "uid",
        "volume_uuid",
        "link_count",
        "mode",
    }
    if type(identity) is not dict or set(identity) != exact:
        _stop("payload FD identity fields mismatch")
    expected_ints = {
        "device": info.st_dev,
        "inode": info.st_ino,
        "size_bytes": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
        "uid": info.st_uid,
        "link_count": info.st_nlink,
    }
    if any(type(identity[key]) is not int for key in expected_ints):
        _stop("payload FD identity integer type mismatch")
    if any(identity[key] != value for key, value in expected_ints.items()):
        _stop("payload FD identity drift")
    if identity["mode"] != _mode_string(info.st_mode):
        _stop("payload FD mode drift")
    if (
        type(identity["volume_uuid"]) is not str
        or UUID_RE.fullmatch(identity["volume_uuid"]) is None
    ):
        _stop("payload FD volume UUID is invalid")


def _validate_spec(spec: Mapping[str, Any], spec_fd: int, control_fd: int) -> None:
    exact = {
        "schema_version",
        "implementation_commit",
        "execution_lock_sha256",
        "synthetic_run_id",
        "attempt_id",
        "logical_time_utc",
        "minimum_stability_seconds",
        "payload_fds",
        "write_directory_fds",
        "control_protocol",
    }
    if set(spec) != exact:
        _stop("worker spec fields mismatch")
    if (
        spec["schema_version"] != WORKER_SPEC_SCHEMA
        or type(spec["implementation_commit"]) is not str
        or COMMIT_RE.fullmatch(spec["implementation_commit"]) is None
        or type(spec["execution_lock_sha256"]) is not str
        or HEX_RE.fullmatch(spec["execution_lock_sha256"]) is None
        or type(spec["synthetic_run_id"]) is not str
        or ID_RE.fullmatch(spec["synthetic_run_id"]) is None
        or type(spec["attempt_id"]) is not str
        or ID_RE.fullmatch(spec["attempt_id"]) is None
        or not core._is_strict_rfc3339_utc_seconds(spec["logical_time_utc"])
        or type(spec["minimum_stability_seconds"]) is not int
        or spec["minimum_stability_seconds"] != 60
        or spec["control_protocol"] != CONTROL_PROTOCOL
    ):
        _stop("worker spec constants or scalar types mismatch")
    records = spec["payload_fds"]
    if type(records) is not list or len(records) != len(PAYLOAD_ROLES):
        _stop("worker payload FD cardinality mismatch")
    if [record.get("role") for record in records if type(record) is dict] != list(
        PAYLOAD_ROLES
    ):
        _stop("worker payload FD role order mismatch")
    used = {spec_fd, control_fd}
    payload_devices: set[int] = set()
    for record in records:
        if type(record) is not dict or set(record) != {
            "role",
            "fd_number",
            "identity",
            "size_bytes",
            "sha256",
            "access",
        }:
            _stop("worker payload FD record fields mismatch")
        fd = record["fd_number"]
        if type(fd) is not int or fd < 0 or fd in used:
            _stop("worker payload FD number is invalid or duplicated")
        used.add(fd)
        if (
            record["access"] != "READ_ONLY"
            or type(record["size_bytes"]) is not int
            or record["size_bytes"] < 0
            or type(record["sha256"]) is not str
            or HEX_RE.fullmatch(record["sha256"]) is None
        ):
            _stop("worker payload FD metadata is invalid")
        info = os.fstat(fd)
        payload_devices.add(info.st_dev)
        _validate_identity(record["identity"], info)
        if info.st_size != record["size_bytes"]:
            _stop("worker payload FD size mismatch")
    directories = spec["write_directory_fds"]
    if type(directories) is not dict or set(directories) != set(WRITE_ROLES):
        _stop("worker write directory FD map mismatch")
    directory_devices: set[int] = set()
    for role in WRITE_ROLES:
        fd = directories[role]
        if type(fd) is not int or fd < 0 or fd in used:
            _stop("worker directory FD number is invalid or duplicated")
        used.add(fd)
        info = os.fstat(fd)
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o700
        ):
            _stop(f"worker directory FD {role} is unsafe")
        directory_devices.add(info.st_dev)
    if len(directory_devices) != 1 or payload_devices != directory_devices:
        _stop("worker payload and write directory FDs do not share one device")


def _payload_snapshot(spec: Mapping[str, Any]) -> tuple[dict[str, bytes], dict[str, tuple[int, ...]]]:
    payloads: dict[str, bytes] = {}
    identities: dict[str, tuple[int, ...]] = {}
    volume_uuids: set[str] = set()
    for record in spec["payload_fds"]:
        fd = record["fd_number"]
        before = os.fstat(fd)
        raw = _read_regular_fd(fd, maximum=4 * 1024 * 1024)
        after = os.fstat(fd)
        _validate_identity(record["identity"], after)
        identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after.st_uid,
            after.st_nlink,
            stat.S_IMODE(after.st_mode),
        )
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
            before.st_uid,
            before.st_nlink,
            stat.S_IMODE(before.st_mode),
        )
        if identity != before_identity:
            _stop(f"payload FD drift during read: {record['role']}")
        if len(raw) != record["size_bytes"] or hashlib.sha256(raw).hexdigest() != record[
            "sha256"
        ]:
            _stop(f"payload FD byte mismatch: {record['role']}")
        payloads[record["role"]] = raw
        identities[record["role"]] = identity
        volume_uuids.add(record["identity"]["volume_uuid"])
    if len(volume_uuids) != 1:
        _stop("payload FD volume UUIDs disagree")
    return payloads, identities


def _message_with_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["message_sha256"] = hashlib.sha256(_canonical_json(result)).hexdigest()
    return result


def _send_frame(control: socket.socket, payload: Mapping[str, Any]) -> None:
    raw = _canonical_json(payload)
    if len(raw) > MAX_FRAME_BYTES:
        _stop("control frame exceeds the closed maximum")
    control.sendall(struct.pack(">I", len(raw)) + raw)


def _ready_message(spec: Mapping[str, Any]) -> dict[str, Any]:
    return _message_with_hash(
        {
            "schema_version": CONTROL_READY_SCHEMA,
            "message_type": "READY",
            "synthetic_run_id": spec["synthetic_run_id"],
            "attempt_id": spec["attempt_id"],
            "worker_pid": os.getpid(),
            "payload_fd_roles": list(PAYLOAD_ROLES),
        }
    )


def _empty_authority() -> dict[str, None]:
    return {
        "sealed_input_payload_manifest_sha256": None,
        "sealed_input_seal_sha256": None,
        "terminal_tree_kind": None,
        "terminal_tree_payload_manifest_sha256": None,
        "terminal_tree_seal_sha256": None,
        "journal_generation": None,
        "journal_generation_manifest_sha256": None,
        "journal_head_event_sha256": None,
    }


def _stability(
    *,
    same_process: bool | None,
    same_fds: bool | None,
    elapsed: float | None,
) -> dict[str, Any]:
    if elapsed is None:
        encoded = None
    else:
        encoded = f"{elapsed:.9f}"
        if DECIMAL_RE.fullmatch(encoded) is None:
            _stop("internal monotonic duration encoding failed")
    return {
        "same_worker_process": same_process,
        "same_five_payload_fds": same_fds,
        "monotonic_elapsed_seconds": encoded,
    }


def _terminal_message(
    spec: Mapping[str, Any],
    *,
    message_type: str,
    reason_code: str,
    terminal_result: str,
    stability: Mapping[str, Any],
    output_authority: Mapping[str, Any],
) -> dict[str, Any]:
    return _message_with_hash(
        {
            "schema_version": CONTROL_RESULT_SCHEMA,
            "message_type": message_type,
            "synthetic_run_id": spec["synthetic_run_id"],
            "attempt_id": spec["attempt_id"],
            "phase": "WORKER",
            "reason_code": reason_code,
            "terminal_result": terminal_result,
            "stability": dict(stability),
            "output_authority": dict(output_authority),
        }
    )


class _FDTreeAuthority:
    """Duck-typed core authority dispatching logical trees to retained dir FDs."""

    def __init__(self, run_id: str, role_fds: Mapping[str, int]):
        self.run_id = run_id
        self._roots = {
            "sealed": os.dup(role_fds["SEALED"]),
            "scan": os.dup(role_fds["SCAN"]),
            "quarantine": os.dup(role_fds["QUARANTINE"]),
            "audit": os.dup(role_fds["AUDIT"]),
            "tmp": os.dup(role_fds["TMP"]),
        }
        infos = [os.fstat(fd) for fd in self._roots.values()]
        if len({info.st_dev for info in infos}) != 1:
            self.close()
            _stop("output authority spans devices")
        self.device = infos[0].st_dev

    def close(self) -> None:
        for fd in self._roots.values():
            try:
                os.close(fd)
            except OSError:
                pass
        self._roots.clear()

    def _resolve(self, relative: str | Sequence[str]) -> tuple[int, tuple[str, ...]]:
        if isinstance(relative, str):
            parts = tuple(relative.split("/"))
        else:
            parts = tuple(relative)
        if (
            len(parts) < 2
            or parts[0] not in self._roots
            or parts[1] != self.run_id
            or any(not part or part in {".", ".."} or "/" in part for part in parts)
        ):
            _stop("logical output path escaped its retained directory FD")
        return self._roots[parts[0]], parts[2:]

    def _validate_dir(self, fd: int) -> None:
        info = os.fstat(fd)
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_dev != self.device
            or stat.S_IMODE(info.st_mode) != 0o700
        ):
            _stop("unsafe output directory below retained FD")

    def open_dir(self, relative: str | Sequence[str], *, create: bool = False) -> int:
        base, parts = self._resolve(relative)
        current = os.dup(base)
        try:
            for part in parts:
                if create:
                    try:
                        os.mkdir(part, 0o700, dir_fd=current)
                    except FileExistsError:
                        pass
                next_fd = os.open(
                    part,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=current,
                )
                os.close(current)
                current = next_fd
                self._validate_dir(current)
            self._validate_dir(current)
            return current
        except Exception:
            os.close(current)
            raise

    def list(self, relative: str | Sequence[str]) -> list[str]:
        fd = self.open_dir(relative)
        try:
            return sorted(os.listdir(fd))
        finally:
            os.close(fd)

    def exists(self, relative: str | Sequence[str], *, directory: bool) -> bool:
        base, parts = self._resolve(relative)
        if not parts:
            return directory
        parent_path = (relative.split("/")[:-1] if isinstance(relative, str) else parts[:-1])
        logical_parent = (
            "/".join(parent_path)
            if isinstance(relative, str)
            else (str(relative[0]), self.run_id, *parts[:-1])
        )
        try:
            parent = self.open_dir(logical_parent)
        except FileNotFoundError:
            return False
        try:
            try:
                info = os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                return False
            expected = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(
                info.st_mode
            )
            expected_mode = 0o700 if directory else 0o600
            if (
                not expected
                or info.st_uid != os.getuid()
                or info.st_dev != self.device
                or (not directory and info.st_nlink != 1)
                or stat.S_IMODE(info.st_mode) != expected_mode
            ):
                _stop("unsafe existing object below retained FD")
            return True
        finally:
            os.close(parent)

    def read_file(self, relative: str | Sequence[str]) -> bytes:
        base, parts = self._resolve(relative)
        if not parts:
            _stop("cannot read a directory as a file")
        parent_logical = (str(relative).split("/")[:-1])
        parent = self.open_dir("/".join(parent_logical))
        try:
            fd = os.open(
                parts[-1],
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent,
            )
            try:
                return _read_regular_fd(fd, maximum=16 * 1024 * 1024)
            finally:
                os.close(fd)
        finally:
            os.close(parent)

    def write_exclusive(self, relative: str | Sequence[str], payload: bytes) -> None:
        _base, parts = self._resolve(relative)
        if not parts:
            _stop("cannot overwrite a retained directory FD")
        parent = self.open_dir("/".join(str(relative).split("/")[:-1]), create=True)
        try:
            fd = os.open(
                parts[-1],
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=parent,
            )
            try:
                view = memoryview(payload)
                while view:
                    count = os.write(fd, view)
                    if count <= 0:
                        _stop("short output write")
                    view = view[count:]
                os.fchmod(fd, 0o600)
                core._sync_fd(fd, full_required=True)
            finally:
                os.close(fd)
            core._sync_fd(parent, full_required=False)
        finally:
            os.close(parent)

    def mkdir_exclusive(self, relative: str | Sequence[str]) -> None:
        _base, parts = self._resolve(relative)
        if not parts:
            _stop("retained run directory already exists")
        parent = self.open_dir("/".join(str(relative).split("/")[:-1]), create=True)
        try:
            os.mkdir(parts[-1], 0o700, dir_fd=parent)
            core._sync_fd(parent, full_required=False)
        finally:
            os.close(parent)

    def rename_exclusive(
        self,
        source: str | Sequence[str],
        destination: str | Sequence[str],
    ) -> None:
        source_base, source_parts = self._resolve(source)
        destination_base, destination_parts = self._resolve(destination)
        if source_base != destination_base and self.device != os.fstat(destination_base).st_dev:
            _stop("cross-device output promotion forbidden")
        source_parent = self.open_dir("/".join(str(source).split("/")[:-1]))
        destination_parent = self.open_dir(
            "/".join(str(destination).split("/")[:-1]), create=True
        )
        try:
            libc = core.ctypes.CDLL(None, use_errno=True)
            function = getattr(libc, "renameatx_np", None)
            if function is None:
                _stop("renameatx_np unavailable")
            function.argtypes = [
                core.ctypes.c_int,
                core.ctypes.c_char_p,
                core.ctypes.c_int,
                core.ctypes.c_char_p,
                core.ctypes.c_uint,
            ]
            function.restype = core.ctypes.c_int
            result = function(
                source_parent,
                os.fsencode(source_parts[-1]),
                destination_parent,
                os.fsencode(destination_parts[-1]),
                0x00000004,
            )
            if result:
                _stop(
                    "exclusive output promotion failed: "
                    + os.strerror(core.ctypes.get_errno())
                )
            core._sync_fd(source_parent, full_required=False)
            core._sync_fd(destination_parent, full_required=False)
        finally:
            os.close(source_parent)
            os.close(destination_parent)


def _load_core_plan() -> tuple[dict[str, Any], bytes]:
    application_root = Path(__file__).resolve().parents[1]
    plan_path = application_root / (
        "config/v4_12_fresh_intake_synthetic_scanner_sealer_plan.json"
    )
    contract_path = application_root / (
        "docs/v4_12_fresh_intake_synthetic_scanner_sealer_contract.md"
    )
    plan_raw = plan_path.read_bytes()
    contract_raw = contract_path.read_bytes()
    if hashlib.sha256(plan_raw).hexdigest() != CORE_PLAN_SHA256:
        _stop("private core plan hash mismatch")
    if hashlib.sha256(contract_raw).hexdigest() != CORE_CONTRACT_SHA256:
        _stop("private core contract hash mismatch")
    plan = _strict_json_object(plan_raw, "private core plan")
    if plan["contract"]["sha256"] != CORE_CONTRACT_SHA256:
        _stop("private core plan/contract binding mismatch")
    return plan, plan_raw


def _decode_control(raw: bytes, plan: Mapping[str, Any]) -> dict[str, Any]:
    control = _strict_json_object(raw, "control manifest")
    if set(control) != set(plan["control_manifest"]["exact_fields"]):
        _stop("control manifest fields mismatch")
    if (
        control["schema_version"] != plan["control_manifest"]["schema"]
        or control["synthetic_fixture"] is not True
        or control["fixture_spec_sha256"]
        != plan["control_manifest"]["fixture_spec_sha256"]
        or control["logical_time_utc"] != plan["fixture"]["logical_time_utc"]
        or control["batch_count"] != 1
        or control["expected_source_row_count"] != 6
        or control["producer_exclusions"] != []
    ):
        _stop("control manifest constants mismatch")
    return control


def _run_canaries(run_id: str) -> list[dict[str, Any]]:
    targets = [
        ("DENY_DATA", "OPEN_READ", ALLOWED_ROOT / "canaries/forbidden/data/sentinel"),
        ("DENY_MODELS", "OPEN_READ", ALLOWED_ROOT / "canaries/forbidden/models/sentinel"),
        ("DENY_REPORTS", "OPEN_READ", ALLOWED_ROOT / "canaries/forbidden/reports/sentinel"),
        (
            "DENY_CHALLENGES",
            "OPEN_READ",
            ALLOWED_ROOT / "canaries/forbidden/challenges/sentinel",
        ),
        (
            "DENY_FINAL_HOLDOUT_INPUTS",
            "OPEN_READ",
            ALLOWED_ROOT / "canaries/forbidden/final_holdout_inputs/sentinel",
        ),
        (
            "DENY_FRESH_HOLDOUT_INTAKE",
            "OPEN_READ",
            ALLOWED_ROOT / "canaries/forbidden/fresh_holdout_intake/sentinel",
        ),
        (
            "DENY_FRESH_HOLDOUT_EVALUATION_LEDGER",
            "OPEN_READ",
            ALLOWED_ROOT
            / "canaries/forbidden/fresh_holdout_evaluation_ledger/sentinel",
        ),
        (
            "DENY_REGISTRIES",
            "OPEN_READ",
            ALLOWED_ROOT / "canaries/forbidden/registries/sentinel",
        ),
    ]
    records: list[dict[str, Any]] = []
    allowed_errnos = {errno.EPERM, errno.EACCES}
    for code, operation, target in targets:
        try:
            fd = os.open(
                target,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
            )
        except OSError as exc:
            if exc.errno not in allowed_errnos:
                _stop(f"canary {code} returned unexpected errno {exc.errno}")
            records.append(
                {
                    "code": code,
                    "operation": operation,
                    "synthetic_target": str(target),
                    "result": "DENIED",
                    "errno": exc.errno,
                }
            )
        else:
            os.close(fd)
            _stop(f"canary {code} unexpectedly opened")
    parent = ALLOWED_ROOT / "canaries/forbidden"
    try:
        iterator = os.scandir(parent)
    except OSError as exc:
        if exc.errno not in allowed_errnos:
            _stop(f"parent enumeration canary returned errno {exc.errno}")
        records.append(
            {
                "code": "DENY_PARENT_ENUMERATION",
                "operation": "ENUMERATE_PARENT",
                "synthetic_target": str(parent),
                "result": "DENIED",
                "errno": exc.errno,
            }
        )
    else:
        iterator.close()
        _stop("parent enumeration canary unexpectedly succeeded")
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.connect(("127.0.0.1", 9))
    except OSError as exc:
        if exc.errno not in allowed_errnos:
            _stop(f"network canary returned unexpected errno {exc.errno}")
        records.append(
            {
                "code": "DENY_NETWORK",
                "operation": "OPEN_NETWORK",
                "synthetic_target": "AF_INET:127.0.0.1:9",
                "result": "DENIED",
                "errno": exc.errno,
            }
        )
    else:
        _stop("network canary unexpectedly connected")
    finally:
        probe.close()
    parent_target = ALLOWED_ROOT / f"audit/{run_id}/parent/worker-write-sentinel"
    try:
        fd = os.open(
            parent_target,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
    except OSError as exc:
        if exc.errno not in allowed_errnos:
            _stop(f"parent-audit write canary returned errno {exc.errno}")
        records.append(
            {
                "code": "DENY_WRITE_PARENT_AUDIT",
                "operation": "WRITE",
                "synthetic_target": str(parent_target),
                "result": "DENIED",
                "errno": exc.errno,
            }
        )
    else:
        os.close(fd)
        _stop("parent-audit write canary unexpectedly succeeded")
    return records


def _write_canary_report(
    spec: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if [record.get("code") for record in records] != [
        "DENY_DATA",
        "DENY_MODELS",
        "DENY_REPORTS",
        "DENY_CHALLENGES",
        "DENY_FINAL_HOLDOUT_INPUTS",
        "DENY_FRESH_HOLDOUT_INTAKE",
        "DENY_FRESH_HOLDOUT_EVALUATION_LEDGER",
        "DENY_REGISTRIES",
        "DENY_PARENT_ENUMERATION",
        "DENY_NETWORK",
        "DENY_WRITE_PARENT_AUDIT",
    ]:
        _stop("canary report record order mismatch")
    ordered = [dict(record) for record in records]
    report = {
        "schema_version": CANARY_REPORT_SCHEMA,
        "synthetic_run_id": spec["synthetic_run_id"],
        "attempt_id": spec["attempt_id"],
        "ordered_records": ordered,
        "record_count": len(ordered),
        "records_sha256": hashlib.sha256(_canonical_json(ordered)).hexdigest(),
    }
    raw = _canonical_json(report, final_lf=True)
    audit_fd = spec["write_directory_fds"]["AUDIT"]
    fd = os.open(
        "canaries.json",
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        0o600,
        dir_fd=audit_fd,
    )
    try:
        view = memoryview(raw)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                _stop("short canary report write")
            view = view[written:]
        os.fchmod(fd, 0o600)
        core._sync_fd(fd, full_required=True)
    finally:
        os.close(fd)
    core._sync_fd(audit_fd, full_required=False)
    return report


def _journal_authority(
    authority: _FDTreeAuthority, plan: Mapping[str, Any], run_id: str
) -> tuple[int, str, str]:
    events, event_hashes = core._validate_journal(plan, authority, run_id)
    generations = authority.list(f"audit/{run_id}/events_manifests")
    if len(events) != 3 or len(event_hashes) != 3 or len(generations) != 3:
        _stop("terminal journal does not contain exactly three generations")
    generation_raw = authority.read_file(
        f"audit/{run_id}/events_manifests/{generations[-1]}"
    )
    generation_hash = hashlib.sha256(generation_raw).hexdigest()
    if generations[-1] != f"00000003-{generation_hash}.json":
        _stop("terminal journal generation filename/hash mismatch")
    return 3, generation_hash, event_hashes[-1]


def _process(
    spec: Mapping[str, Any],
    payload_bytes: Mapping[str, bytes],
) -> tuple[str, dict[str, Any]]:
    plan, plan_raw = _load_core_plan()
    control = _decode_control(payload_bytes["CONTROL_MANIFEST"], plan)
    plan_hash = hashlib.sha256(plan_raw).hexdigest()
    expected_run = core.opaque_digest(
        plan["ids"]["run"]["domain"],
        {
            "fixture_spec_sha256": plan["control_manifest"]["fixture_spec_sha256"],
            "plan_sha256": plan_hash,
        },
    )
    control_hash = hashlib.sha256(payload_bytes["CONTROL_MANIFEST"]).hexdigest()
    expected_attempt = core.opaque_digest(
        plan["ids"]["attempt"]["domain"],
        {
            "synthetic_run_id": expected_run,
            "fixture_control_manifest_sha256": control_hash,
            "logical_time_utc": control["logical_time_utc"],
        },
    )
    if (
        spec["synthetic_run_id"] != expected_run
        or control["synthetic_run_id"] != expected_run
        or spec["attempt_id"] != expected_attempt
        or spec["logical_time_utc"] != control["logical_time_utc"]
    ):
        _stop("worker spec/control deterministic identity mismatch")
    source_payloads = {
        PAYLOAD_NAMES[role]: payload_bytes[role] for role in SOURCE_PAYLOAD_ROLES
    }
    collection_manifest, source_manifest, _evidence = core._validate_source_manifests(
        plan, control, source_payloads, expected_run
    )
    del collection_manifest
    authority = _FDTreeAuthority(expected_run, spec["write_directory_fds"])
    audit_time = core._audit_now()
    old_fresh = core._fresh_opaque_spec
    old_tests = core._tests_sha256
    core._fresh_opaque_spec = lambda _plan: OPAQUE_SPEC
    core._tests_sha256 = lambda: CORE_TESTS_SHA256
    try:
        sealed_path = f"sealed/{expected_run}/input"
        if authority.exists(sealed_path, directory=True):
            sealed = core._validate_tree(
                authority,
                sealed_path,
                package_kind="SEALED_INPUT",
                expected_payload_names=plan["outputs"]["sealed_input"]["payloads"],
                plan=plan,
                run_id=expected_run,
            )
        else:
            core._seal_tree(
                plan=plan,
                authority=authority,
                run_id=expected_run,
                destination=sealed_path,
                package_kind="SEALED_INPUT",
                payloads=source_payloads,
            )
            sealed = core._validate_tree(
                authority,
                sealed_path,
                package_kind="SEALED_INPUT",
                expected_payload_names=plan["outputs"]["sealed_input"]["payloads"],
                plan=plan,
                run_id=expected_run,
            )
        core._validate_source_manifests(plan, control, sealed["payloads"], expected_run)
        core._create_receipts(
            plan=plan,
            authority=authority,
            root=ALLOWED_ROOT,
            run_id=expected_run,
            sealed_path=sealed_path,
            sealed=sealed,
            audit_time=audit_time,
        )
        events, _hashes = core._validate_journal(plan, authority, expected_run)
        manifests, trees = core._hash_maps(sealed["payloads"], sealed_input=sealed)
        if not events:
            core._append_event(
                plan=plan,
                authority=authority,
                run_id=expected_run,
                event=core._event_base(
                    plan,
                    entity_kind="BATCH",
                    previous_state="WAITING_STABLE",
                    new_state="RECEIPTED",
                    manifest_hashes=manifests,
                    tree_hashes=trees,
                ),
                audit_time=audit_time,
            )
            events, _hashes = core._validate_journal(plan, authority, expected_run)
        if len(events) != 1 or events[0]["new_state"] != "RECEIPTED":
            _stop("worker encountered a non-recoverable journal prefix")
        batch_reason, rows = core._batch_parse(sealed["payloads"]["crm_safe.csv"])
        scan_path = f"scan/{expected_run}/output"
        quarantine_path = f"quarantine/{expected_run}/batch"
        if batch_reason is None:
            if authority.exists(quarantine_path, directory=True):
                _stop("batch quarantine conflicts with valid scan")
            if len(rows) != control["expected_source_row_count"]:
                _stop("source row count differs from control")
            if not authority.exists(scan_path, directory=True):
                scan_payloads = core._build_scan_payloads(
                    plan=plan,
                    source_manifest=source_manifest,
                    rows=rows,
                    run_id=expected_run,
                    attempt_id=expected_attempt,
                    control_hash=control_hash,
                    sealed=sealed,
                    plan_hash=plan_hash,
                )
                core._seal_tree(
                    plan=plan,
                    authority=authority,
                    run_id=expected_run,
                    destination=scan_path,
                    package_kind="SCAN_OUTPUT",
                    payloads=scan_payloads,
                )
            branch = core._validate_tree(
                authority,
                scan_path,
                package_kind="SCAN_OUTPUT",
                expected_payload_names=plan["outputs"]["scan_output"]["payloads"],
                plan=plan,
                run_id=expected_run,
            )
            core._validate_scan_binding(
                plan=plan,
                branch=branch,
                sealed=sealed,
                run_id=expected_run,
                attempt_id=expected_attempt,
                control_hash=control_hash,
                plan_hash=plan_hash,
            )
            terminal_state = "INGESTED"
            terminal_result = "INGESTED_SYNTHETIC_SCANNER_SEALER_V412"
            tree_kind = "SCAN_OUTPUT"
            manifests, trees = core._hash_maps(
                sealed["payloads"], sealed_input=sealed, scan=branch
            )
        else:
            if authority.exists(scan_path, directory=True):
                _stop("scan output conflicts with batch quarantine")
            if not authority.exists(quarantine_path, directory=True):
                core._seal_tree(
                    plan=plan,
                    authority=authority,
                    run_id=expected_run,
                    destination=quarantine_path,
                    package_kind="BATCH_QUARANTINE",
                    payloads=core._quarantine_payload(
                        plan, expected_run, batch_reason, sealed
                    ),
                )
            branch = core._validate_tree(
                authority,
                quarantine_path,
                package_kind="BATCH_QUARANTINE",
                expected_payload_names=plan["outputs"]["batch_quarantine"][
                    "payloads"
                ],
                plan=plan,
                run_id=expected_run,
            )
            core._validate_quarantine_binding(
                plan=plan,
                branch=branch,
                sealed=sealed,
                run_id=expected_run,
                observed_reason=batch_reason,
            )
            terminal_state = "QUARANTINED"
            terminal_result = "QUARANTINED_SYNTHETIC_SCANNER_SEALER_V412"
            tree_kind = "BATCH_QUARANTINE"
            manifests, trees = core._hash_maps(
                sealed["payloads"], sealed_input=sealed, quarantine=branch
            )
        events, _hashes = core._validate_journal(plan, authority, expected_run)
        if len(events) == 1:
            core._append_event(
                plan=plan,
                authority=authority,
                run_id=expected_run,
                event=core._event_base(
                    plan,
                    entity_kind="BATCH",
                    previous_state="RECEIPTED",
                    new_state=terminal_state,
                    manifest_hashes=manifests,
                    tree_hashes=trees,
                ),
                audit_time=audit_time,
            )
            events, _hashes = core._validate_journal(plan, authority, expected_run)
        if len(events) == 2:
            core._append_event(
                plan=plan,
                authority=authority,
                run_id=expected_run,
                event=core._event_base(
                    plan,
                    entity_kind="COLLECTION",
                    previous_state="WAITING",
                    new_state=terminal_state,
                    manifest_hashes=manifests,
                    tree_hashes=trees,
                ),
                audit_time=audit_time,
            )
        final_events, _ = core._validate_journal(plan, authority, expected_run)
        if (
            len(final_events) != 3
            or final_events[1]["new_state"] != terminal_state
            or final_events[2]["new_state"] != terminal_state
        ):
            _stop("worker terminal journal state mismatch")
        generation, generation_hash, head_hash = _journal_authority(
            authority, plan, expected_run
        )
        output_authority = {
            "sealed_input_payload_manifest_sha256": sealed[
                "payload_manifest_sha256"
            ],
            "sealed_input_seal_sha256": sealed["seal_sha256"],
            "terminal_tree_kind": tree_kind,
            "terminal_tree_payload_manifest_sha256": branch[
                "payload_manifest_sha256"
            ],
            "terminal_tree_seal_sha256": branch["seal_sha256"],
            "journal_generation": generation,
            "journal_generation_manifest_sha256": generation_hash,
            "journal_head_event_sha256": head_hash,
        }
        return terminal_result, output_authority
    finally:
        core._fresh_opaque_spec = old_fresh
        core._tests_sha256 = old_tests
        authority.close()


def _parse_cli(argv: Sequence[str]) -> tuple[int, int]:
    if len(argv) != 5:
        _stop("worker requires exactly two internal FD options")
    if argv[1:] != [
        "--worker-spec-fd",
        argv[2],
        "--worker-control-fd",
        argv[4],
    ]:
        _stop("worker internal option order mismatch")
    if not argv[2].isascii() or not argv[2].isdigit():
        _stop("worker spec FD must be a decimal integer")
    if not argv[4].isascii() or not argv[4].isdigit():
        _stop("worker control FD must be a decimal integer")
    spec_fd = int(argv[2])
    control_fd = int(argv[4])
    if spec_fd < 3 or control_fd < 3 or spec_fd == control_fd:
        _stop("worker inherited protocol FDs are invalid")
    return spec_fd, control_fd


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv if argv is None else argv)
    spec: dict[str, Any] | None = None
    control: socket.socket | None = None
    started_pid = os.getpid()
    started = time.monotonic()
    first_identities: dict[str, tuple[int, ...]] | None = None
    elapsed: float | None = None
    ready_sent = False
    try:
        spec_fd, control_fd = _parse_cli(arguments)
        spec_raw = _read_regular_fd(spec_fd, maximum=MAX_SPEC_BYTES)
        spec = _strict_json_object(spec_raw, "worker spec")
        _validate_spec(spec, spec_fd, control_fd)
        for fd in [spec_fd, control_fd, *[r["fd_number"] for r in spec["payload_fds"]], *spec["write_directory_fds"].values()]:
            fcntl.fcntl(fd, fcntl.F_SETFD, fcntl.FD_CLOEXEC)
        control = socket.socket(fileno=control_fd)
        if control.family != socket.AF_UNIX or (control.type & 0xF) != socket.SOCK_STREAM:
            _stop("worker control FD is not an AF_UNIX stream socket")
        first_payloads, first_identities = _payload_snapshot(spec)
        _send_frame(control, _ready_message(spec))
        ready_sent = True
        canaries = _run_canaries(spec["synthetic_run_id"])
        _write_canary_report(spec, canaries)
        interval_started = time.monotonic()
        time.sleep(60.0)
        elapsed = time.monotonic() - interval_started
        second_payloads, second_identities = _payload_snapshot(spec)
        same_fds = all(
            first_identities[role] == second_identities[role]
            and first_payloads[role] == second_payloads[role]
            for role in SOURCE_PAYLOAD_ROLES
        )
        same_process = os.getpid() == started_pid
        if elapsed + 1e-9 < 60.0 or not same_fds or not same_process:
            _stop("60-second same-process same-FD stability invariant failed")
        terminal_result, output_authority = _process(spec, second_payloads)
        stability = _stability(
            same_process=same_process, same_fds=same_fds, elapsed=elapsed
        )
        _send_frame(
            control,
            _terminal_message(
                spec,
                message_type="RESULT",
                reason_code="OK",
                terminal_result=terminal_result,
                stability=stability,
                output_authority=output_authority,
            ),
        )
        control.shutdown(socket.SHUT_WR)
        return 0
    except Exception:
        if spec is not None and control is not None and ready_sent:
            try:
                same_process = os.getpid() == started_pid
                same_fds = None
                if first_identities is not None:
                    try:
                        _payloads, current = _payload_snapshot(spec)
                        same_fds = all(
                            first_identities[role] == current[role]
                            for role in SOURCE_PAYLOAD_ROLES
                        )
                    except Exception:
                        same_fds = False
                observed = (
                    elapsed
                    if elapsed is not None
                    else max(0.0, time.monotonic() - started)
                )
                _send_frame(
                    control,
                    _terminal_message(
                        spec,
                        message_type="STOP",
                        reason_code="WORKER_CONTROLLED_STOP",
                        terminal_result="STOP_SYNTHETIC_SCANNER_SEALER_V412",
                        stability=_stability(
                            same_process=same_process,
                            same_fds=same_fds,
                            elapsed=observed,
                        ),
                        output_authority=_empty_authority(),
                    ),
                )
                control.shutdown(socket.SHUT_WR)
            except Exception:
                pass
        return 2
    finally:
        if control is not None:
            try:
                control.detach()
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
