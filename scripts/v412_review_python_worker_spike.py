#!/usr/bin/env python3
"""M3 feasibility spike for a capability-bounded Python business worker.

This module intentionally implements no collection or adjudication logic.  It
only proves that a pinned, relocatable arm64 Python runtime can exchange bounded
frames on one inherited AF_UNIX descriptor while Seatbelt refuses ambient
capabilities.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import socket
import stat
import struct
import subprocess
import sys
import time
from typing import Any, Final


CLAIM: Final = "M3_PYTHON_SANDBOX_FEASIBILITY_ONLY"
PYTHON_SOURCE_ROOT: Final = Path(
    "/Users/nathanjullia/.cache/codex-runtimes/codex-primary-runtime/"
    "dependencies/python"
)
PYTHON_SOURCE_EXECUTABLE: Final = PYTHON_SOURCE_ROOT / "bin/python3.12"
PYTHON_SOURCE_EXECUTABLE_SHA256: Final = "eb9d74b9c7cfdfb2c9b91614edb2c3607360ba46c5aa7fc4557b3a4a23e97cff"
PYTHON_STDLIB_TREE_SHA256: Final = "03054794f7ca52fbf0f03955515f544079fc2f2c84a61f7b0a6c74e0375b7763"
PYTHON_STDLIB_PAYLOAD_SHA256: Final = "6f194cb0b25fb732fafcbbff734ef3edb133310b4ea46c1d1c17645319d86359"
STAGING_PARENT: Final = Path("/Volumes/CATNAT_DATA/0")
STAGING_PATH: Final = STAGING_PARENT / "0"
LATE_ESCAPE_ROOT: Final = Path("/Volumes/CATNAT_DATA/1")
LATE_ESCAPE_PATH: Final = LATE_ESCAPE_ROOT / "late-path-probe.txt"
PIN_PATH: Final = Path(__file__).with_name("v412_review_python_worker_spike.pin.json")
PIN_SCHEMA: Final = "sireto-v4.12-m3-python-worker-spike-pin-1"
SANDBOX_EXEC: Final = Path("/usr/bin/sandbox-exec")
SANDBOX_EXEC_SHA256: Final = "8290e4be7387a0df83cd1559e86afd880464f269450573d012795761fe298f16"

MAGIC: Final = b"S4M3"
PROTOCOL_VERSION: Final = 1
MAX_FRAME_BYTES: Final = 64 * 1024
HEADER: Final = struct.Struct("!4sBBHII")
MAX_PAYLOAD_BYTES: Final = MAX_FRAME_BYTES - HEADER.size
DEFAULT_TIMEOUT_SECONDS: Final = 8.0

FRAME_READY: Final = 1
FRAME_GATE: Final = 2
FRAME_PING: Final = 3
FRAME_PONG: Final = 4
FRAME_RESULT: Final = 5
VALID_FRAME_TYPES: Final = frozenset(
    {FRAME_READY, FRAME_GATE, FRAME_PING, FRAME_PONG, FRAME_RESULT}
)
ROLES: Final = (
    "IDENTITY_DISCOVERY_SPIKE",
    "FROZEN_CANDIDATE_COMPARISON_SPIKE",
)
DENIED_READ_ROOTS: Final = (
    "/Applications", "/Library", "/System/Volumes", "/Users",
    "/bin", "/cores", "/dev", "/etc", "/home", "/opt", "/private",
    "/sbin", "/tmp", "/var", "/workspace", "/usr/bin", "/usr/libexec",
    "/usr/local", "/usr/sbin", "/usr/standalone", "/usr/X11", "/usr/X11R6",
)


class SpikeError(RuntimeError):
    """The feasibility proof failed closed."""


class SpikeIntegrityError(SpikeError):
    """A pinned TCB component or protocol invariant changed."""


class FrameError(SpikeError):
    """A bounded-frame invariant was violated."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_fd(descriptor: int) -> str:
    position = os.lseek(descriptor, 0, os.SEEK_CUR)
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while chunk := os.read(descriptor, 1024 * 1024):
        digest.update(chunk)
    os.lseek(descriptor, position, os.SEEK_SET)
    return digest.hexdigest()


def _load_source_pin() -> tuple[str, str]:
    descriptor = os.open(PIN_PATH, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        linked = os.stat(PIN_PATH, follow_symlinks=False)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (linked.st_dev, linked.st_ino) != (opened.st_dev, opened.st_ino)
            or opened.st_size > 1024
        ):
            raise SpikeIntegrityError("worker source pin identity is unsafe")
        raw = os.read(descriptor, 1025)
    finally:
        os.close(descriptor)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SpikeIntegrityError("worker source pin is invalid JSON") from exc
    if (
        type(value) is not dict
        or set(value) != {"schema", "worker_source_sha256"}
        or value.get("schema") != PIN_SCHEMA
        or type(value.get("worker_source_sha256")) is not str
        or len(value["worker_source_sha256"]) != 64
    ):
        raise SpikeIntegrityError("worker source pin schema mismatch")
    return value["worker_source_sha256"], hashlib.sha256(raw).hexdigest()


def _canonical_json(value: Any) -> bytes:
    try:
        raw = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FrameError("payload is not canonical-JSON serializable") from exc
    if len(raw) > MAX_PAYLOAD_BYTES:
        raise FrameError("payload exceeds the 64 KiB frame ceiling")
    return raw


def _set_socket_deadline(sock: socket.socket, deadline: float) -> None:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("absolute protocol deadline expired")
    sock.settimeout(remaining)


def _send_frame(
    sock: socket.socket,
    frame_type: int,
    sequence: int,
    value: Any,
    *,
    deadline: float | None = None,
) -> None:
    if frame_type not in VALID_FRAME_TYPES:
        raise FrameError("unknown frame type")
    if type(sequence) is not int or not 0 <= sequence <= 0xFFFFFFFF:
        raise FrameError("invalid frame sequence")
    payload = _canonical_json(value)
    header = HEADER.pack(
        MAGIC, PROTOCOL_VERSION, frame_type, 0, sequence, len(payload)
    )
    if deadline is not None:
        _set_socket_deadline(sock, deadline)
    sock.sendall(header + payload)


def _recv_exact(sock: socket.socket, size: int, deadline: float) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        _set_socket_deadline(sock, deadline)
        chunk = sock.recv(remaining)
        if not chunk:
            raise FrameError("truncated frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_frame(
    sock: socket.socket,
    *,
    expected_type: int | None = None,
    expected_sequence: int | None = None,
    deadline: float | None = None,
) -> tuple[int, int, Any]:
    absolute_deadline = (
        time.monotonic() + DEFAULT_TIMEOUT_SECONDS if deadline is None else deadline
    )
    header = _recv_exact(sock, HEADER.size, absolute_deadline)
    magic, version, frame_type, flags, sequence, payload_size = HEADER.unpack(header)
    if magic != MAGIC or version != PROTOCOL_VERSION or flags != 0:
        raise FrameError("invalid frame header")
    if frame_type not in VALID_FRAME_TYPES:
        raise FrameError("unknown frame type")
    if payload_size > MAX_PAYLOAD_BYTES:
        raise FrameError("declared payload exceeds the 64 KiB frame ceiling")
    if expected_type is not None and frame_type != expected_type:
        raise FrameError("unexpected frame type")
    if expected_sequence is not None and sequence != expected_sequence:
        raise FrameError("unexpected frame sequence")
    payload = _recv_exact(sock, payload_size, absolute_deadline)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FrameError("invalid JSON payload") from exc
    if _canonical_json(value) != payload:
        raise FrameError("payload is not canonical JSON")
    return frame_type, sequence, value


def _literal(path: Path) -> str:
    return json.dumps(os.fspath(path))


def _selected_source_files() -> list[tuple[Path, Path]]:
    selected = [
        (PYTHON_SOURCE_EXECUTABLE, Path("bin/python3.12")),
        (PYTHON_SOURCE_ROOT / "lib/libpython3.12.dylib", Path("lib/libpython3.12.dylib")),
    ]
    stdlib = PYTHON_SOURCE_ROOT / "lib/python3.12"
    for source in sorted(stdlib.rglob("*")):
        relative = source.relative_to(stdlib)
        if "site-packages" in relative.parts or "__pycache__" in relative.parts:
            continue
        if source.is_file() and source.suffix != ".pyc":
            selected.append((source, Path("lib/python3.12") / relative))
    return selected


def _manifest_hash(files: list[tuple[Path, Path]], *, require_stage: bool) -> str:
    records: list[dict[str, Any]] = []
    seen_inodes: set[tuple[int, int]] = set()
    for source, relative in files:
        info = os.lstat(source)
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise SpikeIntegrityError("Python TCB contains a non-regular file")
        if require_stage:
            if info.st_uid != os.getuid() or info.st_nlink != 1:
                raise SpikeIntegrityError("staged Python TCB ownership/link count is unsafe")
            if info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
                raise SpikeIntegrityError("staged Python TCB is group/world writable")
            inode = (info.st_dev, info.st_ino)
            if inode in seen_inodes:
                raise SpikeIntegrityError("staged Python TCB contains a hardlink")
            seen_inodes.add(inode)
        records.append(
            {
                "mode": stat.S_IMODE(info.st_mode),
                "path": relative.as_posix(),
                "sha256": _sha256_file(source),
                "size": info.st_size,
            }
        )
    raw = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _prepare_stage() -> tuple[Path, str]:
    STAGING_PARENT.mkdir(mode=0o700, parents=True, exist_ok=True)
    parent_info = os.lstat(STAGING_PARENT)
    if (
        not stat.S_ISDIR(parent_info.st_mode)
        or stat.S_IMODE(parent_info.st_mode) != 0o700
        or parent_info.st_uid != os.getuid()
        or any(STAGING_PARENT.iterdir())
    ):
        raise SpikeIntegrityError("staging parent is not an empty private directory")
    try:
        os.mkdir(STAGING_PATH, 0o700)
    except FileExistsError as exc:
        raise SpikeIntegrityError("fixed staging capability is already locked") from exc
    stage = STAGING_PATH
    try:
        source_files = _selected_source_files()
        for source, relative in source_files:
            destination = stage / relative
            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            shutil.copy2(source, destination, follow_symlinks=False)
        staged_files = [(stage / relative, relative) for _, relative in source_files]
        staged_hash = _manifest_hash(staged_files, require_stage=True)
        if staged_hash != PYTHON_STDLIB_TREE_SHA256:
            raise SpikeIntegrityError("staged Python TCB differs from the pinned source manifest")
        return stage, staged_hash
    except BaseException:
        shutil.rmtree(stage, ignore_errors=False)
        raise


def _assert_stage_anchor(
    stage: Path, catnat_fd: int, parent_fd: int, stage_fd: int
) -> None:
    paths_and_fds = (
        (STAGING_PARENT.parent, catnat_fd),
        (STAGING_PARENT, parent_fd),
        (stage, stage_fd),
    )
    for path, descriptor in paths_and_fds:
        linked = os.stat(path, follow_symlinks=False)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or (linked.st_dev, linked.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            raise SpikeIntegrityError("staging ancestor substitution detected")
    for descriptor in (parent_fd, stage_fd):
        info = os.fstat(descriptor)
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise SpikeIntegrityError("staging capability directory is not private")


def _cleanup_stage(
    stage: Path, anchor: tuple[int, int, int] | None = None
) -> None:
    if stage != STAGING_PATH:
        raise SpikeIntegrityError("refusing cleanup outside the dedicated staging parent")
    if anchor is not None:
        _assert_stage_anchor(stage, *anchor)
    shutil.rmtree(stage, ignore_errors=False)
    parent_entries = os.listdir(anchor[1]) if anchor is not None else os.listdir(STAGING_PARENT)
    if stage.exists() or parent_entries:
        raise SpikeIntegrityError("Python TCB cleanup was incomplete")


def _destroy_stage(
    stage: Path, anchor: tuple[int, int, int] | None = None
) -> None:
    """Backward-compatible test helper; production uses ``_cleanup_stage``."""

    _cleanup_stage(stage, anchor)


def _deny_other_component(base: str, allowed: str) -> str:
    """Seatbelt regex complement for one exact path component, no alternation."""

    rules: list[str] = []
    for index, expected in enumerate(allowed):
        prefix = allowed[:index]
        escaped_expected = expected.replace("-", "\\-").replace("]", "\\]")
        rules.append(
            f'(deny file-read-data (regex #"^{base}/{prefix}[^{escaped_expected}/]"))'
        )
        if prefix:
            rules.append(f'(deny file-read-data (regex #"^{base}/{prefix}/"))')
            rules.append(f'(deny file-read-data (regex #"^{base}/{prefix}$"))')
    rules.append(f'(deny file-read-data (regex #"^{base}/{allowed}[^/]"))')
    return "".join(rules)


def _sandbox_profile(stage: Path) -> str:
    """Return the deny-default profile used by both sequential workers."""

    staged_python = stage / "bin/python3.12"
    denied_roots = "".join(
        f"(deny file-read-data (subpath {json.dumps(root)}))"
        for root in DENIED_READ_ROOTS
    )
    catnat_root = STAGING_PARENT.parent
    return "".join(
        (
            "(version 1)",
            "(deny default)",
            "(deny network*)",
            "(deny process-fork)",
            "(deny process-exec)",
            f"(allow process-exec (literal {_literal(staged_python)}))",
            # This mirrors the audited native runtime: dyld receives its
            # required pathless grant, then all non-TCB roots are closed.
            "(allow file-read-data)",
            "(deny file-read-data (regex #\"^/\\.[^/]+(?:/|$)\"))",
            denied_roots,
            "(deny file-read-data (literal \"/Volumes\"))",
            _deny_other_component("/Volumes", "CATNAT_DATA"),
            f"(deny file-read-data (literal {_literal(catnat_root)}))",
            f"(deny file-read-data (literal {_literal(STAGING_PARENT)}))",
            # Structural complement: only /CATNAT_DATA/0/0 is a capability.
            # Keep alternatives in separate rules: Seatbelt mishandles the
            # grouped alternation form for file-read-data filters.
            "(deny file-read-data (regex #\"^/Volumes/CATNAT_DATA/[^0/]\"))",
            "(deny file-read-data (regex #\"^/Volumes/CATNAT_DATA/0[^/]\"))",
            "(deny file-read-data (regex #\"^/Volumes/CATNAT_DATA/0/[^0/]\"))",
            "(deny file-read-data (regex #\"^/Volumes/CATNAT_DATA/0/0[^/]\"))",
            f"(allow file-read* (subpath {_literal(stage)}) (subpath \"/System\") "
            "(subpath \"/usr/lib\") (subpath \"/usr/share\") "
            "(literal \"/dev/null\"))",
            "(deny file-write*)",
            "(allow file-ioctl)",
            "(allow sysctl-read)",
            "(allow signal (target self))",
        )
    )


def _assert_tcb(stage: Path | None = None) -> dict[str, str]:
    script_path = Path(__file__).resolve(strict=True)
    pinned_source_hash, pin_file_hash = _load_source_pin()
    if _sha256_file(script_path) != pinned_source_hash:
        raise SpikeIntegrityError("worker source differs from its immutable pin")
    if PYTHON_SOURCE_EXECUTABLE.resolve(strict=True) != PYTHON_SOURCE_EXECUTABLE:
        raise SpikeIntegrityError("Python runtime executable path changed")
    if _sha256_file(PYTHON_SOURCE_EXECUTABLE) != PYTHON_SOURCE_EXECUTABLE_SHA256:
        raise SpikeIntegrityError("pinned Python runtime executable hash changed")
    source_hash = _manifest_hash(_selected_source_files(), require_stage=False)
    if source_hash != PYTHON_STDLIB_TREE_SHA256:
        raise SpikeIntegrityError("pinned Python stdlib manifest changed")
    if SANDBOX_EXEC.resolve(strict=True) != SANDBOX_EXEC:
        raise SpikeIntegrityError("sandbox-exec path changed")
    if _sha256_file(SANDBOX_EXEC) != SANDBOX_EXEC_SHA256:
        raise SpikeIntegrityError("pinned sandbox-exec hash changed")
    if platform.machine() != "arm64":
        raise SpikeIntegrityError("M3 spike requires the pinned arm64 host")
    result = {
        "claim": CLAIM,
        "python_source_executable": os.fspath(PYTHON_SOURCE_EXECUTABLE),
        "python_source_executable_sha256": PYTHON_SOURCE_EXECUTABLE_SHA256,
        "python_stdlib_tree_sha256": source_hash,
        "sandbox_exec_path": os.fspath(SANDBOX_EXEC),
        "sandbox_exec_sha256": SANDBOX_EXEC_SHA256,
        "worker_script_path": os.fspath(script_path),
        "worker_script_sha256": pinned_source_hash,
        "worker_source_pin_sha256": pin_file_hash,
    }
    if stage is not None:
        staged_files = [(stage / relative, relative) for _, relative in _selected_source_files()]
        result["staged_tree_sha256"] = _manifest_hash(staged_files, require_stage=True)
        result["sandbox_profile_sha256"] = hashlib.sha256(
            _sandbox_profile(stage).encode()
        ).hexdigest()
    return result


def _denied(callable_: Any) -> bool:
    try:
        result = callable_()
    except OSError as exc:
        return exc.errno in {errno.EPERM, errno.EACCES}
    except subprocess.SubprocessError:
        return False
    if isinstance(result, subprocess.CompletedProcess):
        return False
    return False


def _network_probe(family: int) -> bool:
    probe = socket.socket(family, socket.SOCK_STREAM)
    try:
        destination: tuple[Any, ...]
        if family == socket.AF_INET:
            destination = ("127.0.0.1", 9)
        else:
            destination = ("::1", 9, 0, 0)
        probe.settimeout(0.2)
        probe.connect(destination)
    finally:
        probe.close()
    return False


def _new_unix_socket_probe() -> bool:
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.bind(f"/tmp/sireto-m3-forbidden-{os.getpid()}.sock")
    finally:
        probe.close()
    return False


def _fork_probe() -> bool:
    pid = os.fork()
    if pid == 0:  # pragma: no cover - reached only if Seatbelt failed
        os._exit(91)
    os.waitpid(pid, 0)
    return False


def _spawn_probe() -> bool:
    pid = os.posix_spawn("/usr/bin/true", ["/usr/bin/true"], os.environ.copy())
    os.waitpid(pid, 0)
    return False


def _subprocess_probe() -> bool:
    subprocess.run(
        ["/usr/bin/true"],
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return False


def _outside_read_probe() -> bool:
    with open("/etc/hosts", "rb") as stream:
        stream.read(1)
    return False


def _file_read_probe(path: Path) -> bool:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.read(descriptor, 1)
    finally:
        os.close(descriptor)
    return False


def _directory_list_probe(path: Path) -> bool:
    os.listdir(path)
    return False


def _execve_probe() -> bool:
    # If Seatbelt unexpectedly permits this, the worker is replaced and the
    # parent observes a missing RESULT frame: still a fail-closed outcome.
    os.execve("/usr/bin/true", ["/usr/bin/true"], {})
    return False


def _reexec_probe() -> bool:
    try:
        os.execve(sys.executable, [sys.executable, "-c", "raise SystemExit(92)"], {})
    except OSError as exc:
        return exc.errno in {errno.EPERM, errno.EACCES, errno.ENOENT}
    return False


def _run_worker(role: str, descriptor: int, behavior: str) -> int:
    if role not in ROLES or descriptor < 3:
        return 70
    channel = socket.socket(fileno=descriptor)
    deadline = time.monotonic() + DEFAULT_TIMEOUT_SECONDS
    try:
        _send_frame(
            channel,
            FRAME_READY,
            0,
            {
                "claim": CLAIM,
                "pid": os.getpid(),
                "python_executable": sys.executable,
                "role": role,
            },
            deadline=deadline,
        )
        _recv_frame(
            channel, expected_type=FRAME_GATE, expected_sequence=0, deadline=deadline
        )
        if behavior == "hang":
            time.sleep(DEFAULT_TIMEOUT_SECONDS * 4)
            return 71
        if behavior == "bad-frame":
            channel.sendall(b"INVALID-FRAME")
            return 72
        _, _, ping = _recv_frame(
            channel, expected_type=FRAME_PING, expected_sequence=1, deadline=deadline
        )
        _send_frame(channel, FRAME_PONG, 1, ping, deadline=deadline)
        probes = {
            "af_inet_denied": _denied(lambda: _network_probe(socket.AF_INET)),
            "af_inet6_denied": _denied(lambda: _network_probe(socket.AF_INET6)),
            "new_af_unix_denied": _denied(_new_unix_socket_probe),
            "fork_denied": _denied(_fork_probe),
            "posix_spawn_denied": _denied(_spawn_probe),
            "subprocess_denied": _denied(_subprocess_probe),
            "etc_hosts_denied": _denied(_outside_read_probe),
            "private_denied": _denied(
                lambda: _file_read_probe(Path("/private/etc/hosts"))
            ),
            "users_root_denied": _denied(
                lambda: _directory_list_probe(Path("/Users"))
            ),
            "volumes_root_denied": _denied(
                lambda: _directory_list_probe(Path("/Volumes"))
            ),
            "other_volume_denied": _denied(
                lambda: _file_read_probe(Path("/Volumes/Macintosh HD/etc/hosts"))
            ),
            "business_ssd_denied": _denied(
                lambda: _directory_list_probe(
                    Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100")
                )
            ),
            "late_path_denied": _denied(
                lambda: _file_read_probe(LATE_ESCAPE_PATH)
            ),
            "staging_parent_denied": _denied(
                lambda: _directory_list_probe(STAGING_PARENT)
            ),
            "workspace_data_denied": _denied(
                lambda: _file_read_probe(
                    Path("/Users/nathanjullia/Documents/Projets/SIRETO/AGENTS.md")
                )
            ),
            "worker_parent_denied": _denied(
                lambda: _directory_list_probe(
                    Path("/Users/nathanjullia/Documents/Projets/SIRETO/scripts")
                )
            ),
            "hidden_home_denied": _denied(
                lambda: _directory_list_probe(Path("/Users/nathanjullia/.ssh"))
            ),
            "system_volumes_alias_denied": _denied(
                lambda: _directory_list_probe(Path("/System/Volumes/Data/Users"))
            ),
            # Keep this last: an unexpected allow replaces this process.
            "execve_denied": _denied(_execve_probe),
            "reexec_denied_after_revocation": _reexec_probe(),
        }
        _send_frame(
            channel,
            FRAME_RESULT,
            2,
            {
                "claim": CLAIM,
                "inherited_af_unix_roundtrip": True,
                "pid": os.getpid(),
                "probes": probes,
                "role": role,
            },
            deadline=deadline,
        )
        return 0 if all(probes.values()) else 73
    except (OSError, SpikeError):
        return 74
    finally:
        channel.close()


def _terminate(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.kill()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        os.kill(process.pid, signal.SIGKILL)
        process.wait(timeout=2)


def run_worker(
    role: str,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    behavior: str = "normal",
) -> dict[str, Any]:
    """Launch one sandboxed worker and fail closed on every anomaly."""

    if role not in ROLES or behavior not in {"normal", "hang", "bad-frame"}:
        raise SpikeIntegrityError("invalid closed worker launch parameters")
    if timeout_seconds <= 0:
        raise SpikeIntegrityError("worker timeout must be positive")
    source_tcb = _assert_tcb()
    stage, _ = _prepare_stage()
    catnat_fd: int | None = None
    staging_parent_fd: int | None = None
    stage_fd: int | None = None
    source_fd: int | None = None
    executable_fd: int | None = None
    bin_fd: int | None = None
    parent: socket.socket | None = None
    child: socket.socket | None = None
    process: subprocess.Popen[bytes] | None = None
    late_path_created = False
    result_value: dict[str, Any] | None = None
    primary_error: BaseException | None = None
    deadline = time.monotonic() + timeout_seconds
    try:
        catnat_fd = os.open(
            STAGING_PARENT.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        )
        staging_parent_fd = os.open(
            STAGING_PARENT, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        )
        stage_fd = os.open(stage, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        anchor = (catnat_fd, staging_parent_fd, stage_fd)
        _assert_stage_anchor(stage, *anchor)
        tcb_before = _assert_tcb(stage)
        script_path = Path(tcb_before["worker_script_path"])
        source_fd = os.open(script_path, os.O_RDONLY | os.O_NOFOLLOW)
        linked_source = os.stat(script_path, follow_symlinks=False)
        opened_source = os.fstat(source_fd)
        if (linked_source.st_dev, linked_source.st_ino) != (
            opened_source.st_dev,
            opened_source.st_ino,
        ):
            raise SpikeIntegrityError("worker source descriptor identity mismatch")
        source_chunks: list[bytes] = []
        while chunk := os.read(source_fd, 1024 * 1024):
            source_chunks.append(chunk)
        source_bytes = b"".join(source_chunks)
        script_source = source_bytes.decode("utf-8")
        if (
            hashlib.sha256(source_bytes).hexdigest()
            != tcb_before["worker_script_sha256"]
        ):
            raise SpikeIntegrityError("spawn source bytes differ from the pinned worker")
        staged_python = stage / "bin/python3.12"
        parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        executable_fd = os.open(staged_python, os.O_RDONLY | os.O_NOFOLLOW)
        bin_fd = os.open(
            stage / "bin", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        )
        profile = _sandbox_profile(stage)
        command = [
            os.fspath(SANDBOX_EXEC),
            "-p",
            profile,
            os.fspath(staged_python),
            "-I",
            "-S",
            "-B",
            "-c",
            (
                "import sys;p=sys.argv[1];s=sys.argv[2];"
                "sys.argv=[p,*sys.argv[3:]];globals()['__file__']=p;"
                "exec(compile(s,p,'exec'),globals())"
            ),
            os.fspath(script_path),
            script_source,
            "--worker",
            role,
            "--fd",
            str(child.fileno()),
            "--behavior",
            behavior,
        ]
        if LATE_ESCAPE_ROOT.exists():
            raise SpikeIntegrityError("late-path adversarial root already exists")
        LATE_ESCAPE_ROOT.mkdir(mode=0o700)
        LATE_ESCAPE_PATH.write_bytes(b"must remain unreadable")
        late_path_created = True
        process = subprocess.Popen(
            command,
            close_fds=True,
            pass_fds=(child.fileno(),),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=stage,
            env={
                "LANG": "C",
                "LC_ALL": "C",
                "PATH": "/usr/bin:/bin",
                "PYTHONHASHSEED": "0",
            },
        )
        child.close()
        parent.settimeout(max(0.001, deadline - time.monotonic()))
        _, _, ready = _recv_frame(
            parent, expected_type=FRAME_READY, expected_sequence=0, deadline=deadline
        )
        if ready != {
            "claim": CLAIM,
            "pid": process.pid,
            "python_executable": os.fspath(staged_python),
            "role": role,
        }:
            raise SpikeIntegrityError("worker READY identity mismatch")
        _assert_stage_anchor(stage, *anchor)
        linked_source_after = os.stat(script_path, follow_symlinks=False)
        opened_source_after = os.fstat(source_fd)
        if (
            (linked_source_after.st_dev, linked_source_after.st_ino)
            != (opened_source_after.st_dev, opened_source_after.st_ino)
            or _sha256_fd(source_fd) != hashlib.sha256(source_bytes).hexdigest()
        ):
            raise SpikeIntegrityError("worker source changed across spawn")
        linked = os.stat(staged_python, follow_symlinks=False)
        opened = os.fstat(executable_fd)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (linked.st_dev, linked.st_ino) != (opened.st_dev, opened.st_ino)
            or _sha256_fd(executable_fd) != PYTHON_SOURCE_EXECUTABLE_SHA256
        ):
            raise SpikeIntegrityError("staged executable changed across spawn")
        os.unlink(staged_python)
        os.fsync(bin_fd)
        if staged_python.exists() or os.fstat(executable_fd).st_nlink != 0:
            raise SpikeIntegrityError("staged executable revocation failed")
        _send_frame(
            parent,
            FRAME_GATE,
            0,
            {"authorized": True, "role": role},
            deadline=deadline,
        )
        nonce = hashlib.sha256(f"{role}:{process.pid}".encode()).hexdigest()
        ping = {"nonce": nonce}
        _send_frame(parent, FRAME_PING, 1, ping, deadline=deadline)
        _, _, pong = _recv_frame(
            parent, expected_type=FRAME_PONG, expected_sequence=1, deadline=deadline
        )
        if pong != ping:
            raise SpikeIntegrityError("inherited AF_UNIX roundtrip mismatch")
        _, _, result = _recv_frame(
            parent, expected_type=FRAME_RESULT, expected_sequence=2, deadline=deadline
        )
        remaining = max(0.001, deadline - time.monotonic())
        process.wait(timeout=remaining)
        if process.returncode != 0:
            raise SpikeIntegrityError(f"sandbox worker failed rc={process.returncode}")
        if (
            type(result) is not dict
            or result.get("claim") != CLAIM
            or result.get("role") != role
            or result.get("pid") != process.pid
            or result.get("inherited_af_unix_roundtrip") is not True
            or type(result.get("probes")) is not dict
            or not all(value is True for value in result["probes"].values())
        ):
            raise SpikeIntegrityError("sandbox denial proof is incomplete")
        payload_files = [
            (stage / relative, relative)
            for _, relative in _selected_source_files()[1:]
        ]
        if (
            _assert_tcb() != source_tcb
            or _manifest_hash(payload_files, require_stage=True)
            != PYTHON_STDLIB_PAYLOAD_SHA256
            or _sha256_fd(executable_fd) != PYTHON_SOURCE_EXECUTABLE_SHA256
            or os.fstat(executable_fd).st_nlink != 0
        ):
            raise SpikeIntegrityError("TCB changed across sandbox worker execution")
        _assert_stage_anchor(stage, *anchor)
        result["tcb"] = tcb_before
        result_value = result
    except SpikeIntegrityError as exc:
        primary_error = exc
    except Exception as exc:
        wrapped = SpikeIntegrityError("sandbox worker timed out or violated protocol")
        wrapped.__cause__ = exc
        primary_error = wrapped
    except BaseException as exc:
        primary_error = exc
    finally:
        cleanup_errors: list[BaseException] = []
        if parent is not None:
            try:
                parent.close()
            except BaseException as exc:
                cleanup_errors.append(exc)
        if child is not None:
            try:
                child.close()
            except BaseException as exc:
                cleanup_errors.append(exc)
        if process is not None:
            try:
                _terminate(process)
            except BaseException as exc:
                cleanup_errors.append(exc)
        for descriptor in (executable_fd, bin_fd, source_fd):
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except BaseException as exc:
                    cleanup_errors.append(exc)
        try:
            if late_path_created:
                LATE_ESCAPE_PATH.unlink()
                LATE_ESCAPE_ROOT.rmdir()
        except BaseException as exc:
            cleanup_errors.append(exc)
        anchor_values = (catnat_fd, staging_parent_fd, stage_fd)
        complete_anchor = (
            tuple(anchor_values)
            if all(value is not None for value in anchor_values)
            else None
        )
        try:
            _cleanup_stage(stage, complete_anchor)  # type: ignore[arg-type]
        except BaseException as exc:
            cleanup_errors.append(exc)
        for descriptor in (stage_fd, staging_parent_fd, catnat_fd):
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except BaseException as exc:
                    cleanup_errors.append(exc)
        if cleanup_errors:
            if primary_error is None:
                cleanup_error = SpikeIntegrityError("sandbox worker cleanup failed")
                cleanup_error.__cause__ = cleanup_errors[0]
                primary_error = cleanup_error
            elif hasattr(primary_error, "add_note"):
                primary_error.add_note(
                    f"cleanup also raised {type(cleanup_errors[0]).__name__}"
                )
    if primary_error is not None:
        raise primary_error
    if result_value is None:
        raise SpikeIntegrityError("sandbox worker completed without a result")
    return result_value


def run_spike(*, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Launch exactly two workers, sequentially, with no business claim."""

    tcb = _assert_tcb()
    results: list[dict[str, Any]] = []
    first = run_worker(ROLES[0], timeout_seconds=timeout_seconds)
    results.append(first)
    # ``run_worker`` has reaped worker one and destroyed its TCB before this call.
    second = run_worker(ROLES[1], timeout_seconds=timeout_seconds)
    results.append(second)
    if results[0]["pid"] == results[1]["pid"]:
        raise SpikeIntegrityError("sequential workers unexpectedly share a PID")
    if _assert_tcb() != tcb:
        raise SpikeIntegrityError("source TCB changed across the two-worker spike")
    return {
        "claim": CLAIM,
        "live_network_opened": False,
        "section_5_executed": False,
        "business_logic_executed": False,
        "workers": results,
        "tcb": tcb,
    }


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=ROLES)
    parser.add_argument("--fd", type=int)
    parser.add_argument(
        "--behavior", choices=("normal", "hang", "bad-frame"), default="normal"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.worker is not None:
        if args.fd is None:
            return 64
        return _run_worker(args.worker, args.fd, args.behavior)
    if args.fd is not None or args.behavior != "normal":
        return 64
    print(json.dumps(run_spike(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
