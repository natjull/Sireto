from __future__ import annotations

import importlib.util
import hashlib
import errno
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import threading
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/v412_review_python_worker_spike.py"
SPEC = importlib.util.spec_from_file_location("v412_python_worker_spike", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
spike = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = spike
SPEC.loader.exec_module(spike)


@pytest.mark.skipif(sys.platform != "darwin", reason="Seatbelt is macOS-only")
def test_pinned_relocatable_arm64_python_runtime_and_sandbox_exec() -> None:
    assert spike.PYTHON_SOURCE_EXECUTABLE.resolve(strict=True) == spike.PYTHON_SOURCE_EXECUTABLE
    assert "/opt/homebrew/bin/" not in str(spike.PYTHON_SOURCE_EXECUTABLE)
    assert spike._sha256_file(spike.PYTHON_SOURCE_EXECUTABLE) == spike.PYTHON_SOURCE_EXECUTABLE_SHA256
    assert spike._manifest_hash(
        spike._selected_source_files(), require_stage=False
    ) == spike.PYTHON_STDLIB_TREE_SHA256
    assert spike._sha256_file(spike.SANDBOX_EXEC) == spike.SANDBOX_EXEC_SHA256
    output = subprocess.run(
        ["/usr/bin/file", str(spike.PYTHON_SOURCE_EXECUTABLE)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "Mach-O 64-bit executable arm64" in output


def test_profile_is_deny_default_and_has_no_project_or_network_grant() -> None:
    stage, _ = spike._prepare_stage()
    try:
        profile = spike._sandbox_profile(stage)
        staged_python = stage / "bin/python3.12"
        assert "(deny default)" in profile
        assert "(deny network*)" in profile
        assert "(deny process-fork)" in profile
        assert "(deny process-exec)" in profile
        assert f'(allow process-exec (literal "{staged_python}"))' in profile
        assert "(deny file-write*)" in profile
        assert "(allow network" not in profile
        assert f'(subpath "{ROOT}")' not in profile
        assert str(SCRIPT.resolve()) not in profile
        assert f'(subpath "{stage}")' in profile
        assert "(allow file-read-data)" in profile
        assert '^/Volumes/CATNAT_DATA/[^0/]' in profile
        assert '^/Volumes/CATNAT_DATA/0/0[^/]' in profile
        for root in spike.DENIED_READ_ROOTS:
            assert f'(deny file-read-data (subpath "{root}"))' in profile
    finally:
        spike._destroy_stage(stage)


def test_staged_stdlib_is_private_regular_and_byte_identical() -> None:
    stage, manifest = spike._prepare_stage()
    try:
        assert manifest == spike.PYTHON_STDLIB_TREE_SHA256
        assert (stage.stat().st_mode & 0o777) == 0o700
        files = [(stage / relative, relative) for _, relative in spike._selected_source_files()]
        assert spike._manifest_hash(files, require_stage=True) == manifest
        assert not (stage / "lib/python3.12/site-packages").exists()
    finally:
        spike._destroy_stage(stage)
    assert not stage.exists()


def test_fixed_stage_is_an_atomic_concurrency_lock() -> None:
    stage, _ = spike._prepare_stage()
    try:
        with pytest.raises(spike.SpikeIntegrityError, match="not an empty private"):
            spike._prepare_stage()
    finally:
        spike._destroy_stage(stage)


def test_source_pin_drift_stops_before_staging(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(spike, "_load_source_pin", lambda: ("0" * 64, "1" * 64))
    staged = False

    def forbidden_stage():
        nonlocal staged
        staged = True
        raise AssertionError("staging must not occur after source-pin drift")

    monkeypatch.setattr(spike, "_prepare_stage", forbidden_stage)
    with pytest.raises(spike.SpikeIntegrityError, match="immutable pin"):
        spike.run_worker(spike.ROLES[0])
    assert staged is False


def _open_fd_count() -> int:
    return len(os.listdir("/dev/fd"))


@pytest.mark.parametrize(
    "target_name",
    ("catnat", "parent", "stage", "source", "executable", "bin"),
)
def test_each_prespawn_open_failure_cleans_stage_and_fds(
    target_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    targets = {
        "catnat": spike.STAGING_PARENT.parent,
        "parent": spike.STAGING_PARENT,
        "stage": spike.STAGING_PATH,
        "source": SCRIPT,
        "executable": spike.STAGING_PATH / "bin/python3.12",
        "bin": spike.STAGING_PATH / "bin",
    }
    target = targets[target_name]
    real_open = spike.os.open
    injected = False

    def failing_open(path, flags, *args, **kwargs):
        nonlocal injected
        if not injected and Path(path) == target:
            injected = True
            raise OSError(errno.EIO, f"injected open failure: {target_name}")
        return real_open(path, flags, *args, **kwargs)

    before = _open_fd_count()
    monkeypatch.setattr(spike.os, "open", failing_open)
    with pytest.raises(spike.SpikeIntegrityError):
        spike.run_worker(spike.ROLES[0])
    assert injected is True
    assert not spike.STAGING_PATH.exists()
    assert not spike.LATE_ESCAPE_ROOT.exists()
    assert _open_fd_count() == before


def test_socketpair_failure_cleans_stage_and_fds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = _open_fd_count()

    def failing_socketpair(*args, **kwargs):
        raise OSError(errno.EMFILE, "injected socketpair failure")

    monkeypatch.setattr(spike.socket, "socketpair", failing_socketpair)
    with pytest.raises(spike.SpikeIntegrityError):
        spike.run_worker(spike.ROLES[0])
    assert not spike.STAGING_PATH.exists()
    assert not spike.LATE_ESCAPE_ROOT.exists()
    assert _open_fd_count() == before


def test_popen_failure_cleans_late_path_stage_and_fds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = _open_fd_count()

    def failing_popen(*args, **kwargs):
        raise OSError(errno.EAGAIN, "injected Popen failure")

    monkeypatch.setattr(spike.subprocess, "Popen", failing_popen)
    with pytest.raises(spike.SpikeIntegrityError):
        spike.run_worker(spike.ROLES[0])
    assert not spike.STAGING_PATH.exists()
    assert not spike.LATE_ESCAPE_ROOT.exists()
    assert _open_fd_count() == before


def test_stage_cleanup_error_still_closes_every_acquired_fd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = _open_fd_count()
    real_open = spike.os.open
    real_cleanup = spike._cleanup_stage
    injected_open = False

    def failing_first_open(path, flags, *args, **kwargs):
        nonlocal injected_open
        if not injected_open and Path(path) == spike.STAGING_PARENT.parent:
            injected_open = True
            raise OSError(errno.EIO, "injected anchor failure")
        return real_open(path, flags, *args, **kwargs)

    def cleanup_then_fail(stage, anchor=None):
        real_cleanup(stage, anchor)
        raise OSError(errno.EIO, "injected cleanup failure")

    monkeypatch.setattr(spike.os, "open", failing_first_open)
    monkeypatch.setattr(spike, "_cleanup_stage", cleanup_then_fail)
    with pytest.raises(spike.SpikeIntegrityError, match="protocol"):
        spike.run_worker(spike.ROLES[0])
    assert not spike.STAGING_PATH.exists()
    assert _open_fd_count() == before


@pytest.mark.parametrize("failing_operation", ("unlink", "rmdir"))
def test_late_path_cleanup_failure_does_not_skip_stage_or_fd_cleanup(
    failing_operation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = _open_fd_count()
    real_unlink = Path.unlink
    real_rmdir = Path.rmdir

    def injected_unlink(path: Path, *args, **kwargs):
        if failing_operation == "unlink" and path == spike.LATE_ESCAPE_PATH:
            raise OSError(errno.EIO, "injected late unlink failure")
        return real_unlink(path, *args, **kwargs)

    def injected_rmdir(path: Path, *args, **kwargs):
        if failing_operation == "rmdir" and path == spike.LATE_ESCAPE_ROOT:
            raise OSError(errno.EIO, "injected late rmdir failure")
        return real_rmdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", injected_unlink)
    monkeypatch.setattr(Path, "rmdir", injected_rmdir)
    with pytest.raises(spike.SpikeIntegrityError, match="cleanup failed"):
        spike.run_worker(spike.ROLES[0])
    assert not spike.STAGING_PATH.exists()
    assert _open_fd_count() == before
    monkeypatch.undo()
    if spike.LATE_ESCAPE_PATH.exists():
        spike.LATE_ESCAPE_PATH.unlink()
    if spike.LATE_ESCAPE_ROOT.exists():
        spike.LATE_ESCAPE_ROOT.rmdir()


def test_frame_roundtrip_at_exact_ceiling() -> None:
    left, right = socket.socketpair()
    try:
        # JSON string overhead is two bytes, so this is the largest legal value.
        value = "x" * (spike.MAX_PAYLOAD_BYTES - 2)
        sender = threading.Thread(
            target=spike._send_frame,
            args=(left, spike.FRAME_PING, 17, value),
        )
        sender.start()
        assert spike._recv_frame(
            right, expected_type=spike.FRAME_PING, expected_sequence=17
        ) == (spike.FRAME_PING, 17, value)
        sender.join(timeout=2)
        assert not sender.is_alive()
        with pytest.raises(spike.FrameError, match="ceiling"):
            spike._send_frame(left, spike.FRAME_PING, 18, value + "x")
    finally:
        left.close()
        right.close()


@pytest.mark.parametrize(
    "raw, message",
    [
        (b"short", "truncated"),
        (
            struct.pack(
                "!4sBBHII", b"BAD!", spike.PROTOCOL_VERSION, spike.FRAME_PING, 0, 0, 2
            )
            + b"{}",
            "header",
        ),
        (
            struct.pack(
                "!4sBBHII",
                spike.MAGIC,
                spike.PROTOCOL_VERSION,
                spike.FRAME_PING,
                0,
                0,
                spike.MAX_PAYLOAD_BYTES + 1,
            ),
            "ceiling",
        ),
    ],
)
def test_frame_parser_rejects_malformed_inputs(raw: bytes, message: str) -> None:
    left, right = socket.socketpair()
    try:
        left.sendall(raw)
        left.shutdown(socket.SHUT_WR)
        with pytest.raises(spike.FrameError, match=message):
            spike._recv_frame(right)
    finally:
        left.close()
        right.close()


def test_frame_parser_rejects_out_of_order_sequence() -> None:
    left, right = socket.socketpair()
    try:
        spike._send_frame(left, spike.FRAME_PING, 8, {})
        with pytest.raises(spike.FrameError, match="sequence"):
            spike._recv_frame(right, expected_sequence=9)
    finally:
        left.close()
        right.close()


def test_frame_parser_uses_one_absolute_deadline_against_slow_drip() -> None:
    left, right = socket.socketpair()

    def drip() -> None:
        try:
            raw = spike.HEADER.pack(
                spike.MAGIC, spike.PROTOCOL_VERSION, spike.FRAME_PING, 0, 0, 2
            ) + b"{}"
            for byte in raw:
                left.send(bytes((byte,)))
                time.sleep(0.03)
        except OSError:
            pass

    sender = threading.Thread(target=drip, daemon=True)
    sender.start()
    started = time.monotonic()
    try:
        with pytest.raises((TimeoutError, socket.timeout)):
            spike._recv_frame(right, deadline=started + 0.12)
        assert time.monotonic() - started < 0.30
    finally:
        left.close()
        right.close()
        sender.join(timeout=1)


@pytest.mark.skipif(sys.platform != "darwin", reason="Seatbelt is macOS-only")
def test_full_spike_has_exactly_two_sequential_sandboxed_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    real_popen = spike.subprocess.Popen
    events: list[tuple[str, int]] = []
    real_send_frame = spike._send_frame

    def tracked_send_frame(sock, frame_type, sequence, value, **kwargs):
        if frame_type == spike.FRAME_GATE:
            assert not (spike.STAGING_PATH / "bin/python3.12").exists()
        return real_send_frame(sock, frame_type, sequence, value, **kwargs)

    class TrackedProcess:
        def __init__(self, process: subprocess.Popen[bytes]) -> None:
            self._process = process
            self.pid = process.pid

        @property
        def returncode(self):
            return self._process.returncode

        def poll(self):
            return self._process.poll()

        def wait(self, *args, **kwargs):
            result = self._process.wait(*args, **kwargs)
            events.append(("wait", self.pid))
            return result

        def kill(self):
            return self._process.kill()

    def tracked_popen(command, *args, **kwargs):
        if events and events[-1][0] == "spawn":
            pytest.fail("a second worker was spawned before the first was reaped")
        assert kwargs["close_fds"] is True
        assert len(kwargs["pass_fds"]) == 1
        assert kwargs["stdin"] is subprocess.DEVNULL
        assert kwargs["stdout"] is subprocess.DEVNULL
        assert kwargs["stderr"] is subprocess.DEVNULL
        assert kwargs["cwd"] == spike.STAGING_PATH
        assert kwargs["env"] == {
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/usr/bin:/bin",
            "PYTHONHASHSEED": "0",
        }
        assert command[0] == "/usr/bin/sandbox-exec"
        assert command[3].endswith("/bin/python3.12")
        assert command[3] != str(spike.PYTHON_SOURCE_EXECUTABLE)
        assert str(SCRIPT) not in command[:4]
        assert spike.LATE_ESCAPE_PATH.read_bytes() == b"must remain unreadable"
        pinned_source_hash, _ = spike._load_source_pin()
        assert hashlib.sha256(command[10].encode()).hexdigest() == pinned_source_hash
        process = TrackedProcess(real_popen(command, *args, **kwargs))
        events.append(("spawn", process.pid))
        return process

    monkeypatch.setattr(spike.subprocess, "Popen", tracked_popen)
    monkeypatch.setattr(spike, "_send_frame", tracked_send_frame)
    result = spike.run_spike(timeout_seconds=8)
    assert result["claim"] == "M3_PYTHON_SANDBOX_FEASIBILITY_ONLY"
    assert result["business_logic_executed"] is False
    assert result["live_network_opened"] is False
    assert result["section_5_executed"] is False
    assert [worker["role"] for worker in result["workers"]] == list(spike.ROLES)
    assert len({worker["pid"] for worker in result["workers"]}) == 2
    assert [kind for kind, _ in events] == ["spawn", "wait", "spawn", "wait"]
    for worker in result["workers"]:
        assert worker["inherited_af_unix_roundtrip"] is True
        assert worker["claim"] == result["claim"]
        assert worker["probes"] == {
            "af_inet_denied": True,
            "af_inet6_denied": True,
            "business_ssd_denied": True,
            "etc_hosts_denied": True,
            "execve_denied": True,
            "fork_denied": True,
            "hidden_home_denied": True,
            "late_path_denied": True,
            "new_af_unix_denied": True,
            "other_volume_denied": True,
            "private_denied": True,
            "reexec_denied_after_revocation": True,
            "posix_spawn_denied": True,
            "subprocess_denied": True,
            "staging_parent_denied": True,
            "system_volumes_alias_denied": True,
            "users_root_denied": True,
            "volumes_root_denied": True,
            "worker_parent_denied": True,
            "workspace_data_denied": True,
        }
        for key, value in result["tcb"].items():
            assert worker["tcb"][key] == value


@pytest.mark.skipif(sys.platform != "darwin", reason="Seatbelt is macOS-only")
def test_timeout_kills_and_reaps_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    real_popen = spike.subprocess.Popen
    observed: list[subprocess.Popen[bytes]] = []

    def recording_popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        observed.append(process)
        return process

    monkeypatch.setattr(spike.subprocess, "Popen", recording_popen)
    with pytest.raises(spike.SpikeIntegrityError):
        spike.run_worker(
            spike.ROLES[0], timeout_seconds=0.35, behavior="hang"
        )
    assert len(observed) == 1
    assert observed[0].poll() is not None
    with pytest.raises(ProcessLookupError):
        os.kill(observed[0].pid, 0)


@pytest.mark.skipif(sys.platform != "darwin", reason="Seatbelt is macOS-only")
def test_malformed_worker_frame_fails_closed() -> None:
    with pytest.raises(spike.SpikeIntegrityError):
        spike.run_worker(
            spike.ROLES[0], timeout_seconds=2, behavior="bad-frame"
        )


def test_python_hash_drift_stops_before_spawn(monkeypatch: pytest.MonkeyPatch) -> None:
    real_hash = spike._sha256_file

    def changed_hash(path: Path) -> str:
        if path == spike.PYTHON_SOURCE_EXECUTABLE:
            return "0" * 64
        return real_hash(path)

    spawned = False

    def forbidden_spawn(*args, **kwargs):
        nonlocal spawned
        spawned = True
        raise AssertionError("spawn must not be reached")

    monkeypatch.setattr(spike, "_sha256_file", changed_hash)
    monkeypatch.setattr(spike.subprocess, "Popen", forbidden_spawn)
    with pytest.raises(spike.SpikeIntegrityError, match="runtime executable hash"):
        spike.run_worker(spike.ROLES[0])
    assert spawned is False


def test_source_has_only_the_scoped_feasibility_claim() -> None:
    source = SCRIPT.read_text()
    assert source.count('CLAIM: Final = "M3_PYTHON_SANDBOX_FEASIBILITY_ONLY"') == 1
    assert "v412_review_collection_broker" not in source
    assert "GO_IDENTITY_BROKER_WORKER_PHASE" not in source
    assert "AUTO_MATCH" not in source
    assert "requests." not in source
    assert "urllib" not in source
