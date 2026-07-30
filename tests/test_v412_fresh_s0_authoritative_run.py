from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import socket
import stat
import sys
from pathlib import Path

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
PLAN_PATH = REPOSITORY / "config/v4_12_fresh_s0_authoritative_run_plan.json"
CONTRACT_PATH = REPOSITORY / "docs/v4_12_fresh_s0_authoritative_run_contract.md"
SEALER_PATH = (
    REPOSITORY
    / "scripts/seal_v412_fresh_intake_synthetic_execution_lock.py"
)
LAUNCHER_PATH = (
    REPOSITORY
    / "scripts/launch_v412_fresh_intake_synthetic_scanner_sealer.py"
)
WORKER_PATH = REPOSITORY / "scripts/run_v412_fresh_s0_worker.py"
PROFILE_PATH = (
    REPOSITORY
    / "config/v4_12_fresh_intake_synthetic_scanner_sealer.sb"
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sealer = _load("v412_s0_sealer_integration", SEALER_PATH)
launcher = _load("v412_s0_launcher_integration", LAUNCHER_PATH)
worker = _load("v412_s0_worker_integration", WORKER_PATH)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(fd: int, volume_uuid: str) -> dict[str, object]:
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


class _FixedResolver:
    def __init__(self, value: str) -> None:
        self.value = value

    def for_fd(self, _fd: int) -> str:
        return self.value


def test_authoritative_plan_contract_and_source_roles_are_cross_pinned() -> None:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    assert _sha(PLAN_PATH) == sealer.EXPECTED_PLAN_SHA256
    assert _sha(CONTRACT_PATH) == sealer.EXPECTED_CONTRACT_SHA256
    assert plan["contract"]["sha256"] == _sha(CONTRACT_PATH)
    assert tuple(sealer.IMPLEMENTATION_ROLE_PATHS) == tuple(
        plan["execution_lock"]["implementation_blob_roles"]
    )
    for relative in sealer.IMPLEMENTATION_ROLE_PATHS.values():
        assert (REPOSITORY / relative).is_file()


def test_all_fd_role_orders_match_across_sealer_launcher_and_worker() -> None:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    assert tuple(plan["fd_protocol"]["lock_input_roles_exact_order"]) == (
        launcher.LOCK_INPUT_ROLES
    )
    assert tuple(
        plan["fd_protocol"]["parent_retained_roles_exact_order"]
    ) == launcher.PARENT_RETAINED_ROLES
    assert tuple(
        plan["fd_protocol"]["worker_payload_roles_exact_order"]
    ) == launcher.WORKER_PAYLOAD_ROLES == worker.PAYLOAD_ROLES
    assert launcher.WRITE_DIRECTORY_ROLES == worker.WRITE_ROLES


def test_profile_real_template_renders_with_no_residual_placeholder(
    tmp_path: Path,
) -> None:
    plan, _core = sealer._load_plans()
    rendered = sealer._render_profile(
        PROFILE_PATH.read_bytes(),
        plan,
        tmp_path,
        "a" * 64,
        tmp_path / "runtime" / ("a" * 64),
    )
    assert b"@@" not in rendered
    assert rendered.endswith(b"\n")
    assert b"(deny default)" in rendered


def test_launcher_worker_spec_is_accepted_by_worker(tmp_path: Path) -> None:
    run_id = "a" * 64
    attempt_id = "b" * 64
    volume_uuid = "00000000-0000-0000-0000-000000000001"
    payloads = []
    opened: list[int] = []
    write_fds: dict[str, int] = {}
    parent, child = socket.socketpair()
    try:
        for index, role in enumerate(worker.PAYLOAD_ROLES):
            path = tmp_path / f"payload-{index}"
            path.write_bytes(f"{role}\n".encode())
            path.chmod(0o600)
            fd = os.open(path, os.O_RDONLY)
            opened.append(fd)
            payloads.append(
                launcher.OpenAuthority(
                    role=role,
                    path=path,
                    fd=fd,
                    expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    expected_size=path.stat().st_size,
                    expected_identity=_identity(fd, volume_uuid),
                )
            )
        for role in worker.WRITE_ROLES:
            path = tmp_path / role.lower()
            path.mkdir(mode=0o700)
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            opened.append(fd)
            write_fds[role] = fd
        lock_raw = b'{"synthetic":"lock"}\n'
        lock = {
            "implementation_commit": "1" * 40,
            "synthetic_run_id": run_id,
            "attempt_id": attempt_id,
            "logical_time_utc": "2026-07-30T00:00:00Z",
        }
        value = launcher._worker_spec(
            lock,
            lock_raw,
            payloads,
            write_fds,
            _FixedResolver(volume_uuid),
        )
        spec_path = tmp_path / "worker-spec.json"
        spec_path.write_bytes(launcher.canonical_json(value))
        spec_fd = os.open(spec_path, os.O_RDONLY)
        opened.append(spec_fd)
        worker._validate_spec(value, spec_fd, child.fileno())
    finally:
        parent.close()
        child.close()
        for fd in reversed(opened):
            try:
                os.close(fd)
            except OSError:
                pass


def test_worker_control_messages_are_accepted_by_launcher() -> None:
    run_id = "a" * 64
    attempt_id = "b" * 64
    spec = {"synthetic_run_id": run_id, "attempt_id": attempt_id}
    lock = {"synthetic_run_id": run_id, "attempt_id": attempt_id}
    ready = worker._ready_message(spec)
    launcher._validate_ready(ready, lock, os.getpid())
    terminal = worker._terminal_message(
        spec,
        message_type="RESULT",
        reason_code="OK",
        terminal_result="INGESTED_SYNTHETIC_SCANNER_SEALER_V412",
        stability={
            "same_worker_process": True,
            "same_five_payload_fds": True,
            "monotonic_elapsed_seconds": "60.000000000",
        },
        output_authority={
            "sealed_input_payload_manifest_sha256": "1" * 64,
            "sealed_input_seal_sha256": "2" * 64,
            "terminal_tree_kind": "SCAN_OUTPUT",
            "terminal_tree_payload_manifest_sha256": "3" * 64,
            "terminal_tree_seal_sha256": "4" * 64,
            "journal_generation": 3,
            "journal_generation_manifest_sha256": "5" * 64,
            "journal_head_event_sha256": "6" * 64,
        },
    )
    launcher._validate_terminal(terminal, lock)


def test_canary_proof_schema_and_empty_stop_policy_are_aligned() -> None:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    assert worker.CANARY_REPORT_SCHEMA == (
        "sireto-v4.12-fresh-s0-canary-proof-1"
    )
    declared = plan["launch_receipt"]["types"]["canaries"]
    assert "exact_order" in declared
    assert "empty_array_for_stop_before_complete_canary_proof" in declared
    with pytest.raises(launcher.LauncherStop):
        launcher._validate_canaries([], plan, run_id="a" * 64)
