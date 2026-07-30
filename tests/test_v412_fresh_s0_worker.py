from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import socket
import stat
import struct
from copy import deepcopy
from pathlib import Path

import pytest

from scripts import run_v412_fresh_s0_worker as worker
from scripts import build_v412_fresh_intake_synthetic_fixture as core_builder


RUN_ID = "a" * 64
ATTEMPT_ID = "b" * 64
VOLUME_UUID = "00000000-0000-0000-0000-000000000001"


def _canonical(value: object) -> bytes:
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


def _registered_r3_inputs() -> tuple[str, str, dict, dict[str, bytes]]:
    plan, _plan_raw = worker._load_core_plan()
    identity = deepcopy(worker.EXPECTED_EXECUTION_IDENTITY)
    run_id = identity["run"]["result"]
    csv_bytes = plan["fixture"]["csv"]["exact_utf8_text"].encode("utf-8")
    evidence_bytes = core_builder._empty_evidence_parquet(plan)
    source_payloads = core_builder._manifest_objects(
        plan, run_id, csv_bytes, evidence_bytes
    )
    control = {
        "schema_version": plan["control_manifest"]["schema"],
        "synthetic_fixture": True,
        "fixture_spec_sha256": plan["control_manifest"]["fixture_spec_sha256"],
        "synthetic_run_id": run_id,
        "logical_time_utc": plan["fixture"]["logical_time_utc"],
        "batch_count": 1,
        "expected_source_row_count": 6,
        "producer_exclusions": [],
        "collection_source_manifest_sha256": hashlib.sha256(
            source_payloads["collection_source_manifest.json"]
        ).hexdigest(),
        "source_manifest_sha256": hashlib.sha256(
            source_payloads["source_manifest.json"]
        ).hexdigest(),
        "crm_safe_csv_sha256": hashlib.sha256(
            source_payloads["crm_safe.csv"]
        ).hexdigest(),
        "evidence_source_manifest_sha256": hashlib.sha256(
            source_payloads["evidence_source_manifest.json"]
        ).hexdigest(),
        "evidence_source_parquet_sha256": hashlib.sha256(
            source_payloads["evidence_source.parquet"]
        ).hexdigest(),
    }
    control_raw = _canonical(control)
    assert hashlib.sha256(control_raw).hexdigest() == (
        identity["attempt"]["values"]["fixture_control_manifest_sha256"]
    )
    payloads = {
        "CONTROL_MANIFEST": control_raw,
        **{
            role: source_payloads[name]
            for role, name in worker.PAYLOAD_NAMES.items()
        },
    }
    return run_id, identity["attempt"]["result"], identity, payloads


def _identity(fd: int) -> dict[str, object]:
    info = os.fstat(fd)
    return {
        "device": info.st_dev,
        "inode": info.st_ino,
        "size_bytes": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
        "uid": info.st_uid,
        "volume_uuid": VOLUME_UUID,
        "link_count": info.st_nlink,
        "mode": f"{stat.S_IMODE(info.st_mode):04o}",
    }


def _open_fixture(tmp_path: Path) -> tuple[dict[str, object], list[int], socket.socket]:
    all_fds: list[int] = []
    records = []
    for number, role in enumerate(worker.PAYLOAD_ROLES):
        path = tmp_path / f"payload-{number}"
        payload = f"{role}\n".encode()
        path.write_bytes(payload)
        path.chmod(0o600)
        fd = os.open(path, os.O_RDONLY)
        all_fds.append(fd)
        records.append(
            {
                "role": role,
                "fd_number": fd,
                "identity": _identity(fd),
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "access": "READ_ONLY",
            }
        )
    directories: dict[str, int] = {}
    for role in worker.WRITE_ROLES:
        path = tmp_path / role.lower()
        path.mkdir(mode=0o700)
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        all_fds.append(fd)
        directories[role] = fd
    parent, child = socket.socketpair()
    all_fds.append(child.fileno())
    spec_path = tmp_path / "spec.json"
    spec_path.write_bytes(b"{}\n")
    spec_path.chmod(0o600)
    spec_fd = os.open(spec_path, os.O_RDONLY)
    all_fds.append(spec_fd)
    spec = {
        "schema_version": worker.WORKER_SPEC_SCHEMA,
        "implementation_commit": "1" * 40,
        "execution_lock_sha256": "2" * 64,
        "synthetic_run_id": RUN_ID,
        "attempt_id": ATTEMPT_ID,
        "logical_time_utc": "2026-07-30T00:00:00Z",
        "minimum_stability_seconds": 60,
        "execution_identity": {},
        "payload_fds": records,
        "write_directory_fds": directories,
        "control_protocol": worker.CONTROL_PROTOCOL,
    }
    spec["_spec_fd"] = spec_fd
    spec["_control_fd"] = child.fileno()
    return spec, all_fds, parent


def _close_fixture(spec: dict[str, object], fds: list[int], peer: socket.socket) -> None:
    peer.close()
    control_fd = spec["_control_fd"]
    for fd in reversed(fds):
        if fd == control_fd:
            try:
                socket.socket(fileno=fd).close()
            except OSError:
                pass
        else:
            try:
                os.close(fd)
            except OSError:
                pass


def test_worker_spec_accepts_only_closed_fd_map(tmp_path: Path) -> None:
    spec, fds, peer = _open_fixture(tmp_path)
    try:
        spec_fd = spec.pop("_spec_fd")
        control_fd = spec.pop("_control_fd")
        worker._validate_spec(spec, spec_fd, control_fd)
        spec["write_directory_fds"]["EXTRA"] = spec["write_directory_fds"]["TMP"]
        with pytest.raises(worker.WorkerStop):
            worker._validate_spec(spec, spec_fd, control_fd)
    finally:
        spec.setdefault("_spec_fd", spec_fd)
        spec.setdefault("_control_fd", control_fd)
        _close_fixture(spec, fds, peer)


def test_payload_snapshot_detects_byte_drift(tmp_path: Path) -> None:
    spec, fds, peer = _open_fixture(tmp_path)
    try:
        spec_fd = spec.pop("_spec_fd")
        control_fd = spec.pop("_control_fd")
        worker._validate_spec(spec, spec_fd, control_fd)
        payloads, identities = worker._payload_snapshot(spec)
        assert tuple(payloads) == worker.PAYLOAD_ROLES
        assert tuple(identities) == worker.PAYLOAD_ROLES
        spec["payload_fds"][0]["sha256"] = "0" * 64
        with pytest.raises(worker.WorkerStop):
            worker._payload_snapshot(spec)
    finally:
        spec.setdefault("_spec_fd", spec_fd)
        spec.setdefault("_control_fd", control_fd)
        _close_fixture(spec, fds, peer)


def test_ready_hash_excludes_only_hash_field() -> None:
    spec = {"synthetic_run_id": RUN_ID, "attempt_id": ATTEMPT_ID}
    message = worker._ready_message(spec)
    digest = message.pop("message_sha256")
    assert digest == hashlib.sha256(worker._canonical_json(message)).hexdigest()
    assert message["payload_fd_roles"] == list(worker.PAYLOAD_ROLES)


def test_canary_report_is_canonical_and_worker_audit_only(tmp_path: Path) -> None:
    audit = tmp_path / "audit-worker"
    audit.mkdir(mode=0o700)
    audit_fd = os.open(audit, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    records = [
        {
            "code": code,
            "operation": (
                "ENUMERATE_PARENT"
                if code == "DENY_PARENT_ENUMERATION"
                else "OPEN_NETWORK"
                if code == "DENY_NETWORK"
                else "WRITE"
                if code == "DENY_WRITE_PARENT_AUDIT"
                else "OPEN_READ"
            ),
            "synthetic_target": f"SYNTHETIC:{code}",
            "result": "DENIED",
            "errno": 1,
        }
        for code in (
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
        )
    ]
    try:
        spec = {
            "synthetic_run_id": RUN_ID,
            "attempt_id": ATTEMPT_ID,
            "write_directory_fds": {"AUDIT": audit_fd},
        }
        report = worker._write_canary_report(spec, records)
        raw = (audit / "canaries.json").read_bytes()
        assert raw == _canonical(report)
        assert report["records_sha256"] == hashlib.sha256(
            worker._canonical_json(records)
        ).hexdigest()
        with pytest.raises(FileExistsError):
            worker._write_canary_report(spec, records)
    finally:
        os.close(audit_fd)


def test_fd_tree_authority_never_needs_the_synthetic_root_path(
    tmp_path: Path,
) -> None:
    role_fds: dict[str, int] = {}
    try:
        for role in worker.WRITE_ROLES:
            directory = tmp_path / role.lower()
            directory.mkdir(mode=0o700)
            role_fds[role] = os.open(
                directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            )
        authority = worker._FDTreeAuthority(RUN_ID, role_fds)
        try:
            authority.write_exclusive(
                f"sealed/{RUN_ID}/nested/value.json", b"{}\n"
            )
            assert authority.exists(
                f"sealed/{RUN_ID}/nested/value.json", directory=False
            )
            assert authority.read_file(
                f"sealed/{RUN_ID}/nested/value.json"
            ) == b"{}\n"
            authority.mkdir_exclusive(f"tmp/{RUN_ID}/stage")
            authority.write_exclusive(
                f"tmp/{RUN_ID}/stage/payload", b"payload"
            )
            authority.rename_exclusive(
                f"tmp/{RUN_ID}/stage", f"scan/{RUN_ID}/output"
            )
            assert authority.list(f"scan/{RUN_ID}/output") == ["payload"]
            with pytest.raises(worker.WorkerStop):
                authority.list(f"audit/{RUN_ID}/../parent")
        finally:
            authority.close()
    finally:
        for fd in role_fds.values():
            os.close(fd)


def test_worker_process_reuses_core_over_only_payload_and_directory_fds(
    tmp_path: Path,
) -> None:
    root = tmp_path / "synthetic-root"
    root.mkdir(mode=0o700)
    run_id, attempt_id, execution_identity, payload_bytes = (
        _registered_r3_inputs()
    )
    control = json.loads(payload_bytes["CONTROL_MANIFEST"])
    locations = {
        "SEALED": root / "sealed" / run_id,
        "SCAN": root / "scan" / run_id,
        "QUARANTINE": root / "quarantine" / run_id,
        "AUDIT": root / "audit" / run_id / "worker",
        "TMP": root / "tmp" / run_id,
    }
    role_fds: dict[str, int] = {}
    try:
        for role, directory in locations.items():
            directory.mkdir(mode=0o700, parents=True)
            role_fds[role] = os.open(
                directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            )
        spec = {
            "synthetic_run_id": run_id,
            "attempt_id": attempt_id,
            "logical_time_utc": control["logical_time_utc"],
            "write_directory_fds": role_fds,
        }
        terminal, authority = worker._process(
            spec,
            payload_bytes,
            execution_identity=execution_identity,
            allowed_root=root,
        )
        assert terminal == "INGESTED_SYNTHETIC_SCANNER_SEALER_V412"
        assert authority["terminal_tree_kind"] == "SCAN_OUTPUT"
        assert authority["journal_generation"] == 3
        assert all(value is not None for value in authority.values())
        assert (locations["SEALED"] / "input" / "seal.json").is_file()
        assert (locations["SCAN"] / "output" / "seal.json").is_file()
        assert len(
            list((locations["AUDIT"] / "events_manifests").iterdir())
        ) == 3
    finally:
        for fd in role_fds.values():
            os.close(fd)


def test_main_propagates_closed_identity_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent, child = socket.socketpair()
    identity = {"closed": "successor-authority"}
    spec = {
        "synthetic_run_id": RUN_ID,
        "attempt_id": ATTEMPT_ID,
        "logical_time_utc": "2026-07-30T00:00:00Z",
        "execution_identity": identity,
        "payload_fds": [
            {"role": role, "fd_number": 20 + index}
            for index, role in enumerate(worker.PAYLOAD_ROLES)
        ],
        "write_directory_fds": {
            role: 40 + index
            for index, role in enumerate(worker.WRITE_ROLES)
        },
    }
    payloads = {role: role.encode() for role in worker.PAYLOAD_ROLES}
    payloads["CONTROL_MANIFEST"] = b"{}\n"
    identities = {
        role: (1, 2, 3, 4, 5, 6, 1, 0o400)
        for role in worker.SOURCE_PAYLOAD_ROLES
    }

    monkeypatch.setattr(
        worker, "_parse_cli", lambda _argv: (7, child.fileno())
    )
    monkeypatch.setattr(
        worker,
        "_read_regular_fd",
        lambda _fd, maximum=None: _canonical(spec),
    )
    monkeypatch.setattr(worker, "_validate_spec", lambda *_args: None)
    monkeypatch.setattr(worker.fcntl, "fcntl", lambda *_args: 0)
    monkeypatch.setattr(
        worker,
        "_payload_snapshot",
        lambda _spec: (payloads, identities),
    )
    monkeypatch.setattr(worker, "_run_canaries", lambda _run_id: [])
    monkeypatch.setattr(
        worker,
        "_validate_successor_identity",
        lambda *_args, **_kwargs: (RUN_ID, ATTEMPT_ID),
    )
    monkeypatch.setattr(
        worker, "_write_canary_report", lambda _spec, _canaries: None
    )
    monkeypatch.setattr(worker.time, "sleep", lambda _seconds: None)
    ticks = iter((0.0, 1.0, 61.0))
    monkeypatch.setattr(worker.time, "monotonic", lambda: next(ticks))

    def fail_process(
        _spec: dict,
        _payloads: dict,
        *,
        execution_identity: dict,
        allowed_root: Path = worker.ALLOWED_ROOT,
    ) -> tuple[str, dict]:
        assert execution_identity == identity
        raise worker.WorkerStop(
            "closed failure",
            worker_phase="IDENTITY",
            worker_reason_code="SPEC_CONTROL_IDENTITY_MISMATCH",
        )

    monkeypatch.setattr(worker, "_process", fail_process)
    try:
        assert worker.main(["worker"]) == 2
        child.close()
        buffer = bytearray()
        while True:
            chunk = parent.recv(65_536)
            if not chunk:
                break
            buffer.extend(chunk)
    finally:
        parent.close()
        if child.fileno() >= 0:
            child.close()
    frames = []
    while buffer:
        size = struct.unpack(">I", buffer[:4])[0]
        frames.append(json.loads(bytes(buffer[4 : 4 + size])))
        del buffer[: 4 + size]
    assert [frame["message_type"] for frame in frames] == ["READY", "STOP"]
    assert frames[1]["schema_version"] == (
        "sireto-v4.12-fresh-s0-control-result-2"
    )
    assert frames[1]["worker_failure"] == {
        "schema_version": "sireto-v4.12-fresh-s0-worker-failure-1",
        "worker_phase": "IDENTITY",
        "worker_reason_code": "SPEC_CONTROL_IDENTITY_MISMATCH",
    }


def test_main_rejects_nonplan_identity_before_canaries_or_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent, child = socket.socketpair()
    run_id, attempt_id, identity, payloads = _registered_r3_inputs()
    identity["run"]["domain"] = (
        "SIRETO-V412-FRESH-SYNTHETIC-S0-NONPLAN-RUN-ID\0"
    )
    spec = {
        "synthetic_run_id": run_id,
        "attempt_id": attempt_id,
        "logical_time_utc": "2026-07-30T00:00:00Z",
        "execution_identity": identity,
        "payload_fds": [
            {"role": role, "fd_number": 20 + index}
            for index, role in enumerate(worker.PAYLOAD_ROLES)
        ],
        "write_directory_fds": {
            role: 40 + index
            for index, role in enumerate(worker.WRITE_ROLES)
        },
    }
    identities = {
        role: (1, 2, 3, 4, 5, 6, 1, 0o400)
        for role in worker.SOURCE_PAYLOAD_ROLES
    }
    forbidden: list[str] = []
    monkeypatch.setattr(
        worker, "_parse_cli", lambda _argv: (7, child.fileno())
    )
    monkeypatch.setattr(
        worker,
        "_read_regular_fd",
        lambda _fd, maximum=None: _canonical(spec),
    )
    monkeypatch.setattr(worker, "_validate_spec", lambda *_args: None)
    monkeypatch.setattr(worker.fcntl, "fcntl", lambda *_args: 0)
    monkeypatch.setattr(
        worker,
        "_payload_snapshot",
        lambda _spec: (payloads, identities),
    )
    monkeypatch.setattr(
        worker,
        "_run_canaries",
        lambda _run_id: forbidden.append("canaries"),
    )
    monkeypatch.setattr(
        worker,
        "_write_canary_report",
        lambda *_args: forbidden.append("canary_report"),
    )
    monkeypatch.setattr(
        worker,
        "_process",
        lambda *_args, **_kwargs: forbidden.append("process"),
    )
    ticks = iter((0.0, 0.1))
    monkeypatch.setattr(worker.time, "monotonic", lambda: next(ticks))
    try:
        assert worker.main(["worker"]) == 2
        child.close()
        buffer = bytearray()
        while True:
            chunk = parent.recv(65_536)
            if not chunk:
                break
            buffer.extend(chunk)
    finally:
        parent.close()
        if child.fileno() >= 0:
            child.close()
    frames = []
    while buffer:
        size = struct.unpack(">I", buffer[:4])[0]
        frames.append(json.loads(bytes(buffer[4 : 4 + size])))
        del buffer[: 4 + size]
    assert forbidden == []
    assert [frame["message_type"] for frame in frames] == ["READY", "STOP"]
    assert frames[1]["worker_failure"] == {
        "schema_version": worker.WORKER_FAILURE_SCHEMA,
        "worker_phase": "IDENTITY",
        "worker_reason_code": "RUN_DERIVATION_MISMATCH",
    }


def test_cli_is_exact_and_fd_only() -> None:
    assert worker._parse_cli(
        ["worker", "--worker-spec-fd", "7", "--worker-control-fd", "8"]
    ) == (7, 8)
    with pytest.raises(worker.WorkerStop):
        worker._parse_cli(
            ["worker", "--worker-control-fd", "8", "--worker-spec-fd", "7"]
        )
    with pytest.raises(worker.WorkerStop):
        worker._parse_cli(
            [
                "worker",
                "--worker-spec-fd",
                "7",
                "--worker-control-fd",
                "8",
                "--root",
                "/tmp",
            ]
        )


def test_worker_source_has_no_process_or_keychain_api() -> None:
    source = Path(worker.__file__).read_text()
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert "subprocess" not in imported
    assert not any(
        isinstance(node, ast.Attribute)
        and node.attr in {"fork", "forkpty", "posix_spawn", "posix_spawnp"}
        for node in ast.walk(tree)
    )
    assert "Security.framework" not in source
    assert "keychain" not in source.lower()


def test_sandbox_profile_is_deny_default_and_parent_audit_not_writable() -> None:
    profile = Path(
        "config/v4_12_fresh_intake_synthetic_scanner_sealer.sb"
    ).read_text()
    assert "(deny default)" in profile
    assert "(deny network*)" in profile
    assert "(deny process-fork)" in profile
    assert "com.apple.securityd" in profile
    assert "@@WORKER_AUDIT_ROOT@@" in profile
    assert "@@PRIVATE_RUNTIME_ROOT@@" in profile
    assert "audit/<synthetic_run_id>/parent" not in profile
    assert set(re.findall(r"@@[A-Z_]+@@", profile)) == set(
        json.loads(
            Path("config/v4_12_fresh_s0_authoritative_run_plan.json").read_text()
        )["sandbox_profile_derivation"]["placeholder_order"]
    )
