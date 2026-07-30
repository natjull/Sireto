from __future__ import annotations

import hashlib
import importlib.util
import ast
import json
import os
import struct
import sys
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/launch_v412_fresh_intake_synthetic_scanner_sealer.py"
)
SPEC = importlib.util.spec_from_file_location("v412_s0_launcher", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)


def _hashed_message(value: dict) -> dict:
    result = dict(value)
    result["message_sha256"] = hashlib.sha256(
        launcher.canonical_json(result, final_lf=False)
    ).hexdigest()
    return result


def _ready(run_id: str = "a" * 64, attempt_id: str = "b" * 64) -> dict:
    return _hashed_message(
        {
            "schema_version": "sireto-v4.12-fresh-s0-control-ready-1",
            "message_type": "READY",
            "synthetic_run_id": run_id,
            "attempt_id": attempt_id,
            "worker_pid": 123,
            "payload_fd_roles": list(launcher.WORKER_PAYLOAD_ROLES),
        }
    )


def _authority(*, success: bool) -> dict:
    if not success:
        return {key: None for key in launcher.OUTPUT_AUTHORITY_FIELDS}
    return {
        "sealed_input_payload_manifest_sha256": "1" * 64,
        "sealed_input_seal_sha256": "2" * 64,
        "terminal_tree_kind": "SCAN_OUTPUT",
        "terminal_tree_payload_manifest_sha256": "3" * 64,
        "terminal_tree_seal_sha256": "4" * 64,
        "journal_generation": 3,
        "journal_generation_manifest_sha256": "5" * 64,
        "journal_head_event_sha256": "6" * 64,
    }


def _terminal(*, success: bool) -> dict:
    return _hashed_message(
        {
            "schema_version": "sireto-v4.12-fresh-s0-control-result-1",
            "message_type": "RESULT" if success else "STOP",
            "synthetic_run_id": "a" * 64,
            "attempt_id": "b" * 64,
            "phase": "WORKER",
            "reason_code": "OK" if success else "WORKER_CONTROLLED_STOP",
            "terminal_result": (
                "INGESTED_SYNTHETIC_SCANNER_SEALER_V412"
                if success
                else "STOP_SYNTHETIC_SCANNER_SEALER_V412"
            ),
            "stability": {
                "same_worker_process": True,
                "same_five_payload_fds": True,
                "monotonic_elapsed_seconds": "60.000000000" if success else "0.1",
            },
            "output_authority": _authority(success=success),
        }
    )


def test_public_main_rejects_every_argument_without_launching(monkeypatch, capsys) -> None:
    called = False

    def forbidden() -> dict:
        nonlocal called
        called = True
        raise AssertionError

    monkeypatch.setattr(launcher, "run_authoritative_launch", forbidden)
    assert launcher.main(["--lock", "/tmp/attacker"]) == 2
    assert called is False
    assert "accepts no arguments" in capsys.readouterr().err


def test_public_main_has_fixed_authorization_path() -> None:
    assert launcher.AUTHORIZATION_RELATIVE_PATH.as_posix() == (
        "config/v4_12_fresh_s0_launch_authorization.json"
    )


def test_git_head_and_plan_blob_are_read_without_child_process() -> None:
    head = launcher._git_head()
    assert launcher.COMMIT40.fullmatch(head)
    raw = launcher._git_blob(head, launcher.PLAN_RELATIVE_PATH)
    value = json.loads(raw)
    assert value["status"] == "PREREGISTERED_DO_NOT_IMPLEMENT_UNTIL_AUDIT"
    assert launcher._git_is_ancestor(
        "46b1958b21c2741d9922d252da9f4e81175c385c", head
    )


def test_canonical_json_rejects_duplicate_noncanonical_and_nonfinite() -> None:
    for raw in (
        b'{"a":1,"a":2}\n',
        b'{ "a":1}\n',
        b'{"a":NaN}\n',
        b'[]\n',
    ):
        with pytest.raises(launcher.LauncherStop) as caught:
            launcher.decode_canonical_json(raw, "bad")
        assert caught.value.reason_code == "AUTHORIZATION_INVALID"


def _directory_canary_authorities(directory: Path) -> tuple[bytes, dict, dict]:
    code = "DENY_PARENT_ENUMERATION"
    record = {
        "code": code,
        "kind": "EXISTING_DIRECTORY",
        "absolute_path_or_capability": os.fspath(directory),
        "identity": None,
        "size_bytes": None,
        "sha256": None,
    }
    manifest = {
        "schema_version": "sireto-v4.12-fresh-s0-canary-manifest-1",
        "synthetic_run_id": "a" * 64,
        "ordered_records": [record],
        "record_count": 1,
        "records_sha256": hashlib.sha256(
            launcher.canonical_json([record], final_lf=False)
        ).hexdigest(),
    }
    lock = {
        "synthetic_run_id": manifest["synthetic_run_id"],
        "runtime": {
            "system": {
                "volumes": {
                    "run": {
                        "device": directory.parent.stat().st_dev,
                        "volume_uuid": "synthetic-volume",
                    }
                }
            }
        },
    }
    plan = {
        "schema_definitions": {
            "canary_manifest": {
                "exact_fields": list(manifest),
            }
        },
        "canary_matrix": {
            "runtime_codes_exact_order": [code],
            "synthetic_target_by_runtime_code": {code: os.fspath(directory)},
        },
        "paths": {"allowed_root": os.fspath(directory.parent)},
    }
    return launcher.canonical_json(manifest), lock, plan


def test_directory_canary_is_anchored_and_requires_safe_existing_directory(
    tmp_path: Path,
) -> None:
    target = tmp_path / "forbidden"
    target.mkdir(mode=0o700)
    raw, lock, plan = _directory_canary_authorities(target)
    launcher._validate_canary_manifest(raw, lock, plan)

    target.chmod(0o777)
    with pytest.raises(launcher.LauncherStop) as unsafe:
        launcher._validate_canary_manifest(raw, lock, plan)
    assert unsafe.value.reason_code == "SANDBOX_EXPECTATION_FAILED"

    target.chmod(0o700)
    target.rmdir()
    with pytest.raises(launcher.LauncherStop) as missing:
        launcher._validate_canary_manifest(raw, lock, plan)
    assert missing.value.reason_code == "SANDBOX_EXPECTATION_FAILED"

    replacement = tmp_path / "replacement"
    replacement.mkdir(mode=0o700)
    target.symlink_to(replacement, target_is_directory=True)
    with pytest.raises(launcher.LauncherStop) as symlink:
        launcher._validate_canary_manifest(raw, lock, plan)
    assert symlink.value.reason_code == "SANDBOX_EXPECTATION_FAILED"


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS runtime authority")
def test_root_owned_macos_runtime_files_use_explicit_trusted_owner() -> None:
    for path in (
        Path("/System/Library/CoreServices/SystemVersion.plist"),
        Path("/usr/bin/sandbox-exec"),
    ):
        fd = launcher._open_anchored(path)
        try:
            raw, info = launcher._read_regular_fd(
                fd,
                os.fspath(path),
                expected_uid=0,
            )
        finally:
            os.close(fd)
        assert raw
        assert info.st_uid == 0


@pytest.mark.skipif(sys.platform != "darwin", reason="APFS volume authority")
def test_volume_uuid_resolution_does_not_cache_by_device() -> None:
    repository_fd = launcher._open_anchored(
        launcher.REPOSITORY_ROOT,
        directory=True,
    )
    root_fd = os.open(
        "/",
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        resolver = launcher.VolumeUUIDResolver()
        resolver.for_fd(repository_fd)
        root_after_repository = resolver.for_fd(root_fd)
        root_independent = launcher.VolumeUUIDResolver().for_fd(root_fd)
    finally:
        os.close(repository_fd)
        os.close(root_fd)
    assert root_after_repository == root_independent


def test_control_frames_are_compact_canonical_without_lf() -> None:
    ready = _ready()
    terminal = _terminal(success=True)
    ready_raw = launcher.canonical_json(ready, final_lf=False)
    terminal_raw = launcher.canonical_json(terminal, final_lf=False)
    buffer = bytearray(
        struct.pack(">I", len(ready_raw))
        + ready_raw
        + struct.pack(">I", len(terminal_raw))
        + terminal_raw
    )
    assert launcher._parse_frames(buffer, eof=True) == [ready, terminal]
    assert buffer == bytearray()


def test_control_frame_rejects_partial_extra_and_lf() -> None:
    raw = launcher.canonical_json(_ready(), final_lf=False)
    with pytest.raises(launcher.LauncherStop):
        launcher._parse_frames(bytearray(struct.pack(">I", len(raw) + 1) + raw), eof=True)
    with pytest.raises(launcher.LauncherStop):
        launcher._parse_frames(
            bytearray(struct.pack(">I", len(raw) + 1) + raw + b"\n"), eof=True
        )


def test_ready_and_terminal_validation_are_bound_to_spawn() -> None:
    lock = {"synthetic_run_id": "a" * 64, "attempt_id": "b" * 64}
    launcher._validate_ready(_ready(), lock, 123)
    launcher._validate_terminal(_terminal(success=True), lock)
    launcher._validate_terminal(_terminal(success=False), lock)
    tampered = _ready()
    tampered["worker_pid"] = 124
    with pytest.raises(launcher.LauncherStop) as caught:
        launcher._validate_ready(tampered, lock, 123)
    assert caught.value.reason_code == "WORKER_READY_INVALID"


def test_success_requires_real_sixty_second_same_process_and_fds() -> None:
    assert launcher._success_stability(
        {
            "same_worker_process": True,
            "same_five_payload_fds": True,
            "monotonic_elapsed_seconds": "60.000000000",
        }
    )
    assert not launcher._success_stability(
        {
            "same_worker_process": True,
            "same_five_payload_fds": True,
            "monotonic_elapsed_seconds": "59.999999999",
        }
    )
    assert not launcher._success_stability(
        {
            "same_worker_process": False,
            "same_five_payload_fds": True,
            "monotonic_elapsed_seconds": "61",
        }
    )


def test_claim_is_bound_to_authorization_lock_run_and_attempt() -> None:
    lock = {
        "implementation_commit": "0" * 40,
        "synthetic_run_id": "a" * 64,
        "attempt_id": "b" * 64,
    }
    claim = launcher._claim_value(lock, b"auth\n", b"lock\n")
    assert set(claim) == set(launcher.CLAIM_FIELDS)
    assert claim["authorization_manifest_sha256"] == hashlib.sha256(b"auth\n").hexdigest()
    assert claim["execution_lock_sha256"] == hashlib.sha256(b"lock\n").hexdigest()
    assert claim["claim_status"] == "CLAIMED_PRE_SPAWN"


def test_complete_but_forged_existing_receipt_is_rejected(
    tmp_path: Path,
) -> None:
    auth_raw = b'{"authorization":"synthetic"}\n'
    lock_raw = b'{"lock":"synthetic"}\n'
    claim_raw = b'{"claim":"synthetic"}\n'
    limitations = ["LIMIT"]
    lock = {
        "implementation_commit": "0" * 40,
        "synthetic_run_id": "a" * 64,
        "attempt_id": "b" * 64,
        "runtime": {},
        "paths": {"audit_parent": "/synthetic/audit/parent"},
    }
    implementation_hashes = {"SANDBOX_PROFILE": "1" * 64}
    receipt = {
        "schema_version": "sireto-v4.12-fresh-s0-authoritative-launch-receipt-2",
        "phase": "CLAIM",
        "reason_code": "STOP_NO_RERUN",
        "authorization_manifest_sha256": hashlib.sha256(auth_raw).hexdigest(),
        "execution_lock_path": "/synthetic/execution_lock.json",
        "execution_lock_sha256": hashlib.sha256(lock_raw).hexdigest(),
        "implementation_commit": lock["implementation_commit"],
        "implementation_blob_hashes": implementation_hashes,
        # Complete schema, deliberately forged binding.
        "synthetic_run_id": "c" * 64,
        "attempt_id": lock["attempt_id"],
        "claim_sha256": hashlib.sha256(claim_raw).hexdigest(),
        "lease_path": "/synthetic/audit/parent/leases/" + "b" * 64 + ".lease",
        "lease_held_for_spawn": False,
        "runtime": {},
        "sandbox_profile_sha256": "1" * 64,
        "effective_sandbox_profile_sha256": "2" * 64,
        "parent_before_observations": [],
        "worker_receipt": None,
        "parent_after_observations": None,
        "stability": {
            "same_worker_process": None,
            "same_five_payload_fds": None,
            "monotonic_elapsed_seconds": None,
        },
        "canaries": [],
        "output_authority": {
            key: None for key in launcher.OUTPUT_AUTHORITY_FIELDS
        },
        "macos_limitations": limitations,
        "terminal_result": "STOP_SYNTHETIC_SCANNER_SEALER_V412",
        "verdict": "STOP",
        "started_at_utc": "2026-07-30T00:00:00Z",
        "finished_at_utc": "2026-07-30T00:00:01Z",
    }
    path = tmp_path / "receipt.json"
    path.write_bytes(launcher.canonical_json(receipt))
    path.chmod(0o400)
    plan = {
        "macos_runtime_boundary": {"acknowledged_limitations": limitations},
        "enum_definitions": {
            "launch_phases": ["CLAIM"],
            "reason_codes": ["STOP_NO_RERUN"],
            "scanner_terminal_results": [
                "STOP_SYNTHETIC_SCANNER_SEALER_V412"
            ],
        },
    }
    with pytest.raises(launcher.LauncherStop) as caught:
        launcher._validate_existing_receipt(
            path,
            lock=lock,
            auth_raw=auth_raw,
            lock_raw=lock_raw,
            claim_raw=claim_raw,
            execution_lock_path="/synthetic/execution_lock.json",
            plan=plan,
            implementation_hashes=implementation_hashes,
        )
    assert caught.value.reason_code == "RECEIPT_CONFLICT"


def test_invalid_receipt_under_existing_claim_maps_to_stop_no_rerun() -> None:
    original = launcher.LauncherStop(
        "RECEIPT", "RECEIPT_CONFLICT", "forged"
    )
    with pytest.raises(launcher.LauncherStop) as caught:
        launcher._invalid_recovery_receipt(original)
    assert caught.value.phase == "CLAIM"
    assert caught.value.reason_code == "STOP_NO_RERUN"


def test_existing_invalid_canary_proof_is_not_treated_as_absent(
    tmp_path: Path,
) -> None:
    audit = tmp_path / "audit"
    audit.mkdir(mode=0o700)
    (audit / "canaries.json").write_bytes(b'{"partial":true}\n')
    (audit / "canaries.json").chmod(0o600)
    fd = launcher._open_anchored(audit, directory=True)
    try:
        with pytest.raises(launcher.LauncherStop) as caught:
            launcher._read_stop_canaries(
                fd,
                {"synthetic_run_id": "a" * 64, "attempt_id": "b" * 64},
                {"canary_matrix": {}},
            )
    finally:
        os.close(fd)
    assert caught.value.phase == "POSTWORKER"
    assert caught.value.reason_code == "SANDBOX_EXPECTATION_FAILED"


def test_partial_canary_after_claim_publishes_no_receipt_then_no_rerun(
    tmp_path: Path,
) -> None:
    audit = tmp_path / "audit"
    audit.mkdir(mode=0o700)
    (audit / "canaries.json").write_bytes(b'{"partial":true}\n')
    (audit / "canaries.json").chmod(0o600)
    claim = tmp_path / "claim.json"
    claim.write_bytes(b'{"claimed":true}\n')
    claim.chmod(0o400)
    receipt = tmp_path / "receipt.json"
    fd = launcher._open_anchored(audit, directory=True)
    try:
        with pytest.raises(launcher.LauncherStop) as first:
            launcher._read_stop_canaries(
                fd,
                {"synthetic_run_id": "a" * 64, "attempt_id": "b" * 64},
                {"canary_matrix": {}},
            )
    finally:
        os.close(fd)
    assert first.value.reason_code == "SANDBOX_EXPECTATION_FAILED"
    assert claim.exists()
    assert not receipt.exists()
    with pytest.raises(launcher.LauncherStop) as resumed:
        launcher._claim_without_receipt()
    assert resumed.value.phase == "CLAIM"
    assert resumed.value.reason_code == "STOP_NO_RERUN"


def test_empty_idempotent_canaries_require_proof_file_absence(
    tmp_path: Path,
) -> None:
    audit = tmp_path / "audit"
    audit.mkdir(mode=0o700)
    fd = launcher._open_anchored(audit, directory=True)
    try:
        launcher._require_absent_canary_proof(fd)
        (audit / "canaries.json").write_bytes(b"anything")
        (audit / "canaries.json").chmod(0o600)
        with pytest.raises(launcher.LauncherStop) as caught:
            launcher._require_absent_canary_proof(fd)
    finally:
        os.close(fd)
    assert caught.value.reason_code == "RECEIPT_CONFLICT"


def test_canary_matrix_requires_exact_order_denials_and_errno() -> None:
    codes = [
        "DENY_DATA",
        "DENY_MODELS",
    ]
    plan = {
        "paths": {"allowed_root": "/synthetic"},
        "canary_matrix": {
            "runtime_codes_exact_order": codes,
            "errno_allowed_by_runtime_code": {
                "DENY_DATA": [1, 13],
                "DENY_MODELS": [1, 13],
            },
            "synthetic_target_by_runtime_code": {
                "DENY_DATA": "<allowed_root>/DENY_DATA",
                "DENY_MODELS": "<allowed_root>/DENY_MODELS",
            },
        }
    }
    records = [
        {
            "code": code,
            "operation": "OPEN_READ",
            "synthetic_target": f"/synthetic/{code}",
            "result": "DENIED",
            "errno": 13,
        }
        for code in codes
    ]
    launcher._validate_canaries(records, plan, run_id="a" * 64)
    with pytest.raises(launcher.LauncherStop):
        launcher._validate_canaries(
            list(reversed(records)), plan, run_id="a" * 64
        )
    wrong_operation = [dict(record) for record in records]
    wrong_operation[0]["operation"] = "WRITE"
    with pytest.raises(launcher.LauncherStop):
        launcher._validate_canaries(
            wrong_operation, plan, run_id="a" * 64
        )
    wrong_target = [dict(record) for record in records]
    wrong_target[0]["synthetic_target"] = "/synthetic/other"
    with pytest.raises(launcher.LauncherStop):
        launcher._validate_canaries(wrong_target, plan, run_id="a" * 64)
    bool_errno = [dict(record) for record in records]
    bool_errno[0]["errno"] = True
    with pytest.raises(launcher.LauncherStop):
        launcher._validate_canaries(bool_errno, plan, run_id="a" * 64)
    scalar = list(records)
    scalar[0] = "not-a-record"
    with pytest.raises(launcher.LauncherStop) as caught:
        launcher._validate_canaries(scalar, plan, run_id="a" * 64)
    assert caught.value.reason_code == "SANDBOX_EXPECTATION_FAILED"


def test_result_row_rejects_unlisted_exit_combination() -> None:
    execution = launcher.WorkerExecution(
        pid=10,
        exit_code=1,
        signal_number=None,
        stdout=b"",
        stderr=b"",
        ready=_ready(),
        result=_terminal(success=True),
    )
    assert launcher._result_row(execution) == (
        "STOP",
        "WORKER_EXIT_INVALID",
        "STOP_SYNTHETIC_SCANNER_SEALER_V412",
    )


def test_stability_downgrade_preserves_observed_control_and_output() -> None:
    result = _terminal(success=True)
    result["stability"]["monotonic_elapsed_seconds"] = "59.500000000"
    result["message_sha256"] = launcher._message_hash(result)
    execution = launcher.WorkerExecution(
        pid=10,
        exit_code=0,
        signal_number=None,
        stdout=b"",
        stderr=b"",
        ready=_ready(),
        result=result,
    )
    verdict, reason, terminal, stability = launcher._select_receipt_outcome(
        execution
    )
    assert (verdict, reason, terminal) == (
        "STOP",
        "STABILITY_FAILED",
        "STOP_SYNTHETIC_SCANNER_SEALER_V412",
    )
    assert stability == result["stability"]
    assert execution.result["output_authority"] == _authority(success=True)


def test_parent_validates_exact_sealed_tree_and_rejects_extra(
    tmp_path: Path,
) -> None:
    tree = tmp_path / "tree"
    tree.mkdir(mode=0o700)
    payload = b"synthetic-only\n"
    (tree / "payload.txt").write_bytes(payload)
    (tree / "payload.txt").chmod(0o600)
    records = [
        {
            "relative_path": "payload.txt",
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    ]
    manifest = {
        "schema_version": "sireto-v4.12-fresh-synthetic-payload-manifest-1",
        "package_kind": "SCAN_OUTPUT",
        "synthetic_run_id": "a" * 64,
        "collection_id": "collection-synthetic-001",
        "source_batch_id": "batch-synthetic-001",
        "logical_time_utc": "2026-07-30T00:00:00Z",
        "ordered_payload_records": records,
        "payload_count": 1,
        "payload_tree_sha256": hashlib.sha256(
            launcher.canonical_json(records, final_lf=False)
        ).hexdigest(),
    }
    manifest_raw = launcher.canonical_json(manifest)
    seal = {
        "schema_version": "sireto-v4.12-fresh-synthetic-seal-1",
        "package_kind": manifest["package_kind"],
        "synthetic_run_id": manifest["synthetic_run_id"],
        "collection_id": manifest["collection_id"],
        "source_batch_id": manifest["source_batch_id"],
        "logical_time_utc": manifest["logical_time_utc"],
        "payload_manifest_size_bytes": len(manifest_raw),
        "payload_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "payload_tree_sha256": manifest["payload_tree_sha256"],
    }
    (tree / "payload_manifest.json").write_bytes(manifest_raw)
    (tree / "seal.json").write_bytes(launcher.canonical_json(seal))
    for path in (tree / "payload_manifest.json", tree / "seal.json"):
        path.chmod(0o600)
    assert launcher._validate_sealed_tree(
        tree, expected_package_kind="SCAN_OUTPUT", run_id="a" * 64
    ) == (
        hashlib.sha256(manifest_raw).hexdigest(),
        hashlib.sha256(launcher.canonical_json(seal)).hexdigest(),
    )
    (tree / "extra").write_bytes(b"forbidden")
    (tree / "extra").chmod(0o600)
    with pytest.raises(launcher.LauncherStop):
        launcher._validate_sealed_tree(
            tree, expected_package_kind="SCAN_OUTPUT", run_id="a" * 64
        )


def test_parent_sealed_tree_detects_identity_drift_between_snapshots(
    tmp_path: Path, monkeypatch
) -> None:
    tree = tmp_path / "tree"
    tree.mkdir(mode=0o700)
    payload = b"synthetic-only\n"
    payload_path = tree / "payload.txt"
    payload_path.write_bytes(payload)
    payload_path.chmod(0o600)
    records = [
        {
            "relative_path": "payload.txt",
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    ]
    manifest = {
        "schema_version": "sireto-v4.12-fresh-synthetic-payload-manifest-1",
        "package_kind": "SCAN_OUTPUT",
        "synthetic_run_id": "a" * 64,
        "collection_id": "collection-synthetic-001",
        "source_batch_id": "batch-synthetic-001",
        "logical_time_utc": "2026-07-30T00:00:00Z",
        "ordered_payload_records": records,
        "payload_count": 1,
        "payload_tree_sha256": hashlib.sha256(
            launcher.canonical_json(records, final_lf=False)
        ).hexdigest(),
    }
    manifest_raw = launcher.canonical_json(manifest)
    seal = {
        "schema_version": "sireto-v4.12-fresh-synthetic-seal-1",
        "package_kind": manifest["package_kind"],
        "synthetic_run_id": manifest["synthetic_run_id"],
        "collection_id": manifest["collection_id"],
        "source_batch_id": manifest["source_batch_id"],
        "logical_time_utc": manifest["logical_time_utc"],
        "payload_manifest_size_bytes": len(manifest_raw),
        "payload_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "payload_tree_sha256": manifest["payload_tree_sha256"],
    }
    (tree / "payload_manifest.json").write_bytes(manifest_raw)
    (tree / "seal.json").write_bytes(launcher.canonical_json(seal))
    for path in (tree / "payload_manifest.json", tree / "seal.json"):
        path.chmod(0o600)
    real_listdir = launcher.os.listdir
    calls = 0

    def mutate_on_second_snapshot(fd):
        nonlocal calls
        calls += 1
        if calls == 2:
            payload_path.write_bytes(payload)
            payload_path.chmod(0o600)
        return real_listdir(fd)

    monkeypatch.setattr(launcher.os, "listdir", mutate_on_second_snapshot)
    with pytest.raises(launcher.LauncherStop, match="drift"):
        launcher._validate_sealed_tree(
            tree, expected_package_kind="SCAN_OUTPUT", run_id="a" * 64
        )


def test_static_public_contract_has_one_popen_and_no_argument_parser() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    subprocess_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "subprocess"
    ]
    assert len(subprocess_calls) == 1
    assert subprocess_calls[0].func.attr == "Popen"
    assert source.count("subprocess.Popen(") == 1
    assert "argparse" not in source
    assert "AUTHORIZATION_RELATIVE_PATH = Path(" in source
    assert "os.environ" not in source
