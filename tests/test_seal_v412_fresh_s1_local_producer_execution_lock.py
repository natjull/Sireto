from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
SUBJECT_PATH = (
    REPOSITORY
    / "scripts/seal_v412_fresh_s1_local_producer_execution_lock.py"
)
SPEC = importlib.util.spec_from_file_location("s1_lock_sealer", SUBJECT_PATH)
assert SPEC and SPEC.loader
subject = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = subject
SPEC.loader.exec_module(subject)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )


def _fixture(tmp_path: Path):
    plan = json.loads(subject.PLAN_PATH.read_bytes())
    plan["paths"]["root"] = str(tmp_path / "authority")
    return plan, _canonical(plan)


def test_build_lock_closes_all_material_pins(tmp_path: Path) -> None:
    plan, raw = _fixture(tmp_path)
    lock = subject.build_lock(
        plan, raw, trusted_output_parent=tmp_path
    )
    assert set(lock) == set(
        plan["schemas"]["execution_lock"]["exact_fields"]
    )
    assert lock["plan_sha256"] == hashlib.sha256(raw).hexdigest()
    assert lock["implementation"]["git_commit"] == subject.IMPLEMENTATION_COMMIT
    assert lock["implementation"]["provisioner_sha256"] == hashlib.sha256(
        (
            REPOSITORY
            / lock["implementation"]["provisioner_path"]
        ).read_bytes()
    ).hexdigest()
    assert lock["implementation"]["tests_sha256"] == hashlib.sha256(
        (REPOSITORY / lock["implementation"]["tests_path"]).read_bytes()
    ).hexdigest()
    assert lock["runtime"] == subject.provisioner._expected_runtime()
    assert lock["keychain_policy"] == (
        subject.provisioner._expected_keychain_policy(plan)
    )
    assert lock["volume_device"] == tmp_path.stat().st_dev
    assert lock["output_root"] == plan["paths"]["root"]
    assert "UNIMPLEMENTED" not in _canonical(lock).decode()
    subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            subject.IMPLEMENTATION_COMMIT,
            "HEAD",
        ],
        cwd=REPOSITORY,
        check=True,
    )
    for path_field, hash_field in (
        ("provisioner_path", "provisioner_sha256"),
        ("tests_path", "tests_sha256"),
    ):
        committed = subprocess.run(
            [
                "git",
                "show",
                (
                    f"{subject.IMPLEMENTATION_COMMIT}:"
                    f"{lock['implementation'][path_field]}"
                ),
            ],
            cwd=REPOSITORY,
            check=True,
            capture_output=True,
        ).stdout
        assert hashlib.sha256(committed).hexdigest() == (
            lock["implementation"][hash_field]
        )


def test_seal_lock_is_private_exclusive_and_non_clobbering(
    tmp_path: Path,
) -> None:
    plan, raw = _fixture(tmp_path)
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(raw)
    destination = tmp_path / "execution_lock.json"
    lock = subject.seal_lock(
        plan_path,
        output_path=destination,
        trusted_output_parent=tmp_path,
    )
    assert destination.read_bytes() == _canonical(lock)
    assert destination.stat().st_mode & 0o777 == 0o600
    changed = dict(lock)
    changed["logical_time_utc"] = "2026-07-30T00:00:00Z"
    parent_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(
            subject.provisioner.ProvisionError,
            match="EXISTING_ARTIFACT_DIVERGENCE",
        ):
            subject.provisioner._write_private_at(
                parent_fd, destination.name, _canonical(changed)
            )
    finally:
        os.close(parent_fd)


def test_lock_rejects_noncanonical_plan_and_contract_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, raw = _fixture(tmp_path)
    with pytest.raises(subject.LockSealError, match="PLAN_NON_CANONICAL"):
        subject.build_lock(
            plan, raw.replace(b":", b": ", 1),
            trusted_output_parent=tmp_path,
        )
    plan["authorities"]["contract"]["sha256"] = "0" * 64
    with pytest.raises(subject.LockSealError, match="CONTRACT_HASH"):
        subject.build_lock(
            plan, _canonical(plan), trusted_output_parent=tmp_path
        )


def test_sealer_itself_rejects_live_blob_commit_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, raw = _fixture(tmp_path)
    original = subject.provisioner._read_regular

    def changed(path, label, **kwargs):
        value = original(path, label, **kwargs)
        return value + b"\n" if label == "PROVISIONER" else value

    monkeypatch.setattr(subject.provisioner, "_read_regular", changed)
    with pytest.raises(subject.LockSealError, match="GIT_BLOB_MISMATCH"):
        subject.build_lock(
            plan, raw, trusted_output_parent=tmp_path
        )


def test_sealer_rejects_missing_or_nonancestor_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, raw = _fixture(tmp_path)
    monkeypatch.setattr(subject, "IMPLEMENTATION_COMMIT", "0" * 40)
    with pytest.raises(subject.LockSealError, match="GIT_STATUS"):
        subject.build_lock(
            plan, raw, trusted_output_parent=tmp_path
        )
    monkeypatch.setattr(
        subject,
        "IMPLEMENTATION_COMMIT",
        "ad74b4eaeeae1836e9ca08703d442b60454f2682",
    )
    real_run = subject.subprocess.run

    def nonancestor(command, **kwargs):
        if "merge-base" in command:
            return SimpleNamespace(returncode=1, stdout=b"")
        return real_run(command, **kwargs)

    monkeypatch.setattr(subject.subprocess, "run", nonancestor)
    with pytest.raises(
        subject.LockSealError, match="GIT_COMMIT_NOT_ANCESTOR"
    ):
        subject.build_lock(
            plan, raw, trusted_output_parent=tmp_path
        )


def test_main_rejects_arguments_without_writing(capsys) -> None:
    assert subject.main(["forbidden"]) == 64
    assert capsys.readouterr().err == "STOP:ARGS_FORBIDDEN\n"
    assert not (
        REPOSITORY
        / "config/v4_12_fresh_s1_local_producer_execution_lock.json"
    ).exists()
