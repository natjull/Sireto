from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from src.xgb_matcher import v412_service_execution_lock as lock_module


def _valid_lock() -> dict:
    source_hashes = {
        relative: hashlib.sha256(
            (lock_module.REPOSITORY / relative).read_bytes()
        ).hexdigest()
        for relative in lock_module.SOURCE_CLOSURE
    }
    return {
        "schema_version": lock_module.LOCK_SCHEMA,
        "purpose": lock_module.LOCK_PURPOSE,
        "audit_verdict": lock_module.LOCK_VERDICT,
        "git_commit": "0" * 40,
        "source_hashes": source_hashes,
        "input_hashes": lock_module.INPUT_HASHES,
        "runtime": lock_module.runtime_identity(),
        "output_root": str(lock_module.OUTPUT_ROOT),
        "query_count": 1456,
        "max_rss_bytes": 8 * 1024**3,
    }


def test_runtime_identity_uses_no_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def deny(*args, **kwargs):
        raise AssertionError("runtime identity must not spawn a process")

    monkeypatch.setattr(lock_module.subprocess, "Popen", deny)
    observed = lock_module.runtime_identity()

    uname = lock_module.os.uname()
    assert observed["platform"] == (
        f"{lock_module.sys.platform}:{uname.sysname}:"
        f"{uname.release}:{uname.machine}"
    )
    assert observed["machine"] == uname.machine


def test_execution_lock_requires_external_hash_and_stable_regular_file(
    monkeypatch,
    tmp_path: Path,
) -> None:
    payload = (
        json.dumps(_valid_lock(), sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()
    lock_path = tmp_path / "lock.json"
    lock_path.write_bytes(payload)
    monkeypatch.setattr(lock_module, "LOCK_PATH", lock_path)
    digest = hashlib.sha256(payload).hexdigest()

    _lock, observed = lock_module.validate_execution_lock(
        expected_sha256=digest,
        verify_git=False,
    )
    assert observed == digest
    with pytest.raises(ValueError, match="hash changed"):
        lock_module.validate_execution_lock(
            expected_sha256="f" * 64,
            verify_git=False,
        )

    linked = tmp_path / "linked.json"
    linked.symlink_to(lock_path)
    monkeypatch.setattr(lock_module, "LOCK_PATH", linked)
    with pytest.raises(ValueError, match="symlink"):
        lock_module.validate_execution_lock(
            expected_sha256=digest,
            verify_git=False,
        )


def test_loaded_repository_module_must_be_in_source_closure() -> None:
    name = "_v412_unsealed_test_module"
    sys.modules[name] = SimpleNamespace(
        __file__=str(lock_module.REPOSITORY / "analyze_samples.py")
    )
    try:
        with pytest.raises(ValueError, match="not sealed"):
            lock_module.validate_loaded_repository_modules({})
    finally:
        del sys.modules[name]
