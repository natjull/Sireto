from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from test_preflight_v412_fresh_s1_local_producer_keychain import (
    synthetic_repository,
)


REPOSITORY = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPOSITORY
    / "scripts/seal_v412_fresh_s1_local_producer_preflight_execution_lock.py"
)
SPEC = importlib.util.spec_from_file_location("v412_preflight_sealer", MODULE_PATH)
assert SPEC and SPEC.loader
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def _reader(repo: Path, commit: str, path: str) -> bytes:
    assert commit == "a" * 40
    return (repo / path).read_bytes()


def test_builds_closed_15_9_9_lock(tmp_path: Path) -> None:
    root, plan = synthetic_repository(tmp_path)
    existing = root / plan["preflight_execution_lock"]["path"]
    existing.unlink()
    lock = subject.build_execution_lock(root, "a" * 40, git_reader=_reader)
    schema = plan["preflight_execution_lock"]["schema"]
    assert set(lock) == set(schema["exact_fields"])
    assert len(lock) == 15
    assert set(lock["implementation"]) == set(schema["implementation_exact_fields"])
    assert len(lock["implementation"]) == 9
    assert set(lock["runtime"]) == set(schema["runtime_exact_fields"])
    assert len(lock["runtime"]) == 9
    assert lock["expected_uid"] == os.getuid()


def test_seals_private_exclusive_lock_only_to_injected_destination(
    tmp_path: Path,
) -> None:
    root, plan = synthetic_repository(tmp_path)
    (root / plan["preflight_execution_lock"]["path"]).unlink()
    destination = tmp_path / "sealed" / "lock.json"
    destination.parent.mkdir()
    lock = subject.seal_execution_lock(
        root, destination, "a" * 40, git_reader=_reader
    )
    assert json.loads(destination.read_text()) == lock
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    with pytest.raises(subject.PreflightError):
        subject.seal_execution_lock(
            root, destination, "a" * 40, git_reader=_reader
        )


def test_rejects_worktree_blob_divergence(tmp_path: Path) -> None:
    root, plan = synthetic_repository(tmp_path)
    (root / plan["preflight_execution_lock"]["path"]).unlink()

    def divergent(repo: Path, commit: str, path: str) -> bytes:
        return b"not the worktree blob\n"

    with pytest.raises(subject.PreflightError, match="WORKTREE_DIVERGENCE"):
        subject.build_execution_lock(root, "a" * 40, git_reader=divergent)


def test_rejects_bad_commit_before_git(tmp_path: Path) -> None:
    with pytest.raises(subject.PreflightError, match="GIT_COMMIT_FORMAT"):
        subject.git_file_bytes(tmp_path, "bad", "file")


def test_git_calls_are_absolute_closed_and_disable_replacements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[list[str], dict[str, str]]] = []

    def run(command, **kwargs):
        calls.append((command, kwargs["env"]))
        stdout = (
            "a" * 40 + "\n"
            if command[1] == "rev-list"
            else b"blob\n"
        )
        return SimpleNamespace(returncode=0, stdout=stdout)

    monkeypatch.setattr(subject.subprocess, "run", run)
    assert subject.git_file_bytes(tmp_path, "a" * 40, "path") == b"blob\n"
    assert subject._git_commit(tmp_path) == "a" * 40
    assert len(calls) == 2
    assert all(command[0] == "/usr/bin/git" for command, _ in calls)
    assert all(
        environment["GIT_NO_REPLACE_OBJECTS"] == "1"
        and environment["GIT_CONFIG_NOSYSTEM"] == "1"
        and environment["GIT_CONFIG_GLOBAL"] == "/dev/null"
        for _, environment in calls
    )


def test_main_rejects_arguments_before_any_write() -> None:
    with pytest.raises(subject.PreflightError, match="ARGV_FORBIDDEN"):
        subject.main(["program", "--forbidden"])
