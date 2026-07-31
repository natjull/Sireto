#!/usr/bin/env python3
"""Build the sealed execution lock for the V4.12 Keychain preflight."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable

try:
    from preflight_v412_fresh_s1_local_producer_keychain import (
        PLAN_RELATIVE,
        PreflightError,
        GIT_EXECUTABLE,
        _git_environment,
        _verify_git_commit,
        _load_canonical,
        _read_regular,
        canonical_json,
        parse_canonical_object,
        sha256_bytes,
        validate_plan,
        validate_preflight_lock,
        write_exclusive_durable,
    )
except ImportError:  # pragma: no cover - package-style import in tests
    from scripts.preflight_v412_fresh_s1_local_producer_keychain import (
        PLAN_RELATIVE,
        PreflightError,
        GIT_EXECUTABLE,
        _git_environment,
        _verify_git_commit,
        _load_canonical,
        _read_regular,
        canonical_json,
        parse_canonical_object,
        sha256_bytes,
        validate_plan,
        validate_preflight_lock,
        write_exclusive_durable,
    )


REPOSITORY = Path(__file__).resolve().parents[1]


def _stop(reason: str) -> None:
    raise PreflightError(reason)


def git_file_bytes(repo_root: Path, commit: str, path: str) -> bytes:
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        _stop("GIT_COMMIT_FORMAT")
    process = subprocess.run(
        [GIT_EXECUTABLE, "cat-file", "blob", f"{commit}:{path}"],
        cwd=repo_root,
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=_git_environment(),
    )
    if process.returncode != 0:
        _stop("GIT_BLOB_READ")
    return process.stdout


def _git_commit(repo_root: Path) -> str:
    process = subprocess.run(
        [GIT_EXECUTABLE, "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        stdin=subprocess.DEVNULL,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=_git_environment(),
    )
    commit = process.stdout.strip()
    if process.returncode != 0 or len(commit) != 40:
        _stop("GIT_HEAD")
    return commit


def build_execution_lock(
    repo_root: Path,
    implementation_commit: str,
    *,
    git_reader: Callable[[Path, str, str], bytes] | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    plan, plan_raw = _load_canonical(repo_root / PLAN_RELATIVE, "PREFLIGHT_PLAN")
    validate_plan(plan)
    plan_sha = sha256_bytes(plan_raw)
    contract_raw = _read_regular(repo_root / plan["contract"]["path"], "CONTRACT")
    if sha256_bytes(contract_raw) != plan["contract"]["sha256"]:
        _stop("CONTRACT_HASH")
    authority_raw = _read_regular(
        repo_root / plan["execution_lock"]["path"], "AUTHORITY_LOCK"
    )
    if sha256_bytes(authority_raw) != plan["execution_lock"]["sha256"]:
        _stop("AUTHORITY_LOCK_HASH")
    authority = parse_canonical_object(authority_raw, "AUTHORITY_LOCK")
    if git_reader is None:
        _verify_git_commit(repo_root, implementation_commit)
        git_reader = git_file_bytes
    implementation: dict[str, Any] = {"git_commit": implementation_commit}
    mapping = (
        ("preflight", "preflight_path", "preflight_sha256"),
        ("tests", "tests_path", "tests_sha256"),
        ("sealer", "sealer_path", "sealer_sha256"),
        ("sealer_tests", "sealer_tests_path", "sealer_tests_sha256"),
    )
    for plan_key, path_field, hash_field in mapping:
        path = plan["future_implementation"][plan_key]["path"]
        raw = git_reader(repo_root, implementation_commit, path)
        if _read_regular(repo_root / path, path_field) != raw:
            _stop(f"{path_field.upper()}_WORKTREE_DIVERGENCE")
        implementation[path_field] = path
        implementation[hash_field] = hashlib.sha256(raw).hexdigest()
    lock = {
        "schema_version": plan["preflight_execution_lock"]["schema"][
            "schema_version"
        ],
        "purpose": "READ_ONLY_KEYCHAIN_LOCATOR_ABSENCE_PREFLIGHT",
        "preflight_plan_path": str(PLAN_RELATIVE),
        "preflight_plan_sha256": plan_sha,
        "contract_path": plan["contract"]["path"],
        "contract_sha256": plan["contract"]["sha256"],
        "authority_execution_lock_path": plan["execution_lock"]["path"],
        "authority_execution_lock_sha256": plan["execution_lock"]["sha256"],
        "implementation": implementation,
        "runtime": dict(authority["runtime"]),
        "expected_uid": os.getuid(),
        "logical_time_utc": plan["logical_time_utc"],
        "query_sha256": plan["query_sha256"],
        "claim_path": plan["lifecycle"]["claim"]["path"],
        "output_path": plan["output"]["path"],
    }
    validate_preflight_lock(
        lock,
        plan=plan,
        plan_sha256=plan_sha,
        lock_sha256=sha256_bytes(canonical_json(lock)),
        repo_root=repo_root,
        git_reader=git_reader,
    )
    return lock


def seal_execution_lock(
    repo_root: Path,
    destination: Path,
    implementation_commit: str,
    *,
    git_reader: Callable[[Path, str, str], bytes] | None = None,
) -> dict[str, Any]:
    lock = build_execution_lock(
        repo_root, implementation_commit, git_reader=git_reader
    )
    write_exclusive_durable(destination, canonical_json(lock))
    return lock


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv if argv is None else argv
    if arguments != [arguments[0]]:
        _stop("ARGV_FORBIDDEN")
    plan, _ = _load_canonical(REPOSITORY / PLAN_RELATIVE, "PREFLIGHT_PLAN")
    destination = REPOSITORY / plan["preflight_execution_lock"]["path"]
    lock = seal_execution_lock(REPOSITORY, destination, _git_commit(REPOSITORY))
    print(canonical_json(lock).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
