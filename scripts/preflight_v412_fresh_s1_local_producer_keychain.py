#!/usr/bin/env python3
"""One-shot, status-only Keychain preflight for the V4.12 S1 producer.

The public ``run_preflight`` entry point is deliberately dependency-injected:
tests pass a fake ``native_query`` and a temporary repository.  Importing this
module never loads Security.framework and never performs a Keychain query.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
from typing import Any, Callable, Mapping


REPOSITORY = Path(__file__).resolve().parents[1]
PLAN_RELATIVE = Path(
    "config/v4_12_fresh_s1_local_producer_preflight_plan.json"
)
ERR_SEC_ITEM_NOT_FOUND = -25300
GIT_EXECUTABLE = "/usr/bin/git"
EXPECTED_QUERY_SHA256 = (
    "0d5d2fe817391a4d91e51a57b3eaa447"
    "cad932c8c081b64a8b630bdc566fb96f"
)


class PreflightError(RuntimeError):
    """Fail-closed preflight error containing no Keychain metadata."""


class InjectedCrash(RuntimeError):
    """A synthetic crash boundary used by tests only."""


def _stop(reason: str) -> None:
    raise PreflightError(reason)


def canonical_json(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError):
        _stop("NON_CANONICAL_VALUE")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def parse_canonical_object(raw: bytes, label: str) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _stop(f"{label}_DUPLICATE_KEY")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError):
        _stop(f"{label}_INVALID_JSON")
    if type(value) is not dict or raw != canonical_json(value):
        _stop(f"{label}_NONCANONICAL")
    return value


def _read_regular(path: Path, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    try:
        fd = os.open(path, flags)
    except OSError:
        _stop(f"{label}_OPEN")
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            _stop(f"{label}_IDENTITY")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            _stop(f"{label}_DRIFT")
        raw = b"".join(chunks)
        if len(raw) != after.st_size:
            _stop(f"{label}_SIZE")
        return raw
    finally:
        os.close(fd)


def _load_canonical(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular(path, label)
    return parse_canonical_object(raw, label), raw


def _require_exact(value: Mapping[str, Any], fields: list[str], label: str) -> None:
    if type(value) is not dict or set(value) != set(fields):
        _stop(f"{label}_FIELDS")


def _require_type(value: Any, expected: str, label: str) -> None:
    valid = (
        (expected == "string" and type(value) is str)
        or (expected == "integer" and type(value) is int)
        or (expected == "object" and type(value) is dict)
    )
    if not valid:
        _stop(f"{label}_TYPE")


def _resolve_reference(
    reference: str,
    *,
    plan: Mapping[str, Any],
    plan_sha256: str,
    lock: Mapping[str, Any] | None,
    lock_sha256: str | None,
) -> Any:
    fixed = {
        "canonical_preflight_plan.sha256": plan_sha256,
        "preflight_execution_lock.file_sha256": lock_sha256,
        "runtime.os.getuid": os.getuid(),
    }
    if reference in fixed:
        return fixed[reference]
    if reference.startswith("canonical_preflight_plan."):
        value: Any = plan
        for part in reference.removeprefix("canonical_preflight_plan.").split("."):
            value = value[part]
        return value
    if reference.startswith("preflight_execution_lock."):
        if lock is None:
            _stop("REFERENCE_BEFORE_LOCK")
        value = lock
        for part in reference.removeprefix(
            "preflight_execution_lock."
        ).split("."):
            value = value[part]
        return value
    if reference == "git.commit_containing_all_implementation_blobs":
        if lock is None:
            _stop("REFERENCE_BEFORE_LOCK")
        return lock["implementation"]["git_commit"]
    if reference.startswith("git_blob_sha256_at_commit."):
        if lock is None:
            _stop("REFERENCE_BEFORE_LOCK")
        path_field = reference.removeprefix("git_blob_sha256_at_commit.")
        hash_field = {
            "preflight_path": "preflight_sha256",
            "tests_path": "tests_sha256",
            "sealer_path": "sealer_sha256",
            "sealer_tests_path": "sealer_tests_sha256",
        }.get(path_field)
        if hash_field is None:
            _stop("UNKNOWN_GIT_BLOB_REFERENCE")
        return lock["implementation"][hash_field]
    if reference.startswith("authority_execution_lock.runtime."):
        if lock is None:
            _stop("REFERENCE_BEFORE_LOCK")
        field = reference.removeprefix("authority_execution_lock.runtime.")
        return lock["runtime"][field]
    _stop("UNKNOWN_REFERENCE")


def _validate_rule_object(
    value: Mapping[str, Any],
    schema: Mapping[str, Any],
    *,
    label: str,
    plan: Mapping[str, Any],
    plan_sha256: str,
    lock: Mapping[str, Any] | None,
    lock_sha256: str | None,
) -> None:
    _require_exact(value, schema["exact_fields"], label)
    if set(schema["fields"]) != set(schema["exact_fields"]):
        _stop(f"{label}_SCHEMA_FIELDS")
    for field in schema["exact_fields"]:
        rule = schema["fields"][field]
        _require_type(value[field], rule["type"], f"{label}_{field}")
        if "const" in rule and value[field] != rule["const"]:
            _stop(f"{label}_{field}_VALUE")
        if "equals" in rule:
            expected = _resolve_reference(
                rule["equals"],
                plan=plan,
                plan_sha256=plan_sha256,
                lock=lock,
                lock_sha256=lock_sha256,
            )
            if value[field] != expected:
                _stop(f"{label}_{field}_VALUE")


def validate_plan(plan: Mapping[str, Any]) -> None:
    closed = plan.get("plan_schema", {}).get("closed_object_fields", {})
    _require_exact(plan, closed.get("$", []), "PLAN")
    direct_paths = {
        "absence_semantics": plan["absence_semantics"],
        "contract": plan["contract"],
        "execution_lock": plan["execution_lock"],
        "future_implementation": plan["future_implementation"],
        "implementation_test_requirements": plan[
            "implementation_test_requirements"
        ],
        "lifecycle": plan["lifecycle"],
        "lifecycle.claim": plan["lifecycle"]["claim"],
        "lifecycle.claim.schema": plan["lifecycle"]["claim"]["schema"],
        "lifecycle.existing_state_policy": plan["lifecycle"][
            "existing_state_policy"
        ],
        "output": plan["output"],
        "output.schema": plan["output"]["schema"],
        "plan_schema": plan["plan_schema"],
        "preflight_execution_lock": plan["preflight_execution_lock"],
        "preflight_execution_lock.schema": plan["preflight_execution_lock"][
            "schema"
        ],
        "query_canonicalization": plan["query_canonicalization"],
        "query_exact": plan["query_exact"],
        "success": plan["success"],
    }
    for path, value in direct_paths.items():
        _require_exact(value, closed[path], f"PLAN_{path}")
    for value in plan["future_implementation"].values():
        _require_exact(value, closed["future_implementation.*"], "PLAN_FUTURE")
    for value in plan["gates"].values():
        _require_exact(value, closed["gates.*"], "PLAN_GATE")
    guards = plan["runtime_absence_guards"]
    if (
        type(guards) is not list
        or len(guards) != 3
        or [guard.get("resolution") for guard in guards]
        != ["REPOSITORY", "ABSOLUTE", "ABSOLUTE"]
    ):
        _stop("PLAN_GUARD_SEQUENCE")
    guard_paths: list[str] = []
    for value in guards:
        _require_exact(value, closed["runtime_absence_guards.*"], "PLAN_GUARD")
        if (
            type(value["path"]) is not str
            or not value["path"]
            or value["required_state"] != "ABSENT"
            or value["resolution"] not in {"REPOSITORY", "ABSOLUTE"}
        ):
            _stop("PLAN_GUARD_VALUE")
        guard_path = Path(value["path"])
        if (value["resolution"] == "ABSOLUTE") != guard_path.is_absolute():
            _stop("PLAN_GUARD_RESOLUTION")
        guard_paths.append(value["path"])
    if len(set(guard_paths)) != len(guard_paths):
        _stop("PLAN_GUARD_DUPLICATE")
    expected_cases = plan["implementation_test_requirements"][
        "expected_native_call_count_by_case"
    ]
    _require_exact(
        expected_cases,
        closed[
            "implementation_test_requirements.expected_native_call_count_by_case"
        ],
        "PLAN_NATIVE_COUNTS",
    )
    if any(type(v) is not int or v != 0 for v in expected_cases.values()):
        _stop("PLAN_NATIVE_COUNTS_VALUE")
    if set(plan["query_exact"]) & set(plan["query_forbidden_keys"]):
        _stop("PLAN_QUERY_FORBIDDEN")
    if sha256_bytes(canonical_json(plan["query_exact"])) != plan["query_sha256"]:
        _stop("PLAN_QUERY_HASH")


def validate_preflight_lock(
    lock: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    plan_sha256: str,
    lock_sha256: str,
    repo_root: Path,
    git_reader: Callable[[Path, str, str], bytes] | None = None,
) -> None:
    schema = plan["preflight_execution_lock"]["schema"]
    _validate_rule_object(
        lock,
        {"exact_fields": schema["exact_fields"], "fields": schema["fields"]},
        label="PREFLIGHT_LOCK",
        plan=plan,
        plan_sha256=plan_sha256,
        lock=lock,
        lock_sha256=lock_sha256,
    )
    for name in ("implementation", "runtime"):
        _validate_rule_object(
            lock[name],
            {
                "exact_fields": schema[f"{name}_exact_fields"],
                "fields": schema[f"{name}_fields"],
            },
            label=f"PREFLIGHT_LOCK_{name.upper()}",
            plan=plan,
            plan_sha256=plan_sha256,
            lock=lock,
            lock_sha256=lock_sha256,
        )
    if lock["expected_uid"] != os.getuid():
        _stop("PREFLIGHT_LOCK_UID")
    commit = lock["implementation"]["git_commit"]
    if (
        type(commit) is not str
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        _stop("PREFLIGHT_LOCK_GIT_COMMIT")
    if git_reader is None:
        _verify_git_commit(repo_root, commit)
        git_reader = git_file_bytes
    for path_field, hash_field in (
        ("preflight_path", "preflight_sha256"),
        ("tests_path", "tests_sha256"),
        ("sealer_path", "sealer_sha256"),
        ("sealer_tests_path", "sealer_tests_sha256"),
    ):
        path_text = lock["implementation"][path_field]
        if path_text != {
            "preflight_path": plan["future_implementation"]["preflight"]["path"],
            "tests_path": plan["future_implementation"]["tests"]["path"],
            "sealer_path": plan["future_implementation"]["sealer"]["path"],
            "sealer_tests_path": plan["future_implementation"]["sealer_tests"][
                "path"
            ],
        }[path_field]:
            _stop(f"PREFLIGHT_LOCK_{path_field.upper()}")
        raw = _read_regular(repo_root / path_text, path_field)
        committed_raw = git_reader(repo_root, commit, path_text)
        if raw != committed_raw:
            _stop(f"PREFLIGHT_LOCK_{path_field.upper()}_COMMIT_DIVERGENCE")
        digest = lock["implementation"][hash_field]
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or sha256_bytes(raw) != digest
        ):
            _stop(f"PREFLIGHT_LOCK_{hash_field.upper()}")


def _git_environment() -> dict[str, str]:
    return {
        "PATH": "/usr/bin:/bin",
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_NO_REPLACE_OBJECTS": "1",
    }


def _verify_git_commit(repo_root: Path, commit: str) -> None:
    if (
        type(commit) is not str
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        _stop("PREFLIGHT_LOCK_GIT_COMMIT")
    for command, reason in (
        (
            [GIT_EXECUTABLE, "cat-file", "-e", f"{commit}^{{commit}}"],
            "PREFLIGHT_LOCK_GIT_COMMIT_MISSING",
        ),
        (
            [
                GIT_EXECUTABLE,
                "merge-base",
                "--is-ancestor",
                commit,
                "HEAD",
            ],
            "PREFLIGHT_LOCK_GIT_COMMIT_NOT_ANCESTOR",
        ),
    ):
        process = subprocess.run(
            command,
            cwd=repo_root,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=_git_environment(),
        )
        if process.returncode != 0:
            _stop(reason)


def git_file_bytes(repo_root: Path, commit: str, path: str) -> bytes:
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
        _stop("PREFLIGHT_LOCK_GIT_BLOB")
    return process.stdout


def _current_runtime() -> dict[str, Any]:
    executable = Path(sys.executable).resolve()
    return {
        "platform": platform.system(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "python_executable_path": str(executable),
        "python_executable_sha256": sha256_bytes(
            _read_regular(executable, "PYTHON_EXECUTABLE")
        ),
        "cryptography_version": importlib.metadata.version("cryptography"),
        "security_framework_path": MacStatusOnlyKeychain.SECURITY_PATH,
        "corefoundation_framework_path": MacStatusOnlyKeychain.CORE_PATH,
        "os_build": platform.release(),
    }


def load_and_validate_controls(
    repo_root: Path,
    *,
    git_reader: Callable[[Path, str, str], bytes] | None = None,
) -> tuple[dict[str, Any], str, dict[str, Any], str]:
    repo_root = repo_root.resolve()
    plan_path = repo_root / PLAN_RELATIVE
    plan, plan_raw = _load_canonical(plan_path, "PREFLIGHT_PLAN")
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
    authority_plan_path = repo_root / authority.get("plan_path", "")
    authority_plan, authority_plan_raw = _load_canonical(
        authority_plan_path, "AUTHORITY_PLAN"
    )
    if sha256_bytes(authority_plan_raw) != authority.get("plan_sha256"):
        _stop("AUTHORITY_PLAN_HASH")
    authority_schema = authority_plan.get("schemas", {}).get("execution_lock", {})
    _require_exact(
        authority, authority_schema.get("exact_fields", []), "AUTHORITY_LOCK"
    )
    if set(authority_schema.get("types", {})) != set(authority):
        _stop("AUTHORITY_LOCK_TYPES")
    for nested_name, field in (
        ("implementation_pin_object", "implementation"),
        ("runtime_pin_object", "runtime"),
        ("keychain_policy_object", "keychain_policy"),
    ):
        nested = authority_plan.get("schemas", {}).get(nested_name, {})
        _require_exact(
            authority[field], nested.get("exact_fields", []), f"AUTHORITY_{field}"
        )
    if authority["expected_uid"] != os.getuid():
        _stop("AUTHORITY_LOCK_UID")
    if authority["contract_path"] != authority_plan["authorities"]["contract"]["path"]:
        _stop("AUTHORITY_CONTRACT_PATH")
    authority_contract_raw = _read_regular(
        repo_root / authority["contract_path"], "AUTHORITY_CONTRACT"
    )
    if (
        sha256_bytes(authority_contract_raw) != authority["contract_sha256"]
        or authority["contract_sha256"]
        != authority_plan["authorities"]["contract"]["sha256"]
    ):
        _stop("AUTHORITY_CONTRACT_HASH")
    if authority["output_root"] != authority_plan["paths"]["root"]:
        _stop("AUTHORITY_OUTPUT_ROOT")
    if authority["authorization_path"] != authority_plan["paths"]["authorization"]:
        _stop("AUTHORITY_AUTHORIZATION_PATH")
    for path_field, hash_field in (
        ("provisioner_path", "provisioner_sha256"),
        ("tests_path", "tests_sha256"),
    ):
        implementation_raw = _read_regular(
            repo_root / authority["implementation"][path_field],
            f"AUTHORITY_{path_field}",
        )
        if sha256_bytes(implementation_raw) != authority["implementation"][hash_field]:
            _stop(f"AUTHORITY_{hash_field.upper()}")
    lock_path = repo_root / plan["preflight_execution_lock"]["path"]
    lock, lock_raw = _load_canonical(lock_path, "PREFLIGHT_LOCK")
    lock_sha = sha256_bytes(lock_raw)
    validate_preflight_lock(
        lock,
        plan=plan,
        plan_sha256=plan_sha,
        lock_sha256=lock_sha,
        repo_root=repo_root,
        git_reader=git_reader,
    )
    if lock["authority_execution_lock_sha256"] != sha256_bytes(authority_raw):
        _stop("PREFLIGHT_AUTHORITY_LOCK_HASH")
    if lock["runtime"] != authority["runtime"]:
        _stop("PREFLIGHT_RUNTIME")
    if lock["runtime"] != _current_runtime():
        _stop("PREFLIGHT_RUNTIME_CURRENT")
    return plan, plan_sha, lock, lock_sha


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )


def _open_directory_fd_anchored(path: Path) -> int:
    path = Path(path)
    if not path.is_absolute():
        _stop("DIRECTORY_NOT_ABSOLUTE")
    try:
        current = os.open(Path("/"), _directory_flags())
    except OSError:
        _stop("DIRECTORY_ROOT_OPEN")
    try:
        for component in path.parts[1:]:
            if component in ("", ".", ".."):
                _stop("DIRECTORY_COMPONENT")
            before = os.fstat(current)
            try:
                following = os.open(
                    component, _directory_flags(), dir_fd=current
                )
            except OSError:
                _stop("DIRECTORY_OPEN")
            after = os.fstat(current)
            opened = os.fstat(following)
            if (before.st_dev, before.st_ino) != (
                after.st_dev,
                after.st_ino,
            ):
                os.close(following)
                _stop("DIRECTORY_PARENT_REPLACED")
            if not stat.S_ISDIR(opened.st_mode):
                os.close(following)
                _stop("DIRECTORY_IDENTITY")
            os.close(current)
            current = following
        return current
    except Exception:
        os.close(current)
        raise


def require_absent_fd_anchored(path: Path, *, anchor: Path) -> None:
    """Prove absence using openat/fstatat while rejecting every ambiguity."""
    path = Path(path)
    anchor = Path(anchor)
    if path.is_absolute():
        components = path.parts[1:]
        anchor_path = Path("/")
    else:
        if ".." in path.parts:
            _stop("GUARD_PARENT_COMPONENT")
        components = path.parts
        anchor_path = anchor
    try:
        current = os.open(anchor_path, _directory_flags())
    except OSError:
        _stop("GUARD_ANCHOR_OPEN")
    try:
        for index, component in enumerate(components):
            if component in ("", ".", ".."):
                _stop("GUARD_COMPONENT")
            before = os.fstat(current)
            final = index == len(components) - 1
            if final:
                try:
                    os.stat(component, dir_fd=current, follow_symlinks=False)
                except FileNotFoundError:
                    after = os.fstat(current)
                    if (before.st_dev, before.st_ino) != (
                        after.st_dev,
                        after.st_ino,
                    ):
                        _stop("GUARD_PARENT_REPLACED")
                    return
                except OSError:
                    _stop("GUARD_FSTATAT")
                _stop("GUARD_ENTRY_EXISTS")
            try:
                following = os.open(component, _directory_flags(), dir_fd=current)
            except FileNotFoundError:
                after = os.fstat(current)
                if (before.st_dev, before.st_ino) != (
                    after.st_dev,
                    after.st_ino,
                ):
                    _stop("GUARD_PARENT_REPLACED")
                return
            except OSError:
                _stop("GUARD_PARENT_OPEN")
            after = os.fstat(current)
            if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
                os.close(following)
                _stop("GUARD_PARENT_REPLACED")
            os.close(current)
            current = following
        _stop("GUARD_EMPTY_PATH_EXISTS")
    finally:
        os.close(current)


def _sync_file(fd: int) -> None:
    os.fsync(fd)
    full = getattr(fcntl, "F_FULLFSYNC", None)
    if full is None:
        _stop("F_FULLFSYNC_UNAVAILABLE")
    try:
        fcntl.fcntl(fd, full)
    except OSError:
        _stop("F_FULLFSYNC_FAILED")


def write_exclusive_durable(path: Path, raw: bytes) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    parent_fd = _open_directory_fd_anchored(path.parent)
    try:
        fd = os.open(path.name, flags, 0o600, dir_fd=parent_fd)
        try:
            os.fchmod(fd, 0o600)
            view = memoryview(raw)
            while view:
                count = os.write(fd, view)
                if count <= 0:
                    _stop("OUTPUT_SHORT_WRITE")
                view = view[count:]
            _sync_file(fd)
            info = os.fstat(fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_uid != os.getuid()
                or info.st_nlink != 1
            ):
                _stop("OUTPUT_IDENTITY")
            os.lseek(fd, 0, os.SEEK_SET)
        finally:
            os.close(fd)
        os.fsync(parent_fd)
    except FileExistsError:
        _stop("OUTPUT_ALREADY_EXISTS")
    except OSError:
        _stop("OUTPUT_CREATE")
    finally:
        os.close(parent_fd)
    if _read_regular(path, "OUTPUT_VERIFY") != raw:
        _stop("OUTPUT_VERIFY")


def _expected_claim(
    plan: Mapping[str, Any], plan_sha: str, lock_sha: str
) -> dict[str, Any]:
    return {
        "schema_version": plan["lifecycle"]["claim"]["schema"]["fields"][
            "schema_version"
        ]["const"],
        "state": "KEYCHAIN_QUERY_RESERVED",
        "preflight_plan_sha256": plan_sha,
        "preflight_execution_lock_sha256": lock_sha,
        "logical_time_utc": plan["logical_time_utc"],
    }


def _expected_result(
    plan: Mapping[str, Any],
    plan_sha: str,
    lock: Mapping[str, Any],
    lock_sha: str,
) -> dict[str, Any]:
    implementation = lock["implementation"]
    return {
        "schema_version": plan["output"]["schema"]["fields"]["schema_version"][
            "const"
        ],
        "verdict": plan["success"]["verdict"],
        "authority_execution_lock_sha256": plan["execution_lock"]["sha256"],
        "query_sha256": plan["query_sha256"],
        "osstatus": plan["success"]["osstatus"],
        "logical_time_utc": plan["logical_time_utc"],
        "preflight_plan_sha256": plan_sha,
        "preflight_execution_lock_sha256": lock_sha,
        "implementation_commit": implementation["git_commit"],
        "implementation_sha256": implementation["preflight_sha256"],
        "implementation_tests_sha256": implementation["tests_sha256"],
    }


def _lexists(path: Path) -> bool:
    try:
        os.lstat(path)
        return True
    except FileNotFoundError:
        return False
    except OSError:
        _stop("STATE_LSTAT")


def run_preflight(
    repo_root: Path,
    native_query: Callable[[Mapping[str, Any]], int],
    *,
    checkpoint: Callable[[str], None] | None = None,
    git_reader: Callable[[Path, str, str], bytes] | None = None,
) -> dict[str, Any]:
    """Execute the one-shot state machine with an injected status-only query."""
    plan, plan_sha, lock, lock_sha = load_and_validate_controls(
        repo_root, git_reader=git_reader
    )
    for guard in plan["runtime_absence_guards"]:
        guard_path = Path(guard["path"])
        require_absent_fd_anchored(guard_path, anchor=repo_root)

    claim_path = repo_root / plan["lifecycle"]["claim"]["path"]
    output_path = repo_root / plan["output"]["path"]
    expected_claim = _expected_claim(plan, plan_sha, lock_sha)
    expected_result = _expected_result(plan, plan_sha, lock, lock_sha)
    claim_exists = _lexists(claim_path)
    result_exists = _lexists(output_path)
    if claim_exists and result_exists:
        if (
            parse_canonical_object(_read_regular(claim_path, "CLAIM"), "CLAIM")
            != expected_claim
        ):
            _stop("CLAIM_DIVERGENCE")
        stored = parse_canonical_object(
            _read_regular(output_path, "RESULT"), "RESULT"
        )
        if stored != expected_result:
            _stop("RESULT_DIVERGENCE")
        return stored
    if claim_exists or result_exists:
        _stop("INDETERMINATE_EXISTING_STATE")

    write_exclusive_durable(claim_path, canonical_json(expected_claim))
    if checkpoint is not None:
        checkpoint("AFTER_CLAIM_BEFORE_NATIVE_CALL")
    status = native_query(dict(plan["query_exact"]))
    if type(status) is not int:
        _stop("NATIVE_STATUS_TYPE")
    if checkpoint is not None:
        checkpoint("AFTER_NATIVE_CALL_BEFORE_RESULT")
    if status != ERR_SEC_ITEM_NOT_FOUND:
        _stop("KEYCHAIN_LOCATOR_NOT_PROVEN_ABSENT")
    write_exclusive_durable(output_path, canonical_json(expected_result))
    return expected_result


def build_query_contract(plan: Mapping[str, Any]) -> dict[str, Any]:
    query = dict(plan["query_exact"])
    if set(query) != {
        "kSecClass",
        "kSecAttrService",
        "kSecAttrAccount",
        "kSecAttrSynchronizable",
        "kSecUseDataProtectionKeychain",
        "kSecUseAuthenticationUI",
        "kSecMatchLimit",
    }:
        _stop("QUERY_FIELDS")
    if sha256_bytes(canonical_json(query)) != plan["query_sha256"]:
        _stop("QUERY_HASH")
    if plan["query_sha256"] != EXPECTED_QUERY_SHA256:
        _stop("QUERY_CODE_PIN")
    return query


def _framework_constant(library: Any, name: str) -> ctypes.c_void_p:
    try:
        pointer = ctypes.c_void_p.in_dll(library, name)
    except (ValueError, OSError):
        _stop("CF_CONSTANT")
    if not pointer.value:
        _stop("CF_CONSTANT")
    return pointer


class MacStatusOnlyKeychain:
    """Dedicated seven-key CoreFoundation bridge with a NULL result pointer."""

    SECURITY_PATH = "/System/Library/Frameworks/Security.framework/Security"
    CORE_PATH = (
        "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
    )

    def __init__(
        self,
        security: Any | None = None,
        corefoundation: Any | None = None,
    ) -> None:
        if security is None or corefoundation is None:
            if sys.platform != "darwin":
                _stop("KEYCHAIN_PLATFORM")
            try:
                security = ctypes.CDLL(self.SECURITY_PATH)
                corefoundation = ctypes.CDLL(self.CORE_PATH)
            except OSError:
                _stop("KEYCHAIN_FRAMEWORK")
        self.security = security
        self.core = corefoundation
        self._configure_abi()

    def _configure_abi(self) -> None:
        self.core.CFDictionaryCreateMutable.argtypes = [
            ctypes.c_void_p,
            ctypes.c_long,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.core.CFDictionaryCreateMutable.restype = ctypes.c_void_p
        self.core.CFDictionarySetValue.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.core.CFDictionarySetValue.restype = None
        self.core.CFStringCreateWithCString.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_uint32,
        ]
        self.core.CFStringCreateWithCString.restype = ctypes.c_void_p
        self.core.CFRelease.argtypes = [ctypes.c_void_p]
        self.core.CFRelease.restype = None
        self.security.SecItemCopyMatching.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.security.SecItemCopyMatching.restype = ctypes.c_int32

    def _value_pointer(
        self, value: Any, owned: list[ctypes.c_void_p]
    ) -> ctypes.c_void_p:
        if type(value) is bool:
            return _framework_constant(
                self.core, "kCFBooleanTrue" if value else "kCFBooleanFalse"
            )
        symbolic = {
            "kSecClassGenericPassword",
            "kSecMatchLimitOne",
            "kSecUseAuthenticationUIFail",
        }
        if value in symbolic:
            return _framework_constant(self.security, value)
        pointer = self.core.CFStringCreateWithCString(
            None, value.encode("utf-8"), 0x08000100
        )
        if not pointer:
            _stop("CF_STRING")
        result = ctypes.c_void_p(pointer)
        owned.append(result)
        return result

    def query_status(self, query: Mapping[str, Any]) -> int:
        if set(query) != {
            "kSecClass",
            "kSecAttrService",
            "kSecAttrAccount",
            "kSecAttrSynchronizable",
            "kSecUseDataProtectionKeychain",
            "kSecUseAuthenticationUI",
            "kSecMatchLimit",
        }:
            _stop("QUERY_FIELDS")
        if sha256_bytes(canonical_json(dict(query))) != EXPECTED_QUERY_SHA256:
            _stop("QUERY_VALUE_PIN")
        dictionary_value = self.core.CFDictionaryCreateMutable(None, 0, None, None)
        if not dictionary_value:
            _stop("CF_DICTIONARY")
        dictionary = ctypes.c_void_p(dictionary_value)
        owned: list[ctypes.c_void_p] = []
        try:
            for key, value in query.items():
                key_pointer = _framework_constant(self.security, key)
                value_pointer = self._value_pointer(value, owned)
                self.core.CFDictionarySetValue(
                    dictionary.value, key_pointer.value, value_pointer.value
                )
            # Status-only contract: never request or receive a result object.
            return int(
                self.security.SecItemCopyMatching(dictionary.value, None)
            )
        finally:
            for pointer in owned:
                self.core.CFRelease(pointer.value)
            self.core.CFRelease(dictionary.value)


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv if argv is None else argv
    if len(arguments) != 1:
        _stop("ARGV_FORBIDDEN")
    plan, _, _, _ = load_and_validate_controls(REPOSITORY)
    build_query_contract(plan)
    backend = MacStatusOnlyKeychain()
    result = run_preflight(REPOSITORY, backend.query_status)
    print(canonical_json(result).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
