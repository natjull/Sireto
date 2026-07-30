#!/usr/bin/env python3
"""Fail-closed V4.12 oracle evaluator.

The parent owns policy validation, the durable attempt receipt and publication.
The sandboxed worker receives inherited, already-verified descriptors and
never discovers an oracle path by walking the filesystem.
"""

from __future__ import annotations

import argparse
import array
import contextlib
import ctypes
import datetime as dt
import errno
import fcntl
import hashlib
import json
import math
import os
import platform
import re
import resource
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import duckdb
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy
import sklearn


STOP = "STOP_V412_UNIT_RETRIEVAL_EVALUATION"
PLAN_PATH = Path("config/v4_12_unit_retrieval_evaluator_plan.json")
LOCK_PATH = Path("config/v4_12_unit_retrieval_evaluator_execution_lock.json")
PROFILE_PATH = Path("config/v4_12_unit_retrieval_evaluator.sb")
PLAN_SCHEMA = "sireto-v4.12-unit-retrieval-evaluator-plan-1"
LOCK_SCHEMA = "sireto-v4.12-unit-retrieval-evaluator-execution-lock-1"
LOCK_PURPOSE = "V4.12_UNIT_RETRIEVAL_ORACLE_EVALUATION"
LOCK_VERDICT = "GO_CODE_V412_UNIT_RETRIEVAL_EVALUATOR"
WORKER_SPEC_SCHEMA = "sireto-v4.12-unit-retrieval-evaluator-worker-spec-1"
WORKER_STDOUT_SCHEMA = "sireto-v4.12-unit-retrieval-evaluator-worker-stdout-1"
WORKER_CONTROL_SCHEMA = "sireto-v4.12-unit-retrieval-evaluator-control-1"
MAX_JSON_BYTES = 32 * 1024 * 1024
ADMIN_RSS_LIMIT = 8 * 1024 * 1024 * 1024

PLAN_KEYS = {
    "schema_version",
    "purpose",
    "prerequisite",
    "oracle",
    "input_paths",
    "input_hashes",
    "evaluation_spec",
    "identity_projections",
    "outputs",
    "artifact_contract",
    "attempt_protocol",
    "publication",
    "security",
    "verdicts",
    "external_lock",
    "runtime",
    "max_rss_bytes",
    "future_sources",
}
LOCK_KEYS = {
    "schema_version",
    "purpose",
    "audit_verdict",
    "git_commit",
    "source_hashes",
    "input_paths",
    "input_hashes",
    "evaluation_spec_sha256",
    "runtime",
    "outputs",
    "sandbox",
    "max_rss_bytes",
}
WORKER_SPEC_KEYS = {
    "schema_version",
    "evaluator_build_id",
    "attempt_id",
    "worker_build_id",
    "oracle_build_id",
    "parity_build_id",
    "input_fds",
    "input_paths",
    "input_hashes",
    "git_commit",
    "source_hashes",
    "plan_sha256",
    "lock_sha256",
    "evaluation_spec",
    "artifact_contract",
    "runtime",
    "evaluation_stage",
    "audit_stage",
    "max_rss_bytes",
}
DATA_PROJECTIONS = {
    "worker_candidates_top100": "query_id,candidate_rank,candidate_siret",
    "worker_query_status": "query_id,candidate_count",
    "oracle_dev": (
        "query_id,dev_partition,label_kind,ground_truth_siret,"
        "ground_truth_siren"
    ),
}
ORACLE_ROLE_ORDER = (
    "oracle_manifest",
    "oracle_integrity",
    "oracle_audit_manifest",
    "oracle_dev",
)


class EvaluationStopped(RuntimeError):
    """A fail-closed contract violation."""


def _stop(message: str) -> None:
    raise EvaluationStopped(f"{STOP}: {message}")


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _stop(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def canonical_json(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        _stop(f"non-canonical JSON value: {exc}")


def parse_json(payload: bytes, label: str) -> dict[str, Any]:
    if len(payload) > MAX_JSON_BYTES:
        _stop(f"{label} exceeds JSON limit")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_no_duplicates,
            parse_constant=lambda token: _stop(
                f"{label} contains invalid constant {token}"
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _stop(f"invalid {label}: {exc}")
    if type(value) is not dict:
        _stop(f"{label} must be an object")
    return value


def _runtime() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pa.__version__,
        "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "joblib": joblib.__version__,
        "duckdb": duckdb.__version__,
        "machine": platform.machine(),
        "platform": platform.platform(),
    }


def _check_rss(limit: int) -> None:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    current = int(raw if sys.platform == "darwin" else raw * 1024)
    if current > limit:
        _stop(f"RSS limit exceeded: {current} > {limit}")


def _apply_resource_limits(limit: int) -> dict[str, int]:
    """Apply kernel-enforced address/data ceilings in the sandboxed child."""
    if type(limit) is not int or limit <= 0:
        _stop("invalid process resource limit")
    applied: dict[str, int] = {}
    for name in ("RLIMIT_AS", "RLIMIT_DATA", "RLIMIT_RSS"):
        resource_id = getattr(resource, name, None)
        if resource_id is None:
            continue
        soft, hard = resource.getrlimit(resource_id)
        ceiling = limit if hard == resource.RLIM_INFINITY else min(limit, hard)
        try:
            resource.setrlimit(resource_id, (ceiling, hard))
        except (OSError, ValueError) as exc:
            # Darwin maps hundreds of GiB of shared system regions into a
            # normal scientific-Python process; lowering RLIMIT_AS after the
            # interpreter has loaded is rejected by the kernel. RLIMIT_DATA
            # remains a real enforced allocation ceiling and is mandatory.
            if sys.platform == "darwin" and name in {
                "RLIMIT_AS",
                "RLIMIT_DATA",
                "RLIMIT_RSS",
            }:
                continue
            _stop(f"cannot enforce {name}: {exc}")
        actual, _ = resource.getrlimit(resource_id)
        if actual == resource.RLIM_INFINITY or actual > limit:
            _stop(f"{name} was not enforced")
        applied[name] = int(actual)
    if sys.platform == "darwin":
        # On current Darwin, all three limits reject an 8 GiB ceiling because
        # the interpreter starts with ~436 GiB of shared virtual mappings.
        # The parent therefore enforces the same ceiling with an external RSS
        # kill monitor; retaining this marker makes the fallback auditable.
        if not applied:
            applied["DARWIN_RLIMIT_UNAVAILABLE"] = limit
    elif (
        not applied
        or (
        hasattr(resource, "RLIMIT_AS")
        and "RLIMIT_AS" not in applied
        and hasattr(resource, "RLIMIT_DATA")
        and "RLIMIT_DATA" not in applied
        )
    ):
        _stop("no address/data-space kernel limit was enforced")
    return applied


def _openat_anchored(path: Path, *, directory: bool = False) -> tuple[Path, int]:
    if not path.is_absolute():
        _stop(f"path is not absolute: {path}")
    parts = path.parts
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for index, component in enumerate(parts[1:]):
            last = index == len(parts[1:]) - 1
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            if not last or directory:
                flags |= os.O_DIRECTORY
            flags |= getattr(os, "O_NOFOLLOW", 0)
            next_fd = os.open(component, flags, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        info = os.fstat(fd)
        expected = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(
            info.st_mode
        )
        if not expected:
            _stop(f"unexpected file type: {path}")
        return path, fd
    except Exception:
        os.close(fd)
        raise


def _snapshot_fd(fd: int, limit: int) -> dict[str, Any]:
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode):
        _stop("descriptor is not a regular file")
    digest = hashlib.sha256()
    total = 0
    os.lseek(fd, 0, os.SEEK_SET)
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            _stop("input exceeds byte limit")
        digest.update(chunk)
        _check_rss(limit)
    os.lseek(fd, 0, os.SEEK_SET)
    after = os.fstat(fd)
    if (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        _stop("input mutated while hashing")
    return {
        "size_bytes": total,
        "sha256": digest.hexdigest(),
        "device": info.st_dev,
        "inode": info.st_ino,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
    }


def _same_snapshot(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return dict(left) == dict(right)


def _read_fd(fd: int, limit: int) -> bytes:
    snap = _snapshot_fd(fd, limit)
    os.lseek(fd, 0, os.SEEK_SET)
    payload = bytearray()
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            break
        payload.extend(chunk)
    os.lseek(fd, 0, os.SEEK_SET)
    if len(payload) != snap["size_bytes"]:
        _stop("descriptor read size changed")
    return bytes(payload)


def _read_path(path: Path, limit: int) -> bytes:
    _absolute, fd = _openat_anchored(path)
    try:
        return _read_fd(fd, limit)
    finally:
        os.close(fd)


def _read_small_path(path: Path, maximum: int = MAX_JSON_BYTES) -> bytes:
    payload = _read_path(path, ADMIN_RSS_LIMIT)
    if len(payload) > maximum:
        _stop(f"administrative file exceeds limit: {path}")
    return payload


def _open_locked(
    path: Path, digest: str, limit: int
) -> tuple[int, dict[str, Any]]:
    _absolute, fd = _openat_anchored(path)
    try:
        snapshot = _snapshot_fd(fd, limit)
        if snapshot["sha256"] != digest:
            _stop(f"locked input hash mismatch: {path}")
        return fd, snapshot
    except Exception:
        os.close(fd)
        raise


def _json_from_fd(fd: int, label: str, limit: int) -> dict[str, Any]:
    payload = _read_fd(fd, limit)
    if len(payload) > MAX_JSON_BYTES:
        _stop(f"{label} exceeds JSON limit")
    return parse_json(payload, label)


def _table_from_fd(
    fd: int, columns: Sequence[str], label: str, limit: int
) -> pa.Table:
    before = _snapshot_fd(fd, limit)
    duplicate = os.dup(fd)
    try:
        os.lseek(duplicate, 0, os.SEEK_SET)
        with os.fdopen(duplicate, "rb", closefd=True) as stream:
            duplicate = -1
            table = pq.read_table(stream, columns=list(columns))
    except Exception as exc:
        _stop(f"cannot read {label}: {exc}")
    finally:
        if duplicate >= 0:
            os.close(duplicate)
    after = _snapshot_fd(fd, limit)
    if not _same_snapshot(before, after):
        _stop(f"{label} mutated while consumed")
    return table


def _fsync_dir(path: Path) -> None:
    # Opening the final absolute directory directly lets Seatbelt validate the
    # exact staging-root rule without granting read-data access to every
    # ancestor directory.  `O_NOFOLLOW` still rejects a symlink at the target.
    fd = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _ensure_dir(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    _fsync_dir(path)


def _write_exclusive(path: Path, payload: bytes, mode: int = 0o600) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    fd = os.open(path, flags, mode)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    _fsync_dir(path.parent)


def _atomic_json_cache(
    path: Path,
    value: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> None:
    _ensure_dir(path.parent)
    attempt_root = path.parent.parent
    temporary = attempt_root / (
        f".state-cache-{value['attempt_id']}-{os.urandom(16).hex()}.tmp"
    )
    _write_exclusive(temporary, canonical_json(value))
    current = path.parent / path.name
    if current.is_symlink():
        _stop("state cache is a symlink")
    if (
        protocol["state_cache_temporary_root"] != "<attempt_root>"
        or protocol["state_cache_temporary_outside_slot"] is not True
    ):
        _stop("state cache temporary policy mismatch")
    os.replace(temporary, current)
    _fsync_dir(path.parent)
    _fsync_dir(attempt_root)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_exclusive(path, canonical_json(value))


def _write_parquet(path: Path, table: pa.Table) -> None:
    if table.schema.metadata is not None:
        table = table.replace_schema_metadata(None)
    pq.write_table(table, path, compression="zstd")
    fd = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    _fsync_dir(path.parent)


def _git(repo: Path, *args: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["/usr/bin/git", *args],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=not binary,
    )
    return result.stdout


def _require_keys(value: Mapping[str, Any], keys: Sequence[str], label: str) -> None:
    if set(value) != set(keys):
        _stop(f"{label} keyset mismatch")


def _evaluation_spec_sha(plan: Mapping[str, Any]) -> str:
    projection = plan["identity_projections"]["evaluation_spec_keys"]
    spec = plan["evaluation_spec"]
    if list(spec) != projection or set(spec) != set(projection):
        _stop("evaluation_spec projection mismatch")
    return hashlib.sha256(canonical_json(spec)).hexdigest()


def validate_plan(plan: Mapping[str, Any]) -> None:
    if set(plan) != PLAN_KEYS or plan.get("schema_version") != PLAN_SCHEMA:
        _stop("plan schema/keyset mismatch")
    if plan.get("purpose") != LOCK_PURPOSE:
        _stop("plan purpose mismatch")
    if type(plan.get("max_rss_bytes")) is not int or plan["max_rss_bytes"] <= 0:
        _stop("invalid RSS limit")
    if set(plan["input_paths"]) != set(plan["input_hashes"]):
        _stop("plan input role mismatch")
    if any(
        re.fullmatch(r"[0-9a-f]{64}", digest) is None
        for digest in plan["input_hashes"].values()
    ):
        _stop("invalid input digest")
    spec = plan["evaluation_spec"]
    ids = plan["identity_projections"]
    _require_keys(spec, ids["evaluation_spec_keys"], "evaluation_spec")
    if list(spec) != ids["evaluation_spec_keys"]:
        _stop("evaluation_spec key order mismatch")
    if (
        ids["build_identity_schema_version"]
        != "sireto-v4.12-unit-retrieval-evaluator-build-identity-1"
    ):
        _stop("build identity schema mismatch")
    if spec["gate"]["gate_statistic"] != "OBSERVED_RATE_FROM_RAW_COUNTS":
        _stop("gate statistic mismatch")
    artifact = plan["artifact_contract"]
    if artifact["population_order"] != spec["population_order"]:
        _stop("population order mismatch")
    if artifact["reference_order"] != spec["reference_order"]:
        _stop("reference order mismatch")
    protocol = plan["attempt_protocol"]
    pre = protocol["pre_oracle_revalidation_roles"]
    oracle = protocol["oracle_roles_opened_only_after_commit"]
    if (
        len(pre) != 12
        or len(set(pre)) != 12
        or len(oracle) != 4
        or len(set(oracle)) != 4
        or set(pre).intersection(oracle)
        or set(pre).union(oracle) != set(plan["input_paths"])
        or any(role.startswith("oracle_") for role in pre)
        or any(not role.startswith("oracle_") for role in oracle)
        or tuple(oracle) != ORACLE_ROLE_ORDER
    ):
        _stop("oracle boundary role mismatch")
    if not protocol["event_log_authoritative"]:
        _stop("event log must be authoritative")
    _evaluation_spec_sha(plan)


def validate_lock(
    plan: Mapping[str, Any],
    lock: Mapping[str, Any],
    repo: Path,
    plan_sha256: str,
) -> None:
    if set(lock) != LOCK_KEYS or lock.get("schema_version") != LOCK_SCHEMA:
        _stop("lock schema/keyset mismatch")
    if (
        lock.get("purpose") != LOCK_PURPOSE
        or lock.get("audit_verdict") != LOCK_VERDICT
        or lock.get("input_paths") != plan["input_paths"]
        or lock.get("input_hashes") != plan["input_hashes"]
        or lock.get("runtime") != plan["runtime"]
        or lock.get("outputs") != plan["outputs"]
        or lock.get("max_rss_bytes") != plan["max_rss_bytes"]
        or lock.get("evaluation_spec_sha256") != _evaluation_spec_sha(plan)
    ):
        _stop("lock values mismatch")
    commit = lock.get("git_commit")
    if type(commit) is not str or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        _stop("invalid locked commit")
    if str(_git(repo, "cat-file", "-t", commit)).strip() != "commit":
        _stop("locked commit missing")
    sources = lock.get("source_hashes")
    if type(sources) is not dict or set(sources) != set(plan["future_sources"]):
        _stop("lock source closure mismatch")
    limit = plan["max_rss_bytes"]
    for relative in sorted(sources):
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            _stop("unsafe source path")
        payload = _read_path(repo / relative, limit)
        digest = hashlib.sha256(payload).hexdigest()
        if digest != sources[relative]:
            _stop(f"source worktree differs from lock: {relative}")
        blob = _git(repo, "show", f"{commit}:{relative}", binary=True)
        assert isinstance(blob, bytes)
        if hashlib.sha256(blob).hexdigest() != digest:
            _stop(f"source blob differs from lock: {relative}")
    if sources[str(PLAN_PATH)] != plan_sha256:
        _stop("plan is not sealed by lock")
    sandbox = lock.get("sandbox")
    expected_sandbox = {
        "executable",
        "executable_sha256",
        "python_framework_app",
        "python_framework_app_sha256",
        "python_framework_library",
        "python_framework_library_sha256",
        "git_executable",
        "git_executable_sha256",
        "system_read_roots",
        "device_read_paths",
        "network_allowed",
        "fork_allowed",
        "write_scope",
    }
    if type(sandbox) is not dict or set(sandbox) != expected_sandbox:
        _stop("sandbox lock keyset mismatch")
    if (
        sandbox["network_allowed"] is not False
        or sandbox["fork_allowed"] is not False
        or sandbox["write_scope"] != "PRIVATE_EVALUATOR_STAGING_ONLY"
    ):
        _stop("sandbox policy mismatch")
    role_bindings = {
        "executable": "sandbox_executable",
        "python_framework_app": "python_framework_app",
        "python_framework_library": "python_framework_library",
        "git_executable": "git_executable",
    }
    for path_key, role in role_bindings.items():
        hash_key = f"{path_key}_sha256"
        if (
            sandbox[path_key] != lock["input_paths"][role]
            or sandbox[hash_key] != lock["input_hashes"][role]
        ):
            _stop(f"sandbox role binding mismatch: {role}")
        payload = _read_path(Path(sandbox[path_key]), limit)
        if hashlib.sha256(payload).hexdigest() != sandbox[hash_key]:
            _stop(f"sandbox executable bytes mismatch: {role}")
    system_roots = sandbox["system_read_roots"]
    device_paths = sandbox["device_read_paths"]
    if system_roots != ["/System", "/usr", "/opt/homebrew"] or device_paths != [
        "/dev/null",
        "/dev/urandom",
        "/dev/fd",
    ]:
        _stop("invalid sandbox system/device roots")


def _identity(
    plan: Mapping[str, Any],
    lock: Mapping[str, Any],
    plan_sha: str,
    lock_sha: str,
) -> tuple[str, dict[str, Any]]:
    payload = {
        "schema_version": plan["identity_projections"][
            "build_identity_schema_version"
        ],
        "plan_sha256": plan_sha,
        "lock_sha256": lock_sha,
        "source_hashes": lock["source_hashes"],
        "input_hashes": lock["input_hashes"],
        "worker_build_id": plan["prerequisite"]["worker_build_id"],
        "oracle_build_id": plan["oracle"]["build_id"],
        "parity_build_id": plan["prerequisite"]["parity_build_id"],
        "evaluation_spec": plan["evaluation_spec"],
        "runtime": plan["runtime"],
    }
    _require_keys(
        payload,
        plan["identity_projections"]["build_identity_keys"],
        "build identity",
    )
    return hashlib.sha256(canonical_json(payload)).hexdigest(), payload


def _attempt_identities(
    plan: Mapping[str, Any],
    lock: Mapping[str, Any],
    plan_sha: str,
    lock_sha: str,
) -> tuple[str, str, dict[str, Any]]:
    protocol = plan["attempt_protocol"]
    slot_payload = {
        "schema_version": protocol["schema_versions"][
            "measurement_slot_identity"
        ],
        "purpose": plan["purpose"],
        "worker_build_id": plan["prerequisite"]["worker_build_id"],
        "oracle_build_id": plan["oracle"]["build_id"],
        "parity_build_id": plan["prerequisite"]["parity_build_id"],
    }
    _require_keys(
        slot_payload,
        protocol["measurement_slot_identity_keyset"],
        "measurement slot",
    )
    slot_id = hashlib.sha256(canonical_json(slot_payload)).hexdigest()
    attempt_payload = {
        "schema_version": protocol["schema_versions"]["attempt_identity"],
        "plan_sha256": plan_sha,
        "lock_sha256": lock_sha,
        "input_hashes": lock["input_hashes"],
        "evaluation_spec_sha256": _evaluation_spec_sha(plan),
        "worker_build_id": plan["prerequisite"]["worker_build_id"],
        "oracle_build_id": plan["oracle"]["build_id"],
        "parity_build_id": plan["prerequisite"]["parity_build_id"],
    }
    _require_keys(
        attempt_payload, protocol["attempt_identity_keyset"], "attempt identity"
    )
    attempt_id = hashlib.sha256(canonical_json(attempt_payload)).hexdigest()
    return slot_id, attempt_id, attempt_payload


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _event_to_state(event: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict:
    state = {
        "schema_version": protocol["schema_versions"]["state"],
        "measurement_slot_id": event["measurement_slot_id"],
        "attempt_id": event["attempt_id"],
        "sequence": event["sequence"],
        "state": event["state"],
        "phase": event["phase"],
        "oracle_open_committed": event["oracle_open_committed"],
        "evaluator_build_id": event["evaluator_build_id"],
        "computed_attestation_sha256": event[
            "computed_attestation_sha256"
        ],
        "reason_code": event["reason_code"],
        "updated_at_utc": event["timestamp_utc"],
    }
    _require_keys(state, protocol["state_keyset"], "state cache")
    return state


def _event_projection_matches_state(
    state: Mapping[str, Any],
    event: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> bool:
    return dict(state) == _event_to_state(event, protocol)


def _validate_transition(
    previous: Mapping[str, Any] | None,
    current: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> None:
    if current["state"] not in protocol["states"]:
        _stop("unknown attempt state")
    if current["phase"] not in protocol["phases"]:
        _stop("unknown attempt phase")
    if previous is None:
        if (
            current["sequence"] != 0
            or current["state"] != "STARTED"
            or current["phase"] != "RECEIPT_DURABLE"
            or current["oracle_open_committed"] is not False
            or current["computed_attestation_sha256"] is not None
        ):
            _stop("invalid initial attempt event")
        return
    if current["sequence"] != previous["sequence"] + 1:
        _stop("event sequence is not contiguous")
    identity = ("measurement_slot_id", "attempt_id")
    if any(current[key] != previous[key] for key in identity):
        _stop("event identity changed")
    if previous["oracle_open_committed"] and not current["oracle_open_committed"]:
        _stop("oracle commit regressed")
    previous_attestation = previous["computed_attestation_sha256"]
    current_attestation = current["computed_attestation_sha256"]
    if previous_attestation is not None and (
        current_attestation != previous_attestation
    ):
        _stop("computed attestation hash changed or regressed")
    if previous["state"] in {"FINAL", "STOPPED"}:
        _stop("terminal attempt has an extra event")
    previous_pair = (previous["state"], previous["phase"])
    current_pair = (current["state"], current["phase"])
    forward = {
        ("STARTED", "RECEIPT_DURABLE"): (
            "STARTED",
            "ORACLE_OPEN_COMMITTED",
        ),
        ("STARTED", "ORACLE_OPEN_COMMITTED"): (
            "RECOVERABLE",
            "COMPUTED_STAGING_VALID",
        ),
        ("RECOVERABLE", "COMPUTED_STAGING_VALID"): (
            "RECOVERABLE",
            "PENDING_BOTH_VALID",
        ),
        ("RECOVERABLE", "PENDING_BOTH_VALID"): (
            "RECOVERABLE",
            "AUDIT_FINAL",
        ),
        ("RECOVERABLE", "AUDIT_FINAL"): (
            "RECOVERABLE",
            "EVALUATION_FINAL",
        ),
        ("RECOVERABLE", "EVALUATION_FINAL"): ("FINAL", "TERMINAL"),
    }
    if current["state"] == "STOPPED":
        if current["phase"] != "TERMINAL":
            _stop("STOPPED must be terminal")
        if (
            current["oracle_open_committed"]
            is not previous["oracle_open_committed"]
            or current["evaluator_build_id"] != previous["evaluator_build_id"]
        ):
            _stop("STOPPED event changed oracle/build state")
    elif forward.get(previous_pair) != current_pair:
        _stop(f"invalid attempt transition: {previous_pair} -> {current_pair}")
    if current_pair == ("STARTED", "ORACLE_OPEN_COMMITTED"):
        if (
            previous_pair != ("STARTED", "RECEIPT_DURABLE")
            or previous["oracle_open_committed"] is not False
            or current["oracle_open_committed"] is not True
            or current_attestation is not None
        ):
            _stop("invalid oracle commit transition")
    elif current["state"] != "STOPPED" and current["oracle_open_committed"] is not True:
        _stop("post-oracle transition lacks durable oracle commit")
    if previous_pair != ("STARTED", "RECEIPT_DURABLE") and (
        current["evaluator_build_id"] != previous["evaluator_build_id"]
    ):
        _stop("evaluator build changed")
    if current_pair == ("RECOVERABLE", "COMPUTED_STAGING_VALID"):
        if (
            previous_attestation is not None
            or type(current_attestation) is not str
            or re.fullmatch(r"[0-9a-f]{64}", current_attestation) is None
        ):
            _stop("invalid computed attestation transition")
    elif previous_attestation is None and current_attestation is not None:
        _stop("computed attestation appeared outside computed transition")
    if current["state"] == "FINAL" and current["phase"] != "TERMINAL":
        _stop("FINAL must be terminal")


def _parse_event_chain(
    payload: bytes, protocol: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if not payload or not payload.endswith(b"\n"):
        _stop("event journal is empty or partial")
    lines = payload.splitlines(keepends=True)
    events: list[dict[str, Any]] = []
    previous_hash: str | None = None
    for index, line in enumerate(lines):
        if not line.endswith(b"\n") or line.count(b"\n") != 1:
            _stop("partial event line")
        event = parse_json(line, f"attempt event {index}")
        _require_keys(event, protocol["event_keyset"], "attempt event")
        if line != canonical_json(event):
            _stop("attempt event is not canonical")
        if event["schema_version"] != protocol["schema_versions"]["event"]:
            _stop("attempt event schema mismatch")
        if event["previous_event_sha256"] != previous_hash:
            _stop("event hash chain mismatch")
        _validate_transition(events[-1] if events else None, event, protocol)
        events.append(event)
        previous_hash = hashlib.sha256(line).hexdigest()
    return events


def load_event_chain(
    attempt_dir: Path, protocol: Mapping[str, Any]
) -> list[dict[str, Any]]:
    path = attempt_dir / "events.jsonl"
    if not path.exists():
        _stop("attempt event journal missing")
    return _parse_event_chain(_read_small_path(path), protocol)


def recover_state_cache(
    attempt_dir: Path,
    protocol: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    chain = list(events or load_event_chain(attempt_dir, protocol))
    derived = _event_to_state(chain[-1], protocol)
    path = attempt_dir / "state.json"
    if path.exists():
        existing = parse_json(_read_small_path(path), "state cache")
        _require_keys(existing, protocol["state_keyset"], "state cache")
        if existing == derived:
            return derived
        if not any(
            _event_projection_matches_state(existing, event, protocol)
            for event in chain[:-1]
        ):
            _stop("state cache conflicts with authoritative journal")
    _atomic_json_cache(path, derived, protocol)
    return derived


def append_event(
    attempt_dir: Path,
    protocol: Mapping[str, Any],
    *,
    state: str,
    phase: str,
    oracle_open_committed: bool,
    evaluator_build_id: str | None,
    computed_attestation_sha256: str | None | object = ...,
    reason_code: str | None = None,
    now: Callable[[], str] = _utc_now,
) -> dict[str, Any]:
    path = attempt_dir / "events.jsonl"
    fd = os.open(
        path,
        os.O_RDWR
        | os.O_APPEND
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        # Re-read under the append lock: the previous hash and sequence form
        # a compare-and-swap boundary for concurrent recovery processes.
        journal_payload = _read_fd(fd, ADMIN_RSS_LIMIT)
        if len(journal_payload) > MAX_JSON_BYTES:
            _stop("event journal exceeds administrative limit")
        events = _parse_event_chain(journal_payload, protocol)
        previous_line = canonical_json(events[-1])
        attestation_sha = (
            events[-1]["computed_attestation_sha256"]
            if computed_attestation_sha256 is ...
            else computed_attestation_sha256
        )
        event = {
            "schema_version": protocol["schema_versions"]["event"],
            "measurement_slot_id": events[-1]["measurement_slot_id"],
            "attempt_id": events[-1]["attempt_id"],
            "sequence": events[-1]["sequence"] + 1,
            "state": state,
            "phase": phase,
            "oracle_open_committed": oracle_open_committed,
            "evaluator_build_id": evaluator_build_id,
            "computed_attestation_sha256": attestation_sha,
            "reason_code": reason_code,
            "timestamp_utc": now(),
            "previous_event_sha256": hashlib.sha256(previous_line).hexdigest(),
        }
        _require_keys(event, protocol["event_keyset"], "attempt event")
        _validate_transition(events[-1], event, protocol)
        line = canonical_json(event)
        view = memoryview(line)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                _stop("partial event append")
            view = view[written:]
        os.fsync(fd)
        _atomic_json_cache(
            path.with_name("state.json"),
            _event_to_state(event, protocol),
            protocol,
        )
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(fd)
    _fsync_dir(attempt_dir)
    return event


@contextlib.contextmanager
def _exclusive_slot_lock(
    attempt_root: Path, slot_id: str
) -> Iterable[dict[str, Any]]:
    _ensure_dir(attempt_root)
    _parent_path, parent_fd = _openat_anchored(
        attempt_root.parent, directory=True
    )
    _root_path, root_fd = _openat_anchored(attempt_root, directory=True)
    lock_name = f".slot-{slot_id}.lock"
    fd = os.open(
        lock_name,
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=root_fd,
    )
    try:
        lock_info = os.fstat(fd)
        if (
            not stat.S_ISREG(lock_info.st_mode)
            or lock_info.st_uid != os.geteuid()
            or stat.S_IMODE(lock_info.st_mode) != 0o600
        ):
            _stop("measurement slot lock ownership/mode mismatch")
        try:
            # The parent-directory lock prevents a pathname swap from
            # creating a second independently lockable slot inode.
            fcntl.flock(parent_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _stop("measurement slot is already executing")
        record = {
            "attempt_root": attempt_root,
            "parent_fd": parent_fd,
            "parent_snapshot": os.fstat(parent_fd),
            "root_fd": root_fd,
            "root_snapshot": os.fstat(root_fd),
            "lock_fd": fd,
            "lock_snapshot": os.fstat(fd),
            "lock_name": lock_name,
        }
        _verify_slot_lock(record)
        yield record
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
            try:
                fcntl.flock(parent_fd, fcntl.LOCK_UN)
            finally:
                os.close(root_fd)
                os.close(parent_fd)


def _verify_slot_lock(record: Mapping[str, Any]) -> None:
    parent_path = Path(record["attempt_root"]).parent
    _path, current_parent_fd = _openat_anchored(parent_path, directory=True)
    _root_path, current_root_fd = _openat_anchored(
        Path(record["attempt_root"]), directory=True
    )
    try:
        current_lock_fd = os.open(
            record["lock_name"],
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=record["root_fd"],
        )
        try:
            triples = (
                (os.fstat(current_parent_fd), record["parent_snapshot"]),
                (os.fstat(current_root_fd), record["root_snapshot"]),
                (os.fstat(current_lock_fd), record["lock_snapshot"]),
                (os.fstat(record["lock_fd"]), record["lock_snapshot"]),
            )
            if any(
                (current.st_dev, current.st_ino)
                != (expected.st_dev, expected.st_ino)
                for current, expected in triples
            ):
                _stop("measurement slot pathname/inode substitution")
            for info in (
                os.fstat(current_lock_fd),
                os.fstat(record["lock_fd"]),
            ):
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_uid != os.geteuid()
                    or stat.S_IMODE(info.st_mode) != 0o600
                ):
                    _stop("measurement slot lock ownership/mode mismatch")
        finally:
            os.close(current_lock_fd)
    finally:
        os.close(current_root_fd)
        os.close(current_parent_fd)


def _validate_attempt_tree(
    attempt_dir: Path,
    protocol: Mapping[str, Any],
    *,
    allow_missing_state: bool = False,
) -> bool:
    names = {path.name for path in attempt_dir.iterdir()}
    before = set(protocol["attempt_tree_before_computed_attestation"])
    after = set(protocol["attempt_tree_with_computed_attestation"])
    if names == before:
        return False
    if names == after:
        return True
    if allow_missing_state:
        if names == before - {"state.json"}:
            return False
        if names == after - {"state.json"}:
            return True
    _stop("attempt slot tree mismatch")


def ensure_receipt(
    plan: Mapping[str, Any],
    lock: Mapping[str, Any],
    plan_sha: str,
    lock_sha: str,
    *,
    now: Callable[[], str] = _utc_now,
) -> tuple[Path, str, str]:
    protocol = plan["attempt_protocol"]
    slot_id, attempt_id, attempt_payload = _attempt_identities(
        plan, lock, plan_sha, lock_sha
    )
    root = Path(plan["outputs"]["attempt_root"])
    _ensure_dir(root)
    attempt_dir = root / slot_id
    try:
        attempt_dir.mkdir(mode=0o700)
        created = True
        _fsync_dir(root)
    except FileExistsError:
        created = False
    receipt_path = attempt_dir / "receipt.json"
    receipt_policy = {
        "schema_version": protocol["schema_versions"]["receipt"],
        "measurement_slot_id": slot_id,
        "attempt_id": attempt_id,
        "plan_sha256": attempt_payload["plan_sha256"],
        "lock_sha256": attempt_payload["lock_sha256"],
        "input_hashes": attempt_payload["input_hashes"],
        "evaluation_spec_sha256": attempt_payload["evaluation_spec_sha256"],
        "worker_build_id": attempt_payload["worker_build_id"],
        "oracle_build_id": attempt_payload["oracle_build_id"],
        "parity_build_id": attempt_payload["parity_build_id"],
        "policy_immutable": True,
    }
    if not created:
        _validate_attempt_tree(
            attempt_dir, protocol, allow_missing_state=True
        )
        receipt = parse_json(_read_small_path(receipt_path), "receipt")
        _require_keys(receipt, protocol["receipt_keyset"], "receipt")
        for key, expected in receipt_policy.items():
            if receipt.get(key) != expected:
                _stop("measurement slot already reserved by another policy")
        events = load_event_chain(attempt_dir, protocol)
        recover_state_cache(attempt_dir, protocol, events)
        _validate_attempt_tree(attempt_dir, protocol)
        return attempt_dir, slot_id, attempt_id
    receipt = {
        **receipt_policy,
        "created_at_utc": now(),
    }
    _require_keys(receipt, protocol["receipt_keyset"], "receipt")
    _write_exclusive(receipt_path, canonical_json(receipt), 0o400)
    first = {
        "schema_version": protocol["schema_versions"]["event"],
        "measurement_slot_id": slot_id,
        "attempt_id": attempt_id,
        "sequence": 0,
        "state": "STARTED",
        "phase": "RECEIPT_DURABLE",
        "oracle_open_committed": False,
        "evaluator_build_id": None,
        "computed_attestation_sha256": None,
        "reason_code": None,
        "timestamp_utc": now(),
        "previous_event_sha256": None,
    }
    _require_keys(first, protocol["event_keyset"], "initial event")
    _validate_transition(None, first, protocol)
    _write_exclusive(attempt_dir / "events.jsonl", canonical_json(first))
    _atomic_json_cache(
        attempt_dir / "state.json",
        _event_to_state(first, protocol),
        protocol,
    )
    _fsync_dir(attempt_dir)
    _fsync_dir(root)
    _validate_attempt_tree(attempt_dir, protocol)
    return attempt_dir, slot_id, attempt_id


def _validate_table_schema(
    table: pa.Table, expected: Sequence[tuple[str, pa.DataType, bool]], label: str
) -> None:
    schema = pa.schema(
        [pa.field(name, dtype, nullable=nullable) for name, dtype, nullable in expected],
        metadata=None,
    )
    if not table.schema.equals(schema, check_metadata=True):
        _stop(f"{label} schema mismatch: {table.schema}")


def _clean_text(value: Any, label: str) -> str:
    if type(value) is not str or value == "" or value != value.strip():
        _stop(f"invalid {label}")
    if "\x00" in value or "\n" in value or "\r" in value:
        _stop(f"unsafe {label}")
    return value


def wilson(success: int, denominator: int, z: float) -> tuple[float, float]:
    if (
        type(success) is not int
        or type(denominator) is not int
        or denominator <= 0
        or success < 0
        or success > denominator
        or not math.isfinite(z)
        or z <= 0
    ):
        _stop("invalid Wilson inputs")
    p = success / denominator
    scale = 1.0 + z * z / denominator
    centre = (p + z * z / (2.0 * denominator)) / scale
    radius = (
        z
        * math.sqrt(
            p * (1.0 - p) / denominator
            + z * z / (4.0 * denominator * denominator)
        )
        / scale
    )
    return max(0.0, centre - radius), min(1.0, centre + radius)


def _proportion(
    success: int, denominator: int, spec: Mapping[str, Any]
) -> dict[str, Any]:
    low95, high95 = wilson(success, denominator, spec["z"]["0.95"])
    low99, high99 = wilson(success, denominator, spec["z"]["0.99"])
    value = {
        "success_count": success,
        "denominator_count": denominator,
        "rate": success / denominator,
        "wilson_95_low": low95,
        "wilson_95_high": high95,
        "wilson_99_low": low99,
        "wilson_99_high": high99,
    }
    return value


def _outcomes_payload(rows: Sequence[Mapping[str, Any]]) -> bytes:
    payload = bytearray()
    columns = (
        "query_id",
        "dev_partition",
        "label_kind",
        "candidate_count",
        "exact_rank",
        "hit_at_1",
        "hit_at_10",
        "hit_at_50",
        "hit_at_100",
    )
    for row in rows:
        encoded: list[bytes] = []
        for column in columns:
            value = row[column]
            if value is None:
                encoded.append(b"\\N")
            elif column in {"hit_at_1", "hit_at_10", "hit_at_50", "hit_at_100"}:
                if type(value) is not bool:
                    _stop("outcome hit is not boolean")
                encoded.append(b"1" if value else b"0")
            elif column in {"candidate_count", "exact_rank"}:
                if type(value) is not int or value < 0:
                    _stop("outcome integer is invalid")
                encoded.append(str(value).encode("ascii"))
            elif column == "query_id":
                encoded.append(_clean_text(value, column).encode("utf-8"))
            else:
                encoded.append(_clean_text(value, column).encode("ascii"))
        payload.extend(b"\x00".join(encoded))
        payload.extend(b"\n")
    return bytes(payload)


def evaluate_tables(
    candidates: pa.Table,
    statuses: pa.Table,
    oracle: pa.Table,
    spec: Mapping[str, Any],
) -> tuple[pa.Table, dict[str, Any], bytes, int]:
    _validate_table_schema(
        candidates,
        (
            ("query_id", pa.string(), False),
            ("candidate_rank", pa.uint8(), False),
            ("candidate_siret", pa.string(), False),
        ),
        "candidates",
    )
    _validate_table_schema(
        statuses,
        (("query_id", pa.string(), False), ("candidate_count", pa.uint8(), False)),
        "statuses",
    )
    _validate_table_schema(
        oracle,
        (
            ("query_id", pa.string(), False),
            ("dev_partition", pa.string(), False),
            ("label_kind", pa.string(), False),
            ("ground_truth_siret", pa.string(), True),
            ("ground_truth_siren", pa.string(), True),
        ),
        "oracle",
    )
    candidate_rows = candidates.to_pylist()
    status_rows = statuses.to_pylist()
    oracle_rows = oracle.to_pylist()
    status_ids = [row["query_id"] for row in status_rows]
    oracle_ids = [row["query_id"] for row in oracle_rows]
    if (
        len(status_ids) != len(set(status_ids))
        or len(oracle_ids) != len(set(oracle_ids))
        or status_ids != oracle_ids
        or len(oracle_ids) != spec["join"]["expected_query_count"]
    ):
        _stop("oracle/status population mismatch")
    grouped: dict[str, list[dict[str, Any]]] = {query_id: [] for query_id in oracle_ids}
    previous_query_index = -1
    query_index = {query_id: index for index, query_id in enumerate(oracle_ids)}
    for row in candidate_rows:
        query_id = _clean_text(row["query_id"], "candidate query_id")
        if query_id not in grouped:
            _stop("candidate has an unknown query")
        current_index = query_index[query_id]
        if current_index < previous_query_index:
            _stop("candidate query order mismatch")
        previous_query_index = current_index
        grouped[query_id].append(row)
    if len(candidate_rows) != spec["join"]["expected_candidate_count"]:
        _stop("candidate count mismatch")
    status_by_id = {row["query_id"]: row for row in status_rows}
    max_pool = 0
    min_pool = 256
    under = 0
    empty = 0
    for query_id in oracle_ids:
        rows = grouped[query_id]
        count = status_by_id[query_id]["candidate_count"]
        if type(count) is not int or count != len(rows) or count > 100:
            _stop("candidate_count mismatch or ceiling exceeded")
        ranks = [row["candidate_rank"] for row in rows]
        if ranks != list(range(1, count + 1)):
            _stop("candidate ranks are not contiguous")
        sirets = [row["candidate_siret"] for row in rows]
        if len(sirets) != len(set(sirets)) or any(
            type(siret) is not str or re.fullmatch(r"[0-9]{14}", siret) is None
            for siret in sirets
        ):
            _stop("invalid or duplicate candidate SIRET")
        max_pool = max(max_pool, count)
        min_pool = min(min_pool, count)
        under += int(count < 100)
        empty += int(count == 0)
    expected_join = spec["join"]
    if (
        max_pool != expected_join["expected_maximum_pool_size"]
        or min_pool != expected_join["expected_minimum_pool_size"]
        or under != expected_join["expected_under_ceiling_query_count"]
        or empty != expected_join["expected_empty_query_count"]
    ):
        _stop("pool aggregate mismatch")
    output_rows: list[dict[str, Any]] = []
    for truth in oracle_rows:
        query_id = _clean_text(truth["query_id"], "oracle query_id")
        partition = _clean_text(truth["dev_partition"], "dev_partition")
        kind = _clean_text(truth["label_kind"], "label_kind")
        if partition not in {"threshold_dev", "comparison_dev"}:
            _stop("invalid dev partition")
        if kind not in {"MATCH_EXACT", "AMBIGUOUS"}:
            _stop("invalid label kind")
        count = status_by_id[query_id]["candidate_count"]
        exact_rank: int | None = None
        if kind == "MATCH_EXACT":
            siret = truth["ground_truth_siret"]
            siren = truth["ground_truth_siren"]
            if (
                type(siret) is not str
                or re.fullmatch(r"[0-9]{14}", siret) is None
                or type(siren) is not str
                or re.fullmatch(r"[0-9]{9}", siren) is None
                or siret[:9] != siren
            ):
                _stop("invalid exact truth")
            ranks = [
                row["candidate_rank"]
                for row in grouped[query_id]
                if row["candidate_siret"] == siret
            ]
            exact_rank = ranks[0] if ranks else None
            hits: dict[str, bool | None] = {
                f"hit_at_{k}": exact_rank is not None and exact_rank <= k
                for k in spec["recall_k"]
            }
        else:
            if (
                truth["ground_truth_siret"] is not None
                or truth["ground_truth_siren"] is not None
            ):
                _stop("ambiguous truth is not null")
            hits = {f"hit_at_{k}": None for k in spec["recall_k"]}
        output_rows.append(
            {
                "query_id": query_id,
                "dev_partition": partition,
                "label_kind": kind,
                "candidate_count": count,
                "exact_rank": exact_rank,
                **hits,
            }
        )
    schema = pa.schema(
        [
            pa.field(item["name"], pa.type_for_alias(item["type"]), item["nullable"])
            for item in spec_artifact_schema(spec, output=True)
        ],
        metadata=None,
    )
    arrays = [
        pa.array([row[field.name] for row in output_rows], type=field.type)
        for field in schema
    ]
    table = pa.Table.from_arrays(arrays, schema=schema)
    payload = _outcomes_payload(output_rows)
    missing = sum(
        row["label_kind"] == "MATCH_EXACT" and row["exact_rank"] is None
        for row in output_rows
    )
    return table, {"rows": output_rows, "pool": (min_pool, max_pool, under, empty)}, payload, missing


_ACTIVE_ARTIFACT_CONTRACT: Mapping[str, Any] | None = None


def spec_artifact_schema(
    _spec: Mapping[str, Any], *, output: bool = False
) -> Sequence[Mapping[str, Any]]:
    del _spec, output
    if _ACTIVE_ARTIFACT_CONTRACT is None:
        _stop("artifact contract is not active")
    return _ACTIVE_ARTIFACT_CONTRACT["query_outcomes_schema"]


def build_metrics(
    rows: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any],
    ids: Mapping[str, str],
) -> tuple[dict[str, Any], int]:
    measurements: list[dict[str, Any]] = []
    interval = spec["confidence_interval"]
    missing_global = 0
    for population in spec["population_order"]:
        selected = (
            list(rows)
            if population == "global"
            else [row for row in rows if row["dev_partition"] == population]
        )
        exact = [row for row in selected if row["label_kind"] == "MATCH_EXACT"]
        ambiguous = [row for row in selected if row["label_kind"] == "AMBIGUOUS"]
        expected = spec["population_counts"][population]
        if (
            len(selected) != expected["total"]
            or len(exact) != expected["MATCH_EXACT"]
            or len(ambiguous) != expected["AMBIGUOUS"]
        ):
            _stop(f"population count mismatch: {population}")
        recalls: dict[str, Any] = {}
        for k in spec["recall_k"]:
            success = sum(row[f"hit_at_{k}"] is True for row in exact)
            recalls[str(k)] = _proportion(success, len(exact), interval)
        if population == "global":
            missing_global = len(exact) - recalls["100"]["success_count"]
        measurements.append(
            {
                "population": population,
                "measurement_type": "V412_MEASUREMENT",
                "total_query_count": len(selected),
                "match_exact_count": len(exact),
                "ambiguous_count": len(ambiguous),
                "coverage": _proportion(len(exact), len(selected), interval),
                "recall_at": recalls,
            }
        )
    references: list[dict[str, Any]] = []
    frozen = spec["frozen_references"]
    for row in frozen["rows"]:
        references.append(
            {
                "name": row["name"],
                "reference_type": "FROZEN_REFERENCE",
                "source_build_id": frozen["build_id"],
                "source_population_count": row["coverage_denominator"],
                "coverage": _proportion(
                    row["coverage_success"], row["coverage_denominator"], interval
                ),
                "recall_at_100": _proportion(
                    row["recall_at_100_success"],
                    row["recall_at_100_denominator"],
                    interval,
                ),
            }
        )
    global_measurement = measurements[0]
    gate = spec["gate"]
    coverage_rate = global_measurement["coverage"]["rate"]
    recall_rate = global_measurement["recall_at"]["100"]["rate"]
    gates = {
        "gate_statistic": "OBSERVED_RATE_FROM_RAW_COUNTS",
        "population": "global",
        "coverage_minimum": gate["coverage_minimum"],
        "recall_at_100_minimum": gate["recall_at_100_minimum"],
        "coverage_observed": coverage_rate,
        "recall_at_100_observed": recall_rate,
        "coverage_pass": coverage_rate >= gate["coverage_minimum"],
        "recall_at_100_pass": recall_rate >= gate["recall_at_100_minimum"],
        "all_pass": coverage_rate >= gate["coverage_minimum"]
        and recall_rate >= gate["recall_at_100_minimum"],
    }
    latency_spec = spec["latency"]
    durations_ns = latency_spec["durations_ns"]
    latency = {
        "latency_source": "WORKER_INTEGRITY_AGGREGATE",
        "durations_ns": durations_ns,
        "durations_seconds": {
            key: value / 1_000_000_000 for key, value in durations_ns.items()
        },
        "query_count": latency_spec["query_count"],
        "mean_wall_seconds_per_query_from_aggregate": durations_ns["total"]
        / 1_000_000_000
        / latency_spec["query_count"],
        "per_query_timing_available": False,
        "p95_available": False,
        "latency_gate_evaluated": False,
    }
    declarations = {
        "historical_development_only": True,
        "independent_measurement": False,
        "production_certified": False,
        "models_opened": False,
        "historical_sources_opened": False,
        "challenge_or_final_opened": False,
        "tuning_performed": False,
    }
    verdict = (
        "GO_V412_UNIT_RETRIEVAL_EVALUATION"
        if gates["all_pass"]
        else "PIVOT_V412_UNIT_RETRIEVAL_EVALUATION"
    )
    metrics = {
        "schema_version": _ACTIVE_ARTIFACT_CONTRACT["schema_versions"]["metrics"],
        "evaluator_build_id": ids["evaluator_build_id"],
        "attempt_id": ids["attempt_id"],
        "worker_build_id": ids["worker_build_id"],
        "oracle_build_id": ids["oracle_build_id"],
        "parity_build_id": ids["parity_build_id"],
        "population_order": spec["population_order"],
        "reference_order": spec["reference_order"],
        "v412_measurements": measurements,
        "frozen_references": references,
        "latency": latency,
        "gates": gates,
        "verdict": verdict,
        "declarations": declarations,
    }
    return metrics, missing_global


def _parquet_record(path: Path) -> dict[str, Any]:
    table = pq.read_table(path)
    return {
        "path": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "row_count": table.num_rows,
        "schema": [
            {
                "name": field.name,
                "type": str(field.type),
                "nullable": field.nullable,
            }
            for field in table.schema
        ],
        "metadata": (
            None
            if table.schema.metadata is None
            else {
                key.decode(): value.decode()
                for key, value in table.schema.metadata.items()
            }
        ),
    }


def _json_record(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": path.name,
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _ledger_table(
    snapshots_before: Mapping[str, Mapping[str, Any]],
    snapshots_after: Mapping[str, Mapping[str, Any]],
    paths: Mapping[str, str],
    artifact: Mapping[str, Any],
) -> pa.Table:
    rows = []
    for role in artifact["ledger_role_order"]:
        before = snapshots_before[role]
        after = snapshots_after[role]
        rows.append(
            {
                "role": role,
                "absolute_path": paths[role],
                "projection": DATA_PROJECTIONS.get(role, "FULL_JSON_EXACT_KEYSET"),
                "size_bytes_before": before["size_bytes"],
                "sha256_before": before["sha256"],
                "size_bytes_after": after["size_bytes"],
                "sha256_after": after["sha256"],
            }
        )
    schema = pa.schema(
        [
            pa.field(item["name"], pa.type_for_alias(item["type"]), False)
            for item in artifact["ledger_schema"]
        ],
        metadata=None,
    )
    return pa.Table.from_pylist(rows, schema=schema)


def _check_keysets(
    metrics: Mapping[str, Any],
    integrity: Mapping[str, Any],
    eval_manifest: Mapping[str, Any],
    provenance: Mapping[str, Any],
    audit_manifest: Mapping[str, Any],
    artifact: Mapping[str, Any],
) -> None:
    keys = artifact["json_keysets"]
    _require_keys(metrics, keys["metrics"], "metrics")
    _require_keys(integrity, keys["integrity"], "integrity")
    _require_keys(eval_manifest, keys["evaluation_manifest"], "evaluation manifest")
    _require_keys(provenance, keys["provenance"], "provenance")
    _require_keys(audit_manifest, keys["audit_manifest"], "audit manifest")
    for record in metrics["v412_measurements"]:
        _require_keys(record, keys["measurement_record"], "measurement")
        _require_keys(record["coverage"], keys["proportion_record"], "coverage")
        _require_keys(record["recall_at"], keys["recall_at"], "recall_at")
        for value in record["recall_at"].values():
            _require_keys(value, keys["proportion_record"], "recall")
    for record in metrics["frozen_references"]:
        _require_keys(record, keys["reference_record"], "reference")
        _require_keys(record["coverage"], keys["proportion_record"], "reference coverage")
        _require_keys(
            record["recall_at_100"], keys["proportion_record"], "reference recall"
        )
    _require_keys(metrics["latency"], keys["latency"], "latency")
    _require_keys(metrics["gates"], keys["gates"], "gates")
    for value in (
        metrics["declarations"],
        integrity["declarations"],
        eval_manifest["declarations"],
        provenance["declarations"],
    ):
        _require_keys(value, keys["declarations"], "declarations")


def worker_execute(spec: Mapping[str, Any]) -> tuple[Path, Path]:
    if set(spec) != WORKER_SPEC_KEYS or spec.get("schema_version") != WORKER_SPEC_SCHEMA:
        _stop("worker spec mismatch")
    global _ACTIVE_ARTIFACT_CONTRACT
    _ACTIVE_ARTIFACT_CONTRACT = spec["artifact_contract"]
    limit = spec["max_rss_bytes"]
    # `platform.platform()` may invoke macOS probes forbidden by the
    # deny-by-default worker sandbox, so its value can legitimately differ
    # from the parent process.  The parent has already checked the complete
    # runtime fingerprint against the locked plan.  Re-attest here every
    # stable field that can affect the computation.
    worker_runtime = _runtime()
    for key in (
        "python",
        "numpy",
        "pandas",
        "pyarrow",
        "scikit_learn",
        "scipy",
        "joblib",
        "duckdb",
        "machine",
    ):
        if worker_runtime[key] != spec["runtime"][key]:
            _stop(f"worker runtime mismatch: {key}")
    roles = spec["artifact_contract"]["ledger_role_order"]
    if set(spec["input_fds"]) != set(roles):
        _stop("worker descriptor role mismatch")
    descriptors = {role: int(spec["input_fds"][role]) for role in roles}
    before: dict[str, dict[str, Any]] = {}
    for role, fd in descriptors.items():
        snapshot = _snapshot_fd(fd, limit)
        if snapshot["sha256"] != spec["input_hashes"][role]:
            _stop(f"worker input mismatch: {role}")
        before[role] = snapshot
    json_inputs = {
        role: _json_from_fd(descriptors[role], role, limit)
        for role in roles
        if role not in DATA_PROJECTIONS
    }
    worker_manifest = json_inputs["worker_manifest"]
    worker_integrity = json_inputs["worker_integrity"]
    parity = json_inputs["parity_result"]
    oracle_manifest = json_inputs["oracle_manifest"]
    if (
        worker_manifest.get("worker_build_id") != spec["worker_build_id"]
        or worker_manifest.get("verdict") != "SEALED_V412_UNIT_RETRIEVAL"
        or parity.get("worker_build_id") != spec["worker_build_id"]
        or parity.get("parity_build_id") != spec["parity_build_id"]
        or parity.get("verdict") != "GO_V412_UNIT_RETRIEVAL_PARITY"
        or oracle_manifest.get("build_id") != spec["oracle_build_id"]
        or worker_integrity.get("durations_ns")
        != spec["evaluation_spec"]["latency"]["durations_ns"]
    ):
        _stop("sealed input identity/verdict mismatch")
    candidates = _table_from_fd(
        descriptors["worker_candidates_top100"],
        ("query_id", "candidate_rank", "candidate_siret"),
        "worker candidates",
        limit,
    )
    statuses = _table_from_fd(
        descriptors["worker_query_status"],
        ("query_id", "candidate_count"),
        "worker statuses",
        limit,
    )
    oracle = _table_from_fd(
        descriptors["oracle_dev"],
        (
            "query_id",
            "dev_partition",
            "label_kind",
            "ground_truth_siret",
            "ground_truth_siren",
        ),
        "oracle",
        limit,
    )
    outcomes, detail, logical_payload, missing = evaluate_tables(
        candidates, statuses, oracle, spec["evaluation_spec"]
    )
    ids = {
        "evaluator_build_id": spec["evaluator_build_id"],
        "attempt_id": spec["attempt_id"],
        "worker_build_id": spec["worker_build_id"],
        "oracle_build_id": spec["oracle_build_id"],
        "parity_build_id": spec["parity_build_id"],
    }
    metrics, metric_missing = build_metrics(
        detail["rows"], spec["evaluation_spec"], ids
    )
    if missing != metric_missing:
        _stop("missing truth formula mismatch")
    eval_stage = Path(spec["evaluation_stage"])
    audit_stage = Path(spec["audit_stage"])
    eval_stage.mkdir(mode=0o700, parents=False)
    audit_stage.mkdir(mode=0o700, parents=False)
    outcomes_path = eval_stage / "query_outcomes.parquet"
    metrics_path = eval_stage / "metrics.json"
    integrity_path = eval_stage / "integrity.json"
    _write_parquet(outcomes_path, outcomes)
    _write_json(metrics_path, metrics)
    pool_min, pool_max, under, empty = detail["pool"]
    declarations = metrics["declarations"]
    integrity = {
        "schema_version": spec["artifact_contract"]["schema_versions"]["integrity"],
        **ids,
        "query_count": outcomes.num_rows,
        "match_exact_count": spec["evaluation_spec"]["population_counts"]["global"][
            "MATCH_EXACT"
        ],
        "ambiguous_count": spec["evaluation_spec"]["population_counts"]["global"][
            "AMBIGUOUS"
        ],
        "candidate_count": candidates.num_rows,
        "minimum_pool_size": pool_min,
        "maximum_pool_size": pool_max,
        "under_ceiling_query_count": under,
        "empty_query_count": empty,
        "missing_truth_count": missing,
        "query_outcomes_payload_bytes": len(logical_payload),
        "query_outcomes_payload_sha256": hashlib.sha256(logical_payload).hexdigest(),
        "metrics_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
        "input_snapshot_count": len(spec["input_hashes"]),
        "opened_input_count": len(roles),
        "gate_statistic": "OBSERVED_RATE_FROM_RAW_COUNTS",
        "declarations": declarations,
        "verdict": metrics["verdict"],
    }
    _write_json(integrity_path, integrity)
    eval_manifest = {
        "schema_version": spec["artifact_contract"]["schema_versions"][
            "evaluation_manifest"
        ],
        **ids,
        "files": {
            "query_outcomes.parquet": _parquet_record(outcomes_path),
            "metrics.json": _json_record(metrics_path),
            "integrity.json": _json_record(integrity_path),
        },
        "runtime": spec["runtime"],
        "declarations": declarations,
        "verdict": metrics["verdict"],
    }
    eval_manifest_path = eval_stage / "manifest.json"
    _write_json(eval_manifest_path, eval_manifest)
    after = {role: _snapshot_fd(fd, limit) for role, fd in descriptors.items()}
    for role in roles:
        if not _same_snapshot(before[role], after[role]):
            _stop(f"input mutated after evaluation: {role}")
    ledger = _ledger_table(
        before, after, spec["input_paths"], spec["artifact_contract"]
    )
    ledger_path = audit_stage / "open_ledger.parquet"
    _write_parquet(ledger_path, ledger)
    provenance = {
        "schema_version": spec["artifact_contract"]["schema_versions"]["provenance"],
        "evaluator_build_id": spec["evaluator_build_id"],
        "attempt_id": spec["attempt_id"],
        "git_commit": spec["git_commit"],
        "source_hashes": spec["source_hashes"],
        "plan_sha256": spec["plan_sha256"],
        "lock_sha256": spec["lock_sha256"],
        "input_hashes": spec["input_hashes"],
        "evaluation_spec_sha256": hashlib.sha256(
            canonical_json(spec["evaluation_spec"])
        ).hexdigest(),
        "runtime": spec["runtime"],
        "data_input_count": len(roles),
        "evaluation_manifest_sha256": hashlib.sha256(
            eval_manifest_path.read_bytes()
        ).hexdigest(),
        "declarations": declarations,
    }
    # The worker projection intentionally omits parent-only values. The parent
    # injects them into the already-keyed provenance before publication.
    provenance_path = audit_stage / "provenance.json"
    _write_json(provenance_path, provenance)
    audit_manifest = {
        "schema_version": spec["artifact_contract"]["schema_versions"][
            "audit_manifest"
        ],
        "evaluator_build_id": spec["evaluator_build_id"],
        "attempt_id": spec["attempt_id"],
        "files": {
            "open_ledger.parquet": _json_record(ledger_path),
            "provenance.json": _json_record(provenance_path),
        },
    }
    _check_keysets(
        metrics,
        integrity,
        eval_manifest,
        provenance,
        audit_manifest,
        spec["artifact_contract"],
    )
    _write_json(audit_stage / "manifest.json", audit_manifest)
    _fsync_dir(eval_stage)
    _fsync_dir(audit_stage)
    return eval_stage, audit_stage


def _worker_pre_oracle_attest(spec: Mapping[str, Any]) -> None:
    if set(spec) != WORKER_SPEC_KEYS or spec.get("schema_version") != WORKER_SPEC_SCHEMA:
        _stop("worker pre-oracle spec mismatch")
    oracle_roles = set(ORACLE_ROLE_ORDER)
    expected = set(spec["artifact_contract"]["ledger_role_order"]) - oracle_roles
    if set(spec["input_fds"]) != expected:
        _stop("pre-oracle worker descriptor set mismatch")
    for role, raw_fd in spec["input_fds"].items():
        snapshot = _snapshot_fd(int(raw_fd), spec["max_rss_bytes"])
        if snapshot["sha256"] != spec["input_hashes"][role]:
            _stop(f"pre-oracle worker input mismatch: {role}")
    actual = _runtime()
    for key in (
        "python",
        "numpy",
        "pandas",
        "pyarrow",
        "scikit_learn",
        "scipy",
        "joblib",
        "duckdb",
        "machine",
    ):
        if actual[key] != spec["runtime"][key]:
            _stop(f"pre-oracle runtime mismatch: {key}")


def _worker_receive_oracle_fds(
    control_fd: int, spec: dict[str, Any]
) -> list[int]:
    control = socket.socket(fileno=control_fd)
    ready = {
        "schema_version": WORKER_CONTROL_SCHEMA,
        "status": "READY_PRE_ORACLE",
        "evaluator_build_id": spec["evaluator_build_id"],
        "attempt_id": spec["attempt_id"],
    }
    control.sendall(canonical_json(ready))
    oracle_roles = list(ORACLE_ROLE_ORDER)
    payload, ancillary, flags, _address = control.recvmsg(
        MAX_JSON_BYTES,
        socket.CMSG_SPACE(len(oracle_roles) * array.array("i").itemsize),
    )
    if flags & (getattr(socket, "MSG_TRUNC", 0) | getattr(socket, "MSG_CTRUNC", 0)):
        _stop("oracle FD control message was truncated")
    message = parse_json(payload, "oracle FD control")
    if set(message) != {
        "schema_version",
        "command",
        "evaluator_build_id",
        "attempt_id",
        "roles",
    } or message != {
        "schema_version": WORKER_CONTROL_SCHEMA,
        "command": "ORACLE_FDS_COMMITTED",
        "evaluator_build_id": spec["evaluator_build_id"],
        "attempt_id": spec["attempt_id"],
        "roles": oracle_roles,
    }:
        _stop("oracle FD control attestation mismatch")
    received = array.array("i")
    for level, kind, data in ancillary:
        if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
            usable = len(data) - (len(data) % received.itemsize)
            received.frombytes(data[:usable])
    fds = list(received)
    if len(fds) != len(oracle_roles):
        for fd in fds:
            os.close(fd)
        _stop("oracle FD transfer count mismatch")
    spec["input_fds"].update(dict(zip(oracle_roles, fds, strict=True)))
    control.close()
    return fds


def _exact_tree(root: Path, expected: Sequence[str]) -> None:
    if not root.is_dir() or root.is_symlink():
        _stop(f"invalid package root: {root}")
    names = sorted(item.name for item in root.iterdir())
    if names != sorted(expected):
        _stop(f"package file set mismatch: {root}")
    if any(not item.is_file() or item.is_symlink() for item in root.iterdir()):
        _stop("package contains a non-regular file")


def _validate_outcome_rows(
    rows: Sequence[Mapping[str, Any]], spec: Mapping[str, Any]
) -> None:
    seen: set[str] = set()
    recall_k = list(spec["recall_k"])
    if recall_k != [1, 10, 50, 100]:
        _stop("unexpected Recall@K contract")
    for row in rows:
        query_id = _clean_text(row["query_id"], "published query_id")
        if query_id in seen:
            _stop("duplicate published query_id")
        seen.add(query_id)
        if row["dev_partition"] not in {"threshold_dev", "comparison_dev"}:
            _stop("invalid published partition")
        count = row["candidate_count"]
        if type(count) is not int or not 0 <= count <= 100:
            _stop("invalid published candidate_count")
        rank = row["exact_rank"]
        if rank is not None and (
            type(rank) is not int or not 1 <= rank <= count
        ):
            _stop("invalid published exact_rank")
        if row["label_kind"] == "AMBIGUOUS":
            if rank is not None or any(
                row[f"hit_at_{k}"] is not None for k in recall_k
            ):
                _stop("ambiguous outcome contains a match result")
        elif row["label_kind"] == "MATCH_EXACT":
            for k in recall_k:
                expected = rank is not None and rank <= k
                if row[f"hit_at_{k}"] is not expected:
                    _stop(f"published Hit@{k} invariant mismatch")
        else:
            _stop("invalid published label kind")


def validate_packages(
    evaluation_root: Path,
    audit_root: Path,
    *,
    evaluator_build_id: str,
    attempt_id: str,
    artifact: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> None:
    _exact_tree(evaluation_root, ("query_outcomes.parquet", "metrics.json", "integrity.json", "manifest.json"))
    _exact_tree(audit_root, ("open_ledger.parquet", "provenance.json", "manifest.json"))
    metrics = parse_json((evaluation_root / "metrics.json").read_bytes(), "metrics")
    integrity = parse_json((evaluation_root / "integrity.json").read_bytes(), "integrity")
    eval_manifest = parse_json((evaluation_root / "manifest.json").read_bytes(), "evaluation manifest")
    provenance = parse_json((audit_root / "provenance.json").read_bytes(), "provenance")
    audit_manifest = parse_json((audit_root / "manifest.json").read_bytes(), "audit manifest")
    _check_keysets(
        metrics, integrity, eval_manifest, provenance, audit_manifest, artifact
    )
    if metrics["population_order"] != artifact["population_order"] or [
        row["population"] for row in metrics["v412_measurements"]
    ] != artifact["population_order"]:
        _stop("published population order mismatch")
    if metrics["reference_order"] != artifact["reference_order"] or [
        row["name"] for row in metrics["frozen_references"]
    ] != artifact["reference_order"]:
        _stop("published reference order mismatch")
    interval = {
        "z": {"0.95": 1.959963984540054, "0.99": 2.5758293035489004}
    }
    for measurement in metrics["v412_measurements"]:
        expected_coverage = _proportion(
            measurement["coverage"]["success_count"],
            measurement["coverage"]["denominator_count"],
            interval,
        )
        if measurement["coverage"] != expected_coverage:
            _stop("published coverage calculation mismatch")
        for key in ("1", "10", "50", "100"):
            record = measurement["recall_at"][key]
            if record != _proportion(
                record["success_count"], record["denominator_count"], interval
            ):
                _stop(f"published Recall@{key} calculation mismatch")
    for reference in metrics["frozen_references"]:
        for name in ("coverage", "recall_at_100"):
            record = reference[name]
            if record != _proportion(
                record["success_count"], record["denominator_count"], interval
            ):
                _stop(f"published reference {name} calculation mismatch")
    gates = metrics["gates"]
    global_measurement = metrics["v412_measurements"][0]
    coverage = global_measurement["coverage"]["rate"]
    recall = global_measurement["recall_at"]["100"]["rate"]
    if (
        gates["gate_statistic"] != "OBSERVED_RATE_FROM_RAW_COUNTS"
        or gates["coverage_observed"] != coverage
        or gates["recall_at_100_observed"] != recall
        or gates["coverage_pass"] != (coverage >= gates["coverage_minimum"])
        or gates["recall_at_100_pass"]
        != (recall >= gates["recall_at_100_minimum"])
        or gates["all_pass"]
        != (gates["coverage_pass"] and gates["recall_at_100_pass"])
    ):
        _stop("published gate calculation mismatch")
    for value in (metrics, integrity, eval_manifest, provenance, audit_manifest):
        if (
            value.get("evaluator_build_id") != evaluator_build_id
            or value.get("attempt_id") != attempt_id
        ):
            _stop("published identity mismatch")
    if (
        set(eval_manifest["files"])
        != set(artifact["file_sets"]["evaluation_manifest"])
        or set(audit_manifest["files"])
        != set(artifact["file_sets"]["audit_manifest"])
    ):
        _stop("manifest files keyset mismatch")
    for name in artifact["file_sets"]["evaluation_manifest"]:
        record = eval_manifest["files"].get(name)
        if type(record) is not dict:
            _stop("evaluation manifest file missing")
        actual = (
            _parquet_record(evaluation_root / name)
            if name.endswith(".parquet")
            else _json_record(evaluation_root / name)
        )
        if record != actual:
            _stop(f"evaluation manifest mismatch: {name}")
    for name in artifact["file_sets"]["audit_manifest"]:
        record = audit_manifest["files"].get(name)
        if record != _json_record(audit_root / name):
            _stop(f"audit manifest mismatch: {name}")
    if provenance["evaluation_manifest_sha256"] != hashlib.sha256(
        (evaluation_root / "manifest.json").read_bytes()
    ).hexdigest():
        _stop("provenance does not seal evaluation manifest")
    outcomes = pq.read_table(evaluation_root / "query_outcomes.parquet")
    expected_schema = pa.schema(
        [
            pa.field(item["name"], pa.type_for_alias(item["type"]), item["nullable"])
            for item in artifact["query_outcomes_schema"]
        ],
        metadata=None,
    )
    if not outcomes.schema.equals(expected_schema, check_metadata=True):
        _stop("published outcomes schema mismatch")
    outcome_rows = outcomes.to_pylist()
    _validate_outcome_rows(outcome_rows, validation["evaluation_spec"])
    logical = _outcomes_payload(outcome_rows)
    missing = sum(
        row["label_kind"] == "MATCH_EXACT" and row["exact_rank"] is None
        for row in outcome_rows
    )
    ids = {
        "evaluator_build_id": evaluator_build_id,
        "attempt_id": attempt_id,
        "worker_build_id": validation["worker_build_id"],
        "oracle_build_id": validation["oracle_build_id"],
        "parity_build_id": validation["parity_build_id"],
    }
    global _ACTIVE_ARTIFACT_CONTRACT
    previous_contract = _ACTIVE_ARTIFACT_CONTRACT
    _ACTIVE_ARTIFACT_CONTRACT = artifact
    try:
        expected_metrics, expected_missing = build_metrics(
            outcome_rows, validation["evaluation_spec"], ids
        )
    finally:
        _ACTIVE_ARTIFACT_CONTRACT = previous_contract
    metrics_bytes = (evaluation_root / "metrics.json").read_bytes()
    if (
        metrics != expected_metrics
        or metrics_bytes != canonical_json(expected_metrics)
        or missing != expected_missing
    ):
        _stop("published metrics are not an exact outcome recomputation")
    counts = [row["candidate_count"] for row in outcome_rows]
    expected_integrity = {
        "schema_version": artifact["schema_versions"]["integrity"],
        **ids,
        "query_count": len(outcome_rows),
        "match_exact_count": sum(
            row["label_kind"] == "MATCH_EXACT" for row in outcome_rows
        ),
        "ambiguous_count": sum(
            row["label_kind"] == "AMBIGUOUS" for row in outcome_rows
        ),
        "candidate_count": sum(counts),
        "minimum_pool_size": min(counts),
        "maximum_pool_size": max(counts),
        "under_ceiling_query_count": sum(count < 100 for count in counts),
        "empty_query_count": sum(count == 0 for count in counts),
        "missing_truth_count": missing,
        "query_outcomes_payload_bytes": len(logical),
        "query_outcomes_payload_sha256": hashlib.sha256(logical).hexdigest(),
        "metrics_sha256": hashlib.sha256(metrics_bytes).hexdigest(),
        "input_snapshot_count": len(validation["input_hashes"]),
        "opened_input_count": len(artifact["ledger_role_order"]),
        "gate_statistic": "OBSERVED_RATE_FROM_RAW_COUNTS",
        "declarations": expected_metrics["declarations"],
        "verdict": expected_metrics["verdict"],
    }
    if (
        integrity != expected_integrity
        or (evaluation_root / "integrity.json").read_bytes()
        != canonical_json(expected_integrity)
    ):
        _stop("published integrity is not an exact recomputation")
    ledger = pq.read_table(audit_root / "open_ledger.parquet")
    expected_ledger = pa.schema(
        [
            pa.field(item["name"], pa.type_for_alias(item["type"]), False)
            for item in artifact["ledger_schema"]
        ],
        metadata=None,
    )
    if not ledger.schema.equals(expected_ledger, check_metadata=True):
        _stop("published ledger schema mismatch")
    if ledger.column("role").to_pylist() != artifact["ledger_role_order"]:
        _stop("published ledger role order mismatch")
    ledger_rows = ledger.to_pylist()
    expected_snapshots = validation.get("input_snapshots")
    if (
        type(expected_snapshots) is not dict
        or not set(artifact["ledger_role_order"]).issubset(
            expected_snapshots
        )
    ):
        _stop("independent parent input snapshots are missing")
    for row in ledger_rows:
        role = row["role"]
        expected_projection = DATA_PROJECTIONS.get(
            role, "FULL_JSON_EXACT_KEYSET"
        )
        if (
            row["absolute_path"] != validation["input_paths"][role]
            or row["projection"] != expected_projection
            or row["sha256_before"] != validation["input_hashes"][role]
            or row["sha256_after"] != validation["input_hashes"][role]
            or row["size_bytes_before"]
            != expected_snapshots[role]["size_bytes_before"]
            or row["size_bytes_after"]
            != expected_snapshots[role]["size_bytes_after"]
            or row["sha256_before"]
            != expected_snapshots[role]["sha256_before"]
            or row["sha256_after"]
            != expected_snapshots[role]["sha256_after"]
        ):
            _stop(f"published ledger binding mismatch: {role}")
    expected_declarations = expected_metrics["declarations"]
    expected_provenance = {
        "schema_version": artifact["schema_versions"]["provenance"],
        "evaluator_build_id": evaluator_build_id,
        "attempt_id": attempt_id,
        "git_commit": validation["git_commit"],
        "source_hashes": validation["source_hashes"],
        "plan_sha256": validation["plan_sha256"],
        "lock_sha256": validation["lock_sha256"],
        "input_hashes": validation["input_hashes"],
        "evaluation_spec_sha256": hashlib.sha256(
            canonical_json(validation["evaluation_spec"])
        ).hexdigest(),
        "runtime": validation["runtime"],
        "data_input_count": len(artifact["ledger_role_order"]),
        "evaluation_manifest_sha256": hashlib.sha256(
            (evaluation_root / "manifest.json").read_bytes()
        ).hexdigest(),
        "declarations": expected_declarations,
    }
    if (
        provenance != expected_provenance
        or (audit_root / "provenance.json").read_bytes()
        != canonical_json(expected_provenance)
    ):
        _stop("published provenance binding mismatch")
    if (
        eval_manifest["runtime"] != validation["runtime"]
        or eval_manifest["declarations"] != expected_declarations
        or eval_manifest["verdict"] != expected_metrics["verdict"]
        or integrity["verdict"] != expected_metrics["verdict"]
    ):
        _stop("published declarations/verdict mismatch")


def _promote(source: Path, destination: Path) -> None:
    _ensure_dir(destination.parent)
    _source_parent, source_parent_fd = _openat_anchored(
        source.parent, directory=True
    )
    _destination_parent, destination_parent_fd = _openat_anchored(
        destination.parent, directory=True
    )
    try:
        source_info = os.stat(
            source.name, dir_fd=source_parent_fd, follow_symlinks=False
        )
        if not stat.S_ISDIR(source_info.st_mode):
            _stop("publication source is not a directory")
        if source_info.st_dev != os.fstat(destination_parent_fd).st_dev:
            _stop("publication crosses filesystems")
        libc = ctypes.CDLL(None, use_errno=True)
        if sys.platform == "darwin":
            rename_exclusive = libc.renameatx_np
            rename_exclusive.argtypes = (
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            )
            result = rename_exclusive(
                source_parent_fd,
                os.fsencode(source.name),
                destination_parent_fd,
                os.fsencode(destination.name),
                0x00000004,  # RENAME_EXCL from <sys/stdio.h>
            )
        else:
            rename_exclusive = getattr(libc, "renameat2", None)
            if rename_exclusive is None:
                _stop("atomic exclusive rename is unavailable")
            rename_exclusive.argtypes = (
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            )
            result = rename_exclusive(
                source_parent_fd,
                os.fsencode(source.name),
                destination_parent_fd,
                os.fsencode(destination.name),
                0x00000001,  # RENAME_NOREPLACE
            )
        if result != 0:
            error = ctypes.get_errno()
            if error in {errno.EEXIST, errno.ENOTEMPTY}:
                _stop(f"publication destination already exists: {destination}")
            _stop(f"exclusive publication rename failed: {os.strerror(error)}")
        os.fsync(source_parent_fd)
        if destination_parent_fd != source_parent_fd:
            os.fsync(destination_parent_fd)
    finally:
        os.close(source_parent_fd)
        os.close(destination_parent_fd)


def _freeze(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        os.chmod(path, 0o555 if path.is_dir() else 0o444, follow_symlinks=False)
    os.chmod(root, 0o555, follow_symlinks=False)
    _fsync_dir(root)


def publish_packages(
    plan: Mapping[str, Any],
    attempt_dir: Path,
    evaluator_build_id: str,
    attempt_id: str,
    evaluation_stage: Path,
    audit_stage: Path,
    validation: Mapping[str, Any],
) -> tuple[Path, Path]:
    artifact = plan["artifact_contract"]
    validate_packages(
        evaluation_stage,
        audit_stage,
        evaluator_build_id=evaluator_build_id,
        attempt_id=attempt_id,
        artifact=artifact,
        validation=validation,
    )
    attestation_sha256 = _create_computed_attestation(
        plan,
        attempt_dir,
        evaluator_build_id,
        attempt_id,
        evaluation_stage,
        audit_stage,
        validation,
    )
    last = load_event_chain(attempt_dir, plan["attempt_protocol"])[-1]
    if (
        last["state"],
        last["phase"],
    ) == ("STARTED", "ORACLE_OPEN_COMMITTED"):
        append_event(
            attempt_dir,
            plan["attempt_protocol"],
            state="RECOVERABLE",
            phase="COMPUTED_STAGING_VALID",
            oracle_open_committed=True,
            evaluator_build_id=evaluator_build_id,
            computed_attestation_sha256=attestation_sha256,
            reason_code=None,
        )
    return recover_publication(
        plan,
        attempt_dir,
        evaluator_build_id,
        attempt_id,
        validation,
    ) or _stop("publication recovery returned no result")


def recover_publication(
    plan: Mapping[str, Any],
    attempt_dir: Path,
    evaluator_build_id: str,
    attempt_id: str,
    validation: Mapping[str, Any],
) -> tuple[Path, Path] | None:
    events = load_event_chain(attempt_dir, plan["attempt_protocol"])
    state = recover_state_cache(attempt_dir, plan["attempt_protocol"], events)
    final_eval = Path(plan["outputs"]["evaluation_root"]) / evaluator_build_id
    final_audit = Path(plan["outputs"]["audit_root"]) / evaluator_build_id
    pending_eval = Path(plan["outputs"]["evaluation_root"]) / (
        f".pending-{evaluator_build_id}-{attempt_id}"
    )
    pending_audit = Path(plan["outputs"]["audit_root"]) / (
        f".pending-{evaluator_build_id}-{attempt_id}"
    )
    stage_base = Path(plan["outputs"]["temp_root"]) / attempt_id
    stage_eval = stage_base / "evaluation.stage"
    stage_audit = stage_base / "audit.stage"
    artifact = plan["artifact_contract"]
    protocol = plan["attempt_protocol"]
    has_attestation = _validate_attempt_tree(attempt_dir, protocol)

    def present(path: Path) -> bool:
        return path.exists() or path.is_symlink()

    paths = (stage_eval, stage_audit, pending_eval, pending_audit, final_eval, final_audit)
    if not state["oracle_open_committed"]:
        if has_attestation:
            _stop("computed attestation exists before oracle commit")
        if any(present(path) for path in paths):
            _stop("publication artifact exists before oracle commit")
        return None
    attestation_sha256 = state["computed_attestation_sha256"]
    if (
        not has_attestation
        or type(attestation_sha256) is not str
        or state["phase"] == "ORACLE_OPEN_COMMITTED"
    ):
        _stop("post-oracle recovery lacks committed computed attestation")

    def validate_pair(
        evaluation_path: Path, audit_path: Path
    ) -> dict[str, Any]:
        _attestation, _digest, recovery_validation = (
            _validate_computed_attestation(
                plan,
                attempt_dir,
                evaluator_build_id,
                attempt_id,
                evaluation_path,
                audit_path,
                validation,
                attestation_sha256,
            )
        )
        validate_packages(
            evaluation_path,
            audit_path,
            evaluator_build_id=evaluator_build_id,
            attempt_id=attempt_id,
            artifact=artifact,
            validation=recovery_validation,
        )
        return recovery_validation

    for _ in range(12):
        state = recover_state_cache(attempt_dir, protocol)
        phase = state["phase"]
        have = tuple(present(path) for path in paths)
        se, sa, pe, pa, fe, fa = have
        if fe and not fa:
            _stop("final evaluation exists without final audit")
        if fe and fa:
            if any((se, sa, pe, pa)):
                _stop("final packages coexist with stale publication roots")
            validate_pair(final_eval, final_audit)
            _freeze(final_audit)
            _freeze(final_eval)
            if phase == "AUDIT_FINAL":
                append_event(
                    attempt_dir,
                    protocol,
                    state="RECOVERABLE",
                    phase="EVALUATION_FINAL",
                    oracle_open_committed=True,
                    evaluator_build_id=evaluator_build_id,
                    reason_code="RECOVERED_RENAME_BEFORE_EVENT",
                )
                continue
            if phase == "EVALUATION_FINAL":
                append_event(
                    attempt_dir,
                    protocol,
                    state="FINAL",
                    phase="TERMINAL",
                    oracle_open_committed=True,
                    evaluator_build_id=evaluator_build_id,
                    reason_code="RECOVERED_FINAL_STATE",
                )
                continue
            if state["state"] == "FINAL":
                return final_eval, final_audit
            _stop("final packages conflict with authoritative phase")
        if fa and pe and not any((se, sa, pa, fe)):
            validate_pair(pending_eval, final_audit)
            _freeze(final_audit)
            if phase == "PENDING_BOTH_VALID":
                append_event(
                    attempt_dir,
                    protocol,
                    state="RECOVERABLE",
                    phase="AUDIT_FINAL",
                    oracle_open_committed=True,
                    evaluator_build_id=evaluator_build_id,
                    reason_code="RECOVERED_RENAME_BEFORE_EVENT",
                )
                continue
            if phase != "AUDIT_FINAL":
                _stop("final audit conflicts with authoritative phase")
            _promote(pending_eval, final_eval)
            _freeze(final_eval)
            continue
        if pe and pa and not any((se, sa, fe, fa)):
            validate_pair(pending_eval, pending_audit)
            if phase == "COMPUTED_STAGING_VALID":
                append_event(
                    attempt_dir,
                    protocol,
                    state="RECOVERABLE",
                    phase="PENDING_BOTH_VALID",
                    oracle_open_committed=True,
                    evaluator_build_id=evaluator_build_id,
                    reason_code="RECOVERED_RENAME_BEFORE_EVENT",
                )
                continue
            if phase != "PENDING_BOTH_VALID":
                _stop("pending packages conflict with authoritative phase")
            _promote(pending_audit, final_audit)
            _freeze(final_audit)
            continue
        if se and pa and not any((sa, pe, fe, fa)):
            validate_pair(stage_eval, pending_audit)
            if phase != "COMPUTED_STAGING_VALID":
                _stop("partial pending promotion conflicts with phase")
            _promote(stage_eval, pending_eval)
            continue
        if se and sa and not any((pe, pa, fe, fa)):
            validate_pair(stage_eval, stage_audit)
            if phase == "ORACLE_OPEN_COMMITTED":
                append_event(
                    attempt_dir,
                    protocol,
                    state="RECOVERABLE",
                    phase="COMPUTED_STAGING_VALID",
                    oracle_open_committed=True,
                    evaluator_build_id=evaluator_build_id,
                    reason_code="RECOVERED_STAGE_BEFORE_EVENT",
                )
                continue
            if phase != "COMPUTED_STAGING_VALID":
                _stop("staging packages conflict with authoritative phase")
            _promote(stage_audit, pending_audit)
            continue
        _stop("oracle commit has no complete promotion-only recovery path")
    _stop("publication recovery did not converge")


def mark_attempt_stopped(
    attempt_dir: Path,
    protocol: Mapping[str, Any],
    reason_code: str,
) -> None:
    events = load_event_chain(attempt_dir, protocol)
    last = events[-1]
    if last["state"] in {"FINAL", "STOPPED"}:
        return
    append_event(
        attempt_dir,
        protocol,
        state="STOPPED",
        phase="TERMINAL",
        oracle_open_committed=last["oracle_open_committed"],
        evaluator_build_id=last["evaluator_build_id"],
        reason_code=re.sub(r"[^A-Z0-9_]", "_", reason_code.upper())[:128],
    )


def _sb(value: str) -> str:
    return json.dumps(value)


def render_profile(
    template: str,
    *,
    allowed_files: Sequence[Path],
    staging_root: Path,
    forbidden_roots: Sequence[Path],
    system_read_roots: Sequence[Path] = (
        Path("/System"),
        Path("/usr"),
        Path("/opt/homebrew"),
    ),
    device_read_paths: Sequence[Path] = (
        Path("/dev/null"),
        Path("/dev/urandom"),
        Path("/dev/fd"),
    ),
) -> str:
    markers = {
        "@@ALLOWED_INPUT_RULES@@",
        "@@ANCESTOR_METADATA_RULES@@",
        "@@EXPLICIT_DENY_RULES@@",
        "@@SYSTEM_READ_RULES@@",
        "@@DEVICE_READ_RULES@@",
    }
    if any(marker not in template for marker in markers):
        _stop("sandbox template marker missing")
    allowed = sorted({path.absolute() for path in allowed_files}, key=lambda p: str(p).encode())
    read_rules = "\n".join(f"  (literal {_sb(str(path))})" for path in allowed)
    ancestors = {
        str(parent)
        for path in [*allowed, staging_root.absolute()]
        for parent in [path, *path.parents]
    }
    metadata = "\n".join(
        f"  (literal {_sb(path)})"
        for path in sorted(ancestors, key=lambda value: value.encode())
    )
    denies = "\n".join(
        f"(deny file-read* file-write* (subpath {_sb(str(path.absolute()))}))"
        for path in sorted(set(forbidden_roots), key=lambda p: str(p).encode())
    )
    system_rules = "\n".join(
        f"  (literal {_sb(str(path.absolute()))})\n"
        f"  (subpath {_sb(str(path.absolute()))})"
        for path in sorted(set(system_read_roots), key=lambda p: str(p).encode())
    )
    device_rules = "\n".join(
        f"  (literal {_sb(str(path.absolute()))})\n"
        f"  (subpath {_sb(str(path.absolute()))})"
        for path in sorted(set(device_read_paths), key=lambda p: str(p).encode())
    )
    rendered = (
        template.replace("@@ALLOWED_INPUT_RULES@@", read_rules)
        .replace("@@ANCESTOR_METADATA_RULES@@", metadata)
        .replace("@@EXPLICIT_DENY_RULES@@", denies)
        .replace("@@SYSTEM_READ_RULES@@", system_rules)
        .replace("@@DEVICE_READ_RULES@@", device_rules)
    )
    if "@@" in rendered:
        _stop("unresolved sandbox marker")
    return rendered.rstrip() + "\n"


def _open_roles(
    plan: Mapping[str, Any],
    lock: Mapping[str, Any],
    roles: Sequence[str],
) -> tuple[dict[str, int], dict[str, dict[str, Any]]]:
    descriptors: dict[str, int] = {}
    snapshots: dict[str, dict[str, Any]] = {}
    try:
        for role in roles:
            fd, snapshot = _open_locked(
                Path(lock["input_paths"][role]),
                lock["input_hashes"][role],
                plan["max_rss_bytes"],
            )
            descriptors[role] = fd
            snapshots[role] = snapshot
        return descriptors, snapshots
    except Exception:
        for fd in descriptors.values():
            os.close(fd)
        raise


def _open_oracle_roles_after_commit(
    plan: Mapping[str, Any],
    lock: Mapping[str, Any],
    attempt_dir: Path,
) -> tuple[dict[str, int], dict[str, dict[str, Any]]]:
    events = load_event_chain(attempt_dir, plan["attempt_protocol"])
    last = events[-1]
    if (
        last["oracle_open_committed"] is not True
        or not any(
            event["phase"] == "ORACLE_OPEN_COMMITTED"
            and event["oracle_open_committed"] is True
            for event in events
        )
    ):
        _stop("oracle open attempted before durable commit event")
    if tuple(
        plan["attempt_protocol"]["oracle_roles_opened_only_after_commit"]
    ) != ORACLE_ROLE_ORDER:
        _stop("oracle role order mismatch")
    return _open_roles(plan, lock, ORACLE_ROLE_ORDER)


def _resnapshot(
    descriptors: Mapping[str, int],
    snapshots: Mapping[str, Mapping[str, Any]],
    limit: int,
) -> None:
    for role, fd in descriptors.items():
        if not _same_snapshot(_snapshot_fd(fd, limit), snapshots[role]):
            _stop(f"input changed before oracle commit: {role}")


def _worker_spec(
    plan: Mapping[str, Any],
    lock: Mapping[str, Any],
    evaluator_build_id: str,
    attempt_id: str,
    data_descriptors: Mapping[str, int],
    evaluation_stage: Path,
    audit_stage: Path,
    plan_sha256: str = "",
    lock_sha256: str = "",
) -> dict[str, Any]:
    spec = {
        "schema_version": WORKER_SPEC_SCHEMA,
        "evaluator_build_id": evaluator_build_id,
        "attempt_id": attempt_id,
        "worker_build_id": plan["prerequisite"]["worker_build_id"],
        "oracle_build_id": plan["oracle"]["build_id"],
        "parity_build_id": plan["prerequisite"]["parity_build_id"],
        "input_fds": {role: fd for role, fd in data_descriptors.items()},
        "input_paths": {
            role: lock["input_paths"][role]
            for role in plan["artifact_contract"]["ledger_role_order"]
        },
        "input_hashes": lock["input_hashes"],
        "git_commit": lock["git_commit"],
        "source_hashes": lock["source_hashes"],
        "plan_sha256": plan_sha256,
        "lock_sha256": lock_sha256,
        "evaluation_spec": plan["evaluation_spec"],
        "artifact_contract": plan["artifact_contract"],
        "runtime": plan["runtime"],
        "evaluation_stage": str(evaluation_stage),
        "audit_stage": str(audit_stage),
        "max_rss_bytes": plan["max_rss_bytes"],
    }
    if set(spec) != WORKER_SPEC_KEYS:
        _stop("internal worker spec keyset mismatch")
    return spec


def _package_validation(
    plan: Mapping[str, Any],
    lock: Mapping[str, Any],
    evaluator_build_id: str,
    attempt_id: str,
    plan_sha256: str,
    lock_sha256: str,
) -> dict[str, Any]:
    return {
        "evaluator_build_id": evaluator_build_id,
        "attempt_id": attempt_id,
        "worker_build_id": plan["prerequisite"]["worker_build_id"],
        "oracle_build_id": plan["oracle"]["build_id"],
        "parity_build_id": plan["prerequisite"]["parity_build_id"],
        "input_paths": lock["input_paths"],
        "input_hashes": lock["input_hashes"],
        "git_commit": lock["git_commit"],
        "source_hashes": lock["source_hashes"],
        "plan_sha256": plan_sha256,
        "lock_sha256": lock_sha256,
        "evaluation_spec": plan["evaluation_spec"],
        "runtime": plan["runtime"],
    }


def _computed_input_snapshots(
    plan: Mapping[str, Any],
    lock: Mapping[str, Any],
    before: Mapping[str, Mapping[str, Any]],
    descriptors: Mapping[str, int],
) -> dict[str, Any]:
    protocol = plan["attempt_protocol"]
    roles = protocol["computed_input_role_order"]
    if set(before) != set(roles) or set(descriptors) != set(roles):
        _stop("computed input snapshot role closure mismatch")
    result: dict[str, Any] = {}
    for role in roles:
        after = _snapshot_fd(descriptors[role], plan["max_rss_bytes"])
        if not _same_snapshot(before[role], after):
            _stop(f"input mutated before computed attestation: {role}")
        result[role] = {
            "absolute_path": lock["input_paths"][role],
            "projection": protocol["computed_input_projections"][role],
            "size_bytes_before": before[role]["size_bytes"],
            "sha256_before": before[role]["sha256"],
            "size_bytes_after": after["size_bytes"],
            "sha256_after": after["sha256"],
        }
    return result


def _tree_sha256(root: Path, names: Sequence[str]) -> str:
    _exact_tree(root, names)
    records = {
        name: {
            "size_bytes": (root / name).stat().st_size,
            "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(),
        }
        for name in names
    }
    return hashlib.sha256(canonical_json(records)).hexdigest()


def _validate_computed_attestation(
    plan: Mapping[str, Any],
    attempt_dir: Path,
    evaluator_build_id: str,
    attempt_id: str,
    evaluation_root: Path,
    audit_root: Path,
    validation: Mapping[str, Any],
    expected_sha256: str | None,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    protocol = plan["attempt_protocol"]
    path = attempt_dir / protocol["computed_attestation_filename"]
    payload = _read_small_path(path)
    attestation = parse_json(payload, "computed attestation")
    _require_keys(
        attestation,
        protocol["computed_attestation_keyset"],
        "computed attestation",
    )
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        _stop("computed attestation journal hash mismatch")
    if (
        attestation["schema_version"]
        != protocol["schema_versions"]["computed_attestation"]
        or attestation["measurement_slot_id"] != attempt_dir.name
        or attestation["attempt_id"] != attempt_id
        or attestation["evaluator_build_id"] != evaluator_build_id
        or attestation["plan_sha256"] != validation["plan_sha256"]
        or attestation["lock_sha256"] != validation["lock_sha256"]
    ):
        _stop("computed attestation identity mismatch")
    snapshots = attestation["input_snapshots"]
    roles = protocol["computed_input_role_order"]
    if type(snapshots) is not dict or set(snapshots) != set(roles):
        _stop("computed attestation input role mismatch")
    for role in roles:
        record = snapshots[role]
        _require_keys(
            record,
            protocol["computed_input_snapshot_keyset"],
            f"computed input snapshot {role}",
        )
        if (
            record["absolute_path"] != validation["input_paths"][role]
            or record["projection"]
            != protocol["computed_input_projections"][role]
            or record["sha256_before"] != validation["input_hashes"][role]
            or record["sha256_after"] != validation["input_hashes"][role]
            or record["size_bytes_before"] != record["size_bytes_after"]
        ):
            _stop(f"computed input snapshot mismatch: {role}")
    evaluation_manifest_sha = hashlib.sha256(
        (evaluation_root / "manifest.json").read_bytes()
    ).hexdigest()
    audit_manifest_sha = hashlib.sha256(
        (audit_root / "manifest.json").read_bytes()
    ).hexdigest()
    evaluation_tree_sha = _tree_sha256(
        evaluation_root, plan["outputs"]["runtime_files"]
    )
    audit_tree_sha = _tree_sha256(
        audit_root, plan["outputs"]["audit_files"]
    )
    if (
        attestation["evaluation_manifest_sha256"]
        != evaluation_manifest_sha
        or attestation["audit_manifest_sha256"] != audit_manifest_sha
        or attestation["evaluation_tree_sha256"] != evaluation_tree_sha
        or attestation["audit_tree_sha256"] != audit_tree_sha
    ):
        _stop("computed attestation manifest/tree mismatch")
    recovery_validation = dict(validation)
    recovery_validation["input_snapshots"] = snapshots
    return attestation, digest, recovery_validation


def _create_computed_attestation(
    plan: Mapping[str, Any],
    attempt_dir: Path,
    evaluator_build_id: str,
    attempt_id: str,
    evaluation_root: Path,
    audit_root: Path,
    validation: Mapping[str, Any],
) -> str:
    protocol = plan["attempt_protocol"]
    snapshots = validation.get("input_snapshots")
    if (
        type(snapshots) is not dict
        or set(snapshots) != set(protocol["computed_input_role_order"])
    ):
        _stop("parent snapshots missing for computed attestation")
    payload = {
        "schema_version": protocol["schema_versions"][
            "computed_attestation"
        ],
        "measurement_slot_id": attempt_dir.name,
        "attempt_id": attempt_id,
        "evaluator_build_id": evaluator_build_id,
        "plan_sha256": validation["plan_sha256"],
        "lock_sha256": validation["lock_sha256"],
        "input_snapshots": snapshots,
        "evaluation_manifest_sha256": hashlib.sha256(
            (evaluation_root / "manifest.json").read_bytes()
        ).hexdigest(),
        "audit_manifest_sha256": hashlib.sha256(
            (audit_root / "manifest.json").read_bytes()
        ).hexdigest(),
        "evaluation_tree_sha256": _tree_sha256(
            evaluation_root, plan["outputs"]["runtime_files"]
        ),
        "audit_tree_sha256": _tree_sha256(
            audit_root, plan["outputs"]["audit_files"]
        ),
    }
    _require_keys(
        payload,
        protocol["computed_attestation_keyset"],
        "computed attestation",
    )
    encoded = canonical_json(payload)
    _write_exclusive(
        attempt_dir / protocol["computed_attestation_filename"],
        encoded,
        0o400,
    )
    _validate_attempt_tree(attempt_dir, protocol)
    return hashlib.sha256(encoded).hexdigest()


def _copy_fd_exclusive(fd: int, destination: Path, limit: int, mode: int) -> int:
    payload = _read_fd(fd, limit)
    _write_exclusive(destination, payload, mode)
    _absolute, copied_fd = _openat_anchored(destination)
    if _snapshot_fd(copied_fd, limit)["sha256"] != hashlib.sha256(payload).hexdigest():
        os.close(copied_fd)
        _stop(f"sealed execution copy mismatch: {destination.name}")
    return copied_fd


def _prepare_worker_execution(
    plan: Mapping[str, Any],
    lock: Mapping[str, Any],
    nonoracle_descriptors: Mapping[str, int],
    stage_base: Path,
) -> dict[str, Any]:
    """Seal every path executed/read after the oracle event in a private root."""
    _ensure_dir(stage_base)
    execution_root = stage_base / f"execution.private-{os.getpid()}-{os.urandom(8).hex()}"
    execution_root.mkdir(mode=0o700)
    limit = plan["max_rss_bytes"]
    source_path = Path(__file__).resolve()
    source_role = str(source_path.relative_to(source_path.parents[1]))
    source_fd, source_snapshot = _open_locked(
        source_path, lock["source_hashes"][source_role], limit
    )
    profile_source = source_path.parents[1] / PROFILE_PATH
    profile_fd, _profile_snapshot = _open_locked(
        profile_source,
        lock["source_hashes"][str(PROFILE_PATH)],
        limit,
    )
    sealed_fds: list[int] = [source_fd, profile_fd]
    try:
        # Apple's platform-signed sandbox-exec cannot be copied (AMFI kills a
        # relocated platform binary). Keep its already-verified descriptor
        # open and execute the root-owned canonical path; every mutable worker
        # component remains a private sealed copy.
        sandbox_path = Path(lock["sandbox"]["executable"])
        python_path = execution_root / "python-app"
        library_root = execution_root / "framework"
        library_root.mkdir(mode=0o700)
        library_path = library_root / "Python"
        evaluator_path = execution_root / "evaluator.py"
        sealed_fds.extend(
            [
                _copy_fd_exclusive(
                    nonoracle_descriptors["python_framework_app"],
                    python_path,
                    limit,
                    0o500,
                ),
                _copy_fd_exclusive(
                    nonoracle_descriptors["python_framework_library"],
                    library_path,
                    limit,
                    0o400,
                ),
                _copy_fd_exclusive(source_fd, evaluator_path, limit, 0o400),
            ]
        )
        template = _read_fd(profile_fd, limit).decode("utf-8")
        sandbox = lock["sandbox"]
        profile = render_profile(
            template,
            allowed_files=[
                sandbox_path,
                python_path,
                library_path,
                evaluator_path,
            ],
            staging_root=stage_base,
            forbidden_roots=[
                Path(value) for value in plan["security"]["forbidden_roots"]
            ],
            system_read_roots=[
                Path(value) for value in sandbox["system_read_roots"]
            ],
            device_read_paths=[
                Path(value) for value in sandbox["device_read_paths"]
            ],
        )
        profile_path = execution_root / "evaluator.sb"
        _write_exclusive(profile_path, profile.encode("utf-8"), 0o400)
        _profile_copy_path, profile_copy_fd = _openat_anchored(profile_path)
        sealed_fds.append(profile_copy_fd)
        _fsync_dir(execution_root)
        _freeze(execution_root)
        os.chmod(python_path, 0o500, follow_symlinks=False)
        os.chmod(library_root, 0o500, follow_symlinks=False)
        os.chmod(execution_root, 0o500, follow_symlinks=False)
        sealed_files = {
            str(path): {
                "fd": fd,
                "snapshot": _snapshot_fd(fd, limit),
            }
            for path, fd in (
                (python_path, sealed_fds[2]),
                (library_path, sealed_fds[3]),
                (evaluator_path, sealed_fds[4]),
                (profile_path, profile_copy_fd),
            )
        }
        _fsync_dir(stage_base)
        return {
            "root": execution_root,
            "sandbox": sandbox_path,
            "python": python_path,
            "library": library_path,
            "source": evaluator_path,
            "profile": profile_path,
            "fds": sealed_fds,
            "source_snapshot": source_snapshot,
            "sandbox_fd": nonoracle_descriptors["sandbox_executable"],
            "sandbox_snapshot": _snapshot_fd(
                nonoracle_descriptors["sandbox_executable"], limit
            ),
            "sealed_files": sealed_files,
        }
    except Exception:
        for fd in sealed_fds:
            try:
                os.close(fd)
            except OSError:
                pass
        raise


def _run_with_rss_ceiling(
    command: Sequence[str],
    *,
    pass_fds: Sequence[int],
    env: Mapping[str, str],
    limit: int,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        list(command),
        pass_fds=tuple(pass_fds),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=dict(env),
    )
    peak = 0
    while True:
        waited_pid, status, usage = os.wait4(process.pid, os.WNOHANG)
        if waited_pid == process.pid:
            process.returncode = os.waitstatus_to_exitcode(status)
            final_peak = int(
                usage.ru_maxrss
                if sys.platform == "darwin"
                else usage.ru_maxrss * 1024
            )
            peak = max(peak, final_peak)
            stdout = process.stdout.read() if process.stdout else ""
            stderr = process.stderr.read() if process.stderr else ""
            if peak > limit:
                _stop(f"sandbox worker RSS limit exceeded: {peak} > {limit}")
            return subprocess.CompletedProcess(
                list(command), process.returncode, stdout, stderr
            )
        current, lifetime_peak = _process_resident_bytes(process.pid)
        peak = max(peak, current, lifetime_peak)
        if peak > limit:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            try:
                os.wait4(process.pid, 0)
            except ChildProcessError:
                pass
            process.returncode = -9
            _stop(f"sandbox worker RSS limit exceeded: {peak} > {limit}")
        # A short native proc_pid_rusage poll plus wait4's final high-water
        # mark closes both sustained and sub-poll allocation spikes.
        import time

        time.sleep(0.005)


class _RusageInfoV4(ctypes.Structure):
    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        *[
            (name, ctypes.c_uint64)
            for name in (
                "ri_user_time",
                "ri_system_time",
                "ri_pkg_idle_wkups",
                "ri_interrupt_wkups",
                "ri_pageins",
                "ri_wired_size",
                "ri_resident_size",
                "ri_phys_footprint",
                "ri_proc_start_abstime",
                "ri_proc_exit_abstime",
                "ri_child_user_time",
                "ri_child_system_time",
                "ri_child_pkg_idle_wkups",
                "ri_child_interrupt_wkups",
                "ri_child_pageins",
                "ri_child_elapsed_abstime",
                "ri_diskio_bytesread",
                "ri_diskio_byteswritten",
                "ri_cpu_time_qos_default",
                "ri_cpu_time_qos_maintenance",
                "ri_cpu_time_qos_background",
                "ri_cpu_time_qos_utility",
                "ri_cpu_time_qos_legacy",
                "ri_cpu_time_qos_user_initiated",
                "ri_cpu_time_qos_user_interactive",
                "ri_billed_system_time",
                "ri_serviced_system_time",
                "ri_logical_writes",
                "ri_lifetime_max_phys_footprint",
                "ri_instructions",
                "ri_cycles",
                "ri_billed_energy",
                "ri_serviced_energy",
                "ri_interval_max_phys_footprint",
                "ri_runnable_time",
            )
        ],
    ]


def _process_resident_bytes(pid: int) -> tuple[int, int]:
    if sys.platform == "darwin":
        info = _RusageInfoV4()
        libc = ctypes.CDLL(None, use_errno=True)
        function = libc.proc_pid_rusage
        function.argtypes = (
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(_RusageInfoV4),
        )
        if function(pid, 4, ctypes.byref(info)) != 0:
            _stop("proc_pid_rusage failed for live worker")
        return int(info.ri_resident_size), int(
            info.ri_lifetime_max_phys_footprint
        )
    status = Path(f"/proc/{pid}/status")
    values: dict[str, int] = {}
    for line in status.read_text().splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            key, raw, _unit = line.split()
            values[key.rstrip(":")] = int(raw) * 1024
    if set(values) != {"VmRSS", "VmHWM"}:
        _stop("cannot read live worker RSS/HWM")
    return values["VmRSS"], values["VmHWM"]


def _verify_sealed_execution(
    execution: Mapping[str, Any], limit: int
) -> None:
    for raw_path, record in execution["sealed_files"].items():
        path = Path(raw_path)
        _absolute, path_fd = _openat_anchored(path)
        try:
            path_snapshot = _snapshot_fd(path_fd, limit)
        finally:
            os.close(path_fd)
        if (
            not _same_snapshot(path_snapshot, record["snapshot"])
            or not _same_snapshot(
                _snapshot_fd(record["fd"], limit), record["snapshot"]
            )
        ):
            _stop(f"sealed execution substitution detected: {path.name}")
    sandbox_path = Path(execution["sandbox"])
    _sandbox_absolute, sandbox_path_fd = _openat_anchored(sandbox_path)
    try:
        path_snapshot = _snapshot_fd(sandbox_path_fd, limit)
        held_snapshot = _snapshot_fd(execution["sandbox_fd"], limit)
        info = os.fstat(sandbox_path_fd)
    finally:
        os.close(sandbox_path_fd)
    if (
        not _same_snapshot(path_snapshot, execution["sandbox_snapshot"])
        or not _same_snapshot(held_snapshot, execution["sandbox_snapshot"])
        or info.st_uid != 0
        or info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    ):
        _stop("sandbox binary snapshot/path ownership mismatch")


def _start_worker_pre_oracle(
    plan: Mapping[str, Any],
    lock: Mapping[str, Any],
    spec: Mapping[str, Any],
    descriptors: Mapping[str, int],
    stage_base: Path,
    execution: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    # The attempt staging root is stable, but every pre-oracle launch owns a
    # fresh private spec path. A process loss after READY_PRE_ORACLE must leave
    # its evidence untouched without making the same attempt unrestartable.
    spec_path = stage_base / (
        f"worker_spec.private-{os.getpid()}-{os.urandom(8).hex()}.json"
    )
    _write_exclusive(spec_path, canonical_json(spec))
    _absolute, spec_fd = _openat_anchored(spec_path)
    owns_execution = execution is None
    if execution is None:
        # Synthetic unit entry point: prepare the same private sealed runtime
        # from the already-open descriptors.
        execution = _prepare_worker_execution(
            plan, lock, descriptors, stage_base
        )
    profile_path = Path(execution["profile"])
    sandbox_path = Path(execution["sandbox"])
    python_path = Path(execution["python"])
    source_path = Path(execution["source"])
    parent_control, child_control = socket.socketpair(
        socket.AF_UNIX, socket.SOCK_DGRAM
    )
    command = [
        str(sandbox_path),
        "-D",
        f"PYTHON_EXECUTABLE={python_path}",
        "-D",
        f"STAGING_ROOT={stage_base}",
        "-f",
        str(profile_path),
        str(python_path),
        "-B",
        str(source_path),
        "--worker-spec-fd",
        str(spec_fd),
        "--worker-control-fd",
        str(child_control.fileno()),
    ]
    pass_fds = tuple(
        dict.fromkeys(
            [
                spec_fd,
                child_control.fileno(),
                *descriptors.values(),
                *execution.get("fds", []),
            ]
        )
    )
    _verify_sealed_execution(execution, spec["max_rss_bytes"])
    process = subprocess.Popen(
        command,
        pass_fds=pass_fds,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TMPDIR": str(stage_base),
            "DYLD_LIBRARY_PATH": str(Path(execution["library"]).parent),
        },
    )
    child_control.close()
    os.close(spec_fd)
    peak = 0
    parent_control.settimeout(0.005)
    try:
        while True:
            waited_pid, status, usage = os.wait4(process.pid, os.WNOHANG)
            if waited_pid == process.pid:
                process.returncode = os.waitstatus_to_exitcode(status)
                stderr = process.stderr.read() if process.stderr else ""
                _stop(
                    "sandbox worker exited before oracle commit "
                    f"({process.returncode}): {stderr[-1000:]}"
                )
            current, lifetime_peak = _process_resident_bytes(process.pid)
            peak = max(peak, current, lifetime_peak)
            if peak > spec["max_rss_bytes"]:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                try:
                    os.wait4(process.pid, 0)
                except ChildProcessError:
                    pass
                process.returncode = -9
                _stop(
                    f"sandbox worker RSS limit exceeded: "
                    f"{peak} > {spec['max_rss_bytes']}"
                )
            try:
                ready_payload = parent_control.recv(MAX_JSON_BYTES)
                break
            except TimeoutError:
                continue
        ready = parse_json(ready_payload, "worker pre-oracle READY")
        if ready != {
            "schema_version": WORKER_CONTROL_SCHEMA,
            "status": "READY_PRE_ORACLE",
            "evaluator_build_id": spec["evaluator_build_id"],
            "attempt_id": spec["attempt_id"],
        }:
            _stop("worker pre-oracle READY mismatch")
        return {
            "process": process,
            "control": parent_control,
            "peak": peak,
            "command": command,
            "limit": spec["max_rss_bytes"],
            "spec": spec,
            "owns_execution": owns_execution,
            "execution": execution,
        }
    except Exception:
        parent_control.close()
        if process.returncode is None:
            process.kill()
            try:
                os.wait4(process.pid, 0)
            except ChildProcessError:
                pass
            process.returncode = -9
        if owns_execution:
            for fd in execution.get("fds", []):
                try:
                    os.close(fd)
                except OSError:
                    pass
        raise


def _send_oracle_fds(
    handle: Mapping[str, Any],
    roles: Sequence[str],
    descriptors: Mapping[str, int],
) -> None:
    message = {
        "schema_version": WORKER_CONTROL_SCHEMA,
        "command": "ORACLE_FDS_COMMITTED",
        "evaluator_build_id": handle["spec"]["evaluator_build_id"],
        "attempt_id": handle["spec"]["attempt_id"],
        "roles": list(roles),
    }
    fd_array = array.array("i", [descriptors[role] for role in roles])
    written = handle["control"].sendmsg(
        [canonical_json(message)],
        [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fd_array)],
    )
    if written != len(canonical_json(message)):
        _stop("partial oracle FD control send")
    handle["control"].close()


def _finish_worker(handle: Mapping[str, Any]) -> None:
    process = handle["process"]
    peak = int(handle["peak"])
    limit = int(handle["limit"])
    try:
        while True:
            waited_pid, status, usage = os.wait4(process.pid, os.WNOHANG)
            if waited_pid == process.pid:
                process.returncode = os.waitstatus_to_exitcode(status)
                final_peak = int(
                    usage.ru_maxrss
                    if sys.platform == "darwin"
                    else usage.ru_maxrss * 1024
                )
                peak = max(peak, final_peak)
                stdout = process.stdout.read() if process.stdout else ""
                stderr = process.stderr.read() if process.stderr else ""
                break
            current, lifetime_peak = _process_resident_bytes(process.pid)
            peak = max(peak, current, lifetime_peak)
            if peak > limit:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                try:
                    os.wait4(process.pid, 0)
                except ChildProcessError:
                    pass
                process.returncode = -9
                _stop(f"sandbox worker RSS limit exceeded: {peak} > {limit}")
            import time

            time.sleep(0.005)
    finally:
        if handle["owns_execution"]:
            for fd in handle["execution"].get("fds", []):
                try:
                    os.close(fd)
                except OSError:
                    pass
    if peak > limit:
        _stop(f"sandbox worker RSS limit exceeded: {peak} > {limit}")
    if process.returncode != 0:
        _stop(
            f"sandbox worker failed ({process.returncode}): "
            f"{stderr[-1000:]}"
        )
    stdout = parse_json(stdout.encode(), "worker stdout")
    if (
        set(stdout) != {"schema_version", "evaluator_build_id", "status"}
        or stdout["schema_version"] != WORKER_STDOUT_SCHEMA
        or stdout["evaluator_build_id"]
        != handle["spec"]["evaluator_build_id"]
        or stdout["status"] != "STAGED"
    ):
        _stop("worker stdout mismatch")


def _abort_worker(handle: Mapping[str, Any]) -> None:
    try:
        handle["control"].close()
    except OSError:
        pass
    process = handle["process"]
    if process.returncode is None:
        process.kill()
        try:
            os.wait4(process.pid, 0)
        except ChildProcessError:
            pass
        process.returncode = -9


def _invoke_worker(
    plan: Mapping[str, Any],
    lock: Mapping[str, Any],
    spec: Mapping[str, Any],
    descriptors: Mapping[str, int],
    stage_base: Path,
    execution: Mapping[str, Any] | None = None,
) -> None:
    oracle_roles = ORACLE_ROLE_ORDER
    pre_descriptors = {
        role: fd for role, fd in descriptors.items() if role not in oracle_roles
    }
    pre_spec = dict(spec)
    pre_spec["input_fds"] = dict(pre_descriptors)
    handle = _start_worker_pre_oracle(
        plan, lock, pre_spec, pre_descriptors, stage_base, execution
    )
    try:
        _send_oracle_fds(handle, oracle_roles, descriptors)
        _finish_worker(handle)
    except Exception:
        _abort_worker(handle)
        raise


def run_evaluation(plan_path: Path, lock_path: Path) -> tuple[Path, Path]:
    repo = Path(__file__).resolve().parents[1]
    if plan_path.absolute() != (repo / PLAN_PATH).absolute():
        _stop("only canonical plan may execute")
    if lock_path.absolute() != (repo / LOCK_PATH).absolute():
        _stop("only canonical lock may execute")
    plan_bytes = _read_small_path(plan_path)
    lock_bytes = _read_small_path(lock_path)
    plan = parse_json(plan_bytes, "plan")
    lock = parse_json(lock_bytes, "lock")
    validate_plan(plan)
    validate_lock(plan, lock, repo, hashlib.sha256(plan_bytes).hexdigest())
    if _runtime() != plan["runtime"]:
        _stop("runtime mismatch")
    plan_sha = hashlib.sha256(plan_bytes).hexdigest()
    lock_sha = hashlib.sha256(lock_bytes).hexdigest()
    evaluator_build_id, _payload = _identity(plan, lock, plan_sha, lock_sha)
    slot_id, _attempt_id, _attempt_payload = _attempt_identities(
        plan, lock, plan_sha, lock_sha
    )
    with _exclusive_slot_lock(
        Path(plan["outputs"]["attempt_root"]), slot_id
    ) as slot_lock:
        _verify_slot_lock(slot_lock)
        attempt_dir, _slot_id, attempt_id = ensure_receipt(
            plan, lock, plan_sha, lock_sha
        )
        _verify_slot_lock(slot_lock)
        protocol = plan["attempt_protocol"]
        validation = _package_validation(
            plan,
            lock,
            evaluator_build_id,
            attempt_id,
            plan_sha,
            lock_sha,
        )
        try:
            recovered = recover_publication(
                plan,
                attempt_dir,
                evaluator_build_id,
                attempt_id,
                validation,
            )
            if recovered is not None:
                return recovered
        except EvaluationStopped:
            try:
                mark_attempt_stopped(
                    attempt_dir, protocol, "RECOVERY_VALIDATION_FAILED"
                )
            except (EvaluationStopped, OSError):
                pass
            raise
        nonoracle_fds, nonoracle_snapshots = _open_roles(
            plan, lock, protocol["pre_oracle_revalidation_roles"]
        )
        oracle_fds: dict[str, int] = {}
        execution: dict[str, Any] | None = None
        worker_handle: dict[str, Any] | None = None
        try:
            stage_base = Path(plan["outputs"]["temp_root"]) / attempt_id
            execution = _prepare_worker_execution(
                plan, lock, nonoracle_fds, stage_base
            )
            data_roles = plan["artifact_contract"]["ledger_role_order"]
            if tuple(
                protocol["oracle_roles_opened_only_after_commit"]
            ) != ORACLE_ROLE_ORDER:
                _stop("oracle role order mismatch")
            oracle_roles = ORACLE_ROLE_ORDER
            pre_data_fds = {
                role: nonoracle_fds[role]
                for role in data_roles
                if role not in oracle_roles
            }
            evaluation_stage = stage_base / "evaluation.stage"
            audit_stage = stage_base / "audit.stage"
            spec = _worker_spec(
                plan,
                lock,
                evaluator_build_id,
                attempt_id,
                pre_data_fds,
                evaluation_stage,
                audit_stage,
                plan_sha,
                lock_sha,
            )
            # The sandboxed interpreter, all imports, policy and non-oracle
            # descriptors are live and attested before the irreversible event.
            worker_handle = _start_worker_pre_oracle(
                plan,
                lock,
                spec,
                pre_data_fds,
                stage_base,
                execution,
            )
            _verify_slot_lock(slot_lock)
            # This is the final path-based validation. After the durable
            # event, the already-running worker receives only SCM_RIGHTS FDs.
            if (
                hashlib.sha256(_read_small_path(plan_path)).hexdigest()
                != plan_sha
                or hashlib.sha256(_read_small_path(lock_path)).hexdigest()
                != lock_sha
            ):
                _stop("plan or lock changed before oracle commit")
            validate_lock(plan, lock, repo, plan_sha)
            _resnapshot(
                nonoracle_fds, nonoracle_snapshots, plan["max_rss_bytes"]
            )
            _verify_slot_lock(slot_lock)
            append_event(
                attempt_dir,
                protocol,
                state="STARTED",
                phase="ORACLE_OPEN_COMMITTED",
                oracle_open_committed=True,
                evaluator_build_id=evaluator_build_id,
                reason_code=None,
            )
            _verify_slot_lock(slot_lock)
            oracle_fds, oracle_snapshots = _open_oracle_roles_after_commit(
                plan, lock, attempt_dir
            )
            _verify_slot_lock(slot_lock)
            _send_oracle_fds(worker_handle, oracle_roles, oracle_fds)
            _finish_worker(worker_handle)
            worker_handle = None
            _verify_slot_lock(slot_lock)
            validation["input_snapshots"] = _computed_input_snapshots(
                plan,
                lock,
                {**nonoracle_snapshots, **oracle_snapshots},
                {**nonoracle_fds, **oracle_fds},
            )
            return publish_packages(
                plan,
                attempt_dir,
                evaluator_build_id,
                attempt_id,
                evaluation_stage,
                audit_stage,
                validation,
            )
        except EvaluationStopped as exc:
            try:
                mark_attempt_stopped(
                    attempt_dir,
                    protocol,
                    str(exc).split(":", 1)[-1].strip(),
                )
            except (EvaluationStopped, OSError):
                pass
            raise
        finally:
            if worker_handle is not None:
                _abort_worker(worker_handle)
            for fd in [
                *nonoracle_fds.values(),
                *oracle_fds.values(),
                *(execution.get("fds", []) if execution else []),
            ]:
                try:
                    os.close(fd)
                except OSError:
                    pass


def smoke() -> None:
    spec = {
        "z": {"0.95": 1.959963984540054, "0.99": 2.5758293035489004}
    }
    value = _proportion(99, 100, spec)
    if value["rate"] != 0.99 or not (0.0 <= value["wilson_99_low"] <= 1.0):
        _stop("smoke failed")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    parser.add_argument("--lock", type=Path, default=LOCK_PATH)
    parser.add_argument("--worker-spec-fd", type=int)
    parser.add_argument("--worker-control-fd", type=int)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.smoke:
            smoke()
            print(json.dumps({"status": "SMOKE_OK"}, sort_keys=True))
            return 0
        if args.worker_spec_fd is not None:
            worker_payload = _read_fd(args.worker_spec_fd, ADMIN_RSS_LIMIT)
            if len(worker_payload) > MAX_JSON_BYTES:
                _stop("worker spec exceeds JSON limit")
            spec = parse_json(worker_payload, "worker spec")
            _apply_resource_limits(spec["max_rss_bytes"])
            received_fds: list[int] = []
            try:
                if args.worker_control_fd is not None:
                    _worker_pre_oracle_attest(spec)
                    received_fds = _worker_receive_oracle_fds(
                        args.worker_control_fd, spec
                    )
                worker_execute(spec)
            finally:
                for fd in received_fds:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
            print(
                canonical_json(
                    {
                        "schema_version": WORKER_STDOUT_SCHEMA,
                        "evaluator_build_id": spec["evaluator_build_id"],
                        "status": "STAGED",
                    }
                ).decode(),
                end="",
            )
            return 0
        evaluation, audit = run_evaluation(args.plan, args.lock)
        print(
            canonical_json(
                {
                    "status": "FINAL",
                    "evaluation_root": str(evaluation),
                    "audit_root": str(audit),
                }
            ).decode(),
            end="",
        )
        return 0
    except EvaluationStopped as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
