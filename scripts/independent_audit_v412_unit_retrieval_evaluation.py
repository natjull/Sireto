#!/usr/bin/env python3
"""Independent, read-only audit for the V4.12 evaluator.

This module deliberately does not import the evaluator implementation.
Its default mode is static; artifact auditing requires explicit paths.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq


STOP = "STOP_V412_UNIT_RETRIEVAL_EVALUATION_AUDIT"
ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "config/v4_12_unit_retrieval_evaluator_plan.json"
LOCK = ROOT / "config/v4_12_unit_retrieval_evaluator_execution_lock.json"
CONTRACT = ROOT / "docs/v4_12_unit_retrieval_evaluator_contract.md"
SOURCE = ROOT / "scripts/evaluate_v412_unit_retrieval.py"
PROFILE = ROOT / "config/v4_12_unit_retrieval_evaluator.sb"
MAXIMUM = 32 * 1024 * 1024
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


class IndependentAuditStopped(RuntimeError):
    pass


def _stop(message: str) -> None:
    raise IndependentAuditStopped(f"{STOP}: {message}")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            _stop(f"duplicate JSON key: {key}")
        value[key] = item
    return value


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
        ).encode()
    except (TypeError, ValueError) as exc:
        _stop(f"non-canonical JSON: {exc}")


def _json_payload(
    payload: bytes,
    path: Path,
    *,
    require_canonical: bool = True,
) -> dict[str, Any]:
    if len(payload) > MAXIMUM:
        _stop(f"JSON too large: {path}")
    try:
        value = json.loads(
            payload.decode(),
            object_pairs_hook=_pairs,
            parse_constant=lambda token: _stop(f"invalid constant {token}"),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _stop(f"invalid JSON {path}: {exc}")
    if type(value) is not dict:
        _stop(f"JSON is not an object: {path}")
    if require_canonical and payload != canonical_json(value):
        _stop(f"JSON is not canonical: {path}")
    return value


def _json(path: Path, *, require_canonical: bool = True) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        _stop(f"not a regular file: {path}")
    return _json_payload(
        path.read_bytes(), path, require_canonical=require_canonical
    )


def _keys(value: Mapping[str, Any], expected: Sequence[str], label: str) -> None:
    if set(value) != set(expected):
        _stop(f"{label} keyset mismatch")


def _sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        _stop(f"not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _open_exact_fd(path: Path) -> int:
    if not path.is_absolute():
        _stop(f"exact input path is not absolute: {path}")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_DIRECTORY", 0)
    )
    leaf_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    directory_fd = os.open("/", directory_flags)
    try:
        for component in path.parts[1:-1]:
            next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        fd = os.open(path.name, leaf_flags, dir_fd=directory_fd)
    except OSError as exc:
        os.close(directory_fd)
        _stop(f"cannot open exact input {path}: {exc}")
    os.close(directory_fd)
    return fd


def _open_exact_dir(path: Path) -> int:
    if not path.is_absolute():
        _stop(f"exact root path is not absolute: {path}")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_DIRECTORY", 0)
    )
    fd = os.open("/", flags)
    try:
        for component in path.parts[1:]:
            next_fd = os.open(component, flags, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        if not stat.S_ISDIR(os.fstat(fd).st_mode):
            _stop(f"not an exact directory root: {path}")
        return fd
    except (OSError, IndependentAuditStopped) as exc:
        os.close(fd)
        if isinstance(exc, IndependentAuditStopped):
            raise
        _stop(f"cannot open exact root {path}: {exc}")


def _open_anchored_child(root_fd: int, name: str) -> int:
    if not name or name in {".", ".."} or "/" in name:
        _stop("invalid anchored child name")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        fd = os.open(name, flags, dir_fd=root_fd)
    except OSError as exc:
        _stop(f"cannot open anchored child {name}: {exc}")
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        _stop(f"anchored child is not regular: {name}")
    return fd


def _read_anchored_child(root_fd: int, name: str) -> bytes:
    fd = _open_anchored_child(root_fd, name)
    try:
        output = bytearray()
        while chunk := os.read(fd, 1024 * 1024):
            output.extend(chunk)
        return bytes(output)
    finally:
        os.close(fd)


def _anchored_file_tree(root: Path) -> set[str]:
    root_fd = _open_exact_dir(root)
    try:
        names = set(os.listdir(root_fd))
        for name in names:
            child_fd = _open_anchored_child(root_fd, name)
            os.close(child_fd)
        return names
    finally:
        os.close(root_fd)


def _snapshot_exact_file(path: Path) -> dict[str, Any]:
    fd = _open_exact_fd(path)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            _stop(f"not a regular exact input: {path}")
        digest = hashlib.sha256()
        os.lseek(fd, 0, os.SEEK_SET)
        while chunk := os.read(fd, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(fd)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after:
            _stop(f"exact input mutated during audit: {path}")
        return {
            "size_bytes": before.st_size,
            "sha256": digest.hexdigest(),
        }
    finally:
        os.close(fd)


def _read_exact_small(path: Path) -> bytes:
    fd = _open_exact_fd(path)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAXIMUM:
            _stop(f"invalid administrative file: {path}")
        chunks: list[bytes] = []
        total = 0
        while chunk := os.read(fd, min(1024 * 1024, MAXIMUM + 1 - total)):
            chunks.append(chunk)
            total += len(chunk)
            if total > MAXIMUM:
                _stop(f"administrative file too large: {path}")
        after = os.fstat(fd)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            _stop(f"administrative file mutated during audit: {path}")
        return b"".join(chunks)
    finally:
        os.close(fd)


def _audit_execution_lock(
    plan: Mapping[str, Any],
    plan_path: Path,
    lock_path: Path,
) -> tuple[dict[str, Any], str]:
    lock_payload = _read_exact_small(lock_path)
    lock = _json_payload(lock_payload, lock_path)
    if set(lock) != LOCK_KEYS:
        _stop("execution lock keyset mismatch")
    if (
        lock["schema_version"]
        != "sireto-v4.12-unit-retrieval-evaluator-execution-lock-1"
        or lock["purpose"] != "V4.12_UNIT_RETRIEVAL_ORACLE_EVALUATION"
        or lock["audit_verdict"] != "GO_CODE_V412_UNIT_RETRIEVAL_EVALUATOR"
        or lock["input_paths"] != plan["input_paths"]
        or lock["input_hashes"] != plan["input_hashes"]
        or lock["evaluation_spec_sha256"]
        != hashlib.sha256(canonical_json(plan["evaluation_spec"])).hexdigest()
        or lock["runtime"] != plan["runtime"]
        or lock["outputs"] != plan["outputs"]
        or lock["max_rss_bytes"] != plan["max_rss_bytes"]
    ):
        _stop("execution lock is not bound to the plan")
    commit = lock["git_commit"]
    if type(commit) is not str or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        _stop("invalid execution lock commit")
    sources = lock["source_hashes"]
    if type(sources) is not dict or set(sources) != set(plan["future_sources"]):
        _stop("execution lock source closure mismatch")
    try:
        kind = subprocess.run(
            ["/usr/bin/git", "cat-file", "-t", commit],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        _stop(f"locked commit is unavailable: {exc}")
    if kind != "commit":
        _stop("locked object is not a commit")
    for relative, expected in sorted(sources.items()):
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            _stop("unsafe locked source path")
        worktree = ROOT / relative_path
        if _sha(worktree) != expected:
            _stop(f"locked source differs from worktree: {relative}")
        try:
            blob = subprocess.run(
                ["/usr/bin/git", "show", f"{commit}:{relative}"],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            _stop(f"locked source blob unavailable: {relative}: {exc}")
        if hashlib.sha256(blob).hexdigest() != expected:
            _stop(f"locked source differs from commit: {relative}")
    try:
        relative_plan = str(plan_path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        relative_plan = None
    if relative_plan is not None and sources.get(relative_plan) != _sha(plan_path):
        _stop("canonical plan is not sealed by execution lock")
    sandbox = lock["sandbox"]
    if set(sandbox) != {
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
    } or (
        sandbox["system_read_roots"] != ["/System", "/usr", "/opt/homebrew"]
        or sandbox["device_read_paths"]
        != ["/dev/null", "/dev/urandom", "/dev/fd"]
        or sandbox["network_allowed"] is not False
        or sandbox["fork_allowed"] is not False
        or sandbox["write_scope"] != "PRIVATE_EVALUATOR_STAGING_ONLY"
    ):
        _stop("execution lock sandbox policy mismatch")
    role_bindings = {
        "executable": "sandbox_executable",
        "python_framework_app": "python_framework_app",
        "python_framework_library": "python_framework_library",
        "git_executable": "git_executable",
    }
    for key, role in role_bindings.items():
        if (
            sandbox.get(key) != plan["input_paths"][role]
            or sandbox.get(f"{key}_sha256") != plan["input_hashes"][role]
        ):
            _stop(f"execution lock runtime binding mismatch: {role}")
    return lock, hashlib.sha256(lock_payload).hexdigest()


def _audit_plan_inputs(
    plan: Mapping[str, Any], lock: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    if (
        len(plan["input_paths"]) != 16
        or set(plan["input_paths"]) != set(plan["input_hashes"])
        or plan["input_paths"] != lock["input_paths"]
        or plan["input_hashes"] != lock["input_hashes"]
    ):
        _stop("exact 16-input plan/lock closure mismatch")
    snapshots: dict[str, dict[str, Any]] = {}
    for role, raw_path in plan["input_paths"].items():
        path = Path(raw_path)
        if not path.is_absolute():
            _stop(f"input path is not absolute: {role}")
        snapshot = _snapshot_exact_file(path)
        if snapshot["sha256"] != plan["input_hashes"][role]:
            _stop(f"plan input hash mismatch: {role}")
        snapshots[role] = snapshot
    return snapshots


def _wilson(success: int, denominator: int, z: float) -> tuple[float, float]:
    if denominator <= 0 or not 0 <= success <= denominator:
        _stop("invalid Wilson counts")
    observed = success / denominator
    scale = 1.0 + z * z / denominator
    centre = (observed + z * z / (2 * denominator)) / scale
    radius = (
        z
        * math.sqrt(
            observed * (1 - observed) / denominator
            + z * z / (4 * denominator * denominator)
        )
        / scale
    )
    return max(0.0, centre - radius), min(1.0, centre + radius)


def _proportion(
    record: Mapping[str, Any], z95: float, z99: float, label: str
) -> None:
    expected = (
        "success_count",
        "denominator_count",
        "rate",
        "wilson_95_low",
        "wilson_95_high",
        "wilson_99_low",
        "wilson_99_high",
    )
    _keys(record, expected, label)
    success = record["success_count"]
    denominator = record["denominator_count"]
    if record["rate"] != success / denominator:
        _stop(f"{label} observed rate mismatch")
    low95, high95 = _wilson(success, denominator, z95)
    low99, high99 = _wilson(success, denominator, z99)
    if (
        record["wilson_95_low"],
        record["wilson_95_high"],
        record["wilson_99_low"],
        record["wilson_99_high"],
    ) != (low95, high95, low99, high99):
        _stop(f"{label} Wilson mismatch")


def _proportion_value(
    success: int, denominator: int, z95: float, z99: float
) -> dict[str, Any]:
    low95, high95 = _wilson(success, denominator, z95)
    low99, high99 = _wilson(success, denominator, z99)
    return {
        "success_count": success,
        "denominator_count": denominator,
        "rate": success / denominator,
        "wilson_95_low": low95,
        "wilson_95_high": high95,
        "wilson_99_low": low99,
        "wilson_99_high": high99,
    }


def _recompute_metrics(
    rows: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
    ids: Mapping[str, str],
) -> dict[str, Any]:
    spec = plan["evaluation_spec"]
    z95 = spec["confidence_interval"]["z"]["0.95"]
    z99 = spec["confidence_interval"]["z"]["0.99"]
    measurements = []
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
            _stop(f"population mismatch: {population}")
        measurements.append(
            {
                "population": population,
                "measurement_type": "V412_MEASUREMENT",
                "total_query_count": len(selected),
                "match_exact_count": len(exact),
                "ambiguous_count": len(ambiguous),
                "coverage": _proportion_value(
                    len(exact), len(selected), z95, z99
                ),
                "recall_at": {
                    str(k): _proportion_value(
                        sum(row[f"hit_at_{k}"] is True for row in exact),
                        len(exact),
                        z95,
                        z99,
                    )
                    for k in spec["recall_k"]
                },
            }
        )
    frozen = spec["frozen_references"]
    references = [
        {
            "name": row["name"],
            "reference_type": "FROZEN_REFERENCE",
            "source_build_id": frozen["build_id"],
            "source_population_count": row["coverage_denominator"],
            "coverage": _proportion_value(
                row["coverage_success"], row["coverage_denominator"], z95, z99
            ),
            "recall_at_100": _proportion_value(
                row["recall_at_100_success"],
                row["recall_at_100_denominator"],
                z95,
                z99,
            ),
        }
        for row in frozen["rows"]
    ]
    gate = spec["gate"]
    coverage = measurements[0]["coverage"]["rate"]
    recall = measurements[0]["recall_at"]["100"]["rate"]
    gates = {
        "gate_statistic": "OBSERVED_RATE_FROM_RAW_COUNTS",
        "population": "global",
        "coverage_minimum": gate["coverage_minimum"],
        "recall_at_100_minimum": gate["recall_at_100_minimum"],
        "coverage_observed": coverage,
        "recall_at_100_observed": recall,
        "coverage_pass": coverage >= gate["coverage_minimum"],
        "recall_at_100_pass": recall >= gate["recall_at_100_minimum"],
        "all_pass": coverage >= gate["coverage_minimum"]
        and recall >= gate["recall_at_100_minimum"],
    }
    durations = spec["latency"]["durations_ns"]
    query_count = spec["latency"]["query_count"]
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
    return {
        "schema_version": plan["artifact_contract"]["schema_versions"]["metrics"],
        **ids,
        "population_order": spec["population_order"],
        "reference_order": spec["reference_order"],
        "v412_measurements": measurements,
        "frozen_references": references,
        "latency": {
            "latency_source": "WORKER_INTEGRITY_AGGREGATE",
            "durations_ns": durations,
            "durations_seconds": {
                key: value / 1_000_000_000 for key, value in durations.items()
            },
            "query_count": query_count,
            "mean_wall_seconds_per_query_from_aggregate": durations["total"]
            / 1_000_000_000
            / query_count,
            "per_query_timing_available": False,
            "p95_available": False,
            "latency_gate_evaluated": False,
        },
        "gates": gates,
        "verdict": verdict,
        "declarations": declarations,
    }


def _logical_outcomes(rows: Sequence[Mapping[str, Any]]) -> bytes:
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
    output = bytearray()
    for row in rows:
        fields: list[bytes] = []
        for name in columns:
            value = row[name]
            if value is None:
                fields.append(b"\\N")
            elif name.startswith("hit_at_"):
                if type(value) is not bool:
                    _stop("non-boolean hit")
                fields.append(b"1" if value else b"0")
            elif name in {"candidate_count", "exact_rank"}:
                if type(value) is not int or value < 0:
                    _stop("invalid outcome integer")
                fields.append(str(value).encode("ascii"))
            elif name == "query_id":
                fields.append(value.encode("utf-8"))
            else:
                fields.append(value.encode("ascii"))
        output.extend(b"\x00".join(fields))
        output.extend(b"\n")
    return bytes(output)


def static_audit(repo: Path = ROOT) -> dict[str, Any]:
    plan = _json(repo / PLAN.relative_to(ROOT), require_canonical=False)
    contract = (repo / CONTRACT.relative_to(ROOT)).read_text()
    source_path = repo / SOURCE.relative_to(ROOT)
    source = source_path.read_text()
    profile = (repo / PROFILE.relative_to(ROOT)).read_text()
    tree = ast.parse(source)
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    forbidden_imports = {
        "xgboost",
        "xgb_matcher",
        "transformers",
        "torch",
        "ollama",
    }
    if imports.intersection(forbidden_imports):
        _stop("evaluator imports matching/model code")
    markers = (
        "O_NOFOLLOW",
        "ORACLE_OPEN_COMMITTED",
        "OBSERVED_RATE_FROM_RAW_COUNTS",
        "events.jsonl",
        "state.json",
        "renameatx_np",
        "fcntl.flock",
        "RLIMIT_AS",
        "SCM_RIGHTS",
        "READY_PRE_ORACLE",
        "proc_pid_rusage",
        "wait4",
        "_verify_sealed_execution",
        "_exclusive_slot_lock",
        "ORACLE_ROLE_ORDER",
        "computed_attestation.json",
        "computed_attestation_sha256",
        ".state-cache-",
        "_tree_sha256",
        "size_bytes_before",
        "data_input_count",
        "source_hashes",
    )
    for marker in markers:
        if marker not in source and marker not in contract:
            _stop(f"required marker missing: {marker}")
    if "input_snapshots.json" in source:
        _stop("undeclared attempt snapshot artifact is present")
    if (
        "(deny default)" not in profile
        or "(deny network*)" not in profile
        or "(deny process-fork)" not in profile
        or "(allow file-read* file-write*" not in profile
        or "process-info-setcontrol" not in profile
    ):
        _stop("sandbox profile is not closed")
    spec = plan["evaluation_spec"]
    if list(spec) != plan["identity_projections"]["evaluation_spec_keys"]:
        _stop("evaluation_spec projection mismatch")
    if spec["gate"]["gate_statistic"] != "OBSERVED_RATE_FROM_RAW_COUNTS":
        _stop("gate statistic mismatch")
    if (
        spec["join"]["truth_absent_policy"] != "MISS_AT_ALL_K"
        or spec["join"]["ambiguous_policy"] != "KEEP_EXCLUDE_FROM_RECALL"
    ):
        _stop("miss/ambiguous policy mismatch")
    order = plan["publication"]["promotion_order"]
    if order.index("PROMOTE_FINAL_AUDIT") >= order.index(
        "PROMOTE_FINAL_EVALUATION"
    ):
        _stop("publication order mismatch")
    boundary = plan["attempt_protocol"]
    if (
        len(boundary["pre_oracle_revalidation_roles"]) != 12
        or len(boundary["oracle_roles_opened_only_after_commit"]) != 4
        or not boundary["event_log_authoritative"]
    ):
        _stop("attempt boundary mismatch")
    return {
        "verdict": "GO_CODE_V412_UNIT_RETRIEVAL_EVALUATOR_AUDIT",
        "source_sha256": _sha(source_path),
        "plan_sha256": _sha(repo / PLAN.relative_to(ROOT)),
        "checks": {
            "no_matching_import": True,
            "sandbox_closed": True,
            "evaluation_spec_exact": True,
            "attempt_boundary": True,
        },
    }


def _record(path: Path, parquet: bool) -> dict[str, Any]:
    common = {
        "path": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _sha(path),
    }
    if not parquet:
        return common
    table = pq.read_table(path)
    return {
        **common,
        "row_count": table.num_rows,
        "schema": [
            {"name": f.name, "type": str(f.type), "nullable": f.nullable}
            for f in table.schema
        ],
        "metadata": (
            None
            if table.schema.metadata is None
            else {k.decode(): v.decode() for k, v in table.schema.metadata.items()}
        ),
    }


def _tree_sha256(root: Path, names: Sequence[str]) -> str:
    if set(path.name for path in root.iterdir()) != set(names):
        _stop("attested artifact tree file set mismatch")
    records = {
        name: {
            "size_bytes": (root / name).stat().st_size,
            "sha256": _sha(root / name),
        }
        for name in names
    }
    return hashlib.sha256(canonical_json(records)).hexdigest()


def audit_artifacts(
    evaluation_root: Path,
    audit_root: Path,
    plan_path: Path,
    lock_path: Path,
    attempt_dir: Path,
) -> dict[str, Any]:
    authorization = audit_event_journal(attempt_dir, plan_path, lock_path)
    if (
        authorization["state"] != "FINAL"
        or authorization["oracle_open_committed"] is not True
    ):
        _stop("attempt journal does not authorize independent input audit")
    plan_payload = _read_exact_small(plan_path)
    plan = _json_payload(
        plan_payload, plan_path, require_canonical=False
    )
    if hashlib.sha256(plan_payload).hexdigest() != authorization["plan_sha256"]:
        _stop("plan changed after audit authorization")
    lock, lock_sha256 = _audit_execution_lock(plan, plan_path, lock_path)
    if lock_sha256 != authorization["lock_sha256"]:
        _stop("execution lock changed after audit authorization")
    input_snapshots = _audit_plan_inputs(plan, lock)
    evaluator_identity = {
        "schema_version": plan["identity_projections"][
            "build_identity_schema_version"
        ],
        "plan_sha256": hashlib.sha256(plan_payload).hexdigest(),
        "lock_sha256": lock_sha256,
        "source_hashes": lock["source_hashes"],
        "input_hashes": lock["input_hashes"],
        "worker_build_id": plan["prerequisite"]["worker_build_id"],
        "oracle_build_id": plan["oracle"]["build_id"],
        "parity_build_id": plan["prerequisite"]["parity_build_id"],
        "evaluation_spec": plan["evaluation_spec"],
        "runtime": plan["runtime"],
    }
    _keys(
        evaluator_identity,
        plan["identity_projections"]["build_identity_keys"],
        "evaluator build identity",
    )
    expected_evaluator_build_id = hashlib.sha256(
        canonical_json(evaluator_identity)
    ).hexdigest()
    if authorization["evaluator_build_id"] != expected_evaluator_build_id:
        _stop("attempt evaluator build is not bound to plan and lock")
    artifact = plan["artifact_contract"]
    expected_eval = {
        "query_outcomes.parquet",
        "metrics.json",
        "integrity.json",
        "manifest.json",
    }
    expected_audit = {"open_ledger.parquet", "provenance.json", "manifest.json"}
    if (
        _anchored_file_tree(evaluation_root) != expected_eval
        or _anchored_file_tree(audit_root) != expected_audit
    ):
        _stop("artifact tree mismatch")
    attestation_path = (
        attempt_dir
        / plan["attempt_protocol"]["computed_attestation_filename"]
    )
    attestation = _json(attestation_path)
    protocol = plan["attempt_protocol"]
    _keys(
        attestation,
        protocol["computed_attestation_keyset"],
        "computed attestation",
    )
    if (
        _sha(attestation_path)
        != authorization["computed_attestation_sha256"]
        or attestation["schema_version"]
        != protocol["schema_versions"]["computed_attestation"]
        or attestation["measurement_slot_id"]
        != authorization["measurement_slot_id"]
        or attestation["attempt_id"] != authorization["attempt_id"]
        or attestation["evaluator_build_id"] != expected_evaluator_build_id
        or attestation["plan_sha256"] != authorization["plan_sha256"]
        or attestation["lock_sha256"] != authorization["lock_sha256"]
    ):
        _stop("computed attestation identity/hash mismatch")
    attested_inputs = attestation["input_snapshots"]
    if (
        type(attested_inputs) is not dict
        or set(attested_inputs) != set(protocol["computed_input_role_order"])
    ):
        _stop("computed attestation input closure mismatch")
    for role in protocol["computed_input_role_order"]:
        record = attested_inputs[role]
        _keys(
            record,
            protocol["computed_input_snapshot_keyset"],
            f"computed snapshot {role}",
        )
        actual = input_snapshots[role]
        if (
            record["absolute_path"] != plan["input_paths"][role]
            or record["projection"]
            != protocol["computed_input_projections"][role]
            or record["size_bytes_before"] != actual["size_bytes"]
            or record["size_bytes_after"] != actual["size_bytes"]
            or record["sha256_before"] != actual["sha256"]
            or record["sha256_after"] != actual["sha256"]
            or actual["sha256"] != plan["input_hashes"][role]
        ):
            _stop(f"computed attestation live input mismatch: {role}")
    if (
        attestation["evaluation_manifest_sha256"]
        != _sha(evaluation_root / "manifest.json")
        or attestation["audit_manifest_sha256"]
        != _sha(audit_root / "manifest.json")
        or attestation["evaluation_tree_sha256"]
        != _tree_sha256(evaluation_root, plan["outputs"]["runtime_files"])
        or attestation["audit_tree_sha256"]
        != _tree_sha256(audit_root, plan["outputs"]["audit_files"])
    ):
        _stop("computed attestation final tree/manifest mismatch")
    metrics = _json(evaluation_root / "metrics.json")
    integrity = _json(evaluation_root / "integrity.json")
    manifest = _json(evaluation_root / "manifest.json")
    provenance = _json(audit_root / "provenance.json")
    audit_manifest = _json(audit_root / "manifest.json")
    if (
        metrics.get("attempt_id") != authorization["attempt_id"]
        or metrics.get("evaluator_build_id")
        != authorization["evaluator_build_id"]
    ):
        _stop("artifact identity is not authorized by attempt journal")
    keys = artifact["json_keysets"]
    for value, name in (
        (metrics, "metrics"),
        (integrity, "integrity"),
        (manifest, "evaluation_manifest"),
        (provenance, "provenance"),
        (audit_manifest, "audit_manifest"),
    ):
        _keys(value, keys[name], name)
    if (
        set(manifest["files"])
        != set(artifact["file_sets"]["evaluation_manifest"])
        or set(audit_manifest["files"])
        != set(artifact["file_sets"]["audit_manifest"])
    ):
        _stop("manifest files keyset mismatch")
    if metrics["population_order"] != artifact["population_order"]:
        _stop("population order mismatch")
    if metrics["reference_order"] != artifact["reference_order"]:
        _stop("reference order mismatch")
    interval = plan["evaluation_spec"]["confidence_interval"]["z"]
    for measurement in metrics["v412_measurements"]:
        _keys(measurement, keys["measurement_record"], "measurement")
        _proportion(measurement["coverage"], interval["0.95"], interval["0.99"], "coverage")
        for k in ("1", "10", "50", "100"):
            _proportion(
                measurement["recall_at"][k],
                interval["0.95"],
                interval["0.99"],
                f"recall@{k}",
            )
    outcomes = pq.read_table(evaluation_root / "query_outcomes.parquet")
    schema = pa.schema(
        [
            pa.field(item["name"], pa.type_for_alias(item["type"]), item["nullable"])
            for item in artifact["query_outcomes_schema"]
        ],
        metadata=None,
    )
    if not outcomes.schema.equals(schema, check_metadata=True):
        _stop("outcomes schema mismatch")
    rows = outcomes.to_pylist()
    seen: set[str] = set()
    for row in rows:
        query_id = row["query_id"]
        if (
            type(query_id) is not str
            or not query_id
            or query_id != query_id.strip()
            or query_id in seen
        ):
            _stop("invalid or duplicate outcome query_id")
        seen.add(query_id)
        count = row["candidate_count"]
        rank = row["exact_rank"]
        if type(count) is not int or not 0 <= count <= 100:
            _stop("invalid candidate_count")
        if rank is not None and (
            type(rank) is not int or not 1 <= rank <= count
        ):
            _stop("invalid exact_rank")
        if row["label_kind"] == "AMBIGUOUS":
            if rank is not None or any(
                row[f"hit_at_{k}"] is not None
                for k in plan["evaluation_spec"]["recall_k"]
            ):
                _stop("ambiguous outcome has a result")
        elif row["label_kind"] == "MATCH_EXACT":
            for k in plan["evaluation_spec"]["recall_k"]:
                if row[f"hit_at_{k}"] is not (
                    rank is not None and rank <= k
                ):
                    _stop(f"Hit@{k} invariant mismatch")
        else:
            _stop("invalid label kind")
    payload = _logical_outcomes(rows)
    if (
        len(payload) != integrity["query_outcomes_payload_bytes"]
        or hashlib.sha256(payload).hexdigest()
        != integrity["query_outcomes_payload_sha256"]
    ):
        _stop("outcomes logical payload mismatch")
    missing = sum(
        row["label_kind"] == "MATCH_EXACT" and row["exact_rank"] is None
        for row in rows
    )
    if (
        missing != integrity["missing_truth_count"]
        or missing
        != metrics["v412_measurements"][0]["recall_at"]["100"][
            "denominator_count"
        ]
        - metrics["v412_measurements"][0]["recall_at"]["100"]["success_count"]
    ):
        _stop("missing truth formula mismatch")
    ids = {
        "evaluator_build_id": metrics["evaluator_build_id"],
        "attempt_id": metrics["attempt_id"],
        "worker_build_id": plan["prerequisite"]["worker_build_id"],
        "oracle_build_id": plan["oracle"]["build_id"],
        "parity_build_id": plan["prerequisite"]["parity_build_id"],
    }
    expected_metrics = _recompute_metrics(rows, plan, ids)
    if (
        metrics != expected_metrics
        or (evaluation_root / "metrics.json").read_bytes()
        != canonical_json(expected_metrics)
    ):
        _stop("metrics are not an exact independent recomputation")
    counts = [row["candidate_count"] for row in rows]
    expected_integrity = {
        "schema_version": artifact["schema_versions"]["integrity"],
        **ids,
        "query_count": len(rows),
        "match_exact_count": sum(
            row["label_kind"] == "MATCH_EXACT" for row in rows
        ),
        "ambiguous_count": sum(
            row["label_kind"] == "AMBIGUOUS" for row in rows
        ),
        "candidate_count": sum(counts),
        "minimum_pool_size": min(counts),
        "maximum_pool_size": max(counts),
        "under_ceiling_query_count": sum(count < 100 for count in counts),
        "empty_query_count": sum(count == 0 for count in counts),
        "missing_truth_count": missing,
        "query_outcomes_payload_bytes": len(payload),
        "query_outcomes_payload_sha256": hashlib.sha256(payload).hexdigest(),
        "metrics_sha256": _sha(evaluation_root / "metrics.json"),
        "input_snapshot_count": len(plan["input_hashes"]),
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
        _stop("integrity is not an exact independent recomputation")
    for name in artifact["file_sets"]["evaluation_manifest"]:
        if manifest["files"].get(name) != _record(
            evaluation_root / name, name.endswith(".parquet")
        ):
            _stop(f"evaluation record mismatch: {name}")
    for name in artifact["file_sets"]["audit_manifest"]:
        if audit_manifest["files"].get(name) != _record(
            audit_root / name, False
        ):
            _stop(f"audit record mismatch: {name}")
    expected_audit_manifest = {
        "schema_version": artifact["schema_versions"]["audit_manifest"],
        "evaluator_build_id": ids["evaluator_build_id"],
        "attempt_id": ids["attempt_id"],
        "files": {
            name: _record(audit_root / name, False)
            for name in artifact["file_sets"]["audit_manifest"]
        },
    }
    if (
        audit_manifest != expected_audit_manifest
        or (audit_root / "manifest.json").read_bytes()
        != canonical_json(expected_audit_manifest)
    ):
        _stop("audit manifest is not an exact recomputation")
    if provenance["evaluation_manifest_sha256"] != _sha(
        evaluation_root / "manifest.json"
    ):
        _stop("provenance manifest commitment mismatch")
    ledger = pq.read_table(audit_root / "open_ledger.parquet")
    expected_ledger_schema = pa.schema(
        [
            pa.field(item["name"], pa.type_for_alias(item["type"]), False)
            for item in artifact["ledger_schema"]
        ],
        metadata=None,
    )
    if not ledger.schema.equals(expected_ledger_schema, check_metadata=True):
        _stop("ledger schema mismatch")
    if ledger.column("role").to_pylist() != artifact["ledger_role_order"]:
        _stop("ledger role order mismatch")
    for row in ledger.to_pylist():
        role = row["role"]
        projection = {
            "worker_candidates_top100": "query_id,candidate_rank,candidate_siret",
            "worker_query_status": "query_id,candidate_count",
            "oracle_dev": (
                "query_id,dev_partition,label_kind,ground_truth_siret,"
                "ground_truth_siren"
            ),
        }.get(role, "FULL_JSON_EXACT_KEYSET")
        if (
            row["absolute_path"] != plan["input_paths"][role]
            or row["projection"] != projection
            or row["sha256_before"] != plan["input_hashes"][role]
            or row["sha256_after"] != plan["input_hashes"][role]
            or row["size_bytes_before"]
            != input_snapshots[role]["size_bytes"]
            or row["size_bytes_after"]
            != input_snapshots[role]["size_bytes"]
            or row["sha256_before"] != input_snapshots[role]["sha256"]
            or row["sha256_after"] != input_snapshots[role]["sha256"]
        ):
            _stop(f"ledger binding mismatch: {role}")
    for value in (integrity, manifest):
        for key, expected in ids.items():
            if value.get(key) != expected:
                _stop(f"artifact build identity mismatch: {key}")
    expected_provenance = {
        "schema_version": artifact["schema_versions"]["provenance"],
        "evaluator_build_id": ids["evaluator_build_id"],
        "attempt_id": ids["attempt_id"],
        "git_commit": lock["git_commit"],
        "source_hashes": lock["source_hashes"],
        "plan_sha256": authorization["plan_sha256"],
        "lock_sha256": lock_sha256,
        "input_hashes": plan["input_hashes"],
        "evaluation_spec_sha256": hashlib.sha256(
            canonical_json(plan["evaluation_spec"])
        ).hexdigest(),
        "runtime": plan["runtime"],
        "data_input_count": len(artifact["ledger_role_order"]),
        "evaluation_manifest_sha256": _sha(
            evaluation_root / "manifest.json"
        ),
        "declarations": expected_metrics["declarations"],
    }
    if (
        manifest["runtime"] != plan["runtime"]
        or manifest["declarations"] != expected_metrics["declarations"]
        or manifest["verdict"] != expected_metrics["verdict"]
        or provenance != expected_provenance
        or (audit_root / "provenance.json").read_bytes()
        != canonical_json(expected_provenance)
    ):
        _stop("manifest/provenance plan binding mismatch")
    return {
        "verdict": "GO_ARTIFACTS_V412_UNIT_RETRIEVAL_EVALUATION_AUDIT",
        "evaluator_build_id": metrics["evaluator_build_id"],
        "attempt_id": metrics["attempt_id"],
        "query_count": outcomes.num_rows,
        "missing_truth_count": missing,
        "attempt": authorization,
    }


def audit_event_journal(
    attempt_dir: Path,
    plan_path: Path,
    lock_path: Path,
) -> dict[str, Any]:
    plan = _json(plan_path, require_canonical=False)
    protocol = plan["attempt_protocol"]
    attempt_fd = _open_exact_dir(attempt_dir)
    try:
        slot_names = set(os.listdir(attempt_fd))
        for name in slot_names:
            child_fd = _open_anchored_child(attempt_fd, name)
            os.close(child_fd)
        receipt_payload = _read_anchored_child(attempt_fd, "receipt.json")
        events_payload = _read_anchored_child(attempt_fd, "events.jsonl")
        state_payload = _read_anchored_child(attempt_fd, "state.json")
        attestation_payload = (
            _read_anchored_child(
                attempt_fd, protocol["computed_attestation_filename"]
            )
            if protocol["computed_attestation_filename"] in slot_names
            else None
        )
    finally:
        os.close(attempt_fd)
    before_names = set(protocol["attempt_tree_before_computed_attestation"])
    after_names = set(protocol["attempt_tree_with_computed_attestation"])
    if slot_names not in (before_names, after_names):
        _stop("attempt tree mismatch")
    receipt_path = attempt_dir / "receipt.json"
    receipt = _json_payload(receipt_payload, receipt_path)
    _keys(receipt, protocol["receipt_keyset"], "receipt")
    if (
        receipt["schema_version"] != protocol["schema_versions"]["receipt"]
        or receipt["policy_immutable"] is not True
    ):
        _stop("invalid attempt receipt")
    measurement_payload = {
        "schema_version": protocol["schema_versions"][
            "measurement_slot_identity"
        ],
        "purpose": plan["purpose"],
        "worker_build_id": plan["prerequisite"]["worker_build_id"],
        "oracle_build_id": plan["oracle"]["build_id"],
        "parity_build_id": plan["prerequisite"]["parity_build_id"],
    }
    _keys(
        measurement_payload,
        protocol["measurement_slot_identity_keyset"],
        "measurement slot identity",
    )
    expected_slot = hashlib.sha256(canonical_json(measurement_payload)).hexdigest()
    attempt_payload = {
        "schema_version": protocol["schema_versions"]["attempt_identity"],
        "plan_sha256": receipt["plan_sha256"],
        "lock_sha256": receipt["lock_sha256"],
        "input_hashes": receipt["input_hashes"],
        "evaluation_spec_sha256": receipt["evaluation_spec_sha256"],
        "worker_build_id": receipt["worker_build_id"],
        "oracle_build_id": receipt["oracle_build_id"],
        "parity_build_id": receipt["parity_build_id"],
    }
    _keys(
        attempt_payload,
        protocol["attempt_identity_keyset"],
        "attempt identity",
    )
    expected_attempt = hashlib.sha256(canonical_json(attempt_payload)).hexdigest()
    if (
        attempt_dir.name != expected_slot
        or
        receipt["measurement_slot_id"] != expected_slot
        or receipt["attempt_id"] != expected_attempt
        or receipt["plan_sha256"] != _sha(plan_path)
        or receipt["lock_sha256"] != _sha(lock_path)
        or receipt["input_hashes"] != plan["input_hashes"]
        or receipt["evaluation_spec_sha256"]
        != hashlib.sha256(canonical_json(plan["evaluation_spec"])).hexdigest()
        or receipt["worker_build_id"]
        != plan["prerequisite"]["worker_build_id"]
        or receipt["oracle_build_id"] != plan["oracle"]["build_id"]
        or receipt["parity_build_id"]
        != plan["prerequisite"]["parity_build_id"]
    ):
        _stop("receipt is not bound to plan/attempt identities")
    payload = events_payload
    if not payload.endswith(b"\n"):
        _stop("partial event journal")
    previous_hash = None
    previous = None
    events = []
    for raw in payload.splitlines(keepends=True):
        try:
            event = json.loads(raw, object_pairs_hook=_pairs)
        except json.JSONDecodeError as exc:
            _stop(f"invalid event: {exc}")
        if raw != canonical_json(event):
            _stop("non-canonical event")
        _keys(event, protocol["event_keyset"], "event")
        if event["schema_version"] != protocol["schema_versions"]["event"]:
            _stop("event schema mismatch")
        if (
            event["state"] not in protocol["states"]
            or event["phase"] not in protocol["phases"]
            or event["measurement_slot_id"] != receipt["measurement_slot_id"]
            or event["attempt_id"] != receipt["attempt_id"]
        ):
            _stop("event protocol/identity mismatch")
        if event["previous_event_sha256"] != previous_hash:
            _stop("event chain mismatch")
        if previous is not None:
            if previous["state"] in {"FINAL", "STOPPED"}:
                _stop("terminal event has a successor")
            if event["sequence"] != previous["sequence"] + 1:
                _stop("event sequence mismatch")
            if previous["oracle_open_committed"] and not event["oracle_open_committed"]:
                _stop("oracle commit regressed")
            previous_attestation = previous["computed_attestation_sha256"]
            current_attestation = event["computed_attestation_sha256"]
            if previous_attestation is not None and (
                current_attestation != previous_attestation
            ):
                _stop("computed attestation hash changed")
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
            previous_pair = (previous["state"], previous["phase"])
            pair = (event["state"], event["phase"])
            if event["state"] == "STOPPED":
                if event["phase"] != "TERMINAL":
                    _stop("invalid STOPPED phase")
                if (
                    event["oracle_open_committed"]
                    is not previous["oracle_open_committed"]
                    or event["evaluator_build_id"]
                    != previous["evaluator_build_id"]
                ):
                    _stop("STOPPED changed oracle/build state")
            elif forward.get(previous_pair) != pair:
                _stop("invalid event state transition")
            if pair == ("STARTED", "ORACLE_OPEN_COMMITTED"):
                if (
                    previous["oracle_open_committed"] is not False
                    or event["oracle_open_committed"] is not True
                    or current_attestation is not None
                ):
                    _stop("invalid oracle commit event")
            elif (
                event["state"] != "STOPPED"
                and event["oracle_open_committed"] is not True
            ):
                _stop("post-oracle event lacks commit")
            if previous_pair != ("STARTED", "RECEIPT_DURABLE") and (
                event["evaluator_build_id"]
                != previous["evaluator_build_id"]
            ):
                _stop("event evaluator build changed")
            if pair == ("RECOVERABLE", "COMPUTED_STAGING_VALID"):
                if (
                    previous_attestation is not None
                    or type(current_attestation) is not str
                    or re.fullmatch(r"[0-9a-f]{64}", current_attestation)
                    is None
                ):
                    _stop("invalid computed attestation event")
            elif (
                previous_attestation is None
                and current_attestation is not None
            ):
                _stop("computed attestation appeared outside transition")
        elif (
            event["sequence"] != 0
            or event["state"] != "STARTED"
            or event["phase"] != "RECEIPT_DURABLE"
            or event["oracle_open_committed"] is not False
            or event["computed_attestation_sha256"] is not None
        ):
            _stop("initial event mismatch")
        events.append(event)
        previous = event
        previous_hash = hashlib.sha256(raw).hexdigest()
    state = _json_payload(state_payload, attempt_dir / "state.json")
    last = events[-1]
    if (
        receipt["measurement_slot_id"] != events[0]["measurement_slot_id"]
        or receipt["attempt_id"] != events[0]["attempt_id"]
    ):
        _stop("receipt/event identity mismatch")
    expected_state = {
        "schema_version": protocol["schema_versions"]["state"],
        "measurement_slot_id": last["measurement_slot_id"],
        "attempt_id": last["attempt_id"],
        "sequence": last["sequence"],
        "state": last["state"],
        "phase": last["phase"],
        "oracle_open_committed": last["oracle_open_committed"],
        "evaluator_build_id": last["evaluator_build_id"],
        "computed_attestation_sha256": last[
            "computed_attestation_sha256"
        ],
        "reason_code": last["reason_code"],
        "updated_at_utc": last["timestamp_utc"],
    }
    _keys(state, protocol["state_keyset"], "state")
    if state != expected_state:
        _stop("state cache is not derived from journal")
    attestation_path = (
        attempt_dir / protocol["computed_attestation_filename"]
    )
    if last["computed_attestation_sha256"] is not None:
        if slot_names != after_names or attestation_payload is None:
            _stop("computed event lacks exact attestation tree")
        if (
            hashlib.sha256(attestation_payload).hexdigest()
            != last["computed_attestation_sha256"]
        ):
            _stop("computed attestation hash differs from journal")
    elif slot_names == after_names and last["state"] != "STOPPED":
        _stop("orphan computed attestation is not terminally stopped")
    return {
        "event_count": len(events),
        "state": last["state"],
        "oracle_open_committed": last["oracle_open_committed"],
        "measurement_slot_id": receipt["measurement_slot_id"],
        "attempt_id": receipt["attempt_id"],
        "evaluator_build_id": last["evaluator_build_id"],
        "computed_attestation_sha256": last[
            "computed_attestation_sha256"
        ],
        "plan_sha256": receipt["plan_sha256"],
        "lock_sha256": receipt["lock_sha256"],
    }


def smoke() -> dict[str, Any]:
    low, high = _wilson(99, 100, 1.959963984540054)
    rows = [
        {
            "query_id": "é1",
            "dev_partition": "threshold_dev",
            "label_kind": "MATCH_EXACT",
            "candidate_count": 1,
            "exact_rank": None,
            "hit_at_1": False,
            "hit_at_10": False,
            "hit_at_50": False,
            "hit_at_100": False,
        }
    ]
    payload = _logical_outcomes(rows)
    if not (0 < low < high <= 1) or b"\\N" not in payload:
        _stop("independent smoke failed")
    return {"verdict": "GO_SYNTHETIC_V412_UNIT_RETRIEVAL_EVALUATION_AUDIT"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--evaluation-root", type=Path)
    parser.add_argument("--audit-root", type=Path)
    parser.add_argument("--attempt-root", type=Path)
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--lock", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.smoke:
            result = smoke()
        elif args.static:
            result = static_audit()
        elif (
            args.evaluation_root
            and args.audit_root
            and args.attempt_root
            and args.lock
        ):
            if (
                args.plan.absolute() != PLAN.absolute()
                or args.lock.absolute() != LOCK.absolute()
            ):
                _stop("artifact audit requires canonical plan and lock")
            result = audit_artifacts(
                args.evaluation_root,
                args.audit_root,
                args.plan,
                args.lock,
                args.attempt_root,
            )
        else:
            _stop(
                "explicit --static, --smoke or evaluation/audit/attempt/lock "
                "paths required"
            )
        print(canonical_json(result).decode(), end="")
        return 0
    except IndependentAuditStopped as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
