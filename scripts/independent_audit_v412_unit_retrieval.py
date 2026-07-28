#!/usr/bin/env python3
"""Independent, synthetic-only audit of the V4.12 unit retrieval milestone.

The audit reads repository sources, contracts, plans and sandbox profiles
only.  It never opens the real development queries, stores, historical
candidate file, oracle, models or final-test artifacts.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import types
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq


GO = "GO_V412_UNIT_RETRIEVAL_INDEPENDENT_AUDIT"
STOP = "STOP_V412_UNIT_RETRIEVAL_INDEPENDENT_AUDIT"
SCHEMA = "sireto-v4.12-unit-retrieval-independent-audit-1"
LIMIT = 512 * 1024 * 1024

CONTRACT = "docs/v4_12_unit_retrieval_engine_contract.md"
PLAN = "config/v4_12_unit_retrieval_engine_plan.json"
WORKER = "src/xgb_matcher/v412_unit_retrieval.py"
STRICT_STORES = "src/xgb_matcher/v412_strict_stores.py"
RUNNER = "scripts/run_v412_unit_retrieval.py"
PARITY = "scripts/audit_v412_unit_retrieval_parity.py"
WORKER_PROFILE = "config/v4_12_unit_retrieval.sb"
PARITY_PROFILE = "config/v4_12_unit_retrieval_parity.sb"
SELF = "scripts/independent_audit_v412_unit_retrieval.py"
SELF_TEST = "tests/test_independent_audit_v412_unit_retrieval.py"

STATIC_CHECKS = {
    "contract_and_plan",
    "source_closure",
    "worker_import_boundary",
    "frozen_policy",
    "schema_alignment",
    "manifest_contracts",
    "sandbox_boundaries",
    "sanitized_worker_spec",
    "worker_before_parity",
    "parity_executables_pinned",
    "parity_recovery_checks_derived",
    "parity_promotion_non_clobber",
}
SYNTHETIC_CHECKS = {
    "worker_payloads",
    "worker_manifest",
    "worker_publication",
    "parity_payloads",
    "parity_manifest",
    "parity_publication",
    "mutation_rejected",
    "symlink_rejected",
    "anchored_fd_survives_path_substitution",
    "worker_precedes_parity",
    "parity_recovery_mutation_rejected",
    "parity_destination_race_rejected",
    "parity_pending_substitution_rejected",
}


class IndependentAuditStopped(RuntimeError):
    """Raised on the first audit-contract divergence."""


def _stop(message: str) -> None:
    raise IndependentAuditStopped(f"{STOP}: {message}")


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


def _parse_json(payload: bytes, label: str) -> dict[str, Any]:
    duplicates: list[str] = []

    def hook(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                duplicates.append(key)
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=hook,
            parse_constant=lambda token: _stop(
                f"non-finite JSON token in {label}: {token}"
            ),
        )
    except IndependentAuditStopped:
        raise
    except Exception as exc:
        _stop(f"invalid {label}: {exc}")
    if duplicates or type(value) is not dict:
        _stop(f"{label} must be a duplicate-free JSON object")
    return value


def _safe_relative(relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        _stop(f"unsafe repository-relative path: {relative}")
    return path


def _secure_read(repo: Path, relative: str, maximum: int = LIMIT) -> bytes:
    repo = repo.absolute()
    relative_path = _safe_relative(relative)
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    root_descriptor: int | None = None
    descriptor: int | None = None
    try:
        root_descriptor = os.open(repo.anchor, directory_flags)
        for component in (*repo.parts[1:], *relative_path.parts[:-1]):
            next_descriptor = os.open(
                component,
                directory_flags,
                dir_fd=root_descriptor,
            )
            os.close(root_descriptor)
            root_descriptor = next_descriptor
        descriptor = os.open(
            relative_path.name,
            file_flags,
            dir_fd=root_descriptor,
        )
    except OSError as exc:
        _stop(f"cannot open audited source {relative}: {exc}")
    finally:
        if root_descriptor is not None:
            os.close(root_descriptor)

    def snapshot(open_descriptor: int) -> dict[str, Any]:
        info_before = os.fstat(open_descriptor)
        if not stat.S_ISREG(info_before.st_mode) or info_before.st_size > maximum:
            _stop(f"audited source is not regular or is too large: {relative}")
        digest = hashlib.sha256()
        offset = 0
        while offset < info_before.st_size:
            block = os.pread(
                open_descriptor,
                min(1024 * 1024, info_before.st_size - offset),
                offset,
            )
            if not block:
                _stop(f"short snapshot while auditing {relative}")
            digest.update(block)
            offset += len(block)
        info_after = os.fstat(open_descriptor)
        identity_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_size",
            "st_mtime_ns",
        )
        if any(
            getattr(info_before, field) != getattr(info_after, field)
            for field in identity_fields
        ):
            _stop(f"audited source changed during snapshot: {relative}")
        return {
            field: getattr(info_after, field) for field in identity_fields
        } | {"sha256": digest.hexdigest()}

    assert descriptor is not None
    try:
        before = snapshot(descriptor)
        chunks: list[bytes] = []
        offset = 0
        while offset < before["st_size"]:
            block = os.pread(
                descriptor,
                min(1024 * 1024, before["st_size"] - offset),
                offset,
            )
            if not block:
                _stop(f"short read while auditing {relative}")
            chunks.append(block)
            offset += len(block)
        payload = b"".join(chunks)
        after = snapshot(descriptor)
        if after != before or hashlib.sha256(payload).hexdigest() != before["sha256"]:
            _stop(f"audited source changed during read: {relative}")
        return payload
    finally:
        os.close(descriptor)


def _load_snapshot_module(
    repo: Path,
    relative: str,
    module_name: str,
) -> tuple[types.ModuleType, bytes]:
    payload = _secure_read(repo, relative)
    module = types.ModuleType(module_name)
    module.__file__ = str(repo / relative)
    module.__package__ = ""
    sys.modules[module_name] = module
    try:
        exec(compile(payload, module.__file__, "exec"), module.__dict__)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module, payload


def _literal_assignment(
    tree: ast.Module,
    name: str,
    environment: Mapping[str, Any] | None = None,
) -> Any:
    names = dict(environment or {})

    def evaluate(node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name) and node.id in names:
            return names[node.id]
        if isinstance(node, ast.List):
            return [evaluate(item) for item in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(evaluate(item) for item in node.elts)
        if isinstance(node, ast.Set):
            return {evaluate(item) for item in node.elts}
        if isinstance(node, ast.Dict):
            return {
                evaluate(key): evaluate(value)
                for key, value in zip(node.keys, node.values, strict=True)
            }
        raise ValueError(f"unsupported node: {ast.dump(node, include_attributes=False)}")

    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                value = node.value
                if value is None:
                    break
                try:
                    return evaluate(value)
                except Exception as exc:
                    _stop(f"{name} is not an auditable literal: {exc}")
    _stop(f"missing frozen assignment: {name}")


def _audit_worker_ast(payload: bytes) -> dict[str, Any]:
    try:
        tree = ast.parse(payload, filename=WORKER)
    except SyntaxError as exc:
        _stop(f"worker source is not valid Python: {exc}")
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            prefix = ("." * node.level) + module
            imported.add(prefix)
            for alias in node.names:
                imported.add(alias.name)
                imported.add(f"{prefix}.{alias.name}".strip("."))
    forbidden = {
        "retrieval",
        ".retrieval",
        "xgb_matcher.retrieval",
        "v41_retrieval",
        ".v41_retrieval",
        "xgb_matcher.v41_retrieval",
        "models",
        "xgboost",
    }
    if imported & forbidden:
        _stop(f"worker imports forbidden historical/model code: {sorted(imported & forbidden)}")
    definitions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    required = {
        "UnitRetrievalResult",
        "route_query",
        "retrieve_unit_query",
        "run_child_worker",
        "main",
    }
    if not required <= definitions:
        _stop(f"worker public surface incomplete: {sorted(required - definitions)}")
    strict_imports = {
        name
        for name in imported
        if name.endswith("v412_strict_stores")
    }
    if not strict_imports:
        _stop("worker does not import the certified strict stores")
    cache_namespace = _literal_assignment(tree, "CACHE_NAMESPACE")
    return {
        "tree": tree,
        "imports": sorted(imported),
        "retrieval": _literal_assignment(tree, "_RETRIEVAL_POLICY"),
        "tfidf_cache": _literal_assignment(
            tree,
            "_TFIDF_POLICY",
            {"CACHE_NAMESPACE": cache_namespace},
        ),
    }


def _require_markers(text: str, markers: Sequence[str], label: str) -> None:
    missing = [marker for marker in markers if marker not in text]
    if missing:
        _stop(f"{label} misses mandatory markers: {missing}")


def _source_order_check(runner_source: str, parity_source: str) -> None:
    runner_markers = (
        "result = subprocess.run(",
        "if result.returncode != 0:",
        "_validate_outputs(",
        "_promote(output, pending)",
        "_promote(pending, final_runtime)",
    )
    positions = [runner_source.find(marker) for marker in runner_markers]
    if any(position < 0 for position in positions) or positions != sorted(positions):
        _stop("runner no longer waits, validates and seals in the required order")
    _require_markers(
        parity_source,
        (
            'WORKER_VERDICT = "SEALED_V412_UNIT_RETRIEVAL"',
            "_validate_worker_manifest(",
            "_validate_worker_integrity(",
            "sealed input changed during parity evaluation",
        ),
        "parity controller",
    )


def _audit_controller_p1(parity: Any, parity_source: str) -> dict[str, bool]:
    run_spec_keys = set(parity.RUN_SPEC_KEYS)

    def executable_fields(kind: str) -> tuple[str, str]:
        matching = {
            key for key in run_spec_keys if kind in key and "executable" in key
        }
        hash_fields = {key for key in matching if "sha" in key or "hash" in key}
        path_fields = matching - hash_fields
        if len(hash_fields) != 1 or len(path_fields) != 1:
            _stop(f"parity {kind} executable path/hash are not pinned")
        return next(iter(path_fields)), next(iter(hash_fields))

    executable_keys = [
        *executable_fields("sandbox"),
        *executable_fields("python"),
    ]
    if any(parity_source.count(f'"{key}"') < 2 for key in executable_keys):
        _stop("pinned parity executables are not validated and consumed")

    expected = {
        "query_count": 1,
        "candidate_count": 1,
        "minimum_pool_size": 1,
        "maximum_pool_size": 1,
        "under_ceiling_query_count": 1,
        "empty_query_count": 0,
        "candidate_payload_bytes": 1,
        "candidate_payload_sha256": "e" * 64,
        "status_payload_bytes": 1,
        "status_payload_sha256": "d" * 64,
    }
    inconsistent = {
        "schema_version": parity.PARITY_SCHEMA,
        "parity_build_id": "a" * 64,
        "worker_build_id": "b" * 64,
        "query_count": 1,
        "candidate_count": 1,
        "minimum_pool_size": 1,
        "maximum_pool_size": 1,
        "under_ceiling_query_count": 1,
        "empty_query_count": 0,
        "candidate_payload_bytes": 1,
        "candidate_payload_sha256": "f" * 64,
        "expected_candidate_payload_bytes": 1,
        "expected_candidate_payload_sha256": "e" * 64,
        "status_payload_bytes": 1,
        "status_payload_sha256": "d" * 64,
        "expected_status_payload_bytes": 1,
        "expected_status_payload_sha256": "d" * 64,
        "checks": {key: True for key in parity.CHECK_KEYS},
        "sandbox_checks": {
            key: True for key in parity.SANDBOX_CHECK_KEYS
        },
        "declarations": dict(parity.DECLARATIONS),
        "verdict": parity.GO,
    }
    try:
        parity._validate_parity_report(
            inconsistent,
            parity_id="a" * 64,
            spec={"worker_build_id": "b" * 64, "expected": expected},
        )
    except parity.ParityStopped:
        recovery_derived = True
    else:
        _stop("parity recovery trusts stored checks instead of deriving them")

    tree = ast.parse(parity_source, filename=PARITY)
    promote = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_promote"
        ),
        None,
    )
    if promote is None:
        _stop("parity publication helper is missing")
    direct_clobber = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "os"
        and node.func.attr == "rename"
        for node in ast.walk(promote)
    )
    if direct_clobber:
        _stop("parity promotion still uses clobbering os.rename")
    return {
        "parity_executables_pinned": True,
        "parity_recovery_checks_derived": recovery_derived,
        "parity_promotion_non_clobber": True,
    }


def _source_closure(plan: Mapping[str, Any]) -> set[str]:
    groups = (
        "parent_sources",
        "worker_sources",
        "parity_sources",
        "test_sources",
        "independent_audit_sources",
    )
    values: list[str] = []
    for group in groups:
        entries = plan.get(group)
        if type(entries) is not list or any(type(item) is not str for item in entries):
            _stop(f"invalid source closure group: {group}")
        values.extend(entries)
    if len(values) != len(set(values)):
        _stop("planned source closure contains duplicates")
    if plan["independent_audit_sources"] != [SELF, SELF_TEST]:
        _stop("independent audit source closure changed")
    return set(values)


def static_audit(repo: Path) -> tuple[dict[str, bool], dict[str, str], Any, Any, dict[str, Any]]:
    repo = repo.resolve(strict=True)
    contract_bytes = _secure_read(repo, CONTRACT)
    plan_bytes = _secure_read(repo, PLAN)
    runner, runner_bytes = _load_snapshot_module(
        repo, RUNNER, "_v412_independent_runner"
    )
    parity, parity_bytes = _load_snapshot_module(
        repo, PARITY, "_v412_independent_parity"
    )
    worker_bytes = _secure_read(repo, WORKER)
    strict_bytes = _secure_read(repo, STRICT_STORES)
    worker_profile_bytes = _secure_read(repo, WORKER_PROFILE)
    parity_profile_bytes = _secure_read(repo, PARITY_PROFILE)
    plan = _parse_json(plan_bytes, "V4.12 retrieval plan")
    try:
        runner.validate_plan(plan)
    except Exception as exc:
        _stop(f"runner rejects the frozen plan: {exc}")
    contract = contract_bytes.decode("utf-8")
    _require_markers(
        contract,
        (
            "une requête CRM sûre",
            "au plus 100 SIRET actifs",
            "Le worker doit être terminé",
            "mini-publication limitée à des fixtures synthétiques",
            "Le ranker, le decider, le risk model, l'accepteur et le test final restent",
        ),
        "retrieval contract",
    )
    closure = _source_closure(plan)
    missing = [
        relative
        for relative in sorted(closure)
        if not (repo / _safe_relative(relative)).is_file()
        or (repo / relative).is_symlink()
    ]
    if missing:
        _stop(f"planned source closure is absent or unsafe: {missing}")
    worker_ast = _audit_worker_ast(worker_bytes)
    if worker_ast["retrieval"] != plan["retrieval"]:
        _stop("worker retrieval policy differs from the frozen plan")
    if worker_ast["tfidf_cache"] != plan["tfidf_cache"]:
        _stop("worker TF-IDF policy differs from the frozen plan")
    if runner.STATUS_SCHEMA != parity.STATUS_SCHEMA or runner.CANDIDATE_SCHEMA != parity.CANDIDATE_SCHEMA:
        _stop("worker/parity Arrow schemas diverge")
    if (
        runner.WORKER_INTEGRITY_SCHEMA != parity.WORKER_INTEGRITY_SCHEMA
        or runner.WORKER_MANIFEST_SCHEMA != parity.WORKER_MANIFEST_SCHEMA
        or runner.INTEGRITY_KEYS != parity.WORKER_INTEGRITY_KEYS
        or runner.DECLARATIONS != parity.WORKER_DECLARATIONS
    ):
        _stop("worker manifest/integrity contracts diverge")
    worker_profile = worker_profile_bytes.decode("utf-8")
    parity_profile = parity_profile_bytes.decode("utf-8")
    _require_markers(
        worker_profile,
        (
            "(deny default)",
            "(deny network*)",
            "(deny process-fork)",
            "@@ALLOWED_READ_RULES@@",
            "@@EXPLICIT_DENY_RULES@@",
            'param "RUN_OUTPUT"',
            'param "RUN_TMP"',
        ),
        "worker sandbox",
    )
    _require_markers(
        parity_profile,
        (
            "(deny default)",
            "(deny network*)",
            "(deny process-fork)",
            'param "FORBIDDEN_ORACLE"',
            'param "FORBIDDEN_ORACLE_AUDIT"',
            'param "FORBIDDEN_HISTORICAL"',
            'param "FORBIDDEN_MODEL"',
            'param "FORBIDDEN_STORE"',
            'param "WRITE_SENTINEL"',
        ),
        "parity sandbox",
    )
    rendered = runner.render_profile(
        worker_profile,
        allowed_files=[repo / PLAN],
        forbidden_roots=[Path(path) for path in plan["forbidden_worker_roots"]],
        system_roots=[Path("/System"), Path("/usr"), Path("/opt/homebrew")],
        devices=[Path("/dev/null"), Path("/dev/urandom"), Path("/dev/fd")],
        metadata_extra=[repo],
    )
    if "@@" in rendered:
        _stop("rendered worker sandbox still contains a marker")
    lock = {
        "worker_lock_projection_sha256": "1" * 64,
        "source_hashes": {
            RUNNER: "2" * 64,
            **{relative: "3" * 64 for relative in plan["worker_sources"]},
        },
    }
    worker_spec = runner._worker_run_spec(
        plan,
        lock,
        {"schema_version": "synthetic-gate-a"},
        policy_sha="4" * 64,
    )
    worker_spec_payload = runner.canonical_json(worker_spec).lower()
    forbidden_fragments = (
        b"/oracles/",
        b"/datasets/",
        b"/models/",
        b"is_ground_truth",
        plan["historical_parity"]["candidate_payload_sha256"].encode(),
        plan["historical_parity"]["status_payload_sha256"].encode(),
    )
    if any(fragment in worker_spec_payload for fragment in forbidden_fragments):
        _stop("worker run-spec leaks oracle/model/historical information")
    _source_order_check(
        runner_bytes.decode("utf-8"),
        parity_bytes.decode("utf-8"),
    )
    checks = {
        key: True
        for key in STATIC_CHECKS
        if not key.startswith("parity_")
    }
    checks.update(
        _audit_controller_p1(parity, parity_bytes.decode("utf-8"))
    )
    source_payloads = {
        CONTRACT: contract_bytes,
        PLAN: plan_bytes,
        WORKER: worker_bytes,
        STRICT_STORES: strict_bytes,
        RUNNER: runner_bytes,
        PARITY: parity_bytes,
        WORKER_PROFILE: worker_profile_bytes,
        PARITY_PROFILE: parity_profile_bytes,
        SELF: _secure_read(repo, SELF),
        SELF_TEST: _secure_read(repo, SELF_TEST),
    }
    for relative in closure:
        if relative not in source_payloads:
            source_payloads[relative] = _secure_read(repo, relative)
    hashes = {
        relative: hashlib.sha256(payload).hexdigest()
        for relative, payload in source_payloads.items()
    }
    return checks, hashes, runner, parity, plan


def _write_worker_fixture(root: Path, runner: Any, build_id: str) -> list[str]:
    query_ids = ["synthetic-q1", "synthetic-q2"]
    status = pa.Table.from_pylist(
        [
            {"query_id": query_ids[0], "candidate_count": 2},
            {"query_id": query_ids[1], "candidate_count": 0},
        ],
        schema=runner.STATUS_SCHEMA,
    )
    candidates = pa.Table.from_pylist(
        [
            {
                "query_id": query_ids[0],
                "candidate_rank": 1,
                "candidate_siret": "11111111100011",
            },
            {
                "query_id": query_ids[0],
                "candidate_rank": 2,
                "candidate_siret": "22222222200022",
            },
        ],
        schema=runner.CANDIDATE_SCHEMA,
    )
    pq.write_table(status, root / "query_status.parquet", compression="zstd")
    pq.write_table(candidates, root / "candidates_top100.parquet", compression="zstd")
    candidate_payload = (
        query_ids[0].encode()
        + b"\0"
        + b"11111111100011\0"
        + b"1\n"
        + query_ids[0].encode()
        + b"\0"
        + b"22222222200022\0"
        + b"2\n"
    )
    status_payload = (
        query_ids[0].encode()
        + b"\0"
        + b"2\n"
        + query_ids[1].encode()
        + b"\0"
        + b"0\n"
    )
    integrity = {
        "schema_version": runner.WORKER_INTEGRITY_SCHEMA,
        "worker_build_id": build_id,
        "query_count": 2,
        "candidate_count": 2,
        "minimum_pool_size": 0,
        "maximum_pool_size": 2,
        "under_ceiling_query_count": 2,
        "empty_query_count": 1,
        "lookup_missing_count": 0,
        "candidate_payload_bytes": len(candidate_payload),
        "candidate_payload_sha256": hashlib.sha256(candidate_payload).hexdigest(),
        "status_payload_bytes": len(status_payload),
        "status_payload_sha256": hashlib.sha256(status_payload).hexdigest(),
        "sandbox_checks": {key: True for key in runner.SANDBOX_CHECK_KEYS},
        "peak_rss_bytes": 1,
        "durations_ns": {key: 0 for key in runner.DURATION_KEYS},
        "declarations": dict(runner.DECLARATIONS),
    }
    (root / "integrity.json").write_bytes(runner.canonical_json(integrity))
    return query_ids


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_record(path: Path) -> dict[str, Any]:
    info = os.lstat(path)
    if not stat.S_ISREG(info.st_mode):
        _stop(f"manifest fixture member is not regular: {path.name}")
    return {
        "path": path.name,
        "size_bytes": info.st_size,
        "sha256": _sha(path),
    }


def _parity_spec(
    root: Path,
    worker_root: Path,
    query_ids: Sequence[str],
    runner: Any,
    parity: Any,
    worker_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], Path, Path]:
    safe_queries = root / "safe_queries.parquet"
    rows = [
        {
            "query_id": query_id,
            "crm_name": "SYNTHETIC",
            "crm_address": "1 RUE SYNTHETIQUE",
            "crm_postcode": "75001",
            "crm_city": "PARIS",
            "crm_insee": "75056",
        }
        for query_id in query_ids
    ]
    pq.write_table(
        pa.Table.from_pylist(rows, schema=parity.SAFE_QUERY_SCHEMA),
        safe_queries,
        compression="zstd",
    )
    safe_manifest = root / "safe_manifest.json"
    safe_build_id = "synthetic-safe-input"
    safe_manifest.write_bytes(canonical_json({"build_id": safe_build_id}))
    integrity = _parse_json(
        (worker_root / "integrity.json").read_bytes(),
        "synthetic worker integrity",
    )
    expected_keys = {
        "query_count",
        "candidate_count",
        "minimum_pool_size",
        "maximum_pool_size",
        "under_ceiling_query_count",
        "empty_query_count",
        "candidate_payload_bytes",
        "candidate_payload_sha256",
        "status_payload_bytes",
        "status_payload_sha256",
    }
    expected = {key: integrity[key] for key in expected_keys}
    file_paths = {
        name: str((worker_root / name).resolve())
        for name in parity.WORKER_FILE_NAMES
    }
    file_hashes = {name: _sha(worker_root / name) for name in file_paths}
    query_payload = b"".join(query_id.encode() + b"\n" for query_id in query_ids)
    worker_manifest_path = worker_root / "manifest.json"
    parity_profile = root / "synthetic_parity.sb"
    parity_profile.write_text("(version 1)\n(deny default)\n", encoding="utf-8")
    sandbox_executable = Path("/usr/bin/sandbox-exec")
    python_executable = parity._default_python_executable()
    if not sandbox_executable.is_file() or not python_executable.is_file():
        _stop("pinned executables required by the parity fixture are absent")
    spec = {
        "schema_version": parity.RUN_SPEC_SCHEMA,
        "worker_build_id": worker_manifest["worker_build_id"],
        "worker_manifest_path": str(worker_manifest_path.resolve()),
        "worker_manifest_sha256": _sha(worker_manifest_path),
        "worker_file_paths": file_paths,
        "worker_file_hashes": file_hashes,
        "safe_input_build_id": safe_build_id,
        "safe_queries_path": str(safe_queries.resolve()),
        "safe_queries_sha256": _sha(safe_queries),
        "safe_manifest_path": str(safe_manifest.resolve()),
        "safe_manifest_sha256": _sha(safe_manifest),
        "safe_query_id_payload_sha256": hashlib.sha256(query_payload).hexdigest(),
        "expected": expected,
        "git_commit": "0" * 40,
        "lock_sha256": "1" * 64,
        "parity_source_hashes": {parity.SOURCE_RELATIVE_PATH: "2" * 64},
        "parity_profile_sha256": _sha(parity_profile),
        "sandbox_executable_path": str(sandbox_executable),
        "sandbox_executable_sha256": _sha(sandbox_executable),
        "python_executable_path": str(python_executable),
        "python_executable_sha256": _sha(python_executable),
        "audit_root_path_sha256": parity.path_commitment(
            (root / "synthetic-audit-root").resolve()
        ),
        "runtime": parity.runtime_identity(),
        "temp_root": str((root / "parity-tmp").resolve()),
        "max_rss_bytes": 8 * 1024**3,
        "declarations": dict(parity.DECLARATIONS),
    }
    parity.validate_run_spec(spec)
    return spec, safe_queries, safe_manifest


def _evaluate_synthetic_parity(
    spec: Mapping[str, Any],
    safe_queries: Path,
    safe_manifest: Path,
    worker_root: Path,
    parity: Any,
) -> dict[str, Any]:
    run_spec_payload = parity.canonical_json(spec)
    parity_id = parity.parity_build_id(
        spec, hashlib.sha256(run_spec_payload).hexdigest()
    )
    paths = (
        safe_queries,
        safe_manifest,
        worker_root / "manifest.json",
        worker_root / "integrity.json",
        worker_root / "query_status.parquet",
        worker_root / "candidates_top100.parquet",
    )
    descriptors = [os.open(path, os.O_RDONLY) for path in paths]
    try:
        return parity.evaluate_from_fds(
            spec,
            parity_id=parity_id,
            safe_queries_fd=descriptors[0],
            safe_manifest_fd=descriptors[1],
            worker_manifest_fd=descriptors[2],
            worker_integrity_fd=descriptors[3],
            status_fd=descriptors[4],
            candidates_fd=descriptors[5],
            sandbox_checks={key: True for key in parity.SANDBOX_CHECK_KEYS},
        )
    finally:
        for descriptor in descriptors:
            os.close(descriptor)


def _write_parity_stage(
    stage: Path,
    report: Mapping[str, Any],
    spec: Mapping[str, Any],
    parity: Any,
) -> str:
    stage.mkdir()
    os.chmod(stage, 0o700)
    parity._write_exclusive(stage / "parity.json", parity.canonical_json(report))
    run_spec_sha = hashlib.sha256(parity.canonical_json(spec)).hexdigest()
    provenance = {
        "schema_version": parity.PROVENANCE_SCHEMA,
        "parity_build_id": report["parity_build_id"],
        "worker_build_id": spec["worker_build_id"],
        "git_commit": spec["git_commit"],
        "parity_source_hashes": spec["parity_source_hashes"],
        "lock_sha256": spec["lock_sha256"],
        "parity_run_spec_sha256": run_spec_sha,
        "worker_manifest_sha256": spec["worker_manifest_sha256"],
        "runtime": spec["runtime"],
        "declarations": dict(parity.DECLARATIONS),
    }
    parity._write_exclusive(
        stage / "provenance.json", parity.canonical_json(provenance)
    )
    manifest = {
        "schema_version": parity.MANIFEST_SCHEMA,
        "parity_build_id": report["parity_build_id"],
        "worker_build_id": spec["worker_build_id"],
            "files": sorted(
                [
                    _manifest_record(stage / "parity.json"),
                    _manifest_record(stage / "provenance.json"),
                ],
            key=lambda row: row["path"].encode(),
        ),
        "runtime": spec["runtime"],
        "declarations": dict(parity.DECLARATIONS),
        "verdict": report["verdict"],
    }
    parity._write_exclusive(stage / "manifest.json", parity.canonical_json(manifest))
    return run_spec_sha


def _publish_parity_fixture(
    root: Path,
    report: Mapping[str, Any],
    spec: Mapping[str, Any],
    parity: Any,
) -> Path:
    stage = root / "parity-stage"
    run_spec_sha = _write_parity_stage(stage, report, spec, parity)
    parity.validate_published_audit(
        stage,
        parity_id=report["parity_build_id"],
        spec=spec,
        run_spec_sha256=run_spec_sha,
        limit=8 * 1024**3,
    )
    publication = root / "parity-publication"
    publication.mkdir()
    pending = publication / f".pending-{report['parity_build_id']}"
    final = publication / report["parity_build_id"]
    parity._promote(stage, pending)
    parity._promote(pending, final)
    parity.validate_published_audit(
        final,
        parity_id=report["parity_build_id"],
        spec=spec,
        run_spec_sha256=run_spec_sha,
        limit=8 * 1024**3,
    )
    return final


def synthetic_audit(runner: Any, parity: Any) -> dict[str, Any]:
    checks = {key: False for key in SYNTHETIC_CHECKS}
    trace: list[str] = []
    with tempfile.TemporaryDirectory(prefix="v412-independent-audit-") as raw:
        root = Path(raw).resolve()
        worker_build_id = "b" * 64
        worker_stage = root / "worker-stage"
        worker_stage.mkdir()
        query_ids = _write_worker_fixture(worker_stage, runner, worker_build_id)
        integrity = runner._validate_outputs(
            worker_stage,
            worker_build_id,
            query_ids,
            8 * 1024**3,
        )
        checks["worker_payloads"] = True
        worker_manifest = {
            "schema_version": runner.WORKER_MANIFEST_SCHEMA,
            "worker_build_id": worker_build_id,
            "safe_input_build_id": "synthetic-safe-input",
            "strict_stores_build_id": "synthetic-strict-stores",
            "files": runner._runtime_file_records(worker_stage, 8 * 1024**3),
            "runtime": parity.runtime_identity(),
            "declarations": dict(runner.DECLARATIONS),
            "verdict": runner.SEALED,
        }
        (worker_stage / "manifest.json").write_bytes(
            runner.canonical_json(worker_manifest)
        )
        if worker_manifest["files"] != runner._runtime_file_records(
            worker_stage, 8 * 1024**3
        ):
            _stop("synthetic worker manifest is not self-consistent")
        checks["worker_manifest"] = True
        worker_publication = root / "worker-publication"
        worker_publication.mkdir()
        pending = worker_publication / f".pending-{worker_build_id}"
        final_worker = worker_publication / worker_build_id
        runner._promote(worker_stage, pending)
        runner._promote(pending, final_worker)
        synthetic_publication_plan = {
            "safe_input": {"build_id": "synthetic-safe-input"},
            "prerequisite": {"build_id": "synthetic-strict-stores"},
            "runtime": parity.runtime_identity(),
        }
        runner._validate_runtime_publication(
            final_worker,
            worker_build_id,
            synthetic_publication_plan,
            8 * 1024**3,
        )
        trace.append("worker_final")
        checks["worker_publication"] = True

        spec, safe_queries, safe_manifest = _parity_spec(
            root,
            final_worker,
            query_ids,
            runner,
            parity,
            worker_manifest,
        )
        if "worker_final" not in trace:
            _stop("parity started before the worker publication was final")
        trace.append("parity_started")
        report = _evaluate_synthetic_parity(
            spec, safe_queries, safe_manifest, final_worker, parity
        )
        if report["verdict"] != parity.GO:
            _stop("synthetic parity did not produce GO")
        checks["parity_payloads"] = True
        checks["worker_precedes_parity"] = trace == [
            "worker_final",
            "parity_started",
        ]
        final_parity = _publish_parity_fixture(root, report, spec, parity)
        checks["parity_manifest"] = True
        checks["parity_publication"] = final_parity.is_dir()

        inconsistent_report = copy.deepcopy(report)
        inconsistent_report["candidate_payload_sha256"] = "f" * 64
        inconsistent_stage = root / "inconsistent-parity"
        inconsistent_run_spec_sha = _write_parity_stage(
            inconsistent_stage,
            inconsistent_report,
            spec,
            parity,
        )
        try:
            parity.validate_published_audit(
                inconsistent_stage,
                parity_id=report["parity_build_id"],
                spec=spec,
                run_spec_sha256=inconsistent_run_spec_sha,
                limit=8 * 1024**3,
            )
        except parity.ParityStopped:
            checks["parity_recovery_mutation_rejected"] = True
        else:
            _stop("recovery accepted a parity report with forged stored checks")

        race_source = root / "destination-race-source"
        race_source.mkdir()
        (race_source / "value").write_bytes(b"trusted")
        race_destination = root / "destination-race-final"
        original_seal = parity._seal_tree

        def create_destination_after_precheck(source: Path) -> None:
            race_destination.mkdir()
            (race_destination / "sentinel").write_bytes(b"do-not-clobber")
            original_seal(source)

        parity._seal_tree = create_destination_after_precheck
        try:
            try:
                parity._promote(race_source, race_destination)
            except (parity.ParityStopped, FileExistsError, OSError):
                pass
            else:
                _stop("parity promotion clobbered a raced destination")
        finally:
            parity._seal_tree = original_seal
        if (
            not race_destination.is_dir()
            or not (race_destination / "sentinel").is_file()
            or (race_destination / "sentinel").read_bytes() != b"do-not-clobber"
        ):
            _stop("parity promotion did not preserve the raced destination")
        checks["parity_destination_race_rejected"] = True

        substitution_source = root / "pending-substitution-source"
        substitution_source.mkdir()
        (substitution_source / "value").write_bytes(b"trusted")
        substitution_displaced = root / "pending-substitution-displaced"
        substitution_destination = root / "pending-substitution-final"

        def substitute_after_seal(source: Path) -> None:
            original_seal(source)
            os.rename(source, substitution_displaced)
            source.mkdir()
            (source / "value").write_bytes(b"substituted")

        parity._seal_tree = substitute_after_seal
        try:
            try:
                parity._promote(
                    substitution_source,
                    substitution_destination,
                )
            except (parity.ParityStopped, FileExistsError, OSError):
                pass
            else:
                _stop("parity promotion accepted a substituted pending root")
        finally:
            parity._seal_tree = original_seal
        if substitution_destination.exists():
            _stop("substituted pending root reached the final destination")
        checks["parity_pending_substitution_rejected"] = True

        mutation = root / "mutation"
        mutation.mkdir()
        _write_worker_fixture(mutation, runner, worker_build_id)
        table = pq.read_table(mutation / "query_status.parquet")
        table = table.set_column(
            1,
            "candidate_count",
            pa.array([1, 0], type=pa.uint8()),
        )
        pq.write_table(table, mutation / "query_status.parquet")
        try:
            runner._validate_outputs(
                mutation, worker_build_id, query_ids, 8 * 1024**3
            )
        except runner.RetrievalRunStopped:
            checks["mutation_rejected"] = True
        else:
            _stop("mutated synthetic worker output was accepted")

        symlink_stage = root / "symlink-stage"
        symlink_stage.mkdir()
        target = root / "symlink-target"
        target.write_bytes(b"target")
        (symlink_stage / "link").symlink_to(target)
        try:
            runner._promote(symlink_stage, root / "symlink-final")
        except runner.RetrievalRunStopped:
            checks["symlink_rejected"] = True
        else:
            _stop("symlink publication was accepted")

        anchored = root / "anchored"
        anchored.write_bytes(b"original")
        expected = hashlib.sha256(b"original").hexdigest()
        descriptor, before = parity._open_anchored(
            anchored, expected, 8 * 1024**3
        )
        replacement = root / "replacement"
        replacement.write_bytes(b"replacement")
        os.replace(replacement, anchored)
        try:
            if (
                parity._read_fd(descriptor, 8 * 1024**3) != b"original"
                or parity._snapshot_fd(descriptor, 8 * 1024**3) != before
                or anchored.read_bytes() != b"replacement"
            ):
                _stop("anchored FD did not survive path substitution safely")
        finally:
            os.close(descriptor)
        checks["anchored_fd_survives_path_substitution"] = True

        if set(checks) != SYNTHETIC_CHECKS or any(value is not True for value in checks.values()):
            _stop("synthetic audit check closure incomplete")
        return {
            "checks": checks,
            "query_count": integrity["query_count"],
            "candidate_count": integrity["candidate_count"],
            "worker_build_id": worker_build_id,
            "parity_build_id": report["parity_build_id"],
            "trace": trace,
        }


def audit_repository(repo: Path) -> dict[str, Any]:
    static, hashes, runner, parity, _plan = static_audit(repo)
    synthetic = synthetic_audit(runner, parity)
    if set(static) != STATIC_CHECKS or any(value is not True for value in static.values()):
        _stop("static audit check closure incomplete")
    report = {
        "schema_version": SCHEMA,
        "source_hashes": hashes,
        "static_checks": static,
        "synthetic": synthetic,
        "forbidden_runtime_inputs_opened": False,
        "verdict": GO,
    }
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        report = audit_repository(args.repo)
    except IndependentAuditStopped as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
