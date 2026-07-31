"""Execution-lock validation shared by the V4.12 service parent and workers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any, Mapping

from .v412_service_bundle import _capture_exact


STOP = "STOP_V412_SERVICE_INTEGRITY"
REPOSITORY = Path("/Users/nathanjullia/Documents/Projets/SIRETO")
LOCK_PATH = REPOSITORY / "config/v4_12_service_parity_execution_lock.json"
OUTPUT_ROOT = Path(
    "/Volumes/CATNAT_DATA/SIRETO_RECALL100/runs/"
    "v4_12_service_parity_latency"
)
LOCK_SCHEMA = "sireto-v4.12-service-parity-execution-lock-1"
LOCK_PURPOSE = "RUN_V412_SERVICE_PARITY_LATENCY"
LOCK_VERDICT = "GO_V412_SERVICE_IMPLEMENTATION"
SOURCE_CLOSURE = (
    "docs/v4_12_service_parity_latency_contract.md",
    "scripts/__init__.py",
    "scripts/build_benchmark_v2_qualification.py",
    "scripts/build_benchmark_v3_evidence.py",
    "scripts/build_benchmark_v4_current_snapshot.py",
    "scripts/freeze_v9_closed_benchmark.py",
    "scripts/run_v9_retrieval_experiment.py",
    "scripts/run_v412_persistent_service_bootstrap.py",
    "scripts/run_v412_persistent_service_worker.py",
    "scripts/run_v412_service_parity_latency.py",
    "src/xgb_matcher/__init__.py",
    "src/xgb_matcher/blocking.py",
    "src/xgb_matcher/candidates.py",
    "src/xgb_matcher/contracts.py",
    "src/xgb_matcher/features.py",
    "src/xgb_matcher/fusion.py",
    "src/xgb_matcher/naming.py",
    "src/xgb_matcher/partitioned_store.py",
    "src/xgb_matcher/retrieval.py",
    "src/xgb_matcher/retrieval_config.py",
    "src/xgb_matcher/tfidf_cache.py",
    "src/xgb_matcher/timing.py",
    "src/xgb_matcher/v411_acceptor.py",
    "src/xgb_matcher/v411_scene.py",
    "src/xgb_matcher/v412_direct_evidence.py",
    "src/xgb_matcher/v412_evidence_service.py",
    "src/xgb_matcher/v412_service.py",
    "src/xgb_matcher/v412_service_bundle.py",
    "src/xgb_matcher/v412_service_execution_lock.py",
    "src/xgb_matcher/v412_service_parity.py",
    "src/xgb_matcher/v412_service_retrieval.py",
    "src/xgb_matcher/v412_service_run.py",
    "src/xgb_matcher/v412_service_worker.py",
    "src/xgb_matcher/v412_strict_stores.py",
    "src/xgb_matcher/v412_unit_retrieval.py",
    "src/xgb_matcher/v49_site_function.py",
    "src/xgb_matcher/v9_dataset.py",
    "src/xgb_matcher/v9_features.py",
)
INPUT_HASHES = {
    "safe_queries": (
        "70ded26776bfd56c96501c6033e0e322a6dd11ed296c3309ad89bd9deec84cf9"
    ),
    "reference_manifest": (
        "cbcb3303107cd00f895561b49b8ad3a26e5c8e3df8a07777817e7a6ed97f2340"
    ),
}
_HEX = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _fail(detail: str) -> None:
    raise ValueError(f"{STOP}: {detail}")


def runtime_identity() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": importlib.metadata.version("numpy"),
        "pandas": importlib.metadata.version("pandas"),
        "pyarrow": importlib.metadata.version("pyarrow"),
        "scikit_learn": importlib.metadata.version("scikit-learn"),
        "xgboost": importlib.metadata.version("xgboost"),
        "duckdb": importlib.metadata.version("duckdb"),
        "joblib": importlib.metadata.version("joblib"),
        "scipy": importlib.metadata.version("scipy"),
    }


def _json(payload: bytes) -> dict[str, Any]:
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                _fail("duplicate execution-lock key")
            result[key] = value
        return result

    try:
        value = json.loads(payload, object_pairs_hook=object_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"invalid execution lock: {exc}")
    if type(value) is not dict:
        _fail("execution lock must be an object")
    return value


def validate_execution_lock(
    *,
    expected_sha256: str | None = None,
    verify_git: bool,
) -> tuple[dict[str, Any], str]:
    if expected_sha256 is not None and _HEX.fullmatch(expected_sha256) is None:
        _fail("invalid expected execution-lock hash")
    if not LOCK_PATH.is_file() or LOCK_PATH.is_symlink():
        _fail("execution lock absent or linked")
    payload = LOCK_PATH.read_bytes()
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        _fail("execution-lock hash changed")
    lock = _json(payload)
    expected_fields = {
        "schema_version",
        "purpose",
        "audit_verdict",
        "git_commit",
        "source_hashes",
        "input_hashes",
        "runtime",
        "output_root",
        "query_count",
        "max_rss_bytes",
    }
    source_hashes = lock.get("source_hashes")
    if (
        set(lock) != expected_fields
        or lock.get("schema_version") != LOCK_SCHEMA
        or lock.get("purpose") != LOCK_PURPOSE
        or lock.get("audit_verdict") != LOCK_VERDICT
        or type(source_hashes) is not dict
        or set(source_hashes) != set(SOURCE_CLOSURE)
        or lock.get("input_hashes") != INPUT_HASHES
        or lock.get("runtime") != runtime_identity()
        or lock.get("output_root") != str(OUTPUT_ROOT)
        or lock.get("query_count") != 1456
        or lock.get("max_rss_bytes") != 8 * 1024**3
    ):
        _fail("execution-lock contract changed")
    commit = lock.get("git_commit")
    if type(commit) is not str or _COMMIT.fullmatch(commit) is None:
        _fail("invalid locked commit")
    for relative in SOURCE_CLOSURE:
        digest = source_hashes.get(relative)
        if type(digest) is not str or _HEX.fullmatch(digest) is None:
            _fail(f"invalid source hash: {relative}")
        path = REPOSITORY / relative
        _capture_exact(path, digest)
        if verify_git:
            result = subprocess.run(
                ["git", "show", f"{commit}:{relative}"],
                cwd=REPOSITORY,
                check=False,
                capture_output=True,
            )
            if (
                result.returncode != 0
                or hashlib.sha256(result.stdout).hexdigest() != digest
            ):
                _fail(f"locked commit does not bind source: {relative}")
    return lock, observed_sha256


def validate_loaded_repository_modules(
    source_hashes: Mapping[str, str],
) -> None:
    loaded: set[str] = set()
    repository = REPOSITORY.resolve()
    for module in tuple(sys.modules.values()):
        raw_path = getattr(module, "__file__", None)
        if not raw_path:
            continue
        try:
            path = Path(raw_path).resolve()
        except OSError:
            _fail("cannot resolve an executed module")
        if repository in path.parents and path.suffix == ".py":
            loaded.add(str(path.relative_to(repository)))
    unsealed = loaded - set(source_hashes)
    if unsealed:
        _fail(f"executed repository source is not sealed: {sorted(unsealed)}")
    for relative in loaded:
        _capture_exact(REPOSITORY / relative, source_hashes[relative])


__all__ = [
    "INPUT_HASHES",
    "LOCK_PATH",
    "OUTPUT_ROOT",
    "SOURCE_CLOSURE",
    "runtime_identity",
    "validate_execution_lock",
    "validate_loaded_repository_modules",
]
