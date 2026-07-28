#!/usr/bin/env python3
"""Build the sealed, label-free V4.12 direct-evidence signal.

The build phase deserializes only the exact CRM query projection and the
frozen geographic partitions.  It never opens splits, labels, ranker output,
scenes, models, acceptor output, or a consumed challenge.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_benchmark_v4_current_snapshot import (  # noqa: E402
    POLICY_VERSION,
    ActivePartitionIndex,
    _load_partition,
    _planned_partition_key,
    build_active_partition_index,
    find_direct_active_candidates,
)
from src.xgb_matcher.partitioned_store import (  # noqa: E402
    PartitionedCandidateStore,
)
from src.xgb_matcher.v412_direct_evidence import (  # noqa: E402
    CANDIDATE_EVIDENCE_COLUMNS,
    QUERY_EVIDENCE_COLUMNS,
    build_evidence_frames,
    candidate_evidence_record,
    query_evidence_record,
    validate_evidence,
)
SCHEMA_VERSION = "sireto-v4.12-direct-evidence-1"
LOCK_SCHEMA_VERSION = "sireto-v4.12-evidence-execution-lock-1"
ALLOWLIST_SCHEMA_VERSION = "sireto-v4.12-development-inputs-1"
DENYLIST_SCHEMA_VERSION = "sireto-v4.12-forbidden-artifacts-1"
PURPOSE = "BUILD_V412_DIRECT_EVIDENCE"
AUDIT_VERDICT = "GO_BUILD_V412_EVIDENCE"
PRESEAL_QUERY_COLUMNS = [
    "query_id",
    "crm_name",
    "crm_address",
    "crm_postcode",
    "crm_city",
    "crm_insee",
]
EXPECTED_QUERY_COUNT = 7003
EXPECTED_SNAPSHOT_SHA256 = (
    "c91180cc5bae86948dd57d752c9bae45e58cc64653e99d5a9357664b67300845"
)
EXPECTED_PARTITIONS_SIGNATURE = (
    "2f6668f60da8bc9fe52b683b32ef35641803679c01f8c8fd124e2e86a41e2b82"
)
DEFAULT_ALLOWLIST = Path("config/v4_12_development_inputs.json")
DEFAULT_DENYLIST = Path("config/v4_12_forbidden_artifacts.json")
DEFAULT_CONTRACT = Path("docs/v4_12_direct_evidence_gate_contract.md")
OUTPUT_FILENAMES = {
    "query_evidence.parquet",
    "candidate_evidence.parquet",
    "timing.json",
    "integrity.json",
}
POLICY_SOURCE_HASHES = {
    "scripts/build_benchmark_v4_current_snapshot.py": (
        "b0451766575f0023d42d598caa23aebb0e81cff5fcb60f5071da64c9b3f0b19b"
    ),
    "scripts/build_benchmark_v3_evidence.py": (
        "9ebf636101de6cd73e4079fbcc14b012e655fdd6ff08910e00127ee915718dcc"
    ),
    "src/xgb_matcher/blocking.py": (
        "e6a0fded2f6496c9f4e901d8ba4fca1b912f5410c3c506a170c434ec02a55736"
    ),
    "src/xgb_matcher/features.py": (
        "839f55b0d8c56e22e75758db88647c910fd8158039d1b0175f9c818e5ac0b191"
    ),
    "src/xgb_matcher/naming.py": (
        "b7ef59a8cb7529179567f6e3ffe3b64757383a9e449a0110886abe640a1b5fc1"
    ),
    "src/xgb_matcher/partitioned_store.py": (
        "181d1c8a56539f6b36e01d9fc040a7fb4135e28a0b10147775abd5b33837a39f"
    ),
}
LOCK_FIELDS = {
    "schema_version",
    "purpose",
    "audit_verdict",
    "git_commit",
    "source_hashes",
    "input_paths",
    "input_hashes",
    "runtime",
    "snapshot_sha256",
    "partitions_signature",
}
SOURCE_PATHS = [
    "scripts/__init__.py",
    "scripts/build_v412_direct_evidence.py",
    "src/xgb_matcher/v412_direct_evidence.py",
    "src/xgb_matcher/__init__.py",
    "docs/v4_12_direct_evidence_gate_contract.md",
    "config/v4_12_development_inputs.json",
    "config/v4_12_forbidden_artifacts.json",
    *POLICY_SOURCE_HASHES,
    # Import closure of build_benchmark_v4_current_snapshot.py.  Some are
    # imported by that historical module even though V4.12 calls only its
    # five frozen direct-evidence helpers.
    "scripts/build_benchmark_v2_qualification.py",
    "scripts/freeze_v9_closed_benchmark.py",
    "scripts/run_v9_retrieval_experiment.py",
    "src/xgb_matcher/candidates.py",
    "src/xgb_matcher/contracts.py",
    "src/xgb_matcher/fusion.py",
    "src/xgb_matcher/retrieval.py",
    "src/xgb_matcher/retrieval_config.py",
    "src/xgb_matcher/tfidf_cache.py",
    "src/xgb_matcher/timing.py",
    "src/xgb_matcher/v9_dataset.py",
    "src/xgb_matcher/v9_features.py",
    *[
        str(path.relative_to(Path(__file__).resolve().parent.parent))
        for path in sorted(
            (
                Path(__file__).resolve().parent.parent
                / "src"
                / "xgb_matcher"
            ).glob("*.py")
        )
    ],
]
SOURCE_PATHS = list(dict.fromkeys(SOURCE_PATHS))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _path_signature(path: Path) -> str:
    """Exact local copy of the frozen directory-signature algorithm."""

    path = Path(path)
    if path.is_file():
        return file_sha256(path)
    root_manifest = path / "manifest.json"
    if root_manifest.exists():
        return file_sha256(root_manifest)
    candidates = sorted(
        candidate for candidate in path.rglob("*") if candidate.is_file()
    )
    digest = hashlib.sha256()
    for candidate in candidates:
        digest.update(str(candidate.relative_to(path)).encode())
        digest.update(file_sha256(candidate).encode())
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _runtime() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": importlib.metadata.version("numpy"),
        "pandas": importlib.metadata.version("pandas"),
        "pyarrow": importlib.metadata.version("pyarrow"),
        "rapidfuzz": importlib.metadata.version("rapidfuzz"),
        "scikit-learn": importlib.metadata.version("scikit-learn"),
        "scipy": importlib.metadata.version("scipy"),
        "duckdb": importlib.metadata.version("duckdb"),
    }


def _source_hashes(repo_root: Path) -> dict[str, str]:
    observed = {
        relative: file_sha256(repo_root / relative)
        for relative in SOURCE_PATHS
    }
    for relative, expected in POLICY_SOURCE_HASHES.items():
        if observed.get(relative) != expected:
            raise ValueError(
                f"STOP_V412_POLICY_INTEGRITY: source changed {relative}"
            )
    return observed


def validate_execution_lock(
    lock_path: Path,
    *,
    allowlist_path: Path,
    denylist_path: Path,
    verify_git: bool = True,
) -> tuple[dict[str, Any], str]:
    """Validate the external authorization before parsing any data policy."""

    repo_root = Path(__file__).resolve().parent.parent
    raw = Path(lock_path).read_bytes()
    lock = json.loads(raw)
    if set(lock) != LOCK_FIELDS:
        raise ValueError("STOP_V412_LOCK: execution lock fields changed")
    if (
        lock.get("schema_version") != LOCK_SCHEMA_VERSION
        or lock.get("purpose") != PURPOSE
        or lock.get("audit_verdict") != AUDIT_VERDICT
    ):
        raise ValueError("STOP_V412_LOCK: build is not independently authorized")
    observed_sources = _source_hashes(repo_root)
    if lock.get("source_hashes") != observed_sources:
        raise ValueError("STOP_V412_LOCK: source hashes changed")
    expected_paths = {
        "allowlist": str(Path(allowlist_path).resolve()),
        "denylist": str(Path(denylist_path).resolve()),
    }
    input_paths = lock.get("input_paths")
    if not isinstance(input_paths, Mapping):
        raise ValueError("STOP_V412_LOCK: input paths missing")
    if set(input_paths) != {
        "allowlist",
        "denylist",
        "queries",
        "partitions",
        "snapshot",
    }:
        raise ValueError("STOP_V412_LOCK: input path set changed")
    if {
        key: input_paths.get(key) for key in ("allowlist", "denylist")
    } != expected_paths:
        raise ValueError("STOP_V412_LOCK: policy paths changed")
    input_hashes = lock.get("input_hashes")
    if not isinstance(input_hashes, Mapping):
        raise ValueError("STOP_V412_LOCK: input hashes missing")
    if set(input_hashes) != {
        "allowlist",
        "denylist",
        "queries",
        "partitions",
        "snapshot",
    }:
        raise ValueError("STOP_V412_LOCK: input hash set changed")
    if (
        input_hashes.get("allowlist") != file_sha256(allowlist_path)
        or input_hashes.get("denylist") != file_sha256(denylist_path)
    ):
        raise ValueError("STOP_V412_LOCK: policy hashes changed")
    if lock.get("runtime") != _runtime():
        raise ValueError("STOP_V412_LOCK: runtime changed")
    if (
        lock.get("snapshot_sha256") != EXPECTED_SNAPSHOT_SHA256
        or lock.get("partitions_signature") != EXPECTED_PARTITIONS_SIGNATURE
    ):
        raise ValueError("STOP_V412_LOCK: snapshot or partitions changed")
    commit = str(lock.get("git_commit") or "")
    if not commit:
        raise ValueError("STOP_V412_LOCK: git commit missing")
    if verify_git:
        for relative, expected_hash in observed_sources.items():
            result = subprocess.run(
                ["git", "show", f"{commit}:{relative}"],
                cwd=repo_root,
                check=True,
                capture_output=True,
            )
            if hashlib.sha256(result.stdout).hexdigest() != expected_hash:
                raise ValueError(
                    f"STOP_V412_LOCK: commit does not pin {relative}"
                )
    return lock, hashlib.sha256(raw).hexdigest()


def validate_allowlist(
    path: Path,
) -> tuple[dict[str, Any], Path, str, Path, Path]:
    policy = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(policy) != {"schema_version", "artifacts", "policy"}:
        raise ValueError("STOP_FORBIDDEN_INPUT: allowlist fields changed")
    if policy.get("schema_version") != ALLOWLIST_SCHEMA_VERSION:
        raise ValueError("STOP_FORBIDDEN_INPUT: allowlist schema changed")
    artifacts = policy.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("STOP_FORBIDDEN_INPUT: allowlist artifacts missing")
    roles = [item.get("role") for item in artifacts if isinstance(item, Mapping)]
    if roles != [
        "V411_INPUT_BLIND_DATASET",
        "V411_RANKER_C",
        "V411_ACCEPTOR_SCENES",
        "V411_ACCEPTOR_BUNDLE",
    ]:
        raise ValueError("STOP_FORBIDDEN_INPUT: allowlist roles changed")
    expected_artifact_fields = [
        {"files", "phases", "manifest_sha256", "role", "root"},
        {
            "files",
            "projection",
            "phases",
            "manifest_sha256",
            "role",
            "root",
        },
        {"files", "phases", "manifest_sha256", "role", "root"},
        {"files", "phases", "manifest_sha256", "role", "root"},
    ]
    expected_files = [
        {"queries.parquet", "split_assignments.parquet"},
        {"predictions_ranker_c_oof_dev.parquet", "ranker_c/full_fit.json"},
        {"acceptor_scenes.parquet"},
        {
            "bundle/acceptor_model.joblib",
            "bundle/metadata.json",
            "bundle/stack_manifest.json",
        },
    ]
    for artifact, expected_fields, filenames in zip(
        artifacts, expected_artifact_fields, expected_files, strict=True
    ):
        if set(artifact) != expected_fields:
            raise ValueError("STOP_FORBIDDEN_INPUT: artifact fields changed")
        if set(artifact.get("files") or {}) != filenames:
            raise ValueError("STOP_FORBIDDEN_INPUT: artifact files changed")
        if set(artifact.get("phases") or {}) != filenames:
            raise ValueError("STOP_FORBIDDEN_INPUT: artifact phases changed")
        if not all(
            len(str(value)) == 64
            for value in (artifact.get("files") or {}).values()
        ):
            raise ValueError("STOP_FORBIDDEN_INPUT: artifact hashes changed")
        if len(str(artifact.get("manifest_sha256") or "")) != 64:
            raise ValueError("STOP_FORBIDDEN_INPUT: manifest hash changed")
    ranker_projection = artifacts[1].get("projection") or {}
    if ranker_projection != {
        "predictions_ranker_c_oof_dev.parquet": [
            "query_id",
            "candidate_siret",
            "candidate_siren",
            "retrieval_rank",
            "ranker_score",
            "prediction_origin",
            "oof_fold",
            "ranker_rank",
        ]
    }:
        raise ValueError("STOP_FORBIDDEN_INPUT: ranker projection changed")
    for artifact in artifacts[1:]:
        if set((artifact.get("phases") or {}).values()) != {
            "POST_EVIDENCE_SEAL"
        }:
            raise ValueError("STOP_FORBIDDEN_INPUT: post-seal phase changed")
    blind = artifacts[0]
    if set(blind.get("phases") or {}) != {
        "queries.parquet",
        "split_assignments.parquet",
    }:
        raise ValueError("STOP_FORBIDDEN_INPUT: blind phases changed")
    if blind["phases"].get("queries.parquet") != "PRE_EVIDENCE_SEAL":
        raise ValueError("STOP_FORBIDDEN_INPUT: queries not authorized pre-seal")
    files = blind.get("files") or {}
    query_hash = str(files.get("queries.parquet") or "")
    query_path = Path(str(blind.get("root") or "")).resolve() / "queries.parquet"
    frozen = policy.get("policy") or {}
    if set(frozen) != {
        "partitions_path",
        "partitions_signature",
        "snapshot_path",
        "snapshot_sha256",
    }:
        raise ValueError("STOP_FORBIDDEN_INPUT: policy fields changed")
    partitions_path = Path(str(frozen.get("partitions_path") or "")).resolve()
    snapshot_path = Path(str(frozen.get("snapshot_path") or "")).resolve()
    if (
        frozen.get("partitions_signature") != EXPECTED_PARTITIONS_SIGNATURE
        or frozen.get("snapshot_sha256") != EXPECTED_SNAPSHOT_SHA256
        or not query_hash
    ):
        raise ValueError("STOP_FORBIDDEN_INPUT: frozen policy changed")
    return policy, query_path, query_hash, partitions_path, snapshot_path


def validate_denylist(path: Path) -> tuple[set[Path], set[str]]:
    policy = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(policy) != {"schema_version", "artifacts"}:
        raise ValueError("STOP_FORBIDDEN_INPUT: denylist fields changed")
    if policy.get("schema_version") != DENYLIST_SCHEMA_VERSION:
        raise ValueError("STOP_FORBIDDEN_INPUT: denylist schema changed")
    roots: set[Path] = set()
    hashes: set[str] = set()
    artifacts = policy.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 3:
        raise ValueError("STOP_FORBIDDEN_INPUT: denylist artifacts changed")
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise ValueError("STOP_FORBIDDEN_INPUT: invalid denylist artifact")
        if set(artifact) != {
            "build_id",
            "files",
            "manifest_sha256",
            "path",
            "reason",
        }:
            raise ValueError("STOP_FORBIDDEN_INPUT: denylist fields changed")
        roots.add(Path(str(artifact.get("path") or "")).resolve())
        manifest_hash = str(artifact.get("manifest_sha256") or "")
        if manifest_hash:
            hashes.add(manifest_hash)
        hashes.update(str(value) for value in (artifact.get("files") or {}).values())
    if len(roots) != 3 or not hashes:
        raise ValueError("STOP_FORBIDDEN_INPUT: empty denylist")
    return roots, hashes


def _files(paths: Iterable[Path]) -> Iterable[Path]:
    for raw in paths:
        path = Path(raw).resolve()
        if path.is_file():
            yield path
        elif path.is_dir():
            yield from sorted(item for item in path.rglob("*") if item.is_file())
        else:
            raise ValueError(f"STOP_FORBIDDEN_INPUT: missing input {path}")


def validate_inputs_against_denylist(
    paths: Iterable[Path],
    *,
    forbidden_roots: set[Path],
    forbidden_hashes: set[str],
) -> dict[str, str]:
    """Reject both an original challenge path and any byte-identical copy."""

    inventory: dict[str, str] = {}
    for path in _files(paths):
        if any(path == root or path.is_relative_to(root) for root in forbidden_roots):
            raise ValueError(f"STOP_FORBIDDEN_INPUT: forbidden path {path}")
        digest = file_sha256(path)
        if digest in forbidden_hashes:
            raise ValueError(f"STOP_FORBIDDEN_INPUT: forbidden hash {path}")
        inventory[str(path)] = digest
    return inventory


def validate_preseal_queries(
    frame: pd.DataFrame,
    *,
    expected_count: int = EXPECTED_QUERY_COUNT,
) -> pd.DataFrame:
    if list(frame.columns) != PRESEAL_QUERY_COLUMNS:
        raise ValueError("STOP_V412_PRESEAL: query projection changed")
    output = frame.copy()
    if len(output) != expected_count:
        raise ValueError("STOP_V412_PRESEAL: query count changed")
    output["query_id"] = output["query_id"].fillna("").astype(str)
    if output["query_id"].eq("").any() or output["query_id"].duplicated().any():
        raise ValueError("STOP_V412_PRESEAL: invalid query IDs")
    for column in PRESEAL_QUERY_COLUMNS[1:]:
        output[column] = output[column].fillna("").astype(str)
    return output


def load_preseal_queries(
    path: Path,
    *,
    expected_sha256: str,
    expected_count: int = EXPECTED_QUERY_COUNT,
) -> pd.DataFrame:
    """Read the sole pre-seal parquet and immediately close its TOCTOU window."""

    frame = pd.read_parquet(path, columns=PRESEAL_QUERY_COLUMNS)
    if file_sha256(path) != expected_sha256:
        raise ValueError("STOP_V412_PRESEAL: query changed during read")
    return validate_preseal_queries(frame, expected_count=expected_count)


def validate_frozen_storage(
    *,
    partitions_path: Path,
    snapshot_path: Path,
    expected_partitions_signature: str = EXPECTED_PARTITIONS_SIGNATURE,
    expected_snapshot_sha256: str = EXPECTED_SNAPSHOT_SHA256,
) -> None:
    """Hash, but never deserialize, the frozen partition and snapshot inputs."""

    if _path_signature(partitions_path) != expected_partitions_signature:
        raise ValueError("STOP_FORBIDDEN_INPUT: partition signature changed")
    if file_sha256(snapshot_path) != expected_snapshot_sha256:
        raise ValueError("STOP_FORBIDDEN_INPUT: snapshot hash changed")


def _peak_rss_bytes() -> int:
    observed = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return observed if sys.platform == "darwin" else observed * 1024


def refresh_and_validate_peak_rss(timing: dict[str, Any]) -> None:
    timing["peak_rss_bytes"] = _peak_rss_bytes()
    if timing["peak_rss_bytes"] > timing["peak_rss_limit_bytes"]:
        raise ValueError("STOP_V412_PERFORMANCE: peak RSS exceeds 8 GiB")


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * percentile))
    return float(ordered[index])


def compute_direct_evidence(
    queries: pd.DataFrame,
    *,
    partitions_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run the frozen V4 function over each complete active partition."""

    started = time.perf_counter_ns()
    store = PartitionedCandidateStore(partitions_path)
    work = queries.copy()
    work["split"] = "v412_label_free"
    work["postcode"] = work["crm_postcode"]
    work["insee"] = work["crm_insee"]
    work["partition_key"] = [
        _planned_partition_key(row, store) for row in work.to_dict("records")
    ]
    query_records: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    per_query_ms: list[float] = []
    for partition_key, group in work.groupby("partition_key", sort=True):
        partition_started = time.perf_counter_ns()
        partition_rows = _load_partition(str(partition_key), store)
        index: ActivePartitionIndex = build_active_partition_index(partition_rows)
        partition_load_ms = (
            time.perf_counter_ns() - partition_started
        ) / 1_000_000
        load_share_ms = partition_load_ms / max(1, len(group))
        for query in group.sort_values("query_id", kind="mergesort").to_dict(
            "records"
        ):
            query_started = time.perf_counter_ns()
            direct = find_direct_active_candidates(
                query,
                index,
                partition_key=str(partition_key),
            )
            projected = [
                candidate_evidence_record(str(query["query_id"]), record)
                for record in direct
            ]
            candidate_records.extend(projected)
            query_records.append(
                query_evidence_record(
                    query_id=str(query["query_id"]),
                    partition_key=str(partition_key),
                    active_universe_count=index.active_count,
                    candidates=projected,
                )
            )
            per_query_ms.append(
                load_share_ms
                + (time.perf_counter_ns() - query_started) / 1_000_000
            )
    total_ms = (time.perf_counter_ns() - started) / 1_000_000
    query_evidence, candidate_evidence = build_evidence_frames(
        query_records, candidate_records
    )
    timing = {
        "query_count": len(query_evidence),
        "total_evidence_ms": total_ms,
        "amortized_batch_per_query_ms": {
            "p50": _percentile(per_query_ms, 0.50),
            "p95": _percentile(per_query_ms, 0.95),
            "max": max(per_query_ms, default=0.0),
        },
        "serve_latency_gate_eligible": False,
        "peak_rss_bytes": _peak_rss_bytes(),
        "peak_rss_limit_bytes": 8 * 1024**3,
    }
    return query_evidence, candidate_evidence, timing


def compute_direct_evidence_with_recheck(
    queries: pd.DataFrame,
    *,
    partitions_path: Path,
    snapshot_path: Path,
    expected_partitions_signature: str = EXPECTED_PARTITIONS_SIGNATURE,
    expected_snapshot_sha256: str = EXPECTED_SNAPSHOT_SHA256,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compute evidence, then close the TOCTOU window before any seal."""

    result = compute_direct_evidence(
        queries,
        partitions_path=partitions_path,
    )
    validate_frozen_storage(
        partitions_path=partitions_path,
        snapshot_path=snapshot_path,
        expected_partitions_signature=expected_partitions_signature,
        expected_snapshot_sha256=expected_snapshot_sha256,
    )
    return result


def _external_output(path: Path) -> Path:
    output = Path(path).resolve()
    if not output.is_relative_to(Path("/Volumes/CATNAT_DATA")):
        raise ValueError("output root must be under /Volumes/CATNAT_DATA")
    return output


def build_artifact(
    *,
    execution_lock_path: Path,
    allowlist_path: Path,
    denylist_path: Path,
    output_root: Path,
    expected_query_count: int = EXPECTED_QUERY_COUNT,
    require_external_output: bool = True,
) -> Path:
    lock, lock_sha256 = validate_execution_lock(
        execution_lock_path,
        allowlist_path=allowlist_path,
        denylist_path=denylist_path,
        verify_git=True,
    )
    (
        _,
        query_path,
        expected_query_hash,
        partitions_path,
        snapshot_path,
    ) = validate_allowlist(allowlist_path)
    forbidden_roots, forbidden_hashes = validate_denylist(denylist_path)
    repo_root = Path(__file__).resolve().parent.parent
    scanned = validate_inputs_against_denylist(
        [
            execution_lock_path,
            allowlist_path,
            denylist_path,
            query_path,
            partitions_path,
            snapshot_path,
            *[repo_root / relative for relative in SOURCE_PATHS],
        ],
        forbidden_roots=forbidden_roots,
        forbidden_hashes=forbidden_hashes,
    )
    if file_sha256(query_path) != expected_query_hash:
        raise ValueError("STOP_FORBIDDEN_INPUT: allowlisted query hash changed")
    validate_frozen_storage(
        partitions_path=partitions_path,
        snapshot_path=snapshot_path,
    )
    locked_paths = lock["input_paths"]
    locked_hashes = lock["input_hashes"]
    expected_locked_paths = {
        "allowlist": str(Path(allowlist_path).resolve()),
        "denylist": str(Path(denylist_path).resolve()),
        "queries": str(query_path),
        "partitions": str(partitions_path),
        "snapshot": str(snapshot_path),
    }
    expected_locked_hashes = {
        "allowlist": file_sha256(allowlist_path),
        "denylist": file_sha256(denylist_path),
        "queries": expected_query_hash,
        "partitions": EXPECTED_PARTITIONS_SIGNATURE,
        "snapshot": EXPECTED_SNAPSHOT_SHA256,
    }
    if locked_paths != expected_locked_paths or locked_hashes != expected_locked_hashes:
        raise ValueError("STOP_V412_LOCK: frozen inputs changed")

    # First and only pre-seal dataframe deserialization.
    queries = load_preseal_queries(
        query_path,
        expected_sha256=expected_query_hash,
        expected_count=expected_query_count,
    )
    output_root = (
        _external_output(output_root)
        if require_external_output
        else Path(output_root).resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "execution_lock_sha256": lock_sha256,
        "source_hashes": lock["source_hashes"],
        "query_sha256": expected_query_hash,
        "query_count": len(queries),
        "partitions_signature": EXPECTED_PARTITIONS_SIGNATURE,
        "snapshot_sha256": EXPECTED_SNAPSHOT_SHA256,
        "policy_version": POLICY_VERSION,
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    target = output_root / build_id
    if target.exists():
        raise FileExistsError(f"Immutable V4.12 evidence already exists: {target}")
    staging = Path(tempfile.mkdtemp(prefix=f".{build_id}.tmp-", dir=output_root))
    try:
        (
            query_evidence,
            candidate_evidence,
            timing,
        ) = compute_direct_evidence_with_recheck(
            queries,
            partitions_path=partitions_path,
            snapshot_path=snapshot_path,
        )
        validate_evidence(query_evidence, candidate_evidence)
        query_output = staging / "query_evidence.parquet"
        candidate_output = staging / "candidate_evidence.parquet"
        _write_parquet(query_output, query_evidence)
        _write_parquet(candidate_output, candidate_evidence)
        refresh_and_validate_peak_rss(timing)
        evidence_hashes = {
            "query_evidence.parquet": file_sha256(query_output),
            "candidate_evidence.parquet": file_sha256(candidate_output),
        }
        integrity = {
            "query_count": len(query_evidence),
            "candidate_count": len(candidate_evidence),
            "max_direct_candidate_count": int(
                query_evidence["direct_candidate_count"].max()
            ),
            "query_ids_unique": True,
            "candidate_sirets_unique_per_query": True,
            "evidence_references_bijective": True,
            "active_candidates_only": True,
            "full_partition_universe": True,
            "ranker_pool_opened": False,
            "ranker_pool_modified": False,
            "retrieval_candidate_cap": 100,
            "labels_opened_before_seal": False,
            "split_opened_before_seal": False,
            "scenes_opened_before_seal": False,
            "models_opened_before_seal": False,
            "challenge_opened": False,
            "sealed_evidence_hashes": evidence_hashes,
        }
        _write_json(staging / "timing.json", timing)
        _write_json(staging / "integrity.json", integrity)
        outputs: dict[str, Any] = {}
        for filename in sorted(OUTPUT_FILENAMES):
            path = staging / filename
            record: dict[str, Any] = {
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
            if filename == "query_evidence.parquet":
                record.update(
                    row_count=len(query_evidence),
                    columns=QUERY_EVIDENCE_COLUMNS,
                )
            elif filename == "candidate_evidence.parquet":
                record.update(
                    row_count=len(candidate_evidence),
                    columns=CANDIDATE_EVIDENCE_COLUMNS,
                )
            outputs[filename] = record
        manifest = {
            **identity,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "outputs": outputs,
            "inputs": {
                "queries": {
                    "path": str(query_path),
                    "sha256": expected_query_hash,
                    "projection": PRESEAL_QUERY_COLUMNS,
                },
                "partitions": {
                    "path": str(partitions_path),
                    "signature": EXPECTED_PARTITIONS_SIGNATURE,
                },
                "snapshot": {
                    "opened": False,
                    "sha256": EXPECTED_SNAPSHOT_SHA256,
                },
                "execution_lock": {
                    "path": str(Path(execution_lock_path).resolve()),
                    "sha256": lock_sha256,
                },
            },
            "denylist_scan": {
                "input_file_count": len(scanned),
                "all_paths_clear": True,
                "all_hashes_clear": True,
            },
            "phase_ledger": [
                {
                    "phase": "PRE_EVIDENCE_SEAL",
                    "deserialized": [
                        "queries.parquet:PRESEAL_QUERY_COLUMNS",
                        "frozen_partition_files",
                    ],
                },
                {
                    "phase": "EVIDENCE_SEALED",
                    "sha256": evidence_hashes,
                },
            ],
        }
        _write_json(staging / "manifest.json", manifest)
        os.replace(staging, target)
        directory = os.open(output_root, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def validate_artifact(path: Path) -> None:
    root = Path(path).resolve()
    entries = list(root.iterdir())
    if (
        {item.name for item in entries}
        != OUTPUT_FILENAMES | {"manifest.json"}
        or any(not item.is_file() or item.is_symlink() for item in entries)
    ):
        raise ValueError("STOP_V412_ARTIFACT: unexpected file set")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("STOP_V412_ARTIFACT: schema changed")
    identity = {
        key: manifest[key]
        for key in (
            "schema_version",
            "execution_lock_sha256",
            "source_hashes",
            "query_sha256",
            "query_count",
            "partitions_signature",
            "snapshot_sha256",
            "policy_version",
        )
    }
    expected_build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    if manifest.get("build_id") != expected_build_id or root.name != expected_build_id:
        raise ValueError("STOP_V412_ARTIFACT: identity changed")
    outputs = manifest.get("outputs") or {}
    if set(outputs) != OUTPUT_FILENAMES:
        raise ValueError("STOP_V412_ARTIFACT: output set changed")
    for filename, record in outputs.items():
        output_path = root / filename
        if (
            file_sha256(output_path) != record.get("sha256")
            or output_path.stat().st_size != record.get("size_bytes")
        ):
            raise ValueError(f"STOP_V412_ARTIFACT: output drift {filename}")
    query_evidence = pd.read_parquet(root / "query_evidence.parquet")
    candidate_evidence = pd.read_parquet(root / "candidate_evidence.parquet")
    if outputs["query_evidence.parquet"].get("columns") != QUERY_EVIDENCE_COLUMNS:
        raise ValueError("STOP_V412_ARTIFACT: query declaration changed")
    if (
        outputs["candidate_evidence.parquet"].get("columns")
        != CANDIDATE_EVIDENCE_COLUMNS
    ):
        raise ValueError("STOP_V412_ARTIFACT: candidate declaration changed")
    if (
        manifest.get("query_count") != len(query_evidence)
        or outputs["query_evidence.parquet"].get("row_count")
        != len(query_evidence)
        or outputs["candidate_evidence.parquet"].get("row_count")
        != len(candidate_evidence)
    ):
        raise ValueError("STOP_V412_ARTIFACT: row declarations changed")
    validate_evidence(query_evidence, candidate_evidence)
    integrity = json.loads((root / "integrity.json").read_text(encoding="utf-8"))
    timing = json.loads((root / "timing.json").read_text(encoding="utf-8"))
    evidence_hashes = {
        "query_evidence.parquet": outputs["query_evidence.parquet"]["sha256"],
        "candidate_evidence.parquet": outputs[
            "candidate_evidence.parquet"
        ]["sha256"],
    }
    max_direct = int(
        query_evidence["direct_candidate_count"].max()
        if len(query_evidence)
        else 0
    )
    if (
        integrity.get("query_count") != len(query_evidence)
        or integrity.get("candidate_count") != len(candidate_evidence)
        or integrity.get("max_direct_candidate_count") != max_direct
        or integrity.get("sealed_evidence_hashes") != evidence_hashes
        or any(
            integrity.get(name) is not expected
            for name, expected in {
                "query_ids_unique": True,
                "candidate_sirets_unique_per_query": True,
                "evidence_references_bijective": True,
                "active_candidates_only": True,
                "full_partition_universe": True,
                "ranker_pool_opened": False,
                "ranker_pool_modified": False,
                "labels_opened_before_seal": False,
                "split_opened_before_seal": False,
                "scenes_opened_before_seal": False,
                "models_opened_before_seal": False,
                "challenge_opened": False,
            }.items()
        )
        or integrity.get("retrieval_candidate_cap") != 100
    ):
        raise ValueError("STOP_V412_ARTIFACT: integrity declarations changed")
    amortized = timing.get("amortized_batch_per_query_ms")
    numeric_timing = [
        timing.get("total_evidence_ms"),
        timing.get("peak_rss_bytes"),
        *((amortized or {}).get(name) for name in ("p50", "p95", "max")),
    ]
    if (
        timing.get("query_count") != len(query_evidence)
        or timing.get("serve_latency_gate_eligible") is not False
        or not isinstance(amortized, Mapping)
        or set(amortized) != {"p50", "p95", "max"}
        or "per_query_evidence_ms" in timing
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
            for value in numeric_timing
        )
        or not (
            float(amortized["p50"])
            <= float(amortized["p95"])
            <= float(amortized["max"])
        )
        or not isinstance(timing.get("peak_rss_bytes"), int)
        or timing.get("peak_rss_limit_bytes") != 8 * 1024**3
        or timing["peak_rss_bytes"] > timing["peak_rss_limit_bytes"]
    ):
        raise ValueError("STOP_V412_ARTIFACT: timing declarations changed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-lock", type=Path)
    parser.add_argument("--allowlist", type=Path, default=DEFAULT_ALLOWLIST)
    parser.add_argument("--denylist", type=Path, default=DEFAULT_DENYLIST)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--validate-artifact", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_artifact:
        validate_artifact(args.validate_artifact)
        print(args.validate_artifact)
        return
    if args.execution_lock is None or args.output_root is None:
        raise SystemExit("--execution-lock and --output-root are required")
    print(
        build_artifact(
            execution_lock_path=args.execution_lock,
            allowlist_path=args.allowlist,
            denylist_path=args.denylist,
            output_root=args.output_root,
        )
    )


if __name__ == "__main__":
    main()
