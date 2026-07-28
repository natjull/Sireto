#!/usr/bin/env python3
"""Evaluate the sealed V4.12-G guard on frozen V4.11 historical populations.

This runner is post-seal only.  It requires an independently audited external
execution lock and cannot open any consumed challenge, retrieval label, or the
``is_ground_truth`` column present in the Ranker-C parquet.
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
from typing import Any, Mapping

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.v412_direct_evidence import (  # noqa: E402
    CANDIDATE_EVIDENCE_COLUMNS,
    QUERY_EVIDENCE_COLUMNS,
    apply_guard,
    validate_evidence,
)
from src.xgb_matcher.v412_evaluation import (  # noqa: E402
    CANONICAL_COUNTS,
    DECISION_COLUMNS,
    FIXED_THRESHOLD,
    RANKER_COLUMNS,
    SCENE_METADATA_COLUMNS,
    SPLIT_COLUMNS,
    apply_guard_frame,
    evaluate_comparison_gate,
    score_v411,
    validate_population_parity,
    validate_ranker_projection,
)


SCHEMA_VERSION = "sireto-v4.12-guard-historical-evaluation-1"
LOCK_SCHEMA_VERSION = "sireto-v4.12-guard-evaluation-lock-1"
PURPOSE = "EVALUATE_V412_GUARD_HISTORICAL_ONCE"
AUDIT_VERDICT = "GO_EVALUATE_V412_GUARD"
EVIDENCE_BUILD_ID = "10f16403795ccee6"
EXPECTED_EVIDENCE_MANIFEST_SHA256 = (
    "2006184308fc412af944b4752f8fd0dbbc9ea167943681238215b46ea5fbc12a"
)
EXPECTED_EVIDENCE_QUERY_SHA256 = (
    "6fd7d441a9f6aaac99555c1083b90a61e52e02a291d6cffccc07d061f60242ed"
)
EXPECTED_EVIDENCE_CANDIDATE_SHA256 = (
    "36a3b0042a30c852f1a0595d2a31a91858b4fa5ea405cf7ccc32330f345f7d97"
)
EXPECTED_INPUT_HASHES = {
    "split": "33fa52af7a740124235c151efb5b9a8834ffd1c83c65d1af56c75b2eff271193",
    "ranker": "f14828aafa146dc4ad0399697c9477e57930ba618a5b2d7d0a903e52c2d879c0",
    "scenes": "c0f3d670e50cb43cdd6fed3b976c95e51d70f0313ae07ed0e1e2ed01eca5bed3",
    "acceptor_model": "a804feb64f28c417adda4418724f53df50b20d3d308b3e7c778c7189d368e3cf",
    "acceptor_metadata": "e4b99676e695d19748b71a7657ff5a1f5c7dfa2879754dd2e1b15c8906a61d6b",
    "stack_manifest": "81279978f47e1e2b1b4a1ea85d595b8dedd8ee8a073e34a19b3ffd340c945d5a",
    "retrieval_manifest": "445bf15d0a8f950c213764a104c05f8263bcfba7b7391c9df247d0e5873e6280",
    "ranker_manifest": "1552ab2623580f1ae68e31ec1497be8a93a1bb1f2d33114dd34cfea07a864053",
    "scene_manifest": "8faaf2761bb280f1ba559ea3f2c579fd5d91531a202b6b54dff79e38f0d2757e",
    "acceptor_manifest": "a7fc765fe439392baec61fa8a35a941bb1f778281ccdbb54b55c699e9f0c11d9",
    "evidence_manifest": EXPECTED_EVIDENCE_MANIFEST_SHA256,
    "evidence_query": EXPECTED_EVIDENCE_QUERY_SHA256,
    "evidence_candidate": EXPECTED_EVIDENCE_CANDIDATE_SHA256,
}
ROOTS = {
    "retrieval": Path(
        "/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/"
        "v4_11_input_blind/ec4326ec57e4411d"
    ),
    "ranker": Path(
        "/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/"
        "v4_11_ranker_c/e13eb3ac7498256e"
    ),
    "scenes": Path(
        "/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/"
        "v4_11_acceptor/52ea3faba9a56aff"
    ),
    "acceptor": Path(
        "/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/"
        "v4_11_acceptor/9d23bf3deb6b63de"
    ),
    "evidence": Path(
        "/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/"
        f"v4_12_direct_evidence/{EVIDENCE_BUILD_ID}"
    ),
}
DEFAULT_ALLOWLIST = Path("config/v4_12_development_inputs.json")
DEFAULT_DENYLIST = Path("config/v4_12_forbidden_artifacts.json")
DEFAULT_CONTRACT = Path("docs/v4_12_direct_evidence_gate_contract.md")
INPUT_PATHS = {
    "split": ROOTS["retrieval"] / "split_assignments.parquet",
    "ranker": ROOTS["ranker"] / "predictions_ranker_c_oof_dev.parquet",
    "scenes": ROOTS["scenes"] / "acceptor_scenes.parquet",
    "acceptor_model": ROOTS["acceptor"] / "bundle/acceptor_model.joblib",
    "acceptor_metadata": ROOTS["acceptor"] / "bundle/metadata.json",
    "stack_manifest": ROOTS["acceptor"] / "bundle/stack_manifest.json",
    "retrieval_manifest": ROOTS["retrieval"] / "manifest.json",
    "ranker_manifest": ROOTS["ranker"] / "manifest.json",
    "scene_manifest": ROOTS["scenes"] / "manifest.json",
    "acceptor_manifest": ROOTS["acceptor"] / "manifest.json",
    "evidence_manifest": ROOTS["evidence"] / "manifest.json",
    "evidence_query": ROOTS["evidence"] / "query_evidence.parquet",
    "evidence_candidate": ROOTS["evidence"] / "candidate_evidence.parquet",
}
SOURCE_PATHS = [
    "scripts/__init__.py",
    "scripts/evaluate_v412_guard.py",
    "src/xgb_matcher/v412_evaluation.py",
    "src/xgb_matcher/v412_direct_evidence.py",
    "src/xgb_matcher/v411_acceptor.py",
    "src/xgb_matcher/v411_scene.py",
    "src/xgb_matcher/v49_site_function.py",
    "src/xgb_matcher/__init__.py",
    "docs/v4_12_direct_evidence_gate_contract.md",
    "config/v4_12_development_inputs.json",
    "config/v4_12_forbidden_artifacts.json",
]
LOCK_FIELDS = {
    "schema_version",
    "purpose",
    "audit_verdict",
    "git_commit",
    "source_hashes",
    "input_paths",
    "input_hashes",
    "runtime",
    "threshold",
}
OUTPUTS = {
    "decisions.parquet",
    "metrics.json",
    "segments.json",
    "integrity.json",
}
EXPECTED_ALLOW_ARTIFACTS = [
    {
        "files": {
            "queries.parquet": (
                "3a47aef768cee1436ad77a6e114defe50e685b7495f0e75137e9fd06dfe9fc68"
            ),
            "split_assignments.parquet": EXPECTED_INPUT_HASHES["split"],
        },
        "phases": {
            "queries.parquet": "PRE_EVIDENCE_SEAL",
            "split_assignments.parquet": "POST_EVIDENCE_SEAL",
        },
        "manifest_sha256": EXPECTED_INPUT_HASHES["retrieval_manifest"],
        "role": "V411_INPUT_BLIND_DATASET",
        "root": str(ROOTS["retrieval"]),
    },
    {
        "files": {
            "predictions_ranker_c_oof_dev.parquet": EXPECTED_INPUT_HASHES[
                "ranker"
            ],
            "ranker_c/full_fit.json": (
                "f4b71b49ed4f879b88e05e4fb84229d0306c5e8ca96958ac20ad97fcc04349c0"
            ),
        },
        "projection": {
            "predictions_ranker_c_oof_dev.parquet": RANKER_COLUMNS,
        },
        "phases": {
            "predictions_ranker_c_oof_dev.parquet": "POST_EVIDENCE_SEAL",
            "ranker_c/full_fit.json": "POST_EVIDENCE_SEAL",
        },
        "manifest_sha256": EXPECTED_INPUT_HASHES["ranker_manifest"],
        "role": "V411_RANKER_C",
        "root": str(ROOTS["ranker"]),
    },
    {
        "files": {
            "acceptor_scenes.parquet": EXPECTED_INPUT_HASHES["scenes"],
        },
        "phases": {
            "acceptor_scenes.parquet": "POST_EVIDENCE_SEAL",
        },
        "manifest_sha256": EXPECTED_INPUT_HASHES["scene_manifest"],
        "role": "V411_ACCEPTOR_SCENES",
        "root": str(ROOTS["scenes"]),
    },
    {
        "files": {
            "bundle/acceptor_model.joblib": EXPECTED_INPUT_HASHES[
                "acceptor_model"
            ],
            "bundle/metadata.json": EXPECTED_INPUT_HASHES[
                "acceptor_metadata"
            ],
            "bundle/stack_manifest.json": EXPECTED_INPUT_HASHES[
                "stack_manifest"
            ],
        },
        "phases": {
            "bundle/acceptor_model.joblib": "POST_EVIDENCE_SEAL",
            "bundle/metadata.json": "POST_EVIDENCE_SEAL",
            "bundle/stack_manifest.json": "POST_EVIDENCE_SEAL",
        },
        "manifest_sha256": EXPECTED_INPUT_HASHES["acceptor_manifest"],
        "role": "V411_ACCEPTOR_BUNDLE",
        "root": str(ROOTS["acceptor"]),
    },
]
EXPECTED_ALLOW_POLICY = {
    "partitions_path": (
        "/Users/nathanjullia/Documents/Projets/SIRETO/data/candidates_v7_all"
    ),
    "partitions_signature": (
        "2f6668f60da8bc9fe52b683b32ef35641803679c01f8c8fd124e2e86a41e2b82"
    ),
    "snapshot_path": (
        "/Users/nathanjullia/Documents/Projets/SIRETO/"
        "data/StockEtablissement_utf8.parquet"
    ),
    "snapshot_sha256": (
        "c91180cc5bae86948dd57d752c9bae45e58cc64653e99d5a9357664b67300845"
    ),
}
EXPECTED_DENY_ARTIFACTS = [
    {
        "build_id": "1c994c852c10acaf",
        "files": {
            "queries_sanitized.parquet": (
                "68b9a9c59bfc91f42f1242510137b43c903bcf3687933020867a5c4bd59f0074"
            ),
            "sealed_mapping.parquet": (
                "3710a48aa522e69d6687bdea4ab8df25b8bb14e739219b0968ea166c2f21a540"
            ),
        },
        "manifest_sha256": (
            "449bed70276f31728357c173a5d17a3f646c3975306a2488dabd95083cc7dae3"
        ),
        "path": (
            "/Volumes/CATNAT_DATA/SIRETO_RECALL100/challenges/"
            "v4_11_unseen_sanitized/1c994c852c10acaf"
        ),
    },
    {
        "build_id": "4f9ef46516b89ab8",
        "files": {
            "evidence.parquet": (
                "5180c3b88339149bf8141defc7c9540322dbcd70fab7c6b515caaa7b4c71477b"
            ),
            "labels_frozen.parquet": (
                "747426ea3ce2a9188c6f35591ca1186c87f9534844707882b56cafd09b1c9b15"
            ),
        },
        "manifest_sha256": (
            "17c7915725cea978278f1699832e5c17405dbab8cd21ef407f6d96916a5c89e7"
        ),
        "path": (
            "/Volumes/CATNAT_DATA/SIRETO_RECALL100/challenges/"
            "v4_11_unseen_qualification/4f9ef46516b89ab8"
        ),
    },
    {
        "build_id": "ddb7336e8c2e042d",
        "files": {
            "errors.parquet": (
                "87dd3171c99e2c9438249ee16695e70c132246f684dd9e52bbb6111b9212c986"
            ),
            "evaluated_predictions.parquet": (
                "286c4ef4028d252bd9561b7a89da27636e6c3d93164acae0796a0bd95d8bf9be"
            ),
            "evaluation.json": (
                "509c64d12e27b3efefb12cf5aaebe2a4446ec8a062d2adf2d85200bd9a4bce9e"
            ),
            "label_overlay.parquet": (
                "88940328714bbb39592ee4dee5198387123e411e6875bfbcefc6114a5ac873b5"
            ),
            "predictions_blind.parquet": (
                "ac4e9d2ee8cb112039f4242d51bb10c7eee95b771f8ef1d2a358bb8d8fa1b392"
            ),
            "ranked_candidates.parquet": (
                "ee8690fc27162157a2907f584371b84a892fafbe73a55b2aa602620d4c439c6f"
            ),
            "retrieval_candidates.parquet": (
                "9b3cacc50f92178d376fbc241a88b0f26cc786d61e1467ec238a6e38864dc438"
            ),
            "scenes_blind.parquet": (
                "b4e8bca02485793849b73f8fa1eee70dcc14b578f970680b8f64f8d04b7140e7"
            ),
        },
        "manifest_sha256": (
            "37f4957052493b3aa1e8b2e3ba5f156816cb33121aa5915f88c9b581306c71e6"
        ),
        "path": (
            "/Volumes/CATNAT_DATA/SIRETO_RECALL100/challenges/"
            "v4_11_unseen_execution/ddb7336e8c2e042d"
        ),
    },
]
MANIFEST_FIELDS = {
    "schema_version",
    "execution_lock_sha256",
    "evidence_seal",
    "input_hashes",
    "source_hashes",
    "threshold",
    "build_id",
    "created_at",
    "outputs",
    "verdict",
    "gate",
    "phase_ledger",
    "latency_gate_evaluated",
    "production_certified",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any] | list[Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fsync_file(path: Path) -> None:
    with Path(path).open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(Path(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _runtime() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": importlib.metadata.version("numpy"),
        "pandas": importlib.metadata.version("pandas"),
        "pyarrow": importlib.metadata.version("pyarrow"),
        "joblib": importlib.metadata.version("joblib"),
        "scikit-learn": importlib.metadata.version("scikit-learn"),
        "scipy": importlib.metadata.version("scipy"),
        "xgboost": importlib.metadata.version("xgboost"),
    }


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent.parent
    return {relative: file_sha256(root / relative) for relative in SOURCE_PATHS}


def validate_execution_lock(
    path: Path,
    *,
    verify_git: bool = True,
) -> tuple[dict[str, Any], str]:
    raw = Path(path).read_bytes()
    lock = json.loads(raw)
    if set(lock) != LOCK_FIELDS:
        raise ValueError("STOP_V412_EVAL_LOCK: fields changed")
    if (
        lock.get("schema_version") != LOCK_SCHEMA_VERSION
        or lock.get("purpose") != PURPOSE
        or lock.get("audit_verdict") != AUDIT_VERDICT
    ):
        raise ValueError("STOP_V412_EVAL_LOCK: independent GO missing")
    sources = _source_hashes()
    if lock.get("source_hashes") != sources:
        raise ValueError("STOP_V412_EVAL_LOCK: sources changed")
    expected_paths = {
        **{name: str(path.resolve()) for name, path in INPUT_PATHS.items()},
        "allowlist": str(DEFAULT_ALLOWLIST.resolve()),
        "denylist": str(DEFAULT_DENYLIST.resolve()),
    }
    expected_hashes = {
        **EXPECTED_INPUT_HASHES,
        "allowlist": file_sha256(DEFAULT_ALLOWLIST),
        "denylist": file_sha256(DEFAULT_DENYLIST),
    }
    if lock.get("input_paths") != expected_paths:
        raise ValueError("STOP_V412_EVAL_LOCK: input paths changed")
    if lock.get("input_hashes") != expected_hashes:
        raise ValueError("STOP_V412_EVAL_LOCK: input hashes changed")
    if lock.get("runtime") != _runtime():
        raise ValueError("STOP_V412_EVAL_LOCK: runtime changed")
    if float(lock.get("threshold", -1)) != FIXED_THRESHOLD:
        raise ValueError("STOP_V412_EVAL_LOCK: threshold changed")
    commit = str(lock.get("git_commit") or "")
    if not commit:
        raise ValueError("STOP_V412_EVAL_LOCK: commit missing")
    if verify_git:
        repo = Path(__file__).resolve().parent.parent
        for relative, expected in sources.items():
            result = subprocess.run(
                ["git", "show", f"{commit}:{relative}"],
                cwd=repo,
                check=True,
                capture_output=True,
            )
            if hashlib.sha256(result.stdout).hexdigest() != expected:
                raise ValueError(
                    f"STOP_V412_EVAL_LOCK: commit does not pin {relative}"
                )
    return lock, hashlib.sha256(raw).hexdigest()


def validate_allowlist_and_denylist() -> tuple[set[Path], set[str]]:
    allow = json.loads(DEFAULT_ALLOWLIST.read_text(encoding="utf-8"))
    if (
        set(allow) != {"schema_version", "artifacts", "policy"}
        or
        allow.get("schema_version") != "sireto-v4.12-development-inputs-1"
        or allow.get("artifacts") != EXPECTED_ALLOW_ARTIFACTS
        or allow.get("policy") != EXPECTED_ALLOW_POLICY
    ):
        raise ValueError("STOP_FORBIDDEN_INPUT: allowlist changed")
    deny = json.loads(DEFAULT_DENYLIST.read_text(encoding="utf-8"))
    if (
        set(deny) != {"schema_version", "artifacts"}
        or
        deny.get("schema_version") != "sireto-v4.12-forbidden-artifacts-1"
        or len(deny.get("artifacts") or []) != 3
    ):
        raise ValueError("STOP_FORBIDDEN_INPUT: denylist changed")
    for observed, expected in zip(
        deny["artifacts"], EXPECTED_DENY_ARTIFACTS, strict=True
    ):
        if (
            set(observed)
            != {
                "build_id",
                "files",
                "manifest_sha256",
                "path",
                "reason",
            }
            or {key: observed.get(key) for key in expected} != expected
            or not str(observed.get("reason") or "").strip()
        ):
            raise ValueError("STOP_FORBIDDEN_INPUT: denylist artifact changed")
    roots = {
        Path(str(artifact.get("path") or "")).resolve()
        for artifact in deny["artifacts"]
    }
    hashes = {
        str(value)
        for artifact in deny["artifacts"]
        for value in [
            artifact.get("manifest_sha256"),
            *(artifact.get("files") or {}).values(),
        ]
        if value
    }
    return roots, hashes


def validate_inputs(
    *,
    forbidden_roots: set[Path],
    forbidden_hashes: set[str],
    additional_paths: Mapping[str, Path] | None = None,
) -> dict[str, str]:
    named_paths = list(INPUT_PATHS.items()) + [
        (f"additional:{name}", Path(path))
        for name, path in (additional_paths or {}).items()
    ]
    inventory: dict[str, str] = {}
    for name, path in named_paths:
        resolved = path.resolve()
        if path.is_symlink() or not resolved.is_file():
            raise ValueError(f"STOP_V412_EVAL_INPUT: non-regular input {name}")
        if any(
            resolved == root or resolved.is_relative_to(root)
            for root in forbidden_roots
        ):
            raise ValueError(f"STOP_FORBIDDEN_INPUT: forbidden path {name}")
        digest = file_sha256(resolved)
        if digest in forbidden_hashes:
            raise ValueError(f"STOP_FORBIDDEN_INPUT: forbidden hash {name}")
        if name in EXPECTED_INPUT_HASHES and digest != EXPECTED_INPUT_HASHES[name]:
            raise ValueError(f"STOP_V412_EVAL_INPUT: hash changed {name}")
        inventory[name] = digest
    return inventory


def assert_inventory_unchanged(
    expected: Mapping[str, str],
    observed: Mapping[str, str],
    *,
    phase: str,
) -> None:
    if dict(observed) != dict(expected):
        raise ValueError(f"STOP_V412_EVAL_TOCTOU: input changed {phase}")


def load_sealed_evidence() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    manifest = json.loads(INPUT_PATHS["evidence_manifest"].read_text("utf-8"))
    if (
        manifest.get("schema_version") != "sireto-v4.12-direct-evidence-1"
        or manifest.get("build_id") != EVIDENCE_BUILD_ID
        or manifest.get("query_count") != 7003
    ):
        raise ValueError("STOP_V412_EVAL_INPUT: evidence manifest changed")
    outputs = manifest.get("outputs") or {}
    expected = {
        "query_evidence.parquet": (
            EXPECTED_EVIDENCE_QUERY_SHA256,
            QUERY_EVIDENCE_COLUMNS,
            7003,
        ),
        "candidate_evidence.parquet": (
            EXPECTED_EVIDENCE_CANDIDATE_SHA256,
            CANDIDATE_EVIDENCE_COLUMNS,
            10275,
        ),
    }
    for filename, (digest, columns, rows) in expected.items():
        record = outputs.get(filename) or {}
        if (
            record.get("sha256") != digest
            or record.get("columns") != columns
            or int(record.get("row_count", -1)) != rows
        ):
            raise ValueError("STOP_V412_EVAL_INPUT: evidence declaration changed")
    query_evidence = pd.read_parquet(
        INPUT_PATHS["evidence_query"], columns=QUERY_EVIDENCE_COLUMNS
    )
    candidate_evidence = pd.read_parquet(
        INPUT_PATHS["evidence_candidate"], columns=CANDIDATE_EVIDENCE_COLUMNS
    )
    validate_evidence(query_evidence, candidate_evidence)
    return query_evidence, candidate_evidence, manifest


def _validate_bundle_metadata() -> tuple[dict[str, Any], list[str]]:
    metadata = json.loads(INPUT_PATHS["acceptor_metadata"].read_text("utf-8"))
    expected_fields = {
        "binary_features",
        "decision_rule",
        "feature_order",
        "model_bundle_id",
        "model_family",
        "monotonic_constraints",
        "scaled_features",
        "scene_dataset_manifest_sha256",
        "schema_version",
        "threshold",
        "training_plan_sha256",
        "unresolved_policy",
    }
    if set(metadata) != expected_fields:
        raise ValueError("STOP_V412_EVAL_INPUT: acceptor metadata fields changed")
    if (
        metadata.get("schema_version") != "sireto-v4.11-acceptor-bundle-1"
        or metadata.get("model_bundle_id") != "9d23bf3deb6b63de"
        or metadata.get("model_family") != "COMPACT_LOGIT"
        or float(metadata.get("threshold", -1)) != FIXED_THRESHOLD
        or metadata.get("unresolved_policy") != "FORCE_REVIEW"
        or metadata.get("decision_rule")
        != "AUTO_MATCH if score >= threshold else REVIEW"
    ):
        raise ValueError("STOP_V412_EVAL_INPUT: frozen acceptor changed")
    feature_order = [str(value) for value in metadata.get("feature_order") or []]
    if len(feature_order) != 80 or len(set(feature_order)) != 80:
        raise ValueError("STOP_V412_EVAL_INPUT: feature order changed")
    return metadata, feature_order


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _external_output(path: Path) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(Path("/Volumes/CATNAT_DATA")):
        raise ValueError("output root must be under /Volumes/CATNAT_DATA")
    return resolved


def evaluate(
    *,
    execution_lock_path: Path,
    output_root: Path,
) -> Path:
    lock, lock_sha256 = validate_execution_lock(execution_lock_path)
    forbidden_roots, forbidden_hashes = validate_allowlist_and_denylist()
    extra_inputs = {
        "execution_lock": execution_lock_path,
        "allowlist": DEFAULT_ALLOWLIST,
        "denylist": DEFAULT_DENYLIST,
        **{
            f"source:{relative}": (
                Path(__file__).resolve().parent.parent / relative
            )
            for relative in SOURCE_PATHS
        },
    }
    pre_read_inventory = validate_inputs(
        forbidden_roots=forbidden_roots,
        forbidden_hashes=forbidden_hashes,
        additional_paths=extra_inputs,
    )

    # Phase boundary: no historical split, model output, scene or model has
    # been deserialized before both sealed evidence parquets pass validation.
    query_evidence, candidate_evidence, evidence_manifest = (
        load_sealed_evidence()
    )
    evidence_seal = {
        "manifest_sha256": EXPECTED_EVIDENCE_MANIFEST_SHA256,
        "query_sha256": EXPECTED_EVIDENCE_QUERY_SHA256,
        "candidate_sha256": EXPECTED_EVIDENCE_CANDIDATE_SHA256,
    }

    split = pd.read_parquet(INPUT_PATHS["split"], columns=SPLIT_COLUMNS)
    ranker = pd.read_parquet(INPUT_PATHS["ranker"], columns=RANKER_COLUMNS)
    metadata, feature_order = _validate_bundle_metadata()
    scene_columns = list(dict.fromkeys([*SCENE_METADATA_COLUMNS, *feature_order]))
    scenes = pd.read_parquet(INPUT_PATHS["scenes"], columns=scene_columns)
    populations = validate_population_parity(
        split, scenes, enforce_canonical=True
    )
    ranker_sirets = validate_ranker_projection(ranker, split, scenes)
    model = joblib.load(INPUT_PATHS["acceptor_model"])
    v411 = score_v411(
        model,
        scenes,
        feature_order=feature_order,
        threshold=FIXED_THRESHOLD,
    )
    decisions = apply_guard_frame(
        v411,
        query_evidence,
        ranker_sirets=ranker_sirets,
        populations=populations,
    )
    metrics, segments = evaluate_comparison_gate(
        decisions, enforce_canonical=True
    )
    post_read_inventory = validate_inputs(
        forbidden_roots=forbidden_roots,
        forbidden_hashes=forbidden_hashes,
        additional_paths=extra_inputs,
    )
    assert_inventory_unchanged(
        pre_read_inventory,
        post_read_inventory,
        phase="during scoring",
    )

    output_root = _external_output(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "execution_lock_sha256": lock_sha256,
        "evidence_seal": evidence_seal,
        "input_hashes": lock["input_hashes"],
        "source_hashes": lock["source_hashes"],
        "threshold": FIXED_THRESHOLD,
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    target = output_root / build_id
    if target.exists():
        raise FileExistsError(f"Immutable V4.12 evaluation exists: {target}")
    staging = Path(tempfile.mkdtemp(prefix=f".{build_id}.tmp-", dir=output_root))
    try:
        decisions.to_parquet(staging / "decisions.parquet", index=False)
        _write_json(staging / "metrics.json", metrics)
        _write_json(staging / "segments.json", segments)
        peak_rss_bytes = _peak_rss_bytes()
        if peak_rss_bytes > 8 * 1024**3:
            raise ValueError("STOP_V412_EVAL: peak RSS exceeds 8 GiB")
        integrity = {
            "evidence_closed_before_historical_inputs": True,
            "evidence_build_id": evidence_manifest["build_id"],
            "split_projection": SPLIT_COLUMNS,
            "ranker_projection": RANKER_COLUMNS,
            "ranker_is_ground_truth_opened": False,
            "retrieval_labels_opened": False,
            "challenge_opened": False,
            "ranker_pool_cap": 100,
            "ranker_pool_modified": False,
            "model_retrained": False,
            "threshold_changed": False,
            "v412_is_pure_veto": True,
            "comparison_dev_only_gate": True,
            "peak_rss_bytes": peak_rss_bytes,
            "peak_rss_limit_bytes": 8 * 1024**3,
        }
        _write_json(staging / "integrity.json", integrity)
        outputs: dict[str, Any] = {}
        for filename in sorted(OUTPUTS):
            path = staging / filename
            record: dict[str, Any] = {
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
            if filename == "decisions.parquet":
                record["row_count"] = len(decisions)
                record["columns"] = DECISION_COLUMNS
            outputs[filename] = record
        manifest = {
            **identity,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "outputs": outputs,
            "verdict": metrics["verdict"],
            "gate": {
                "comparison_dev_only": True,
                "minimum_auto": 600,
                "maximum_error_auto": 0,
                "maximum_ambiguous_auto": 0,
                "segment_minimum_rows": 100,
                "segment_maximum_coverage_loss_points": 2,
            },
            "phase_ledger": [
                {
                    "phase": "EVIDENCE_SEAL_VALIDATED",
                    "hashes": evidence_seal,
                },
                {
                    "phase": "POST_EVIDENCE_SEAL",
                    "deserialized": [
                        "split_assignments.parquet:SPLIT_COLUMNS",
                        "predictions_ranker_c_oof_dev.parquet:RANKER_COLUMNS",
                        "acceptor_scenes.parquet:SCENE_METADATA+FEATURE_ORDER",
                        "acceptor_model.joblib",
                    ],
                },
            ],
            "latency_gate_evaluated": False,
            "production_certified": False,
        }
        _write_json(staging / "manifest.json", manifest)
        for path in sorted(staging.iterdir()):
            _fsync_file(path)
        _fsync_directory(staging)
        _fsync_directory(output_root)
        final_inventory = validate_inputs(
            forbidden_roots=forbidden_roots,
            forbidden_hashes=forbidden_hashes,
            additional_paths=extra_inputs,
        )
        assert_inventory_unchanged(
            pre_read_inventory,
            final_inventory,
            phase="before publication",
        )
        os.replace(staging, target)
        _fsync_directory(target)
        _fsync_directory(output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def validate_decision_coherence(decisions: pd.DataFrame) -> None:
    """Recompute every stored correctness and V4.11/V4.12 decision field."""

    if not pd.api.types.is_bool_dtype(
        decisions["correct_exact_siret"].dtype
    ):
        raise ValueError("STOP_V412_EVAL_ARTIFACT: correctness type changed")
    predicted = decisions["predicted_siret"].fillna("").astype(str)
    truth = decisions["ground_truth_siret"].fillna("").astype(str)
    expected_correct = decisions["label_kind"].eq(
        "MATCH_EXACT"
    ) & predicted.eq(truth)
    if not decisions["correct_exact_siret"].eq(expected_correct).all():
        raise ValueError("STOP_V412_EVAL_ARTIFACT: correctness was not recomputed")
    if (
        decisions["acceptor_target"].isna().any()
        or not pd.api.types.is_integer_dtype(
            decisions["acceptor_target"].dtype
        )
        or not set(decisions["acceptor_target"].astype(int)).issubset({0, 1})
        or not decisions["acceptor_target"].astype(bool).eq(
            expected_correct
        ).all()
    ):
        raise ValueError("STOP_V412_EVAL_ARTIFACT: acceptor target changed")
    if (
        not pd.api.types.is_float_dtype(decisions["acceptor_score"].dtype)
        or not decisions["acceptor_score"].map(math.isfinite).all()
    ):
        raise ValueError("STOP_V412_EVAL_ARTIFACT: acceptor score type changed")
    valid_top1 = predicted.str.fullmatch(r"\d{14}")
    expected_v411_auto = valid_top1 & decisions["acceptor_score"].ge(
        FIXED_THRESHOLD
    )
    expected_v411_decision = expected_v411_auto.map(
        {True: "AUTO_MATCH", False: "REVIEW"}
    )
    expected_v411_reason = [
        None if auto else ("LOW_CONFIDENCE" if valid else "NO_CANDIDATE")
        for auto, valid in zip(
            expected_v411_auto.tolist(), valid_top1.tolist(), strict=True
        )
    ]
    if not decisions["decision_v411"].astype(str).eq(
        expected_v411_decision
    ).all():
        raise ValueError("STOP_V412_EVAL_ARTIFACT: V4.11 decision changed")

    def nullable(value: Any) -> str | None:
        return None if pd.isna(value) else str(value)

    if [
        nullable(value) for value in decisions["review_reason_v411"]
    ] != expected_v411_reason:
        raise ValueError("STOP_V412_EVAL_ARTIFACT: V4.11 reason changed")
    expected_v412 = [
        apply_guard(
            decision_v411=str(row.decision_v411),
            review_reason_v411=nullable(row.review_reason_v411),
            predicted_siret=nullable(row.predicted_siret),
            direct_candidate_count=row.direct_candidate_count,
            sole_direct_siret=nullable(row.sole_direct_siret),
        )
        for row in decisions.itertuples(index=False)
    ]
    if decisions["decision_v412"].astype(str).tolist() != [
        decision for decision, _ in expected_v412
    ] or [
        nullable(value) for value in decisions["review_reason_v412"]
    ] != [
        reason for _, reason in expected_v412
    ]:
        raise ValueError("STOP_V412_EVAL_ARTIFACT: V4.12 decision changed")


def validate_artifact(path: Path) -> None:
    root = Path(path).resolve()
    if Path(path).is_symlink() or not root.is_dir():
        raise ValueError("STOP_V412_EVAL_ARTIFACT: root is not a regular directory")
    entries = list(root.iterdir())
    expected_names = {"manifest.json", *OUTPUTS}
    if {entry.name for entry in entries} != expected_names:
        raise ValueError("STOP_V412_EVAL_ARTIFACT: top-level files changed")
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise ValueError("STOP_V412_EVAL_ARTIFACT: non-regular output")
    manifest = json.loads((root / "manifest.json").read_text("utf-8"))
    if set(manifest) != MANIFEST_FIELDS:
        raise ValueError("STOP_V412_EVAL_ARTIFACT: manifest fields changed")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("STOP_V412_EVAL_ARTIFACT: schema changed")
    identity = {
        key: manifest[key]
        for key in (
            "schema_version",
            "execution_lock_sha256",
            "evidence_seal",
            "input_hashes",
            "source_hashes",
            "threshold",
        )
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    if root.name != build_id or manifest.get("build_id") != build_id:
        raise ValueError("STOP_V412_EVAL_ARTIFACT: identity changed")
    if set(manifest.get("outputs") or {}) != OUTPUTS:
        raise ValueError("STOP_V412_EVAL_ARTIFACT: outputs changed")
    for filename, record in manifest["outputs"].items():
        expected_record_fields = {"sha256", "size_bytes"}
        if filename == "decisions.parquet":
            expected_record_fields |= {"row_count", "columns"}
        if set(record) != expected_record_fields:
            raise ValueError(
                f"STOP_V412_EVAL_ARTIFACT: output declaration changed {filename}"
            )
        output = root / filename
        if (
            file_sha256(output) != record.get("sha256")
            or output.stat().st_size != int(record.get("size_bytes", -1))
        ):
            raise ValueError(f"STOP_V412_EVAL_ARTIFACT: hash changed {filename}")
    decisions = pd.read_parquet(root / "decisions.parquet")
    decision_record = manifest["outputs"]["decisions.parquet"]
    if (
        list(decisions.columns) != DECISION_COLUMNS
        or decision_record.get("columns") != DECISION_COLUMNS
        or int(decision_record.get("row_count", -1)) != len(decisions)
        or len(decisions) != sum(CANONICAL_COUNTS.values())
    ):
        raise ValueError("STOP_V412_EVAL_ARTIFACT: decision schema changed")
    if decisions["query_id"].astype(str).duplicated().any():
        raise ValueError("STOP_V412_EVAL_ARTIFACT: duplicate decision")
    if decisions["population"].value_counts().to_dict() != CANONICAL_COUNTS:
        raise ValueError("STOP_V412_EVAL_ARTIFACT: population counts changed")
    validate_decision_coherence(decisions)
    metrics = json.loads((root / "metrics.json").read_text("utf-8"))
    segments = json.loads((root / "segments.json").read_text("utf-8"))
    expected_metrics, expected_segments = evaluate_comparison_gate(
        decisions, enforce_canonical=True
    )
    if metrics != expected_metrics or segments != expected_segments:
        raise ValueError("STOP_V412_EVAL_ARTIFACT: metrics differ from decisions")
    integrity = json.loads((root / "integrity.json").read_text("utf-8"))
    expected_integrity_fields = {
        "evidence_closed_before_historical_inputs",
        "evidence_build_id",
        "split_projection",
        "ranker_projection",
        "ranker_is_ground_truth_opened",
        "retrieval_labels_opened",
        "challenge_opened",
        "ranker_pool_cap",
        "ranker_pool_modified",
        "model_retrained",
        "threshold_changed",
        "v412_is_pure_veto",
        "comparison_dev_only_gate",
        "peak_rss_bytes",
        "peak_rss_limit_bytes",
    }
    required_integrity = {
        "evidence_closed_before_historical_inputs": True,
        "ranker_is_ground_truth_opened": False,
        "retrieval_labels_opened": False,
        "challenge_opened": False,
        "ranker_pool_cap": 100,
        "ranker_pool_modified": False,
        "model_retrained": False,
        "threshold_changed": False,
        "v412_is_pure_veto": True,
        "comparison_dev_only_gate": True,
        "peak_rss_limit_bytes": 8 * 1024**3,
    }
    if (
        set(integrity) != expected_integrity_fields
        or any(
            integrity.get(key) != value
            for key, value in required_integrity.items()
        )
        or integrity.get("evidence_build_id") != EVIDENCE_BUILD_ID
        or integrity.get("split_projection") != SPLIT_COLUMNS
        or integrity.get("ranker_projection") != RANKER_COLUMNS
    ):
        raise ValueError("STOP_V412_EVAL_ARTIFACT: integrity declaration changed")
    if int(integrity.get("peak_rss_bytes", 8 * 1024**3 + 1)) > 8 * 1024**3:
        raise ValueError("STOP_V412_EVAL_ARTIFACT: RSS gate failed")
    expected_gate = {
        "comparison_dev_only": True,
        "minimum_auto": 600,
        "maximum_error_auto": 0,
        "maximum_ambiguous_auto": 0,
        "segment_minimum_rows": 100,
        "segment_maximum_coverage_loss_points": 2,
    }
    expected_phases = [
        {
            "phase": "EVIDENCE_SEAL_VALIDATED",
            "hashes": manifest["evidence_seal"],
        },
        {
            "phase": "POST_EVIDENCE_SEAL",
            "deserialized": [
                "split_assignments.parquet:SPLIT_COLUMNS",
                "predictions_ranker_c_oof_dev.parquet:RANKER_COLUMNS",
                "acceptor_scenes.parquet:SCENE_METADATA+FEATURE_ORDER",
                "acceptor_model.joblib",
            ],
        },
    ]
    if (
        manifest.get("evidence_seal")
        != {
            "manifest_sha256": EXPECTED_EVIDENCE_MANIFEST_SHA256,
            "query_sha256": EXPECTED_EVIDENCE_QUERY_SHA256,
            "candidate_sha256": EXPECTED_EVIDENCE_CANDIDATE_SHA256,
        }
        or float(manifest.get("threshold", -1)) != FIXED_THRESHOLD
        or manifest.get("input_hashes", {}).get("evidence_manifest")
        != EXPECTED_EVIDENCE_MANIFEST_SHA256
        or
        manifest.get("gate") != expected_gate
        or manifest.get("phase_ledger") != expected_phases
        or manifest.get("verdict") != metrics["verdict"]
        or manifest.get("latency_gate_evaluated") is not False
        or manifest.get("production_certified") is not False
        or metrics.get("latency_gate_evaluated") is not False
        or metrics.get("production_certified") is not False
    ):
        raise ValueError("STOP_V412_EVAL_ARTIFACT: gate declaration changed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-lock", type=Path)
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
        evaluate(
            execution_lock_path=args.execution_lock,
            output_root=args.output_root,
        )
    )


if __name__ == "__main__":
    main()
