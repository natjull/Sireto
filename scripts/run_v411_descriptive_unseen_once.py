#!/usr/bin/env python3
"""Execute the frozen V4.11 stack once on a sealed descriptive challenge."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_v411_input_blind_dataset import (  # noqa: E402
    CANDIDATE_CEILING,
    RANKER_C_FEATURE_ORDER,
    RawCandidateWriter,
    bulk_hydrate_snapshot,
    finalize_hydrated_pools,
    input_blind_retrieval_config,
    retrieve_raw_input_blind_query,
    tfidf_cache_namespace,
)
from scripts.build_v411_unseen_qualification import (  # noqa: E402
    LABEL_COLUMNS,
)
from scripts.build_v411_unseen_sanitized import (  # noqa: E402
    QUERY_COLUMNS,
    SCHEMA_VERSION as SANITIZED_SCHEMA_VERSION,
    validate_query_schema,
)
from scripts.build_v46_aligned_dataset import (  # noqa: E402
    _path_signature,
    validate_v42_runtime,
)
from src.xgb_matcher.partitioned_store import PartitionedCandidateStore  # noqa: E402
from src.xgb_matcher.tfidf_cache import TfidfPersistentCache  # noqa: E402
from src.xgb_matcher.v41_retrieval import (  # noqa: E402
    V41CurrentStateStore,
    V41RetrievalConfig,
    V41RetrievalVariant,
)
from src.xgb_matcher.v411_scene import (  # noqa: E402
    V411_ACCEPTOR_FEATURE_NAMES,
    build_v411_compact_scene,
)
from src.xgb_matcher.v49_site_function import SiteFunctionTaxonomy  # noqa: E402
from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


SCHEMA_VERSION = "sireto-v4.11-descriptive-unseen-one-shot-1"
LOCK_SCHEMA_VERSION = "sireto-v4.11-descriptive-unseen-execution-lock-1"
EXPERIMENT_ID = "V411_DESCRIPTIVE_UNSEEN_225_ONE_SHOT"
EXPECTED_ROWS = 225
EXPECTED_COHORT_COUNTS = {
    "DESCRIPTIVE_UNSEEN_BLIND_222": 222,
    "EXPOSED_3": 3,
}
FIXED_THRESHOLD = 0.8720916706888049
EXPECTED_STACK_SHA256 = (
    "81279978f47e1e2b1b4a1ea85d595b8dedd8ee8a073e34a19b3ffd340c945d5a"
)
EXPECTED_CANDIDATE_MANIFEST_SHA256 = (
    "a7fc765fe439392baec61fa8a35a941bb1f778281ccdbb54b55c699e9f0c11d9"
)
EXPECTED_RANKER_MODEL_SHA256 = (
    "f4b71b49ed4f879b88e05e4fb84229d0306c5e8ca96958ac20ad97fcc04349c0"
)
EXPECTED_ACCEPTOR_MODEL_SHA256 = (
    "a804feb64f28c417adda4418724f53df50b20d3d308b3e7c778c7189d368e3cf"
)
EXPECTED_ACCEPTOR_METADATA_SHA256 = (
    "e4b99676e695d19748b71a7657ff5a1f5c7dfa2879754dd2e1b15c8906a61d6b"
)
EXPECTED_SNAPSHOT_SHA256 = (
    "c91180cc5bae86948dd57d752c9bae45e58cc64653e99d5a9357664b67300845"
)
EXPECTED_PARTITIONS_SIGNATURE = (
    "2f6668f60da8bc9fe52b683b32ef35641803679c01f8c8fd124e2e86a41e2b82"
)
EXPECTED_TFIDF_NAMESPACE = (
    "296c7891107249a073c00d93c7310c55a652243de4bcfa7165d09dbfc3349a82"
)
EXPECTED_CONTRACT_SHA256 = (
    "28785be1c776f27b9dc9357fe543049bb70d6937b6b03d6f59c33eee67f43026"
)
EXPECTED_RUNTIME = {
    "numpy": "2.4.2",
    "pandas": "2.3.3",
    "pyarrow": "23.0.1",
    "scikit-learn": "1.8.0",
    "xgboost": "3.1.2",
    "joblib": "1.5.3",
    "duckdb": "1.4.3",
    "rapidfuzz": "3.14.3",
    "scipy": "1.16.3",
}
DEFAULT_BUNDLE_ROOT = Path(
    "/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/"
    "v4_11_acceptor/9d23bf3deb6b63de"
)
DEFAULT_V42_MANIFEST = Path(
    "/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/"
    "v4_2_retrieval_integrity_7c4b957/manifest.json"
)
DEFAULT_GLOBAL_STORE = Path(
    "/Volumes/CATNAT_DATA/SIRETO_V9/stores/"
    "siren_candidates_v7_v2_43f2c64"
)
DEFAULT_PARTITIONS = Path("data/candidates_v7_all")
DEFAULT_SNAPSHOT = Path("data/StockEtablissement_utf8.parquet")
DEFAULT_TAXONOMY = Path("config/v4_9_site_function_taxonomy.json")
DEFAULT_CONTRACT = Path("docs/v4_11_descriptive_unseen_225_contract.md")
GLOBAL_OPENING_LEDGER = Path(
    "/Volumes/CATNAT_DATA/SIRETO_RECALL100/challenges/"
    "v4_11_unseen_control/OPENING_LEDGER.json"
)

RAW_PREDICTION_COLUMNS = [
    "query_id",
    "crm_record_id",
    "predicted_siret",
    "predicted_siren",
    "candidate_count",
    "ranker_score_top1",
    "acceptor_score",
    "threshold",
    "decision",
    "review_reason",
]
SCORED_CANDIDATE_COLUMNS = [
    "query_id",
    "candidate_siret",
    "candidate_siren",
    "retrieval_rank",
    "ranker_score",
    "ranker_rank",
]
_REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
INFERENCE_SOURCE_PATHS = [
    "scripts/build_v411_input_blind_dataset.py",
    "scripts/build_v411_unseen_qualification.py",
    "scripts/build_v411_unseen_sanitized.py",
    "scripts/build_v46_aligned_dataset.py",
    "scripts/build_v41_training_dataset.py",
    *[
        str(path.relative_to(_REPOSITORY_ROOT))
        for path in sorted(
            (_REPOSITORY_ROOT / "src" / "xgb_matcher").glob("*.py")
        )
    ],
]


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _durable_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = _json_bytes(payload)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def create_exclusive_ledger(path: Path, payload: Mapping[str, Any]) -> None:
    """Create the global one-shot receipt atomically; existence forbids rerun."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o444,
    )
    try:
        os.write(descriptor, _json_bytes(payload))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def update_ledger(path: Path, *, expected_run_id: str, **changes: Any) -> None:
    current = json.loads(Path(path).read_text(encoding="utf-8"))
    if current.get("run_id") != expected_run_id:
        raise ValueError("STOP_DESCRIPTIVE_INTEGRITY: ledger run mismatch")
    current.update(changes)
    current["updated_at"] = datetime.now(timezone.utc).isoformat()
    _durable_json(Path(path), current)


def assert_predictions_sealed(
    ledger_path: Path,
    predictions_path: Path,
    *,
    expected_query_ids: set[str],
) -> dict[str, Any]:
    ledger = json.loads(Path(ledger_path).read_text(encoding="utf-8"))
    if (
        ledger.get("phase") != "PREDICTIONS_SEALED"
        or not ledger.get("predictions_sha256")
    ):
        raise ValueError(
            "STOP_DESCRIPTIVE_INTEGRITY: labels cannot open before predictions seal"
        )
    if file_sha256(predictions_path) != ledger["predictions_sha256"]:
        raise ValueError(
            "STOP_DESCRIPTIVE_INTEGRITY: sealed predictions hash mismatch"
        )
    schema = list(pd.read_parquet(predictions_path).columns)
    query_ids = set(
        pd.read_parquet(predictions_path, columns=["query_id"])[
            "query_id"
        ].astype(str)
    )
    if schema != RAW_PREDICTION_COLUMNS or query_ids != expected_query_ids:
        raise ValueError(
            "STOP_DESCRIPTIVE_INTEGRITY: sealed predictions population changed"
        )
    return ledger


def open_labels_after_seal(
    labels_path: Path,
    ledger_path: Path,
    predictions_path: Path,
    *,
    expected_query_ids: set[str],
    loader: Callable[..., pd.DataFrame] = pd.read_parquet,
) -> pd.DataFrame:
    """The sole labels deserialization boundary."""

    assert_predictions_sealed(
        ledger_path,
        predictions_path,
        expected_query_ids=expected_query_ids,
    )
    return loader(labels_path, columns=LABEL_COLUMNS)


def _runtime_versions() -> dict[str, str]:
    return {
        package: importlib.metadata.version(package)
        for package in EXPECTED_RUNTIME
    }


def _output_path(
    root: Path,
    manifest: Mapping[str, Any],
    filename: str,
) -> Path:
    record = (manifest.get("outputs") or {}).get(filename)
    if not isinstance(record, Mapping):
        raise ValueError(
            f"STOP_DESCRIPTIVE_INTEGRITY: undeclared output {filename}"
        )
    path = root / filename
    if file_sha256(path) != str(record.get("sha256") or ""):
        raise ValueError(
            f"STOP_DESCRIPTIVE_INTEGRITY: output hash mismatch {filename}"
        )
    return path


def validate_sanitized_artifact(root: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != SANITIZED_SCHEMA_VERSION
        or manifest.get("experiment_id") != "V411_DESCRIPTIVE_UNSEEN_SANITIZED"
    ):
        raise ValueError("STOP_DESCRIPTIVE_INTEGRITY: sanitized artifact changed")
    queries_path = _output_path(root, manifest, "queries_sanitized.parquet")
    queries = pd.read_parquet(queries_path, columns=QUERY_COLUMNS)
    validate_query_schema(queries)
    if len(queries) != EXPECTED_ROWS:
        raise ValueError("STOP_DESCRIPTIVE_INTEGRITY: sanitized count changed")
    # The mapping is deliberately not opened here.
    _output_path(root, manifest, "sealed_mapping.parquet")
    return manifest, queries


def validate_qualification_manifest_only(
    root: Path,
    *,
    sanitized_manifest_sha256: str,
) -> tuple[dict[str, Any], Path, Path]:
    """Validate and hash qualification outputs without deserializing either."""

    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version")
        != "sireto-v4.11-unseen-mechanical-qualification-1"
        or manifest.get("experiment_id")
        != "V411_DESCRIPTIVE_UNSEEN_225_QUALIFICATION"
    ):
        raise ValueError(
            "STOP_DESCRIPTIVE_INTEGRITY: qualification artifact changed"
        )
    identity = manifest.get("build_identity") or {}
    expected_build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    if manifest.get("build_id") != expected_build_id or root.name != expected_build_id:
        raise ValueError(
            "STOP_DESCRIPTIVE_INTEGRITY: qualification identity mismatch"
        )
    if identity.get("sanitized_manifest_sha256") != sanitized_manifest_sha256:
        raise ValueError(
            "STOP_DESCRIPTIVE_INTEGRITY: qualification/sanitized mismatch"
        )
    invariants = manifest.get("invariants") or {}
    required = {
        "source_registry_opened": False,
        "models_or_scores_opened": False,
        "retrieval_topk_used": False,
        "full_geographic_universe_used": True,
        "active_only": True,
        "no_match_created": False,
        "labels_and_evidence_closed_atomically": True,
        "human_validated": False,
    }
    if any(invariants.get(key) is not value for key, value in required.items()):
        raise ValueError(
            "STOP_DESCRIPTIVE_INTEGRITY: qualification invariants changed"
        )
    labels_path = _output_path(root, manifest, "labels_frozen.parquet")
    evidence_path = _output_path(root, manifest, "evidence.parquet")
    labels_record = manifest["outputs"]["labels_frozen.parquet"]
    if (
        int(labels_record.get("row_count", -1)) != EXPECTED_ROWS
        or list(labels_record.get("columns") or []) != LABEL_COLUMNS
    ):
        raise ValueError(
            "STOP_DESCRIPTIVE_INTEGRITY: frozen label declaration changed"
        )
    return manifest, labels_path, evidence_path


def validate_frozen_bundle(bundle_root: Path) -> dict[str, Path]:
    bundle_root = Path(bundle_root).resolve()
    manifest_path = bundle_root / "manifest.json"
    if file_sha256(manifest_path) != EXPECTED_CANDIDATE_MANIFEST_SHA256:
        raise ValueError("STOP_DESCRIPTIVE_INTEGRITY: candidate manifest drift")
    parent = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        parent.get("build_id") != "9d23bf3deb6b63de"
        or parent.get("verdict") != "GO_FREEZE_V411_CANDIDATE"
        or parent.get("winner") != "COMPACT_LOGIT"
    ):
        raise ValueError("STOP_DESCRIPTIVE_INTEGRITY: candidate is not frozen GO")
    stack_path = bundle_root / "bundle/stack_manifest.json"
    if file_sha256(stack_path) != EXPECTED_STACK_SHA256:
        raise ValueError("STOP_DESCRIPTIVE_INTEGRITY: stack manifest drift")
    stack = json.loads(stack_path.read_text(encoding="utf-8"))
    components = stack.get("components") or {}
    ranker = components.get("ranker_c") or {}
    acceptor = components.get("acceptor") or {}
    scene = components.get("scene") or {}
    ranker_model = Path(str(ranker.get("model_path") or "")).resolve()
    acceptor_model = bundle_root / "bundle" / str(
        acceptor.get("model_path") or ""
    )
    acceptor_metadata = bundle_root / "bundle" / str(
        acceptor.get("metadata_path") or ""
    )
    taxonomy = Path(str(scene.get("taxonomy_path") or "")).resolve()
    paths = {
        "candidate_manifest": manifest_path,
        "stack_manifest": stack_path,
        "ranker_model": ranker_model,
        "acceptor_model": acceptor_model,
        "acceptor_metadata": acceptor_metadata,
        "taxonomy": taxonomy,
        "scene_source": Path(str(scene.get("source_path") or "")).resolve(),
        "site_function_source": Path(
            str(scene.get("site_function_source_path") or "")
        ).resolve(),
    }
    expected_hashes = {
        "ranker_model": EXPECTED_RANKER_MODEL_SHA256,
        "acceptor_model": EXPECTED_ACCEPTOR_MODEL_SHA256,
        "acceptor_metadata": EXPECTED_ACCEPTOR_METADATA_SHA256,
        "taxonomy": str(scene.get("taxonomy_sha256") or ""),
        "scene_source": str(scene.get("source_sha256") or ""),
        "site_function_source": str(
            scene.get("site_function_source_sha256") or ""
        ),
    }
    for name, expected in expected_hashes.items():
        if not expected or file_sha256(paths[name]) != expected:
            raise ValueError(
                f"STOP_DESCRIPTIVE_INTEGRITY: frozen component drift {name}"
            )
    metadata = json.loads(acceptor_metadata.read_text(encoding="utf-8"))
    if (
        metadata.get("model_family") != "COMPACT_LOGIT"
        or float(metadata.get("threshold", -1)) != FIXED_THRESHOLD
        or list(metadata.get("feature_order") or [])
        != V411_ACCEPTOR_FEATURE_NAMES
    ):
        raise ValueError("STOP_DESCRIPTIVE_INTEGRITY: acceptor metadata changed")
    return paths


def validate_execution_lock(
    path: Path,
    *,
    sanitized_manifest_sha256: str,
    qualification_manifest_sha256: str,
    labels_sha256: str,
    verify_git: bool = True,
) -> tuple[dict[str, Any], str]:
    raw = Path(path).read_bytes()
    lock = json.loads(raw)
    required_fields = {
        "schema_version",
        "purpose",
        "runner_sha256",
        "runner_commit",
        "sanitized_manifest_sha256",
        "qualification_manifest_sha256",
        "labels_sha256",
        "stack_manifest_sha256",
        "ranker_model_sha256",
        "acceptor_model_sha256",
        "threshold",
        "snapshot_sha256",
        "partitions_signature",
        "tfidf_cache_namespace",
        "runtime",
        "source_hashes",
    }
    if set(lock) != required_fields:
        raise ValueError("STOP_DESCRIPTIVE_INTEGRITY: execution lock fields changed")
    expected = {
        "schema_version": LOCK_SCHEMA_VERSION,
        "purpose": "DESCRIPTIVE_UNSEEN_225_ONCE",
        "runner_sha256": file_sha256(Path(__file__).resolve()),
        "sanitized_manifest_sha256": sanitized_manifest_sha256,
        "qualification_manifest_sha256": qualification_manifest_sha256,
        "labels_sha256": labels_sha256,
        "stack_manifest_sha256": EXPECTED_STACK_SHA256,
        "ranker_model_sha256": EXPECTED_RANKER_MODEL_SHA256,
        "acceptor_model_sha256": EXPECTED_ACCEPTOR_MODEL_SHA256,
        "threshold": FIXED_THRESHOLD,
        "snapshot_sha256": EXPECTED_SNAPSHOT_SHA256,
        "partitions_signature": EXPECTED_PARTITIONS_SIGNATURE,
        "tfidf_cache_namespace": EXPECTED_TFIDF_NAMESPACE,
        "runtime": EXPECTED_RUNTIME,
        "source_hashes": {
            relative: file_sha256(
                Path(__file__).resolve().parent.parent / relative
            )
            for relative in INFERENCE_SOURCE_PATHS
        },
    }
    for key, value in expected.items():
        if lock.get(key) != value:
            raise ValueError(
                f"STOP_DESCRIPTIVE_INTEGRITY: execution lock mismatch {key}"
            )
    commit = str(lock.get("runner_commit") or "")
    if not commit:
        raise ValueError("STOP_DESCRIPTIVE_INTEGRITY: runner commit missing")
    if verify_git:
        committed_paths = {
            "scripts/run_v411_descriptive_unseen_once.py":
                expected["runner_sha256"],
            **expected["source_hashes"],
        }
        for relative, expected_sha in committed_paths.items():
            result = subprocess.run(
                ["git", "show", f"{commit}:{relative}"],
                cwd=Path(__file__).resolve().parent.parent,
                check=True,
                capture_output=True,
            )
            if hashlib.sha256(result.stdout).hexdigest() != expected_sha:
                raise ValueError(
                    f"STOP_DESCRIPTIVE_INTEGRITY: commit does not pin {relative}"
                )
    return lock, hashlib.sha256(raw).hexdigest()


def validate_candidate_pools(candidates: pd.DataFrame) -> None:
    required = {
        "query_id",
        "candidate_siret",
        "candidate_siren",
        "candidate_state",
        "retrieval_rank",
        *RANKER_C_FEATURE_ORDER,
    }
    if not required.issubset(candidates.columns):
        raise ValueError("STOP_DESCRIPTIVE_INTEGRITY: candidate schema incomplete")
    if candidates.duplicated(["query_id", "candidate_siret"]).any():
        raise ValueError("STOP_DESCRIPTIVE_INTEGRITY: duplicate candidate SIRET")
    if not candidates["candidate_state"].astype(str).eq("A").all():
        raise ValueError("STOP_DESCRIPTIVE_INTEGRITY: non-active candidate")
    sizes = candidates.groupby("query_id").size()
    if not sizes.empty and int(sizes.max()) > CANDIDATE_CEILING:
        raise ValueError("STOP_DESCRIPTIVE_INTEGRITY: candidate ceiling exceeded")
    for _, group in candidates.groupby("query_id", sort=False):
        ranks = sorted(group["retrieval_rank"].astype(int).tolist())
        if ranks != list(range(1, len(group) + 1)):
            raise ValueError(
                "STOP_DESCRIPTIVE_INTEGRITY: retrieval ranks are not contiguous"
            )
    matrix = candidates[RANKER_C_FEATURE_ORDER].to_numpy(dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("STOP_DESCRIPTIVE_INTEGRITY: non-finite ranker feature")


def score_ranker_candidates(model: Any, candidates: pd.DataFrame) -> pd.DataFrame:
    """Score a label-free pool; truth metadata is explicitly forbidden."""

    forbidden = {
        "is_ground_truth",
        "ground_truth_siret",
        "ground_truth_siren",
        "label_kind",
    }
    if forbidden & set(candidates.columns):
        raise ValueError(
            "STOP_DESCRIPTIVE_INTEGRITY: ranker scorer received truth metadata"
        )
    validate_candidate_pools(candidates)
    output = candidates.copy()
    output["ranker_score"] = np.asarray(
        model.predict(
            output[RANKER_C_FEATURE_ORDER].to_numpy(dtype=np.float32)
        ),
        dtype=np.float32,
    )
    if not np.isfinite(output["ranker_score"].to_numpy(dtype=np.float64)).all():
        raise ValueError("STOP_DESCRIPTIVE_INTEGRITY: non-finite ranker score")
    output = output.sort_values(
        ["query_id", "ranker_score", "retrieval_rank", "candidate_siret"],
        ascending=[True, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    output["ranker_rank"] = (
        output.groupby("query_id", sort=False).cumcount() + 1
    ).astype(np.int16)
    if forbidden & set(output.columns):
        raise AssertionError("truth metadata escaped label-free ranker scorer")
    return output


def build_blind_predictions(
    *,
    queries: pd.DataFrame,
    scored_candidates: pd.DataFrame,
    taxonomy: SiteFunctionTaxonomy,
    acceptor: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pools = {
        str(query_id): frame
        for query_id, frame in scored_candidates.groupby("query_id", sort=False)
    }
    scene_rows: list[dict[str, Any]] = []
    for query in queries.sort_values("query_id", kind="mergesort").to_dict(
        "records"
    ):
        query_id = str(query["query_id"])
        pool = pools.get(query_id, scored_candidates.iloc[0:0])
        scene = build_v411_compact_scene(query, pool, taxonomy)
        top1_score = (
            None
            if pool.empty
            else float(
                pool.sort_values(
                    ["ranker_score", "retrieval_rank", "candidate_siret"],
                    ascending=[False, True, True],
                    kind="mergesort",
                ).iloc[0]["ranker_score"]
            )
        )
        scene_rows.append(
            {
                "query_id": query_id,
                "crm_record_id": str(query["crm_record_id"]),
                "ranker_score_top1": top1_score,
                **scene,
            }
        )
    scenes = pd.DataFrame(scene_rows)
    matrix = scenes[V411_ACCEPTOR_FEATURE_NAMES].to_numpy(dtype=np.float64)
    scores = np.asarray(acceptor.predict_proba(matrix)[:, 1], dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("STOP_DESCRIPTIVE_INTEGRITY: non-finite acceptor score")
    predictions = scenes[
        [
            "query_id",
            "crm_record_id",
            "predicted_siret",
            "predicted_siren",
            "candidate_count",
            "ranker_score_top1",
        ]
    ].copy()
    predictions["acceptor_score"] = scores
    predictions["threshold"] = FIXED_THRESHOLD
    no_candidate = predictions["candidate_count"].astype(float).eq(0)
    predictions["decision"] = np.where(
        ~no_candidate & predictions["acceptor_score"].ge(FIXED_THRESHOLD),
        "AUTO_MATCH",
        "REVIEW",
    )
    predictions["review_reason"] = np.where(
        no_candidate,
        "NO_CANDIDATE",
        np.where(
            predictions["decision"].eq("REVIEW"),
            "LOW_CONFIDENCE",
            None,
        ),
    )
    return predictions[RAW_PREDICTION_COLUMNS], scenes


def _wilson(successes: int, total: int) -> list[float]:
    if total <= 0:
        return [0.0, 1.0]
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    centre = p + z * z / (2 * total)
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return [(centre - margin) / denominator, (centre + margin) / denominator]


def evaluate_after_seal(
    *,
    predictions: pd.DataFrame,
    scored_candidates: pd.DataFrame,
    labels: pd.DataFrame,
    mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if list(labels.columns) != LABEL_COLUMNS or len(labels) != EXPECTED_ROWS:
        raise ValueError("STOP_DESCRIPTIVE_INTEGRITY: label schema/count changed")
    if list(mapping.columns) != ["query_id", "source_row_number", "cohort"]:
        raise ValueError("STOP_DESCRIPTIVE_INTEGRITY: cohort mapping changed")
    prediction_ids = set(predictions["query_id"].astype(str))
    label_ids = set(labels["query_id"].astype(str))
    mapping_ids = set(mapping["query_id"].astype(str))
    if (
        len(predictions) != EXPECTED_ROWS
        or len(labels) != EXPECTED_ROWS
        or len(mapping) != EXPECTED_ROWS
        or predictions["query_id"].astype(str).duplicated().any()
        or labels["query_id"].astype(str).duplicated().any()
        or mapping["query_id"].astype(str).duplicated().any()
        or not (prediction_ids == label_ids == mapping_ids)
    ):
        raise ValueError(
            "STOP_DESCRIPTIVE_INTEGRITY: evaluation populations differ"
        )
    if mapping["cohort"].value_counts().to_dict() != EXPECTED_COHORT_COUNTS:
        raise ValueError("STOP_DESCRIPTIVE_INTEGRITY: cohort counts changed")
    evaluated = (
        predictions.merge(labels, on="query_id", validate="one_to_one")
        .merge(mapping, on="query_id", validate="one_to_one")
    )
    pool_sirets = (
        scored_candidates.groupby("query_id")["candidate_siret"]
        .agg(lambda values: set(values.astype(str)))
        .to_dict()
    )
    evaluated["retrieval_hit"] = [
        bool(
            label == "MATCH_EXACT"
            and str(truth) in pool_sirets.get(str(query_id), set())
        )
        for query_id, label, truth in zip(
            evaluated["query_id"],
            evaluated["label_kind"],
            evaluated["ground_truth_siret"],
            strict=True,
        )
    ]
    evaluated["ranker_hit"] = (
        evaluated["label_kind"].eq("MATCH_EXACT")
        & evaluated["predicted_siret"].astype(str).eq(
            evaluated["ground_truth_siret"].astype(str)
        )
    )
    evaluated["correct_auto"] = (
        evaluated["decision"].eq("AUTO_MATCH") & evaluated["ranker_hit"]
    )
    evaluated["confirmed_error_auto"] = (
        evaluated["decision"].eq("AUTO_MATCH")
        & (
            evaluated["label_kind"].eq("AMBIGUOUS")
            | (
                evaluated["label_kind"].eq("MATCH_EXACT")
                & ~evaluated["ranker_hit"]
            )
        )
    )
    evaluated["unverifiable_auto"] = (
        evaluated["decision"].eq("AUTO_MATCH")
        & evaluated["label_kind"].eq("UNRESOLVED")
    )
    overlay = evaluated[["query_id", "decision", "label_kind"]].copy()
    overlay = overlay.rename(columns={"decision": "raw_decision"})
    overlay["overlay_decision"] = np.where(
        overlay["label_kind"].eq("UNRESOLVED"),
        "REVIEW",
        overlay["raw_decision"],
    )
    overlay["overlay_reason"] = np.where(
        overlay["label_kind"].eq("UNRESOLVED"),
        "UNRESOLVED_LABEL_OVERLAY",
        None,
    )
    cohorts = {
        "ALL_225": pd.Series(True, index=evaluated.index),
        "DESCRIPTIVE_UNSEEN_BLIND_222": evaluated["cohort"].eq(
            "DESCRIPTIVE_UNSEEN_BLIND_222"
        ),
        "EXPOSED_3": evaluated["cohort"].eq("EXPOSED_3"),
    }
    metrics: dict[str, Any] = {}
    for name, mask in cohorts.items():
        frame = evaluated[mask]
        exact = frame["label_kind"].eq("MATCH_EXACT")
        auto = frame["decision"].eq("AUTO_MATCH")
        auto_count = int(auto.sum())
        correct = int(frame["correct_auto"].sum())
        confirmed_errors = int(frame["confirmed_error_auto"].sum())
        unverifiable = int(frame["unverifiable_auto"].sum())
        evaluable_auto = correct + confirmed_errors
        exact_count = int(exact.sum())
        identifiable = exact
        identifiable_count = exact_count
        identifiable_auto = int((auto & identifiable).sum())
        non_unresolved = ~frame["label_kind"].eq("UNRESOLVED")
        non_unresolved_count = int(non_unresolved.sum())
        non_unresolved_auto = int((auto & non_unresolved).sum())
        metrics[name] = {
            "row_count": len(frame),
            "label_counts": {
                str(key): int(value)
                for key, value in frame["label_kind"].value_counts().to_dict().items()
            },
            "retrieval_recall_at_100": (
                int(frame.loc[exact, "retrieval_hit"].sum()) / exact_count
                if exact_count
                else None
            ),
            "ranker_hit_at_1": (
                int(frame.loc[exact, "ranker_hit"].sum()) / exact_count
                if exact_count
                else None
            ),
            "auto_count": auto_count,
            "correct_auto": correct,
            "confirmed_error_auto": confirmed_errors,
            "unverifiable_auto": unverifiable,
            "evaluable_auto_count": evaluable_auto,
            "precision_evaluable": (
                correct / evaluable_auto if evaluable_auto else None
            ),
            "precision_evaluable_wilson_95": _wilson(
                correct, evaluable_auto
            ),
            "precision_conservative_lower_bound": (
                correct / auto_count if auto_count else None
            ),
            "raw_coverage_all": auto_count / len(frame) if len(frame) else 0.0,
            "identifiable_count": identifiable_count,
            "identifiable_auto_count": identifiable_auto,
            "coverage_identifiable": (
                identifiable_auto / identifiable_count
                if identifiable_count
                else None
            ),
            "non_unresolved_count": non_unresolved_count,
            "non_unresolved_auto_count": non_unresolved_auto,
            "coverage_non_unresolved": (
                non_unresolved_auto / non_unresolved_count
                if non_unresolved_count
                else None
            ),
            "ambiguous_auto": int(
                (auto & frame["label_kind"].eq("AMBIGUOUS")).sum()
            ),
            "unresolved_auto": int(
                (auto & frame["label_kind"].eq("UNRESOLVED")).sum()
            ),
        }
    return evaluated, overlay, metrics


def _seal_parquet(path: Path, frame: pd.DataFrame) -> str:
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
    return file_sha256(path)


def validate_blind_predictions(
    predictions: pd.DataFrame,
    *,
    expected_query_ids: set[str],
) -> None:
    if (
        list(predictions.columns) != RAW_PREDICTION_COLUMNS
        or len(predictions) != len(expected_query_ids)
        or predictions["query_id"].astype(str).duplicated().any()
        or set(predictions["query_id"].astype(str)) != expected_query_ids
        or not predictions["decision"].isin(["AUTO_MATCH", "REVIEW"]).all()
    ):
        raise ValueError(
            "STOP_DESCRIPTIVE_INTEGRITY: blind predictions invalid"
        )


def run_once(  # noqa: C901 - orchestration is intentionally explicit
    *,
    sanitized_artifact: Path,
    qualification_artifact: Path,
    execution_lock_path: Path,
    bundle_root: Path,
    partitions_dir: Path,
    global_store_path: Path,
    snapshot_path: Path,
    v42_manifest_path: Path,
    cache_dir: Path,
    work_dir: Path,
    output_root: Path,
    contract_path: Path = DEFAULT_CONTRACT,
    verify_git: bool = True,
) -> Path:
    if file_sha256(contract_path) != EXPECTED_CONTRACT_SHA256:
        raise ValueError("STOP_DESCRIPTIVE_INTEGRITY: contract drift")
    if _runtime_versions() != EXPECTED_RUNTIME:
        raise ValueError("STOP_DESCRIPTIVE_INTEGRITY: runtime drift")
    sanitized_manifest, queries = validate_sanitized_artifact(sanitized_artifact)
    sanitized_manifest_path = Path(sanitized_artifact) / "manifest.json"
    sanitized_manifest_sha = file_sha256(sanitized_manifest_path)
    qualification_manifest, labels_path, _ = validate_qualification_manifest_only(
        qualification_artifact,
        sanitized_manifest_sha256=sanitized_manifest_sha,
    )
    qualification_manifest_path = Path(qualification_artifact) / "manifest.json"
    qualification_manifest_sha = file_sha256(qualification_manifest_path)
    labels_sha = file_sha256(labels_path)  # bytes only; no deserialization
    frozen = validate_frozen_bundle(bundle_root)
    _, lock_sha = validate_execution_lock(
        execution_lock_path,
        sanitized_manifest_sha256=sanitized_manifest_sha,
        qualification_manifest_sha256=qualification_manifest_sha,
        labels_sha256=labels_sha,
        verify_git=verify_git,
    )
    source_config = V41RetrievalConfig(
        variant=V41RetrievalVariant.B_INPUT_EVIDENCE,
        max_candidates=CANDIDATE_CEILING,
    )
    runtime = validate_v42_runtime(
        manifest_path=v42_manifest_path,
        config=source_config,
        partitions_dir=partitions_dir,
        global_store_path=global_store_path,
        state_snapshot_path=snapshot_path,
    )
    if (
        runtime["partitions_signature"] != EXPECTED_PARTITIONS_SIGNATURE
        or runtime["state_snapshot_sha256"] != EXPECTED_SNAPSHOT_SHA256
    ):
        raise ValueError("STOP_DESCRIPTIVE_INTEGRITY: retrieval runtime drift")
    config = input_blind_retrieval_config()
    if config.sparse_config().to_dict() != source_config.sparse_config().to_dict():
        raise ValueError("STOP_DESCRIPTIVE_INTEGRITY: sparse policy drift")
    namespace = tfidf_cache_namespace(
        sparse_config_hash=config.sparse_config().signature().hash,
        tfidf_artifact_hash=config.sparse_config().tfidf_artifact_hash(),
        partitions_signature=runtime["partitions_signature"],
    )
    if namespace != EXPECTED_TFIDF_NAMESPACE:
        raise ValueError("STOP_DESCRIPTIVE_INTEGRITY: TF-IDF namespace drift")

    identity = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "sanitized_manifest_sha256": sanitized_manifest_sha,
        "qualification_manifest_sha256": qualification_manifest_sha,
        "labels_sha256": labels_sha,
        "execution_lock_sha256": lock_sha,
        "stack_manifest_sha256": EXPECTED_STACK_SHA256,
        "snapshot_sha256": EXPECTED_SNAPSHOT_SHA256,
        "partitions_signature": EXPECTED_PARTITIONS_SIGNATURE,
        "runner_sha256": file_sha256(Path(__file__).resolve()),
        "threshold": FIXED_THRESHOLD,
    }
    run_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / run_id
    if target.exists():
        raise FileExistsError(f"immutable one-shot artifact exists: {target}")
    ledger_path = GLOBAL_OPENING_LEDGER
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{run_id}.tmp-", dir=output_root))
    ledger_created = False
    scoring_started = False
    labels_opened = False
    try:
        create_exclusive_ledger(
            ledger_path,
            {
                "schema_version": "sireto-v4.11-descriptive-opening-ledger-1",
                "challenge_id": "DESCRIPTIVE_UNSEEN_225",
                "run_id": run_id,
                "phase": "PREFLIGHT_PASSED",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "retry_forbidden_after_scoring": True,
                "labels_deserialized": False,
                "predictions_sha256": None,
            },
        )
        ledger_created = True
        raw_path = staging / ".raw_candidates.parquet"
        hydrated_path = staging / ".hydrated_candidates.parquet"
        candidate_path = staging / "retrieval_candidates.parquet"
        writer = RawCandidateWriter(raw_path)
        persistent = TfidfPersistentCache(
            config_hash=namespace,
            cache_dir=Path(cache_dir) / "v411_verified",
            require_verified=True,
        )
        store = PartitionedCandidateStore(partitions_dir)
        duckdb_temp = Path(work_dir) / f"duckdb-{run_id}"
        duckdb_temp.mkdir(parents=True, exist_ok=False)
        try:
            with V41CurrentStateStore(snapshot_path) as state:
                state._connection.execute(  # noqa: SLF001
                    f"SET temp_directory = "
                    f"'{str(duckdb_temp).replace(chr(39), chr(39) * 2)}'"
                )
                memory_cache: dict[tuple[str, str], tuple] = {}
                buffer: list[dict[str, Any]] = []
                for query in queries.sort_values("query_id").to_dict("records"):
                    rows, _ = retrieve_raw_input_blind_query(
                        query=query,
                        partitioned_store=store,
                        config=config,
                        tfidf_cache=memory_cache,
                        persistent_cache=persistent,
                    )
                    buffer.extend(rows)
                    if len(buffer) >= 10_000:
                        writer.write(buffer)
                        buffer.clear()
                writer.write(buffer)
                writer.close()
                bulk_hydrate_snapshot(
                    connection=state._connection,  # noqa: SLF001
                    raw_candidates_path=raw_path,
                    state_snapshot_path=snapshot_path,
                    output_path=hydrated_path,
                )
            finalize_hydrated_pools(
                hydrated_path=hydrated_path,
                output_path=candidate_path,
                queries=queries,
            )
        finally:
            shutil.rmtree(duckdb_temp, ignore_errors=True)
        raw_path.unlink()
        hydrated_path.unlink()
        candidates = pd.read_parquet(candidate_path)
        validate_candidate_pools(candidates)
        label_free = candidates.drop(columns=["is_ground_truth"], errors="raise")

        update_ledger(
            ledger_path,
            expected_run_id=run_id,
            phase="SCORING_STARTED",
        )
        scoring_started = True
        ranker = xgb.XGBRanker()
        ranker.load_model(frozen["ranker_model"])
        scored = score_ranker_candidates(ranker, label_free)
        scored[SCORED_CANDIDATE_COLUMNS].to_parquet(
            staging / "ranked_candidates.parquet", index=False
        )
        taxonomy = SiteFunctionTaxonomy.load(frozen["taxonomy"])
        acceptor = joblib.load(frozen["acceptor_model"])
        predictions, scenes = build_blind_predictions(
            queries=queries,
            scored_candidates=scored,
            taxonomy=taxonomy,
            acceptor=acceptor,
        )
        predictions_path = staging / "predictions_blind.parquet"
        query_ids = set(queries["query_id"].astype(str))
        validate_blind_predictions(
            predictions,
            expected_query_ids=query_ids,
        )
        predictions_sha = _seal_parquet(predictions_path, predictions)
        scenes.to_parquet(staging / "scenes_blind.parquet", index=False)
        update_ledger(
            ledger_path,
            expected_run_id=run_id,
            phase="PREDICTIONS_SEALED",
            predictions_sha256=predictions_sha,
            labels_deserialized=False,
        )

        labels = open_labels_after_seal(
            labels_path,
            ledger_path,
            predictions_path,
            expected_query_ids=query_ids,
        )
        labels_opened = True
        mapping_path = _output_path(
            Path(sanitized_artifact),
            sanitized_manifest,
            "sealed_mapping.parquet",
        )
        mapping = pd.read_parquet(
            mapping_path,
            columns=["query_id", "source_row_number", "cohort"],
        )
        evaluated, overlay, metrics = evaluate_after_seal(
            predictions=predictions,
            scored_candidates=scored,
            labels=labels,
            mapping=mapping,
        )
        evaluated.to_parquet(staging / "evaluated_predictions.parquet", index=False)
        overlay.to_parquet(staging / "label_overlay.parquet", index=False)
        errors = evaluated[
            evaluated["confirmed_error_auto"]
            | (
                evaluated["label_kind"].eq("MATCH_EXACT")
                & ~evaluated["ranker_hit"]
            )
        ].copy()
        errors.to_parquet(staging / "errors.parquet", index=False)
        _durable_json(staging / "evaluation.json", {"cohorts": metrics})
        output_files = [
            path
            for path in staging.iterdir()
            if path.is_file() and not path.name.startswith(".")
        ]
        outputs = {
            path.name: {
                "sha256": file_sha256(path),
                "size_bytes": int(path.stat().st_size),
            }
            for path in output_files
        }
        manifest = {
            **identity,
            "build_identity": identity,
            "build_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "verdict": "DESCRIPTIVE_UNSEEN_COMPLETED",
            "inputs": {
                "sanitized_manifest": sanitized_manifest_sha,
                "qualification_manifest": qualification_manifest_sha,
                "labels_bytes": labels_sha,
                "execution_lock": lock_sha,
            },
            "outputs": outputs,
            "metrics": metrics,
            "invariants": {
                "training_performed": False,
                "threshold_selected": False,
                "fixed_threshold": FIXED_THRESHOLD,
                "labels_deserialized_after_predictions_sealed": True,
                "cohort_mapping_hash_verified_before_predictions": True,
                "cohort_mapping_deserialized_after_predictions_sealed": True,
                "ranker_scored_without_truth_metadata": True,
                "raw_decisions_preserved": True,
                "label_overlay_separate": True,
                "single_run": True,
            },
        }
        _durable_json(staging / "manifest.json", manifest)
        os.replace(staging, target)
        update_ledger(
            ledger_path,
            expected_run_id=run_id,
            phase="COMPLETED",
            labels_deserialized=True,
            artifact_path=str(target),
            manifest_sha256=file_sha256(target / "manifest.json"),
        )
        return target
    except BaseException as error:
        if ledger_created:
            update_ledger(
                ledger_path,
                expected_run_id=run_id,
                phase=(
                    "CONSUMED_FAILED"
                    if scoring_started
                    else "FAILED_BEFORE_SCORING"
                ),
                labels_deserialized=labels_opened,
                error_type=type(error).__name__,
                error=str(error),
                retry_forbidden=bool(scoring_started),
            )
            _durable_json(
                staging / "failure.json",
                {
                    "verdict": "STOP_DESCRIPTIVE_INTEGRITY",
                    "run_id": run_id,
                    "scoring_started": scoring_started,
                    "retry_forbidden": bool(scoring_started),
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            if not target.exists():
                os.replace(staging, target)
        else:
            shutil.rmtree(staging, ignore_errors=True)
        raise


def run_historical_parity(report_path: Path) -> Path:
    """Reproduce frozen dev serve scores without touching challenge artifacts."""

    dataset_root = Path(
        "/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/"
        "v4_11_input_blind/ec4326ec57e4411d"
    )
    ranker_root = Path(
        "/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/"
        "v4_11_ranker_c/e13eb3ac7498256e"
    )
    scene_root = Path(
        "/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/"
        "v4_11_acceptor/52ea3faba9a56aff"
    )
    bundle_root = DEFAULT_BUNDLE_ROOT
    assignments = pd.read_parquet(dataset_root / "split_assignments.parquet")
    dev_ids = set(
        assignments.loc[assignments["split"].eq("dev"), "query_id"].astype(str)
    )
    queries = pd.read_parquet(dataset_root / "queries.parquet")
    queries = queries[queries["query_id"].astype(str).isin(dev_ids)]
    candidates = pd.read_parquet(dataset_root / "candidates_sparse_top100.parquet")
    candidates = candidates[candidates["query_id"].astype(str).isin(dev_ids)]
    expected_ranker = pd.read_parquet(
        ranker_root / "predictions_ranker_c_oof_dev.parquet"
    )
    expected_ranker = expected_ranker[
        expected_ranker["query_id"].astype(str).isin(dev_ids)
    ]
    candidate_keys = set(
        zip(
            candidates["query_id"].astype(str),
            candidates["candidate_siret"].astype(str),
            strict=True,
        )
    )
    expected_ranker_keys = set(
        zip(
            expected_ranker["query_id"].astype(str),
            expected_ranker["candidate_siret"].astype(str),
            strict=True,
        )
    )
    if (
        len(queries) != 1_456
        or len(candidates) != 145_236
        or len(expected_ranker) != 145_236
        or candidate_keys != expected_ranker_keys
    ):
        raise ValueError(
            "STOP_DESCRIPTIVE_INTEGRITY: parity ranker population changed"
        )
    label_free = candidates.drop(columns=["is_ground_truth"])
    model = xgb.XGBRanker()
    frozen = validate_frozen_bundle(bundle_root)
    model.load_model(frozen["ranker_model"])
    scored = score_ranker_candidates(model, label_free)
    paired = scored.merge(
        expected_ranker[
            ["query_id", "candidate_siret", "ranker_score", "ranker_rank"]
        ],
        on=["query_id", "candidate_siret"],
        suffixes=("_serve", "_frozen"),
        validate="one_to_one",
    )
    ranker_scores_equal = np.array_equal(
        paired["ranker_score_serve"].to_numpy(dtype=np.float32),
        paired["ranker_score_frozen"].to_numpy(dtype=np.float32),
    )
    ranker_ranks_equal = np.array_equal(
        paired["ranker_rank_serve"].to_numpy(),
        paired["ranker_rank_frozen"].to_numpy(),
    )
    taxonomy = SiteFunctionTaxonomy.load(frozen["taxonomy"])
    acceptor = joblib.load(frozen["acceptor_model"])
    predictions, scenes = build_blind_predictions(
        queries=queries,
        scored_candidates=scored,
        taxonomy=taxonomy,
        acceptor=acceptor,
    )
    expected_scenes = pd.read_parquet(scene_root / "acceptor_scenes.parquet")
    expected_scenes = expected_scenes[
        expected_scenes["query_id"].astype(str).isin(dev_ids)
    ]
    if (
        len(predictions) != 1_456
        or len(scenes) != 1_456
        or len(expected_scenes) != 1_456
        or set(scenes["query_id"].astype(str))
        != set(expected_scenes["query_id"].astype(str))
    ):
        raise ValueError(
            "STOP_DESCRIPTIVE_INTEGRITY: parity scene population changed"
        )
    scene_pair = scenes.merge(
        expected_scenes[["query_id", *V411_ACCEPTOR_FEATURE_NAMES]],
        on="query_id",
        suffixes=("_serve", "_frozen"),
        validate="one_to_one",
    )
    scenes_equal = all(
        np.array_equal(
            scene_pair[f"{name}_serve"].to_numpy(dtype=np.float64),
            scene_pair[f"{name}_frozen"].to_numpy(dtype=np.float64),
        )
        for name in V411_ACCEPTOR_FEATURE_NAMES
    )
    expected_acceptor = pd.read_parquet(
        bundle_root / "predictions.parquet"
    )
    expected_acceptor = expected_acceptor[
        expected_acceptor["model_family"].eq("COMPACT_LOGIT")
    ][["query_id", "score", "decision"]].rename(
        columns={
            "score": "expected_acceptor_score",
            "decision": "expected_decision",
        }
    )
    if (
        len(expected_acceptor) != 1_456
        or set(predictions["query_id"].astype(str))
        != set(expected_acceptor["query_id"].astype(str))
    ):
        raise ValueError(
            "STOP_DESCRIPTIVE_INTEGRITY: parity acceptor population changed"
        )
    accept_pair = predictions.merge(
        expected_acceptor,
        on="query_id",
        validate="one_to_one",
    )
    acceptor_scores_equal = np.array_equal(
        accept_pair["acceptor_score"].to_numpy(dtype=np.float64),
        accept_pair["expected_acceptor_score"].to_numpy(dtype=np.float64),
    )
    expected_decisions = np.where(
        accept_pair["expected_decision"].astype(str).eq("AUTO_MATCH"),
        "AUTO_MATCH",
        "REVIEW",
    )
    decisions_equal = np.array_equal(
        accept_pair["decision"].to_numpy(dtype=str),
        expected_decisions,
    )
    checks = {
        "ranker_scores_bit_exact": ranker_scores_equal,
        "ranker_ranks_exact": ranker_ranks_equal,
        "scenes_bit_exact": scenes_equal,
        "acceptor_scores_bit_exact": acceptor_scores_equal,
        "decisions_exact": decisions_equal,
    }
    if not all(checks.values()):
        raise ValueError(f"STOP_DESCRIPTIVE_INTEGRITY: parity failed {checks}")
    report_path = Path(report_path).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _durable_json(
        report_path,
        {
            "schema_version": "sireto-v4.11-descriptive-parity-1",
            "row_count": len(queries),
            "candidate_count": len(scored),
            "checks": checks,
            "challenge_artifact_opened": False,
        },
    )
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parity-artifact", type=Path)
    parser.add_argument("--sanitized-artifact", type=Path)
    parser.add_argument("--qualification-artifact", type=Path)
    parser.add_argument("--execution-lock", type=Path)
    parser.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    parser.add_argument("--partitions", type=Path, default=DEFAULT_PARTITIONS)
    parser.add_argument("--global-store", type=Path, default=DEFAULT_GLOBAL_STORE)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--v42-manifest", type=Path, default=DEFAULT_V42_MANIFEST)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.parity_artifact is not None:
        print(run_historical_parity(args.parity_artifact))
        return
    required = (
        "sanitized_artifact",
        "qualification_artifact",
        "execution_lock",
        "cache_dir",
        "work_dir",
        "output_root",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise SystemExit(f"Missing required arguments: {', '.join(missing)}")
    print(
        run_once(
            sanitized_artifact=args.sanitized_artifact,
            qualification_artifact=args.qualification_artifact,
            execution_lock_path=args.execution_lock,
            bundle_root=args.bundle_root,
            partitions_dir=args.partitions,
            global_store_path=args.global_store,
            snapshot_path=args.snapshot,
            v42_manifest_path=args.v42_manifest,
            cache_dir=args.cache_dir,
            work_dir=args.work_dir,
            output_root=args.output_root,
            contract_path=args.contract,
        )
    )


if __name__ == "__main__":
    main()
