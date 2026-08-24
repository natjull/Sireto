#!/usr/bin/env python3
"""Run leakage-safe v12 admission workflows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_retrieval_ltr_streaming import iter_complete_query_frames  # noqa: E402
from src.xgb_matcher.retrieval_hierarchical_admission_v12 import (  # noqa: E402
    AdmissionConfigV12,
    HierarchicalModelsV12,
    SCHEMA_VERSION,
    adapt_cascade_parent_rows,
    canonicalize_cascade_parent_rows,
    canonicalize_union,
    evaluate_all_human,
    fit_hierarchical_models,
    fit_one_outer_site_model,
    fit_parent_ranker_from_cascade_rows,
    missing_label_outcomes,
    normalize_siret,
    parent_feature_names,
    sample_precomputed_candidates,
    score_canonical_candidates,
    select_top100,
    site_feature_names,
    summarize_outcomes,
)


DEFAULT_CONFIG = ROOT / "config/retrieval_all_human_v12.json"
PARENT_OOF_BUNDLE_SCHEMA = "sireto-parent-oof-model-bundle-v12-1"
NESTED_CASCADE_SCORER_KIND = "NESTED_OOF_FOLD_MODEL_MAPPING"
_ALLOWED_FOLDS = frozenset({2, 3, 4})
_LABEL_LEDGER_SCHEMA = "sireto-retrieval-all-human-v12-label-ledger-1"
_LABEL_POPULATION = "ALL_USABLE_HUMAN_LABELS"
_LABEL_ONLY_EXCLUSION = "NO_VALID_CURRENT_OR_HISTORICAL_HUMAN_SIRET"
_RUNTIME_PROFILE_SCHEMA = "sireto-retrieval-all-human-v12-runtime-profile-1"
_RUNTIME_SAMPLE_RULE = "SHA256_QUERY_ID_ASC_FIRST_512"
_OPERATIONAL_EQUIVALENCE_MODE = (
    "EXACT_ONLY_PENDING_SEALED_SAME_SITE_EVIDENCE"
)
_LABEL_SOURCE_KINDS = frozenset(
    {
        "SEALED_V11_DEVELOPMENT",
        "SEALED_RAW_ALL_HUMAN_DEVELOPMENT",
        "EXPLICIT_RAW_ALL_HUMAN_OOF",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_PARENT_FIT_FOLDS: Mapping[str, tuple[int, ...]] = {
    "fit_2": (2,),
    "fit_3": (3,),
    "fit_4": (4,),
    "fit_23": (2, 3),
    "fit_24": (2, 4),
    "fit_34": (3, 4),
}
_OUTER_PARENT_MAPS: Mapping[str, Mapping[str, Mapping[str, Any]]] = {
    "2": {
        "2": {"model": "fit_34", "fit_folds": [3, 4], "role": "OUTER_HELDOUT"},
        "3": {"model": "fit_4", "fit_folds": [4], "role": "INNER_TRAIN"},
        "4": {"model": "fit_3", "fit_folds": [3], "role": "INNER_TRAIN"},
    },
    "3": {
        "3": {"model": "fit_24", "fit_folds": [2, 4], "role": "OUTER_HELDOUT"},
        "2": {"model": "fit_4", "fit_folds": [4], "role": "INNER_TRAIN"},
        "4": {"model": "fit_2", "fit_folds": [2], "role": "INNER_TRAIN"},
    },
    "4": {
        "4": {"model": "fit_23", "fit_folds": [2, 3], "role": "OUTER_HELDOUT"},
        "2": {"model": "fit_3", "fit_folds": [3], "role": "INNER_TRAIN"},
        "3": {"model": "fit_2", "fit_folds": [2], "role": "INNER_TRAIN"},
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_manifest(path: Path) -> tuple[Path, Mapping[str, Any]]:
    manifest_path = next(
        (
            candidate
            for candidate in (
                path.with_suffix(path.suffix + ".manifest.json"),
                path.parent / "manifest.json",
            )
            if candidate.is_file()
        ),
        None,
    )
    if manifest_path is None:
        raise FileNotFoundError(f"v12 requires a sealed manifest for {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    return manifest_path, manifest


def _declared_file_hash(manifest: Mapping[str, Any], path: Path) -> str | None:
    direct = manifest.get("candidate_parquet_sha256")
    if isinstance(direct, str):
        return direct
    for container_name in ("files", "outputs"):
        entry = (manifest.get(container_name) or {}).get(path.name)
        if isinstance(entry, Mapping):
            value = entry.get("sha256")
        else:
            value = entry
        if isinstance(value, str):
            return value
    return None


def verify_candidate_manifest(
    path: Path, *, labelled: bool, require_master: bool = True
) -> tuple[Path, Mapping[str, Any]]:
    manifest_path, manifest = _artifact_manifest(path)
    if _declared_file_hash(manifest, path) != sha256_file(path):
        raise ValueError("v12 candidate hash mismatch")
    if manifest.get("positive_injection") is not False:
        raise ValueError("v12 manifest does not prove zero positive injection")
    if int(manifest.get("maximum_union_candidates", 2_000)) > 2_000:
        raise ValueError("v12 candidate union exceeds Master@2000")
    if labelled:
        if manifest.get("labels_attached_after_retrieval") is not True:
            raise ValueError("v12 fit requires labels attached after retrieval")
        if not isinstance(manifest.get("source_labels_sha256"), str):
            raise ValueError("v12 labelled candidates lack sealed ledger lineage")
    elif (
        manifest.get("human_labels_opened_by_retrieval") is not False
        and manifest.get("human_labels_opened") is not False
    ):
        raise ValueError("v12 prediction requires label-blind retrieval")
    opened = {int(value) for value in manifest.get("opened_folds", ())}
    if opened and not opened.issubset(_ALLOWED_FOLDS):
        raise ValueError("v12 candidate manifest opened fold 0/1")
    view = (manifest.get("views") or {}).get(path.name)
    if not isinstance(view, Mapping):
        view = manifest.get("candidate_view")
    if require_master:
        if not isinstance(view, Mapping):
            raise ValueError("v12 candidate manifest does not seal a Master view")
        if view.get("union_rank_semantics") != "MASTER_RANK_WITHIN_MAX_2000":
            raise ValueError("hydration-only input cannot impersonate Master@2000")
        if view.get("hydration_rank_preserved") is not True:
            raise ValueError("Master view does not preserve natural PreRank marks")
        if int(manifest.get("maximum_master_references", 2_001)) > 2_000:
            raise ValueError("v12 Master view exceeds 2,000")
        if int(manifest.get("maximum_hydration_shortlist", 501)) > 500:
            raise ValueError("v12 PreRank view exceeds 500")
    return manifest_path, manifest


def verify_parent_candidates(
    path: Path,
    *,
    require_nested_oof: bool,
    expected_outer_fold: int | None = None,
) -> tuple[Path, Mapping[str, Any]]:
    """Verify a complete pre-expansion parent pool and its score provenance."""
    manifest_path, manifest = _artifact_manifest(path)
    if _declared_file_hash(manifest, path) != sha256_file(path):
        raise ValueError("v12 pre-expansion parent candidate hash mismatch")
    if manifest.get("positive_injection") is not False:
        raise ValueError("v12 parent candidate manifest does not prove zero injection")
    if manifest.get("human_labels_opened") is not False:
        raise ValueError("v12 parent pool must be label-blind")
    view = (manifest.get("views") or {}).get(path.name)
    if not isinstance(view, Mapping):
        raise ValueError("v12 parent pool has no sealed view metadata")
    if view.get("contains_all_bounded_parent_training_candidates") is not True:
        raise ValueError("v12 parent pool is not the complete bounded pre-expansion view")
    if int(view.get("maximum_candidates", 2_001)) > 2_000:
        raise ValueError("v12 parent pool exceeds its cap")
    if int(manifest.get("maximum_parent_candidates", 2_001)) > 2_000:
        raise ValueError("v12 parent pool observed cap exceeds 2,000")
    opened = {int(value) for value in manifest.get("opened_folds", ())}
    if opened and not opened.issubset(_ALLOWED_FOLDS):
        raise ValueError("v12 parent pool opened fold 0/1")
    parent_scores = manifest.get("candidate_parent_scores") or {}
    if not all(
        parent_scores.get(key) is True
        for key in (
            "computed_before_parent_expansion",
            "complete_for_all_master_and_hydration_candidates",
        )
    ):
        raise ValueError("cascade does not preserve complete pass-A parent scores")
    if require_nested_oof:
        proof = manifest.get("nested_parent_oof") or {}
        if (
            manifest.get("rescue_hits_sha256") is not None
            or manifest.get("upstream_rescue_reference_count") != 0
            or manifest.get("rescue_execution_queries") != 0
        ):
            raise ValueError(
                "initial v12 nested cascade must exclude every deferred rescue channel"
            )
        training_lineage = proof.get("training_lineage")
        source_retrieval = (
            training_lineage.get("source_retrieval")
            if isinstance(training_lineage, Mapping)
            else None
        )
        if (
            not isinstance(training_lineage, Mapping)
            or set(training_lineage)
            != {
                "source_retrieval",
                "label_ledger_sha256",
                "label_ledger_manifest_sha256",
            }
            or not isinstance(source_retrieval, Mapping)
            or set(source_retrieval)
            != {
                "source_candidate_sha256",
                "source_manifest_sha256",
                "sidecar_manifest_sha256",
            }
            or any(
                not _is_sha256(value)
                for value in (
                    *source_retrieval.values(),
                    training_lineage.get("label_ledger_sha256"),
                    training_lineage.get("label_ledger_manifest_sha256"),
                )
            )
        ):
            raise ValueError("cascade lacks sealed parent-training lineage")
        if not all(
            parent_scores.get(key) is True
            for key in ("sealed_by_nested_model_bundle",)
        ):
            raise ValueError("cascade does not seal pass-A parent scores for admission")
        if manifest.get("parent_scorer_kind") != NESTED_CASCADE_SCORER_KIND:
            raise ValueError("simple/bootstrap parent cascade is not gate-eligible")
        if proof.get("gate_eligible") is not True:
            raise ValueError("cascade does not prove nested parent OOF generation")
        if proof.get("parent_feature_origin") != "CASCADE_PRE_EXPANSION":
            raise ValueError("cascade parent scorer origin is not pre-expansion")
        if tuple(proof.get("parent_feature_names") or ()) != tuple(
            parent_feature_names(AdmissionConfigV12.load(DEFAULT_CONFIG))
        ):
            raise ValueError("cascade parent feature schema/order drift")
        outer = int(proof.get("outer_heldout_fold", -1))
        if expected_outer_fold is not None and outer != expected_outer_fold:
            raise ValueError("cascade nested OOF outer context mismatch")
        expected_map = _OUTER_PARENT_MAPS.get(str(outer))
        observed_map = proof.get("scorer_by_query_fold") or {}
        for score_fold, expected in expected_map.items():
            observed = observed_map.get(score_fold, observed_map.get(int(score_fold)))
            if not isinstance(observed, Mapping):
                raise ValueError("cascade nested OOF fold/model map is incomplete")
            for key in ("model", "fit_folds", "role"):
                if observed.get(key) != expected[key]:
                    raise ValueError("cascade nested OOF fold/model map mismatch")
            if not isinstance(observed.get("sha256"), str):
                raise ValueError("cascade nested OOF route lacks model hash")
        top_hashes = manifest.get("parent_model_hashes") or {}
        if not all(
            top_hashes.get(score_fold) == (
                observed_map.get(score_fold, observed_map.get(int(score_fold)))
            ).get("sha256")
            for score_fold in expected_map
        ):
            raise ValueError("cascade nested OOF parent model hashes disagree")
    return manifest_path, manifest


def _prepare_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.mkdir(parents=True)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _parquet_schema_descriptor(path: Path) -> list[dict[str, Any]]:
    return [
        {"name": field.name, "type": str(field.type), "nullable": bool(field.nullable)}
        for field in pq.read_schema(path)
    ]


def _parity_aggregate_digest(rows: pd.DataFrame, hash_column: str) -> str:
    digest = hashlib.sha256()
    ordered = rows.sort_values(["query_id", "fold"], kind="stable")
    for row in ordered.itertuples(index=False):
        digest.update(str(row.query_id).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(int(row.fold)).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(getattr(row, hash_column)).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_label_ledger_manifest_contract(
    *,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    ledger_path: Path,
    config_path: Path,
) -> Mapping[str, Path]:
    """Validate the exact builder contract before reading any human labels."""
    if manifest.get("schema_version") != _LABEL_LEDGER_SCHEMA:
        raise ValueError("v12 requires the exact all-human label-ledger schema")
    identity = manifest.get("build_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("v12 label ledger lacks sealed build_identity")
    exact_identity_keys = {
        "schema_version",
        "population",
        "only_exclusion",
        "folds",
        "source_kind",
        "source_hashes",
        "contract_sha256",
        "config_sha256",
        "builder_sha256",
        "label_precedence",
        "operational_equivalence_mode",
        "runtime_query_sample_rule",
        "positive_injection",
        "retrieval_outputs_read",
        "model_outputs_read",
        "fold0_opened",
        "fold1_opened",
        "official_catalog",
    }
    if set(identity) != exact_identity_keys:
        raise ValueError("v12 label ledger build_identity keys drifted")
    expected_identity_claims = {
        "schema_version": _LABEL_LEDGER_SCHEMA,
        "population": _LABEL_POPULATION,
        "only_exclusion": _LABEL_ONLY_EXCLUSION,
        "folds": [2, 3, 4],
        "label_precedence": [
            "ground_truth_siret",
            "historical_ground_truth_siret",
        ],
        "operational_equivalence_mode": _OPERATIONAL_EQUIVALENCE_MODE,
        "runtime_query_sample_rule": _RUNTIME_SAMPLE_RULE,
        "positive_injection": False,
        "retrieval_outputs_read": False,
        "model_outputs_read": False,
        "fold0_opened": False,
        "fold1_opened": False,
    }
    if any(identity.get(key) != value for key, value in expected_identity_claims.items()):
        raise ValueError("v12 label ledger build_identity claims drifted")
    source_kind = identity.get("source_kind")
    if source_kind not in _LABEL_SOURCE_KINDS:
        raise ValueError("v12 label ledger source_kind is not allowlisted")
    source_hashes = identity.get("source_hashes")
    if (
        not isinstance(source_hashes, Mapping)
        or not source_hashes
        or any(
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or not _is_sha256(value)
            for name, value in source_hashes.items()
        )
    ):
        raise ValueError("v12 label ledger source_hashes are not fully sealed")
    expected_config_sha = sha256_file(config_path)
    if identity.get("config_sha256") != expected_config_sha:
        raise ValueError("v12 label ledger was built with a different config")
    config_value = json.loads(config_path.read_text(encoding="utf-8"))
    contract_raw = config_value.get("contract_path")
    if not isinstance(contract_raw, str) or not contract_raw:
        raise ValueError("v12 config lacks its retrieval contract path")
    contract_path = Path(contract_raw)
    if not contract_path.is_absolute():
        contract_path = ROOT / contract_path
    if not contract_path.is_file() or identity.get("contract_sha256") != sha256_file(
        contract_path
    ):
        raise ValueError("v12 label ledger contract hash drifted")
    builder_path = ROOT / "scripts/build_retrieval_all_human_v12_label_ledger.py"
    if identity.get("builder_sha256") != sha256_file(builder_path):
        raise ValueError("v12 label ledger builder hash drifted")
    official_identity = identity.get("official_catalog")
    if not isinstance(official_identity, Mapping) or set(official_identity) != {
        "available",
        "manifest_sha256",
        "establishments_sha256",
    }:
        raise ValueError("v12 label ledger official-catalog identity drifted")
    official_available = official_identity.get("available")
    if not isinstance(official_available, bool):
        raise ValueError("v12 label ledger official-catalog availability is invalid")
    official_hashes = (
        official_identity.get("manifest_sha256"),
        official_identity.get("establishments_sha256"),
    )
    if official_available:
        if not all(_is_sha256(value) for value in official_hashes):
            raise ValueError("available official catalog lacks sealed hashes")
    elif official_hashes != (None, None):
        raise ValueError("unavailable official catalog must not declare hashes")
    build_id = manifest.get("build_id")
    expected_build_id = _canonical_json_sha256(identity)[:20]
    if build_id != expected_build_id or manifest_path.parent.name != expected_build_id:
        raise ValueError("v12 label ledger build_id does not match build_identity")

    root_claims = {
        "population": _LABEL_POPULATION,
        "only_exclusion": _LABEL_ONLY_EXCLUSION,
        "positive_injection": False,
        "retrieval_outputs_read": False,
        "model_outputs_read": False,
        "folds": [2, 3, 4],
        "fold0_opened": False,
        "fold1_opened": False,
        "candidate_group_dependency": "NONE",
        "queries_with_zero_candidate_groups_remain_in_denominator": True,
        "operational_equivalence_mode": _OPERATIONAL_EQUIVALENCE_MODE,
        "runtime_query_sample_rule": _RUNTIME_SAMPLE_RULE,
        "runtime_query_sample_rows": 512,
    }
    if any(manifest.get(key) != value for key, value in root_claims.items()):
        raise ValueError("v12 label ledger top-level claims drifted")
    unverified_count = manifest.get("unverified_operational_alternative_count")
    unverified_by_fold = manifest.get(
        "unverified_operational_alternative_count_by_fold"
    )
    if (
        not isinstance(unverified_count, int)
        or unverified_count < 0
        or not isinstance(unverified_by_fold, Mapping)
        or set(unverified_by_fold) != {"2", "3", "4"}
        or any(
            not isinstance(value, int) or value < 0
            for value in unverified_by_fold.values()
        )
        or sum(unverified_by_fold.values()) != unverified_count
    ):
        raise ValueError("v12 unverified operational audit counts do not reconcile")
    if manifest.get("official_catalog_presence_available") is not official_available:
        raise ValueError("v12 official-catalog root/identity claims disagree")
    official_lineage = manifest.get("official_catalog_lineage")
    if not isinstance(official_lineage, Mapping) or official_lineage.get(
        "available"
    ) is not official_available:
        raise ValueError("v12 official-catalog lineage is incomplete")
    if official_available and (
        official_lineage.get("manifest_sha256") != official_hashes[0]
        or official_lineage.get("establishments_sha256") != official_hashes[1]
    ):
        raise ValueError("v12 official-catalog lineage hashes disagree")

    source_scan = manifest.get("source_scan")
    expected_source_scan_keys = {
        "kind",
        "predicate",
        "materialised_folds",
        "fold0_rows_materialised",
        "fold1_rows_materialised",
        "locked_fold1_artifacts_opened",
        "source_hashes",
    }
    if not isinstance(source_scan, Mapping) or set(source_scan) != expected_source_scan_keys:
        raise ValueError("v12 label ledger source_scan contract drifted")
    expected_source_scan = {
        "kind": source_kind,
        "predicate": "fold IN (2,3,4)",
        "materialised_folds": [2, 3, 4],
        "fold0_rows_materialised": False,
        "fold1_rows_materialised": False,
        "locked_fold1_artifacts_opened": False,
        "source_hashes": dict(source_hashes),
    }
    if dict(source_scan) != expected_source_scan:
        raise ValueError("v12 label ledger source_scan disagrees with build_identity")

    counts = manifest.get("counts_by_fold")
    excluded_counts = manifest.get("excluded_counts_by_fold")
    query_count = manifest.get("query_count")
    excluded_count = manifest.get("excluded_count")
    if (
        not isinstance(query_count, int)
        or query_count <= 0
        or not isinstance(excluded_count, int)
        or excluded_count < 0
        or not isinstance(counts, Mapping)
        or set(counts) != {"2", "3", "4"}
        or not isinstance(excluded_counts, Mapping)
        or set(excluded_counts) != {"2", "3", "4"}
        or any(not isinstance(value, int) or value < 0 for value in counts.values())
        or any(
            not isinstance(value, int) or value < 0
            for value in excluded_counts.values()
        )
        or sum(counts.values()) != query_count
        or sum(excluded_counts.values()) != excluded_count
    ):
        raise ValueError("v12 label ledger population counts do not reconcile")

    artifacts = manifest.get("artifacts")
    expected_artifacts = {
        "label_ledger": ("label_ledger_path", "label_ledger_sha256", query_count),
        "operational_sidecar": (
            "operational_sidecar_path",
            "operational_sidecar_sha256",
            query_count,
        ),
        "excluded_sidecar": (
            "excluded_sidecar_path",
            "excluded_sidecar_sha256",
            excluded_count,
        ),
        "runtime_query_sample": (
            "runtime_query_sample_path",
            "runtime_query_sample_sha256",
            512,
        ),
    }
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(expected_artifacts):
        raise ValueError("v12 label ledger artifacts map is incomplete")
    resolved: dict[str, Path] = {}
    for name, (root_path_key, root_hash_key, expected_rows) in expected_artifacts.items():
        metadata = artifacts.get(name)
        if not isinstance(metadata, Mapping) or set(metadata) != {
            "path",
            "rows",
            "sha256",
            "size_bytes",
            "schema",
        }:
            raise ValueError(f"v12 {name} metadata contract drifted")
        relative = metadata.get("path")
        if not isinstance(relative, str) or Path(relative).name != relative:
            raise ValueError(f"v12 {name} path must stay inside the build directory")
        artifact_path = (manifest_path.parent / relative).resolve()
        if artifact_path.parent != manifest_path.parent.resolve() or not artifact_path.is_file():
            raise ValueError(f"v12 {name} artifact is missing")
        if (
            manifest.get(root_path_key) != relative
            or manifest.get(root_hash_key) != metadata.get("sha256")
            or metadata.get("rows") != expected_rows
            or metadata.get("size_bytes") != artifact_path.stat().st_size
            or not _is_sha256(metadata.get("sha256"))
            or metadata.get("sha256") != sha256_file(artifact_path)
            or metadata.get("schema") != _parquet_schema_descriptor(artifact_path)
            or pq.ParquetFile(artifact_path).metadata.num_rows != expected_rows
        ):
            raise ValueError(f"v12 {name} metadata/hash/rows mismatch")
        resolved[name] = artifact_path
    if resolved["label_ledger"] != ledger_path.resolve():
        raise ValueError("v12 loader path is not the sealed label ledger artifact")
    return resolved


def _load_sealed_query_labels(
    path: Path, *, config_path: Path
) -> tuple[pd.DataFrame, Path]:
    manifest_path, manifest = _artifact_manifest(path)
    resolved_artifacts = _validate_label_ledger_manifest_contract(
        manifest=manifest,
        manifest_path=manifest_path,
        ledger_path=path,
        config_path=config_path,
    )
    if manifest.get("label_ledger_sha256") != sha256_file(path):
        raise ValueError("v12 label-side parquet hash mismatch")
    declared_ledger_path = Path(str(manifest.get("label_ledger_path") or ""))
    if not declared_ledger_path.is_absolute():
        declared_ledger_path = manifest_path.parent / declared_ledger_path
    if declared_ledger_path.resolve() != path.resolve():
        raise ValueError("v12 manifest label_ledger_path mismatch")
    required_claims = {
        "population": "ALL_USABLE_HUMAN_LABELS",
        "only_exclusion": "NO_VALID_CURRENT_OR_HISTORICAL_HUMAN_SIRET",
        "positive_injection": False,
        "retrieval_outputs_read": False,
        "model_outputs_read": False,
        "ground_truth_siren_disjoint_across_folds": True,
        "fold0_opened": False,
        "fold1_opened": False,
    }
    if any(manifest.get(key) != value for key, value in required_claims.items()):
        raise ValueError("v12 label ledger population/lineage claims are incomplete")
    if {int(value) for value in manifest.get("folds", ())} != _ALLOWED_FOLDS:
        raise ValueError("v12 label ledger must contain exactly folds 2/3/4")
    excluded_path_raw = manifest.get("excluded_sidecar_path")
    excluded_sha = manifest.get("excluded_sidecar_sha256")
    if not isinstance(excluded_path_raw, str) or not isinstance(excluded_sha, str):
        raise ValueError("v12 label ledger lacks its sealed exclusions sidecar")
    excluded_path = resolved_artifacts["excluded_sidecar"]
    if not excluded_path.is_file() or sha256_file(excluded_path) != excluded_sha:
        raise ValueError("v12 label exclusion sidecar hash mismatch")
    excluded = pd.read_parquet(excluded_path)
    required_excluded_columns = {
        "query_id",
        "fold",
        "missing_reason",
        "current_siret_present_but_invalid",
        "historical_siret_present_but_invalid",
        "label_kind",
        "qualification_v2",
        "qualification_v3",
        "source_kind",
        "data_origin",
        "split",
    }
    if not required_excluded_columns.issubset(excluded.columns):
        raise ValueError("v12 exclusions sidecar schema is incomplete")
    if not excluded.empty and (
        excluded["query_id"].astype(str).duplicated().any()
        or not excluded["missing_reason"].eq(_LABEL_ONLY_EXCLUSION).all()
        or not set(excluded["fold"].astype(int)).issubset(_ALLOWED_FOLDS)
    ):
        raise ValueError("v12 exclusions sidecar population is invalid")
    if int(manifest.get("excluded_count", -1)) != len(excluded):
        raise ValueError("v12 label exclusion count mismatch")
    excluded_counts = {
        str(fold): int((excluded.get("fold", pd.Series(dtype=int)) == fold).sum())
        for fold in sorted(_ALLOWED_FOLDS)
    }
    if manifest.get("excluded_counts_by_fold") != excluded_counts:
        raise ValueError("v12 label exclusion fold counts mismatch")
    labels = pd.read_parquet(path)
    required_ledger_columns = {
        "query_id",
        "fold",
        "human_ground_truth_siret",
        "human_ground_truth_siren",
        "human_ground_truth_source",
        "ground_truth_state",
        "label_kind",
        "qualification_v2",
        "qualification_v3",
        "source_kind",
        "data_origin",
        "split",
        "unseen_siren",
        "new_site_known_siren",
        "mega_base_pool",
        "pool_size",
        "insee",
        "official_catalog_present",
    }
    if not required_ledger_columns.issubset(labels.columns):
        raise ValueError("v12 label ledger does not match the exact builder schema")
    label_column = next(
        (
            name
            for name in (
                "human_ground_truth_siret",
                "ground_truth_siret",
                "historical_ground_truth_siret",
                "human_siret",
            )
            if name in labels.columns
        ),
        None,
    )
    if label_column is None or not {"query_id", "fold"}.issubset(labels.columns):
        raise ValueError("v12 label-side artifact lacks query_id/fold/human SIRET")
    retained = [
        "query_id",
        "fold",
        label_column,
        "human_ground_truth_siren",
        "human_ground_truth_source",
    ]
    retained.extend(
        name
        for name in (
            "ground_truth_state",
            "label_kind",
            "qualification_v2",
            "qualification_v3",
            "source_kind",
            "data_origin",
            "split",
            "unseen_siren",
            "new_site_known_siren",
            "mega_base_pool",
            "pool_size",
            "insee",
            "official_catalog_present",
        )
        if name in labels.columns
    )
    labels = labels[retained].rename(columns={label_column: "human_siret"})
    labels["query_id"] = labels["query_id"].astype(str)
    labels["fold"] = labels["fold"].astype(int)
    if labels["query_id"].duplicated().any():
        raise ValueError("v12 label-side artifact must be unique by query")
    if set(labels["fold"].unique()) != _ALLOWED_FOLDS:
        raise ValueError("v12 parent fit refuses fold 0/1 or missing folds 2/3/4")
    if int(manifest.get("query_count", -1)) != len(labels):
        raise ValueError("v12 label ledger query count mismatch")
    observed_counts = {
        str(int(key)): int(value)
        for key, value in labels["fold"].value_counts().sort_index().items()
    }
    if manifest.get("counts_by_fold") != observed_counts:
        raise ValueError("v12 label ledger fold counts mismatch")
    normalized_exact = labels["human_siret"].map(normalize_siret)
    if not normalized_exact.eq(labels["human_siret"].astype(str)).all() or not labels[
        "human_ground_truth_siren"
    ].astype(str).eq(normalized_exact.str[:9]).all():
        raise ValueError("v12 exact SIRET/SIREN label columns disagree")
    observed_source_counts = {
        str(key): int(value)
        for key, value in labels["human_ground_truth_source"]
        .astype(str)
        .value_counts()
        .sort_index()
        .items()
    }
    if manifest.get("human_ground_truth_source_counts") != observed_source_counts:
        raise ValueError("v12 human label source counts mismatch")
    catalog_available = manifest.get("official_catalog_presence_available") is True
    if "official_catalog_present" not in labels.columns:
        raise ValueError("v12 label ledger lacks official catalog audit column")
    catalog_values = labels["official_catalog_present"].astype("boolean")
    if catalog_available and catalog_values.isna().any():
        raise ValueError("official catalog availability claim has unknown rows")
    if not catalog_available and catalog_values.notna().any():
        raise ValueError("official catalog values exist without sealed availability")
    if catalog_available:
        if int(manifest.get("official_catalog_present_count", -1)) != int(
            catalog_values.fillna(False).sum()
        ) or int(manifest.get("official_catalog_absent_count", -1)) != int(
            catalog_values.eq(False).sum()
        ):
            raise ValueError("official catalog audit counts mismatch")
    truth_parents = labels["human_siret"].astype(str).str.replace(
        r"\D", "", regex=True
    ).str.zfill(14).str[:9]
    if pd.DataFrame({"parent": truth_parents, "fold": labels["fold"]}).groupby(
        "parent"
    )["fold"].nunique().gt(1).any():
        raise ValueError("v12 ground-truth SIREN overlap across folds")
    sirens_by_fold = {
        fold: set(truth_parents[labels["fold"].eq(fold)])
        for fold in sorted(_ALLOWED_FOLDS)
    }
    observed_disjoint_proof = {
        "ground_truth_siren_disjoint_across_folds": True,
        "pairwise_overlap_counts": {
            "2:3": len(sirens_by_fold[2].intersection(sirens_by_fold[3])),
            "2:4": len(sirens_by_fold[2].intersection(sirens_by_fold[4])),
            "3:4": len(sirens_by_fold[3].intersection(sirens_by_fold[4])),
        },
        "unique_ground_truth_sirens_by_fold": {
            str(fold): len(sirens_by_fold[fold]) for fold in sorted(_ALLOWED_FOLDS)
        },
    }
    if manifest.get("siren_disjoint_proof") != observed_disjoint_proof:
        raise ValueError("v12 SIREN-disjoint proof does not match the ledger")
    if set(excluded["query_id"].astype(str)).intersection(
        set(labels["query_id"].astype(str))
    ):
        raise ValueError("v12 included and excluded query populations overlap")

    operational_path_raw = manifest.get("operational_sidecar_path")
    operational_sha = manifest.get("operational_sidecar_sha256")
    if not isinstance(operational_path_raw, str) or not isinstance(operational_sha, str):
        raise ValueError("v12 label ledger lacks sealed operational sidecar")
    operational_path = resolved_artifacts["operational_sidecar"]
    if not operational_path.is_file() or sha256_file(operational_path) != operational_sha:
        raise ValueError("v12 operational label sidecar hash mismatch")
    operational = pd.read_parquet(operational_path)
    if set(operational.columns) != {
        "query_id",
        "acceptable_sirets_operational_json",
    } or operational["query_id"].astype(str).duplicated().any():
        raise ValueError("v12 operational label sidecar schema is invalid")
    operational = operational[
        ["query_id", "acceptable_sirets_operational_json"]
    ].copy()
    operational["query_id"] = operational["query_id"].astype(str)
    if set(operational["query_id"]) != set(labels["query_id"]):
        raise ValueError("v12 operational sidecar must cover exactly the label ledger")
    labels = labels.merge(operational, on="query_id", how="left", validate="one_to_one")
    if labels["acceptable_sirets_operational_json"].isna().any():
        raise ValueError("v12 operational sidecar omits a labelled query")
    for row in labels.to_dict("records"):
        exact = normalize_siret(row["human_siret"])
        try:
            raw_operational = json.loads(row["acceptable_sirets_operational_json"])
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("v12 operational sidecar contains invalid JSON") from error
        if not isinstance(raw_operational, list):
            raise ValueError("v12 operational sidecar must contain a SIRET list")
        try:
            acceptable = {normalize_siret(value) for value in raw_operational}
        except ValueError as error:
            raise ValueError("v12 operational sidecar contains an invalid SIRET") from error
        if acceptable != {exact}:
            raise ValueError(
                "v12 operational sidecar must remain exact-only until sealed "
                "same-site evidence exists"
            )
    runtime_sample = pd.read_parquet(resolved_artifacts["runtime_query_sample"])
    if (
        list(runtime_sample.columns) != ["query_id", "fold"]
        or len(runtime_sample) != 512
        or runtime_sample["query_id"].astype(str).duplicated().any()
    ):
        raise ValueError("v12 runtime query sample schema/population is invalid")
    runtime_sample["query_id"] = runtime_sample["query_id"].astype(str)
    runtime_sample["fold"] = runtime_sample["fold"].astype(int)
    expected_runtime_sample = labels[["query_id", "fold"]].copy()
    expected_runtime_sample["_sample_key"] = expected_runtime_sample[
        "query_id"
    ].map(lambda value: hashlib.sha256(str(value).encode("utf-8")).hexdigest())
    expected_runtime_sample = (
        expected_runtime_sample.sort_values(
            ["_sample_key", "query_id"], kind="stable"
        )
        .head(512)[["query_id", "fold"]]
        .reset_index(drop=True)
    )
    if not runtime_sample.reset_index(drop=True).equals(expected_runtime_sample):
        raise ValueError("v12 runtime query sample violates deterministic selection")
    labels["seen_unseen_siren"] = labels.get(
        "unseen_siren", pd.Series(pd.NA, index=labels.index)
    ).map({True: "UNSEEN_SIREN", False: "SEEN_SIREN"}).fillna("UNKNOWN")
    labels["known_new_site"] = labels.get(
        "new_site_known_siren", pd.Series(pd.NA, index=labels.index)
    ).map({True: "NEW_SITE_KNOWN_SIREN", False: "KNOWN_SITE"}).fillna("UNKNOWN")
    labels["megapolis"] = labels.get(
        "mega_base_pool", pd.Series(pd.NA, index=labels.index)
    ).astype("boolean").map(
        {True: "MEGA_BASE_POOL", False: "NON_MEGA_BASE_POOL"}
    ).fillna("UNKNOWN")
    labels["qualification"] = (
        labels.get("qualification_v2", "").astype(str)
        + "|"
        + labels.get("qualification_v3", "").astype(str)
    )
    return labels, manifest_path


def _validate_runtime_profile_manifest(
    path: Path,
    *,
    config_path: Path,
    labels: pd.DataFrame,
    label_manifest_path: Path,
    cascade_manifests: Mapping[int, Path] | None = None,
    candidate_paths: Mapping[int, Path] | None = None,
    model_paths: Mapping[int, Path] | None = None,
    selected_paths: Mapping[int, Path] | None = None,
    sealed_lineage: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Validate a sealed, warm, end-to-end Mac M4 Pro runtime replay.

    Per-query latency carried by candidate rows is deliberately not accepted as
    an operational SLO proof.  The profile must replay the exact OOF artifacts,
    publish a label-free deterministic 512-query sample, and prove byte parity
    of ``candidate_siret + selected_rank`` before its percentiles can gate OOF.
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    profile = json.loads(path.read_text(encoding="utf-8"))
    if profile.get("schema_version") != _RUNTIME_PROFILE_SCHEMA:
        raise ValueError("v12 runtime profile schema mismatch")
    required_claims = {
        "run_kind": "WARM_END_TO_END_OOF_REPLAY",
        "folds": [2, 3, 4],
        "fold0_opened": False,
        "fold1_opened": False,
        "positive_injection": False,
        "gate_eligible": True,
    }
    if any(profile.get(key) != value for key, value in required_claims.items()):
        raise ValueError("v12 runtime profile integrity claims are incomplete")
    if profile.get("config_sha256") != sha256_file(config_path):
        raise ValueError("v12 runtime profile config hash mismatch")
    config_value = json.loads(config_path.read_text(encoding="utf-8"))
    contract_path = Path(str(config_value.get("contract_path") or ""))
    if not contract_path.is_absolute():
        contract_path = ROOT / contract_path
    if not contract_path.is_file() or profile.get("contract_sha256") != sha256_file(
        contract_path
    ):
        raise ValueError("v12 runtime profile contract hash mismatch")
    code_hashes = profile.get("code_sha256")
    expected_code_hashes = {
        "admission_cli": sha256_file(Path(__file__).resolve()),
        "admission_module": sha256_file(
            ROOT / "src/xgb_matcher/retrieval_hierarchical_admission_v12.py"
        ),
        "cascade_module": sha256_file(
            ROOT / "src/xgb_matcher/retrieval_top100_cascade_v12.py"
        ),
        "runtime_profiler": sha256_file(
            ROOT / "scripts/profile_retrieval_all_human_v12.py"
        ),
    }
    if code_hashes != expected_code_hashes:
        raise ValueError("v12 runtime profile code hashes do not match this run")

    producer = profile.get("producer_lineage")
    producer_keys = {
        "query_source_sha256",
        "query_source_manifest_sha256",
        "query_sample_source_sha256",
        "label_ledger_manifest_sha256",
        "v11_config_sha256",
        "official_config_sha256",
        "v11_candidate_manifest_sha256",
        "sidecar_manifest_sha256",
        "nested_parent_bundle_sha256",
        "index_manifest_sha256",
        "code_sha256",
    }
    if not isinstance(producer, Mapping) or set(producer) != producer_keys:
        raise ValueError("v12 runtime profile producer lineage is incomplete")
    direct_hash_keys = producer_keys.difference(
        {"index_manifest_sha256", "code_sha256"}
    )
    if any(not _is_sha256(producer.get(key)) for key in direct_hash_keys):
        raise ValueError("v12 runtime producer artifact hashes are invalid")
    if producer.get("label_ledger_manifest_sha256") != sha256_file(
        label_manifest_path
    ):
        raise ValueError("v12 runtime profile uses another label-ledger build")
    label_manifest = json.loads(label_manifest_path.read_text(encoding="utf-8"))
    if producer.get("query_sample_source_sha256") != label_manifest.get(
        "runtime_query_sample_sha256"
    ):
        raise ValueError("v12 runtime profile did not use the sealed ledger sample")
    expected_producer_code = {
        "profiler": sha256_file(ROOT / "scripts/profile_retrieval_all_human_v12.py"),
        "query_projection_builder": sha256_file(
            ROOT / "scripts/build_retrieval_all_human_v12_runtime_queries.py"
        ),
        "v11_retriever": sha256_file(
            ROOT / "src/xgb_matcher/retrieval_all_human_v11.py"
        ),
        "official_retriever": sha256_file(
            ROOT / "src/xgb_matcher/official_evidence_retrieval.py"
        ),
        "cascade_runner": sha256_file(
            ROOT / "scripts/run_retrieval_top100_cascade_v12.py"
        ),
    }
    if producer.get("code_sha256") != expected_producer_code:
        raise ValueError("v12 runtime producer code hashes drifted")
    index_hashes = producer.get("index_manifest_sha256")
    expected_index_keys = {
        "base",
        "overlay",
        "temporal_overlay",
        "legal_entity_role",
        "sector_official",
    }
    if (
        not isinstance(index_hashes, Mapping)
        or set(index_hashes) != expected_index_keys
        or any(
            not isinstance(values, list)
            or any(not _is_sha256(value) for value in values)
            for values in index_hashes.values()
        )
        or len(index_hashes["base"]) != 1
    ):
        raise ValueError("v12 runtime official-index lineage is incomplete")

    lineage = profile.get("lineage")
    if not isinstance(lineage, Mapping):
        raise ValueError("v12 runtime profile lacks artifact lineage")
    if sealed_lineage is None:
        if any(
            value is None
            for value in (
                cascade_manifests,
                candidate_paths,
                model_paths,
                selected_paths,
            )
        ):
            raise ValueError("v12 runtime validation lacks current OOF lineage")
        expected_cascades = {
            str(fold): {
                "manifest_sha256": sha256_file(cascade_manifests[fold]),
                "master_sha256": sha256_file(candidate_paths[fold]),
            }
            for fold in sorted(_ALLOWED_FOLDS)
        }
        expected_models = {
            str(fold): {"site_model_sha256": sha256_file(model_paths[fold])}
            for fold in sorted(_ALLOWED_FOLDS)
        }
        expected_selected = {
            str(fold): {"selected_output_sha256": sha256_file(selected_paths[fold])}
            for fold in sorted(_ALLOWED_FOLDS)
        }
    else:
        expected_cascades = sealed_lineage.get("outer_cascades")
        expected_models = sealed_lineage.get("site_models")
        expected_selected = sealed_lineage.get("selected_outputs")
    if lineage.get("outer_cascades") != expected_cascades:
        raise ValueError("v12 runtime profile cascade lineage mismatch")
    if lineage.get("site_models") != expected_models:
        raise ValueError("v12 runtime profile site-model lineage mismatch")
    if lineage.get("selected_outputs") != expected_selected:
        raise ValueError("v12 runtime profile selected-output lineage mismatch")
    if cascade_manifests is not None:
        cascade_values = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in cascade_manifests.values()
        ]
        expected_producer_lineage = {
            "v11_candidate_manifest_sha256": {
                value.get("source_manifest_sha256") for value in cascade_values
            },
            "sidecar_manifest_sha256": {
                value.get("sidecar_manifest_sha256") for value in cascade_values
            },
            "nested_parent_bundle_sha256": {
                value.get("parent_model_bundle_sha256") for value in cascade_values
            },
        }
        for key, values in expected_producer_lineage.items():
            if len(values) != 1 or producer.get(key) != next(iter(values)):
                raise ValueError(f"v12 runtime producer lineage mismatch for {key}")

    sample = profile.get("query_sample")
    if not isinstance(sample, Mapping) or set(sample) != {
        "path",
        "sha256",
        "rows",
        "fold_counts",
        "selection_rule",
        "labels_present",
    }:
        raise ValueError("v12 runtime profile query-sample contract is incomplete")
    sample_name = sample.get("path")
    if not isinstance(sample_name, str) or Path(sample_name).name != sample_name:
        raise ValueError("v12 runtime profile query sample must be local")
    sample_path = (path.parent / sample_name).resolve()
    if sample_path.parent != path.parent.resolve() or not sample_path.is_file():
        raise ValueError("v12 runtime profile query sample is missing")
    if (
        sample.get("rows") != 512
        or sample.get("selection_rule") != _RUNTIME_SAMPLE_RULE
        or sample.get("labels_present") is not False
        or sample.get("sha256") != sha256_file(sample_path)
    ):
        raise ValueError("v12 runtime profile query sample is not sealed")
    sample_rows = pd.read_parquet(sample_path)
    if set(sample_rows.columns) != {"query_id", "fold"} or len(sample_rows) != 512:
        raise ValueError("v12 runtime sample must contain 512 label-free query/fold rows")
    sample_rows["query_id"] = sample_rows["query_id"].astype(str)
    sample_rows["fold"] = sample_rows["fold"].astype(int)
    if sample_rows["query_id"].duplicated().any() or set(sample_rows["fold"]) != _ALLOWED_FOLDS:
        raise ValueError("v12 runtime sample query/fold population is invalid")
    fold_counts = {
        str(int(key)): int(value)
        for key, value in sample_rows["fold"].value_counts().sort_index().items()
    }
    if sample.get("fold_counts") != fold_counts:
        raise ValueError("v12 runtime sample fold counts mismatch")
    label_lookup = labels.set_index("query_id")
    if not label_lookup.index.is_unique:
        raise ValueError("v12 runtime ledger query_id values must be unique")
    label_folds = label_lookup["fold"]
    if not set(sample_rows["query_id"]).issubset(set(label_folds.index)) or any(
        int(row.fold) != int(label_folds.loc[row.query_id])
        for row in sample_rows.itertuples(index=False)
    ):
        raise ValueError("v12 runtime sample is not a subset of the sealed OOF ledger")
    expected_sample = labels[["query_id", "fold"]].copy()
    expected_sample["query_id"] = expected_sample["query_id"].astype(str)
    expected_sample["fold"] = expected_sample["fold"].astype(int)
    expected_sample["_sample_key"] = expected_sample["query_id"].map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )
    expected_sample = (
        expected_sample.sort_values(["_sample_key", "query_id"], kind="stable")
        .head(512)[["query_id", "fold"]]
        .reset_index(drop=True)
    )
    if not sample_rows[["query_id", "fold"]].reset_index(drop=True).equals(
        expected_sample
    ):
        raise ValueError("v12 runtime sample does not follow its deterministic rule")

    hardware = profile.get("hardware")
    if not isinstance(hardware, Mapping) or (
        hardware.get("computer") != "Mac"
        or hardware.get("chip") != "Apple M4 Pro"
        or not isinstance(hardware.get("os"), str)
        or not hardware.get("os")
        or not isinstance(hardware.get("python"), str)
        or not hardware.get("python")
        or not isinstance(hardware.get("xgboost"), str)
        or not hardware.get("xgboost")
    ):
        raise ValueError("v12 runtime profile was not measured on the Mac M4 Pro")
    protocol = profile.get("protocol")
    required_stages = {
        "candidate_generation",
        "cascade",
        "canonicalization",
        "feature_computation",
        "site_lambdamart",
        "top100_selection",
    }
    if not isinstance(protocol, Mapping) or (
        protocol.get("cache_state") != "WARM"
        or protocol.get("warm") is not True
        or not isinstance(protocol.get("warmup_queries"), int)
        or protocol.get("warmup_queries") < 1
        or not isinstance(protocol.get("repetitions"), int)
        or protocol.get("repetitions") < 1
        or protocol.get("concurrency") != 1
        or protocol.get("end_to_end") is not True
        or protocol.get("measured_queries") != 512
        or protocol.get("missing_queries") != 0
        or set(protocol.get("stages") or ()) != required_stages
    ):
        raise ValueError("v12 runtime profile protocol is not a complete warm replay")

    parity = profile.get("parity")
    if not isinstance(parity, Mapping) or set(parity) != {
        "path",
        "sha256",
        "rows",
        "columns",
        "candidates_and_order_byte_identical",
        "reference_sha256",
        "replay_sha256",
    } or (
        parity.get("columns") != ["query_id", "candidate_siret", "selected_rank"]
        or parity.get("rows") != 512
        or parity.get("candidates_and_order_byte_identical") is not True
        or not _is_sha256(parity.get("reference_sha256"))
        or parity.get("reference_sha256") != parity.get("replay_sha256")
    ):
        raise ValueError("v12 runtime profile does not prove byte-identical output")
    parity_name = parity.get("path")
    if not isinstance(parity_name, str) or Path(parity_name).name != parity_name:
        raise ValueError("v12 runtime parity sidecar must be local")
    parity_path = (path.parent / parity_name).resolve()
    if (
        parity_path.parent != path.parent.resolve()
        or not parity_path.is_file()
        or parity.get("sha256") != sha256_file(parity_path)
    ):
        raise ValueError("v12 runtime parity sidecar is not content sealed")
    parity_rows = pd.read_parquet(parity_path)
    parity_columns = {
        "query_id",
        "fold",
        "reference_sha256",
        "replay_sha256",
        "byte_identical",
    }
    if set(parity_rows.columns) != parity_columns or len(parity_rows) != 512:
        raise ValueError("v12 runtime parity sidecar schema/population mismatch")
    parity_rows["query_id"] = parity_rows["query_id"].astype(str)
    parity_rows["fold"] = parity_rows["fold"].astype(int)
    if not parity_rows[["query_id", "fold"]].reset_index(drop=True).equals(
        sample_rows[["query_id", "fold"]].reset_index(drop=True)
    ):
        raise ValueError("v12 runtime parity sidecar does not match query sample")
    if (
        not pd.api.types.is_bool_dtype(parity_rows["byte_identical"])
        or not parity_rows["byte_identical"].eq(True).all()
        or any(
            not _is_sha256(reference)
            or reference != replay
            for reference, replay in zip(
                parity_rows["reference_sha256"],
                parity_rows["replay_sha256"],
            )
        )
        or parity.get("reference_sha256")
        != _parity_aggregate_digest(parity_rows, "reference_sha256")
        or parity.get("replay_sha256")
        != _parity_aggregate_digest(parity_rows, "replay_sha256")
    ):
        raise ValueError("v12 runtime parity sidecar contains a replay mismatch")
    latency = profile.get("latency_ms")
    if not isinstance(latency, Mapping) or set(latency) != {"p50", "p95", "p99"}:
        raise ValueError("v12 runtime profile latency percentiles are incomplete")
    values = [latency.get(name) for name in ("p50", "p95", "p99")]
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(float(value))
        or float(value) < 0.0
        for value in values
    ) or not (float(values[0]) <= float(values[1]) <= float(values[2])):
        raise ValueError("v12 runtime profile latency values are invalid")
    return profile


def _apply_runtime_profile_gate(
    summary: Mapping[str, Any],
    *,
    config: AdmissionConfigV12,
    profile: Mapping[str, Any] | None,
    manifest_path: Path | None,
) -> dict[str, Any]:
    """Replace embedded timing diagnostics with the sealed runtime SLO gate."""
    updated = dict(summary)
    updated["embedded_latency_diagnostic"] = dict(updated.get("latency") or {})
    gates = {key: dict(value) for key, value in (updated.get("gates") or {}).items()}
    if profile is None:
        latency = {"measured_queries": 0, "total_queries": 512, "p50_ms": None, "p95_ms": None, "p99_ms": None}
        profile_summary = {
            "provided": False,
            "gate_eligible": False,
            "required_schema_version": _RUNTIME_PROFILE_SCHEMA,
            "reason": "SEALED_WARM_RUNTIME_PROFILE_REQUIRED",
        }
    else:
        values = profile["latency_ms"]
        latency = {
            "measured_queries": 512,
            "total_queries": 512,
            "p50_ms": float(values["p50"]),
            "p95_ms": float(values["p95"]),
            "p99_ms": float(values["p99"]),
        }
        profile_summary = {
            "provided": True,
            "gate_eligible": True,
            "schema_version": profile["schema_version"],
            "manifest_sha256": sha256_file(manifest_path) if manifest_path else None,
            "hardware": dict(profile["hardware"]),
            "protocol": dict(profile["protocol"]),
        }
    gates["latency_p95_ms"] = {
        "observed": latency["p95_ms"],
        "maximum": config.gates["latency_p95_ms"],
        "passed": bool(
            profile is not None
            and latency["p95_ms"] is not None
            and latency["p95_ms"] <= config.gates["latency_p95_ms"]
        ),
        "source": "SEALED_WARM_RUNTIME_PROFILE",
    }
    gates["latency_p99_ms"] = {
        "observed": latency["p99_ms"],
        "maximum": config.gates["latency_p99_ms"],
        "passed": bool(
            profile is not None
            and latency["p99_ms"] is not None
            and latency["p99_ms"] <= config.gates["latency_p99_ms"]
        ),
        "source": "SEALED_WARM_RUNTIME_PROFILE",
    }
    updated["latency"] = latency
    updated["runtime_profile"] = profile_summary
    updated["gates"] = gates
    recall_passed = all(
        bool(gates[name]["passed"])
        for name in (
            "prerank_at_500",
            "selected_recall_at_100",
            "wilson95_lower_at_100",
            "worst_fold_at_100",
        )
    )
    official_known = bool(gates["official_catalog_loss_taxonomy"]["passed"])
    latency_passed = bool(
        gates["latency_p95_ms"]["passed"] and gates["latency_p99_ms"]["passed"]
    )
    updated["decision_eligible"] = bool(profile is not None and official_known)
    if not official_known:
        updated["verdict"] = "STOP"
        updated["stop_reason"] = "INTEGRITY_OFFICIAL_CATALOG_UNAVAILABLE"
    elif not gates["prerank_at_500"]["passed"]:
        updated["verdict"] = "PIVOT_GENERATION"
        updated["stop_reason"] = None
    elif not recall_passed:
        updated["verdict"] = "PIVOT_ADMISSION"
        updated["stop_reason"] = None
    elif profile is None or not latency_passed:
        updated["verdict"] = "PIVOT_LATENCY"
        updated["stop_reason"] = None
    else:
        updated["verdict"] = "READY_FOR_FOLD1"
        updated["stop_reason"] = None
    return updated


def _label_index(labels: pd.DataFrame) -> pd.DataFrame:
    if labels.index.name == "query_id" and "query_id" in labels.columns:
        if labels.index.has_duplicates:
            raise ValueError("v12 label-side artifact must be unique by query")
        return labels
    if labels["query_id"].astype(str).duplicated().any():
        raise ValueError("v12 label-side artifact must be unique by query")
    indexed = labels.copy()
    indexed["query_id"] = indexed["query_id"].astype(str)
    return indexed.set_index("query_id", drop=False)


def _attach_query_label(raw: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    query_id = str(raw.iloc[0]["query_id"])
    indexed = _label_index(labels)
    if query_id not in indexed.index:
        raise ValueError(f"v12 has no unique sealed label for query {query_id!r}")
    label = indexed.loc[query_id]
    if isinstance(label, pd.DataFrame):
        raise ValueError(f"v12 has no unique sealed label for query {query_id!r}")
    if int(raw.iloc[0]["fold"]) != int(label["fold"]):
        raise ValueError("v12 cascade fold disagrees with sealed labels")
    output = raw.copy()
    output["human_ground_truth_siret"] = str(label["human_siret"])
    for name in (
        "acceptable_sirets_operational_json",
        "ground_truth_state",
        "label_kind",
        "qualification_v2",
        "qualification_v3",
        "source_kind",
        "data_origin",
        "split",
        "unseen_siren",
        "new_site_known_siren",
        "mega_base_pool",
        "pool_size",
        "insee",
        "seen_unseen_siren",
        "known_new_site",
        "megapolis",
        "qualification",
        "official_catalog_present",
    ):
        if name in label.index:
            output[name] = label[name]
    return output


def _load_parent_fit_sample(
    path: Path, labels: pd.DataFrame, config: AdmissionConfigV12
) -> tuple[pd.DataFrame, Mapping[str, int]]:
    """Stream pass-A parents and retain at most the pinned fit cap/query."""
    cap = int(config.fit_candidate_sampling["maximum_candidates_per_query"])
    pieces: list[pd.DataFrame] = []
    rows_before = rows_after = natural_positives = 0
    seen_folds: set[int] = set()
    label_index = labels.set_index("query_id", drop=False)
    for raw in iter_complete_query_frames(path, batch_size=131_072):
        query_id = str(raw.iloc[0]["query_id"])
        if query_id not in label_index.index:
            raise ValueError("pass-A parent pool contains a query outside label ledger")
        label = label_index.loc[query_id]
        if isinstance(label, pd.DataFrame):
            raise ValueError("label ledger is not unique by query")
        fold = int(raw.iloc[0]["fold"])
        if fold != int(label["fold"]) or fold not in _ALLOWED_FOLDS:
            raise ValueError("pass-A parent pool fold/label mismatch")
        seen_folds.add(fold)
        adapted = canonicalize_cascade_parent_rows(raw)
        adapted["fold"] = fold
        adapted["human_siret"] = str(label["human_siret"])
        sealed_rank = pd.to_numeric(
            adapted["sealed_parent_rank"], errors="coerce"
        )
        adapted["__rank"] = sealed_rank.where(
            sealed_rank.notna(),
            pd.Series(np.arange(1, len(adapted) + 1), index=adapted.index),
        )
        truth_parent = str(label["human_siret"])[:9]
        positive = adapted[adapted["candidate_parent"].eq(truth_parent)]
        selected = adapted.nsmallest(cap, "__rank")
        if not positive.empty and truth_parent not in set(selected["candidate_parent"]):
            selected = pd.concat(
                [selected.iloc[: max(0, cap - 1)], positive.iloc[[0]]],
                ignore_index=True,
            )
        selected = selected.drop(columns="__rank").drop_duplicates(
            ["query_id", "candidate_parent"]
        )
        pieces.append(selected)
        rows_before += len(adapted)
        rows_after += len(selected)
        natural_positives += int(not positive.empty)
    if not pieces or seen_folds != _ALLOWED_FOLDS:
        raise ValueError("pass-A parent pool must cover folds 2/3/4")
    output = pd.concat(pieces, ignore_index=True)
    if int(output.groupby("query_id").size().max()) > cap:
        raise AssertionError("parent fit sample cap exceeded")
    return output, {
        "rows_before_sampling": rows_before,
        "rows_after_sampling": rows_after,
        "maximum_candidates_per_query": cap,
        "natural_parent_positive_queries": natural_positives,
    }


def run_fit_parent_oof(args: argparse.Namespace) -> Path:
    """Fit the six frozen parent models needed by nested cascade generation."""
    config = AdmissionConfigV12.load(args.config)
    parent_manifest, pass_a_manifest = verify_parent_candidates(
        args.parent_candidates, require_nested_oof=False
    )
    if (
        pass_a_manifest.get("parent_scorer_kind")
        != "DETERMINISTIC_BOOTSTRAP_NOT_GATE_ELIGIBLE"
        or pass_a_manifest.get("bootstrap") is not True
        or set(int(value) for value in pass_a_manifest.get("opened_folds", ()))
        != _ALLOWED_FOLDS
        or pass_a_manifest.get("rescue_hits_sha256") is not None
        or pass_a_manifest.get("gate_eligible") is not False
        or any(
            not _is_sha256(pass_a_manifest.get(key))
            for key in (
                "source_candidate_sha256",
                "source_manifest_sha256",
                "sidecar_manifest_sha256",
            )
        )
        or pass_a_manifest.get("config_sha256") != sha256_file(args.config)
    ):
        raise ValueError(
            "fit-parent-oof requires the sealed deterministic pass-A bootstrap "
            "ledger with no rescue"
        )
    query_labels, labels_manifest = _load_sealed_query_labels(
        args.labels, config_path=args.config
    )
    _prepare_output(args.output_dir)
    cascade_parents, parent_sample_diagnostics = _load_parent_fit_sample(
        args.parent_candidates, query_labels, config
    )
    if set(cascade_parents["fold"].astype(int).unique()) != _ALLOWED_FOLDS:
        raise ValueError("v12 parent candidate pool does not cover folds 2/3/4")

    models: dict[str, Mapping[str, Any]] = {}
    diagnostics: dict[str, Mapping[str, int]] = {}
    for name, fit_folds in _PARENT_FIT_FOLDS.items():
        fit_rows = cascade_parents[cascade_parents["fold"].isin(fit_folds)]
        model, diag = fit_parent_ranker_from_cascade_rows(fit_rows, config)
        model_path = args.output_dir / f"parent_{name}.json"
        model.save_model(model_path)
        models[name] = {
            "path": model_path.name,
            "sha256": sha256_file(model_path),
            "fit_folds": list(fit_folds),
        }
        diagnostics[name] = dict(diag)

    outer_maps = {
        f"outer{outer}": {
            "outer_heldout_fold": int(outer),
            "scorer_by_query_fold": mapping,
        }
        for outer, mapping in _OUTER_PARENT_MAPS.items()
    }
    manifest = {
        "schema_version": PARENT_OOF_BUNDLE_SCHEMA,
        "run_kind": "PARENT_NESTED_OOF_MODEL_BUNDLE",
        "gate_eligible_for_nested_candidate_generation": True,
        "parent_feature_origin": "CASCADE_PRE_EXPANSION",
        "parent_feature_names": list(parent_feature_names(config)),
        "opened_folds": [2, 3, 4],
        "fold0_opened": False,
        "fold1_opened": False,
        "positive_injection": False,
        "source_parent_candidates_sha256": sha256_file(args.parent_candidates),
        "source_parent_manifest_sha256": sha256_file(parent_manifest),
        "source_retrieval_lineage": {
            key: pass_a_manifest[key]
            for key in (
                "source_candidate_sha256",
                "source_manifest_sha256",
                "sidecar_manifest_sha256",
            )
        },
        "source_labels_sha256": sha256_file(args.labels),
        "source_labels_manifest_sha256": sha256_file(labels_manifest),
        "config_sha256": sha256_file(args.config),
        "models": models,
        "outer_maps": outer_maps,
        "diagnostics": diagnostics,
        "fit_sampling": parent_sample_diagnostics,
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    return args.output_dir


def run_attach_labels(args: argparse.Namespace) -> Path:
    """Join the sealed benchmark only after label-blind cascade generation."""
    source_manifest_path, source_manifest = verify_candidate_manifest(
        args.master_candidates, labelled=False, require_master=True
    )
    labels, labels_manifest_path = _load_sealed_query_labels(
        args.labels, config_path=args.config
    )
    labels_by_query = _label_index(labels)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    seen: set[str] = set()
    query_count = natural_positive_queries = 0
    maximum_master = maximum_prerank = 0
    try:
        for raw in iter_complete_query_frames(
            args.master_candidates, batch_size=131_072
        ):
            labelled = _attach_query_label(raw, labels_by_query)
            query_id = str(labelled.iloc[0]["query_id"])
            seen.add(query_id)
            truth = str(labelled.iloc[0]["human_ground_truth_siret"])
            natural_positive_queries += int(
                truth in set(labelled["siret"].astype(str))
            )
            query_count += 1
            maximum_master = max(maximum_master, len(labelled))
            maximum_prerank = max(
                maximum_prerank,
                int(pd.to_numeric(labelled["hydration_rank"], errors="coerce").notna().sum()),
            )
            table = pa.Table.from_pandas(
                labelled, schema=writer.schema if writer else None, preserve_index=False
            )
            if writer is None:
                writer = pq.ParquetWriter(args.output, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    expected = set(labels["query_id"].astype(str))
    if writer is None:
        raise ValueError("v12 label attachment source has no candidate group")
    missing_ids = expected.difference(seen)
    missing_path = args.output.with_suffix(
        args.output.suffix + ".missing_label_queries.parquet"
    )
    labels[labels["query_id"].isin(missing_ids)].to_parquet(
        missing_path, index=False, compression="zstd"
    )
    manifest = {
        "schema_version": "sireto-retrieval-v12-labelled-master-manifest-1",
        "run_kind": "LABEL_SIDE_JOIN_AFTER_LABEL_BLIND_RETRIEVAL",
        "candidate_parquet_sha256": sha256_file(args.output),
        "source_master_sha256": sha256_file(args.master_candidates),
        "source_master_manifest_sha256": sha256_file(source_manifest_path),
        "source_labels_sha256": sha256_file(args.labels),
        "source_labels_manifest_sha256": sha256_file(labels_manifest_path),
        "positive_injection": False,
        "human_labels_opened_by_retrieval": False,
        "labels_attached_after_retrieval": True,
        "all_label_queries_represented": not missing_ids,
        "missing_label_query_count": len(missing_ids),
        "missing_label_queries_sha256": sha256_file(missing_path),
        "all_label_queries_have_natural_candidates": natural_positive_queries == query_count,
        "query_count": query_count,
        "natural_positive_queries": natural_positive_queries,
        "opened_folds": [2, 3, 4],
        "fold0_opened": False,
        "fold1_opened": False,
        "maximum_master_references": maximum_master,
        "maximum_hydration_shortlist": maximum_prerank,
        "candidate_view": {
            "union_rank_semantics": "MASTER_RANK_WITHIN_MAX_2000",
            "hydration_rank_preserved": True,
        },
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    _write_json(manifest_path, manifest)
    return args.output


def _load_feature_sample(
    path: Path, config: AdmissionConfigV12
) -> tuple[pd.DataFrame, Mapping[str, int]]:
    """Stream complete queries, compute full-union features, then sample."""
    pieces: list[pd.DataFrame] = []
    before = after = positives = queries = 0
    folds: set[int] = set()
    for raw in iter_complete_query_frames(path, batch_size=131_072):
        fold = int(raw.iloc[0]["fold"])
        if fold not in config.train_folds:
            raise ValueError("v12 fit attempted to open fold 0/1 or an unknown fold")
        folds.add(fold)
        complete = canonicalize_union(
            raw, config, labels_allowed=True, require_labels=True
        )
        sampled = sample_precomputed_candidates(complete, config)
        fit_columns = [
            "query_id",
            "fold",
            "human_siret",
            "operational_sirets_json",
            "candidate_siret",
            "candidate_parent",
            "heuristic_hydration_rank",
            "features_computed_on_complete_union",
            "sealed_parent_features_present",
            "sealed_parent_rank",
            "sealed_parent_score",
            "parent_preexpansion_missing",
            *site_feature_names(config),
        ]
        fit_columns = list(dict.fromkeys(fit_columns))
        sampled = sampled[fit_columns].copy()
        numeric_features = list(site_feature_names(config))
        sampled[numeric_features] = sampled[numeric_features].astype(np.float32)
        pieces.append(sampled)
        queries += 1
        before += len(complete)
        after += len(sampled)
        positives += int(sampled["candidate_siret"].eq(sampled["human_siret"]).sum())
    if not pieces or folds != set(config.train_folds):
        raise ValueError("v12 fit input must contain exactly folds 2/3/4")
    return pd.concat(pieces, ignore_index=True), {
        "queries": queries,
        "rows_before_sampling": before,
        "rows_after_sampling": after,
        "natural_positives_retained": positives,
        "features_computed_before_sampling": True,
    }


def _load_outer_training_features(
    path: Path,
    labels: pd.DataFrame,
    *,
    heldout: int,
    config: AdmissionConfigV12,
) -> tuple[pd.DataFrame, Mapping[str, int]]:
    """Load only the two fit folds from one sealed outer cascade context."""
    fit_folds = set(config.train_folds).difference({heldout})
    pieces: list[pd.DataFrame] = []
    labels_by_query = _label_index(labels)
    observed: set[int] = set()
    rows_before = rows_after = queries = 0
    for raw_blind in iter_complete_query_frames(path, batch_size=131_072):
        fold = int(raw_blind.iloc[0]["fold"])
        if fold == heldout:
            continue
        if fold not in fit_folds:
            raise ValueError("nested v12 training encountered a forbidden fold")
        observed.add(fold)
        raw = _attach_query_label(raw_blind, labels_by_query)
        complete = canonicalize_union(
            raw, config, labels_allowed=True, require_labels=True
        )
        sampled = sample_precomputed_candidates(complete, config)
        # Keep only the columns consumed by the site LambdaMART.  The Master
        # rows may carry large evidence payloads; retaining those payloads for
        # every sampled candidate would multiply memory use across queries.
        fit_columns = list(
            dict.fromkeys(
                [
                    "query_id",
                    "fold",
                    "human_siret",
                    "operational_sirets_json",
                    "candidate_siret",
                    "candidate_parent",
                    "heuristic_hydration_rank",
                    "features_computed_on_complete_union",
                    "sealed_parent_features_present",
                    "sealed_parent_rank",
                    "sealed_parent_score",
                    "parent_preexpansion_missing",
                    *site_feature_names(config),
                ]
            )
        )
        sampled = sampled[fit_columns].copy()
        sampled[list(site_feature_names(config))] = sampled[
            list(site_feature_names(config))
        ].astype(np.float32)
        pieces.append(sampled)
        queries += 1
        rows_before += len(complete)
        rows_after += len(sampled)
    if not pieces or observed != fit_folds:
        raise ValueError("nested v12 outer context lacks one of its two fit folds")
    return pd.concat(pieces, ignore_index=True), {
        "queries": queries,
        "rows_before_sampling": rows_before,
        "rows_after_sampling": rows_after,
        "features_computed_before_sampling": True,
    }


def _selected_projection(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        [
            "query_id",
            "fold",
            "candidate_siret",
            "candidate_parent",
            "union_rank",
            "master_rank",
            "heuristic_hydration_rank",
            "prerank_rank",
            "ranked_rank",
            "selected_rank",
            "final_score",
            "marginal_score",
            "parent_rank",
            "site_rank",
            "selection_reason",
            "channels_json",
            "official_evidence_ids_json",
            "official_evidence_sources_json",
            "admission_latency_ms",
        ]
    ].copy()


def _write_selected(
    writer: pq.ParquetWriter | None, path: Path, selected: pd.DataFrame
) -> pq.ParquetWriter:
    table = pa.Table.from_pandas(
        _selected_projection(selected),
        schema=writer.schema if writer else None,
        preserve_index=False,
    )
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema, compression="zstd")
    writer.write_table(table)
    return writer


def _stream_oof_validation(
    path: Path,
    models: Mapping[int, HierarchicalModelsV12],
    config: AdmissionConfigV12,
    selected_path: Path,
) -> tuple[pd.DataFrame, Mapping[str, Any]]:
    writer: pq.ParquetWriter | None = None
    chunks: list[pd.DataFrame] = []
    by_fold = {fold: 0 for fold in config.train_folds}
    maximum_master = maximum_prerank = maximum_selected = 0
    try:
        for raw in iter_complete_query_frames(path, batch_size=131_072):
            admission_started = time.perf_counter()
            fold = int(raw.iloc[0]["fold"])
            if fold not in config.train_folds:
                raise ValueError("v12 validation attempted to open fold 0/1")
            complete = canonicalize_union(
                raw, config, labels_allowed=True, require_labels=True
            )
            blind = complete.copy()
            blind["human_siret"] = ""
            scored = score_canonical_candidates(
                models[fold], blind, config
            )
            selected = select_top100(scored, config)
            admission_latency_ms = (time.perf_counter() - admission_started) * 1_000.0
            scored["admission_latency_ms"] = admission_latency_ms
            selected["admission_latency_ms"] = admission_latency_ms
            outcomes, _ = evaluate_all_human(
                complete, scored=scored, selected=selected, config=config
            )
            chunks.append(outcomes)
            writer = _write_selected(writer, selected_path, selected)
            by_fold[fold] += 1
            maximum_master = max(maximum_master, len(complete))
            maximum_prerank = max(
                maximum_prerank, int(scored["prerank_rank"].notna().sum())
            )
            maximum_selected = max(maximum_selected, len(selected))
    finally:
        if writer is not None:
            writer.close()
    if writer is None or not chunks:
        raise ValueError("v12 OOF produced no validation output")
    return pd.concat(chunks, ignore_index=True), {
        "queries_by_fold": {str(key): value for key, value in by_fold.items()},
        "maximum_master_references": maximum_master,
        "maximum_prerank_references": maximum_prerank,
        "maximum_published_candidates": maximum_selected,
        "validation_candidate_sampling": False,
    }


def _stream_one_outer_validation(
    path: Path,
    model: HierarchicalModelsV12,
    labels: pd.DataFrame,
    *,
    heldout: int,
    config: AdmissionConfigV12,
    selected_path: Path,
) -> tuple[pd.DataFrame, Mapping[str, Any]]:
    writer: pq.ParquetWriter | None = None
    chunks: list[pd.DataFrame] = []
    labels_by_query = _label_index(labels)
    maximum_master = maximum_prerank = maximum_selected = 0
    try:
        for raw_blind in iter_complete_query_frames(path, batch_size=131_072):
            fold = int(raw_blind.iloc[0]["fold"])
            if fold != heldout:
                continue
            raw = _attach_query_label(raw_blind, labels_by_query)
            # Label attachment is evaluation-only and absent from production.
            # Time every production operation from canonicalisation onward.
            admission_started = time.perf_counter()
            complete = canonicalize_union(
                raw, config, labels_allowed=True, require_labels=True
            )
            blind = complete.copy()
            blind["human_siret"] = ""
            scored = score_canonical_candidates(
                model, blind, config
            )
            selected = select_top100(scored, config)
            admission_latency_ms = (time.perf_counter() - admission_started) * 1_000.0
            scored["admission_latency_ms"] = admission_latency_ms
            selected["admission_latency_ms"] = admission_latency_ms
            outcomes, _ = evaluate_all_human(
                complete, scored=scored, selected=selected, config=config
            )
            chunks.append(outcomes)
            writer = _write_selected(writer, selected_path, selected)
            maximum_master = max(maximum_master, len(complete))
            maximum_prerank = max(
                maximum_prerank, int(scored["prerank_rank"].notna().sum())
            )
            maximum_selected = max(maximum_selected, len(selected))
    finally:
        if writer is not None:
            writer.close()
    if writer is None or not chunks:
        raise ValueError(f"outer{heldout} produced no heldout validation rows")
    return pd.concat(chunks, ignore_index=True), {
        "heldout_fold": heldout,
        "queries": len(chunks),
        "maximum_master_references": maximum_master,
        "maximum_prerank_references": maximum_prerank,
        "maximum_published_candidates": maximum_selected,
        "validation_candidate_sampling": False,
    }


def _base_manifest(
    *,
    run_kind: str,
    candidates: Path,
    source_manifest: Path,
    config_path: Path,
    config: AdmissionConfigV12,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_kind": run_kind,
        "policy_id": config.policy_id,
        "source_candidates_sha256": sha256_file(candidates),
        "source_manifest_sha256": sha256_file(source_manifest),
        "config_sha256": sha256_file(config_path),
        "opened_folds": list(config.train_folds),
        "fold0_opened": False,
        "fold1_opened": False,
        "positive_injection": False,
    }


def _reconcile_oof_population(
    outcomes: pd.DataFrame, labels: pd.DataFrame
) -> None:
    """Fail closed unless OOF outcomes are exactly the sealed label population."""

    required = {"query_id", "fold"}
    if not required.issubset(outcomes.columns) or not required.issubset(labels.columns):
        raise ValueError("OOF population reconciliation requires query_id and fold")
    observed = outcomes[["query_id", "fold"]].copy()
    expected = labels[["query_id", "fold"]].copy()
    for frame, name in ((observed, "outcomes"), (expected, "label ledger")):
        frame["query_id"] = frame["query_id"].astype(str)
        frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
        if frame["query_id"].eq("").any() or frame["query_id"].duplicated().any():
            raise ValueError(f"OOF {name} query_id values must be non-empty and unique")
    if len(observed) != len(expected) or set(observed["query_id"]) != set(
        expected["query_id"]
    ):
        raise ValueError("OOF outcomes do not exactly match the sealed label denominator")
    observed_folds = observed.set_index("query_id")["fold"].sort_index()
    expected_folds = expected.set_index("query_id")["fold"].sort_index()
    if not observed_folds.equals(expected_folds):
        raise ValueError("OOF outcome folds do not match the sealed label ledger")


def run_oof(args: argparse.Namespace) -> Path:
    config = AdmissionConfigV12.load(args.config)
    labels, labels_manifest = _load_sealed_query_labels(
        args.labels, config_path=args.config
    )
    cascade_dirs = {
        2: args.outer2_cascade,
        3: args.outer3_cascade,
        4: args.outer4_cascade,
    }
    candidate_paths: dict[int, Path] = {}
    parent_paths: dict[int, Path] = {}
    cascade_manifests: dict[int, Path] = {}
    cascade_manifest_values: dict[int, Mapping[str, Any]] = {}
    for heldout, directory in cascade_dirs.items():
        candidate = directory / "master_references.parquet"
        parent = directory / "parent_candidates.parquet"
        candidate_manifest, candidate_manifest_value = verify_candidate_manifest(
            candidate, labelled=False, require_master=True
        )
        verify_parent_candidates(
            parent, require_nested_oof=True, expected_outer_fold=heldout
        )
        candidate_paths[heldout] = candidate
        parent_paths[heldout] = parent
        cascade_manifests[heldout] = candidate_manifest
        cascade_manifest_values[heldout] = candidate_manifest_value
    expected_config_sha = sha256_file(args.config)
    expected_label_lineage = {
        "label_ledger_sha256": sha256_file(args.labels),
        "label_ledger_manifest_sha256": sha256_file(labels_manifest),
    }
    sealed_identity_fields = (
        "source_candidate_sha256",
        "source_manifest_sha256",
        "sidecar_manifest_sha256",
        "parent_model_bundle_sha256",
    )
    for manifest in cascade_manifest_values.values():
        if manifest.get("config_sha256") != expected_config_sha:
            raise ValueError("outer cascade config hash differs from admission config")
        training_lineage = (manifest.get("nested_parent_oof") or {}).get(
            "training_lineage"
        )
        if not isinstance(training_lineage, Mapping) or any(
            training_lineage.get(key) != value
            for key, value in expected_label_lineage.items()
        ):
            raise ValueError(
                "outer parent models were fitted on another label ledger"
            )
    for field in sealed_identity_fields:
        values = {manifest.get(field) for manifest in cascade_manifest_values.values()}
        if len(values) != 1 or None in values:
            raise ValueError(f"outer cascade lineage mismatch for {field}")
    _prepare_output(args.output_dir)
    sample_diagnostics: dict[str, Mapping[str, int]] = {}
    model_diagnostics: dict[str, Mapping[str, Any]] = {}
    model_paths: dict[int, Path] = {}
    outcome_parts: list[pd.DataFrame] = []
    selected_paths: dict[int, Path] = {}
    stream_diagnostics: dict[str, Mapping[str, Any]] = {}
    for heldout in config.train_folds:
        training, fit_diagnostics = _load_outer_training_features(
            candidate_paths[heldout], labels, heldout=heldout, config=config
        )
        model = fit_one_outer_site_model(
            training, heldout=heldout, config=config, pre_sampled=True
        )
        site_path = args.output_dir / f"fold{heldout}_site_ranker.json"
        model.site.save_model(site_path)
        model_paths[heldout] = site_path
        sample_diagnostics[str(heldout)] = fit_diagnostics
        model_diagnostics[str(heldout)] = dict(model.diagnostics)
        selected_path = args.output_dir / f"fold{heldout}_selected_candidates.parquet"
        outcomes, diagnostics = _stream_one_outer_validation(
            candidate_paths[heldout],
            model,
            labels,
            heldout=heldout,
            config=config,
            selected_path=selected_path,
        )
        outcome_parts.append(outcomes)
        selected_paths[heldout] = selected_path
        stream_diagnostics[str(heldout)] = diagnostics
        del training, model
    outcomes = pd.concat(outcome_parts, ignore_index=True)
    missing = missing_label_outcomes(labels, outcomes["query_id"], config)
    if not missing.empty:
        outcomes = pd.concat([outcomes, missing], ignore_index=True)
    _reconcile_oof_population(outcomes, labels)
    summary = dict(summarize_outcomes(outcomes, config))
    summary["nested_oof"] = {
        "outer_folds": list(config.train_folds),
        "model_diagnostics": model_diagnostics,
        "fit_sample_by_outer": sample_diagnostics,
        "stream_validation": stream_diagnostics,
        "fold0_opened": False,
        "fold1_opened": False,
    }
    runtime_manifest_path = getattr(args, "runtime_profile_manifest", None)
    runtime_profile = None
    if runtime_manifest_path is not None:
        runtime_profile = _validate_runtime_profile_manifest(
            runtime_manifest_path,
            config_path=args.config,
            labels=labels,
            label_manifest_path=labels_manifest,
            cascade_manifests=cascade_manifests,
            candidate_paths=candidate_paths,
            model_paths=model_paths,
            selected_paths=selected_paths,
        )
    summary = _apply_runtime_profile_gate(
        summary,
        config=config,
        profile=runtime_profile,
        manifest_path=runtime_manifest_path,
    )
    outcomes_path = args.output_dir / "query_outcomes.parquet"
    summary_path = args.output_dir / "summary.json"
    outcomes.to_parquet(outcomes_path, index=False, compression="zstd")
    _write_json(summary_path, summary)
    manifest = _base_manifest(
        run_kind="NESTED_OUTER_OOF",
        candidates=candidate_paths[2],
        source_manifest=cascade_manifests[2],
        config_path=args.config,
        config=config,
    )
    manifest.update(
        {
            "features_computed_before_sampling": True,
            "validation_candidate_sampling": False,
            "source_labels_sha256": sha256_file(args.labels),
            "source_labels_manifest_sha256": sha256_file(labels_manifest),
            "outer_cascades": {
                str(fold): {
                    "master_sha256": sha256_file(candidate_paths[fold]),
                    "parent_candidates_sha256": sha256_file(parent_paths[fold]),
                    "manifest_sha256": sha256_file(cascade_manifests[fold]),
                }
                for fold in config.train_folds
            },
            "outputs": {
                path.name: sha256_file(path)
                for path in (
                    *selected_paths.values(),
                    outcomes_path,
                    summary_path,
                    *model_paths.values(),
                )
            },
            "runtime_profile_manifest_sha256": (
                sha256_file(runtime_manifest_path)
                if runtime_manifest_path is not None
                else None
            ),
            "decision_eligible": summary["decision_eligible"],
            "verdict": summary["verdict"],
        }
    )
    _write_json(args.output_dir / "manifest.json", manifest)
    return args.output_dir


def run_apply_runtime_profile(args: argparse.Namespace) -> Path:
    """Apply the runtime gate to an existing sealed OOF run without refitting."""
    config = AdmissionConfigV12.load(args.config)
    labels, labels_manifest = _load_sealed_query_labels(
        args.labels, config_path=args.config
    )
    oof_manifest_path = args.oof_dir / "manifest.json"
    oof_summary_path = args.oof_dir / "summary.json"
    if not oof_manifest_path.is_file() or not oof_summary_path.is_file():
        raise FileNotFoundError("sealed OOF manifest/summary is missing")
    oof_manifest = json.loads(oof_manifest_path.read_text(encoding="utf-8"))
    if (
        oof_manifest.get("schema_version") != SCHEMA_VERSION
        or oof_manifest.get("run_kind") != "NESTED_OUTER_OOF"
        or oof_manifest.get("config_sha256") != sha256_file(args.config)
        or oof_manifest.get("positive_injection") is not False
        or oof_manifest.get("fold0_opened") is not False
        or oof_manifest.get("fold1_opened") is not False
        or set(oof_manifest.get("opened_folds") or ()) != _ALLOWED_FOLDS
        or oof_manifest.get("source_labels_sha256") != sha256_file(args.labels)
        or oof_manifest.get("source_labels_manifest_sha256")
        != sha256_file(labels_manifest)
    ):
        raise ValueError("runtime profile target is not the same sealed v12 OOF run")
    outputs = oof_manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("sealed OOF run lacks output hashes")
    required_output_names = {
        "summary.json",
        "query_outcomes.parquet",
        *(f"fold{fold}_site_ranker.json" for fold in sorted(_ALLOWED_FOLDS)),
        *(
            f"fold{fold}_selected_candidates.parquet"
            for fold in sorted(_ALLOWED_FOLDS)
        ),
    }
    if not required_output_names.issubset(outputs):
        raise ValueError("sealed OOF run lacks a model, selection or metrics artifact")
    for name in required_output_names:
        artifact = args.oof_dir / name
        if (
            not artifact.is_file()
            or not _is_sha256(outputs.get(name))
            or sha256_file(artifact) != outputs.get(name)
        ):
            raise ValueError(f"sealed OOF output hash mismatch: {name}")
    outer_cascades = oof_manifest.get("outer_cascades")
    if not isinstance(outer_cascades, Mapping) or set(outer_cascades) != {
        "2",
        "3",
        "4",
    }:
        raise ValueError("sealed OOF run lacks all three outer cascade lineages")
    expected_cascades: dict[str, Mapping[str, str]] = {}
    for fold in sorted(_ALLOWED_FOLDS):
        entry = outer_cascades.get(str(fold))
        if not isinstance(entry, Mapping) or not all(
            _is_sha256(entry.get(key))
            for key in ("master_sha256", "parent_candidates_sha256", "manifest_sha256")
        ):
            raise ValueError("sealed OOF outer cascade lineage is incomplete")
        expected_cascades[str(fold)] = {
            "manifest_sha256": entry["manifest_sha256"],
            "master_sha256": entry["master_sha256"],
        }
    sealed_lineage = {
        "outer_cascades": expected_cascades,
        "site_models": {
            str(fold): {
                "site_model_sha256": outputs[f"fold{fold}_site_ranker.json"]
            }
            for fold in sorted(_ALLOWED_FOLDS)
        },
        "selected_outputs": {
            str(fold): {
                "selected_output_sha256": outputs[
                    f"fold{fold}_selected_candidates.parquet"
                ]
            }
            for fold in sorted(_ALLOWED_FOLDS)
        },
    }
    profile = _validate_runtime_profile_manifest(
        args.runtime_profile_manifest,
        config_path=args.config,
        labels=labels,
        label_manifest_path=labels_manifest,
        sealed_lineage=sealed_lineage,
    )
    profile_lineage = profile.get("lineage") or {}
    if (
        profile_lineage.get("oof_manifest_sha256")
        != sha256_file(oof_manifest_path)
        or profile_lineage.get("query_outcomes_sha256")
        != outputs["query_outcomes.parquet"]
    ):
        raise ValueError("runtime profile is not bound to this exact OOF result")
    source_summary = json.loads(oof_summary_path.read_text(encoding="utf-8"))
    updated = _apply_runtime_profile_gate(
        source_summary,
        config=config,
        profile=profile,
        manifest_path=args.runtime_profile_manifest,
    )
    _prepare_output(args.output_dir)
    summary_path = args.output_dir / "summary.json"
    _write_json(summary_path, updated)
    decision_manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_kind": "APPLY_SEALED_RUNTIME_PROFILE_NO_REFIT",
        "config_sha256": sha256_file(args.config),
        "source_oof_manifest_sha256": sha256_file(oof_manifest_path),
        "source_oof_summary_sha256": sha256_file(oof_summary_path),
        "source_labels_sha256": sha256_file(args.labels),
        "source_labels_manifest_sha256": sha256_file(labels_manifest),
        "runtime_profile_manifest_sha256": sha256_file(
            args.runtime_profile_manifest
        ),
        "output_summary_sha256": sha256_file(summary_path),
        "model_refit": False,
        "candidate_rescore": False,
        "fold0_opened": False,
        "fold1_opened": False,
        "decision_eligible": updated["decision_eligible"],
        "verdict": updated["verdict"],
    }
    _write_json(args.output_dir / "manifest.json", decision_manifest)
    return args.output_dir


def run_oracle(args: argparse.Namespace) -> Path:
    config = AdmissionConfigV12.load(args.config)
    source_manifest, _ = verify_candidate_manifest(
        args.candidates, labelled=False, require_master=True
    )
    labels, labels_manifest = _load_sealed_query_labels(
        args.labels, config_path=args.config
    )
    labels_by_query = _label_index(labels)
    _prepare_output(args.output_dir)
    chunks: list[pd.DataFrame] = []
    folds: set[int] = set()
    for raw_blind in iter_complete_query_frames(args.candidates, batch_size=131_072):
        fold = int(raw_blind.iloc[0]["fold"])
        if fold not in config.train_folds:
            raise ValueError("v12 oracle attempted to open fold 0/1")
        folds.add(fold)
        raw = _attach_query_label(raw_blind, labels_by_query)
        complete = canonicalize_union(
            raw, config, labels_allowed=True, require_labels=True
        )
        outcomes, _ = evaluate_all_human(
            complete, scored=None, selected=None, config=config
        )
        chunks.append(outcomes)
    if not chunks or folds != set(config.train_folds):
        raise ValueError("v12 oracle requires exactly folds 2/3/4")
    outcomes = pd.concat(chunks, ignore_index=True)
    missing = missing_label_outcomes(labels, outcomes["query_id"], config)
    if not missing.empty:
        outcomes = pd.concat([outcomes, missing], ignore_index=True)
    _reconcile_oof_population(outcomes, labels)
    summary = dict(summarize_outcomes(outcomes, config))
    summary.update(
        {
            "run_kind": "MASTER_AND_HEURISTIC_HYDRATION_DIAGNOSTIC",
            "admission_model_fit": False,
            "source_labels_sha256": sha256_file(args.labels),
            "source_labels_manifest_sha256": sha256_file(labels_manifest),
            "fold0_opened": False,
            "fold1_opened": False,
            # A standalone cascade is not an outer/inner OOF population.  Its
            # curve is useful for debugging generation only and can never
            # authorize admission, fold1, or a scientific gate decision.
            "decision_eligible": False,
            "verdict": "DIAGNOSTIC_ONLY",
        }
    )
    outcomes_path = args.output_dir / "query_outcomes.parquet"
    summary_path = args.output_dir / "summary.json"
    outcomes.to_parquet(outcomes_path, index=False, compression="zstd")
    _write_json(summary_path, summary)
    manifest = _base_manifest(
        run_kind="MASTER_AND_HEURISTIC_HYDRATION_DIAGNOSTIC",
        candidates=args.candidates,
        source_manifest=source_manifest,
        config_path=args.config,
        config=config,
    )
    manifest.update(
        {
            "admission_model_fit": False,
            "source_labels_sha256": sha256_file(args.labels),
            "source_labels_manifest_sha256": sha256_file(labels_manifest),
            "outputs": {
                outcomes_path.name: sha256_file(outcomes_path),
                summary_path.name: sha256_file(summary_path),
            },
            "verdict": summary["verdict"],
        }
    )
    _write_json(args.output_dir / "manifest.json", manifest)
    return args.output_dir


def run_fit(args: argparse.Namespace) -> Path:
    raise RuntimeError(
        "AFTER_READY_ONLY: production refit is intentionally disabled until "
        "OOF gates, latency, freeze and explicit fold1 authorization are complete"
    )


def run_predict(args: argparse.Namespace) -> Path:
    raise RuntimeError(
        "AFTER_READY_ONLY: production prediction is intentionally disabled until "
        "a frozen parent bundle, final site refit and model manifest are sealed"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    parent_oof = subparsers.add_parser("fit-parent-oof")
    parent_oof.add_argument("--parent-candidates", type=Path, required=True)
    parent_oof.add_argument("--labels", type=Path, required=True)
    parent_oof.add_argument("--output-dir", type=Path, required=True)
    parent_oof.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    attach = subparsers.add_parser("attach-labels")
    attach.add_argument("--master-candidates", type=Path, required=True)
    attach.add_argument("--labels", type=Path, required=True)
    attach.add_argument("--output", type=Path, required=True)
    attach.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    for command in ("oracle", "fit"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--candidates", type=Path, required=True)
        sub.add_argument("--output-dir", type=Path, required=True)
        sub.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        if command == "oracle":
            sub.add_argument("--labels", type=Path, required=True)
        if command == "fit":
            sub.add_argument("--parent-candidates", type=Path, required=True)
    oof = subparsers.add_parser("oof")
    oof.add_argument("--outer2-cascade", type=Path, required=True)
    oof.add_argument("--outer3-cascade", type=Path, required=True)
    oof.add_argument("--outer4-cascade", type=Path, required=True)
    oof.add_argument("--labels", type=Path, required=True)
    oof.add_argument("--output-dir", type=Path, required=True)
    oof.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    oof.add_argument(
        "--runtime-profile-manifest",
        type=Path,
        help=(
            "sealed 512-query warm Mac M4 Pro replay; when absent the OOF run "
            "is decision-ineligible and cannot return READY_FOR_FOLD1"
        ),
    )
    apply_runtime = subparsers.add_parser("apply-runtime-profile")
    apply_runtime.add_argument("--oof-dir", type=Path, required=True)
    apply_runtime.add_argument("--labels", type=Path, required=True)
    apply_runtime.add_argument("--runtime-profile-manifest", type=Path, required=True)
    apply_runtime.add_argument("--output-dir", type=Path, required=True)
    apply_runtime.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    predict = subparsers.add_parser("predict")
    predict.add_argument("--candidates", type=Path, required=True)
    predict.add_argument("--models-dir", type=Path, required=True)
    predict.add_argument("--parent-candidates", type=Path, required=True)
    predict.add_argument("--output-dir", type=Path, required=True)
    predict.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    runner = {
        "fit-parent-oof": run_fit_parent_oof,
        "attach-labels": run_attach_labels,
        "oracle": run_oracle,
        "oof": run_oof,
        "apply-runtime-profile": run_apply_runtime_profile,
        "fit": run_fit,
        "predict": run_predict,
    }[args.command]
    print(runner(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
