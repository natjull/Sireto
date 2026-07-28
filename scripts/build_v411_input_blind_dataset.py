#!/usr/bin/env python3
"""Build the V4.11 input-blind sparse top-100 dataset for ranker C."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_v46_aligned_dataset import (  # noqa: E402
    _directory_record,
    _external_path,
    _input_record,
    _path_signature,
    _physical_parquet_row_count,
    assert_authorized_canonical_table,
    validate_v42_runtime,
)
from src.xgb_matcher.features import (  # noqa: E402
    make_feature_rows_from_preprocessed,
    preprocess_crm_row,
    set_global_name_idf_map,
)
from src.xgb_matcher.partitioned_store import PartitionedCandidateStore  # noqa: E402
from src.xgb_matcher.retrieval import build_candidate_pool  # noqa: E402
from src.xgb_matcher.tfidf_cache import TfidfPersistentCache  # noqa: E402
from src.xgb_matcher.v41_retrieval import (  # noqa: E402
    V41CurrentStateStore,
    V41RetrievalConfig,
    V41RetrievalVariant,
)
from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


SCHEMA_VERSION = "sireto-v4.11-input-blind-ranker-dataset-1"
EXPERIMENT_ID = "V411_INPUT_BLIND_ALIGNED_STACK"
RETRIEVAL_POLICY_VERSION = "v4.11-input-blind-sparse-v42-1"
TFIDF_CACHE_POLICY_VERSION = "v4.11-verified-partition-cache-1"
SEED = 42
CANDIDATE_CEILING = 100
EXPECTED_QUERY_COUNT = 7_003
EXPECTED_INPUT_HASHES = {
    "queries.parquet": "6a12f1c4ca9ec33636ebcf7748c208595c6168d7cdb8c068e1434af3fe22abb0",
    "labels.parquet": "69032b745817959422ef26e4c0c1228686260c1daa272ca5d6aba1d7be087b04",
    "candidates_v42b.parquet": "0b7fc90e045da10033f0ae4b598963505d76c16710e2efc9dbe728a93a6536dc",
    "split_assignments.parquet": "33fa52af7a740124235c151efb5b9a8834ffd1c83c65d1af56c75b2eff271193",
    "state_snapshot": "c91180cc5bae86948dd57d752c9bae45e58cc64653e99d5a9357664b67300845",
    "v42_manifest": "63b52c3a1466070410881b0ea61b833ff5d413262239920abbc6b04e3f153f54",
}
EXPECTED_V42_SOURCE_CONFIG_SIGNATURE = (
    "021f928e21e2360186217862b4310be90fe0f705c1bfbf43b39a8b41e644e40c"
)
DEFAULT_CONTRACT = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "v4_11_input_blind_aligned_stack_contract.md"
)

INPUT_BLIND_QUERY_COLUMNS = [
    "query_id",
    "crm_record_id",
    "crm_name",
    "crm_address",
    "crm_postcode",
    "crm_city",
    "crm_insee",
    "crm_name_norm",
    "crm_address_norm",
    "crm_city_norm",
]
QUERY_AUDIT_COLUMNS = [
    "query_id",
    "input_siret_state",
    "source_segment",
]
SOURCE_QUERY_READ_COLUMNS = [
    *INPUT_BLIND_QUERY_COLUMNS,
    *[
        column
        for column in QUERY_AUDIT_COLUMNS
        if column not in INPUT_BLIND_QUERY_COLUMNS
    ],
]
LABEL_COLUMNS = [
    "query_id",
    "label_kind",
    "ground_truth_siret",
    "ground_truth_siren",
]
RANKER_C_FEATURE_ORDER = [
    "has_any_name",
    "name_count",
    "name_jaro_max",
    "name_jaro_second",
    "name_jaro_gap",
    "name_levenshtein_max",
    "name_token_overlap_max",
    "idf_name",
    "numeric_token_match",
    "name_first_word_match_max",
    "name_contains_crm_max",
    "name_crm_contains_cand_max",
    "acronym_match_max",
    "name_sim_max_etab",
    "name_sim_max_ul",
    "name_sim_max_sigle",
    "name_sim_max_pm_dirigeant",
    "type_of_max_name",
    "is_ul_name_max",
    "is_sigle_max",
    "name_length_max",
    "has_person_name",
    "person_name_jaro_max",
    "name_city_overlap_max",
    "name_is_city_like_max",
    "addr_jaro",
    "addr_levenshtein",
    "postcode_match",
    "city_match",
    "street_number_diff",
    "addr_token_overlap",
    "address_density",
    "street_name_jaro",
    "name_addr_consistency",
    "legal_form_category",
    "is_siege",
    "is_association",
    "alias_match",
    "token_overlap_ul",
    "ul_vs_pm_indicator",
    "is_crm_school",
    "geo_exact_match",
    "name_norm_exact",
    "street_number_match",
    "retrieval_rank_recip",
]
CANDIDATE_METADATA_COLUMNS = [
    "query_id",
    "candidate_siret",
    "candidate_siren",
    "candidate_state",
    "is_ground_truth",
    "retrieval_rank",
    "retrieval_source",
    "retrieval_channel_count",
    "retrieval_agreement",
]
CANDIDATE_ROLE_SOURCE_COLUMNS = [
    "enseigne1",
    "enseigne2",
    "enseigne3",
    "denomination_usuelle",
    "activity_code",
]
CANDIDATE_COLUMNS = (
    CANDIDATE_METADATA_COLUMNS
    + CANDIDATE_ROLE_SOURCE_COLUMNS
    + RANKER_C_FEATURE_ORDER
)
RAW_CANDIDATE_SOURCE_COLUMNS = [
    "denomination",
    "enseigne1",
    "enseigne2",
    "enseigne3",
    "etablissementSiege",
    "is_siege",
    "numeroVoie",
    "typeVoie",
    "libelleVoie",
    "complementAdresse",
    "postcode",
    "city",
    "insee",
    "cj_ul",
    "etat_admin",
    "sigle_ul",
    "denomination_ul",
    "denomination_usuelle_ul",
    "nom_ul",
    "prenom_usuel_ul",
    "pm_dirigeant_names",
    "_xgb_addr_density_insee",
    "_xgb_addr_density_cp",
]
RAW_CANDIDATE_COLUMNS = [
    "query_id",
    "candidate_siret",
    "candidate_siren",
    "rrf_score",
    "retrieval_source",
    "retrieval_channel_count",
    "idf_payload",
    "idf_default",
    *RAW_CANDIDATE_SOURCE_COLUMNS,
]
HYDRATED_SNAPSHOT_COLUMNS = [
    "snapshot_candidate_state",
    "snapshot_enseigne1",
    "snapshot_enseigne2",
    "snapshot_enseigne3",
    "snapshot_denomination_usuelle",
    "snapshot_activity_code",
]
MAX_IN_MEMORY_TFIDF_PARTITIONS = 20
FORBIDDEN_PREDICTION_COLUMNS = {
    "input_siret",
    "input_siren",
    "input_siret_state",
    "ground_truth_siret",
    "ground_truth_siren",
    "is_ground_truth",
    "candidate_from_input_siret",
    "candidate_from_input_siren",
    "candidate_from_closed_alias",
    "input_siret_exact_match",
    "input_siren_exact_match",
}

if len(RANKER_C_FEATURE_ORDER) != 45:  # pragma: no cover - import-time guard
    raise RuntimeError("V4.11 ranker C requires exactly 45 features")
if FORBIDDEN_PREDICTION_COLUMNS & set(RANKER_C_FEATURE_ORDER):
    raise RuntimeError("V4.11 ranker C contains an input/label leakage feature")


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def feature_order_sha256(features: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(
            list(features),
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def read_projected_source_queries(
    path: Path,
    *,
    reader: Any = pd.read_parquet,
) -> pd.DataFrame:
    """Physically read only prediction-safe and audit-only query columns."""

    return reader(path, columns=SOURCE_QUERY_READ_COLUMNS)


def read_sha256_pinned_parquet(
    path: Path,
    *,
    expected_sha256: str,
    reader: Any = pd.read_parquet,
) -> pd.DataFrame:
    """Refuse an immutable Parquet input before invoking its deserializer."""

    observed_sha256 = file_sha256(Path(path))
    if observed_sha256 != expected_sha256:
        raise ValueError(
            f"Pinned Parquet hash mismatch: {Path(path).name}"
        )
    return reader(path)


def tfidf_cache_namespace(
    *,
    sparse_config_hash: str,
    tfidf_artifact_hash: str,
    partitions_signature: str,
) -> str:
    payload = {
        "policy_version": TFIDF_CACHE_POLICY_VERSION,
        "sparse_config_hash": sparse_config_hash,
        "tfidf_artifact_hash": tfidf_artifact_hash,
        "partitions_signature": partitions_signature,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def verified_tfidf_cache_dir(
    *,
    cache_root: Path,
    namespace: str,
) -> Path:
    root = Path(cache_root).resolve()
    directory = (root / "v411_verified" / namespace).resolve()
    if not directory.is_relative_to(root):
        raise ValueError("V4.11 TF-IDF cache escaped its controlled root")
    return directory


def input_blind_retrieval_config() -> V41RetrievalConfig:
    """Reuse the V4.2 sparse configuration without enabling identifier branches."""

    return V41RetrievalConfig(
        variant=V41RetrievalVariant.A_SPARSE_ACTIVE,
        max_candidates=CANDIDATE_CEILING,
    )


def input_blind_retrieval_signature(config: V41RetrievalConfig) -> str:
    payload = {
        "policy_version": RETRIEVAL_POLICY_VERSION,
        "candidate_ceiling": CANDIDATE_CEILING,
        "allowed_crm_fields": INPUT_BLIND_QUERY_COLUMNS[2:7],
        "source_sparse_config": config.sparse_config().to_dict(),
        "forbidden_channels": [
            "input_siret_active",
            "input_siren_active_sites",
            "closed_alias_name",
            "closed_alias_address",
            "siren_head",
        ],
        "truth_argument": None,
        "tie_break": ["rrf_score_desc", "candidate_siret_asc"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _candidate_schema() -> pa.Schema:
    string_columns = {
        "query_id",
        "candidate_siret",
        "candidate_siren",
        "candidate_state",
        "retrieval_source",
        *CANDIDATE_ROLE_SOURCE_COLUMNS,
    }
    integer_columns = {
        "is_ground_truth",
        "retrieval_rank",
        "retrieval_channel_count",
        "retrieval_agreement",
    }
    return pa.schema(
        [
            pa.field(
                column,
                (
                    pa.string()
                    if column in string_columns
                    else pa.int32() if column in integer_columns else pa.float32()
                ),
            )
            for column in CANDIDATE_COLUMNS
        ]
    )


class CandidateWriter:
    def __init__(self, path: Path) -> None:
        self.schema = _candidate_schema()
        self._writer = pq.ParquetWriter(path, self.schema)
        self.count = 0

    def write(self, rows: Sequence[Mapping[str, Any]]) -> None:
        if not rows:
            return
        frame = pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)
        self._writer.write_table(
            pa.Table.from_pandas(frame, schema=self.schema, preserve_index=False)
        )
        self.count += len(frame)

    def close(self) -> None:
        self._writer.close()


def _raw_candidate_schema() -> pa.Schema:
    bool_columns = {"etablissementSiege", "is_siege"}
    float_columns = {
        "rrf_score",
        "idf_default",
        "_xgb_addr_density_insee",
        "_xgb_addr_density_cp",
    }
    integer_columns = {"retrieval_channel_count"}
    return pa.schema(
        [
            pa.field(
                column,
                (
                    pa.bool_()
                    if column in bool_columns
                    else (
                        pa.float64()
                        if column in float_columns
                        else pa.int32() if column in integer_columns else pa.string()
                    )
                ),
            )
            for column in RAW_CANDIDATE_COLUMNS
        ]
    )


class RawCandidateWriter:
    """Stream the label-free sparse result before authoritative hydration."""

    def __init__(self, path: Path) -> None:
        self.schema = _raw_candidate_schema()
        self._writer = pq.ParquetWriter(path, self.schema)
        self.count = 0

    def write(self, rows: Sequence[Mapping[str, Any]]) -> None:
        if not rows:
            return
        frame = pd.DataFrame(rows, columns=RAW_CANDIDATE_COLUMNS)
        self._writer.write_table(
            pa.Table.from_pandas(frame, schema=self.schema, preserve_index=False)
        )
        self.count += len(frame)

    def close(self) -> None:
        self._writer.close()


def load_source_population(
    source_dataset_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Load the pinned V4.6 population and immediately split prediction/audit data."""

    source_dataset_dir = Path(source_dataset_dir).resolve()
    manifest_path = source_dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "sireto-v4.6-aligned-ranker-dataset-1":
        raise ValueError("Unsupported V4.6 source dataset schema")
    if manifest.get("build_id") != "301b24f47820f992":
        raise ValueError("Unexpected V4.6 source dataset build")
    if (manifest.get("invariants") or {}).get("positive_injection") is not False:
        raise ValueError("V4.6 source permits positive injection")
    for filename in (
        "queries.parquet",
        "labels.parquet",
        "candidates_v42b.parquet",
        "split_assignments.parquet",
    ):
        path = source_dataset_dir / filename
        expected = EXPECTED_INPUT_HASHES[filename]
        declared = ((manifest.get("outputs") or {}).get(filename) or {}).get("sha256")
        if declared != expected:
            raise ValueError(f"V4.6 source hash mismatch: {filename}")
        # Labels and comparison candidates remain physically unopened until
        # the final label-free V4.11 pools have been closed.
        if filename in {"queries.parquet", "split_assignments.parquet"}:
            if file_sha256(path) != expected:
                raise ValueError(f"V4.6 source hash mismatch: {filename}")

    raw_queries = read_projected_source_queries(
        source_dataset_dir / "queries.parquet"
    )
    assert_authorized_canonical_table(raw_queries, name="source_queries")
    required = set(INPUT_BLIND_QUERY_COLUMNS + QUERY_AUDIT_COLUMNS)
    missing = required - set(raw_queries.columns)
    if missing:
        raise ValueError(f"V4.6 source queries missing: {sorted(missing)}")
    if len(raw_queries) != EXPECTED_QUERY_COUNT:
        raise ValueError("V4.11 requires exactly 7,003 source queries")
    if raw_queries["query_id"].astype(str).duplicated().any():
        raise ValueError("Source query_id must be unique")
    queries = raw_queries[INPUT_BLIND_QUERY_COLUMNS].copy()
    audit = raw_queries[QUERY_AUDIT_COLUMNS].copy()
    if FORBIDDEN_PREDICTION_COLUMNS & set(queries.columns):
        raise ValueError("Input-blind query projection contains forbidden fields")
    return manifest, queries, audit


def validate_assignments(
    queries: pd.DataFrame,
    assignments: pd.DataFrame,
) -> None:
    required = {"query_id", "siren_component_id", "split", "oof_fold"}
    missing = required - set(assignments.columns)
    if missing:
        raise ValueError(f"Split assignments missing: {sorted(missing)}")
    if assignments["query_id"].astype(str).duplicated().any():
        raise ValueError("Split assignment query_id must be unique")
    if set(assignments["query_id"].astype(str)) != set(
        queries["query_id"].astype(str)
    ):
        raise ValueError("Split assignments differ from the query population")
    if not set(assignments["split"].astype(str)).issubset({"fit", "dev"}):
        raise ValueError("V4.11 assignments contain an unauthorized split")
    if assignments["oof_fold"].isna().any():
        raise ValueError("Every frozen V4.11 assignment must carry an OOF fold")
    folds = set(assignments["oof_fold"].astype(int))
    if folds != {0, 1, 2, 3, 4}:
        raise ValueError(f"V4.11 requires five frozen folds, got {sorted(folds)}")
    component_fold_counts = assignments.groupby("siren_component_id")[
        "oof_fold"
    ].nunique()
    if not component_fold_counts.eq(1).all():
        raise ValueError("A SIREN component spans multiple frozen OOF folds")


def _finite_float(value: Any) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return 0.0
    return output if np.isfinite(output) else 0.0


def _trim_tfidf_cache(
    cache: dict[tuple[str, str], tuple],
    *,
    ceiling: int = MAX_IN_MEMORY_TFIDF_PARTITIONS,
) -> None:
    """Apply the same deterministic FIFO ceiling as V4.1."""

    if ceiling <= 0:
        raise ValueError("TF-IDF cache ceiling must be positive")
    while len(cache) > ceiling:
        cache.pop(next(iter(cache)), None)


def _idf_payload(
    crm_pre: Mapping[str, Any],
    idf_map: Mapping[str, float],
    default_idf: float,
) -> str:
    """Persist only IDFs that can participate in an overlap with this CRM."""

    tokens = sorted(set(str(crm_pre.get("crm_name") or "").split()))
    payload = {
        token: float(idf_map.get(token, default_idf))
        for token in tokens
    }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def retrieve_raw_input_blind_query(
    *,
    query: Mapping[str, Any],
    partitioned_store: Any,
    config: V41RetrievalConfig,
    tfidf_cache: dict[tuple[str, str], tuple],
    persistent_cache: Any,
    sparse_pool_builder: Any = build_candidate_pool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Retrieve a label-free raw pool; no snapshot or truth is accepted."""

    unexpected = FORBIDDEN_PREDICTION_COLUMNS & set(query)
    if unexpected:
        raise ValueError(
            f"Input-blind retrieval received forbidden fields: {sorted(unexpected)}"
        )
    query_id = str(query["query_id"])
    crm_row = {
        "query_id": query_id,
        "crm_id": query_id,
        "crm_name": str(query.get("crm_name") or ""),
        "crm_address": str(query.get("crm_address") or ""),
        "crm_city": str(query.get("crm_city") or ""),
        "postcode": str(query.get("crm_postcode") or ""),
        "insee": str(query.get("crm_insee") or ""),
    }
    crm_pre = preprocess_crm_row(crm_row)
    result = sparse_pool_builder(
        partitioned_store,
        crm_row,
        crm_pre,
        config.sparse_config(),
        tfidf_cache,
        None,
        persistent_cache=persistent_cache,
    )
    if result.gt_was_injected:
        raise ValueError("Positive injection is forbidden in V4.11")

    candidate_by_siret: dict[str, dict[str, Any]] = {}
    for raw_candidate in result.candidates:
        candidate = dict(raw_candidate)
        siret = str(candidate.get("siret") or "")
        if siret:
            candidate_by_siret.setdefault(siret, candidate)
    if len(candidate_by_siret) > CANDIDATE_CEILING:
        raise ValueError("Raw V4.11 sparse pool exceeded 100 unique SIRETs")
    idf_payload = _idf_payload(
        crm_pre,
        result.idf_map,
        result.default_idf,
    )
    rows = []
    for siret, candidate in candidate_by_siret.items():
        rows.append(
            {
                "query_id": query_id,
                "candidate_siret": siret,
                "candidate_siren": str(candidate.get("siren") or siret[:9]),
                "rrf_score": _finite_float(candidate.get("rrf_score")),
                "retrieval_source": str(
                    candidate.get("retrieval_source") or "v4.11-sparse"
                ),
                "retrieval_channel_count": int(
                    candidate.get("retrieval_channel_count") or 0
                ),
                "idf_payload": idf_payload,
                "idf_default": _finite_float(result.default_idf),
                **{
                    column: candidate.get(column)
                    for column in RAW_CANDIDATE_SOURCE_COLUMNS
                },
            }
        )
    _trim_tfidf_cache(tfidf_cache)
    return rows, {
        "query_id": query_id,
        "raw_sparse_count": len(result.candidates),
    }


def _finalize_query_candidates(
    *,
    query: Mapping[str, Any],
    hydrated_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the exact final pool after authoritative bulk hydration."""

    query_id = str(query["query_id"])
    crm_row = {
        "query_id": query_id,
        "crm_id": query_id,
        "crm_name": str(query.get("crm_name") or ""),
        "crm_address": str(query.get("crm_address") or ""),
        "crm_city": str(query.get("crm_city") or ""),
        "postcode": str(query.get("crm_postcode") or ""),
        "insee": str(query.get("crm_insee") or ""),
    }
    crm_pre = preprocess_crm_row(crm_row)
    active_candidates: list[dict[str, Any]] = []
    for raw_row in hydrated_rows:
        siret = str(raw_row.get("candidate_siret") or "")
        state = str(raw_row.get("snapshot_candidate_state") or "").upper()
        if state != "A":
            continue
        candidate = {
            column: raw_row.get(column)
            for column in RAW_CANDIDATE_SOURCE_COLUMNS
            if raw_row.get(column) is not None
        }
        candidate["siret"] = siret
        candidate["siren"] = str(raw_row.get("candidate_siren") or siret[:9])
        candidate["rrf_score"] = _finite_float(raw_row.get("rrf_score"))
        candidate["retrieval_source"] = str(
            raw_row.get("retrieval_source") or "v4.11-sparse"
        )
        candidate["retrieval_channel_count"] = int(
            raw_row.get("retrieval_channel_count") or 0
        )
        candidate["etat_admin"] = "A"
        candidate["enseigne1"] = raw_row.get("snapshot_enseigne1")
        candidate["enseigne2"] = raw_row.get("snapshot_enseigne2")
        candidate["enseigne3"] = raw_row.get("snapshot_enseigne3")
        candidate["denomination_usuelle"] = raw_row.get(
            "snapshot_denomination_usuelle"
        )
        candidate["activity_code"] = raw_row.get("snapshot_activity_code")
        active_candidates.append(candidate)

    active_candidates.sort(
        key=lambda candidate: (
            -_finite_float(candidate.get("rrf_score")),
            str(candidate.get("siret") or ""),
        )
    )
    active_candidates = active_candidates[:CANDIDATE_CEILING]
    first = hydrated_rows[0] if hydrated_rows else {}
    idf_payload = json.loads(str(first.get("idf_payload") or "{}"))
    idf_default = _finite_float(first.get("idf_default"))
    set_global_name_idf_map(idf_payload, idf_default)
    legacy_rows = make_feature_rows_from_preprocessed(
        crm_pre,
        active_candidates,
        include_semantic=False,
    )
    rows: list[dict[str, Any]] = []
    for rank, (candidate, legacy) in enumerate(
        zip(active_candidates, legacy_rows, strict=True),
        start=1,
    ):
        siret = str(candidate["siret"])
        channel_count = int(candidate.get("retrieval_channel_count") or 0)
        feature_values = {
            feature: _finite_float(legacy.get(feature, 0.0))
            for feature in RANKER_C_FEATURE_ORDER[:-1]
        }
        feature_values["retrieval_rank_recip"] = 1.0 / rank
        rows.append(
            {
                "query_id": query_id,
                "candidate_siret": siret,
                "candidate_siren": str(candidate.get("siren") or siret[:9]),
                "candidate_state": "A",
                "is_ground_truth": 0,
                "retrieval_rank": rank,
                "retrieval_source": str(
                    candidate.get("retrieval_source") or "v4.11-sparse"
                ),
                "retrieval_channel_count": channel_count,
                "retrieval_agreement": int(channel_count >= 2),
                **{
                    column: candidate.get(column)
                    for column in CANDIDATE_ROLE_SOURCE_COLUMNS
                },
                **feature_values,
            }
        )
    if len(rows) > CANDIDATE_CEILING:
        raise AssertionError("V4.11 candidate ceiling violated")
    return rows, {
        "query_id": query_id,
        "candidate_count": len(rows),
        "raw_sparse_count": len(hydrated_rows),
        "authoritative_non_active_removed": len(hydrated_rows) - len(rows),
    }


def _bulk_snapshot_details_for_testing(
    current_state_store: Any,
    sirets: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Narrow fixture adapter; production uses ``bulk_hydrate_snapshot`` only."""

    custom = getattr(current_state_store, "get_candidate_scene_details", None)
    if not callable(custom):
        raise TypeError(
            "Per-query snapshot access is disabled; use bulk_hydrate_snapshot"
        )
    return custom(sirets)


def retrieve_input_blind_query(
    *,
    query: Mapping[str, Any],
    partitioned_store: Any,
    current_state_store: Any,
    config: V41RetrievalConfig,
    tfidf_cache: dict[tuple[str, str], tuple],
    persistent_cache: Any,
    sparse_pool_builder: Any = build_candidate_pool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fixture-compatible paired path using already indexed in-memory details."""

    raw_rows, raw_diagnostic = retrieve_raw_input_blind_query(
        query=query,
        partitioned_store=partitioned_store,
        config=config,
        tfidf_cache=tfidf_cache,
        persistent_cache=persistent_cache,
        sparse_pool_builder=sparse_pool_builder,
    )
    details = _bulk_snapshot_details_for_testing(
        current_state_store,
        [str(row["candidate_siret"]) for row in raw_rows],
    )
    hydrated_rows = []
    for row in raw_rows:
        detail = details.get(str(row["candidate_siret"]), {})
        hydrated_rows.append(
            {
                **row,
                "snapshot_candidate_state": detail.get("candidate_state"),
                "snapshot_enseigne1": detail.get("enseigne1"),
                "snapshot_enseigne2": detail.get("enseigne2"),
                "snapshot_enseigne3": detail.get("enseigne3"),
                "snapshot_denomination_usuelle": detail.get(
                    "denomination_usuelle"
                ),
                "snapshot_activity_code": detail.get("activity_code"),
            }
        )
    rows, diagnostic = _finalize_query_candidates(
        query=query,
        hydrated_rows=hydrated_rows,
    )
    diagnostic["raw_sparse_count"] = raw_diagnostic["raw_sparse_count"]
    return rows, diagnostic


def _sql_path(path: Path) -> str:
    return str(Path(path).resolve()).replace("'", "''")


def bulk_hydrate_snapshot(
    *,
    connection: Any,
    raw_candidates_path: Path,
    state_snapshot_path: Path,
    output_path: Path,
) -> dict[str, int]:
    """Scan the SIRENE snapshot once and materialize only requested SIRETs."""

    raw_sql = _sql_path(raw_candidates_path)
    snapshot_sql = _sql_path(state_snapshot_path)
    output_sql = _sql_path(output_path)
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE v411_requested_sirets AS
        SELECT DISTINCT candidate_siret AS siret
        FROM read_parquet('{raw_sql}')
        """
    )
    requested_count = int(
        connection.execute(
            "SELECT count(*) FROM v411_requested_sirets"
        ).fetchone()[0]
    )
    # This is the single complete snapshot scan in the V4.11 builder.
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE v411_snapshot_details AS
        SELECT DISTINCT
            CAST(snapshot.siret AS VARCHAR) AS siret,
            upper(trim(CAST(
                snapshot.etatAdministratifEtablissement AS VARCHAR
            ))) AS snapshot_candidate_state,
            CAST(snapshot.enseigne1Etablissement AS VARCHAR)
                AS snapshot_enseigne1,
            CAST(snapshot.enseigne2Etablissement AS VARCHAR)
                AS snapshot_enseigne2,
            CAST(snapshot.enseigne3Etablissement AS VARCHAR)
                AS snapshot_enseigne3,
            CAST(snapshot.denominationUsuelleEtablissement AS VARCHAR)
                AS snapshot_denomination_usuelle,
            CAST(snapshot.activitePrincipaleEtablissement AS VARCHAR)
                AS snapshot_activity_code
        FROM read_parquet('{snapshot_sql}') AS snapshot
        INNER JOIN v411_requested_sirets AS requested
            ON CAST(snapshot.siret AS VARCHAR) = requested.siret
        """
    )
    conflicts = connection.execute(
        """
        SELECT siret
        FROM v411_snapshot_details
        GROUP BY siret
        HAVING count(*) > 1
        ORDER BY siret
        LIMIT 10
        """
    ).fetchall()
    if conflicts:
        raise ValueError(
            "STOP_DATASET_INTEGRITY: conflicting snapshot details for "
            f"{[row[0] for row in conflicts]}"
        )
    matched_count = int(
        connection.execute(
            "SELECT count(*) FROM v411_snapshot_details"
        ).fetchone()[0]
    )
    connection.execute(
        f"""
        COPY (
            SELECT
                raw.*,
                details.snapshot_candidate_state,
                details.snapshot_enseigne1,
                details.snapshot_enseigne2,
                details.snapshot_enseigne3,
                details.snapshot_denomination_usuelle,
                details.snapshot_activity_code
            FROM read_parquet('{raw_sql}') AS raw
            LEFT JOIN v411_snapshot_details AS details
                ON raw.candidate_siret = details.siret
            ORDER BY raw.query_id, raw.candidate_siret
        ) TO '{output_sql}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    return {
        "requested_unique_siret_count": requested_count,
        "snapshot_matched_unique_siret_count": matched_count,
        "snapshot_full_scan_count": 1,
    }


def finalize_hydrated_pools(
    *,
    hydrated_path: Path,
    output_path: Path,
    queries: pd.DataFrame,
) -> tuple[int, list[dict[str, Any]]]:
    """Stream hydrated rows query-by-query into final label-free candidate pools."""

    query_by_id = {
        str(row["query_id"]): row
        for row in queries.to_dict("records")
    }
    writer = CandidateWriter(output_path)
    count = 0
    diagnostics: list[dict[str, Any]] = []
    current_query_id: str | None = None
    current_rows: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal count, current_rows
        if current_query_id is None:
            return
        query = query_by_id.get(current_query_id)
        if query is None:
            raise ValueError(
                f"Hydrated candidates contain unknown query {current_query_id}"
            )
        rows, diagnostic = _finalize_query_candidates(
            query=query,
            hydrated_rows=current_rows,
        )
        writer.write(rows)
        count += len(rows)
        diagnostics.append(diagnostic)
        current_rows = []

    try:
        parquet = pq.ParquetFile(hydrated_path)
        for batch in parquet.iter_batches(batch_size=25_000):
            for row in batch.to_pylist():
                query_id = str(row["query_id"])
                if current_query_id is not None and query_id != current_query_id:
                    flush()
                current_query_id = query_id
                current_rows.append(row)
        flush()
    finally:
        writer.close()
    return count, diagnostics


def load_labels_after_final_pool_closed(
    *,
    final_unlabelled_path: Path,
    labels_path: Path,
    expected_sha256: str,
    loader: Any = pd.read_parquet,
) -> pd.DataFrame:
    """Enforce the physical phase boundary before any label is opened."""

    parquet = pq.ParquetFile(final_unlabelled_path)
    if parquet.metadata is None:
        raise ValueError("Final label-free candidate pool is not closed")
    return read_sha256_pinned_parquet(
        labels_path,
        expected_sha256=expected_sha256,
        reader=loader,
    )


def label_closed_candidate_file(
    *,
    unlabelled_path: Path,
    output_path: Path,
    labels: pd.DataFrame,
) -> int:
    """Join exact truth after every retrieval pool has been closed."""

    exact = labels.loc[
        labels["label_kind"].astype(str).eq("MATCH_EXACT"),
        ["query_id", "ground_truth_siret"],
    ]
    truth = dict(
        zip(
            exact["query_id"].astype(str),
            exact["ground_truth_siret"].fillna("").astype(str),
            strict=True,
        )
    )
    source = pq.ParquetFile(unlabelled_path)
    writer = CandidateWriter(output_path)
    count = 0
    try:
        for batch in source.iter_batches(batch_size=25_000):
            frame = batch.to_pandas()
            frame["is_ground_truth"] = [
                int(str(siret) == truth.get(str(query_id), ""))
                for query_id, siret in zip(
                    frame["query_id"],
                    frame["candidate_siret"],
                    strict=True,
                )
            ]
            writer.write(frame.to_dict("records"))
            count += len(frame)
    finally:
        writer.close()
    return count


def ordered_pool_content_sha256(candidates: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    ordered = candidates.sort_values(
        ["query_id", "retrieval_rank", "candidate_siret"],
        kind="stable",
    )
    for row in ordered[
        ["query_id", "retrieval_rank", "candidate_siret"]
    ].itertuples(index=False):
        digest.update(
            f"{row.query_id}\0{int(row.retrieval_rank)}\0"
            f"{row.candidate_siret}\n".encode()
        )
    return digest.hexdigest()


def candidate_content_sha256(candidates: pd.DataFrame) -> str:
    if list(candidates.columns) != CANDIDATE_COLUMNS:
        raise ValueError("Candidate columns are not in the V4.11 canonical order")
    ordered = candidates.sort_values(
        ["query_id", "retrieval_rank", "candidate_siret"],
        kind="stable",
    ).reset_index(drop=True)
    hashes = pd.util.hash_pandas_object(
        ordered[CANDIDATE_COLUMNS],
        index=False,
        categorize=False,
    ).to_numpy()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(CANDIDATE_COLUMNS, separators=(",", ":")).encode()
    )
    digest.update(str(ordered.dtypes.astype(str).tolist()).encode())
    digest.update(hashes.tobytes())
    return digest.hexdigest()


def _pool_metrics(
    *,
    query_ids: set[str],
    exact_labels: pd.DataFrame,
    candidates: pd.DataFrame,
) -> dict[str, Any]:
    relevant = exact_labels.loc[
        exact_labels["query_id"].astype(str).isin(query_ids)
    ].copy()
    ranked = candidates.loc[
        candidates["query_id"].astype(str).isin(query_ids)
    ]
    truth_by_query = dict(
        zip(
            relevant["query_id"].astype(str),
            relevant["ground_truth_siret"].astype(str),
            strict=True,
        )
    )
    truth_siren_by_query = {
        query_id: siret[:9] for query_id, siret in truth_by_query.items()
    }
    by_query = {
        str(query_id): group.sort_values("retrieval_rank", kind="stable")
        for query_id, group in ranked.groupby("query_id", sort=False)
    }
    output: dict[str, Any] = {"exact_total": len(relevant)}
    for cutoff in (1, 10, 50, 100):
        hits = sum(
            truth_by_query[query_id]
            in set(
                by_query.get(query_id, pd.DataFrame())
                .head(cutoff)
                .get("candidate_siret", pd.Series(dtype=str))
                .astype(str)
            )
            for query_id in truth_by_query
        )
        output[f"recall_siret_at_{cutoff}"] = {
            "successes": hits,
            "total": len(relevant),
            "rate": hits / len(relevant) if len(relevant) else 0.0,
        }
    siren_hits = sum(
        truth_siren_by_query[query_id]
        in set(
            by_query.get(query_id, pd.DataFrame())
            .head(100)
            .get("candidate_siren", pd.Series(dtype=str))
            .astype(str)
        )
        for query_id in truth_by_query
    )
    output["recall_siren_at_100"] = {
        "successes": siren_hits,
        "total": len(relevant),
        "rate": siren_hits / len(relevant) if len(relevant) else 0.0,
    }
    output["miss_query_ids_at_100"] = sorted(
        query_id
        for query_id, truth in truth_by_query.items()
        if truth
        not in set(
            by_query.get(query_id, pd.DataFrame())
            .head(100)
            .get("candidate_siret", pd.Series(dtype=str))
            .astype(str)
        )
    )
    return output


def _comparison_metrics(
    *,
    labels: pd.DataFrame,
    assignments: pd.DataFrame,
    v42b_candidates: pd.DataFrame,
) -> dict[str, Any]:
    full = v42b_candidates.copy()
    sparse = full.loc[full["candidate_from_sparse"].astype(float).eq(1.0)].copy()
    exact = labels.loc[labels["label_kind"].eq("MATCH_EXACT")]
    result: dict[str, Any] = {}
    for split in ("fit", "dev"):
        ids = set(
            assignments.loc[assignments["split"].eq(split), "query_id"].astype(str)
        )
        result[split] = {
            "v42b_full": _pool_metrics(
                query_ids=ids,
                exact_labels=exact,
                candidates=full,
            ),
            "v42b_sparse_subset_not_reconstruction": _pool_metrics(
                query_ids=ids,
                exact_labels=exact,
                candidates=sparse,
            ),
        }
    return result


def compute_integrity(
    *,
    queries: pd.DataFrame,
    labels: pd.DataFrame,
    assignments: pd.DataFrame,
    candidates: pd.DataFrame,
) -> dict[str, Any]:
    if list(candidates.columns) != CANDIDATE_COLUMNS:
        raise ValueError("V4.11 candidate schema or feature order changed")
    if FORBIDDEN_PREDICTION_COLUMNS & set(RANKER_C_FEATURE_ORDER):
        raise ValueError("Ranker C feature order contains forbidden input signals")
    if not set(candidates["query_id"].astype(str)).issubset(
        set(queries["query_id"].astype(str))
    ):
        raise ValueError("Candidates contain an unknown query")
    if candidates["candidate_state"].astype(str).ne("A").any():
        raise ValueError("V4.11 contains a non-active candidate")
    if candidates.duplicated(["query_id", "candidate_siret"]).any():
        raise ValueError("V4.11 contains duplicate SIRETs within a pool")
    counts = candidates.groupby("query_id").size()
    if len(counts) and int(counts.max()) > CANDIDATE_CEILING:
        raise ValueError("V4.11 candidate ceiling exceeded")
    ranks_valid = candidates.groupby("query_id")["retrieval_rank"].apply(
        lambda values: list(values.astype(int))
        == list(range(1, len(values) + 1))
    )
    if not ranks_valid.all():
        raise ValueError("V4.11 candidate ranks are not contiguous")
    if not np.isfinite(
        candidates[RANKER_C_FEATURE_ORDER].to_numpy(dtype=float)
    ).all():
        raise ValueError("V4.11 candidate feature matrix contains non-finite values")
    expected_recip = 1.0 / candidates["retrieval_rank"].astype(float)
    if not np.allclose(
        candidates["retrieval_rank_recip"].astype(float),
        expected_recip,
        rtol=0.0,
        atol=1e-7,
    ):
        raise ValueError("retrieval_rank_recip is inconsistent with final pool rank")

    truth = labels.set_index("query_id")["ground_truth_siret"]
    expected_target = [
        int(
            str(candidate_siret)
            == (
                ""
                if pd.isna(truth.get(str(query_id)))
                else str(truth.get(str(query_id)))
            )
        )
        for query_id, candidate_siret in zip(
            candidates["query_id"],
            candidates["candidate_siret"],
            strict=True,
        )
    ]
    if expected_target != candidates["is_ground_truth"].astype(int).tolist():
        raise ValueError("V4.11 candidate targets disagree with frozen labels")
    exact = labels.loc[labels["label_kind"].eq("MATCH_EXACT")]
    retrieval: dict[str, Any] = {}
    for split in ("fit", "dev"):
        ids = set(
            assignments.loc[assignments["split"].eq(split), "query_id"].astype(str)
        )
        retrieval[split] = _pool_metrics(
            query_ids=ids,
            exact_labels=exact,
            candidates=candidates,
        )
    gate_passed = all(
        retrieval[split]["recall_siret_at_100"]["rate"] >= 0.99
        for split in ("fit", "dev")
    )
    all_pool_sizes = (
        queries[["query_id"]]
        .merge(
            counts.rename("candidate_count"),
            left_on="query_id",
            right_index=True,
            how="left",
        )["candidate_count"]
        .fillna(0)
        .astype(int)
    )
    return {
        "query_count": len(queries),
        "label_count": len(labels),
        "assignment_count": len(assignments),
        "candidate_count": len(candidates),
        "pool_size": {
            "min": int(all_pool_sizes.min()),
            "median": float(all_pool_sizes.median()),
            "mean": float(all_pool_sizes.mean()),
            "max": int(all_pool_sizes.max()),
        },
        "zero_candidate_query_count": int(all_pool_sizes.eq(0).sum()),
        "closed_candidate_count": 0,
        "duplicate_candidate_count": 0,
        "positive_injection": False,
        "ranker_c_feature_count": len(RANKER_C_FEATURE_ORDER),
        "ranker_c_feature_order_sha256": feature_order_sha256(
            RANKER_C_FEATURE_ORDER
        ),
        "retrieval": retrieval,
        "retrieval_gate_passed": gate_passed,
        "verdict": (
            "GO_TRAIN_INPUT_BLIND_RANKER"
            if gate_passed
            else "PIVOT_INPUT_BLIND_RETRIEVAL"
        ),
        "ordered_pool_content_sha256": ordered_pool_content_sha256(candidates),
        "candidate_content_sha256": candidate_content_sha256(candidates),
    }


def _dependency_versions() -> dict[str, str]:
    output = {}
    for package in ("pandas", "pyarrow", "duckdb", "scikit-learn"):
        try:
            output[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            output[package] = "missing"
    return output


def build_artifact(
    *,
    source_dataset_dir: Path,
    partitions_dir: Path,
    global_store_path: Path,
    state_snapshot_path: Path,
    v42_manifest_path: Path,
    cache_dir: Path,
    work_dir: Path,
    output_root: Path,
    contract_path: Path = DEFAULT_CONTRACT,
) -> Path:
    output_root = _external_path(output_root, name="output_root")
    cache_dir = _external_path(cache_dir, name="cache_dir")
    work_dir = _external_path(work_dir, name="work_dir")
    output_root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    source_manifest, queries, query_audit = load_source_population(
        source_dataset_dir
    )
    assignments_path = Path(source_dataset_dir) / "split_assignments.parquet"
    assignments = pd.read_parquet(assignments_path)
    validate_assignments(queries, assignments)

    source_config = V41RetrievalConfig(
        variant=V41RetrievalVariant.B_INPUT_EVIDENCE,
        max_candidates=CANDIDATE_CEILING,
    )
    if source_config.signature() != EXPECTED_V42_SOURCE_CONFIG_SIGNATURE:
        raise ValueError("V4.2 source retrieval configuration changed")
    runtime = validate_v42_runtime(
        manifest_path=v42_manifest_path,
        config=source_config,
        partitions_dir=partitions_dir,
        global_store_path=global_store_path,
        state_snapshot_path=state_snapshot_path,
    )
    config = input_blind_retrieval_config()
    if config.sparse_config().to_dict() != source_config.sparse_config().to_dict():
        raise ValueError("V4.11 sparse configuration drifted from V4.2")
    sparse_config_hash = config.sparse_config().signature().hash
    tfidf_artifact_hash = config.sparse_config().tfidf_artifact_hash()
    cache_namespace = tfidf_cache_namespace(
        sparse_config_hash=sparse_config_hash,
        tfidf_artifact_hash=tfidf_artifact_hash,
        partitions_signature=runtime["partitions_signature"],
    )
    cache_namespace_dir = verified_tfidf_cache_dir(
        cache_root=cache_dir,
        namespace=cache_namespace,
    )

    script_path = Path(__file__).resolve()
    identity = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "seed": SEED,
        "contract_sha256": file_sha256(contract_path),
        "builder_source_sha256": file_sha256(script_path),
        "source_manifest_sha256": file_sha256(
            Path(source_dataset_dir) / "manifest.json"
        ),
        "queries_sha256": EXPECTED_INPUT_HASHES["queries.parquet"],
        "labels_sha256": EXPECTED_INPUT_HASHES["labels.parquet"],
        "v42b_candidates_context_sha256": EXPECTED_INPUT_HASHES[
            "candidates_v42b.parquet"
        ],
        "split_assignments_sha256": EXPECTED_INPUT_HASHES[
            "split_assignments.parquet"
        ],
        "v42_manifest_sha256": runtime["manifest_sha256"],
        "v42_source_config_signature": source_config.signature(),
        "input_blind_retrieval_signature": input_blind_retrieval_signature(config),
        "partitions_signature": runtime["partitions_signature"],
        "tfidf_cache_policy_version": TFIDF_CACHE_POLICY_VERSION,
        "tfidf_cache_namespace": cache_namespace,
        "tfidf_sparse_config_hash": sparse_config_hash,
        "tfidf_artifact_hash": tfidf_artifact_hash,
        "state_snapshot_sha256": runtime["state_snapshot_sha256"],
        "ranker_c_feature_order": RANKER_C_FEATURE_ORDER,
        "ranker_c_feature_order_sha256": feature_order_sha256(
            RANKER_C_FEATURE_ORDER
        ),
        "positive_injection": False,
        "input_identifiers_visible_to_retrieval": False,
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    target = output_root / build_id
    if target.exists():
        raise FileExistsError(f"Immutable V4.11 dataset exists: {target}")
    staging = Path(tempfile.mkdtemp(prefix=f".{build_id}.tmp-", dir=output_root))
    raw_candidates_path = staging / ".raw_sparse_candidates.parquet"
    hydrated_path = staging / ".hydrated_sparse_candidates.parquet"
    unlabelled_path = staging / ".candidates_unlabelled.parquet"
    raw_writer: RawCandidateWriter | None = RawCandidateWriter(
        raw_candidates_path
    )
    raw_diagnostics: list[dict[str, Any]] = []
    final_diagnostics: list[dict[str, Any]] = []
    hydration_diagnostics: dict[str, int] = {}
    persistent_cache_stats: dict[str, int] = {
        "hits": 0,
        "misses": 0,
        "verification_rejections": 0,
    }
    duckdb_temp = work_dir / f"duckdb-{build_id}"
    started = time.perf_counter()
    try:
        persistent_cache = TfidfPersistentCache(
            config_hash=cache_namespace,
            cache_dir=cache_dir / "v411_verified",
            require_verified=True,
        )
        if persistent_cache.cache_dir.resolve() != cache_namespace_dir:
            raise ValueError("V4.11 TF-IDF cache namespace path mismatch")
        partitioned_store = PartitionedCandidateStore(partitions_dir)
        with V41CurrentStateStore(state_snapshot_path) as state_store:
            duckdb_temp.mkdir(exist_ok=False)
            state_store._connection.execute(  # noqa: SLF001
                f"SET temp_directory = '{str(duckdb_temp).replace(chr(39), chr(39) * 2)}'"
            )
            in_memory_cache: dict[tuple[str, str], tuple] = {}
            buffer: list[dict[str, Any]] = []
            for query in queries.sort_values("query_id", kind="stable").to_dict(
                "records"
            ):
                rows, diagnostic = retrieve_raw_input_blind_query(
                    query=query,
                    partitioned_store=partitioned_store,
                    config=config,
                    tfidf_cache=in_memory_cache,
                    persistent_cache=persistent_cache,
                )
                buffer.extend(rows)
                raw_diagnostics.append(diagnostic)
                if len(buffer) >= 10_000:
                    raw_writer.write(buffer)
                    buffer.clear()
            raw_writer.write(buffer)
            raw_writer.close()
            raw_writer = None
            if len(in_memory_cache) > MAX_IN_MEMORY_TFIDF_PARTITIONS:
                raise ValueError("V4.11 TF-IDF cache ceiling was not enforced")
            persistent_cache_stats = persistent_cache.stats()

            hydration_diagnostics = bulk_hydrate_snapshot(
                connection=state_store._connection,  # noqa: SLF001
                raw_candidates_path=raw_candidates_path,
                state_snapshot_path=state_snapshot_path,
                output_path=hydrated_path,
            )
            unlabelled_count, final_diagnostics = finalize_hydrated_pools(
                hydrated_path=hydrated_path,
                output_path=unlabelled_path,
                queries=queries,
            )
            if unlabelled_count > sum(
                item["raw_sparse_count"] for item in raw_diagnostics
            ):
                raise ValueError("Bulk hydration increased candidate cardinality")
        raw_candidates_path.unlink()
        hydrated_path.unlink()

        # Labels open only after the hydrated, filtered and ranked pools are closed.
        labels_path = Path(source_dataset_dir) / "labels.parquet"
        labels = load_labels_after_final_pool_closed(
            final_unlabelled_path=unlabelled_path,
            labels_path=labels_path,
            expected_sha256=EXPECTED_INPUT_HASHES["labels.parquet"],
        )
        if list(labels.columns) != LABEL_COLUMNS:
            raise ValueError("Frozen label schema changed")
        if labels["query_id"].astype(str).duplicated().any():
            raise ValueError("Frozen labels contain duplicate query IDs")
        if set(labels["query_id"].astype(str)) != set(
            queries["query_id"].astype(str)
        ):
            raise ValueError("Frozen labels differ from the query population")
        candidates_path = staging / "candidates_sparse_top100.parquet"
        labelled_count = label_closed_candidate_file(
            unlabelled_path=unlabelled_path,
            output_path=candidates_path,
            labels=labels,
        )
        unlabelled_path.unlink()
        candidates = pd.read_parquet(candidates_path)
        if labelled_count != len(candidates):
            raise ValueError("Candidate labelling cardinality changed")
        integrity = compute_integrity(
            queries=queries,
            labels=labels,
            assignments=assignments,
            candidates=candidates,
        )
        v42b_candidates_path = Path(source_dataset_dir) / "candidates_v42b.parquet"
        v42b_candidates = read_sha256_pinned_parquet(
            v42b_candidates_path,
            expected_sha256=EXPECTED_INPUT_HASHES[
                "candidates_v42b.parquet"
            ],
        )
        comparisons = _comparison_metrics(
            labels=labels,
            assignments=assignments,
            v42b_candidates=v42b_candidates,
        )

        queries.to_parquet(staging / "queries.parquet", index=False)
        query_audit.to_parquet(staging / "query_audit.parquet", index=False)
        shutil.copyfile(labels_path, staging / "labels.parquet")
        shutil.copyfile(
            assignments_path,
            staging / "split_assignments.parquet",
        )
        total_seconds = time.perf_counter() - started
        summary = {
            **integrity,
            "comparison_context": comparisons,
            "retrieval_diagnostics": {
                "raw_sparse_candidate_count": int(
                    sum(item["raw_sparse_count"] for item in raw_diagnostics)
                ),
                "authoritative_non_active_removed": int(
                    sum(item["raw_sparse_count"] for item in raw_diagnostics)
                    - unlabelled_count
                ),
                "max_in_memory_tfidf_partitions": (
                    MAX_IN_MEMORY_TFIDF_PARTITIONS
                ),
                "persistent_tfidf_cache": {
                    "policy_version": TFIDF_CACHE_POLICY_VERSION,
                    "namespace": cache_namespace,
                    "directory": str(cache_namespace_dir),
                    **persistent_cache_stats,
                },
                **hydration_diagnostics,
            },
            "total_seconds": total_seconds,
        }
        summary_path = staging / "integrity_report.json"
        _json_dump(summary_path, summary)
        output_paths = [
            staging / "queries.parquet",
            staging / "query_audit.parquet",
            staging / "labels.parquet",
            candidates_path,
            staging / "split_assignments.parquet",
            summary_path,
        ]
        outputs = {
            path.name: {
                "sha256": file_sha256(path),
                "size_bytes": int(path.stat().st_size),
                "row_count": (
                    int(pq.ParquetFile(path).metadata.num_rows)
                    if path.suffix == ".parquet"
                    else None
                ),
            }
            for path in output_paths
        }
        manifest = {
            **identity,
            "build_identity": identity,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_dataset_build_id": source_manifest["build_id"],
            "inputs": {
                "source_manifest": _input_record(
                    Path(source_dataset_dir) / "manifest.json"
                ),
                "queries": _input_record(
                    Path(source_dataset_dir) / "queries.parquet",
                    row_count=len(queries),
                ),
                "labels": _input_record(labels_path, row_count=len(labels)),
                "v42b_candidates_comparison_only": {
                    **_input_record(
                        v42b_candidates_path,
                        row_count=len(v42b_candidates),
                    ),
                    "opened_after_all_v411_pools_closed": True,
                },
                "split_assignments": _input_record(
                    assignments_path,
                    row_count=len(assignments),
                ),
                "v42_manifest": _input_record(v42_manifest_path),
                "partitions": _directory_record(
                    partitions_dir,
                    runtime_signature=runtime["partitions_signature"],
                    row_count=_physical_parquet_row_count(partitions_dir),
                ),
                "global_store_context_only": _directory_record(
                    global_store_path,
                    runtime_signature=runtime["global_store_signature"],
                    row_count=14_378_332,
                ),
                "state_snapshot": _input_record(
                    state_snapshot_path,
                    row_count=42_322_035,
                ),
                "contract": _input_record(contract_path),
                "builder_source": _input_record(script_path),
            },
            "retrieval": {
                "policy_version": RETRIEVAL_POLICY_VERSION,
                "source_sparse_config": config.sparse_config().to_dict(),
                "input_blind_signature": input_blind_retrieval_signature(config),
                "allowed_input_fields": INPUT_BLIND_QUERY_COLUMNS[2:7],
                "forbidden_identifier_branches": [
                    "input_siret_active",
                    "input_siren_active_sites",
                    "closed_alias_name",
                    "closed_alias_address",
                    "siren_head",
                ],
                "candidate_ceiling": CANDIDATE_CEILING,
                "tie_break": ["rrf_score_desc", "candidate_siret_asc"],
                "tfidf_cache": {
                    "policy_version": TFIDF_CACHE_POLICY_VERSION,
                    "namespace": cache_namespace,
                    "directory": str(cache_namespace_dir),
                    "sparse_config_hash": sparse_config_hash,
                    "tfidf_artifact_hash": tfidf_artifact_hash,
                    "partitions_signature": runtime[
                        "partitions_signature"
                    ],
                    "verified_sidecar_required_before_pickle_load": True,
                    "stats": persistent_cache_stats,
                },
            },
            "ranker_c": {
                "feature_order": RANKER_C_FEATURE_ORDER,
                "feature_count": len(RANKER_C_FEATURE_ORDER),
                "feature_order_sha256": feature_order_sha256(
                    RANKER_C_FEATURE_ORDER
                ),
                "acceptor_role_source_columns": CANDIDATE_ROLE_SOURCE_COLUMNS,
            },
            "outputs": outputs,
            "integrity": summary,
            "timing": {"total_seconds": total_seconds},
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "logical_cpu_count": os.cpu_count(),
                "dependencies": _dependency_versions(),
                "work_dir": str(work_dir),
            },
            "invariants": {
                "positive_injection": False,
                "truth_joined_after_all_pools_closed": True,
                "input_siret_opened_by_retrieval": False,
                "input_siren_opened_by_retrieval": False,
                "identifier_dependent_branch_enabled": False,
                "v42b_candidates_used_for_retrieval": False,
                "v42b_candidates_opened_after_all_pools_closed": True,
                "ranker_or_acceptor_loaded_or_scored": False,
                "training_performed": False,
                "candidate_ceiling": CANDIDATE_CEILING,
                "active_candidates_only": True,
                "snapshot_full_scan_count": 1,
                "snapshot_hydration_completed_before_labels_opened": True,
                "max_in_memory_tfidf_partitions": (
                    MAX_IN_MEMORY_TFIDF_PARTITIONS
                ),
                "tfidf_disk_cache_namespace_verified": True,
                "tfidf_pickle_requires_verified_hash_sidecar": True,
            },
        }
        _json_dump(staging / "manifest.json", manifest)
        os.replace(staging, target)
    except BaseException:
        if raw_writer is not None:
            raw_writer.close()
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(duckdb_temp, ignore_errors=True)
    return target


def validate_artifact(artifact_dir: Path) -> None:
    artifact_dir = Path(artifact_dir)
    manifest = json.loads(
        (artifact_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported V4.11 dataset schema")
    if manifest.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("Unexpected V4.11 experiment identifier")
    identity = manifest.get("build_identity") or {}
    expected_build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    if manifest.get("build_id") != expected_build_id:
        raise ValueError("V4.11 content-addressed build identity mismatch")
    if artifact_dir.name != expected_build_id:
        raise ValueError("V4.11 artifact directory is not its build ID")
    expected_cache_namespace = tfidf_cache_namespace(
        sparse_config_hash=str(identity.get("tfidf_sparse_config_hash") or ""),
        tfidf_artifact_hash=str(identity.get("tfidf_artifact_hash") or ""),
        partitions_signature=str(identity.get("partitions_signature") or ""),
    )
    if identity.get("tfidf_cache_namespace") != expected_cache_namespace:
        raise ValueError("V4.11 TF-IDF cache namespace identity mismatch")
    if identity.get("tfidf_cache_policy_version") != (
        TFIDF_CACHE_POLICY_VERSION
    ):
        raise ValueError("V4.11 TF-IDF cache policy changed")
    for filename, record in (manifest.get("outputs") or {}).items():
        path = artifact_dir / filename
        if file_sha256(path) != record.get("sha256"):
            raise ValueError(f"V4.11 output hash mismatch: {filename}")
    queries = pd.read_parquet(artifact_dir / "queries.parquet")
    labels = pd.read_parquet(artifact_dir / "labels.parquet")
    assignments = pd.read_parquet(artifact_dir / "split_assignments.parquet")
    candidates = pd.read_parquet(
        artifact_dir / "candidates_sparse_top100.parquet"
    )
    if list(queries.columns) != INPUT_BLIND_QUERY_COLUMNS:
        raise ValueError("V4.11 query projection changed")
    validate_assignments(queries, assignments)
    observed = compute_integrity(
        queries=queries,
        labels=labels,
        assignments=assignments,
        candidates=candidates,
    )
    declared = manifest.get("integrity") or {}
    for key, value in observed.items():
        if declared.get(key) != value:
            raise ValueError(f"V4.11 integrity mismatch: {key}")
    invariants = manifest.get("invariants") or {}
    for key in (
        "positive_injection",
        "input_siret_opened_by_retrieval",
        "input_siren_opened_by_retrieval",
        "identifier_dependent_branch_enabled",
        "v42b_candidates_used_for_retrieval",
        "ranker_or_acceptor_loaded_or_scored",
        "training_performed",
    ):
        if invariants.get(key) is not False:
            raise ValueError(f"V4.11 invariant changed: {key}")
    if invariants.get("snapshot_full_scan_count") != 1:
        raise ValueError("V4.11 snapshot was not hydrated in one bulk scan")
    if invariants.get("snapshot_hydration_completed_before_labels_opened") is not True:
        raise ValueError("V4.11 label phase boundary changed")
    if invariants.get("max_in_memory_tfidf_partitions") != (
        MAX_IN_MEMORY_TFIDF_PARTITIONS
    ):
        raise ValueError("V4.11 TF-IDF cache ceiling changed")
    for key in (
        "tfidf_disk_cache_namespace_verified",
        "tfidf_pickle_requires_verified_hash_sidecar",
    ):
        if invariants.get(key) is not True:
            raise ValueError(f"V4.11 verified cache invariant changed: {key}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", type=Path)
    parser.add_argument("--partitions", type=Path)
    parser.add_argument("--global-store", type=Path)
    parser.add_argument("--state-snapshot", type=Path)
    parser.add_argument("--v42-manifest", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--validate-artifact", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_artifact:
        validate_artifact(args.validate_artifact)
        print(args.validate_artifact)
        return
    names = (
        "source_dataset",
        "partitions",
        "global_store",
        "state_snapshot",
        "v42_manifest",
        "cache_dir",
        "work_dir",
        "output_root",
    )
    missing = [name for name in names if getattr(args, name) is None]
    if missing:
        raise SystemExit(f"Missing required arguments: {', '.join(missing)}")
    target = build_artifact(
        source_dataset_dir=args.source_dataset,
        partitions_dir=args.partitions,
        global_store_path=args.global_store,
        state_snapshot_path=args.state_snapshot,
        v42_manifest_path=args.v42_manifest,
        cache_dir=args.cache_dir,
        work_dir=args.work_dir,
        output_root=args.output_root,
        contract_path=args.contract,
    )
    print(target)


if __name__ == "__main__":
    main()
