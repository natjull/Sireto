#!/usr/bin/env python3
"""Build the label-free V4.12 service-parity reference package.

Only explicitly listed Parquet projections are read.  In particular, this
builder never requests labels, targets, correctness flags or
``is_ground_truth`` even when a frozen source file physically contains them.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any, Mapping, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.xgb_matcher.v412_direct_evidence import apply_guard, validate_evidence


SCHEMA_VERSION = "sireto-v4.12-service-parity-reference-1"
EXPECTED_QUERY_COUNT = 1_456
EXPECTED_CANDIDATE_COUNT = 145_236
MAX_CANDIDATES = 100
COMPACT_LOGIT = "COMPACT_LOGIT"
FIXED_THRESHOLD = 0.8720916706888049

INPUT_BLIND_ROOT = Path(
    "/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/"
    "v4_11_input_blind/ec4326ec57e4411d"
)
RANKER_ROOT = Path(
    "/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/"
    "v4_11_ranker_c/e13eb3ac7498256e"
)
SCENES_ROOT = Path(
    "/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/"
    "v4_11_acceptor/52ea3faba9a56aff"
)
ACCEPTOR_ROOT = Path(
    "/Volumes/CATNAT_DATA/SIRETO_RECALL100/models/"
    "v4_11_acceptor/9d23bf3deb6b63de"
)
EVIDENCE_ROOT = Path(
    "/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/"
    "v4_12_direct_evidence/10f16403795ccee6"
)
GUARD_ROOT = Path(
    "/Volumes/CATNAT_DATA/SIRETO_RECALL100/evaluations/"
    "v4_12_guard_historical/fedcd1d512bfd269"
)

QUERY_COLUMNS = [
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
SPLIT_COLUMNS = ["query_id", "split"]
CANDIDATE_COLUMNS = [
    "query_id",
    "candidate_siret",
    "candidate_siren",
    "candidate_state",
    "retrieval_rank",
    "retrieval_source",
    "retrieval_channel_count",
    "retrieval_agreement",
    "enseigne1",
    "enseigne2",
    "enseigne3",
    "denomination_usuelle",
    "activity_code",
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
RANKER_COLUMNS = [
    "query_id",
    "candidate_siret",
    "candidate_siren",
    "retrieval_rank",
    "ranker_score",
    "prediction_origin",
    "oof_fold",
    "ranker_rank",
]
SCENE_FEATURES = [
    "candidate_count",
    "ranker_gap_fraction",
    "ranker_top3_gap_fraction",
    "ranker_score_std_fraction",
    "ranker_score_entropy",
    "unique_siren_count",
    "top1_siren_candidate_count",
    "same_siren_top2",
    "siren_gap_fraction",
    "retrieval_rank_top1_recip",
    "retrieval_rank_gap_recip",
    "same_siren_best_sibling_gap_fraction",
    "crm_is_school",
    "top1_name_jaro_max",
    "delta_name_jaro_max",
    "top1_name_jaro_gap",
    "delta_name_jaro_gap",
    "top1_name_token_overlap_max",
    "delta_name_token_overlap_max",
    "top1_idf_name",
    "delta_idf_name",
    "top1_numeric_token_match",
    "delta_numeric_token_match",
    "top1_name_first_word_match_max",
    "delta_name_first_word_match_max",
    "top1_name_contains_crm_max",
    "delta_name_contains_crm_max",
    "top1_name_crm_contains_cand_max",
    "delta_name_crm_contains_cand_max",
    "top1_acronym_match_max",
    "delta_acronym_match_max",
    "top1_name_sim_max_etab",
    "delta_name_sim_max_etab",
    "top1_name_sim_max_ul",
    "delta_name_sim_max_ul",
    "top1_name_sim_max_sigle",
    "delta_name_sim_max_sigle",
    "top1_name_sim_max_pm_dirigeant",
    "delta_name_sim_max_pm_dirigeant",
    "top1_is_ul_name_max",
    "delta_is_ul_name_max",
    "top1_is_sigle_max",
    "delta_is_sigle_max",
    "top1_person_name_jaro_max",
    "delta_person_name_jaro_max",
    "top1_name_is_city_like_max",
    "delta_name_is_city_like_max",
    "top1_addr_jaro",
    "delta_addr_jaro",
    "top1_postcode_match",
    "delta_postcode_match",
    "top1_city_match",
    "delta_city_match",
    "top1_street_number_diff",
    "delta_street_number_diff",
    "top1_addr_token_overlap",
    "delta_addr_token_overlap",
    "top1_address_density",
    "delta_address_density",
    "top1_street_name_jaro",
    "delta_street_name_jaro",
    "top1_name_addr_consistency",
    "delta_name_addr_consistency",
    "top1_geo_exact_match",
    "delta_geo_exact_match",
    "top1_name_norm_exact",
    "delta_name_norm_exact",
    "top1_street_number_match",
    "delta_street_number_match",
    "top1_is_siege",
    "delta_is_siege",
    "top1_is_association",
    "delta_is_association",
    "role_crm_count",
    "role_top1_count",
    "role_crm_top1_conflict",
    "role_top1_top2_conflict",
    "same_siren_distinct_role_count",
    "same_siren_role_plurality",
    "naf_top1_top2_division_equal",
]
SCENE_COLUMNS = [
    "query_id",
    "crm_record_id",
    "predicted_siret",
    "predicted_siren",
    *SCENE_FEATURES,
]
ACCEPTOR_COLUMNS = [
    "model_family",
    "evaluation_partition",
    "query_id",
    "predicted_siret",
    "score",
    "threshold",
    "decision",
]
QUERY_EVIDENCE_COLUMNS = [
    "query_id",
    "partition_key",
    "active_universe_count",
    "direct_candidate_count",
    "direct_siren_count",
    "sole_direct_siret",
    "sole_direct_siren",
    "cross_siren_direct_collision",
    "same_siren_direct_multisite",
    "evidence_refs_json",
]
CANDIDATE_EVIDENCE_COLUMNS = [
    "evidence_ref",
    "query_id",
    "candidate_siret",
    "candidate_siren",
    "candidate_state",
    "exact_name_anchor",
    "exact_address_anchor",
    "strong_name_evidence",
    "strong_address_evidence",
    "direct_evidence_class",
    "direct_match_rule",
]
GUARD_SOURCE_COLUMNS = [
    "query_id",
    "predicted_siret",
    "acceptor_score",
    "decision_v411",
    "review_reason_v411",
    "direct_candidate_count",
    "direct_siren_count",
    "sole_direct_siret",
    "sole_direct_siren",
    "decision_v412",
    "review_reason_v412",
]
GUARD_COLUMNS = [
    "query_id",
    "predicted_siret",
    "predicted_siren",
    "acceptor_score",
    "acceptor_threshold",
    "decision_v411",
    "direct_candidate_count",
    "direct_siren_count",
    "sole_direct_siret",
    "sole_direct_siren",
    "decision_v412",
    "review_reason_v412",
]

FORBIDDEN_COLUMN_TOKENS = (
    "ground_truth",
    "is_ground_truth",
    "label_kind",
    "acceptor_target",
    "correct",
    "truth_present",
    "truth_absent",
    "input_siret",
    "input_siren",
    "siren_component_id",
)
OUTPUT_FILES = {
    "queries.parquet": QUERY_COLUMNS,
    "candidates_features.parquet": CANDIDATE_COLUMNS,
    "ranker_reference.parquet": RANKER_COLUMNS,
    "scenes_reference.parquet": SCENE_COLUMNS,
    "acceptor_reference.parquet": ACCEPTOR_COLUMNS,
    "guard_reference.parquet": GUARD_COLUMNS,
    "query_evidence.parquet": QUERY_EVIDENCE_COLUMNS,
    "candidate_evidence.parquet": CANDIDATE_EVIDENCE_COLUMNS,
}


class ReferenceBuildError(RuntimeError):
    """Fail-closed reference package build error."""


@dataclass(frozen=True)
class SourceFile:
    path: Path
    sha256: str


@dataclass(frozen=True)
class ReferenceSources:
    queries: SourceFile
    splits: SourceFile
    candidates: SourceFile
    ranker: SourceFile
    scenes: SourceFile
    acceptor: SourceFile
    guard: SourceFile
    query_evidence: SourceFile
    candidate_evidence: SourceFile


DEFAULT_SOURCES = ReferenceSources(
    queries=SourceFile(
        INPUT_BLIND_ROOT / "queries.parquet",
        "3a47aef768cee1436ad77a6e114defe50e685b7495f0e75137e9fd06dfe9fc68",
    ),
    splits=SourceFile(
        INPUT_BLIND_ROOT / "split_assignments.parquet",
        "33fa52af7a740124235c151efb5b9a8834ffd1c83c65d1af56c75b2eff271193",
    ),
    candidates=SourceFile(
        INPUT_BLIND_ROOT / "candidates_sparse_top100.parquet",
        "78b2f78ddeac863ac39ca64301d42312c7fb766ac51e2b5d19dde5c5910aedac",
    ),
    ranker=SourceFile(
        RANKER_ROOT / "predictions_ranker_c_oof_dev.parquet",
        "f14828aafa146dc4ad0399697c9477e57930ba618a5b2d7d0a903e52c2d879c0",
    ),
    scenes=SourceFile(
        SCENES_ROOT / "acceptor_scenes.parquet",
        "c0f3d670e50cb43cdd6fed3b976c95e51d70f0313ae07ed0e1e2ed01eca5bed3",
    ),
    acceptor=SourceFile(
        ACCEPTOR_ROOT / "predictions.parquet",
        "bfed032d097cfda4a6730adb9cf04e970ae0d70f97fb0acf388cc985dadd3d3d",
    ),
    guard=SourceFile(
        GUARD_ROOT / "decisions.parquet",
        "c5b79d95503a7d10a1ca7a1a2e2c82aeb76f835347b0af8866c438809bfd944a",
    ),
    query_evidence=SourceFile(
        EVIDENCE_ROOT / "query_evidence.parquet",
        "6fd7d441a9f6aaac99555c1083b90a61e52e02a291d6cffccc07d061f60242ed",
    ),
    candidate_evidence=SourceFile(
        EVIDENCE_ROOT / "candidate_evidence.parquet",
        "36a3b0042a30c852f1a0595d2a31a91858b4fa5ea405cf7ccc32330f345f7d97",
    ),
)

SOURCE_PROJECTIONS = {
    "queries": QUERY_COLUMNS,
    "splits": SPLIT_COLUMNS,
    "candidates": CANDIDATE_COLUMNS,
    "ranker": RANKER_COLUMNS,
    "scenes": SCENE_COLUMNS,
    "acceptor": ACCEPTOR_COLUMNS,
    "guard": GUARD_SOURCE_COLUMNS,
    "query_evidence": QUERY_EVIDENCE_COLUMNS,
    "candidate_evidence": CANDIDATE_EVIDENCE_COLUMNS,
}


def canonical_json(value: Any) -> bytes:
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ReferenceBuildError("STOP_REFERENCE_INPUT_OPEN") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ReferenceBuildError("STOP_REFERENCE_INPUT_IDENTITY")
        while chunk := os.read(fd, 1 << 20):
            digest.update(chunk)
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
            raise ReferenceBuildError("STOP_REFERENCE_INPUT_DRIFT")
    finally:
        os.close(fd)
    return digest.hexdigest()


def _validate_source(source: SourceFile) -> None:
    if (
        len(source.sha256) != 64
        or any(character not in "0123456789abcdef" for character in source.sha256)
        or file_sha256(source.path) != source.sha256
    ):
        raise ReferenceBuildError("STOP_REFERENCE_INPUT_HASH")


def read_projection(
    source: SourceFile,
    columns: Sequence[str],
    *,
    reader: Any = pd.read_parquet,
) -> pd.DataFrame:
    """Validate a pin, then request exactly the declared projection."""
    _validate_source(source)
    frame = reader(source.path, columns=list(columns))
    if type(frame) is not pd.DataFrame or list(frame.columns) != list(columns):
        raise ReferenceBuildError("STOP_REFERENCE_PROJECTION")
    if file_sha256(source.path) != source.sha256:
        raise ReferenceBuildError("STOP_REFERENCE_INPUT_DRIFT")
    return frame


def _require_exact_ids(
    frame: pd.DataFrame, ids: set[str], label: str, *, unique: bool
) -> None:
    observed = frame["query_id"].astype(str)
    if set(observed) != ids or (unique and observed.duplicated().any()):
        raise ReferenceBuildError(f"STOP_REFERENCE_{label}_IDS")


def _require_non_null(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    if frame[list(columns)].isna().any(axis=None):
        raise ReferenceBuildError(f"STOP_REFERENCE_{label}_NULL")


def _validate_safe_columns(filename: str, columns: Sequence[str]) -> None:
    lowered = [column.lower() for column in columns]
    for column in lowered:
        if any(token in column for token in FORBIDDEN_COLUMN_TOKENS):
            raise ReferenceBuildError(
                f"STOP_REFERENCE_FORBIDDEN_COLUMN:{filename}:{column}"
            )


def _build_guard(
    acceptor: pd.DataFrame,
    scenes: pd.DataFrame,
    evidence: pd.DataFrame,
) -> pd.DataFrame:
    base = acceptor.merge(
        scenes[["query_id", "predicted_siren"]],
        on="query_id",
        how="inner",
        validate="one_to_one",
    ).merge(
        evidence[
            [
                "query_id",
                "direct_candidate_count",
                "direct_siren_count",
                "sole_direct_siret",
                "sole_direct_siren",
            ]
        ],
        on="query_id",
        how="inner",
        validate="one_to_one",
    )
    decisions: list[str] = []
    reasons: list[str | None] = []
    for row in base.itertuples(index=False):
        decision, reason = apply_guard(
            decision_v411=row.decision,
            review_reason_v411=(
                "LOW_CONFIDENCE" if row.decision == "REVIEW" else None
            ),
            predicted_siret=row.predicted_siret,
            direct_candidate_count=row.direct_candidate_count,
            sole_direct_siret=row.sole_direct_siret,
        )
        decisions.append(decision)
        reasons.append(reason)
    output = pd.DataFrame(
        {
            "query_id": base["query_id"].astype(str),
            "predicted_siret": base["predicted_siret"],
            "predicted_siren": base["predicted_siren"],
            "acceptor_score": base["score"].astype(float),
            "acceptor_threshold": base["threshold"].astype(float),
            "decision_v411": base["decision"].astype(str),
            "direct_candidate_count": base["direct_candidate_count"].astype(
                "int64"
            ),
            "direct_siren_count": base["direct_siren_count"].astype("int64"),
            "sole_direct_siret": base["sole_direct_siret"],
            "sole_direct_siren": base["sole_direct_siren"],
            "decision_v412": decisions,
            "review_reason_v412": reasons,
        }
    )
    return output[GUARD_COLUMNS].sort_values("query_id").reset_index(drop=True)


def _nullable_strings_equal(left: pd.Series, right: pd.Series) -> bool:
    """Compare text fields while treating every missing representation alike."""
    return bool(
        left.astype("string").fillna("<NULL>").eq(
            right.astype("string").fillna("<NULL>")
        ).all()
    )


def _validate_guard_parity(
    rebuilt: pd.DataFrame,
    frozen: pd.DataFrame,
) -> None:
    """Require exact agreement with the independently frozen V4.12 batch."""
    aligned = rebuilt.merge(
        frozen,
        on="query_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_rebuilt", "_frozen"),
    )
    if len(aligned) != len(rebuilt) or len(aligned) != len(frozen):
        raise ReferenceBuildError("STOP_REFERENCE_GUARD_PARITY")

    string_columns = [
        "predicted_siret",
        "decision_v411",
        "direct_candidate_count",
        "direct_siren_count",
        "sole_direct_siret",
        "sole_direct_siren",
        "decision_v412",
        "review_reason_v412",
    ]
    for column in string_columns:
        if not _nullable_strings_equal(
            aligned[f"{column}_rebuilt"],
            aligned[f"{column}_frozen"],
        ):
            raise ReferenceBuildError("STOP_REFERENCE_GUARD_PARITY")

    expected_v411_reason = aligned["decision_v411_rebuilt"].map(
        {"AUTO_MATCH": None, "REVIEW": "LOW_CONFIDENCE"}
    )
    if not _nullable_strings_equal(
        expected_v411_reason,
        aligned["review_reason_v411"],
    ):
        raise ReferenceBuildError("STOP_REFERENCE_GUARD_PARITY")

    score_delta = (
        aligned["acceptor_score_rebuilt"].astype(float)
        - aligned["acceptor_score_frozen"].astype(float)
    ).abs()
    if score_delta.isna().any() or score_delta.gt(1e-15).any():
        raise ReferenceBuildError("STOP_REFERENCE_GUARD_PARITY")


def build_frames(
    sources: ReferenceSources,
    *,
    expected_query_count: int = EXPECTED_QUERY_COUNT,
    expected_candidate_count: int = EXPECTED_CANDIDATE_COUNT,
    reader: Any = pd.read_parquet,
) -> dict[str, pd.DataFrame]:
    raw: dict[str, pd.DataFrame] = {}
    for field in fields(sources):
        name = field.name
        raw[name] = read_projection(
            getattr(sources, name), SOURCE_PROJECTIONS[name], reader=reader
        )

    split = raw["splits"]
    if split["query_id"].astype(str).duplicated().any():
        raise ReferenceBuildError("STOP_REFERENCE_SPLIT_DUPLICATE")
    dev_ids = set(
        split.loc[split["split"].astype(str).eq("dev"), "query_id"].astype(str)
    )
    if len(dev_ids) != expected_query_count:
        raise ReferenceBuildError("STOP_REFERENCE_DEV_COUNT")

    queries = raw["queries"][
        raw["queries"]["query_id"].astype(str).isin(dev_ids)
    ].copy()
    candidates = raw["candidates"][
        raw["candidates"]["query_id"].astype(str).isin(dev_ids)
    ].copy()
    ranker = raw["ranker"][
        raw["ranker"]["query_id"].astype(str).isin(dev_ids)
    ].copy()
    scenes = raw["scenes"][
        raw["scenes"]["query_id"].astype(str).isin(dev_ids)
    ].copy()
    acceptor = raw["acceptor"][
        raw["acceptor"]["query_id"].astype(str).isin(dev_ids)
        & raw["acceptor"]["model_family"].astype(str).eq(COMPACT_LOGIT)
    ].copy()
    guard = raw["guard"][
        raw["guard"]["query_id"].astype(str).isin(dev_ids)
    ].copy()
    query_evidence = raw["query_evidence"][
        raw["query_evidence"]["query_id"].astype(str).isin(dev_ids)
    ].copy()
    candidate_evidence = raw["candidate_evidence"][
        raw["candidate_evidence"]["query_id"].astype(str).isin(dev_ids)
    ].copy()

    for frame, label in (
        (queries, "QUERIES"),
        (scenes, "SCENES"),
        (acceptor, "ACCEPTOR"),
        (guard, "GUARD"),
        (query_evidence, "QUERY_EVIDENCE"),
    ):
        _require_exact_ids(frame, dev_ids, label, unique=True)
    for frame, label in (
        (candidates, "CANDIDATES"),
        (ranker, "RANKER"),
    ):
        _require_exact_ids(frame, dev_ids, label, unique=False)
        if len(frame) != expected_candidate_count:
            raise ReferenceBuildError(f"STOP_REFERENCE_{label}_COUNT")
    candidate_keys = set(
        zip(
            candidates["query_id"].astype(str),
            candidates["candidate_siret"].astype(str),
            strict=False,
        )
    )
    ranker_keys = set(
        zip(
            ranker["query_id"].astype(str),
            ranker["candidate_siret"].astype(str),
            strict=False,
        )
    )
    if (
        candidate_keys != ranker_keys
        or len(candidate_keys) != expected_candidate_count
    ):
        raise ReferenceBuildError("STOP_REFERENCE_CANDIDATE_PARITY")
    for query_id, group in candidates.groupby("query_id", sort=False):
        ranks = sorted(group["retrieval_rank"].astype(int).tolist())
        if len(ranks) > MAX_CANDIDATES or ranks != list(range(1, len(ranks) + 1)):
            raise ReferenceBuildError("STOP_REFERENCE_POOL_CAP_OR_RANKS")
    for query_id, group in ranker.groupby("query_id", sort=False):
        ranks = sorted(group["ranker_rank"].astype(int).tolist())
        if len(ranks) > MAX_CANDIDATES or ranks != list(range(1, len(ranks) + 1)):
            raise ReferenceBuildError("STOP_REFERENCE_RANKER_RANKS")
    if (
        acceptor["model_family"].ne(COMPACT_LOGIT).any()
        or not bool(acceptor["decision"].isin(["AUTO_MATCH", "REVIEW"]).all())
        or acceptor["threshold"].astype(float).ne(FIXED_THRESHOLD).any()
    ):
        raise ReferenceBuildError("STOP_REFERENCE_ACCEPTOR_POLICY")
    _require_non_null(
        scenes,
        ["query_id", "crm_record_id", "predicted_siret", "predicted_siren"],
        "SCENES",
    )
    _require_non_null(
        acceptor,
        ["query_id", "predicted_siret", "score", "threshold", "decision"],
        "ACCEPTOR",
    )
    ranker_top1 = ranker.loc[
        ranker["ranker_rank"].astype(int).eq(1),
        ["query_id", "candidate_siret", "candidate_siren"],
    ].rename(
        columns={
            "candidate_siret": "ranker_predicted_siret",
            "candidate_siren": "ranker_predicted_siren",
        }
    )
    if (
        len(ranker_top1) != expected_query_count
        or ranker_top1["query_id"].astype(str).duplicated().any()
    ):
        raise ReferenceBuildError("STOP_REFERENCE_RANKER_TOP1")
    prediction_parity = (
        scenes[["query_id", "predicted_siret", "predicted_siren"]]
        .merge(ranker_top1, on="query_id", how="inner", validate="one_to_one")
        .merge(
            acceptor[["query_id", "predicted_siret"]].rename(
                columns={"predicted_siret": "acceptor_predicted_siret"}
            ),
            on="query_id",
            how="inner",
            validate="one_to_one",
        )
    )
    if (
        len(prediction_parity) != expected_query_count
        or prediction_parity["predicted_siret"]
        .astype(str)
        .ne(prediction_parity["ranker_predicted_siret"].astype(str))
        .any()
        or prediction_parity["predicted_siren"]
        .astype(str)
        .ne(prediction_parity["ranker_predicted_siren"].astype(str))
        .any()
        or prediction_parity["predicted_siret"]
        .astype(str)
        .ne(prediction_parity["acceptor_predicted_siret"].astype(str))
        .any()
    ):
        raise ReferenceBuildError("STOP_REFERENCE_PREDICTION_PARITY")
    if set(candidate_evidence["query_id"].astype(str)) - dev_ids:
        raise ReferenceBuildError("STOP_REFERENCE_CANDIDATE_EVIDENCE_IDS")
    if candidate_evidence[["query_id", "evidence_ref"]].duplicated().any():
        raise ReferenceBuildError("STOP_REFERENCE_EVIDENCE_DUPLICATE")
    declared_evidence: set[tuple[str, str]] = set()
    for row in query_evidence[["query_id", "evidence_refs_json"]].itertuples(
        index=False
    ):
        try:
            refs = json.loads(str(row.evidence_refs_json))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ReferenceBuildError("STOP_REFERENCE_EVIDENCE_JSON") from exc
        if (
            not isinstance(refs, list)
            or any(not isinstance(ref, str) or not ref for ref in refs)
            or len(refs) != len(set(refs))
        ):
            raise ReferenceBuildError("STOP_REFERENCE_EVIDENCE_JSON")
        declared_evidence.update((str(row.query_id), ref) for ref in refs)
    observed_evidence = set(
        zip(
            candidate_evidence["query_id"].astype(str),
            candidate_evidence["evidence_ref"].astype(str),
            strict=False,
        )
    )
    if declared_evidence != observed_evidence:
        raise ReferenceBuildError("STOP_REFERENCE_EVIDENCE_PARITY")
    try:
        validate_evidence(query_evidence, candidate_evidence)
    except ValueError as exc:
        raise ReferenceBuildError("STOP_REFERENCE_EVIDENCE_INVALID") from exc

    outputs = {
        "queries.parquet": queries.sort_values("query_id").reset_index(drop=True),
        "candidates_features.parquet": candidates.sort_values(
            ["query_id", "retrieval_rank", "candidate_siret"]
        ).reset_index(drop=True),
        "ranker_reference.parquet": ranker.sort_values(
            ["query_id", "ranker_rank", "candidate_siret"]
        ).reset_index(drop=True),
        "scenes_reference.parquet": scenes.sort_values("query_id").reset_index(
            drop=True
        ),
        "acceptor_reference.parquet": acceptor.sort_values("query_id").reset_index(
            drop=True
        ),
        "query_evidence.parquet": query_evidence.sort_values("query_id").reset_index(
            drop=True
        ),
        "candidate_evidence.parquet": candidate_evidence.sort_values(
            ["query_id", "candidate_siret", "evidence_ref"]
        ).reset_index(drop=True),
    }
    rebuilt_guard = _build_guard(
        outputs["acceptor_reference.parquet"],
        outputs["scenes_reference.parquet"],
        outputs["query_evidence.parquet"],
    )
    _validate_guard_parity(rebuilt_guard, guard)
    outputs["guard_reference.parquet"] = rebuilt_guard
    for filename, frame in outputs.items():
        expected_columns = OUTPUT_FILES[filename]
        if list(frame.columns) != expected_columns:
            raise ReferenceBuildError(f"STOP_REFERENCE_SCHEMA:{filename}")
        _validate_safe_columns(filename, frame.columns)
    return outputs


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(
        table,
        path,
        compression="zstd",
        compression_level=9,
        use_dictionary=False,
        write_statistics=True,
        version="2.6",
        data_page_version="1.0",
        row_group_size=65_536,
    )
    os.chmod(path, 0o600)
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(fd)
        full = getattr(__import__("fcntl"), "F_FULLFSYNC", None)
        if full is None:
            raise ReferenceBuildError("STOP_REFERENCE_FULLFSYNC_UNAVAILABLE")
        __import__("fcntl").fcntl(fd, full)
    finally:
        os.close(fd)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    raw = canonical_json(value)
    fd = os.open(path, flags, 0o600)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise ReferenceBuildError("STOP_REFERENCE_SHORT_WRITE")
            view = view[written:]
        os.fsync(fd)
        full = getattr(__import__("fcntl"), "F_FULLFSYNC", None)
        if full is None:
            raise ReferenceBuildError("STOP_REFERENCE_FULLFSYNC_UNAVAILABLE")
        __import__("fcntl").fcntl(fd, full)
    finally:
        os.close(fd)


def _promote_exclusive(staging: Path, target: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "renameatx_np", None)
    if function is None:
        raise ReferenceBuildError("STOP_REFERENCE_RENAME_EXCL_UNAVAILABLE")
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    function.restype = ctypes.c_int
    result = function(
        -2,
        os.fsencode(staging),
        -2,
        os.fsencode(target),
        0x00000004,
    )
    if result != 0:
        error = ctypes.get_errno()
        if error in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(f"Immutable reference exists: {target}")
        raise ReferenceBuildError("STOP_REFERENCE_PROMOTION")


def _source_manifest(sources: ReferenceSources) -> dict[str, Any]:
    return {
        field.name: {
            "path": str(getattr(sources, field.name).path),
            "sha256": getattr(sources, field.name).sha256,
            "projection": SOURCE_PROJECTIONS[field.name],
        }
        for field in fields(sources)
    }


def build_reference(
    output_root: Path,
    *,
    sources: ReferenceSources = DEFAULT_SOURCES,
    expected_query_count: int = EXPECTED_QUERY_COUNT,
    expected_candidate_count: int = EXPECTED_CANDIDATE_COUNT,
    reader: Any = pd.read_parquet,
) -> Path:
    output_root_input = Path(output_root)
    if output_root_input.is_symlink():
        raise ReferenceBuildError("STOP_REFERENCE_OUTPUT_ROOT")
    output_root = output_root_input.resolve()
    if not output_root.is_dir() or output_root.is_symlink():
        raise ReferenceBuildError("STOP_REFERENCE_OUTPUT_ROOT")
    spec = {
        "schema_version": SCHEMA_VERSION,
        "sources": _source_manifest(sources),
        "outputs": OUTPUT_FILES,
        "expected_query_count": expected_query_count,
        "expected_candidate_count": expected_candidate_count,
        "max_candidates": MAX_CANDIDATES,
        "model_family": COMPACT_LOGIT,
        "threshold": FIXED_THRESHOLD,
    }
    build_id = hashlib.sha256(canonical_json(spec)).hexdigest()[:16]
    target = output_root / build_id
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Immutable reference exists: {target}")
    staging: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{build_id}.tmp-", dir=output_root)
    )
    os.chmod(staging, 0o700)
    try:
        frames = build_frames(
            sources,
            expected_query_count=expected_query_count,
            expected_candidate_count=expected_candidate_count,
            reader=reader,
        )
        outputs: dict[str, Any] = {}
        for filename in OUTPUT_FILES:
            path = staging / filename
            _write_parquet(path, frames[filename])
            outputs[filename] = {
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
                "row_count": len(frames[filename]),
                "columns": OUTPUT_FILES[filename],
            }
        manifest = {
            **spec,
            "build_id": build_id,
            "outputs": outputs,
            "invariants": {
                "dev_filter_from_split_assignments_only": True,
                "query_count": expected_query_count,
                "candidate_count": expected_candidate_count,
                "candidate_maximum_absolute": MAX_CANDIDATES,
                "labels_opened": False,
                "challenge_opened": False,
                "test_or_final_opened": False,
                "model_retrained": False,
                "reference_only": True,
            },
        }
        _write_json(staging / "manifest.json", manifest)
        parent_fd = os.open(output_root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent_fd)
            _promote_exclusive(staging, target)
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        staging = None
        return target
    finally:
        if staging is not None and staging.exists():
            if (
                staging.parent != output_root
                or not staging.name.startswith(f".{build_id}.tmp-")
                or staging.is_symlink()
                or not staging.is_dir()
            ):
                raise ReferenceBuildError("STOP_REFERENCE_STAGING_IDENTITY")
            shutil.rmtree(staging)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(build_reference(args.output_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
