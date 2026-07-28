"""Pure V4.11 compact query-scene features shared by train and serve.

The module intentionally has no model or dataset I/O.  A caller supplies one
CRM query, the candidates scored by the ranker, and the frozen V4.9 site
function taxonomy.  The same function is therefore usable when materialising
OOF training scenes and when serving a frozen bundle.
"""

from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .v49_site_function import FunctionDetection, SiteFunctionTaxonomy


V411_SCENE_FEATURE_NAMES = [
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
]

V411_EVIDENCE_BASE_FEATURE_NAMES = [
    "name_jaro_max",
    "name_jaro_gap",
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
    "is_ul_name_max",
    "is_sigle_max",
    "person_name_jaro_max",
    "name_is_city_like_max",
    "addr_jaro",
    "postcode_match",
    "city_match",
    "street_number_diff",
    "addr_token_overlap",
    "address_density",
    "street_name_jaro",
    "name_addr_consistency",
    "geo_exact_match",
    "name_norm_exact",
    "street_number_match",
    "is_siege",
    "is_association",
]

V411_EVIDENCE_FEATURE_NAMES = [
    name
    for base in V411_EVIDENCE_BASE_FEATURE_NAMES
    for name in (f"top1_{base}", f"delta_{base}")
]

V411_ROLE_FEATURE_NAMES = [
    "role_crm_count",
    "role_top1_count",
    "role_crm_top1_conflict",
    "role_top1_top2_conflict",
    "same_siren_distinct_role_count",
    "same_siren_role_plurality",
    "naf_top1_top2_division_equal",
]

V411_ACCEPTOR_FEATURE_NAMES = (
    V411_SCENE_FEATURE_NAMES
    + V411_EVIDENCE_FEATURE_NAMES
    + V411_ROLE_FEATURE_NAMES
)

# Candidate bases that are genuinely binary in the frozen candidate feature
# implementation.  A delta between two binary values is kept raw too, even
# though its domain is {-1, 0, 1}.
V411_BINARY_EVIDENCE_BASES = frozenset(
    {
        "name_first_word_match_max",
        "name_contains_crm_max",
        "name_crm_contains_cand_max",
        "acronym_match_max",
        "is_ul_name_max",
        "is_sigle_max",
        "name_is_city_like_max",
        "postcode_match",
        "city_match",
        "geo_exact_match",
        "name_norm_exact",
        "street_number_match",
        "is_siege",
        "is_association",
    }
)

V411_BINARY_FEATURE_NAMES = [
    "same_siren_top2",
    "crm_is_school",
    *[
        name
        for base in V411_EVIDENCE_BASE_FEATURE_NAMES
        if base in V411_BINARY_EVIDENCE_BASES
        for name in (f"top1_{base}", f"delta_{base}")
    ],
    "role_crm_top1_conflict",
    "role_top1_top2_conflict",
    "same_siren_role_plurality",
    "naf_top1_top2_division_equal",
]

# The compact logistic model standardises every non-binary field.  Counts are
# intentionally included; V4.10b showed that leaving them raw distorted the
# regularisation strength.
V411_SCALED_FEATURE_NAMES = [
    name
    for name in V411_ACCEPTOR_FEATURE_NAMES
    if name not in set(V411_BINARY_FEATURE_NAMES)
]

_POSITIVE_MONOTONIC_BASES = frozenset(
    {
        "name_jaro_max",
        "name_jaro_gap",
        "name_token_overlap_max",
        "numeric_token_match",
        "name_first_word_match_max",
        "name_contains_crm_max",
        "name_crm_contains_cand_max",
        "acronym_match_max",
        "name_sim_max_etab",
        "name_sim_max_ul",
        "name_sim_max_sigle",
        "name_sim_max_pm_dirigeant",
        "person_name_jaro_max",
        "addr_jaro",
        "postcode_match",
        "city_match",
        "addr_token_overlap",
        "street_name_jaro",
        "name_addr_consistency",
        "geo_exact_match",
        "name_norm_exact",
        "street_number_match",
    }
)

V411_MONOTONIC_CONSTRAINTS = [
    (
        1
        if name
        in {
            "ranker_gap_fraction",
            "siren_gap_fraction",
            "retrieval_rank_top1_recip",
            "retrieval_rank_gap_recip",
            "same_siren_best_sibling_gap_fraction",
        }
        or any(
            name in {f"top1_{base}", f"delta_{base}"}
            for base in _POSITIVE_MONOTONIC_BASES
        )
        else -1
        if name
        in {
            "ranker_score_entropy",
            "top1_street_number_diff",
            "delta_street_number_diff",
            "role_crm_top1_conflict",
            "role_top1_top2_conflict",
            "same_siren_role_plurality",
        }
        else 0
    )
    for name in V411_ACCEPTOR_FEATURE_NAMES
]

_FORBIDDEN_FEATURE_FRAGMENTS = (
    "input_siret",
    "input_siren",
    "candidate_from_input",
    "candidate_from_closed_alias",
    "ground_truth",
    "label",
    "split",
    "fold",
    "population",
    "prediction_origin",
    "score_top",
    "scene_score",
)


def _missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _candidate_feature(value: Any, *, name: str) -> float:
    """Apply the frozen candidate-feature missing-value policy.

    Missing and non-numeric historical candidate features are zero.  An
    explicitly infinite value is corrupted data rather than a missing value.
    """

    if _missing(value):
        return 0.0
    try:
        output = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(output):
        raise ValueError(
            f"STOP_DATASET_INTEGRITY: non-finite candidate feature {name}"
        )
    return output


def _required_finite(value: Any, *, name: str) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"STOP_DATASET_INTEGRITY: invalid numeric value for {name}"
        ) from error
    if not math.isfinite(output):
        raise ValueError(
            f"STOP_DATASET_INTEGRITY: non-finite numeric value for {name}"
        )
    return output


def _normalise_siret(value: Any) -> str:
    if _missing(value):
        raise ValueError("STOP_DATASET_INTEGRITY: candidate SIRET is missing")
    digits = "".join(character for character in str(value) if character.isdigit())
    if not digits or len(digits) > 14:
        raise ValueError("STOP_DATASET_INTEGRITY: invalid candidate SIRET")
    output = digits.zfill(14)
    if len(output) != 14:
        raise ValueError("STOP_DATASET_INTEGRITY: invalid candidate SIRET")
    return output


def _normalise_siren(value: Any, *, siret: str) -> str:
    if _missing(value) or not str(value).strip():
        return siret[:9]
    digits = "".join(character for character in str(value) if character.isdigit())
    output = digits.zfill(9)
    if len(output) != 9 or output != siret[:9]:
        raise ValueError(
            "STOP_DATASET_INTEGRITY: candidate SIREN/SIRET incoherence"
        )
    return output


def rank_v411_candidates(
    candidates: pd.DataFrame,
    *,
    score_column: str = "ranker_score",
    retrieval_rank_column: str = "retrieval_rank",
) -> pd.DataFrame:
    """Validate and order candidates by the preregistered V4.11 tie-break."""

    if candidates.empty:
        return candidates.copy()
    required = {
        "candidate_siret",
        score_column,
        retrieval_rank_column,
    }
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(
            "STOP_DATASET_INTEGRITY: missing candidate scene columns "
            f"{sorted(missing)}"
        )
    ranked = candidates.copy()
    ranked["candidate_siret"] = ranked["candidate_siret"].map(_normalise_siret)
    ranked["candidate_siren"] = [
        _normalise_siren(raw, siret=siret)
        for raw, siret in zip(
            (
                ranked["candidate_siren"]
                if "candidate_siren" in ranked.columns
                else pd.Series([None] * len(ranked), index=ranked.index)
            ),
            ranked["candidate_siret"],
            strict=True,
        )
    ]
    ranked["_v411_ranker_score"] = [
        _required_finite(value, name=score_column)
        for value in ranked[score_column]
    ]
    ranked["_v411_retrieval_rank"] = [
        _required_finite(value, name=retrieval_rank_column)
        for value in ranked[retrieval_rank_column]
    ]
    invalid_ranks = (
        ranked["_v411_retrieval_rank"].le(0)
        | ranked["_v411_retrieval_rank"].mod(1).ne(0)
    )
    if invalid_ranks.any():
        raise ValueError(
            "STOP_DATASET_INTEGRITY: retrieval ranks must be positive integers"
        )
    if ranked["candidate_siret"].duplicated().any():
        raise ValueError(
            "STOP_DATASET_INTEGRITY: duplicate SIRET in a V4.11 candidate pool"
        )
    return ranked.sort_values(
        ["_v411_ranker_score", "_v411_retrieval_rank", "candidate_siret"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _normalised_score_features(ranked: pd.DataFrame) -> dict[str, float]:
    if ranked.empty:
        return {name: 0.0 for name in V411_SCENE_FEATURE_NAMES}

    scores = ranked["_v411_ranker_score"].to_numpy(dtype=np.float64)
    count = len(scores)
    score_range = float(scores[0] - scores[-1])
    if count == 1:
        gap = 1.0
        top3_gap = 1.0
        score_std = 0.0
        entropy = 0.0
    elif score_range <= 1e-12:
        gap = 0.0
        top3_gap = 0.0
        score_std = 0.0
        entropy = 1.0
    else:
        gap = float((scores[0] - scores[1]) / score_range)
        top3_tail = scores[1 : min(3, count)]
        top3_gap = float((scores[0] - float(top3_tail.mean())) / score_range)
        score_std = float(scores.std(ddof=0) / score_range)
        normalised = (scores - scores[-1]) / score_range
        shifted = normalised - float(normalised.max())
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum()
        raw_entropy = -float(
            np.sum(probabilities * np.log(np.clip(probabilities, 1e-300, None)))
        )
        entropy = raw_entropy / math.log(count)

    top1_siren = str(ranked.iloc[0]["candidate_siren"])
    top1_siren_rows = ranked[ranked["candidate_siren"].eq(top1_siren)]
    other_sirens = ranked[~ranked["candidate_siren"].eq(top1_siren)]
    siblings = top1_siren_rows.iloc[1:]
    if count == 1:
        siren_gap = 1.0
        sibling_gap = 1.0
    elif score_range <= 1e-12:
        siren_gap = 0.0
        sibling_gap = 0.0
    else:
        siren_gap = (
            1.0
            if other_sirens.empty
            else float(
                (scores[0] - other_sirens["_v411_ranker_score"].max())
                / score_range
            )
        )
        sibling_gap = (
            1.0
            if siblings.empty
            else float(
                (scores[0] - siblings["_v411_ranker_score"].max())
                / score_range
            )
        )

    first_rank_recip = 1.0 / float(ranked.iloc[0]["_v411_retrieval_rank"])
    second_rank_recip = (
        1.0 / float(ranked.iloc[1]["_v411_retrieval_rank"])
        if count > 1
        else 0.0
    )
    crm_school = _candidate_feature(
        ranked.iloc[0].get("is_crm_school"),
        name="is_crm_school",
    )
    return {
        "candidate_count": float(count),
        "ranker_gap_fraction": gap,
        "ranker_top3_gap_fraction": top3_gap,
        "ranker_score_std_fraction": score_std,
        "ranker_score_entropy": entropy,
        "unique_siren_count": float(ranked["candidate_siren"].nunique()),
        "top1_siren_candidate_count": float(len(top1_siren_rows)),
        "same_siren_top2": float(
            count > 1 and str(ranked.iloc[1]["candidate_siren"]) == top1_siren
        ),
        "siren_gap_fraction": siren_gap,
        "retrieval_rank_top1_recip": first_rank_recip,
        "retrieval_rank_gap_recip": first_rank_recip - second_rank_recip,
        "same_siren_best_sibling_gap_fraction": sibling_gap,
        "crm_is_school": crm_school,
    }


def _candidate_texts(row: Mapping[str, Any]) -> list[Any]:
    # Keep the exact V4.10/V4.9 site-function serialization.
    return [
        row.get("enseigne1"),
        row.get("enseigne2"),
        row.get("enseigne3"),
        row.get("denomination_usuelle"),
    ]


def _activity_code(row: Mapping[str, Any]) -> Any:
    for name in (
        "activity_code",
        "activitePrincipaleEtablissement",
        "activite_principale",
    ):
        value = row.get(name)
        if not _missing(value) and str(value).strip():
            return value
    return None


def _naf_division(value: Any) -> str:
    code = re.sub(r"[^A-Z0-9]", "", "" if _missing(value) else str(value).upper())
    digits = "".join(character for character in code if character.isdigit())
    return digits[:2] if len(digits) >= 2 else "UNKNOWN"


def _empty_detection() -> FunctionDetection:
    return FunctionDetection(
        roles=(),
        matched_patterns=(),
        matched_activity_codes=(),
    )


def _role_features(
    query: Mapping[str, Any],
    ranked: pd.DataFrame,
    taxonomy: SiteFunctionTaxonomy,
) -> dict[str, float]:
    crm = taxonomy.detect(
        [
            query.get("crm_name"),
            query.get("crm_address"),
            query.get("crm_city"),
        ]
    )
    if ranked.empty:
        return {
            "role_crm_count": float(len(crm.roles)),
            "role_top1_count": 0.0,
            "role_crm_top1_conflict": 0.0,
            "role_top1_top2_conflict": 0.0,
            "same_siren_distinct_role_count": 0.0,
            "same_siren_role_plurality": 0.0,
            "naf_top1_top2_division_equal": 0.0,
        }

    records = ranked.to_dict("records")
    top1_detection = taxonomy.detect(
        _candidate_texts(records[0]),
        activity_code=_activity_code(records[0]),
    )
    top2_detection = (
        taxonomy.detect(
            _candidate_texts(records[1]),
            activity_code=_activity_code(records[1]),
        )
        if len(records) > 1
        else _empty_detection()
    )
    top1_siren = str(records[0]["candidate_siren"])
    sibling_roles: set[str] = set()
    for record in records:
        if str(record["candidate_siren"]) == top1_siren:
            sibling_roles.update(
                taxonomy.detect(
                    _candidate_texts(record),
                    activity_code=_activity_code(record),
                ).roles
            )
    return {
        "role_crm_count": float(len(crm.roles)),
        "role_top1_count": float(len(top1_detection.roles)),
        "role_crm_top1_conflict": float(
            taxonomy.guard(crm, top1_detection).review
        ),
        "role_top1_top2_conflict": float(
            len(records) > 1
            and taxonomy.guard(top1_detection, top2_detection).review
        ),
        "same_siren_distinct_role_count": float(len(sibling_roles)),
        "same_siren_role_plurality": float(len(sibling_roles) > 1),
        "naf_top1_top2_division_equal": float(
            len(records) > 1
            and _naf_division(_activity_code(records[0]))
            == _naf_division(_activity_code(records[1]))
        ),
    }


def build_v411_compact_scene_features(
    query: Mapping[str, Any],
    candidates: pd.DataFrame,
    taxonomy: SiteFunctionTaxonomy,
    *,
    score_column: str = "ranker_score",
    retrieval_rank_column: str = "retrieval_rank",
) -> dict[str, float]:
    """Return the exact ordered 80-feature V4.11 acceptor scene."""

    ranked = rank_v411_candidates(
        candidates,
        score_column=score_column,
        retrieval_rank_column=retrieval_rank_column,
    )
    scene = _normalised_score_features(ranked)
    top1 = ranked.iloc[0] if not ranked.empty else None
    top2 = ranked.iloc[1] if len(ranked) > 1 else None
    for base in V411_EVIDENCE_BASE_FEATURE_NAMES:
        first = (
            _candidate_feature(top1.get(base), name=base)
            if top1 is not None
            else 0.0
        )
        second = (
            _candidate_feature(top2.get(base), name=base)
            if top2 is not None
            else 0.0
        )
        scene[f"top1_{base}"] = first
        scene[f"delta_{base}"] = first - second
    scene.update(_role_features(query, ranked, taxonomy))
    validate_v411_feature_mapping(scene)
    return {name: float(scene[name]) for name in V411_ACCEPTOR_FEATURE_NAMES}


def build_v411_compact_scene(
    query: Mapping[str, Any],
    candidates: pd.DataFrame,
    taxonomy: SiteFunctionTaxonomy,
    *,
    score_column: str = "ranker_score",
    retrieval_rank_column: str = "retrieval_rank",
) -> dict[str, Any]:
    """Return prediction metadata and the same 80 features used by the model."""

    ranked = rank_v411_candidates(
        candidates,
        score_column=score_column,
        retrieval_rank_column=retrieval_rank_column,
    )
    features = build_v411_compact_scene_features(
        query,
        ranked,
        taxonomy,
        score_column="_v411_ranker_score",
        retrieval_rank_column="_v411_retrieval_rank",
    )
    return {
        "predicted_siret": (
            None if ranked.empty else str(ranked.iloc[0]["candidate_siret"])
        ),
        "predicted_siren": (
            None if ranked.empty else str(ranked.iloc[0]["candidate_siren"])
        ),
        **features,
    }


def validate_v411_feature_order(feature_order: Sequence[str]) -> None:
    """Fail closed on drift, aliases, leakage, or duplicated feature names."""

    observed = list(feature_order)
    if observed != V411_ACCEPTOR_FEATURE_NAMES:
        raise ValueError(
            "STOP_DATASET_INTEGRITY: V4.11 acceptor feature order drift"
        )
    if len(observed) != 80 or len(set(observed)) != 80:
        raise ValueError(
            "STOP_DATASET_INTEGRITY: V4.11 requires 80 unique features"
        )
    leaked = sorted(
        name
        for name in observed
        if name.startswith("top2_")
        or any(fragment in name for fragment in _FORBIDDEN_FEATURE_FRAGMENTS)
    )
    if leaked:
        raise ValueError(
            "STOP_DATASET_INTEGRITY: forbidden or aliased V4.11 features "
            f"{leaked}"
        )
    for base in V411_EVIDENCE_BASE_FEATURE_NAMES:
        if observed.count(f"top1_{base}") != 1 or observed.count(f"delta_{base}") != 1:
            raise ValueError(
                "STOP_DATASET_INTEGRITY: evidence must have one top1/delta pair"
            )


def validate_v411_feature_mapping(features: Mapping[str, Any]) -> None:
    """Validate the exact train/serve feature contract for one scene."""

    validate_v411_feature_order(V411_ACCEPTOR_FEATURE_NAMES)
    observed = list(features)
    if observed != V411_ACCEPTOR_FEATURE_NAMES:
        raise ValueError(
            "STOP_DATASET_INTEGRITY: V4.11 scene keys/order changed"
        )
    for name in V411_ACCEPTOR_FEATURE_NAMES:
        _required_finite(features[name], name=name)


def validate_v411_scene_frame(frame: pd.DataFrame) -> None:
    """Validate a materialised scene frame while allowing metadata columns."""

    missing = [
        name for name in V411_ACCEPTOR_FEATURE_NAMES if name not in frame.columns
    ]
    if missing:
        raise ValueError(
            f"STOP_DATASET_INTEGRITY: missing V4.11 scene features {missing}"
        )
    matrix = frame[V411_ACCEPTOR_FEATURE_NAMES].to_numpy(dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError(
            "STOP_DATASET_INTEGRITY: non-finite V4.11 scene feature matrix"
        )
    for name in V411_BINARY_FEATURE_NAMES:
        allowed = {-1.0, 0.0, 1.0} if name.startswith("delta_") else {0.0, 1.0}
        if not set(frame[name].astype(float).unique()).issubset(allowed):
            raise ValueError(
                f"STOP_DATASET_INTEGRITY: invalid binary feature domain for {name}"
            )


def assert_v411_train_serve_parity(
    train_features: Mapping[str, Any],
    serve_features: Mapping[str, Any],
) -> None:
    """Require bit-identical ordered values from the two call sites."""

    validate_v411_feature_mapping(train_features)
    validate_v411_feature_mapping(serve_features)
    train = np.asarray(
        [train_features[name] for name in V411_ACCEPTOR_FEATURE_NAMES],
        dtype=np.float64,
    )
    serve = np.asarray(
        [serve_features[name] for name in V411_ACCEPTOR_FEATURE_NAMES],
        dtype=np.float64,
    )
    if not np.array_equal(train, serve):
        changed = [
            name
            for name, left, right in zip(
                V411_ACCEPTOR_FEATURE_NAMES, train, serve, strict=True
            )
            if left != right
        ]
        raise ValueError(
            "STOP_DATASET_INTEGRITY: V4.11 train/serve scene drift "
            f"{changed}"
        )


validate_v411_feature_order(V411_ACCEPTOR_FEATURE_NAMES)
if len(V411_MONOTONIC_CONSTRAINTS) != 80:
    raise AssertionError("V4.11 monotonic vector must contain exactly 80 values")
if set(V411_MONOTONIC_CONSTRAINTS) - {-1, 0, 1}:
    raise AssertionError("V4.11 monotonic vector has an invalid value")


__all__ = [
    "V411_ACCEPTOR_FEATURE_NAMES",
    "V411_BINARY_EVIDENCE_BASES",
    "V411_BINARY_FEATURE_NAMES",
    "V411_EVIDENCE_BASE_FEATURE_NAMES",
    "V411_EVIDENCE_FEATURE_NAMES",
    "V411_MONOTONIC_CONSTRAINTS",
    "V411_ROLE_FEATURE_NAMES",
    "V411_SCALED_FEATURE_NAMES",
    "V411_SCENE_FEATURE_NAMES",
    "assert_v411_train_serve_parity",
    "build_v411_compact_scene",
    "build_v411_compact_scene_features",
    "rank_v411_candidates",
    "validate_v411_feature_mapping",
    "validate_v411_feature_order",
    "validate_v411_scene_frame",
]
