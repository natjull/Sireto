#!/usr/bin/env python3
"""Build the canonical V4.10 structured acceptor matrices.

This milestone only materialises features.  It deliberately does not fit,
score, calibrate, or open any random/test/holdout population.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_v41_training_dataset import FEATURE_ORDER as V41_CANDIDATE_FEATURES  # noqa: E402
from src.xgb_matcher.features import (  # noqa: E402
    NAME_STOPWORDS,
    jaro_sim,
    normalize_name,
    preprocess_crm_row,
)
from src.xgb_matcher.naming import normalize_text  # noqa: E402
from src.xgb_matcher.v49_site_function import SiteFunctionTaxonomy  # noqa: E402
from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402
from src.xgb_matcher.v9_scene import V9_SCENE_FEATURE_NAMES  # noqa: E402


SCHEMA_VERSION = "sireto-v4.10-structured-acceptor-dataset-1"
EXPERIMENT_ID = "V410_STRUCTURED_ACCEPTOR"
FEATURE_POLICY_VERSION = "v4.10-structured-features-1"
SEED = 42
MAX_CANDIDATES = 100
EXPECTED_CONTRACT_SHA256 = (
    "91527a57271e5a9410dc6555b6264c817dd2c20d3ce4af1a1903abb6b1f878c4"
)
EXPECTED_CONTRACT_COMMIT = "b19abed55f76861e5fe5b78c59143ee01f584402"
EXPECTED_INPUT_HASHES = {
    "historical_scenes": "8f3bc4633ada9eb6347e47a1029f0e69fa8946b1c3c1df38c72232f572088dc9",
    "historical_predictions": "eea22c58378d8adc232a7f2723c0a84323963db9633a7bb9af2e2485cd6329d2",
    "historical_queries": "6a12f1c4ca9ec33636ebcf7748c208595c6168d7cdb8c068e1434af3fe22abb0",
    "historical_candidates": "34b526fe49e3451c05248294305e4a8d6ccf4db92277eb36dc03cc6231420c67",
    "hard_candidates": "9f48a558bc77bf9db835e7689963989ba99d2914fb1add32be4988ec3cab3242",
    "hard_scenes": "72540dcdba6f33da0eb1875ef4bcdc8c44a2cd10083589b5e1683098cd954a08",
    "current_labels": "e5e592d4dcd540273378dada7128f957b1d335df63fbc88f4c1377c0f9337bd2",
    "hard_queue": "47af4887769a2edb11f1e629c38077edccd035dd96cb3a6d39620714361fdecc",
    "partition_assignments": "f828249172c36ce33a3279d294dfc5030e6d8eeb58baee9cf9e08130f13593b9",
    "partition_manifest": "f0e255b891dfb6b24d57f3b7423dd64a227908dbf68559b2da4572ea37791d33",
    "sirene_snapshot": "c91180cc5bae86948dd57d752c9bae45e58cc64653e99d5a9357664b67300845",
    "taxonomy": "48bbb7e1795a0731f1f12df41aeb971667c10d03c879bf06d5ba15b65f8b121d",
}
DEFAULT_CONTRACT = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "v4_10_structured_acceptor_contract.md"
)
SEMANTIC_SCENE_FEATURES = tuple(
    name
    for name in V9_SCENE_FEATURE_NAMES
    if "name_semantic_" in name
)
CURRENT80_FEATURES = tuple(V9_SCENE_FEATURE_NAMES)
SCENE_FEATURES = tuple(
    name for name in V9_SCENE_FEATURE_NAMES if name not in SEMANTIC_SCENE_FEATURES
)
CATEGORICAL_CANDIDATE_FEATURES = {
    "type_of_max_name": tuple(str(value) for value in range(9)),
    "legal_form_category": ("-1", "0", "1"),
    "candidate_state": ("A", "F"),
}
NUMERIC_CANDIDATE_FEATURES = tuple(
    [
        "ranker_score",
        "rank",
        "retrieval_channel_count",
        "retrieval_agreement",
    ]
    + [
        name
        for name in V41_CANDIDATE_FEATURES
        if name not in CATEGORICAL_CANDIDATE_FEATURES
    ]
)
NAF_SECTIONS = tuple("ABCDEFGHIJKLMNOPQRSTU")
NAF_DIVISIONS = tuple(f"{value:02d}" for value in range(100))
NAF_SECTION_RANGES = (
    ("A", 1, 3), ("B", 5, 9), ("C", 10, 33), ("D", 35, 35),
    ("E", 36, 39), ("F", 41, 43), ("G", 45, 47), ("H", 49, 53),
    ("I", 55, 56), ("J", 58, 63), ("K", 64, 66), ("L", 68, 68),
    ("M", 69, 75), ("N", 77, 82), ("O", 84, 84), ("P", 85, 85),
    ("Q", 86, 88), ("R", 90, 93), ("S", 94, 96), ("T", 97, 98),
    ("U", 99, 99),
)
ROLE_NAMES = (
    "ADMIN_MAIRIE",
    "EDU_MATERNELLE",
    "EDU_PRIMAIRE",
    "EDU_COLLEGE",
    "EDU_LYCEE",
    "CHILDCARE_CRECHE",
    "MED_FAM",
    "MED_MAS",
    "MED_EAM",
    "MED_IME",
    "MED_EHPAD",
    "MED_FOYER",
    "HEALTH_HOSPITAL",
    "HEALTH_CLINIC",
    "HEALTH_PHARMACY",
)
INTERACTION_THRESHOLDS = {
    "strong_address_full_score_min": 0.85,
    "weak_name_jaro_max": 0.65,
    "weak_name_token_overlap_max": 0.50,
}
DRIFT_AUDIT_ONLY_BASE_FEATURES = {
    *(
        name
        for name in V41_CANDIDATE_FEATURES
        if name.startswith("admission_")
    ),
    "candidate_from_sparse",
    "candidate_from_input_siret",
    "candidate_from_input_siren",
    "candidate_from_closed_alias",
}
BOOLEAN_CANDIDATE_BASE_FEATURES = {
    "has_any_name",
    "numeric_token_match",
    "name_first_word_match_max",
    "name_contains_crm_max",
    "name_crm_contains_cand_max",
    "acronym_match_max",
    "is_ul_name_max",
    "is_sigle_max",
    "has_person_name",
    "postcode_match",
    "city_match",
    "is_siege",
    "is_association",
    "alias_match",
    "is_crm_school",
    "geo_exact_match",
    "name_norm_exact",
    "street_number_match",
    "input_siret_exact_match",
    "input_siren_exact_match",
    "candidate_is_active",
    "candidate_is_closed",
    "candidate_state_unknown",
    "candidate_from_sparse",
    "candidate_from_input_siret",
    "candidate_from_input_siren",
    "candidate_from_closed_alias",
    "retrieval_agreement",
}
CURRENT80_BINARY_FEATURES = {
    "has_candidate",
    "same_siren_top2",
    "top1_retrieval_agreement",
    "sparse_dense_top1_agreement",
    "retrieval_disagreement",
    "retrieval_miss",
    *(
        f"{position}_{base}"
        for base in (
            "numeric_token_match",
            "postcode_match",
            "city_match",
        )
        for position in ("top1", "top2")
    ),
}
CURRENT80_COUNT_FEATURES = {
    "candidate_count",
    "unique_siren_count",
    "top1_siren_candidate_count",
    "top1_retrieval_channel_count",
}
FORBIDDEN_SOURCE_PATH = re.compile(
    r"(?:^|[/_.-])(test|holdout|fresh|random)(?:$|[/_.-])",
    re.IGNORECASE,
)


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_bytes(_json_bytes(payload))


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _siret(value: Any) -> str | None:
    digits = "".join(character for character in _text(value) if character.isdigit())
    return digits.zfill(14) if digits else None


def _finite(value: Any) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return 0.0
    return output if math.isfinite(output) else 0.0


def _strict_finite(value: Any, *, name: str) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"STOP_DATASET_INTEGRITY: {name} is not numeric") from error
    if not math.isfinite(output):
        raise ValueError(f"STOP_DATASET_INTEGRITY: {name} is not finite")
    return output


def _numeric_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return True
    if math.isnan(numeric):
        return True
    if math.isinf(numeric):
        raise ValueError(
            "STOP_DATASET_INTEGRITY: infinite candidate feature value"
        )
    return False


def _assert_authorized_path(path: Path, *, name: str) -> Path:
    resolved = Path(path).resolve()
    if FORBIDDEN_SOURCE_PATH.search(str(resolved)):
        raise ValueError(f"{name} points to a forbidden population: {resolved}")
    return resolved


def validate_frozen_inputs(paths: Mapping[str, Path], contract_path: Path) -> dict[str, Any]:
    """Hash every authorised source before any population is materialised."""

    if file_sha256(contract_path) != EXPECTED_CONTRACT_SHA256:
        raise ValueError("V4.10 contract hash mismatch")
    if set(paths) != set(EXPECTED_INPUT_HASHES):
        raise ValueError("V4.10 source set differs from the preregistered contract")
    records: dict[str, Any] = {}
    for name, raw_path in paths.items():
        path = _assert_authorized_path(raw_path, name=name)
        observed = file_sha256(path)
        if observed != EXPECTED_INPUT_HASHES[name]:
            raise ValueError(f"Frozen V4.10 input hash mismatch: {name}")
        records[name] = {
            "path": str(path),
            "sha256": observed,
            "size_bytes": int(path.stat().st_size),
        }
    return records


def read_targeted_parquet(path: Path, *, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Read only targeted hard cases at the Parquet scan boundary."""

    output = pd.read_parquet(
        path,
        columns=list(columns) if columns is not None else None,
        filters=[("sampling_stratum", "!=", "RANDOM_POPULATION")],
    )
    if "sampling_stratum" not in output.columns:
        raise ValueError("Filtered hard source lost sampling_stratum")
    if output["sampling_stratum"].astype(str).eq("RANDOM_POPULATION").any():
        raise ValueError("A random V4.8 row crossed the scan boundary")
    return output


def prepare_historical_candidates(
    predictions: pd.DataFrame,
    frozen_candidates: pd.DataFrame,
    *,
    enforce_contract_counts: bool = True,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Bind frozen ranker-A order to the exact V4.1 candidate rows."""

    keys = ["query_id", "candidate_siret"]
    left = predictions.copy()
    right = frozen_candidates.copy()
    sentinel_count = int(left["candidate_siret"].isna().sum())
    left = left[left["candidate_siret"].notna()].copy()
    for frame in (left, right):
        frame["query_id"] = frame["query_id"].astype(str)
        frame["candidate_siret"] = frame["candidate_siret"].map(_siret)
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise ValueError("Historical candidate keys must be unique")
    left = left.rename(columns={"score": "ranker_score"})[
        keys + ["ranker_score", "rank", "prediction_origin", "fold"]
    ]
    right_columns = keys + [
        "candidate_siren",
        "candidate_state",
        "retrieval_rank",
        "retrieval_source",
        "retrieval_channel_count",
        "retrieval_agreement",
    ] + [
        column
        for column in right.columns
        if column in V41_CANDIDATE_FEATURES
    ]
    merged = left.merge(
        right[right_columns],
        on=keys,
        how="left",
        validate="one_to_one",
        indicator=True,
        suffixes=("", "_aligned"),
    )
    candidate_counts = merged.groupby("query_id").size()
    if int(candidate_counts.max()) > MAX_CANDIDATES:
        raise ValueError("Historical candidate ceiling exceeds 100")
    report = {
        "row_count": int(len(merged)),
        "candidate_prediction_joined_count": int(merged["_merge"].eq("both").sum()),
        "candidate_prediction_join_rate": float(merged["_merge"].eq("both").mean()),
        "no_candidate_sentinel_count": sentinel_count,
    }
    if (
        (enforce_contract_counts and len(merged) != 698_428)
        or (enforce_contract_counts and sentinel_count != 2)
        or not merged["_merge"].eq("both").all()
    ):
        raise ValueError(
            "The 698,428 V4.1 candidate/prediction pairs must join exactly"
        )
    return merged.drop(columns="_merge"), report


def prepare_hard_candidates(candidates: pd.DataFrame, allowed_ids: set[str]) -> pd.DataFrame:
    output = candidates[
        candidates["audit_case_id"].astype(str).isin(allowed_ids)
    ].copy()
    output["query_id"] = output["audit_case_id"].astype(str)
    output["candidate_siret"] = output["candidate_siret"].map(_siret)
    output = output.rename(columns={"ranker_score": "ranker_score"})
    if output.duplicated(["query_id", "candidate_siret"]).any():
        raise ValueError("Hard candidate keys must be unique")
    if not output.empty and int(output.groupby("query_id").size().max()) > MAX_CANDIDATES:
        raise ValueError("Hard candidate ceiling exceeds 100")
    return output


def validate_query_coverage(
    scenes: pd.DataFrame,
    queries: pd.DataFrame,
    *,
    name: str,
) -> dict[str, Any]:
    scene_ids = scenes["query_id"].astype(str)
    query_ids = queries["query_id"].astype(str)
    if scene_ids.duplicated().any() or query_ids.duplicated().any():
        raise ValueError(f"STOP_DATASET_INTEGRITY: duplicate query_id in {name}")
    missing = sorted(set(scene_ids) - set(query_ids))
    extra = sorted(set(query_ids) - set(scene_ids))
    if missing or extra:
        raise ValueError(
            f"STOP_DATASET_INTEGRITY: CRM query coverage differs for {name}; "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    return {
        "scene_count": int(len(scene_ids)),
        "query_count": int(len(query_ids)),
        "joined_count": int(len(scene_ids)),
        "join_rate": 1.0,
    }


def hydrate_sirene(
    candidates: pd.DataFrame,
    snapshot_path: Path,
    *,
    temp_directory: Path,
) -> pd.DataFrame:
    """Read only SIRETs present in actual pools from the authoritative snapshot."""

    ids = pd.DataFrame(
        {"siret": sorted(set(candidates["candidate_siret"].dropna().astype(str)))}
    )
    if ids.empty:
        return pd.DataFrame(columns=["siret"])
    connection = duckdb.connect()
    try:
        connection.execute(
            f"SET temp_directory = '{str(temp_directory).replace(chr(39), chr(39) * 2)}'"
        )
        connection.register("wanted_sirets", ids)
        escaped = str(Path(snapshot_path).resolve()).replace("'", "''")
        return connection.execute(
            f"""
            SELECT
                CAST(s.siret AS VARCHAR) AS siret,
                CAST(s.siren AS VARCHAR) AS siren,
                s.etatAdministratifEtablissement AS registry_state,
                s.enseigne1Etablissement AS enseigne1,
                s.enseigne2Etablissement AS enseigne2,
                s.enseigne3Etablissement AS enseigne3,
                s.denominationUsuelleEtablissement AS denomination_usuelle,
                s.activitePrincipaleEtablissement AS activity_code,
                s.codePostalEtablissement AS registry_postcode,
                s.libelleCommuneEtablissement AS registry_city,
                s.numeroVoieEtablissement AS registry_street_number
            FROM read_parquet('{escaped}') s
            INNER JOIN wanted_sirets w
              ON CAST(s.siret AS VARCHAR) = w.siret
            """
        ).fetchdf()
    finally:
        connection.close()


def validate_registry_coverage(
    candidates: pd.DataFrame,
    registry: pd.DataFrame,
    *,
    name: str,
) -> dict[str, Any]:
    normalized_registry = registry.copy()
    normalized_registry["siret"] = normalized_registry["siret"].map(_siret)
    normalized_registry["siren"] = normalized_registry["siren"].map(
        lambda value: (_siret(value) or "")[:9]
        if len("".join(character for character in _text(value) if character.isdigit())) > 9
        else "".join(character for character in _text(value) if character.isdigit()).zfill(9)
    )
    if normalized_registry["siret"].isna().any():
        raise ValueError("SIRENE snapshot returned an invalid SIRET")
    if normalized_registry["siret"].duplicated().any():
        raise ValueError("SIRENE snapshot join is not unique by SIRET")
    required_sirets: set[str] = set()
    candidate_siren_by_siret: dict[str, str] = {}
    for _, rows in candidates.groupby("query_id", sort=False):
        ranked = _ranked(rows)
        top = ranked.head(2)
        required_sirets.update(top["candidate_siret"].dropna().map(_siret))
        if not ranked.empty:
            top1_siren = _text(ranked.iloc[0]["candidate_siren"])
            constellation = ranked[
                ranked["candidate_siren"].astype(str).eq(top1_siren)
            ]
            required_sirets.update(
                constellation["candidate_siret"].dropna().map(_siret)
            )
        for record in ranked.to_dict("records"):
            candidate_siren_by_siret[_siret(record["candidate_siret"])] = _text(
                record.get("candidate_siren")
            )
    required_sirets.discard(None)
    available = set(normalized_registry["siret"])
    missing = sorted(required_sirets - available)
    if missing:
        raise ValueError(
            f"SIRENE top-1/top-2 coverage is below 100% for {name}: "
            f"{len(missing)} missing"
        )
    selected = normalized_registry[
        normalized_registry["siret"].isin(required_sirets)
    ]
    inconsistent = selected[
        selected["siren"].ne(selected["siret"].str[:9])
        | selected.apply(
            lambda row: candidate_siren_by_siret.get(row["siret"], "")
            != row["siren"],
            axis=1,
        )
    ]
    if not inconsistent.empty:
        raise ValueError("SIRENE SIRET/SIREN coherence failed in required constellation")
    return {
        "required_top1_top2_and_constellation_count": len(required_sirets),
        "joined_top1_top2_and_constellation_count": len(required_sirets),
        "join_rate": 1.0,
        "siren_coherence_rate": 1.0,
    }


def _naf_parts(value: Any) -> tuple[str, str]:
    code = re.sub(r"[^A-Z0-9]", "", _text(value).upper())
    digits = "".join(character for character in code if character.isdigit())
    division = digits[:2] if len(digits) >= 2 else "UNKNOWN"
    section = "UNKNOWN"
    if division != "UNKNOWN":
        number = int(division)
        section = next(
            (
                name
                for name, lower, upper in NAF_SECTION_RANGES
                if lower <= number <= upper
            ),
            "UNKNOWN",
        )
    return section, division


def _candidate_texts(row: Mapping[str, Any]) -> list[Any]:
    return [
        row.get("enseigne1"),
        row.get("enseigne2"),
        row.get("enseigne3"),
        row.get("denomination_usuelle"),
    ]


def _ranked(rows: pd.DataFrame) -> pd.DataFrame:
    output = rows.copy()
    rank_column = "rank" if "rank" in output.columns else "retrieval_rank"
    output["_rank"] = pd.to_numeric(output[rank_column], errors="coerce").fillna(10_000)
    output["ranker_score"] = pd.to_numeric(
        output.get("ranker_score"), errors="coerce"
    ).fillna(0.0)
    return output.sort_values(["_rank", "ranker_score"], ascending=[True, False])


def _encode_numeric_pair(
    output: dict[str, Any],
    top1: Mapping[str, Any],
    top2: Mapping[str, Any] | None,
    feature: str,
) -> None:
    values: list[float] = []
    for position, row in (("top1", top1), ("top2", top2)):
        raw = None if row is None else row.get(feature)
        missing = _numeric_missing(raw)
        value = 0.0 if missing else float(raw)
        output[f"candidate_{position}_{feature}"] = value
        output[f"candidate_{position}_{feature}_missing"] = float(missing)
        values.append(value)
    output[f"candidate_delta_{feature}"] = values[0] - values[1]


def _category(value: Any, allowed: Sequence[str]) -> str:
    text = _text(value)
    if text:
        try:
            numeric = float(text)
            text = str(int(numeric)) if numeric.is_integer() else text
        except ValueError:
            pass
    return text if text in allowed else "UNKNOWN"


def _encode_category_pair(
    output: dict[str, Any],
    top1: Mapping[str, Any],
    top2: Mapping[str, Any] | None,
    feature: str,
    categories: Sequence[str],
) -> None:
    first = _category(top1.get(feature), categories)
    second = _category(None if top2 is None else top2.get(feature), categories)
    for position, value in (("top1", first), ("top2", second)):
        for category in (*categories, "UNKNOWN"):
            output[f"candidate_{position}_{feature}__{category}"] = float(
                value == category
            )
    output[f"candidate_{feature}_top1_top2_equal"] = float(first == second)


def _v8_interactions(row: Mapping[str, Any], query: Mapping[str, Any]) -> dict[str, float]:
    addr = _finite(row.get("addr_jaro"))
    name = _finite(row.get("name_jaro_max"))
    density = _finite(row.get("address_density"))
    postcode = _finite(row.get("postcode_match"))
    street = _finite(row.get("street_name_jaro"))
    street_number = _finite(row.get("street_number_match"))
    enseigne = _text(row.get("enseigne1") or row.get("denomination_usuelle"))
    crm_preprocessed = preprocess_crm_row(query) if query else {}
    crm_name = _text(crm_preprocessed.get("crm_name"))
    crm_city_tokens = (
        set(normalize_text(_text(query.get("crm_city"))).split()) - NAME_STOPWORDS
    )
    enseigne_tokens = set(normalize_text(enseigne).split())
    return {
        "addr_unsupported_by_name": addr * (1.0 - name),
        "name_density_penalty": min(density, 100.0) / 100.0 * (1.0 - name),
        "addr_jaro_per_density": addr / math.log1p(max(1.0, density)),
        "postcode_match_without_addr": postcode * (1.0 - street),
        "full_addr_match_score": 0.5 * street + 0.3 * street_number + 0.2 * postcode,
        "name_jaro_vs_enseigne": (
            jaro_sim(crm_name, normalize_name(enseigne)) if crm_name and enseigne else 0.0
        ),
        "name_city_suffix_match": (
            len(crm_city_tokens & enseigne_tokens) / len(crm_city_tokens)
            if crm_city_tokens and enseigne_tokens
            else 0.0
        ),
    }


def _activity_and_roles(
    output: dict[str, Any],
    query: Mapping[str, Any],
    top1: Mapping[str, Any],
    top2: Mapping[str, Any] | None,
    siblings: pd.DataFrame,
    taxonomy: SiteFunctionTaxonomy,
) -> tuple[set[str], set[str]]:
    crm = taxonomy.detect(
        [query.get("crm_name"), query.get("crm_address"), query.get("crm_city")]
    )
    detections = []
    for row in (top1, top2):
        detections.append(
            taxonomy.detect(
                [] if row is None else _candidate_texts(row),
                activity_code=None if row is None else row.get("activity_code"),
            )
        )
    first_section, first_division = _naf_parts(top1.get("activity_code"))
    second_section, second_division = _naf_parts(
        None if top2 is None else top2.get("activity_code")
    )
    for position, value in (("top1", first_section), ("top2", second_section)):
        for category in (*NAF_SECTIONS, "UNKNOWN"):
            output[f"naf_{position}_section__{category}"] = float(value == category)
    for position, value in (("top1", first_division), ("top2", second_division)):
        for category in (*NAF_DIVISIONS, "UNKNOWN"):
            output[f"naf_{position}_division__{category}"] = float(value == category)
    output["naf_top1_top2_section_equal"] = float(first_section == second_section)
    output["naf_top1_top2_division_equal"] = float(first_division == second_division)
    output["naf_top1_division_missing"] = float(first_division == "UNKNOWN")
    output["naf_top2_division_missing"] = float(second_division == "UNKNOWN")
    for role in ROLE_NAMES:
        output[f"role_crm__{role}"] = float(role in crm.roles)
        output[f"role_top1__{role}"] = float(role in detections[0].roles)
        output[f"role_top2__{role}"] = float(role in detections[1].roles)
    output["role_crm_count"] = float(len(crm.roles))
    output["role_top1_count"] = float(len(detections[0].roles))
    output["role_top2_count"] = float(len(detections[1].roles))
    output["role_crm_top1_conflict"] = float(
        taxonomy.guard(crm, detections[0]).review
    )
    output["role_top1_top2_conflict"] = float(
        taxonomy.guard(detections[0], detections[1]).review
    )
    sibling_divisions = {_naf_parts(value)[1] for value in siblings["activity_code"]}
    sibling_divisions.discard("UNKNOWN")
    sibling_roles: set[str] = set()
    for record in siblings.to_dict("records"):
        sibling_roles.update(
            taxonomy.detect(
                _candidate_texts(record),
                activity_code=record.get("activity_code"),
            ).roles
        )
    output["same_siren_distinct_division_count"] = float(len(sibling_divisions))
    output["same_siren_distinct_role_count"] = float(len(sibling_roles))
    output["same_siren_role_plurality"] = float(len(sibling_roles) > 1)
    return set(crm.roles), set(detections[0].roles)


def _constellation(
    output: dict[str, Any],
    top1: Mapping[str, Any],
    same_siren_group: pd.DataFrame,
) -> None:
    top1_siret = _siret(top1.get("candidate_siret"))
    siblings = same_siren_group[
        same_siren_group["candidate_siret"].map(_siret).ne(top1_siret)
    ].copy()
    output["same_siren_candidate_count"] = float(len(same_siren_group))
    output["same_siren_sibling_count"] = float(len(siblings))
    output["same_siren_has_sibling"] = float(not siblings.empty)
    output["same_siren_active_count"] = float(
        pd.to_numeric(
            same_siren_group.get("candidate_is_active"), errors="coerce"
        )
        .fillna(0)
        .sum()
    )
    def best_site(rows: pd.DataFrame, metric: str) -> str | None:
        if rows.empty:
            return None
        ordered = rows.assign(
            _metric=pd.to_numeric(rows[metric], errors="coerce").fillna(0.0),
            _score=pd.to_numeric(
                rows["ranker_score"], errors="coerce"
            ).fillna(0.0),
            _siret=rows["candidate_siret"].fillna("").astype(str),
        ).sort_values(
            ["_metric", "_score", "_rank", "_siret"],
            ascending=[False, False, True, True],
            kind="mergesort",
        )
        return str(ordered.iloc[0]["candidate_siret"])

    for feature in ("name_jaro_max", "addr_jaro", "ranker_score"):
        values = pd.to_numeric(
            same_siren_group[feature], errors="coerce"
        ).fillna(0.0)
        ordered = values.sort_values(ascending=False).to_numpy()
        best = float(ordered[0]) if len(ordered) else 0.0
        second = float(ordered[1]) if len(ordered) > 1 else 0.0
        output[f"same_siren_best_{feature}"] = best
        output[f"same_siren_second_{feature}"] = second
        sibling_values = pd.to_numeric(
            siblings[feature], errors="coerce"
        ).fillna(0.0)
        sibling_best = (
            float(sibling_values.max()) if not sibling_values.empty else 0.0
        )
        output[f"same_siren_best_sibling_{feature}"] = sibling_best
        output[f"top1_gap_to_same_siren_best_{feature}"] = (
            _finite(top1.get(feature)) - sibling_best
        )
        output[f"top1_is_same_siren_best_{feature}"] = float(
            _finite(top1.get(feature)) >= best
        )
    for feature in ("postcode_match", "city_match", "street_number_match"):
        output[f"same_siren_{feature}_count"] = float(
            (
                pd.to_numeric(
                    same_siren_group[feature], errors="coerce"
                ).fillna(0)
                >= 1.0
            ).sum()
        )
    if siblings.empty:
        output["same_siren_best_sibling_name_geo_disagreement"] = 0.0
    else:
        output["same_siren_best_sibling_name_geo_disagreement"] = float(
            best_site(siblings, "name_jaro_max")
            != best_site(siblings, "addr_jaro")
        )
    output["same_siren_name_geo_best_disagreement"] = float(
        best_site(same_siren_group, "name_jaro_max")
        != best_site(same_siren_group, "addr_jaro")
    )


def build_structured_scenes(
    scenes: pd.DataFrame,
    candidates: pd.DataFrame,
    queries: pd.DataFrame,
    registry: pd.DataFrame,
    taxonomy: SiteFunctionTaxonomy,
    *,
    population: str,
) -> pd.DataFrame:
    """Create one numeric V4.10 row per supplied scene."""

    query_map = {
        str(record["query_id"]): record for record in queries.to_dict("records")
    }
    registry_for_join = registry.rename(
        columns={"siret": "candidate_siret", "siren": "registry_siren"}
    )
    enriched = candidates.merge(
        registry_for_join,
        on="candidate_siret",
        how="left",
        validate="many_to_one",
    )
    grouped = {
        str(query_id): _ranked(group)
        for query_id, group in enriched.groupby("query_id", sort=False)
    }
    output_rows: list[dict[str, Any]] = []
    for scene in scenes.to_dict("records"):
        query_id = str(scene["query_id"])
        ranked = grouped.get(query_id)
        no_candidate = ranked is None or ranked.empty
        if no_candidate:
            ranked = enriched.iloc[0:0].copy()
            top1 = {
                "candidate_siret": None,
                "candidate_siren": "",
                **{name: None for name in NUMERIC_CANDIDATE_FEATURES},
                **{name: None for name in CATEGORICAL_CANDIDATE_FEATURES},
            }
            top2 = None
        else:
            top1 = ranked.iloc[0].to_dict()
            top2 = ranked.iloc[1].to_dict() if len(ranked) > 1 else None
        expected_top1 = _siret(
            scene.get("predicted_siret")
            or scene.get("replayed_top1_siret")
            or scene.get("current_top1_siret")
        )
        if expected_top1 and _siret(top1.get("candidate_siret")) != expected_top1:
            raise ValueError(f"Frozen top-1 mismatch for {query_id}")
        if query_id not in query_map:
            raise ValueError(
                f"STOP_DATASET_INTEGRITY: CRM query missing for {query_id}"
            )
        query = query_map[query_id]
        row: dict[str, Any] = {
            "query_id": query_id,
            "population_role": population,
            "split": scene.get("split"),
            "ranker_prediction_is_out_of_sample": bool(
                scene.get("ranker_prediction_is_out_of_sample")
            ),
            "prediction_origin": _text(scene.get("prediction_origin")),
            "ranker_oof_fold": scene.get("ranker_oof_fold"),
            "top1_siret": _siret(top1.get("candidate_siret")),
            "top1_siren": _text(top1.get("candidate_siren")),
        }
        for feature in CURRENT80_FEATURES:
            row[feature] = _strict_finite(
                scene.get(feature),
                name=f"{query_id}.current80.{feature}",
            )
        for feature in SCENE_FEATURES:
            row[f"scene_{feature}"] = row[feature]
        for feature in NUMERIC_CANDIDATE_FEATURES:
            _encode_numeric_pair(row, top1, top2, feature)
        for feature, categories in CATEGORICAL_CANDIDATE_FEATURES.items():
            _encode_category_pair(row, top1, top2, feature, categories)
        for position, candidate in (("top1", top1), ("top2", top2)):
            interactions = _v8_interactions(candidate or {}, query)
            for name, value in interactions.items():
                row[f"{position}_{name}"] = value
        for name in _v8_interactions(top1, query):
            row[f"delta_{name}"] = (
                row[f"top1_{name}"] - row[f"top2_{name}"]
            )
        normalized_top1 = _siret(top1.get("candidate_siret"))
        top1_siren = _text(top1.get("candidate_siren")) or (
            normalized_top1[:9] if normalized_top1 else ""
        )
        same_siren_group = ranked[
            ranked["candidate_siren"].astype(str).eq(top1_siren)
        ].copy()
        crm_roles, candidate_roles = _activity_and_roles(
            row, query, top1, top2, same_siren_group, taxonomy
        )
        _constellation(row, top1, same_siren_group)
        full_address = row["top1_full_addr_match_score"]
        strong_address = full_address >= INTERACTION_THRESHOLDS[
            "strong_address_full_score_min"
        ]
        weak_name = (
            _finite(top1.get("name_jaro_max"))
            < INTERACTION_THRESHOLDS["weak_name_jaro_max"]
            and _finite(top1.get("name_token_overlap_max"))
            < INTERACTION_THRESHOLDS["weak_name_token_overlap_max"]
        )
        input_siren = _text(query.get("input_siren"))
        row["interaction_strong_address_weak_name"] = float(
            strong_address and weak_name
        )
        row["interaction_different_siren_strong_address"] = float(
            bool(input_siren) and input_siren != top1_siren and strong_address
        )
        row["interaction_same_siren_geo_incompatible"] = float(
            bool(input_siren)
            and input_siren == top1_siren
            and (
                _finite(top1.get("city_match")) < 1.0
                or _finite(top1.get("postcode_match")) < 1.0
            )
        )
        function_conflict = any(
            taxonomy.incompatible(left, right)
            for left in crm_roles
            for right in candidate_roles
        )
        row["interaction_exact_input_siret_function_incompatible"] = float(
            _finite(top1.get("input_siret_exact_match")) >= 1.0
            and function_conflict
        )
        output_rows.append(row)
    output = pd.DataFrame(output_rows)
    if output["query_id"].duplicated().any():
        raise ValueError("V4.10 output must contain one row per query")
    numeric = output.drop(
        columns=["ranker_oof_fold"], errors="ignore"
    ).select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("STOP_DATASET_INTEGRITY: structured matrix contains non-finite")
    return output


def make_feature_catalog(feature_order: Sequence[str]) -> dict[str, Any]:
    metadata = {
        "query_id",
        "population_role",
        "split",
        "acceptor_target",
        "adjudication_label",
        "training_eligible",
        "top1_siret",
        "top1_siren",
        "hard_component_id",
        "hard_fold",
        "role",
        "ranker_prediction_is_out_of_sample",
        "prediction_origin",
        "ranker_oof_fold",
    }
    all_features = [name for name in feature_order if name not in metadata]
    missing_current80 = [
        name for name in CURRENT80_FEATURES if name not in all_features
    ]
    if missing_current80:
        raise ValueError(
            "Feature catalog requires the exact current80 block: "
            f"{missing_current80}"
        )

    def is_drift_feature(name: str) -> bool:
        return any(
            name == f"candidate_delta_{base}"
            or name.startswith(f"candidate_top1_{base}")
            or name.startswith(f"candidate_top2_{base}")
            for base in DRIFT_AUDIT_ONLY_BASE_FEATURES
        )

    audit_only = [name for name in all_features if is_drift_feature(name)]
    structured_order = [
        name
        for name in all_features
        if name not in CURRENT80_FEATURES and name not in audit_only
    ]
    binary_exact = {
        "interaction_strong_address_weak_name",
        "interaction_different_siren_strong_address",
        "interaction_same_siren_geo_incompatible",
        "interaction_exact_input_siret_function_incompatible",
        "naf_top1_top2_section_equal",
        "naf_top1_top2_division_equal",
        "naf_top1_division_missing",
        "naf_top2_division_missing",
        "role_crm_top1_conflict",
        "role_top1_top2_conflict",
        "same_siren_role_plurality",
        "same_siren_has_sibling",
        "same_siren_name_geo_best_disagreement",
        "same_siren_best_sibling_name_geo_disagreement",
    }
    count_exact = {
        "role_crm_count",
        "role_top1_count",
        "role_top2_count",
        "same_siren_distinct_division_count",
        "same_siren_distinct_role_count",
        "same_siren_candidate_count",
        "same_siren_sibling_count",
        "same_siren_active_count",
        "same_siren_postcode_match_count",
        "same_siren_city_match_count",
        "same_siren_street_number_match_count",
    }
    binary_candidate_projection = {
        f"candidate_{position}_{base}"
        for base in BOOLEAN_CANDIDATE_BASE_FEATURES
        for position in ("top1", "top2")
    }
    binary_dynamic = {
        *binary_exact,
        *binary_candidate_projection,
        *(
            f"candidate_{feature}_top1_top2_equal"
            for feature in CATEGORICAL_CANDIDATE_FEATURES
        ),
        *(
            f"top1_is_same_siren_best_{feature}"
            for feature in ("name_jaro_max", "addr_jaro", "ranker_score")
        ),
    }

    def explicit_spec(name: str, *, order: str, model_allowed: bool) -> dict[str, Any]:
        is_one_hot = "__" in name
        is_missing = name.endswith("_missing")
        baseline_name = name.removeprefix("scene_")
        if (
            name in binary_dynamic
            or is_one_hot
            or is_missing
            or (
                order == "current80"
                and name in CURRENT80_BINARY_FEATURES
            )
            or (
                name.startswith("scene_")
                and baseline_name in CURRENT80_BINARY_FEATURES
            )
        ):
            kind = "binary"
        elif (
            name in count_exact
            or (
                order == "current80"
                and name in CURRENT80_COUNT_FEATURES
            )
            or (
                name.startswith("scene_")
                and baseline_name in CURRENT80_COUNT_FEATURES
            )
        ):
            kind = "count"
        else:
            kind = "continuous"
        if order == "current80":
            source = "frozen_v4.1_scene"
            source_block = "current80_scene"
            formula = f"identity:{name}"
            missing_policy = "forbidden"
        elif name.startswith("scene_"):
            source = "frozen_v4.1_scene_non_semantic"
            source_block = "scene_v41"
            formula = f"identity:{name.removeprefix('scene_')}"
            missing_policy = "forbidden"
        elif name.startswith("candidate_"):
            source = "frozen_ranker_candidate_top1_top2"
            source_block = "candidate_ranker"
            formula = "top1_top2_delta_projection_v1"
            missing_policy = (
                "indicator_column"
                if is_missing
                else (
                    "pinned_UNKNOWN_one_hot"
                    if is_one_hot
                    else "zero_plus_explicit_indicator"
                )
            )
        elif name.startswith("naf_"):
            source = "authoritative_sirene_activity"
            source_block = "sirene_activity"
            formula = "pinned_naf_projection_v1"
            missing_policy = "pinned_UNKNOWN_one_hot"
        elif name.startswith("role_"):
            source = "v4.9_site_function_taxonomy"
            source_block = "site_function"
            formula = "pinned_role_projection_v1"
            missing_policy = "pinned_UNKNOWN_or_zero_count"
        elif (
            name.startswith("same_siren_")
            or name.startswith("top1_gap_")
            or name.startswith("top1_is_same_siren_best_")
        ):
            source = "actual_pool_same_siren_constellation"
            source_block = "siren_constellation"
            formula = "constellation_aggregate_v1"
            missing_policy = "deterministic_empty_constellation_value"
        elif name.startswith(("top1_", "top2_", "delta_")):
            source = "frozen_candidate_crm_and_sirene"
            source_block = "v8_interaction"
            formula = "v8_interaction_v1"
            missing_policy = "deterministic_from_encoded_inputs"
        elif name.startswith("interaction_"):
            source = "structured_general_interaction"
            source_block = "structured_interaction"
            formula = "interaction_thresholds_v1"
            missing_policy = "forbidden"
        else:
            raise ValueError(
                f"Feature catalog has no explicit specification for {name}"
            )
        return {
            "name": name,
            "dtype": "float64",
            "kind": kind,
            "source": source,
            "source_block": source_block,
            "formula": formula,
            "formula_version": FEATURE_POLICY_VERSION,
            "nullable": False,
            "missing_policy": missing_policy,
            "model_allowed": model_allowed,
            "order": order,
            "leakage_class": (
                "retrieval_population_drift" if not model_allowed else "none"
            ),
        }

    entries = [
        *[
            explicit_spec(name, order="current80", model_allowed=True)
            for name in CURRENT80_FEATURES
        ],
        *[
            explicit_spec(name, order="structured", model_allowed=True)
            for name in structured_order
        ],
        *[
            explicit_spec(name, order="audit_only", model_allowed=False)
            for name in audit_only
        ],
    ]
    current80_hash = hashlib.sha256(
        "\n".join(CURRENT80_FEATURES).encode("utf-8")
    ).hexdigest()
    structured_hash = hashlib.sha256(
        "\n".join(structured_order).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "sireto-v4.10-feature-catalog-1",
        "experiment_id": EXPERIMENT_ID,
        "feature_policy_version": FEATURE_POLICY_VERSION,
        "current80_feature_order": list(CURRENT80_FEATURES),
        "current80_feature_order_sha256": current80_hash,
        "structured_feature_order": structured_order,
        "structured_feature_order_sha256": structured_hash,
        "feature_order": structured_order,
        "feature_order_sha256": structured_hash,
        "feature_count": len(structured_order),
        "output_feature_order": all_features,
        "output_feature_count": len(all_features),
        "features": entries,
        "metadata_columns": sorted(metadata),
        "audit_only_features": [
            {
                "name": name,
                "model_allowed": False,
                "reason": "retrieval_v41_v42b_distribution_drift",
            }
            for name in audit_only
        ],
        "excluded_features": {
            "semantic_scene_features": list(SEMANTIC_SCENE_FEATURES),
            "raw_identifiers": ["query_id", "top1_siret", "top1_siren"],
        },
        "categorical_encodings": {
            name: [*categories, "UNKNOWN"]
            for name, categories in CATEGORICAL_CANDIDATE_FEATURES.items()
        },
        "naf_sections": [*NAF_SECTIONS, "UNKNOWN"],
        "naf_divisions": [*NAF_DIVISIONS, "UNKNOWN"],
        "site_function_roles": list(ROLE_NAMES),
        "interaction_thresholds": INTERACTION_THRESHOLDS,
        "interaction_semantics": "features_only_never_post_model_veto",
        "missing_indicator_features": sorted(
            name for name in all_features if name.endswith("_missing")
        ),
        "unknown_category_features": sorted(
            name for name in all_features if name.endswith("__UNKNOWN")
        ),
    }


def _missingness(frame: pd.DataFrame, feature_order: Sequence[str]) -> dict[str, float]:
    return {
        column: float(frame[column].isna().mean())
        for column in feature_order
    }


def _counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    observed = {
        str(key): int(value)
        for key, value in frame[column].value_counts(dropna=False).items()
    }
    return dict(sorted(observed.items()))


def _stable_rows_sha256(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    rows = (
        frame[list(columns)]
        .fillna("")
        .astype(str)
        .sort_values(list(columns), kind="mergesort")
        .itertuples(index=False, name=None)
    )
    digest = hashlib.sha256()
    for row in rows:
        digest.update(("\x1f".join(row) + "\n").encode("utf-8"))
    return digest.hexdigest()


def dataset_population_audit(
    historical: pd.DataFrame,
    consumed: pd.DataFrame,
    locked: pd.DataFrame,
) -> dict[str, Any]:
    hard_labels = _counts(consumed, "adjudication_label")
    expected_hard = {"AMBIGUOUS": 1, "TOP1_CORRECT": 68, "TOP1_WRONG": 25}
    if hard_labels != expected_hard:
        raise ValueError(
            f"STOP_DATASET_INTEGRITY: hard_oof labels {hard_labels} != {expected_hard}"
        )
    support_count = int(
        historical["role"].astype(str).eq("historical_hard_support").sum()
    )
    if support_count != 20:
        raise ValueError(
            f"STOP_DATASET_INTEGRITY: historical hard supports {support_count} != 20"
        )
    if len(consumed) != 94 or len(locked) != 4 or len(historical) != 7_003:
        raise ValueError("STOP_DATASET_INTEGRITY: canonical population volumes changed")
    component_fold_counts = consumed.groupby("hard_component_id")[
        "hard_fold"
    ].nunique(dropna=False)
    if (component_fold_counts != 1).any():
        raise ValueError(
            "STOP_DATASET_INTEGRITY: a hard component crosses OOF folds"
        )
    hard_component_folds = (
        consumed[["hard_component_id", "hard_fold"]]
        .drop_duplicates()
        .set_index("hard_component_id")["hard_fold"]
    )
    supports = historical[
        historical["role"].astype(str).eq("historical_hard_support")
    ]
    linked_support_folds = supports["hard_component_id"].map(hard_component_folds)
    if (
        linked_support_folds.isna().any()
        or not (
            linked_support_folds.astype("Int64")
            == supports["hard_fold"].astype("Int64")
        ).all()
    ):
        raise ValueError(
            "STOP_DATASET_INTEGRITY: historical hard support fold linkage changed"
        )
    frames = {
        "historical": historical,
        "development_consumed": consumed,
        "descriptive_locked": locked,
    }
    return {
        name: {
            "row_count": int(len(frame)),
            "split_counts": _counts(frame, "split"),
            "role_counts": _counts(frame, "role"),
            "fold_counts": _counts(frame, "hard_fold"),
            "target_counts": _counts(frame, "acceptor_target"),
            "label_counts": _counts(frame, "adjudication_label"),
            "oos_proof_counts": _counts(
                frame, "ranker_prediction_is_out_of_sample"
            ),
            "prediction_origin_counts": _counts(frame, "prediction_origin"),
            "query_ids_sha256": _stable_rows_sha256(frame, ["query_id"]),
            "component_assignments_sha256": _stable_rows_sha256(
                frame,
                ["query_id", "hard_component_id", "hard_fold", "role"],
            ),
        }
        for name, frame in frames.items()
    }


def validate_feature_matrix(
    frame: pd.DataFrame,
    feature_order: Sequence[str],
    *,
    name: str,
) -> None:
    if frame[list(feature_order)].isna().any().any():
        raise ValueError(f"{name} contains nullable encoded features")
    try:
        values = frame[list(feature_order)].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} contains a non-numeric encoded feature") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains a non-finite encoded feature")


def runtime_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parent.parent
    package_versions: dict[str, str] = {}
    for package in (
        "pandas",
        "numpy",
        "duckdb",
        "pyarrow",
        "scikit-learn",
        "xgboost",
    ):
        package_versions[package] = importlib.metadata.version(package)
    source_files = {
        "builder": Path(__file__).resolve(),
        "feature_implementation": root / "src/xgb_matcher/features.py",
        "site_function_implementation": root / "src/xgb_matcher/v49_site_function.py",
    }
    return {
        "contract_commit": EXPECTED_CONTRACT_COMMIT,
        "source_hashes": {
            name: {
                "path": str(path),
                "sha256": file_sha256(path),
            }
            for name, path in source_files.items()
        },
        "package_versions": package_versions,
    }


def build(args: argparse.Namespace) -> Path:
    source_paths = {
        "historical_scenes": args.historical_scenes,
        "historical_predictions": args.historical_predictions,
        "historical_queries": args.historical_queries,
        "historical_candidates": args.historical_candidates,
        "hard_candidates": args.hard_candidates,
        "hard_scenes": args.hard_scenes,
        "current_labels": args.current_labels,
        "hard_queue": args.hard_queue,
        "partition_assignments": args.partition_assignments,
        "partition_manifest": args.partition_manifest,
        "sirene_snapshot": args.sirene_snapshot,
        "taxonomy": args.taxonomy,
    }
    input_records = validate_frozen_inputs(source_paths, args.contract)
    runtime = runtime_provenance()
    historical_scene_columns = [
        "query_id",
        "split",
        "predicted_siret",
        "predicted_siren",
        "prediction_origin",
        "ranker_prediction_is_out_of_sample",
        "acceptor_eligible",
        *CURRENT80_FEATURES,
    ]
    historical_scenes = pd.read_parquet(
        args.historical_scenes, columns=historical_scene_columns
    )
    historical_targets = pd.read_parquet(
        args.historical_scenes,
        columns=["query_id", "label_kind", "is_exact_siret_correct"],
    )
    historical_predictions = pd.read_parquet(
        args.historical_predictions,
        columns=[
            "query_id",
            "candidate_siret",
            "score",
            "prediction_origin",
            "fold",
            "rank",
        ],
    )
    historical_queries = pd.read_parquet(args.historical_queries)
    prediction_proof = (
        historical_predictions.groupby("query_id", sort=False)
        .agg(
            prediction_origin_from_candidates=("prediction_origin", "first"),
            prediction_origin_count=("prediction_origin", "nunique"),
            ranker_oof_fold=("fold", "first"),
            ranker_oof_fold_count=("fold", "nunique"),
        )
        .reset_index()
    )
    if (
        prediction_proof["prediction_origin_count"].ne(1).any()
        or prediction_proof["ranker_oof_fold_count"].gt(1).any()
    ):
        raise ValueError("Historical ranker proof differs within a query")
    historical_scenes = historical_scenes.merge(
        prediction_proof[
            [
                "query_id",
                "prediction_origin_from_candidates",
                "ranker_oof_fold",
            ]
        ],
        on="query_id",
        validate="one_to_one",
    )
    if not historical_scenes["ranker_prediction_is_out_of_sample"].astype(bool).all():
        raise ValueError("All historical V4.10 scenes require ranker OOS proof")
    if not set(historical_scenes["prediction_origin"].astype(str)).issubset(
        {"oof", "out_of_sample_dev"}
    ):
        raise ValueError("Historical prediction_origin is not authorised")
    if not (
        historical_scenes["prediction_origin"].astype(str)
        == historical_scenes["prediction_origin_from_candidates"].astype(str)
    ).all():
        raise ValueError("Historical scene/candidate OOS origins differ")
    historical_scenes = historical_scenes.drop(
        columns="prediction_origin_from_candidates"
    )
    historical_candidate_columns = [
        "query_id",
        "candidate_siret",
        "candidate_siren",
        "candidate_state",
        "retrieval_rank",
        "retrieval_source",
        "retrieval_channel_count",
        "retrieval_agreement",
        *V41_CANDIDATE_FEATURES,
    ]
    if "is_ground_truth" in historical_candidate_columns:
        raise AssertionError("Target leakage column entered candidate projection")
    frozen_candidates = pd.read_parquet(
        args.historical_candidates,
        columns=historical_candidate_columns,
    )
    partition_columns = [
        "population",
        "query_id",
        "audit_case_id",
        "component_id",
        "hard_fold",
        "role",
        "evidence_validated",
    ]
    partitions = pd.read_parquet(
        args.partition_assignments,
        columns=partition_columns,
        filters=[
            [
                ("population", "=", "historical"),
            ],
            [
                ("population", "=", "current"),
                ("role", "in", ["hard_oof", "hard_dev_locked"]),
                ("evidence_validated", "=", True),
            ],
        ],
    )
    current_partitions = partitions[partitions["population"].eq("current")].copy()
    role_counts = current_partitions["role"].value_counts().to_dict()
    if role_counts != {"hard_oof": 94, "hard_dev_locked": 4}:
        raise ValueError(f"Unexpected reliable targeted roles: {role_counts}")
    allowed_hard_ids = set(current_partitions["audit_case_id"].astype(str))
    current = read_targeted_parquet(
        args.current_labels,
        columns=[
            "audit_case_id",
            "sampling_stratum",
            "current_top1_siret",
            "current_adjudication_label",
            "current_acceptor_target",
            "current_training_eligible",
        ],
    )
    current = current[current["audit_case_id"].astype(str).isin(allowed_hard_ids)].copy()
    if set(current["audit_case_id"].astype(str)) != allowed_hard_ids:
        raise ValueError("V4.7 labels do not cover the 98 authorised hard cases")
    hard_scenes = read_targeted_parquet(
        args.hard_scenes,
        columns=[
            "audit_case_id",
            "sampling_stratum",
            "replayed_top1_siret",
            "replayed_top1_siren",
            "input_siret",
            "candidate_pool_sirens_json",
            *CURRENT80_FEATURES,
        ],
    )
    hard_scenes = hard_scenes[
        hard_scenes["audit_case_id"].astype(str).isin(allowed_hard_ids)
    ].copy()
    if set(hard_scenes["audit_case_id"].astype(str)) != allowed_hard_ids:
        raise ValueError("V4.5 scenes do not cover the 98 authorised hard cases")
    top1_binding = hard_scenes[
        ["audit_case_id", "replayed_top1_siret"]
    ].merge(
        current[["audit_case_id", "current_top1_siret"]],
        on="audit_case_id",
        validate="one_to_one",
    )
    if not (
        top1_binding["replayed_top1_siret"].map(_siret)
        == top1_binding["current_top1_siret"].map(_siret)
    ).all():
        raise ValueError("A current hard label is no longer bound to replayed top-1")
    hard_candidates = prepare_hard_candidates(
        pd.read_parquet(
            args.hard_candidates,
            columns=[
                column
                for column in [
                    "audit_case_id",
                    "service_id",
                    "candidate_siret",
                    "candidate_siren",
                    "candidate_state",
                    "rank",
                    "ranker_score",
                    "rrf_score",
                    "retrieval_channel_count",
                    "retrieval_agreement",
                    *V41_CANDIDATE_FEATURES,
                ]
                if column != "is_ground_truth"
            ],
            filters=[("audit_case_id", "in", sorted(allowed_hard_ids))],
        ),
        allowed_hard_ids,
    )
    hard_queue = read_targeted_parquet(
        args.hard_queue,
        columns=[
            "audit_case_id",
            "sampling_stratum",
            "SITE",
            "SITE_CLI_ADRESSE",
            "CODE_POSTAL",
            "COMMUNE",
            "CODE_INSEE",
            "input_siret",
            "input_siren",
        ],
    )
    hard_queue = hard_queue[
        hard_queue["audit_case_id"].astype(str).isin(allowed_hard_ids)
    ].copy()
    hard_queue["query_id"] = hard_queue["audit_case_id"].astype(str)
    hard_queue = hard_queue.rename(
        columns={
            "SITE": "crm_name",
            "SITE_CLI_ADRESSE": "crm_address",
            "CODE_POSTAL": "crm_postcode",
            "COMMUNE": "crm_city",
            "CODE_INSEE": "crm_insee",
        }
    )
    hard_scenes["query_id"] = hard_scenes["audit_case_id"].astype(str)
    query_coverage = {
        "historical": validate_query_coverage(
            historical_scenes, historical_queries, name="historical"
        ),
        "hard_authorised": validate_query_coverage(
            hard_scenes, hard_queue, name="hard_authorised"
        ),
    }
    historical_candidates, join_report = prepare_historical_candidates(
        historical_predictions, frozen_candidates
    )
    hard_scenes["ranker_prediction_is_out_of_sample"] = True
    hard_scenes["prediction_origin"] = "frozen_v41_ranker_on_v42b_experimental"
    hard_scenes["ranker_oof_fold"] = pd.NA
    current["query_id"] = current["audit_case_id"].astype(str)
    if not (
        current_partitions["query_id"].astype(str)
        == current_partitions["audit_case_id"].astype(str)
    ).all():
        raise ValueError("V4.8 hard partition query/audit identifiers differ")
    hard_partition_metadata = current_partitions.rename(
        columns={"component_id": "hard_component_id"}
    )[["query_id", "hard_component_id", "hard_fold", "role"]]
    all_candidates = pd.concat(
        [
            historical_candidates[["candidate_siret"]],
            hard_candidates[["candidate_siret"]],
        ],
        ignore_index=True,
    )
    output_root = Path(args.output_root).resolve()
    if not output_root.is_relative_to(Path("/Volumes/CATNAT_DATA")):
        raise ValueError("V4.10 output must be on /Volumes/CATNAT_DATA")
    output_root.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=".v410-", dir=output_root))
    try:
        registry = hydrate_sirene(
            all_candidates,
            args.sirene_snapshot,
            temp_directory=work_dir,
        )
        registry_reports = {
            "historical": validate_registry_coverage(
                historical_candidates, registry, name="historical"
            ),
            "development_consumed": validate_registry_coverage(
                hard_candidates, registry, name="hard"
            ),
        }
        taxonomy = SiteFunctionTaxonomy.load(args.taxonomy)
        historical = build_structured_scenes(
            historical_scenes,
            historical_candidates,
            historical_queries,
            registry,
            taxonomy,
            population="historical_v41",
        )
        consumed = build_structured_scenes(
            hard_scenes,
            hard_candidates,
            hard_queue,
            registry,
            taxonomy,
            population="development_consumed",
        )
        hard_targets = current[
            [
                "query_id",
                "current_adjudication_label",
                "current_acceptor_target",
                "current_training_eligible",
            ]
        ].rename(
            columns={
                "current_adjudication_label": "adjudication_label",
                "current_acceptor_target": "acceptor_target",
                "current_training_eligible": "training_eligible",
            }
        )
        hard_all = consumed.merge(
            hard_targets, on="query_id", validate="one_to_one"
        ).merge(
            hard_partition_metadata, on="query_id", validate="one_to_one"
        )
        consumed = hard_all[hard_all["role"].eq("hard_oof")].copy()
        locked = hard_all[hard_all["role"].eq("hard_dev_locked")].copy()
        consumed["population_role"] = "development_consumed"
        locked["population_role"] = "descriptive_locked"
        historical = historical.merge(
            historical_targets.rename(
                columns={
                    "label_kind": "adjudication_label",
                    "is_exact_siret_correct": "acceptor_target",
                }
            ),
            on="query_id",
            validate="one_to_one",
        ).merge(
            partitions[partitions["population"].eq("historical")][
                ["query_id", "component_id", "hard_fold", "role"]
            ].rename(columns={"component_id": "hard_component_id"}),
            on="query_id",
            validate="one_to_one",
        )
        eligibility = historical_scenes.set_index(
            historical_scenes["query_id"].astype(str)
        )["acceptor_eligible"]
        historical["training_eligible"] = (
            historical["query_id"].astype(str).map(eligibility).astype(bool)
        )
        population_audit = dataset_population_audit(
            historical, consumed, locked
        )
        metadata = {
            "query_id",
            "population_role",
            "split",
            "acceptor_target",
            "adjudication_label",
            "training_eligible",
            "top1_siret",
            "top1_siren",
            "hard_component_id",
            "hard_fold",
            "role",
            "ranker_prediction_is_out_of_sample",
            "prediction_origin",
            "ranker_oof_fold",
        }
        feature_order = [
            column
            for column in historical.columns
            if column not in metadata
        ]
        consumed_feature_order = [
            column for column in consumed.columns if column not in metadata
        ]
        locked_feature_order = [
            column for column in locked.columns if column not in metadata
        ]
        if feature_order != consumed_feature_order:
            raise ValueError("Historical and hard structured feature schemas differ")
        if feature_order != locked_feature_order:
            raise ValueError("Locked structured feature schema differs")
        catalog = make_feature_catalog([*metadata, *feature_order])
        for frame_name, frame in (
            ("historical", historical),
            ("development_consumed", consumed),
            ("descriptive_locked", locked),
        ):
            validate_feature_matrix(frame, feature_order, name=frame_name)
        catalog_bytes = _json_bytes(catalog)
        build_payload = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "contract_sha256": EXPECTED_CONTRACT_SHA256,
            "contract_commit": EXPECTED_CONTRACT_COMMIT,
            "input_hashes": EXPECTED_INPUT_HASHES,
            "runtime_provenance": runtime,
            "feature_catalog_sha256": hashlib.sha256(catalog_bytes).hexdigest(),
        }
        build_id = hashlib.sha256(
            json.dumps(build_payload, sort_keys=True).encode()
        ).hexdigest()[:16]
        stage = work_dir / build_id
        stage.mkdir()
        historical_path = stage / "historical_scenes.parquet"
        consumed_path = stage / "consumed_hard_scenes.parquet"
        locked_path = stage / "descriptive_locked_scenes.parquet"
        historical.to_parquet(historical_path, index=False)
        consumed.to_parquet(consumed_path, index=False)
        locked.to_parquet(locked_path, index=False)
        _json_dump(stage / "feature_catalog.json", catalog)
        manifest = {
            **build_payload,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "inputs": input_records,
            "outputs": {
                path.name: file_sha256(path)
                for path in (
                    historical_path,
                    consumed_path,
                    locked_path,
                    stage / "feature_catalog.json",
                )
            },
            "volumes": {
                "historical_scenes": int(len(historical)),
                "development_consumed": int(len(consumed)),
                "descriptive_locked": int(len(locked)),
                "historical_candidates": int(len(historical_candidates)),
                "hard_candidates": int(len(hard_candidates)),
            },
            "join_rates": {
                **join_report,
                "crm_query_coverage": query_coverage,
                "sirene_candidate_join_rate": float(
                    all_candidates["candidate_siret"].isin(registry["siret"]).mean()
                ),
                "sirene_top1_top2": registry_reports,
            },
            "missingness": {
                "historical": _missingness(historical, feature_order),
                "development_consumed": _missingness(consumed, feature_order),
                "descriptive_locked": _missingness(locked, feature_order),
            },
            "output_feature_order": feature_order,
            "model_feature_order": catalog["feature_order"],
            "model_feature_order_sha256": catalog["feature_order_sha256"],
            "current80_feature_order": catalog["current80_feature_order"],
            "current80_feature_order_sha256": catalog[
                "current80_feature_order_sha256"
            ],
            "structured_feature_order": catalog["structured_feature_order"],
            "structured_feature_order_sha256": catalog[
                "structured_feature_order_sha256"
            ],
            "population_audit": population_audit,
            "encoding_audit": {
                "missing_indicator_count": len(
                    catalog["missing_indicator_features"]
                ),
                "missing_indicator_features_sha256": hashlib.sha256(
                    "\n".join(catalog["missing_indicator_features"]).encode()
                ).hexdigest(),
                "unknown_category_count": len(
                    catalog["unknown_category_features"]
                ),
                "unknown_category_features_sha256": hashlib.sha256(
                    "\n".join(catalog["unknown_category_features"]).encode()
                ).hexdigest(),
            },
            "invariants": {
                "max_candidates": MAX_CANDIDATES,
                "positive_injection": False,
                "ranker_a_frozen": True,
                "historical_retrieval": "v4.1_oof_frozen",
                "hard_retrieval": "v4.2-b_frozen",
                "mixed_retrieval_feasibility_only": True,
                "model_trained": False,
                "threshold_selected": False,
                "random_v48_rows_read_or_scored": 0,
                "test_rows_read_or_scored": 0,
                "fresh_rows_read_or_scored": 0,
                "consumed_hard_role": "development_consumed",
                "raw_identifiers_are_model_features": False,
                "is_ground_truth_loaded_or_used": False,
                "semantic_scene_feature_count_excluded": len(SEMANTIC_SCENE_FEATURES),
            },
        }
        _json_dump(stage / "manifest.json", manifest)
        final = output_root / build_id
        if final.exists():
            raise FileExistsError(f"V4.10 build already exists: {final}")
        stage.rename(final)
        return final
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--historical-scenes",
        type=Path,
        default=base / "models/v4_1/f938abf6b8a87155/acceptor_scenes.parquet",
    )
    parser.add_argument(
        "--historical-predictions",
        type=Path,
        default=base / "models/v4_1/f938abf6b8a87155/ranker_predictions.parquet",
    )
    parser.add_argument(
        "--historical-queries",
        type=Path,
        default=base / "datasets/v4_1/f938abf6b8a87155/queries.parquet",
    )
    parser.add_argument(
        "--historical-candidates",
        type=Path,
        default=base / "datasets/v4_1/f938abf6b8a87155/candidates.parquet",
    )
    parser.add_argument(
        "--hard-candidates",
        type=Path,
        default=base / "datasets/v4_5_hard_scenes/21f8c0b0b172b907/candidates.parquet",
    )
    parser.add_argument(
        "--hard-scenes",
        type=Path,
        default=base / "datasets/v4_5_hard_scenes/21f8c0b0b172b907/scene_compatibility.parquet",
    )
    parser.add_argument(
        "--current-labels",
        type=Path,
        default=base / "audits/v4_7_current_adjudications/4cc5420fb5da0683/current_labels.parquet",
    )
    parser.add_argument(
        "--hard-queue",
        type=Path,
        default=base / "audits/v4_3_hard_labels/0f832305ab199267/hard_label_queue.parquet",
    )
    parser.add_argument(
        "--partition-assignments",
        type=Path,
        default=base
        / "datasets/v4_8_acceptor_partitions/1c78764d5263afca/partition_assignments.parquet",
    )
    parser.add_argument(
        "--partition-manifest",
        type=Path,
        default=base
        / "datasets/v4_8_acceptor_partitions/1c78764d5263afca/manifest.json",
    )
    parser.add_argument(
        "--sirene-snapshot",
        type=Path,
        default=Path("data/StockEtablissement_utf8.parquet"),
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=Path("config/v4_9_site_function_taxonomy.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=base / "datasets/v4_10_structured_acceptor",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    output = build(parse_args(argv))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
