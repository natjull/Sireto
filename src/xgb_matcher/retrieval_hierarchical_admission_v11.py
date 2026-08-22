"""Hierarchical, all-human SIRET retrieval admission (v11).

This module deliberately leaves the historical flat admission untouched.  It
implements a two-level learned retrieval policy:

1. rank candidate legal entities (parents) with LambdaMART;
2. rank candidate establishments with a second LambdaMART using only
   cross-fitted parent scores during training;
3. select a deterministic, protected portfolio of at most 100 SIRETs.

Human SIRET labels are accepted only by the training and evaluation entry
points.  Candidate preparation, model scoring and portfolio selection are
label-blind.  Validation is always performed on the complete input union (up
to 2,000 candidates); no positive is ever appended to a candidate set.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from statistics import NormalDist
from typing import Any, Iterable, Mapping, Sequence
import unicodedata

import numpy as np
import pandas as pd
import xgboost as xgb


SCHEMA_VERSION = "sireto-retrieval-hierarchical-admission-v11"
CONFIG_SCHEMA_VERSION = "sireto-retrieval-all-human-v11-config-1"
LABEL_COLUMNS = (
    "human_siret",
    "human_ground_truth_siret",
    "historical_ground_truth_siret",
    "ground_truth_siret",
    "gt_siret",
)
_FEATURE_ID_TOKENS = ("query_id", "siret", "siren", "identifier")
_SPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^A-Z0-9]+")
_SOURCE_SPLIT_RE = re.compile(r"[+,|;\s]+")
_LEADING_NUMBER_RE = re.compile(r"^(\d{1,5})(?:\s|$)")


PARENT_FEATURES = (
    "parent_best_union_rank_reciprocal",
    "parent_best_union_rank_log_inverse",
    "parent_best_retrieval_rank_reciprocal",
    "parent_best_retrieval_rank_log_inverse",
    "parent_max_retrieval_score_relative",
    "parent_candidate_count_log",
    "parent_channel_coverage_log",
    "parent_anchor_candidate_count_log",
    "parent_consensus_candidate_count_log",
    "parent_max_name_token_jaccard",
    "parent_max_name_char_dice",
    "parent_max_address_token_jaccard",
    "parent_max_address_char_dice",
    "parent_any_street_number_match",
    "parent_any_postcode_match",
    "parent_any_insee_match",
    "parent_active_fraction",
    "parent_historical_signal_max",
    "parent_current_signal_max",
    "parent_official_source_max",
)


SIRET_FEATURES = (
    "union_rank_reciprocal",
    "union_rank_log_inverse",
    "union_rank_percentile_inverse",
    "retrieval_rank_reciprocal",
    "retrieval_rank_log_inverse",
    "retrieval_score_relative",
    "retrieval_score_percentile",
    "channel_count_log",
    "exact_signal_count",
    "is_anchor",
    "is_consensus",
    "name_token_jaccard",
    "name_char_dice",
    "address_token_jaccard",
    "address_char_dice",
    "name_address_joint",
    "street_number_match",
    "postcode_match",
    "insee_match",
    "candidate_active",
    "candidate_is_siege",
    "official_source_count_log",
    "current_signal_count",
    "historical_signal_count",
    "relation_signal_count",
    "within_parent_rank_reciprocal",
    "within_parent_candidate_count_log",
    "parent_rank_reciprocal",
    "parent_score_probability",
    "parent_score_gap_to_best",
)


def _default_xgb(seed: int) -> dict[str, Any]:
    return {
        "objective": "rank:ndcg",
        "eval_metric": "ndcg@100",
        "tree_method": "hist",
        "n_estimators": 180,
        "learning_rate": 0.04,
        "max_depth": 4,
        "min_child_weight": 25.0,
        "subsample": 1.0,
        "colsample_bytree": 0.85,
        "reg_alpha": 1.0,
        "reg_lambda": 75.0,
        "random_state": seed,
        "n_jobs": 0,
        "lambdarank_pair_method": "topk",
        "lambdarank_num_pair_per_sample": 64,
    }


@dataclass(frozen=True)
class HierarchicalAdmissionConfig:
    """Immutable v11 experiment contract."""

    raw: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "HierarchicalAdmissionConfig":
        return cls(dict(raw))

    @classmethod
    def load(cls, path: Path | str) -> "HierarchicalAdmissionConfig":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def default(cls) -> "HierarchicalAdmissionConfig":
        return cls(
            {
                "schema_version": CONFIG_SCHEMA_VERSION,
                "policy_id": "retrieval-v11-all-human-exact-primary",
                "folds": {
                    "train": [2, 3, 4],
                    "development_diagnostic": 0,
                    "test_locked": 1,
                },
                "labels": {
                    "exact_target_field": "human_ground_truth_siret",
                    "precedence": [
                        "ground_truth_siret",
                        "historical_ground_truth_siret",
                    ],
                    "only_exclusion": "NO_VALID_CURRENT_OR_HISTORICAL_HUMAN_SIRET",
                },
                "retrieval": {
                    "maximum_internal_candidates": 2000,
                    "maximum_final_candidates": 100,
                    "anchor_protection_slots": 20,
                    "consensus_protection_slots": 20,
                    "maximum_final_sites_per_siren": 3,
                    "positive_injection": False,
                },
                "admission": {
                    "architecture": "PARENT_LAMBDAMART_THEN_SIRET_LAMBDAMART",
                    "parent_scores_for_siret_training": "NESTED_INNER_OOF_ONLY",
                    "validation_candidate_sampling": False,
                    "anchor_name_channels": [
                        "legal_name_exact",
                        "trade_name_exact",
                        "site_name_exact",
                        "name_exact",
                    ],
                    "anchor_address_channels": [
                        "address_exact",
                        "geo_exact_cell",
                    ],
                    "consensus_min_channels": 2,
                    "same_siren_same_site_exact_negative": False,
                    "fit_candidate_sampling": {
                        "maximum_candidates_per_query": 128,
                        "union_head": 96,
                        "channel_head": 2,
                        "retain_natural_positive_if_present": True,
                        "validation_sampling": False,
                    },
                    "xgboost": {
                        "parent": _default_xgb(20260831),
                        "siret": _default_xgb(20260901),
                    },
                },
                "gates": {
                    "minimum_oof_oracle_recall_at_2000": 0.995,
                    "minimum_oof_exact_recall_at_100": 0.992,
                    "minimum_oof_exact_recall_at_100_wilson95_lower": 0.99,
                    "minimum_each_outer_fold_exact_recall_at_100": 0.988,
                    "maximum_warm_p95_ms": 1000,
                    "maximum_warm_p99_ms": 2000,
                },
            }
        )

    def __post_init__(self) -> None:
        if self.raw.get("schema_version") != CONFIG_SCHEMA_VERSION:
            raise ValueError("Unsupported hierarchical admission schema")
        if self.train_folds != (2, 3, 4) or self.dev_fold != 0 or self.test_fold != 1:
            raise ValueError("v11 requires train=2/3/4, dev=0 and test=1")
        if self.internal_union_cap != 2000:
            raise ValueError("v11 internal union cap must remain 2,000")
        if self.max_candidates != 100:
            raise ValueError("v11 output ceiling must remain 100")
        if self.anchor_slots != 20 or self.consensus_slots != 20:
            raise ValueError("v11 requires 20 anchor and 20 consensus slots")
        if self.max_unprotected_per_parent != 3:
            raise ValueError("v11 requires a three-site unprotected parent cap")
        admission = self.raw.get("admission")
        if not isinstance(admission, Mapping):
            raise ValueError("The single v11 config requires an admission section")
        if admission.get("architecture") != "PARENT_LAMBDAMART_THEN_SIRET_LAMBDAMART":
            raise ValueError("Unsupported v11 admission architecture")
        if admission.get("parent_scores_for_siret_training") != "NESTED_INNER_OOF_ONLY":
            raise ValueError("SIRET fitting requires nested inner-OOF parent scores")
        if admission.get("validation_candidate_sampling") is not False:
            raise ValueError("v11 validation must score the complete candidate union")
        if admission.get("same_siren_same_site_exact_negative") is not False:
            raise ValueError("Operational same-site siblings cannot be exact negatives")
        sampling = admission.get("fit_candidate_sampling") or {}
        if sampling != {
            "maximum_candidates_per_query": 128,
            "union_head": 96,
            "channel_head": 2,
            "retain_natural_positive_if_present": True,
            "validation_sampling": False,
        }:
            raise ValueError("v11 fit-only candidate sampling contract drift")
        if not self.anchor_name_channels or not self.anchor_address_channels:
            raise ValueError("v11 anchors require separate name and address channels")
        if set(self.anchor_name_channels) & set(self.anchor_address_channels):
            raise ValueError("v11 name/address anchor channels must be disjoint")
        if self.raw.get("retrieval", {}).get("positive_injection") is not False:
            raise ValueError("v11 forbids positive injection")
        if self.anchor_slots + self.consensus_slots > self.max_candidates:
            raise ValueError("Protected slots exceed output ceiling")
        for name in ("parent", "siret"):
            if self.xgb_params(name).get("objective") != "rank:ndcg":
                raise ValueError(f"{name} model must use rank:ndcg")
        assert_identifier_free_features(PARENT_FEATURES)
        assert_identifier_free_features(SIRET_FEATURES)

    @property
    def policy_id(self) -> str:
        return str(self.raw["policy_id"])

    @property
    def train_folds(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.raw["folds"]["train"])

    @property
    def dev_fold(self) -> int:
        return int(self.raw["folds"]["development_diagnostic"])

    @property
    def test_fold(self) -> int:
        return int(self.raw["folds"]["test_locked"])

    @property
    def internal_union_cap(self) -> int:
        return int(self.raw["retrieval"]["maximum_internal_candidates"])

    @property
    def max_candidates(self) -> int:
        return int(self.raw["retrieval"]["maximum_final_candidates"])

    @property
    def anchor_slots(self) -> int:
        return int(self.raw["retrieval"]["anchor_protection_slots"])

    @property
    def consensus_slots(self) -> int:
        return int(self.raw["retrieval"]["consensus_protection_slots"])

    @property
    def max_unprotected_per_parent(self) -> int:
        return int(self.raw["retrieval"]["maximum_final_sites_per_siren"])

    @property
    def anchor_name_channels(self) -> tuple[str, ...]:
        return tuple(
            str(value) for value in self.raw["admission"]["anchor_name_channels"]
        )

    @property
    def anchor_address_channels(self) -> tuple[str, ...]:
        return tuple(
            str(value)
            for value in self.raw["admission"]["anchor_address_channels"]
        )

    @property
    def consensus_min_channels(self) -> int:
        return int(self.raw["admission"]["consensus_min_channels"])

    @property
    def gates(self) -> Mapping[str, float]:
        gates = self.raw["gates"]
        return {
            "minimum_oracle_at_2000": float(
                gates["minimum_oof_oracle_recall_at_2000"]
            ),
            "minimum_recall_at_100": float(
                gates["minimum_oof_exact_recall_at_100"]
            ),
            "minimum_wilson95_lower": float(
                gates["minimum_oof_exact_recall_at_100_wilson95_lower"]
            ),
            "minimum_outer_fold_recall": float(
                gates["minimum_each_outer_fold_exact_recall_at_100"]
            ),
            "maximum_warm_p95_ms": float(gates["maximum_warm_p95_ms"]),
            "maximum_warm_p99_ms": float(gates["maximum_warm_p99_ms"]),
        }

    def xgb_params(self, level: str) -> dict[str, Any]:
        return dict(self.raw["admission"]["xgboost"][level])

    @property
    def fit_candidate_sampling(self) -> Mapping[str, Any]:
        return dict(self.raw["admission"]["fit_candidate_sampling"])

    @property
    def human_label_columns(self) -> tuple[str, ...]:
        labels = self.raw["labels"]
        return tuple(
            dict.fromkeys(
                [
                    str(labels["exact_target_field"]),
                    *[str(value) for value in labels["precedence"]],
                ]
            )
        )


def assert_identifier_free_features(features: Sequence[str]) -> None:
    offending = [
        feature
        for feature in features
        if any(token in feature.lower() for token in _FEATURE_ID_TOKENS)
    ]
    if offending:
        raise ValueError(f"Identifier-bearing features are forbidden: {offending}")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(character for character in text if not unicodedata.combining(character))
    return _SPACE_RE.sub(" ", _NON_ALNUM_RE.sub(" ", text.upper())).strip()


def normalize_siret(value: Any, *, allow_empty: bool = False) -> str:
    text = re.sub(r"\D", "", str(value or ""))
    if not text and allow_empty:
        return ""
    if not text or len(text) > 14:
        raise ValueError(f"Invalid SIRET: {value!r}")
    return text.zfill(14)


def _as_bool(value: Any) -> bool:
    if value is None:
        return False
    try:
        if bool(pd.isna(value)):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.number)):
        return bool(value)
    return str(value).strip().upper() in {"1", "TRUE", "T", "YES", "Y", "OUI", "A"}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return default
    return output if math.isfinite(output) else default


def _values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    try:
        if bool(pd.isna(value)):
            return []
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item).strip()]
    return [text]


def _first(row: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name not in row:
            continue
        value = row.get(name)
        if _values(value):
            return value
    return ""


def _candidate_values(row: Mapping[str, Any], names: Sequence[str]) -> list[str]:
    output: set[str] = set()
    for name in names:
        if name in row:
            output.update(normalize_text(value) for value in _values(row.get(name)))
    output.discard("")
    return sorted(output)


def _channels(row: Mapping[str, Any]) -> set[str]:
    output: set[str] = set()
    for column in ("channels", "sources", "retrieval_source", "channel"):
        for raw in _values(row.get(column)):
            output.update(filter(None, _SOURCE_SPLIT_RE.split(raw.strip())))
    for column, value in row.items():
        if not column.endswith("_rank") or column in {"union_rank", "retrieval_rank"}:
            continue
        if _float(value, 0.0) >= 1:
            output.add(column[: -len("_rank")])
    return output


def _token_jaccard(left: str, right: str) -> float:
    a, b = set(normalize_text(left).split()), set(normalize_text(right).split())
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _char_dice(left: str, right: str) -> float:
    def grams(value: str) -> set[str]:
        normalized = f" {normalize_text(value).replace(' ', '_')} "
        return {
            normalized[index : index + 3]
            for index in range(max(0, len(normalized) - 2))
        }

    a, b = grams(left), grams(right)
    if not a or not b:
        return 0.0
    return 2.0 * len(a & b) / (len(a) + len(b))


def _maximum(query: str, candidates: Iterable[str], metric: Any) -> float:
    return max((metric(query, candidate) for candidate in candidates), default=0.0)


def _leading_number(address: str) -> str:
    match = _LEADING_NUMBER_RE.match(normalize_text(address))
    return match.group(1) if match else ""


def _human_label(
    row: Mapping[str, Any], config: HierarchicalAdmissionConfig
) -> str:
    for column in config.human_label_columns:
        if column not in row:
            continue
        try:
            value = normalize_siret(row.get(column), allow_empty=True)
        except ValueError:
            continue
        if value:
            return value
    return ""


def _operational_labels(row: Mapping[str, Any], human: str) -> tuple[str, ...]:
    raw = row.get("acceptable_sirets_operational_json")
    if raw in (None, ""):
        raw = row.get("acceptable_sirets_operational")
    values = _values(raw)
    normalized: set[str] = {human} if human else set()
    for value in values:
        try:
            normalized.add(normalize_siret(value))
        except ValueError:
            continue
    return tuple(sorted(normalized))


def _reject_labels(frame: pd.DataFrame) -> None:
    populated: list[str] = []
    for column in LABEL_COLUMNS:
        if column not in frame.columns:
            continue
        if frame[column].map(lambda value: bool(normalize_siret(value, allow_empty=True))).any():
            populated.append(column)
    if populated:
        raise ValueError(
            "Production candidate scoring cannot receive human labels: "
            + ", ".join(populated)
        )


def canonicalize_union(
    frame: pd.DataFrame,
    config: HierarchicalAdmissionConfig,
    *,
    labels_allowed: bool,
    require_labels: bool = False,
) -> pd.DataFrame:
    """Create numeric, label-blind candidate features from a capped union."""
    required = {"query_id", "fold", "union_rank", "crm_name", "crm_address"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Candidate union is missing columns: {missing}")
    if frame.empty:
        raise ValueError("Candidate union is empty")
    if not labels_allowed:
        _reject_labels(frame)

    records: list[dict[str, Any]] = []
    for raw in frame.to_dict("records"):
        candidate_siret = normalize_siret(
            _first(raw, ("candidate_siret", "siret"))
        )
        candidate_parent = re.sub(
            r"\D", "", str(_first(raw, ("candidate_siren", "siren")) or "")
        )
        candidate_parent = candidate_parent.zfill(9) if candidate_parent else candidate_siret[:9]
        if candidate_parent != candidate_siret[:9]:
            raise ValueError("Candidate SIRET and parent identifier disagree")
        names = _candidate_values(
            raw,
            (
                "candidate_names",
                "candidate_names_json",
                "candidate_name",
                "names",
                "denomination",
                "denominationUniteLegale",
                "enseigne",
            ),
        )
        addresses = _candidate_values(
            raw,
            (
                "candidate_addresses",
                "candidate_addresses_json",
                "candidate_address",
                "addresses",
                "address",
                "adresse",
            ),
        )
        crm_name = normalize_text(raw.get("crm_name"))
        crm_address = normalize_text(raw.get("crm_address"))
        candidate_insee = normalize_text(
            _first(raw, ("candidate_insee", "insee", "codeCommuneEtablissement"))
        )
        candidate_postcode = normalize_text(
            _first(raw, ("candidate_postcode", "postcode", "codePostalEtablissement"))
        )
        candidate_number = normalize_text(
            _first(raw, ("candidate_number", "number", "numeroVoieEtablissement"))
        )
        channels = _channels(raw)
        exact_name_count = len(channels.intersection(config.anchor_name_channels))
        exact_address_count = len(
            channels.intersection(config.anchor_address_channels)
        )
        current_count = len(
            channels.intersection(
                {
                    "legal_name_exact",
                    "trade_name_exact",
                    "site_name_exact",
                    "name_exact",
                    "address_exact",
                    "number_exact",
                    "fielded_name_bm25",
                    "legal_name_word",
                    "trade_name_word",
                    "site_name_word",
                    "address_word",
                    "current_name_char",
                    "address_char",
                    "rne_name",
                    "rne_address",
                }
            )
        )
        historical_count = len(
            channels.intersection(
                {
                    "historical_name_word",
                    "historical_address_word",
                    "supporting_name_word",
                    "supporting_address_word",
                    "bodacc_name",
                    "bodacc_address",
                }
            )
        )
        relation_count = len(
            channels.intersection(
                {
                    "official_successor",
                    "bodacc_relation",
                    "legal_entity_role_name",
                    "sector_parent",
                    "direct_parent_expansion",
                    "consensus_parent_expansion",
                }
            )
        )
        evidence_sources = set(
            normalize_text(value)
            for value in _values(raw.get("official_evidence_sources"))
            if normalize_text(value)
        )
        address_for_site = addresses[0] if addresses else ""
        geography = candidate_insee or candidate_postcode
        site_number = candidate_number or _leading_number(address_for_site)
        explicit_site = normalize_text(
            _first(raw, ("candidate_site_key", "site_key", "official_site_key"))
        )
        site_key = (
            f"EXPLICIT:{explicit_site}"
            if explicit_site
            else "|".join((geography, site_number, address_for_site))
            if geography and address_for_site
            else ""
        )
        state = normalize_text(
            _first(raw, ("candidate_state", "state", "etatAdministratifEtablissement"))
            or "A"
        )
        union_rank = int(_float(raw.get("union_rank"), 0.0))
        if union_rank < 1:
            raise ValueError("union_rank must be positive")
        retrieval_rank = int(_float(raw.get("retrieval_rank"), union_rank))
        retrieval_rank = max(1, retrieval_rank)
        crm_number = normalize_text(
            _first(raw, ("crm_number", "numeroVoie", "numero"))
        ) or _leading_number(crm_address)
        crm_postcode = normalize_text(
            _first(raw, ("crm_postcode", "postcode", "crm_cp"))
        )
        crm_insee = normalize_text(_first(raw, ("crm_insee", "insee")))
        street_number_match = int(bool(crm_number) and crm_number == site_number)
        postcode_match = int(bool(crm_postcode) and crm_postcode == candidate_postcode)
        insee_match = int(bool(crm_insee) and crm_insee == candidate_insee)
        is_anchor = int(
            exact_name_count > 0
            and (exact_address_count > 0 or insee_match > 0 or postcode_match > 0)
        )
        human = _human_label(raw, config) if labels_allowed else ""
        operational = _operational_labels(raw, human) if labels_allowed else ()
        relation_channels = {
            "official_successor",
            "bodacc_relation",
            "establishment_relation",
            "siren_relation_expansion",
            "legal_entity_role_name",
            "sector_parent",
        }
        geo_channels = {"geo_insee_census", "geo_exact_cell", "geo_number_cell"}
        parent_channels = {"v11_parent_expansion"}
        baseline_channels = channels - relation_channels - geo_channels - parent_channels
        record = {
            "query_id": str(raw["query_id"]),
            "fold": int(raw["fold"]),
            "candidate_siret": candidate_siret,
            "candidate_parent": candidate_parent,
            "candidate_site_key": site_key,
            "human_siret": human,
            "operational_sirets_json": json.dumps(operational),
            "union_rank": union_rank,
            "retrieval_rank": retrieval_rank,
            "retrieval_score": _float(raw.get("retrieval_score"), 0.0),
            "channel_count": len(channels),
            "channels_json": json.dumps(sorted(channels)),
            "official_evidence_ids_json": json.dumps(
                sorted(_values(raw.get("official_evidence_ids")))
            ),
            "official_evidence_sources_json": json.dumps(
                sorted(_values(raw.get("official_evidence_sources")))
            ),
            "exact_signal_count": exact_name_count + exact_address_count,
            "is_anchor": is_anchor,
            "is_consensus": int(len(channels) >= config.consensus_min_channels),
            "name_token_jaccard": _maximum(crm_name, names, _token_jaccard),
            "name_char_dice": _maximum(crm_name, names, _char_dice),
            "address_token_jaccard": _maximum(crm_address, addresses, _token_jaccard),
            "address_char_dice": _maximum(crm_address, addresses, _char_dice),
            "street_number_match": street_number_match,
            "postcode_match": postcode_match,
            "insee_match": insee_match,
            "candidate_active": int(state != "F"),
            "candidate_is_siege": int(
                _as_bool(_first(raw, ("candidate_is_siege", "is_siege", "etablissementSiege")))
            ),
            "official_source_count": len(evidence_sources),
            "current_signal_count": current_count,
            "historical_signal_count": historical_count,
            "relation_signal_count": relation_count,
            "retrieval_latency_ms": _float(raw.get("retrieval_latency_ms"), 0.0),
            "stage_baseline": int(bool(baseline_channels)),
            "stage_parent": int(bool(baseline_channels | (channels & parent_channels))),
            "stage_parent_geo": int(
                bool(baseline_channels | (channels & parent_channels) | (channels & geo_channels))
            ),
            "stage_full": 1,
            "segment_state": normalize_text(raw.get("ground_truth_state")),
            "segment_origin": normalize_text(
                _first(raw, ("data_origin", "source_kind"))
            ),
            "segment_unseen_siren": int(_as_bool(raw.get("unseen_siren"))),
            "segment_new_site_known_siren": int(
                _as_bool(raw.get("new_site_known_siren"))
            ),
            "segment_mega": int(
                _as_bool(raw.get("mega_base_pool"))
                or _float(raw.get("pool_size"), 0.0) >= 100_000
            ),
        }
        records.append(record)

    output = pd.DataFrame(records)
    if output["query_id"].eq("").any():
        raise ValueError("query_id cannot be empty")
    if output.duplicated(["query_id", "candidate_siret"]).any():
        raise ValueError("Candidate union must be deduplicated by query and SIRET")
    counts = output.groupby("query_id").size()
    if int(counts.max()) > config.internal_union_cap:
        raise ValueError("Candidate union exceeds the full-2,000 validation cap")
    for column in (
        "fold",
        "human_siret",
        "operational_sirets_json",
        "retrieval_latency_ms",
        "segment_state",
        "segment_origin",
        "segment_unseen_siren",
        "segment_new_site_known_siren",
        "segment_mega",
    ):
        if column == "human_siret" and not labels_allowed:
            continue
        if output.groupby("query_id")[column].nunique(dropna=False).gt(1).any():
            raise ValueError(f"Query-level field varies within query: {column}")
    if require_labels:
        labelled = output.groupby("query_id")["human_siret"].first()
        if labelled.eq("").any():
            raise ValueError("Every training/evaluation query needs a usable human SIRET")

    output["union_rank_reciprocal"] = 1.0 / output["union_rank"]
    output["union_rank_log_inverse"] = 1.0 / np.log2(output["union_rank"] + 1.0)
    output["retrieval_rank_reciprocal"] = 1.0 / output["retrieval_rank"]
    output["retrieval_rank_log_inverse"] = 1.0 / np.log2(output["retrieval_rank"] + 1.0)
    query_sizes = output.groupby("query_id")["candidate_siret"].transform("size")
    output["union_rank_percentile_inverse"] = 1.0 - (
        (output["union_rank"] - 1) / query_sizes.clip(lower=1)
    )
    best_score = output.groupby("query_id")["retrieval_score"].transform("max")
    output["retrieval_score_relative"] = output["retrieval_score"] - best_score
    output["retrieval_score_percentile"] = output.groupby("query_id")[
        "retrieval_score"
    ].rank(method="average", pct=True)
    output["channel_count_log"] = np.log1p(output["channel_count"])
    output["official_source_count_log"] = np.log1p(output["official_source_count"])
    output["name_address_joint"] = np.sqrt(
        output["name_char_dice"].clip(lower=0.0)
        * output["address_char_dice"].clip(lower=0.0)
    )
    within_rank = output.groupby(["query_id", "candidate_parent"])["union_rank"].rank(
        method="first", ascending=True
    )
    output["within_parent_rank_reciprocal"] = 1.0 / within_rank
    output["site_rank"] = within_rank.astype(np.int32)
    output["within_parent_candidate_count_log"] = np.log1p(
        output.groupby(["query_id", "candidate_parent"])["candidate_siret"].transform("size")
    )
    return output.sort_values(
        ["query_id", "union_rank", "candidate_siret"], kind="stable"
    ).reset_index(drop=True)


def build_parent_rows(candidates: pd.DataFrame) -> pd.DataFrame:
    """Aggregate establishment evidence into one row per query and parent."""
    records: list[dict[str, Any]] = []
    for (query_id, parent), group in candidates.groupby(
        ["query_id", "candidate_parent"], sort=True
    ):
        best_score = float(group["retrieval_score"].max())
        records.append(
            {
                "query_id": str(query_id),
                "fold": int(group.iloc[0]["fold"]),
                "candidate_parent": str(parent),
                "human_siret": str(group.iloc[0].get("human_siret") or ""),
                "parent_best_union_rank_reciprocal": float(
                    group["union_rank_reciprocal"].max()
                ),
                "parent_best_union_rank_log_inverse": float(
                    group["union_rank_log_inverse"].max()
                ),
                "parent_best_retrieval_rank_reciprocal": float(
                    group["retrieval_rank_reciprocal"].max()
                ),
                "parent_best_retrieval_rank_log_inverse": float(
                    group["retrieval_rank_log_inverse"].max()
                ),
                "_parent_raw_retrieval_score": best_score,
                "parent_candidate_count_log": math.log1p(len(group)),
                "parent_channel_coverage_log": math.log1p(
                    int(group["channel_count"].max())
                ),
                "parent_anchor_candidate_count_log": math.log1p(
                    int(group["is_anchor"].sum())
                ),
                "parent_consensus_candidate_count_log": math.log1p(
                    int(group["is_consensus"].sum())
                ),
                "parent_max_name_token_jaccard": float(
                    group["name_token_jaccard"].max()
                ),
                "parent_max_name_char_dice": float(group["name_char_dice"].max()),
                "parent_max_address_token_jaccard": float(
                    group["address_token_jaccard"].max()
                ),
                "parent_max_address_char_dice": float(
                    group["address_char_dice"].max()
                ),
                "parent_any_street_number_match": int(
                    group["street_number_match"].max()
                ),
                "parent_any_postcode_match": int(group["postcode_match"].max()),
                "parent_any_insee_match": int(group["insee_match"].max()),
                "parent_active_fraction": float(group["candidate_active"].mean()),
                "parent_historical_signal_max": float(
                    group["historical_signal_count"].max()
                ),
                "parent_current_signal_max": float(group["current_signal_count"].max()),
                "parent_official_source_max": float(
                    group["official_source_count"].max()
                ),
            }
        )
    output = pd.DataFrame(records)
    best = output.groupby("query_id")["_parent_raw_retrieval_score"].transform("max")
    output["parent_max_retrieval_score_relative"] = (
        output["_parent_raw_retrieval_score"] - best
    )
    return output.sort_values(
        ["query_id", "candidate_parent"], kind="stable"
    ).reset_index(drop=True)


def _ranker_matrix(
    rows: pd.DataFrame, features: Sequence[str], label_column: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert_identifier_free_features(features)
    ordered = rows.sort_values(["query_id", label_column], ascending=[True, False], kind="stable")
    raw_ids = ordered["query_id"].astype(str).to_numpy()
    if len(raw_ids) == 0:
        raise ValueError("No trainable ranking rows")
    starts = np.empty(len(raw_ids), dtype=bool)
    starts[0] = True
    starts[1:] = raw_ids[1:] != raw_ids[:-1]
    qid = np.cumsum(starts, dtype=np.int64) - 1
    if len(set(raw_ids)) != int(qid[-1]) + 1:
        raise ValueError("Ranking query groups are not contiguous")
    matrix = ordered[list(features)].fillna(0.0).to_numpy(dtype=np.float32)
    labels = ordered[label_column].to_numpy(dtype=np.int8)
    return matrix, labels, qid


def _fit_ranker(
    rows: pd.DataFrame,
    features: Sequence[str],
    label_column: str,
    params: Mapping[str, Any],
) -> xgb.XGBRanker:
    matrix, labels, qid = _ranker_matrix(rows, features, label_column)
    model = xgb.XGBRanker(**dict(params))
    model.fit(matrix, labels, qid=qid, verbose=False)
    return model


def _parent_training_rows(parent_rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    output: list[pd.DataFrame] = []
    missing = 0
    no_negative = 0
    for _query_id, group in parent_rows.groupby("query_id", sort=True):
        human = normalize_siret(group.iloc[0]["human_siret"])
        positive = group["candidate_parent"].eq(human[:9])
        if int(positive.sum()) != 1:
            missing += 1
            continue
        if len(group) < 2:
            no_negative += 1
            continue
        retained = group.copy()
        retained["parent_label"] = positive.astype(np.int8)
        output.append(retained)
    if not output:
        raise ValueError("No natural parent positives available for training")
    return pd.concat(output, ignore_index=True), {
        "queries_missing_natural_parent": missing,
        "queries_without_parent_negative": no_negative,
    }


def fit_parent_ranker(
    candidates: pd.DataFrame, config: HierarchicalAdmissionConfig
) -> tuple[xgb.XGBRanker, dict[str, int]]:
    training, diagnostics = _parent_training_rows(build_parent_rows(candidates))
    return (
        _fit_ranker(
            training,
            PARENT_FEATURES,
            "parent_label",
            config.xgb_params("parent"),
        ),
        diagnostics,
    )


def attach_parent_scores(
    model: xgb.XGBRanker, candidates: pd.DataFrame
) -> pd.DataFrame:
    """Attach parent predictions without reading any human label column."""
    parents = build_parent_rows(candidates)
    parents["_parent_model_score"] = model.predict(
        parents[list(PARENT_FEATURES)].fillna(0.0).to_numpy(dtype=np.float32)
    ).astype(np.float64)
    parents["parent_rank"] = parents.groupby("query_id")["_parent_model_score"].rank(
        method="first", ascending=False
    )
    parents["parent_rank_reciprocal"] = 1.0 / parents["parent_rank"]
    best = parents.groupby("query_id")["_parent_model_score"].transform("max")
    parents["parent_score_gap_to_best"] = parents["_parent_model_score"] - best

    shifted = (
        parents["_parent_model_score"]
        - parents.groupby("query_id")["_parent_model_score"].transform("max")
    ).clip(lower=-30.0, upper=0.0)
    exponent = np.exp(shifted)
    parents["parent_score_probability"] = exponent / exponent.groupby(
        parents["query_id"]
    ).transform("sum")
    columns = [
        "query_id",
        "candidate_parent",
        "_parent_model_score",
        "parent_rank",
        "parent_rank_reciprocal",
        "parent_score_probability",
        "parent_score_gap_to_best",
    ]
    return candidates.merge(
        parents[columns], on=["query_id", "candidate_parent"], how="left", validate="many_to_one"
    )


def _site_training_rows(
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    output: list[pd.DataFrame] = []
    missing = 0
    same_site_excluded = 0
    no_negative = 0
    for _query_id, group in candidates.groupby("query_id", sort=True):
        human = normalize_siret(group.iloc[0]["human_siret"])
        positive = group["candidate_siret"].eq(human)
        if int(positive.sum()) != 1:
            missing += 1
            continue
        operational = set(
            _operational_labels(
                {
                    "acceptable_sirets_operational_json": group.iloc[0][
                        "operational_sirets_json"
                    ]
                },
                human,
            )
        )
        equivalent = group["candidate_siret"].isin(operational - {human})
        same_site_excluded += int(equivalent.sum())
        eligible = group[~equivalent].copy()
        if len(eligible) < 2:
            no_negative += 1
            continue
        eligible["siret_label"] = eligible["candidate_siret"].eq(human).astype(np.int8)
        output.append(eligible)
    if not output:
        raise ValueError("No natural exact SIRET positives available for training")
    return pd.concat(output, ignore_index=True), {
        "queries_missing_natural_siret": missing,
        "same_parent_same_site_negatives_excluded": same_site_excluded,
        "queries_without_siret_negative": no_negative,
    }


def fit_siret_ranker(
    candidates_with_crossfit_parent_scores: pd.DataFrame,
    config: HierarchicalAdmissionConfig,
) -> tuple[xgb.XGBRanker, dict[str, int]]:
    training, diagnostics = _site_training_rows(candidates_with_crossfit_parent_scores)
    return (
        _fit_ranker(
            training,
            SIRET_FEATURES,
            "siret_label",
            config.xgb_params("siret"),
        ),
        diagnostics,
    )


def attach_siret_scores(model: xgb.XGBRanker, candidates: pd.DataFrame) -> pd.DataFrame:
    scored = candidates.copy()
    scored["siret_score"] = model.predict(
        scored[list(SIRET_FEATURES)].fillna(0.0).to_numpy(dtype=np.float32)
    ).astype(np.float64)
    scored["final_score"] = scored["siret_score"]
    return scored


def inner_oof_parent_scores(
    candidates: pd.DataFrame,
    folds: Sequence[int],
    config: HierarchicalAdmissionConfig,
) -> pd.DataFrame:
    """Generate parent scores for site fitting without in-sample predictions."""
    observed = set(int(value) for value in candidates["fold"].unique())
    expected = set(int(value) for value in folds)
    if observed != expected or len(expected) < 2:
        raise ValueError("Inner parent cross-fitting needs at least two complete folds")
    pieces: list[pd.DataFrame] = []
    for heldout in sorted(expected):
        fit_rows = candidates[candidates["fold"].ne(heldout)]
        validation = candidates[candidates["fold"].eq(heldout)]
        parent_model, _ = fit_parent_ranker(fit_rows, config)
        labels = validation[["query_id", "human_siret"]].drop_duplicates("query_id")
        blind_validation = validation.copy()
        blind_validation["human_siret"] = ""
        scored = attach_parent_scores(parent_model, blind_validation).drop(
            columns=["human_siret"]
        )
        pieces.append(
            scored.merge(labels, on="query_id", how="left", validate="many_to_one")
        )
    output = pd.concat(pieces, ignore_index=True)
    if len(output) != len(candidates):
        raise AssertionError("Inner OOF parent scoring changed candidate population")
    return output.sort_values(
        ["query_id", "union_rank", "candidate_siret"], kind="stable"
    ).reset_index(drop=True)


@dataclass
class HierarchicalModels:
    parent: xgb.XGBRanker
    siret: xgb.XGBRanker
    diagnostics: Mapping[str, Any]


def fit_hierarchical_models(
    labelled_union: pd.DataFrame, config: HierarchicalAdmissionConfig
) -> HierarchicalModels:
    candidates = canonicalize_union(
        labelled_union, config, labels_allowed=True, require_labels=False
    )
    usable_ids = set(
        candidates.groupby("query_id")["human_siret"]
        .first()
        .loc[lambda values: values.ne("")]
        .index.astype(str)
    )
    unusable_queries = int(candidates["query_id"].nunique() - len(usable_ids))
    candidates = candidates[candidates["query_id"].isin(usable_ids)].copy()
    if candidates.empty:
        raise ValueError("No usable human-labelled query remains for v11 fit")
    observed = set(int(value) for value in candidates["fold"].unique())
    if observed != set(config.train_folds):
        raise ValueError("Final v11 fit may open folds 2/3/4 only")
    parent_oof = inner_oof_parent_scores(candidates, config.train_folds, config)
    siret_model, siret_diagnostics = fit_siret_ranker(parent_oof, config)
    parent_model, parent_diagnostics = fit_parent_ranker(candidates, config)
    return HierarchicalModels(
        parent=parent_model,
        siret=siret_model,
        diagnostics={
            "parent": parent_diagnostics,
            "siret": siret_diagnostics,
            "train_folds": list(config.train_folds),
            "parent_scores_for_siret_fit": "three_fold_oof",
            "unusable_human_queries_excluded": unusable_queries,
            "positive_injection": False,
        },
    )


def score_canonical_candidates(
    models: HierarchicalModels, candidates: pd.DataFrame
) -> pd.DataFrame:
    parent_scored = attach_parent_scores(models.parent, candidates)
    return attach_siret_scores(models.siret, parent_scored)


def select_top100(
    scored_candidates: pd.DataFrame, config: HierarchicalAdmissionConfig
) -> pd.DataFrame:
    """Select protected anchors/consensus, then a parent-diverse learned tail."""
    selected_groups: list[pd.DataFrame] = []
    order = ["siret_score", "_parent_model_score", "union_rank", "candidate_siret"]
    ascending = [False, False, True, True]
    for _query_id, group in scored_candidates.groupby("query_id", sort=True):
        group = group.sort_values(order, ascending=ascending, kind="stable")
        selected: list[int] = []
        selected_set: set[int] = set()
        reasons: dict[int, str] = {}

        anchors = group[group["is_anchor"].eq(1)].head(config.anchor_slots)
        for index in anchors.index:
            selected.append(index)
            selected_set.add(index)
            reasons[index] = "anchor"

        consensus_pool = group[
            group["is_consensus"].eq(1) & ~group.index.isin(selected_set)
        ]
        consensus_parents: set[str] = set()
        for index, row in consensus_pool.iterrows():
            parent = str(row["candidate_parent"])
            if parent in consensus_parents:
                continue
            selected.append(index)
            selected_set.add(index)
            reasons[index] = "consensus"
            consensus_parents.add(parent)
            if len(consensus_parents) >= config.consensus_slots:
                break

        parent_counts: dict[str, int] = {}
        for index in selected:
            parent = str(group.loc[index, "candidate_parent"])
            parent_counts[parent] = parent_counts.get(parent, 0) + 1
        for index, row in group[~group.index.isin(selected_set)].iterrows():
            if len(selected) >= config.max_candidates:
                break
            parent = str(row["candidate_parent"])
            if parent_counts.get(parent, 0) >= config.max_unprotected_per_parent:
                continue
            selected.append(index)
            selected_set.add(index)
            parent_counts[parent] = parent_counts.get(parent, 0) + 1
            reasons[index] = "ranked_parent_diverse"

        output = group.loc[selected].copy()
        output["selection_reason"] = [reasons[index] for index in selected]
        output["selected_rank"] = np.arange(1, len(output) + 1, dtype=np.int16)
        if len(output) > config.max_candidates:
            raise AssertionError("v11 emitted more than 100 candidates")
        selected_groups.append(output)
    return pd.concat(selected_groups, ignore_index=True)


def predict_top100(
    models: HierarchicalModels,
    unlabeled_union: pd.DataFrame,
    config: HierarchicalAdmissionConfig,
) -> pd.DataFrame:
    candidates = canonicalize_union(
        unlabeled_union, config, labels_allowed=False, require_labels=False
    )
    return select_top100(score_canonical_candidates(models, candidates), config)


def _wilson(successes: int, total: int, confidence: float) -> list[float] | None:
    if total <= 0:
        return None
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    observed = successes / total
    denominator = 1.0 + z * z / total
    center = (observed + z * z / (2.0 * total)) / denominator
    spread = (
        z
        * math.sqrt(observed * (1.0 - observed) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return [max(0.0, center - spread), min(1.0, center + spread)]


def _metric(successes: int, total: int) -> dict[str, Any]:
    return {
        "successes": int(successes),
        "total": int(total),
        "rate": (successes / total) if total else None,
        "wilson_95": _wilson(successes, total, 0.95),
        "wilson_99": _wilson(successes, total, 0.99),
    }


def evaluate_all_human(
    labelled_candidates: pd.DataFrame,
    selected: pd.DataFrame,
    config: HierarchicalAdmissionConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate every usable human label; no qualification changes the denominator."""
    outcomes: list[dict[str, Any]] = []
    selected_by_query = {
        str(query_id): group
        for query_id, group in selected.groupby("query_id", sort=False)
    }
    for query_id, group in labelled_candidates.groupby("query_id", sort=True):
        human = normalize_siret(group.iloc[0]["human_siret"])
        selected_group = selected_by_query.get(str(query_id), group.iloc[0:0])
        union_sirets = set(group["candidate_siret"].astype(str))
        selected_sirets = set(selected_group["candidate_siret"].astype(str))
        parent_present = human[:9] in set(group["candidate_parent"].astype(str))
        operational = set(
            _operational_labels(
                {
                    "acceptable_sirets_operational_json": group.iloc[0][
                        "operational_sirets_json"
                    ]
                },
                human,
            )
        )
        truth = group[group["candidate_siret"].eq(human)]
        truth_stage = truth.iloc[0] if not truth.empty else None
        outcomes.append(
            {
                "query_id": str(query_id),
                "fold": int(group.iloc[0]["fold"]),
                "human_siret": human,
                "parent_oracle_hit": bool(parent_present),
                "exact_oracle_hit": human in union_sirets,
                "operational_oracle_hit": bool(operational & union_sirets),
                "exact_hit_at_100": human in selected_sirets,
                "operational_hit_at_100": bool(operational & selected_sirets),
                "union_candidate_count": int(len(group)),
                "selected_candidate_count": int(len(selected_group)),
                "retrieval_latency_ms": float(group.iloc[0]["retrieval_latency_ms"]),
                "oracle_baseline_hit": bool(
                    truth_stage is not None and truth_stage["stage_baseline"]
                ),
                "oracle_parent_hit": bool(
                    truth_stage is not None and truth_stage["stage_parent"]
                ),
                "oracle_parent_geo_hit": bool(
                    truth_stage is not None and truth_stage["stage_parent_geo"]
                ),
                "segment_state": str(group.iloc[0]["segment_state"]),
                "segment_origin": str(group.iloc[0]["segment_origin"]),
                "segment_unseen_siren": int(group.iloc[0]["segment_unseen_siren"]),
                "segment_new_site_known_siren": int(
                    group.iloc[0]["segment_new_site_known_siren"]
                ),
                "segment_mega": int(group.iloc[0]["segment_mega"]),
            }
        )
    frame = pd.DataFrame(outcomes)
    return frame, summarize_outcomes(frame, config)


def summarize_outcomes(
    frame: pd.DataFrame, config: HierarchicalAdmissionConfig
) -> dict[str, Any]:
    """Aggregate complete all-human query outcomes without changing population."""
    if frame.empty:
        raise ValueError("Cannot summarize an empty all-human outcome set")
    def metrics_for(part: pd.DataFrame) -> dict[str, Any]:
        total = len(part)
        return {
            "parent_oracle": _metric(int(part["parent_oracle_hit"].sum()), total),
            "exact_oracle_at_2000": _metric(int(part["exact_oracle_hit"].sum()), total),
            "exact_recall_at_100": _metric(int(part["exact_hit_at_100"].sum()), total),
            "operational_recall_at_100": _metric(
                int(part["operational_hit_at_100"].sum()), total
            ),
        }

    overall = metrics_for(frame)
    by_fold = {
        str(int(fold)): metrics_for(part)
        for fold, part in frame.groupby("fold", sort=True)
    }
    max_candidates = int(frame["selected_candidate_count"].max())
    latency_p95 = float(np.quantile(frame["retrieval_latency_ms"], 0.95))
    latency_p99 = float(np.quantile(frame["retrieval_latency_ms"], 0.99))
    recall = float(overall["exact_recall_at_100"]["rate"] or 0.0)
    oracle = float(overall["exact_oracle_at_2000"]["rate"] or 0.0)
    lower95 = float((overall["exact_recall_at_100"]["wilson_95"] or [0.0])[0])
    worst_fold = min(
        float(value["exact_recall_at_100"]["rate"] or 0.0)
        for value in by_fold.values()
    )
    gates = {
        "oracle_at_2000": {
            "observed": oracle,
            "minimum": float(config.gates["minimum_oracle_at_2000"]),
            "passed": oracle >= float(config.gates["minimum_oracle_at_2000"]),
        },
        "recall_at_100": {
            "observed": recall,
            "minimum": float(config.gates["minimum_recall_at_100"]),
            "passed": recall >= float(config.gates["minimum_recall_at_100"]),
        },
        "wilson95_lower": {
            "observed": lower95,
            "minimum": float(config.gates["minimum_wilson95_lower"]),
            "passed": lower95 >= float(config.gates["minimum_wilson95_lower"]),
        },
        "worst_outer_fold_recall": {
            "observed": worst_fold,
            "minimum": float(config.gates["minimum_outer_fold_recall"]),
            "passed": worst_fold >= float(config.gates["minimum_outer_fold_recall"]),
        },
        "candidate_ceiling": {
            "observed": max_candidates,
            "maximum": config.max_candidates,
            "passed": max_candidates <= config.max_candidates,
        },
        "warm_p95_ms": {
            "observed": latency_p95,
            "maximum": float(config.gates["maximum_warm_p95_ms"]),
            "passed": latency_p95 <= float(config.gates["maximum_warm_p95_ms"]),
        },
        "warm_p99_ms": {
            "observed": latency_p99,
            "maximum": float(config.gates["maximum_warm_p99_ms"]),
            "passed": latency_p99 <= float(config.gates["maximum_warm_p99_ms"]),
        },
    }
    segment_columns = (
        "segment_state",
        "segment_origin",
        "segment_unseen_siren",
        "segment_new_site_known_siren",
        "segment_mega",
    )
    segments = {
        column: {
            str(value): metrics_for(part)
            for value, part in frame.groupby(column, sort=True, dropna=False)
        }
        for column in segment_columns
    }
    oracle_explanatory = {
        "baseline_current": _metric(int(frame["oracle_baseline_hit"].sum()), len(frame)),
        "parent_complete": _metric(int(frame["oracle_parent_hit"].sum()), len(frame)),
        "parent_plus_geography": _metric(
            int(frame["oracle_parent_geo_hit"].sum()), len(frame)
        ),
        "official_stack_complete": overall["exact_oracle_at_2000"],
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "policy_id": config.policy_id,
        "population": "ALL_USABLE_HUMAN_LABELS",
        "query_count": int(len(frame)),
        "metrics": overall,
        "outer_folds": by_fold,
        "metrics_by_segment": segments,
        "oracle_explanatory": oracle_explanatory,
        "latency_ms": {"p95": latency_p95, "p99": latency_p99},
        "gates": gates,
        "integrity": {
            "full_union_validation": True,
            "maximum_union_candidates": int(frame["union_candidate_count"].max()),
            "positive_injection": False,
            "same_parent_same_site_is_exact_negative": False,
            "feature_identifiers": False,
        },
        "verdict": (
            "PASS_OOF_GATES" if all(value["passed"] for value in gates.values()) else "PIVOT"
        ),
    }
    return summary


def fit_outer_oof_models(
    sampled_candidates: pd.DataFrame,
    config: HierarchicalAdmissionConfig,
) -> tuple[dict[int, HierarchicalModels], dict[str, Any]]:
    """Fit the three outer models from a preregistered fit-only sample."""
    observed = set(int(value) for value in sampled_candidates["fold"].unique())
    if observed != set(config.train_folds):
        raise ValueError("Outer fit sample must contain exactly folds 2/3/4")
    models: dict[int, HierarchicalModels] = {}
    diagnostics: dict[str, Any] = {}
    for heldout in config.train_folds:
        fit_folds = tuple(fold for fold in config.train_folds if fold != heldout)
        outer_train = sampled_candidates[
            sampled_candidates["fold"].isin(fit_folds)
        ].copy()
        inner_parent = inner_oof_parent_scores(outer_train, fit_folds, config)
        siret_model, siret_diag = fit_siret_ranker(inner_parent, config)
        parent_model, parent_diag = fit_parent_ranker(outer_train, config)
        models[heldout] = HierarchicalModels(
            parent=parent_model,
            siret=siret_model,
            diagnostics={
                "fit_folds": list(fit_folds),
                "parent": parent_diag,
                "siret": siret_diag,
                "inner_parent_scores": "OOF_ONLY",
            },
        )
        diagnostics[str(heldout)] = dict(models[heldout].diagnostics)
    return models, diagnostics


def outer_oof_evaluate(
    labelled_union: pd.DataFrame, config: HierarchicalAdmissionConfig
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Three-fold nested OOF evaluation on complete candidate unions."""
    candidates = canonicalize_union(
        labelled_union, config, labels_allowed=True, require_labels=False
    )
    all_query_count = int(candidates["query_id"].nunique())
    usable_ids = set(
        candidates.groupby("query_id")["human_siret"]
        .first()
        .loc[lambda values: values.ne("")]
        .index.astype(str)
    )
    candidates = candidates[candidates["query_id"].isin(usable_ids)].copy()
    if candidates.empty:
        raise ValueError("No usable human-labelled query remains for outer OOF")
    observed = set(int(value) for value in candidates["fold"].unique())
    if observed != set(config.train_folds):
        raise ValueError("Outer OOF must contain exactly folds 2/3/4")
    selected_parts: list[pd.DataFrame] = []
    scored_parts: list[pd.DataFrame] = []
    fold_diagnostics: dict[str, Any] = {}
    for heldout in config.train_folds:
        outer_train_folds = tuple(fold for fold in config.train_folds if fold != heldout)
        outer_train = candidates[candidates["fold"].isin(outer_train_folds)].copy()
        outer_validation = candidates[candidates["fold"].eq(heldout)].copy()

        inner_parent = inner_oof_parent_scores(
            outer_train, outer_train_folds, config
        )
        siret_model, siret_diag = fit_siret_ranker(inner_parent, config)
        parent_model, parent_diag = fit_parent_ranker(outer_train, config)
        blind_validation = outer_validation.copy()
        blind_validation["human_siret"] = ""
        validation_scored = attach_siret_scores(
            siret_model, attach_parent_scores(parent_model, blind_validation)
        )
        validation_selected = select_top100(validation_scored, config)
        selected_parts.append(validation_selected)
        scored_parts.append(validation_scored)
        fold_diagnostics[str(heldout)] = {
            "fit_folds": list(outer_train_folds),
            "validation_fold": heldout,
            "validation_candidates": int(len(outer_validation)),
            "validation_queries": int(outer_validation["query_id"].nunique()),
            "validation_union_max": int(
                outer_validation.groupby("query_id").size().max()
            ),
            "parent": parent_diag,
            "siret": siret_diag,
            "inner_parent_scores": "OOF_ONLY",
        }

    selected = pd.concat(selected_parts, ignore_index=True)
    scored = pd.concat(scored_parts, ignore_index=True)
    outcomes, summary = evaluate_all_human(candidates, selected, config)
    summary["nested_oof"] = {
        "outer_folds": list(config.train_folds),
        "fold_diagnostics": fold_diagnostics,
        "fold0_opened": False,
        "fold1_opened": False,
        "validation_candidate_sampling": False,
        "input_query_count": all_query_count,
        "usable_human_query_count": len(usable_ids),
        "unusable_human_queries_excluded": all_query_count - len(usable_ids),
    }
    return selected, outcomes, summary


__all__ = [
    "HierarchicalAdmissionConfig",
    "HierarchicalModels",
    "PARENT_FEATURES",
    "SIRET_FEATURES",
    "assert_identifier_free_features",
    "attach_parent_scores",
    "attach_siret_scores",
    "build_parent_rows",
    "canonicalize_union",
    "evaluate_all_human",
    "fit_hierarchical_models",
    "fit_outer_oof_models",
    "fit_parent_ranker",
    "fit_siret_ranker",
    "inner_oof_parent_scores",
    "outer_oof_evaluate",
    "predict_top100",
    "score_canonical_candidates",
    "select_top100",
    "summarize_outcomes",
]
