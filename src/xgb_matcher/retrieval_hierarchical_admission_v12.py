"""Leakage-safe hierarchical admission for retrieval v12.

The v12 admission is deliberately independent from the historical v11
implementation.  Its important ordering invariant is::

    complete query union -> feature computation -> fit-only row sampling

Consequently, sampling can never change a rank percentile, a parent aggregate
or an intra-parent rank.  Parent scores used by the establishment ranker are
always cross-fitted.  Production prediction accepts label-free frames only.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import lru_cache
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

from .retrieval_top100_cascade_v12 import CASCADE_PARENT_FEATURES


SCHEMA_VERSION = "sireto-retrieval-hierarchical-admission-v12"
OPERATIONAL_EQUIVALENCE_MODE = (
    "EXACT_ONLY_PENDING_SEALED_SAME_SITE_EVIDENCE"
)
METRIC_CUTOFFS = (20, 50, 100, 200, 500, 1_000, 2_000)
EXCLUSIVE_LOSS_REASONS = (
    "OFFICIAL_CATALOG_ABSENT",
    "PARENT_NOT_RETRIEVED",
    "SITE_NOT_EXPANDED",
    "EVICTED_MASTER_TO_500",
    "EVICTED_500_TO_100",
)
AUDIT_ONLY_LOSS_REASONS = ("OFFICIAL_CATALOG_STATUS_UNKNOWN",)
LABEL_COLUMNS = frozenset(
    {
        "human_siret",
        "human_ground_truth_siret",
        "historical_ground_truth_siret",
        "ground_truth_siret",
        "gt_siret",
        "acceptable_sirets_operational",
        "acceptable_sirets_operational_json",
    }
)

# The schema is pinned in configuration for a production run.  This default
# covers the complete sparse/open-data union currently emitted by SIRETO.
DEFAULT_CHANNELS = (
    "legal_name_exact",
    "trade_name_exact",
    "site_name_exact",
    "number_exact",
    "fielded_name_bm25",
    "legal_name_word",
    "trade_name_word",
    "site_name_word",
    "current_name_char",
    "historical_name_word",
    "historical_address_word",
    "supporting_name_word",
    "supporting_address_word",
    "name_exact",
    "address_exact",
    "name_word",
    "address_word",
    "name_char",
    "address_char",
    "rne_name",
    "rne_address",
    "bodacc_name",
    "bodacc_address",
    "hierarchical",
    "direct_parent_expansion",
    "consensus_parent_expansion",
    "legal_entity_role_name",
    "sector_name",
    "sector_address",
    "sector_parent",
    "contrastive_siren_rescue",
    "v12_conditional_siren_rescue",
    "official_successor",
    "bodacc_relation",
    "geo_insee_census",
    "geo_exact_cell",
    "geo_number_cell",
    "v11_parent_expansion",
    "v12_parent_local_expansion",
    "v12_rescue_local_expansion",
)

# Consensus is counted once per evidence family, never once per raw channel or
# source string.  Derived variants from the same underlying evidence therefore
# cannot manufacture consensus.
DEFAULT_CHANNEL_FAMILIES: Mapping[str, tuple[str, ...]] = {
    "current_name": (
        "legal_name_exact",
        "trade_name_exact",
        "site_name_exact",
        "fielded_name_bm25",
        "legal_name_word",
        "trade_name_word",
        "site_name_word",
        "current_name_char",
        "name_exact",
        "name_word",
        "name_char",
    ),
    "current_address": (
        "number_exact",
        "address_exact",
        "address_word",
        "address_char",
    ),
    "historical_official": (
        "historical_name_word",
        "historical_address_word",
        "supporting_name_word",
        "supporting_address_word",
    ),
    "rne": ("rne_name", "rne_address", "legal_entity_role_name"),
    "bodacc": ("bodacc_name", "bodacc_address", "bodacc_relation"),
    "geographic_cell": (
        "geo_insee_census",
        "geo_exact_cell",
        "geo_number_cell",
    ),
    "parent_sparse": (
        "hierarchical",
        "direct_parent_expansion",
        "consensus_parent_expansion",
        "v11_parent_expansion",
        "v12_parent_local_expansion",
        "v12_rescue_local_expansion",
        "contrastive_siren_rescue",
        "v12_conditional_siren_rescue",
    ),
    "official_relation": ("official_successor",),
    "sector_official": ("sector_name", "sector_address", "sector_parent"),
}

EXACT_NAME_CHANNELS = frozenset(
    {"legal_name_exact", "trade_name_exact", "site_name_exact", "name_exact"}
)
EXACT_GEO_CHANNELS = frozenset(
    {"address_exact", "geo_exact_cell", "geo_number_cell"}
)
CURRENT_CHANNELS = frozenset(
    set(DEFAULT_CHANNEL_FAMILIES["current_name"])
    | set(DEFAULT_CHANNEL_FAMILIES["current_address"])
)
HISTORICAL_CHANNELS = frozenset(DEFAULT_CHANNEL_FAMILIES["historical_official"])
RELATION_CHANNELS = frozenset(
    set(DEFAULT_CHANNEL_FAMILIES["official_relation"])
    | {"bodacc_relation", "legal_entity_role_name"}
)
NAME_EVIDENCE_CHANNELS = frozenset(
    set(DEFAULT_CHANNEL_FAMILIES["current_name"])
    | {"historical_name_word", "supporting_name_word", "rne_name", "bodacc_name", "legal_entity_role_name", "sector_name", "contrastive_siren_rescue", "v12_conditional_siren_rescue"}
)
ADDRESS_EVIDENCE_CHANNELS = frozenset(
    set(DEFAULT_CHANNEL_FAMILIES["current_address"])
    | {"historical_address_word", "supporting_address_word", "rne_address", "bodacc_address", "sector_address"}
    | set(DEFAULT_CHANNEL_FAMILIES["geographic_cell"])
)

_SPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^A-Z0-9]+")
_LEADING_NUMBER_RE = re.compile(r"^(\d{1,5})(?:\s|$)")
_SOURCE_SPLIT_RE = re.compile(r"[+,|;\s]+")


@lru_cache(maxsize=16_384)
def _parse_iso_date(value: str) -> date | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return date.fromisoformat(raw[:10])
    except ValueError:
        return None


def _source_families(values: Iterable[Any]) -> set[str]:
    """Collapse backend aliases to independent official source families."""
    families: set[str] = set()
    for value in values:
        raw = normalize_text(value)
        for token in filter(None, _SOURCE_SPLIT_RE.split(raw)):
            if "BODACC" in token:
                families.add("BODACC")
            elif token in {"RNE", "INPI"} or "RNE" in token:
                families.add("RNE")
            elif token in {"SIRENE", "SIREN", "INSEE"} or "SIRENE" in token:
                families.add("SIRENE")
            elif "SECTOR" in token or "SECTEUR" in token:
                families.add("SECTOR")
    return families


def _temporal_features(
    valid_from: Any, valid_to: Any, reference: date
) -> dict[str, float]:
    start = _parse_iso_date(str(valid_from or ""))
    end = _parse_iso_date(str(valid_to or ""))
    return {
        "temporal_valid_from_present": float(start is not None),
        "temporal_valid_to_present": float(end is not None),
        "temporal_open_ended": float(end is None),
        "temporal_valid_at_reference": float(
            (start is None or start <= reference) and (end is None or end >= reference)
        ),
        "temporal_starts_after_reference": float(start is not None and start > reference),
        "temporal_ended_before_reference": float(end is not None and end < reference),
        "temporal_start_distance_days_log1p": float(
            math.log1p(abs((reference - start).days)) if start is not None else 0.0
        ),
        "temporal_end_distance_days_log1p": float(
            math.log1p(abs((reference - end).days)) if end is not None else 0.0
        ),
    }


def _relation_features(
    signals: Sequence[Mapping[str, Any]], reference: date
) -> dict[str, float]:
    relation_signals = [
        signal
        for signal in signals
        if normalize_text(signal.get("relation_direction"))
    ]
    directions = {
        normalize_text(signal.get("relation_direction"))
        for signal in relation_signals
    }
    dates = [
        parsed
        for parsed in (
            _parse_iso_date(
                str(
                    signal.get("relation_date")
                    or signal.get("event_date")
                    or signal.get("date")
                    or ""
                )
            )
            for signal in relation_signals
        )
        if parsed is not None
    ]
    nearest_age = min((abs((reference - value).days) for value in dates), default=0)
    known = {"SUCCESSOR", "PREDECESSOR", "BOTH"}
    return {
        "relation_successor_present": float("SUCCESSOR" in directions or "BOTH" in directions),
        "relation_predecessor_present": float("PREDECESSOR" in directions or "BOTH" in directions),
        "relation_both_present": float("BOTH" in directions),
        "relation_direction_other_present": float(bool(directions - known)),
        "relation_dated_present": float(bool(dates)),
        "relation_date_missing": float(bool(relation_signals) and not dates),
        "relation_nearest_age_days_log1p": float(math.log1p(nearest_age)),
    }


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
class AdmissionConfigV12:
    """Flexible loader for a single, pinned v12 experiment configuration.

    A future v12 config may live next to the v11 config.  Until then, the
    loader accepts the common all-human fields from v11 and overlays explicit
    ``admission_v12`` / ``retrieval_v12`` sections when present.  It never
    mutates the source mapping.
    """

    raw: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "AdmissionConfigV12":
        return cls(dict(raw))

    @classmethod
    def load(cls, path: Path | str) -> "AdmissionConfigV12":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def default(cls) -> "AdmissionConfigV12":
        return cls(
            {
                "schema_version": "sireto-retrieval-all-human-v12-config-1",
                "policy_id": "retrieval-v12-all-human-exact-primary",
                "folds": {"train": [2, 3, 4], "development_diagnostic": 0, "test_locked": 1},
                "retrieval": {
                    "maximum_internal_candidates": 2_000,
                    "maximum_final_candidates": 100,
                },
                "admission_v12": {
                    "channels": list(DEFAULT_CHANNELS),
                    "channel_families": {
                        key: list(value) for key, value in DEFAULT_CHANNEL_FAMILIES.items()
                    },
                    "independent_consensus_min_families": 2,
                    "fit_candidate_sampling": {
                        "maximum_candidates_per_query": 128,
                        "union_head": 64,
                        "channel_head": 4,
                        "retain_natural_positive_if_present": True,
                    },
                    "selection": {
                        "parent_probability_weight": 0.20,
                        "repeat_parent_penalty": 0.035,
                    },
                    "temporal_reference_date": "2026-08-24",
                    "xgboost": {
                        "parent": _default_xgb(20260912),
                        "site": _default_xgb(20260913),
                    },
                },
                "gates": {
                    "minimum_oof_oracle_recall_at_2000": 0.995,
                    "minimum_oof_exact_recall_at_100": 0.992,
                    "minimum_oof_exact_recall_at_100_wilson95_lower": 0.99,
                    "minimum_each_outer_fold_exact_recall_at_100": 0.988,
                },
            }
        )

    def __post_init__(self) -> None:
        if not isinstance(self.raw, Mapping):
            raise ValueError("v12 configuration must be a mapping")
        if self.train_folds != (2, 3, 4) or self.dev_fold != 0 or self.test_fold != 1:
            raise ValueError("v12 requires train folds 2/3/4, diagnostic fold 0 and locked fold 1")
        if self.internal_cap != 2_000 or self.final_cap != 100:
            raise ValueError("v12 requires strict 2,000/100 candidate ceilings")
        if self.prerank_cap != 500:
            raise ValueError("v12 requires a strict PreRank/hydration ceiling of 500")
        if not self.channels or len(set(self.channels)) != len(self.channels):
            raise ValueError("v12 channels must be non-empty and unique")
        family_channels = [channel for values in self.channel_families.values() for channel in values]
        unknown = sorted(set(family_channels) - set(self.channels))
        if unknown:
            raise ValueError(f"v12 evidence families reference unknown channels: {unknown}")
        sampling = self.fit_candidate_sampling
        if not 1 <= int(sampling["maximum_candidates_per_query"]) <= self.internal_cap:
            raise ValueError("invalid v12 fit-only sample cap")
        if sampling.get("retain_natural_positive_if_present") is not True:
            raise ValueError("v12 sampling must retain a naturally present positive")
        if self.consensus_min_families < 2:
            raise ValueError("independent consensus needs at least two evidence families")
        if _parse_iso_date(self.temporal_reference_date) is None:
            raise ValueError("v12 requires an ISO frozen temporal reference date")
        for level in ("parent", "site"):
            if self.xgb_params(level).get("objective") != "rank:ndcg":
                raise ValueError("v12 rankers must use rank:ndcg")
        model_features = site_feature_names(self)
        if len(model_features) != len(set(model_features)):
            raise ValueError("v12 site feature schema contains duplicate names")

    @property
    def policy_id(self) -> str:
        return str(self.raw.get("policy_id") or "retrieval-v12-all-human-exact-primary")

    @property
    def train_folds(self) -> tuple[int, ...]:
        folds = self.raw.get("folds") or {}
        return tuple(int(value) for value in folds.get("outer_oof", folds.get("train", (2, 3, 4))))

    @property
    def dev_fold(self) -> int:
        folds = self.raw.get("folds") or {}
        return int(folds.get("development_contaminated", folds.get("development_diagnostic", 0)))

    @property
    def test_fold(self) -> int:
        return int((self.raw.get("folds") or {}).get("test_locked", 1))

    @property
    def _admission(self) -> Mapping[str, Any]:
        return self.raw.get("admission_v12") or self.raw.get("admission") or {}

    @property
    def _retrieval(self) -> Mapping[str, Any]:
        return self.raw.get("retrieval_v12") or self.raw.get("retrieval") or {}

    @property
    def internal_cap(self) -> int:
        return int(self._retrieval.get("maximum_internal_candidates", 2_000))

    @property
    def final_cap(self) -> int:
        architecture = self.raw.get("architecture") or {}
        return int(architecture.get("maximum_final_sirets", self._retrieval.get("maximum_final_candidates", 100)))

    @property
    def prerank_cap(self) -> int:
        architecture = self.raw.get("architecture") or {}
        return int(architecture.get("maximum_hydrated_sirets", architecture.get("normal_shortlist_siret_cap", 500)))

    @property
    def channels(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self._admission.get("channels", DEFAULT_CHANNELS))

    @property
    def channel_families(self) -> Mapping[str, tuple[str, ...]]:
        raw = self._admission.get("channel_families") or DEFAULT_CHANNEL_FAMILIES
        return {str(key): tuple(str(value) for value in values) for key, values in raw.items()}

    @property
    def consensus_min_families(self) -> int:
        return int(self._admission.get("independent_consensus_min_families", 2))

    @property
    def temporal_reference_date(self) -> str:
        return str(self._admission.get("temporal_reference_date") or "2026-08-24")

    @property
    def fit_candidate_sampling(self) -> Mapping[str, Any]:
        defaults = {
            "maximum_candidates_per_query": 128,
            "union_head": 64,
            "channel_head": 4,
            "retain_natural_positive_if_present": True,
        }
        defaults.update(dict(self._admission.get("fit_candidate_sampling") or {}))
        defaults.pop("validation_sampling", None)
        return defaults

    @property
    def selection(self) -> Mapping[str, float]:
        defaults = {"parent_probability_weight": 0.20, "repeat_parent_penalty": 0.035}
        defaults.update(dict(self._admission.get("selection") or {}))
        return {key: float(value) for key, value in defaults.items()}

    def xgb_params(self, level: str) -> Mapping[str, Any]:
        defaults = _default_xgb(20260912 if level == "parent" else 20260913)
        raw_xgb = self._admission.get("xgboost") or {}
        source = raw_xgb.get(level)
        if source is None and level == "site":
            source = raw_xgb.get("siret")
        defaults.update(dict(source or {}))
        return defaults

    @property
    def gates(self) -> Mapping[str, float]:
        raw = self.raw.get("gates") or {}
        master = raw.get("master_exact_recall_at_2000") or {}
        prerank = raw.get("prerank_exact_recall_at_500") or {}
        product = raw.get("product_exact_recall_at_100") or {}
        return {
            "oracle_2000": float(master.get("minimum", raw.get("minimum_oof_oracle_recall_at_2000", 0.995))),
            "prerank_500": float(prerank.get("minimum", 0.994)),
            "recall_100": float(product.get("minimum", raw.get("minimum_oof_exact_recall_at_100", 0.992))),
            "wilson_100": float(product.get("wilson95_lower_minimum", raw.get("minimum_oof_exact_recall_at_100_wilson95_lower", 0.99))),
            "worst_fold_100": float(product.get("minimum_each_outer_fold", raw.get("minimum_each_outer_fold_exact_recall_at_100", 0.988))),
            "latency_p95_ms": float(
                (raw.get("warm_end_to_end_latency_ms") or {}).get(
                    "p95_maximum", 1_000.0
                )
            ),
            "latency_p99_ms": float(
                (raw.get("warm_end_to_end_latency_ms") or {}).get(
                    "p99_maximum", 2_000.0
                )
            ),
        }


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


def _float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


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
    return normalize_text(value) in {"1", "TRUE", "T", "YES", "Y", "OUI", "A"}


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
        if name in row and _values(row.get(name)):
            return row.get(name)
    return ""


def _candidate_values(row: Mapping[str, Any], names: Sequence[str]) -> list[str]:
    values: set[str] = set()
    for name in names:
        for raw in _values(row.get(name)):
            normalized = normalize_text(raw)
            if normalized:
                values.add(normalized)
    return sorted(values)


def _token_jaccard(left: str, right: str) -> float:
    a, b = set(normalize_text(left).split()), set(normalize_text(right).split())
    return len(a & b) / len(a | b) if a and b else 0.0


def _char_dice(left: str, right: str) -> float:
    def grams(value: str) -> set[str]:
        normalized = f" {normalize_text(value).replace(' ', '_')} "
        return {normalized[index : index + 3] for index in range(max(0, len(normalized) - 2))}

    a, b = grams(left), grams(right)
    return 2.0 * len(a & b) / (len(a) + len(b)) if a and b else 0.0


def _maximum(query: str, candidates: Iterable[str], metric: Any) -> float:
    return max((metric(query, candidate) for candidate in candidates), default=0.0)


def _leading_number(address: str) -> str:
    match = _LEADING_NUMBER_RE.match(normalize_text(address))
    return match.group(1) if match else ""


def _human_label(row: Mapping[str, Any]) -> str:
    for column in ("human_ground_truth_siret", "ground_truth_siret", "historical_ground_truth_siret", "gt_siret"):
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
    raw = row.get("acceptable_sirets_operational_json", row.get("acceptable_sirets_operational"))
    output = {human} if human else set()
    for value in _values(raw):
        try:
            output.add(normalize_siret(value))
        except ValueError:
            pass
    return tuple(sorted(output))


def reject_label_columns(frame: pd.DataFrame) -> None:
    present = sorted(set(frame.columns) & LABEL_COLUMNS)
    if present:
        raise ValueError("Production v12 input contains forbidden label columns: " + ", ".join(present))


def _raw_channels(row: Mapping[str, Any], configured: Sequence[str]) -> set[str]:
    # Only actual channel names are accepted.  Source/backend strings are not
    # channels and can never inflate consensus.
    configured_set = set(configured)
    output = {value for value in _values(row.get("channels")) if value in configured_set}
    output.update(
        str(signal.get("channel"))
        for signal in _signals(row.get("signals"))
        if str(signal.get("channel")) in configured_set
    )
    for channel in configured:
        if _float(row.get(f"{channel}_rank"), 0.0) >= 1:
            output.add(channel)
    return output


def _signals(value: Any) -> list[Mapping[str, Any]]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _raw_candidate_records(
    frame: pd.DataFrame,
    config: AdmissionConfigV12,
    *,
    labels_allowed: bool,
) -> pd.DataFrame:
    required = {"query_id", "fold", "crm_name", "crm_address"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"v12 candidate union is missing columns: {missing}")
    if frame.empty:
        raise ValueError("v12 candidate union is empty")
    if not labels_allowed:
        reject_label_columns(frame)

    reference_date = _parse_iso_date(config.temporal_reference_date)
    if reference_date is None:  # guarded by config validation; keep fail-closed
        raise ValueError("invalid frozen v12 temporal reference date")
    records: list[dict[str, Any]] = []
    for raw in frame.to_dict("records"):
        candidate = normalize_siret(_first(raw, ("candidate_siret", "siret")))
        parent = re.sub(r"\D", "", str(_first(raw, ("candidate_parent", "candidate_siren", "siren"))))
        parent = parent.zfill(9) if parent else candidate[:9]
        if parent != candidate[:9]:
            raise ValueError("candidate establishment and parent disagree")
        # Master and PreRank are separate views.  ``master_rank`` measures
        # coverage in the <=2,000 ledger; ``hydration_rank`` identifies the
        # <=500 rows eligible for rich site ranking.  When an explicit
        # hydration column exists, a null value means "not in PreRank" and
        # must never fall back to master rank.
        master_rank = int(
            _float(_first(raw, ("master_rank", "union_rank")), 0.0)
        )
        if master_rank < 1:
            raise ValueError("master/union rank must be positive")
        if "hydration_rank" in raw:
            hydration_value = _float(raw.get("hydration_rank"), 0.0)
            heuristic_hydration_rank = (
                int(hydration_value) if hydration_value >= 1 else None
            )
        else:
            fallback_rank = int(_float(raw.get("union_rank"), 0.0))
            heuristic_hydration_rank = (
                fallback_rank
                if 1 <= fallback_rank <= config.prerank_cap
                else None
            )
        names = _candidate_values(
            raw,
            ("candidate_names", "candidate_names_json", "candidate_name", "names", "denomination", "enseigne"),
        )
        addresses = _candidate_values(
            raw,
            ("candidate_addresses", "candidate_addresses_json", "candidate_address", "addresses", "address", "adresse"),
        )
        crm_name = normalize_text(raw.get("crm_name"))
        crm_address = normalize_text(raw.get("crm_address"))
        crm_number = normalize_text(_first(raw, ("crm_number", "numeroVoie", "numero"))) or _leading_number(crm_address)
        crm_postcode = normalize_text(_first(raw, ("crm_postcode", "postcode", "crm_cp")))
        crm_insee = normalize_text(_first(raw, ("crm_insee", "insee")))
        candidate_insee = normalize_text(_first(raw, ("candidate_insee", "codeCommuneEtablissement")))
        candidate_postcode = normalize_text(_first(raw, ("candidate_postcode", "codePostalEtablissement")))
        candidate_number = normalize_text(_first(raw, ("candidate_number", "numeroVoieEtablissement")))
        if not candidate_number and addresses:
            candidate_number = _leading_number(addresses[0])
        channels = _raw_channels(raw, config.channels)
        signals = _signals(raw.get("signals"))
        families = {
            family for family, members in config.channel_families.items() if channels.intersection(members)
        }
        exact_name = bool(channels & EXACT_NAME_CHANNELS)
        geo_match = bool(channels & EXACT_GEO_CHANNELS) or (
            bool(crm_insee) and candidate_insee == crm_insee
        ) or (bool(crm_postcode) and candidate_postcode == crm_postcode)
        state = normalize_text(_first(raw, ("candidate_state", "state", "etatAdministratifEtablissement")))
        human = _human_label(raw) if labels_allowed else ""
        operational = _operational_labels(raw, human) if labels_allowed else ()
        evidence_sources = {
            normalize_text(value) for value in _values(raw.get("official_evidence_sources")) if normalize_text(value)
        }
        evidence_sources.update(
            normalize_text(value)
            for value in _values(raw.get("sources"))
            if normalize_text(value)
        )
        evidence_sources.update(
            normalize_text(signal.get("source"))
            for signal in signals
            if normalize_text(signal.get("source"))
        )
        source_families = _source_families(evidence_sources)
        temporal = _temporal_features(
            raw.get("temporal_valid_from"),
            raw.get("temporal_valid_to"),
            reference_date,
        )
        relation = _relation_features(signals, reference_date)
        official_catalog_raw = raw.get("official_catalog_present")
        try:
            official_catalog_known = official_catalog_raw is not None and not bool(
                pd.isna(official_catalog_raw)
            )
        except (TypeError, ValueError):
            official_catalog_known = False
        parent_names = tuple(_values(raw.get("parent_feature_names")))
        parent_values = raw.get("parent_feature_values")
        if isinstance(parent_values, np.ndarray):
            parent_values = parent_values.tolist()
        if parent_names or parent_values not in (None, [], ()):
            if parent_names != tuple(CASCADE_PARENT_FEATURES) or not isinstance(
                parent_values, (list, tuple)
            ) or len(parent_values) != len(CASCADE_PARENT_FEATURES):
                raise ValueError("candidate row parent feature schema/order drift")
            sealed_parent_features = {
                name: _float(value, math.nan)
                for name, value in zip(parent_names, parent_values)
            }
        else:
            # Unit fixtures and pre-bridge diagnostics remain readable, but
            # gate entry points later require the sealed marker to be present.
            sealed_parent_features = {
                name: 0.0 for name in CASCADE_PARENT_FEATURES
            }
        record: dict[str, Any] = {
            "query_id": str(raw["query_id"]),
            "fold": int(raw["fold"]),
            "candidate_siret": candidate,
            "candidate_parent": parent,
            "human_siret": human,
            "operational_sirets_json": json.dumps(operational),
            # Keep ``union_rank`` as the complete-ledger rank for backwards
            # compatibility with feature names and fit-only sampling.
            "union_rank": master_rank,
            "master_rank": master_rank,
            # The upstream hydration rank is only a label-blind heuristic
            # feature.  Learned PreRank@500 is assigned after the site
            # LambdaMART scores the complete Master<=2,000 population.
            "heuristic_hydration_rank": heuristic_hydration_rank,
            "crm_name_normalized": crm_name,
            "crm_address_normalized": crm_address,
            "candidate_names_normalized_json": json.dumps(names),
            "candidate_addresses_normalized_json": json.dumps(addresses),
            "channels_json": json.dumps(sorted(channels)),
            "official_evidence_ids_json": json.dumps(sorted(_values(raw.get("official_evidence_ids")))),
            "official_evidence_sources_json": json.dumps(sorted(_values(raw.get("official_evidence_sources")))),
            "name_token_jaccard": _maximum(crm_name, names, _token_jaccard),
            "name_char_dice": _maximum(crm_name, names, _char_dice),
            "address_token_jaccard": _maximum(crm_address, addresses, _token_jaccard),
            "address_char_dice": _maximum(crm_address, addresses, _char_dice),
            "street_number_match": int(bool(crm_number) and crm_number == candidate_number),
            "postcode_match": int(bool(crm_postcode) and crm_postcode == candidate_postcode),
            "insee_match": int(bool(crm_insee) and crm_insee == candidate_insee),
            "exact_name_present": int(exact_name),
            "exact_geo_present": int(geo_match),
            "joint_name_geo": int(exact_name and geo_match),
            "site_name_evidence": int(bool(channels & NAME_EVIDENCE_CHANNELS)),
            "site_address_evidence": int(bool(channels & ADDRESS_EVIDENCE_CHANNELS)),
            "evidence_family_count": len(families),
            "independent_consensus": int(len(families) >= config.consensus_min_families),
            "official_source_count_log": math.log1p(len(source_families)),
            "site_source_registry_present": int("SIRENE" in source_families),
            "site_source_rne_present": int("RNE" in source_families),
            "site_source_bodacc_present": int("BODACC" in source_families),
            "site_source_sector_present": int("SECTOR" in source_families),
            "current_signal_count_log": math.log1p(len(channels & CURRENT_CHANNELS)),
            "historical_signal_count_log": math.log1p(len(channels & HISTORICAL_CHANNELS)),
            "relation_signal_count_log": math.log1p(len(channels & RELATION_CHANNELS)),
            "candidate_active": int(state == "A"),
            "candidate_closed": int(state == "F"),
            "candidate_state_unknown": int(state not in {"A", "F"}),
            "candidate_is_siege": int(
                _as_bool(
                    _first(
                        raw,
                        (
                            "candidate_is_headquarters",
                            "candidate_is_siege",
                            "is_siege",
                            "etablissementSiege",
                        ),
                    )
                )
            ),
            "retrieval_latency_ms": _float(
                raw.get("retrieval_latency_ms"), math.nan
            ),
            "segment_state": normalize_text(raw.get("ground_truth_state")),
            "segment_origin": normalize_text(_first(raw, ("data_origin", "source_kind"))),
            "segment_seen_siren": normalize_text(
                _first(raw, ("seen_unseen_siren", "siren_novelty"))
            ),
            "segment_new_site": normalize_text(
                _first(raw, ("known_new_site", "site_novelty"))
            ),
            "segment_megapolis": normalize_text(
                _first(raw, ("megapolis", "is_megapolis"))
            ),
            "segment_qualification": normalize_text(
                _first(raw, ("qualification", "qualification_status"))
            ),
            "segment_label_kind": normalize_text(raw.get("label_kind")),
            "segment_qualification_v2": normalize_text(raw.get("qualification_v2")),
            "segment_qualification_v3": normalize_text(raw.get("qualification_v3")),
            "official_catalog_present": (
                _as_bool(official_catalog_raw)
                if official_catalog_known
                else None
            ),
            "sealed_parent_features_present": bool(parent_names),
            "sealed_parent_rank": _float(raw.get("parent_rank"), math.nan),
            "sealed_parent_score": _float(raw.get("parent_score"), math.nan),
            **temporal,
            **relation,
            **sealed_parent_features,
        }
        for family, members in config.channel_families.items():
            record[f"family__{family}__present"] = int(bool(channels.intersection(members)))
        for channel in config.channels:
            channel_signals = [signal for signal in signals if str(signal.get("channel")) == channel]
            selected_signal = (
                min(
                    channel_signals,
                    key=lambda signal: (
                        int(_float(signal.get("rank"), 2_147_483_647)),
                        -_float(signal.get("score"), 0.0),
                    ),
                )
                if channel_signals
                else None
            )
            rank = _float(
                raw.get(f"{channel}_rank"),
                _float(selected_signal.get("rank"), 0.0) if selected_signal else 0.0,
            )
            score = _float(
                raw.get(f"{channel}_score"),
                _float(selected_signal.get("score"), 0.0) if selected_signal else 0.0,
            )
            present = int(rank >= 1)
            record[f"channel__{channel}__present"] = present
            record[f"__raw_rank__{channel}"] = rank if present else np.nan
            record[f"__raw_score__{channel}"] = score if present else np.nan
        records.append(record)

    output = pd.DataFrame(records)
    if output["query_id"].eq("").any() or output.duplicated(["query_id", "candidate_siret"]).any():
        raise ValueError("v12 union requires unique non-empty query/SIRET pairs")
    if output.duplicated(["query_id", "master_rank"]).any():
        raise ValueError("v12 master ranks must be unique within query")
    hydrated = output[output["heuristic_hydration_rank"].notna()]
    if hydrated.duplicated(["query_id", "heuristic_hydration_rank"]).any():
        raise ValueError("v12 hydration ranks must be unique within query")
    if (
        not hydrated.empty
        and int(hydrated["heuristic_hydration_rank"].max()) > config.prerank_cap
    ):
        raise ValueError("v12 hydration shortlist exceeds PreRank@500")
    counts = output.groupby("query_id").size()
    if int(counts.max()) > config.internal_cap:
        raise ValueError("v12 union exceeds 2,000 candidates")
    query_columns = (
        "fold",
        "human_siret",
        "operational_sirets_json",
        "retrieval_latency_ms",
        "segment_state",
        "segment_origin",
        "segment_seen_siren",
        "segment_new_site",
        "segment_megapolis",
        "segment_qualification",
        "segment_label_kind",
        "segment_qualification_v2",
        "segment_qualification_v3",
        "official_catalog_present",
    )
    for column in query_columns:
        if output.groupby("query_id")[column].nunique(dropna=False).gt(1).any():
            raise ValueError(f"query-level field varies within query: {column}")
    return output


def _attach_complete_union_features(
    candidates: pd.DataFrame, config: AdmissionConfigV12
) -> pd.DataFrame:
    output = candidates.copy()
    query_sizes = output.groupby("query_id")["candidate_siret"].transform("size")
    output["union_rank_reciprocal"] = 1.0 / output["union_rank"]
    output["union_rank_log_inverse"] = 1.0 / np.log2(output["union_rank"] + 1.0)
    output["union_rank_percentile_inverse"] = 1.0 - ((output["union_rank"] - 1.0) / query_sizes.clip(lower=1))
    heuristic_present = output["heuristic_hydration_rank"].notna()
    output["heuristic_hydration_rank_reciprocal"] = 0.0
    output["heuristic_hydration_rank_log_inverse"] = 0.0
    output["heuristic_hydration_rank_percentile_inverse"] = 0.0
    if heuristic_present.any():
        heuristic_rank = output.loc[
            heuristic_present, "heuristic_hydration_rank"
        ].astype(float)
        output.loc[
            heuristic_present, "heuristic_hydration_rank_reciprocal"
        ] = 1.0 / heuristic_rank
        output.loc[
            heuristic_present, "heuristic_hydration_rank_log_inverse"
        ] = 1.0 / np.log2(
            heuristic_rank + 1.0
        )
        heuristic_sizes = output.loc[heuristic_present].groupby("query_id")[
            "candidate_siret"
        ].transform("size")
        output.loc[
            heuristic_present, "heuristic_hydration_rank_percentile_inverse"
        ] = 1.0 - (
            (heuristic_rank - 1.0) / heuristic_sizes.clip(lower=1)
        )

    for channel in config.channels:
        present_column = f"channel__{channel}__present"
        rank_column = f"__raw_rank__{channel}"
        score_column = f"__raw_score__{channel}"
        present = output[present_column].eq(1)
        rank = output[rank_column]
        output[f"channel__{channel}__rank_reciprocal"] = np.where(present, 1.0 / rank, 0.0)
        output[f"channel__{channel}__rank_log_inverse"] = np.where(present, 1.0 / np.log2(rank + 1.0), 0.0)
        rank_pct = output.loc[present].groupby("query_id")[rank_column].rank(method="average", pct=True, ascending=True)
        output[f"channel__{channel}__rank_percentile_inverse"] = 0.0
        output.loc[present, f"channel__{channel}__rank_percentile_inverse"] = 1.0 - rank_pct + 1.0 / output.loc[present].groupby("query_id")[rank_column].transform("size")
        score_best = output.loc[present].groupby("query_id")[score_column].transform("max")
        output[f"channel__{channel}__score_relative"] = 0.0
        output.loc[present, f"channel__{channel}__score_relative"] = output.loc[present, score_column] - score_best
        score_pct = output.loc[present].groupby("query_id")[score_column].rank(method="average", pct=True, ascending=True)
        output[f"channel__{channel}__score_percentile"] = 0.0
        output.loc[present, f"channel__{channel}__score_percentile"] = score_pct

    output["name_address_joint"] = np.sqrt(
        output["name_char_dice"].clip(lower=0.0) * output["address_char_dice"].clip(lower=0.0)
    )
    output["within_parent_rank"] = output.groupby(["query_id", "candidate_parent"])["union_rank"].rank(method="first", ascending=True)
    output["within_parent_rank_reciprocal"] = 1.0 / output["within_parent_rank"]
    output["within_parent_candidate_count_log"] = np.log1p(
        output.groupby(["query_id", "candidate_parent"])["candidate_siret"].transform("size")
    )

    parent_columns = [
        "query_id",
        "candidate_parent",
        "sealed_parent_features_present",
        "sealed_parent_rank",
        "sealed_parent_score",
        *CASCADE_PARENT_FEATURES,
    ]
    for column in parent_columns[2:]:
        if output.groupby(["query_id", "candidate_parent"])[column].nunique(
            dropna=False
        ).gt(1).any():
            raise ValueError(f"sealed parent field varies within parent: {column}")
    parents = output[parent_columns].drop_duplicates(
        ["query_id", "candidate_parent"]
    ).copy()
    scored_parent = parents[
        parents["sealed_parent_rank"].notna()
        & parents["sealed_parent_score"].notna()
    ].copy()
    if (
        (scored_parent["sealed_parent_rank"] < 1).any()
        or scored_parent.duplicated(["query_id", "sealed_parent_rank"]).any()
    ):
        raise ValueError("sealed cascade parent score/rank is invalid")
    scored_parent["parent_rank"] = scored_parent["sealed_parent_rank"]
    scored_parent["_parent_model_score"] = scored_parent["sealed_parent_score"]
    scored_parent["parent_rank_reciprocal"] = 1.0 / scored_parent["parent_rank"]
    best_parent = scored_parent.groupby("query_id")["_parent_model_score"].transform("max")
    scored_parent["parent_score_gap_to_best"] = (
        scored_parent["_parent_model_score"] - best_parent
    )
    parent_exp = np.exp(
        (scored_parent["_parent_model_score"] - best_parent).clip(
            lower=-30.0, upper=0.0
        )
    )
    scored_parent["parent_score_probability"] = parent_exp / parent_exp.groupby(
        scored_parent["query_id"]
    ).transform("sum")
    output = output.merge(
        scored_parent[
            [
                "query_id",
                "candidate_parent",
                "_parent_model_score",
                "parent_rank",
                *PARENT_SCORE_FEATURES,
            ]
        ],
        on=["query_id", "candidate_parent"],
        how="left",
        validate="many_to_one",
    )
    output["parent_preexpansion_missing"] = output["_parent_model_score"].isna().astype(np.int8)
    for column in (
        "_parent_model_score",
        "parent_rank",
        *PARENT_SCORE_FEATURES,
    ):
        output[column] = output[column].fillna(0.0)

    output["features_computed_on_complete_union"] = True
    return output.sort_values(["query_id", "union_rank", "candidate_siret"], kind="stable").reset_index(drop=True)


def canonicalize_union(
    frame: pd.DataFrame,
    config: AdmissionConfigV12,
    *,
    labels_allowed: bool,
    require_labels: bool = False,
) -> pd.DataFrame:
    """Compute all rank, score and parent features on complete query unions."""
    output = _attach_complete_union_features(
        _raw_candidate_records(frame, config, labels_allowed=labels_allowed), config
    )
    if require_labels and output.groupby("query_id")["human_siret"].first().eq("").any():
        raise ValueError("every v12 training/evaluation query needs a human SIRET")
    return output


def channel_feature_names(config: AdmissionConfigV12) -> tuple[str, ...]:
    suffixes = (
        "present",
        "rank_reciprocal",
        "rank_log_inverse",
        "rank_percentile_inverse",
        "score_relative",
        "score_percentile",
    )
    return tuple(f"channel__{channel}__{suffix}" for channel in config.channels for suffix in suffixes)


BASE_SITE_FEATURES = (
    "union_rank_reciprocal",
    "union_rank_log_inverse",
    "union_rank_percentile_inverse",
    "heuristic_hydration_rank_reciprocal",
    "heuristic_hydration_rank_log_inverse",
    "heuristic_hydration_rank_percentile_inverse",
    "name_token_jaccard",
    "name_char_dice",
    "address_token_jaccard",
    "address_char_dice",
    "name_address_joint",
    "street_number_match",
    "postcode_match",
    "insee_match",
    "exact_name_present",
    "exact_geo_present",
    "joint_name_geo",
    "evidence_family_count",
    "independent_consensus",
    "official_source_count_log",
    "site_source_registry_present",
    "site_source_rne_present",
    "site_source_bodacc_present",
    "site_source_sector_present",
    "current_signal_count_log",
    "historical_signal_count_log",
    "relation_signal_count_log",
    "candidate_active",
    "candidate_closed",
    "candidate_state_unknown",
    "candidate_is_siege",
    "temporal_valid_from_present",
    "temporal_valid_to_present",
    "temporal_open_ended",
    "temporal_valid_at_reference",
    "temporal_starts_after_reference",
    "temporal_ended_before_reference",
    "temporal_start_distance_days_log1p",
    "temporal_end_distance_days_log1p",
    "relation_successor_present",
    "relation_predecessor_present",
    "relation_both_present",
    "relation_direction_other_present",
    "relation_dated_present",
    "relation_date_missing",
    "relation_nearest_age_days_log1p",
    "within_parent_rank_reciprocal",
    "within_parent_candidate_count_log",
    "parent_preexpansion_missing",
)

BASE_PARENT_FEATURES = (
    *CASCADE_PARENT_FEATURES,
)

PARENT_SCORE_FEATURES = (
    "parent_rank_reciprocal",
    "parent_score_probability",
    "parent_score_gap_to_best",
)


def parent_feature_names(config: AdmissionConfigV12) -> tuple[str, ...]:
    del config
    return tuple(CASCADE_PARENT_FEATURES)


def site_feature_names(config: AdmissionConfigV12) -> tuple[str, ...]:
    family = tuple(f"family__{name}__present" for name in config.channel_families)
    return (
        *BASE_SITE_FEATURES,
        *channel_feature_names(config),
        *family,
        *parent_feature_names(config),
        *PARENT_SCORE_FEATURES,
    )


def assert_identifier_free_feature_values(frame: pd.DataFrame, features: Sequence[str]) -> None:
    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise ValueError(f"missing v12 model features: {missing}")
    non_numeric = [feature for feature in features if not pd.api.types.is_numeric_dtype(frame[feature])]
    if non_numeric:
        raise ValueError(f"v12 model features must be numeric: {non_numeric}")


def diagnostic_parent_rows_from_hydrated_union(
    candidates: pd.DataFrame, config: AdmissionConfigV12
) -> pd.DataFrame:
    """Aggregate hydrated rows for diagnostics, never for parent fitting.

    The learned parent scorer must consume the cascade's pre-expansion parent
    rows.  These aggregates intentionally carry no pre-expansion provenance,
    so every parent fit/scoring entry point rejects them.
    """
    records: list[dict[str, Any]] = []
    for (query_id, parent), group in candidates.groupby(["query_id", "candidate_parent"], sort=True):
        channels: set[str] = set()
        sources: set[str] = set()
        for value in group["channels_json"]:
            channels.update(_values(value))
        for value in group["official_evidence_sources_json"]:
            sources.update(_values(value))
        best_ranks = {
            channel: float(group[f"__raw_rank__{channel}"].min(skipna=True))
            for channel in config.channels
            if group[f"__raw_rank__{channel}"].notna().any()
        }
        rrf_constant = float(config._admission.get("parent_rrf_constant", 60.0))
        exact_name = bool(channels & EXACT_NAME_CHANNELS)
        name_evidence = bool(channels & NAME_EVIDENCE_CHANNELS)
        exact_address = bool(channels & EXACT_GEO_CHANNELS)
        address_evidence = bool(channels & ADDRESS_EVIDENCE_CHANNELS)
        rescue_channel = "contrastive_siren_rescue"
        rescue = rescue_channel in channels
        rescue_scores = group[f"__raw_score__{rescue_channel}"] if rescue_channel in config.channels else pd.Series(dtype=float)
        record: dict[str, Any] = {
            "query_id": str(query_id),
            "fold": int(group.iloc[0]["fold"]),
            "candidate_parent": str(parent),
            "human_siret": str(group.iloc[0].get("human_siret") or ""),
            "exact_name_and_address": float(exact_name and address_evidence),
            "name_and_exact_address": float(name_evidence and exact_address),
            "exact_name_evidence": float(exact_name),
            "name_evidence": float(name_evidence),
            "exact_address_evidence": float(exact_address),
            "address_evidence": float(address_evidence),
            "geographic_support": float(bool(group[["insee_match", "postcode_match"]].max(axis=1).max())),
            "joint_site_log1p": float(math.log1p(int((group["site_name_evidence"].eq(1) & group["site_address_evidence"].eq(1)).sum()))),
            "official_source_count": float(len(sources)),
            "channel_count": float(len(best_ranks)),
            "support_site_log1p": float(math.log1p(len(group))),
            "channel_rrf": float(sum(1.0 / (rrf_constant + max(1.0, rank)) for rank in best_ranks.values())),
            "rescue_evidence": float(rescue),
            "rescue_score": float(rescue_scores.max(skipna=True)) if rescue and rescue_scores.notna().any() else 0.0,
        }
        records.append(record)
    return pd.DataFrame(records).sort_values(["query_id", "candidate_parent"], kind="stable").reset_index(drop=True)


def canonicalize_cascade_parent_rows(parent_rows: pd.DataFrame) -> pd.DataFrame:
    """Validate label-blind, pre-expansion cascade parent feature rows."""
    required_parent = {"query_id", "siren", "feature_names", "feature_values"}
    missing_parent = sorted(required_parent - set(parent_rows.columns))
    if missing_parent:
        raise ValueError(f"cascade parent rows missing columns: {missing_parent}")
    records: list[dict[str, Any]] = []
    for row in parent_rows.to_dict("records"):
        names = tuple(_values(row.get("feature_names")))
        values = row.get("feature_values")
        if isinstance(values, np.ndarray):
            values = values.tolist()
        if not isinstance(values, (list, tuple)):
            raise ValueError("cascade feature_values must be an ordered list")
        if names != tuple(CASCADE_PARENT_FEATURES) or len(values) != len(CASCADE_PARENT_FEATURES):
            raise ValueError("cascade/admission parent feature schema drift")
        record = {
            "query_id": str(row["query_id"]),
            "candidate_parent": str(row["siren"]),
            "parent_feature_origin": "CASCADE_PRE_EXPANSION",
            "sealed_parent_rank": _float(row.get("parent_rank"), math.nan),
            "sealed_parent_score": _float(row.get("parent_score"), math.nan),
        }
        record.update({name: float(value) for name, value in zip(names, values)})
        records.append(record)
    adapted = pd.DataFrame(records)
    if adapted.duplicated(["query_id", "candidate_parent"]).any():
        raise ValueError("cascade parent rows must be unique by query and parent")
    return adapted.sort_values(["query_id", "candidate_parent"], kind="stable").reset_index(drop=True)


def attach_sealed_parent_scores(
    candidates: pd.DataFrame,
    config: AdmissionConfigV12,
) -> pd.DataFrame:
    """Validate scores/features transported on every master candidate row.

    Pass-B parent pools have themselves been bounded by the nested parent
    models.  Re-fitting a parent model on those pools would create an indirect
    OOF leak.  Admission is therefore site-only and never loads the global
    parent-candidate parquet on its hot path.
    """
    required = {
        "sealed_parent_features_present",
        "sealed_parent_rank",
        "sealed_parent_score",
        *parent_feature_names(config),
        *PARENT_SCORE_FEATURES,
    }
    missing = sorted(required - set(candidates.columns))
    if missing or not candidates["sealed_parent_features_present"].all():
        raise ValueError("master rows lack sealed pre-expansion parent features")
    if candidates["parent_preexpansion_missing"].any():
        raise ValueError("master rows lack a complete sealed parent score")
    return candidates.copy()


def adapt_cascade_parent_rows(
    parent_rows: pd.DataFrame, query_labels: pd.DataFrame
) -> pd.DataFrame:
    """Attach sealed label-side metadata to pre-expansion parent rows."""
    required_labels = {"query_id", "fold", "human_siret"}
    missing_labels = sorted(required_labels - set(query_labels.columns))
    if missing_labels:
        raise ValueError(f"cascade query labels missing columns: {missing_labels}")
    if query_labels["query_id"].astype(str).duplicated().any():
        raise ValueError("cascade query labels must be unique")
    adapted = canonicalize_cascade_parent_rows(parent_rows)
    labels = query_labels[["query_id", "fold", "human_siret"]].copy()
    labels["query_id"] = labels["query_id"].astype(str)
    output = adapted.merge(labels, on="query_id", how="left", validate="many_to_one")
    if output[["fold", "human_siret"]].isna().any().any():
        raise ValueError("cascade parent row has no sealed query label")
    return output.sort_values(["query_id", "candidate_parent"], kind="stable").reset_index(drop=True)


def sample_precomputed_candidates(
    complete_features: pd.DataFrame, config: AdmissionConfigV12
) -> pd.DataFrame:
    """Sample fit rows without recomputing or renormalising any feature."""
    if "features_computed_on_complete_union" not in complete_features or not complete_features["features_computed_on_complete_union"].all():
        raise ValueError("v12 sampling requires features computed on the complete union")
    sampling = config.fit_candidate_sampling
    cap = int(sampling["maximum_candidates_per_query"])
    union_head = int(sampling["union_head"])
    channel_head = int(sampling["channel_head"])
    pieces: list[pd.DataFrame] = []
    for _query_id, group in complete_features.groupby("query_id", sort=True):
        if len(group) <= cap:
            pieces.append(group.copy())
            continue
        eligible = group[group["heuristic_hydration_rank"].notna()]
        selected: set[int] = set(
            eligible.nsmallest(union_head, "heuristic_hydration_rank").index
        )
        for channel in config.channels:
            column = f"__raw_rank__{channel}"
            selected.update(group[group[column].notna()].nsmallest(channel_head, column).index)
        human = str(group.iloc[0].get("human_siret") or "")
        positives = (
            set(group.index[group["candidate_siret"].eq(human)])
            if human
            else set()
        )
        selected.update(positives)
        if len(selected) < cap:
            remainder = group.drop(index=list(selected), errors="ignore")
            remainder = remainder.assign(
                __eligible_priority=remainder["heuristic_hydration_rank"].notna(),
                __prerank_priority=remainder["heuristic_hydration_rank"].fillna(
                    config.prerank_cap + remainder["union_rank"]
                ),
            ).sort_values(
                ["__eligible_priority", "__prerank_priority", "union_rank"],
                ascending=[False, True, True],
                kind="stable",
            )
            selected.update(remainder.head(cap - len(selected)).index)
        selected_frame = group.loc[list(selected)].copy()
        selected_frame["__positive_priority"] = selected_frame.index.isin(positives)
        selected_frame["__eligible_priority"] = selected_frame[
            "heuristic_hydration_rank"
        ].notna()
        selected_frame["__prerank_priority"] = selected_frame[
            "heuristic_hydration_rank"
        ].fillna(config.prerank_cap + selected_frame["union_rank"])
        selected_frame["__union_priority"] = selected_frame["union_rank"]
        selected_frame = selected_frame.sort_values(
            [
                "__positive_priority",
                "__eligible_priority",
                "__prerank_priority",
                "__union_priority",
                "candidate_siret",
            ],
            ascending=[False, False, True, True, True],
            kind="stable",
        ).head(cap).drop(
            columns=[
                "__positive_priority",
                "__eligible_priority",
                "__prerank_priority",
                "__union_priority",
            ]
        )
        pieces.append(selected_frame.sort_values(["union_rank", "candidate_siret"], kind="stable"))
    output = pd.concat(pieces, ignore_index=True)
    if not output["features_computed_on_complete_union"].all():
        raise AssertionError("v12 sampling changed feature provenance")
    return output


def _ranker_matrix(rows: pd.DataFrame, features: Sequence[str], label: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    assert_identifier_free_feature_values(rows, features)
    ordered = rows.sort_values(["query_id", label], ascending=[True, False], kind="stable")
    ids = ordered["query_id"].astype(str).to_numpy()
    if not len(ids):
        raise ValueError("no trainable v12 ranking rows")
    starts = np.empty(len(ids), dtype=bool)
    starts[0] = True
    starts[1:] = ids[1:] != ids[:-1]
    qid = np.cumsum(starts, dtype=np.int64) - 1
    return (
        ordered[list(features)].fillna(0.0).astype(np.float32),
        ordered[label].to_numpy(dtype=np.int8),
        qid,
    )


def _fit_ranker(rows: pd.DataFrame, features: Sequence[str], label: str, params: Mapping[str, Any]) -> xgb.XGBRanker:
    matrix, target, qid = _ranker_matrix(rows, features, label)
    model = xgb.XGBRanker(**dict(params))
    model.fit(matrix, target, qid=qid, verbose=False)
    return model


def _parent_training_rows(candidates: pd.DataFrame, config: AdmissionConfigV12) -> tuple[pd.DataFrame, Mapping[str, int]]:
    if "parent_feature_origin" not in candidates or not candidates["parent_feature_origin"].eq("CASCADE_PRE_EXPANSION").all():
        raise ValueError("parent LambdaMART may fit only pre-expansion cascade parent rows")
    parent_columns = ["query_id", "fold", "candidate_parent", "human_siret", *parent_feature_names(config)]
    parents = candidates[parent_columns].drop_duplicates(["query_id", "candidate_parent"])
    pieces: list[pd.DataFrame] = []
    missing = no_negative = 0
    for _query_id, group in parents.groupby("query_id", sort=True):
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
        pieces.append(retained)
    if not pieces:
        raise ValueError("no natural parent positives for v12")
    return pd.concat(pieces, ignore_index=True), {"missing_natural_parent": missing, "without_parent_negative": no_negative}


def fit_parent_ranker(candidates: pd.DataFrame, config: AdmissionConfigV12) -> tuple[xgb.XGBRanker, Mapping[str, int]]:
    rows, diagnostics = _parent_training_rows(candidates, config)
    return _fit_ranker(rows, parent_feature_names(config), "parent_label", config.xgb_params("parent")), diagnostics


def fit_parent_ranker_from_cascade_rows(
    cascade_parent_rows: pd.DataFrame, config: AdmissionConfigV12
) -> tuple[xgb.XGBRanker, Mapping[str, int]]:
    return fit_parent_ranker(cascade_parent_rows, config)


def _site_training_rows(
    candidates: pd.DataFrame, config: AdmissionConfigV12
) -> tuple[pd.DataFrame, Mapping[str, int]]:
    pieces: list[pd.DataFrame] = []
    missing = neutral = no_negative = 0
    for _query_id, raw_group in candidates.groupby("query_id", sort=True):
        # The site model learns Master<=2,000 -> PreRank<=500 directly.  The
        # old upstream hydration shortlist is a feature, never an eligibility
        # filter for fit or validation.
        group = raw_group.copy()
        if group.empty:
            missing += 1
            continue
        human = normalize_siret(group.iloc[0]["human_siret"])
        positive = group["candidate_siret"].eq(human)
        if int(positive.sum()) != 1:
            missing += 1
            continue
        operational = set(_operational_labels({"acceptable_sirets_operational_json": group.iloc[0]["operational_sirets_json"]}, human))
        equivalent = group["candidate_siret"].isin(operational - {human})
        neutral += int(equivalent.sum())
        retained = group[~equivalent].copy()
        if len(retained) < 2:
            no_negative += 1
            continue
        retained["site_label"] = retained["candidate_siret"].eq(human).astype(np.int8)
        pieces.append(retained)
    if not pieces:
        raise ValueError("no natural site positives for v12")
    return pd.concat(pieces, ignore_index=True), {
        "missing_natural_site": missing,
        "operational_neutrals_excluded": neutral,
        "without_site_negative": no_negative,
    }


def fit_site_ranker(candidates: pd.DataFrame, config: AdmissionConfigV12) -> tuple[xgb.XGBRanker, Mapping[str, int]]:
    rows, diagnostics = _site_training_rows(candidates, config)
    return _fit_ranker(rows, site_feature_names(config), "site_label", config.xgb_params("site")), diagnostics


@dataclass
class HierarchicalModelsV12:
    # Parent scoring happens inside the sealed cascade.  The optional field is
    # retained only for loading a production bundle alongside the site model;
    # admission OOF never fits or reapplies it on pass-B pools.
    parent: xgb.XGBRanker | None
    site: xgb.XGBRanker
    diagnostics: Mapping[str, Any]


def fit_hierarchical_models(
    complete_candidates: pd.DataFrame,
    config: AdmissionConfigV12,
) -> HierarchicalModelsV12:
    """Fit the frozen production recipe on folds 2/3/4 only."""
    assert_ground_truth_parent_disjoint(complete_candidates, config.train_folds)
    sampled = sample_precomputed_candidates(complete_candidates, config)
    sealed_parent = attach_sealed_parent_scores(sampled, config)
    site_model, site_diag = fit_site_ranker(sealed_parent, config)
    return HierarchicalModelsV12(
        None,
        site_model,
        {
            "fit_folds": list(config.train_folds),
            "site": dict(site_diag),
            "parent_scores_for_site_fit": "SEALED_CASCADE_OOF_ONLY",
            "parent_refit_on_pass_b": False,
            "features_computed_before_sampling": True,
        },
    )


def attach_site_scores(model: xgb.XGBRanker, candidates: pd.DataFrame, config: AdmissionConfigV12) -> pd.DataFrame:
    output = candidates.copy()
    output["site_score"] = model.predict(
        output[list(site_feature_names(config))].fillna(0.0).astype(np.float32)
    ).astype(np.float64)
    output["final_score"] = output["site_score"]
    global_order = output.sort_values(
        ["query_id", "site_score", "candidate_siret"],
        ascending=[True, False, True],
        kind="stable",
    )
    output.loc[global_order.index, "ranked_rank"] = (
        global_order.groupby("query_id", sort=False).cumcount() + 1
    ).to_numpy(dtype=np.int32)
    output["prerank_rank"] = output["ranked_rank"].where(
        output["ranked_rank"].le(config.prerank_cap)
    )
    site_order = output.sort_values(
        ["query_id", "candidate_parent", "site_score", "candidate_siret"],
        ascending=[True, True, False, True],
        kind="stable",
    )
    output.loc[site_order.index, "site_rank"] = (
        site_order.groupby(["query_id", "candidate_parent"], sort=False).cumcount()
        + 1
    ).to_numpy(dtype=np.int32)
    return output


def score_canonical_candidates(
    models: HierarchicalModelsV12,
    candidates: pd.DataFrame,
    config: AdmissionConfigV12,
) -> pd.DataFrame:
    parent_scored = attach_sealed_parent_scores(candidates, config)
    return attach_site_scores(models.site, parent_scored, config)


def select_top100(scored: pd.DataFrame, config: AdmissionConfigV12) -> pd.DataFrame:
    """Adaptive marginal allocation with joint name+geography protection.

    There is deliberately no fixed per-parent cap and no 20/20 lane quota.
    Repeated selections from one parent receive a smooth penalty, while a high
    cross-fitted parent posterior continues to earn additional sites.
    """
    pieces: list[pd.DataFrame] = []
    parent_weight = config.selection["parent_probability_weight"]
    repeat_penalty = config.selection["repeat_parent_penalty"]
    for _query_id, raw_group in scored.groupby("query_id", sort=True):
        # Top100 is allocated only inside the learned PreRank500.  The same
        # site LambdaMART score determines both stages; there is no hidden
        # third ranker or heuristic hydration eligibility gate.
        group = raw_group[raw_group["prerank_rank"].notna()].copy()
        if group.empty:
            continue
        # ``ranked_rank`` was assigned from (score desc, SIRET asc), so this
        # utility is invariant to the incoming Master row order, including
        # XGBoost score ties.
        group["_site_percentile"] = 1.0 - (
            (group["ranked_rank"].astype(float) - 1.0) / max(1, len(group))
        )
        group["_base_utility"] = group["_site_percentile"] + parent_weight * group["parent_score_probability"].fillna(0.0)
        chosen: list[int] = []
        reasons: dict[int, str] = {}
        marginal: dict[int, float] = {}
        parent_counts: dict[str, int] = {}

        protected = group[group["joint_name_geo"].eq(1)].sort_values(
            ["_base_utility", "site_score", "candidate_siret"],
            ascending=[False, False, True], kind="stable",
        ).head(config.final_cap)
        for index, row in protected.iterrows():
            chosen.append(index)
            parent = str(row["candidate_parent"])
            parent_counts[parent] = parent_counts.get(parent, 0) + 1
            reasons[index] = "protected_joint_name_geo"
            marginal[index] = float(row["_base_utility"])

        remaining = set(group.index) - set(chosen)
        while len(chosen) < config.final_cap and remaining:
            best_index: int | None = None
            best_key: tuple[float, float, str] | None = None
            best_marginal = -math.inf
            for index in remaining:
                row = group.loc[index]
                parent = str(row["candidate_parent"])
                count = parent_counts.get(parent, 0)
                value = float(row["_base_utility"]) - repeat_penalty * math.log1p(count)
                key = (value, float(row["site_score"]), -int(str(row["candidate_siret"])))
                if best_key is None or key > best_key:
                    best_key = key
                    best_index = int(index)
                    best_marginal = value
            assert best_index is not None
            row = group.loc[best_index]
            parent = str(row["candidate_parent"])
            chosen.append(best_index)
            remaining.remove(best_index)
            parent_counts[parent] = parent_counts.get(parent, 0) + 1
            reasons[best_index] = "adaptive_marginal"
            marginal[best_index] = best_marginal

        selected = group.loc[chosen].copy()
        selected["selection_reason"] = [reasons[index] for index in chosen]
        selected["marginal_score"] = [marginal[index] for index in chosen]
        selected["selected_rank"] = np.arange(1, len(selected) + 1, dtype=np.int16)
        pieces.append(selected.drop(columns=["_site_percentile", "_base_utility"]))
    output = pd.concat(pieces, ignore_index=True)
    if int(output.groupby("query_id").size().max()) > config.final_cap:
        raise AssertionError("v12 emitted more than 100 candidates")
    return output


def predict_top100(
    models: HierarchicalModelsV12,
    raw_unlabelled_union: pd.DataFrame,
    config: AdmissionConfigV12,
) -> pd.DataFrame:
    reject_label_columns(raw_unlabelled_union)
    candidates = canonicalize_union(raw_unlabelled_union, config, labels_allowed=False)
    return select_top100(
        score_canonical_candidates(models, candidates, config), config
    )


def assert_ground_truth_parent_disjoint(candidates: pd.DataFrame, folds: Sequence[int]) -> None:
    labels = candidates.groupby("query_id", sort=False).agg(fold=("fold", "first"), human=("human_siret", "first"))
    labels["truth_parent"] = labels["human"].str[:9]
    overlap = labels.groupby("truth_parent")["fold"].nunique().gt(1)
    if overlap.any():
        raise ValueError(f"v12 requires ground-truth-parent-disjoint folds; overlaps={int(overlap.sum())}")
    if set(labels["fold"].astype(int)) != set(int(value) for value in folds):
        raise ValueError("v12 candidates do not contain the exact configured training folds")


def fit_outer_oof_models(
    complete_candidates: Mapping[int, pd.DataFrame],
    config: AdmissionConfigV12,
) -> tuple[Mapping[int, HierarchicalModelsV12], Mapping[str, Any]]:
    """Fit one model per outer context from split-specific candidate pools.

    A single consolidated cascade is intentionally rejected: although its
    parent scores could be recomputed OOF, its <=500 hydrated population was
    already chosen by one parent scorer.  Each mapping value must therefore be
    the *training* population generated inside the corresponding outer
    context (the two fit folds scored inner-OOF).
    """
    if not isinstance(complete_candidates, Mapping):
        raise ValueError(
            "outer OOF requires split-specific cascade mappings; a single "
            "cascade parquet is not gate-eligible"
        )
    expected_outer = set(config.train_folds)
    if set(int(key) for key in complete_candidates) != expected_outer:
        raise ValueError("outer OOF mappings must contain heldout folds 2/3/4")
    models: dict[int, HierarchicalModelsV12] = {}
    diagnostics: dict[str, Any] = {}
    for heldout in config.train_folds:
        model = fit_one_outer_site_model(
            complete_candidates[heldout], heldout=heldout, config=config
        )
        models[heldout] = model
        diagnostics[str(heldout)] = dict(model.diagnostics)
    return models, diagnostics


def fit_one_outer_site_model(
    complete_candidates: pd.DataFrame,
    *,
    heldout: int,
    config: AdmissionConfigV12,
    pre_sampled: bool = False,
) -> HierarchicalModelsV12:
    if heldout not in config.train_folds:
        raise ValueError("outer heldout fold must be one of 2/3/4")
    fit_folds = tuple(fold for fold in config.train_folds if fold != heldout)
    assert_ground_truth_parent_disjoint(complete_candidates, fit_folds)
    train = (
        complete_candidates.copy()
        if pre_sampled
        else sample_precomputed_candidates(complete_candidates, config)
    )
    if not train["features_computed_on_complete_union"].all():
        raise ValueError("outer site fit lacks complete-union feature provenance")
    sealed_parent = attach_sealed_parent_scores(train, config)
    site_model, site_diag = fit_site_ranker(sealed_parent, config)
    return HierarchicalModelsV12(
        None,
        site_model,
        {
            "fit_folds": list(fit_folds),
            "site": dict(site_diag),
            "parent_scores_for_site_fit": "SEALED_CASCADE_INNER_OOF_ONLY",
            "parent_refit_on_pass_b": False,
            "candidate_generation": "INNER_OOF_SPLIT_SPECIFIC",
            "outer_heldout_fold": heldout,
            "features_computed_before_sampling": True,
        },
    )


def _wilson(successes: int, total: int, confidence: float = 0.95) -> list[float] | None:
    if total <= 0:
        return None
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    spread = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return [max(0.0, center - spread), min(1.0, center + spread)]


def _metric(successes: int, total: int) -> Mapping[str, Any]:
    return {"successes": int(successes), "total": int(total), "rate": successes / total if total else None, "wilson_95": _wilson(successes, total)}


def evaluate_all_human(
    labelled_candidates: pd.DataFrame,
    *,
    scored: pd.DataFrame | None,
    selected: pd.DataFrame | None,
    config: AdmissionConfigV12,
) -> tuple[pd.DataFrame, Mapping[str, Any]]:
    scored_groups = {} if scored is None else {str(key): value for key, value in scored.groupby("query_id", sort=False)}
    selected_groups = {} if selected is None else {str(key): value for key, value in selected.groupby("query_id", sort=False)}
    records: list[dict[str, Any]] = []
    for query_id, group in labelled_candidates.groupby("query_id", sort=True):
        human = normalize_siret(group.iloc[0]["human_siret"])
        truth = group[group["candidate_siret"].eq(human)]
        parent_present = human[:9] in set(group["candidate_parent"].astype(str))
        official_catalog_raw = group.iloc[0]["official_catalog_present"]
        try:
            official_catalog_known = official_catalog_raw is not None and not bool(
                pd.isna(official_catalog_raw)
            )
        except (TypeError, ValueError):
            official_catalog_known = False
        official_catalog_present = (
            bool(official_catalog_raw) if official_catalog_known else None
        )
        truth_master_rank = (
            int(truth.iloc[0]["master_rank"]) if not truth.empty else None
        )
        truth_heuristic_hydration_rank = (
            int(truth.iloc[0]["heuristic_hydration_rank"])
            if not truth.empty
            and pd.notna(truth.iloc[0]["heuristic_hydration_rank"])
            else None
        )
        scored_group = scored_groups.get(str(query_id), group.iloc[0:0])
        selected_group = selected_groups.get(str(query_id), group.iloc[0:0])
        ranked_truth = scored_group[scored_group["candidate_siret"].eq(human)] if not scored_group.empty else scored_group
        ranked_rank = int(ranked_truth.iloc[0]["ranked_rank"]) if not ranked_truth.empty else None
        truth_prerank_rank = (
            int(ranked_truth.iloc[0]["prerank_rank"])
            if not ranked_truth.empty
            and pd.notna(ranked_truth.iloc[0]["prerank_rank"])
            else None
        )
        selected_hit = human in set(selected_group.get("candidate_siret", pd.Series(dtype=str)).astype(str))
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
        selected_ids = set(
            selected_group.get("candidate_siret", pd.Series(dtype=str)).astype(str)
        )
        selected_operational_hit = bool(selected_ids.intersection(operational))
        latency_source = selected_group if not selected_group.empty else scored_group
        admission_latency_ms = (
            _float(latency_source.iloc[0].get("admission_latency_ms"), math.nan)
            if not latency_source.empty
            else math.nan
        )
        retrieval_latency_ms = float(group.iloc[0]["retrieval_latency_ms"])
        end_to_end_latency_ms = (
            retrieval_latency_ms + admission_latency_ms
            if math.isfinite(admission_latency_ms)
            else math.nan
        )
        raw: dict[str, Any] = {
            "query_id": str(query_id),
            "fold": int(group.iloc[0]["fold"]),
            "human_siret": human,
            "truth_union_rank": truth_master_rank,
            "truth_master_rank": truth_master_rank,
            "truth_heuristic_hydration_rank": truth_heuristic_hydration_rank,
            "truth_prerank_rank": truth_prerank_rank,
            "truth_ranked_rank": ranked_rank,
            "truth_parent_present": bool(parent_present),
            "selected_hit_at_100": bool(selected_hit),
            "selected_operational_hit_at_100": selected_operational_hit,
            "retrieval_latency_ms": retrieval_latency_ms,
            "admission_latency_ms": admission_latency_ms,
            "end_to_end_latency_ms": end_to_end_latency_ms,
            "union_candidate_count": int(len(group)),
            "selected_candidate_count": int(len(selected_group)),
            "segment_state": str(group.iloc[0]["segment_state"]),
            "segment_origin": str(group.iloc[0]["segment_origin"]),
            "segment_seen_siren": str(group.iloc[0]["segment_seen_siren"]),
            "segment_new_site": str(group.iloc[0]["segment_new_site"]),
            "segment_megapolis": str(group.iloc[0]["segment_megapolis"]),
            "segment_qualification": str(
                group.iloc[0]["segment_qualification"]
            ),
            "segment_label_kind": str(group.iloc[0]["segment_label_kind"]),
            "segment_qualification_v2": str(
                group.iloc[0]["segment_qualification_v2"]
            ),
            "segment_qualification_v3": str(
                group.iloc[0]["segment_qualification_v3"]
            ),
            "official_catalog_present": official_catalog_present,
        }
        for cutoff in METRIC_CUTOFFS:
            raw[f"union_hit_at_{cutoff}"] = bool(
                truth_master_rank is not None and truth_master_rank <= cutoff
            )
            raw[f"prerank_hit_at_{cutoff}"] = bool(
                truth_prerank_rank is not None and truth_prerank_rank <= cutoff
            )
            raw[f"heuristic_hydration_hit_at_{cutoff}"] = bool(
                truth_heuristic_hydration_rank is not None
                and truth_heuristic_hydration_rank <= cutoff
            )
            raw[f"ranked_hit_at_{cutoff}"] = bool(ranked_rank is not None and ranked_rank <= cutoff)
        if truth_master_rank is None and official_catalog_present is False:
            raw["loss_reason"] = "OFFICIAL_CATALOG_ABSENT"
        elif truth_master_rank is None and official_catalog_present is None:
            raw["loss_reason"] = "OFFICIAL_CATALOG_STATUS_UNKNOWN"
        elif truth_master_rank is None and not parent_present:
            raw["loss_reason"] = "PARENT_NOT_RETRIEVED"
        elif truth_master_rank is None:
            raw["loss_reason"] = "SITE_NOT_EXPANDED"
        elif truth_prerank_rank is None:
            raw["loss_reason"] = "EVICTED_MASTER_TO_500"
        elif selected_hit:
            raw["loss_reason"] = "HIT"
        else:
            raw["loss_reason"] = "EVICTED_500_TO_100"
        raw["rescued_by_adaptive_allocation"] = bool(
            selected_hit and ranked_rank is not None and ranked_rank > config.final_cap
        )
        records.append(raw)
    outcomes = pd.DataFrame(records)
    return outcomes, summarize_outcomes(outcomes, config)


def missing_label_outcomes(
    query_labels: pd.DataFrame,
    seen_query_ids: Iterable[str],
    config: AdmissionConfigV12,
) -> pd.DataFrame:
    """Create denominator rows for labelled queries with no candidate group."""
    seen = {str(value) for value in seen_query_ids}
    records: list[dict[str, Any]] = []
    for row in query_labels.to_dict("records"):
        query_id = str(row.get("query_id") or "")
        if query_id in seen:
            continue
        record: dict[str, Any] = {
            "query_id": query_id,
            "fold": int(row["fold"]),
            "human_siret": normalize_siret(row.get("human_siret")),
            "truth_union_rank": None,
            "truth_master_rank": None,
            "truth_heuristic_hydration_rank": None,
            "truth_prerank_rank": None,
            "truth_ranked_rank": None,
            "truth_parent_present": False,
            "selected_hit_at_100": False,
            "selected_operational_hit_at_100": False,
            "retrieval_latency_ms": math.nan,
            "admission_latency_ms": math.nan,
            "end_to_end_latency_ms": math.nan,
            "union_candidate_count": 0,
            "selected_candidate_count": 0,
            "segment_state": normalize_text(row.get("ground_truth_state")),
            "segment_origin": normalize_text(row.get("data_origin")),
            "segment_seen_siren": normalize_text(
                row.get("seen_unseen_siren", row.get("siren_novelty"))
            ),
            "segment_new_site": normalize_text(
                row.get("known_new_site", row.get("site_novelty"))
            ),
            "segment_megapolis": normalize_text(
                row.get("megapolis", row.get("is_megapolis"))
            ),
            "segment_qualification": normalize_text(
                row.get("qualification", row.get("qualification_status"))
            ),
            "segment_label_kind": normalize_text(row.get("label_kind")),
            "segment_qualification_v2": normalize_text(row.get("qualification_v2")),
            "segment_qualification_v3": normalize_text(row.get("qualification_v3")),
            "official_catalog_present": row.get("official_catalog_present"),
            "loss_reason": (
                "OFFICIAL_CATALOG_ABSENT"
                if row.get("official_catalog_present") is False
                else "PARENT_NOT_RETRIEVED"
                if row.get("official_catalog_present") is True
                else "OFFICIAL_CATALOG_STATUS_UNKNOWN"
            ),
            "rescued_by_adaptive_allocation": False,
        }
        for cutoff in METRIC_CUTOFFS:
            record[f"union_hit_at_{cutoff}"] = False
            record[f"prerank_hit_at_{cutoff}"] = False
            record[f"heuristic_hydration_hit_at_{cutoff}"] = False
            record[f"ranked_hit_at_{cutoff}"] = False
        records.append(record)
    if not records:
        return pd.DataFrame()
    output = pd.DataFrame(records)
    if not set(output["fold"].astype(int)).issubset(set(config.train_folds)):
        raise ValueError("missing-label ledger attempted to open fold 0/1")
    return output


def summarize_outcomes(outcomes: pd.DataFrame, config: AdmissionConfigV12) -> Mapping[str, Any]:
    if outcomes.empty:
        raise ValueError("cannot summarize empty v12 outcomes")
    operational_mode = OPERATIONAL_EQUIVALENCE_MODE

    def loss_taxonomy(part: pd.DataFrame) -> Mapping[str, Any]:
        total = len(part)
        failures = part.loc[~part["selected_hit_at_100"].astype(bool)]
        failure_total = len(failures)
        by_reason: dict[str, Mapping[str, Any]] = {}
        for reason in (*EXCLUSIVE_LOSS_REASONS, *AUDIT_ONLY_LOSS_REASONS):
            count = int(failures["loss_reason"].eq(reason).sum())
            by_reason[reason] = {
                "count": count,
                "rate_all_queries": count / total if total else None,
                "rate_exact_failures": count / failure_total if failure_total else None,
            }
        classified = int(
            failures["loss_reason"].isin(EXCLUSIVE_LOSS_REASONS).sum()
        )
        unattributed = failure_total - classified
        hit_rows_valid = bool(
            part.loc[part["selected_hit_at_100"].astype(bool), "loss_reason"]
            .eq("HIT")
            .all()
        )
        return {
            "denominator_queries": int(total),
            "exact_failures": int(failure_total),
            "exact_failure_rate": failure_total / total if total else None,
            "classified_failures": classified,
            "unattributed_failures": int(unattributed),
            "exclusive_assignment_complete": bool(
                unattributed == 0 and hit_rows_valid
            ),
            "by_reason": by_reason,
        }

    def metrics(part: pd.DataFrame) -> Mapping[str, Any]:
        total = len(part)
        union = {str(cutoff): _metric(int(part[f"union_hit_at_{cutoff}"].sum()), total) for cutoff in METRIC_CUTOFFS}
        prerank = {str(cutoff): _metric(int(part[f"prerank_hit_at_{cutoff}"].sum()), total) for cutoff in METRIC_CUTOFFS}
        heuristic_hydration = {
            str(cutoff): _metric(
                int(part[f"heuristic_hydration_hit_at_{cutoff}"].sum()), total
            )
            for cutoff in METRIC_CUTOFFS
        }
        ranked = {str(cutoff): _metric(int(part[f"ranked_hit_at_{cutoff}"].sum()), total) for cutoff in METRIC_CUTOFFS}
        return {
            "union_recall": union,
            "prerank_recall": prerank,
            "heuristic_hydration_recall": heuristic_hydration,
            "ranked_recall": ranked,
            "selected_recall_at_100": _metric(int(part["selected_hit_at_100"].sum()), total),
            "selected_operational_recall_at_100": _metric(
                int(part["selected_operational_hit_at_100"].sum()), total
            ),
            "loss_taxonomy": loss_taxonomy(part),
        }

    overall = metrics(outcomes)
    by_fold = {str(int(fold)): metrics(part) for fold, part in outcomes.groupby("fold", sort=True)}
    segment_columns = {
        "active_closed": "segment_state",
        "crm_origin": "segment_origin",
        "seen_unseen_siren": "segment_seen_siren",
        "known_new_site": "segment_new_site",
        "megapolis": "segment_megapolis",
        "qualification": "segment_qualification",
        "label_kind": "segment_label_kind",
        "qualification_v2": "segment_qualification_v2",
        "qualification_v3": "segment_qualification_v3",
    }
    segments: dict[str, Mapping[str, Any]] = {}
    for public_name, column in segment_columns.items():
        if column not in outcomes:
            continue
        values = outcomes[column].fillna("").astype(str).replace("", "UNKNOWN")
        segments[public_name] = {
            str(value): metrics(outcomes.loc[index])
            for value, index in values.groupby(values).groups.items()
        }
    selected_metric = overall["selected_recall_at_100"]
    oracle_metric = overall["union_recall"]["2000"]
    worst_fold = min(float(value["selected_recall_at_100"]["rate"] or 0.0) for value in by_fold.values())
    latency_values = outcomes["end_to_end_latency_ms"].dropna().to_numpy(dtype=float)
    latency_complete = len(latency_values) == len(outcomes)
    latency = {
        "measured_queries": int(len(latency_values)),
        "total_queries": int(len(outcomes)),
        "p50_ms": float(np.percentile(latency_values, 50)) if len(latency_values) else None,
        "p95_ms": float(np.percentile(latency_values, 95)) if len(latency_values) else None,
        "p99_ms": float(np.percentile(latency_values, 99)) if len(latency_values) else None,
    }
    gates = {
        "oracle_at_2000": {"observed": oracle_metric["rate"], "minimum": config.gates["oracle_2000"], "passed": float(oracle_metric["rate"] or 0.0) >= config.gates["oracle_2000"], "decision_blocking": False, "below_target_status": "MASTER_HEADROOM_BELOW_TARGET"},
        "prerank_at_500": {"observed": overall["prerank_recall"]["500"]["rate"], "minimum": config.gates["prerank_500"], "passed": float(overall["prerank_recall"]["500"]["rate"] or 0.0) >= config.gates["prerank_500"]},
        "selected_recall_at_100": {"observed": selected_metric["rate"], "minimum": config.gates["recall_100"], "passed": float(selected_metric["rate"] or 0.0) >= config.gates["recall_100"]},
        "wilson95_lower_at_100": {"observed": (selected_metric["wilson_95"] or [0.0])[0], "minimum": config.gates["wilson_100"], "passed": float((selected_metric["wilson_95"] or [0.0])[0]) >= config.gates["wilson_100"]},
        "worst_fold_at_100": {"observed": worst_fold, "minimum": config.gates["worst_fold_100"], "passed": worst_fold >= config.gates["worst_fold_100"]},
        "latency_p95_ms": {
            "observed": latency["p95_ms"],
            "maximum": config.gates["latency_p95_ms"],
            "passed": bool(
                latency_complete
                and latency["p95_ms"] is not None
                and latency["p95_ms"] <= config.gates["latency_p95_ms"]
            ),
        },
        "latency_p99_ms": {
            "observed": latency["p99_ms"],
            "maximum": config.gates["latency_p99_ms"],
            "passed": bool(
                latency_complete
                and latency["p99_ms"] is not None
                and latency["p99_ms"] <= config.gates["latency_p99_ms"]
            ),
        },
    }
    losses = {str(key): int(value) for key, value in outcomes["loss_reason"].value_counts().to_dict().items()}
    for reason in (*EXCLUSIVE_LOSS_REASONS, *AUDIT_ONLY_LOSS_REASONS):
        losses.setdefault(reason, 0)
    losses["rescued_by_adaptive_allocation"] = int(outcomes["rescued_by_adaptive_allocation"].sum())
    official_catalog_known = bool(
        "official_catalog_present" in outcomes
        and outcomes["official_catalog_present"].notna().all()
    )
    gates["official_catalog_loss_taxonomy"] = {
        "observed": official_catalog_known,
        "required": True,
        "passed": official_catalog_known,
    }
    recall_passed = all(
        gates[name]["passed"]
        for name in (
            "prerank_at_500",
            "selected_recall_at_100",
            "wilson95_lower_at_100",
            "worst_fold_at_100",
        )
    )
    verdict = (
        "STOP"
        if not gates["official_catalog_loss_taxonomy"]["passed"]
        else "READY_FOR_FOLD1"
        if recall_passed
        and gates["latency_p95_ms"]["passed"]
        and gates["latency_p99_ms"]["passed"]
        else "PIVOT_GENERATION"
        if not gates["prerank_at_500"]["passed"]
        else "PIVOT_ADMISSION"
        if not recall_passed
        else "PIVOT_LATENCY"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "policy_id": config.policy_id,
        "population": "ALL_USABLE_HUMAN_LABELS",
        "operational_equivalence_mode": operational_mode,
        "query_count": int(len(outcomes)),
        "metrics": overall,
        "master_exact_recall_at_2000": overall["union_recall"]["2000"],
        "prerank_exact_recall_at_500": overall["prerank_recall"]["500"],
        "product_exact_recall_at_100": overall["selected_recall_at_100"],
        "product_operational_recall_at_100": overall[
            "selected_operational_recall_at_100"
        ],
        "latency": latency,
        "outer_folds": by_fold,
        "segments": segments,
        "losses": losses,
        "loss_taxonomy": overall["loss_taxonomy"],
        "loss_taxonomy_availability": {
            "official_catalog_present_available": official_catalog_known,
            "official_catalog_absent_attribution_available": official_catalog_known,
        },
        "gates": gates,
        "integrity": {
            "features_computed_before_sampling": True,
            "parent_scores_for_site_fit": "SEALED_CASCADE_OOF_ONLY",
            "parent_refit_on_pass_b": False,
            "validation_candidate_sampling": False,
            "positive_injection": False,
            "operational_equivalence_mode": operational_mode,
            "official_catalog_loss_taxonomy_available": official_catalog_known,
            "fixed_parent_cap": False,
            "fixed_anchor_or_consensus_quotas": False,
            "only_protected_evidence": "JOINT_NAME_AND_GEOGRAPHY",
        },
        "verdict": verdict,
        "stop_reason": (
            "INTEGRITY_OFFICIAL_CATALOG_UNAVAILABLE"
            if verdict == "STOP"
            else None
        ),
    }


def outer_oof_evaluate(
    raw_labelled_union: Mapping[int, pd.DataFrame],
    config: AdmissionConfigV12,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Mapping[str, Any]]:
    if not isinstance(raw_labelled_union, Mapping):
        raise ValueError(
            "outer_oof_evaluate requires three split-specific cascade contexts"
        )
    complete_by_outer: dict[int, pd.DataFrame] = {}
    train_by_outer: dict[int, pd.DataFrame] = {}
    for heldout in config.train_folds:
        complete = canonicalize_union(
            raw_labelled_union[heldout],
            config,
            labels_allowed=True,
            require_labels=True,
        )
        fit_folds = set(config.train_folds).difference({heldout})
        complete_by_outer[heldout] = complete
        train_by_outer[heldout] = complete[complete["fold"].isin(fit_folds)]
    models, diagnostics = fit_outer_oof_models(train_by_outer, config)
    scored_parts: list[pd.DataFrame] = []
    selected_parts: list[pd.DataFrame] = []
    for heldout in config.train_folds:
        validation = complete_by_outer[heldout][
            complete_by_outer[heldout]["fold"].eq(heldout)
        ].copy()
        blind = validation.copy()
        blind["human_siret"] = ""
        scored = score_canonical_candidates(models[heldout], blind, config)
        selected = select_top100(scored, config)
        scored_parts.append(scored)
        selected_parts.append(selected)
    scored = pd.concat(scored_parts, ignore_index=True)
    selected = pd.concat(selected_parts, ignore_index=True)
    labelled_validation = pd.concat(
        [
            complete_by_outer[heldout][
                complete_by_outer[heldout]["fold"].eq(heldout)
            ]
            for heldout in config.train_folds
        ],
        ignore_index=True,
    )
    outcomes, summary = evaluate_all_human(
        labelled_validation, scored=scored, selected=selected, config=config
    )
    summary = dict(summary)
    summary["nested_oof"] = {
        "outer_folds": list(config.train_folds),
        "model_diagnostics": diagnostics,
        "fold0_opened": False,
        "fold1_opened": False,
        "ground_truth_parent_disjoint": True,
    }
    return scored, selected, outcomes, summary


__all__ = [
    "AdmissionConfigV12",
    "BASE_PARENT_FEATURES",
    "BASE_SITE_FEATURES",
    "DEFAULT_CHANNELS",
    "DEFAULT_CHANNEL_FAMILIES",
    "HierarchicalModelsV12",
    "METRIC_CUTOFFS",
    "SCHEMA_VERSION",
    "assert_ground_truth_parent_disjoint",
    "adapt_cascade_parent_rows",
    "attach_sealed_parent_scores",
    "attach_site_scores",
    "canonicalize_union",
    "canonicalize_cascade_parent_rows",
    "channel_feature_names",
    "diagnostic_parent_rows_from_hydrated_union",
    "evaluate_all_human",
    "fit_outer_oof_models",
    "fit_one_outer_site_model",
    "fit_hierarchical_models",
    "fit_parent_ranker",
    "fit_parent_ranker_from_cascade_rows",
    "fit_site_ranker",
    "missing_label_outcomes",
    "outer_oof_evaluate",
    "parent_feature_names",
    "predict_top100",
    "reject_label_columns",
    "sample_precomputed_candidates",
    "score_canonical_candidates",
    "select_top100",
    "site_feature_names",
    "summarize_outcomes",
]
