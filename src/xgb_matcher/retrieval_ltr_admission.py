"""Label-blind union, LambdaMART admission and selective retrieval metrics.

The module is deliberately independent from the retrieval indexes.  It consumes
lexical/hierarchical candidate rows, builds at most 2,000 candidates per query,
and trains one fixed ``rank:ndcg`` XGBRanker on human-labelled folds 2/3/4.
Fold 0 is development and fold 1 can only be opened by the dedicated runner.

SIRET, SIREN and query identifiers remain audit columns; none is a model
feature.  Operational same-SIREN/same-site alternatives are reported as a
separate metric and are never taught to the exact-SIRET ranker as negatives.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import time
from statistics import NormalDist
from typing import Any, Iterable, Mapping, Sequence
import unicodedata

import numpy as np
import pandas as pd
import xgboost as xgb


SCHEMA_VERSION = "sireto-retrieval-ltr-admission-v1"
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config/retrieval_ltr_admission_v1.json"
)
MINIMUM_INPUT_COLUMNS = {
    "query_id",
    "siret",
    "siren",
    "fold",
    "gt_siret",
    "crm_name",
    "crm_address",
    "crm_number",
    "crm_insee",
    "crm_postcode",
}
BASE_FEATURES = (
    "retrieval_rank_reciprocal",
    "retrieval_rank_log_inverse",
    "retrieval_score",
    "retrieval_channel_count",
    "exact_signal_count",
    "is_exact_protected",
    "is_consensus_protected",
    "name_token_jaccard",
    "name_char_trigram_dice",
    "address_token_jaccard",
    "address_char_trigram_dice",
    "street_number_match",
    "postcode_match",
    "insee_match",
    "candidate_active",
    "candidate_is_siege",
)
_ID_TOKENS = ("query_id", "siret", "siren")
_SPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^A-Z0-9]+")
_SOURCE_SPLIT_RE = re.compile(r"[+,|;\s]+")
_LEADING_NUMBER_RE = re.compile(r"^(\d{1,5})(?:\s|$)")


@dataclass(frozen=True)
class AdmissionConfig:
    """Validated immutable view of the single checked-in experiment config."""

    raw: Mapping[str, Any]

    @classmethod
    def load(cls, path: Path | str = DEFAULT_CONFIG_PATH) -> "AdmissionConfig":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def __post_init__(self) -> None:
        if self.raw.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported retrieval LTR config schema")
        if self.train_folds != (2, 3, 4) or self.dev_fold != 0 or self.test_fold != 1:
            raise ValueError("The frozen fold policy must be train=2/3/4, dev=0, test=1")
        if self.internal_union_cap != 2000:
            raise ValueError("The frozen internal union cap must be 2000")
        if self.max_candidates != 100:
            raise ValueError("The absolute output ceiling must be 100")
        if self.xgb_params.get("objective") != "rank:ndcg":
            raise ValueError("Only LambdaMART rank:ndcg is allowed")
        if self.exact_slots + self.consensus_slots > self.max_candidates:
            raise ValueError("Protection slots exceed the output ceiling")
        if not set(self.exact_channels).issubset(self.channels):
            raise ValueError("Exact channels must be declared retrieval channels")
        assert_feature_order_is_identifier_free(feature_order(self))

    @property
    def policy_id(self) -> str:
        return str(self.raw["policy_id"])

    @property
    def train_folds(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.raw["folds"]["train"])

    @property
    def dev_fold(self) -> int:
        return int(self.raw["folds"]["dev"])

    @property
    def test_fold(self) -> int:
        return int(self.raw["folds"]["test"])

    @property
    def internal_union_cap(self) -> int:
        return int(self.raw["limits"]["internal_union_cap"])

    @property
    def max_training_candidates(self) -> int:
        return int(self.raw["limits"]["max_training_candidates_per_query"])

    @property
    def max_candidates(self) -> int:
        return int(self.raw["limits"]["max_candidates"])

    @property
    def exact_slots(self) -> int:
        return int(self.raw["limits"]["exact_protection_slots"])

    @property
    def consensus_slots(self) -> int:
        return int(self.raw["limits"]["consensus_protection_slots"])

    @property
    def channels(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.raw["channels"])

    @property
    def exact_channels(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.raw["exact_channels"])

    @property
    def consensus_min_channels(self) -> int:
        return int(self.raw["consensus_min_channels"])

    @property
    def xgb_params(self) -> dict[str, Any]:
        return dict(self.raw["xgboost"])

    @property
    def gates(self) -> Mapping[str, float]:
        return self.raw["gates"]

    @property
    def synthetic_markers(self) -> tuple[str, ...]:
        return tuple(str(value).lower() for value in self.raw["forbidden"]["synthetic_markers"])

    @property
    def dense_markers(self) -> tuple[str, ...]:
        return tuple(str(value).lower() for value in self.raw["forbidden"]["dense_channel_markers"])


def feature_order(config: AdmissionConfig) -> list[str]:
    output = list(BASE_FEATURES)
    for channel in config.channels:
        output.extend(
            [
                f"source_{channel}",
                f"rank_reciprocal_{channel}",
                f"score_{channel}",
            ]
        )
    return output


def assert_feature_order_is_identifier_free(features: Sequence[str]) -> None:
    offending = [
        name
        for name in features
        if any(token in name.lower() for token in _ID_TOKENS)
    ]
    if offending:
        raise ValueError(f"Identifier-bearing model features are forbidden: {offending}")


def normalize_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(character for character in text if not unicodedata.combining(character))
    return _SPACE_RE.sub(" ", _NON_ALNUM_RE.sub(" ", text.upper())).strip()


def normalize_siret(value: Any, *, allow_empty: bool = False) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        value = ""
    text = re.sub(r"\D", "", str(value))
    if not text and allow_empty:
        return ""
    if not text or len(text) > 14:
        raise ValueError(f"Invalid SIRET: {value!r}")
    return text.zfill(14)


def normalize_siren(value: Any, *, siret: str = "") -> str:
    text = re.sub(r"\D", "", str(value or ""))
    if not text and siret:
        return siret[:9]
    if not text or len(text) > 9:
        raise ValueError(f"Invalid SIREN: {value!r}")
    return text.zfill(9)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().upper() in {"1", "TRUE", "T", "YES", "Y", "OUI", "A"}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _positive_rank(value: Any) -> int | None:
    result = _finite_float(value, default=0.0)
    return int(result) if result >= 1 else None


def _parse_values(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [str(item) for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = []
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item).strip()]
    return [text]


def _candidate_names(row: Mapping[str, Any]) -> list[str]:
    output: list[str] = []
    for column in (
        "names",
        "candidate_names",
        "candidate_name",
        "denomination",
        "denominationUniteLegale",
        "enseigne",
        "enseigne1Etablissement",
    ):
        output.extend(_parse_values(row.get(column)))
    return sorted(set(filter(None, map(normalize_text, output))))


def _candidate_addresses(row: Mapping[str, Any]) -> list[str]:
    output: list[str] = []
    for column in (
        "addresses",
        "candidate_addresses",
        "candidate_address",
        "address",
        "adresse",
        "adresseEtablissement",
    ):
        output.extend(_parse_values(row.get(column)))
    if not output:
        assembled = " ".join(
            str(row.get(column) or "")
            for column in (
                "number",
                "numeroVoieEtablissement",
                "typeVoieEtablissement",
                "libelleVoieEtablissement",
                "postcode",
                "codePostalEtablissement",
            )
        )
        output.append(assembled)
    return sorted(set(filter(None, map(normalize_text, output))))


def _first_value(row: Mapping[str, Any], columns: Sequence[str]) -> Any:
    for column in columns:
        value = row.get(column)
        if not _is_missing(value) and str(value).strip():
            return value
    return ""


def _channel_sources(row: Mapping[str, Any], config: AdmissionConfig) -> set[str]:
    raw_sources: list[str] = []
    raw_sources.extend(_parse_values(row.get("retrieval_source")))
    raw_sources.extend(_parse_values(row.get("channel")))
    split: set[str] = set()
    for raw in raw_sources:
        split.update(filter(None, _SOURCE_SPLIT_RE.split(str(raw).strip())))
    for channel in config.channels:
        for column in (
            f"{channel}_rank",
            f"rank_{channel}",
            f"retrieval_rank_{channel}",
            f"{channel}_retrieval_rank",
        ):
            if column in row and _positive_rank(row.get(column)) is not None:
                split.add(channel)
        for column in (
            f"{channel}_score",
            f"score_{channel}",
            f"retrieval_score_{channel}",
            f"{channel}_retrieval_score",
        ):
            if column in row and row.get(column) is not None and not pd.isna(row.get(column)):
                split.add(channel)
    return split.intersection(config.channels)


def validate_candidate_input(
    frame: pd.DataFrame,
    config: AdmissionConfig,
    *,
    allowed_folds: Iterable[int],
) -> None:
    missing = sorted(MINIMUM_INPUT_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Candidate input is missing columns: {missing}")
    if frame.empty:
        raise ValueError("Candidate input is empty")
    allowed = set(int(value) for value in allowed_folds)
    observed_folds = set(pd.to_numeric(frame["fold"], errors="raise").astype(int))
    if not observed_folds.issubset(allowed):
        raise ValueError(f"Forbidden fold rows were opened: {sorted(observed_folds - allowed)}")

    column_names = " ".join(str(column).lower() for column in frame.columns)
    dense_columns = [marker for marker in config.dense_markers if marker in column_names]
    if dense_columns:
        raise ValueError(f"Dense/learned-sparse columns are forbidden: {dense_columns}")
    source_columns = [
        column for column in ("retrieval_source", "channel") if column in frame.columns
    ]
    source_text = " ".join(
        " ".join(frame[column].fillna("").astype(str).str.lower().unique())
        for column in source_columns
    )
    dense_sources = [marker for marker in config.dense_markers if marker in source_text]
    if dense_sources:
        raise ValueError(f"Dense/learned-sparse channels are forbidden: {dense_sources}")

    provenance_columns = [
        column
        for column in ("source_kind", "label_source", "provenance", "dataset_kind")
        if column in frame.columns
    ]
    provenance_text = " ".join(
        " ".join(frame[column].fillna("").astype(str).str.lower().unique())
        for column in provenance_columns
    )
    synthetic = [marker for marker in config.synthetic_markers if marker in provenance_text]
    if synthetic:
        raise ValueError(f"Synthetic rows are forbidden in this experiment: {synthetic}")
    if "is_synthetic" in frame.columns and frame["is_synthetic"].map(_as_bool).any():
        raise ValueError("Synthetic rows are forbidden in this experiment")

    normalized_ids = frame["query_id"].astype(str)
    if normalized_ids.eq("").any():
        raise ValueError("query_id cannot be empty")
    invariant_columns = [
        "fold",
        "gt_siret",
        "crm_name",
        "crm_address",
        "crm_number",
        "crm_insee",
        "crm_postcode",
    ]
    invariant_columns.extend(
        column
        for column in (
            "retrieval_latency_ms",
            "historical_ground_truth_siret",
            "v2_exact",
            "v2_label_kind",
            "qualification_v2",
            "v3_exact",
            "v3_label_kind",
            "qualification_v3",
            "label_kind",
            "ground_truth_state",
            "gt_state",
            "pool_size",
            "mega_base_pool",
            "unseen_siren",
            "is_unseen_siren",
        )
        if column in frame.columns
    )
    for column in invariant_columns:
        varies = (
            frame.assign(_query_id=normalized_ids)
            .groupby("_query_id")[column]
            .nunique(dropna=False)
            .gt(1)
            .any()
        )
        if varies:
            raise ValueError(f"Query-level column varies within query: {column}")


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = set(normalize_text(left).split())
    right_tokens = set(normalize_text(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _trigram_dice(left: str, right: str) -> float:
    def grams(value: str) -> set[str]:
        normalized = f" {normalize_text(value).replace(' ', '_')} "
        return {normalized[index : index + 3] for index in range(max(0, len(normalized) - 2))}

    left_grams = grams(left)
    right_grams = grams(right)
    if not left_grams or not right_grams:
        return 0.0
    return 2 * len(left_grams & right_grams) / (len(left_grams) + len(right_grams))


def _max_similarity(query: str, values: Iterable[str], metric: Any) -> float:
    return max((metric(query, value) for value in values), default=0.0)


def _leading_number(address: str) -> str:
    match = _LEADING_NUMBER_RE.match(normalize_text(address))
    return match.group(1) if match else ""


def _site_key(
    *,
    insee: str,
    postcode: str,
    number: str,
    addresses: Sequence[str],
    explicit: Any = "",
) -> str:
    if str(explicit or "").strip():
        return f"EXPLICIT:{normalize_text(explicit)}"
    address = normalize_text(addresses[0]) if addresses else ""
    normalized_number = normalize_text(number) or _leading_number(address)
    geography = normalize_text(insee) or normalize_text(postcode)
    if not geography or not address:
        return ""
    return "|".join((geography, normalized_number, address))


def _per_channel_value(row: Mapping[str, Any], channel: str, kind: str) -> Any:
    for column in (
        f"{channel}_{kind}",
        f"{kind}_{channel}",
        f"retrieval_{kind}_{channel}",
        f"{channel}_retrieval_{kind}",
    ):
        if column in row:
            return row.get(column)
    return None


def _base_candidate(
    row: Mapping[str, Any], config: AdmissionConfig
) -> dict[str, Any]:
    siret = normalize_siret(row.get("siret"))
    siren = normalize_siren(row.get("siren"), siret=siret)
    if siren != siret[:9]:
        raise ValueError(f"Candidate SIREN/SIRET mismatch: {siren}/{siret}")
    names = _candidate_names(row)
    addresses = _candidate_addresses(row)
    postcode = normalize_text(
        _first_value(row, ("postcode", "candidate_postcode", "codePostalEtablissement"))
    )
    insee = normalize_text(
        _first_value(row, ("insee", "candidate_insee", "codeCommuneEtablissement"))
    )
    number = normalize_text(
        _first_value(row, ("number", "candidate_number", "numeroVoieEtablissement"))
    )
    state = str(
        _first_value(row, ("state", "candidate_state", "etatAdministratifEtablissement"))
        or "A"
    ).upper()
    return {
        "siret": siret,
        "siren": siren,
        "sources": set(),
        "retrieval_rank": 2_147_483_647,
        "retrieval_score": -math.inf,
        "channel_ranks": {},
        "channel_scores": {},
        "candidate_names": set(names),
        "candidate_addresses": set(addresses),
        "candidate_postcode": postcode,
        "candidate_insee": insee,
        "candidate_number": number,
        "candidate_state": state,
        "candidate_is_siege": _as_bool(
            _first_value(row, ("is_siege", "candidate_is_siege", "etablissementSiege"))
        ),
        "explicit_site_key": _first_value(row, ("site_key", "address_id", "official_site_key")),
    }


def _merge_candidate_hit(
    candidate: dict[str, Any],
    row: Mapping[str, Any],
    config: AdmissionConfig,
) -> None:
    sources = _channel_sources(row, config)
    candidate["sources"].update(sources)
    rank = _positive_rank(row.get("retrieval_rank"))
    if rank is not None:
        candidate["retrieval_rank"] = min(candidate["retrieval_rank"], rank)
    score = _finite_float(row.get("retrieval_score"), default=-math.inf)
    candidate["retrieval_score"] = max(candidate["retrieval_score"], score)
    candidate["candidate_names"].update(_candidate_names(row))
    candidate["candidate_addresses"].update(_candidate_addresses(row))
    for channel in sources:
        channel_rank = _positive_rank(_per_channel_value(row, channel, "rank"))
        if channel_rank is None:
            channel_rank = rank
        if channel_rank is not None:
            previous = candidate["channel_ranks"].get(channel, 2_147_483_647)
            candidate["channel_ranks"][channel] = min(previous, channel_rank)
        channel_score = _per_channel_value(row, channel, "score")
        if channel_score is None:
            channel_score = row.get("retrieval_score")
        numeric_score = _finite_float(channel_score, default=-math.inf)
        previous_score = candidate["channel_scores"].get(channel, -math.inf)
        candidate["channel_scores"][channel] = max(previous_score, numeric_score)


def _query_metadata(group: pd.DataFrame) -> dict[str, Any]:
    row = group.iloc[0]
    gt_siret = normalize_siret(row.get("gt_siret"), allow_empty=True)
    label_kind = str(_first_value(row, ("label_kind",)) or "MATCH_EXACT")
    historical = normalize_siret(
        row.get("historical_ground_truth_siret"), allow_empty=True
    ) if "historical_ground_truth_siret" in group.columns else gt_siret
    historical = historical or gt_siret

    def qualification(columns: Sequence[str]) -> bool | None:
        for column in columns:
            if column not in group.columns:
                continue
            value = row.get(column)
            if _is_missing(value) or str(value).strip() == "":
                continue
            normalized = str(value).strip().upper()
            if normalized in {"MATCH_EXACT", "EXACT", "EXACT_IDENTIFIABLE"}:
                return True
            if normalized in {
                "AMBIGUOUS",
                "UNRESOLVED",
                "NO_MATCH",
                "FALSE",
                "F",
                "0",
            }:
                return False
            return _as_bool(value)
        return None

    v2_exact = qualification(("v2_exact", "v2_label_kind", "qualification_v2"))
    v3_exact = qualification(
        ("v3_exact", "v3_label_kind", "qualification_v3", "label_kind")
    )
    latency_measured = (
        "retrieval_latency_ms" in group.columns
        and not _is_missing(row.get("retrieval_latency_ms"))
    )
    retrieval_latency = (
        _finite_float(row.get("retrieval_latency_ms"), default=0.0)
        if latency_measured
        else 0.0
    )
    if retrieval_latency < 0:
        raise ValueError("retrieval_latency_ms cannot be negative")
    pool_size = (
        _finite_float(row.get("pool_size"), default=-1.0)
        if "pool_size" in group.columns and not _is_missing(row.get("pool_size"))
        else -1.0
    )
    return {
        "query_id": str(row["query_id"]),
        "fold": int(row["fold"]),
        "gt_siret": gt_siret,
        "historical_gt_siret": historical,
        "crm_name": normalize_text(row.get("crm_name")),
        "crm_address": normalize_text(row.get("crm_address")),
        "crm_number": normalize_text(row.get("crm_number")),
        "crm_insee": normalize_text(row.get("crm_insee")),
        "crm_postcode": normalize_text(row.get("crm_postcode")),
        "label_kind": label_kind,
        "identifiable_exact": (
            _as_bool(row.get("identifiable_exact"), default=bool(gt_siret))
            if "identifiable_exact" in group.columns
            else (
                bool(v3_exact) and bool(gt_siret)
                if v3_exact is not None
                else label_kind == "MATCH_EXACT" and bool(gt_siret)
            )
        ),
        "acceptable_sirets_operational_json": str(
            _first_value(
                row,
                (
                    "acceptable_sirets_operational_json",
                    "acceptable_sirets_operational",
                ),
            )
            or "[]"
        ),
        "v2_exact_available": v2_exact is not None,
        "v2_exact": bool(v2_exact) if v2_exact is not None else False,
        "v3_exact_available": v3_exact is not None,
        "v3_exact": bool(v3_exact) if v3_exact is not None else False,
        "retrieval_latency_measured": latency_measured,
        "retrieval_latency_ms": retrieval_latency,
        "ground_truth_state_available": any(
            column in group.columns for column in ("ground_truth_state", "gt_state")
        ),
        "ground_truth_state": str(
            _first_value(row, ("ground_truth_state", "gt_state")) or ""
        ).upper(),
        "pool_size_available": "pool_size" in group.columns,
        "pool_size": pool_size,
        "mega_base_pool_available": "mega_base_pool" in group.columns,
        "mega_base_pool": _as_bool(row.get("mega_base_pool"), default=False),
        "unseen_siren_available": any(
            column in group.columns for column in ("unseen_siren", "is_unseen_siren")
        ),
        "unseen_siren": _as_bool(
            _first_value(row, ("unseen_siren", "is_unseen_siren")), default=False
        ),
    }


def _materialize_candidate(
    candidate: dict[str, Any],
    metadata: Mapping[str, Any],
    config: AdmissionConfig,
) -> dict[str, Any]:
    sources = set(candidate["sources"])
    names = sorted(candidate["candidate_names"])
    addresses = sorted(candidate["candidate_addresses"])
    rank = int(candidate["retrieval_rank"])
    if rank == 2_147_483_647:
        rank = min(candidate["channel_ranks"].values(), default=config.internal_union_cap + 1)
    score = float(candidate["retrieval_score"])
    if not math.isfinite(score):
        score = 0.0
    exact_count = len(sources.intersection(config.exact_channels))
    row: dict[str, Any] = {
        **metadata,
        "candidate_siret": candidate["siret"],
        "candidate_siren": candidate["siren"],
        "retrieval_source": "+".join(sorted(sources)),
        "retrieval_rank": rank,
        "retrieval_score": score,
        "retrieval_channel_count": len(sources),
        "candidate_names_json": json.dumps(
            names, ensure_ascii=False, separators=(",", ":")
        ),
        "candidate_addresses_json": json.dumps(
            addresses, ensure_ascii=False, separators=(",", ":")
        ),
        "candidate_postcode": candidate["candidate_postcode"],
        "candidate_insee": candidate["candidate_insee"],
        "candidate_number": candidate["candidate_number"],
        "candidate_state": candidate["candidate_state"],
        "candidate_site_key": _site_key(
            insee=candidate["candidate_insee"],
            postcode=candidate["candidate_postcode"],
            number=candidate["candidate_number"],
            addresses=addresses,
            explicit=candidate["explicit_site_key"],
        ),
        "candidate_is_siege_audit": bool(candidate["candidate_is_siege"]),
        "retrieval_rank_reciprocal": 1.0 / max(1, rank),
        "retrieval_rank_log_inverse": 1.0 / math.log2(max(2, rank + 1)),
        "exact_signal_count": exact_count,
        "is_exact_protected": int(exact_count > 0),
        "is_consensus_protected": int(len(sources) >= config.consensus_min_channels),
        "name_token_jaccard": _max_similarity(
            metadata["crm_name"], names, _token_jaccard
        ),
        "name_char_trigram_dice": _max_similarity(
            metadata["crm_name"], names, _trigram_dice
        ),
        "address_token_jaccard": _max_similarity(
            metadata["crm_address"], addresses, _token_jaccard
        ),
        "address_char_trigram_dice": _max_similarity(
            metadata["crm_address"], addresses, _trigram_dice
        ),
        "street_number_match": int(
            bool(metadata["crm_number"])
            and metadata["crm_number"]
            == (candidate["candidate_number"] or _leading_number(addresses[0] if addresses else ""))
        ),
        "postcode_match": int(
            bool(metadata["crm_postcode"])
            and metadata["crm_postcode"] == candidate["candidate_postcode"]
        ),
        "insee_match": int(
            bool(metadata["crm_insee"])
            and metadata["crm_insee"] == candidate["candidate_insee"]
        ),
        "candidate_active": int(candidate["candidate_state"] != "F"),
        "candidate_is_siege": int(candidate["candidate_is_siege"]),
    }
    for channel in config.channels:
        channel_rank = candidate["channel_ranks"].get(channel)
        channel_score = candidate["channel_scores"].get(channel, 0.0)
        if not math.isfinite(channel_score):
            channel_score = 0.0
        row[f"source_{channel}"] = int(channel in sources)
        row[f"rank_reciprocal_{channel}"] = (
            1.0 / channel_rank if channel_rank else 0.0
        )
        row[f"score_{channel}"] = float(channel_score)
        row[f"rank_{channel}"] = channel_rank
    return row


def _union_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """A label-blind cap: exact and multi-channel evidence survive first."""
    return (
        -int(row["is_exact_protected"]),
        -int(row["exact_signal_count"]),
        -int(row["is_consensus_protected"]),
        -int(row["retrieval_channel_count"]),
        int(row["retrieval_rank"]),
        -float(row["retrieval_score"]),
        str(row["candidate_siret"]),
    )


def build_internal_union(
    frame: pd.DataFrame,
    config: AdmissionConfig,
    *,
    allowed_folds: Iterable[int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Deduplicate hits and cap each label-blind candidate union at 2,000."""
    validate_candidate_input(frame, config, allowed_folds=allowed_folds)
    working = frame.copy()
    working["query_id"] = working["query_id"].astype(str)
    records: list[dict[str, Any]] = []
    latencies: list[float] = []
    before_counts: list[int] = []
    for _query_id, group in working.groupby("query_id", sort=True):
        started = time.perf_counter()
        metadata = _query_metadata(group)
        by_siret: dict[str, dict[str, Any]] = {}
        for raw in group.to_dict("records"):
            siret = normalize_siret(raw.get("siret"))
            if siret not in by_siret:
                by_siret[siret] = _base_candidate(raw, config)
            candidate = by_siret[siret]
            _merge_candidate_hit(candidate, raw, config)
        materialized = [
            _materialize_candidate(candidate, metadata, config)
            for candidate in by_siret.values()
        ]
        materialized.sort(key=_union_sort_key)
        before_counts.append(len(materialized))
        kept = materialized[: config.internal_union_cap]
        elapsed = (time.perf_counter() - started) * 1000.0
        latencies.append(elapsed)
        for position, row in enumerate(kept, start=1):
            row["union_rank"] = position
            row["union_latency_ms"] = elapsed
            records.append(row)
    output = pd.DataFrame(records)
    if output.empty:
        raise ValueError("No candidates survived union construction")
    counts = output.groupby("query_id").size()
    duplicate_count = int(output.duplicated(["query_id", "candidate_siret"]).sum())
    if duplicate_count or int(counts.max()) > config.internal_union_cap:
        raise AssertionError("Internal union integrity failure")
    diagnostics = {
        "query_count": int(output["query_id"].nunique()),
        "raw_row_count": int(len(frame)),
        "deduplicated_candidate_count": int(sum(before_counts)),
        "retained_candidate_count": int(len(output)),
        "queries_capped_at_2000": int(
            sum(count > config.internal_union_cap for count in before_counts)
        ),
        "max_candidates_before_cap": int(max(before_counts)),
        "max_candidates_after_cap": int(counts.max()),
        "duplicate_query_candidate_pairs": duplicate_count,
        "union_latency_ms": _latency_summary(latencies),
        "positive_injection": False,
        "synthetic_rows": 0,
        "dense_channels": 0,
    }
    return output, diagnostics


def _parse_operational_sirets(value: Any) -> set[str]:
    output: set[str] = set()
    for item in _parse_values(value):
        try:
            output.add(normalize_siret(item))
        except ValueError:
            continue
    return output


def operational_sirets_for_group(group: pd.DataFrame) -> set[str]:
    """Return exact plus policy-declared/strict same-site SIRETs."""
    gt_siret = normalize_siret(group.iloc[0]["gt_siret"], allow_empty=True)
    if not gt_siret:
        return set()
    output = {gt_siret}
    declared = _parse_operational_sirets(
        group.iloc[0].get("acceptable_sirets_operational_json", "[]")
    )
    cross_siren = {value for value in declared if value[:9] != gt_siret[:9]}
    if cross_siren:
        raise ValueError("Operational alternatives must share the exact SIREN")
    output.update(declared)
    truth = group[group["candidate_siret"].eq(gt_siret)]
    if truth.empty:
        return output
    truth_site = str(truth.iloc[0].get("candidate_site_key") or "")
    if not truth_site:
        return output
    same_site = group[
        group["candidate_siren"].eq(gt_siret[:9])
        & group["candidate_site_key"].fillna("").astype(str).eq(truth_site)
    ]["candidate_siret"]
    output.update(same_site.astype(str))
    return output


def prepare_training_rows(
    union: pd.DataFrame,
    config: AdmissionConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build exact ranking groups and remove operational siblings as negatives."""
    expected_folds = set(config.train_folds)
    observed = set(union["fold"].astype(int))
    if not observed.issubset(expected_folds):
        raise ValueError("Non-training folds reached LambdaMART fit preparation")
    output: list[pd.DataFrame] = []
    excluded_same_site = 0
    missing_positive = 0
    groups_without_negative = 0
    non_identifiable_groups = 0
    for _query_id, group in union.groupby("query_id", sort=True):
        group = group.copy()
        if not _as_bool(group.iloc[0]["identifiable_exact"]):
            non_identifiable_groups += 1
            continue
        exact = normalize_siret(group.iloc[0]["gt_siret"], allow_empty=True)
        positive = group["candidate_siret"].eq(exact)
        if not exact or int(positive.sum()) != 1:
            missing_positive += 1
            continue
        operational = operational_sirets_for_group(group)
        excluded = group["candidate_siret"].isin(operational - {exact})
        excluded_same_site += int(excluded.sum())
        eligible = group[~excluded].copy()
        negatives = eligible[~eligible["candidate_siret"].eq(exact)].sort_values(
            [
                "is_exact_protected",
                "is_consensus_protected",
                "retrieval_channel_count",
                "union_rank",
                "candidate_siret",
            ],
            ascending=[False, False, False, True, True],
            kind="stable",
        )
        if negatives.empty:
            groups_without_negative += 1
            continue
        positive_row = eligible[eligible["candidate_siret"].eq(exact)]
        retained = pd.concat(
            [
                positive_row,
                negatives.head(config.max_training_candidates - 1),
            ],
            ignore_index=True,
        )
        retained["ltr_label"] = retained["candidate_siret"].eq(exact).astype(np.int8)
        output.append(retained)
    if not output:
        raise ValueError("No trainable ranking groups with a natural positive")
    training = pd.concat(output, ignore_index=True).sort_values(
        ["query_id", "ltr_label", "union_rank", "candidate_siret"],
        ascending=[True, False, True, True],
        kind="stable",
    )
    diagnostics = {
        "trainable_query_count": int(training["query_id"].nunique()),
        "training_row_count": int(len(training)),
        "natural_positive_count": int(training["ltr_label"].sum()),
        "queries_missing_natural_positive": missing_positive,
        "groups_without_negative": groups_without_negative,
        "non_identifiable_groups_excluded": non_identifiable_groups,
        "same_siren_same_site_negatives_excluded": excluded_same_site,
        "max_training_candidates_per_query": int(training.groupby("query_id").size().max()),
        "positive_injection": False,
    }
    return training.reset_index(drop=True), diagnostics


def train_ranker(
    union: pd.DataFrame,
    config: AdmissionConfig,
) -> tuple[xgb.XGBRanker, dict[str, Any]]:
    """Fit the single frozen LambdaMART configuration on folds 2/3/4 only."""
    started = time.perf_counter()
    training, diagnostics = prepare_training_rows(union, config)
    features = feature_order(config)
    assert_feature_order_is_identifier_free(features)
    qid = training["query_id"].astype("category").cat.codes.to_numpy(dtype=np.int64)
    if np.any(qid[1:] < qid[:-1]):
        raise AssertionError("XGBoost qid values must be sorted")
    model = xgb.XGBRanker(**config.xgb_params)
    model.fit(
        training[features].fillna(0.0).to_numpy(dtype=np.float32),
        training["ltr_label"].to_numpy(dtype=np.int8),
        qid=qid,
        verbose=False,
    )
    diagnostics = {
        **diagnostics,
        "train_folds": list(config.train_folds),
        "feature_order": features,
        "xgboost": config.xgb_params,
        "training_latency_ms": (time.perf_counter() - started) * 1000.0,
        "test_fold_read": False,
    }
    return model, diagnostics


def protected_top100(group: pd.DataFrame, config: AdmissionConfig) -> pd.DataFrame:
    """Reserve exact/consensus evidence, then fill by LambdaMART score."""
    if group["candidate_siret"].duplicated().any():
        raise ValueError("Duplicate candidates reached final admission")
    score_order = ["ltr_score", "retrieval_channel_count", "union_rank", "candidate_siret"]
    score_ascending = [False, False, True, True]
    exact = group[group["is_exact_protected"].eq(1)].sort_values(
        ["exact_signal_count", *score_order],
        ascending=[False, *score_ascending],
        kind="stable",
    )
    selected_indices: list[int] = list(exact.head(config.exact_slots).index)
    selected_set = set(selected_indices)
    consensus = group[
        group["is_consensus_protected"].eq(1) & ~group.index.isin(selected_set)
    ].sort_values(score_order, ascending=score_ascending, kind="stable")
    consensus_indices = list(consensus.head(config.consensus_slots).index)
    selected_indices.extend(consensus_indices)
    selected_set.update(consensus_indices)
    remainder = group[~group.index.isin(selected_set)].sort_values(
        score_order, ascending=score_ascending, kind="stable"
    )
    slots = config.max_candidates - len(selected_indices)
    selected_indices.extend(list(remainder.head(slots).index))
    output = group.loc[selected_indices].copy()
    output["selected_rank"] = np.arange(1, len(output) + 1, dtype=np.int16)
    if len(output) > config.max_candidates or output["candidate_siret"].duplicated().any():
        raise AssertionError("Absolute top-100 contract was violated")
    return output


def score_and_select(
    model: xgb.XGBRanker,
    union: pd.DataFrame,
    config: AdmissionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Score all union candidates and publish strict protected top-100 lists."""
    features = feature_order(config)
    missing_features = sorted(set(features).difference(union.columns))
    if missing_features:
        raise ValueError(f"Union is missing model features: {missing_features}")
    selected_frames: list[pd.DataFrame] = []
    outcomes: list[dict[str, Any]] = []
    scoring_latencies: list[float] = []
    total_latencies: list[float] = []
    for query_id, group in union.groupby("query_id", sort=True):
        started = time.perf_counter()
        scored = group.copy()
        score_started = time.perf_counter()
        scored["ltr_score"] = model.predict(
            scored[features].fillna(0.0).to_numpy(dtype=np.float32)
        ).astype(np.float32)
        scoring_ms = (time.perf_counter() - score_started) * 1000.0
        selected = protected_top100(scored, config)
        total_ms = (time.perf_counter() - started) * 1000.0
        scoring_latencies.append(scoring_ms)
        total_latencies.append(total_ms)
        selected_frames.append(selected)
        exact = normalize_siret(group.iloc[0]["gt_siret"], allow_empty=True)
        historical = normalize_siret(
            group.iloc[0]["historical_gt_siret"], allow_empty=True
        )
        operational = operational_sirets_for_group(group)
        union_sirets = set(group["candidate_siret"].astype(str))
        selected_sirets = set(selected["candidate_siret"].astype(str))
        outcomes.append(
            {
                "query_id": str(query_id),
                "fold": int(group.iloc[0]["fold"]),
                "gt_siret": exact,
                "historical_gt_siret": str(group.iloc[0]["historical_gt_siret"]),
                "identifiable_exact": _as_bool(group.iloc[0]["identifiable_exact"]),
                "v2_exact_available": _as_bool(group.iloc[0]["v2_exact_available"]),
                "v2_exact": _as_bool(group.iloc[0]["v2_exact"]),
                "v3_exact_available": _as_bool(group.iloc[0]["v3_exact_available"]),
                "v3_exact": _as_bool(group.iloc[0]["v3_exact"]),
                "historical_oracle_hit": bool(
                    historical and historical in union_sirets
                ),
                "historical_hit_at_100": bool(
                    historical and historical in selected_sirets
                ),
                "exact_oracle_hit": bool(exact and exact in union_sirets),
                "operational_oracle_hit": bool(operational & union_sirets),
                "exact_hit_at_100": bool(exact and exact in selected_sirets),
                "operational_hit_at_100": bool(operational & selected_sirets),
                "union_candidate_count": int(len(group)),
                "selected_candidate_count": int(len(selected)),
                "retrieval_latency_measured": _as_bool(
                    group.iloc[0]["retrieval_latency_measured"]
                ),
                "retrieval_latency_ms": float(
                    group.iloc[0]["retrieval_latency_ms"]
                ),
                "admission_scoring_ms": scoring_ms,
                "admission_total_ms": total_ms,
                "union_latency_ms": float(group.iloc[0].get("union_latency_ms") or 0.0),
                "ground_truth_state": str(group.iloc[0].get("ground_truth_state") or ""),
                "ground_truth_state_available": _as_bool(
                    group.iloc[0].get("ground_truth_state_available")
                ),
                "pool_size": float(group.iloc[0].get("pool_size", -1.0)),
                "pool_size_available": _as_bool(
                    group.iloc[0].get("pool_size_available")
                ),
                "mega_base_pool": _as_bool(group.iloc[0].get("mega_base_pool")),
                "mega_base_pool_available": _as_bool(
                    group.iloc[0].get("mega_base_pool_available")
                ),
                "unseen_siren": _as_bool(group.iloc[0].get("unseen_siren")),
                "unseen_siren_available": _as_bool(
                    group.iloc[0].get("unseen_siren_available")
                ),
            }
        )
    selected_output = pd.concat(selected_frames, ignore_index=True)
    outcome_frame = pd.DataFrame(outcomes)
    if selected_output.groupby("query_id").size().gt(config.max_candidates).any():
        raise AssertionError("At least one final candidate set exceeds 100")
    measured = outcome_frame["retrieval_latency_measured"].astype(bool)
    measured_end_to_end = (
        outcome_frame.loc[measured, "retrieval_latency_ms"]
        + outcome_frame.loc[measured, "union_latency_ms"]
        + outcome_frame.loc[measured, "admission_total_ms"]
    )
    latency = {
        "query_count": int(len(outcome_frame)),
        "retrieval_measured_query_count": int(measured.sum()),
        "retrieval_latency_complete": bool(measured.all()),
        "retrieval_ms": _latency_summary(
            outcome_frame.loc[measured, "retrieval_latency_ms"].tolist()
        ),
        "scoring_ms": _latency_summary(scoring_latencies),
        "score_and_select_ms": _latency_summary(total_latencies),
        "union_ms": _latency_summary(outcome_frame["union_latency_ms"].tolist()),
        "end_to_end_admission_ms": _latency_summary(
            measured_end_to_end.tolist()
        ),
    }
    return selected_output, outcome_frame, latency


def wilson_interval(successes: int, total: int, confidence: float) -> list[float] | None:
    if total <= 0:
        return None
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def _metric(values: pd.Series) -> dict[str, Any]:
    clean = values.fillna(False).astype(bool)
    successes = int(clean.sum())
    total = int(len(clean))
    return {
        "successes": successes,
        "total": total,
        "rate": successes / total if total else None,
        "wilson_95": wilson_interval(successes, total, 0.95),
        "wilson_99": wilson_interval(successes, total, 0.99),
    }


def _latency_summary(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p95": None, "p99": None, "mean": None, "max": None}
    raw = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.quantile(raw, 0.50)),
        "p95": float(np.quantile(raw, 0.95)),
        "p99": float(np.quantile(raw, 0.99)),
        "mean": float(raw.mean()),
        "max": float(raw.max()),
    }


def evaluate_outcomes(
    outcomes: pd.DataFrame,
    config: AdmissionConfig,
    *,
    integrity: Mapping[str, bool] | None = None,
    latency: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish exact and operational views and a GO/PIVOT/STOP decision."""
    integrity_checks = {
        "nonempty_evaluation": not outcomes.empty,
        "union_cap_at_most_2000": bool(outcomes["union_candidate_count"].le(2000).all()),
        "strict_top100": bool(outcomes["selected_candidate_count"].le(100).all()),
        **dict(integrity or {}),
    }
    identifiable = outcomes["identifiable_exact"].fillna(False).astype(bool)
    coverage = _metric(identifiable)
    exact_oracle = _metric(outcomes.loc[identifiable, "exact_oracle_hit"])
    operational_oracle = _metric(outcomes.loc[identifiable, "operational_oracle_hit"])
    exact_recall = _metric(outcomes.loc[identifiable, "exact_hit_at_100"])
    operational_recall = _metric(outcomes.loc[identifiable, "operational_hit_at_100"])

    historical_hit_column = (
        "historical_hit_at_100"
        if "historical_hit_at_100" in outcomes.columns
        else "exact_hit_at_100"
    )
    historical_oracle_column = (
        "historical_oracle_hit"
        if "historical_oracle_hit" in outcomes.columns
        else "exact_oracle_hit"
    )
    historical_scope = (
        outcomes["historical_gt_siret"].fillna("").astype(str).ne("")
        if "historical_gt_siret" in outcomes.columns
        else identifiable
    )

    def qualification_view(version: str) -> dict[str, Any]:
        available_column = f"{version}_exact_available"
        exact_column = f"{version}_exact"
        if (
            available_column not in outcomes.columns
            or exact_column not in outcomes.columns
            or not outcomes[available_column].fillna(False).astype(bool).all()
        ):
            return {
                "status": "NOT_AVAILABLE",
                "coverage": None,
                "exact_oracle_recall": None,
                "exact_recall_at_100": None,
            }
        scope = outcomes[exact_column].fillna(False).astype(bool)
        return {
            "status": "AVAILABLE",
            "coverage": _metric(scope),
            "exact_oracle_recall": _metric(outcomes.loc[scope, "exact_oracle_hit"]),
            "exact_recall_at_100": _metric(outcomes.loc[scope, "exact_hit_at_100"]),
        }

    qualification_views = {
        "historical": {
            "status": "AVAILABLE",
            "coverage": _metric(historical_scope),
            "exact_oracle_recall": _metric(
                outcomes.loc[historical_scope, historical_oracle_column]
            ),
            "exact_recall_at_100": _metric(
                outcomes.loc[historical_scope, historical_hit_column]
            ),
        },
        "v2": qualification_view("v2"),
        "v3": qualification_view("v3"),
    }
    qualifications_available = all(
        qualification_views[version]["status"] == "AVAILABLE"
        for version in ("v2", "v3")
    )

    latency_payload = dict(latency or {})
    latency_complete = bool(latency_payload.get("retrieval_latency_complete", False))
    end_to_end = latency_payload.get("end_to_end_admission_ms") or {}
    latency_p95 = end_to_end.get("p95")
    latency_p99 = end_to_end.get("p99")
    gates = {
        "identifiable_coverage": {
            "minimum": float(config.gates["minimum_identifiable_coverage"]),
            "observed": coverage["rate"],
            "passed": bool(
                coverage["rate"] is not None
                and coverage["rate"] >= float(config.gates["minimum_identifiable_coverage"])
            ),
        },
        "exact_recall_at_100": {
            "minimum": float(config.gates["minimum_exact_recall_at_100"]),
            "observed": exact_recall["rate"],
            "passed": bool(
                exact_recall["rate"] is not None
                and exact_recall["rate"] >= float(config.gates["minimum_exact_recall_at_100"])
            ),
        },
        "exact_oracle_recall": {
            "minimum": float(config.gates["minimum_exact_oracle_recall"]),
            "observed": exact_oracle["rate"],
            "passed": bool(
                exact_oracle["rate"] is not None
                and exact_oracle["rate"] >= float(config.gates["minimum_exact_oracle_recall"])
            ),
        },
        "candidate_ceiling": {
            "maximum": config.max_candidates,
            "observed": int(outcomes["selected_candidate_count"].max()) if len(outcomes) else None,
            "passed": bool(outcomes["selected_candidate_count"].le(config.max_candidates).all()),
        },
        "qualification_views_available": {
            "required": ["historical", "v2", "v3"],
            "observed": [
                version
                for version, payload in qualification_views.items()
                if payload["status"] == "AVAILABLE"
            ],
            "passed": qualifications_available,
        },
        "latency_measured": {
            "required": True,
            "observed": latency_complete,
            "reason": None if latency_complete else "latency_not_measured",
            "passed": latency_complete,
        },
        "end_to_end_p95_ms": {
            "maximum": float(config.gates["maximum_end_to_end_p95_ms"]),
            "observed": latency_p95,
            "passed": bool(
                latency_complete
                and latency_p95 is not None
                and latency_p95 <= float(config.gates["maximum_end_to_end_p95_ms"])
            ),
        },
        "end_to_end_p99_ms": {
            "maximum": float(config.gates["maximum_end_to_end_p99_ms"]),
            "observed": latency_p99,
            "passed": bool(
                latency_complete
                and latency_p99 is not None
                and latency_p99 <= float(config.gates["maximum_end_to_end_p99_ms"])
            ),
        },
    }
    if not all(integrity_checks.values()):
        verdict = "STOP"
    elif all(value["passed"] for value in gates.values()):
        verdict = "GO"
    else:
        verdict = "PIVOT"
    segments: dict[str, Any] = {}

    def add_segment(name: str, mask: pd.Series) -> None:
        if not mask.any():
            return
        scoped_identifiable = mask & identifiable
        segments[name] = {
            "query_count": int(mask.sum()),
            "identifiable_coverage": _metric(identifiable.loc[mask]),
            "exact_oracle_recall": _metric(
                outcomes.loc[scoped_identifiable, "exact_oracle_hit"]
            ),
            "exact_recall_at_100": _metric(
                outcomes.loc[scoped_identifiable, "exact_hit_at_100"]
            ),
            "operational_recall_at_100": _metric(
                outcomes.loc[scoped_identifiable, "operational_hit_at_100"]
            ),
        }

    if (
        "ground_truth_state" in outcomes.columns
        and outcomes.get("ground_truth_state_available", pd.Series(False, index=outcomes.index))
        .fillna(False)
        .astype(bool)
        .all()
    ):
        states = outcomes["ground_truth_state"].fillna("").astype(str)
        for state in sorted(set(states) - {""}):
            add_segment(f"ground_truth_state={state}", states.eq(state))
    if (
        "mega_base_pool" in outcomes.columns
        and outcomes.get("mega_base_pool_available", pd.Series(False, index=outcomes.index))
        .fillna(False)
        .astype(bool)
        .all()
    ):
        mega = outcomes["mega_base_pool"].fillna(False).astype(bool)
        add_segment("mega_base_pool=true", mega)
        add_segment("mega_base_pool=false", ~mega)
    if (
        "pool_size" in outcomes.columns
        and outcomes.get("pool_size_available", pd.Series(False, index=outcomes.index))
        .fillna(False)
        .astype(bool)
        .all()
    ):
        pool = pd.to_numeric(outcomes["pool_size"], errors="coerce")
        measured_pool = pool.ge(0)
        add_segment("pool_size<1000", measured_pool & pool.lt(1000))
        add_segment("pool_size=1000-9999", measured_pool & pool.ge(1000) & pool.lt(10000))
        add_segment("pool_size>=10000", measured_pool & pool.ge(10000))
    if (
        "unseen_siren" in outcomes.columns
        and outcomes.get("unseen_siren_available", pd.Series(False, index=outcomes.index))
        .fillna(False)
        .astype(bool)
        .all()
    ):
        unseen = outcomes["unseen_siren"].fillna(False).astype(bool)
        add_segment("unseen_siren=true", unseen)
        add_segment("unseen_siren=false", ~unseen)

    return {
        "schema_version": SCHEMA_VERSION,
        "policy_id": config.policy_id,
        "verdict": verdict,
        "metrics": {
            "identifiable_coverage": coverage,
            "exact_oracle_recall": exact_oracle,
            "operational_oracle_recall": operational_oracle,
            "exact_recall_at_100": exact_recall,
            "operational_recall_at_100": operational_recall,
        },
        "qualification_views": qualification_views,
        "segments": segments,
        "latency_ms": latency_payload,
        "gates": gates,
        "integrity": integrity_checks,
        "query_count": int(len(outcomes)),
        "exact_oracle_misses": int((identifiable & ~outcomes["exact_oracle_hit"]).sum()),
        "exact_pruned_by_admission": int(
            (identifiable & outcomes["exact_oracle_hit"] & ~outcomes["exact_hit_at_100"]).sum()
        ),
    }


__all__ = [
    "AdmissionConfig",
    "DEFAULT_CONFIG_PATH",
    "MINIMUM_INPUT_COLUMNS",
    "SCHEMA_VERSION",
    "assert_feature_order_is_identifier_free",
    "build_internal_union",
    "evaluate_outcomes",
    "feature_order",
    "operational_sirets_for_group",
    "prepare_training_rows",
    "protected_top100",
    "score_and_select",
    "train_ranker",
    "validate_candidate_input",
]
