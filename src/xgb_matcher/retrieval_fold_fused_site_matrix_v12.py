"""Fast, exact site-feature matrices for the fold-fused v12 executor.

The historical v12 canonicalizer is intentionally general: it accepts several
physical encodings and returns a wide, audit-friendly pandas frame.  The
fold-fused executor has a narrower hot path.  It already sealed the complete
Master statistics and only needs the frozen 319 float32 model features.

This module keeps the scientific calculation unchanged while avoiding a
2,000 x 319 pandas frame for every held-out scene.  Row-local evidence is
parsed once, complete-pool normalizers come from ``LazySiteStatisticsV12``,
and the final matrix is filled column-wise in the canonical feature order.
"""

from __future__ import annotations

from datetime import date
import json
import math
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .retrieval_hierarchical_admission_v12 import (
    ADDRESS_EVIDENCE_CHANNELS,
    CASCADE_PARENT_FEATURES,
    CURRENT_CHANNELS,
    EXACT_GEO_CHANNELS,
    EXACT_NAME_CHANNELS,
    HISTORICAL_CHANNELS,
    LABEL_COLUMNS,
    NAME_EVIDENCE_CHANNELS,
    PARENT_SCORE_FEATURES,
    RELATION_CHANNELS,
    AdmissionConfigV12,
    _as_bool,
    _candidate_values,
    _first,
    _float,
    _leading_number,
    _parse_iso_date,
    _signals,
    _source_families,
    _values,
    normalize_siret,
    normalize_text,
    reject_label_columns,
    site_feature_names,
)
from .retrieval_v12_lazy_site_features import (
    LazySiteFeatureError,
    LazySiteStatisticsV12,
    _statistics_from_base,
)


SCHEMA_VERSION = "sireto-retrieval-v12-fold-fused-site-matrix-1"
_TRAIN_FOLDS = frozenset({2, 3, 4})
_KNOWN_RELATION_DIRECTIONS = frozenset({"SUCCESSOR", "PREDECESSOR", "BOTH"})


def _normalized_cache() -> tuple[dict[str, str], Any]:
    cache: dict[str, str] = {}

    def cached(value: Any) -> str:
        # Candidate values have already passed through ``_values`` and are
        # strings.  Query/geo/state values may still be None/NaN, for which
        # the canonical normalizer has special semantics.
        if value is None:
            return ""
        try:
            if bool(pd.isna(value)):
                return ""
        except (TypeError, ValueError):
            pass
        key = str(value)
        observed = cache.get(key)
        if observed is None:
            observed = normalize_text(value)
            cache[key] = observed
        return observed

    return cache, cached


def _token_set(value: str) -> set[str]:
    return set(value.split())


def _char_grams(value: str) -> set[str]:
    padded = f" {value.replace(' ', '_')} "
    return {
        padded[index : index + 3]
        for index in range(max(0, len(padded) - 2))
    }


def _maximum_similarity(
    query_tokens: set[str],
    query_grams: set[str],
    candidates: Sequence[str],
) -> tuple[float, float]:
    best_token = 0.0
    best_char = 0.0
    for candidate in candidates:
        candidate_tokens = _token_set(candidate)
        if query_tokens and candidate_tokens:
            union = query_tokens | candidate_tokens
            score = len(query_tokens & candidate_tokens) / len(union)
            if score > best_token:
                best_token = score
        candidate_grams = _char_grams(candidate)
        if query_grams and candidate_grams:
            score = (
                2.0
                * len(query_grams & candidate_grams)
                / (len(query_grams) + len(candidate_grams))
            )
            if score > best_char:
                best_char = score
    return best_token, best_char


def _row_channels(
    raw: Mapping[str, Any],
    signals: Sequence[Mapping[str, Any]],
    configured: tuple[str, ...],
    present_channels: frozenset[str],
) -> frozenset[str]:
    configured_set = frozenset(configured)
    channels = {
        value
        for value in _values(raw.get("channels"))
        if value in configured_set
    }
    channels.update(
        str(signal.get("channel"))
        for signal in signals
        if str(signal.get("channel")) in configured_set
    )
    channels.update(present_channels)
    return frozenset(channels)


def _temporal_values(valid_from: Any, valid_to: Any, reference: date) -> tuple[float, ...]:
    start = _parse_iso_date(str(valid_from or ""))
    end = _parse_iso_date(str(valid_to or ""))
    return (
        float(start is not None),
        float(end is not None),
        float(end is None),
        float((start is None or start <= reference) and (end is None or end >= reference)),
        float(start is not None and start > reference),
        float(end is not None and end < reference),
        float(math.log1p(abs((reference - start).days)) if start is not None else 0.0),
        float(math.log1p(abs((reference - end).days)) if end is not None else 0.0),
    )


def _relation_values(
    signals: Sequence[Mapping[str, Any]], reference: date, normalize: Any
) -> tuple[float, ...]:
    relation_signals = [
        signal for signal in signals if normalize(signal.get("relation_direction"))
    ]
    directions = {
        normalize(signal.get("relation_direction")) for signal in relation_signals
    }
    dates = []
    for signal in relation_signals:
        parsed = _parse_iso_date(
            str(
                signal.get("relation_date")
                or signal.get("event_date")
                or signal.get("date")
                or ""
            )
        )
        if parsed is not None:
            dates.append(parsed)
    nearest_age = min(
        (abs((reference - value).days) for value in dates), default=0
    )
    return (
        float("SUCCESSOR" in directions or "BOTH" in directions),
        float("PREDECESSOR" in directions or "BOTH" in directions),
        float("BOTH" in directions),
        float(bool(directions - _KNOWN_RELATION_DIRECTIONS)),
        float(bool(dates)),
        float(bool(relation_signals) and not dates),
        float(math.log1p(nearest_age)),
    )


def _selected_signals_by_channel(
    signals: Sequence[Mapping[str, Any]], configured: frozenset[str]
) -> dict[str, Mapping[str, Any]]:
    selected: dict[str, Mapping[str, Any]] = {}
    for signal in signals:
        channel = str(signal.get("channel"))
        if channel not in configured:
            continue
        current = selected.get(channel)
        key = (
            int(_float(signal.get("rank"), 2_147_483_647)),
            -_float(signal.get("score"), 0.0),
        )
        if current is None or key < (
            int(_float(current.get("rank"), 2_147_483_647)),
            -_float(current.get("score"), 0.0),
        ):
            selected[channel] = signal
    return selected


def compute_fold_fused_site_statistics_v12(
    raw_rows: Sequence[Mapping[str, Any]], config: AdmissionConfigV12
) -> LazySiteStatisticsV12:
    """Build exact lazy statistics with channel columns in vector blocks.

    This entry point is restricted to the fold-fused executor.  It accepts the
    already materialized Master mappings directly, avoiding a DataFrame ->
    records round-trip and replacing the 40-channel inner Python loop with
    column operations.  ``_statistics_from_base`` remains the single source
    of truth for complete-pool distributions and hashes.
    """

    if not raw_rows:
        raise LazySiteFeatureError("v12 candidate union is empty")
    frame = pd.DataFrame(raw_rows)
    required = {"query_id", "fold", "crm_name", "crm_address"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise LazySiteFeatureError(f"v12 candidate union is missing columns: {missing}")
    reject_label_columns(frame)
    channels = tuple(config.channels)
    configured = frozenset(channels)
    row_count = len(raw_rows)

    query_ids: list[str] = []
    folds = np.empty(row_count, dtype=np.int64)
    candidates: list[str] = []
    parents: list[str] = []
    union_rank = np.empty(row_count, dtype=np.int64)
    hydration_rank = np.full(row_count, np.nan, dtype=np.float64)
    parent_feature_present = np.zeros(row_count, dtype=bool)
    sealed_parent_rank = np.full(row_count, np.nan, dtype=np.float64)
    sealed_parent_score = np.full(row_count, np.nan, dtype=np.float64)
    parent_features = {
        name: np.zeros(row_count, dtype=np.float64)
        for name in CASCADE_PARENT_FEATURES
    }
    signal_values: dict[str, dict[int, tuple[float, float]]] = {}
    raw_by_candidate: dict[tuple[str, str], Mapping[str, Any]] = {}

    for index, raw in enumerate(raw_rows):
        query_id = str(raw["query_id"])
        fold = int(raw["fold"])
        candidate = normalize_siret(_first(raw, ("candidate_siret", "siret")))
        parent = re.sub(
            r"\D",
            "",
            str(_first(raw, ("candidate_parent", "candidate_siren", "siren"))),
        )
        parent = parent.zfill(9) if parent else candidate[:9]
        if parent != candidate[:9]:
            raise LazySiteFeatureError("candidate establishment and parent disagree")
        rank = int(_float(_first(raw, ("master_rank", "union_rank")), 0.0))
        if rank < 1:
            raise LazySiteFeatureError("master/union rank must be positive")
        if "hydration_rank" in raw:
            hydration_value = _float(raw.get("hydration_rank"), 0.0)
            if hydration_value >= 1:
                hydration_rank[index] = int(hydration_value)
        else:
            fallback = int(_float(raw.get("union_rank"), 0.0))
            if 1 <= fallback <= config.prerank_cap:
                hydration_rank[index] = fallback

        names = tuple(_values(raw.get("parent_feature_names")))
        values = raw.get("parent_feature_values")
        if isinstance(values, np.ndarray):
            values = values.tolist()
        if names or values not in (None, [], ()):
            if (
                names != tuple(CASCADE_PARENT_FEATURES)
                or not isinstance(values, (list, tuple))
                or len(values) != len(CASCADE_PARENT_FEATURES)
            ):
                raise LazySiteFeatureError("candidate row parent feature schema/order drift")
            parent_feature_present[index] = True
            for name, value in zip(CASCADE_PARENT_FEATURES, values):
                parent_features[name][index] = _float(value, math.nan)

        signals = _signals(raw.get("signals"))
        for channel, signal in _selected_signals_by_channel(
            signals, configured
        ).items():
            signal_values.setdefault(channel, {})[index] = (
                _float(signal.get("rank"), 0.0),
                _float(signal.get("score"), 0.0),
            )
        query_ids.append(query_id)
        folds[index] = fold
        candidates.append(candidate)
        parents.append(parent)
        union_rank[index] = rank
        sealed_parent_rank[index] = _float(raw.get("parent_rank"), math.nan)
        sealed_parent_score[index] = _float(raw.get("parent_score"), math.nan)
        key = (query_id, candidate)
        if key in raw_by_candidate:
            raise LazySiteFeatureError("v12 union requires unique query/SIRET pairs")
        raw_by_candidate[key] = MappingProxyType(dict(raw))

    block: dict[str, Any] = {
        "query_id": query_ids,
        "fold": folds,
        "candidate_siret": candidates,
        "candidate_parent": parents,
        "human_siret": [""] * row_count,
        "operational_sirets_json": ["[]"] * row_count,
        "union_rank": union_rank,
        "master_rank": union_rank.copy(),
        "heuristic_hydration_rank": hydration_rank,
        "sealed_parent_features_present": parent_feature_present,
        "sealed_parent_rank": sealed_parent_rank,
        "sealed_parent_score": sealed_parent_score,
        **parent_features,
    }

    for channel in channels:
        signal_rank = np.zeros(row_count, dtype=np.float64)
        signal_score = np.zeros(row_count, dtype=np.float64)
        for index, (rank_value, score_value) in signal_values.get(
            channel, {}
        ).items():
            signal_rank[index] = rank_value
            signal_score[index] = score_value
        rank_name = f"{channel}_rank"
        score_name = f"{channel}_score"
        if rank_name in frame:
            wide_rank = pd.to_numeric(frame[rank_name], errors="coerce").to_numpy(
                dtype=np.float64, copy=False
            )
            rank = np.where(np.isfinite(wide_rank), wide_rank, signal_rank)
        else:
            rank = signal_rank
        if score_name in frame:
            wide_score = pd.to_numeric(frame[score_name], errors="coerce").to_numpy(
                dtype=np.float64, copy=False
            )
            score = np.where(np.isfinite(wide_score), wide_score, signal_score)
        else:
            score = signal_score
        present = rank >= 1.0
        # Match the canonical lightweight frame's consolidated integer block;
        # this also keeps parent group slicing cheap when a scene has many
        # two-site parents.
        block[f"channel__{channel}__present"] = present.astype(np.int64)
        block[f"__raw_rank__{channel}"] = np.where(present, rank, np.nan)
        block[f"__raw_score__{channel}"] = np.where(present, score, np.nan)

    base = pd.DataFrame(block, copy=False)
    if base["query_id"].eq("").any():
        raise LazySiteFeatureError("v12 union requires non-empty query IDs")
    if base.duplicated(["query_id", "master_rank"]).any():
        raise LazySiteFeatureError("v12 master ranks must be unique within query")
    hydrated = base[base["heuristic_hydration_rank"].notna()]
    if hydrated.duplicated(["query_id", "heuristic_hydration_rank"]).any():
        raise LazySiteFeatureError("v12 hydration ranks must be unique within query")
    if (
        not hydrated.empty
        and int(hydrated["heuristic_hydration_rank"].max()) > config.prerank_cap
    ):
        raise LazySiteFeatureError("v12 hydration shortlist exceeds PreRank@500")
    counts = base.groupby("query_id", sort=False).size()
    if int(counts.max()) > config.internal_cap:
        raise LazySiteFeatureError("v12 union exceeds 2,000 candidates")
    if base.groupby("query_id", sort=False)["fold"].nunique(dropna=False).gt(1).any():
        raise LazySiteFeatureError("query-level field varies within query: fold")
    observed_folds = frozenset(int(value) for value in folds)
    if not observed_folds or not observed_folds.issubset(_TRAIN_FOLDS):
        raise LazySiteFeatureError("lazy training features refuse fold 0/1")
    return _statistics_from_base(base, MappingProxyType(raw_by_candidate), config)


def materialize_fold_fused_site_matrix_v12(
    statistics: LazySiteStatisticsV12,
    config: AdmissionConfigV12,
    *,
    selected_indices: Sequence[int] | None = None,
) -> np.ndarray:
    """Return exact canonical float32 features for full Master or a fit slice.

    All normalizers and parent summaries still come from the complete sealed
    Master.  ``selected_indices`` therefore changes only the materialized rows,
    never any feature value.
    """

    feature_names = tuple(site_feature_names(config))
    if statistics.feature_names != feature_names:
        raise LazySiteFeatureError("fold-fused matrix/config feature schema drift")
    if len(statistics.query_ids) != 1:
        raise LazySiteFeatureError("fold-fused heldout matrix requires exactly one query")
    complete_base = statistics._base
    complete_raw_rows = statistics._raw_rows
    if selected_indices is None:
        indices = np.arange(len(complete_base), dtype=np.int64)
    else:
        indices = np.asarray(tuple(int(value) for value in selected_indices), dtype=np.int64)
        if (
            indices.ndim != 1
            or len(indices) != len(set(int(value) for value in indices))
            or (len(indices) and (int(indices.min()) < 0 or int(indices.max()) >= len(complete_base)))
        ):
            raise LazySiteFeatureError("fold-fused site matrix selection is invalid")
    base = complete_base.iloc[indices].reset_index(drop=True)
    raw_rows = tuple(complete_raw_rows[int(index)] for index in indices)
    row_count = len(base)
    if row_count != len(raw_rows) or (
        selected_indices is None and row_count != statistics.complete_rows
    ):
        raise LazySiteFeatureError("fold-fused Master payload/statistics cardinality drift")

    reference = _parse_iso_date(config.temporal_reference_date)
    if reference is None:
        raise LazySiteFeatureError("invalid frozen v12 temporal reference date")
    channels = tuple(config.channels)
    families = config.channel_families
    consensus_min = config.consensus_min_families
    _cache, normalize = _normalized_cache()

    values: dict[str, np.ndarray] = {
        name: np.zeros(row_count, dtype=np.float64) for name in feature_names
    }
    query_id = statistics.query_ids[0]
    query_size = float(statistics._query_sizes[query_id])
    union_rank = base["union_rank"].to_numpy(dtype=np.float64, copy=False)
    values["union_rank_reciprocal"] = 1.0 / union_rank
    values["union_rank_log_inverse"] = 1.0 / np.log2(union_rank + 1.0)
    values["union_rank_percentile_inverse"] = 1.0 - (
        (union_rank - 1.0) / max(query_size, 1.0)
    )

    hydration = base["heuristic_hydration_rank"].to_numpy(
        dtype=np.float64, copy=False
    )
    hydration_present = np.isfinite(hydration)
    if hydration_present.any():
        rank = hydration[hydration_present]
        hydration_size = float(statistics._hydration_sizes[query_id])
        values["heuristic_hydration_rank_reciprocal"][hydration_present] = 1.0 / rank
        values["heuristic_hydration_rank_log_inverse"][hydration_present] = 1.0 / np.log2(
            rank + 1.0
        )
        values["heuristic_hydration_rank_percentile_inverse"][hydration_present] = 1.0 - (
            (rank - 1.0) / max(hydration_size, 1.0)
        )

    present_by_row: list[set[str]] = [set() for _ in range(row_count)]
    for channel in channels:
        present = base[f"channel__{channel}__present"].to_numpy(
            dtype=np.int8, copy=False
        ) == 1
        rank = base[f"__raw_rank__{channel}"].to_numpy(
            dtype=np.float64, copy=False
        )
        score = base[f"__raw_score__{channel}"].to_numpy(
            dtype=np.float64, copy=False
        )
        values[f"channel__{channel}__present"] = present.astype(np.float64)
        if present.any():
            selected_rank = rank[present]
            selected_score = score[present]
            summary = statistics._channel_distributions[channel][query_id]
            values[f"channel__{channel}__rank_reciprocal"][present] = 1.0 / selected_rank
            values[f"channel__{channel}__rank_log_inverse"][present] = 1.0 / np.log2(
                selected_rank + 1.0
            )
            left = np.searchsorted(summary.sorted_ranks, selected_rank, side="left")
            right = np.searchsorted(summary.sorted_ranks, selected_rank, side="right")
            percentile = (
                left + (right - left + 1.0) / 2.0
            ) / float(len(summary.sorted_ranks))
            values[f"channel__{channel}__rank_percentile_inverse"][present] = (
                1.0 - percentile + 1.0 / float(len(summary.sorted_ranks))
            )
            values[f"channel__{channel}__score_relative"][present] = (
                selected_score - summary.best_score
            )
            left = np.searchsorted(summary.sorted_scores, selected_score, side="left")
            right = np.searchsorted(summary.sorted_scores, selected_score, side="right")
            values[f"channel__{channel}__score_percentile"][present] = (
                left + (right - left + 1.0) / 2.0
            ) / float(len(summary.sorted_scores))
            for index in np.flatnonzero(present):
                present_by_row[int(index)].add(channel)

    query_name = normalize(raw_rows[0].get("crm_name"))
    query_address = normalize(raw_rows[0].get("crm_address"))
    query_name_tokens = _token_set(query_name)
    query_name_grams = _char_grams(query_name)
    query_address_tokens = _token_set(query_address)
    query_address_grams = _char_grams(query_address)

    for index, raw in enumerate(raw_rows):
        if str(raw.get("query_id")) != query_id:
            raise LazySiteFeatureError("fold-fused raw row query identity drift")
        crm_name = normalize(raw.get("crm_name"))
        crm_address = normalize(raw.get("crm_address"))
        if crm_name != query_name or crm_address != query_address:
            raise LazySiteFeatureError("query-level name/address varies within query")
        crm_number = normalize(_first(raw, ("crm_number", "numeroVoie", "numero")))
        if not crm_number:
            crm_number = _leading_number(query_address)
        crm_postcode = normalize(_first(raw, ("crm_postcode", "postcode", "crm_cp")))
        crm_insee = normalize(_first(raw, ("crm_insee", "insee")))

        names = _candidate_values(
            raw,
            (
                "candidate_names",
                "candidate_names_json",
                "candidate_name",
                "names",
                "denomination",
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
        name_token, name_char = _maximum_similarity(
            query_name_tokens, query_name_grams, names
        )
        address_token, address_char = _maximum_similarity(
            query_address_tokens, query_address_grams, addresses
        )
        values["name_token_jaccard"][index] = name_token
        values["name_char_dice"][index] = name_char
        values["address_token_jaccard"][index] = address_token
        values["address_char_dice"][index] = address_char
        values["name_address_joint"][index] = math.sqrt(
            max(0.0, name_char) * max(0.0, address_char)
        )

        candidate_insee = normalize(
            _first(raw, ("candidate_insee", "codeCommuneEtablissement"))
        )
        candidate_postcode = normalize(
            _first(raw, ("candidate_postcode", "codePostalEtablissement"))
        )
        candidate_number = normalize(
            _first(raw, ("candidate_number", "numeroVoieEtablissement"))
        )
        if not candidate_number and addresses:
            candidate_number = _leading_number(addresses[0])

        signals = _signals(raw.get("signals"))
        row_channels = _row_channels(
            raw, signals, channels, frozenset(present_by_row[index])
        )
        exact_name = bool(row_channels & EXACT_NAME_CHANNELS)
        geo_match = bool(row_channels & EXACT_GEO_CHANNELS) or (
            bool(crm_insee) and candidate_insee == crm_insee
        ) or (bool(crm_postcode) and candidate_postcode == crm_postcode)
        row_families = {
            family
            for family, members in families.items()
            if row_channels.intersection(members)
        }
        values["street_number_match"][index] = float(
            bool(crm_number) and crm_number == candidate_number
        )
        values["postcode_match"][index] = float(
            bool(crm_postcode) and crm_postcode == candidate_postcode
        )
        values["insee_match"][index] = float(
            bool(crm_insee) and crm_insee == candidate_insee
        )
        values["exact_name_present"][index] = float(exact_name)
        values["exact_geo_present"][index] = float(geo_match)
        values["joint_name_geo"][index] = float(exact_name and geo_match)
        values["evidence_family_count"][index] = float(len(row_families))
        values["independent_consensus"][index] = float(
            len(row_families) >= consensus_min
        )
        for family, members in families.items():
            values[f"family__{family}__present"][index] = float(
                bool(row_channels.intersection(members))
            )

        evidence_sources = {
            normalize(value)
            for value in _values(raw.get("official_evidence_sources"))
            if normalize(value)
        }
        evidence_sources.update(
            normalize(value)
            for value in _values(raw.get("sources"))
            if normalize(value)
        )
        evidence_sources.update(
            normalize(signal.get("source"))
            for signal in signals
            if normalize(signal.get("source"))
        )
        source_families = _source_families(evidence_sources)
        values["official_source_count_log"][index] = math.log1p(
            len(source_families)
        )
        values["site_source_registry_present"][index] = float(
            "SIRENE" in source_families
        )
        values["site_source_rne_present"][index] = float("RNE" in source_families)
        values["site_source_bodacc_present"][index] = float(
            "BODACC" in source_families
        )
        values["site_source_sector_present"][index] = float(
            "SECTOR" in source_families
        )
        values["current_signal_count_log"][index] = math.log1p(
            len(row_channels & CURRENT_CHANNELS)
        )
        values["historical_signal_count_log"][index] = math.log1p(
            len(row_channels & HISTORICAL_CHANNELS)
        )
        values["relation_signal_count_log"][index] = math.log1p(
            len(row_channels & RELATION_CHANNELS)
        )

        state = normalize(
            _first(raw, ("candidate_state", "state", "etatAdministratifEtablissement"))
        )
        values["candidate_active"][index] = float(state == "A")
        values["candidate_closed"][index] = float(state == "F")
        values["candidate_state_unknown"][index] = float(state not in {"A", "F"})
        values["candidate_is_siege"][index] = float(
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
        )

        temporal = _temporal_values(
            raw.get("temporal_valid_from"), raw.get("temporal_valid_to"), reference
        )
        for name, value in zip(
            (
                "temporal_valid_from_present",
                "temporal_valid_to_present",
                "temporal_open_ended",
                "temporal_valid_at_reference",
                "temporal_starts_after_reference",
                "temporal_ended_before_reference",
                "temporal_start_distance_days_log1p",
                "temporal_end_distance_days_log1p",
            ),
            temporal,
        ):
            values[name][index] = value
        relation = _relation_values(signals, reference, normalize)
        for name, value in zip(
            (
                "relation_successor_present",
                "relation_predecessor_present",
                "relation_both_present",
                "relation_direction_other_present",
                "relation_dated_present",
                "relation_date_missing",
                "relation_nearest_age_days_log1p",
            ),
            relation,
        ):
            values[name][index] = value

    parent_values = base[list(CASCADE_PARENT_FEATURES)].to_numpy(
        dtype=np.float64, copy=False
    )
    for column, name in enumerate(CASCADE_PARENT_FEATURES):
        values[name] = parent_values[:, column]

    parents = base["candidate_parent"].astype(str).to_numpy(copy=False)
    for index, (parent, row_rank) in enumerate(zip(parents, union_rank)):
        key = (query_id, str(parent))
        parent_ranks = statistics._parent_union_ranks[key]
        within_rank = np.searchsorted(parent_ranks, row_rank, side="left") + 1
        values["within_parent_rank_reciprocal"][index] = 1.0 / float(within_rank)
        values["within_parent_candidate_count_log"][index] = math.log1p(
            statistics._parent_sizes[key]
        )

    parent_score_by_parent = {
        str(row.candidate_parent): (
            float(row.parent_rank_reciprocal),
            float(row.parent_score_probability),
            float(row.parent_score_gap_to_best),
        )
        for row in statistics._parent_scores.itertuples(index=False)
    }
    for index, parent in enumerate(parents):
        score_values = parent_score_by_parent.get(str(parent))
        if score_values is None:
            values["parent_preexpansion_missing"][index] = 1.0
            continue
        for name, value in zip(PARENT_SCORE_FEATURES, score_values):
            values[name][index] = value

    matrix = np.empty((row_count, len(feature_names)), dtype=np.float32)
    for column, name in enumerate(feature_names):
        matrix[:, column] = values[name]
    if not np.isfinite(matrix).all():
        # Canonical ``fillna(0)`` also replaces NaN, but infinity would be a
        # scientific drift.  NaNs are converted explicitly before this gate.
        matrix[np.isnan(matrix)] = 0.0
    if not np.isfinite(matrix).all():
        raise LazySiteFeatureError("fold-fused site matrix contains infinity")
    return np.ascontiguousarray(matrix, dtype=np.float32)


__all__ = [
    "SCHEMA_VERSION",
    "compute_fold_fused_site_statistics_v12",
    "materialize_fold_fused_site_matrix_v12",
]
