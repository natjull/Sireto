"""Exact lazy training features for the fold-fused retrieval-v12 executor.

The canonical v12 implementation deliberately computes every normalized site
feature on the complete Master<=2,000 union before fit-only sampling.  That is
the right scientific contract, but materialising a dense 319-column matrix for
rows that the frozen sampler immediately discards is unnecessary.

This module preserves the contract in two explicit phases:

``compute_lazy_site_statistics``
    parses the complete, label-blind Master and seals all sufficient statistics
    on the complete population.  It does *not* build the 319-column float32
    matrix;

``materialize_training_sample``
    attaches sealed human labels, applies the byte-for-byte-equivalent frozen
    fit sampler, and materialises the exact 319 float32 features only for the
    selected rows (at most 128 per query in the preregistered configuration).

The held-out/validation path must keep using ``canonicalize_union`` on the full
Master.  This adapter is intentionally training-only and fail-closes on folds
0/1, missing labels, candidate injection, or schema drift.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .retrieval_hierarchical_admission_v12 import (
    AdmissionConfigV12,
    CASCADE_PARENT_FEATURES,
    LABEL_COLUMNS,
    PARENT_SCORE_FEATURES,
    _first,
    _float,
    _human_label,
    _operational_labels,
    _raw_candidate_records,
    _signals,
    _values,
    normalize_siret,
    reject_label_columns,
    site_feature_names,
)


SCHEMA_VERSION = "sireto-retrieval-v12-lazy-site-features-1"
FROZEN_RULE_ID = "V12_SAMPLE_PRECOMPUTED_CANDIDATES_FROZEN"
TRAIN_FOLDS = frozenset({2, 3, 4})


class LazySiteFeatureError(ValueError):
    """Raised when lazy materialisation cannot preserve the v12 contract."""


@dataclass(frozen=True)
class HumanSiteLabelV12:
    """Label-side payload attached only after complete-Master statistics."""

    query_id: str
    human_siret: str
    operational_sirets: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        query_id = str(self.query_id)
        human = normalize_siret(self.human_siret)
        operational = {human}
        for value in self.operational_sirets:
            operational.add(normalize_siret(value))
        object.__setattr__(self, "query_id", query_id)
        object.__setattr__(self, "human_siret", human)
        object.__setattr__(self, "operational_sirets", tuple(sorted(operational)))


@dataclass(frozen=True)
class _ChannelDistribution:
    """Complete-union sufficient statistics for one query/channel."""

    sorted_ranks: np.ndarray
    sorted_scores: np.ndarray
    best_score: float


@dataclass(frozen=True)
class LazySiteStatisticsV12:
    """Opaque complete-Master state consumed by the training materializer.

    ``base`` contains parsed static/raw columns, never the full normalized
    319-feature float matrix.  Private summaries retain only cardinalities,
    empirical channel distributions, and one row per parent.
    """

    schema_version: str
    rule_id: str
    query_ids: tuple[str, ...]
    folds: tuple[int, ...]
    complete_rows: int
    maximum_rows_per_query: int
    candidate_order_sha256: str
    statistics_sha256: str
    feature_names: tuple[str, ...]
    _base: pd.DataFrame
    _query_sizes: Mapping[str, int]
    _hydration_sizes: Mapping[str, int]
    _channel_distributions: Mapping[str, Mapping[str, _ChannelDistribution]]
    _parent_sizes: Mapping[tuple[str, str], int]
    _parent_union_ranks: Mapping[tuple[str, str], np.ndarray]
    _parent_scores: pd.DataFrame
    _raw_rows: tuple[Mapping[str, Any], ...]

    @property
    def base_column_count(self) -> int:
        """Number of static/raw columns retained for the complete Master."""

        return int(len(self._base.columns))

    @property
    def dense_complete_feature_cells(self) -> int:
        """Dense 319-feature cells materialised before label attachment."""

        return 0


@dataclass(frozen=True)
class LazyTrainingSampleV12:
    """The exact fit projection plus auditable materialisation accounting."""

    frame: pd.DataFrame
    feature_names: tuple[str, ...]
    complete_rows: int
    selected_rows: int
    query_count: int
    maximum_selected_per_query: int
    materialized_feature_cells: int
    avoided_dense_feature_cells: int
    source_statistics_sha256: str
    rule_id: str = FROZEN_RULE_ID


def _json_default(value: Any) -> Any:
    """Fast JSON fallback for numpy and read-only row payloads."""

    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, set):
        return sorted(value, key=str)
    return str(value)


def _sha256_base(
    base: pd.DataFrame,
    raw_rows: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
) -> str:
    """Seal light statistics and all deferred row-local source values."""

    digest = hashlib.sha256()
    digest.update(SCHEMA_VERSION.encode("ascii"))
    digest.update(json.dumps(list(feature_names), separators=(",", ":")).encode("utf-8"))
    columns = tuple(sorted(str(column) for column in base.columns))
    digest.update(json.dumps(columns, separators=(",", ":")).encode("utf-8"))
    # ``hash_pandas_object`` streams a fixed-width digest per row and avoids a
    # second JSON/Arrow copy of a potentially 2,000-row Master.
    row_hashes = pd.util.hash_pandas_object(
        base.loc[:, list(columns)], index=False, categorize=False
    ).to_numpy(dtype=np.uint64, copy=False)
    digest.update(row_hashes.tobytes(order="C"))
    for row in raw_rows:
        digest.update(
            json.dumps(
                row,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                # Python's stable NaN/Infinity literals preserve the physical
                # payload without an expensive recursive normalization pass.
                allow_nan=True,
                default=_json_default,
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _candidate_order_sha256(base: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in base[["query_id", "candidate_siret", "candidate_parent", "union_rank"]].itertuples(
        index=False, name=None
    ):
        digest.update(
            json.dumps(row, separators=(",", ":"), ensure_ascii=True).encode("ascii")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _lightweight_candidate_records(
    frame: pd.DataFrame, config: AdmissionConfigV12
) -> tuple[pd.DataFrame, Mapping[tuple[str, str], Mapping[str, Any]]]:
    """Parse only complete-population fields needed before fit sampling.

    Channel precedence deliberately mirrors ``_raw_candidate_records``:
    a finite wide rank/score wins, otherwise the best structured signal is
    used.  Names, addresses, temporal data, provenance and relation features
    remain untouched in the raw payload until their row survives sampling.
    """

    required = {"query_id", "fold", "crm_name", "crm_address"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise LazySiteFeatureError(
            f"v12 candidate union is missing columns: {missing}"
        )
    if frame.empty:
        raise LazySiteFeatureError("v12 candidate union is empty")
    reject_label_columns(frame)

    records: list[dict[str, Any]] = []
    raw_by_candidate: dict[tuple[str, str], Mapping[str, Any]] = {}
    for raw in frame.to_dict("records"):
        query_id = str(raw["query_id"])
        candidate = normalize_siret(_first(raw, ("candidate_siret", "siret")))
        parent = re.sub(
            r"\D",
            "",
            str(_first(raw, ("candidate_parent", "candidate_siren", "siren"))),
        )
        parent = parent.zfill(9) if parent else candidate[:9]
        if parent != candidate[:9]:
            raise LazySiteFeatureError(
                "candidate establishment and parent disagree"
            )
        master_rank = int(
            _float(_first(raw, ("master_rank", "union_rank")), 0.0)
        )
        if master_rank < 1:
            raise LazySiteFeatureError("master/union rank must be positive")
        if "hydration_rank" in raw:
            hydration_value = _float(raw.get("hydration_rank"), 0.0)
            hydration_rank: int | None = (
                int(hydration_value) if hydration_value >= 1 else None
            )
        else:
            fallback_rank = int(_float(raw.get("union_rank"), 0.0))
            hydration_rank = (
                fallback_rank
                if 1 <= fallback_rank <= config.prerank_cap
                else None
            )

        parent_names = tuple(_values(raw.get("parent_feature_names")))
        parent_values = raw.get("parent_feature_values")
        if isinstance(parent_values, np.ndarray):
            parent_values = parent_values.tolist()
        if parent_names or parent_values not in (None, [], ()):
            if (
                parent_names != tuple(CASCADE_PARENT_FEATURES)
                or not isinstance(parent_values, (list, tuple))
                or len(parent_values) != len(CASCADE_PARENT_FEATURES)
            ):
                raise LazySiteFeatureError(
                    "candidate row parent feature schema/order drift"
                )
            sealed_parent_features = {
                name: _float(value, math.nan)
                for name, value in zip(parent_names, parent_values)
            }
        else:
            sealed_parent_features = {
                name: 0.0 for name in CASCADE_PARENT_FEATURES
            }

        signals = _signals(raw.get("signals"))
        record: dict[str, Any] = {
            "query_id": query_id,
            "fold": int(raw["fold"]),
            "candidate_siret": candidate,
            "candidate_parent": parent,
            "human_siret": "",
            "operational_sirets_json": "[]",
            "union_rank": master_rank,
            "master_rank": master_rank,
            "heuristic_hydration_rank": hydration_rank,
            "sealed_parent_features_present": bool(parent_names),
            "sealed_parent_rank": _float(raw.get("parent_rank"), math.nan),
            "sealed_parent_score": _float(raw.get("parent_score"), math.nan),
            **sealed_parent_features,
        }
        for channel in config.channels:
            channel_signals = [
                signal
                for signal in signals
                if str(signal.get("channel")) == channel
            ]
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
                _float(selected_signal.get("rank"), 0.0)
                if selected_signal
                else 0.0,
            )
            score = _float(
                raw.get(f"{channel}_score"),
                _float(selected_signal.get("score"), 0.0)
                if selected_signal
                else 0.0,
            )
            present = rank >= 1
            record[f"channel__{channel}__present"] = int(present)
            record[f"__raw_rank__{channel}"] = rank if present else np.nan
            record[f"__raw_score__{channel}"] = score if present else np.nan
        key = (query_id, candidate)
        if key in raw_by_candidate:
            raise LazySiteFeatureError(
                "v12 union requires unique query/SIRET pairs"
            )
        records.append(record)
        raw_by_candidate[key] = MappingProxyType(dict(raw))

    output = pd.DataFrame(records)
    if output["query_id"].eq("").any():
        raise LazySiteFeatureError("v12 union requires non-empty query IDs")
    if output.duplicated(["query_id", "master_rank"]).any():
        raise LazySiteFeatureError("v12 master ranks must be unique within query")
    hydrated = output[output["heuristic_hydration_rank"].notna()]
    if hydrated.duplicated(["query_id", "heuristic_hydration_rank"]).any():
        raise LazySiteFeatureError("v12 hydration ranks must be unique within query")
    if (
        not hydrated.empty
        and int(hydrated["heuristic_hydration_rank"].max())
        > config.prerank_cap
    ):
        raise LazySiteFeatureError("v12 hydration shortlist exceeds PreRank@500")
    counts = output.groupby("query_id").size()
    if int(counts.max()) > config.internal_cap:
        raise LazySiteFeatureError("v12 union exceeds 2,000 candidates")
    if output.groupby("query_id")["fold"].nunique(dropna=False).gt(1).any():
        raise LazySiteFeatureError("query-level field varies within query: fold")
    return output, MappingProxyType(raw_by_candidate)


def _channel_summaries(
    base: pd.DataFrame, config: AdmissionConfigV12
) -> Mapping[str, Mapping[str, _ChannelDistribution]]:
    output: dict[str, Mapping[str, _ChannelDistribution]] = {}
    for channel in config.channels:
        present = base[f"channel__{channel}__present"].eq(1)
        per_query: dict[str, _ChannelDistribution] = {}
        if present.any():
            columns = ["query_id", f"__raw_rank__{channel}", f"__raw_score__{channel}"]
            for query_id, group in base.loc[present, columns].groupby(
                "query_id", sort=True
            ):
                ranks = np.sort(
                    group[f"__raw_rank__{channel}"].to_numpy(dtype=np.float64),
                    kind="stable",
                )
                scores = np.sort(
                    group[f"__raw_score__{channel}"].to_numpy(dtype=np.float64),
                    kind="stable",
                )
                ranks.setflags(write=False)
                scores.setflags(write=False)
                per_query[str(query_id)] = _ChannelDistribution(
                    sorted_ranks=ranks,
                    sorted_scores=scores,
                    best_score=float(scores[-1]),
                )
        output[channel] = MappingProxyType(per_query)
    return MappingProxyType(output)


def _parent_summaries(
    base_original_order: pd.DataFrame,
) -> tuple[
    Mapping[tuple[str, str], int],
    Mapping[tuple[str, str], np.ndarray],
    pd.DataFrame,
]:
    parent_columns = [
        "query_id",
        "candidate_parent",
        "sealed_parent_features_present",
        "sealed_parent_rank",
        "sealed_parent_score",
        *CASCADE_PARENT_FEATURES,
    ]
    grouped = base_original_order.groupby(["query_id", "candidate_parent"])
    for column in parent_columns[2:]:
        if grouped[column].nunique(dropna=False).gt(1).any():
            raise LazySiteFeatureError(
                f"sealed parent field varies within parent: {column}"
            )

    sizes: dict[tuple[str, str], int] = {}
    union_ranks: dict[tuple[str, str], np.ndarray] = {}
    for (query_id, parent), group in grouped:
        key = (str(query_id), str(parent))
        ranks = np.sort(group["union_rank"].to_numpy(dtype=np.float64), kind="stable")
        ranks.setflags(write=False)
        sizes[key] = int(len(group))
        union_ranks[key] = ranks

    # Deliberately mirror the canonical pandas operations and original parent
    # encounter order.  This keeps float32 parity even in extreme score ranges.
    parents = base_original_order[parent_columns].drop_duplicates(
        ["query_id", "candidate_parent"]
    ).copy()
    scored = parents[
        parents["sealed_parent_rank"].notna()
        & parents["sealed_parent_score"].notna()
    ].copy()
    if (
        (scored["sealed_parent_rank"] < 1).any()
        or scored.duplicated(["query_id", "sealed_parent_rank"]).any()
    ):
        raise LazySiteFeatureError("sealed cascade parent score/rank is invalid")
    scored["parent_rank_reciprocal"] = 1.0 / scored["sealed_parent_rank"]
    best = scored.groupby("query_id")["sealed_parent_score"].transform("max")
    scored["parent_score_gap_to_best"] = scored["sealed_parent_score"] - best
    parent_exp = np.exp(
        (scored["sealed_parent_score"] - best).clip(lower=-30.0, upper=0.0)
    )
    scored["parent_score_probability"] = parent_exp / parent_exp.groupby(
        scored["query_id"]
    ).transform("sum")
    parent_scores = scored[
        [
            "query_id",
            "candidate_parent",
            "parent_rank_reciprocal",
            "parent_score_probability",
            "parent_score_gap_to_best",
        ]
    ].copy()
    return (
        MappingProxyType(sizes),
        MappingProxyType(union_ranks),
        parent_scores,
    )


def _statistics_from_base(
    base_original_order: pd.DataFrame,
    raw_by_candidate: Mapping[tuple[str, str], Mapping[str, Any]],
    config: AdmissionConfigV12,
) -> LazySiteStatisticsV12:
    if base_original_order.empty:
        raise LazySiteFeatureError("lazy site statistics require a non-empty Master")
    observed_folds = frozenset(int(value) for value in base_original_order["fold"].unique())
    if not observed_folds or not observed_folds.issubset(TRAIN_FOLDS):
        raise LazySiteFeatureError("lazy training features refuse fold 0/1")

    feature_names = tuple(site_feature_names(config))
    # This is the exact ordering returned by canonicalize_union and consumed by
    # sample_precomputed_candidates.  Statistics that are order-sensitive were
    # computed above/on the original parse order.
    base = base_original_order.sort_values(
        ["query_id", "union_rank", "candidate_siret"], kind="stable"
    ).reset_index(drop=True)
    raw_rows = tuple(
        raw_by_candidate[(str(query_id), str(candidate))]
        for query_id, candidate in base[
            ["query_id", "candidate_siret"]
        ].itertuples(index=False, name=None)
    )
    query_sizes = {
        str(query_id): int(size)
        for query_id, size in base.groupby("query_id", sort=True).size().items()
    }
    hydration_sizes = {
        str(query_id): int(size)
        for query_id, size in base.loc[
            base["heuristic_hydration_rank"].notna()
        ].groupby("query_id", sort=True).size().items()
    }
    parent_sizes, parent_ranks, parent_scores = _parent_summaries(
        base_original_order
    )
    channels = _channel_summaries(base, config)
    candidate_hash = _candidate_order_sha256(base)
    statistics_hash = _sha256_base(base, raw_rows, feature_names)
    return LazySiteStatisticsV12(
        schema_version=SCHEMA_VERSION,
        rule_id=FROZEN_RULE_ID,
        query_ids=tuple(sorted(query_sizes)),
        folds=tuple(sorted(observed_folds)),
        complete_rows=int(len(base)),
        maximum_rows_per_query=max(query_sizes.values()),
        candidate_order_sha256=candidate_hash,
        statistics_sha256=statistics_hash,
        feature_names=feature_names,
        _base=base,
        _query_sizes=MappingProxyType(query_sizes),
        _hydration_sizes=MappingProxyType(hydration_sizes),
        _channel_distributions=channels,
        _parent_sizes=parent_sizes,
        _parent_union_ranks=parent_ranks,
        _parent_scores=parent_scores,
        _raw_rows=raw_rows,
    )


def compute_lazy_site_statistics(
    raw_master: pd.DataFrame, config: AdmissionConfigV12
) -> LazySiteStatisticsV12:
    """Parse and summarize a complete label-blind training Master.

    This is the method intended for ``FoldFusedContextAdapterV12``.  Passing a
    label column fails inside the canonical parser before any statistic exists.
    """

    base, raw_by_candidate = _lightweight_candidate_records(raw_master, config)
    return _statistics_from_base(base, raw_by_candidate, config)


def _labels_by_query(
    statistics: LazySiteStatisticsV12,
    labels: Iterable[HumanSiteLabelV12],
) -> Mapping[str, HumanSiteLabelV12]:
    output: dict[str, HumanSiteLabelV12] = {}
    for label in labels:
        if label.query_id in output:
            raise LazySiteFeatureError(f"duplicate human label: {label.query_id}")
        output[label.query_id] = label
    expected = set(statistics.query_ids)
    observed = set(output)
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise LazySiteFeatureError(
            f"label/Master query mismatch; missing={missing}, unexpected={unexpected}"
        )
    return MappingProxyType(output)


def _select_fit_indices(
    base: pd.DataFrame,
    labels: Mapping[str, HumanSiteLabelV12],
    config: AdmissionConfigV12,
) -> tuple[int, ...]:
    """Exact index-only port of ``sample_precomputed_candidates``."""

    sampling = config.fit_candidate_sampling
    cap = int(sampling["maximum_candidates_per_query"])
    union_head = int(sampling["union_head"])
    channel_head = int(sampling["channel_head"])
    pieces: list[int] = []
    for query_id, group in base.groupby("query_id", sort=True):
        if len(group) <= cap:
            pieces.extend(int(value) for value in group.index)
            continue
        eligible = group[group["heuristic_hydration_rank"].notna()]
        selected: set[int] = set(
            int(value)
            for value in eligible.nsmallest(
                union_head, "heuristic_hydration_rank"
            ).index
        )
        for channel in config.channels:
            column = f"__raw_rank__{channel}"
            selected.update(
                int(value)
                for value in group[group[column].notna()].nsmallest(
                    channel_head, column
                ).index
            )
        human = labels[str(query_id)].human_siret
        positives = set(
            int(value)
            for value in group.index[group["candidate_siret"].eq(human)]
        )
        # This retains a natural positive.  It never adds a candidate absent
        # from Master, and absence remains visible to the training caller.
        selected.update(positives)
        if len(selected) < cap:
            remainder = group.drop(index=list(selected), errors="ignore")
            remainder = remainder.assign(
                __eligible_priority=remainder[
                    "heuristic_hydration_rank"
                ].notna(),
                __prerank_priority=remainder[
                    "heuristic_hydration_rank"
                ].fillna(config.prerank_cap + remainder["union_rank"]),
            ).sort_values(
                ["__eligible_priority", "__prerank_priority", "union_rank"],
                ascending=[False, True, True],
                kind="stable",
            )
            selected.update(
                int(value) for value in remainder.head(cap - len(selected)).index
            )
        selected_frame = group.loc[list(selected)].copy()
        selected_frame["__positive_priority"] = selected_frame.index.isin(
            positives
        )
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
        ).head(cap)
        selected_frame = selected_frame.sort_values(
            ["union_rank", "candidate_siret"], kind="stable"
        )
        pieces.extend(int(value) for value in selected_frame.index)
    return tuple(pieces)


def _average_percentile(sorted_values: np.ndarray, value: float) -> float:
    """Equivalent to pandas ``rank(method='average', pct=True)``."""

    left = int(np.searchsorted(sorted_values, value, side="left"))
    right = int(np.searchsorted(sorted_values, value, side="right"))
    # one-based average rank of the equal interval, divided by population
    return (left + (right - left + 1.0) / 2.0) / float(len(sorted_values))


def _materialize_selected_features(
    statistics: LazySiteStatisticsV12,
    selected_indices: Sequence[int],
    labels: Mapping[str, HumanSiteLabelV12],
    config: AdmissionConfigV12,
) -> pd.DataFrame:
    base = statistics._base
    expected = base.loc[
        list(selected_indices), ["query_id", "candidate_siret", "candidate_parent"]
    ].reset_index(drop=True)
    output = _raw_candidate_records(
        pd.DataFrame([statistics._raw_rows[int(index)] for index in selected_indices]),
        config,
        labels_allowed=False,
    )
    observed = output[
        ["query_id", "candidate_siret", "candidate_parent"]
    ].reset_index(drop=True)
    if not observed.equals(expected):
        raise LazySiteFeatureError(
            "deferred local feature rows differ from the sealed Master order"
        )
    output["human_siret"] = output["query_id"].map(
        {query_id: label.human_siret for query_id, label in labels.items()}
    )
    output["operational_sirets_json"] = output["query_id"].map(
        {
            query_id: json.dumps(label.operational_sirets)
            for query_id, label in labels.items()
        }
    )

    derived: dict[str, Any] = {}
    query_sizes = output["query_id"].map(statistics._query_sizes).astype(float)
    union_rank = output["union_rank"].astype(float)
    derived["union_rank_reciprocal"] = 1.0 / union_rank
    derived["union_rank_log_inverse"] = 1.0 / np.log2(union_rank + 1.0)
    derived["union_rank_percentile_inverse"] = 1.0 - (
        (union_rank - 1.0) / query_sizes.clip(lower=1)
    )

    hydration_present = output["heuristic_hydration_rank"].notna()
    hydration_reciprocal = pd.Series(0.0, index=output.index)
    hydration_log_inverse = pd.Series(0.0, index=output.index)
    hydration_percentile_inverse = pd.Series(0.0, index=output.index)
    if hydration_present.any():
        hydration_rank = output.loc[
            hydration_present, "heuristic_hydration_rank"
        ].astype(float)
        hydration_size = output.loc[hydration_present, "query_id"].map(
            statistics._hydration_sizes
        ).astype(float)
        hydration_reciprocal.loc[hydration_present] = 1.0 / hydration_rank
        hydration_log_inverse.loc[hydration_present] = 1.0 / np.log2(
            hydration_rank + 1.0
        )
        hydration_percentile_inverse.loc[hydration_present] = 1.0 - (
            (hydration_rank - 1.0) / hydration_size.clip(lower=1)
        )
    derived["heuristic_hydration_rank_reciprocal"] = hydration_reciprocal
    derived["heuristic_hydration_rank_log_inverse"] = hydration_log_inverse
    derived[
        "heuristic_hydration_rank_percentile_inverse"
    ] = hydration_percentile_inverse

    for channel in config.channels:
        present_column = f"channel__{channel}__present"
        rank_column = f"__raw_rank__{channel}"
        score_column = f"__raw_score__{channel}"
        present = output[present_column].eq(1)
        rank = output[rank_column]
        derived[f"channel__{channel}__rank_reciprocal"] = np.where(
            present, 1.0 / rank, 0.0
        )
        derived[f"channel__{channel}__rank_log_inverse"] = np.where(
            present, 1.0 / np.log2(rank + 1.0), 0.0
        )
        rank_percentile = np.zeros(len(output), dtype=np.float64)
        score_relative = np.zeros(len(output), dtype=np.float64)
        score_percentile = np.zeros(len(output), dtype=np.float64)
        if present.any():
            for position, (query_id, raw_rank, raw_score, is_present) in enumerate(
                zip(
                    output["query_id"],
                    output[rank_column],
                    output[score_column],
                    present,
                )
            ):
                if not is_present:
                    continue
                summary = statistics._channel_distributions[channel][str(query_id)]
                rank_pct = _average_percentile(summary.sorted_ranks, float(raw_rank))
                rank_percentile[position] = (
                    1.0
                    - rank_pct
                    + 1.0 / float(len(summary.sorted_ranks))
                )
                score_relative[position] = float(raw_score) - summary.best_score
                score_percentile[position] = _average_percentile(
                    summary.sorted_scores, float(raw_score)
                )
        derived[f"channel__{channel}__rank_percentile_inverse"] = rank_percentile
        derived[f"channel__{channel}__score_relative"] = score_relative
        derived[f"channel__{channel}__score_percentile"] = score_percentile

    derived["name_address_joint"] = np.sqrt(
        output["name_char_dice"].clip(lower=0.0)
        * output["address_char_dice"].clip(lower=0.0)
    )
    within_parent_rank: list[float] = []
    parent_count_log: list[float] = []
    for query_id, parent, row_union_rank in output[
        ["query_id", "candidate_parent", "union_rank"]
    ].itertuples(index=False, name=None):
        key = (str(query_id), str(parent))
        ranks = statistics._parent_union_ranks[key]
        within_parent_rank.append(
            float(np.searchsorted(ranks, float(row_union_rank), side="left") + 1)
        )
        parent_count_log.append(math.log1p(statistics._parent_sizes[key]))
    derived["within_parent_rank_reciprocal"] = 1.0 / np.asarray(
        within_parent_rank, dtype=np.float64
    )
    derived["within_parent_candidate_count_log"] = np.asarray(
        parent_count_log, dtype=np.float64
    )
    # Build normalized columns in one block.  Besides avoiding pandas
    # fragmentation, this is the actual lazy boundary: the block has exactly
    # ``selected_rows`` rows, never ``complete_master_rows`` rows.
    output = pd.concat(
        [output, pd.DataFrame(derived, index=output.index)], axis=1
    )

    output = output.merge(
        statistics._parent_scores,
        on=["query_id", "candidate_parent"],
        how="left",
        validate="many_to_one",
    )
    output["parent_preexpansion_missing"] = output[
        "parent_rank_reciprocal"
    ].isna().astype(np.int8)
    for column in PARENT_SCORE_FEATURES:
        output[column] = output[column].fillna(0.0)
    output["features_computed_on_complete_union"] = True

    feature_names = list(statistics.feature_names)
    missing = sorted(set(feature_names) - set(output.columns))
    if missing:
        raise LazySiteFeatureError(f"lazy v12 feature schema is incomplete: {missing}")
    output[feature_names] = output[feature_names].fillna(0.0).astype(np.float32)
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
                *feature_names,
            ]
        )
    )
    return output[fit_columns].reset_index(drop=True)


def materialize_training_sample(
    statistics: LazySiteStatisticsV12,
    labels: Iterable[HumanSiteLabelV12],
    config: AdmissionConfigV12,
) -> LazyTrainingSampleV12:
    """Attach labels, apply the frozen sampler, and build selected features."""

    if statistics.schema_version != SCHEMA_VERSION or statistics.rule_id != FROZEN_RULE_ID:
        raise LazySiteFeatureError("lazy site statistics schema/rule drift")
    if tuple(site_feature_names(config)) != statistics.feature_names:
        raise LazySiteFeatureError("lazy statistics/config feature schema drift")
    labels_by_query = _labels_by_query(statistics, labels)
    selected_indices = _select_fit_indices(
        statistics._base, labels_by_query, config
    )
    selected = _materialize_selected_features(
        statistics, selected_indices, labels_by_query, config
    )
    counts = selected.groupby("query_id", sort=True).size()
    maximum = int(counts.max()) if not counts.empty else 0
    configured_cap = int(
        config.fit_candidate_sampling["maximum_candidates_per_query"]
    )
    if maximum > configured_cap:
        raise LazySiteFeatureError("lazy sampler exceeded the frozen fit cap")
    # Absence is not repaired.  Retention is required only when the positive
    # was naturally present in the complete Master.
    complete_pairs = set(
        statistics._base[["query_id", "candidate_siret"]].itertuples(
            index=False, name=None
        )
    )
    sampled_pairs = set(
        selected[["query_id", "candidate_siret"]].itertuples(
            index=False, name=None
        )
    )
    for query_id, label in labels_by_query.items():
        pair = (query_id, label.human_siret)
        if pair in complete_pairs and pair not in sampled_pairs:
            raise LazySiteFeatureError("frozen sampler evicted a natural positive")
    feature_count = len(statistics.feature_names)
    return LazyTrainingSampleV12(
        frame=selected,
        feature_names=statistics.feature_names,
        complete_rows=statistics.complete_rows,
        selected_rows=int(len(selected)),
        query_count=len(statistics.query_ids),
        maximum_selected_per_query=maximum,
        materialized_feature_cells=int(len(selected) * feature_count),
        avoided_dense_feature_cells=int(
            (statistics.complete_rows - len(selected)) * feature_count
        ),
        source_statistics_sha256=statistics.statistics_sha256,
    )


def materialize_labelled_master_training_sample(
    raw_master: pd.DataFrame, config: AdmissionConfigV12
) -> LazyTrainingSampleV12:
    """Differential/reference-friendly wrapper for an already labelled Master.

    Production fold-fused code should use the split statistics/label APIs.  The
    wrapper exists to compare directly with
    ``canonicalize_union(...)->sample_precomputed_candidates(...)``.
    """

    per_query: dict[str, HumanSiteLabelV12] = {}
    for raw in raw_master.to_dict("records"):
        query_id = str(raw.get("query_id") or "")
        human = _human_label(raw)
        if not human:
            raise LazySiteFeatureError(
                "every lazy v12 training query needs a human SIRET"
            )
        label = HumanSiteLabelV12(
            query_id=query_id,
            human_siret=human,
            operational_sirets=_operational_labels(raw, human),
        )
        previous = per_query.get(query_id)
        if previous is not None and previous != label:
            raise LazySiteFeatureError(
                "query-level human/operational label varies within query"
            )
        per_query[query_id] = label
    labels = [per_query[query_id] for query_id in sorted(per_query)]
    # Statistics remain label-blind and row-local parsing remains deferred.
    # Strip every recognized truth column before invoking the strict API.
    blind = raw_master.drop(
        columns=sorted(set(raw_master.columns) & LABEL_COLUMNS)
    )
    statistics = compute_lazy_site_statistics(blind, config)
    return materialize_training_sample(statistics, labels, config)


class FoldFusedLazySiteFeatureAdapterV12:
    """Small dependency object for ``FoldFusedContextAdapterV12`` integration."""

    def __init__(self, config: AdmissionConfigV12) -> None:
        self.config = config
        self.feature_names = tuple(site_feature_names(config))
        self.maximum_training_rows = int(
            config.fit_candidate_sampling["maximum_candidates_per_query"]
        )
        if self.maximum_training_rows > 128:
            raise LazySiteFeatureError("fold-fused lazy fit cap may not exceed 128")

    def compute_full_master_statistics(
        self, raw_master: pd.DataFrame
    ) -> LazySiteStatisticsV12:
        return compute_lazy_site_statistics(raw_master, self.config)

    def materialize_training_sample(
        self,
        statistics: LazySiteStatisticsV12,
        labels: Iterable[HumanSiteLabelV12],
    ) -> LazyTrainingSampleV12:
        return materialize_training_sample(statistics, labels, self.config)


__all__ = [
    "FROZEN_RULE_ID",
    "FoldFusedLazySiteFeatureAdapterV12",
    "HumanSiteLabelV12",
    "LazySiteFeatureError",
    "LazySiteStatisticsV12",
    "LazyTrainingSampleV12",
    "SCHEMA_VERSION",
    "compute_lazy_site_statistics",
    "materialize_labelled_master_training_sample",
    "materialize_training_sample",
]
