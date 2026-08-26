"""Leakage-safe primitives for the v19 parent@2000 fusion workbench."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb


SCHEMA_VERSION = "sireto-parent-fusion-v19-1"
ALLOWED_FOLDS = (2, 3, 4)
IDENTIFIER_FEATURE_NAMES = frozenset(
    {"query_id", "fold", "siren", "siret", "nic", "label", "target", "truth", "gt_siren"}
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(dict(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def feature_names(channels: Sequence[str]) -> tuple[str, ...]:
    base = (
        "channel_count",
        "best_rank_reciprocal",
        "rrf_score",
        "best_seed_rank_reciprocal",
        "structured_channel_count",
        "native_channel_count",
    )
    per_channel = tuple(
        item
        for channel in channels
        for item in (f"has__{channel}", f"rr__{channel}")
    )
    names = (*base, *per_channel)
    assert_identifier_free_features(names)
    return names


def assert_identifier_free_features(names: Iterable[str]) -> None:
    names = tuple(str(name) for name in names)
    if not names or len(names) != len(set(names)):
        raise ValueError("v19 feature order is empty or duplicated")
    for name in names:
        lowered = name.lower()
        if (
            lowered in IDENTIFIER_FEATURE_NAMES
            or lowered.startswith(("query_", "candidate_siren", "candidate_siret", "gt_", "truth_", "target_"))
            or lowered.endswith("_query_id")
        ):
            raise ValueError(f"identifier-derived v19 feature forbidden: {name}")


def feature_sql(channels: Sequence[str], *, rrf_k: int) -> str:
    if not channels or len(channels) != len(set(channels)):
        raise ValueError("v19 channels are empty or duplicated")
    if rrf_k <= 0:
        raise ValueError("v19 RRF k must be positive")
    expressions = [
        "count(*)::INTEGER AS channel_count",
        "max(1.0 / (? + channel_rank))::FLOAT AS best_rank_reciprocal",
        "sum(1.0 / (? + channel_rank))::FLOAT AS rrf_score",
        "max(CASE WHEN seed_rank IS NULL THEN 0.0 ELSE 1.0 / (? + seed_rank) END)::FLOAT AS best_seed_rank_reciprocal",
        "sum(CASE WHEN relation_type <> 'SIRET_NATIVE' THEN 1 ELSE 0 END)::INTEGER AS structured_channel_count",
        "sum(CASE WHEN relation_type = 'SIRET_NATIVE' THEN 1 ELSE 0 END)::INTEGER AS native_channel_count",
    ]
    parameters = [rrf_k, rrf_k, rrf_k]
    for channel in channels:
        escaped = channel.replace("'", "''")
        expressions.extend(
            [
                f"max(CASE WHEN channel='{escaped}' THEN 1 ELSE 0 END)::TINYINT AS has__{channel}",
                f"max(CASE WHEN channel='{escaped}' THEN 1.0 / (? + channel_rank) ELSE 0.0 END)::FLOAT AS rr__{channel}",
            ]
        )
        parameters.append(rrf_k)
    sql = ",\n       ".join(expressions)
    # DuckDB positional parameters are returned alongside the SQL so callers
    # cannot interpolate numeric configuration into identifiers.
    return json.dumps({"sql": sql, "parameters": parameters}, separators=(",", ":"))


def decode_feature_sql(payload: str) -> tuple[str, list[int]]:
    raw = json.loads(payload)
    return str(raw["sql"]), [int(value) for value in raw["parameters"]]


def validate_fold_isolation(truth: pd.DataFrame) -> None:
    required = {"query_id", "fold", "gt_siren"}
    if not required.issubset(truth.columns):
        raise ValueError("v19 truth surface is incomplete")
    if set(truth["fold"].astype(int).unique()) != set(ALLOWED_FOLDS):
        raise ValueError("v19 truth opened forbidden or missing folds")
    if truth["query_id"].duplicated().any():
        raise ValueError("v19 query population is not unique")
    by_fold = {
        fold: set(truth.loc[truth["fold"].eq(fold), "gt_siren"].astype(str))
        for fold in ALLOWED_FOLDS
    }
    for index, left in enumerate(ALLOWED_FOLDS):
        for right in ALLOWED_FOLDS[index + 1 :]:
            if by_fold[left] & by_fold[right]:
                raise ValueError("v19 outer folds share a ground-truth SIREN")


def prepare_training_frame(
    rows: pd.DataFrame,
    *,
    features: Sequence[str],
    negative_limit: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    assert_identifier_free_features(features)
    if negative_limit < 1:
        raise ValueError("v19 negative limit must be positive")
    required = {"query_id", "is_positive", "negative_rank", *features}
    if not required.issubset(rows.columns):
        raise ValueError("v19 training frame is incomplete")
    rows = rows.loc[rows["is_positive"].eq(1) | rows["negative_rank"].le(negative_limit)].copy()
    positive_per_query = rows.groupby("query_id", sort=False)["is_positive"].sum()
    eligible = positive_per_query[positive_per_query.eq(1)].index
    rows = rows.loc[rows["query_id"].isin(eligible)].copy()
    rows.sort_values(["query_id", "is_positive", "negative_rank"], ascending=[True, False, True], inplace=True, kind="stable")
    sizes = rows.groupby("query_id", sort=False).size()
    if rows.empty or sizes.min() < 2 or not rows.groupby("query_id")["is_positive"].sum().eq(1).all():
        raise ValueError("v19 training groups need exactly one positive and a negative")
    qid = np.repeat(np.arange(len(sizes), dtype=np.int64), sizes.to_numpy(dtype=np.int64))
    X = rows[list(features)].fillna(0.0).to_numpy(dtype=np.float32)
    y = rows["is_positive"].to_numpy(dtype=np.int8)
    diagnostics = {
        "trainable_query_count": int(len(sizes)),
        "training_row_count": int(len(rows)),
        "natural_positive_count": int(y.sum()),
        "max_rows_per_query": int(sizes.max()),
        "positive_injection": False,
    }
    return X, y, qid, diagnostics


def fit_ranker(
    rows: pd.DataFrame,
    *,
    features: Sequence[str],
    params: Mapping[str, Any],
    negative_limit: int,
) -> tuple[xgb.XGBRanker, dict[str, Any]]:
    X, y, qid, diagnostics = prepare_training_frame(
        rows, features=features, negative_limit=negative_limit
    )
    model = xgb.XGBRanker(**dict(params))
    model.fit(X, y, qid=qid, verbose=False)
    return model, diagnostics


def file_record(path: Path) -> dict[str, Any]:
    metadata = pq.ParquetFile(path).metadata
    return {
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "rows": metadata.num_rows,
        "row_groups": metadata.num_row_groups,
    }


OOF_SCHEMA = pa.schema(
    [
        pa.field("query_id", pa.string(), nullable=False),
        pa.field("fold", pa.int8(), nullable=False),
        pa.field("siren", pa.string(), nullable=False),
        pa.field("parent_rank", pa.int32(), nullable=False),
        pa.field("ltr_score", pa.float32(), nullable=False),
        pa.field("rrf_score", pa.float32(), nullable=False),
    ]
)


__all__ = [
    "ALLOWED_FOLDS",
    "OOF_SCHEMA",
    "SCHEMA_VERSION",
    "assert_identifier_free_features",
    "canonical_json_sha256",
    "decode_feature_sql",
    "feature_names",
    "feature_sql",
    "file_record",
    "fit_ranker",
    "prepare_training_frame",
    "sha256_file",
    "validate_fold_isolation",
]
