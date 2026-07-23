"""Canonical, hash-addressed dataset contract for SIRETO V9."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from .contracts import GroundTruthKind
from .features import V9_BASELINE_FEATURE_NAMES, normalize_text
from .retrieval_config import RetrievalConfigV1
from .v9_features import (
    V9_RETRIEVAL_FEATURE_NAMES,
    inject_retrieval_siren_features,
)


SCHEMA_VERSION = "v9.0"
V9_CANDIDATE_FEATURE_NAMES = (
    V9_BASELINE_FEATURE_NAMES + V9_RETRIEVAL_FEATURE_NAMES
)
QUERY_COLUMNS = [
    "query_id",
    "crm_name",
    "crm_address",
    "crm_postcode",
    "crm_city",
    "crm_insee",
    "crm_name_norm",
    "crm_address_norm",
    "crm_city_norm",
    "reference_date",
    "split",
]
LABEL_COLUMNS = [
    "query_id",
    "label_kind",
    "ground_truth_siret",
    "ground_truth_siren",
    "label_source",
    "validator",
    "validated_at",
    "sirene_snapshot_id",
    "split",
]


def normalize_siret(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    raw = str(value).strip()
    if not raw or raw.lower() == "nan":
        return None
    if "e" in raw.lower():
        try:
            raw = str(int(float(raw)))
        except (ValueError, OverflowError):
            pass
    digits = "".join(char for char in raw if char.isdigit())
    return digits.zfill(14) if digits else None


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    sample = path.read_text(encoding="utf-8", errors="ignore")[:4096]
    delimiter = ";" if sample.count(";") > sample.count(",") else ","
    return pd.read_csv(path, sep=delimiter, dtype=str)


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_split(key: str, seed: int = 42) -> str:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    if value < 0.70:
        return "train"
    if value < 0.85:
        return "dev"
    return "test"


def _first_column(df: pd.DataFrame, names: Iterable[str], default: Any = "") -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series([default] * len(df), index=df.index)


def canonicalize_labels(
    source: pd.DataFrame,
    *,
    snapshot_id: str,
    seed: int,
    default_source: str,
) -> pd.DataFrame:
    query_ids = _first_column(source, ("query_id", "crm_id", "SERVICE ID"))
    sirets = _first_column(
        source,
        ("ground_truth_siret", "siret_gt", "SIRET", "siret"),
        default=None,
    ).map(normalize_siret)

    if "label_kind" in source.columns:
        kinds = source["label_kind"].fillna(GroundTruthKind.UNRESOLVED.value).astype(str)
    else:
        kinds = sirets.map(
            lambda value: (
                GroundTruthKind.MATCH_EXACT.value
                if value
                else GroundTruthKind.UNRESOLVED.value
            )
        )
    allowed = {kind.value for kind in GroundTruthKind}
    invalid = sorted(set(kinds) - allowed)
    if invalid:
        raise ValueError(f"Unsupported label_kind values: {invalid}")

    labels = pd.DataFrame(
        {
            "query_id": query_ids.astype(str),
            "label_kind": kinds,
            "ground_truth_siret": sirets,
            "label_source": _first_column(
                source,
                ("label_source",),
                default_source,
            ).fillna(default_source),
            "validator": _first_column(source, ("validator",), "").fillna(""),
            "validated_at": _first_column(source, ("validated_at",), "").fillna(""),
            "sirene_snapshot_id": snapshot_id,
        }
    )
    if labels["query_id"].duplicated().any():
        raise ValueError("query_id must be unique in the canonical label source")
    labels["ground_truth_siren"] = labels["ground_truth_siret"].map(
        lambda value: value[:9] if value else None
    )
    invalid_match = (
        labels["label_kind"].eq(GroundTruthKind.MATCH_EXACT.value)
        & labels["ground_truth_siret"].isna()
    )
    if invalid_match.any():
        raise ValueError("MATCH_EXACT labels require ground_truth_siret")
    invalid_open = (
        ~labels["label_kind"].eq(GroundTruthKind.MATCH_EXACT.value)
        & labels["ground_truth_siret"].notna()
    )
    if invalid_open.any():
        raise ValueError("Only MATCH_EXACT labels may carry a ground_truth_siret")

    split_keys = labels["ground_truth_siren"].fillna(labels["query_id"])
    labels["split"] = split_keys.map(lambda key: stable_split(str(key), seed))
    return labels[LABEL_COLUMNS]


def canonicalize_queries(source: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    queries = pd.DataFrame(
        {
            "query_id": _first_column(
                source,
                ("query_id", "crm_id", "SERVICE ID"),
            ).astype(str),
            "crm_name": _first_column(source, ("crm_name", "SITE", "name")).fillna(""),
            "crm_address": _first_column(
                source,
                ("crm_address", "SITE_CLI_ADRESSE", "address"),
            ).fillna(""),
            "crm_postcode": _first_column(
                source,
                ("crm_postcode", "postcode", "CODE_POSTAL"),
            ).fillna(""),
            "crm_city": _first_column(
                source,
                ("crm_city", "crm_city_addr", "SITE_CLI_COMMUNE", "COMMUNE"),
            ).fillna(""),
            "crm_insee": _first_column(
                source,
                ("crm_insee", "insee", "CODE_INSEE"),
            ).fillna(""),
            "reference_date": _first_column(
                source,
                ("reference_date", "observed_at"),
            ).fillna(""),
        }
    )
    if queries["query_id"].duplicated().any():
        raise ValueError("query_id must be unique in the canonical query source")
    queries["crm_name_norm"] = queries["crm_name"].map(normalize_text)
    queries["crm_address_norm"] = queries["crm_address"].map(normalize_text)
    queries["crm_city_norm"] = queries["crm_city"].map(normalize_text)
    split_map = labels.set_index("query_id")["split"]
    queries["split"] = queries["query_id"].map(split_map)
    if queries["split"].isna().any():
        raise ValueError("Every query must have exactly one canonical label")
    return queries[QUERY_COLUMNS]


def canonicalize_candidates(
    source: pd.DataFrame | None,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    base_columns = [
        "query_id",
        "candidate_siret",
        "candidate_siren",
        "split",
        "is_ground_truth",
    ]
    if source is None:
        return pd.DataFrame(columns=base_columns + V9_CANDIDATE_FEATURE_NAMES)

    candidates = source.copy()
    if "query_id" not in candidates.columns and "crm_id" in candidates.columns:
        candidates = candidates.rename(columns={"crm_id": "query_id"})
    if "candidate_siret" not in candidates.columns:
        for column in ("siret", "siret_candidate"):
            if column in candidates.columns:
                candidates = candidates.rename(columns={column: "candidate_siret"})
                break
    required = {"query_id", "candidate_siret"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Missing candidate columns: {sorted(missing)}")

    candidates["query_id"] = candidates["query_id"].astype(str)
    unknown_queries = sorted(set(candidates["query_id"]) - set(labels["query_id"]))
    if unknown_queries:
        raise ValueError(
            f"Candidates reference {len(unknown_queries)} unknown query_id values"
        )
    candidates["candidate_siret"] = candidates["candidate_siret"].map(normalize_siret)
    candidates = candidates[candidates["candidate_siret"].notna()].copy()
    candidates["candidate_siren"] = candidates["candidate_siret"].str[:9]
    label_index = labels.set_index("query_id")
    candidates["split"] = candidates["query_id"].map(label_index["split"])
    candidates["is_ground_truth"] = [
        int(
            label_index.at[query_id, "label_kind"] == GroundTruthKind.MATCH_EXACT.value
            and siret == label_index.at[query_id, "ground_truth_siret"]
        )
        for query_id, siret in zip(
            candidates["query_id"],
            candidates["candidate_siret"],
            strict=True,
        )
    ]
    for feature in V9_BASELINE_FEATURE_NAMES:
        if feature not in candidates.columns:
            candidates[feature] = 0.0
        candidates[feature] = pd.to_numeric(candidates[feature], errors="coerce").fillna(0.0)
    passthrough = [
        column
        for column in (
            "sparse_score",
            "sparse_rank",
            "dense_score",
            "dense_rank",
            "rrf_score",
            "retrieval_rank",
            "retrieval_source",
            "retrieval_channel_count",
            "retrieval_agreement",
            "global_siren_rank",
            "denomination",
            "denomination_usuelle_ul",
            "enseigne1",
            "enseigne2",
            "enseigne3",
            "address",
            "postcode",
            "city",
            "forme_juridique",
        )
        if column in candidates.columns
    ]
    for feature in V9_RETRIEVAL_FEATURE_NAMES:
        candidates[feature] = 0.0
    for _query_id, query_rows in candidates.groupby("query_id", sort=False):
        indices = query_rows.index.tolist()
        feature_rows = candidates.loc[indices, V9_RETRIEVAL_FEATURE_NAMES].to_dict(
            "records"
        )
        candidate_rows = candidates.loc[indices].to_dict("records")
        inject_retrieval_siren_features(feature_rows, candidate_rows)
        candidates.loc[indices, V9_RETRIEVAL_FEATURE_NAMES] = pd.DataFrame(
            feature_rows,
            index=indices,
        )
    return candidates[base_columns + passthrough + V9_CANDIDATE_FEATURE_NAMES]


def assert_entity_disjoint(labels: pd.DataFrame) -> None:
    matches = labels[labels["ground_truth_siren"].notna()]
    per_siren = matches.groupby("ground_truth_siren")["split"].nunique()
    leaking = per_siren[per_siren > 1]
    if not leaking.empty:
        raise ValueError(f"SIREN split leakage detected for {len(leaking)} entities")


@dataclass(frozen=True)
class V9DatasetManifest:
    schema_version: str
    build_id: str
    created_at: str
    seed: int
    sirene_snapshot_id: str
    input_hashes: Mapping[str, str]
    retrieval_config: Mapping[str, Any]
    retrieval_signature: str
    tokenizer_fingerprint: str | None
    feature_order: list[str]
    row_counts: Mapping[str, int]
    legacy_artifacts_allowed: bool = False

    @classmethod
    def load(cls, path: Path) -> "V9DatasetManifest":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def validate(
        self,
        *,
        retrieval_config: RetrievalConfigV1 | None = None,
        feature_order: Iterable[str] = V9_CANDIDATE_FEATURE_NAMES,
    ) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported dataset schema {self.schema_version}; expected {SCHEMA_VERSION}"
            )
        expected_features = list(feature_order)
        if self.feature_order != expected_features:
            raise ValueError("Dataset feature order is incompatible with the requested model")
        if retrieval_config is not None:
            expected_signature = retrieval_config.signature().hash
            if self.retrieval_signature != expected_signature:
                raise ValueError("Dataset retrieval signature mismatch")
        if self.legacy_artifacts_allowed:
            raise ValueError("V9 training refuses manifests that allow legacy artefacts")


def tokenizer_fingerprint(model_path: Path | None) -> str | None:
    if model_path is None or not model_path.exists():
        return None
    digest = hashlib.sha256()
    found = False
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        path = model_path / name
        if path.exists():
            found = True
            digest.update(name.encode("utf-8"))
            digest.update(file_sha256(path).encode("ascii"))
    return digest.hexdigest() if found else None


def build_canonical_dataset(
    *,
    query_source_path: Path,
    output_root: Path,
    sirene_snapshot_id: str,
    candidate_source_path: Path | None = None,
    label_source_path: Path | None = None,
    sirene_snapshot_path: Path | None = None,
    tokenizer_model_path: Path | None = None,
    retrieval_config: RetrievalConfigV1 | None = None,
    seed: int = 42,
    default_label_source: str = "provided_ground_truth",
) -> Path:
    retrieval_config = retrieval_config or RetrievalConfigV1()
    query_source = read_table(query_source_path)
    label_source = read_table(label_source_path) if label_source_path else query_source
    labels = canonicalize_labels(
        label_source,
        snapshot_id=sirene_snapshot_id,
        seed=seed,
        default_source=default_label_source,
    )
    queries = canonicalize_queries(query_source, labels)
    candidates_source = read_table(candidate_source_path) if candidate_source_path else None
    candidates = canonicalize_candidates(candidates_source, labels)
    assert_entity_disjoint(labels)

    input_paths = {
        "queries": query_source_path,
        **({"labels": label_source_path} if label_source_path else {}),
        **({"candidates": candidate_source_path} if candidate_source_path else {}),
        **({"sirene_snapshot": sirene_snapshot_path} if sirene_snapshot_path else {}),
    }
    input_hashes = {
        name: file_sha256(path)
        for name, path in input_paths.items()
        if path is not None
    }
    manifest_payload = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "sirene_snapshot_id": sirene_snapshot_id,
        "input_hashes": input_hashes,
        "retrieval_config": retrieval_config.to_dict(),
        "retrieval_signature": retrieval_config.signature().hash,
        "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer_model_path),
        "feature_order": V9_CANDIDATE_FEATURE_NAMES,
        "row_counts": {
            "queries": len(queries),
            "labels": len(labels),
            "candidates": len(candidates),
        },
        "legacy_artifacts_allowed": False,
    }
    build_id = hashlib.sha256(
        json.dumps(
            manifest_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]
    output_dir = output_root / build_id
    output_dir.mkdir(parents=True, exist_ok=False)
    queries.to_parquet(output_dir / "queries.parquet", index=False)
    labels.to_parquet(output_dir / "labels.parquet", index=False)
    candidates.to_parquet(output_dir / "candidates.parquet", index=False)
    manifest = V9DatasetManifest(
        build_id=build_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        **manifest_payload,
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(asdict(manifest), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_dir


__all__ = [
    "SCHEMA_VERSION",
    "QUERY_COLUMNS",
    "LABEL_COLUMNS",
    "V9_CANDIDATE_FEATURE_NAMES",
    "V9DatasetManifest",
    "normalize_siret",
    "stable_split",
    "canonicalize_labels",
    "canonicalize_queries",
    "canonicalize_candidates",
    "assert_entity_disjoint",
    "build_canonical_dataset",
]
