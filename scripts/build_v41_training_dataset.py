#!/usr/bin/env python3
"""Build the canonical V4.1 fit/dev training dataset without positive injection."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.features import (  # noqa: E402
    SEMANTIC_FEATURE_NAMES,
    V9_BASELINE_FEATURE_NAMES,
    make_feature_rows_from_preprocessed,
    normalize_text,
    preprocess_crm_row,
    set_global_name_idf_map,
)
from src.xgb_matcher.partitioned_store import (  # noqa: E402
    PartitionedCandidateStore,
)
from src.xgb_matcher.tfidf_cache import TfidfPersistentCache  # noqa: E402
from src.xgb_matcher.v41_features import (  # noqa: E402
    V41_CANDIDATE_FEATURE_NAMES,
    build_v41_candidate_features,
    validate_v41_model_feature_order,
)
from src.xgb_matcher.v41_retrieval import (  # noqa: E402
    V41CandidateRetriever,
    V41GlobalCandidateStore,
    V41RetrievalConfig,
    V41RetrievalVariant,
    normalize_input_siret,
)
from src.xgb_matcher.v9_dataset import file_sha256, read_table  # noqa: E402
from src.xgb_matcher.v9_features import (  # noqa: E402
    SELECTIVE_RETRIEVAL_CHANNELS,
    SELECTIVE_RETRIEVAL_FEATURE_NAMES,
)


SCHEMA_VERSION = "sireto-v4.1-training-dataset-1"
ALLOWED_LABEL_KINDS = {"MATCH_EXACT", "AMBIGUOUS"}
AUTHORIZED_INPUT_LABEL_KINDS = ALLOWED_LABEL_KINDS | {"UNRESOLVED"}
LEGACY_55_FEATURE_NAMES = [
    name
    for name in V9_BASELINE_FEATURE_NAMES
    if name not in SEMANTIC_FEATURE_NAMES
] + SELECTIVE_RETRIEVAL_FEATURE_NAMES
FEATURE_ORDER = LEGACY_55_FEATURE_NAMES + V41_CANDIDATE_FEATURE_NAMES
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
    "input_siret",
    "input_siren",
    "input_siret_state",
    "source_segment",
]
LABEL_COLUMNS = [
    "query_id",
    "label_kind",
    "ground_truth_siret",
    "ground_truth_siren",
]
CANDIDATE_METADATA_COLUMNS = [
    "query_id",
    "candidate_siret",
    "candidate_siren",
    "candidate_state",
    "is_ground_truth",
    "retrieval_rank",
    "retrieval_source",
    "retrieval_channel_count",
    "retrieval_agreement",
]
CANDIDATE_COLUMNS = CANDIDATE_METADATA_COLUMNS + FEATURE_ORDER

if len(LEGACY_55_FEATURE_NAMES) != 55:  # pragma: no cover - import-time guard
    raise RuntimeError("V4.1 requires the frozen 55-feature V4 legacy order")
validate_v41_model_feature_order(FEATURE_ORDER)


def _normalized_id(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    output = str(value).strip()
    return output if output and output.lower() != "nan" else None


def _normalized_siret(value: Any) -> str | None:
    return normalize_input_siret(value)


def _first_series(
    frame: pd.DataFrame,
    names: Sequence[str],
    *,
    default: Any = "",
) -> pd.Series:
    for name in names:
        if name in frame:
            return frame[name]
    return pd.Series([default] * len(frame), index=frame.index)


def _coalesced_series(
    frame: pd.DataFrame,
    names: Sequence[str],
    *,
    default: Any = "",
) -> pd.Series:
    output = pd.Series([None] * len(frame), index=frame.index, dtype=object)
    for name in names:
        if name not in frame:
            continue
        values = frame[name]
        usable = values.notna() & values.astype(str).str.strip().ne("")
        output.loc[output.isna() & usable] = values.loc[output.isna() & usable]
    return output.fillna(default)


_FORBIDDEN_COLUMN = re.compile(r"(?:test|holdout)", re.IGNORECASE)
_FORBIDDEN_VALUE = re.compile(
    r"^(?:test|holdout)(?:$|[_:/-])|(?:[_:/-])(?:test|holdout)$",
    re.IGNORECASE,
)


def assert_authorized_benchmark(
    frame: pd.DataFrame,
    *,
    source_name: str,
    denied_crm_ids: set[str],
) -> None:
    """Reject consumed roles, suspicious schemas and denylisted CRM IDs."""

    forbidden_columns = sorted(
        column for column in frame.columns if _FORBIDDEN_COLUMN.search(str(column))
    )
    if forbidden_columns:
        raise ValueError(
            f"{source_name}: test/holdout columns are forbidden: "
            f"{forbidden_columns}"
        )
    for column in frame.columns:
        if not (
            pd.api.types.is_object_dtype(frame[column])
            or isinstance(frame[column].dtype, pd.StringDtype)
        ):
            continue
        values = frame[column].dropna().astype(str).str.strip()
        forbidden = sorted(
            {
                value
                for value in values.unique()
                if _FORBIDDEN_VALUE.search(value)
            }
        )
        if forbidden:
            raise ValueError(
                f"{source_name}: test/holdout values are forbidden "
                f"in {column!r}: {forbidden[:5]}"
            )
    required = {"query_id", "crm_record_id", "label_kind"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{source_name}: benchmark columns missing: {sorted(missing)}"
        )
    crm_ids = frame["crm_record_id"].map(_normalized_id)
    overlap = sorted(set(crm_ids) & denied_crm_ids)
    if overlap:
        raise ValueError(
            f"{source_name}: consumed CRM IDs are forbidden: {overlap[:5]}"
        )
    label_kinds = set(frame["label_kind"].fillna("").astype(str).str.upper())
    unsupported = sorted(label_kinds - AUTHORIZED_INPUT_LABEL_KINDS)
    if unsupported:
        raise ValueError(
            f"{source_name}: unsupported training benchmark labels; "
            f"found {unsupported}"
        )


def load_denylist_ids(paths: Sequence[Path]) -> tuple[set[str], dict[str, str]]:
    if not paths:
        raise ValueError("At least one consumed-evaluation denylist is required")
    output: set[str] = set()
    hashes: dict[str, str] = {}
    for position, path in enumerate(paths):
        frame = read_table(path)
        # The historical closed benchmark is a single parquet containing
        # train/dev/test.  When used as a denylist, only its consumed test rows
        # are relevant.  A sealed-holdout file contains only holdout rows and
        # therefore falls through unchanged.
        if "split" in frame:
            split = frame["split"].fillna("").astype(str).str.lower()
            if split.eq("test").any():
                frame = frame.loc[split.eq("test")].copy()
        if "fresh_role" in frame:
            role = frame["fresh_role"].fillna("").astype(str).str.lower()
            if role.eq("holdout_sealed").any():
                frame = frame.loc[role.eq("holdout_sealed")].copy()
        id_column = next(
            (
                name
                for name in ("crm_record_id", "SERVICE ID", "service_id")
                if name in frame
            ),
            None,
        )
        if id_column is None:
            raise ValueError(f"Denylist has no CRM ID column: {path}")
        output.update(
            value
            for value in frame[id_column].map(_normalized_id)
            if value is not None
        )
        hashes[f"denylist_{position}:{path.name}"] = file_sha256(path)
    return output, hashes


def load_authorized_benchmarks(
    paths: Sequence[Path],
    *,
    denied_crm_ids: set[str],
) -> tuple[pd.DataFrame, dict[str, str], dict[str, Any]]:
    if not paths:
        raise ValueError("At least one authorized fit/dev benchmark is required")
    frames: list[pd.DataFrame] = []
    hashes: dict[str, str] = {}
    excluded_missing_ids = 0
    input_label_counts: dict[str, int] = {}
    for position, path in enumerate(paths):
        if path.suffix.lower() != ".parquet":
            raise ValueError(f"Authorized benchmarks must be parquet: {path}")
        # A consumed artifact must not be disguised as an authorized input.
        if _FORBIDDEN_VALUE.search(path.stem):
            raise ValueError(f"test/holdout benchmark path is forbidden: {path}")
        frame = read_table(path)
        assert_authorized_benchmark(
            frame,
            source_name=str(path),
            denied_crm_ids=denied_crm_ids,
        )
        frame = frame.copy()
        counts = (
            frame["label_kind"]
            .fillna("")
            .astype(str)
            .str.upper()
            .value_counts()
            .to_dict()
        )
        for label, count in counts.items():
            input_label_counts[label] = input_label_counts.get(label, 0) + int(count)
        frame = frame.loc[
            frame["label_kind"]
            .fillna("")
            .astype(str)
            .str.upper()
            .isin(ALLOWED_LABEL_KINDS)
        ].copy()
        missing_id = frame["crm_record_id"].map(_normalized_id).isna()
        excluded_missing_ids += int(missing_id.sum())
        frame = frame.loc[~missing_id].copy()
        if frame.empty:
            raise ValueError(
                f"{path}: no MATCH_EXACT or AMBIGUOUS rows remain"
            )
        frame["_benchmark_source"] = f"{position}:{path.name}"
        frames.append(frame)
        hashes[f"benchmark_{position}:{path.name}"] = file_sha256(path)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["query_id"] = combined["query_id"].astype(str)
    if combined["query_id"].duplicated().any():
        examples = combined.loc[
            combined["query_id"].duplicated(keep=False), "query_id"
        ].unique()
        raise ValueError(
            "query_id must be unique across authorized benchmarks: "
            f"{list(examples[:5])}"
        )
    return combined, hashes, {
        "input_label_counts": input_label_counts,
        "excluded_missing_crm_record_id_count": excluded_missing_ids,
    }


def canonicalize_benchmark(
    benchmark: pd.DataFrame,
    *,
    crm_source: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Join suspect input SIRETs while preserving all future grouping IDs."""

    if "SERVICE ID" not in crm_source or "SIRET" not in crm_source:
        raise ValueError("CRM source requires SERVICE ID and SIRET columns")
    crm = crm_source.copy()
    crm["_crm_record_id"] = crm["SERVICE ID"].map(_normalized_id)
    non_empty = crm["_crm_record_id"].dropna()
    if non_empty.duplicated().any():
        raise ValueError("CRM source SERVICE ID values must be unique")
    crm_by_id = crm.dropna(subset=["_crm_record_id"]).set_index("_crm_record_id")

    crm_ids = benchmark["crm_record_id"].map(_normalized_id)
    missing_source = sorted(set(crm_ids) - set(crm_by_id.index))
    if missing_source:
        raise ValueError(
            "Authorized benchmark CRM IDs are absent from the CRM source: "
            f"{missing_source[:5]}"
        )
    raw_input_siret = crm_ids.map(crm_by_id["SIRET"])
    normalized_input = raw_input_siret.map(_normalized_siret)
    label_kind = benchmark["label_kind"].astype(str).str.upper()
    truth = _first_series(
        benchmark,
        ("ground_truth_siret",),
        default=None,
    ).map(_normalized_siret)
    exact = label_kind.eq("MATCH_EXACT")
    if truth.loc[exact].isna().any():
        raise ValueError("MATCH_EXACT rows require a valid ground_truth_siret")
    if truth.loc[~exact].notna().any():
        raise ValueError("AMBIGUOUS rows cannot carry ground_truth_siret")

    source_segment = _coalesced_series(
        benchmark,
        ("source_segment", "subset", "fresh_role", "split"),
        default="authorized_fit_dev",
    ).fillna("authorized_fit_dev")
    queries = pd.DataFrame(
        {
            "query_id": benchmark["query_id"].astype(str),
            "crm_record_id": crm_ids,
            "crm_name": _first_series(
                benchmark, ("crm_name", "SITE")
            ).fillna(""),
            "crm_address": _first_series(
                benchmark, ("crm_address", "SITE_CLI_ADRESSE")
            ).fillna(""),
            "crm_postcode": _first_series(
                benchmark, ("postcode", "crm_postcode", "CODE_POSTAL")
            ).fillna(""),
            "crm_city": _first_series(
                benchmark, ("crm_city", "COMMUNE", "SITE_CLI_COMMUNE")
            ).fillna(""),
            "crm_insee": _first_series(
                benchmark, ("insee", "crm_insee", "CODE_INSEE")
            ).fillna(""),
            "crm_name_norm": "",
            "crm_address_norm": "",
            "crm_city_norm": "",
            # Existence/state is filled from the exact global store used by
            # retrieval, not trusted from the CRM or a geographic partition.
            "input_siret": normalized_input,
            "input_siren": normalized_input.map(
                lambda value: value[:9] if value else None
            ),
            "input_siret_state": "UNQUALIFIED",
            "source_segment": source_segment.astype(str),
        }
    )
    queries["crm_name_norm"] = queries["crm_name"].map(normalize_text)
    queries["crm_address_norm"] = queries["crm_address"].map(normalize_text)
    queries["crm_city_norm"] = queries["crm_city"].map(normalize_text)
    labels = pd.DataFrame(
        {
            "query_id": benchmark["query_id"].astype(str),
            "label_kind": label_kind,
            "ground_truth_siret": truth,
            "ground_truth_siren": truth.map(
                lambda value: value[:9] if value else None
            ),
        }
    )
    diagnostics = {
        "raw_invalid_input_siret_count": int(normalized_input.isna().sum()),
        "label_counts": label_kind.value_counts().to_dict(),
    }
    return queries[QUERY_COLUMNS], labels[LABEL_COLUMNS], diagnostics


def _rank_recip(value: Any) -> float:
    try:
        rank = int(value)
    except (TypeError, ValueError):
        return 0.0
    return 1.0 / rank if rank > 0 else 0.0


_LEGACY_CHANNEL_MAP = {
    "current_sparse": "sparse_active",
    "name_word": "closed_alias_name",
    "name_char": None,
    "address_word": "closed_alias_address",
    "siren_head": "input_siren_active_sites",
    "name_exact": "input_siret_active",
    "address_exact": None,
}


def build_legacy_55_features(
    legacy_row: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, float]:
    """Project V4.1 provenance into the frozen 55-column V4 model contract."""

    ranks = candidate.get("v41_channel_ranks") or {}
    rank = int(candidate.get("retrieval_rank") or 0)
    values = {
        name: float(legacy_row.get(name, 0.0))
        for name in LEGACY_55_FEATURE_NAMES
    }
    values.update(
        {
            "admission_rank_recip": 1.0 / rank if rank > 0 else 0.0,
            "admission_fusion_score": float(candidate.get("rrf_score") or 0.0),
            "admission_channel_count": float(
                candidate.get("retrieval_channel_count") or len(ranks)
            ),
            "admission_overlay_quota": 0.0,
        }
    )
    for legacy_channel in SELECTIVE_RETRIEVAL_CHANNELS:
        v41_channel = _LEGACY_CHANNEL_MAP[legacy_channel]
        values[f"admission_{legacy_channel}_rank_recip"] = (
            _rank_recip(ranks.get(v41_channel))
            if v41_channel is not None
            else 0.0
        )
    return values


def _candidate_schema() -> pa.Schema:
    string_columns = {
        "query_id",
        "candidate_siret",
        "candidate_siren",
        "candidate_state",
        "retrieval_source",
    }
    integer_columns = {
        "is_ground_truth",
        "retrieval_rank",
        "retrieval_channel_count",
        "retrieval_agreement",
    }
    return pa.schema(
        [
            pa.field(
                column,
                (
                    pa.string()
                    if column in string_columns
                    else (
                        pa.int32()
                        if column in integer_columns
                        else pa.float32()
                    )
                ),
            )
            for column in CANDIDATE_COLUMNS
        ]
    )


class CandidateWriter:
    def __init__(self, path: Path) -> None:
        self.schema = _candidate_schema()
        self._writer = pq.ParquetWriter(path, self.schema)
        self.count = 0

    def write(self, rows: Sequence[Mapping[str, Any]]) -> None:
        if not rows:
            return
        frame = pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)
        self._writer.write_table(
            pa.Table.from_pandas(
                frame,
                schema=self.schema,
                preserve_index=False,
            )
        )
        self.count += len(frame)

    def close(self) -> None:
        self._writer.close()


def retrieval_signature(
    config: V41RetrievalConfig,
    *,
    partitions_signature: str,
    global_store_signature: str,
) -> str:
    payload = {
        "config": {
            **asdict(config),
            "variant": config.variant.value,
        },
        "sparse_signature": config.sparse_config().signature().hash,
        "partitions_signature": partitions_signature,
        "global_store_signature": global_store_signature,
        "legacy_channel_projection": _LEGACY_CHANNEL_MAP,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _path_signature(path: Path) -> str:
    path = Path(path)
    if path.is_file():
        return file_sha256(path)
    root_manifest = path / "manifest.json"
    if root_manifest.exists():
        return file_sha256(root_manifest)
    candidates = sorted(
        candidate for candidate in path.rglob("*") if candidate.is_file()
    )
    digest = hashlib.sha256()
    for candidate in candidates:
        digest.update(str(candidate.relative_to(path)).encode())
        digest.update(file_sha256(candidate).encode())
    return digest.hexdigest()


def validate_retrieval_gate(
    *,
    gate_manifest_path: Path,
    retrieval_config: V41RetrievalConfig,
    crm_source_sha256: str,
    partitions_signature: str,
    global_store_signature: str,
) -> dict[str, Any]:
    """Fail closed unless the exact dev-gated retrieval is being reused."""

    gate_manifest_path = Path(gate_manifest_path)
    gate = json.loads(gate_manifest_path.read_text(encoding="utf-8"))
    if gate.get("schema_version") != "sireto-v4.1-retrieval-evaluation-1":
        raise ValueError("Unsupported V4.1 retrieval gate manifest")
    if gate.get("split") != "dev" or gate.get("positive_injection") is not False:
        raise ValueError("Retrieval gate must be leak-safe and evaluated on dev")
    outputs = gate.get("outputs") or {}
    summary_path = gate_manifest_path.parent / "summary.json"
    if (
        not summary_path.is_file()
        or outputs.get("summary.json") != file_sha256(summary_path)
    ):
        raise ValueError("Retrieval gate summary hash mismatch")
    raw_path = gate_manifest_path.parent / "raw_results.parquet"
    if (
        not raw_path.is_file()
        or outputs.get("raw_results.parquet") != file_sha256(raw_path)
    ):
        raise ValueError("Retrieval gate raw-results hash mismatch")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    selection = summary.get("selection") or {}
    variant = retrieval_config.variant.value
    if selection.get("verdict") != "GO":
        raise ValueError("Retrieval gate verdict is not GO")
    if selection.get("selected_variant") != variant:
        raise ValueError("Retrieval gate selected variant differs from --variant")
    inputs = gate.get("inputs") or {}
    if (inputs.get("crm_source") or {}).get("sha256") != crm_source_sha256:
        raise ValueError("Retrieval gate CRM source hash mismatch")
    if (inputs.get("partitions") or {}).get(
        "runtime_signature"
    ) != partitions_signature:
        raise ValueError("Retrieval gate partitions signature mismatch")
    if (inputs.get("global_store") or {}).get(
        "runtime_signature"
    ) != global_store_signature:
        raise ValueError("Retrieval gate global-store signature mismatch")
    variant_gate = (gate.get("retrieval") or {}).get(variant) or {}
    if variant_gate.get("v41_signature") != retrieval_config.signature():
        raise ValueError("Retrieval gate V4.1 config signature mismatch")
    expected_full_signature = retrieval_signature(
        retrieval_config,
        partitions_signature=partitions_signature,
        global_store_signature=global_store_signature,
    )
    if (
        variant_gate.get("dataset_retrieval_signature")
        != expected_full_signature
    ):
        raise ValueError("Retrieval gate full dataset signature mismatch")
    return {
        "manifest_path": str(gate_manifest_path),
        "manifest_sha256": file_sha256(gate_manifest_path),
        "summary_path": str(summary_path),
        "summary_sha256": file_sha256(summary_path),
        "selected_variant": variant,
        "retrieval_signature": expected_full_signature,
    }


def build_dataset(
    *,
    benchmark_paths: Sequence[Path],
    crm_source_path: Path,
    denylist_paths: Sequence[Path],
    retriever: Any,
    retrieval_config: V41RetrievalConfig,
    persistent_cache: Any,
    output_root: Path,
    partitions_signature: str,
    global_store_signature: str,
    retrieval_gate_manifest_path: Path,
) -> Path:
    """Build hash-addressed canonical artifacts using a frozen retriever."""

    denied_ids, deny_hashes = load_denylist_ids(denylist_paths)
    benchmark, benchmark_hashes, benchmark_diagnostics = load_authorized_benchmarks(
        benchmark_paths,
        denied_crm_ids=denied_ids,
    )
    crm_source = read_table(crm_source_path)
    crm_source_sha256 = file_sha256(crm_source_path)
    retrieval_gate = validate_retrieval_gate(
        gate_manifest_path=retrieval_gate_manifest_path,
        retrieval_config=retrieval_config,
        crm_source_sha256=crm_source_sha256,
        partitions_signature=partitions_signature,
        global_store_signature=global_store_signature,
    )
    queries, labels, diagnostics = canonicalize_benchmark(
        benchmark,
        crm_source=crm_source,
    )
    queries = queries.sort_values(
        ["crm_insee", "crm_postcode", "query_id"],
        kind="stable",
    ).reset_index(drop=True)
    signature = retrieval_signature(
        retrieval_config,
        partitions_signature=partitions_signature,
        global_store_signature=global_store_signature,
    )
    input_hashes = {
        **benchmark_hashes,
        **deny_hashes,
        "crm_source": crm_source_sha256,
        "partitions": partitions_signature,
        "global_store": global_store_signature,
        "retrieval_gate_manifest": retrieval_gate["manifest_sha256"],
    }
    identity = {
        "schema_version": SCHEMA_VERSION,
        "input_hashes": input_hashes,
        "retrieval_signature": signature,
        "feature_order": FEATURE_ORDER,
        "positive_injection": False,
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / build_id
    if target.exists():
        raise FileExistsError(f"Immutable V4.1 dataset already exists: {target}")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{build_id}.tmp-", dir=output_root)
    )

    labels_by_query = labels.set_index("query_id")
    query_state: dict[str, tuple[str | None, str | None, str]] = {}
    qualification_by_query: dict[str, Any] = {}
    if hasattr(retriever, "global_store") and hasattr(
        retriever.global_store, "qualify_input_sirets"
    ):
        qualifications = retriever.global_store.qualify_input_sirets(
            queries["input_siret"].tolist()
        )
        qualification_by_query = dict(
            zip(queries["query_id"].astype(str), qualifications, strict=True)
        )
    candidate_writer = CandidateWriter(staging / "candidates.parquet")
    candidate_buffer: list[dict[str, Any]] = []
    retrieval_misses = 0
    zero_candidate_queries = 0
    max_candidates = 0
    try:
        for query in queries.itertuples(index=False):
            query_id = str(query.query_id)
            label = labels_by_query.loc[query_id]
            crm_row = {
                "query_id": query_id,
                "crm_id": query_id,
                "crm_name": query.crm_name,
                "crm_address": query.crm_address,
                "crm_city": query.crm_city,
                "postcode": query.crm_postcode,
                "insee": query.crm_insee,
            }
            retrieval_kwargs = {
                "crm_row": crm_row,
                "crm_pre": preprocess_crm_row(crm_row),
                "input_siret": query.input_siret,
                # Ground truth is deliberately absent from the retrieval call.
                "gt_siret": None,
                "persistent_cache": persistent_cache,
            }
            if query_id in qualification_by_query:
                retrieval_kwargs["input_qualification"] = (
                    qualification_by_query[query_id]
                )
            result = retriever.build(**retrieval_kwargs)
            if result.sparse_result.gt_was_injected:
                raise ValueError("Positive injection is forbidden in V4.1")
            candidates = list(result.candidates)
            if len(candidates) > 100:
                raise ValueError("V4.1 candidate budget exceeds 100")
            sirets = [str(candidate.get("siret") or "") for candidate in candidates]
            if len(sirets) != len(set(sirets)):
                raise ValueError("V4.1 candidate SIRETs must be unique")
            if any(
                str(candidate.get("etat_admin") or "").upper() != "A"
                for candidate in candidates
            ):
                raise ValueError("V4.1 training candidates must all be active")
            max_candidates = max(max_candidates, len(candidates))
            zero_candidate_queries += int(not candidates)
            qualification = result.input_siret
            query_state[query_id] = (
                qualification.normalized_siret,
                qualification.siren,
                qualification.state.value,
            )

            set_global_name_idf_map(
                result.sparse_result.idf_map,
                result.sparse_result.default_idf,
            )
            legacy_rows = make_feature_rows_from_preprocessed(
                preprocess_crm_row(crm_row),
                candidates,
                include_semantic=False,
            )
            truth = (
                str(label["ground_truth_siret"])
                if label["label_kind"] == "MATCH_EXACT"
                and pd.notna(label["ground_truth_siret"])
                else None
            )
            hit = truth is not None and truth in set(sirets)
            retrieval_misses += int(truth is not None and not hit)
            for position, (candidate, legacy_row) in enumerate(
                zip(candidates, legacy_rows, strict=True),
                start=1,
            ):
                candidate = dict(candidate)
                ranks = candidate.get("v41_channel_ranks") or {}
                candidate["candidate_from_sparse"] = "sparse_active" in ranks
                candidate["candidate_from_input_siret"] = (
                    "input_siret_active" in ranks
                )
                candidate["candidate_from_input_siren"] = (
                    "input_siren_active_sites" in ranks
                )
                candidate["candidate_from_closed_alias"] = bool(
                    {"closed_alias_name", "closed_alias_address"} & set(ranks)
                )
                siret = str(candidate["siret"])
                channel_count = int(
                    candidate.get("retrieval_channel_count") or len(ranks)
                )
                row = {
                    "query_id": query_id,
                    "candidate_siret": siret,
                    "candidate_siren": str(
                        candidate.get("siren") or siret[:9]
                    ),
                    "candidate_state": str(
                        candidate.get("etat_admin") or ""
                    ).upper(),
                    "is_ground_truth": int(siret == truth),
                    "retrieval_rank": int(
                        candidate.get("retrieval_rank") or position
                    ),
                    "retrieval_source": str(
                        candidate.get("retrieval_source") or "v41"
                    ),
                    "retrieval_channel_count": channel_count,
                    "retrieval_agreement": int(channel_count >= 2),
                    **build_legacy_55_features(legacy_row, candidate),
                    **build_v41_candidate_features(
                        candidate,
                        input_siret=qualification.normalized_siret,
                    ),
                }
                candidate_buffer.append(row)
            if len(candidate_buffer) >= 10_000:
                candidate_writer.write(candidate_buffer)
                candidate_buffer.clear()
        candidate_writer.write(candidate_buffer)
        candidate_buffer.clear()
    except BaseException:
        candidate_writer.close()
        shutil.rmtree(staging, ignore_errors=True)
        raise
    candidate_writer.close()

    try:
        queries = queries.copy()
        for index, query_id in enumerate(queries["query_id"]):
            input_siret, input_siren, input_state = query_state[str(query_id)]
            queries.at[index, "input_siret"] = input_siret
            queries.at[index, "input_siren"] = input_siren
            queries.at[index, "input_siret_state"] = input_state
        queries.to_parquet(staging / "queries.parquet", index=False)
        labels.to_parquet(staging / "labels.parquet", index=False)

        diagnostics.update(
            {
                **benchmark_diagnostics,
                "query_count": int(len(queries)),
                "candidate_count": int(candidate_writer.count),
                "max_candidates": int(max_candidates),
                "zero_candidate_query_count": int(zero_candidate_queries),
                "match_exact_retrieval_miss_count": int(retrieval_misses),
                "input_siret_state_counts": queries[
                    "input_siret_state"
                ].value_counts().to_dict(),
                "consumed_crm_ids_scored": 0,
            }
        )
        output_hashes = {
            name: file_sha256(staging / name)
            for name in (
                "queries.parquet",
                "labels.parquet",
                "candidates.parquet",
            )
        }
        manifest = {
            **identity,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "retrieval_config": {
                **asdict(retrieval_config),
                "variant": retrieval_config.variant.value,
            },
            "retrieval_gate": retrieval_gate,
            "legacy_55_channel_projection": _LEGACY_CHANNEL_MAP,
            "row_counts": {
                "queries": int(len(queries)),
                "labels": int(len(labels)),
                "candidates": int(candidate_writer.count),
            },
            "outputs": output_hashes,
            "diagnostics": diagnostics,
            "output_hashes": output_hashes,
            "invariants": {
                "positive_injection": False,
                "candidate_budget": 100,
                "active_candidates_only": True,
                "ground_truth_miss_preserved": True,
                "consumed_evaluation_ids_scored": 0,
                "final_split_assigned": False,
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(
                manifest,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        type=Path,
        action="append",
        required=True,
        help="Authorized fit/dev benchmark parquet; repeat for multiple inputs.",
    )
    parser.add_argument("--crm-source", type=Path, required=True)
    parser.add_argument(
        "--denylist",
        type=Path,
        action="append",
        required=True,
        help="Consumed benchmark IDs that must never be scored.",
    )
    parser.add_argument("--partitions", type=Path, required=True)
    parser.add_argument("--global-store", type=Path, required=True)
    parser.add_argument(
        "--variant",
        choices=[variant.value for variant in V41RetrievalVariant],
        required=True,
    )
    parser.add_argument(
        "--retrieval-gate-manifest",
        type=Path,
        required=True,
        help="Frozen dev retrieval manifest whose selected variant is GO.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Persistent TF-IDF cache root, normally on /Volumes/CATNAT_DATA.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    config = V41RetrievalConfig(
        variant=V41RetrievalVariant(args.variant),
        max_candidates=100,
    )
    partition_signature = _path_signature(args.partitions)
    global_signature = _path_signature(args.global_store)
    cache = TfidfPersistentCache(
        config_hash=config.sparse_config().tfidf_artifact_hash(),
        cache_dir=args.cache_dir,
    )
    partitioned_store = PartitionedCandidateStore(args.partitions)
    with V41GlobalCandidateStore(args.global_store) as global_store:
        retriever = V41CandidateRetriever(
            partitioned_store=partitioned_store,
            global_store=global_store,
            config=config,
        )
        target = build_dataset(
            benchmark_paths=args.benchmark,
            crm_source_path=args.crm_source,
            denylist_paths=args.denylist,
            retriever=retriever,
            retrieval_config=config,
            persistent_cache=cache,
            output_root=args.output_root,
            partitions_signature=partition_signature,
            global_store_signature=global_signature,
            retrieval_gate_manifest_path=args.retrieval_gate_manifest,
        )
    print(target)


if __name__ == "__main__":
    main()
