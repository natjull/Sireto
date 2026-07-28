#!/usr/bin/env python3
"""Build the frozen V4.6 training population with V4.2-B candidate pools."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_v41_training_dataset import (  # noqa: E402
    CANDIDATE_COLUMNS,
    FEATURE_ORDER,
    CandidateWriter,
    _path_signature,
    build_legacy_55_features,
)
from src.xgb_matcher.features import (  # noqa: E402
    make_feature_rows_from_preprocessed,
    preprocess_crm_row,
    set_global_name_idf_map,
)
from src.xgb_matcher.partitioned_store import PartitionedCandidateStore  # noqa: E402
from src.xgb_matcher.tfidf_cache import TfidfPersistentCache  # noqa: E402
from src.xgb_matcher.v41_features import (  # noqa: E402
    build_v41_candidate_features,
    validate_v41_model_feature_order,
)
from src.xgb_matcher.v41_retrieval import (  # noqa: E402
    V41CandidateRetriever,
    V41CurrentStateStore,
    V41GlobalCandidateStore,
    V41RetrievalConfig,
    V41RetrievalVariant,
)
from src.xgb_matcher.v41_split import (  # noqa: E402
    assign_connected_siren_splits,
    validate_connected_siren_split,
)
from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


SCHEMA_VERSION = "sireto-v4.6-aligned-ranker-dataset-1"
EXPERIMENT_ID = "V46_ALIGNED_RANKER_V42B"
SEED = 42
EXPECTED_QUERY_COUNT = 7_003
EXPECTED_FIT_COUNT = 5_547
EXPECTED_DEV_COUNT = 1_456
EXPECTED_DEV_EXACT_COUNT = 1_217
EXPECTED_INPUT_HASHES = {
    "queries.parquet": "6a12f1c4ca9ec33636ebcf7748c208595c6168d7cdb8c068e1434af3fe22abb0",
    "labels.parquet": "69032b745817959422ef26e4c0c1228686260c1daa272ca5d6aba1d7be087b04",
    "legacy_candidates.parquet": "34b526fe49e3451c05248294305e4a8d6ccf4db92277eb36dc03cc6231420c67",
    "split_assignments.parquet": "33fa52af7a740124235c151efb5b9a8834ffd1c83c65d1af56c75b2eff271193",
    "state_snapshot": "c91180cc5bae86948dd57d752c9bae45e58cc64653e99d5a9357664b67300845",
    "v42_manifest": "63b52c3a1466070410881b0ea61b833ff5d413262239920abbc6b04e3f153f54",
}
EXPECTED_V42_CONFIG_SIGNATURE = (
    "021f928e21e2360186217862b4310be90fe0f705c1bfbf43b39a8b41e644e40c"
)
EXPECTED_CONTRACT_SHA256 = (
    "6ed52cb8fab8d1634c80ef936f37c5529d732db71547c8ed8c2390b9e99d4d80"
)
DEFAULT_CONTRACT = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "v4_6_aligned_ranker_contract.md"
)
ROLE_COLUMNS = {"source_segment", "split", "partition", "subset", "fresh_role"}
FORBIDDEN_ROLE_VALUE = re.compile(
    r"(?:^|[_:/-])(test|holdout|v4_4|random|adjudication)(?:$|[_:/-])",
    re.IGNORECASE,
)
FORBIDDEN_SCHEMA_TOKEN = re.compile(
    r"(?:^|_)(test|holdout|v4_4|random|adjudication)(?:$|_)",
    re.IGNORECASE,
)


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _external_path(path: Path, *, name: str) -> Path:
    resolved = Path(path).resolve()
    external_root = Path("/Volumes/CATNAT_DATA")
    if not resolved.is_relative_to(external_root):
        raise ValueError(f"{name} must be located under {external_root}")
    return resolved


def _input_record(path: Path, *, row_count: int | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(Path(path).resolve()),
        "size_bytes": int(Path(path).stat().st_size),
        "sha256": file_sha256(Path(path)),
    }
    if row_count is not None:
        record["row_count"] = int(row_count)
    return record


def _directory_record(
    path: Path,
    *,
    runtime_signature: str,
    row_count: int | None,
) -> dict[str, Any]:
    path = Path(path)
    files = [item for item in path.rglob("*") if item.is_file()]
    return {
        "path": str(path.resolve()),
        "runtime_signature": runtime_signature,
        "file_count": len(files),
        "size_bytes": int(sum(item.stat().st_size for item in files)),
        "row_count": row_count,
    }


def _physical_parquet_row_count(path: Path) -> int:
    root = Path(path)
    physical_partitions = [
        item
        for item in root.rglob("*.parquet")
        if "manifest" not in item.relative_to(root).parts
    ]
    return int(
        sum(
            pq.ParquetFile(item).metadata.num_rows
            for item in physical_partitions
        )
    )


def assert_authorized_canonical_table(frame: pd.DataFrame, *, name: str) -> None:
    """Reject forbidden boundary metadata without scanning free business text."""

    forbidden_columns = [
        str(column)
        for column in frame.columns
        if FORBIDDEN_SCHEMA_TOKEN.search(str(column))
    ]
    if forbidden_columns:
        raise ValueError(f"{name} contains forbidden columns: {forbidden_columns}")
    for column in frame.columns:
        column_name = str(column)
        if column_name not in ROLE_COLUMNS and not column_name.endswith("_role"):
            continue
        values = frame[column].dropna().astype(str).str.strip()
        forbidden = sorted(
            value for value in values.unique() if FORBIDDEN_ROLE_VALUE.search(value)
        )
        if forbidden:
            raise ValueError(
                f"{name}.{column_name} contains forbidden roles: {forbidden}"
            )


def load_frozen_population(
    *,
    source_dataset_dir: Path,
    split_assignments_path: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Load V4.1 queries only; labels remain closed until pools are frozen."""

    source_dataset_dir = Path(source_dataset_dir).resolve()
    manifest_path = source_dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "sireto-v4.1-training-dataset-1":
        raise ValueError("Unsupported frozen V4.1 dataset schema")
    if manifest.get("build_id") != "f938abf6b8a87155":
        raise ValueError("Unexpected frozen V4.1 dataset build")
    if manifest.get("positive_injection") is not False:
        raise ValueError("Frozen V4.1 population permits positive injection")
    if list(manifest.get("feature_order") or []) != list(FEATURE_ORDER):
        raise ValueError("Frozen V4.1 candidate feature order changed")
    queries_path = source_dataset_dir / "queries.parquet"
    labels_path = source_dataset_dir / "labels.parquet"
    for filename, path in (
        ("queries.parquet", queries_path),
        ("labels.parquet", labels_path),
    ):
        expected = EXPECTED_INPUT_HASHES[filename]
        declared = (manifest.get("outputs") or {}).get(filename)
        if declared != expected or file_sha256(path) != expected:
            raise ValueError(f"Frozen V4.1 input hash mismatch: {filename}")
    # The old candidate file is deliberately not opened. Its declared hash
    # identifies ranker A's context and must remain frozen.
    if (manifest.get("outputs") or {}).get("candidates.parquet") != (
        EXPECTED_INPUT_HASHES["legacy_candidates.parquet"]
    ):
        raise ValueError("V4.1 legacy candidate context hash mismatch")
    if file_sha256(split_assignments_path) != EXPECTED_INPUT_HASHES[
        "split_assignments.parquet"
    ]:
        raise ValueError("Frozen V4.1 split assignments hash mismatch")

    queries = pd.read_parquet(queries_path)
    assert_authorized_canonical_table(queries, name="queries")
    label_row_count = pq.ParquetFile(labels_path).metadata.num_rows
    if len(queries) != EXPECTED_QUERY_COUNT or label_row_count != EXPECTED_QUERY_COUNT:
        raise ValueError("V4.6 requires exactly 7,003 queries and labels")
    if queries["query_id"].astype(str).duplicated().any():
        raise ValueError("V4.1 query_id must be unique")
    return manifest, queries


def validate_frozen_assignments(
    *,
    queries: pd.DataFrame,
    labels: pd.DataFrame,
    assignments: pd.DataFrame,
    enforce_contract_counts: bool = True,
) -> None:
    """Rebuild the input/target-SIREN graph and require identical boundaries."""

    required = {"query_id", "siren_component_id", "split", "oof_fold"}
    missing = required - set(assignments.columns)
    if missing:
        raise ValueError(f"Split assignments missing: {sorted(missing)}")
    assert_authorized_canonical_table(assignments, name="split_assignments")
    if assignments["query_id"].astype(str).duplicated().any():
        raise ValueError("Split assignment query_id must be unique")
    if set(assignments["query_id"].astype(str)) != set(
        queries["query_id"].astype(str)
    ):
        raise ValueError("Split assignment population differs from queries")
    validate_connected_siren_split(assignments)
    source = queries[
        ["query_id", "input_siret", "input_siren"]
    ].merge(
        labels[
            ["query_id", "ground_truth_siret", "ground_truth_siren"]
        ],
        on="query_id",
        validate="one_to_one",
    ).rename(
        columns={
            "ground_truth_siret": "target_siret",
            "ground_truth_siren": "target_siren",
        }
    )
    rebuilt = assign_connected_siren_splits(
        source,
        dev_fraction=0.2,
        oof_folds=5,
        seed=SEED,
    )[list(required)]
    columns = ["query_id", "siren_component_id", "split", "oof_fold"]
    observed = assignments[columns].sort_values("query_id").reset_index(drop=True)
    expected = rebuilt[columns].sort_values("query_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(observed, expected, check_dtype=False)
    if enforce_contract_counts:
        split_counts = assignments["split"].value_counts().to_dict()
        if split_counts != {"fit": EXPECTED_FIT_COUNT, "dev": EXPECTED_DEV_COUNT}:
            raise ValueError(f"Frozen split counts changed: {split_counts}")
        dev_ids = set(
            assignments.loc[assignments["split"].eq("dev"), "query_id"].astype(str)
        )
        dev_exact = labels[
            labels["query_id"].astype(str).isin(dev_ids)
            & labels["label_kind"].astype(str).eq("MATCH_EXACT")
        ]
        if len(dev_exact) != EXPECTED_DEV_EXACT_COUNT:
            raise ValueError("Frozen dev exact count changed")


def validate_v42_runtime(
    *,
    manifest_path: Path,
    config: V41RetrievalConfig,
    partitions_dir: Path,
    global_store_path: Path,
    state_snapshot_path: Path,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    if file_sha256(manifest_path) != EXPECTED_INPUT_HASHES["v42_manifest"]:
        raise ValueError("Frozen V4.2 manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version")
        != "sireto-v4.2-representative-retrieval-evaluation-1"
    ):
        raise ValueError("Unsupported V4.2 retrieval manifest")
    if manifest.get("positive_injection") is not False:
        raise ValueError("V4.2 retrieval manifest permits positive injection")
    if manifest.get("verdict") != "GO_HARD_LABELS":
        raise ValueError("V4.2 retrieval integrity verdict is not GO_HARD_LABELS")
    retrieval = manifest.get("retrieval") or {}
    if config.signature() != EXPECTED_V42_CONFIG_SIGNATURE:
        raise ValueError("Runtime is not the frozen V4.2-B configuration")
    if retrieval.get("v41_signature") != config.signature():
        raise ValueError("V4.2 retrieval configuration signature mismatch")
    if retrieval.get("v41_config") != config.to_dict():
        raise ValueError("V4.2 retrieval configuration payload mismatch")
    inputs = manifest.get("inputs") or {}
    partition_signature = _path_signature(partitions_dir)
    global_signature = _path_signature(global_store_path)
    state_hash = file_sha256(state_snapshot_path)
    checks = {
        "partitions": (
            partition_signature,
            (inputs.get("partitions") or {}).get("runtime_signature"),
        ),
        "global_store": (
            global_signature,
            (inputs.get("global_store") or {}).get("runtime_signature"),
        ),
        "state_snapshot": (
            state_hash,
            (inputs.get("current_state_snapshot") or {}).get("sha256"),
        ),
    }
    for name, (observed, expected) in checks.items():
        if observed != expected:
            raise ValueError(f"V4.2 runtime input mismatch: {name}")
    paths = {
        "partitions": (
            Path(partitions_dir),
            Path(str((inputs.get("partitions") or {}).get("path") or "")),
        ),
        "global_store": (
            Path(global_store_path),
            Path(str((inputs.get("global_store") or {}).get("path") or "")),
        ),
        "state_snapshot": (
            Path(state_snapshot_path),
            Path(
                str(
                    (inputs.get("current_state_snapshot") or {}).get("path")
                    or ""
                )
            ),
        ),
    }
    for name, (observed, expected) in paths.items():
        if observed.resolve() != expected.resolve():
            raise ValueError(f"V4.2 runtime path mismatch: {name}")
    if state_hash != EXPECTED_INPUT_HASHES["state_snapshot"]:
        raise ValueError("Authoritative SIRENE snapshot hash mismatch")
    return {
        "manifest_sha256": file_sha256(manifest_path),
        "config_signature": config.signature(),
        "partitions_signature": partition_signature,
        "global_store_signature": global_signature,
        "state_snapshot_sha256": state_hash,
        "verdict": str(manifest.get("verdict") or ""),
    }


def _annotate_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(candidate)
    ranks = set((output.get("v41_channel_ranks") or {}).keys())
    output["candidate_from_sparse"] = "sparse_active" in ranks
    output["candidate_from_input_siret"] = "input_siret_active" in ranks
    output["candidate_from_input_siren"] = "input_siren_active_sites" in ranks
    output["candidate_from_closed_alias"] = bool(
        {"closed_alias_name", "closed_alias_address"} & ranks
    )
    return output


def retrieve_unlabelled_query(
    *,
    query: Mapping[str, Any],
    retriever: Any,
    persistent_cache: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Retrieve and featurize one query without accepting any truth argument."""

    query_id = str(query["query_id"])
    crm_row = {
        "query_id": query_id,
        "crm_id": query_id,
        "crm_name": str(query.get("crm_name") or ""),
        "crm_address": str(query.get("crm_address") or ""),
        "crm_city": str(query.get("crm_city") or ""),
        "postcode": str(query.get("crm_postcode") or ""),
        "insee": str(query.get("crm_insee") or ""),
    }
    crm_pre = preprocess_crm_row(crm_row)
    result = retriever.build(
        crm_row=crm_row,
        crm_pre=crm_pre,
        input_siret=query.get("input_siret"),
        gt_siret=None,
        persistent_cache=persistent_cache,
    )
    if result.sparse_result.gt_was_injected:
        raise ValueError("Positive injection is forbidden in V4.6")
    candidates = [_annotate_candidate(item) for item in result.candidates]
    if len(candidates) > 100:
        raise ValueError("V4.6 retrieval exceeded 100 candidates")
    sirets = [str(item.get("siret") or "") for item in candidates]
    if len(sirets) != len(set(sirets)):
        raise ValueError("V4.6 retrieval returned duplicate SIRETs")
    if any(str(item.get("etat_admin") or "").upper() != "A" for item in candidates):
        raise ValueError("V4.6 retrieval returned a non-active candidate")
    set_global_name_idf_map(
        result.sparse_result.idf_map,
        result.sparse_result.default_idf,
    )
    legacy_rows = make_feature_rows_from_preprocessed(
        crm_pre, candidates, include_semantic=False
    )
    rows: list[dict[str, Any]] = []
    for rank, (candidate, legacy) in enumerate(
        zip(candidates, legacy_rows, strict=True), start=1
    ):
        siret = str(candidate["siret"])
        channel_count = int(
            candidate.get("retrieval_channel_count")
            or len(candidate.get("v41_channel_ranks") or {})
        )
        rows.append(
            {
                "query_id": query_id,
                "candidate_siret": siret,
                "candidate_siren": str(candidate.get("siren") or siret[:9]),
                "candidate_state": str(
                    candidate.get("etat_admin") or ""
                ).upper(),
                # It remains zero until every pool has been closed.
                "is_ground_truth": 0,
                "retrieval_rank": rank,
                "retrieval_source": str(
                    candidate.get("retrieval_source") or "v4.2-b"
                ),
                "retrieval_channel_count": channel_count,
                "retrieval_agreement": int(channel_count >= 2),
                **build_legacy_55_features(legacy, candidate),
                **build_v41_candidate_features(
                    candidate,
                    input_siret=result.input_siret.normalized_siret,
                ),
            }
        )
    return rows, {
        "query_id": query_id,
        "candidate_count": len(rows),
        "input_siret_state_runtime": result.input_siret.state.value,
    }


def label_closed_candidate_file(
    *,
    unlabelled_path: Path,
    output_path: Path,
    labels: pd.DataFrame,
) -> int:
    """Join exact truth only after retrieval has closed every candidate pool."""

    exact = labels.loc[
        labels["label_kind"].astype(str).eq("MATCH_EXACT"),
        ["query_id", "ground_truth_siret"],
    ].copy()
    truth = dict(
        zip(
            exact["query_id"].astype(str),
            exact["ground_truth_siret"].fillna("").astype(str),
            strict=True,
        )
    )
    source = pq.ParquetFile(unlabelled_path)
    writer: pq.ParquetWriter | None = None
    count = 0
    try:
        for batch in source.iter_batches(batch_size=25_000):
            frame = batch.to_pandas()
            frame["is_ground_truth"] = [
                int(str(siret) == truth.get(str(query_id), ""))
                for query_id, siret in zip(
                    frame["query_id"], frame["candidate_siret"], strict=True
                )
            ]
            table = pa.Table.from_pandas(
                frame[CANDIDATE_COLUMNS],
                schema=batch.schema,
                preserve_index=False,
            )
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)
            count += len(frame)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        CandidateWriter(output_path).close()
    return count


def ordered_pool_content_sha256(candidates: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    ordered = candidates.sort_values(
        ["query_id", "retrieval_rank"], kind="stable"
    )
    for row in ordered[
        ["query_id", "retrieval_rank", "candidate_siret"]
    ].itertuples(index=False):
        digest.update(
            f"{row.query_id}\0{int(row.retrieval_rank)}\0"
            f"{row.candidate_siret}\n".encode()
        )
    return digest.hexdigest()


def candidate_content_sha256(candidates: pd.DataFrame) -> str:
    """Hash every canonical value independently from Parquet metadata."""

    if list(candidates.columns) != list(CANDIDATE_COLUMNS):
        raise ValueError("Candidate columns are not in the frozen canonical order")
    ordered = candidates.sort_values(
        ["query_id", "retrieval_rank", "candidate_siret"],
        kind="stable",
    ).reset_index(drop=True)
    row_hashes = pd.util.hash_pandas_object(
        ordered[CANDIDATE_COLUMNS],
        index=False,
        categorize=False,
    ).to_numpy()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            list(CANDIDATE_COLUMNS),
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode()
    )
    digest.update(str(ordered.dtypes.astype(str).tolist()).encode())
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()


def compute_integrity(
    *,
    queries: pd.DataFrame,
    labels: pd.DataFrame,
    assignments: pd.DataFrame,
    candidates: pd.DataFrame,
) -> dict[str, Any]:
    if list(candidates.columns) != list(CANDIDATE_COLUMNS):
        raise ValueError("V4.6 candidate schema differs from the frozen 64 features")
    if not set(candidates["query_id"].astype(str)).issubset(
        set(queries["query_id"].astype(str))
    ):
        raise ValueError("Candidates contain an unknown query")
    if int(candidates["retrieval_rank"].max()) > 100:
        raise ValueError("Candidate rank exceeds 100")
    duplicates = candidates.duplicated(["query_id", "candidate_siret"])
    if duplicates.any():
        raise ValueError("A V4.6 pool contains duplicate SIRETs")
    if candidates["candidate_state"].astype(str).ne("A").any():
        raise ValueError("A V4.6 candidate is not active")
    rank_counts = candidates.groupby("query_id")["retrieval_rank"].apply(
        lambda values: list(sorted(values.astype(int)))
        == list(range(1, len(values) + 1))
    )
    if not rank_counts.all():
        raise ValueError("Candidate ranks are not contiguous within a pool")
    truth = labels.set_index("query_id")["ground_truth_siret"]
    def truth_text(query_id: Any) -> str:
        value = truth.get(str(query_id))
        return "" if value is None or pd.isna(value) else str(value)
    expected_target = [
        int(
            str(candidate_siret)
            == truth_text(query_id)
        )
        for query_id, candidate_siret in zip(
            candidates["query_id"], candidates["candidate_siret"], strict=True
        )
    ]
    if expected_target != candidates["is_ground_truth"].astype(int).tolist():
        raise ValueError("Candidate targets are inconsistent with frozen labels")
    merged_labels = labels.merge(
        assignments[["query_id", "split"]],
        on="query_id",
        validate="one_to_one",
    )
    hit_queries = set(
        candidates.loc[candidates["is_ground_truth"].eq(1), "query_id"].astype(str)
    )
    recall: dict[str, Any] = {}
    for split in ("fit", "dev"):
        exact_ids = set(
            merged_labels.loc[
                merged_labels["split"].eq(split)
                & merged_labels["label_kind"].eq("MATCH_EXACT"),
                "query_id",
            ].astype(str)
        )
        successes = len(exact_ids & hit_queries)
        recall[split] = {
            "successes": successes,
            "total": len(exact_ids),
            "rate": successes / len(exact_ids) if exact_ids else 0.0,
        }
    counts = candidates.groupby("query_id").size()
    return {
        "query_count": int(len(queries)),
        "label_count": int(len(labels)),
        "assignment_count": int(len(assignments)),
        "candidate_count": int(len(candidates)),
        "max_candidate_count": int(counts.max()) if len(counts) else 0,
        "zero_candidate_query_count": int(
            len(queries) - candidates["query_id"].nunique()
        ),
        "closed_candidate_count": 0,
        "duplicate_candidate_count": 0,
        "positive_injection": False,
        "recall_at_100": recall,
        "ordered_pool_content_sha256": ordered_pool_content_sha256(candidates),
        "candidate_content_sha256": candidate_content_sha256(candidates),
    }


def _dependency_versions() -> dict[str, str]:
    output = {}
    for package in ("pandas", "pyarrow", "duckdb", "scikit-learn"):
        try:
            output[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            output[package] = "missing"
    return output


def build_artifact(
    *,
    source_dataset_dir: Path,
    split_assignments_path: Path,
    partitions_dir: Path,
    global_store_path: Path,
    state_snapshot_path: Path,
    v42_manifest_path: Path,
    cache_dir: Path,
    work_dir: Path,
    output_root: Path,
    contract_path: Path = DEFAULT_CONTRACT,
) -> Path:
    output_root = _external_path(output_root, name="output_root")
    cache_dir = _external_path(cache_dir, name="cache_dir")
    work_dir = _external_path(work_dir, name="work_dir")
    output_root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    source_manifest, queries = load_frozen_population(
        source_dataset_dir=source_dataset_dir,
        split_assignments_path=split_assignments_path,
    )
    assignments = pd.read_parquet(split_assignments_path)
    config = V41RetrievalConfig(
        variant=V41RetrievalVariant.B_INPUT_EVIDENCE,
        max_candidates=100,
    )
    runtime = validate_v42_runtime(
        manifest_path=v42_manifest_path,
        config=config,
        partitions_dir=partitions_dir,
        global_store_path=global_store_path,
        state_snapshot_path=state_snapshot_path,
    )
    validate_v41_model_feature_order(FEATURE_ORDER)
    script_path = Path(__file__).resolve()
    identity = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "seed": SEED,
        "contract_sha256": file_sha256(contract_path),
        "builder_source_sha256": file_sha256(script_path),
        "source_manifest_sha256": file_sha256(
            Path(source_dataset_dir) / "manifest.json"
        ),
        "queries_sha256": EXPECTED_INPUT_HASHES["queries.parquet"],
        "labels_sha256": EXPECTED_INPUT_HASHES["labels.parquet"],
        "split_assignments_sha256": EXPECTED_INPUT_HASHES[
            "split_assignments.parquet"
        ],
        "legacy_candidates_context_sha256": EXPECTED_INPUT_HASHES[
            "legacy_candidates.parquet"
        ],
        "v42_manifest_sha256": runtime["manifest_sha256"],
        "retrieval_config_signature": config.signature(),
        "partitions_signature": runtime["partitions_signature"],
        "global_store_signature": runtime["global_store_signature"],
        "state_snapshot_sha256": runtime["state_snapshot_sha256"],
        "feature_order": list(FEATURE_ORDER),
        "positive_injection": False,
    }
    if identity["contract_sha256"] != EXPECTED_CONTRACT_SHA256:
        raise ValueError("Active V4.6 contract hash mismatch")
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    target = output_root / build_id
    if target.exists():
        raise FileExistsError(f"Immutable V4.6 dataset exists: {target}")
    staging = Path(tempfile.mkdtemp(prefix=f".{build_id}.tmp-", dir=output_root))
    unlabelled_path = staging / ".candidates_unlabelled.parquet"
    writer = CandidateWriter(unlabelled_path)
    retrieval_diagnostics: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        cache = TfidfPersistentCache(
            config_hash=config.sparse_config().tfidf_artifact_hash(),
            cache_dir=cache_dir,
        )
        partitioned_store = PartitionedCandidateStore(partitions_dir)
        with (
            V41GlobalCandidateStore(global_store_path) as global_store,
            V41CurrentStateStore(state_snapshot_path) as state_store,
        ):
            duckdb_temp = work_dir / f"duckdb-{build_id}"
            duckdb_temp.mkdir(exist_ok=False)
            escaped_temp = str(duckdb_temp).replace("'", "''")
            state_store._connection.execute(  # noqa: SLF001
                f"SET temp_directory = '{escaped_temp}'"
            )
            retriever = V41CandidateRetriever(
                partitioned_store=partitioned_store,
                global_store=global_store,
                current_state_store=state_store,
                config=config,
            )
            buffer: list[dict[str, Any]] = []
            for query in queries.sort_values("query_id", kind="stable").to_dict(
                "records"
            ):
                rows, diagnostics = retrieve_unlabelled_query(
                    query=query,
                    retriever=retriever,
                    persistent_cache=cache,
                )
                buffer.extend(rows)
                retrieval_diagnostics.append(diagnostics)
                if len(buffer) >= 10_000:
                    writer.write(buffer)
                    buffer.clear()
            writer.write(buffer)
            writer.close()
            writer = None

        # Retrieval is now closed. Only at this boundary are labels reopened.
        labels = pd.read_parquet(Path(source_dataset_dir) / "labels.parquet")
        assert_authorized_canonical_table(labels, name="labels")
        if labels["query_id"].astype(str).duplicated().any():
            raise ValueError("V4.1 label query_id must be unique")
        if set(queries["query_id"].astype(str)) != set(
            labels["query_id"].astype(str)
        ):
            raise ValueError("V4.1 query and label identifiers differ")
        validate_frozen_assignments(
            queries=queries,
            labels=labels,
            assignments=assignments,
        )
        candidates_path = staging / "candidates_v42b.parquet"
        labelled_count = label_closed_candidate_file(
            unlabelled_path=unlabelled_path,
            output_path=candidates_path,
            labels=labels,
        )
        unlabelled_path.unlink()
        candidates = pd.read_parquet(candidates_path)
        if labelled_count != len(candidates):
            raise ValueError("Candidate labelling cardinality changed")
        integrity = compute_integrity(
            queries=queries,
            labels=labels,
            assignments=assignments,
            candidates=candidates,
        )
        shutil.copyfile(
            Path(source_dataset_dir) / "queries.parquet",
            staging / "queries.parquet",
        )
        shutil.copyfile(
            Path(source_dataset_dir) / "labels.parquet",
            staging / "labels.parquet",
        )
        shutil.copyfile(
            split_assignments_path,
            staging / "split_assignments.parquet",
        )
        total_seconds = time.perf_counter() - started
        summary = {
            **integrity,
            "total_seconds": total_seconds,
            "input_siret_state_runtime_counts": pd.Series(
                [
                    item["input_siret_state_runtime"]
                    for item in retrieval_diagnostics
                ]
            ).value_counts().to_dict(),
            "integrity_verdict": "GO_TRAIN_RANKER",
        }
        summary_path = staging / "integrity_report.json"
        _json_dump(summary_path, summary)
        output_paths = [
            staging / "queries.parquet",
            staging / "labels.parquet",
            candidates_path,
            staging / "split_assignments.parquet",
            summary_path,
        ]
        outputs = {
            path.name: {
                "sha256": file_sha256(path),
                "size_bytes": int(path.stat().st_size),
                "row_count": (
                    int(
                        pq.ParquetFile(path).metadata.num_rows
                    )
                    if path.suffix == ".parquet"
                    else None
                ),
            }
            for path in output_paths
        }
        manifest = {
            **identity,
            "build_identity": identity,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_dataset_build_id": str(source_manifest.get("build_id") or ""),
            "inputs": {
                "source_manifest": _input_record(
                    Path(source_dataset_dir) / "manifest.json"
                ),
                "queries": _input_record(
                    Path(source_dataset_dir) / "queries.parquet",
                    row_count=len(queries),
                ),
                "labels": _input_record(
                    Path(source_dataset_dir) / "labels.parquet",
                    row_count=len(labels),
                ),
                "split_assignments": _input_record(
                    split_assignments_path,
                    row_count=len(assignments),
                ),
                "legacy_candidates_context": {
                    "path": str(
                        (Path(source_dataset_dir) / "candidates.parquet").resolve()
                    ),
                    "sha256": EXPECTED_INPUT_HASHES[
                        "legacy_candidates.parquet"
                    ],
                    "size_bytes": int(
                        (
                            Path(source_dataset_dir) / "candidates.parquet"
                        ).stat().st_size
                    ),
                    "row_count": int(
                        source_manifest.get("row_counts", {}).get(
                            "candidates", 0
                        )
                    ),
                    "opened": False,
                },
                "v42_manifest": _input_record(v42_manifest_path),
                "partitions": _directory_record(
                    partitions_dir,
                    runtime_signature=runtime["partitions_signature"],
                    row_count=_physical_parquet_row_count(partitions_dir),
                ),
                "global_store": _directory_record(
                    global_store_path,
                    runtime_signature=runtime["global_store_signature"],
                    row_count=14_378_332,
                ),
                "state_snapshot": _input_record(
                    state_snapshot_path,
                    row_count=42_322_035,
                ),
                "contract": _input_record(contract_path),
                "builder_source": _input_record(script_path),
            },
            "retrieval": {
                "config": config.to_dict(),
                "config_signature": config.signature(),
                "candidate_ceiling": 100,
                "state_authority": "COMPLETE_SIRENE_SNAPSHOT",
            },
            "outputs": outputs,
            "integrity": summary,
            "timing": {"total_seconds": total_seconds},
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "logical_cpu_count": os.cpu_count(),
                "dependencies": _dependency_versions(),
                "work_dir": str(work_dir),
            },
            "invariants": {
                "positive_injection": False,
                "truth_joined_after_all_pools_closed": True,
                "legacy_candidates_opened": False,
                "acceptor_loaded_or_scored": False,
                "training_performed": False,
                "threshold_selected_or_applied": False,
                "test_holdout_v44_random_opened": False,
                "candidate_ceiling": 100,
                "active_candidates_only": True,
                "frozen_splits_reused_and_reaudited": True,
            },
        }
        _json_dump(staging / "manifest.json", manifest)
        os.replace(staging, target)
        shutil.rmtree(duckdb_temp, ignore_errors=True)
    except BaseException:
        if writer is not None:
            writer.close()
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(work_dir / f"duckdb-{build_id}", ignore_errors=True)
        raise
    return target


def validate_artifact(artifact_dir: Path) -> None:
    artifact_dir = Path(artifact_dir)
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported V4.6 dataset schema")
    if manifest.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("Unexpected V4.6 experiment identifier")
    identity = manifest.get("build_identity") or {}
    expected_build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    if manifest.get("build_id") != expected_build_id:
        raise ValueError("V4.6 content-addressed build identity mismatch")
    if artifact_dir.name != expected_build_id:
        raise ValueError("V4.6 artifact directory is not its build ID")
    for filename, record in (manifest.get("outputs") or {}).items():
        path = artifact_dir / filename
        if file_sha256(path) != record.get("sha256"):
            raise ValueError(f"V4.6 output hash mismatch: {filename}")
        if int(path.stat().st_size) != int(record.get("size_bytes")):
            raise ValueError(f"V4.6 output size mismatch: {filename}")
    queries = pd.read_parquet(artifact_dir / "queries.parquet")
    labels = pd.read_parquet(artifact_dir / "labels.parquet")
    assignments = pd.read_parquet(artifact_dir / "split_assignments.parquet")
    candidates = pd.read_parquet(artifact_dir / "candidates_v42b.parquet")
    if file_sha256(artifact_dir / "queries.parquet") != EXPECTED_INPUT_HASHES[
        "queries.parquet"
    ]:
        raise ValueError("V4.6 queries are not byte-identical to V4.1")
    if file_sha256(artifact_dir / "labels.parquet") != EXPECTED_INPUT_HASHES[
        "labels.parquet"
    ]:
        raise ValueError("V4.6 labels are not byte-identical to V4.1")
    if file_sha256(
        artifact_dir / "split_assignments.parquet"
    ) != EXPECTED_INPUT_HASHES["split_assignments.parquet"]:
        raise ValueError("V4.6 split assignments are not byte-identical to V4.1")
    assert_authorized_canonical_table(queries, name="queries")
    assert_authorized_canonical_table(labels, name="labels")
    validate_frozen_assignments(
        queries=queries,
        labels=labels,
        assignments=assignments,
    )
    observed = compute_integrity(
        queries=queries,
        labels=labels,
        assignments=assignments,
        candidates=candidates,
    )
    declared = manifest.get("integrity") or {}
    for key, value in observed.items():
        if declared.get(key) != value:
            raise ValueError(f"V4.6 integrity mismatch: {key}")
    for name, record in (manifest.get("inputs") or {}).items():
        if name == "legacy_candidates_context":
            continue
        if name in {"partitions", "global_store"}:
            path = Path(str(record.get("path") or ""))
            if _path_signature(path) != record.get("runtime_signature"):
                raise ValueError(f"V4.6 input signature mismatch: {name}")
            continue
        path_text = record.get("path")
        expected_hash = record.get("sha256")
        if path_text and expected_hash:
            path = Path(path_text)
            if not path.is_file() or file_sha256(path) != expected_hash:
                raise ValueError(f"V4.6 input hash mismatch: {name}")
    if (manifest.get("invariants") or {}).get("training_performed") is not False:
        raise ValueError("V4.6 builder manifest suggests training occurred")
    total_seconds = (manifest.get("timing") or {}).get("total_seconds")
    if not isinstance(total_seconds, (int, float)) or total_seconds <= 0:
        raise ValueError("V4.6 total construction timing is missing")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", type=Path)
    parser.add_argument("--split-assignments", type=Path)
    parser.add_argument("--partitions", type=Path)
    parser.add_argument("--global-store", type=Path)
    parser.add_argument("--state-snapshot", type=Path)
    parser.add_argument("--v42-manifest", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--validate-artifact", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_artifact:
        validate_artifact(args.validate_artifact)
        print(args.validate_artifact)
        return
    names = (
        "source_dataset",
        "split_assignments",
        "partitions",
        "global_store",
        "state_snapshot",
        "v42_manifest",
        "cache_dir",
        "work_dir",
        "output_root",
    )
    missing = [name for name in names if getattr(args, name) is None]
    if missing:
        raise SystemExit(f"Missing required arguments: {', '.join(missing)}")
    target = build_artifact(
        source_dataset_dir=args.source_dataset,
        split_assignments_path=args.split_assignments,
        partitions_dir=args.partitions,
        global_store_path=args.global_store,
        state_snapshot_path=args.state_snapshot,
        v42_manifest_path=args.v42_manifest,
        cache_dir=args.cache_dir,
        work_dir=args.work_dir,
        output_root=args.output_root,
        contract_path=args.contract,
    )
    print(target)


if __name__ == "__main__":
    main()
