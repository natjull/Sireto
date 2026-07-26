#!/usr/bin/env python3
"""Build train/dev candidate features from the frozen selective top-100 lists."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate_retrieval_admission import V7_WEIGHTS  # noqa: E402
from scripts.run_v9_retrieval_experiment import _git_commit  # noqa: E402
from src.xgb_matcher.blocking import attach_address_density  # noqa: E402
from src.xgb_matcher.candidates import compute_name_idf_map  # noqa: E402
from src.xgb_matcher.features import (  # noqa: E402
    SEMANTIC_FEATURE_NAMES,
    V9_BASELINE_FEATURE_NAMES,
    make_feature_rows_from_preprocessed,
    preprocess_crm_row,
    set_global_name_idf_map,
)
from src.xgb_matcher.partitioned_store import PartitionedCandidateStore  # noqa: E402
from src.xgb_matcher.v9_dataset import (  # noqa: E402
    LABEL_COLUMNS,
    QUERY_COLUMNS,
    SCHEMA_VERSION,
    V9DatasetManifest,
    file_sha256,
)
from src.xgb_matcher.v9_features import (  # noqa: E402
    SELECTIVE_RETRIEVAL_CHANNELS,
    SELECTIVE_RETRIEVAL_FEATURE_NAMES,
)


DATASET_FEATURE_ORDER = [
    feature
    for feature in V9_BASELINE_FEATURE_NAMES
    if feature not in SEMANTIC_FEATURE_NAMES
] + SELECTIVE_RETRIEVAL_FEATURE_NAMES
OUTPUT_CANDIDATE_COLUMNS = [
    "query_id",
    "candidate_siret",
    "candidate_siren",
    "split",
    "is_ground_truth",
    "sparse_rank",
    "dense_rank",
    "rrf_score",
    "retrieval_rank",
    "retrieval_source",
    "retrieval_channel_count",
    "retrieval_agreement",
    *V9_BASELINE_FEATURE_NAMES,
    *SELECTIVE_RETRIEVAL_FEATURE_NAMES,
]


def _load_json_list(row: pd.Series, channel: str) -> list[str]:
    return [
        str(value).zfill(14)
        for value in json.loads(row[f"{channel}_sirets_json"])
    ]


def selective_provenance(
    selected: list[str],
    v7_row: pd.Series,
    overlay_row: pd.Series,
) -> dict[str, dict[str, float]]:
    v7_lists = {
        channel: _load_json_list(v7_row, channel)
        for channel in SELECTIVE_RETRIEVAL_CHANNELS
    }
    overlay_lists = {
        channel: _load_json_list(overlay_row, channel)
        for channel in SELECTIVE_RETRIEVAL_CHANNELS
    }
    v7_ranks = {
        channel: {
            siret: rank
            for rank, siret in enumerate(values, start=1)
        }
        for channel, values in v7_lists.items()
    }
    overlay_sets = {
        channel: set(values)
        for channel, values in overlay_lists.items()
    }
    overlay_quota_candidates: list[str] = []
    overlay_quota_seen: set[str] = set()
    for channel, quota in (("name_word", 1), ("name_char", 10)):
        for siret in overlay_lists[channel][:quota]:
            if siret not in overlay_quota_seen:
                overlay_quota_seen.add(siret)
                overlay_quota_candidates.append(siret)
    overlay_quota_set = set(overlay_quota_candidates)
    output: dict[str, dict[str, float]] = {}
    for admission_rank, siret in enumerate(selected, start=1):
        fusion_score = sum(
            V7_WEIGHTS[channel] / rank_map[siret]
            for channel, rank_map in v7_ranks.items()
            if siret in rank_map
        )
        channel_count = sum(
            siret in v7_ranks[channel] or siret in overlay_sets[channel]
            for channel in SELECTIVE_RETRIEVAL_CHANNELS
        )
        overlay_quota = float(siret in overlay_quota_set)
        features = {
            "admission_rank_recip": 1.0 / admission_rank,
            "admission_fusion_score": float(fusion_score),
            "admission_channel_count": float(channel_count),
            "admission_overlay_quota": overlay_quota,
        }
        for channel in SELECTIVE_RETRIEVAL_CHANNELS:
            rank = v7_ranks[channel].get(siret)
            features[f"admission_{channel}_rank_recip"] = (
                1.0 / rank if rank else 0.0
            )
        output[siret] = features
    return output


def _partition_rows(
    store: PartitionedCandidateStore,
    partition_key: str,
) -> list[dict[str, Any]]:
    if not partition_key or partition_key.lower() == "nan":
        return []
    if partition_key.endswith("_") and not partition_key.startswith("_"):
        return store.load_by_insee(partition_key[:-1])
    if partition_key.startswith("_"):
        return store.load_by_postcode(partition_key[1:])
    raise ValueError(f"Unsupported partition key: {partition_key!r}")


def _candidate_schema() -> pa.Schema:
    string_columns = {
        "query_id",
        "candidate_siret",
        "candidate_siren",
        "split",
        "retrieval_source",
    }
    integer_columns = {"is_ground_truth", "retrieval_rank"}
    fields = []
    for column in OUTPUT_CANDIDATE_COLUMNS:
        if column in string_columns:
            fields.append(pa.field(column, pa.string()))
        elif column in integer_columns:
            fields.append(pa.field(column, pa.int32()))
        else:
            fields.append(pa.field(column, pa.float32()))
    return pa.schema(fields)


class CandidateWriter:
    def __init__(self, path: Path):
        self.path = path
        self.schema = _candidate_schema()
        self.writer = pq.ParquetWriter(path, self.schema)
        self.count = 0

    def write(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        frame = pd.DataFrame(rows, columns=OUTPUT_CANDIDATE_COLUMNS)
        self.writer.write_table(
            pa.Table.from_pandas(
                frame,
                schema=self.schema,
                preserve_index=False,
            )
        )
        self.count += len(rows)

    def close(self) -> None:
        self.writer.close()


def _canonical_queries(benchmark: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(
        {
            "query_id": benchmark["query_id"].astype(str),
            "crm_name": benchmark["crm_name"].fillna(""),
            "crm_address": benchmark["crm_address"].fillna(""),
            "crm_postcode": benchmark["postcode"].fillna(""),
            "crm_city": benchmark["crm_city"].fillna(""),
            "crm_insee": benchmark["insee"].fillna(""),
            "crm_name_norm": benchmark["crm_name"].fillna(""),
            "crm_address_norm": benchmark["crm_address"].fillna(""),
            "crm_city_norm": benchmark["crm_city"].fillna(""),
            "reference_date": benchmark["reference_date"].fillna(""),
            "split": benchmark["split"],
        }
    )
    from src.xgb_matcher.features import normalize_text

    output["crm_name_norm"] = output["crm_name"].map(normalize_text)
    output["crm_address_norm"] = output["crm_address"].map(normalize_text)
    output["crm_city_norm"] = output["crm_city"].map(normalize_text)
    return output[QUERY_COLUMNS]


def _canonical_labels(qualified: pd.DataFrame, snapshot_id: str) -> pd.DataFrame:
    output = pd.DataFrame(
        {
            "query_id": qualified["query_id"].astype(str),
            "label_kind": qualified["label_kind"].astype(str),
            "ground_truth_siret": qualified["ground_truth_siret"],
            "ground_truth_siren": qualified["ground_truth_siren"],
            "label_source": "qualification_v3_direct_evidence",
            "validator": qualified.get("validator", ""),
            "validated_at": "",
            "sirene_snapshot_id": snapshot_id,
            "split": qualified["split"],
        }
    )
    return output[LABEL_COLUMNS]


def build_split_candidates(
    *,
    split: str,
    benchmark: pd.DataFrame,
    labels: pd.DataFrame,
    admission: pd.DataFrame,
    v7_channels: pd.DataFrame,
    overlay_channels: pd.DataFrame,
    v7_store: PartitionedCandidateStore,
    overlay_store: PartitionedCandidateStore,
    writer: CandidateWriter,
) -> dict[str, int]:
    benchmark_by_query = benchmark.set_index("query_id", drop=False)
    labels_by_query = labels.set_index("query_id", drop=False)
    admission_by_query = admission.set_index("query_id", drop=False)
    v7_by_query = v7_channels.set_index("query_id", drop=False)
    overlay_by_query = overlay_channels.set_index("query_id", drop=False)
    expected = set(benchmark_by_query.index)
    for name, frame in (
        ("labels", labels_by_query),
        ("admission", admission_by_query),
        ("v7 channels", v7_by_query),
        ("overlay channels", overlay_by_query),
    ):
        if set(frame.index) != expected:
            raise ValueError(f"{split}: {name} query IDs do not align")

    missing_details = 0
    duplicate_sirets = 0
    over_budget = 0
    partition_pairs = v7_channels[["query_id", "partition_key"]].merge(
        overlay_channels[["query_id", "partition_key"]],
        on="query_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_v7", "_overlay"),
    )
    for position, (partition_keys, partition_queries) in enumerate(
        partition_pairs.groupby(
            ["partition_key_v7", "partition_key_overlay"],
            sort=False,
            dropna=False,
        ),
        start=1,
    ):
        v7_partition_key, overlay_partition_key = partition_keys
        v7_pool = _partition_rows(v7_store, str(v7_partition_key))
        overlay_pool = _partition_rows(
            overlay_store,
            str(overlay_partition_key),
        )
        attach_address_density(v7_pool)
        attach_address_density(overlay_pool)
        candidate_map = {
            str(candidate["siret"]).zfill(14): candidate
            for candidate in [*overlay_pool, *v7_pool]
            if candidate.get("siret")
        }
        idf_map, default_idf = compute_name_idf_map(candidate_map)
        set_global_name_idf_map(idf_map, default_idf)

        output_rows: list[dict[str, Any]] = []
        for query_id in partition_queries["query_id"].astype(str):
            query = benchmark_by_query.loc[query_id]
            label = labels_by_query.loc[query_id]
            admission_row = admission_by_query.loc[query_id]
            selected = [
                str(value).zfill(14)
                for value in json.loads(admission_row["candidate_sirets_json"])
            ]
            if len(selected) > 100:
                over_budget += 1
            if len(selected) != len(set(selected)):
                duplicate_sirets += 1
            selected = list(dict.fromkeys(selected))
            selected_candidates = []
            for siret in selected:
                candidate = candidate_map.get(siret)
                if candidate is None:
                    missing_details += 1
                    continue
                selected_candidates.append(candidate)
            if len(selected_candidates) != len(selected):
                continue

            crm_row = pd.Series(
                {
                    "crm_name": query["crm_name"],
                    "crm_address": query["crm_address"],
                    "crm_city": query["crm_city"],
                    "postcode": query["postcode"],
                    "insee": query["insee"],
                }
            )
            feature_rows = make_feature_rows_from_preprocessed(
                preprocess_crm_row(crm_row),
                selected_candidates,
                include_semantic=False,
            )
            provenance = selective_provenance(
                selected,
                v7_by_query.loc[query_id],
                overlay_by_query.loc[query_id],
            )
            truth = (
                str(label["ground_truth_siret"]).zfill(14)
                if label["label_kind"] == "MATCH_EXACT"
                and pd.notna(label["ground_truth_siret"])
                else None
            )
            for rank, (candidate, features) in enumerate(
                zip(selected_candidates, feature_rows, strict=True),
                start=1,
            ):
                siret = str(candidate["siret"]).zfill(14)
                retrieval = provenance[siret]
                row = {
                    "query_id": query_id,
                    "candidate_siret": siret,
                    "candidate_siren": siret[:9],
                    "split": split,
                    "is_ground_truth": int(siret == truth),
                    "sparse_rank": (
                        1.0
                        / retrieval["admission_current_sparse_rank_recip"]
                        if retrieval["admission_current_sparse_rank_recip"] > 0
                        else 0.0
                    ),
                    "dense_rank": 0.0,
                    "rrf_score": retrieval["admission_fusion_score"],
                    "retrieval_rank": rank,
                    "retrieval_source": (
                        "overlay_quota"
                        if retrieval["admission_overlay_quota"] > 0
                        else "v7_fusion"
                    ),
                    "retrieval_channel_count": retrieval[
                        "admission_channel_count"
                    ],
                    "retrieval_agreement": float(
                        retrieval["admission_channel_count"] >= 2
                    ),
                    **{
                        feature: float(features.get(feature, 0.0))
                        for feature in V9_BASELINE_FEATURE_NAMES
                    },
                    **retrieval,
                }
                output_rows.append(row)
        writer.write(output_rows)
        if position % 100 == 0:
            print(
                f"[dataset] {split} partitions={position} "
                f"rows={writer.count}",
                flush=True,
            )
    return {
        "missing_candidate_details": missing_details,
        "duplicate_candidate_lists": duplicate_sirets,
        "over_budget_queries": over_budget,
    }


def _read_with_hash(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--train-qualification", type=Path, required=True)
    parser.add_argument("--dev-qualification", type=Path, required=True)
    parser.add_argument("--train-admission", type=Path, required=True)
    parser.add_argument("--dev-admission", type=Path, required=True)
    parser.add_argument("--train-v7-channels", type=Path, required=True)
    parser.add_argument("--dev-v7-channels", type=Path, required=True)
    parser.add_argument("--train-overlay-channels", type=Path, required=True)
    parser.add_argument("--dev-overlay-channels", type=Path, required=True)
    parser.add_argument("--v7-partitions", type=Path, required=True)
    parser.add_argument("--overlay-partitions", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    input_paths = {
        key: value
        for key, value in vars(args).items()
        if isinstance(value, Path) and value.is_file()
    }
    benchmark_manifest = json.loads(
        args.benchmark_manifest.read_text(encoding="utf-8")
    )
    snapshot_id = benchmark_manifest["establishment_snapshot_sha256"]
    benchmark = _read_with_hash(args.benchmark)
    benchmark = benchmark[benchmark["split"].isin(["train", "dev"])].copy()
    benchmark["query_id"] = benchmark["query_id"].astype(str)
    qualification = pd.concat(
        [
            _read_with_hash(args.train_qualification),
            _read_with_hash(args.dev_qualification),
        ],
        ignore_index=True,
    )
    qualification["query_id"] = qualification["query_id"].astype(str)
    queries = _canonical_queries(benchmark)
    labels = _canonical_labels(qualification, snapshot_id)
    if set(queries["query_id"]) != set(labels["query_id"]):
        raise ValueError("Benchmark and V3 qualification do not align")

    sources = {}
    for split in ("train", "dev"):
        sources[split] = {
            "admission": _read_with_hash(getattr(args, f"{split}_admission")),
            "v7": _read_with_hash(getattr(args, f"{split}_v7_channels")),
            "overlay": _read_with_hash(
                getattr(args, f"{split}_overlay_channels")
            ),
        }
        for frame in sources[split].values():
            frame["query_id"] = frame["query_id"].astype(str)

    identity = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_build_id": benchmark_manifest["build_id"],
        "input_hashes": {
            name: file_sha256(path)
            for name, path in input_paths.items()
        },
        "feature_order": DATASET_FEATURE_ORDER,
        "retrieval": "selective_admission_frozen_v3",
        "git_commit": _git_commit(),
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    output_dir = args.output_root / build_id
    output_dir.mkdir(parents=True, exist_ok=False)
    writer = CandidateWriter(output_dir / "candidates.parquet")
    v7_store = PartitionedCandidateStore(args.v7_partitions)
    overlay_store = PartitionedCandidateStore(args.overlay_partitions)
    diagnostics = {}
    try:
        for split in ("train", "dev"):
            diagnostics[split] = build_split_candidates(
                split=split,
                benchmark=benchmark[benchmark["split"].eq(split)],
                labels=labels[labels["split"].eq(split)],
                admission=sources[split]["admission"],
                v7_channels=sources[split]["v7"],
                overlay_channels=sources[split]["overlay"],
                v7_store=v7_store,
                overlay_store=overlay_store,
                writer=writer,
            )
    finally:
        writer.close()
    if any(
        value
        for split_diagnostics in diagnostics.values()
        for value in split_diagnostics.values()
    ):
        raise ValueError(f"Candidate build contract failed: {diagnostics}")

    queries.to_parquet(output_dir / "queries.parquet", index=False)
    labels.to_parquet(output_dir / "labels.parquet", index=False)
    manifest = V9DatasetManifest(
        schema_version=SCHEMA_VERSION,
        build_id=build_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        seed=42,
        sirene_snapshot_id=snapshot_id,
        input_hashes=identity["input_hashes"],
        retrieval_config={
            "name": "selective_admission_frozen_v3",
            "candidate_budget": 100,
            "channels": SELECTIVE_RETRIEVAL_CHANNELS,
            "weights": V7_WEIGHTS,
        },
        retrieval_signature=hashlib.sha256(
            json.dumps(
                {
                    "train": identity["input_hashes"]["train_admission"],
                    "dev": identity["input_hashes"]["dev_admission"],
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
        tokenizer_fingerprint=None,
        feature_order=DATASET_FEATURE_ORDER,
        row_counts={
            "queries": len(queries),
            "labels": len(labels),
            "candidates": writer.count,
        },
        legacy_artifacts_allowed=False,
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    build_report = {
        "schema_version": "sireto-downstream-selective-dataset-build-1",
        "dataset_manifest_id": build_id,
        "git_commit": _git_commit(),
        "current_selective_test_read": False,
        "diagnostics": diagnostics,
        "output_hashes": {
            name: file_sha256(output_dir / name)
            for name in (
                "queries.parquet",
                "labels.parquet",
                "candidates.parquet",
            )
        },
    }
    (output_dir / "build_report.json").write_text(
        json.dumps(build_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output_dir)


if __name__ == "__main__":
    main()
