#!/usr/bin/env python3
"""Materialize deterministic candidate features for all V4.12-L top-100 pools."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_downstream_selective_dataset import (  # noqa: E402
    CandidateWriter,
    DATASET_FEATURE_ORDER,
    build_split_candidates,
)
from src.xgb_matcher.partitioned_store import PartitionedCandidateStore  # noqa: E402
from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


SCHEMA_VERSION = "sireto-v4.12-learned-candidate-features-1"
DEFAULT_POPULATION = Path(
    "/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/"
    "v4_12_learned_unified_population/2d29be3ccd8fcc3e"
)
DEFAULT_RETRIEVAL = Path(
    "/Volumes/CATNAT_DATA/SIRETO_RECALL100/evaluations/"
    "v4_12_learned_unified_retrieval/cce1bc83f82a1c3f"
)
DEFAULT_V7_CHANNELS = (
    Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/channel_audit_k5000_train_c33b80855f560074_aeeaf0f"),
    Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/channel_audit_k5000_dev_c33b80855f560074_d4255de"),
    Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/channel_audit_k5000_test_c33b80855f560074_eb0e6a3"),
    Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_12_learned_fresh_channels_v7/1465a04be44b13c3"),
)
DEFAULT_OVERLAY_CHANNELS = (
    Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/closed_overlay_channel_audit_k5000_train_c33b80855f560074_aeeaf0f"),
    Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/closed_overlay_channel_audit_k5000_dev_c33b80855f560074_d4255de"),
    Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/closed_overlay_channel_audit_k5000_test_c33b80855f560074_eb0e6a3"),
    Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/v4_12_learned_fresh_channels_overlay/1465a04be44b13c3"),
)


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _verified_output(directory: Path, name: str) -> tuple[Path, dict[str, Any]]:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = directory / name
    expected = manifest.get("outputs", {}).get(name)
    if expected != file_sha256(path):
        raise ValueError(f"Input hash mismatch: {path}")
    return path, manifest


def _load_many(
    directories: Iterable[Path], name: str
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    frames: list[pd.DataFrame] = []
    records: list[dict[str, str]] = []
    for directory in directories:
        path, _manifest = _verified_output(directory, name)
        manifest_path = directory / "manifest.json"
        frames.append(pd.read_parquet(path))
        records.extend(
            [
                {"path": str(path), "sha256": file_sha256(path)},
                {"path": str(manifest_path), "sha256": file_sha256(manifest_path)},
            ]
        )
    output = pd.concat(frames, ignore_index=True)
    output["query_id"] = output["query_id"].astype(str)
    if output["query_id"].duplicated().any():
        raise ValueError(f"Duplicate query_id in {name} inputs")
    return output, records


def build(args: argparse.Namespace) -> Path:
    population_manifest_path = args.population / "manifest.json"
    population_manifest = json.loads(
        population_manifest_path.read_text(encoding="utf-8")
    )
    queries_path = args.population / "queries.parquet"
    labels_path = args.population / "labels.parquet"
    for path in (queries_path, labels_path):
        if population_manifest.get("outputs", {}).get(path.name) != file_sha256(path):
            raise ValueError(f"Population hash mismatch: {path}")
    queries = pd.read_parquet(queries_path)
    labels = pd.read_parquet(labels_path)
    queries["query_id"] = queries["query_id"].astype(str)
    labels["query_id"] = labels["query_id"].astype(str)

    pools_path, retrieval_manifest = _verified_output(
        args.retrieval, "candidate_pools.parquet"
    )
    pools = pd.read_parquet(pools_path)
    pools["query_id"] = pools["query_id"].astype(str)
    if retrieval_manifest.get("decision") != "GO_RANKER_TRAINING":
        raise ValueError("Candidate features require a passed retrieval gate")
    if int(pools["candidate_count"].max()) > 100:
        raise ValueError("Candidate pool ceiling exceeds 100")

    v7, v7_records = _load_many(args.v7_channels, "raw_results.parquet")
    overlay, overlay_records = _load_many(
        args.overlay_channels, "raw_results.parquet"
    )
    expected_ids = set(queries["query_id"])
    for name, frame in (("labels", labels), ("pools", pools), ("v7", v7), ("overlay", overlay)):
        if set(frame["query_id"]) != expected_ids:
            raise ValueError(f"{name} query IDs differ from the population")

    benchmark = queries.rename(
        columns={"crm_postcode": "postcode", "crm_insee": "insee"}
    )[
        ["query_id", "crm_name", "crm_address", "crm_city", "postcode", "insee"]
    ].copy()
    benchmark["split"] = "oof"
    labels_for_builder = labels[[
        "query_id", "label_kind", "ground_truth_siret", "ground_truth_siren"
    ]].copy()
    labels_for_builder["split"] = "oof"

    input_records = [
        {"path": str(population_manifest_path), "sha256": file_sha256(population_manifest_path)},
        {"path": str(queries_path), "sha256": file_sha256(queries_path)},
        {"path": str(labels_path), "sha256": file_sha256(labels_path)},
        {"path": str(args.retrieval / "manifest.json"), "sha256": file_sha256(args.retrieval / "manifest.json")},
        {"path": str(pools_path), "sha256": file_sha256(pools_path)},
        *v7_records,
        *overlay_records,
    ]
    identity = {
        "schema_version": SCHEMA_VERSION,
        "builder_sha256": file_sha256(Path(__file__)),
        "feature_order": DATASET_FEATURE_ORDER,
        "candidate_ceiling": 100,
        "positive_injection": False,
        "v7_partitions": str(args.v7_partitions),
        "overlay_partitions": str(args.overlay_partitions),
        "inputs": sorted(input_records, key=lambda record: record["path"]),
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    destination = args.output_root / build_id
    if destination.exists():
        manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("build_identity") != identity:
            raise FileExistsError(f"Conflicting immutable output: {destination}")
        return destination

    args.output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{build_id}.", dir=args.output_root))
    writer = CandidateWriter(temporary / "candidates.parquet")
    diagnostics: dict[str, int] = {}
    try:
        diagnostics = build_split_candidates(
            split="oof",
            benchmark=benchmark,
            labels=labels_for_builder,
            admission=pools,
            v7_channels=v7,
            overlay_channels=overlay,
            v7_store=PartitionedCandidateStore(args.v7_partitions),
            overlay_store=PartitionedCandidateStore(args.overlay_partitions),
            writer=writer,
        )
    finally:
        writer.close()
    if any(diagnostics.values()):
        shutil.rmtree(temporary, ignore_errors=True)
        raise ValueError(f"Candidate build diagnostics failed: {diagnostics}")
    expected_candidate_count = int(pools["candidate_count"].sum())
    if writer.count != expected_candidate_count:
        shutil.rmtree(temporary, ignore_errors=True)
        raise ValueError(
            f"Candidate row count differs: {writer.count} != {expected_candidate_count}"
        )

    try:
        queries.to_parquet(temporary / "queries.parquet", index=False)
        labels.to_parquet(temporary / "labels.parquet", index=False)
        report = (
            "# Features candidat V4.12-L\n\n"
            f"- requêtes : {len(queries)} ;\n"
            f"- candidats : {writer.count} ;\n"
            f"- features modèle : {len(DATASET_FEATURE_ORDER)} ;\n"
            "- injection positive : non ;\n"
            "- plafond par requête : 100.\n"
        )
        (temporary / "report.md").write_text(report, encoding="utf-8")
        output_names = ["queries.parquet", "labels.parquet", "candidates.parquet", "report.md"]
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "build_identity": identity,
            "row_counts": {
                "queries": len(queries),
                "labels": len(labels),
                "candidates": writer.count,
            },
            "feature_order": DATASET_FEATURE_ORDER,
            "feature_count": len(DATASET_FEATURE_ORDER),
            "candidate_ceiling": 100,
            "positive_injection": False,
            "diagnostics": diagnostics,
            "outputs": {name: file_sha256(temporary / name) for name in output_names},
        }
        _json_dump(temporary / "manifest.json", manifest)
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", type=Path, default=DEFAULT_POPULATION)
    parser.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument("--v7-channels", type=Path, action="append", default=None)
    parser.add_argument("--overlay-channels", type=Path, action="append", default=None)
    parser.add_argument("--v7-partitions", type=Path, default=Path("data/candidates_v7_all"))
    parser.add_argument(
        "--overlay-partitions",
        type=Path,
        default=Path(
            "/Volumes/CATNAT_DATA/SIRETO_RECALL100/stores/"
            "legacy_closed_overlay_c33b80855f560074_e39fddd"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/"
            "v4_12_learned_candidate_features"
        ),
    )
    args = parser.parse_args()
    if args.v7_channels is None:
        args.v7_channels = list(DEFAULT_V7_CHANNELS)
    if args.overlay_channels is None:
        args.overlay_channels = list(DEFAULT_OVERLAY_CHANNELS)
    return args


if __name__ == "__main__":
    print(build(parse_args()))
