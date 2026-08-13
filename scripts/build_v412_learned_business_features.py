#!/usr/bin/env python3
"""Enrich the V4.12-L candidate table with learnable business features."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_downstream_selective_dataset import DATASET_FEATURE_ORDER  # noqa: E402
from scripts.evaluate_v412_ranker_business_features import (  # noqa: E402
    RELATIONAL_FEATURES,
    SOURCE_FEATURES,
    _read_enriched_sources,
    _relational_features,
    _source_features,
)
from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


SCHEMA_VERSION = "sireto-v4.12-learned-business-features-1"
DEFAULT_DATASET = Path(
    "/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/"
    "v4_12_learned_candidate_features/e22aa96feb6ac16f"
)
DEFAULT_ETABLISSEMENTS = Path("data/StockEtablissement_utf8.parquet")
DEFAULT_UNITES_LEGALES = Path("data/StockUniteLegale_utf8.parquet")
BUSINESS_FEATURE_ORDER = [*DATASET_FEATURE_ORDER, *SOURCE_FEATURES, *RELATIONAL_FEATURES]
METADATA_COLUMNS = [
    "query_id",
    "candidate_siret",
    "candidate_siren",
    "is_ground_truth",
    "retrieval_rank",
    "retrieval_source",
]


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build(args: argparse.Namespace) -> Path:
    manifest_path = args.dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidates_path = args.dataset / "candidates.parquet"
    queries_path = args.dataset / "queries.parquet"
    labels_path = args.dataset / "labels.parquet"
    for path in (candidates_path, queries_path, labels_path):
        if manifest.get("outputs", {}).get(path.name) != file_sha256(path):
            raise ValueError(f"Candidate dataset hash mismatch: {path}")
    if manifest.get("positive_injection") is not False:
        raise ValueError("Business features require a non-injected candidate dataset")
    if list(manifest.get("feature_order") or []) != DATASET_FEATURE_ORDER:
        raise ValueError("Candidate dataset feature order differs")

    identity = {
        "schema_version": SCHEMA_VERSION,
        "builder_sha256": file_sha256(Path(__file__)),
        "business_helper_sha256": file_sha256(
            Path(__file__).with_name("evaluate_v412_ranker_business_features.py")
        ),
        "dataset_manifest_sha256": file_sha256(manifest_path),
        "establishments_sha256": file_sha256(args.etablissements),
        "legal_units_sha256": file_sha256(args.unites_legales),
        "feature_order": BUSINESS_FEATURE_ORDER,
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    destination = args.output_root / build_id
    if destination.exists():
        existing = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        if existing.get("build_identity") != identity:
            raise FileExistsError(f"Conflicting immutable output: {destination}")
        return destination

    frame = _read_enriched_sources(
        args.dataset,
        args.etablissements,
        args.unites_legales,
        candidate_filename="candidates.parquet",
    )
    frame = _relational_features(_source_features(frame))
    required = [*METADATA_COLUMNS, *BUSINESS_FEATURE_ORDER]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Business enrichment is missing columns: {missing}")
    matrix = frame[BUSINESS_FEATURE_ORDER].to_numpy(dtype=np.float32)
    if not np.isfinite(matrix).all():
        bad = frame[BUSINESS_FEATURE_ORDER].columns[
            ~np.isfinite(matrix).all(axis=0)
        ].tolist()
        raise ValueError(f"Business features contain non-finite values: {bad}")
    output = frame[required].copy()
    output = output.sort_values(
        ["query_id", "retrieval_rank", "candidate_siret"], kind="mergesort"
    ).reset_index(drop=True)
    if output.duplicated(["query_id", "candidate_siret"]).any():
        raise ValueError("Business candidate rows contain duplicate SIRETs")
    if output.groupby("query_id", sort=False).size().max() > 100:
        raise ValueError("Business candidate pool exceeds 100")
    if len(output) != int(manifest["row_counts"]["candidates"]):
        raise ValueError("Business enrichment changed the candidate row count")

    args.output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{build_id}.", dir=args.output_root))
    try:
        output.to_parquet(temporary / "candidates_business.parquet", index=False)
        shutil.copyfile(queries_path, temporary / "queries.parquet")
        shutil.copyfile(labels_path, temporary / "labels.parquet")
        report = (
            "# Features métier apprises V4.12-L\n\n"
            f"- candidats : {len(output)} ;\n"
            f"- features retrieval/identité : {len(DATASET_FEATURE_ORDER)} ;\n"
            f"- features métier source : {len(SOURCE_FEATURES)} ;\n"
            f"- features relationnelles : {len(RELATIONAL_FEATURES)} ;\n"
            f"- total BUSINESS_LEARNED : {len(BUSINESS_FEATURE_ORDER)}.\n\n"
            "Ces colonnes sont des entrées du ranker XGBoost. Aucune d'elles "
            "n'applique directement une décision ou une promotion de candidat.\n"
        )
        (temporary / "report.md").write_text(report, encoding="utf-8")
        output_names = [
            "candidates_business.parquet",
            "queries.parquet",
            "labels.parquet",
            "report.md",
        ]
        output_manifest = {
            "schema_version": SCHEMA_VERSION,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "build_identity": identity,
            "row_counts": {
                "queries": int(manifest["row_counts"]["queries"]),
                "labels": int(manifest["row_counts"]["labels"]),
                "candidates": len(output),
            },
            "base_feature_order": DATASET_FEATURE_ORDER,
            "source_feature_order": SOURCE_FEATURES,
            "relational_feature_order": RELATIONAL_FEATURES,
            "business_feature_order": BUSINESS_FEATURE_ORDER,
            "candidate_ceiling": 100,
            "positive_injection": False,
            "deterministic_promotions": False,
            "outputs": {
                name: file_sha256(temporary / name) for name in output_names
            },
        }
        _json_dump(temporary / "manifest.json", output_manifest)
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--etablissements", type=Path, default=DEFAULT_ETABLISSEMENTS)
    parser.add_argument("--unites-legales", type=Path, default=DEFAULT_UNITES_LEGALES)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/"
            "v4_12_learned_business_features"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(build(parse_args()))
