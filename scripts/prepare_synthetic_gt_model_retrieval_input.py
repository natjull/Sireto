#!/usr/bin/env python3
"""Freeze promoted synthetic CRM rows as a non-injected retrieval benchmark."""

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

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.features import normalize_text  # noqa: E402
from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


BASE = Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100")
DEFAULT_PROMOTED_ROOT = BASE / "datasets/synthetic_gt_corpus/balanced_v1"
DEFAULT_REAL_LABELS = BASE / "datasets/v4_12_learned_business_features/8800ef53f6927215"
DEFAULT_SNAPSHOT = Path("data/StockEtablissement_utf8.parquet")
DEFAULT_RETRIEVAL_MANIFEST = Path(
    "/Volumes/CATNAT_DATA/SIRETO_V9/benchmarks/closed/c33b80855f560074/manifest.json"
)
DEFAULT_OUTPUT_ROOT = BASE / "datasets/synthetic_gt_model_retrieval_input_v1"
SCHEMA_VERSION = "sireto-synthetic-gt-model-retrieval-input-1"
SPLIT = "synthetic_train"
TRAIN_FOLDS = (2, 3, 4)
SEED = 42


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _stable_train_fold(siren: str, seed: int = SEED) -> int:
    digest = hashlib.sha256(f"synthetic-model-fold:{seed}:{siren}".encode()).digest()
    return TRAIN_FOLDS[int.from_bytes(digest[:8], "big") % len(TRAIN_FOLDS)]


def _discover_promoted(root: Path, explicit: Iterable[Path]) -> list[Path]:
    paths = [Path(path) for path in explicit]
    if not paths:
        paths = sorted(root.glob("P*_promoted/promoted.jsonl"))
    paths = sorted({path.resolve() for path in paths})
    if not paths:
        raise FileNotFoundError(f"No promoted.jsonl found under {root}")
    return paths


def _read_promoted(paths: list[Path]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for path in paths:
        manifest_path = path.with_name("manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        observed = file_sha256(path)
        expected_sha256 = manifest.get("promoted_sha256")
        expected_rows = manifest.get("promoted_variants")
        if expected_sha256 is None:
            # The balanced finalizer publishes a single, audited corpus with a
            # different manifest envelope from the per-batch promotion
            # manifests.  Accept it only through its explicit filename hash
            # and final row count; this lets model builds bind directly to the
            # cleaned final corpus instead of rediscovering superseded batch
            # promotions.
            expected_sha256 = manifest.get("files", {}).get(path.name)
            expected_rows = manifest.get("rows")
        if expected_sha256 != observed:
            raise ValueError(f"Promoted hash mismatch: {path}")
        count = 0
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("final_decision") != "ACCEPT":
                    raise ValueError(f"Non-ACCEPT promoted row: {path}:{line_number}")
                qualification = row.get("full_sirene_qualification", {})
                if qualification.get("decision") != "EXACT_IDENTIFIABLE":
                    raise ValueError(f"Non-exact promoted row: {path}:{line_number}")
                if not qualification.get("target_naturally_returned"):
                    raise ValueError(f"Injected/absent promoted target: {path}:{line_number}")
                rows.append(row)
                count += 1
        if count != int(expected_rows if expected_rows is not None else -1):
            raise ValueError(f"Promoted row count mismatch: {path}")
        sources.append(
            {
                "path": str(path),
                "sha256": observed,
                "manifest": str(manifest_path),
                "manifest_sha256": file_sha256(manifest_path),
                "rows": count,
            }
        )
    frame = pd.DataFrame(rows)
    required = {
        "variant_contract_sha256", "target_siret", "target_siren", "crm",
        "difficulty", "augmentation_stratum", "source_kind",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Promoted rows are missing: {sorted(missing)}")
    if frame["variant_contract_sha256"].duplicated().any():
        raise ValueError("Promoted corpus contains duplicate variant contracts")
    return frame, sources


def _snapshot_states(snapshot: Path, sirets: pd.Series) -> dict[str, str]:
    wanted = pd.DataFrame({"siret": sorted(set(sirets.astype(str).str.zfill(14)))})
    with duckdb.connect() as connection:
        connection.register("wanted", wanted)
        rows = connection.execute(
            """
            SELECT CAST(e.siret AS VARCHAR) AS siret,
                   CAST(e.etatAdministratifEtablissement AS VARCHAR) AS state
            FROM read_parquet(?) e
            INNER JOIN wanted w ON CAST(e.siret AS VARCHAR) = w.siret
            """,
            [str(snapshot.resolve())],
        ).fetch_df()
    rows["siret"] = rows["siret"].astype(str).str.zfill(14)
    if rows["siret"].duplicated().any():
        raise ValueError("SIRENE snapshot contains duplicate SIRETs")
    states = dict(zip(rows["siret"], rows["state"], strict=True))
    missing = sorted(set(wanted["siret"]) - set(states))
    if missing:
        raise ValueError(f"Promoted SIRETs missing from snapshot: {missing[:5]}")
    return states


def build(args: argparse.Namespace) -> Path:
    promoted_paths = _discover_promoted(args.promoted_root, args.promoted or [])
    promoted, promoted_sources = _read_promoted(promoted_paths)
    if len(promoted) < args.minimum_variants:
        raise ValueError(
            f"Corpus has {len(promoted)} promoted variants; {args.minimum_variants} required"
        )
    promoted["target_siret"] = promoted["target_siret"].astype(str).str.zfill(14)
    promoted["target_siren"] = promoted["target_siren"].astype(str).str.zfill(9)
    if ~promoted["target_siret"].str[:9].eq(promoted["target_siren"]).all():
        raise ValueError("Promoted SIRET/SIREN bindings are inconsistent")

    real_manifest = json.loads((args.real_labels / "manifest.json").read_text(encoding="utf-8"))
    real_labels_path = args.real_labels / "labels.parquet"
    if real_manifest.get("outputs", {}).get("labels.parquet") != file_sha256(real_labels_path):
        raise ValueError("Real label manifest mismatch")
    real_labels = pd.read_parquet(real_labels_path)
    real_sirens = set(
        real_labels.loc[real_labels["label_kind"].eq("MATCH_EXACT"), "ground_truth_siren"]
        .dropna().astype(str).str.zfill(9)
    )
    overlap = real_sirens & set(promoted["target_siren"])
    if overlap:
        raise ValueError(f"Synthetic SIRENs overlap protected real labels: {sorted(overlap)[:5]}")

    states = _snapshot_states(args.establishments, promoted["target_siret"])
    records: list[dict[str, Any]] = []
    label_records: list[dict[str, Any]] = []
    for row in promoted.itertuples(index=False):
        crm = dict(row.crm)
        contract = str(row.variant_contract_sha256)
        query_id = f"synthetic:{contract}"
        fold = _stable_train_fold(str(row.target_siren))
        record = {
            "query_id": query_id,
            "crm_name": str(crm.get("name") or ""),
            "crm_address": str(crm.get("address") or ""),
            "crm_city": str(crm.get("city") or ""),
            "postcode": str(crm.get("postcode") or ""),
            "insee": str(crm.get("insee") or ""),
            "reference_date": "",
            "split": SPLIT,
            "ground_truth_siret": str(row.target_siret),
            "ground_truth_siren": str(row.target_siren),
            "ground_truth_state": states[str(row.target_siret)],
            "location_match_type": "insee" if str(crm.get("insee") or "") else "cp_only",
            "difficulty": str(row.difficulty),
            "augmentation_stratum": str(row.augmentation_stratum),
            "variant_contract_sha256": contract,
            "oof_fold": fold,
        }
        records.append(record)
        label_records.append(
            {
                "query_id": query_id,
                "label_kind": "MATCH_EXACT",
                "ground_truth_siret": str(row.target_siret),
                "ground_truth_siren": str(row.target_siren),
                "ground_truth_state": states[str(row.target_siret)],
                "oof_fold": fold,
                "difficulty": str(row.difficulty),
                "augmentation_stratum": str(row.augmentation_stratum),
                "variant_contract_sha256": contract,
                "source_kind": "SYNTHETIC_GT",
            }
        )
    benchmark = pd.DataFrame(records).sort_values("query_id", kind="mergesort")
    labels = pd.DataFrame(label_records).sort_values("query_id", kind="mergesort")
    if benchmark["query_id"].duplicated().any() or labels["query_id"].duplicated().any():
        raise ValueError("Synthetic query IDs are not unique")
    if labels.groupby("ground_truth_siren")["oof_fold"].nunique().max() != 1:
        raise ValueError("A synthetic SIREN crosses folds 2/3/4")
    if set(labels["oof_fold"].astype(int)) != set(TRAIN_FOLDS):
        raise ValueError("Synthetic folds 2/3/4 are not all populated")

    retrieval_manifest = json.loads(args.retrieval_benchmark_manifest.read_text(encoding="utf-8"))
    partitions_sha256 = str(retrieval_manifest.get("partitions_sha256") or "")
    if not partitions_sha256:
        raise ValueError("Pinned retrieval benchmark lacks partitions_sha256")
    if retrieval_manifest.get("establishment_snapshot_sha256") != file_sha256(args.establishments):
        raise ValueError("Establishment snapshot differs from the pinned retrieval benchmark")
    identity = {
        "schema_version": SCHEMA_VERSION,
        "builder_sha256": file_sha256(Path(__file__)),
        "promoted_sources": promoted_sources,
        "real_labels_manifest_sha256": file_sha256(args.real_labels / "manifest.json"),
        "establishments_sha256": file_sha256(args.establishments),
        "retrieval_benchmark_manifest_sha256": file_sha256(args.retrieval_benchmark_manifest),
        "partitions_sha256": partitions_sha256,
        "minimum_variants": args.minimum_variants,
        "fold_seed": SEED,
        "train_folds": list(TRAIN_FOLDS),
        "positive_injection": False,
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    destination = args.output_root / build_id
    if destination.exists():
        existing = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        if existing.get("build_identity") != identity:
            raise FileExistsError(f"Conflicting immutable output: {destination}")
        return destination

    args.output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{build_id}.", dir=args.output_root))
    try:
        benchmark.to_parquet(temporary / "benchmark.parquet", index=False)
        labels.to_parquet(temporary / "labels.parquet", index=False)
        queries = benchmark.rename(columns={"postcode": "crm_postcode", "insee": "crm_insee"})[
            ["query_id", "crm_name", "crm_address", "crm_postcode", "crm_city", "crm_insee", "reference_date"]
        ].copy()
        queries["crm_record_id"] = queries["query_id"]
        queries["crm_name_norm"] = queries["crm_name"].map(normalize_text)
        queries["crm_address_norm"] = queries["crm_address"].map(normalize_text)
        queries["crm_city_norm"] = queries["crm_city"].map(normalize_text)
        queries.to_parquet(temporary / "queries.parquet", index=False)
        outputs = ("benchmark.parquet", "labels.parquet", "queries.parquet")
        output_hashes = {name: file_sha256(temporary / name) for name in outputs}
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "build_identity": identity,
            "query_count": len(benchmark),
            "target_count": int(labels["ground_truth_siret"].nunique()),
            "siren_count": int(labels["ground_truth_siren"].nunique()),
            "fold_counts": {str(k): int(v) for k, v in labels["oof_fold"].value_counts().sort_index().items()},
            "partitions_sha256": partitions_sha256,
            "positive_injection": False,
            "qualification_uses_retrieval_or_model_scores": False,
            "output_sha256": output_hashes,
            "outputs": output_hashes,
        }
        _json_dump(temporary / "manifest.json", manifest)
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promoted-root", type=Path, default=DEFAULT_PROMOTED_ROOT)
    parser.add_argument("--promoted", type=Path, action="append")
    parser.add_argument("--minimum-variants", type=int, default=20_000)
    parser.add_argument("--real-labels", type=Path, default=DEFAULT_REAL_LABELS)
    parser.add_argument("--establishments", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--retrieval-benchmark-manifest", type=Path, default=DEFAULT_RETRIEVAL_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    print(build(parse_args()))
