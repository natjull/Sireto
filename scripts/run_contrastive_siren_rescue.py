#!/usr/bin/env python3
"""Build, fit and execute the bounded contrastive SIREN rescue."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xgb_matcher.contrastive_siren_rescue import (  # noqa: E402
    AdapterConfig,
    CONTRASTIVE_RESCUE_SCHEMA_VERSION,
    apply_adapter,
    fit_linear_adapters,
    model_fingerprint,
    normalize_text,
    query_text,
    rank_sites,
    sha256_file,
    siren_documents,
    top_sirens,
)


def _cached_model_path() -> Path:
    root = Path.home() / ".cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots"
    candidates = sorted(path for path in root.glob("*") if path.is_dir())
    if not candidates:
        raise FileNotFoundError("cached multilingual MiniLM model not found")
    return candidates[-1]


def build_training(union: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    union_sql = str(union.resolve()).replace("'", "''")
    output_sql = str(output.resolve()).replace("'", "''")
    connection = duckdb.connect()
    query = f"""
        COPY (
          WITH source AS (
            SELECT query_id, fold, ground_truth_siren,
                   any_value(crm_name) crm_name,
                   any_value(crm_address) crm_address,
                   any_value(crm_postcode) crm_postcode,
                   any_value(crm_insee) crm_insee,
                   any_value(crm_number) crm_number,
                   min_by(
                     'noms: ' || array_to_string(candidate_names, ' | ') ||
                     ' adresse: ' || array_to_string(candidate_addresses, ' | '),
                     union_rank
                   ) FILTER (WHERE siren=ground_truth_siren) positive_text,
                   min_by(
                     'noms: ' || array_to_string(candidate_names, ' | ') ||
                     ' adresse: ' || array_to_string(candidate_addresses, ' | '),
                     union_rank
                   ) FILTER (WHERE siren<>ground_truth_siren) negative_text,
                   bool_or(coalesce(is_synthetic, false)) synthetic
            FROM read_parquet('{union_sql}')
            WHERE fold IN (2,3,4) AND coalesce(identifiable_exact, false)
            GROUP BY query_id, fold, ground_truth_siren
          )
          SELECT * EXCLUDE(synthetic) FROM source
          WHERE NOT synthetic AND positive_text IS NOT NULL AND negative_text IS NOT NULL
          ORDER BY query_id
        ) TO '{output_sql}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    connection.execute(query)
    connection.close()
    print(output)


def _encoder(model_path: Path, device: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(str(model_path), device=device, local_files_only=True)


def _encode(encoder, texts: list[str], batch_size: int) -> np.ndarray:
    return np.asarray(
        encoder.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )


def train(args: argparse.Namespace) -> None:
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    frame = pd.read_parquet(args.training)
    records = frame.to_dict("records")
    q_texts = [query_text(row) for row in records]
    encoder = _encoder(args.model, args.device)
    queries = _encode(encoder, q_texts, args.encode_batch_size)
    positives = _encode(encoder, frame["positive_text"].fillna("").astype(str).tolist(), args.encode_batch_size)
    negatives = _encode(encoder, frame["negative_text"].fillna("").astype(str).tolist(), args.encode_batch_size)
    config = AdapterConfig()
    q_weight, d_weight, history = fit_linear_adapters(queries, positives, negatives, config)
    np.save(output / "query_adapter.npy", q_weight)
    np.save(output / "document_adapter.npy", d_weight)
    manifest = {
        "schema_version": CONTRASTIVE_RESCUE_SCHEMA_VERSION,
        "training_sha256": sha256_file(args.training),
        "training_rows": int(len(frame)),
        "training_folds": sorted(int(value) for value in frame["fold"].unique()),
        "model_fingerprint": model_fingerprint(args.model),
        "model_path": str(args.model),
        "identifiers_used_as_features": False,
        "synthetic_training_rows": 0,
        "adapter_config": config.__dict__,
        "history": history,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(output)


def _partition_paths(root: Path, row: dict[str, Any]) -> list[Path]:
    insee = str(row.get("crm_insee") or "").strip()
    postcode = str(row.get("crm_postcode") or "").strip()
    if insee:
        paths = sorted((root / "insee" / f"insee={insee}").glob("*.parquet"))
        if paths:
            return paths
    if postcode:
        return sorted((root / "cp" / f"postcode={postcode}").glob("*.parquet"))
    return []


def retrieve(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads((args.adapter / "manifest.json").read_text())
    if manifest["training_folds"] != [2, 3, 4]:
        raise RuntimeError("contrastive rescue must be trained only on folds 2/3/4")
    if manifest["model_fingerprint"] != model_fingerprint(args.model):
        raise RuntimeError("contrastive rescue model fingerprint mismatch")
    queries = pd.read_parquet(args.queries)
    if set(queries["fold"].astype(int).unique()) - {0}:
        raise RuntimeError("rescue evaluation accepts dev fold 0 only")
    encoder = _encoder(args.model, args.device)
    q_weight = np.load(args.adapter / "query_adapter.npy")
    d_weight = np.load(args.adapter / "document_adapter.npy")
    output_rows: list[dict[str, Any]] = []
    for partition_key, group in queries.groupby(
        [queries["crm_insee"].fillna(""), queries["crm_postcode"].fillna("")],
        sort=True,
    ):
        first = group.iloc[0].to_dict()
        paths = _partition_paths(args.partitions, first)
        if not paths:
            continue
        columns = pq.ParquetFile(paths[0]).schema_arrow.names
        frames = [pd.read_parquet(path, columns=columns) for path in paths]
        sites = pd.concat(frames, ignore_index=True).drop_duplicates("siret")
        documents = siren_documents(sites)
        embeddings = _encode(
            encoder,
            documents["document_text"].fillna("").astype(str).tolist(),
            args.encode_batch_size,
        )
        document_embeddings = apply_adapter(embeddings, d_weight)
        q_records = group.to_dict("records")
        q_embeddings = apply_adapter(
            _encode(encoder, [query_text(row) for row in q_records], args.encode_batch_size),
            q_weight,
        )
        rankings = top_sirens(
            q_embeddings,
            document_embeddings,
            documents["siren"].astype(str).tolist(),
            args.siren_limit,
        )
        for row, ranking in zip(q_records, rankings, strict=True):
            rescue_rank = 0
            for parent_rank, (siren, score) in enumerate(ranking, start=1):
                selected = rank_sites(sites, row, siren, args.sites_per_siren)
                for site_rank, site in enumerate(selected.to_dict("records"), start=1):
                    rescue_rank += 1
                    output_rows.append(
                        {
                            "query_id": str(row["query_id"]),
                            "fold": int(row["fold"]),
                            "ground_truth_siret": str(row.get("ground_truth_siret") or ""),
                            "ground_truth_siren": str(row.get("ground_truth_siren") or ""),
                            "siret": str(site.get("siret") or "").zfill(14),
                            "siren": str(siren).zfill(9),
                            "contrastive_siren_rescue_rank": rescue_rank,
                            "contrastive_siren_rescue_score": score,
                            "parent_rank": parent_rank,
                            "site_rank": site_rank,
                            "candidate_state": str(site.get("etat_admin") or ""),
                            "candidate_is_siege": bool(site.get("is_siege") or False),
                            "candidate_names": [
                                value
                                for key in (
                                    "denomination", "enseigne1", "enseigne2", "enseigne3",
                                    "sigle_ul", "denomination_ul", "denomination_usuelle_ul",
                                    "nom_ul", "prenom_usuel_ul", "pm_dirigeant_names",
                                )
                                if (value := normalize_text(site.get(key)))
                            ],
                            "candidate_address": " ".join(
                                value
                                for key in (
                                    "numeroVoie", "typeVoie", "libelleVoie",
                                    "complementAdresse", "postcode", "city",
                                )
                                if (value := normalize_text(site.get(key)))
                            ),
                            "candidate_postcode": normalize_text(site.get("postcode")),
                        }
                    )
        # Mega-communes can materialize hundreds of thousands of SIREN
        # documents.  Release both host and MPS allocator state before the
        # next geographic block; keeping these tensors until process exit can
        # exceed the 24 GB unified-memory budget.
        del frames, sites, documents, embeddings, document_embeddings
        gc.collect()
        try:
            import torch

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except (ImportError, RuntimeError):
            pass
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows).to_parquet(args.output, index=False)
    payload = {
        "schema_version": CONTRASTIVE_RESCUE_SCHEMA_VERSION,
        "queries_sha256": sha256_file(args.queries),
        "adapter_manifest_sha256": sha256_file(args.adapter / "manifest.json"),
        "query_count": int(queries["query_id"].nunique()),
        "candidate_rows": len(output_rows),
        "siren_limit": args.siren_limit,
        "sites_per_siren": args.sites_per_siren,
        "positive_injection": False,
        "test_opened": False,
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(args.output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-training")
    build.add_argument("--union", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    fit = sub.add_parser("train")
    fit.add_argument("--training", type=Path, required=True)
    fit.add_argument("--model", type=Path, default=_cached_model_path())
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--device", default="mps")
    fit.add_argument("--encode-batch-size", type=int, default=256)
    run = sub.add_parser("retrieve")
    run.add_argument("--queries", type=Path, required=True)
    run.add_argument("--partitions", type=Path, default=ROOT / "data/candidates_v7_all")
    run.add_argument("--adapter", type=Path, required=True)
    run.add_argument("--model", type=Path, default=_cached_model_path())
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--device", default="mps")
    run.add_argument("--encode-batch-size", type=int, default=256)
    run.add_argument("--siren-limit", type=int, default=50)
    run.add_argument("--sites-per-siren", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build-training":
        build_training(args.union, args.output)
    elif args.command == "train":
        train(args)
    else:
        retrieve(args)


if __name__ == "__main__":
    main()
