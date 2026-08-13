#!/usr/bin/env python3
"""Build the immutable raw-text corpus for the V4.12-N reranker experiment."""

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

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


BASE = Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100")
DEFAULT_DATASET = BASE / "datasets/v4_12_learned_candidate_features/e22aa96feb6ac16f"
DEFAULT_POPULATION = BASE / "datasets/v4_12_learned_unified_population/2d29be3ccd8fcc3e"
DEFAULT_RANKER = BASE / "experiments/v4_12_learned_oof_rankers/839ef55308d5077e"
DEFAULT_OUTPUT_ROOT = BASE / "datasets/v4_12_neural_text_corpus"
SCHEMA_VERSION = "sireto-v4.12-neural-text-corpus-1"


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _duckdb_path(path: Path) -> str:
    return str(path).replace("'", "''")


def _validate_dataset(dataset: Path) -> dict[str, Any]:
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in ("queries.parquet", "labels.parquet", "candidates.parquet"):
        expected = manifest.get("outputs", {}).get(name)
        if not expected or file_sha256(dataset / name) != expected:
            raise ValueError(f"Dataset hash mismatch: {name}")
    if manifest.get("positive_injection") is not False:
        raise ValueError("Neural corpus requires a candidate pool without injection")
    if int(manifest.get("candidate_ceiling", 0)) != 100:
        raise ValueError("Neural corpus requires the frozen top-100 pool")
    return manifest


def _baseline_metrics(labels: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    exact = labels[labels["label_kind"].eq("MATCH_EXACT")].copy()
    top1 = predictions[predictions["ranker_rank"].eq(1)][
        ["query_id", "candidate_siret"]
    ].copy()
    detail = exact.merge(top1, on="query_id", how="left", validate="one_to_one")
    detail["correct"] = detail["candidate_siret"].astype("string").eq(
        detail["ground_truth_siret"].astype("string")
    ).fillna(False)
    detail["difficult"] = detail["label_is_human_validated"].astype(bool)
    detail["active"] = detail["ground_truth_state"].eq("A")
    detail["closed"] = detail["ground_truth_state"].eq("F")
    rows: list[dict[str, Any]] = []
    for fold_name, fold_mask in [
        ("ALL", pd.Series(True, index=detail.index)),
        *[(str(fold), detail["oof_fold"].astype(int).eq(fold)) for fold in range(5)],
    ]:
        for segment, segment_mask in [
            ("exact", pd.Series(True, index=detail.index)),
            ("difficult", detail["difficult"]),
            ("active", detail["active"]),
            ("closed", detail["closed"]),
        ]:
            selected = detail[fold_mask & segment_mask]
            rows.append(
                {
                    "fold": fold_name,
                    "segment": segment,
                    "correct": int(selected["correct"].sum()),
                    "total": len(selected),
                    "hit_at_1": (
                        float(selected["correct"].mean()) if len(selected) else None
                    ),
                }
            )
    return pd.DataFrame(rows)


def build(args: argparse.Namespace) -> Path:
    dataset_manifest = _validate_dataset(args.dataset)
    population_manifest = json.loads(
        (args.population / "manifest.json").read_text(encoding="utf-8")
    )
    inputs = {
        "dataset_manifest": file_sha256(args.dataset / "manifest.json"),
        "population_manifest": file_sha256(args.population / "manifest.json"),
        "ranker_manifest": file_sha256(args.ranker / "manifest.json"),
        "establishments": file_sha256(args.establishments),
        "legal_units": file_sha256(args.legal_units),
    }
    identity = {
        "schema_version": SCHEMA_VERSION,
        "builder_sha256": file_sha256(Path(__file__)),
        "inputs": inputs,
        "candidate_ceiling": 100,
        "positive_injection": False,
        "fold_roles": {"train": [2, 3, 4], "selection": 0, "confirmation": 1},
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    destination = args.output_root / build_id
    if destination.exists():
        existing = json.loads((destination / "manifest.json").read_text())
        if existing.get("build_identity") != identity:
            raise FileExistsError(f"Conflicting immutable output: {destination}")
        return destination

    args.output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{build_id}.", dir=args.output_root))
    try:
        query_output = temporary / "queries_text.parquet"
        candidate_output = temporary / "candidates_text.parquet"
        with duckdb.connect() as connection:
            connection.execute("SET threads TO 8")
            connection.execute("SET preserve_insertion_order TO false")
            connection.execute(
                f"""
                COPY (
                    SELECT
                        CAST(query_id AS VARCHAR) AS query_id,
                        CAST(oof_fold AS TINYINT) AS oof_fold,
                        crm_record_id,
                        crm_name,
                        crm_address,
                        crm_postcode,
                        crm_city,
                        crm_insee,
                        trim(concat(
                            'Nom CRM : ', coalesce(crm_name, ''),
                            '. Adresse CRM : ', coalesce(crm_address, ''),
                            ', ', coalesce(crm_postcode, ''),
                            ' ', coalesce(crm_city, ''),
                            '. Code commune : ', coalesce(crm_insee, ''), '.'
                        )) AS query_text
                    FROM read_parquet('{_duckdb_path(args.dataset / "queries.parquet")}')
                    ORDER BY query_id
                ) TO '{_duckdb_path(query_output)}'
                    (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
                """
            )
            connection.execute(
                f"""
                COPY (
                    SELECT
                        CAST(c.query_id AS VARCHAR) AS query_id,
                        CAST(c.candidate_siret AS VARCHAR) AS candidate_siret,
                        CAST(c.candidate_siren AS VARCHAR) AS candidate_siren,
                        CAST(c.retrieval_rank AS SMALLINT) AS retrieval_rank,
                        c.retrieval_source,
                        CAST(c.is_ground_truth AS TINYINT) AS is_ground_truth,
                        trim(concat(
                            'Dénomination légale : ', coalesce(u.denominationUniteLegale, ''),
                            '. Sigle : ', coalesce(u.sigleUniteLegale, ''),
                            '. Noms usuels unité légale : ',
                            concat_ws(' ; ', u.denominationUsuelle1UniteLegale,
                                u.denominationUsuelle2UniteLegale,
                                u.denominationUsuelle3UniteLegale),
                            '. Personne : ', concat_ws(' ', u.prenomUsuelUniteLegale,
                                u.prenom1UniteLegale, u.nomUniteLegale, u.nomUsageUniteLegale),
                            '. Enseignes et nom établissement : ',
                            concat_ws(' ; ', e.enseigne1Etablissement,
                                e.enseigne2Etablissement, e.enseigne3Etablissement,
                                e.denominationUsuelleEtablissement),
                            '. Adresse établissement : ',
                            concat_ws(' ', e.complementAdresseEtablissement,
                                e.numeroVoieEtablissement, e.indiceRepetitionEtablissement,
                                e.typeVoieEtablissement, e.libelleVoieEtablissement),
                            ', ', coalesce(e.codePostalEtablissement, ''),
                            ' ', coalesce(e.libelleCommuneEtablissement, ''),
                            '. Code commune : ', coalesce(e.codeCommuneEtablissement, ''),
                            '. État établissement : ', coalesce(e.etatAdministratifEtablissement, ''),
                            '. Siège : ', coalesce(e.etablissementSiege, ''),
                            '. Activité établissement : ', coalesce(e.activitePrincipaleEtablissement, ''),
                            '. Activité unité légale : ', coalesce(u.activitePrincipaleUniteLegale, ''),
                            '. Catégorie juridique : ', coalesce(u.categorieJuridiqueUniteLegale, ''),
                            '. Employeur : ', coalesce(e.caractereEmployeurEtablissement, ''),
                            '. Effectif : ', coalesce(e.trancheEffectifsEtablissement, ''),
                            '. Début activité : ', coalesce(CAST(e.dateDebut AS VARCHAR), ''), '.'
                        )) AS candidate_text
                    FROM read_parquet('{_duckdb_path(args.dataset / "candidates.parquet")}') c
                    LEFT JOIN read_parquet('{_duckdb_path(args.establishments.resolve())}') e
                        ON CAST(c.candidate_siret AS VARCHAR) = CAST(e.siret AS VARCHAR)
                    LEFT JOIN read_parquet('{_duckdb_path(args.legal_units.resolve())}') u
                        ON CAST(c.candidate_siren AS VARCHAR) = CAST(u.siren AS VARCHAR)
                    ORDER BY c.query_id, c.retrieval_rank, c.candidate_siret
                ) TO '{_duckdb_path(candidate_output)}'
                    (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
                """
            )

            diagnostics = connection.execute(
                """
                SELECT
                    count(*) AS candidate_rows,
                    count(DISTINCT query_id) AS candidate_queries,
                    max(pool_size) AS max_pool_size,
                    sum(CASE WHEN candidate_text IS NULL OR candidate_text = '' THEN 1 ELSE 0 END)
                        AS blank_candidate_texts
                FROM (
                    SELECT *, count(*) OVER (PARTITION BY query_id) AS pool_size
                    FROM read_parquet(?)
                )
                """,
                [str(candidate_output)],
            ).fetchone()

        labels = pd.read_parquet(args.dataset / "labels.parquet")
        predictions = pd.read_parquet(
            args.ranker / "business_learned_oof_candidates.parquet",
            columns=["query_id", "candidate_siret", "ranker_rank"],
        )
        labels["query_id"] = labels["query_id"].astype(str)
        predictions["query_id"] = predictions["query_id"].astype(str)
        baseline = _baseline_metrics(labels, predictions)
        baseline.to_csv(temporary / "baseline_by_fold.csv", index=False)
        shutil.copyfile(args.dataset / "labels.parquet", temporary / "labels.parquet")
        shutil.copyfile(
            args.population / "fold_assignments.parquet",
            temporary / "fold_assignments.parquet",
        )

        candidate_rows, candidate_queries, max_pool, blank_texts = map(int, diagnostics)
        expected_candidates = int(dataset_manifest["row_counts"]["candidates"])
        expected_queries = int(dataset_manifest["row_counts"]["queries"])
        if candidate_rows != expected_candidates or candidate_queries != expected_queries:
            raise ValueError("Text projection changed the candidate population")
        if max_pool > 100:
            raise ValueError("Text corpus exceeds the candidate ceiling")
        if blank_texts:
            raise ValueError("Text corpus contains blank candidate evidence")
        fold_assignment = pd.read_parquet(args.population / "fold_assignments.parquet")
        if fold_assignment.groupby("siren_component_id")["oof_fold"].nunique().max() != 1:
            raise ValueError("SIREN component crosses folds")

        report = (
            "# Corpus texte V4.12-N\n\n"
            f"- requêtes : {expected_queries:,} ;\n"
            f"- candidats : {candidate_rows:,} ;\n"
            f"- maximum par requête : {max_pool} ;\n"
            "- folds : 2/3/4 entraînement, 0 sélection, 1 confirmation ;\n"
            "- positif injecté : non ;\n"
            "- test final ouvert : non.\n\n"
            "Le texte contient uniquement les champs CRM et SIRENE bruts. Le SIRET "
            "candidat reste une clé de sortie et n'est pas sérialisé dans l'entrée modèle.\n"
        )
        (temporary / "report.md").write_text(report, encoding="utf-8")
        output_names = [
            "queries_text.parquet",
            "candidates_text.parquet",
            "labels.parquet",
            "fold_assignments.parquet",
            "baseline_by_fold.csv",
            "report.md",
        ]
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "build_identity": identity,
            "source_population_build_id": population_manifest.get("build_id"),
            "row_counts": {
                "queries": expected_queries,
                "candidates": candidate_rows,
                "labels": len(labels),
            },
            "candidate_ceiling": max_pool,
            "positive_injection": False,
            "candidate_siret_serialized_in_text": False,
            "final_test_opened": False,
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
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--population", type=Path, default=DEFAULT_POPULATION)
    parser.add_argument("--ranker", type=Path, default=DEFAULT_RANKER)
    parser.add_argument(
        "--establishments", type=Path, default=Path("data/StockEtablissement_utf8.parquet")
    )
    parser.add_argument(
        "--legal-units", type=Path, default=Path("data/StockUniteLegale_utf8.parquet")
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    print(build(parse_args()))
