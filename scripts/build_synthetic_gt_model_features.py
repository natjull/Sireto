#!/usr/bin/env python3
"""Materialize the non-injected XGBoost/BGE bundle for synthetic GT."""

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
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_downstream_selective_dataset import (  # noqa: E402
    CandidateWriter,
    build_split_candidates,
)
from scripts.build_v412_learned_business_features import (  # noqa: E402
    BUSINESS_FEATURE_ORDER,
    METADATA_COLUMNS,
)
from scripts.evaluate_retrieval_admission import ORACLE_CHANNELS, select_candidates  # noqa: E402
from scripts.evaluate_v412_ranker_business_features import (  # noqa: E402
    _read_enriched_sources,
    _relational_features,
    _source_features,
)
from src.xgb_matcher.partitioned_store import PartitionedCandidateStore  # noqa: E402
from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


BASE = Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100")
DEFAULT_V7_PARTITIONS = Path("data/candidates_v7_all")
DEFAULT_OVERLAY_PARTITIONS = BASE / "stores/legacy_closed_overlay_c33b80855f560074_e39fddd"
DEFAULT_ETABLISSEMENTS = Path("data/StockEtablissement_utf8.parquet")
DEFAULT_UNITES_LEGALES = Path("data/StockUniteLegale_utf8.parquet")
DEFAULT_RANKER = BASE / "experiments/v4_12_learned_oof_rankers/839ef55308d5077e"
DEFAULT_REAL_BUSINESS = BASE / "datasets/v4_12_learned_business_features/8800ef53f6927215"
DEFAULT_OUTPUT_ROOT = BASE / "datasets/synthetic_gt_model_features"
SCHEMA_VERSION = "sireto-synthetic-gt-model-features-1"
SOURCE_SCHEMA = "sireto-synthetic-gt-model-retrieval-input-1"
CHANNEL_SCHEMA = "sireto-retrieval-channel-audit-1"
SPLIT = "synthetic_train"
INTERNAL_K = 5000
CANDIDATE_BUDGET = 100
MAX_NEGATIVES = 15
HOMONYM_THRESHOLD = 0.90


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def _verified(root: Path, names: tuple[str, ...]) -> dict[str, Any]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for name in names:
        if manifest.get("outputs", {}).get(name) != file_sha256(root / name):
            raise ValueError(f"Manifest mismatch: {root / name}")
    return manifest


def _validate_channels(root: Path, source: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    manifest = _verified(root, ("raw_results.parquet", "summary.json"))
    if manifest.get("schema_version") != CHANNEL_SCHEMA:
        raise ValueError(f"Unexpected channel schema: {root}")
    if manifest.get("split") != SPLIT:
        raise ValueError(f"Channel audit is not synthetic_train: {root}")
    if int(manifest.get("per_channel_k", -1)) != INTERNAL_K or INTERNAL_K not in manifest.get("cutoffs", []):
        raise ValueError(f"Channel audit is not the frozen k=5000 run: {root}")
    if manifest.get("benchmark_manifest_sha256") != file_sha256(source / "manifest.json"):
        raise ValueError(f"Channel/source manifest mismatch: {root}")
    return manifest, pd.read_parquet(root / "raw_results.parquet")


def _admission(v7: pd.DataFrame, overlay: pd.DataFrame) -> pd.DataFrame:
    for frame in (v7, overlay):
        frame["query_id"] = frame["query_id"].astype(str)
    if list(v7["query_id"]) != list(overlay["query_id"]):
        raise ValueError("V7 and overlay channel query order differs")
    records = []
    for position, row in v7.iterrows():
        overlay_row = overlay.iloc[position]
        v7_lists = {
            channel: json.loads(row[f"{channel}_sirets_json"])
            for channel in ORACLE_CHANNELS
        }
        overlay_lists = {
            channel: json.loads(overlay_row[f"{channel}_sirets_json"])
            for channel in ORACLE_CHANNELS
        }
        selected = select_candidates(
            v7_channels=v7_lists,
            overlay_channels=overlay_lists,
            budget=CANDIDATE_BUDGET,
            internal_k=INTERNAL_K,
        )
        if len(selected) > CANDIDATE_BUDGET or len(selected) != len(set(selected)):
            raise ValueError("Frozen admission emitted an invalid candidate list")
        records.append(
            {
                "query_id": str(row["query_id"]),
                "candidate_sirets_json": json.dumps(selected, separators=(",", ":")),
                "candidate_count": len(selected),
            }
        )
    return pd.DataFrame(records)


def _candidate_text_projection(
    *,
    candidates: Path,
    queries: Path,
    etablissements: Path,
    unites_legales: Path,
    output_queries: Path,
    output_candidates: Path,
) -> None:
    with duckdb.connect() as connection:
        connection.execute("SET threads TO 8")
        connection.execute(
            f"""
            COPY (
                SELECT CAST(query_id AS VARCHAR) AS query_id,
                       CAST(oof_fold AS TINYINT) AS oof_fold,
                       trim(concat(
                           'Nom CRM : ', coalesce(crm_name, ''),
                           '. Adresse CRM : ', coalesce(crm_address, ''),
                           ', ', coalesce(crm_postcode, ''), ' ', coalesce(crm_city, ''),
                           '. Code commune : ', coalesce(crm_insee, ''), '.'
                       )) AS query_text
                FROM read_parquet(?) ORDER BY query_id
            ) TO '{_sql_path(output_queries)}'
              (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
            """,
            [str(queries)],
        )
        connection.execute(
            f"""
            COPY (
                SELECT CAST(c.query_id AS VARCHAR) AS query_id,
                       CAST(c.candidate_siret AS VARCHAR) AS candidate_siret,
                       CAST(c.candidate_siren AS VARCHAR) AS candidate_siren,
                       CAST(c.retrieval_rank AS SMALLINT) AS retrieval_rank,
                       CAST(c.is_ground_truth AS TINYINT) AS is_ground_truth,
                       CAST(e.etatAdministratifEtablissement AS VARCHAR) AS candidate_state,
                       CAST(e.identifiantAdresseEtablissement AS VARCHAR) AS address_id,
                       trim(concat(
                           'Dénomination légale : ', coalesce(u.denominationUniteLegale, ''),
                           '. Sigle : ', coalesce(u.sigleUniteLegale, ''),
                           '. Noms usuels unité légale : ', concat_ws(' ; ',
                               u.denominationUsuelle1UniteLegale,
                               u.denominationUsuelle2UniteLegale,
                               u.denominationUsuelle3UniteLegale),
                           '. Personne : ', concat_ws(' ', u.prenomUsuelUniteLegale,
                               u.prenom1UniteLegale, u.nomUniteLegale, u.nomUsageUniteLegale),
                           '. Enseignes et nom établissement : ', concat_ws(' ; ',
                               e.enseigne1Etablissement, e.enseigne2Etablissement,
                               e.enseigne3Etablissement, e.denominationUsuelleEtablissement),
                           '. Adresse établissement : ', concat_ws(' ',
                               e.complementAdresseEtablissement, e.numeroVoieEtablissement,
                               e.indiceRepetitionEtablissement, e.typeVoieEtablissement,
                               e.libelleVoieEtablissement), ', ',
                           coalesce(e.codePostalEtablissement, ''), ' ',
                           coalesce(e.libelleCommuneEtablissement, ''),
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
                FROM read_parquet(?) c
                LEFT JOIN read_parquet(?) e
                  ON CAST(c.candidate_siret AS VARCHAR) = CAST(e.siret AS VARCHAR)
                LEFT JOIN read_parquet(?) u
                  ON CAST(c.candidate_siren AS VARCHAR) = CAST(u.siren AS VARCHAR)
                ORDER BY c.query_id, c.retrieval_rank, c.candidate_siret
            ) TO '{_sql_path(output_candidates)}'
              (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
            """,
            [str(candidates), str(etablissements.resolve()), str(unites_legales.resolve())],
        )


def _select_negative_indices(frame: pd.DataFrame) -> tuple[list[int], int]:
    """Reproduce BGE hard-negative quotas while excluding same-site siblings."""
    positive = frame[frame["is_ground_truth"].astype(int).eq(1)]
    if len(positive) != 1:
        return [], 0
    truth = positive.iloc[0]
    negative = frame[~frame.index.isin(positive.index)].copy()
    truth_address = "" if pd.isna(truth.get("address_id")) else str(truth.get("address_id"))
    same_site = (
        negative["candidate_siren"].astype(str).eq(str(truth["candidate_siren"]))
        & negative["address_id"].fillna("").astype(str).ne("")
        & negative["address_id"].fillna("").astype(str).eq(truth_address)
    )
    excluded = int(same_site.sum())
    negative = negative[~same_site].sort_values(
        ["business_ranker_rank", "retrieval_rank", "candidate_siret"], kind="mergesort"
    )
    chosen: list[int] = []

    def take(mask: pd.Series, count: int) -> None:
        remaining = negative[mask & ~negative.index.isin(chosen)].head(count)
        chosen.extend(remaining.index.tolist())

    take(pd.Series(True, index=negative.index), 5)
    take(negative["candidate_siren"].astype(str).eq(str(truth["candidate_siren"])), 3)
    take(
        negative[["source_name_score", "name_jaro_max"]].max(axis=1).ge(HOMONYM_THRESHOLD)
        | (negative["addr_jaro"].ge(0.98) & negative["postcode_match"].eq(1)),
        3,
    )
    take(
        negative["candidate_state"].isin(["A", "F"])
        & negative["candidate_state"].ne(str(truth["candidate_state"])),
        2,
    )
    take(pd.Series(True, index=negative.index), MAX_NEGATIVES - len(chosen))
    return chosen[:MAX_NEGATIVES], excluded


def _score_and_build_groups(
    *,
    labels: pd.DataFrame,
    business: pd.DataFrame,
    queries_text: pd.DataFrame,
    candidates_text: pd.DataFrame,
    model_path: Path,
) -> tuple[pd.DataFrame, dict[str, int]]:
    model = xgb.XGBRanker()
    model.load_model(model_path)
    scored = business.copy()
    scored["business_ranker_score"] = model.predict(
        scored[BUSINESS_FEATURE_ORDER].to_numpy(dtype=np.float32)
    ).astype(np.float32)
    scored = scored.sort_values(
        ["query_id", "business_ranker_score", "retrieval_rank", "candidate_siret"],
        ascending=[True, False, True, True], kind="mergesort",
    )
    scored["business_ranker_rank"] = (
        scored.groupby("query_id", sort=False).cumcount() + 1
    ).astype(np.int16)
    text = candidates_text[
        ["query_id", "candidate_siret", "candidate_text", "candidate_state", "address_id"]
    ].copy()
    for frame in (scored, text, labels, queries_text):
        frame["query_id"] = frame["query_id"].astype(str)
    scored["candidate_siret"] = scored["candidate_siret"].astype(str).str.zfill(14)
    text["candidate_siret"] = text["candidate_siret"].astype(str).str.zfill(14)
    merged = scored.merge(text, on=["query_id", "candidate_siret"], validate="one_to_one")
    label_by_id = labels.set_index("query_id")
    query_text_by_id = queries_text.set_index("query_id")["query_text"]
    output: list[dict[str, Any]] = []
    excluded_same_site = 0
    eligible = 0
    for query_id, frame in merged.groupby("query_id", sort=False):
        truth_siret = str(label_by_id.loc[query_id, "ground_truth_siret"]).zfill(14)
        frame = frame.copy()
        frame["is_ground_truth"] = frame["candidate_siret"].eq(truth_siret).astype(np.int8)
        if int(frame["is_ground_truth"].sum()) != 1:
            continue
        selected, excluded = _select_negative_indices(frame)
        excluded_same_site += excluded
        positive = frame[frame["is_ground_truth"].eq(1)].iloc[0]
        ordered = [positive, *[frame.loc[index] for index in selected]]
        eligible += 1
        for position, row in enumerate(ordered):
            output.append(
                {
                    "query_id": query_id,
                    "oof_fold": int(label_by_id.loc[query_id, "oof_fold"]),
                    "label_is_human_validated": False,
                    "ranker_weight": np.float32(1.0),
                    "candidate_siret": str(row.candidate_siret),
                    "candidate_siren": str(row.candidate_siren),
                    "retrieval_rank": int(row.retrieval_rank),
                    "business_ranker_score": np.float32(row.business_ranker_score),
                    "business_ranker_rank": int(row.business_ranker_rank),
                    "group_position": position,
                    "is_positive": int(position == 0),
                    "negative_category": "positive" if position == 0 else "synthetic_hard_negative",
                    "query_text": str(query_text_by_id.loc[query_id]),
                    "candidate_text": str(row.candidate_text),
                }
            )
    return pd.DataFrame(output), {
        "eligible_bge_scenes": eligible,
        "same_siren_same_site_negatives_excluded": excluded_same_site,
    }


def build(args: argparse.Namespace) -> Path:
    source_manifest = _verified(args.source, ("benchmark.parquet", "labels.parquet", "queries.parquet"))
    if source_manifest.get("schema_version") != SOURCE_SCHEMA:
        raise ValueError("Unexpected synthetic retrieval input schema")
    if source_manifest.get("positive_injection") is not False:
        raise ValueError("Synthetic source suggests positive injection")
    v7_manifest, v7 = _validate_channels(args.v7_channels, args.source)
    overlay_manifest, overlay = _validate_channels(args.overlay_channels, args.source)
    source_ids = set(pd.read_parquet(args.source / "labels.parquet", columns=["query_id"])["query_id"].astype(str))
    if set(v7["query_id"].astype(str)) != source_ids or set(overlay["query_id"].astype(str)) != source_ids:
        raise ValueError("Channel audits do not cover the complete synthetic corpus")
    admission = _admission(v7, overlay)

    ranker_manifest = _verified(args.ranker, ("models_business_learned/full.json",))
    if ranker_manifest.get("positive_injection") is not False:
        raise ValueError("Published ranker suggests positive injection")
    real_business_manifest = _verified(
        args.real_business_contract, ("candidates_business.parquet",)
    )
    real_identity = real_business_manifest.get("build_identity", {})
    if real_identity.get("establishments_sha256") != file_sha256(args.etablissements):
        raise ValueError("Establishment snapshot differs from frozen BUSINESS features")
    if real_identity.get("legal_units_sha256") != file_sha256(args.unites_legales):
        raise ValueError("Legal-unit snapshot differs from frozen BUSINESS features")
    if list(real_business_manifest.get("business_feature_order", [])) != BUSINESS_FEATURE_ORDER:
        raise ValueError("Frozen BUSINESS feature order changed")
    identity = {
        "schema_version": SCHEMA_VERSION,
        "builder_sha256": file_sha256(Path(__file__)),
        "source_manifest_sha256": file_sha256(args.source / "manifest.json"),
        "v7_manifest_sha256": file_sha256(args.v7_channels / "manifest.json"),
        "overlay_manifest_sha256": file_sha256(args.overlay_channels / "manifest.json"),
        "ranker_manifest_sha256": file_sha256(args.ranker / "manifest.json"),
        "real_business_manifest_sha256": file_sha256(args.real_business_contract / "manifest.json"),
        "establishments_sha256": file_sha256(args.etablissements),
        "legal_units_sha256": file_sha256(args.unites_legales),
        "candidate_budget": CANDIDATE_BUDGET,
        "internal_channel_k": INTERNAL_K,
        "business_feature_order": BUSINESS_FEATURE_ORDER,
        "train_folds": [2, 3, 4],
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

    benchmark = pd.read_parquet(args.source / "benchmark.parquet")
    labels = pd.read_parquet(args.source / "labels.parquet")
    queries = pd.read_parquet(args.source / "queries.parquet")
    for frame in (benchmark, labels, queries, admission, v7, overlay):
        frame["query_id"] = frame["query_id"].astype(str)
    labels_for_candidates = labels[["query_id", "label_kind", "ground_truth_siret", "ground_truth_siren"]].copy()
    labels_for_candidates["split"] = SPLIT

    args.output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{build_id}.", dir=args.output_root))
    try:
        base_candidates_path = temporary / "candidates.parquet"
        writer = CandidateWriter(base_candidates_path)
        try:
            diagnostics = build_split_candidates(
                split=SPLIT,
                benchmark=benchmark,
                labels=labels_for_candidates,
                admission=admission,
                v7_channels=v7,
                overlay_channels=overlay,
                v7_store=PartitionedCandidateStore(args.v7_partitions),
                overlay_store=PartitionedCandidateStore(args.overlay_partitions),
                writer=writer,
            )
        finally:
            writer.close()
        if any(diagnostics.values()):
            raise ValueError(f"Synthetic candidate feature diagnostics failed: {diagnostics}")
        shutil.copyfile(args.source / "queries.parquet", temporary / "queries.parquet")
        shutil.copyfile(args.source / "labels.parquet", temporary / "labels.parquet")

        enriched = _read_enriched_sources(
            temporary,
            args.etablissements,
            args.unites_legales,
            candidate_filename="candidates.parquet",
        )
        enriched = _relational_features(_source_features(enriched))
        missing = set(METADATA_COLUMNS + BUSINESS_FEATURE_ORDER) - set(enriched.columns)
        if missing:
            raise ValueError(f"Synthetic business projection missing: {sorted(missing)}")
        if not np.isfinite(enriched[BUSINESS_FEATURE_ORDER].to_numpy(dtype=np.float32)).all():
            raise ValueError("Synthetic BUSINESS features contain non-finite values")
        business = enriched[METADATA_COLUMNS + BUSINESS_FEATURE_ORDER].copy()
        business = business.sort_values(
            ["query_id", "retrieval_rank", "candidate_siret"], kind="mergesort"
        ).reset_index(drop=True)
        business.to_parquet(temporary / "candidates_business.parquet", index=False)

        queries_with_folds = queries.merge(
            labels[["query_id", "oof_fold"]], on="query_id", validate="one_to_one"
        )
        queries_with_folds.to_parquet(temporary / "queries_with_folds.parquet", index=False)
        _candidate_text_projection(
            candidates=base_candidates_path,
            queries=temporary / "queries_with_folds.parquet",
            etablissements=args.etablissements,
            unites_legales=args.unites_legales,
            output_queries=temporary / "queries_text.parquet",
            output_candidates=temporary / "candidates_text.parquet",
        )
        queries_text = pd.read_parquet(temporary / "queries_text.parquet")
        candidates_text = pd.read_parquet(temporary / "candidates_text.parquet")
        groups, group_diagnostics = _score_and_build_groups(
            labels=labels,
            business=business,
            queries_text=queries_text,
            candidates_text=candidates_text,
            model_path=args.ranker / "models_business_learned/full.json",
        )
        if groups.empty or groups.groupby("query_id")["is_positive"].sum().ne(1).any():
            raise ValueError("Synthetic BGE groups are empty or invalid")
        groups.to_parquet(temporary / "training_groups.parquet", index=False)

        all_query_ids = labels["query_id"].astype(str)
        candidate_counts = business.groupby("query_id").size().reindex(
            all_query_ids, fill_value=0
        )
        truth_hits = business.groupby("query_id")["is_ground_truth"].sum().reindex(
            all_query_ids, fill_value=0
        ).eq(1)
        outputs = (
            "candidates_business.parquet", "labels.parquet", "training_groups.parquet",
            "queries_text.parquet", "candidates_text.parquet",
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "build_identity": identity,
            "row_counts": {
                "queries": len(labels),
                "candidate_rows": len(business),
                "bge_scenes": int(groups["query_id"].nunique()),
                "bge_pairs": len(groups),
            },
            "retrieval": {
                "policy": "frozen_v7_weighted_fusion_plus_closed_overlay_quotas",
                "candidate_ceiling": int(candidate_counts.max()) if len(candidate_counts) else 0,
                "natural_truth_hits": int(truth_hits.sum()),
                "natural_truth_misses": int((~truth_hits).sum()),
                "recall_at_100": float(truth_hits.mean()),
                "positive_injection": False,
            },
            "bge_group_diagnostics": group_diagnostics,
            "business_ranker_use": "negative_mining_only_published_full_real_model",
            "business_feature_order": BUSINESS_FEATURE_ORDER,
            "candidate_ceiling": CANDIDATE_BUDGET,
            "positive_injection": False,
            "allowed_consumers": {
                "ranker": True,
                "bge": True,
                "risk_model": False,
                "calibration": False,
                "auto_thresholds": False,
            },
            "confirmation_fold_opened": False,
            "final_test_opened": False,
            "outputs": {name: file_sha256(temporary / name) for name in outputs},
        }
        _json_dump(temporary / "manifest.json", manifest)
        # Staging-only projections are deliberately omitted from the immutable bundle.
        for name in ("candidates.parquet", "queries.parquet", "queries_with_folds.parquet"):
            (temporary / name).unlink(missing_ok=True)
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--v7-channels", type=Path, required=True)
    parser.add_argument("--overlay-channels", type=Path, required=True)
    parser.add_argument("--v7-partitions", type=Path, default=DEFAULT_V7_PARTITIONS)
    parser.add_argument("--overlay-partitions", type=Path, default=DEFAULT_OVERLAY_PARTITIONS)
    parser.add_argument("--etablissements", type=Path, default=DEFAULT_ETABLISSEMENTS)
    parser.add_argument("--unites-legales", type=Path, default=DEFAULT_UNITES_LEGALES)
    parser.add_argument("--ranker", type=Path, default=DEFAULT_RANKER)
    parser.add_argument("--real-business-contract", type=Path, default=DEFAULT_REAL_BUSINESS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    print(build(parse_args()))
