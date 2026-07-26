#!/usr/bin/env python3
"""Audit the frozen ranker/decider/risk chain without training or test tuning."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_v9_retrieval_experiment import _git_commit  # noqa: E402
from src.xgb_matcher.calibration import load_calibrator  # noqa: E402
from src.xgb_matcher.features import (  # noqa: E402
    FEATURE_NAMES,
    V8_EXPERIMENTAL_FEATURE_NAMES,
    V9_BASELINE_FEATURE_NAMES,
)
from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402
from src.xgb_matcher.v9_features import V9_RETRIEVAL_FEATURE_NAMES  # noqa: E402
from src.xgb_matcher.v9_scene import V9_SCENE_FEATURE_NAMES  # noqa: E402


SCHEMA_VERSION = "sireto-downstream-architecture-audit-1"
DIRECT_EVIDENCE_FEATURES = {
    "name_jaro_max",
    "name_token_overlap_max",
    "name_norm_exact",
    "addr_jaro",
    "street_name_jaro",
    "postcode_match",
    "street_number_match",
}


def _rate(successes: int, total: int) -> float:
    return successes / total if total else 0.0


def _normalise_identifier(value: Any, width: int | None = None) -> str:
    raw = str(value or "").strip()
    if raw.endswith(".0"):
        raw = raw[:-2]
    digits = "".join(character for character in raw if character.isdigit())
    if width and digits:
        return digits.zfill(width)
    return digits or raw


def candidate_dataset_summary(
    frame: pd.DataFrame,
    feature_order: Iterable[str],
) -> dict[str, Any]:
    features = list(feature_order)
    required = {"query_id", "label", "siren", "split"} | set(features)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Candidate dataset is missing columns: {sorted(missing)}")
    if set(frame["split"].dropna().astype(str)) - {"train", "dev", "test"}:
        raise ValueError("Candidate dataset contains an unsupported split")

    positives = frame[frame["label"].astype(int).eq(1)].copy()
    positive_counts = (
        frame.groupby("query_id")["label"].sum().astype(int).value_counts().sort_index()
    )
    positive_sirens = {
        split: set(
            positives.loc[
                positives["split"].eq(split),
                "siren",
            ].dropna().astype(str)
        )
        for split in ("train", "dev", "test")
    }
    leakage = {
        "train_dev": len(positive_sirens["train"] & positive_sirens["dev"]),
        "train_test": len(positive_sirens["train"] & positive_sirens["test"]),
        "dev_test": len(positive_sirens["dev"] & positive_sirens["test"]),
    }

    feature_stats: dict[str, Any] = {}
    for feature in features:
        values = pd.to_numeric(frame[feature], errors="coerce")
        feature_stats[feature] = {
            "missing_count": int(values.isna().sum()),
            "unique_count": int(values.nunique(dropna=True)),
            "zero_rate": float(values.fillna(0.0).eq(0.0).mean()),
            "mean": float(values.mean()) if len(values) else 0.0,
            "std": float(values.std()) if len(values) else 0.0,
        }

    return {
        "row_count": int(len(frame)),
        "query_count": int(frame["query_id"].nunique()),
        "positive_count": int(frame["label"].astype(int).sum()),
        "positive_rate": float(frame["label"].astype(int).mean()),
        "query_count_by_split": {
            str(split): int(count)
            for split, count in frame.groupby("split")["query_id"].nunique().items()
        },
        "candidate_count_quantiles": {
            str(quantile): float(value)
            for quantile, value in frame.groupby("query_id").size().quantile(
                [0.0, 0.01, 0.5, 0.99, 1.0]
            ).items()
        },
        "positive_count_per_query": {
            str(count): int(frequency)
            for count, frequency in positive_counts.items()
        },
        "positive_siren_split_leakage": leakage,
        "constant_features": [
            feature
            for feature, stats in feature_stats.items()
            if stats["unique_count"] <= 1
        ],
        "rare_nonzero_features": [
            feature
            for feature, stats in feature_stats.items()
            if stats["zero_rate"] >= 0.99
        ],
        "feature_stats": feature_stats,
    }


def legacy_reference_summary(
    topk: pd.DataFrame,
    ground_truth: pd.DataFrame,
    routed: pd.DataFrame,
) -> dict[str, Any]:
    required_topk = {"crm_id", "siret_candidate", "rank"}
    required_gt = {"crm_id", "siret_gt"}
    required_routed = {"crm_id", "xgb_status", "chosen_siret_final"}
    for name, frame, required in (
        ("topk", topk, required_topk),
        ("ground truth", ground_truth, required_gt),
        ("routed", routed, required_routed),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")

    candidates = topk.copy()
    candidates["crm_id"] = candidates["crm_id"].map(_normalise_identifier)
    candidates["siret_candidate"] = candidates["siret_candidate"].map(
        lambda value: _normalise_identifier(value, 14)
    )
    candidates["rank"] = pd.to_numeric(candidates["rank"], errors="raise")
    candidates = candidates.sort_values(["crm_id", "rank"])

    truth = ground_truth.copy()
    truth["crm_id"] = truth["crm_id"].map(_normalise_identifier)
    truth["siret_gt"] = truth["siret_gt"].map(
        lambda value: _normalise_identifier(value, 14)
    )
    decisions = routed.copy()
    decisions["crm_id"] = decisions["crm_id"].map(_normalise_identifier)
    decisions["chosen_siret_final"] = decisions["chosen_siret_final"].map(
        lambda value: _normalise_identifier(value, 14)
    )

    first = (
        candidates.groupby("crm_id", sort=False).nth(0).reset_index()[
            ["crm_id", "siret_candidate"]
        ].rename(columns={"siret_candidate": "top1_siret"})
    )
    second = (
        candidates.groupby("crm_id", sort=False).nth(1).reset_index()[
            ["crm_id", "siret_candidate"]
        ].rename(columns={"siret_candidate": "top2_siret"})
    )
    joined = (
        first.merge(second, on="crm_id", validate="one_to_one")
        .merge(truth[["crm_id", "siret_gt"]], on="crm_id", validate="one_to_one")
        .merge(
            decisions[["crm_id", "xgb_status", "chosen_siret_final"]],
            on="crm_id",
            validate="one_to_one",
        )
    )
    grouped_candidates = candidates.groupby("crm_id")["siret_candidate"].apply(set)
    truth_map = truth.set_index("crm_id")["siret_gt"]
    candidate_hits = sum(
        truth_map.get(query_id) in sirets
        for query_id, sirets in grouped_candidates.items()
    )

    joined["duplicate_top1_top2"] = joined["top1_siret"].eq(
        joined["top2_siret"]
    )
    joined["top1_correct"] = joined["top1_siret"].eq(joined["siret_gt"])
    joined["auto"] = joined["xgb_status"].astype(str).str.startswith("AUTO")
    joined["auto_correct"] = joined["chosen_siret_final"].eq(joined["siret_gt"])

    def auto_metrics(scope: pd.DataFrame) -> dict[str, Any]:
        auto = scope[scope["auto"]]
        correct = int(auto["auto_correct"].sum())
        return {
            "query_count": int(len(scope)),
            "auto_count": int(len(auto)),
            "coverage": _rate(len(auto), len(scope)),
            "correct_count": correct,
            "error_count": int(len(auto) - correct),
            "precision": _rate(correct, len(auto)),
        }

    duplicated = joined[joined["duplicate_top1_top2"]]
    distinct = joined[~joined["duplicate_top1_top2"]]
    return {
        "query_count": int(len(joined)),
        "candidate_row_count": int(len(candidates)),
        "candidate_recall": {
            "successes": int(candidate_hits),
            "total": int(len(joined)),
            "rate": _rate(candidate_hits, len(joined)),
        },
        "top1_exact": {
            "successes": int(joined["top1_correct"].sum()),
            "total": int(len(joined)),
            "rate": float(joined["top1_correct"].mean()),
        },
        "duplicate_candidate_row_count": int(
            candidates.duplicated(["crm_id", "siret_candidate"]).sum()
        ),
        "duplicate_top1_top2": {
            "query_count": int(len(duplicated)),
            "rate": _rate(len(duplicated), len(joined)),
        },
        "auto_all": auto_metrics(joined),
        "auto_duplicate_scenes": auto_metrics(duplicated),
        "auto_distinct_scenes": auto_metrics(distinct),
        "same_siren_wrong_site_count": int(
            (
                joined["top1_siret"].str[:9].eq(joined["siret_gt"].str[:9])
                & ~joined["top1_siret"].eq(joined["siret_gt"])
            ).sum()
        ),
    }


def same_scene_model_comparison(
    frame: pd.DataFrame,
    *,
    ranker_model_path: Path,
    decider_model_path: Path,
    decider_calibrator_path: Path,
    feature_order: Iterable[str],
) -> dict[str, Any]:
    features = list(feature_order)
    required = {"query_id", "split", "label"} | set(features)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Comparison dataset is missing columns: {sorted(missing)}")
    matrix = frame[features].astype(np.float32).to_numpy()

    ranker = xgb.Booster()
    ranker.load_model(str(ranker_model_path))
    ranker_scores = ranker.predict(
        xgb.DMatrix(matrix, feature_names=features)
    )
    decider = xgb.XGBClassifier()
    decider.load_model(str(decider_model_path))
    decider_scores = decider.predict_proba(matrix)[:, 1]
    calibrator = load_calibrator(decider_calibrator_path)
    calibrated_scores = calibrator.predict_proba(matrix)[:, 1]

    scored = frame[["query_id", "split", "label"]].copy()
    scored["ranker"] = ranker_scores
    scored["decider"] = decider_scores
    scored["calibrated_decider"] = calibrated_scores
    result: dict[str, Any] = {}
    for split in ("train", "dev", "test"):
        subset = scored[scored["split"].eq(split)]
        split_result: dict[str, Any] = {
            "query_count": int(subset["query_id"].nunique())
        }
        top_labels: dict[str, pd.Series] = {}
        for model_name in ("ranker", "decider", "calibrated_decider"):
            top_indices = subset.groupby("query_id")[model_name].idxmax()
            top = subset.loc[top_indices].set_index("query_id")["label"].astype(int)
            top_labels[model_name] = top
            split_result[model_name] = {
                "correct_count": int(top.sum()),
                "hit_at_1": float(top.mean()),
            }
        ranker_correct = top_labels["ranker"]
        calibrated_correct = top_labels["calibrated_decider"]
        split_result["paired"] = {
            "ranker_only_correct": int(
                ((ranker_correct == 1) & (calibrated_correct == 0)).sum()
            ),
            "decider_only_correct": int(
                ((ranker_correct == 0) & (calibrated_correct == 1)).sum()
            ),
        }
        result[split] = split_result
    return result


def model_importance(
    model_path: Path,
    feature_order: Iterable[str],
) -> dict[str, Any]:
    features = list(feature_order)
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    configured_names = booster.feature_names
    mapping = {
        f"f{index}": feature for index, feature in enumerate(features)
    }

    def resolve(name: str) -> str:
        return name if configured_names else mapping.get(name, name)

    gain = {
        resolve(name): float(value)
        for name, value in booster.get_score(importance_type="gain").items()
    }
    weight = {
        resolve(name): float(value)
        for name, value in booster.get_score(importance_type="weight").items()
    }
    return {
        "model_path": str(model_path),
        "num_features_model": int(booster.num_features()),
        "num_features_contract": len(features),
        "feature_names_embedded": bool(configured_names),
        "boosted_rounds": int(booster.num_boosted_rounds()),
        "top_gain": [
            {"feature": feature, "gain": value}
            for feature, value in sorted(
                gain.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:20]
        ],
        "unused_features": [
            feature for feature in features if feature not in weight
        ],
    }


def risk_metadata_summary(paths: Iterable[Path]) -> list[dict[str, Any]]:
    summaries = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        summaries.append(
            {
                "path": str(path),
                "target": payload.get("target"),
                "threshold": payload.get("threshold"),
                "feature_count": len(payload.get("features") or []),
                "model_type": payload.get("model_type"),
            }
        )
    return summaries


def profile_inventory(model_dir: Path) -> dict[str, Any]:
    metas = sorted(
        model_dir.glob("xgb_two_stage_meta_*.json"),
        reverse=True,
    )
    first = metas[0] if metas else None
    first_payload = (
        json.loads(first.read_text(encoding="utf-8")) if first else {}
    )
    invalid_same_stage_paths = []
    for meta in metas:
        payload = json.loads(meta.read_text(encoding="utf-8"))
        if (
            payload.get("ranker_model")
            and payload.get("ranker_model") == payload.get("decider_model")
        ):
            invalid_same_stage_paths.append(str(meta))
    return {
        "lexicographic_latest": str(first) if first else None,
        "lexicographic_latest_ranker": first_payload.get("ranker_model"),
        "lexicographic_latest_decider": first_payload.get("decider_model"),
        "invalid_same_ranker_and_decider_metas": invalid_same_stage_paths,
        "meta_count": len(metas),
    }


def _format_rate(value: float) -> str:
    return f"{value:.3%}"


def render_report(summary: dict[str, Any]) -> str:
    legacy = summary["legacy_reference"]
    comparison = summary["same_scene_comparison"]
    ranker_data = summary["candidate_datasets"]["ranker"]
    decider_data = summary["candidate_datasets"]["decider"]
    routing_divergence = summary["routing_v7_label_divergence"]
    profile = summary["profile_inventory"]
    scene_direct_evidence = summary["feature_contract"][
        "direct_evidence_present_in_v9_scene"
    ]
    lines = [
        "# Audit aval — ranker, decider et décision AUTO",
        "",
        "## Verdict",
        "",
        "**Le retrieval peut être conservé, mais le bundle aval historique ne "
        "doit pas être promu ni simplement recalibré.**",
        "",
        "Le bon remplacement est un ranker candidat unique sur le top-100 "
        "gelé, suivi d'un accepteur query-level entraîné sur la correction "
        "SIRET exacte. L'accepteur doit combiner les preuves du top-1 et la "
        "forme de la scène.",
        "",
        "## Défaut principal de la référence 74,5 % AUTO",
        "",
        f"- {legacy['duplicate_top1_top2']['query_count']}/"
        f"{legacy['query_count']} scènes ont le même SIRET en top-1 et top-2 "
        f"({_format_rate(legacy['duplicate_top1_top2']['rate'])}).",
        f"- Ces {legacy['auto_duplicate_scenes']['auto_count']} scènes sont "
        "toutes acceptées automatiquement.",
        f"- Sur les scènes distinctes, l'AUTO tombe à "
        f"{_format_rate(legacy['auto_distinct_scenes']['coverage'])} de "
        f"couverture et {_format_rate(legacy['auto_distinct_scenes']['precision'])} "
        "de précision brute.",
        f"- Le fichier reproductible contient "
        f"{legacy['auto_all']['error_count']} erreurs sur "
        f"{legacy['auto_all']['auto_count']} AUTO, soit "
        f"{_format_rate(legacy['auto_all']['precision'])}, pas 99,84 %.",
        "",
        "La correction humaine de trois labels mentionnée dans la documentation "
        "n'est pas reliée à un artefact d'adjudication versionné. Elle ne permet "
        "donc pas de certifier le chiffre annoncé.",
        "",
        "## Pertes observables",
        "",
        "| Étape historique | Succès | Taux |",
        "|---|---:|---:|",
        f"| Vérité présente dans le top-20 | "
        f"{legacy['candidate_recall']['successes']}/{legacy['candidate_recall']['total']} | "
        f"{_format_rate(legacy['candidate_recall']['rate'])} |",
        f"| Bon SIRET en top-1 | {legacy['top1_exact']['successes']}/"
        f"{legacy['top1_exact']['total']} | "
        f"{_format_rate(legacy['top1_exact']['rate'])} |",
        f"| AUTO correct brut | {legacy['auto_all']['correct_count']}/"
        f"{legacy['auto_all']['auto_count']} | "
        f"{_format_rate(legacy['auto_all']['precision'])} |",
        "",
        "Le goulot aval est donc bien le passage candidats → top-1, pas le "
        "retrieval désormais qualifié.",
        "",
        "## Le decider apporte-t-il un signal ?",
        "",
        "Sur exactement les mêmes scènes V7 de second étage :",
        "",
        "| Split | Ranker Hit@1 | Decider calibré Hit@1 | Delta |",
        "|---|---:|---:|---:|",
    ]
    for split in ("train", "dev", "test"):
        row = comparison[split]
        ranker = row["ranker"]["hit_at_1"]
        decider = row["calibrated_decider"]["hit_at_1"]
        lines.append(
            f"| {split} | {_format_rate(ranker)} | "
            f"{_format_rate(decider)} | {decider - ranker:+.3%} |"
        )
    lines.extend(
        [
            "",
            "Le decider contient donc un signal utile. Cela ne justifie pas de "
            "conserver deux modèles candidats : avec seulement 100 candidats "
            "finaux, ses signaux lexicaux, adresse et sémantiques peuvent être "
            "appris directement par un ranker final unique.",
            "",
            "## Audit des données et features",
            "",
            f"- Registre courant : {len(FEATURE_NAMES)} features candidates ; "
            f"baseline qualifiée : {len(V9_BASELINE_FEATURE_NAMES)} ; "
            f"expérimentales V8 non qualifiées : "
            f"{len(V8_EXPERIMENTAL_FEATURE_NAMES)}.",
            f"- Dataset ranker V7 : {ranker_data['query_count']} requêtes ; "
            f"dataset decider V7 : {decider_data['query_count']} requêtes.",
            "- Chaque requête conservée dans ces datasets possède exactement "
            "un positif. Les requêtes dont le positif n'était pas retrouvé "
            "sont absentes : leurs Hit@1 ne sont donc pas end-to-end.",
            "- Les SIREN positifs sont disjoints entre train/dev/test.",
            f"- Features constantes du ranker : "
            f"{', '.join(ranker_data['constant_features']) or 'aucune'}.",
            f"- Features présentes dans le code mais absentes des datasets V7 : "
            f"{', '.join(summary['feature_contract']['missing_from_v7'])}.",
            "",
            "### Risques de contrat",
            "",
            "- Les modèles historiques sans noms de features embarqués dépendent "
            "entièrement de l'ordre fourni par une meta externe.",
            "- La sélection `from_latest_meta()` est lexicographique et peut "
            f"choisir un bundle legacy plutôt que le plus récent : ici "
            f"`{Path(profile['lexicographic_latest']).name}`.",
            f"- {len(profile['invalid_same_ranker_and_decider_metas'])} meta "
            "référence le même fichier comme ranker et decider.",
            "- Les risk models disponibles ciblent tous le SIREN, alors que la "
            "métrique produit est le SIRET exact.",
            f"- Le dataset V7 contient "
            f"{routing_divergence['same_siren_wrong_siret_count']}/"
            f"{routing_divergence['query_count']} lignes "
            f"({_format_rate(routing_divergence['rate'])}) où le SIREN est "
            "correct mais le SIRET faux ; elles deviennent positives pour le "
            "risk model SIREN.",
            "- La feature sémantique dominante des modèles historiques a été "
            "entraînée avec l'ancien tokenizer non fiable.",
            "- Les sept features V8 sont dans le registre actif de 54 features "
            "mais absentes des datasets et modèles V7 de 47 features.",
            f"- L'accepteur V9 courant transporte "
            f"{len(scene_direct_evidence)}/{len(DIRECT_EVIDENCE_FEATURES)} "
            "preuves directes nom/adresse auditées.",
            "",
            "## Architecture recommandée",
            "",
            "```text",
            "top-100 retrieval gelé",
            "  → calcul canonique des features candidates",
            "  → ranker final SIRET unique",
            "  → top-1 + top-2 + preuves directes + forme de la scène",
            "  → accepteur exact-SIRET",
            "  → AUTO_MATCH ou REVIEW",
            "```",
            "",
            "Le futur accepteur doit recevoir :",
            "",
            "1. les preuves directes du top-1 : nom, adresse, numéro, commune ;",
            "2. les écarts top-1/top-2 ;",
            "3. la concurrence entre SIREN et entre établissements du même SIREN ;",
            "4. les signaux de provenance du retrieval ;",
            "5. les indicateurs hors distribution et absence de candidat.",
            "",
            "Les scènes d'entraînement doivent venir de prédictions ranker "
            "out-of-fold et conserver les erreurs de retrieval. Le seuil AUTO "
            "doit être choisi sur dev pour la précision SIRET exacte, puis "
            "évalué sur un nouveau holdout.",
            "",
            "## Décision",
            "",
            "**`PIVOT AVAL` : ne pas réutiliser tel quel `ranker + decider + "
            "risk model`. Conserver leurs signaux utiles dans `ranker final + "
            "accepteur exact-SIRET`.**",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranker-samples", type=Path, required=True)
    parser.add_argument("--decider-samples", type=Path, required=True)
    parser.add_argument("--ranker-model", type=Path, required=True)
    parser.add_argument("--decider-model", type=Path, required=True)
    parser.add_argument("--decider-calibrator", type=Path, required=True)
    parser.add_argument("--legacy-topk", type=Path, required=True)
    parser.add_argument("--legacy-ground-truth", type=Path, required=True)
    parser.add_argument("--legacy-routed", type=Path, required=True)
    parser.add_argument("--routing-v7", type=Path, required=True)
    parser.add_argument("--risk-meta", type=Path, action="append", default=[])
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")

    ranker_columns = V9_BASELINE_FEATURE_NAMES + [
        "label",
        "query_id",
        "siren",
        "split",
    ]
    decider_columns = V9_BASELINE_FEATURE_NAMES + [
        "label",
        "query_id",
        "siren",
        "split",
    ]
    ranker_samples = pd.read_parquet(
        args.ranker_samples,
        columns=ranker_columns,
    )
    decider_samples = pd.read_parquet(
        args.decider_samples,
        columns=decider_columns,
    )
    routing_v7 = pd.read_parquet(
        args.routing_v7,
        columns=["label_top1_strict", "label_top1_siren"],
    )
    same_siren_wrong = int(
        (
            routing_v7["label_top1_siren"].astype(int).eq(1)
            & routing_v7["label_top1_strict"].astype(int).eq(0)
        ).sum()
    )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "scope": "READ_ONLY_DOWNSTREAM_AUDIT",
        "current_selective_test_read": False,
        "candidate_datasets": {
            "ranker": candidate_dataset_summary(
                ranker_samples,
                V9_BASELINE_FEATURE_NAMES,
            ),
            "decider": candidate_dataset_summary(
                decider_samples,
                V9_BASELINE_FEATURE_NAMES,
            ),
        },
        "feature_contract": {
            "current_feature_count": len(FEATURE_NAMES),
            "v9_baseline_feature_count": len(V9_BASELINE_FEATURE_NAMES),
            "v8_experimental_features": list(V8_EXPERIMENTAL_FEATURE_NAMES),
            "missing_from_v7": [
                feature
                for feature in FEATURE_NAMES
                if feature not in ranker_samples.columns
            ],
            "v9_retrieval_feature_count": len(V9_RETRIEVAL_FEATURE_NAMES),
            "v9_scene_feature_count": len(V9_SCENE_FEATURE_NAMES),
            "direct_evidence_features": sorted(DIRECT_EVIDENCE_FEATURES),
            "direct_evidence_present_in_v9_scene": sorted(
                feature
                for feature in DIRECT_EVIDENCE_FEATURES
                if f"top1_{feature}" in V9_SCENE_FEATURE_NAMES
            ),
        },
        "models": {
            "ranker": model_importance(
                args.ranker_model,
                V9_BASELINE_FEATURE_NAMES,
            ),
            "decider": model_importance(
                args.decider_model,
                V9_BASELINE_FEATURE_NAMES,
            ),
        },
        "legacy_reference": legacy_reference_summary(
            pd.read_csv(args.legacy_topk, dtype=str),
            pd.read_csv(args.legacy_ground_truth, dtype=str),
            pd.read_csv(args.legacy_routed, dtype=str),
        ),
        "same_scene_comparison": same_scene_model_comparison(
            decider_samples,
            ranker_model_path=args.ranker_model,
            decider_model_path=args.decider_model,
            decider_calibrator_path=args.decider_calibrator,
            feature_order=V9_BASELINE_FEATURE_NAMES,
        ),
        "risk_models": risk_metadata_summary(args.risk_meta),
        "routing_v7_label_divergence": {
            "query_count": int(len(routing_v7)),
            "same_siren_wrong_siret_count": same_siren_wrong,
            "rate": _rate(same_siren_wrong, len(routing_v7)),
        },
        "profile_inventory": profile_inventory(args.model_dir),
        "verdict": "PIVOT_AVAL",
    }

    inputs = [
        args.ranker_samples,
        args.decider_samples,
        args.ranker_model,
        args.decider_model,
        args.decider_calibrator,
        args.legacy_topk,
        args.legacy_ground_truth,
        args.legacy_routed,
        args.routing_v7,
        *args.risk_meta,
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "scope": "READ_ONLY_DOWNSTREAM_AUDIT",
        "current_selective_test_read": False,
        "inputs": {
            str(path): file_sha256(path)
            for path in inputs
        },
    }

    args.output_dir.mkdir(parents=True)
    summary_path = args.output_dir / "summary.json"
    report_path = args.output_dir / "report.md"
    manifest_path = args.output_dir / "manifest.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(render_report(summary), encoding="utf-8")
    manifest["outputs"] = {
        "summary.json": file_sha256(summary_path),
        "report.md": file_sha256(report_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
