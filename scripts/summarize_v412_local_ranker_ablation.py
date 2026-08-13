#!/usr/bin/env python3
"""Score local-241 ranker ablations on the frozen 1,127 positive controls.

The control identities come from the frozen conservative ensemble
(``scope == CONTROL``), not from the business-feature evaluator's dynamically
computed ``non_trusted_dev`` subset.  The four counter-audited control truths
are overlaid in memory.  Canonical labels and dataset parquets are read-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate_v412_ranker_business_features import (
    BASE,
    DEFAULT_DATASET,
    DEFAULT_ETABLISSEMENTS,
    DEFAULT_UNITES_LEGALES,
    _read_enriched_sources,
    _relational_features,
    _source_features,
)


DEFAULT_EXPERIMENT_ROOT = (
    BASE / "experiments/v4_12_ranker_business_features_local241"
)
DEFAULT_ENSEMBLE = (
    BASE / "experiments/v4_12_conservative_ensemble/9ba1012722cc4b3f"
)
DEFAULT_CURRENT = (
    BASE / "experiments/v4_12_trusted_label_ranker/2f57628196fefce0"
)
DEFAULT_DERIVED = Path("reports/v412_review_local_identifiable_labels_279.csv")
DEFAULT_CONTROL_OVERLAY = Path("reports/v412_control_label_counteraudit_4.csv")
DEFAULT_SUMMARY = Path("reports/v412_local_ranker_ablation_results.csv")
DEFAULT_REPORT = Path("reports/v412_local_ranker_ablation.md")


def _top1(frame: pd.DataFrame, score: str) -> pd.DataFrame:
    ranked = frame.sort_values(
        ["query_id", score, "retrieval_rank", "candidate_siret"],
        ascending=[True, False, True, True],
        kind="mergesort",
    ).copy()
    ranked["ranker_rank"] = ranked.groupby("query_id", sort=False).cumcount() + 1
    return ranked


def _comparison(
    prediction: pd.Series, reference_prediction: pd.Series, truth: pd.Series
) -> dict[str, int]:
    prediction = prediction.reindex(truth.index)
    reference_prediction = reference_prediction.reindex(truth.index)
    if prediction.isna().any() or reference_prediction.isna().any():
        raise ValueError("Predictions are incomplete on the requested scope")
    correct = prediction.eq(truth)
    reference_correct = reference_prediction.eq(truth)
    return {
        "correct": int(correct.sum()),
        "reference_correct": int(reference_correct.sum()),
        "fixed": int((~reference_correct & correct).sum()),
        "regressed": int((reference_correct & ~correct).sum()),
    }


def run(args: argparse.Namespace) -> pd.DataFrame:
    experiment_root = args.experiment_root.resolve()
    runs = sorted(path for path in experiment_root.iterdir() if (path / "evaluation.json").exists())
    if len(runs) != 8:
        raise ValueError(f"Expected 8 experiment runs, found {len(runs)}")

    decisions = pd.read_parquet(args.ensemble.resolve() / "decisions.parquet")
    decisions["query_id"] = decisions["query_id"].astype(str)
    control_ids = set(
        decisions.loc[decisions["scope"].eq("CONTROL"), "query_id"].astype(str)
    )
    if len(control_ids) != 1_127:
        raise ValueError(f"Frozen CONTROL scope changed: {len(control_ids)} != 1127")

    dataset = args.dataset.resolve()
    labels = pd.read_parquet(dataset / "labels.parquet")
    labels["query_id"] = labels["query_id"].astype(str)
    control_truth = (
        labels[labels["query_id"].isin(control_ids)]
        .set_index("query_id")["ground_truth_siret"]
        .astype(str)
    )
    overlay = pd.read_csv(args.control_overlay, dtype=str, keep_default_na=False)
    correction = overlay.set_index("query_id")["corrected_ground_truth_siret"]
    if set(correction.index.astype(str)) - control_ids:
        raise ValueError("Control overlay contains IDs outside frozen CONTROL scope")
    control_truth.loc[correction.index.astype(str)] = correction.astype(str)
    if len(control_truth) != 1_127:
        raise ValueError("Control truth is incomplete")

    current_control = pd.read_parquet(
        args.current.resolve() / "non_trusted_dev_comparison.parquet"
    )
    current_control["query_id"] = current_control["query_id"].astype(str)
    current_control_prediction = current_control.set_index("query_id")[
        "predicted_siret_candidate"
    ].astype(str)
    if set(current_control_prediction.index) != control_ids:
        raise ValueError("Current-ranker control identity differs from frozen CONTROL scope")
    current_control_correct = int(
        current_control_prediction.reindex(control_truth.index).eq(control_truth).sum()
    )
    if current_control_correct != 1_123:
        raise ValueError(
            f"Expected corrected current control score 1123, got {current_control_correct}"
        )

    derived = pd.read_csv(args.derived_labels, dtype=str, keep_default_na=False)
    local_truth = (
        derived[derived["label_kind"].eq("MATCH_EXACT")]
        .set_index("query_id")["ground_truth_siret"]
        .astype(str)
    )
    if len(local_truth) != 241:
        raise ValueError(f"Expected 241 locally identifiable truths, got {len(local_truth)}")
    current_oof = pd.read_parquet(args.current.resolve() / "trusted_oof_comparison.parquet")
    current_oof["query_id"] = current_oof["query_id"].astype(str)
    current_oof_prediction = current_oof.set_index("query_id")[
        "predicted_siret_candidate"
    ].astype(str)
    current_oof_correct = int(
        current_oof_prediction.reindex(local_truth.index).eq(local_truth).sum()
    )
    if current_oof_correct != 212:
        raise ValueError(f"Expected current local OOF score 212, got {current_oof_correct}")

    enriched = _read_enriched_sources(
        dataset, args.etablissements, args.unites_legales
    )
    enriched = enriched[enriched["query_id"].isin(control_ids)].copy()
    enriched = _relational_features(_source_features(enriched))
    if enriched["query_id"].nunique() != 1_127:
        raise ValueError("Candidate features do not cover all 1,127 controls")

    four_ids = set(correction.index.astype(str))
    rows: list[dict[str, object]] = []
    for run_dir in runs:
        evaluation = json.loads((run_dir / "evaluation.json").read_text())
        if len(evaluation["variants"]) != 1:
            raise ValueError(f"Run {run_dir.name} does not contain exactly one variant")
        variant, payload = next(iter(evaluation["variants"].items()))
        features = payload["features"]
        model = xgb.XGBRanker()
        model.load_model(run_dir / f"{variant}_ranker.json")
        scored = enriched[
            ["query_id", "candidate_siret", "candidate_siren", "retrieval_rank"]
        ].copy()
        scored["ranker_score"] = model.predict(
            enriched[features].to_numpy(dtype=np.float32)
        ).astype("float32")
        ranked = _top1(scored, "ranker_score")
        ranked.to_parquet(
            run_dir / f"{variant}_control1127_corrected_ranked_candidates.parquet",
            index=False,
        )
        control_prediction = (
            ranked[ranked["ranker_rank"].eq(1)]
            .set_index("query_id")["candidate_siret"]
            .astype(str)
        )
        control = _comparison(
            control_prediction, current_control_prediction, control_truth
        )

        oof_path = run_dir / f"{variant}_trusted_oof_comparison.parquet"
        oof = pd.read_parquet(oof_path)
        oof["query_id"] = oof["query_id"].astype(str)
        if set(oof["query_id"]) != set(local_truth.index):
            raise ValueError(f"OOF identity mismatch in {run_dir.name}")
        oof_prediction = oof.set_index("query_id")["predicted_siret_candidate"].astype(str)
        oof_metrics = _comparison(oof_prediction, current_oof_prediction, local_truth)

        four_correct = int(
            control_prediction.reindex(sorted(four_ids))
            .eq(control_truth.reindex(sorted(four_ids)))
            .sum()
        )
        base_fit = payload["base_fit_oof"]
        rows.append(
            {
                "variant": variant,
                "hard_weight": float(evaluation["hard_weight"]),
                "run_id": run_dir.name,
                "feature_count": int(payload["feature_count"]),
                "oof_correct_241": oof_metrics["correct"],
                "oof_hit_at_1": oof_metrics["correct"] / 241,
                "oof_delta_vs_current": oof_metrics["correct"] - 212,
                "oof_fixed_vs_current": oof_metrics["fixed"],
                "oof_regressed_vs_current": oof_metrics["regressed"],
                "control_correct_1127": control["correct"],
                "control_hit_at_1": control["correct"] / 1_127,
                "control_delta_vs_current": control["correct"] - 1_123,
                "control_fixed_vs_current": control["fixed"],
                "control_regressed_vs_current": control["regressed"],
                "counteraudit_correct_4": four_correct,
                "base_fit_oof_correct": int(base_fit["top1_correct_count"]),
                "base_fit_oof_count": int(base_fit["query_count"]),
            }
        )

    summary = pd.DataFrame(rows).sort_values(
        ["control_regressed_vs_current", "oof_correct_241", "control_correct_1127"],
        ascending=[True, False, False],
        kind="mergesort",
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary, index=False)

    lines = [
        "# V4.12 — ablation ranker sur 241 labels localement identifiables",
        "",
        "Le score OOF courant recalculé sur le même périmètre est `212/241`.",
        "Le contrôle est l'ensemble figé de `1 127` IDs `scope=CONTROL`; les quatre vérités du contre-audit sont appliquées en mémoire. Le ranker courant vaut `1 123/1 127` après cette correction.",
        "",
        "| Variante | Poids | OOF 241 | Delta | Régressions OOF | Contrôle 1 127 | Fixes contrôle | Régressions contrôle | 4 contre-audits | Base fit OOF |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.to_dict("records"):
        lines.append(
            "| {variant} | {hard_weight:.2f} | {oof_correct_241}/241 ({oof_pct:.2f} %) | {oof_delta_vs_current:+d} | {oof_regressed_vs_current} | {control_correct_1127}/1127 ({control_pct:.2f} %) | {control_fixed_vs_current} | {control_regressed_vs_current} | {counteraudit_correct_4}/4 | {base_fit_oof_correct}/{base_fit_oof_count} |".format(
                **row,
                oof_pct=100 * row["oof_hit_at_1"],
                control_pct=100 * row["control_hit_at_1"],
            )
        )
    eligible = summary[summary["control_regressed_vs_current"].eq(0)]
    lines.extend(["", "## Verdict", ""])
    if eligible.empty:
        lines.append(
            "Aucune variante n'améliore le périmètre local sans vraie régression sur les 1 127 contrôles corrigés."
        )
    else:
        best = eligible.iloc[0]
        lines.append(
            f"La meilleure variante sans régression contrôle est `{best['variant']}` au poids `{best['hard_weight']:.2f}` : "
            f"`{int(best['oof_correct_241'])}/241` OOF et `{int(best['control_correct_1127'])}/1127` contrôles."
        )
        lines.append(
            f"Elle effectue `{int(best['oof_fixed_vs_current'])}` corrections et `{int(best['oof_regressed_vs_current'])}` régressions à l'intérieur des 241 cas, soit un gain net de `{int(best['oof_delta_vs_current']):+d}`. "
            f"Sur les 1 127 contrôles corrigés, elle effectue `{int(best['control_fixed_vs_current'])}` corrections sans perdre aucun des 1 123 cas déjà corrects."
        )
        lines.append(
            f"Le contrôle historique base-fit vaut `{int(best['base_fit_oof_correct'])}/{int(best['base_fit_oof_count'])}` contre `4655/4666` pour le ranker courant : le gate métier est propre, mais il ne faut pas présenter le résultat comme une absence de régression universelle."
        )
        best_candidates = pd.read_parquet(
            experiment_root
            / str(best["run_id"])
            / f"{best['variant']}_control1127_corrected_ranked_candidates.parquet"
        )
        best_top1 = (
            best_candidates[best_candidates["ranker_rank"].eq(1)]
            .set_index("query_id")["candidate_siret"]
            .astype(str)
        )
        lines.extend(
            [
                "",
                "### Quatre contrôles contre-audités — meilleure variante",
                "",
                "| Query | Ranker courant | Variante retenue | Vérité corrigée | Correct |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for query_id in overlay["query_id"].astype(str):
            predicted = best_top1.loc[query_id]
            corrected = control_truth.loc[query_id]
            lines.append(
                f"| `{query_id}` | `{current_control_prediction.loc[query_id]}` | `{predicted}` | `{corrected}` | {'oui' if predicted == corrected else 'non'} |"
            )
    lines.extend(
        [
            "",
            "Les sorties `non_trusted_dev` natives des runs portent sur 1 135 cas et ne sont pas utilisées pour ce gate. Chaque répertoire de run contient désormais une sortie candidats recalculée exactement sur les 1 127 contrôles figés.",
            "",
        ]
    )
    args.report.write_text("\n".join(lines), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--ensemble", type=Path, default=DEFAULT_ENSEMBLE)
    parser.add_argument("--current", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--derived-labels", type=Path, default=DEFAULT_DERIVED)
    parser.add_argument("--control-overlay", type=Path, default=DEFAULT_CONTROL_OVERLAY)
    parser.add_argument("--etablissements", type=Path, default=DEFAULT_ETABLISSEMENTS)
    parser.add_argument("--unites-legales", type=Path, default=DEFAULT_UNITES_LEGALES)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()).to_string(index=False))
