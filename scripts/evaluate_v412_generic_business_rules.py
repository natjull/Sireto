#!/usr/bin/env python3
"""Ablate generic business gates on the conservative V4.12 ensemble.

The rules never inspect query identifiers or ground truth when selecting a
candidate.  Label overlays are used only after prediction, for evaluation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
import pandas as pd

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


DEFAULT_ENSEMBLE = (
    BASE / "experiments/v4_12_conservative_ensemble/9ba1012722cc4b3f"
)
DEFAULT_QUALITY_OVERLAY = Path("reports/v412_trusted_label_quality_overlay.csv")
DEFAULT_OUTPUT_ROOT = BASE / "experiments/v4_12_generic_business_rules"
RULE_ORDER = ("seat_guard", "role_aware", "operating_over_holding", "same_siren_site")

Rule = Callable[[pd.Series, pd.DataFrame, pd.Series], tuple[str, str] | None]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_candidates(args: argparse.Namespace, query_ids: set[str]) -> pd.DataFrame:
    frame = _read_enriched_sources(
        args.dataset.resolve(),
        args.etablissements.resolve(),
        args.unites_legales.resolve(),
    )
    frame = frame[frame["query_id"].isin(query_ids)].copy()
    frame = _relational_features(_source_features(frame))

    # The historical feature treated 86.90 as neutral.  For a CRM asking for
    # a hospital/clinic, 86.10 is the establishment role and 86.22/86.90 are
    # explicit practitioner/other-health conflicts.
    hospital_query = frame["_crm"].str.contains(
        r"\b(?:HOPITAL|HOSPITALIER|CLINIQUE)\b", regex=True
    )
    activity = (
        frame["source_etab_activity"]
        .fillna(frame["activity_code"])
        .fillna("")
        .astype(str)
    )
    frame.loc[
        hospital_query & activity.str.startswith("86.10"), "business_role_match"
    ] = 1.0
    frame.loc[
        hospital_query & activity.str.startswith(("86.22", "86.90")),
        "business_role_conflict",
    ] = 1.0
    frame["source_activity"] = activity
    frame["etab_exact"] = frame[
        ["etab_name_exact", "etab_name_compact_exact"]
    ].max(axis=1)
    frame["ul_exact"] = frame[["ul_name_exact", "ul_name_compact_exact"]].max(
        axis=1
    )
    frame["source_name_score"] = frame[
        ["name_sim_max_ul", "name_sim_max_etab"]
    ].max(axis=1)
    frame["source_exact"] = frame[["etab_exact", "ul_exact"]].max(axis=1)
    frame["business_selection_score"] = (
        frame["name_addr_consistency"].astype(float)
        + 0.3 * frame["is_employer"].astype(float)
        + 0.1 * frame["has_known_effectif"].astype(float)
    ).astype("float32")
    return frame


def _seat_guard(
    decision: pd.Series, candidates: pd.DataFrame, current: pd.Series
) -> tuple[str, str] | None:
    base = candidates[candidates["candidate_siret"].eq(decision["base_siret"])]
    if decision["predicted_siret"] == decision["base_siret"] or len(base) != 1:
        return None
    base = base.iloc[0]
    if (
        str(base["candidate_siren"]) == str(current["candidate_siren"])
        and float(base["is_siege"]) == 1.0
        and float(current["is_siege"]) == 0.0
        and float(current["etab_exact"]) == 0.0
    ):
        return str(base["candidate_siret"]), "secondary_has_no_exact_crm_establishment_name"
    return None


def _role_aware(
    decision: pd.Series, candidates: pd.DataFrame, current: pd.Series
) -> tuple[str, str] | None:
    if float(current["business_role_conflict"]) != 1.0:
        return None
    alternatives = candidates[candidates["business_role_match"].eq(1)].copy()
    same_siren = alternatives[
        alternatives["candidate_siren"].astype(str).eq(str(current["candidate_siren"]))
    ]
    if len(same_siren):
        alternatives = same_siren
        reason = "role_match_same_siren"
    elif float(current["name_sim_max_ul"]) < 0.90:
        reason = "role_match_cross_siren_low_top_ul_similarity"
    else:
        return None
    if alternatives.empty:
        return None
    selected = alternatives.sort_values(
        ["business_selection_score", "retrieval_rank", "candidate_siret"],
        ascending=[False, True, True],
        kind="mergesort",
    ).iloc[0]
    selected_siret = str(selected["candidate_siret"])
    if selected_siret == str(current["candidate_siret"]):
        return None
    return selected_siret, reason


def _operating_over_holding(
    decision: pd.Series, candidates: pd.DataFrame, current: pd.Series
) -> tuple[str, str] | None:
    if (
        not (
            float(current["activity_is_holding"]) == 1.0
            or float(current["activity_is_property"]) == 1.0
        )
        or float(current["query_says_group_or_holding"]) == 1.0
    ):
        return None
    alternatives = candidates[
        candidates["activity_is_holding"].eq(0)
        & candidates["activity_is_property"].eq(0)
        & candidates["source_exact"].eq(1)
        & candidates["source_name_score"].ge(
            float(current["source_name_score"]) + 0.10
        )
    ].copy()
    if alternatives.empty:
        return None
    selected = alternatives.sort_values(
        ["source_name_score", "name_addr_consistency", "retrieval_rank", "candidate_siret"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).iloc[0]
    return str(selected["candidate_siret"]), "exact_operating_name_beats_holding_by_0.10"


def _same_siren_site(
    decision: pd.Series, candidates: pd.DataFrame, current: pd.Series
) -> tuple[str, str] | None:
    same_siren = candidates[
        candidates["candidate_siren"].astype(str).eq(str(current["candidate_siren"]))
    ].copy()

    # Prefer an explicitly named establishment, or a role-correct site when
    # the current site has an explicit role conflict.
    site = same_siren[
        same_siren["etab_exact"].eq(1)
        | (
            (float(current["business_role_conflict"]) == 1.0)
            & same_siren["business_role_match"].eq(1)
        )
    ].copy()
    if float(current["etab_exact"]) == 0.0 and len(site):
        selected = site.sort_values(
            [
                "etab_exact",
                "business_role_match",
                "business_selection_score",
                "retrieval_rank",
                "candidate_siret",
            ],
            ascending=[False, False, False, True, True],
            kind="mergesort",
        ).iloc[0]
        selected_siret = str(selected["candidate_siret"])
        if selected_siret != str(current["candidate_siret"]):
            return selected_siret, "same_siren_exact_establishment_or_role"

    # A legal-name-like CRM with no exact establishment name is generic.  In
    # that narrow case, choose the seat only when its address is not worse.
    seats = same_siren[same_siren["is_siege"].eq(1)]
    generic = (
        float(current["is_siege"]) == 0.0
        and not same_siren["etab_exact"].eq(1).any()
        and len(seats) == 1
        and float(current["name_sim_max_ul"]) >= 0.85
        and float(same_siren["name_sim_max_etab"].max()) < 0.90
    )
    if generic:
        seat = seats.iloc[0]
        if (
            float(seat["name_sim_max_ul"]) >= 0.85
            and float(seat["addr_jaro"]) >= float(current["addr_jaro"]) - 0.001
        ):
            return str(seat["candidate_siret"]), "same_siren_generic_crm_prefers_seat"
    return None


RULES: dict[str, Rule] = {
    "seat_guard": _seat_guard,
    "role_aware": _role_aware,
    "operating_over_holding": _operating_over_holding,
    "same_siren_site": _same_siren_site,
}


def _apply_rules(
    decisions: pd.DataFrame,
    groups: dict[str, pd.DataFrame],
    rule_names: tuple[str, ...],
) -> tuple[pd.Series, pd.Series]:
    prediction = decisions.set_index("query_id")["predicted_siret"].astype(str).copy()
    traces: dict[str, list[str]] = {query_id: [] for query_id in prediction.index}
    decision_rows = decisions.set_index("query_id", drop=False)
    for rule_name in rule_names:
        rule = RULES[rule_name]
        for query_id in prediction.index:
            candidates = groups[query_id]
            current_rows = candidates[
                candidates["candidate_siret"].eq(prediction.loc[query_id])
            ]
            if len(current_rows) != 1:
                raise ValueError(f"Missing current candidate for {query_id}")
            result = rule(
                decision_rows.loc[query_id], candidates, current_rows.iloc[0]
            )
            if result is None:
                continue
            selected_siret, reason = result
            if selected_siret != prediction.loc[query_id]:
                prediction.loc[query_id] = selected_siret
                traces[query_id].append(f"{rule_name}:{reason}")
    return prediction, pd.Series(
        {query_id: "|".join(items) if items else "KEEP_CONSERVATIVE_ENSEMBLE"
         for query_id, items in traces.items()},
        name="applied_business_rules",
    )


def _with_evaluation_truth(
    decisions: pd.DataFrame, quality_overlay: Path
) -> pd.DataFrame:
    output = decisions.copy()
    overlay = pd.read_csv(quality_overlay, dtype=str).fillna("")
    excluded = set(
        overlay.loc[
            overlay["scope_action"].isin(["EXCLUDE_LOCAL", "QUARANTINE_EXTERNAL"])
            | (
                overlay["scope_action"].eq("CORRECT")
                & ~overlay["recommended_kind"].eq("MATCH_EXACT")
            ),
            "query_id",
        ]
    )
    corrections = overlay[
        overlay["scope_action"].eq("CORRECT")
        & overlay["recommended_kind"].eq("MATCH_EXACT")
    ].set_index("query_id")["recommended_siret"]
    output["strict_truth_siret"] = output["query_id"].map(corrections).fillna(
        output["corrected_truth_siret"]
    )
    output["strict_local_241"] = output["scope"].eq("TRUSTED") & ~output[
        "query_id"
    ].isin(excluded)
    if int(output["strict_local_241"].sum()) != 241:
        raise ValueError("Expected exactly 241 strict local labels")
    return output


def _metrics(
    decisions: pd.DataFrame, prediction: pd.Series
) -> dict[str, dict[str, Any]]:
    predicted = decisions["query_id"].map(prediction)
    views = {
        "trusted_historical_254": (
            decisions["scope"].eq("TRUSTED"),
            decisions["historical_truth_siret"],
        ),
        "control_corrected_1127": (
            decisions["scope"].eq("CONTROL"),
            decisions["corrected_truth_siret"],
        ),
        "strict_local_241": (
            decisions["strict_local_241"],
            decisions["strict_truth_siret"],
        ),
        "strict_plus_controls_1368": (
            decisions["strict_local_241"] | decisions["scope"].eq("CONTROL"),
            decisions["strict_truth_siret"],
        ),
    }
    result: dict[str, dict[str, Any]] = {}
    base = decisions["predicted_siret"]
    changed = predicted.ne(base)
    for name, (mask, truth) in views.items():
        baseline_correct = base[mask].eq(truth[mask])
        candidate_correct = predicted[mask].eq(truth[mask])
        changed_view = changed[mask]
        result[name] = {
            "query_count": int(mask.sum()),
            "top1_correct_count": int(candidate_correct.sum()),
            "hit_at_1": float(candidate_correct.mean()),
            "changed_count": int(changed_view.sum()),
            "fixed_count": int((~baseline_correct & candidate_correct).sum()),
            "regressed_count": int((baseline_correct & ~candidate_correct).sum()),
            "wrong_to_wrong_count": int(
                (~baseline_correct & ~candidate_correct & changed_view).sum()
            ),
        }
    return result


def _ranked_candidates(
    enriched: pd.DataFrame,
    base_ranked: pd.DataFrame,
    decisions: pd.DataFrame,
) -> pd.DataFrame:
    base_columns = base_ranked[
        ["query_id", "candidate_siret", "ranker_score", "ranker_rank"]
    ].rename(
        columns={
            "ranker_score": "pre_rule_ranker_score",
            "ranker_rank": "pre_rule_ranker_rank",
        }
    )
    output = enriched.merge(
        base_columns,
        on=["query_id", "candidate_siret"],
        how="left",
        validate="one_to_one",
    )
    output["score_origin"] = np.where(
        output["pre_rule_ranker_rank"].notna(),
        "CONSERVATIVE_TOP20",
        "DETERMINISTIC_TAIL",
    )
    top20_max_rank = output.groupby("query_id")["pre_rule_ranker_rank"].transform(
        "max"
    )
    tail = output["pre_rule_ranker_rank"].isna()
    tail_order = (
        output.loc[tail]
        .sort_values(
            ["query_id", "retrieval_rank", "candidate_siret"],
            kind="mergesort",
        )
        .groupby("query_id", sort=False)
        .cumcount()
        .add(1)
    )
    output.loc[tail, "pre_rule_ranker_rank"] = (
        top20_max_rank[tail].fillna(0).astype(int) + tail_order
    )
    minimum_score = output.groupby("query_id")["pre_rule_ranker_score"].transform(
        "min"
    )
    output.loc[tail, "pre_rule_ranker_score"] = (
        minimum_score[tail]
        - 1.0
        - output.loc[tail, "pre_rule_ranker_rank"].astype(float) / 1000.0
    ).astype("float32")
    output["pre_rule_ranker_rank"] = output["pre_rule_ranker_rank"].astype("int16")
    output["ranker_score"] = output["pre_rule_ranker_score"].astype("float32")

    decision_index = decisions.set_index("query_id")
    changes = decision_index[
        decision_index["final_siret"].ne(decision_index["predicted_siret"])
    ]
    for row in changes.itertuples():
        query_mask = output["query_id"].eq(row.Index)
        previous = query_mask & output["candidate_siret"].eq(row.predicted_siret)
        selected = query_mask & output["candidate_siret"].eq(row.final_siret)
        if int(previous.sum()) != 1 or int(selected.sum()) != 1:
            raise ValueError(f"Cannot swap business-rule selection for {row.Index}")
        previous_score = float(output.loc[previous, "ranker_score"].iloc[0])
        selected_score = float(output.loc[selected, "ranker_score"].iloc[0])
        output.loc[previous, "ranker_score"] = selected_score
        output.loc[selected, "ranker_score"] = previous_score

    output = output.sort_values(
        ["query_id", "ranker_score", "retrieval_rank", "candidate_siret"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )
    output["ranker_rank"] = (
        output.groupby("query_id", sort=False).cumcount().add(1).astype("int16")
    )
    output = output.merge(
        decisions[
            ["query_id", "final_siret", "applied_business_rules"]
        ],
        on="query_id",
        validate="many_to_one",
    )
    output["selected_by_business_rules"] = output["candidate_siret"].eq(
        output["final_siret"]
    )
    observed = output[output["ranker_rank"].eq(1)].set_index("query_id")[
        "candidate_siret"
    ]
    expected = decision_index["final_siret"]
    if not observed.sort_index().equals(expected.sort_index()):
        raise ValueError("Ranked candidates do not reproduce final decisions")

    columns = [
        "query_id",
        "candidate_siret",
        "candidate_siren",
        "retrieval_rank",
        "ranker_score",
        "ranker_rank",
        "pre_rule_ranker_score",
        "pre_rule_ranker_rank",
        "score_origin",
        "selected_by_business_rules",
        "applied_business_rules",
        "is_siege",
        "name_sim_max_ul",
        "name_sim_max_etab",
        "addr_jaro",
        "name_addr_consistency",
        "etab_exact",
        "ul_exact",
        "business_role_match",
        "business_role_conflict",
        "business_selection_score",
        "activity_is_holding",
        "activity_is_property",
        "query_says_group_or_holding",
        "is_employer",
        "has_known_effectif",
        "source_activity",
        "source_ul_name",
        "source_enseigne1",
        "source_etab_usual",
    ]
    return output[columns].reset_index(drop=True)


def run(args: argparse.Namespace) -> Path:
    ensemble = args.ensemble.resolve()
    decisions = pd.read_parquet(ensemble / "decisions.parquet")
    decisions["query_id"] = decisions["query_id"].astype(str)
    decisions = _with_evaluation_truth(decisions, args.quality_overlay.resolve())
    query_ids = set(decisions["query_id"])
    enriched = _prepare_candidates(args, query_ids)
    groups = {
        str(query_id): group.copy()
        for query_id, group in enriched.groupby("query_id", sort=False)
    }
    if set(groups) != query_ids:
        raise ValueError("Enriched candidate scenes are incomplete")

    ablations: dict[str, dict[str, Any]] = {}
    all_predictions: dict[str, pd.Series] = {}
    all_traces: dict[str, pd.Series] = {}
    baseline_prediction = decisions.set_index("query_id")["predicted_siret"]
    ablations["conservative_baseline"] = _metrics(decisions, baseline_prediction)
    for rule_name in RULE_ORDER:
        prediction, traces = _apply_rules(decisions, groups, (rule_name,))
        all_predictions[rule_name] = prediction
        all_traces[rule_name] = traces
        ablations[rule_name] = _metrics(decisions, prediction)

    final_prediction, final_traces = _apply_rules(decisions, groups, RULE_ORDER)
    ablations["combined"] = _metrics(decisions, final_prediction)
    selected_rules = [
        rule_name
        for rule_name in RULE_ORDER
        if ablations[rule_name]["strict_local_241"]["regressed_count"] == 0
        and ablations[rule_name]["control_corrected_1127"]["regressed_count"] == 0
    ]
    if selected_rules != list(RULE_ORDER):
        raise ValueError(f"A business rule regressed a gated view: {selected_rules}")

    decisions["final_siret"] = decisions["query_id"].map(final_prediction)
    decisions["applied_business_rules"] = decisions["query_id"].map(final_traces)
    decisions["final_strict_correct"] = decisions["final_siret"].eq(
        decisions["strict_truth_siret"]
    )
    for rule_name in RULE_ORDER:
        decisions[f"{rule_name}_siret"] = decisions["query_id"].map(
            all_predictions[rule_name]
        )
        decisions[f"{rule_name}_trace"] = decisions["query_id"].map(
            all_traces[rule_name]
        )

    base_ranked = pd.read_parquet(ensemble / "ranked_candidates.parquet")
    ranked = _ranked_candidates(enriched, base_ranked, decisions)

    flat_rows = []
    for variant, variant_metrics in ablations.items():
        for view, values in variant_metrics.items():
            flat_rows.append({"variant": variant, "view": view, **values})
    ablation_frame = pd.DataFrame(flat_rows)

    payload = {
        "schema_version": "sireto-v4.12-generic-business-rules-development-1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "CONSUMED_DEVELOPMENT_ONLY",
        "final_test_opened": False,
        "selection_policy": {
            "uses_query_id_conditions": False,
            "uses_ground_truth_for_selection": False,
            "rule_order": list(RULE_ORDER),
            "promotion_gate": "zero regression on strict_local_241 and control_corrected_1127",
            "selected_rules": selected_rules,
            "tail_score_policy": "top20 conservative scores, deterministic lower tail, swap prior top1 and selected candidate scores",
        },
        "ablations": ablations,
        "objective": {
            "minimum_strict_correct": 225,
            "observed_strict_correct": ablations["combined"]["strict_local_241"][
                "top1_correct_count"
            ],
            "observed_control_correct": ablations["combined"][
                "control_corrected_1127"
            ]["top1_correct_count"],
            "passed": (
                ablations["combined"]["strict_local_241"]["top1_correct_count"]
                >= 225
                and ablations["combined"]["control_corrected_1127"][
                    "top1_correct_count"
                ]
                == 1127
            ),
        },
        "inputs": {
            "ensemble_evaluation_sha256": _sha256(ensemble / "evaluation.json"),
            "quality_overlay_sha256": _sha256(args.quality_overlay.resolve()),
            "dataset_manifest_sha256": _sha256(args.dataset.resolve() / "manifest.json"),
            "etablissements": {
                "path": str(args.etablissements.resolve()),
                "size": args.etablissements.resolve().stat().st_size,
                "mtime_ns": args.etablissements.resolve().stat().st_mtime_ns,
            },
            "unites_legales": {
                "path": str(args.unites_legales.resolve()),
                "size": args.unites_legales.resolve().stat().st_size,
                "mtime_ns": args.unites_legales.resolve().stat().st_mtime_ns,
            },
        },
    }
    identity = hashlib.sha256(
        json.dumps(
            {
                "schema": payload["schema_version"],
                "selection_policy": payload["selection_policy"],
                "inputs": payload["inputs"],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    output = args.output_root.resolve() / identity
    output.mkdir(parents=True, exist_ok=False)
    (output / "evaluation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    ablation_frame.to_csv(output / "ablation_metrics.csv", index=False)
    decisions.sort_values(["scope", "query_id"], kind="mergesort").to_parquet(
        output / "decisions.parquet", index=False
    )
    ranked.to_parquet(output / "ranked_candidates.parquet", index=False)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble", type=Path, default=DEFAULT_ENSEMBLE)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--quality-overlay", type=Path, default=DEFAULT_QUALITY_OVERLAY
    )
    parser.add_argument("--etablissements", type=Path, default=DEFAULT_ETABLISSEMENTS)
    parser.add_argument("--unites-legales", type=Path, default=DEFAULT_UNITES_LEGALES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
