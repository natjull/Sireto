"""Pure post-seal evaluation logic for the deterministic V4.12-G veto."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .v412_direct_evidence import apply_guard


SPLIT_COLUMNS = ["query_id", "siren_component_id", "split", "oof_fold"]
RANKER_COLUMNS = [
    "query_id",
    "candidate_siret",
    "candidate_siren",
    "retrieval_rank",
    "ranker_score",
    "prediction_origin",
    "oof_fold",
    "ranker_rank",
]
SCENE_METADATA_COLUMNS = [
    "query_id",
    "split",
    "dev_partition",
    "oof_fold",
    "siren_component_id",
    "label_kind",
    "ground_truth_siret",
    "predicted_siret",
    "acceptor_target",
    "ranker_prediction_is_out_of_sample",
    "prediction_origin",
    "input_siret_state",
    "source_segment",
    "top1_siren_candidate_count",
    "role_crm_count",
]
DECISION_COLUMNS = [
    "query_id",
    "population",
    "label_kind",
    "ground_truth_siret",
    "predicted_siret",
    "acceptor_target",
    "acceptor_score",
    "decision_v411",
    "review_reason_v411",
    "direct_candidate_count",
    "direct_siren_count",
    "sole_direct_siret",
    "sole_direct_siren",
    "sole_direct_in_top100",
    "decision_v412",
    "review_reason_v412",
    "correct_exact_siret",
    "input_siret_state",
    "source_segment",
    "top1_siren_candidate_count",
    "role_crm_count",
]
FIXED_THRESHOLD = 0.8720916706888049
CANONICAL_COUNTS = {
    "fit": 5547,
    "threshold_dev": 710,
    "comparison_dev": 746,
}
CANONICAL_LABEL_COUNTS = {
    "fit": {"MATCH_EXACT": 4666, "AMBIGUOUS": 881},
    "threshold_dev": {"MATCH_EXACT": 583, "AMBIGUOUS": 127},
    "comparison_dev": {"MATCH_EXACT": 634, "AMBIGUOUS": 112},
}
_SIRET = re.compile(r"^\d{14}$")
_SIREN = re.compile(r"^\d{9}$")


def _string_column(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].fillna("").astype(str)


def assign_populations(split: pd.DataFrame) -> pd.DataFrame:
    """Recompute all three populations without trusting scene partitions."""

    if list(split.columns) != SPLIT_COLUMNS:
        raise ValueError("STOP_V412_EVAL: split projection changed")
    if split["query_id"].duplicated().any():
        raise ValueError("STOP_V412_EVAL: duplicate split query")
    for column in ("query_id", "siren_component_id", "split"):
        if _string_column(split, column).str.strip().eq("").any():
            raise ValueError(f"STOP_V412_EVAL: empty split field {column}")
    if not set(_string_column(split, "split")).issubset({"fit", "dev"}):
        raise ValueError("STOP_V412_EVAL: unsupported split")
    component_splits = split.groupby("siren_component_id")["split"].nunique()
    if component_splits.gt(1).any():
        raise ValueError("STOP_V412_EVAL: SIREN component crosses split")
    output = split.copy()
    output["population"] = "fit"
    dev = output["split"].eq("dev")
    output.loc[dev, "population"] = output.loc[
        dev, "siren_component_id"
    ].map(
        lambda value: (
            "threshold_dev"
            if hashlib.sha256(
                f"v411-threshold:{value}".encode("utf-8")
            ).digest()[0]
            < 128
            else "comparison_dev"
        )
    )
    return output


def validate_population_parity(
    split: pd.DataFrame,
    scenes: pd.DataFrame,
    *,
    enforce_canonical: bool,
) -> pd.DataFrame:
    populations = assign_populations(split)
    missing = set(SCENE_METADATA_COLUMNS) - set(scenes.columns)
    if missing:
        raise ValueError(f"STOP_V412_EVAL: scene metadata missing {sorted(missing)}")
    if scenes["query_id"].duplicated().any():
        raise ValueError("STOP_V412_EVAL: duplicate scene query")
    if not scenes["ranker_prediction_is_out_of_sample"].map(
        lambda value: type(value) is bool
    ).all() or not pd.api.types.is_bool_dtype(
        scenes["ranker_prediction_is_out_of_sample"].dtype
    ) or not scenes["ranker_prediction_is_out_of_sample"].all():
        raise ValueError("STOP_V412_EVAL: ranker prediction is not OOS")
    if set(scenes["label_kind"].astype(str)) - {"MATCH_EXACT", "AMBIGUOUS"}:
        raise ValueError("STOP_V412_EVAL: unresolved scene imported")
    joined = populations.merge(
        scenes[
            [
                "query_id",
                "split",
                "dev_partition",
                "siren_component_id",
                "oof_fold",
            ]
        ],
        on="query_id",
        suffixes=("_split", "_scene"),
        validate="one_to_one",
    )
    if len(joined) != len(populations) or len(joined) != len(scenes):
        raise ValueError("STOP_V412_EVAL: split and scene query sets differ")
    for column in ("split", "siren_component_id"):
        if (
            _string_column(joined, f"{column}_split")
            != _string_column(joined, f"{column}_scene")
        ).any():
            raise ValueError(f"STOP_V412_EVAL: scene {column} differs")
    fold_equal = (
        joined["oof_fold_split"].fillna(-1).astype(int)
        == joined["oof_fold_scene"].fillna(-1).astype(int)
    )
    if not fold_equal.all():
        raise ValueError("STOP_V412_EVAL: scene OOF fold differs")
    dev_rows = _string_column(joined, "split_split").eq("dev")
    if (
        _string_column(joined.loc[dev_rows], "population")
        != _string_column(joined.loc[dev_rows], "dev_partition")
    ).any():
        raise ValueError("STOP_V412_EVAL: dev partition was not reproduced")
    populations = populations[["query_id", "population"]]
    for population in CANONICAL_COUNTS:
        expected = set(
            populations.loc[
                populations["population"].eq(population), "query_id"
            ].astype(str)
        )
        observed_mask = (
            scenes["split"].eq("fit")
            if population == "fit"
            else scenes["dev_partition"].eq(population)
        )
        observed = set(scenes.loc[observed_mask, "query_id"].astype(str))
        if expected != observed:
            raise ValueError(
                f"STOP_V412_EVAL: {population} query set differs"
            )
    if enforce_canonical:
        for population, expected_count in CANONICAL_COUNTS.items():
            subset = scenes[
                scenes["split"].eq("fit")
                if population == "fit"
                else scenes["dev_partition"].eq(population)
            ]
            if len(subset) != expected_count:
                raise ValueError(
                    f"STOP_V412_EVAL: canonical {population} count changed"
                )
            if (
                subset["label_kind"].value_counts().to_dict()
                != CANONICAL_LABEL_COUNTS[population]
            ):
                raise ValueError(
                    f"STOP_V412_EVAL: canonical {population} labels changed"
                )
    return populations


def validate_ranker_projection(
    ranker: pd.DataFrame,
    split: pd.DataFrame,
    scenes: pd.DataFrame,
) -> dict[str, set[str]]:
    if list(ranker.columns) != RANKER_COLUMNS:
        raise ValueError("STOP_V412_EVAL: ranker projection changed")
    if "is_ground_truth" in ranker.columns:
        raise ValueError("STOP_V412_EVAL: forbidden ranker label opened")
    if ranker.duplicated(["query_id", "candidate_siret"]).any():
        raise ValueError("STOP_V412_EVAL: duplicate ranker candidate")
    if not np.isfinite(ranker["ranker_score"].to_numpy(dtype=float)).all():
        raise ValueError("STOP_V412_EVAL: non-finite ranker score")
    if not _string_column(ranker, "candidate_siret").map(_SIRET.fullmatch).all():
        raise ValueError("STOP_V412_EVAL: invalid ranker SIRET")
    if not _string_column(ranker, "candidate_siren").map(_SIREN.fullmatch).all():
        raise ValueError("STOP_V412_EVAL: invalid ranker SIREN")
    if (
        _string_column(ranker, "candidate_siret").str[:9]
        != _string_column(ranker, "candidate_siren")
    ).any():
        raise ValueError("STOP_V412_EVAL: ranker SIRET/SIREN mismatch")
    grouped = ranker.groupby("query_id", sort=False)
    counts = grouped.size()
    if counts.gt(100).any():
        raise ValueError("STOP_V412_EVAL: ranker pool exceeds 100")
    for column in ("ranker_rank", "retrieval_rank"):
        if (
            ranker[column].isna().any()
            or not pd.api.types.is_integer_dtype(ranker[column].dtype)
            or ranker[column].le(0).any()
        ):
            raise ValueError(f"STOP_V412_EVAL: invalid {column}")
        if ranker[column].gt(100).any():
            raise ValueError(f"STOP_V412_EVAL: {column} exceeds 100")
    for query_id, group in grouped:
        ranks = sorted(group["ranker_rank"].astype(int).tolist())
        if ranks != list(range(1, len(group) + 1)):
            raise ValueError(
                f"STOP_V412_EVAL: non-contiguous ranker ranks {query_id}"
            )
    populations = assign_populations(split).set_index("query_id")
    ranker_with_split = ranker.merge(
        populations[["split", "oof_fold"]],
        left_on="query_id",
        right_index=True,
        suffixes=("_ranker", "_split"),
        validate="many_to_one",
    )
    if len(ranker_with_split) != len(ranker):
        raise ValueError("STOP_V412_EVAL: ranker query outside split")
    fit = ranker_with_split["split"].eq("fit")
    if not ranker_with_split.loc[fit, "prediction_origin"].eq(
        "ranker_c_oof"
    ).all():
        raise ValueError("STOP_V412_EVAL: fit ranker is not OOF")
    if not ranker_with_split.loc[~fit, "prediction_origin"].eq(
        "ranker_c_dev"
    ).all():
        raise ValueError("STOP_V412_EVAL: dev ranker origin changed")
    if (
        ranker_with_split.loc[fit, "oof_fold_ranker"].astype(int)
        != ranker_with_split.loc[fit, "oof_fold_split"].astype(int)
    ).any():
        raise ValueError("STOP_V412_EVAL: fit OOF fold changed")
    if ranker_with_split.loc[~fit, "oof_fold_ranker"].notna().any():
        raise ValueError("STOP_V412_EVAL: dev ranker carries OOF fold")
    top1 = ranker[ranker["ranker_rank"].astype(int).eq(1)][
        ["query_id", "candidate_siret", "prediction_origin", "oof_fold"]
    ]
    checked = scenes[
        ["query_id", "predicted_siret", "prediction_origin", "oof_fold"]
    ].merge(
        top1,
        on="query_id",
        how="left",
        suffixes=("_scene", "_ranker"),
        validate="one_to_one",
    )
    if checked["candidate_siret"].isna().any():
        raise ValueError("STOP_V412_EVAL: scene has no ranker top1")
    if (
        _string_column(checked, "predicted_siret")
        != _string_column(checked, "candidate_siret")
    ).any():
        raise ValueError("STOP_V412_EVAL: scene top1 differs from ranker")
    if (
        _string_column(checked, "prediction_origin_scene")
        != _string_column(checked, "prediction_origin_ranker")
    ).any() or (
        checked["oof_fold_scene"].fillna(-1).astype(int)
        != checked["oof_fold_ranker"].fillna(-1).astype(int)
    ).any():
        raise ValueError("STOP_V412_EVAL: scene/ranker provenance differs")
    return {
        str(query_id): set(group["candidate_siret"].astype(str))
        for query_id, group in grouped
    }


def score_v411(
    model: Any,
    scenes: pd.DataFrame,
    *,
    feature_order: Sequence[str],
    threshold: float = FIXED_THRESHOLD,
) -> pd.DataFrame:
    if float(threshold) != FIXED_THRESHOLD:
        raise ValueError("STOP_V412_EVAL: V4.11 threshold changed")
    if len(feature_order) != 80 or len(set(feature_order)) != 80:
        raise ValueError("STOP_V412_EVAL: feature order changed")
    missing = set(feature_order) - set(scenes.columns)
    if missing:
        raise ValueError("STOP_V412_EVAL: scene feature missing")
    matrix = scenes[list(feature_order)].to_numpy(dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError("STOP_V412_EVAL: non-finite acceptor feature")
    first = np.asarray(model.predict_proba(matrix)[:, 1], dtype=np.float64)
    second = np.asarray(model.predict_proba(matrix)[:, 1], dtype=np.float64)
    if not np.array_equal(first, second):
        raise ValueError("STOP_V412_EVAL: acceptor scores are not bit-exact")
    scores = first
    if len(scores) != len(scenes) or not np.isfinite(scores).all():
        raise ValueError("STOP_V412_EVAL: invalid acceptor score")
    output = scenes.copy()
    if (
        output["acceptor_target"].isna().any()
        or not pd.api.types.is_integer_dtype(output["acceptor_target"].dtype)
        or not set(output["acceptor_target"].astype(int)).issubset({0, 1})
    ):
        raise ValueError("STOP_V412_EVAL: invalid acceptor target")
    output["acceptor_score"] = scores
    valid_top1 = _string_column(output, "predicted_siret").map(
        lambda value: bool(_SIRET.fullmatch(value))
    )
    output["decision_v411"] = np.where(
        valid_top1 & output["acceptor_score"].ge(threshold),
        "AUTO_MATCH",
        "REVIEW",
    )
    output["review_reason_v411"] = np.where(
        output["decision_v411"].eq("AUTO_MATCH"),
        None,
        np.where(valid_top1, "LOW_CONFIDENCE", "NO_CANDIDATE"),
    )
    exact_correct = (
        output["label_kind"].eq("MATCH_EXACT")
        & _string_column(output, "predicted_siret").eq(
            _string_column(output, "ground_truth_siret")
        )
    )
    if (
        output["acceptor_target"].astype(int).astype(bool)
        != exact_correct
    ).any():
        raise ValueError("STOP_V412_EVAL: acceptor target differs from exact SIRET")
    output["correct_exact_siret"] = exact_correct
    return output


def apply_guard_frame(
    v411: pd.DataFrame,
    query_evidence: pd.DataFrame,
    *,
    ranker_sirets: Mapping[str, set[str]],
    populations: pd.DataFrame,
) -> pd.DataFrame:
    required_evidence = [
        "query_id",
        "direct_candidate_count",
        "direct_siren_count",
        "sole_direct_siret",
        "sole_direct_siren",
    ]
    if not set(required_evidence).issubset(query_evidence.columns):
        raise ValueError("STOP_V412_EVAL: sealed evidence fields missing")
    if query_evidence["query_id"].duplicated().any():
        raise ValueError("STOP_V412_EVAL: duplicate sealed query evidence")
    output = v411.merge(
        query_evidence[required_evidence],
        on="query_id",
        validate="one_to_one",
    ).merge(populations, on="query_id", validate="one_to_one")
    if len(output) != len(v411) or len(output) != len(query_evidence):
        raise ValueError("STOP_V412_EVAL: sealed evidence query set differs")
    output["sole_direct_in_top100"] = [
        (
            row.sole_direct_siret in ranker_sirets.get(str(row.query_id), set())
            if int(row.direct_candidate_count) == 1
            else None
        )
        for row in output.itertuples(index=False)
    ]
    decisions = [
        apply_guard(
            decision_v411=str(row.decision_v411),
            review_reason_v411=(
                None
                if pd.isna(row.review_reason_v411)
                else str(row.review_reason_v411)
            ),
            predicted_siret=(
                None if pd.isna(row.predicted_siret) else str(row.predicted_siret)
            ),
            direct_candidate_count=row.direct_candidate_count,
            sole_direct_siret=(
                None
                if pd.isna(row.sole_direct_siret)
                else str(row.sole_direct_siret)
            ),
        )
        for row in output.itertuples(index=False)
    ]
    output["decision_v412"] = [decision for decision, _ in decisions]
    output["review_reason_v412"] = [reason for _, reason in decisions]
    if (
        output["decision_v411"].eq("REVIEW")
        & output["decision_v412"].ne("REVIEW")
    ).any():
        raise ValueError("STOP_V412_EVAL: guard is not a veto")
    allowed = output["decision_v411"].eq("AUTO_MATCH") & output[
        "direct_candidate_count"
    ].eq(1) & _string_column(output, "sole_direct_siret").eq(
        _string_column(output, "predicted_siret")
    )
    if output.loc[allowed, "decision_v412"].ne("AUTO_MATCH").any():
        raise ValueError("STOP_V412_EVAL: allowed V4.11 decision changed")
    return output[DECISION_COLUMNS].sort_values(
        "query_id", kind="mergesort"
    ).reset_index(drop=True)


def _decision_metrics(frame: pd.DataFrame, decision_column: str) -> dict[str, Any]:
    auto = frame[decision_column].eq("AUTO_MATCH")
    correct = frame["correct_exact_siret"].astype(bool)
    auto_count = int(auto.sum())
    correct_auto = int((auto & correct).sum())
    return {
        "row_count": len(frame),
        "auto_count": auto_count,
        "correct_auto": correct_auto,
        "error_auto": auto_count - correct_auto,
        "ambiguous_auto": int((auto & frame["label_kind"].eq("AMBIGUOUS")).sum()),
        "coverage": auto_count / len(frame) if len(frame) else 0.0,
        "precision": correct_auto / auto_count if auto_count else None,
    }


def _family_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    masks: dict[str, pd.Series] = {}
    for column in ("input_siret_state", "source_segment"):
        for value in sorted(frame[column].astype(str).unique()):
            masks[f"{column}={value}"] = frame[column].astype(str).eq(value)
    siren_count = frame["top1_siren_candidate_count"].astype(float)
    masks["top1_siren_candidate_count>1"] = siren_count.gt(1)
    masks["top1_siren_candidate_count=1"] = siren_count.eq(1)
    role_count = frame["role_crm_count"].astype(float)
    masks["role_crm_count>0"] = role_count.gt(0)
    masks["role_crm_count=0"] = role_count.eq(0)
    return masks


def evaluate_comparison_gate(
    decisions: pd.DataFrame,
    *,
    enforce_canonical: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    comparison = decisions[
        decisions["population"].eq("comparison_dev")
    ].copy()
    baseline = _decision_metrics(comparison, "decision_v411")
    candidate = _decision_metrics(comparison, "decision_v412")
    if enforce_canonical and (
        baseline["row_count"] != 746
        or baseline["auto_count"] != 614
        or baseline["error_auto"] != 0
        or baseline["ambiguous_auto"] != 0
    ):
        raise ValueError("STOP_V412_EVAL: V4.11 reference not reproduced")
    global_gate = (
        candidate["auto_count"] >= 600
        and candidate["error_auto"] == 0
        and candidate["ambiguous_auto"] == 0
    )
    families: list[dict[str, Any]] = []
    segment_gate = True
    for family, mask in _family_masks(comparison).items():
        rows = comparison[mask]
        baseline_auto = int(rows["decision_v411"].eq("AUTO_MATCH").sum())
        candidate_auto = int(rows["decision_v412"].eq("AUTO_MATCH").sum())
        gated = len(rows) >= 100
        passed = (
            100 * (baseline_auto - candidate_auto) <= 2 * len(rows)
            if gated
            else True
        )
        segment_gate = segment_gate and passed
        families.append(
            {
                "family": family,
                "row_count": len(rows),
                "v411_auto_count": baseline_auto,
                "v412_auto_count": candidate_auto,
                "auto_loss_count": baseline_auto - candidate_auto,
                "gated": gated,
                "coverage_noninferiority_pass": passed,
            }
        )
    reasons = {
        str(reason): int(count)
        for reason, count in comparison.loc[
            comparison["decision_v412"].eq("REVIEW"), "review_reason_v412"
        ].value_counts(dropna=False).sort_index().items()
    }
    metrics = {
        "population": "comparison_dev",
        "v411": baseline,
        "v412_g": candidate,
        "review_reason_counts": reasons,
        "global_gate_pass": global_gate,
        "segment_gate_pass": segment_gate,
        "verdict": (
            "GO_V412_HISTORICAL_GATE"
            if global_gate and segment_gate
            else "STOP_V412_GUARD"
        ),
        "latency_gate_evaluated": False,
        "production_certified": False,
        "gate_formula": {"minimum_auto": 600, "maximum_errors": 0,
                         "maximum_ambiguous_auto": 0},
    }
    return metrics, families


__all__ = [
    "CANONICAL_COUNTS",
    "CANONICAL_LABEL_COUNTS",
    "DECISION_COLUMNS",
    "FIXED_THRESHOLD",
    "RANKER_COLUMNS",
    "SCENE_METADATA_COLUMNS",
    "SPLIT_COLUMNS",
    "apply_guard_frame",
    "assign_populations",
    "evaluate_comparison_gate",
    "score_v411",
    "validate_population_parity",
    "validate_ranker_projection",
]
