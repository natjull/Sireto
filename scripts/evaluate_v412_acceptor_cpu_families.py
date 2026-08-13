#!/usr/bin/env python3
"""Short CPU-family ablation for the role-aware V4.12 acceptor scenes.

All family comparisons use the same 104 features and the same nested
component-OOF protocol.  Model selection consumes only development labels.
The frozen 1,127 positive controls are evaluated once after selection; the
final test is never opened.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate_v412_acceptor_business_competition import (
    ACCEPTOR_FEATURES,
    _decision_metrics,
)
from scripts.run_v411_acceptor_development import (
    EXPECTED_MODEL_CONFIGS,
    select_threshold,
)
from src.xgb_matcher.v9_dataset import file_sha256


BASE = Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100")
DEFAULT_SCENES = (
    BASE
    / "experiments/v4_12_acceptor_business_competition/35b7ca460456a40b/"
    "acceptor_scenes_business.parquet"
)
DEFAULT_TRUSTED = Path("reports/v412_review_local_identifiable_labels_279.csv")
DEFAULT_ENSEMBLE = (
    BASE / "experiments/v4_12_conservative_ensemble/9ba1012722cc4b3f"
)
DEFAULT_CONTROL_OVERLAY = Path("reports/v412_control_label_counteraudit_4.csv")
DEFAULT_QUERIES = (
    BASE
    / "datasets/v4_11_input_blind/ec4326ec57e4411d/queries.parquet"
)
DEFAULT_OUTPUT_ROOT = BASE / "experiments/v4_12_acceptor_cpu_families"
SCHEMA_VERSION = "sireto-v4.12-acceptor-cpu-families-development-1"
TRUSTED_WEIGHT = 10.0
MINIMUM_CORRECT_AUTO = 148
SEED = 42


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["query_id"] = output["query_id"].astype(str)
    return output


def _xgb_unconstrained() -> Any:
    config = dict(EXPECTED_MODEL_CONFIGS["MONOTONIC_XGB"])
    config.pop("monotone_constraints", None)
    return xgb.XGBClassifier(**config)


def _extra_trees(min_leaf: int) -> Any:
    return ExtraTreesClassifier(
        n_estimators=240,
        min_samples_leaf=min_leaf,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=8,
        random_state=SEED,
    )


def _random_forest(min_leaf: int) -> Any:
    return RandomForestClassifier(
        n_estimators=240,
        min_samples_leaf=min_leaf,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=8,
        random_state=SEED,
    )


MODEL_FACTORIES: dict[str, Callable[[], Any]] = {
    "XGB_UNCONSTRAINED": _xgb_unconstrained,
    "EXTRA_TREES_LEAF_3": lambda: _extra_trees(3),
    "EXTRA_TREES_LEAF_5": lambda: _extra_trees(5),
    "EXTRA_TREES_LEAF_10": lambda: _extra_trees(10),
    "RANDOM_FOREST_LEAF_3": lambda: _random_forest(3),
    "RANDOM_FOREST_LEAF_5": lambda: _random_forest(5),
    "RANDOM_FOREST_LEAF_10": lambda: _random_forest(10),
}

NORTH_STAR_DIRECT_RULES = [
    {
        "feature": "delta_name_sim_max_ul",
        "operator": ">=",
        "value": 0.43,
    },
    {
        "feature": "delta_business_role_net_business",
        "operator": ">=",
        "value": 1.0,
    },
    {"feature": "delta_name_contains_crm_max", "operator": ">=", "value": 1.0},
    {"feature": "delta_addr_token_overlap", "operator": ">=", "value": 0.21},
    {"feature": "delta_idf_name", "operator": ">=", "value": 0.35},
    {"feature": "role_top1_count", "operator": ">=", "value": 1.0},
    {"feature": "delta_acronym_match_max", "operator": ">=", "value": 1.0},
    {
        "feature": "start_year_margin_other_siren_same_address_business",
        "operator": ">=",
        "value": 0.01,
    },
]
RULE_REASON_NAMES = {
    "delta_name_sim_max_ul": "LEGAL_NAME_SIMILARITY_ADVANTAGE",
    "delta_business_role_net_business": "BUSINESS_ROLE_ADVANTAGE",
    "delta_name_contains_crm_max": "CRM_NAME_CONTAINMENT_ADVANTAGE",
    "delta_addr_token_overlap": "ADDRESS_OVERLAP_ADVANTAGE",
    "delta_idf_name": "RARE_NAME_TOKEN_ADVANTAGE",
    "role_top1_count": "EXPLICIT_TOP1_ROLE",
    "delta_acronym_match_max": "ACRONYM_ADVANTAGE",
    "start_year_margin_other_siren_same_address_business": (
        "RECENCY_AT_SHARED_ADDRESS_ADVANTAGE"
    ),
}


def _sample_weight(frame: pd.DataFrame, trusted_ids: set[str]) -> np.ndarray:
    return np.where(
        frame["query_id"].astype(str).isin(trusted_ids), TRUSTED_WEIGHT, 1.0
    ).astype(np.float32)


def _fit(
    factory: Callable[[], Any], frame: pd.DataFrame, trusted_ids: set[str]
) -> Any:
    model = factory()
    model.fit(
        frame[ACCEPTOR_FEATURES].to_numpy(dtype=np.float32),
        frame["acceptor_target"].astype(int).to_numpy(),
        sample_weight=_sample_weight(frame, trusted_ids),
    )
    return model


def _scores(model: Any, frame: pd.DataFrame) -> np.ndarray:
    scores = np.asarray(
        model.predict_proba(frame[ACCEPTOR_FEATURES].to_numpy(dtype=np.float32))[
            :, 1
        ],
        dtype=float,
    )
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite acceptor scores")
    return scores


def _metrics(frame: pd.DataFrame) -> dict[str, Any]:
    metrics = _decision_metrics(frame)
    metrics["unresolved_auto"] = int(
        (
            frame["decision"].eq("AUTO_MATCH")
            & frame["label_kind"].eq("UNRESOLVED")
        ).sum()
    )
    return metrics


def _or_rules(frame: pd.DataFrame, rules: list[dict[str, Any]]) -> np.ndarray:
    accepted = np.zeros(len(frame), dtype=bool)
    for rule in rules:
        values = frame[str(rule["feature"])].to_numpy(dtype=float)
        threshold = float(rule["value"])
        if rule["operator"] == ">=":
            accepted |= values >= threshold
        elif rule["operator"] == "<=":
            accepted |= values <= threshold
        else:
            raise ValueError(f"Unsupported rule operator {rule['operator']}")
    return accepted


def _rule_reasons(frame: pd.DataFrame, rules: list[dict[str, Any]]) -> pd.Series:
    reasons: list[str] = []
    for _, row in frame.iterrows():
        matched: list[str] = []
        for rule in rules:
            feature = str(rule["feature"])
            value = float(row[feature])
            threshold = float(rule["value"])
            passes = value >= threshold if rule["operator"] == ">=" else value <= threshold
            if passes:
                matched.append(RULE_REASON_NAMES[feature])
        reasons.append("|".join(matched))
    return pd.Series(reasons, index=frame.index, dtype=str)


def _nested_oof(
    base: pd.DataFrame,
    trusted: pd.DataFrame,
    factory: Callable[[], Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Fit, calibrate and evaluate on disjoint SIREN-component folds."""

    outer_parts: list[pd.DataFrame] = []
    fold_results: list[dict[str, Any]] = []
    for outer in range(5):
        calibration_parts: list[pd.DataFrame] = []
        for inner in range(5):
            if inner == outer:
                continue
            fit_base = base[~base["oof_fold"].astype(int).isin([outer, inner])]
            fit_trusted = trusted[
                ~trusted["oof_fold"].astype(int).isin([outer, inner])
            ]
            model = _fit(
                factory,
                pd.concat([fit_base, fit_trusted], ignore_index=True),
                set(fit_trusted["query_id"].astype(str)),
            )
            held = trusted[trusted["oof_fold"].astype(int).eq(inner)].copy()
            held["acceptor_score"] = _scores(model, held)
            calibration_parts.append(held)

        calibration = pd.concat(calibration_parts, ignore_index=True)
        selected = select_threshold(
            calibration["acceptor_score"].to_numpy(),
            calibration["acceptor_target"].astype(int).to_numpy(),
            calibration["label_kind"].astype(str).to_numpy(),
        )
        if selected is None:
            raise ValueError(f"No inner threshold for outer fold {outer}")
        threshold, calibration_metrics, _ = selected

        outer_base = base[base["oof_fold"].astype(int).ne(outer)]
        outer_trusted = trusted[trusted["oof_fold"].astype(int).ne(outer)]
        model = _fit(
            factory,
            pd.concat([outer_base, outer_trusted], ignore_index=True),
            set(outer_trusted["query_id"].astype(str)),
        )
        held = trusted[trusted["oof_fold"].astype(int).eq(outer)].copy()
        held["acceptor_score"] = _scores(model, held)
        held["nested_threshold"] = float(threshold)
        held["decision"] = np.where(
            held["acceptor_score"].ge(threshold), "AUTO_MATCH", "REVIEW"
        )
        outer_parts.append(held)
        fold_results.append(
            {
                "outer_fold": outer,
                "inner_calibration_count": len(calibration),
                "threshold": float(threshold),
                "inner_calibration_metrics": calibration_metrics,
                "outer_metrics": _metrics(held),
            }
        )
    return pd.concat(outer_parts, ignore_index=True), fold_results


def _load_populations(
    scenes_path: Path, labels_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, set[str]]:
    scenes = _normalise(pd.read_parquet(scenes_path))
    if not set(ACCEPTOR_FEATURES).issubset(scenes.columns):
        raise ValueError("The 104-feature scene contract is incomplete")
    matrix = scenes[ACCEPTOR_FEATURES].to_numpy(dtype=np.float32)
    if matrix.shape[1] != 104 or not np.isfinite(matrix).all():
        raise ValueError("The 104-feature scene matrix is invalid")

    labels = pd.read_csv(labels_path, dtype=str, keep_default_na=False)
    labels["query_id"] = labels["query_id"].astype(str)
    if labels["label_kind"].value_counts().to_dict() != {
        "MATCH_EXACT": 241,
        "AMBIGUOUS": 31,
        "UNRESOLVED": 7,
    }:
        raise ValueError("Local quality labels changed")
    all_trusted_ids = set(labels["query_id"])
    base = scenes[
        scenes["split"].eq("fit") & scenes["label_kind"].eq("MATCH_EXACT")
    ].copy()
    trusted = scenes[scenes["query_id"].isin(all_trusted_ids)].copy()
    trusted.loc[trusted["label_kind"].eq("UNRESOLVED"), "acceptor_target"] = 0
    trusted["acceptor_target"] = trusted["acceptor_target"].astype(np.int8)
    if (
        len(base) != 4_666
        or len(trusted) != 279
        or int(trusted["acceptor_target"].sum()) != 227
        or int(trusted["acceptor_target"].eq(0).sum()) != 52
    ):
        raise ValueError("Role-aware acceptor populations changed")
    return scenes, base, trusted, all_trusted_ids


def _load_controls(
    scenes: pd.DataFrame,
    trusted_ids: set[str],
    ensemble_dir: Path,
    correction_path: Path,
) -> pd.DataFrame:
    frozen = _normalise(pd.read_parquet(ensemble_dir / "decisions.parquet"))
    control_ids = set(
        frozen.loc[frozen["scope"].eq("CONTROL"), "query_id"].astype(str)
    )
    if len(control_ids) != 1_127:
        raise ValueError("Frozen control identity changed")
    trusted_components = set(
        scenes.loc[
            scenes["query_id"].isin(trusted_ids), "siren_component_id"
        ].astype(str)
    )
    controls = scenes[
        scenes["query_id"].isin(control_ids)
        & scenes["split"].eq("dev")
        & ~scenes["siren_component_id"].astype(str).isin(trusted_components)
    ].copy()
    if set(controls["query_id"]) != control_ids:
        raise ValueError("The 1,127 controls are incomplete or share a trusted component")

    corrections = pd.read_csv(correction_path, dtype=str, keep_default_na=False)
    corrections["query_id"] = corrections["query_id"].astype(str)
    if set(corrections["query_id"]) - control_ids:
        raise ValueError("Control correction outside frozen scope")
    corrected = corrections.set_index("query_id")["corrected_ground_truth_siret"]
    controls = controls.set_index("query_id")
    controls.loc[corrected.index, "ground_truth_siret"] = corrected.astype(str)
    controls["acceptor_target"] = controls["predicted_siret"].astype(str).eq(
        controls["ground_truth_siret"].astype(str)
    ).astype(np.int8)
    controls = controls.reset_index()
    if len(controls) != 1_127 or not controls["acceptor_target"].eq(1).all():
        raise ValueError("Role-aware top1 is not correct on all 1,127 controls")
    return controls


def _selection_key(item: dict[str, Any]) -> tuple[int, int, float, str]:
    metrics = item["nested_oof_metrics"]
    return (
        0 if metrics["error_auto"] == 0 else 1,
        int(metrics["error_auto"]),
        -float(metrics["correct_top1_acceptance"]),
        str(item["family"]),
    )


def _combined_projection(
    difficult_metrics: dict[str, Any],
    control_metrics: dict[str, Any],
    policy: str,
) -> dict[str, Any]:
    auto_count = int(difficult_metrics["auto_count"] + control_metrics["auto_count"])
    correct_auto = int(
        difficult_metrics["correct_auto"] + control_metrics["correct_auto"]
    )
    return {
        "all_query_count": 1_406,
        "identifiable_query_count": 1_368,
        "correct_top1_count": 1_354,
        "auto_count": auto_count,
        "correct_auto": correct_auto,
        "error_auto": auto_count - correct_auto,
        "precision": correct_auto / auto_count if auto_count else None,
        "coverage_all_1406": auto_count / 1_406,
        "coverage_identifiable_1368": correct_auto / 1_368,
        "acceptance_available_correct_top1_1354": correct_auto / 1_354,
        "unresolved_query_count": 7,
        "policy": policy,
    }


def run(args: argparse.Namespace) -> Path:
    scenes_path = args.scenes.resolve()
    labels_path = args.trusted_labels.resolve()
    scenes, base, trusted, all_trusted_ids = _load_populations(
        scenes_path, labels_path
    )

    variants: dict[str, dict[str, Any]] = {}
    details: dict[str, pd.DataFrame] = {}
    for name, factory in MODEL_FACTORIES.items():
        detail, folds = _nested_oof(base, trusted, factory)
        metrics = _metrics(detail)
        variants[name] = {
            "family": name,
            "nested_oof_metrics": metrics,
            "outer_folds": folds,
            "eligible": bool(
                metrics["correct_auto"] >= MINIMUM_CORRECT_AUTO
                and metrics["error_auto"] == 0
                and metrics["ambiguous_auto"] == 0
            ),
        }
        details[name] = detail

    eligible = [item for item in variants.values() if item["eligible"]]
    selected = sorted(eligible or list(variants.values()), key=_selection_key)[0]
    selected_name = str(selected["family"])
    selected_detail = details[selected_name]
    global_selected = select_threshold(
        selected_detail["acceptor_score"].to_numpy(),
        selected_detail["acceptor_target"].astype(int).to_numpy(),
        selected_detail["label_kind"].astype(str).to_numpy(),
    )
    if global_selected is None:
        raise ValueError("No consumed-OOF threshold for selected model")
    threshold, single_threshold_metrics, _ = global_selected
    trusted_ids = set(trusted["query_id"].astype(str))
    final_model = _fit(
        MODEL_FACTORIES[selected_name],
        pd.concat([base, trusted], ignore_index=True),
        trusted_ids,
    )

    controls = _load_controls(
        scenes,
        all_trusted_ids,
        args.ensemble.resolve(),
        args.control_overlay.resolve(),
    )
    controls["acceptor_score"] = _scores(final_model, controls)
    controls["decision"] = np.where(
        controls["acceptor_score"].ge(threshold), "AUTO_MATCH", "REVIEW"
    )
    control_metrics = _metrics(controls)

    projection = _combined_projection(
        single_threshold_metrics,
        control_metrics,
        "FINAL_MODEL_AND_SINGLE_THRESHOLD_FROM_CONSUMED_NESTED_OOF",
    )

    # North Star policy: union of the safest nested-OOF CPU model and only
    # eight direct, interpretable evidence rules.  The rule thresholds were
    # selected on these consumed development labels; this is a projection to
    # freeze for a new holdout, not an OOF/certification claim.
    north_trusted = selected_detail.copy()
    north_trusted["model_auto"] = north_trusted["decision"].eq("AUTO_MATCH")
    north_trusted["rule_auto"] = _or_rules(
        north_trusted, NORTH_STAR_DIRECT_RULES
    )
    north_trusted["rule_reasons"] = _rule_reasons(
        north_trusted, NORTH_STAR_DIRECT_RULES
    )
    north_trusted["decision_reason"] = np.select(
        [
            north_trusted["model_auto"] & north_trusted["rule_auto"],
            north_trusted["model_auto"],
            north_trusted["rule_auto"],
        ],
        ["MODEL_AND_DIRECT_EVIDENCE", "MODEL", "DIRECT_EVIDENCE"],
        default="REVIEW",
    )
    north_trusted["decision"] = np.where(
        north_trusted["model_auto"] | north_trusted["rule_auto"],
        "AUTO_MATCH",
        "REVIEW",
    )
    north_trusted_metrics = _metrics(north_trusted)

    north_controls = controls.copy()
    north_controls["model_auto"] = north_controls["decision"].eq("AUTO_MATCH")
    north_controls["nested_threshold"] = float(threshold)
    north_controls["rule_auto"] = _or_rules(
        north_controls, NORTH_STAR_DIRECT_RULES
    )
    north_controls["rule_reasons"] = _rule_reasons(
        north_controls, NORTH_STAR_DIRECT_RULES
    )
    north_controls["decision_reason"] = np.select(
        [
            north_controls["model_auto"] & north_controls["rule_auto"],
            north_controls["model_auto"],
            north_controls["rule_auto"],
        ],
        ["MODEL_AND_DIRECT_EVIDENCE", "MODEL", "DIRECT_EVIDENCE"],
        default="REVIEW",
    )
    north_controls["decision"] = np.where(
        north_controls["model_auto"] | north_controls["rule_auto"],
        "AUTO_MATCH",
        "REVIEW",
    )
    north_control_metrics = _metrics(north_controls)
    north_projection = _combined_projection(
        north_trusted_metrics,
        north_control_metrics,
        "NESTED_MODEL_OR_EIGHT_DIRECT_RULES_SELECTED_ON_CONSUMED_DEV",
    )
    north_prudent_pass = bool(
        north_trusted_metrics["correct_auto"] >= MINIMUM_CORRECT_AUTO
        and north_trusted_metrics["error_auto"] == 0
        and north_trusted_metrics["ambiguous_auto"] == 0
        and north_trusted_metrics["unresolved_auto"] == 0
        and north_projection["error_auto"] == 0
    )
    north_star_pass = bool(
        north_prudent_pass
        and north_projection["precision"] >= 0.998
        and 0.88 <= north_projection["coverage_all_1406"] <= 0.92
        and 0.65 <= north_trusted_metrics["correct_top1_acceptance"] <= 0.75
    )

    identity_payload = {
        "schema": SCHEMA_VERSION,
        "scenes": file_sha256(scenes_path),
        "trusted": file_sha256(labels_path),
        "ensemble_decisions": file_sha256(
            args.ensemble.resolve() / "decisions.parquet"
        ),
        "control_overlay": file_sha256(args.control_overlay.resolve()),
        "features": ACCEPTOR_FEATURES,
        "families": list(MODEL_FACTORIES),
        "north_star_direct_rules": NORTH_STAR_DIRECT_RULES,
        "queries": file_sha256(args.queries.resolve()),
        "trusted_weight": TRUSTED_WEIGHT,
        "builder": file_sha256(Path(__file__)),
    }
    identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True).encode()
    ).hexdigest()[:16]
    output = args.output_root.resolve() / identity
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "CONSUMED_DEVELOPMENT_PLUS_FROZEN_POSITIVE_CONTROL",
        "final_test_opened": False,
        "production_promotion_authorized": False,
        "inputs": identity_payload,
        "feature_count": len(ACCEPTOR_FEATURES),
        "trusted_weight": TRUSTED_WEIGHT,
        "variants": variants,
        "selected_model": selected,
        "selected_threshold_from_consumed_oof": float(threshold),
        "selected_consumed_oof_at_single_threshold": single_threshold_metrics,
        "selected_positive_control_1127": control_metrics,
        "combined_projection": projection,
        "north_star_development_policy": {
            "combination": "NESTED_XGB_UNCONSTRAINED_OR_EIGHT_DIRECT_RULES",
            "rules_combination": "OR",
            "rules": NORTH_STAR_DIRECT_RULES,
            "trusted_consumed_metrics": north_trusted_metrics,
            "positive_control_1127": north_control_metrics,
            "combined_projection": north_projection,
            "observed_precision_gate": 0.998,
            "observed_coverage_all_gate": [0.88, 0.92],
            "observed_hard_correct_acceptance_gate": [0.65, 0.75],
            "status": "GO_NORTH_STAR_DEV_ZERO_ERROR" if north_star_pass else "STOP_NORTH_STAR_DEV",
        },
        "prudent_gate": {
            "minimum_correct_auto_over_227": MINIMUM_CORRECT_AUTO,
            "observed_small_sample_requirement": "ZERO_ERROR",
            "passed_by_final_union": north_prudent_pass,
        },
        "cpu_families_alone_verdict": (
            "GO_CPU_FAMILY" if eligible else "STOP_CPU_FAMILIES_ALONE"
        ),
        "prudent_zero_error_verdict": (
            "GO_PRUDENT_DEV_ZERO_ERROR"
            if north_prudent_pass
            else "STOP_PRUDENT_ZERO_ERROR"
        ),
        "verdict": "GO_NORTH_STAR_DEV_ZERO_ERROR" if north_star_pass else "STOP_NORTH_STAR_DEV",
        "limitations": {
            "trusted_labels_used_for_family_selection": True,
            "trusted_oof_used_for_final_development_threshold": True,
            "positive_control_contains_no_negative_cases": True,
            "combined_result_is_a_development_projection": True,
            "direct_rule_thresholds_selected_on_consumed_dev": True,
            "precision_99_8_certified": False,
        },
    }
    (output / "evaluation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    selected_detail[
        [
            "query_id",
            "label_kind",
            "ground_truth_siret",
            "predicted_siret",
            "acceptor_target",
            "oof_fold",
            "acceptor_score",
            "nested_threshold",
            "decision",
        ]
    ].to_parquet(output / "selected_trusted_nested_oof.parquet", index=False)
    controls[
        [
            "query_id",
            "ground_truth_siret",
            "predicted_siret",
            "acceptor_target",
            "acceptor_score",
            "decision",
        ]
    ].to_parquet(output / "selected_positive_control_1127.parquet", index=False)
    queries = _normalise(pd.read_parquet(args.queries.resolve()))[
        ["query_id", "crm_name", "crm_address", "crm_postcode", "crm_city"]
    ]
    north_decisions = pd.concat(
        [
            north_trusted.assign(evaluation_role="TRUSTED_CONSUMED_NESTED_OOF"),
            north_controls.assign(evaluation_role="POSITIVE_CONTROL_1127"),
        ],
        ignore_index=True,
    ).merge(queries, on="query_id", validate="one_to_one")
    north_decisions[
        [
            "query_id",
            "crm_name",
            "crm_address",
            "crm_postcode",
            "crm_city",
            "evaluation_role",
            "label_kind",
            "ground_truth_siret",
            "predicted_siret",
            "acceptor_target",
            "acceptor_score",
            "nested_threshold",
            "model_auto",
            "rule_auto",
            "rule_reasons",
            "decision_reason",
            "decision",
        ]
    ].to_parquet(output / "north_star_development_decisions.parquet", index=False)
    joblib.dump(final_model, output / "selected_acceptor.joblib")
    artifacts = {}
    for name in (
        "evaluation.json",
        "selected_trusted_nested_oof.parquet",
        "selected_positive_control_1127.parquet",
        "north_star_development_decisions.parquet",
        "selected_acceptor.joblib",
    ):
        path = output / name
        artifacts[name] = {
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_id": identity,
        "created_at": result["created_at"],
        "inputs": identity_payload,
        "artifacts": artifacts,
        "final_test_opened": False,
        "production_promotion_authorized": False,
        "verdict": result["verdict"],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=Path, default=DEFAULT_SCENES)
    parser.add_argument("--trusted-labels", type=Path, default=DEFAULT_TRUSTED)
    parser.add_argument("--ensemble", type=Path, default=DEFAULT_ENSEMBLE)
    parser.add_argument(
        "--control-overlay", type=Path, default=DEFAULT_CONTROL_OVERLAY
    )
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
