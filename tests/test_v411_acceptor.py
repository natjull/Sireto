from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.run_v411_acceptor_development import (
    _critical_family_masks,
    _fit_repeated,
    _load_scenes,
    _stack_dependencies,
    _variant_gate,
    decision_metrics,
    family_comparison,
    select_threshold,
)
from src.xgb_matcher.v9_dataset import file_sha256
from src.xgb_matcher.v411_acceptor import (
    COMPACT_LOGIT,
    MONOTONIC_XGB,
    OrderPreservingSelectiveStandardScaler,
    build_v411_acceptor,
)
from src.xgb_matcher.v411_scene import (
    V411_ACCEPTOR_FEATURE_NAMES,
    V411_MONOTONIC_CONSTRAINTS,
    V411_SCALED_FEATURE_NAMES,
)


LOGIT_CONFIG = {
    "C": 0.1,
    "solver": "lbfgs",
    "tol": 0.0001,
    "class_weight": None,
    "max_iter": 5000,
    "random_state": 42,
}
XGB_CONFIG = {
    "n_estimators": 4,
    "learning_rate": 0.03,
    "max_depth": 2,
    "min_child_weight": 1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_lambda": 10,
    "reg_alpha": 1,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "n_jobs": 1,
    "random_state": 42,
}


def _frame(rows: int = 12) -> pd.DataFrame:
    matrix = np.zeros((rows, len(V411_ACCEPTOR_FEATURE_NAMES)), dtype=float)
    matrix[:, 1] = np.linspace(0, 1, rows)
    matrix[:, 4] = np.linspace(1, 0, rows)
    frame = pd.DataFrame(matrix, columns=V411_ACCEPTOR_FEATURE_NAMES)
    frame["query_id"] = [f"q{index}" for index in range(rows)]
    frame["acceptor_target"] = [0, 1] * (rows // 2)
    frame["label_kind"] = np.where(
        frame["acceptor_target"].eq(1), "MATCH_EXACT", "AMBIGUOUS"
    )
    frame["input_siret_state"] = "ACTIVE"
    frame["source_segment"] = "train"
    return frame


def test_selective_scaler_preserves_order_and_binary_columns() -> None:
    scaled_indices = [
        V411_ACCEPTOR_FEATURE_NAMES.index(name)
        for name in V411_SCALED_FEATURE_NAMES
    ]
    scaler = OrderPreservingSelectiveStandardScaler(scaled_indices)
    frame = _frame()
    matrix = frame[V411_ACCEPTOR_FEATURE_NAMES].to_numpy()
    transformed = scaler.fit_transform(matrix)
    binary_index = V411_ACCEPTOR_FEATURE_NAMES.index("same_siren_top2")
    assert np.array_equal(transformed[:, binary_index], matrix[:, binary_index])
    assert transformed.shape == matrix.shape


def test_factories_pin_selective_scaling_and_monotonic_vector() -> None:
    logit = build_v411_acceptor(COMPACT_LOGIT, LOGIT_CONFIG)
    xgb = build_v411_acceptor(MONOTONIC_XGB, XGB_CONFIG)
    assert logit.named_steps["model"].C == 0.1
    assert tuple(xgb.get_params()["monotone_constraints"]) == tuple(
        V411_MONOTONIC_CONSTRAINTS
    )
    with pytest.raises(ValueError, match="Unsupported"):
        build_v411_acceptor("GRID_SEARCH", {})


def test_threshold_uses_exact_integer_998_per_thousand_rule() -> None:
    # The lowest score admits 500 rows with exactly one error: 99.8%.
    scores = np.arange(500, dtype=float)
    targets = np.ones(500, dtype=int)
    targets[0] = 0
    labels = np.where(targets == 1, "MATCH_EXACT", "AMBIGUOUS")
    selected = select_threshold(scores, targets, labels)
    assert selected is not None
    threshold, metrics, _ = selected
    assert threshold == 0.0
    assert metrics["auto_count"] == 500
    assert metrics["correct_auto"] == 499
    assert 1000 * metrics["correct_auto"] == 998 * metrics["auto_count"]


def test_threshold_rejects_zero_auto_even_though_precision_is_vacuous() -> None:
    scores = np.array([0.1, 0.2])
    targets = np.array([0, 0])
    labels = np.array(["AMBIGUOUS", "AMBIGUOUS"])
    assert select_threshold(scores, targets, labels) is None


def test_decision_gate_uses_precision_coverage_and_ambiguous_counts() -> None:
    metrics = decision_metrics(
        np.ones(500),
        np.r_[np.ones(499, dtype=int), 0],
        np.concatenate([np.repeat("MATCH_EXACT", 499), ["AMBIGUOUS"]]),
        0.5,
    )
    gate = _variant_gate(metrics, family_gate=True)
    assert gate["eligible"] is False
    assert gate["checks"]["precision_at_least_99_8"]
    assert not gate["checks"]["ambiguous_auto_zero"]


def test_scene_loader_excludes_unresolved_and_returns_forced_review_population(
    tmp_path: Path,
) -> None:
    scenes = _frame(12)
    scenes["split"] = ["fit"] * 7 + ["dev"] * 5
    scenes["dev_partition"] = (
        [""] * 7 + ["threshold_dev"] * 3 + ["comparison_dev"] * 2
    )
    scenes["predicted_siret"] = "11111111100001"
    scenes["ground_truth_siret"] = "11111111100001"
    scenes["ranker_prediction_is_out_of_sample"] = True
    scenes["input_siret_state"] = "ACTIVE"
    scenes["source_segment"] = "train"
    scenes.loc[11, "label_kind"] = "UNRESOLVED"
    scenes.loc[11, "acceptor_target"] = np.nan
    scenes_path = tmp_path / "scenes.parquet"
    scenes.to_parquet(scenes_path, index=False)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {"schema_version": "sireto-v4.11-compact-acceptor-dataset-1"}
        ),
        encoding="utf-8",
    )
    fit, threshold, comparison, unresolved = _load_scenes(
        {},
        {"scene_manifest": manifest_path, "scenes": scenes_path},
        enforce_canonical=False,
    )
    assert len(fit) == 7
    assert len(threshold) == 3
    assert len(comparison) == 1
    assert unresolved["query_id"].tolist() == ["q11"]
    assert unresolved["acceptor_target"].isna().all()


def test_stack_dependencies_rejects_non_promotable_ranker(
    tmp_path: Path,
) -> None:
    retrieval_manifest = tmp_path / "retrieval-manifest.json"
    retrieval_manifest.write_text(
        json.dumps({"build_id": "retrieval-build"}), encoding="utf-8"
    )
    ranker_dir = tmp_path / "ranker"
    (ranker_dir / "ranker_c").mkdir(parents=True)
    ranker_model = ranker_dir / "ranker_c/full_fit.json"
    ranker_model.write_text("model", encoding="utf-8")
    ranker_manifest = ranker_dir / "manifest.json"
    ranker_manifest.write_text(
        json.dumps(
            {
                "schema_version":
                    "sireto-v4.11-input-blind-ranker-c-development-1",
                "build_id": "ranker-build",
                "verdict": "GO_RANKER_C",
                "outputs": {
                    "ranker_c/full_fit.json": {
                        "sha256": file_sha256(ranker_model)
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    taxonomy = tmp_path / "taxonomy.json"
    taxonomy.write_text("{}", encoding="utf-8")
    contract = tmp_path / "contract.md"
    contract.write_text("contract", encoding="utf-8")
    scene_source = tmp_path / "v411_scene.py"
    scene_source.write_text("scene", encoding="utf-8")
    site_function_source = tmp_path / "v49_site_function.py"
    site_function_source.write_text("site-function", encoding="utf-8")

    def record(path: Path) -> dict[str, str]:
        return {"path": str(path), "sha256": file_sha256(path)}

    scene_manifest = tmp_path / "scene-manifest.json"
    payload = {
        "inputs": {
            "retrieval_dataset_manifest": record(retrieval_manifest),
            "ranker_artifact_manifest": record(ranker_manifest),
            "taxonomy": record(taxonomy),
            "contract": record(contract),
            "scene_source": record(scene_source),
            "site_function_source": record(site_function_source),
        }
    }
    scene_manifest.write_text(json.dumps(payload), encoding="utf-8")
    dependencies = _stack_dependencies({"scene_manifest": scene_manifest})
    assert dependencies["ranker_c"]["model_sha256"] == file_sha256(ranker_model)
    assert dependencies["scene"]["source_sha256"] == file_sha256(scene_source)
    assert dependencies["scene"]["site_function_source_sha256"] == file_sha256(
        site_function_source
    )
    payload["inputs"]["ranker_artifact_manifest"]["sha256"] = ""
    ranker_payload = json.loads(ranker_manifest.read_text(encoding="utf-8"))
    ranker_payload["verdict"] = "PIVOT_INPUT_BLIND_RANKER"
    ranker_manifest.write_text(json.dumps(ranker_payload), encoding="utf-8")
    payload["inputs"]["ranker_artifact_manifest"] = record(ranker_manifest)
    scene_manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="not promotable"):
        _stack_dependencies({"scene_manifest": scene_manifest})


def test_family_comparison_is_integer_exact_and_publishes_empty_pool_group() -> None:
    comparison = _frame(100)
    comparison["acceptor_target"] = 1
    comparison["label_kind"] = "MATCH_EXACT"
    comparison["top1_siren_candidate_count"] = 1
    comparison["role_crm_count"] = 0
    baseline = pd.DataFrame(
        {
            "query_id": comparison["query_id"],
            "baseline_auto": True,
            "baseline_correct": True,
        }
    )
    rows, passed = family_comparison(
        comparison,
        np.ones(100, dtype=bool),
        baseline,
        minimum_rows=100,
    )
    assert passed
    by_name = {row["family"]: row for row in rows}
    assert by_name["top1_siren_candidate_count=1"]["gated"]
    assert not by_name["top1_siren_candidate_count=0"]["gated"]
    assert by_name["role_crm_count=0"]["passed"]
    assert set(_critical_family_masks(comparison)) >= {
        "input_siret_state=ACTIVE",
        "source_segment=train",
        "top1_siren_candidate_count>1",
        "top1_siren_candidate_count=1",
        "role_crm_count>0",
        "role_crm_count=0",
    }


@pytest.mark.parametrize(
    ("family", "config"),
    [(COMPACT_LOGIT, LOGIT_CONFIG), (MONOTONIC_XGB, XGB_CONFIG)],
)
def test_two_repetitions_produce_bit_exact_scores(
    family: str,
    config: dict[str, object],
) -> None:
    frame = _frame()
    _, scores = _fit_repeated(
        family,
        config,
        frame,
        [frame],
        repetitions=2,
    )
    assert len(scores) == 1
    assert len(scores[0]) == len(frame)
