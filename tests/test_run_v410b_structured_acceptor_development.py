from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from scripts import run_v410b_structured_acceptor_development as subject
from src.xgb_matcher.v410b_acceptor import (
    OrderPreservingSelectiveStandardScaler,
)


def test_selective_scaler_preserves_width_order_and_serializes(tmp_path: Path) -> None:
    matrix = np.array(
        [
            [10.0, 0.0, 100.0, 1.0],
            [20.0, 1.0, 300.0, 0.0],
            [30.0, 0.0, 500.0, 1.0],
        ]
    )
    scaler = OrderPreservingSelectiveStandardScaler([0, 2]).fit(matrix)
    transformed = scaler.transform(matrix)

    assert transformed.shape == matrix.shape
    np.testing.assert_array_equal(transformed[:, [1, 3]], matrix[:, [1, 3]])
    np.testing.assert_allclose(transformed[:, [0, 2]].mean(axis=0), 0.0)
    assert matrix[0, 0] == 10.0  # transform must not mutate its caller.

    artifact = tmp_path / "scaler.joblib"
    joblib.dump(scaler, artifact)
    restored = joblib.load(artifact)
    assert restored.__class__.__module__ == "src.xgb_matcher.v410b_acceptor"
    np.testing.assert_array_equal(restored.transform(matrix), transformed)


def test_selective_scaler_is_sklearn_cloneable() -> None:
    indices = [0, 2]
    scaler = OrderPreservingSelectiveStandardScaler(indices)

    cloned = clone(scaler)

    assert scaler.scaled_indices is indices
    assert cloned.scaled_indices == indices
    assert isinstance(cloned.scaled_indices, list)
    assert not hasattr(cloned, "scaled_indices_")
    cloned.fit(np.array([[1.0, 0.0, 3.0], [2.0, 1.0, 5.0]]))
    assert cloned.scaled_indices_ == (0, 2)


def test_training_frame_excludes_exact_held_fold_and_keeps_weights_aligned() -> None:
    historical = pd.DataFrame(
        {
            "query_id": ["z", "a", "m"],
            "role": ["historical_fit", "historical_hard_support", "historical_hard_support"],
            "hard_fold": [pd.NA, 2, 1],
            "acceptor_target": [1, 0, 1],
        }
    )
    hard = pd.DataFrame(
        {
            "query_id": ["y", "b"],
            "role": ["hard_oof", "hard_oof"],
            "hard_fold": [2, 1],
            "acceptor_target": [0, 1],
        }
    )

    frame, weights = subject._training_frame(
        historical, hard, hard_weight=4.0, held_out_fold=2
    )

    assert frame["query_id"].tolist() == ["b", "m", "z"]
    assert weights.tolist() == [4.0, 1.0, 1.0]
    assert "a" not in set(frame["query_id"])
    assert "y" not in set(frame["query_id"])


def test_threshold_selection_uses_integer_precision_and_stable_tie_break() -> None:
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    targets = np.array([1, 1, 1, 0])

    selected = subject.select_threshold(scores, targets, minimum_auto_count=2)

    assert selected is not None
    threshold, metrics, _ = selected
    assert threshold == 0.7
    assert metrics["auto_count"] == 3
    assert metrics["correct_auto"] == 3


def test_gate_uses_exact_integer_boundaries() -> None:
    plan = {
        "development_gate": {
            "minimum_wrong_hard_rejected": 24,
            "maximum_ambiguous_hard_auto": 0,
            "minimum_correct_hard_auto": 58,
        }
    }
    baseline = {"auto_count": 1184, "correct_auto": 1182}
    hard = {
        "wrong_hard_rejected": 24,
        "ambiguous_hard_auto": 0,
        "correct_hard_auto": 58,
    }

    passing = subject._variant_gate(
        {"auto_count": 1155, "correct_auto": 1154}, baseline, hard, plan
    )
    failing_coverage = subject._variant_gate(
        {"auto_count": 1154, "correct_auto": 1152}, baseline, hard, plan
    )

    assert passing["admissible"]
    assert not failing_coverage["admissible"]
    assert not failing_coverage["checks"]["historical_coverage"]


def test_execution_lock_rejects_extra_fields_and_accepts_exact_fixture(
    tmp_path: Path,
) -> None:
    runner_hash = subject.file_sha256(Path(subject.__file__).resolve())
    paths = {
        "dataset_manifest": tmp_path / "manifest.json",
        "feature_catalog.json": tmp_path / "feature_catalog.json",
    }
    for path in paths.values():
        path.write_text("{}\n", encoding="utf-8")
    hashes = {
        "plan": "plan-hash",
        "dataset_manifest": subject.file_sha256(paths["dataset_manifest"]),
        "feature_catalog.json": subject.file_sha256(paths["feature_catalog.json"]),
    }
    plan = {
        "runtime": {
            "execution_lock": {
                "schema_version": "sireto-v4.10b-execution-lock-1",
                "exact_fields": [
                    "schema_version",
                    "training_plan_sha256",
                    "runner_sha256",
                    "runner_commit",
                    "dataset_manifest_sha256",
                    "feature_catalog_sha256",
                ],
            }
        }
    }
    lock = {
        "schema_version": "sireto-v4.10b-execution-lock-1",
        "training_plan_sha256": "plan-hash",
        "runner_sha256": runner_hash,
        "runner_commit": "fixture-commit",
        "dataset_manifest_sha256": hashes["dataset_manifest"],
        "feature_catalog_sha256": hashes["feature_catalog.json"],
    }
    lock_path = tmp_path / "execution-lock.json"
    raw = (json.dumps(lock, sort_keys=True) + "\n").encode()
    lock_path.write_bytes(raw)

    observed, observed_hash = subject.validate_execution_lock(
        plan, paths, hashes, lock_path, verify_git_commit=False
    )
    assert observed == lock
    assert observed_hash == hashlib.sha256(raw).hexdigest()

    lock["unexpected"] = True
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(ValueError, match="fields differ"):
        subject.validate_execution_lock(
            plan, paths, hashes, lock_path, verify_git_commit=False
        )

    lock.pop("unexpected")
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    def committed_hash(_commit: str, relative_path: str) -> str:
        if relative_path == str(subject.SCALER_SOURCE):
            return "mutable-scaler-source"
        return runner_hash

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(subject, "_git_blob_sha256", committed_hash)
    try:
        with pytest.raises(ValueError, match="does not pin scaler source"):
            subject.validate_execution_lock(
                plan, paths, hashes, lock_path, verify_git_commit=True
            )
    finally:
        monkeypatch.undo()


def test_run_validates_external_lock_before_any_semantic_data_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        subject,
        "_load_plan",
        lambda *args, **kwargs: (
            {"runtime": {"required_versions": {}}},
            {},
            {},
        ),
    )
    monkeypatch.setattr(subject, "_validate_runtime", lambda plan: {})
    monkeypatch.setattr(
        subject,
        "validate_execution_lock",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("STOP_EXECUTION_LOCK: fixture")
        ),
    )
    semantic_read_attempted = False

    def forbidden_semantic_read(*args: object, **kwargs: object) -> None:
        nonlocal semantic_read_attempted
        semantic_read_attempted = True
        raise AssertionError("semantic data must not be read")

    monkeypatch.setattr(subject, "_load_frames", forbidden_semantic_read)

    with pytest.raises(ValueError, match="STOP_EXECUTION_LOCK"):
        subject.run_development(
            tmp_path / "plan.json",
            execution_lock_path=tmp_path / "lock.json",
            enforce_canonical=False,
        )
    assert not semantic_read_attempted


def test_fit_twice_compares_dev_hard_threshold_and_decisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FixedModel:
        def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
            score = np.asarray(matrix[:, 0], dtype=float)
            return np.column_stack([1.0 - score, score])

    calls: list[int] = []

    def fake_fit(*args: object, **kwargs: object) -> FixedModel:
        calls.append(1)
        return FixedModel()

    monkeypatch.setattr(subject, "_fit_model", fake_fit)
    plan = {
        "determinism": {"repeat_fit_score_absolute_tolerance": 1e-12},
        "threshold_selection": {"minimum_auto_count": 2},
    }
    train = pd.DataFrame({"f": [0.1, 0.9], "acceptor_target": [0, 1]})
    dev = pd.DataFrame(
        {"f": [0.9, 0.8, 0.2], "acceptor_target": [1, 1, 0]}
    )
    hard = pd.DataFrame({"f": [0.7, 0.1], "acceptor_target": [1, 0]})

    outcome = subject._fit_twice(
        "fixture",
        train,
        feature_order=["f"],
        continuous_features=["f"],
        plan=plan,
        base_weights=np.ones(2),
        dev=dev,
        held_hard=hard,
    )

    assert len(calls) == 2
    assert outcome[3] == 0.8
    assert outcome[-1]["passed"]
    assert outcome[-1]["threshold_identical"]
    assert outcome[-1]["held_hard_auto_decisions_identical"]


def test_fit_twice_still_repeats_when_no_safe_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FixedModel:
        def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
            score = np.asarray(matrix[:, 0], dtype=float)
            return np.column_stack([1.0 - score, score])

    calls: list[int] = []
    monkeypatch.setattr(
        subject,
        "_fit_model",
        lambda *args, **kwargs: calls.append(1) or FixedModel(),
    )
    plan = {
        "determinism": {"repeat_fit_score_absolute_tolerance": 1e-12},
        "threshold_selection": {"minimum_auto_count": 100},
    }
    train = pd.DataFrame({"f": [0.1, 0.9], "acceptor_target": [0, 1]})
    dev = pd.DataFrame({"f": [0.9, 0.2], "acceptor_target": [1, 0]})

    outcome = subject._fit_twice(
        "fixture",
        train,
        feature_order=["f"],
        continuous_features=["f"],
        plan=plan,
        base_weights=np.ones(2),
        dev=dev,
        held_hard=None,
    )

    assert len(calls) == 2
    assert outcome[3] is None
    assert outcome[-1]["passed"]


def test_stop_is_a_scientific_verdict_not_an_exception() -> None:
    plan = {"selection": {"success_verdict": "GO_FRESH_DEV_V410B"}}
    assert subject._development_verdict([], [], plan) == "STOP_STRUCTURED_ACCEPTOR"
    assert subject._development_verdict(["control"], [], plan) == "PIVOT_STRUCTURED_FEATURES"
    assert (
        subject._development_verdict(["candidate"], ["candidate"], plan)
        == "GO_FRESH_DEV_V410B"
    )


def test_baseline_has_an_explicit_non_promotable_variant_result() -> None:
    result = subject._baseline_variant_result(
        {"auto_count": 1188},
        {"auto_count": 1184},
        {"wrong_hard_rejected": 7},
    )

    assert result["role"] == "comparator"
    assert result["complete_safe_thresholds"] is False
    assert result["passes_development_gate"] is False
    assert result["promotion_eligible"] is False
    assert result["historical_metrics"]["auto_count"] == 1184
    assert result["hard_diagnostic"]["used_for_gate_or_selection"] is False


def test_variant_results_require_all_ten_ids_including_baseline() -> None:
    variants = [{"id": "BASE_FROZEN"}] + [
        {"id": f"TRAINED_{index}"} for index in range(9)
    ]
    results = {str(item["id"]): {} for item in variants}

    subject._assert_variant_results_complete(results, variants)
    results.pop("BASE_FROZEN")
    with pytest.raises(ValueError, match="variant results are incomplete"):
        subject._assert_variant_results_complete(results, variants)


def test_population_usage_names_every_excluded_population_explicitly() -> None:
    usage = subject._population_usage(
        {
            "dataset": {
                "expected_rows": {
                    "descriptive_locked": 123,
                }
            }
        },
        historical_fit_rows=20,
        effective_dev_rows=8,
        hard_rows=5,
        excluded_dev_rows=4,
    )

    assert usage["random_v48"] == {"read": 0, "scored": 0}
    assert usage["historical_excluded_dev"] == {
        "read": 4,
        "baseline_scored": 4,
        "trained_variant_scored": 0,
        "threshold_used": 0,
        "gate_used": 0,
    }
    assert usage["historical_random_excluded_other_44"]["materialized"] == 0
    assert "random_scored" not in usage


def test_model_diagnostics_map_values_to_true_input_order() -> None:
    logit = SimpleNamespace(
        named_steps={
            "model": SimpleNamespace(
                coef_=np.array([[2.0, -3.0]]),
                intercept_=np.array([0.25]),
            )
        }
    )
    xgb_model = SimpleNamespace(
        feature_importances_=np.array([0.75, 0.25]),
        importance_type="gain",
    )

    logit_diagnostic = subject._model_diagnostic(
        logit, "structured_logit", ["binary_first", "continuous_second"]
    )
    xgb_diagnostic = subject._model_diagnostic(
        xgb_model, "structured_xgb", ["feature_b", "feature_a"]
    )

    assert logit_diagnostic["by_feature"] == {
        "binary_first": 2.0,
        "continuous_second": -3.0,
    }
    assert xgb_diagnostic["by_feature"] == {
        "feature_b": 0.75,
        "feature_a": 0.25,
    }
