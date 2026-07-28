from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_v48_acceptor_development import (
    _training_frame,
    _variant_gate,
    decision_metrics,
    select_threshold,
)


def test_select_threshold_uses_exact_precision_and_maximum_coverage() -> None:
    scores = np.array([0.9] * 499 + [0.8] + [0.1, 0.1])
    targets = np.array([1] * 500 + [0, 0])
    ambiguous = np.zeros(len(scores), dtype=bool)
    selected = select_threshold(
        scores,
        targets,
        ambiguous=ambiguous,
        max_ambiguous_auto=0,
        min_auto_count=100,
    )
    assert selected is not None
    threshold, metrics = selected
    assert threshold == 0.8
    assert metrics["auto_count"] == 500
    assert metrics["correct_auto"] == 500


def test_select_threshold_respects_ambiguous_cap() -> None:
    scores = np.array([0.9, 0.8, 0.7])
    targets = np.array([1, 1, 0])
    ambiguous = np.array([False, False, True])
    selected = select_threshold(
        scores,
        targets,
        ambiguous=ambiguous,
        max_ambiguous_auto=0,
        min_auto_count=1,
    )
    assert selected is not None
    threshold, metrics = selected
    assert threshold == 0.8
    assert metrics["ambiguous_auto"] == 0


def test_training_frame_excludes_held_out_component_support() -> None:
    historical = pd.DataFrame(
        [
            {"query_id": "h0", "hard_fold": 0, "acceptor_target": 1},
            {"query_id": "h1", "hard_fold": 1, "acceptor_target": 1},
            {"query_id": "h-free", "hard_fold": pd.NA, "acceptor_target": 0},
        ]
    )
    hard = pd.DataFrame(
        [
            {"query_id": "c0", "hard_fold": 0, "acceptor_target": 0},
            {"query_id": "c1", "hard_fold": 1, "acceptor_target": 1},
        ]
    )
    frame, weights = _training_frame(
        historical,
        hard,
        hard_weight=4,
        held_out_fold=0,
    )
    assert set(frame["query_id"]) == {"h1", "h-free", "c1"}
    assert weights.tolist() == [1.0, 1.0, 4.0]


def test_variant_gate_requires_four_additional_wrong_rejections() -> None:
    frozen = {"auto_count": 1000, "correct_auto": 998, "coverage": 0.8}
    variant = {"auto_count": 1000, "correct_auto": 998, "coverage": 0.8}
    base_hard = {
        "wrong_rejected": 10,
        "correct_acceptance_rate": 0.8,
        "ambiguous_auto": 0,
    }
    candidate = {
        "wrong_rejected": 14,
        "correct_acceptance_rate": 0.76,
        "ambiguous_auto": 0,
    }
    gate = _variant_gate(
        frozen=frozen,
        variant=variant,
        hard_metrics=candidate,
        base_hard_metrics=base_hard,
    )
    assert gate["admissible"]
    candidate["wrong_rejected"] = 13
    gate = _variant_gate(
        frozen=frozen,
        variant=variant,
        hard_metrics=candidate,
        base_hard_metrics=base_hard,
    )
    assert not gate["admissible"]


def test_decision_rule_includes_equal_score() -> None:
    metrics = decision_metrics(
        np.array([0.5, 0.49]),
        np.array([1, 0]),
        0.5,
    )
    assert metrics["auto_count"] == 1
    assert metrics["correct_auto"] == 1
