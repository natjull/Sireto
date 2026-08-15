from __future__ import annotations

import pandas as pd

from scripts.prepare_synthetic_augmented_model_mix import (
    _eligible_synthetic,
    _select_synthetic,
    _validate_plan,
)


def test_plan_freezes_train_dev_ratio_and_all_prohibitions() -> None:
    plan = {
        "seed": 42,
        "fold_roles": {
            "train": [2, 3, 4],
            "development": 0,
            "confirmation_closed": 1,
        },
        "sampling": {"real_to_synthetic_scene_ratio": [2, 1]},
        "weights": {
            "synthetic_scene_formula": (
                "0.5 / variants_for_target_siret_in_complete_eligible_corpus"
            )
        },
        "prohibitions": {
            "risk": True,
            "calibration": True,
            "auto": True,
            "dev": True,
            "test": True,
        },
    }
    assert _validate_plan(plan) == 42


def test_synthetic_selection_is_deterministic_and_proportional() -> None:
    labels = pd.DataFrame(
        {
            "query_id": [f"q{i}" for i in range(12)],
            "ground_truth_siren": [f"{i:09d}" for i in range(12)],
            "difficulty": ["EASY"] * 6 + ["HARD"] * 6,
            "augmentation_stratum": ["CONTROL"] * 3 + ["FAIL"] * 3
            + ["CONTROL"] * 3 + ["FAIL"] * 3,
        }
    )
    first = _select_synthetic(labels, 6, 42)
    second = _select_synthetic(labels.sample(frac=1, random_state=3), 6, 42)

    assert first["query_id"].tolist() == second["query_id"].tolist()
    assert first.groupby(["difficulty", "augmentation_stratum"]).size().to_dict() == {
        ("EASY", "CONTROL"): 2,
        ("EASY", "FAIL"): 2,
        ("HARD", "CONTROL"): 1,
        ("HARD", "FAIL"): 1,
    }


def test_eligible_synthetic_requires_natural_positive_in_both_views() -> None:
    labels = pd.DataFrame(
        {
            "query_id": ["q1", "q2", "q3"],
            "ground_truth_siret": ["1", "2", "3"],
        }
    )
    candidates = pd.DataFrame(
        {
            "query_id": ["q1", "q1", "q2", "q3"],
            "candidate_siret": ["1", "9", "8", "3"],
        }
    )
    groups = pd.DataFrame(
        {
            "query_id": ["q1", "q1", "q2", "q3"],
            "is_positive": [1, 0, 0, 0],
        }
    )

    eligible, diagnostics = _eligible_synthetic(labels, candidates, groups)

    assert eligible["query_id"].tolist() == ["q1"]
    assert diagnostics["eligible_non_injected_scenes"] == 1
