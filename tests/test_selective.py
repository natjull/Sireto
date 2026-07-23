import numpy as np
import pytest

from src.xgb_matcher.selective import (
    certified_precision_lower,
    risk_coverage_curve,
    select_threshold,
)


def test_risk_coverage_curve_and_threshold_selection():
    scores = [0.99, 0.95, 0.90, 0.20]
    correct = [1, 1, 0, 0]
    curve = risk_coverage_curve(scores, correct)

    assert curve.iloc[0]["precision"] == 1.0
    selected = select_threshold(
        scores,
        correct,
        target_precision=1.0,
        min_auto_count=2,
    )
    assert selected is not None
    assert selected.threshold == pytest.approx(0.95)
    assert selected.coverage == pytest.approx(0.5)


def test_certification_requires_about_2300_zero_error_decisions():
    assert certified_precision_lower(0, 2299, confidence_level=0.99) < 0.998
    assert certified_precision_lower(0, 2301, confidence_level=0.99) >= 0.998


def test_curve_rejects_misaligned_arrays():
    with pytest.raises(ValueError):
        risk_coverage_curve(np.array([0.1]), np.array([1, 0]))
