from __future__ import annotations

import numpy as np

from scripts.train_v412_learned_acceptor import _select_threshold


def test_threshold_maximises_safe_calibration_coverage() -> None:
    scores = np.asarray([0.99, 0.98, 0.97, 0.96], dtype=float)
    correct = np.asarray([True, True, False, True])
    audited_open = np.asarray([False, False, True, False])

    threshold, metric = _select_threshold(
        scores,
        correct,
        audited_open,
        target_precision=0.998,
    )

    assert threshold == 0.98
    assert metric["accepted"] == 2
    assert metric["precision"] == 1.0
    assert metric["audited_open_auto"] == 0


def test_threshold_abstains_when_highest_score_is_wrong() -> None:
    threshold, metric = _select_threshold(
        np.asarray([0.9, 0.8]),
        np.asarray([False, True]),
        np.asarray([False, False]),
        target_precision=0.998,
    )

    assert np.isinf(threshold)
    assert metric["accepted"] == 0
