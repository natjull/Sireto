from __future__ import annotations

import pytest

from scripts.build_v412_neural_training_groups import _validate_group_stats


def test_group_stats_require_one_truth_and_three_training_folds() -> None:
    _validate_group_stats((160, 10, 16, 16, 1, 1, 3), 15)

    with pytest.raises(ValueError, match="exactly one"):
        _validate_group_stats((160, 10, 16, 16, 0, 1, 3), 15)
    with pytest.raises(ValueError, match="folds 2/3/4"):
        _validate_group_stats((160, 10, 16, 16, 1, 1, 2), 15)
