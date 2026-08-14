from __future__ import annotations

import pytest

from scripts.build_v412_bge_training_groups import _validate_stats


def test_bge_group_stats_require_one_truth_and_three_training_folds() -> None:
    result = _validate_stats((160, 10, 16, 16, 1, 1, 3), 15)
    assert result["scenes"] == 10

    with pytest.raises(ValueError, match="exactly one positive"):
        _validate_stats((160, 10, 16, 16, 0, 1, 3), 15)
    with pytest.raises(ValueError, match="folds 2/3/4"):
        _validate_stats((160, 10, 16, 16, 1, 1, 2), 15)
    with pytest.raises(ValueError, match="group size"):
        _validate_stats((170, 10, 17, 17, 1, 1, 3), 15)
