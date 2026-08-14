from __future__ import annotations

import argparse

import pytest

from scripts.train_v412_bge_groupwise import _parse_folds, _validate_fold_roles


def test_bge_fold_roles_are_cross_fitted_and_confirmation_closed() -> None:
    assert _parse_folds("4,2,3") == (2, 3, 4)
    _validate_fold_roles((2, 3, 4), 0)
    _validate_fold_roles((3, 4), 2)

    with pytest.raises(ValueError, match="cannot be used"):
        _validate_fold_roles((2, 3, 4), 3)
    with pytest.raises(ValueError, match="closed"):
        _validate_fold_roles((2, 3, 4), 1)
    with pytest.raises(argparse.ArgumentTypeError, match="subset"):
        _parse_folds("0,2")
