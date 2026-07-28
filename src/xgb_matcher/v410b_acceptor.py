"""Stable, serializable preprocessing primitives for the V4.10b acceptor."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class OrderPreservingSelectiveStandardScaler(BaseEstimator, TransformerMixin):
    """Scale selected columns in place without changing master feature order."""

    def __init__(self, scaled_indices: Sequence[int]) -> None:
        # sklearn estimators must preserve constructor parameters byte-for-byte
        # so clone() can reconstruct them without detecting a mutation.
        self.scaled_indices = scaled_indices

    def fit(
        self,
        matrix: np.ndarray,
        y: Any = None,
    ) -> "OrderPreservingSelectiveStandardScaler":
        values = np.asarray(matrix, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError("STOP_INPUT_INTEGRITY: scaler expects a 2D matrix")
        self.n_features_in_ = int(values.shape[1])
        self.scaled_indices_ = tuple(int(value) for value in self.scaled_indices)
        if (
            not self.scaled_indices_
            or len(set(self.scaled_indices_)) != len(self.scaled_indices_)
            or min(self.scaled_indices_) < 0
            or max(self.scaled_indices_) >= self.n_features_in_
        ):
            raise ValueError("STOP_INPUT_INTEGRITY: invalid scaled feature indices")
        self.scaler_ = StandardScaler().fit(values[:, self.scaled_indices_])
        return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        values = np.asarray(matrix, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise ValueError("STOP_INPUT_INTEGRITY: scaler feature width changed")
        output = values.copy()
        output[:, self.scaled_indices_] = self.scaler_.transform(
            values[:, self.scaled_indices_]
        )
        return output


__all__ = ["OrderPreservingSelectiveStandardScaler"]
