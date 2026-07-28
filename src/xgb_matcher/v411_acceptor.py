"""Stable preprocessing and model factories for the V4.11 acceptor."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from .v411_scene import (
    V411_ACCEPTOR_FEATURE_NAMES,
    V411_MONOTONIC_CONSTRAINTS,
    V411_SCALED_FEATURE_NAMES,
)


COMPACT_LOGIT = "COMPACT_LOGIT"
MONOTONIC_XGB = "MONOTONIC_XGB"
V411_ACCEPTOR_FAMILIES = (COMPACT_LOGIT, MONOTONIC_XGB)


class OrderPreservingSelectiveStandardScaler(BaseEstimator, TransformerMixin):
    """Scale selected columns in place while preserving the frozen order."""

    def __init__(self, scaled_indices: Sequence[int]) -> None:
        # sklearn clone requires constructor parameters to remain untouched.
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
            raise ValueError("STOP_INPUT_INTEGRITY: invalid scaled indices")
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


def build_compact_logit(config: dict[str, Any]) -> Pipeline:
    """Build the one preregistered compact logistic estimator."""

    scaled = set(V411_SCALED_FEATURE_NAMES)
    indices = [
        index
        for index, name in enumerate(V411_ACCEPTOR_FEATURE_NAMES)
        if name in scaled
    ]
    return Pipeline(
        [
            (
                "preprocessing",
                OrderPreservingSelectiveStandardScaler(indices),
            ),
            (
                "model",
                LogisticRegression(
                    C=float(config["C"]),
                    solver=str(config["solver"]),
                    tol=float(config["tol"]),
                    class_weight=config.get("class_weight"),
                    max_iter=int(config["max_iter"]),
                    random_state=int(config["random_state"]),
                ),
            ),
        ]
    )


def build_monotonic_xgb(config: dict[str, Any]) -> xgb.XGBClassifier:
    """Build the one preregistered monotonic XGBoost estimator."""

    values = dict(config)
    values["monotone_constraints"] = tuple(V411_MONOTONIC_CONSTRAINTS)
    return xgb.XGBClassifier(**values)


def build_v411_acceptor(family: str, config: dict[str, Any]) -> Any:
    """Return exactly one of the two frozen V4.11 model families."""

    if family == COMPACT_LOGIT:
        return build_compact_logit(config)
    if family == MONOTONIC_XGB:
        return build_monotonic_xgb(config)
    raise ValueError(f"Unsupported V4.11 acceptor family: {family}")


__all__ = [
    "COMPACT_LOGIT",
    "MONOTONIC_XGB",
    "OrderPreservingSelectiveStandardScaler",
    "V411_ACCEPTOR_FAMILIES",
    "build_compact_logit",
    "build_monotonic_xgb",
    "build_v411_acceptor",
]
