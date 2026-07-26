"""Training and inference primitives for the V9 selective query acceptor."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .contracts import MatchDecision, ReviewReason, V9MatchResult
from .selective import (
    SelectiveThreshold,
    certified_precision_lower,
    risk_coverage_curve,
    select_threshold,
)
from .v9_scene import V9_SCENE_FEATURE_NAMES


@dataclass
class V9AcceptorBundle:
    model: Any
    calibrator: Any
    threshold: float
    feature_order: list[str]
    model_bundle_id: str
    dataset_manifest_id: str
    model_family: str
    calibration_method: str = "isotonic"

    def confidence(self, scenes: pd.DataFrame) -> np.ndarray:
        matrix = scenes[self.feature_order].astype(float).to_numpy()
        raw = self.model.predict_proba(matrix)[:, 1]
        return np.asarray(self.calibrator.predict(raw), dtype=np.float64)

    def decide(self, scene: Mapping[str, Any]) -> V9MatchResult:
        frame = pd.DataFrame([{name: scene.get(name, 0.0) for name in self.feature_order}])
        confidence = float(self.confidence(frame)[0])
        siret = scene.get("predicted_siret")
        if siret and confidence >= self.threshold:
            return V9MatchResult(
                crm_id=str(scene["query_id"]),
                decision=MatchDecision.AUTO_MATCH,
                predicted_siret=str(siret),
                predicted_siren=str(scene.get("predicted_siren") or str(siret)[:9]),
                confidence=confidence,
                review_reason=None,
                model_bundle_id=self.model_bundle_id,
                dataset_manifest_id=self.dataset_manifest_id,
            )

        if not siret:
            reason = ReviewReason.NO_CANDIDATE
        elif float(scene.get("retrieval_disagreement") or 0.0) > 0:
            reason = ReviewReason.RETRIEVAL_DISAGREEMENT
        elif float(scene.get("same_siren_top2") or 0.0) > 0:
            reason = ReviewReason.AMBIGUOUS_SITE
        elif float(scene.get("siren_score_gap") or 0.0) < 0.05:
            reason = ReviewReason.AMBIGUOUS_SIREN
        else:
            reason = ReviewReason.LOW_CONFIDENCE
        return V9MatchResult(
            crm_id=str(scene["query_id"]),
            decision=MatchDecision.REVIEW,
            predicted_siret=str(siret) if siret else None,
            predicted_siren=(
                str(scene.get("predicted_siren") or str(siret)[:9]) if siret else None
            ),
            confidence=confidence,
            review_reason=reason,
            model_bundle_id=self.model_bundle_id,
            dataset_manifest_id=self.dataset_manifest_id,
        )

    def save(self, output_dir: Path, metadata: Mapping[str, Any]) -> None:
        output_dir.mkdir(parents=True, exist_ok=False)
        joblib.dump(self.model, output_dir / "acceptor_model.joblib")
        joblib.dump(self.calibrator, output_dir / "acceptor_calibrator.joblib")
        payload = {
            "schema_version": "v9-acceptor-2",
            "threshold": self.threshold,
            "feature_order": self.feature_order,
            "model_bundle_id": self.model_bundle_id,
            "dataset_manifest_id": self.dataset_manifest_id,
            "model_family": self.model_family,
            "calibration_method": self.calibration_method,
            **dict(metadata),
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, output_dir: Path) -> "V9AcceptorBundle":
        metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
        return cls(
            model=joblib.load(output_dir / "acceptor_model.joblib"),
            calibrator=joblib.load(output_dir / "acceptor_calibrator.joblib"),
            threshold=float(metadata["threshold"]),
            feature_order=list(metadata["feature_order"]),
            model_bundle_id=str(metadata["model_bundle_id"]),
            dataset_manifest_id=str(metadata["dataset_manifest_id"]),
            model_family=str(metadata["model_family"]),
            calibration_method=str(
                metadata.get("calibration_method", "isotonic")
            ),
        )


class IdentityCalibrator:
    """Keep model probabilities unchanged while exposing a calibrator API."""

    def fit(
        self,
        scores: Iterable[float],
        labels: Iterable[int],
    ) -> "IdentityCalibrator":
        del scores, labels
        return self

    def predict(self, scores: Iterable[float]) -> np.ndarray:
        return np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0)


class SigmoidCalibrator:
    """One-dimensional logistic (Platt-style) probability calibration."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.model = LogisticRegression(
            C=1.0,
            max_iter=3000,
            random_state=seed,
        )

    def fit(
        self,
        scores: Iterable[float],
        labels: Iterable[int],
    ) -> "SigmoidCalibrator":
        matrix = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
        self.model.fit(matrix, np.asarray(labels, dtype=np.int8))
        return self

    def predict(self, scores: Iterable[float]) -> np.ndarray:
        matrix = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
        return self.model.predict_proba(matrix)[:, 1]


def candidate_models(seed: int = 42) -> dict[str, Any]:
    from xgboost import XGBClassifier

    return {
        "logistic_scaled": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=3000,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "xgboost": XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_lambda=5.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        ),
    }


def fit_calibrator(
    method: str,
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
) -> Any:
    """Fit an explicitly named score transformation on calibration data."""
    if method == "raw":
        return IdentityCalibrator().fit(scores, labels)
    if method == "sigmoid":
        return SigmoidCalibrator(seed=seed).fit(scores, labels)
    if method == "isotonic":
        return IsotonicRegression(out_of_bounds="clip").fit(scores, labels)
    raise ValueError(f"Unsupported calibration method: {method}")


def _metrics(
    confidence: np.ndarray,
    correct: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    auto = confidence >= threshold
    auto_count = int(auto.sum())
    errors = int((1 - correct[auto]).sum()) if auto_count else 0
    precision = float(correct[auto].mean()) if auto_count else 0.0
    return {
        "count": int(len(correct)),
        "auto_count": auto_count,
        "coverage": auto_count / len(correct) if len(correct) else 0.0,
        "precision": precision,
        "error_count": errors,
        "precision_lower_99": (
            certified_precision_lower(errors, auto_count) if auto_count else 0.0
        ),
    }


def train_selective_acceptor(
    scenes: pd.DataFrame,
    *,
    dataset_manifest_id: str,
    target_precision: float = 0.998,
    min_auto_count: int = 1,
    seed: int = 42,
    models: Mapping[str, Any] | None = None,
    calibration_methods: Iterable[str] = ("isotonic",),
    minimum_gate_coverage: float = 0.25,
    evaluate_test: bool = False,
) -> tuple[V9AcceptorBundle, dict[str, Any]]:
    """Compare acceptors without consulting test for model or threshold choice."""
    required = set(V9_SCENE_FEATURE_NAMES) | {
        "split",
        "dev_role",
        "is_exact_siret_correct",
    }
    missing = required - set(scenes.columns)
    if missing:
        raise ValueError(f"Missing scene columns: {sorted(missing)}")

    train = scenes[scenes["split"].eq("train")]
    calibration = scenes[
        scenes["split"].eq("dev") & scenes["dev_role"].eq("calibration")
    ]
    threshold_set = scenes[
        scenes["split"].eq("dev") & scenes["dev_role"].eq("threshold")
    ]
    test = scenes[scenes["split"].eq("test")]
    if min(map(len, (train, calibration, threshold_set))) == 0:
        raise ValueError("train, dev/calibration and dev/threshold must be non-empty")
    if evaluate_test and test.empty:
        raise ValueError("The explicitly authorized final holdout is empty")

    feature_order = list(V9_SCENE_FEATURE_NAMES)
    X_train = train[feature_order].astype(float).to_numpy()
    y_train = train["is_exact_siret_correct"].astype(int).to_numpy()
    if len(np.unique(y_train)) < 2:
        raise ValueError("Acceptor training requires both correct and incorrect train scenes")

    methods = tuple(dict.fromkeys(str(method) for method in calibration_methods))
    if not methods:
        raise ValueError("At least one calibration method is required")
    unsupported = set(methods) - {"raw", "sigmoid", "isotonic"}
    if unsupported:
        raise ValueError(f"Unsupported calibration methods: {sorted(unsupported)}")
    if not 0.0 <= minimum_gate_coverage <= 1.0:
        raise ValueError("minimum_gate_coverage must be between 0 and 1")

    results: dict[str, Any] = {}
    fitted: dict[
        str,
        tuple[str, str, Any, Any, SelectiveThreshold],
    ] = {}
    for name, model in dict(models or candidate_models(seed)).items():
        model.fit(X_train, y_train)
        calibration_raw = model.predict_proba(
            calibration[feature_order].astype(float).to_numpy()
        )[:, 1]
        calibration_y = calibration["is_exact_siret_correct"].astype(int).to_numpy()
        threshold_raw = model.predict_proba(
            threshold_set[feature_order].astype(float).to_numpy()
        )[:, 1]
        threshold_y = threshold_set["is_exact_siret_correct"].astype(int).to_numpy()
        for method in methods:
            variant = name if len(methods) == 1 else f"{name}__{method}"
            calibrator = fit_calibrator(
                method,
                calibration_raw,
                calibration_y,
                seed=seed,
            )
            threshold_confidence = np.asarray(
                calibrator.predict(threshold_raw),
                dtype=np.float64,
            )
            operating_points = {}
            for precision_target in (0.99, 0.995, 0.998):
                point = select_threshold(
                    threshold_confidence,
                    threshold_y,
                    target_precision=precision_target,
                    min_auto_count=min_auto_count,
                )
                operating_points[f"{precision_target:.3f}"] = (
                    point.__dict__ if point is not None else None
                )
            selected = select_threshold(
                threshold_confidence,
                threshold_y,
                target_precision=target_precision,
                min_auto_count=min_auto_count,
            )
            if selected is None:
                results[variant] = {
                    "eligible": False,
                    "model_family": name,
                    "calibration_method": method,
                    "operating_points": operating_points,
                }
                continue
            results[variant] = {
                "eligible": True,
                "model_family": name,
                "calibration_method": method,
                "unique_threshold_scores": int(
                    np.unique(threshold_confidence).size
                ),
                "threshold_selection": selected.__dict__,
                "operating_points": operating_points,
                "threshold_risk_coverage_curve": risk_coverage_curve(
                    threshold_confidence,
                    threshold_y,
                ).to_dict("records"),
            }
            fitted[variant] = (
                name,
                method,
                model,
                calibrator,
                selected,
            )

    if not fitted:
        raise ValueError("No acceptor satisfies target precision on threshold split")
    winner = sorted(
        fitted,
        key=lambda name: (
            -fitted[name][4].coverage,
            -fitted[name][4].precision,
            name,
        ),
    )[0]
    model_family, calibration_method, model, calibrator, selected = fitted[winner]
    test_metrics = None
    test_curve: list[dict[str, Any]] = []
    if evaluate_test:
        test_confidence = np.asarray(
            calibrator.predict(
                model.predict_proba(test[feature_order].astype(float).to_numpy())[:, 1]
            )
        )
        test_y = test["is_exact_siret_correct"].astype(int).to_numpy()
        test_metrics = _metrics(test_confidence, test_y, selected.threshold)
        test_curve = risk_coverage_curve(test_confidence, test_y).to_dict("records")

    identity = {
        "dataset_manifest_id": dataset_manifest_id,
        "model_family": model_family,
        "calibration_method": calibration_method,
        "feature_order": feature_order,
        "threshold": selected.threshold,
        "seed": seed,
    }
    model_bundle_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    bundle = V9AcceptorBundle(
        model=model,
        calibrator=calibrator,
        threshold=selected.threshold,
        feature_order=feature_order,
        model_bundle_id=model_bundle_id,
        dataset_manifest_id=dataset_manifest_id,
        model_family=model_family,
        calibration_method=calibration_method,
    )
    report = {
        "schema_version": "v9-acceptor-report-2",
        "dataset_manifest_id": dataset_manifest_id,
        "target_precision": target_precision,
        "min_auto_count": min_auto_count,
        "minimum_gate_coverage": minimum_gate_coverage,
        "winner": winner,
        "winner_model_family": model_family,
        "winner_calibration_method": calibration_method,
        "verdict": (
            "PASS_E2B"
            if selected.coverage >= minimum_gate_coverage
            else "STOP_E2B"
        ),
        "models": results,
        "test": test_metrics,
        "test_risk_coverage_curve": test_curve,
        "test_used_for_selection": False,
        "final_holdout_evaluated": bool(evaluate_test),
    }
    return bundle, report


__all__ = [
    "IdentityCalibrator",
    "SigmoidCalibrator",
    "V9AcceptorBundle",
    "candidate_models",
    "fit_calibrator",
    "train_selective_acceptor",
]
