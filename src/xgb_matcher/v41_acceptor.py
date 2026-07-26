"""V4.1 query-level acceptor: scaled logistic regression, raw score only."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .selective import select_threshold


V41_CONFIDENCE_KIND = "ROUTING_SCORE_UNCALIBRATED"


@dataclass
class V41RawLogisticAcceptor:
    """A raw logistic routing score with a threshold selected on dev only."""

    model: Any
    threshold: float
    feature_order: list[str]
    model_bundle_id: str
    dataset_manifest_id: str
    retrieval_signature: str
    confidence_kind: str = V41_CONFIDENCE_KIND
    model_family: str = "logistic_scaled"
    calibration_method: str = "raw"

    def __post_init__(self) -> None:
        if self.model_family != "logistic_scaled":
            raise ValueError("V4.1 only supports the scaled logistic acceptor")
        if self.calibration_method != "raw":
            raise ValueError("V4.1 forbids score calibration, including isotonic")
        if self.confidence_kind != V41_CONFIDENCE_KIND:
            raise ValueError("Unexpected V4.1 confidence kind")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("Acceptor threshold must be between 0 and 1")

    def confidence(self, scenes: pd.DataFrame) -> np.ndarray:
        missing = set(self.feature_order) - set(scenes.columns)
        if missing:
            raise ValueError(f"Missing acceptor features: {sorted(missing)}")
        matrix = scenes[self.feature_order].astype(float).to_numpy()
        return np.asarray(self.model.predict_proba(matrix)[:, 1], dtype=np.float64)

    def score(self, scene: Mapping[str, Any]) -> float:
        frame = pd.DataFrame(
            [{name: float(scene.get(name, 0.0) or 0.0) for name in self.feature_order}]
        )
        return float(self.confidence(frame)[0])

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=False)
        joblib.dump(self.model, output_dir / "acceptor_model.joblib")
        metadata = {
            "schema_version": "v4.1-acceptor-1",
            "threshold": self.threshold,
            "feature_order": self.feature_order,
            "model_bundle_id": self.model_bundle_id,
            "dataset_manifest_id": self.dataset_manifest_id,
            "retrieval_signature": self.retrieval_signature,
            "confidence_kind": self.confidence_kind,
            "model_family": self.model_family,
            "calibration_method": self.calibration_method,
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, output_dir: Path) -> "V41RawLogisticAcceptor":
        metadata = json.loads((output_dir / "metadata.json").read_text("utf-8"))
        if metadata.get("schema_version") != "v4.1-acceptor-1":
            raise ValueError("Unsupported V4.1 acceptor schema")
        return cls(
            model=joblib.load(output_dir / "acceptor_model.joblib"),
            threshold=float(metadata["threshold"]),
            feature_order=list(metadata["feature_order"]),
            model_bundle_id=str(metadata["model_bundle_id"]),
            dataset_manifest_id=str(metadata["dataset_manifest_id"]),
            retrieval_signature=str(metadata["retrieval_signature"]),
            confidence_kind=str(metadata["confidence_kind"]),
            model_family=str(metadata["model_family"]),
            calibration_method=str(metadata["calibration_method"]),
        )


def fit_v41_raw_logistic_acceptor(
    scenes: pd.DataFrame,
    *,
    feature_order: Sequence[str],
    dataset_manifest_id: str,
    retrieval_signature: str,
    target_precision: float = 0.998,
    min_auto_count: int = 100,
    seed: int = 42,
) -> tuple[V41RawLogisticAcceptor, dict[str, Any]]:
    """Fit V4.1 without calibration and without consulting a final test.

    Every ranker prediction used here must be out-of-sample.  For fit rows that
    means OOF prediction; for dev rows it means prediction by a ranker trained
    without the dev component.
    """
    required = {
        "split",
        "is_exact_siret_correct",
        "ranker_prediction_is_out_of_sample",
        *feature_order,
    }
    missing = required - set(scenes.columns)
    if missing:
        raise ValueError(f"Missing acceptor columns: {sorted(missing)}")
    if not scenes["split"].isin({"fit", "dev"}).all():
        raise ValueError("V4.1 acceptor fitting accepts fit/dev only; test is forbidden")
    if not scenes["ranker_prediction_is_out_of_sample"].astype(bool).all():
        raise ValueError("Acceptor scenes must use out-of-sample ranker predictions")

    fit = scenes[scenes["split"].eq("fit")]
    dev = scenes[scenes["split"].eq("dev")]
    if fit.empty or dev.empty:
        raise ValueError("Both fit and dev scenes are required")
    y_fit = fit["is_exact_siret_correct"].astype(int).to_numpy()
    if np.unique(y_fit).size < 2:
        raise ValueError("Fit scenes require both correct and incorrect predictions")

    model = Pipeline(
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
    )
    ordered_features = list(feature_order)
    model.fit(fit[ordered_features].astype(float).to_numpy(), y_fit)
    dev_scores = np.asarray(
        model.predict_proba(dev[ordered_features].astype(float).to_numpy())[:, 1]
    )
    dev_y = dev["is_exact_siret_correct"].astype(int).to_numpy()
    selected = select_threshold(
        dev_scores,
        dev_y,
        target_precision=target_precision,
        min_auto_count=min_auto_count,
    )
    if selected is None:
        raise ValueError("No raw-logistic threshold satisfies the V4.1 dev target")

    identity = {
        "dataset_manifest_id": dataset_manifest_id,
        "retrieval_signature": retrieval_signature,
        "feature_order": ordered_features,
        "threshold": selected.threshold,
        "seed": seed,
        "model_family": "logistic_scaled",
        "calibration_method": "raw",
    }
    bundle_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    bundle = V41RawLogisticAcceptor(
        model=model,
        threshold=selected.threshold,
        feature_order=ordered_features,
        model_bundle_id=bundle_id,
        dataset_manifest_id=dataset_manifest_id,
        retrieval_signature=retrieval_signature,
    )
    report = {
        "schema_version": "v4.1-acceptor-report-1",
        "dataset_manifest_id": dataset_manifest_id,
        "retrieval_signature": retrieval_signature,
        "feature_order": ordered_features,
        "model_family": "logistic_scaled",
        "calibration_method": "raw",
        "confidence_kind": V41_CONFIDENCE_KIND,
        "target_precision": target_precision,
        "min_auto_count": min_auto_count,
        "dev_selection": selected.__dict__,
        "final_test_evaluated": False,
    }
    return bundle, report


__all__ = [
    "V41_CONFIDENCE_KIND",
    "V41RawLogisticAcceptor",
    "fit_v41_raw_logistic_acceptor",
]
