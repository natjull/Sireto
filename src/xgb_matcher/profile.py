"""Inference Profile: Single Source of Truth for inference configuration.

This module ensures train/serve alignment by loading configuration from
the same metadata files used during training.

Usage:
    from xgb_matcher.profile import InferenceProfile

    profile = InferenceProfile.from_meta("models/xgb_two_stage_meta_*.json")
    # profile now contains all knobs aligned with training
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class InferenceProfile:
    """Configuration profile for inference, aligned with training."""

    # Model paths
    ranker_path: Path | None = None
    ranker_fast_path: Path | None = None
    decider_path: Path | None = None
    calibrator_path: Path | None = None
    risk_model_path: Path | None = None
    risk_meta_path: Path | None = None

    # Feature configuration
    feature_order: List[str] = field(default_factory=list)
    semantic_features: List[str] = field(default_factory=list)

    # TF-IDF / candidate retrieval knobs
    tfidf_name_mode: str = "bag"  # "primary" or "bag"
    siren_siblings: bool = True  # Variant C: enrich with SIREN siblings
    prefilter_k: int = 500
    char_top_k: int = 200
    min_candidates: int = 150

    # Candidate filtering
    drop_unnamed: bool = True
    exclude_closed: bool = False

    # Partitions
    partitions_dir: Path = field(default_factory=lambda: Path("data/candidates_v4_active"))

    # Semantic
    semantic_required: bool = True
    semantic_enabled: bool = False  # Set from env at runtime

    # Stage 1 configuration
    use_ranker_fast: bool = True  # Use ranker_fast for Stage 1 (skip_semantic=True)
    stage1_top_n: int = 200

    # Risk model
    risk_threshold: float = 0.835

    # Metadata
    samples_file: str | None = None
    meta_path: Path | None = None

    def __post_init__(self):
        """Check semantic env and validate configuration."""
        self.semantic_enabled = os.getenv("XGB_SEMANTIC_ENABLED", "0") == "1"

    def validate(self, strict: bool = True) -> List[str]:
        """Validate profile configuration. Returns list of warnings/errors."""
        issues = []

        # Semantic check
        if self.semantic_required and not self.semantic_enabled:
            msg = (
                "SEMANTIC SKEW RISK: semantic_required=True but XGB_SEMANTIC_ENABLED != '1'. "
                "Set XGB_SEMANTIC_ENABLED=1 before imports to enable semantic features."
            )
            if strict:
                raise RuntimeError(msg)
            issues.append(msg)

        # Ranker selection check
        if self.use_ranker_fast and not self.ranker_fast_path:
            if self.ranker_path:
                issues.append(
                    "use_ranker_fast=True but ranker_fast_path not set. "
                    "Falling back to ranker_path (may cause skew if Stage 1 uses skip_semantic=True)."
                )

        # TF-IDF mode check
        if self.tfidf_name_mode not in ("primary", "bag"):
            issues.append(f"Invalid tfidf_name_mode: {self.tfidf_name_mode}. Must be 'primary' or 'bag'.")

        return issues

    def get_stage1_ranker_path(self) -> Path | None:
        """Get the appropriate ranker for Stage 1."""
        if self.use_ranker_fast and self.ranker_fast_path:
            return self.ranker_fast_path
        return self.ranker_path

    @classmethod
    def from_meta(
        cls,
        meta_path: Path | str,
        samples_meta_path: Path | str | None = None,
        strict: bool = True,
    ) -> "InferenceProfile":
        """Load profile from model metadata JSON.

        Args:
            meta_path: Path to xgb_two_stage_meta_*.json
            samples_meta_path: Optional path to samples_*.json for training knobs
            strict: If True, raise on semantic mismatch
        """
        meta_path = Path(meta_path)
        if not meta_path.exists():
            raise FileNotFoundError(f"Meta file not found: {meta_path}")

        with open(meta_path) as f:
            meta = json.load(f)

        # Extract paths
        model_dir = meta_path.parent
        ranker_path = _resolve_path(meta.get("ranker_model"), model_dir)
        ranker_fast_path = _resolve_path(meta.get("ranker_fast_model"), model_dir)
        decider_path = _resolve_path(meta.get("decider_model"), model_dir)
        calibrator_path = _resolve_path(meta.get("decider_calibrator"), model_dir)
        risk_model_path = _resolve_path(
            meta.get("risk_model_path") or meta.get("routing_risk_model") or meta.get("risk_model"),
            model_dir,
        )
        risk_meta_path = _resolve_path(
            meta.get("risk_meta_path") or meta.get("routing_risk_meta") or meta.get("risk_meta"),
            model_dir,
        )

        # Default risk assets if present
        default_risk_model = Path("models/routing_risk_model.pkl")
        default_risk_meta = Path("models/routing_risk_meta.json")
        if risk_model_path is None and default_risk_model.exists():
            risk_model_path = default_risk_model
        if risk_meta_path is None and default_risk_meta.exists():
            risk_meta_path = default_risk_meta

        # Feature order
        feature_order = meta.get("feature_order") or meta.get("feature_names") or []
        semantic_features = meta.get("semantic_features_zeroed_for_ranker_fast") or []

        # Load samples metadata if available
        samples_meta: Dict[str, Any] = {}
        samples_file = meta.get("samples_file")
        if samples_file and not samples_meta_path:
            # Try to find sidecar JSON
            samples_json = Path(samples_file).with_suffix(".json")
            if samples_json.exists():
                samples_meta_path = samples_json

        if samples_meta_path:
            samples_meta_path = Path(samples_meta_path)
            if samples_meta_path.exists():
                with open(samples_meta_path) as f:
                    samples_meta = json.load(f)

        # Extract training knobs from samples metadata
        tfidf_name_mode = samples_meta.get("tfidf_name_mode", cls().tfidf_name_mode)
        siren_siblings = samples_meta.get("siren_siblings", cls().siren_siblings)
        prefilter_k = samples_meta.get("prefilter_k", cls().prefilter_k)
        drop_unnamed = samples_meta.get("drop_unnamed_candidates", cls().drop_unnamed)
        exclude_closed = samples_meta.get("exclude_closed_candidates", cls().exclude_closed)
        partitions_dir = samples_meta.get("partitions_dir", str(cls().partitions_dir))

        # Semantic was enabled during training?
        semantic_enabled_train = meta.get("semantic_enabled_samples", 1) == 1

        profile = cls(
            ranker_path=ranker_path,
            ranker_fast_path=ranker_fast_path,
            decider_path=decider_path,
            calibrator_path=calibrator_path,
            risk_model_path=risk_model_path,
            risk_meta_path=risk_meta_path,
            feature_order=feature_order,
            semantic_features=semantic_features,
            tfidf_name_mode=tfidf_name_mode,
            siren_siblings=siren_siblings,
            prefilter_k=prefilter_k,
            drop_unnamed=drop_unnamed,
            exclude_closed=exclude_closed,
            partitions_dir=Path(partitions_dir),
            semantic_required=semantic_enabled_train,
            samples_file=samples_file,
            meta_path=meta_path,
        )

        profile.validate(strict=strict)
        return profile

    @classmethod
    def from_latest_meta(
        cls,
        model_dir: Path | str = Path("models"),
        strict: bool = True,
    ) -> "InferenceProfile":
        """Load profile from the latest xgb_two_stage_meta_*.json in model_dir."""
        model_dir = Path(model_dir)
        metas = sorted(model_dir.glob("xgb_two_stage_meta_*.json"), reverse=True)
        if not metas:
            raise FileNotFoundError(f"No xgb_two_stage_meta_*.json found in {model_dir}")
        return cls.from_meta(metas[0], strict=strict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize profile to dict (for logging/debugging)."""
        return {
            "ranker_path": str(self.ranker_path) if self.ranker_path else None,
            "ranker_fast_path": str(self.ranker_fast_path) if self.ranker_fast_path else None,
            "decider_path": str(self.decider_path) if self.decider_path else None,
            "feature_order": self.feature_order[:5] + ["..."] if len(self.feature_order) > 5 else self.feature_order,
            "tfidf_name_mode": self.tfidf_name_mode,
            "siren_siblings": self.siren_siblings,
            "prefilter_k": self.prefilter_k,
            "semantic_required": self.semantic_required,
            "semantic_enabled": self.semantic_enabled,
            "use_ranker_fast": self.use_ranker_fast,
            "partitions_dir": str(self.partitions_dir),
        }


def _resolve_path(path_str: str | None, base_dir: Path) -> Path | None:
    """Resolve a path string relative to base_dir if needed."""
    if not path_str:
        return None
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return p
    # Try relative to base_dir
    rel = base_dir / p.name
    if rel.exists():
        return rel
    # Try as-is
    if p.exists():
        return p
    return None


def ensure_semantic_enabled() -> None:
    """Ensure XGB_SEMANTIC_ENABLED=1 is set. Call at module top before imports."""
    current = os.getenv("XGB_SEMANTIC_ENABLED", "0")
    if current != "1":
        os.environ["XGB_SEMANTIC_ENABLED"] = "1"


__all__ = [
    "InferenceProfile",
    "ensure_semantic_enabled",
]
