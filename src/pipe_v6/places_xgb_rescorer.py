"""XGBoost rescoring for Places-guided candidate pools.

Phase 4: Places-as-CRM
======================
This module loads the existing XGBoost decider and computes scores for
candidate pools. The decider uses the SAME model, features, calibrator,
and semantic gating as standard inference.

Key principle: Places is used for candidate generation AND for the
CRM input to the decider (Places-as-CRM). This reuses the exact same
feature pipeline as standard inference, but substitutes CRM with Places.

Model paths are EXPLICIT (no auto-latest detection) per Phase 4 requirements.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from xgb_matcher.features import (
    FEATURE_NAMES,
    preprocess_crm_row,
    make_features_from_preprocessed,
    build_semantic_name_pool,
    semantic_gate_allows,
)
from xgb_matcher.naming import build_candidate_names, normalize_text

from .config import PipelineConfig
from .places_candidate_generator import SireneCandidate
from .serper_places_client import PlacesResult

LOGGER = logging.getLogger(__name__)


# ============================================================================
# Phase 4: Explicit Model Paths (no auto-latest)
# ============================================================================

# Default model paths from handover (Phase 4)
DEFAULT_DECIDER_MODEL = "models/xgb_decider_20260103_132351.json"
DEFAULT_CALIBRATOR_PATH = "models/xgb_decider_calibrator_isotonic_20260103_132351.pkl"
DEFAULT_META_PATH = "models/xgb_two_stage_meta_20260103_132351.json"


def _semantic_enabled() -> bool:
    return os.getenv("XGB_SEMANTIC_ENABLED", "0") == "1"


# ============================================================================
# Calibrator Classes (for pickle compatibility)
# ============================================================================


class IsotonicCalibrator:
    """Isotonic regression calibrator wrapper."""

    def __init__(self, base_estimator, iso_reg):
        self.base_estimator = base_estimator
        self.iso_reg = iso_reg

    def predict_proba(self, X):
        proba = self.base_estimator.predict_proba(X)[:, 1]
        calibrated = self.iso_reg.predict(proba)
        calibrated = np.clip(calibrated, 0, 1)
        return np.column_stack([1 - calibrated, calibrated])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class SigmoidCalibrator:
    """Platt scaling calibrator wrapper."""

    def __init__(self, base_estimator, lr):
        self.base_estimator = base_estimator
        self.lr = lr

    def predict_proba(self, X):
        proba = self.base_estimator.predict_proba(X)[:, 1]
        calibrated = self.lr.predict_proba(proba.reshape(-1, 1))
        return calibrated

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ============================================================================
# Model Bundle
# ============================================================================


@dataclass
class XgbModelBundle:
    """Container for XGB model artifacts."""

    classifier: XGBClassifier
    calibrator: Any | None
    feature_order: List[str]
    meta: dict


def _load_places_config(config: PipelineConfig) -> dict:
    """Load Places thresholds config if available."""
    try:
        import yaml
    except ImportError:
        return {}

    config_path = Path("configs/places_thresholds.yaml")
    if not config_path.exists():
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        LOGGER.warning("Failed to load places_thresholds.yaml: %s", e)
        return {}


def _resolve_crm_mode(config: PipelineConfig, places_cfg: dict) -> str:
    """Resolve CRM mode for scoring: 'places' or 'original'."""
    mode = places_cfg.get("crm_mode") or getattr(config, "places_crm_mode", None) or "places"
    return str(mode).strip().lower()


def _load_model_bundle(config: PipelineConfig) -> XgbModelBundle:
    """Load XGB model bundle with explicit paths (Phase 4).

    Priority:
    1. Places config YAML (configs/places_thresholds.yaml)
    2. PipelineConfig attributes
    3. Default paths from handover
    """
    model_dir = Path(getattr(config, "xgb_model_dir", "models")).resolve()
    places_cfg = _load_places_config(config)
    models_cfg = places_cfg.get("models", {})

    # Resolve paths with priority
    decider_path = (
        models_cfg.get("decider_model")
        or getattr(config, "xgb_decider_model", None)
        or DEFAULT_DECIDER_MODEL
    )
    calibrator_path = (
        models_cfg.get("calibrator_path")
        or getattr(config, "xgb_calibrator_path", None)
        or DEFAULT_CALIBRATOR_PATH
    )
    meta_path = (
        models_cfg.get("meta_path")
        or getattr(config, "xgb_meta_path", None)
        or DEFAULT_META_PATH
    )

    # Convert to absolute paths
    decider_path = Path(decider_path)
    if not decider_path.is_absolute():
        decider_path = model_dir.parent / decider_path

    calibrator_path = Path(calibrator_path)
    if not calibrator_path.is_absolute():
        calibrator_path = model_dir.parent / calibrator_path

    meta_path = Path(meta_path)
    if not meta_path.is_absolute():
        meta_path = model_dir.parent / meta_path

    # Validate paths exist
    if not decider_path.exists():
        raise FileNotFoundError(f"XGB decider model not found: {decider_path}")

    # Load classifier
    classifier = XGBClassifier()
    classifier.load_model(str(decider_path))
    LOGGER.info("Loaded XGB decider: %s", decider_path)

    # Load calibrator (optional)
    calibrator = None
    if calibrator_path.exists():
        try:
            with open(calibrator_path, "rb") as f:
                calibrator = pickle.load(f)
            LOGGER.info("Loaded calibrator: %s", calibrator_path)
        except Exception as e:
            LOGGER.warning("Failed to load calibrator: %s", e)

    # Load metadata for feature order
    meta: dict = {}
    feature_order = FEATURE_NAMES
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            feature_order = meta.get("feature_order") or meta.get("feature_names") or FEATURE_NAMES
            LOGGER.info("Loaded meta: %s (features=%d)", meta_path, len(feature_order))
        except Exception as e:
            LOGGER.warning("Failed to load meta: %s (using default features)", e)

    # Validate semantic consistency
    meta_semantic = meta.get("semantic_enabled_samples")
    env_semantic = int(_semantic_enabled())
    if meta_semantic is not None and int(meta_semantic) != env_semantic:
        LOGGER.warning(
            "Semantic mismatch: meta=%s vs env=%s (train/serve skew risk)",
            meta_semantic,
            env_semantic,
        )

    return XgbModelBundle(
        classifier=classifier,
        calibrator=calibrator,
        feature_order=feature_order,
        meta=meta,
    )


# ============================================================================
# Candidate Mapping
# ============================================================================


def _candidate_to_xgb_dict(candidate: SireneCandidate) -> dict[str, Any]:
    """Map SireneCandidate to the feature schema expected by xgb_matcher."""
    return {
        "siret": candidate.siret,
        "siren": candidate.siren,
        "denomination": candidate.denomination,
        "denomination_ci": candidate.denomination_ci,
        "enseigne1": candidate.enseigne1,
        "enseigne2": candidate.enseigne2,
        "enseigne3": candidate.enseigne3,
        "denomination_ul": candidate.denomination_unite_legale,
        "denomination_usuelle_ul": candidate.denomination_unite_legale,
        "sigle_ul": None,
        "nom_ul": candidate.nom_unite_legale,
        "prenom_usuel_ul": candidate.prenom1_unite_legale,
        "cj_ul": candidate.legal_nature,
        "numeroVoie": candidate.street_number,
        "typeVoie": candidate.street_type,
        "libelleVoie": candidate.street_name,
        "complementAdresse": None,
        "postcode": candidate.postcode,
        "city": candidate.city,
        "is_siege": bool(candidate.etablissement_siege) if candidate.etablissement_siege is not None else False,
        "etat_admin": candidate.etat_administratif,
        # UL fields for Phase 4
        "pm_dirigeant": getattr(candidate, "pm_dirigeant", None),
    }


def _build_original_crm_dict(crm_row: Any) -> dict[str, Any]:
    """Build CRM dict from original CRM row data (for proper XGB scoring).

    Uses the original CRM name/address, NOT the Places-canonicalized version.
    This ensures XGB scores match the original inference.
    """
    # Build address from components or use raw address
    crm_address = getattr(crm_row, "crm_address", None) or ""
    if not crm_address:
        parts = [
            getattr(crm_row, "street_number", None),
            getattr(crm_row, "street_name", None),
        ]
        crm_address = " ".join(p for p in parts if p)

    crm_city = getattr(crm_row, "city", None) or ""
    postcode = getattr(crm_row, "postcode", None)
    insee = getattr(crm_row, "insee_code", None) or getattr(crm_row, "insee", None)

    return {
        "crm_id": getattr(crm_row, "crm_id", None),
        "crm_name": getattr(crm_row, "crm_name", "") or "",
        "crm_address": crm_address,
        "crm_city": crm_city,
        "postcode": postcode,
        "insee": insee,
    }


def _build_places_crm_dict(places: PlacesResult, crm_row: Any) -> dict[str, Any]:
    """Build CRM dict from Places result (Places-as-CRM).

    Uses Places title/address as the canonical CRM input and falls back
    to original CRM fields when Places fields are missing.
    """
    street_parts = [places.street_number, places.street_name]
    street_addr = " ".join(p for p in street_parts if p)
    places_addr = street_addr or places.address or ""

    crm_city = places.city or getattr(crm_row, "city", None) or getattr(crm_row, "crm_city", None)
    postcode = places.postcode or getattr(crm_row, "postcode", None)
    insee = getattr(crm_row, "insee_code", None) or getattr(crm_row, "insee", None)

    return {
        "crm_id": getattr(crm_row, "crm_id", None),
        "crm_name": places.title or getattr(crm_row, "crm_name", "") or "",
        "crm_address": places_addr or getattr(crm_row, "crm_address", "") or "",
        "crm_city": crm_city,
        "postcode": postcode,
        "insee": insee,
    }


# ============================================================================
# Main Rescorer Class
# ============================================================================


class PlacesXgbRescorer:
    """XGB scorer for Places-guided candidate pools.

    Phase 4 implementation:
    - Uses explicit model paths (no auto-latest)
    - Uses Places-as-CRM by default (configurable via places_thresholds.yaml)
    - Applies same semantic gating as training/inference
    - Uses isotonic calibration if available
    """

    def __init__(self, config: PipelineConfig):
        """Initialize with explicit model paths."""
        self._bundle = _load_model_bundle(config)
        self._semantic_cache: dict[str, Any] = {}
        self._places_cfg = _load_places_config(config)
        self._crm_mode = _resolve_crm_mode(config, self._places_cfg)

    @property
    def feature_order(self) -> List[str]:
        return self._bundle.feature_order

    def score_candidates(
        self,
        places: PlacesResult,
        crm_row: Any,
        candidates: Iterable[SireneCandidate],
    ) -> List[float]:
        """Return XGB probabilities for the candidate pool.

        Uses ORIGINAL CRM data for scoring (not Places pseudo-CRM).
        Places is only used for candidate generation, not feature computation.

        Args:
            places: Places result (for logging/observability only)
            crm_row: Original CRM row data
            candidates: Iterable of SireneCandidate objects

        Returns:
            List of calibrated probabilities for each candidate
        """
        # Use Places-as-CRM by default (fallback to original CRM if configured)
        if self._crm_mode == "original":
            crm_dict = _build_original_crm_dict(crm_row)
        else:
            crm_dict = _build_places_crm_dict(places, crm_row)
        crm_ctx = preprocess_crm_row(crm_dict)

        cand_list = list(candidates)
        if not cand_list:
            return []

        cand_dicts: List[dict] = []
        feats: List[dict] = []
        semantic_pools: List[List[str]] = []

        sem_enabled = _semantic_enabled()

        for cand in cand_list:
            cand_dict = _candidate_to_xgb_dict(cand)
            cand_dicts.append(cand_dict)
            feat = make_features_from_preprocessed(crm_ctx, cand_dict, skip_semantic=True)
            feats.append(feat)

            if sem_enabled:
                cand_city_norm = normalize_text(cand_dict.get("city"))
                semantic_pools.append(
                    build_semantic_name_pool(
                        build_candidate_names(cand_dict),
                        crm_city_norm=crm_ctx.get("crm_city_norm", ""),
                        cand_city_norm=cand_city_norm,
                    )
                )

        # Batch semantic computation with gating
        if sem_enabled and semantic_pools:
            from xgb_matcher.semantic import top2_semantic_similarities_batch

            sem = top2_semantic_similarities_batch(
                crm_ctx.get("crm_name_semantic", ""), semantic_pools
            )
            for feat, (sem_max, sem_second, sem_gap) in zip(feats, sem, strict=True):
                # Apply same gating as training/inference
                if semantic_gate_allows(
                    feat.get("name_jaro_max", 0.0),
                    feat.get("name_token_overlap_max", 0.0),
                ):
                    feat["name_semantic_max"] = sem_max
                    feat["name_semantic_second"] = sem_second
                    feat["name_semantic_gap"] = sem_gap
                else:
                    feat["name_semantic_max"] = 0.0
                    feat["name_semantic_second"] = 0.0
                    feat["name_semantic_gap"] = 0.0

        # Build feature matrix
        X = pd.DataFrame(feats)
        for col in self._bundle.feature_order:
            if col not in X.columns:
                X[col] = 0.0
        X = X[self._bundle.feature_order].fillna(0.0)
        X = X.astype(float)

        # Predict with calibration
        if self._bundle.calibrator is not None:
            probs = self._bundle.calibrator.predict_proba(X.values)[:, 1]
        else:
            probs = self._bundle.classifier.predict_proba(X.values)[:, 1]

        return [float(x) for x in probs]

    def score_candidates_with_features(
        self,
        places: PlacesResult,
        crm_row: Any,
        candidates: Iterable[SireneCandidate],
    ) -> List[tuple[float, dict]]:
        """Return (score, features) tuples for each candidate.

        Useful for debugging and observability.
        """
        if self._crm_mode == "original":
            crm_dict = _build_original_crm_dict(crm_row)
        else:
            crm_dict = _build_places_crm_dict(places, crm_row)
        crm_ctx = preprocess_crm_row(crm_dict)

        cand_list = list(candidates)
        if not cand_list:
            return []

        cand_dicts: List[dict] = []
        feats: List[dict] = []
        semantic_pools: List[List[str]] = []

        sem_enabled = _semantic_enabled()

        for cand in cand_list:
            cand_dict = _candidate_to_xgb_dict(cand)
            cand_dicts.append(cand_dict)
            feat = make_features_from_preprocessed(crm_ctx, cand_dict, skip_semantic=True)
            feats.append(feat)

            if sem_enabled:
                cand_city_norm = normalize_text(cand_dict.get("city"))
                semantic_pools.append(
                    build_semantic_name_pool(
                        build_candidate_names(cand_dict),
                        crm_city_norm=crm_ctx.get("crm_city_norm", ""),
                        cand_city_norm=cand_city_norm,
                    )
                )

        if sem_enabled and semantic_pools:
            from xgb_matcher.semantic import top2_semantic_similarities_batch

            sem = top2_semantic_similarities_batch(
                crm_ctx.get("crm_name_semantic", ""), semantic_pools
            )
            for feat, (sem_max, sem_second, sem_gap) in zip(feats, sem, strict=True):
                if semantic_gate_allows(
                    feat.get("name_jaro_max", 0.0),
                    feat.get("name_token_overlap_max", 0.0),
                ):
                    feat["name_semantic_max"] = sem_max
                    feat["name_semantic_second"] = sem_second
                    feat["name_semantic_gap"] = sem_gap
                else:
                    feat["name_semantic_max"] = 0.0
                    feat["name_semantic_second"] = 0.0
                    feat["name_semantic_gap"] = 0.0

        X = pd.DataFrame(feats)
        for col in self._bundle.feature_order:
            if col not in X.columns:
                X[col] = 0.0
        X = X[self._bundle.feature_order].fillna(0.0)
        X = X.astype(float)

        if self._bundle.calibrator is not None:
            probs = self._bundle.calibrator.predict_proba(X.values)[:, 1]
        else:
            probs = self._bundle.classifier.predict_proba(X.values)[:, 1]

        return [(float(p), f) for p, f in zip(probs, feats)]


__all__ = [
    "PlacesXgbRescorer",
    "XgbModelBundle",
    "IsotonicCalibrator",
    "SigmoidCalibrator",
    "DEFAULT_DECIDER_MODEL",
    "DEFAULT_CALIBRATOR_PATH",
    "DEFAULT_META_PATH",
]
