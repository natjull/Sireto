"""XGBoost rescoring for Places-guided candidate pools.

This module loads the existing XGBoost classifier and computes scores for
candidate pools using Places-normalized CRM data (pseudo-CRM).
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
)
from xgb_matcher.naming import build_candidate_names, normalize_text
from xgb_matcher.semantic import top2_semantic_similarities_batch

from .config import PipelineConfig
from .places_candidate_generator import SireneCandidate
from .serper_places_client import PlacesResult

LOGGER = logging.getLogger(__name__)


def _semantic_enabled() -> bool:
    return os.getenv("XGB_SEMANTIC_ENABLED", "0") == "1"


@dataclass
class XgbModelBundle:
    classifier: XGBClassifier
    scaler: Any | None
    feature_order: List[str]


def _find_latest_models(model_dir: Path) -> dict[str, Path]:
    """Find the most recent model set in `model_dir`."""
    pattern = list(model_dir.glob("xgb_matcher_features_[0-9]*_[0-9]*.json"))
    if pattern:
        latest_meta = sorted(pattern, reverse=True)[0]
        timestamp = latest_meta.stem.replace("xgb_matcher_features_", "")
        return {
            "classifier": model_dir / f"xgbclassifier_{timestamp}.json",
            "scaler": model_dir / f"xgb_matcher_scaler_{timestamp}.pkl",
            "meta": latest_meta,
            "timestamp": timestamp,
        }
    return {
        "classifier": model_dir / "xgbclassifier.json",
        "scaler": model_dir / "xgb_matcher_scaler.pkl",
        "meta": model_dir / "xgb_matcher_features.json",
        "timestamp": "default",
    }


def _load_model_bundle(config: PipelineConfig) -> XgbModelBundle:
    model_dir = Path(getattr(config, "xgb_model_dir", "models")).resolve()
    paths = _find_latest_models(model_dir)

    if not paths["classifier"].exists():
        raise FileNotFoundError(f"XGB classifier not found: {paths['classifier']}")
    if not paths["meta"].exists():
        raise FileNotFoundError(f"XGB feature metadata not found: {paths['meta']}")

    classifier = XGBClassifier()
    classifier.load_model(str(paths["classifier"]))

    scaler = None
    scaler_path = paths.get("scaler")
    if scaler_path and scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

    with open(paths["meta"], "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_order = meta.get("feature_order") or meta.get("feature_names") or FEATURE_NAMES

    LOGGER.info(
        "Loaded XGB classifier (%s) | features=%d | scaler=%s",
        paths.get("timestamp", "default"),
        len(feature_order),
        "on" if scaler is not None else "off",
    )

    return XgbModelBundle(classifier=classifier, scaler=scaler, feature_order=feature_order)


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
    }


def _build_places_crm_dict(places: PlacesResult, crm_row: Any) -> dict[str, Any]:
    """Build pseudo-CRM dict from Places result (canonicalized).
    
    DEPRECATED: Use _build_original_crm_dict instead for proper scoring.
    """
    street_parts = [places.street_number, places.street_name]
    street_addr = " ".join(p for p in street_parts if p)
    crm_address = street_addr or places.address or getattr(crm_row, "crm_address", "") or ""

    crm_city = places.city or getattr(crm_row, "city", None) or getattr(crm_row, "crm_city", None)
    postcode = places.postcode or getattr(crm_row, "postcode", None)
    insee = getattr(crm_row, "insee_code", None) or getattr(crm_row, "insee", None)

    return {
        "crm_id": getattr(crm_row, "crm_id", None),
        "crm_name": places.title or getattr(crm_row, "crm_name", ""),
        "crm_address": crm_address,
        "crm_city": crm_city,
        "postcode": postcode,
        "insee": insee,
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


class PlacesXgbRescorer:
    """XGB scorer for Places-guided candidate pools."""

    def __init__(self, config: PipelineConfig):
        self._bundle = _load_model_bundle(config)

    def score_candidates(
        self,
        places: PlacesResult,
        crm_row: Any,
        candidates: Iterable[SireneCandidate],
    ) -> List[float]:
        """Return XGB probabilities for the candidate pool.
        
        Uses ORIGINAL CRM data for scoring (not Places pseudo-CRM).
        Places is only used for candidate generation, not feature computation.
        """
        # Use original CRM data for proper XGB scoring
        crm_dict = _build_original_crm_dict(crm_row)
        crm_ctx = preprocess_crm_row(crm_dict)

        cand_dicts: List[dict] = []
        feats: List[dict] = []
        semantic_pools: List[List[str]] = []

        sem_enabled = _semantic_enabled()

        for cand in candidates:
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
            sem = top2_semantic_similarities_batch(
                crm_ctx.get("crm_name_semantic", ""), semantic_pools
            )
            for feat, (sem_max, sem_second, sem_gap) in zip(feats, sem, strict=True):
                feat["name_semantic_max"] = sem_max
                feat["name_semantic_second"] = sem_second
                feat["name_semantic_gap"] = sem_gap

        X = pd.DataFrame(feats)
        for col in self._bundle.feature_order:
            if col not in X.columns:
                X[col] = 0.0
        X = X[self._bundle.feature_order].fillna(0.0)
        X = X.astype(float)

        Xs = self._bundle.scaler.transform(X) if self._bundle.scaler is not None else X.values
        probs = self._bundle.classifier.predict_proba(Xs)[:, 1]
        return [float(x) for x in probs]


__all__ = ["PlacesXgbRescorer"]
