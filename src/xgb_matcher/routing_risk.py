from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from .calibration import load_calibrator


BASE_FEATURES: List[str] = [
    "name_jaro_max",
    "name_token_overlap_max",
    "name_semantic_max",
    "name_semantic_second",
    "name_semantic_gap",
    "addr_jaro",
    "addr_token_overlap",
    "street_number_diff",
    "address_density",
    "idf_name",
    "name_length_max",
    "numeric_token_match",
    "name_sim_max_etab",
    "name_sim_max_pm_dirigeant",
    "name_crm_contains_cand_max",
    "name_contains_crm_max",
    "postcode_match",
    "city_match",
    "street_name_jaro",
    "name_addr_consistency",
]


def default_feature_columns(base_features: Iterable[str] | None = None) -> List[str]:
    feats = list(base_features) if base_features is not None else BASE_FEATURES
    cols: List[str] = ["score_top1", "score_top2", "score_gap", "score_ratio"]
    for feat in feats:
        cols.append(f"top1_{feat}")
        cols.append(f"top2_{feat}")
        cols.append(f"delta_{feat}")
    return cols


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def build_feature_row(
    top1: Dict | pd.Series,
    top2: Dict | pd.Series | None,
    feature_names: Iterable[str],
    base_features: Iterable[str] | None = None,
) -> List[float]:
    feats = list(base_features) if base_features is not None else BASE_FEATURES
    values: Dict[str, float] = {}

    score_top1 = _safe_float(top1.get("score"))
    score_top2 = _safe_float(top2.get("score")) if top2 is not None else 0.0
    values["score_top1"] = score_top1
    values["score_top2"] = score_top2
    values["score_gap"] = score_top1 - score_top2
    # Use floor 0.001 to align with training dataset
    denom = score_top2 if score_top2 > 1e-6 else 0.001
    values["score_ratio"] = score_top1 / denom

    for feat in feats:
        v1 = _safe_float(top1.get(feat))
        v2 = _safe_float(top2.get(feat)) if top2 is not None else 0.0
        values[f"top1_{feat}"] = v1
        values[f"top2_{feat}"] = v2
        values[f"delta_{feat}"] = v1 - v2

    return [values.get(name, 0.0) for name in feature_names]


def load_risk_model(
    model_path: Path,
    meta_path: Path,
    calibrator_path: Path | None = None
) -> Tuple[object, object | None, List[str], float]:
    """Load risk model assets (model, calibrator, meta)."""
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    features = meta.get("features", default_feature_columns())
    threshold = meta.get("threshold", 0.835)
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    calibrator = None
    if calibrator_path and calibrator_path.exists():
        try:
            calibrator = load_calibrator(calibrator_path)
        except Exception:
            # Fallback for standard pickles if not wrapped
            with open(calibrator_path, "rb") as f:
                calibrator = pickle.load(f)
                
    return model, calibrator, features, threshold
