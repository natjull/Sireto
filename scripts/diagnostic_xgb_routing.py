#!/usr/bin/env python3
"""
Reproducible diagnostic for XGBoost routing (risk-coverage + calibration).

Inputs:
  - CSV or parquet with ground truth SIRET (default: data/samples_v4_with_ranker.parquet)
  - XGBoost models (ranker + classifier)

NOTE: Parquet mode requires CSV with CRM fields (crm_name, crm_address, etc.) for
feature recomputation. Use --input-path with a CSV file for full functionality.

Outputs:
  - reports/diagnostic_analysis.json
  - reports/diagnostic_report.md
  - reports/diagnostic_plots.png

Notes:
  - Coverage is computed as auto_count / total_count (total_count = rows evaluated).
  - Also computes coverage_eligible = auto_count / eligible_count
    (eligible: ground truth present in candidate pool).
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import sys

# Ensure repo root is on sys.path for src/ and scripts/ imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import xgboost as xgb

from src.xgb_matcher.candidates import (
    load_candidates_for_locations,
    get_candidates_for_query,
    build_location_index,
)
from src.xgb_matcher.features import (
    FEATURE_NAMES,
    preprocess_crm_row,
    make_features_from_preprocessed,
    build_semantic_name_pool,
    normalize_text,
)
from src.xgb_matcher.naming import build_candidate_names, primary_name
from src.xgb_matcher.semantic import top2_semantic_similarities_batch

from scripts.old.infer_xgb_matcher_topk import apply_rerank_rules, find_latest_models


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _semantic_enabled() -> bool:
    return os.getenv("XGB_SEMANTIC_ENABLED", "0") == "1"


def _load_models(model_dir: Path | None = None) -> Tuple[xgb.Booster, xgb.XGBClassifier, List[str], Any]:
    paths = find_latest_models()
    if model_dir is not None:
        # Override model dir if provided
        pattern = list(Path(model_dir).glob("xgb_matcher_features_[0-9]*_[0-9]*.json"))
        if pattern:
            latest_meta = sorted(pattern, reverse=True)[0]
            timestamp = latest_meta.stem.replace("xgb_matcher_features_", "")
            paths = {
                "ranker": Path(model_dir) / f"xgbranker_{timestamp}.json",
                "classifier": Path(model_dir) / f"xgbclassifier_{timestamp}.json",
                "scaler": Path(model_dir) / f"xgb_matcher_scaler_{timestamp}.pkl",
                "meta": latest_meta,
                "timestamp": timestamp,
            }

    ranker = xgb.Booster()
    ranker.load_model(str(paths["ranker"]))

    classifier = xgb.XGBClassifier()
    classifier.load_model(str(paths["classifier"]))

    scaler = None
    scaler_path = paths.get("scaler")
    if scaler_path and Path(scaler_path).exists():
        import pickle

        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

    with open(paths["meta"], "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_order = meta.get("feature_order") or meta.get("feature_names") or FEATURE_NAMES
    return ranker, classifier, feature_order, scaler


def _build_crm_dict(row: pd.Series) -> Dict[str, Any]:
    crm_address = row.get("crm_address") or row.get("Adresse") or row.get("address") or ""
    crm_name = row.get("crm_name") or row.get("Client final") or row.get("name") or ""
    crm_city = row.get("crm_city") or row.get("Commune") or row.get("city") or ""
    postcode = row.get("postcode") or row.get("Code Postal") or row.get("cp") or ""
    insee = row.get("insee") or row.get("Code INSEE") or ""
    return {
        "crm_name": str(crm_name),
        "crm_address": str(crm_address),
        "crm_city": str(crm_city),
        "postcode": str(postcode) if not pd.isna(postcode) else "",
        "insee": str(insee) if not pd.isna(insee) else "",
    }


def _pool_candidates(
    row: pd.Series,
    all_candidates: Dict[str, dict],
    index: Dict[str, Dict[str, List[Tuple[str, dict]]]],
    pool_mode: str,
) -> List[Tuple[str, dict]]:
    postcode = row.get("postcode") or row.get("Code Postal") or ""
    insee = row.get("insee") or row.get("Code INSEE") or ""

    if pool_mode == "insee_then_postcode":
        if insee and insee in index["insee"]:
            return index["insee"][insee]
        if postcode and postcode in index["postcode"]:
            return index["postcode"][postcode]
        return []

    # union mode (dedupe by siret)
    seen = set()
    results: List[Tuple[str, dict]] = []
    if insee and insee in index["insee"]:
        for siret, cand in index["insee"][insee]:
            if siret not in seen:
                results.append((siret, cand))
                seen.add(siret)
    if postcode and postcode in index["postcode"]:
        for siret, cand in index["postcode"][postcode]:
            if siret not in seen:
                results.append((siret, cand))
                seen.add(siret)
    return results


def _address_rescue_indices(feats: List[Dict[str, float]], top_idx: Iterable[int]) -> List[int]:
    top_set = set(top_idx)
    rescue = []
    for i, feat in enumerate(feats):
        if i in top_set:
            continue
        addr_jaro = float(feat.get("addr_jaro", 0.0))
        street_name_jaro = float(feat.get("street_name_jaro", 0.0))
        street_number_diff = float(feat.get("street_number_diff", 9999))
        if addr_jaro >= 0.96 and street_name_jaro >= 0.95 and street_number_diff <= 2:
            rescue.append(i)
    return rescue


def _token_count(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


def _format_pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def _calibration_bins(scores: List[float], correct: List[int], bins: List[float]) -> List[Dict[str, Any]]:
    rows = []
    scores_arr = np.array(scores, dtype=float)
    correct_arr = np.array(correct, dtype=int)

    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i + 1]
        mask = (scores_arr > low) & (scores_arr <= high)
        if not mask.any():
            rows.append(
                {
                    "score_bin": f"({low}, {high}]",
                    "errors": 0,
                    "count": 0,
                    "mean_score": 0.0,
                    "error_rate": 0.0,
                    "expected_correct": 0.0,
                    "actual_correct": 0.0,
                }
            )
            continue
        subset_scores = scores_arr[mask]
        subset_correct = correct_arr[mask]
        count = len(subset_scores)
        errors = int(count - subset_correct.sum())
        mean_score = float(subset_scores.mean())
        error_rate = errors / count if count else 0.0
        rows.append(
            {
                "score_bin": f"({low}, {high}]",
                "errors": errors,
                "count": count,
                "mean_score": mean_score,
                "error_rate": error_rate,
                "expected_correct": mean_score,
                "actual_correct": 1.0 - error_rate,
            }
        )
    return rows


def _choose_threshold(scores: List[float], correct: List[int], thresholds: List[float], target_err: float) -> Dict[str, Any]:
    best = None
    scores_arr = np.array(scores, dtype=float)
    correct_arr = np.array(correct, dtype=int)

    for thr in thresholds:
        mask = scores_arr >= thr
        auto_count = int(mask.sum())
        if auto_count == 0:
            continue
        err = auto_count - int(correct_arr[mask].sum())
        err_rate = err / auto_count if auto_count else 0.0
        coverage = auto_count / len(scores_arr) if len(scores_arr) else 0.0
        if err_rate <= target_err:
            if best is None or coverage > best["coverage"]:
                best = {
                    "threshold": thr,
                    "coverage": coverage,
                    "error_rate": err_rate,
                    "auto_count": auto_count,
                }

    if best is None and thresholds:
        # Fallback: pick max threshold for lowest risk
        thr = max(thresholds)
        mask = scores_arr >= thr
        auto_count = int(mask.sum())
        err = auto_count - int(correct_arr[mask].sum())
        err_rate = err / auto_count if auto_count else 0.0
        coverage = auto_count / len(scores_arr) if len(scores_arr) else 0.0
        best = {
            "threshold": thr,
            "coverage": coverage,
            "error_rate": err_rate,
            "auto_count": auto_count,
        }

    return best or {"threshold": None, "coverage": 0.0, "error_rate": 0.0, "auto_count": 0}


def _plot_risk_coverage(thresholds: List[float], coverage: List[float], error_rate: List[float], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(thresholds, coverage, color="#1f77b4", label="Coverage")
    ax1.set_xlabel("Seuil θ")
    ax1.set_ylabel("Coverage", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax2 = ax1.twinx()
    ax2.plot(thresholds, error_rate, color="#d62728", label="Error rate")
    ax2.set_ylabel("Taux d'erreur", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    ax1.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnostic XGB routing (risk-coverage + calibration).")
    parser.add_argument("--input-path", type=Path, default=Path("data/samples_v4_with_ranker.parquet"))
    parser.add_argument("--split", choices=["train", "dev", "test"], default="test", help="Split to evaluate (only used if input is parquet)")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--pool-mode", choices=["insee_then_postcode", "union"], default="insee_then_postcode")
    parser.add_argument("--thresholds", type=str, default="0.90,0.95,0.99,0.995")
    parser.add_argument("--max-topn", type=int, default=200)
    parser.add_argument("--limit", type=int, default=0, help="Limit number of rows for quick runs.")
    parser.add_argument("--include-closed-candidates", action="store_true")
    parser.add_argument("--keep-unnamed-candidates", action="store_true")
    parser.add_argument("--policy", choices=["on", "off"], default="on")
    parser.add_argument("--semantic", action="store_true", help="Force semantic features on.")
    parser.add_argument("--calibration-scope", choices=["eligible", "all"], default="eligible")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--model-dir", type=Path, default=None)
    args = parser.parse_args()

    # Load input data (CSV or parquet)
    if args.input_path.suffix == ".parquet":
        # Parquet mode: filter by split, but warn about limitations
        print("=" * 60)
        print("ERROR: Parquet mode is not fully supported for this script.")
        print("=" * 60)
        print()
        print("This script recomputes features from CRM fields (crm_name, crm_address, etc.)")
        print("which are not present in the parquet file.")
        print()
        print("Please use a CSV file with CRM fields instead:")
        print("  python scripts/diagnostic_xgb_routing.py --input-path data/old/2026-01-11_splits/test.csv")
        print()
        print("Or use evaluate_xgb_comprehensive.py for parquet-based evaluation.")
        sys.exit(1)
    else:
        df = pd.read_csv(args.input_path, dtype=str)
    
    if args.limit and args.limit > 0:
        df = df.head(args.limit)

    # Build candidate pool
    postcodes = set(df.get("postcode", pd.Series(dtype=str)).dropna().unique())
    insee_codes = set(df.get("insee", pd.Series(dtype=str)).dropna().unique())
    if "Code Postal" in df.columns:
        postcodes.update(df["Code Postal"].dropna().unique())
    if "Code INSEE" in df.columns:
        insee_codes.update(df["Code INSEE"].dropna().unique())

    candidates = load_candidates_for_locations(
        postcodes,
        insee_codes,
        include_closed_establishments=args.include_closed_candidates,
        drop_unnamed_candidates=not args.keep_unnamed_candidates,
    )
    index = build_location_index(candidates)

    ranker, classifier, feature_order, scaler = _load_models(args.model_dir)

    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    thresholds = sorted(set(thresholds))

    results = []

    force_semantic = args.semantic
    if force_semantic:
        os.environ["XGB_SEMANTIC_ENABLED"] = "1"

    for idx, row in df.iterrows():
        crm_dict = _build_crm_dict(row)
        crm_ctx = preprocess_crm_row(crm_dict)
        cand_list = _pool_candidates(row, candidates, index, args.pool_mode)

        if not cand_list:
            results.append(
                {
                    "crm_id": row.get("crm_id", idx),
                    "status": "no_candidates",
                    "eligible": False,
                }
            )
            continue

        gt_siret = row.get("ground_truth_siret") or row.get("SIRET") or ""
        gt_siret = str(gt_siret).strip()
        gt_in_pool = bool(gt_siret) and any(s == gt_siret for s, _ in cand_list)

        # Stage 1 features
        feats_stage1 = []
        for siret, cand in cand_list:
            feat = make_features_from_preprocessed(crm_ctx, cand, skip_semantic=True)
            feats_stage1.append(feat)

        X1 = pd.DataFrame(feats_stage1)
        for col in feature_order:
            if col not in X1.columns:
                X1[col] = 0.0
        X1 = X1[feature_order].fillna(0.0)
        Xs1 = scaler.transform(X1) if scaler is not None else X1.values
        dmatrix = xgb.DMatrix(Xs1, feature_names=feature_order)
        rank_scores = ranker.predict(dmatrix)

        # Top-N selection
        stage1_top_n = min(args.max_topn, len(rank_scores))
        top_n_idx = np.argsort(rank_scores)[::-1][:stage1_top_n]

        # Address rescue (align with inference)
        rescue = _address_rescue_indices(feats_stage1, top_n_idx)
        if rescue:
            rescue = rescue[:50]
            top_n_idx = np.unique(np.concatenate([top_n_idx, np.array(rescue, dtype=int)]))

        # Stage 2 features (semantic optional)
        feats_n = []
        cand_list_n = []
        semantic_pools = []

        for idx_n in top_n_idx:
            siret, cand = cand_list[idx_n]
            feat = feats_stage1[idx_n]
            feat["_siret"] = siret
            feat["_cand_name"] = primary_name(cand) or f"SIRET {siret}"
            feat["_ul_denoms"] = [
                normalize_text(cand.get("denomination_ul") or ""),
                normalize_text(cand.get("denomination_usuelle_ul") or ""),
                normalize_text(cand.get("sigle_ul") or ""),
            ]
            feat["_is_siege"] = bool(cand.get("is_siege", False))

            if _semantic_enabled():
                cand_city_norm = cand.get("_xgb_cached_city_norm") or normalize_text(cand.get("city"))
                semantic_pools.append(
                    build_semantic_name_pool(
                        build_candidate_names(cand),
                        crm_city_norm=crm_ctx.get("crm_city_norm", ""),
                        cand_city_norm=cand_city_norm,
                    )
                )
            feats_n.append(feat)
            cand_list_n.append((siret, cand))

        if _semantic_enabled() and semantic_pools:
            sem = top2_semantic_similarities_batch(crm_ctx.get("crm_name_semantic", ""), semantic_pools)
            for feat, (sem_max, sem_second, sem_gap) in zip(feats_n, sem, strict=True):
                feat["name_semantic_max"] = sem_max
                feat["name_semantic_second"] = sem_second
                feat["name_semantic_gap"] = sem_gap

        X_n = pd.DataFrame(feats_n)
        for col in feature_order:
            if col not in X_n.columns:
                X_n[col] = 0.0
        X_n = X_n[feature_order].fillna(0.0)
        Xs_n = scaler.transform(X_n) if scaler is not None else X_n.values
        class_probs = classifier.predict_proba(Xs_n)[:, 1]

        scores = np.array(class_probs, dtype=float)
        if args.policy == "on":
            scores = apply_rerank_rules(scores, feats_n, row)

        # Top-1 selection
        order = np.argsort(scores)[::-1]
        top1_idx = int(order[0])
        top2_idx = int(order[1]) if len(order) > 1 else None

        siret_pred = cand_list_n[top1_idx][0]
        score_top1 = float(scores[top1_idx])
        score_top2 = float(scores[top2_idx]) if top2_idx is not None else 0.0

        correct = bool(gt_siret and siret_pred == gt_siret)
        status = "ok" if gt_in_pool else "gt_not_in_pool"

        name_word_count = _token_count(crm_ctx.get("crm_name", ""))
        addr_complete = bool(crm_ctx.get("crm_street_num") and crm_ctx.get("crm_street_name"))

        # Extract routing features from top1
        top_feat = feats_n[top1_idx]
        results.append(
            {
                "crm_id": row.get("crm_id", idx),
                "status": status,
                "eligible": gt_in_pool,
                "ground_truth": gt_siret,
                "pred_siret": siret_pred,
                "score_top1": score_top1,
                "score_top2": score_top2,
                "correct": int(correct),
                "name_word_count": name_word_count,
                "addr_complete": int(addr_complete),
                "name_jaro_max": float(top_feat.get("name_jaro_max", 0.0)),
                "name_token_overlap_max": float(top_feat.get("name_token_overlap_max", 0.0)),
                "name_sim_max_etab": float(top_feat.get("name_sim_max_etab", 0.0)),
                "name_crm_contains_cand_max": float(top_feat.get("name_crm_contains_cand_max", 0.0)),
                "name_sim_max_pm_dirigeant": float(top_feat.get("name_sim_max_pm_dirigeant", 0.0)),
                "addr_jaro": float(top_feat.get("addr_jaro", 0.0)),
                "addr_token_overlap": float(top_feat.get("addr_token_overlap", 0.0)),
                "street_number_diff": float(top_feat.get("street_number_diff", 9999.0)),
                "name_length_max": float(top_feat.get("name_length_max", 0.0)),
            }
        )

    # Aggregate metrics
    total_count = len(results)
    eligible_results = [r for r in results if r.get("eligible")]
    eligible_count = len(eligible_results)
    no_candidates = sum(1 for r in results if r.get("status") == "no_candidates")
    gt_not_in_pool = sum(1 for r in results if r.get("status") == "gt_not_in_pool")
    correct_eligible = sum(1 for r in eligible_results if r.get("correct") == 1)
    precision_at_1 = correct_eligible / eligible_count if eligible_count else 0.0

    # Risk-coverage
    risk_coverage = []
    eligible_scores = [r["score_top1"] for r in eligible_results]
    eligible_correct = [r["correct"] for r in eligible_results]

    scores_arr = np.array(eligible_scores, dtype=float)
    correct_arr = np.array(eligible_correct, dtype=int)

    for thr in thresholds:
        mask = scores_arr >= thr
        auto_count = int(mask.sum())
        errors = auto_count - int(correct_arr[mask].sum()) if auto_count else 0
        error_rate = errors / auto_count if auto_count else 0.0
        coverage_total = auto_count / total_count if total_count else 0.0
        coverage_eligible = auto_count / eligible_count if eligible_count else 0.0
        risk_coverage.append(
            {
                "threshold": thr,
                "coverage": coverage_total,
                "coverage_eligible": coverage_eligible,
                "auto_count": auto_count,
                "errors": errors,
                "error_rate": error_rate,
            }
        )

    # Calibration
    if args.calibration_scope == "eligible":
        calib_scores = eligible_scores
        calib_correct = eligible_correct
    else:
        calib_scores = [r["score_top1"] for r in results if r.get("status") != "no_candidates"]
        calib_correct = [r["correct"] for r in results if r.get("status") != "no_candidates"]

    bins = [0.0, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0]
    calibration = _calibration_bins(calib_scores, calib_correct, bins)

    # Segment recommendations
    thresholds_dense = sorted(set([round(x, 3) for x in np.linspace(0.90, 0.995, 20)] + thresholds))
    target_err = 0.05

    def _segment(predicate):
        seg = [r for r in eligible_results if predicate(r)]
        seg_scores = [r["score_top1"] for r in seg]
        seg_correct = [r["correct"] for r in seg]
        if not seg_scores:
            return {"threshold": None, "coverage": 0.0, "error_rate": 0.0, "auto_count": 0}
        return _choose_threshold(seg_scores, seg_correct, thresholds_dense, target_err)

    short_seg = _segment(lambda r: r["name_word_count"] <= 2)
    medium_seg = _segment(lambda r: 3 <= r["name_word_count"] <= 4)
    long_seg = _segment(lambda r: r["name_word_count"] >= 5)
    incomplete_addr_seg = _segment(lambda r: r["addr_complete"] == 0)

    recommendations = {
        "short_name_threshold": short_seg["threshold"],
        "medium_name_threshold": medium_seg["threshold"],
        "long_name_threshold": long_seg["threshold"],
        "incomplete_addr_threshold": incomplete_addr_seg["threshold"],
        "segments": {
            "short_name": short_seg,
            "medium_name": medium_seg,
            "long_name": long_seg,
            "incomplete_addr": incomplete_addr_seg,
        },
    }

    analysis = {
        "meta": {
            "generated_at": _utc_now(),
            "input_path": str(args.input_path),
            "total_count": total_count,
            "eligible_count": eligible_count,
            "no_candidates": no_candidates,
            "gt_not_in_pool": gt_not_in_pool,
            "precision_at_1": precision_at_1,
            "pool_mode": args.pool_mode,
            "policy": args.policy,
            "semantic_enabled": _semantic_enabled(),
            "thresholds": thresholds,
            "calibration_scope": args.calibration_scope,
        },
        "risk_coverage": risk_coverage,
        "calibration": calibration,
        "recommendations": recommendations,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = args.output_dir / "diagnostic_analysis.json"
    report_path = args.output_dir / "diagnostic_report.md"
    plot_path = args.output_dir / "diagnostic_plots.png"

    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    # Build report (simple)
    rows = []
    for thr in thresholds:
        entry = min(risk_coverage, key=lambda r: abs(_safe_float(r["threshold"]) - thr))
        rows.append(entry)

    lines = []
    lines.append("# Diagnostic Quantifie - XGBoost Entity Matching")
    lines.append("")
    lines.append("## Resume Executif")
    lines.append("")
    lines.append("| Metrique | Valeur |")
    lines.append("|---|---|")
    lines.append(f"| **Precision@1** | {_format_pct(precision_at_1, 2)} |")
    lines.append(
        f"| **Taux d'erreur (FP)** | {_format_pct(1 - precision_at_1, 2)} "
        f"({eligible_count - correct_eligible}/{eligible_count}) |"
        if eligible_count
        else "| **Taux d'erreur (FP)** | n/a |"
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Courbe Risk-Coverage")
    lines.append("")
    lines.append(f"![Risk-Coverage]({plot_path.name})")
    lines.append("")
    lines.append("| Seuil theta | Couverture | Erreurs | Taux d'erreur |")
    lines.append("|---|---|---|---|")
    for row in rows:
        lines.append(
            f"| {row['threshold']} | {_format_pct(row['coverage'])} | {row['errors']} | {_format_pct(row['error_rate'])} |"
        )
    lines.append("")
    lines.append(
        f"> Couverture calculee comme auto_count / total_count avec total_count={total_count}."
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Plot
    _plot_risk_coverage(
        thresholds=[r["threshold"] for r in risk_coverage],
        coverage=[r["coverage"] for r in risk_coverage],
        error_rate=[r["error_rate"] for r in risk_coverage],
        out_path=plot_path,
    )

    if args.save_predictions:
        pred_path = args.output_dir / "diagnostic_predictions.csv"
        pd.DataFrame(results).to_csv(pred_path, index=False)


if __name__ == "__main__":
    main()
