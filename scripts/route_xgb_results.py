"""
Route XGBoost top-k results into AUTO / REVIEW / NO_MATCH.

Expected input: CSV with top-k rows per CRM (from infer_xgb_matcher_topk.py).
Required columns (minimum):
  - crm_id
  - rank (optional but recommended)
  - siret_candidate
  - score
Optional columns used for AUTO routing:
  - name_semantic_max
  - name_semantic_second
Optional columns for explainability:
  - shap
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Route XGB top-k results into AUTO/REVIEW/NO_MATCH.")
    p.add_argument("--input-path", type=Path, required=True)
    p.add_argument("--output-path", type=Path, required=True)
    p.add_argument("--only-review", action="store_true", help="Export only REVIEW/NO_MATCH rows.")
    return p.parse_args()


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _pick_top_rows(group: pd.DataFrame) -> tuple[pd.Series, pd.Series | None]:
    if "rank" in group.columns:
        group_sorted = group.sort_values(["rank", "score"], ascending=[True, False])
    else:
        group_sorted = group.sort_values(["score"], ascending=[False])
    top1 = group_sorted.iloc[0]
    top2 = group_sorted.iloc[1] if len(group_sorted) > 1 else None
    return top1, top2


def _route_xgb(top1: pd.Series) -> str:
    score = _safe_float(top1.get("score"))
    name_semantic_max = _safe_float(top1.get("name_semantic_max"))
    name_semantic_second = _safe_float(top1.get("name_semantic_second"))

    auto = (
        score >= 0.99
        or (score >= 0.95 and name_semantic_max >= 0.75)
        or (score >= 0.98 and name_semantic_second >= 0.65)
    )
    if auto:
        return "AUTO"
    if score >= 0.70:
        return "REVIEW"
    return "NO_MATCH"


def main() -> None:
    args = _parse_args()
    df = pd.read_csv(args.input_path)

    if "crm_id" not in df.columns:
        raise ValueError("Missing required column: crm_id")
    if "score" not in df.columns:
        raise ValueError("Missing required column: score")

    rows_out: list[dict] = []
    for crm_id, group in df.groupby("crm_id", dropna=False):
        top1, top2 = _pick_top_rows(group)
        status = _route_xgb(top1)
        if args.only_review and status == "AUTO":
            continue

        row_out = {
            "crm_id": crm_id,
            "xgb_status": status,
            "chosen_siret_xgb": top1.get("siret_candidate"),
            "score_top1": _safe_float(top1.get("score")),
            "score_top2": _safe_float(top2.get("score") if top2 is not None else None),
            "name_semantic_max": _safe_float(top1.get("name_semantic_max")),
            "name_semantic_second": _safe_float(top1.get("name_semantic_second")),
            "xgb_shap_top1": top1.get("shap"),
        }
        rows_out.append(row_out)

    out_df = pd.DataFrame(rows_out)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    main()
