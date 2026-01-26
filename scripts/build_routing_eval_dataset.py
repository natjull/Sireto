#!/usr/bin/env python3
"""
Build a query-level routing evaluation dataset (one row per crm_id).

Outputs top1/top2 features, deltas, and GT labels (STRICT + SAME_SIREN).
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd

from src.xgb_matcher.routing_risk import BASE_FEATURES


def _read_csv_auto(path: Path) -> pd.DataFrame:
    sample = path.read_text(encoding="utf-8", errors="ignore")[:2048]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=";,	")
        delimiter = dialect.delimiter
    except Exception:
        delimiter = ";"
    return pd.read_csv(path, dtype=str, sep=delimiter)


def _normalize_siret(value) -> str | None:
    if pd.isna(value):
        return None
    s = str(value).strip().replace(" ", "")
    if not s or s.lower() == "nan":
        return None
    if "e" in s.lower():
        try:
            s = str(int(float(s)))
        except (ValueError, OverflowError):
            pass
    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        return None
    return digits.zfill(14)


def _normalize_siren(value) -> str | None:
    if pd.isna(value):
        return None
    s = str(value).strip().replace(" ", "")
    if not s or s.lower() == "nan":
        return None
    if "e" in s.lower():
        try:
            s = str(int(float(s)))
        except (ValueError, OverflowError):
            pass
    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        return None
    if len(digits) >= 9:
        return digits[:9].zfill(9)
    return digits.zfill(9)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _has_complete_address(row: pd.Series) -> bool:
    street_number = str(row.get("street_number", "") or "").strip()
    street_name = str(row.get("street_name", "") or "").strip()
    crm_address = str(row.get("crm_address", "") or "").strip()

    has_number = bool(street_number) or bool(re.match(r"^\s*\d+", crm_address))
    if street_name:
        has_street = True
    else:
        addr_tokens = re.findall(r"[A-Z0-9]+", crm_address.upper()) if crm_address else []
        has_street = len(addr_tokens) >= 2
    return bool(has_number and has_street)


def load_topk(path: Path, base_features: Iterable[str]) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    if "crm_id" not in df.columns and "query_id" in df.columns:
        df = df.rename(columns={"query_id": "crm_id"})
    df["crm_id"] = df["crm_id"].astype(str)

    if "rank" in df.columns:
        df["rank"] = pd.to_numeric(df["rank"], errors="coerce").fillna(0).astype(int)
        if df["rank"].min() == 1:
            df["rank"] = df["rank"] - 1

    numeric_cols = ["score", "score_gap", "score_ratio"] + list(base_features)
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "siret_candidate" in df.columns:
        df["siret_candidate"] = df["siret_candidate"].apply(_normalize_siret)
    elif "siret" in df.columns:
        df = df.rename(columns={"siret": "siret_candidate"})
        df["siret_candidate"] = df["siret_candidate"].apply(_normalize_siret)

    return df


def load_ground_truth(path: Path) -> Tuple[Dict[str, set], Dict[str, set]]:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = _read_csv_auto(path)

    if "label" in df.columns and "siret" in df.columns:
        id_col = "crm_id" if "crm_id" in df.columns else "query_id"
        df = df[df["label"].astype(str) == "1"][[id_col, "siret"]]
        df = df.rename(columns={id_col: "crm_id", "siret": "siret_gt"})
    else:
        if "siret" in df.columns and "siret_gt" not in df.columns:
            df = df.rename(columns={"siret": "siret_gt"})
        if "ground_truth_siret" in df.columns:
            df = df.rename(columns={"ground_truth_siret": "siret_gt"})
        if "gt_siret" in df.columns and "siret_gt" not in df.columns:
            df = df.rename(columns={"gt_siret": "siret_gt"})
        if "crm_id" not in df.columns and "query_id" in df.columns:
            df = df.rename(columns={"query_id": "crm_id"})

    if "crm_id" not in df.columns or "siret_gt" not in df.columns:
        raise ValueError("Ground truth must contain crm_id and siret_gt")

    df["crm_id"] = df["crm_id"].astype(str)
    df["siret_gt_norm"] = df["siret_gt"].apply(_normalize_siret)
    df = df[df["siret_gt_norm"].notna()].copy()
    df["siren_gt_norm"] = df["siret_gt_norm"].apply(lambda s: s[:9] if s else None)

    siret_map = df.groupby("crm_id")["siret_gt_norm"].apply(set).to_dict()
    siren_map = df.groupby("crm_id")["siren_gt_norm"].apply(set).to_dict()
    return siret_map, siren_map


def build_dataset(
    topk: pd.DataFrame,
    siret_map: Dict[str, set],
    siren_map: Dict[str, set],
    base_features: Iterable[str],
) -> pd.DataFrame:
    if "rank" in topk.columns:
        topk = topk.sort_values(["crm_id", "rank", "score"], ascending=[True, True, False])
    else:
        topk = topk.sort_values(["crm_id", "score"], ascending=[True, False])

    top1 = topk.groupby("crm_id", as_index=False).first()
    top2 = topk.groupby("crm_id").nth(1).reset_index()

    top1 = top1.rename(columns={"siret_candidate": "top1_siret"})
    top2 = top2.rename(columns={"siret_candidate": "top2_siret"})

    base_cols = ["score"] + [c for c in base_features if c in top1.columns]
    top1_cols = {c: f"top1_{c}" for c in base_cols}
    top2_cols = {c: f"top2_{c}" for c in base_cols}

    top1 = top1.rename(columns=top1_cols)
    top2 = top2.rename(columns=top2_cols)

    out = top1.merge(top2[["crm_id"] + list(top2_cols.values()) + ["top2_siret"]], on="crm_id", how="left")

    out["score_top1"] = out.get("top1_score").apply(_safe_float)
    out["score_top2"] = out.get("top2_score").apply(_safe_float)
    out["score_gap"] = out["score_top1"] - out["score_top2"]
    out["score_ratio"] = out["score_top1"] / out["score_top2"].replace(0, 0.001)

    for feat in base_features:
        c1 = f"top1_{feat}"
        c2 = f"top2_{feat}"
        if c1 in out.columns:
            out[c1] = pd.to_numeric(out[c1], errors="coerce")
        if c2 in out.columns:
            out[c2] = pd.to_numeric(out[c2], errors="coerce")
        if c1 in out.columns and c2 in out.columns:
            out[f"delta_{feat}"] = out[c1].fillna(0.0) - out[c2].fillna(0.0)

    def _is_correct_strict(row) -> int:
        valid = siret_map.get(row["crm_id"], set())
        chosen = row.get("top1_siret")
        return int(chosen in valid if chosen else False)

    def _is_correct_siren(row) -> int:
        valid = siren_map.get(row["crm_id"], set())
        chosen = row.get("top1_siret")
        siren = chosen[:9] if isinstance(chosen, str) and chosen else None
        return int(siren in valid if siren else False)

    out["label_top1_strict"] = out.apply(_is_correct_strict, axis=1)
    out["label_top1_siren"] = out.apply(_is_correct_siren, axis=1)

    return out


def attach_split(out: pd.DataFrame, samples_path: Path | None) -> pd.DataFrame:
    if not samples_path:
        return out
    df = pd.read_parquet(samples_path)
    if "query_id" not in df.columns or "split" not in df.columns:
        return out
    splits = df[["query_id", "split"]].drop_duplicates()
    splits = splits.rename(columns={"query_id": "crm_id"})
    splits["crm_id"] = splits["crm_id"].astype(str)
    return out.merge(splits, on="crm_id", how="left")


def attach_crm_meta(out: pd.DataFrame, crm_meta_path: Path | None, crm_id_col: str) -> pd.DataFrame:
    if not crm_meta_path:
        return out
    df = _read_csv_auto(crm_meta_path)
    if crm_id_col not in df.columns:
        return out
    df = df.rename(columns={crm_id_col: "crm_id"})
    df["crm_id"] = df["crm_id"].astype(str)
    merged = out.merge(df, on="crm_id", how="left")

    if "crm_name" in merged.columns:
        merged["crm_name_word_count"] = merged["crm_name"].astype(str).apply(
            lambda x: len(re.findall(r"[A-Z0-9]+", x.upper())) if x else 0
        )
    if "crm_address" in merged.columns:
        merged["address_complete"] = merged.apply(_has_complete_address, axis=1)
    return merged


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build routing evaluation dataset (query-level).")
    p.add_argument("--topk-path", type=Path, required=True)
    p.add_argument("--gt-path", type=Path, required=True)
    p.add_argument("--output-path", type=Path, default=Path("reports/routing_eval_dataset.parquet"))
    p.add_argument("--samples-path", type=Path, default=None, help="Optional samples parquet to attach split")
    p.add_argument("--crm-meta-path", type=Path, default=None, help="Optional CRM meta CSV to join")
    p.add_argument("--crm-id-col", type=str, default="crm_id")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    topk = load_topk(args.topk_path, BASE_FEATURES)
    siret_map, siren_map = load_ground_truth(args.gt_path)

    out = build_dataset(topk, siret_map, siren_map, BASE_FEATURES)
    out = attach_split(out, args.samples_path)
    out = attach_crm_meta(out, args.crm_meta_path, args.crm_id_col)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.output_path.suffix == ".csv":
        out.to_csv(args.output_path, index=False)
    else:
        out.to_parquet(args.output_path, index=False)

    print(f"Wrote {len(out)} rows to {args.output_path}")


if __name__ == "__main__":
    main()
