#!/usr/bin/env python3
"""
Analyze missing ground-truth coverage in samples V4.

Outputs:
  - summary counts by reason
  - optional CSV with detailed rows
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import pyarrow.dataset as ds


DEFAULT_TRAINING = Path("data/entrainements.csv")
DEFAULT_SAMPLES = Path("data/samples_v4.parquet")
DEFAULT_ETAB = Path("data/StockEtablissement_utf8.parquet")
DEFAULT_OUT = Path("reports/missing_gt_v4.csv")
CHUNK = 2000


def _norm_code(x: object) -> str | None:
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return None
    try:
        return str(int(float(s)))
    except Exception:
        return s


def load_training(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8-sig")
    df = df.rename(
        columns={
            "SITE": "crm_name",
            "SITE_CLI_ADRESSE": "crm_address",
            "SITE_CLI_COMMUNE": "crm_city",
            "CODE_POSTAL": "postcode",
            "CODE_INSEE": "insee",
            "SIRET": "ground_truth_siret",
        }
    )
    df = df[df["ground_truth_siret"].notna() & (df["ground_truth_siret"].str.len() == 14)]
    df["crm_id"] = range(len(df))
    df["insee"] = df["insee"].apply(_norm_code)
    df["postcode"] = df["postcode"].apply(_norm_code)
    return df


def iter_chunks(values: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(values), size):
        yield values[i : i + size]


def lookup_gt_locations(parquet_path: Path, sirets: List[str]) -> pd.DataFrame:
    dataset = ds.dataset(parquet_path, format="parquet")
    rows = []
    for chunk in iter_chunks(sirets, CHUNK):
        filt = ds.field("siret").isin(chunk)
        table = dataset.to_table(
            filter=filt,
            columns=["siret", "codeCommuneEtablissement", "codePostalEtablissement"],
        )
        if table.num_rows:
            rows.append(table.to_pandas())
    if not rows:
        return pd.DataFrame(columns=["siret", "codeCommuneEtablissement", "codePostalEtablissement"])
    df = pd.concat(rows, ignore_index=True)
    df = df.rename(
        columns={
            "codeCommuneEtablissement": "gt_insee",
            "codePostalEtablissement": "gt_postcode",
        }
    )
    df["gt_insee"] = df["gt_insee"].apply(_norm_code)
    df["gt_postcode"] = df["gt_postcode"].apply(_norm_code)
    return df


def lookup_in_store(partitions_dir: Path, sirets: List[str]) -> pd.DataFrame:
    rows = []
    for sub in ["insee", "cp"]:
        root = partitions_dir / sub
        if not root.exists():
            continue
        dataset = ds.dataset(root, format="parquet", partitioning="hive")
        for chunk in iter_chunks(sirets, CHUNK):
            filt = ds.field("siret").isin(chunk)
            table = dataset.to_table(filter=filt, columns=["siret"])
            if table.num_rows:
                rows.append(table.to_pandas())
    if not rows:
        return pd.DataFrame(columns=["siret"])
    df = pd.concat(rows, ignore_index=True).drop_duplicates()
    df["in_store"] = True
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze missing GT in samples v4.")
    p.add_argument("--training-csv", type=Path, default=DEFAULT_TRAINING)
    p.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    p.add_argument("--etab-parquet", type=Path, default=DEFAULT_ETAB)
    p.add_argument("--partitions-dir", type=Path, default=None)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    train = load_training(args.training_csv)
    samples = pd.read_parquet(args.samples, columns=["query_id", "label"])
    pos_queries = set(samples[samples["label"] == 1]["query_id"].unique())
    missing = train[~train["crm_id"].isin(pos_queries)].copy()
    missing_sirets = missing["ground_truth_siret"].dropna().unique().tolist()

    gt_loc = lookup_gt_locations(args.etab_parquet, missing_sirets)
    missing = missing.merge(gt_loc, left_on="ground_truth_siret", right_on="siret", how="left")
    missing.drop(columns=["siret"], inplace=True, errors="ignore")

    if args.partitions_dir:
        store_df = lookup_in_store(args.partitions_dir, missing_sirets)
        missing = missing.merge(store_df, left_on="ground_truth_siret", right_on="siret", how="left")
        missing.drop(columns=["siret"], inplace=True, errors="ignore")
        missing["in_store"] = missing["in_store"].fillna(False)

    def reason(row: pd.Series) -> str:
        if pd.isna(row.get("gt_insee")) and pd.isna(row.get("gt_postcode")):
            return "GT_NOT_IN_SIRENE"
        crm_insee = row.get("insee")
        crm_cp = row.get("postcode")
        gt_insee = row.get("gt_insee")
        gt_cp = row.get("gt_postcode")
        if crm_insee and gt_insee and crm_insee == gt_insee:
            return "CRM_INSEE_MATCH_BUT_MISSING"
        if crm_cp and gt_cp and crm_cp == gt_cp:
            return "CRM_CP_MATCH_BUT_MISSING"
        return "CRM_LOC_MISMATCH"

    missing["reason"] = missing.apply(reason, axis=1)

    print("=" * 60)
    print("MISSING GT ANALYSIS (V4)")
    print("=" * 60)
    print(f"Missing queries: {len(missing)}")
    print("\nReason counts:")
    print(missing["reason"].value_counts())

    if args.partitions_dir and "in_store" in missing.columns:
        print("\nIn store counts:")
        print(missing["in_store"].value_counts())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    missing.to_csv(args.output, index=False)
    print(f"\nSaved details to {args.output}")


if __name__ == "__main__":
    main()
