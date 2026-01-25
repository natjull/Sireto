#!/usr/bin/env python3
"""
Audit Ground Truth: Identify CRM entries with location mismatch vs SIRENE.

Rule: GT is valid if INSEE matches OR CP matches. Otherwise excluded from training.

Outputs:
  - data/crm_ok_gt.csv   : Clean GT (can be used for training)
  - data/crm_bad_gt.csv  : Bad GT (location mismatch, excluded)
  - data/crm_no_gt.csv   : GT SIRET not found in SIRENE
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Audit GT location match")
    parser.add_argument(
        "--crm",
        type=Path,
        default=Path("data/entrainements.csv"),
        help="CRM CSV with GT SIRET",
    )
    parser.add_argument(
        "--sirene",
        type=Path,
        default=Path("data/StockEtablissement_utf8.parquet"),
        help="SIRENE establishments parquet",
    )
    parser.add_argument(
        "--out-ok",
        type=Path,
        default=Path("data/crm_ok_gt.csv"),
        help="Output: valid GT entries",
    )
    parser.add_argument(
        "--out-bad",
        type=Path,
        default=Path("data/crm_bad_gt.csv"),
        help="Output: invalid GT (location mismatch)",
    )
    parser.add_argument(
        "--out-missing",
        type=Path,
        default=Path("data/crm_no_gt.csv"),
        help="Output: GT SIRET not found in SIRENE",
    )
    args = parser.parse_args()

    # Load CRM
    print(f"Loading CRM from {args.crm}...")
    crm = pd.read_csv(args.crm, sep=";", dtype=str, encoding="utf-8-sig")
    crm.columns = crm.columns.str.strip()
    
    # Normalize column names for consistency
    col_map = {
        "SIRET": "gt_siret",
        "CODE_POSTAL": "crm_cp",
        "CODE_INSEE": "crm_insee",
        "SITE": "crm_name",
        "SERVICE ID": "crm_id",
        "COMMUNE": "crm_commune",
        "SITE_CLI_ADRESSE": "crm_adresse",
    }
    crm = crm.rename(columns=col_map)
    
    # Clean SIRET (remove spaces, ensure 14 chars)
    crm["gt_siret"] = crm["gt_siret"].str.replace(r"\s+", "", regex=True)
    crm["gt_siret"] = crm["gt_siret"].str.zfill(14)
    
    # Clean codes
    crm["crm_cp"] = crm["crm_cp"].str.strip().str.zfill(5)
    crm["crm_insee"] = crm["crm_insee"].str.strip().str.zfill(5)
    
    print(f"  Total CRM entries: {len(crm)}")
    
    # Load SIRENE (only needed columns)
    print(f"Loading SIRENE from {args.sirene}...")
    sirene_cols = [
        "siret",
        "codeCommuneEtablissement",
        "codePostalEtablissement",
        "etatAdministratifEtablissement",
    ]
    sirene = pd.read_parquet(args.sirene, columns=sirene_cols)
    sirene = sirene.rename(columns={
        "siret": "gt_siret",
        "codeCommuneEtablissement": "sirene_insee",
        "codePostalEtablissement": "sirene_cp",
        "etatAdministratifEtablissement": "sirene_etat",
    })
    sirene["gt_siret"] = sirene["gt_siret"].str.zfill(14)
    sirene["sirene_insee"] = sirene["sirene_insee"].str.zfill(5)
    sirene["sirene_cp"] = sirene["sirene_cp"].str.zfill(5)
    
    print(f"  SIRENE establishments: {len(sirene):,}")
    
    # Merge
    print("Merging CRM with SIRENE...")
    merged = crm.merge(sirene, on="gt_siret", how="left")
    
    # Categorize
    # 1. GT not found in SIRENE
    mask_missing = merged["sirene_insee"].isna()
    
    # 2. Location match: INSEE matches OR CP matches
    mask_insee_match = merged["crm_insee"] == merged["sirene_insee"]
    mask_cp_match = merged["crm_cp"] == merged["sirene_cp"]
    mask_loc_ok = mask_insee_match | mask_cp_match
    
    # Final categories
    df_missing = merged[mask_missing].copy()
    df_ok = merged[~mask_missing & mask_loc_ok].copy()
    df_bad = merged[~mask_missing & ~mask_loc_ok].copy()
    
    # Add diagnostic columns
    df_ok["loc_match_type"] = "insee"
    df_ok.loc[~mask_insee_match[~mask_missing & mask_loc_ok], "loc_match_type"] = "cp_only"
    
    df_bad["mismatch_detail"] = (
        "CRM(" + df_bad["crm_insee"] + "/" + df_bad["crm_cp"] + ") vs "
        "SIRENE(" + df_bad["sirene_insee"] + "/" + df_bad["sirene_cp"] + ")"
    )
    
    # Stats
    print("\n" + "=" * 60)
    print("AUDIT RESULTS")
    print("=" * 60)
    print(f"  Total CRM entries:     {len(crm):>6}")
    print(f"  GT not in SIRENE:      {len(df_missing):>6} ({100*len(df_missing)/len(crm):.1f}%)")
    print(f"  GT location OK:        {len(df_ok):>6} ({100*len(df_ok)/len(crm):.1f}%)")
    print(f"  GT location MISMATCH:  {len(df_bad):>6} ({100*len(df_bad)/len(crm):.1f}%)")
    print("=" * 60)
    
    # Breakdown of OK
    if len(df_ok) > 0:
        insee_exact = (df_ok["loc_match_type"] == "insee").sum()
        cp_only = (df_ok["loc_match_type"] == "cp_only").sum()
        print(f"\n  OK breakdown:")
        print(f"    INSEE match:  {insee_exact:>6}")
        print(f"    CP only:      {cp_only:>6}")
    
    # Breakdown of BAD by state
    if len(df_bad) > 0:
        print(f"\n  BAD breakdown by SIRENE state:")
        for etat, cnt in df_bad["sirene_etat"].value_counts().items():
            label = "Active" if etat == "A" else "Fermé" if etat == "F" else etat
            print(f"    {label} ({etat}): {cnt:>6}")
        
        # Show sample of mismatches
        print(f"\n  Sample BAD entries:")
        sample = df_bad.head(10)[["crm_id", "crm_name", "mismatch_detail", "sirene_etat"]]
        for _, row in sample.iterrows():
            print(f"    {row['crm_id']}: {row['crm_name'][:40]:<40} | {row['mismatch_detail']} | {row['sirene_etat']}")
    
    # Save outputs
    print(f"\nSaving outputs...")
    
    # Keep original columns for OK/BAD
    out_cols = [c for c in crm.columns if c in df_ok.columns]
    out_cols += ["sirene_insee", "sirene_cp", "sirene_etat", "loc_match_type"]
    
    df_ok[[c for c in out_cols if c in df_ok.columns]].to_csv(args.out_ok, index=False, sep=";")
    print(f"  {args.out_ok}: {len(df_ok)} entries")
    
    bad_cols = [c for c in crm.columns if c in df_bad.columns]
    bad_cols += ["sirene_insee", "sirene_cp", "sirene_etat", "mismatch_detail"]
    df_bad[[c for c in bad_cols if c in df_bad.columns]].to_csv(args.out_bad, index=False, sep=";")
    print(f"  {args.out_bad}: {len(df_bad)} entries")
    
    missing_cols = [c for c in crm.columns if c in df_missing.columns]
    df_missing[missing_cols].to_csv(args.out_missing, index=False, sep=";")
    print(f"  {args.out_missing}: {len(df_missing)} entries")
    
    print("\nDone.")


if __name__ == "__main__":
    main()
