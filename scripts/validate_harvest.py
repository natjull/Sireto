#!/usr/bin/env python3
"""
Validation du harvest - Compare avec la référence parquet.
"""

import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path("data/harvest_full.sqlite")
REF_PATH = Path("data/StockUniteLegale_utf8.parquet")

def main():
    print("=" * 60)
    print("         VALIDATION HARVEST FULL")
    print("=" * 60)
    print()
    
    # Load harvest
    conn = sqlite3.connect(DB_PATH)
    harvested = pd.read_sql("SELECT siren FROM entreprises", conn)
    harvested_set = set(harvested["siren"].astype(str))
    
    # Stats from DB
    ent_count = len(harvested_set)
    dir_count = pd.read_sql("SELECT COUNT(*) FROM dirigeants", conn).iloc[0, 0]
    etab_count = pd.read_sql("SELECT COUNT(*) FROM etablissements", conn).iloc[0, 0]
    siege_count = pd.read_sql("SELECT COUNT(*) FROM sieges", conn).iloc[0, 0]
    fin_count = pd.read_sql("SELECT COUNT(*) FROM finances", conn).iloc[0, 0]
    conn.close()
    
    print("📊 DONNÉES RÉCOLTÉES:")
    print(f"   Entreprises:    {ent_count:>12,}")
    print(f"   Sièges:         {siege_count:>12,}")
    print(f"   Établissements: {etab_count:>12,}")
    print(f"   Dirigeants:     {dir_count:>12,}")
    print(f"   Finances:       {fin_count:>12,}")
    print()
    
    # Load reference
    print("📂 Chargement référence parquet...")
    ref = pd.read_parquet(REF_PATH, columns=["siren", "statutDiffusionUniteLegale"])
    
    total_ref = len(ref)
    diffusibles = ref[ref["statutDiffusionUniteLegale"] == "O"]
    diffusibles_set = set(diffusibles["siren"].astype(str))
    non_diffusibles = total_ref - len(diffusibles_set)
    
    print(f"   Total référence:    {total_ref:>12,}")
    print(f"   Diffusibles (O):    {len(diffusibles_set):>12,}")
    print(f"   Non-diffusibles (P): {non_diffusibles:>12,}")
    print()
    
    # Coverage analysis
    covered = harvested_set & diffusibles_set
    missing = diffusibles_set - harvested_set
    extra = harvested_set - set(ref["siren"].astype(str))
    
    coverage = len(covered) / len(diffusibles_set) * 100 if diffusibles_set else 0
    
    print("📈 COUVERTURE:")
    print(f"   SIREN récupérés:    {len(covered):>12,}")
    print(f"   SIREN manquants:    {len(missing):>12,}")
    print(f"   SIREN extra:        {len(extra):>12,}")
    print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"   COUVERTURE:         {coverage:>11.2f}%")
    print()
    
    if coverage >= 99.9:
        print("✅ HARVEST COMPLET!")
    elif coverage >= 95:
        print("⚠️  Harvest quasi-complet, quelques SIREN manquants.")
    else:
        print(f"❌ Harvest incomplet - {len(missing):,} SIREN à récupérer.")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
