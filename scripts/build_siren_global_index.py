#!/usr/bin/env python3
"""
Build global SIREN-level TF-IDF index for Route B architecture.

Creates:
- data/siren_index/word_vectorizer.pkl (word n-gram 1-2 TfidfVectorizer, L2 norm)
- data/siren_index/word_matrix.npz (7M × V sparse CSR, L2 normalized rows)
- data/siren_index/char_vectorizer.pkl (char_wb 3-5 TfidfVectorizer, L2 norm)
- data/siren_index/char_matrix.npz (7M × V sparse CSR, L2 normalized rows)
- data/siren_index/siren_ids.npy (array of 7M SIREN strings)
- data/siren_index/siren_to_geo.parquet (siren, insee, postcode, siret_count)

Execution time: ~30-45 minutes on full dataset (14M establishments).
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import duckdb
import joblib
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Physical person CJ codes (for person names in UL)
PHYSICAL_PERSON_CJ_CODES = {
    '1000',  # EIRL
    '1100',  # Entrepreneur individuel
    '1210',  # SARL unipersonnelle
    '1300',  # EURL
    '2110',  # Indivision
    '2140',  # Groupement de personnes
    '2300',  # Succursale de société étrangère
    '2400',  # Établissement public
    '3110',  # Association déclarée
    '3220',  # Fondation
    '5410',  # SARL de famille
    '5500',  # Civile immobilière
    '5520',  # Civile mobilière
    '5540',  # Civile d'exploitation agricole
    '5560',  # Civile professionnelle
    '5630',  # De droit local
    '5710',  # SCP
    '5720',  # Groupement agricole de productivité
    '5800',  # Syndicat
    '8210',  # Établissement de crédit
    '8230',  # Entreprise d'assurance
    '8250',  # Institution financière
    '9110',  # Commune
    '9150',  # Région
    '9210',  # Association de communes
    '9220',  # Établissement intercommunal
}


def normalize_name(s: str) -> str:
    """Normalize name for comparison."""
    if not s or not str(s).strip():
        return ""
    s = str(s).upper().strip()
    # Remove punctuation and normalize whitespace
    import re
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def build_siren_index(
    etab_path: Path,
    ul_path: Path,
    output_dir: Path,
) -> None:
    """Build SIREN global index using DuckDB aggregation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading and aggregating SIREN bags via DuckDB...")
    con = duckdb.connect(memory=True)

    # Aggregation: for each SIREN, collect all name variants
    df = con.execute(f"""
        SELECT
            CAST(e.siren AS VARCHAR) AS siren,
            COALESCE(STRING_AGG(DISTINCT COALESCE(e.enseigne1Etablissement, ''), ' '), '') ||
            ' ' ||
            COALESCE(STRING_AGG(DISTINCT COALESCE(e.enseigne2Etablissement, ''), ' '), '') ||
            ' ' ||
            COALESCE(STRING_AGG(DISTINCT COALESCE(e.enseigne3Etablissement, ''), ' '), '') ||
            ' ' ||
            COALESCE(STRING_AGG(DISTINCT COALESCE(e.denominationUsuelleEtablissement, ''), ' '), '') ||
            ' ' ||
            COALESCE(STRING_AGG(DISTINCT COALESCE(ul.denominationUniteLegale, ''), ' '), '') ||
            ' ' ||
            COALESCE(STRING_AGG(DISTINCT COALESCE(ul.denominationUsuelle1UniteLegale, ''), ' '), '') ||
            ' ' ||
            COALESCE(STRING_AGG(DISTINCT COALESCE(ul.denominationUsuelle2UniteLegale, ''), ' '), '') ||
            ' ' ||
            COALESCE(STRING_AGG(DISTINCT COALESCE(ul.sigleUniteLegale, ''), ' '), '')
                AS name_bag_raw
        FROM read_parquet('{etab_path}') e
        LEFT JOIN read_parquet('{ul_path}') ul
          ON CAST(e.siren AS VARCHAR) = CAST(ul.siren AS VARCHAR)
        GROUP BY e.siren
    """).df()

    logger.info(f"Aggregated {len(df)} unique SIRENs")

    # Normalize names (vectorized)
    df["name_bag"] = (
        df["name_bag_raw"]
        .fillna("")
        .str.upper()
        .str.replace(r'[^\w\s]', ' ', regex=True)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )

    siren_ids = df["siren"].values.astype(str)
    name_bags = df["name_bag"].values

    logger.info("Building Word TF-IDF (n-gram 1-2, L2 norm)...")
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        lowercase=False,
        token_pattern=r"(?u)\b\w+\b",
        min_df=2,
        max_df=0.95,
        norm='l2'  # L2 normalization critical for global SIREN index
    )
    word_matrix = word_vec.fit_transform(name_bags)
    logger.info(f"Word matrix: {word_matrix.shape}, nnz={word_matrix.nnz}")

    logger.info("Building Char TF-IDF (char_wb 3-5, L2 norm)...")
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=False,
        min_df=2,
        max_df=0.95,
        norm='l2'
    )
    char_matrix = char_vec.fit_transform(name_bags)
    logger.info(f"Char matrix: {char_matrix.shape}, nnz={char_matrix.nnz}")

    # Persist vectorizers and matrices
    logger.info(f"Saving to {output_dir}...")
    joblib.dump(word_vec, output_dir / "word_vectorizer.pkl")
    joblib.dump(char_vec, output_dir / "char_vectorizer.pkl")
    scipy.sparse.save_npz(output_dir / "word_matrix.npz", word_matrix.tocsr())
    scipy.sparse.save_npz(output_dir / "char_matrix.npz", char_matrix.tocsr())
    np.save(output_dir / "siren_ids.npy", siren_ids)
    logger.info("Vectorizers and matrices saved.")

    # Build siren_to_geo index: for each SIREN, all its geo locations
    logger.info("Building siren_to_geo index...")
    geo_df = con.execute(f"""
        SELECT
            CAST(e.siren AS VARCHAR) AS siren,
            CAST(e.codeCommuneEtablissement AS VARCHAR) AS insee,
            CAST(e.codePostalEtablissement AS VARCHAR) AS postcode,
            COUNT(*) AS siret_count
        FROM read_parquet('{etab_path}') e
        WHERE e.etatAdministratifEtablissement = 'A'
        GROUP BY e.siren, e.codeCommuneEtablissement, e.codePostalEtablissement
    """).df()

    logger.info(f"Geo index: {len(geo_df)} siren-geo combinations")
    geo_df.to_parquet(output_dir / "siren_to_geo.parquet", index=False)
    logger.info("siren_to_geo.parquet saved.")

    logger.info("Done! Index ready at " + str(output_dir))


def main():
    parser = argparse.ArgumentParser(
        description="Build global SIREN-level TF-IDF index for Route B."
    )
    parser.add_argument(
        "--etab",
        type=Path,
        default=Path("data/StockEtablissement_utf8.parquet"),
        help="Path to StockEtablissement parquet",
    )
    parser.add_argument(
        "--ul",
        type=Path,
        default=Path("data/StockUniteLegale_utf8.parquet"),
        help="Path to StockUniteLegale parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/siren_index"),
        help="Output directory for index artifacts",
    )

    args = parser.parse_args()

    if not args.etab.exists():
        logger.error(f"Establishment file not found: {args.etab}")
        exit(1)
    if not args.ul.exists():
        logger.error(f"Legal unit file not found: {args.ul}")
        exit(1)

    build_siren_index(args.etab, args.ul, args.output)


if __name__ == "__main__":
    main()
