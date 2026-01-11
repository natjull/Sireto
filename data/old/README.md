# Archived Data - SIRETO

This directory contains obsolete data files that have been superseded by canonical versions.

## Archive Structure

### 2026-01-11_splits/
CSV splits that duplicated information already in the parquet `split` column.
- `train.csv`, `dev.csv`, `test.csv` - now use `samples_v4_with_ranker.parquet` with `split` column

### 2026-01-11_candidates/
Obsolete candidate partition directories:
- `candidates_v4/` - included closed establishments (open + closed)
- `candidates_v4_fixed/` - intermediate fix attempt

**Current canonical:** `data/candidates_v4_active/` (open-only, used for training)

### 2026-01-11_samples/
Obsolete sample files from various pipeline iterations:
- `samples_aligned_qw.*` - early Qwant-aligned version
- `samples_aligned_v3.*` - v3 alignment
- `samples_aligned_v3_compat.parquet` - compatibility layer
- `samples_v4.*` - v4 without active-only filter
- `samples_v4_active.*` - intermediate (same content as canonical but different path)
- `samples_v4_fixed.*` - fixed candidates version
- `samples_v4_nd_fix.parquet` - ND fix attempt
- `samples_v4_no_ranker.*` - without ranker scores
- `samples_v4_sem.*` - semantic features version

**Current canonical:** `data/samples_v4_with_ranker.parquet` + `data/samples_v4_with_ranker.json`

## Canonical Files (DO NOT ARCHIVE)

```
data/
  samples_v4_with_ranker.parquet   # Training samples with ranker scores
  samples_v4_with_ranker.json      # Generation metadata
  candidates_v4_active/            # Partitioned candidates (open-only)
  StockEtablissement_utf8.parquet  # Raw SIRENE establishments
  StockUniteLegale_utf8.parquet    # Raw SIRENE legal units
  entrainements.csv                # CRM ground truth
```

## Why Archive Instead of Delete?

These files are kept for:
1. Reproducibility of past experiments
2. Debugging if issues arise with new canonical files
3. Historical reference

Safe to delete after validation period (e.g., 3 months post-production).
