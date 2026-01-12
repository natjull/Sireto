# SIRETO Routing SSOT (Single Source of Truth)

> Last updated: 2026-01-11  
> Canonical models: `xgb_two_stage_meta_20260103_132351.json`

## 1. Definitions

### Routing Statuses

| Status | Definition | Action |
|--------|------------|--------|
| **AUTO** | Score >= threshold, high confidence | Direct SIRET assignment |
| **REVIEW** | Score < threshold | Fallback to Places API |

### Key Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| **AUTO Rate** | `count(AUTO) / total` | >= 75% (cost constraint) |
| **Precision** | `TP / (TP + FP)` among AUTO | >= 95% |
| **FP (False Positive)** | Wrong SIRET in AUTO | Minimize |
| **FN (False Negative)** | Correct SIRET sent to REVIEW | Acceptable (Places fixes) |

### SIRET Normalization

```python
def normalize_siret(siret: str | int | None) -> str | None:
    if siret is None:
        return None
    s = str(siret).strip()
    if not s or s in ('', 'nan', 'None'):
        return None
    return s.zfill(14)  # Left-pad with zeros to 14 chars
```

## 2. Canonical Artefacts

| Artefact | Path | Description |
|----------|------|-------------|
| Meta | `models/xgb_two_stage_meta_20260103_132351.json` | Model metadata |
| Ranker | `models/xgbranker_fast_20260103_132351.json` | Stage 1 ranking |
| Decider | `models/xgb_decider_20260103_132351.json` | Stage 2 classification |
| Calibrator | `models/xgb_decider_calibrator_isotonic_20260103_132351.pkl` | Isotonic calibration |
| Test samples | `data/samples_v4_with_ranker.parquet` (split=test) | 2,193 queries, 111,769 samples |
| Candidates | `data/candidates_v4_active/` | Open establishments only |

## 3. Routing Scenarios (Calibrated Scores)

### 3.1 Canonical (parquet test split)

Evaluated on canonical test split (2,193 queries, Hit@1 = 90.7%):

| Threshold | AUTO% | Precision | FP | FN |
|-----------|-------|-----------|----|----|
| 0.350 | 83.3 | 95.3 | 86 | 248 |
| **0.450** | **75.5** | **95.9** | **68** | **401** |
| 0.550 | 74.8 | 95.9 | 68 | 416 |
| 0.650 | 68.7 | 96.5 | 52 | 534 |
| 0.750 | 63.8 | 97.1 | 40 | 629 |
| 0.850 | 51.6 | 97.7 | 26 | 884 |
| 0.950 | 32.9 | 98.1 | 14 | 1,281 |
| 0.995 | 2.1 | 100.0 | 0 | 1,942 |

**Recommended threshold: 0.45** (75.5% AUTO, 95.9% precision, 68 FP)

### 3.2 End-to-end inference (`reports/xgb_two_stage_topk_test.csv`)

Generated with the IDF fix using `data/old/2026-01-11_splits/test.csv` (2,853 queries).  
Note: 929 queries (32.6%) do not have the GT SIRET in the candidate pool, which caps precision.

**All queries (GT may be missing):**

| Threshold | AUTO% | Precision | FP | FN |
|-----------|-------|-----------|----|----|
| 0.35 | 77.0 | 73.0 | 594 | 106 |
| **0.40** | **75.8** | **73.5** | **573** | **120** |
| 0.45 | 69.6 | 75.4 | 489 | 213 |
| 0.50 | 69.2 | 75.4 | 485 | 222 |
| 0.65 | 63.7 | 76.5 | 427 | 319 |
| 0.75 | 58.2 | 77.0 | 381 | 431 |
| 0.85 | 45.7 | 79.0 | 274 | 679 |
| 0.95 | 30.2 | 81.5 | 160 | 1,007 |

**GT in pool only (1,924 queries):**

| Threshold | AUTO% | Precision | FP | FN |
|-----------|-------|-----------|----|----|
| 0.35 | 91.8 | 90.8 | 163 | 106 |
| 0.40 | 91.0 | 90.8 | 161 | 120 |
| 0.45 | 85.0 | 91.5 | 139 | 213 |
| 0.50 | 84.5 | 91.6 | 137 | 222 |
| **0.65** | **78.4** | **92.2** | **118** | **319** |
| 0.75 | 71.8 | 92.5 | 103 | 431 |
| 0.85 | 56.7 | 94.5 | 60 | 679 |
| 0.95 | 38.1 | 95.8 | 31 | 1,007 |

**Recommended threshold (end-to-end):**
- `0.40` if prioritizing AUTO rate >= 75% on full queries
- `0.65` if evaluating only cases where GT exists in pool

## 4. Inference Pipeline

```
CRM Query
    |
    v
[TF-IDF Prefilter] -> candidate pool (~500)
    |
    v
[XGBoost Ranker] -> Stage 1 top-200
    |
    v
[XGBoost Decider + Isotonic] -> calibrated score
    |
    v
[Routing] score >= 0.45 ? AUTO : REVIEW
```

### IDF Computation (CRITICAL)

The `idf_name` feature requires per-pool IDF computation:

```python
from src.xgb_matcher.candidates import compute_name_idf_map
from src.xgb_matcher.features import set_global_name_idf_map

# For each query's candidate pool:
candidates_dict = {str(c['siret']): c for c in candidates}
idf_map, default_idf = compute_name_idf_map(candidates_dict)
set_global_name_idf_map(idf_map, default_idf)
```

Without this, `idf_name = 0` for all candidates (bug fixed 2026-01-11).

## 5. Feature Importance (Routing-Relevant)

| Feature | Importance | Notes |
|---------|------------|-------|
| `name_jaro_max` | High | Best lexical name match |
| `name_semantic_max` | High | Semantic similarity (gated) |
| `addr_jaro` | Medium | Address similarity |
| `idf_name` | Medium | Name rarity (requires pool IDF) |
| `score_gap` | Medium | Top1 - Top2 score difference |
| `address_density` | Low | Candidates at same address |

## 6. Known Limitations

1. **Closed establishments**: Model trained on open-only; can match closed in inference
2. **CRM-location mismatch**: ~5% queries where GT is in different commune
3. **Equipment entities**: Low SIRENE coverage for street furniture, sensors, etc.

## 7. Scripts

| Script | Purpose |
|--------|---------|
| `scripts/infer_xgb_two_stage.py` | Full inference pipeline |
| `scripts/route_xgb_results.py` | Apply routing thresholds (needs refactor) |
| `scripts/train_xgb_two_stage.py` | Train ranker + decider |

## 8. Validation Checklist

Before deploying routing changes:

- [ ] Reproduce metrics on `data/samples_v4_with_ranker.parquet` (split=test)
- [ ] Verify `idf_name` has variance (mean > 0)
- [ ] Check AUTO rate >= 75%
- [ ] Check precision >= 95%
- [ ] No regression on Hit@1

---

## Appendix: Historical Context

Previous routing attempts:
- `configs/routing_thresholds.yaml`: Segment-based thresholds (broken due to idf=0)
- `models/router_confidence_model.pkl`: Orphan ML router (never integrated)
- Various `reports/routing_*.json`: Inconsistent evaluation universes

All archived to `models/old/` and `data/old/` as of 2026-01-11.
