# Documentation Update Summary: data/splits/ Archive

## Date
2026-01-11

## Overview
Updated all documentation files to reflect that `data/splits/` has been archived to `data/old/2026-01-11_splits/` and the canonical source for train/dev/test splits is now `data/samples_v4_with_ranker.parquet` (column `split`).

## Files Updated

### 1. reports/handover.md
**Lines affected:** ~294, ~434-440

**Changes:**
- Session 2026-01-03T17:30:00Z: Added note about data archival
- Session 2026-01-04T11:18:30Z: Updated all command examples referencing `data/splits/test.csv` to use `data/old/2026-01-11_splits/test.csv`
- Added guidance to prefer extracting split from parquet for new workflows

**Context:** Historical session log - updated to reflect backward compatibility paths

---

### 2. reports/entity_matching_audit.md
**Lines affected:** ~271-288, ~354-357, ~450-453, ~507-509, ~920-960, ~1069-1106

**Changes:**
- **Phase 0 diagnostic section (lines 271-288):** Updated input path examples from `data/splits/test.csv` to `data/old/2026-01-11_splits/test.csv` with note about parquet being preferred
- **Phase 1 validation commands (lines 354-357):** Added legacy note with parquet preference
- **Phase 2 validation commands (lines 450-453):** Added legacy note with parquet preference  
- **Phase 3 validation commands (lines 507-509):** Added legacy note with parquet preference
- **Phase 4 commands section (lines 920-960):** Added comprehensive note about archival and parquet canonical source
- **Phase 4 validation commands (lines 1069-1106):** Updated command examples to use actual CRM files instead of splits, added archival note

**Context:** Technical planning document - clarified data source expectations across all phases

---

### 3. docs/XGBOOST_MATCHER_V2.md
**Lines affected:** ~46-50

**Changes:**
- Added `[DEPRECATED 2026-01-11]` tag to CSV splits output
- Added note explaining archival and pointing to parquet canonical source

**Context:** Technical procedure document - marked old outputs as deprecated

---

### 4. prompts/ctx7_consultant_handover.md
**Lines affected:** ~91-96, ~166-187, ~189-224

**Changes:**
- **Data section (lines 91-96):** Completely rewrote to show parquet as canonical, CSV splits as archived, with important note
- **CSV vs parquet confusion section (lines 166-187):** Changed title to "RÉSOLU", added resolution date and migration guidance
- **Commands section (lines 189-224):** Updated all command examples to use archived paths with note preferring parquet extraction

**Context:** Consultant handover document - most critical for preventing context rot

---

## Key Messages Added

### For Legacy Compatibility
- Commands using `data/splits/test.csv` should now use `data/old/2026-01-11_splits/test.csv`
- This maintains backward compatibility for existing scripts

### For New Workflows
- Always use `data/samples_v4_with_ranker.parquet` as the canonical source
- Filter by `split == 'test'` or `split == 'train'` etc. to get specific splits
- Reconstruct CRM CSV from parquet if needed for scripts requiring CSV input

### Migration Path
```python
# Extract test split from canonical parquet
import pandas as pd
df = pd.read_parquet('data/samples_v4_with_ranker.parquet')
test_df = df[df['split'] == 'test']
# Reconstruct CRM columns if needed for inference scripts
```

## Consistency Notes

All documentation now consistently:
1. Points to `data/samples_v4_with_ranker.parquet` as the canonical source
2. Provides `data/old/2026-01-11_splits/` as the legacy/backward-compat path
3. Recommends parquet-first workflows for new development
4. Maintains clarity that model metrics in `models/xgb_two_stage_meta_20260103_132351.json` are based on the parquet splits

## Files NOT Changed

- Model metadata files (already reference parquet correctly)
- Python scripts (will be updated separately if needed)
- Configuration files (no split references)

