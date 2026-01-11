# Archived Routing Files (2026-01-11)

Files archived during routing cleanup session.

## Reason for archival

These files were produced before the IDF bug was fixed in `infer_xgb_two_stage.py`.
All had `idf_name = 0` everywhere, making segment-based routing unreliable.

## Contents

### From `models/`
- `router_confidence_model.pkl` - Orphan ML router never integrated into pipeline

### From `configs/`
- `routing_thresholds.yaml` - Segment-based thresholds (broken due to idf=0)

### From `reports/`
- `routing_evaluation_*.json` - Evaluation reports with inconsistent universes
- `routed_*.csv` - Routing results with broken idf_name

## Canonical replacement

See `docs/routing/SSOT.md` for the current source of truth.
