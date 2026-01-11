#!/usr/bin/env python3
"""
Extract high-confidence silver labels from Places observability logs.

Input: JSON array from --log-places (route_xgb_results.py)
Output: CSV with crm_id, siret, confidence, source
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict

import pandas as pd


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build silver labels from Places logs.")
    p.add_argument("--input-path", type=Path, required=True, help="JSON file from --log-places")
    p.add_argument("--output-path", type=Path, required=True, help="Output CSV for silver labels")
    p.add_argument("--min-score-after", type=float, default=0.97)
    p.add_argument("--min-gap-after", type=float, default=0.03)
    p.add_argument("--min-name-sim", type=float, default=0.70)
    p.add_argument("--min-addr-overlap", type=float, default=0.70)
    p.add_argument("--max-distance-m", type=float, default=200.0)
    p.add_argument("--require-gate", action="store_true", default=True)
    return p.parse_args()


def _is_silver(rec: Dict, args: argparse.Namespace) -> bool:
    if rec.get("places_decision") != "MATCH_PLACES":
        return False
    if args.require_gate and rec.get("places_gate_valid") is False:
        return False
    if rec.get("score_after") is not None and float(rec.get("score_after")) < args.min_score_after:
        return False
    if rec.get("gap_after") is not None and float(rec.get("gap_after")) < args.min_gap_after:
        return False
    if rec.get("validation_name_sim") is not None and float(rec.get("validation_name_sim")) < args.min_name_sim:
        return False
    if rec.get("validation_addr_overlap") is not None and float(rec.get("validation_addr_overlap")) < args.min_addr_overlap:
        return False
    if rec.get("validation_distance_m") is not None and float(rec.get("validation_distance_m")) > args.max_distance_m:
        return False
    return True


def main() -> None:
    args = _parse_args()
    with open(args.input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        records = [data]
    else:
        records = data

    rows: List[Dict] = []
    for rec in records:
        if not _is_silver(rec, args):
            continue
        siret = rec.get("chosen_siret")
        crm_id = rec.get("crm_id")
        if not siret or crm_id is None:
            continue
        rows.append(
            {
                "crm_id": crm_id,
                "siret": siret,
                "source": "places_silver",
                "confidence": rec.get("score_after"),
                "gap_after": rec.get("gap_after"),
                "name_sim": rec.get("validation_name_sim"),
                "addr_overlap": rec.get("validation_addr_overlap"),
                "distance_m": rec.get("validation_distance_m"),
            }
        )

    out_df = pd.DataFrame(rows)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_path, index=False)
    print(f"Saved {len(out_df)} silver labels to {args.output_path}")


if __name__ == "__main__":
    main()

