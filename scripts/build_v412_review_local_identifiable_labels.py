#!/usr/bin/env python3
"""Build the 279-row local-identifiability label view from the canonical labels.

The canonical file is read-only.  Every change must be declared in the quality
overlay, whose ``current_label`` value is checked against the canonical row
before applying it.  Rows depending on external evidence are deliberately
mapped to UNRESOLVED in this local-only training/evaluation view.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_CANONICAL = Path("reports/v412_review_trusted_labels_279.csv")
DEFAULT_OVERLAY = Path("reports/v412_trusted_label_quality_overlay.csv")
DEFAULT_OUTPUT = Path("reports/v412_review_local_identifiable_labels_279.csv")

CANONICAL_COLUMNS = [
    "query_id",
    "label_kind",
    "ground_truth_siret",
    "reliability",
    "error_family",
    "evidence_reference",
    "cohort",
    "source_file",
]
OVERLAY_COLUMNS = [
    "query_id",
    "current_label",
    "recommended_kind",
    "recommended_siret",
    "scope_action",
    "confidence",
    "evidence_summary",
    "evidence_urls",
]
ALLOWED_ACTIONS = {"CORRECT", "EXCLUDE_LOCAL", "QUARANTINE_EXTERNAL"}
ALLOWED_KINDS = {"MATCH_EXACT", "NO_MATCH", "AMBIGUOUS", "UNRESOLVED"}


def _current_label(kind: str, siret: str) -> str:
    return f"{kind}:{siret}" if siret else kind


def build(canonical_path: Path, overlay_path: Path, output_path: Path) -> Path:
    canonical = pd.read_csv(
        canonical_path, dtype=str, keep_default_na=False
    )
    overlay = pd.read_csv(overlay_path, dtype=str, keep_default_na=False)

    if list(canonical.columns) != CANONICAL_COLUMNS:
        raise ValueError(f"Unexpected canonical schema: {list(canonical.columns)}")
    if list(overlay.columns) != OVERLAY_COLUMNS:
        raise ValueError(f"Unexpected overlay schema: {list(overlay.columns)}")
    if len(canonical) != 279 or canonical["query_id"].duplicated().any():
        raise ValueError("Canonical labels must contain 279 unique query_id values")
    if overlay["query_id"].duplicated().any():
        raise ValueError("Quality overlay query_id values must be unique")
    unknown_ids = set(overlay["query_id"]) - set(canonical["query_id"])
    if unknown_ids:
        raise ValueError(f"Overlay contains unknown query_id values: {sorted(unknown_ids)}")
    unknown_actions = set(overlay["scope_action"]) - ALLOWED_ACTIONS
    if unknown_actions:
        raise ValueError(f"Unknown scope actions: {sorted(unknown_actions)}")

    result = canonical.copy()
    row_by_id = {query_id: index for index, query_id in result["query_id"].items()}
    for item in overlay.to_dict("records"):
        index = row_by_id[item["query_id"]]
        canonical_label = _current_label(
            result.at[index, "label_kind"], result.at[index, "ground_truth_siret"]
        )
        if canonical_label != item["current_label"]:
            raise ValueError(
                f"Stale overlay for {item['query_id']}: expected {canonical_label}, "
                f"got {item['current_label']}"
            )

        action = item["scope_action"]
        if action == "QUARANTINE_EXTERNAL":
            new_kind, new_siret = "UNRESOLVED", ""
        else:
            new_kind = item["recommended_kind"]
            new_siret = item["recommended_siret"] if new_kind == "MATCH_EXACT" else ""
        if new_kind not in ALLOWED_KINDS:
            raise ValueError(f"Invalid recommended kind for {item['query_id']}: {new_kind}")
        if new_kind == "MATCH_EXACT" and len(new_siret) != 14:
            raise ValueError(f"MATCH_EXACT requires a 14-digit SIRET for {item['query_id']}")
        if action in {"EXCLUDE_LOCAL", "QUARANTINE_EXTERNAL"} and new_kind == "MATCH_EXACT":
            raise ValueError(f"{action} must leave MATCH_EXACT for {item['query_id']}")

        result.at[index, "label_kind"] = new_kind
        result.at[index, "ground_truth_siret"] = new_siret
        result.at[index, "reliability"] = item["confidence"]
        result.at[index, "error_family"] = f"QUALITY_OVERLAY_{action}"
        result.at[index, "evidence_reference"] = (
            f"{overlay_path.as_posix()}#{item['query_id']}"
        )
        result.at[index, "source_file"] = overlay_path.as_posix()

    counts = result["label_kind"].value_counts().to_dict()
    expected_counts = {"MATCH_EXACT": 241, "AMBIGUOUS": 31, "UNRESOLVED": 7}
    if counts != expected_counts:
        raise ValueError(f"Unexpected derived label counts: {counts} != {expected_counts}")
    if len(result) != 279 or result["query_id"].duplicated().any():
        raise ValueError("Derived labels lost the 279-row one-query-one-label contract")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(build(args.canonical, args.overlay, args.output))
