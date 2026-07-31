#!/usr/bin/env python3
"""Build the canonical 279-query trusted REVIEW label overlay."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_ASSIGNMENTS = Path(
    "/Volumes/CATNAT_DATA/SIRETO_RECALL100/datasets/"
    "v4_11_input_blind/ec4326ec57e4411d/split_assignments.parquet"
)
DEFAULT_OUTPUT = Path("reports/v412_review_trusted_labels_279.csv")


def _part(
    path: Path,
    cohort: str,
    *,
    label_column: str,
    siret_column: str,
    reliability_column: str,
    error_column: str | None = None,
    evidence_column: str | None = None,
) -> pd.DataFrame:
    source = pd.read_csv(path, dtype=str).fillna("")
    output = pd.DataFrame(
        {
            "query_id": source["query_id"].astype(str),
            "label_kind": source[label_column],
            "ground_truth_siret": source[siret_column],
            "reliability": source[reliability_column],
            "error_family": source[error_column] if error_column else "ADJUDICATED_REVIEW",
            "evidence_reference": (
                source[evidence_column] if evidence_column else str(path)
            ),
            "cohort": cohort,
            "source_file": str(path),
        }
    )
    return output


def build(assignments_path: Path) -> pd.DataFrame:
    parts = [
        _part(
            Path("reports/v412_review_adjudication_labels.csv"),
            "R30",
            label_column="label",
            siret_column="validated_siret",
            reliability_column="reliability",
        ),
        _part(
            Path("reports/v412_review_rerank_counteraudit_53.csv"),
            "R53",
            label_column="label",
            siret_column="adjudication",
            reliability_column="confidence",
            error_column="error_family",
        ),
        _part(
            Path("reports/v412_ranker_independent_validation_labels.csv"),
            "R7",
            label_column="label",
            siret_column="validated_siret",
            reliability_column="confidence",
            error_column="error_family",
        ),
        _part(
            Path("reports/v412_corrected_review_overlay_60.csv"),
            "OVERLAY60",
            label_column="label_kind",
            siret_column="ground_truth_siret",
            reliability_column="reliability",
            error_column="error_family",
            evidence_column="evidence_reference",
        ),
    ]
    for cohort, filename in (
        ("ACCEPTOR30", "v412_corrected_acceptor_independent_labels_30.csv"),
        ("BLIND_B1", "v412_clean_target_independent_labels_30.csv"),
        ("BLIND_B2", "v412_clean_target_independent_b2_labels_30.csv"),
        ("BLIND_FINAL39", "v412_clean_target_independent_final39_labels.csv"),
    ):
        parts.append(
            _part(
                Path("reports") / filename,
                cohort,
                label_column="label_kind",
                siret_column="ground_truth_siret",
                reliability_column="reliability",
                error_column="error_family",
                evidence_column="evidence_url",
            )
        )

    output = pd.concat(parts, ignore_index=True)
    if len(output) != 279 or output["query_id"].duplicated().any():
        raise ValueError("Trusted overlay must contain 279 unique queries")
    if output["label_kind"].value_counts().to_dict() != {
        "MATCH_EXACT": 254,
        "AMBIGUOUS": 25,
    }:
        raise ValueError("Trusted label counts changed")
    if output["reliability"].value_counts().to_dict() != {
        "HIGH": 257,
        "MEDIUM": 22,
    }:
        raise ValueError("Trusted reliability counts changed")
    exact = output["label_kind"].eq("MATCH_EXACT")
    if not output.loc[exact, "ground_truth_siret"].str.fullmatch(r"\d{14}").all():
        raise ValueError("Every exact label must contain a 14-digit SIRET")
    if not output.loc[~exact, "ground_truth_siret"].eq("").all():
        raise ValueError("Ambiguous labels must not contain a SIRET")
    assignments = pd.read_parquet(assignments_path)
    assignments["query_id"] = assignments["query_id"].astype(str)
    joined = output[["query_id"]].merge(
        assignments[["query_id", "split", "oof_fold"]],
        on="query_id",
        how="left",
        validate="one_to_one",
    )
    if joined["split"].value_counts(dropna=False).to_dict() != {"dev": 279}:
        raise ValueError("Every trusted REVIEW label must belong to dev")
    output = output.sort_values("query_id", kind="mergesort").reset_index(drop=True)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    frame = build(args.assignments)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(args.output)
