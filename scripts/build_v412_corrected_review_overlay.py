#!/usr/bin/env python3
"""Build the 60-case corrected REVIEW label overlay from audited reports."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


BASE = Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100")
DEFAULT_DATASET = BASE / "datasets/v4_1/f938abf6b8a87155"
DEFAULT_OUTPUT = Path("reports/v412_corrected_review_overlay_60.csv")


def _normalise_siret(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith(".0"):
        text = text[:-2]
    if not text.isdigit() or len(text) > 14:
        raise ValueError(f"Invalid SIRET: {value!r}")
    return text.zfill(14)


def _load_audit(path: Path, evidence_reference: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str).fillna("")
    required = {
        "query_id",
        "label",
        "validated_siret",
        "reliability",
        "legacy_label",
        "pipeline_predicted_siret",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    output = pd.DataFrame(
        {
            "query_id": frame["query_id"].astype(str),
            "label_kind": frame["label"].astype(str),
            "ground_truth_siret": frame["validated_siret"].map(_normalise_siret),
            "reliability": frame["reliability"].astype(str),
            "reported_previous_label_kind": frame["legacy_label"].astype(str),
            "pipeline_predicted_siret": frame["pipeline_predicted_siret"].map(
                _normalise_siret
            ),
            "error_family": frame.get(
                "error_family", frame.get("pipeline_error", "")
            ).astype(str),
            "evidence_reference": frame.get(
                "evidence_url", pd.Series([evidence_reference] * len(frame))
            ).replace("", evidence_reference),
            "source_audit": str(path),
        }
    )
    return output


def build_overlay(
    audit_paths: list[tuple[Path, str]],
    dataset: Path,
) -> pd.DataFrame:
    overlay = pd.concat(
        [_load_audit(path, reference) for path, reference in audit_paths],
        ignore_index=True,
    )
    if len(overlay) != 60 or overlay["query_id"].duplicated().any():
        raise ValueError("The corrected overlay must contain 60 unique queries")
    counts = overlay["label_kind"].value_counts().to_dict()
    if counts != {"MATCH_EXACT": 56, "AMBIGUOUS": 4}:
        raise ValueError(f"Unexpected adjudicated label counts: {counts}")
    exact = overlay["label_kind"].eq("MATCH_EXACT")
    if overlay.loc[exact, "ground_truth_siret"].isna().any():
        raise ValueError("Every exact label requires a SIRET")
    if overlay.loc[~exact, "ground_truth_siret"].notna().any():
        raise ValueError("Ambiguous labels cannot carry a SIRET")
    if not overlay["reliability"].isin(["HIGH", "MEDIUM"]).all():
        raise ValueError("Every corrected label requires explicit reliability")

    queries = pd.read_parquet(dataset / "queries.parquet")
    labels = pd.read_parquet(dataset / "labels.parquet")
    for frame in (queries, labels):
        frame["query_id"] = frame["query_id"].astype(str)
    context = queries[["query_id", "input_siret"]].merge(
        labels[
            ["query_id", "label_kind", "ground_truth_siret"]
        ].rename(
            columns={
                "label_kind": "previous_label_kind",
                "ground_truth_siret": "previous_ground_truth_siret",
            }
        ),
        on="query_id",
        validate="one_to_one",
    )
    output = overlay.merge(context, on="query_id", how="left", validate="one_to_one")
    if output["previous_label_kind"].isna().any():
        raise ValueError("Every corrected query must exist in the V4.11 dataset")
    output["input_siret"] = output["input_siret"].map(_normalise_siret)
    output["previous_ground_truth_siret"] = output[
        "previous_ground_truth_siret"
    ].map(_normalise_siret)
    output["ground_truth_siren"] = output["ground_truth_siret"].map(
        lambda value: value[:9] if value else None
    )
    output["truth_equals_input_siret"] = (
        exact & output["ground_truth_siret"].eq(output["input_siret"])
    )
    output["ranker_top1_correct"] = (
        exact
        & output["ground_truth_siret"].eq(output["pipeline_predicted_siret"])
    )
    output["label_changed"] = (
        output["label_kind"].ne(output["previous_label_kind"])
        | output["ground_truth_siret"].fillna("").ne(
            output["previous_ground_truth_siret"].fillna("")
        )
    )
    columns = [
        "query_id",
        "label_kind",
        "ground_truth_siret",
        "ground_truth_siren",
        "reliability",
        "previous_label_kind",
        "previous_ground_truth_siret",
        "input_siret",
        "pipeline_predicted_siret",
        "truth_equals_input_siret",
        "ranker_top1_correct",
        "label_changed",
        "error_family",
        "evidence_reference",
        "source_audit",
    ]
    return output[columns].sort_values("query_id", key=lambda x: x.astype(int))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audits = [
        (
            Path("reports/v412_remaining_review_audit_first5.csv"),
            "reports/v412_remaining_review_audit_first5.md",
        ),
        (
            Path("reports/v412_remaining_review_audit_next25.csv"),
            "reports/v412_remaining_review_audit_30.md",
        ),
        (
            Path("reports/v412_remaining_review_audit_batch2.csv"),
            "reports/v412_remaining_review_audit_batch2.md",
        ),
    ]
    output = build_overlay(audits, args.dataset.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(args.output)


if __name__ == "__main__":
    main()
