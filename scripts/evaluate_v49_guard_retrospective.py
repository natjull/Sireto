#!/usr/bin/env python3
"""Evaluate the frozen V4.9 site-function guard on the consumed V4.7 labels."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.v49_site_function import SiteFunctionTaxonomy  # noqa: E402
from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


SCHEMA_VERSION = "sireto-v4.9-site-function-retrospective-1"
CANONICAL_OUTPUT_ROOT = Path(
    "/Volumes/CATNAT_DATA/SIRETO_RECALL100/experiments/"
    "v4_9_site_function_retrospective"
)
EXPECTED_HASHES = {
    "taxonomy": "48bbb7e1795a0731f1f12df41aeb971667c10d03c879bf06d5ba15b65f8b121d",
    "guard_code": "8463086d2ce404e5c83140df8ea7351cfb363793edfa7e74db95fe202d9c54e2",
    "current_labels": "e5e592d4dcd540273378dada7128f957b1d335df63fbc88f4c1377c0f9337bd2",
    "crm_queue": "47af4887769a2edb11f1e629c38077edccd035dd96cb3a6d39620714361fdecc",
    "sirene_snapshot": "c91180cc5bae86948dd57d752c9bae45e58cc64653e99d5a9357664b67300845",
}
RELIABLE_LABELS = {"TOP1_CORRECT", "TOP1_WRONG", "AMBIGUOUS"}
NEGATIVE_LABELS = {"TOP1_WRONG", "AMBIGUOUS"}
V48_RANDOM_ERROR_IDS = {
    "008373b595622d22",
    "00ebcafaaa0a8bf5",
    "01d50f2a608bb3bb",
}


def _json_dump(path: Path, payload: Any) -> None:
    encoded = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _validate_hashes(paths: dict[str, Path]) -> dict[str, str]:
    actual = {name: file_sha256(path) for name, path in paths.items()}
    mismatches = {
        name: {"expected": EXPECTED_HASHES[name], "actual": digest}
        for name, digest in actual.items()
        if digest != EXPECTED_HASHES[name]
    }
    if mismatches:
        raise ValueError(f"V4.9 retrospective input mismatch: {mismatches}")
    return actual


def load_sirene_top1(snapshot_path: Path, sirets: set[str]) -> pd.DataFrame:
    columns = [
        "siret",
        "enseigne1Etablissement",
        "enseigne2Etablissement",
        "enseigne3Etablissement",
        "denominationUsuelleEtablissement",
        "activitePrincipaleEtablissement",
    ]
    table = ds.dataset(snapshot_path, format="parquet").to_table(
        columns=columns,
        filter=ds.field("siret").isin(sorted(sirets)),
    )
    frame = table.to_pandas()
    frame["siret"] = frame["siret"].astype(str)
    if frame["siret"].duplicated().any():
        raise ValueError("SIRENE snapshot contains duplicate requested SIRETs")
    missing = sirets - set(frame["siret"])
    if missing:
        raise ValueError(f"SIRENE snapshot misses {len(missing)} top1 SIRETs")
    return frame


def build_predictions(
    labels: pd.DataFrame,
    crm_queue: pd.DataFrame,
    sirene: pd.DataFrame,
    taxonomy: SiteFunctionTaxonomy,
) -> pd.DataFrame:
    if len(labels) != 172 or labels["audit_case_id"].astype(str).duplicated().any():
        raise ValueError("V4.9 expects 172 unique V4.7 current-label cases")
    crm = crm_queue[["audit_case_id", "SITE"]].copy()
    crm["audit_case_id"] = crm["audit_case_id"].astype(str)
    if crm["audit_case_id"].duplicated().any():
        raise ValueError("CRM queue contains duplicate audit_case_id")
    merged = labels.copy()
    merged["audit_case_id"] = merged["audit_case_id"].astype(str)
    merged["current_top1_siret"] = merged["current_top1_siret"].astype(str)
    merged = merged.merge(crm, on="audit_case_id", how="left", validate="one_to_one")
    if merged["SITE"].isna().any():
        raise ValueError("CRM queue does not cover every V4.7 current-label case")
    sirene = sirene.rename(columns={"siret": "current_top1_siret"})
    merged = merged.merge(
        sirene,
        on="current_top1_siret",
        how="left",
        validate="many_to_one",
    )
    if merged["activitePrincipaleEtablissement"].isna().any():
        raise ValueError("SIRENE names/activity missing for at least one top1")

    output_rows: list[dict[str, Any]] = []
    candidate_columns = [
        "enseigne1Etablissement",
        "enseigne2Etablissement",
        "enseigne3Etablissement",
        "denominationUsuelleEtablissement",
    ]
    for row in merged.to_dict("records"):
        crm_detection = taxonomy.detect([row["SITE"]])
        candidate_detection = taxonomy.detect(
            [row[column] for column in candidate_columns],
            activity_code=row["activitePrincipaleEtablissement"],
        )
        decision = taxonomy.guard(crm_detection, candidate_detection)
        label = str(row["current_adjudication_label"])
        output_rows.append(
            {
                "audit_case_id": row["audit_case_id"],
                "query_id": str(row["query_id"]),
                "service_id": str(row["service_id"]),
                "sampling_stratum": str(row["sampling_stratum"]),
                "current_label_origin": str(row["current_label_origin"]),
                "current_adjudication_label": label,
                "reliable_label": label in RELIABLE_LABELS,
                "is_negative_or_ambiguous": label in NEGATIVE_LABELS,
                "is_v48_random_error": row["audit_case_id"]
                in V48_RANDOM_ERROR_IDS,
                "current_top1_siret": row["current_top1_siret"],
                "crm_site": row["SITE"],
                "candidate_name": " | ".join(
                    str(row[column])
                    for column in candidate_columns
                    if pd.notna(row[column]) and str(row[column]).strip()
                ),
                "candidate_activity_code": str(
                    row["activitePrincipaleEtablissement"]
                ),
                "crm_roles_json": json.dumps(
                    crm_detection.roles, ensure_ascii=False
                ),
                "candidate_roles_json": json.dumps(
                    candidate_detection.roles, ensure_ascii=False
                ),
                "crm_pattern_hits_json": json.dumps(
                    crm_detection.matched_patterns, ensure_ascii=False
                ),
                "candidate_pattern_hits_json": json.dumps(
                    candidate_detection.matched_patterns, ensure_ascii=False
                ),
                "candidate_activity_hits_json": json.dumps(
                    candidate_detection.matched_activity_codes,
                    ensure_ascii=False,
                ),
                "guard_review": decision.review,
                "guard_reason": decision.reason,
            }
        )
    return pd.DataFrame(output_rows).sort_values("audit_case_id").reset_index(drop=True)


def summarize(predictions: pd.DataFrame) -> dict[str, Any]:
    reliable = predictions[predictions["reliable_label"]].copy()
    correct = reliable["current_adjudication_label"].eq("TOP1_CORRECT")
    negative = reliable["is_negative_or_ambiguous"]
    rejected = reliable["guard_review"]
    correct_total = int(correct.sum())
    correct_rejected = int((correct & rejected).sum())
    negative_rejected = int((negative & rejected).sum())
    random_errors_rejected = int(
        (reliable["is_v48_random_error"] & rejected).sum()
    )
    correct_rejection_rate = (
        correct_rejected / correct_total if correct_total else None
    )
    checks = {
        "at_least_five_negative_or_ambiguous_rejected": negative_rejected >= 5,
        "at_least_one_v48_random_error_rejected": random_errors_rejected >= 1,
        "at_most_five_percent_correct_rejected": (
            correct_rejection_rate is not None and correct_rejection_rate <= 0.05
        ),
        "no_case_specific_rules": True,
    }

    def grouped(columns: list[str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for keys, group in reliable.groupby(columns, dropna=False):
            values = keys if isinstance(keys, tuple) else (keys,)
            item = dict(zip(columns, values))
            item.update(
                {
                    "count": int(len(group)),
                    "correct": int(
                        group["current_adjudication_label"].eq("TOP1_CORRECT").sum()
                    ),
                    "negative_or_ambiguous": int(
                        group["is_negative_or_ambiguous"].sum()
                    ),
                    "guard_review": int(group["guard_review"].sum()),
                }
            )
            rows.append(item)
        return rows

    passed = all(checks.values())
    return {
        "verdict": "GO_FRESH_V49" if passed else "STOP_SITE_FUNCTION_GUARD",
        "passed": passed,
        "checks": checks,
        "counts": {
            "all_rows": int(len(predictions)),
            "reliable_rows": int(len(reliable)),
            "unresolved_rows": int((~predictions["reliable_label"]).sum()),
            "correct_total": correct_total,
            "negative_or_ambiguous_total": int(negative.sum()),
            "correct_rejected": correct_rejected,
            "negative_or_ambiguous_rejected": negative_rejected,
            "v48_random_errors_rejected": random_errors_rejected,
            "all_reliable_rejected": int(rejected.sum()),
        },
        "rates": {
            "correct_rejection_rate": correct_rejection_rate,
            "reliable_rejection_rate": float(rejected.mean()),
        },
        "by_origin": grouped(["current_label_origin"]),
        "by_sampling_stratum": grouped(["sampling_stratum"]),
        "by_origin_and_label": grouped(
            ["current_label_origin", "current_adjudication_label"]
        ),
    }


def evaluate(
    *,
    taxonomy_path: Path,
    guard_code_path: Path,
    current_labels_path: Path,
    crm_queue_path: Path,
    sirene_snapshot_path: Path,
    output_root: Path,
) -> Path:
    paths = {
        "taxonomy": Path(taxonomy_path).resolve(),
        "guard_code": Path(guard_code_path).resolve(),
        "current_labels": Path(current_labels_path).resolve(),
        "crm_queue": Path(crm_queue_path).resolve(),
        "sirene_snapshot": Path(sirene_snapshot_path).resolve(),
    }
    hashes = _validate_hashes(paths)
    artifact_id = hashlib.sha256(
        json.dumps(
            {"schema_version": SCHEMA_VERSION, "inputs": hashes},
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    output_dir = Path(output_root).resolve() / artifact_id
    output_dir.mkdir(parents=True, exist_ok=False)

    try:
        labels = pd.read_parquet(paths["current_labels"])
        crm_queue = pd.read_parquet(paths["crm_queue"])
        sirets = set(labels["current_top1_siret"].astype(str))
        sirene = load_sirene_top1(paths["sirene_snapshot"], sirets)
        taxonomy = SiteFunctionTaxonomy.load(paths["taxonomy"])
        predictions = build_predictions(labels, crm_queue, sirene, taxonomy)
        report = summarize(predictions)
        predictions_path = output_dir / "retrospective_predictions.parquet"
        predictions.to_parquet(predictions_path, index=False)
        report_path = output_dir / "retrospective_report.json"
        _json_dump(report_path, report)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifact_id": artifact_id,
            "inputs": {
                name: {"path": str(paths[name]), "sha256": digest}
                for name, digest in hashes.items()
            },
            "outputs": {
                "predictions": {
                    "path": str(predictions_path),
                    "sha256": file_sha256(predictions_path),
                },
                "report": {
                    "path": str(report_path),
                    "sha256": file_sha256(report_path),
                },
            },
            "invariants": {
                "taxonomy_frozen_before_measurement": True,
                "retrospective_only": True,
                "fresh_population_opened": False,
                "test_final_opened": False,
                "model_retrained": False,
                "threshold_changed": False,
            },
        }
        _json_dump(output_dir / "manifest.json", manifest)
    except Exception:
        output_dir.rmdir()
        raise
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=Path("config/v4_9_site_function_taxonomy.json"),
    )
    parser.add_argument(
        "--guard-code",
        type=Path,
        default=Path("src/xgb_matcher/v49_site_function.py"),
    )
    parser.add_argument(
        "--current-labels",
        type=Path,
        default=Path(
            "/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/"
            "v4_7_current_adjudications/4cc5420fb5da0683/current_labels.parquet"
        ),
    )
    parser.add_argument(
        "--crm-queue",
        type=Path,
        default=Path(
            "/Volumes/CATNAT_DATA/SIRETO_RECALL100/audits/"
            "v4_3_hard_labels/0f832305ab199267/hard_label_queue.parquet"
        ),
    )
    parser.add_argument(
        "--sirene-snapshot",
        type=Path,
        default=Path("data/StockEtablissement_utf8.parquet"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=CANONICAL_OUTPUT_ROOT,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = evaluate(
        taxonomy_path=args.taxonomy,
        guard_code_path=args.guard_code,
        current_labels_path=args.current_labels,
        crm_queue_path=args.crm_queue,
        sirene_snapshot_path=args.sirene_snapshot,
        output_root=args.output_root,
    )
    print(output_dir)


if __name__ == "__main__":
    main()
