#!/usr/bin/env python3
"""Open the V4-Fresh holdout once and prepare immutable evaluation inputs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_v9_retrieval_experiment import _git_commit  # noqa: E402
from src.xgb_matcher.v9_dataset import LABEL_COLUMNS, file_sha256  # noqa: E402


SCHEMA_VERSION = "sireto-v4-final-holdout-input-1"
EXPECTED_COUNTS = {
    "MATCH_EXACT": 302,
    "AMBIGUOUS": 52,
    "UNRESOLVED": 991,
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_authorization(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("purpose") != "v4_final_holdout_once":
        raise ValueError("Invalid final holdout purpose")
    if payload.get("status") != "FROZEN_AUTHORIZED":
        raise ValueError("Final holdout is not authorized")
    if payload.get("git_commit") != _git_commit():
        raise ValueError("Code commit differs from frozen authorization")
    if payload.get("expected_counts") != EXPECTED_COUNTS:
        raise ValueError("Authorized holdout counts differ from contract")
    for name, frozen_path in payload["frozen_paths"].items():
        observed = file_sha256(Path(frozen_path))
        expected = payload["frozen_file_hashes"][name]
        if observed != expected:
            raise ValueError(f"Frozen artifact changed: {name}")
    return payload


def prepare_evaluation_rows(
    benchmark: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    counts = benchmark["label_kind"].value_counts().to_dict()
    if counts != EXPECTED_COUNTS:
        raise ValueError(
            f"Final holdout counts differ from contract: {counts}"
        )
    evaluation = benchmark[
        benchmark["label_kind"].isin(["MATCH_EXACT", "AMBIGUOUS"])
    ].copy()
    evaluation["query_id"] = evaluation["query_id"].astype(str)
    evaluation["evaluation_label_kind"] = evaluation["label_kind"]
    evaluation["evaluation_ground_truth_siret"] = evaluation[
        "ground_truth_siret"
    ]
    evaluation["evaluation_ground_truth_siren"] = evaluation[
        "ground_truth_siren"
    ]
    ambiguous = evaluation["label_kind"].eq("AMBIGUOUS")
    direct_lists = evaluation.loc[
        ambiguous, "direct_active_sirets_json"
    ].map(json.loads)
    if not direct_lists.map(len).ge(2).all():
        raise ValueError("Final ambiguous rows require at least two SIRETs")
    evaluation.loc[
        ambiguous, "diagnostic_probe_siret"
    ] = direct_lists.map(lambda values: str(values[0]).zfill(14))
    evaluation.loc[
        ambiguous, "ground_truth_siret"
    ] = evaluation.loc[ambiguous, "diagnostic_probe_siret"]
    evaluation.loc[
        ambiguous, "ground_truth_siren"
    ] = evaluation.loc[ambiguous, "diagnostic_probe_siret"].str[:9]
    evaluation["retrieval_uses_diagnostic_probe"] = False
    evaluation["split"] = "final_holdout"

    snapshot_id = str(benchmark["sirene_snapshot_id"].dropna().iloc[0])
    canonical_labels = pd.DataFrame(
        {
            "query_id": evaluation["query_id"],
            "label_kind": evaluation["evaluation_label_kind"],
            "ground_truth_siret": evaluation[
                "evaluation_ground_truth_siret"
            ],
            "ground_truth_siren": evaluation[
                "evaluation_ground_truth_siren"
            ],
            "label_source": "qualification_v4_current_snapshot",
            "validator": evaluation["validator"].fillna(""),
            "validated_at": "",
            "sirene_snapshot_id": snapshot_id,
            "split": "test",
        }
    )[LABEL_COLUMNS]
    return evaluation, canonical_labels, {
        key: int(value) for key, value in counts.items()
    }


def prepare(
    *,
    authorization_path: Path,
    output_root: Path,
) -> Path:
    authorization = validate_authorization(authorization_path)
    run_id = authorization["authorization_id"]
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    opened_path = output_dir / "HOLDOUT_OPENED.json"
    opened_path.write_text(
        json.dumps(
            {
                "authorization_id": run_id,
                "opened_at": datetime.now(timezone.utc).isoformat(),
                "git_commit": _git_commit(),
                "irreversible": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    fresh_dir = Path(authorization["paths"]["fresh_dir"])
    benchmark_path = fresh_dir / "holdout_sealed/benchmark.parquet"
    labels_path = fresh_dir / "holdout_sealed/labels.parquet"
    for name, path in (
        ("benchmark.parquet", benchmark_path),
        ("labels.parquet", labels_path),
    ):
        if (
            file_sha256(path)
            != authorization["holdout_declared_hashes"][name]
        ):
            raise ValueError(f"Holdout hash mismatch: {name}")
    benchmark = pd.read_parquet(benchmark_path)
    source_labels = pd.read_parquet(labels_path)
    if not benchmark["query_id"].astype(str).equals(
        source_labels["query_id"].astype(str)
    ):
        raise ValueError("Holdout benchmark and labels are misaligned")
    for column in ("label_kind", "ground_truth_siret", "ground_truth_siren"):
        left = benchmark[column].fillna("").astype(str)
        right = source_labels[column].fillna("").astype(str)
        if not left.equals(right):
            raise ValueError(f"Holdout label mismatch: {column}")

    evaluation, labels, counts = prepare_evaluation_rows(benchmark)
    acceptor_dataset_dir = Path(
        authorization["paths"]["acceptor_dataset_dir"]
    )
    known_labels = pd.read_parquet(
        acceptor_dataset_dir / "labels.parquet",
        columns=["ground_truth_siren", "label_kind"],
    )
    known_sirens = set(
        known_labels.loc[
            known_labels["label_kind"].eq("MATCH_EXACT"),
            "ground_truth_siren",
        ]
        .dropna()
        .astype(str)
    )
    holdout_sirens = set(
        labels.loc[
            labels["label_kind"].eq("MATCH_EXACT"),
            "ground_truth_siren",
        ]
        .dropna()
        .astype(str)
    )
    overlap = sorted(known_sirens & holdout_sirens)
    if overlap:
        raise ValueError(
            f"Final holdout shares exact SIRENs with fit/dev: {overlap[:5]}"
        )

    benchmark_output = output_dir / "evaluation_benchmark.parquet"
    labels_output = output_dir / "labels.parquet"
    evaluation.to_parquet(benchmark_output, index=False)
    labels.to_parquet(labels_output, index=False)
    benchmark_hash = file_sha256(benchmark_output)
    common = {
        "schema_version": SCHEMA_VERSION,
        "build_id": run_id,
        "benchmark_build_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "split": "final_holdout",
        "partitions_sha256": None,
        "output_sha256": {
            benchmark_output.name: benchmark_hash,
        },
    }
    for name, partitions_hash in (
        (
            "manifest_v7.json",
            authorization["active_partitions_sha256"],
        ),
        (
            "manifest_overlay.json",
            authorization["overlay_partitions_sha256"],
        ),
    ):
        (output_dir / name).write_text(
            json.dumps(
                {**common, "partitions_sha256": partitions_hash},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "build_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "authorization_sha256": file_sha256(authorization_path),
        "holdout_opened": True,
        "old_test_read": False,
        "positive_injection": False,
        "diagnostic_probe_affects_retrieval": False,
        "source_query_count": int(len(benchmark)),
        "evaluation_query_count": int(len(evaluation)),
        "source_label_counts": counts,
        "exact_siren_overlap_with_fit_dev": 0,
        "outputs": {
            path.name: file_sha256(path)
            for path in (
                opened_path,
                benchmark_output,
                labels_output,
                output_dir / "manifest_v7.json",
                output_dir / "manifest_overlay.json",
            )
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(
        prepare(
            authorization_path=args.authorization,
            output_root=args.output_root,
        )
    )


if __name__ == "__main__":
    main()
