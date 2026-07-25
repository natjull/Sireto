#!/usr/bin/env python3
"""Certify the single selective Recall@100 test run against frozen gates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import sys
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_v9_retrieval_experiment import (  # noqa: E402
    _binary_metric,
    _git_commit,
)
from src.xgb_matcher.contracts import GroundTruthKind  # noqa: E402
from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


SCHEMA_VERSION = "sireto-selective-retrieval-certification-1"
MIN_COVERAGE = 0.80
MIN_RECALL = 0.99
MAX_CANDIDATES = 100
SEGMENT_REFERENCES = {
    "active": {"coverage": 0.8498556304, "recall": 0.9988674972},
    "closed": {"coverage": 0.6940451745, "recall": 0.9792899408},
    "mega": {"coverage": 0.8727272727, "recall": 0.9861111111},
    "multi_site": {"coverage": 0.7947882736, "recall": 0.9938524590},
    "insee": {"coverage": 0.8218527316, "recall": 0.9956647399},
}


def _metric_set(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "admission_at_100": _binary_metric(frame["hit_at_100"]),
        "frozen_sparse_at_100": _binary_metric(
            frame["baseline_hit_at_100"]
        ),
        "internal_oracle": _binary_metric(frame["oracle_hit"]),
    }


def certify(
    qualified: pd.DataFrame,
    admission: pd.DataFrame,
) -> dict[str, Any]:
    required_qualified = {
        "query_id",
        "label_kind",
        "v2_label_kind",
        "ground_truth_state",
    }
    required_admission = {
        "query_id",
        "hit_at_100",
        "baseline_hit_at_100",
        "oracle_hit",
        "candidate_count",
        "mega_base_pool",
        "multi_site_siren",
        "location_match_type",
    }
    if required_qualified - set(qualified.columns):
        raise ValueError("Qualified benchmark is missing certification columns")
    if required_admission - set(admission.columns):
        raise ValueError("Admission artifact is missing certification columns")

    labels = qualified.copy()
    raw = admission.copy()
    labels["query_id"] = labels["query_id"].astype(str)
    raw["query_id"] = raw["query_id"].astype(str)
    if labels["query_id"].duplicated().any() or raw["query_id"].duplicated().any():
        raise ValueError("query_id must be unique")
    if set(labels["query_id"]) != set(raw["query_id"]):
        raise ValueError("Qualification and admission query IDs differ")

    raw = raw.drop(columns=["ground_truth_state"], errors="ignore")
    merged = raw.merge(
        labels[
            [
                "query_id",
                "label_kind",
                "v2_label_kind",
                "ground_truth_state",
            ]
        ],
        on="query_id",
        validate="one_to_one",
    )
    v2_exact = merged[
        merged["v2_label_kind"].eq(GroundTruthKind.MATCH_EXACT.value)
    ]
    v3_exact = merged[
        merged["label_kind"].eq(GroundTruthKind.MATCH_EXACT.value)
    ]
    coverage = len(v3_exact) / len(merged) if len(merged) else 0.0
    recall = (
        float(v3_exact["hit_at_100"].astype(bool).mean())
        if len(v3_exact)
        else 0.0
    )
    unseen = int((~v3_exact["oracle_hit"].astype(bool)).sum())
    over_budget = int(
        merged["candidate_count"].astype(int).gt(MAX_CANDIDATES).sum()
    )

    gates: dict[str, Any] = {
        "coverage": {
            "observed": coverage,
            "minimum": MIN_COVERAGE,
            "passed": coverage >= MIN_COVERAGE,
        },
        "recall_at_100": {
            "observed": recall,
            "minimum": MIN_RECALL,
            "passed": recall >= MIN_RECALL,
        },
        "internal_oracle_unseen": {
            "observed": unseen,
            "maximum": 0,
            "passed": unseen == 0,
        },
        "candidate_ceiling": {
            "observed_max": int(
                merged["candidate_count"].astype(int).max()
            ),
            "over_100": over_budget,
            "passed": over_budget == 0,
        },
    }

    segment_masks = {
        "active": merged["ground_truth_state"].fillna("").eq("A"),
        "closed": merged["ground_truth_state"].fillna("").eq("F"),
        "mega": merged["mega_base_pool"].astype(bool),
        "multi_site": merged["multi_site_siren"].astype(bool),
        "insee": merged["location_match_type"].astype(str).eq("insee"),
    }
    segments: dict[str, Any] = {}
    for name, mask in segment_masks.items():
        segment = merged[mask]
        exact = segment[
            segment["label_kind"].eq(GroundTruthKind.MATCH_EXACT.value)
        ]
        segment_coverage = len(exact) / len(segment) if len(segment) else 0.0
        segment_recall = (
            float(exact["hit_at_100"].astype(bool).mean())
            if len(exact)
            else 0.0
        )
        reference = SEGMENT_REFERENCES[name]
        gated = len(exact) >= 100
        coverage_floor = reference["coverage"] - 0.05
        recall_floor = reference["recall"] - 0.02
        coverage_passed = segment_coverage >= coverage_floor
        recall_passed = segment_recall >= recall_floor
        segments[name] = {
            "query_count": int(len(segment)),
            "exact_query_count": int(len(exact)),
            "coverage": segment_coverage,
            "recall_at_100": segment_recall,
            "dev_reference": reference,
            "coverage_floor": coverage_floor,
            "recall_floor": recall_floor,
            "gated": gated,
            "passed": (
                coverage_passed and recall_passed if gated else True
            ),
        }

    global_passed = all(gate["passed"] for gate in gates.values())
    segments_passed = all(segment["passed"] for segment in segments.values())
    decision = "GO" if global_passed and segments_passed else "PIVOT"
    return {
        "decision": decision,
        "query_count": int(len(merged)),
        "v3_exact_query_count": int(len(v3_exact)),
        "metrics": {
            "historical_all_queries": _metric_set(merged),
            "v2_exact": _metric_set(v2_exact),
            "v3_exact": _metric_set(v3_exact),
        },
        "gates": gates,
        "segments": segments,
        "review_query_count": int(
            (
                ~merged["label_kind"].eq(
                    GroundTruthKind.MATCH_EXACT.value
                )
            ).sum()
        ),
        "review_routing_not_measured": True,
    }


def _markdown(result: dict[str, Any], manifest: dict[str, Any]) -> str:
    v3 = result["metrics"]["v3_exact"]["admission_at_100"]
    coverage = result["gates"]["coverage"]["observed"]
    lines = [
        "# Certification finale du retrieval sélectif",
        "",
        f"## Décision : **{result['decision']}**",
        "",
        f"- Test : {result['query_count']} requêtes ;",
        f"- périmètre exact : {result['v3_exact_query_count']} "
        f"({coverage:.3%}) ;",
        f"- Recall@100 V3 : {v3['successes']}/{v3['total']} = "
        f"{v3['rate']:.3%} ;",
        f"- SIRET exacts invisibles en amont : "
        f"{result['gates']['internal_oracle_unseen']['observed']} ;",
        f"- sorties au-dessus de 100 : "
        f"{result['gates']['candidate_ceiling']['over_100']}.",
        "",
        "## Gates",
        "",
        "| Gate | Résultat |",
        "|---|---:|",
    ]
    for name, gate in result["gates"].items():
        lines.append(f"| {name} | {'PASS' if gate['passed'] else 'FAIL'} |")
    lines.extend(
        [
            "",
            "## Segments",
            "",
            "| Segment | N exact | Couverture | Recall@100 | Gate |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, segment in result["segments"].items():
        lines.append(
            f"| {name} | {segment['exact_query_count']} | "
            f"{segment['coverage']:.3%} | "
            f"{segment['recall_at_100']:.3%} | "
            f"{'PASS' if segment['passed'] else 'FAIL'}"
            f"{'' if segment['gated'] else ' (informatif)'} |"
        )
    lines.extend(
        [
            "",
            "La précision AUTO et le comportement `REVIEW` ne sont pas "
            "certifiés par ce test de retrieval.",
            "",
            f"Manifeste : `{manifest['schema_version']}`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualified-benchmark", type=Path, required=True)
    parser.add_argument("--qualification-manifest", type=Path, required=True)
    parser.add_argument("--admission-raw", type=Path, required=True)
    parser.add_argument("--admission-manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(
            f"Immutable output already exists: {args.output_dir}"
        )

    qualification_manifest = json.loads(
        args.qualification_manifest.read_text(encoding="utf-8")
    )
    admission_manifest = json.loads(
        args.admission_manifest.read_text(encoding="utf-8")
    )
    if (
        qualification_manifest.get("outputs", {}).get(
            args.qualified_benchmark.name
        )
        != file_sha256(args.qualified_benchmark)
    ):
        raise ValueError("Qualification artifact hash mismatch")
    if (
        admission_manifest.get("outputs", {}).get(args.admission_raw.name)
        != file_sha256(args.admission_raw)
    ):
        raise ValueError("Admission artifact hash mismatch")
    if qualification_manifest.get("split") != "test":
        raise ValueError("Certification requires the test qualification")
    if admission_manifest.get("split") != "test":
        raise ValueError("Certification requires the test admission")
    if qualification_manifest.get(
        "benchmark_build_id"
    ) != admission_manifest.get("benchmark_build_id"):
        raise ValueError("Qualification and admission benchmark IDs differ")

    result = certify(
        pd.read_parquet(args.qualified_benchmark),
        pd.read_parquet(args.admission_raw),
    )
    args.output_dir.mkdir(parents=True)
    summary_path = args.output_dir / "summary.json"
    report_path = args.output_dir / "report.md"
    summary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "command": [sys.executable, *sys.argv],
        "benchmark_build_id": qualification_manifest["benchmark_build_id"],
        "split": "test",
        "decision": result["decision"],
        "inputs": {
            "qualification_manifest_sha256": file_sha256(
                args.qualification_manifest
            ),
            "qualified_benchmark_sha256": file_sha256(
                args.qualified_benchmark
            ),
            "admission_manifest_sha256": file_sha256(
                args.admission_manifest
            ),
            "admission_raw_sha256": file_sha256(args.admission_raw),
            "contract_sha256": file_sha256(args.contract),
        },
    }
    report_path.write_text(
        _markdown(result, manifest),
        encoding="utf-8",
    )
    manifest["outputs"] = {
        "summary.json": file_sha256(summary_path),
        "report.md": file_sha256(report_path),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
