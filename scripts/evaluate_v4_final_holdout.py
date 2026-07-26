#!/usr/bin/env python3
"""Evaluate the frozen V4 ranker and acceptor on prepared final inputs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_downstream_selective_dataset import (  # noqa: E402
    DATASET_FEATURE_ORDER,
    CandidateWriter,
    build_split_candidates,
)
from scripts.run_v9_retrieval_experiment import (  # noqa: E402
    _git_commit,
    wilson_interval,
)
from scripts.train_v9_ranker import ranking_metrics, score_rows  # noqa: E402
from src.xgb_matcher.partitioned_store import (  # noqa: E402
    PartitionedCandidateStore,
)
from src.xgb_matcher.v9_acceptor import V9AcceptorBundle  # noqa: E402
from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402
from src.xgb_matcher.v9_scene import build_query_scenes  # noqa: E402


SCHEMA_VERSION = "sireto-v4-final-evaluation-1"
EXPECTED = {
    "source": 1345,
    "exact": 302,
    "ambiguous": 52,
    "evaluation": 354,
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(successes: int, total: int) -> dict[str, Any]:
    low95, high95 = wilson_interval(successes, total, confidence=0.95)
    low99, high99 = wilson_interval(successes, total, confidence=0.99)
    return {
        "successes": int(successes),
        "total": int(total),
        "rate": successes / total if total else 0.0,
        "wilson_95": [low95, high95],
        "wilson_99": [low99, high99],
    }


def final_verdict(
    *,
    integrity_pass: bool,
    source_coverage_pass: bool,
    technical_pass: bool,
) -> tuple[str, str]:
    if not integrity_pass:
        return "STOP", "TECHNICAL_INVALID"
    technical = "TECHNICAL_GO" if technical_pass else "TECHNICAL_PIVOT"
    if source_coverage_pass and technical_pass:
        return "GO", technical
    return "PIVOT", technical


def _validate_manifest_output(path: Path) -> None:
    manifest = _read_json(path.parent / "manifest.json")
    expected = manifest.get("outputs", {}).get(path.name)
    if expected != file_sha256(path):
        raise ValueError(f"Input hash mismatch: {path}")


def _markdown(summary: dict[str, Any]) -> str:
    metrics = summary["metrics"]
    gates = summary["gates"]
    return "\n".join(
        [
            "# Évaluation finale V4",
            "",
            f"**Verdict global : `{summary['verdict']}`**",
            "",
            f"Sous-verdict matching : `{summary['technical_verdict']}`.",
            "",
            "## Résultats",
            "",
            "- Couverture identifiable source : "
            f"{metrics['source_identifiable_coverage']['successes']}/"
            f"{metrics['source_identifiable_coverage']['total']} = "
            f"{metrics['source_identifiable_coverage']['rate']:.3%}.",
            "- Recall@100 SIRET exact : "
            f"{metrics['retrieval_recall_at_100']['successes']}/"
            f"{metrics['retrieval_recall_at_100']['total']} = "
            f"{metrics['retrieval_recall_at_100']['rate']:.3%}.",
            "- Hit@1 SIRET exact : "
            f"{metrics['ranker_hit_at_1_siret']['successes']}/"
            f"{metrics['ranker_hit_at_1_siret']['total']} = "
            f"{metrics['ranker_hit_at_1_siret']['rate']:.3%}.",
            "- AUTO_MATCH : "
            f"{metrics['auto_coverage']['successes']}/"
            f"{metrics['auto_coverage']['total']} = "
            f"{metrics['auto_coverage']['rate']:.3%}.",
            "- Précision SIRET exacte des AUTO : "
            f"{metrics['auto_precision']['successes']}/"
            f"{metrics['auto_precision']['total']} = "
            f"{metrics['auto_precision']['rate']:.3%}.",
            "",
            "## Gates",
            "",
            *[
                f"- {name} : {'PASS' if passed else 'FAIL'}"
                for name, passed in gates.items()
            ],
            "",
            "Ce résultat n'est pas une garantie statistique de 99,8 %.",
            "",
        ]
    )


def evaluate(
    *,
    authorization_path: Path,
    prepared_dir: Path,
    v7_raw_path: Path,
    overlay_raw_path: Path,
    admission_raw_path: Path,
    output_root: Path,
) -> Path:
    authorization = _read_json(authorization_path)
    if (
        authorization.get("purpose") != "v4_final_holdout_once"
        or authorization.get("status") != "FROZEN_AUTHORIZED"
    ):
        raise ValueError("Invalid final authorization")
    prepared_manifest = _read_json(prepared_dir / "manifest.json")
    if not prepared_manifest.get("holdout_opened"):
        raise ValueError("Final holdout was not prepared")
    if (
        prepared_manifest.get("authorization_sha256")
        != file_sha256(authorization_path)
    ):
        raise ValueError("Prepared input uses another authorization")
    for name, expected_hash in prepared_manifest["outputs"].items():
        if file_sha256(prepared_dir / name) != expected_hash:
            raise ValueError(f"Prepared artifact changed: {name}")
    for path in (v7_raw_path, overlay_raw_path, admission_raw_path):
        _validate_manifest_output(path)

    benchmark = pd.read_parquet(
        prepared_dir / "evaluation_benchmark.parquet"
    )
    labels = pd.read_parquet(prepared_dir / "labels.parquet")
    v7_raw = pd.read_parquet(v7_raw_path)
    overlay_raw = pd.read_parquet(overlay_raw_path)
    admission = pd.read_parquet(admission_raw_path)
    for frame in (benchmark, labels, v7_raw, overlay_raw, admission):
        frame["query_id"] = frame["query_id"].astype(str)
    query_ids = set(labels["query_id"])
    if any(
        set(frame["query_id"]) != query_ids
        for frame in (benchmark, v7_raw, overlay_raw, admission)
    ):
        raise ValueError("Final retrieval inputs do not align")

    label_counts = labels["label_kind"].value_counts().to_dict()
    if label_counts != {
        "MATCH_EXACT": EXPECTED["exact"],
        "AMBIGUOUS": EXPECTED["ambiguous"],
    }:
        raise ValueError("Final evaluation label counts differ")

    output_dir = output_root / authorization["authorization_id"]
    output_dir.mkdir(parents=True, exist_ok=False)
    candidates_path = output_dir / "candidates.parquet"
    writer = CandidateWriter(candidates_path)
    try:
        diagnostics = build_split_candidates(
            split="test",
            benchmark=benchmark,
            labels=labels,
            admission=admission,
            v7_channels=v7_raw,
            overlay_channels=overlay_raw,
            v7_store=PartitionedCandidateStore(
                Path(authorization["paths"]["active_partitions"])
            ),
            overlay_store=PartitionedCandidateStore(
                Path(authorization["paths"]["overlay_partitions"])
            ),
            writer=writer,
        )
    finally:
        writer.close()
    candidates = pd.read_parquet(candidates_path)
    counts = candidates.groupby("query_id").size()
    duplicate_pairs = int(
        candidates.duplicated(["query_id", "candidate_siret"]).sum()
    )

    ranker_dir = Path(authorization["paths"]["ranker_dir"])
    ranker_metadata = _read_json(ranker_dir / "metadata.json")
    if ranker_metadata["feature_order"] != DATASET_FEATURE_ORDER:
        raise ValueError("Frozen ranker features changed")
    ranker = xgb.XGBRanker()
    ranker.load_model(ranker_dir / "ranker.json")
    predictions = score_rows(
        ranker,
        candidates,
        DATASET_FEATURE_ORDER,
        origin="final_holdout",
        fold=None,
    )
    predictions_path = output_dir / "ranker_predictions.parquet"
    predictions.to_parquet(predictions_path, index=False)
    ranker_summary = ranking_metrics(predictions, labels)

    scenes = build_query_scenes(predictions, labels)
    acceptor_dir = Path(authorization["paths"]["acceptor_dir"])
    bundle = V9AcceptorBundle.load(acceptor_dir)
    if (
        bundle.model_bundle_id
        != authorization["acceptor_model_bundle_id"]
        or bundle.threshold != authorization["acceptor_threshold"]
    ):
        raise ValueError("Frozen acceptor bundle changed")
    scenes["confidence"] = bundle.confidence(scenes)
    scenes["decision"] = "REVIEW"
    scenes.loc[
        scenes["confidence"].ge(bundle.threshold), "decision"
    ] = "AUTO_MATCH"
    scenes["auto_correct"] = (
        scenes["decision"].eq("AUTO_MATCH")
        & scenes["is_exact_siret_correct"].eq(1)
    )
    scenes_path = output_dir / "scenes.parquet"
    scenes.to_parquet(scenes_path, index=False)

    exact_scenes = scenes[scenes["label_kind"].eq("MATCH_EXACT")]
    ambiguous_scenes = scenes[scenes["label_kind"].eq("AMBIGUOUS")]
    auto = scenes["decision"].eq("AUTO_MATCH")
    auto_count = int(auto.sum())
    auto_correct = int(scenes.loc[auto, "is_exact_siret_correct"].sum())
    exact_top1_siret = int(
        exact_scenes["is_exact_siret_correct"].sum()
    )
    exact_top1_siren = int(
        (
            exact_scenes["predicted_siren"].fillna("")
            == exact_scenes["ground_truth_siren"].fillna("").astype(str)
        ).sum()
    )
    exact_truth = labels[
        labels["label_kind"].eq("MATCH_EXACT")
    ][["query_id", "ground_truth_siret"]]
    retrieval = exact_truth.merge(
        admission[["query_id", "candidate_sirets_json"]],
        on="query_id",
        how="left",
        validate="one_to_one",
    )
    retrieval["hit"] = retrieval.apply(
        lambda row: str(row["ground_truth_siret"]).zfill(14)
        in [
            str(value).zfill(14)
            for value in json.loads(row["candidate_sirets_json"])
        ],
        axis=1,
    )
    retrieval_hits = int(retrieval["hit"].sum())
    candidate_count_max = int(counts.max()) if len(counts) else 0
    above_100 = int(counts.gt(100).sum())
    missing_scenes = EXPECTED["evaluation"] - int(
        scenes["query_id"].nunique()
    )

    metrics = {
        "source_identifiable_coverage": _metric(
            EXPECTED["exact"], EXPECTED["source"]
        ),
        "evaluated_exact_share": _metric(
            EXPECTED["exact"], EXPECTED["evaluation"]
        ),
        "retrieval_recall_at_100": _metric(
            retrieval_hits, EXPECTED["exact"]
        ),
        "ranker_hit_at_1_siret": _metric(
            exact_top1_siret, EXPECTED["exact"]
        ),
        "ranker_hit_at_1_siren": _metric(
            exact_top1_siren, EXPECTED["exact"]
        ),
        "auto_coverage": _metric(auto_count, EXPECTED["evaluation"]),
        "auto_coverage_exact_only": _metric(
            int(exact_scenes["decision"].eq("AUTO_MATCH").sum()),
            EXPECTED["exact"],
        ),
        "auto_precision": _metric(auto_correct, auto_count),
    }
    integrity = {
        "prepared_counts_match": (
            prepared_manifest["source_query_count"] == EXPECTED["source"]
            and prepared_manifest["evaluation_query_count"]
            == EXPECTED["evaluation"]
        ),
        "zero_exact_siren_overlap": (
            prepared_manifest["exact_siren_overlap_with_fit_dev"] == 0
        ),
        "zero_missing_candidate_details": (
            diagnostics["missing_candidate_details"] == 0
        ),
        "zero_duplicate_candidate_lists": (
            diagnostics["duplicate_candidate_lists"] == 0
        ),
        "zero_over_budget_diagnostic": (
            diagnostics["over_budget_queries"] == 0
        ),
        "zero_duplicate_candidate_pairs": duplicate_pairs == 0,
        "candidate_budget_at_most_100": (
            candidate_count_max <= 100 and above_100 == 0
        ),
        "all_scenes_scored": missing_scenes == 0,
        "old_test_read": False,
        "positive_injection": False,
    }
    gates = {
        "source_identifiable_coverage_at_least_80pct": (
            metrics["source_identifiable_coverage"]["rate"] >= 0.80
        ),
        "retrieval_recall_at_100_at_least_99pct": (
            metrics["retrieval_recall_at_100"]["rate"] >= 0.99
        ),
        "ranker_hit_at_1_at_least_96_033pct": (
            metrics["ranker_hit_at_1_siret"]["rate"] >= 0.96033
        ),
        "auto_coverage_at_least_25pct": (
            metrics["auto_coverage"]["rate"] >= 0.25
        ),
        "auto_precision_at_least_99_8pct": (
            metrics["auto_precision"]["rate"] >= 0.998
        ),
        "auto_count_at_least_25": auto_count >= 25,
    }
    integrity_pass = all(integrity.values())
    technical_pass = all(
        gates[name]
        for name in (
            "retrieval_recall_at_100_at_least_99pct",
            "ranker_hit_at_1_at_least_96_033pct",
            "auto_coverage_at_least_25pct",
            "auto_precision_at_least_99_8pct",
            "auto_count_at_least_25",
        )
    )
    verdict, technical_verdict = final_verdict(
        integrity_pass=integrity_pass,
        source_coverage_pass=gates[
            "source_identifiable_coverage_at_least_80pct"
        ],
        technical_pass=technical_pass,
    )
    errors = scenes[
        (scenes["label_kind"].eq("MATCH_EXACT") & ~scenes["auto_correct"])
        | (
            scenes["label_kind"].eq("AMBIGUOUS")
            & scenes["decision"].eq("AUTO_MATCH")
        )
    ].copy()
    errors_path = output_dir / "errors_and_reviews.parquet"
    errors.to_parquet(errors_path, index=False)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "authorization_id": authorization["authorization_id"],
        "verdict": verdict,
        "technical_verdict": technical_verdict,
        "metrics": metrics,
        "counts": {
            "ambiguous_auto": int(
                ambiguous_scenes["decision"].eq("AUTO_MATCH").sum()
            ),
            "exact_auto_errors": int(
                (
                    exact_scenes["decision"].eq("AUTO_MATCH")
                    & exact_scenes["is_exact_siret_correct"].eq(0)
                ).sum()
            ),
            "candidate_count_max": candidate_count_max,
            "candidate_lists_above_100": above_100,
        },
        "gates": gates,
        "integrity": integrity,
        "diagnostics": diagnostics,
        "ranker_raw_summary": ranker_summary,
        "statistical_warning": (
            "Observed precision is not a 99.8% production guarantee."
        ),
    }
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(_markdown(summary), encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "authorization_sha256": file_sha256(authorization_path),
        "prepared_manifest_sha256": file_sha256(
            prepared_dir / "manifest.json"
        ),
        "inputs": {
            "v7_raw": file_sha256(v7_raw_path),
            "overlay_raw": file_sha256(overlay_raw_path),
            "admission_raw": file_sha256(admission_raw_path),
            "ranker_model": file_sha256(ranker_dir / "ranker.json"),
            "acceptor_model": file_sha256(
                acceptor_dir / "acceptor_model.joblib"
            ),
            "acceptor_calibrator": file_sha256(
                acceptor_dir / "acceptor_calibrator.joblib"
            ),
        },
        "outputs": {
            path.name: file_sha256(path)
            for path in (
                candidates_path,
                predictions_path,
                scenes_path,
                errors_path,
                summary_path,
                report_path,
            )
        },
        "holdout_opened_once": True,
        "old_test_read": False,
        "positive_injection": False,
        "verdict": verdict,
        "technical_verdict": technical_verdict,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--v7-raw", type=Path, required=True)
    parser.add_argument("--overlay-raw", type=Path, required=True)
    parser.add_argument("--admission-raw", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(
        evaluate(
            authorization_path=args.authorization,
            prepared_dir=args.prepared_dir,
            v7_raw_path=args.v7_raw,
            overlay_raw_path=args.overlay_raw,
            admission_raw_path=args.admission_raw,
            output_root=args.output_root,
        )
    )


if __name__ == "__main__":
    main()
