#!/usr/bin/env python3
"""Produce an immutable paired comparison for two V9 retrieval modes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.v9_dataset import file_sha256
from src.xgb_matcher.v9_evaluation import (
    paired_binary_comparison,
    retrieval_promotion_gate,
)


def _segment_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    masks = {
        "gt_active": frame["ground_truth_state"].fillna("").eq("A"),
        "gt_closed": frame["ground_truth_state"].fillna("").eq("F"),
        "missing_insee": frame["missing_insee"].astype(bool),
        "mega_base_pool": frame["mega_base_pool"].astype(bool),
        "multi_site_siren": frame["multi_site_siren"].astype(bool),
    }
    for value in sorted(
        str(item)
        for item in frame["location_match_type"].dropna().unique()
        if str(item)
    ):
        masks[f"location_match_type={value}"] = (
            frame["location_match_type"].astype(str).eq(value)
        )
    return masks


def _validate_experiment(
    experiment_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    manifest_path = experiment_dir / "manifest.json"
    summary_path = experiment_dir / "summary.json"
    raw_path = experiment_dir / "raw_results.parquet"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = manifest.get("outputs", {})
    if expected.get("raw_results.parquet") != file_sha256(raw_path):
        raise ValueError("raw_results.parquet hash differs from experiment manifest")
    if expected.get("summary.json") != file_sha256(summary_path):
        raise ValueError("summary.json hash differs from experiment manifest")
    return pd.read_parquet(raw_path), summary, manifest


def compare_modes(
    raw: pd.DataFrame,
    summary: dict[str, Any],
    *,
    baseline_mode: str,
    variant_mode: str,
    bootstrap_samples: int = 100_000,
    seed: int = 42,
) -> dict[str, Any]:
    required = {
        "mode",
        "query_id",
        "ground_truth_siret",
        "hit_at_budget_siret",
        "budget_compliant",
        "latency_ms",
        "ground_truth_state",
        "location_match_type",
        "missing_insee",
        "mega_base_pool",
        "multi_site_siren",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Raw experiment is missing columns: {missing}")
    if baseline_mode not in summary or variant_mode not in summary:
        raise ValueError("Requested modes are absent from summary.json")

    baseline = raw[raw["mode"].eq(baseline_mode)].copy()
    variant = raw[raw["mode"].eq(variant_mode)].copy()
    for mode, frame in ((baseline_mode, baseline), (variant_mode, variant)):
        if frame.empty:
            raise ValueError(f"Mode has no raw rows: {mode}")
        if frame["query_id"].duplicated().any():
            raise ValueError(f"Mode has duplicate query_id values: {mode}")

    paired = baseline.merge(
        variant,
        on="query_id",
        how="outer",
        suffixes=("_baseline", "_variant"),
        validate="one_to_one",
        indicator=True,
    )
    if not paired["_merge"].eq("both").all():
        raise ValueError("Modes do not contain exactly the same query IDs")
    if not paired["ground_truth_siret_baseline"].eq(
        paired["ground_truth_siret_variant"]
    ).all():
        raise ValueError("Ground truth differs between paired modes")

    paired = paired.sort_values("query_id").reset_index(drop=True)
    overall = paired_binary_comparison(
        paired["hit_at_budget_siret_baseline"].to_numpy(),
        paired["hit_at_budget_siret_variant"].to_numpy(),
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )

    baseline_for_segments = baseline.set_index("query_id").loc[
        paired["query_id"]
    ].reset_index()
    segments: dict[str, Any] = {}
    for name, mask in _segment_masks(baseline_for_segments).items():
        if not mask.any():
            continue
        segment = paired_binary_comparison(
            paired.loc[mask.to_numpy(), "hit_at_budget_siret_baseline"].to_numpy(),
            paired.loc[mask.to_numpy(), "hit_at_budget_siret_variant"].to_numpy(),
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        segments[name] = segment

    segment_deltas = {
        name: values["delta"] for name, values in segments.items()
    }
    baseline_p95 = float(
        summary[baseline_mode]["latency_ms"]["p95"]
    )
    variant_p95 = float(summary[variant_mode]["latency_ms"]["p95"])
    baseline_budget_violations = int(
        (~paired["budget_compliant_baseline"].astype(bool)).sum()
    )
    variant_budget_violations = int(
        (~paired["budget_compliant_variant"].astype(bool)).sum()
    )
    gate = retrieval_promotion_gate(
        baseline_recall_at_50=overall["baseline_rate"],
        variant_recall_at_50=overall["variant_rate"],
        baseline_latency_p95_ms=baseline_p95,
        variant_latency_p95_ms=variant_p95,
        segment_recall_deltas=segment_deltas,
        baseline_budget_violations=baseline_budget_violations,
        variant_budget_violations=variant_budget_violations,
    )
    return {
        "baseline_mode": baseline_mode,
        "variant_mode": variant_mode,
        "overall": overall,
        "segments": segments,
        "latency_ms": {
            "baseline_p95": baseline_p95,
            "variant_p95": variant_p95,
            "ratio": gate["latency_ratio"],
        },
        "gate": gate,
    }


def _markdown(report: dict[str, Any], manifest: dict[str, Any]) -> str:
    overall = report["overall"]
    gate = report["gate"]
    rows = [
        "# V9 — comparaison retrieval appariée",
        "",
        f"- Benchmark : `{manifest['benchmark_build_id']}` / split `{manifest['split']}`",
        f"- Baseline : `{report['baseline_mode']}`",
        f"- Variante : `{report['variant_mode']}`",
        f"- Décision Gate 2 : **{'PASS' if gate['promote'] else 'FAIL'}**",
        "",
        "| Mesure | Résultat |",
        "|---|---:|",
        f"| Requêtes appariées | {overall['total']} |",
        f"| Recall baseline | {overall['baseline_rate']:.4%} |",
        f"| Recall variante | {overall['variant_rate']:.4%} |",
        f"| Delta | {overall['delta']:+.4%} |",
        f"| Misses récupérés | {overall['recovered']} |",
        f"| Hits déplacés/perdus | {overall['displaced']} |",
        (
            "| IC95 bootstrap apparié | "
            f"[{overall['paired_bootstrap_95'][0]:+.4%}, "
            f"{overall['paired_bootstrap_95'][1]:+.4%}] |"
        ),
        f"| p exact McNemar bilatéral | {overall['mcnemar_exact_two_sided_p']:.6g} |",
        f"| Ratio latence p95 | {gate['latency_ratio']:.3f}× |",
        "",
        "## Gates pré-enregistrées",
        "",
    ]
    for name, passed in gate["checks"].items():
        rows.append(f"- {'PASS' if passed else 'FAIL'} — `{name}`")
    rows.extend(
        [
            "",
            "## Segments",
            "",
            "| Segment | n | Delta | Récupérés | Déplacés |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, values in report["segments"].items():
        rows.append(
            f"| {name} | {values['total']} | {values['delta']:+.4%} | "
            f"{values['recovered']} | {values['displaced']} |"
        )
    rows.append("")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--baseline-mode", default="sparse")
    parser.add_argument("--variant-mode", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(
            f"Immutable comparison directory exists: {args.output_dir}"
        )

    raw, summary, manifest = _validate_experiment(args.experiment_dir)
    report = compare_modes(
        raw,
        summary,
        baseline_mode=args.baseline_mode,
        variant_mode=args.variant_mode,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    report["experiment_manifest_sha256"] = file_sha256(
        args.experiment_dir / "manifest.json"
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    json_path = args.output_dir / "comparison.json"
    markdown_path = args.output_dir / "comparison.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    markdown_path.write_text(
        _markdown(report, manifest),
        encoding="utf-8",
    )
    comparison_manifest = {
        "experiment_manifest_sha256": report["experiment_manifest_sha256"],
        "baseline_mode": args.baseline_mode,
        "variant_mode": args.variant_mode,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "outputs": {
            "comparison.json": file_sha256(json_path),
            "comparison.md": file_sha256(markdown_path),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(comparison_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
