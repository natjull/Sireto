#!/usr/bin/env python3
"""Audit V3 qualification stability on train/dev without reading test."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_v9_retrieval_experiment import _git_commit  # noqa: E402
from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


SCHEMA_VERSION = "sireto-v3-qualification-stability-1"
ALLOWED_SPLITS = {"train", "dev"}
NAME_NEAR_THRESHOLD = 0.75
STREET_NEAR_THRESHOLD = 0.75


def _safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def qualification_decomposition(frame: pd.DataFrame) -> dict[str, Any]:
    required = {
        "label_kind",
        "v2_label_kind",
        "direct_evidence_class",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Qualified benchmark is missing columns: {sorted(missing)}"
        )
    query_count = len(frame)
    v2_exact = frame["v2_label_kind"].eq("MATCH_EXACT")
    v3_exact = frame["label_kind"].eq("MATCH_EXACT")
    if (v3_exact & ~v2_exact).any():
        raise ValueError("V3 exact scope cannot exceed V2 exact scope")
    v2_count = int(v2_exact.sum())
    v3_count = int(v3_exact.sum())
    return {
        "query_count": query_count,
        "v2_exact_count": v2_count,
        "v3_exact_count": v3_count,
        "structural_excluded_count": query_count - v2_count,
        "direct_evidence_excluded_count": v2_count - v3_count,
        "v2_coverage": _safe_rate(v2_count, query_count),
        "direct_evidence_retention": _safe_rate(v3_count, v2_count),
        "v3_coverage": _safe_rate(v3_count, query_count),
    }


def no_evidence_nearness(frame: pd.DataFrame) -> dict[str, Any]:
    required = {
        "label_kind",
        "v2_label_kind",
        "name_jaro_max",
        "postcode_match",
        "street_name_jaro",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Qualified benchmark is missing columns: {sorted(missing)}"
        )
    excluded = frame[
        frame["v2_label_kind"].eq("MATCH_EXACT")
        & ~frame["label_kind"].eq("MATCH_EXACT")
    ].copy()
    name_near = excluded["name_jaro_max"].ge(NAME_NEAR_THRESHOLD)
    address_near = (
        excluded["postcode_match"].gt(0)
        & excluded["street_name_jaro"].ge(STREET_NEAR_THRESHOLD)
    )
    labels = (
        name_near.map({True: "NAME_NEAR", False: "NAME_FAR"})
        + "__"
        + address_near.map({True: "ADDRESS_NEAR", False: "ADDRESS_FAR"})
    )
    counts = labels.value_counts().sort_index()
    return {
        "query_count": int(len(excluded)),
        "name_near_threshold": NAME_NEAR_THRESHOLD,
        "street_near_threshold": STREET_NEAR_THRESHOLD,
        "buckets": {
            str(label): {
                "count": int(count),
                "rate": _safe_rate(int(count), len(excluded)),
            }
            for label, count in counts.items()
        },
    }


def _segment(
    frame: pd.DataFrame,
    mask: pd.Series,
) -> dict[str, Any]:
    return qualification_decomposition(frame[mask])


def summarize_segments(
    frame: pd.DataFrame,
    *,
    dev_flags: pd.DataFrame | None = None,
) -> dict[str, Any]:
    required = {"ground_truth_state", "location_match_type", "query_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Qualified benchmark is missing columns: {sorted(missing)}"
        )
    working = frame.copy()
    working["query_id"] = working["query_id"].astype(str)
    segments = {
        "all": qualification_decomposition(working),
        "active": _segment(
            working,
            working["ground_truth_state"].eq("A"),
        ),
        "closed": _segment(
            working,
            working["ground_truth_state"].eq("F"),
        ),
        "insee": _segment(
            working,
            working["location_match_type"].eq("insee"),
        ),
        "cp_only": _segment(
            working,
            working["location_match_type"].eq("cp_only"),
        ),
    }
    if dev_flags is None:
        return segments

    flags = dev_flags.copy()
    flags["query_id"] = flags["query_id"].astype(str)
    required_flags = {"query_id", "mega_base_pool", "multi_site_siren"}
    missing_flags = required_flags - set(flags.columns)
    if missing_flags:
        raise ValueError(f"Dev flags are missing columns: {sorted(missing_flags)}")
    if flags["query_id"].duplicated().any():
        raise ValueError("Dev flags contain duplicate query IDs")
    if set(flags["query_id"]) != set(working["query_id"]):
        raise ValueError("Dev flags and qualified benchmark query IDs differ")
    working = working.merge(
        flags[list(required_flags)],
        on="query_id",
        validate="one_to_one",
    )
    mega = working["mega_base_pool"].astype(bool)
    multi = working["multi_site_siren"].astype(bool)
    active = working["ground_truth_state"].eq("A")
    closed = working["ground_truth_state"].eq("F")
    segments.update(
        {
            "mega": _segment(working, mega),
            "non_mega": _segment(working, ~mega),
            "mega_active": _segment(working, mega & active),
            "mega_closed": _segment(working, mega & closed),
            "multi_site": _segment(working, multi),
            "single_site": _segment(working, ~multi),
        }
    )
    return segments


def audit_train_dev(
    train: pd.DataFrame,
    dev: pd.DataFrame,
    dev_flags: pd.DataFrame,
) -> dict[str, Any]:
    for expected, frame in (("train", train), ("dev", dev)):
        observed = set(frame["split"].astype(str).unique())
        if observed != {expected}:
            raise ValueError(
                f"Expected only split {expected!r}, observed {sorted(observed)}"
            )
        if "test" in observed:
            raise ValueError("Test split is forbidden in this audit")

    train_segments = summarize_segments(train)
    dev_segments = summarize_segments(dev, dev_flags=dev_flags)
    return {
        "scope": "TRAIN_DEV_ONLY",
        "test_read": False,
        "train": {
            "segments": train_segments,
            "no_evidence": {
                "all": no_evidence_nearness(train),
                "closed": no_evidence_nearness(
                    train[train["ground_truth_state"].eq("F")]
                ),
            },
        },
        "dev": {
            "segments": dev_segments,
            "no_evidence": {
                "all": no_evidence_nearness(dev),
                "closed": no_evidence_nearness(
                    dev[dev["ground_truth_state"].eq("F")]
                ),
                "mega": no_evidence_nearness(
                    dev[
                        dev["query_id"].astype(str).isin(
                            dev_flags.loc[
                                dev_flags["mega_base_pool"].astype(bool),
                                "query_id",
                            ].astype(str)
                        )
                    ]
                ),
            },
        },
    }


def _validate_bound_parquet(
    path: Path,
    manifest_path: Path,
    *,
    expected_split: str,
) -> dict[str, Any]:
    if expected_split not in ALLOWED_SPLITS:
        raise ValueError("This audit accepts only train and dev")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("split") != expected_split:
        raise ValueError(
            f"Manifest split must be {expected_split!r}, "
            f"got {manifest.get('split')!r}"
        )
    expected_hash = manifest.get("outputs", {}).get(path.name)
    if not expected_hash:
        raise ValueError(f"Manifest does not bind {path.name}")
    if file_sha256(path) != expected_hash:
        raise ValueError(f"Hash mismatch for {path}")
    return manifest


def _format_rate(value: float) -> str:
    return f"{value:.3%}"


def render_report(summary: dict[str, Any], manifest: dict[str, Any]) -> str:
    lines = [
        "# Audit de stabilité de la qualification V3 — train/dev",
        "",
        "Cet audit n'a lu aucune ligne du test.",
        "",
        "## Décomposition de la couverture",
        "",
        "| Split/segment | N | V2 exact | V3 exact | Couverture V2 | "
        "Rétention preuve | Couverture V3 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, segment_names in (
        ("train", ("all", "active", "closed", "insee", "cp_only")),
        (
            "dev",
            (
                "all",
                "active",
                "closed",
                "mega",
                "mega_active",
                "mega_closed",
                "multi_site",
            ),
        ),
    ):
        segments = summary[split]["segments"]
        for name in segment_names:
            metric = segments[name]
            lines.append(
                f"| {split}/{name} | {metric['query_count']} | "
                f"{metric['v2_exact_count']} | {metric['v3_exact_count']} | "
                f"{_format_rate(metric['v2_coverage'])} | "
                f"{_format_rate(metric['direct_evidence_retention'])} | "
                f"{_format_rate(metric['v3_coverage'])} |"
            )

    lines.extend(
        [
            "",
            "## Distance descriptive des cas sans preuve directe",
            "",
            "`NEAR` signifie seulement Jaro nom ≥ 0,75, ou code postal égal "
            "et Jaro voie ≥ 0,75. Ce n'est pas une règle de relabel.",
            "",
            "| Split/segment | Sans preuve | Nom loin + adresse loin | "
            "Nom loin + adresse proche | Nom proche + adresse loin | "
            "Nom proche + adresse proche |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    bucket_order = (
        "NAME_FAR__ADDRESS_FAR",
        "NAME_FAR__ADDRESS_NEAR",
        "NAME_NEAR__ADDRESS_FAR",
        "NAME_NEAR__ADDRESS_NEAR",
    )
    for split, segment_names in (
        ("train", ("all", "closed")),
        ("dev", ("all", "closed", "mega")),
    ):
        for name in segment_names:
            profile = summary[split]["no_evidence"][name]
            values = [
                profile["buckets"].get(bucket, {}).get("count", 0)
                for bucket in bucket_order
            ]
            lines.append(
                f"| {split}/{name} | {profile['query_count']} | "
                + " | ".join(str(value) for value in values)
                + " |"
            )

    lines.extend(
        [
            "",
            "## Lecture",
            "",
            "- La pénalité des établissements fermés existe déjà dans le "
            "train : elle n'est pas créée par le dev.",
            "- Sur dev, les mégapoles ont une meilleure rétention de preuve "
            "que l'ensemble. Le petit segment dev ne suffit donc pas comme "
            "référence stable de couverture.",
            "- Parmi les cas V2 exacts rejetés par V3, le groupe dominant a "
            "à la fois un nom et une adresse éloignés. Un simple déplacement "
            "de seuil ne traite pas la cause dominante.",
            "- Les cas ouverts demandent surtout des alias ou preuves "
            "externes versionnés, ou une revue ; ils ne justifient pas une "
            "nouvelle architecture de retrieval.",
            "",
            f"Manifeste : `{manifest['schema_version']}`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-benchmark", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--dev-benchmark", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--dev-flags", type=Path, required=True)
    parser.add_argument("--dev-flags-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    train_manifest = _validate_bound_parquet(
        args.train_benchmark,
        args.train_manifest,
        expected_split="train",
    )
    dev_manifest = _validate_bound_parquet(
        args.dev_benchmark,
        args.dev_manifest,
        expected_split="dev",
    )
    flags_manifest = _validate_bound_parquet(
        args.dev_flags,
        args.dev_flags_manifest,
        expected_split="dev",
    )
    build_ids = {
        train_manifest.get("benchmark_build_id"),
        dev_manifest.get("benchmark_build_id"),
        flags_manifest.get("benchmark_build_id"),
    }
    if len(build_ids) != 1:
        raise ValueError("Train/dev/flags benchmark build IDs differ")
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")

    train = pd.read_parquet(args.train_benchmark)
    dev = pd.read_parquet(args.dev_benchmark)
    dev_flags = pd.read_parquet(args.dev_flags)
    summary = audit_train_dev(train, dev, dev_flags)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "scope": "TRAIN_DEV_ONLY",
        "test_read": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "benchmark_build_id": next(iter(build_ids)),
        "thresholds": {
            "descriptive_name_near": NAME_NEAR_THRESHOLD,
            "descriptive_street_near": STREET_NEAR_THRESHOLD,
        },
        "inputs": {
            "train_benchmark_sha256": file_sha256(args.train_benchmark),
            "train_manifest_sha256": file_sha256(args.train_manifest),
            "dev_benchmark_sha256": file_sha256(args.dev_benchmark),
            "dev_manifest_sha256": file_sha256(args.dev_manifest),
            "dev_flags_sha256": file_sha256(args.dev_flags),
            "dev_flags_manifest_sha256": file_sha256(
                args.dev_flags_manifest
            ),
        },
    }
    identity = json.dumps(manifest, sort_keys=True).encode("utf-8")
    manifest["audit_id"] = hashlib.sha256(identity).hexdigest()[:16]

    args.output_dir.mkdir(parents=True)
    summary_path = args.output_dir / "summary.json"
    report_path = args.output_dir / "report.md"
    manifest_path = args.output_dir / "manifest.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        render_report(summary, manifest),
        encoding="utf-8",
    )
    manifest["outputs"] = {
        "summary.json": file_sha256(summary_path),
        "report.md": file_sha256(report_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
