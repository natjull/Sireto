#!/usr/bin/env python3
"""Build an immutable, provisional V2 qualification of exact-SIRET labels."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
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


SCHEMA_VERSION = "sireto-benchmark-v2-qualification-1"
POLICY_VERSION = "site-consistency-v2.0"

AMBIGUOUS_CLASSES = {
    "ACTIVE_GT_HAS_ACTIVE_EXACT_SIBLING",
    "MULTIPLE_ACTIVE_EXACT_SIBLINGS",
}
UNRESOLVED_CLASSES = {
    "CLOSED_GT_UNIQUE_ACTIVE_EXACT_SIBLING",
}
KNOWN_SITE_CLASSES = {
    *AMBIGUOUS_CLASSES,
    *UNRESOLVED_CLASSES,
    "HISTORICAL_REFERENCE_DATE_PRESENT",
    "INACTIVE_EXACT_SIBLING_ONLY",
    "NO_EXACT_SIBLING",
    "NO_USABLE_CRM_ADDRESS",
}


def qualify_site_label(site_label_class: str) -> tuple[str, str]:
    """Apply the retrieval-independent V2 qualification policy."""
    if site_label_class not in KNOWN_SITE_CLASSES:
        raise ValueError(f"Unsupported site label class: {site_label_class}")
    if site_label_class == "ACTIVE_GT_HAS_ACTIVE_EXACT_SIBLING":
        return (
            GroundTruthKind.AMBIGUOUS.value,
            "ACTIVE_LABEL_AND_ACTIVE_EXACT_ADDRESS_SIBLING",
        )
    if site_label_class == "MULTIPLE_ACTIVE_EXACT_SIBLINGS":
        return (
            GroundTruthKind.AMBIGUOUS.value,
            "MULTIPLE_ACTIVE_EXACT_ADDRESS_SIBLINGS",
        )
    if site_label_class == "CLOSED_GT_UNIQUE_ACTIVE_EXACT_SIBLING":
        return (
            GroundTruthKind.UNRESOLVED.value,
            "CLOSED_LABEL_WITH_ACTIVE_EXACT_ADDRESS_SIBLING",
        )
    return (
        GroundTruthKind.MATCH_EXACT.value,
        "NO_STRUCTURAL_CONTRADICTION_ESTABLISHED",
    )


def _normalized_identifier(value: Any, width: int) -> str | None:
    if value is None or pd.isna(value):
        return None
    digits = "".join(character for character in str(value) if character.isdigit())
    return digits.zfill(width) if digits else None


def apply_policy(
    benchmark: pd.DataFrame,
    site_audit: pd.DataFrame,
) -> pd.DataFrame:
    """Return a full provisional benchmark without mutating its inputs."""
    required_benchmark = {
        "query_id",
        "split",
        "label_kind",
        "ground_truth_siret",
        "ground_truth_siren",
    }
    required_audit = {
        "query_id",
        "split",
        "ground_truth_siret",
        "ground_truth_siren",
        "site_label_class",
        "exact_sibling_count",
        "active_exact_sibling_count",
        "exact_sibling_sirets_json",
        "active_exact_sibling_sirets_json",
    }
    missing_benchmark = required_benchmark - set(benchmark.columns)
    missing_audit = required_audit - set(site_audit.columns)
    if missing_benchmark:
        raise ValueError(
            f"Benchmark is missing columns: {sorted(missing_benchmark)}"
        )
    if missing_audit:
        raise ValueError(
            f"Site audit is missing columns: {sorted(missing_audit)}"
        )

    base = benchmark.copy()
    audit = site_audit.copy()
    base["query_id"] = base["query_id"].astype(str)
    audit["query_id"] = audit["query_id"].astype(str)
    if base["query_id"].duplicated().any() or audit["query_id"].duplicated().any():
        raise ValueError("query_id must be unique in benchmark and site audit")
    if set(base["query_id"]) != set(audit["query_id"]):
        raise ValueError("Benchmark and site audit query IDs differ")
    if set(base["split"].astype(str)) != set(audit["split"].astype(str)):
        raise ValueError("Benchmark and site audit splits differ")
    if not base["label_kind"].eq(GroundTruthKind.MATCH_EXACT.value).all():
        raise ValueError("V2 qualification expects historical MATCH_EXACT labels")

    base["_normalized_historical_siret"] = base["ground_truth_siret"].map(
        lambda value: _normalized_identifier(value, 14)
    )
    audit["_normalized_audit_siret"] = audit["ground_truth_siret"].map(
        lambda value: _normalized_identifier(value, 14)
    )
    base["_normalized_historical_siren"] = base["ground_truth_siren"].map(
        lambda value: _normalized_identifier(value, 9)
    )
    audit["_normalized_audit_siren"] = audit["ground_truth_siren"].map(
        lambda value: _normalized_identifier(value, 9)
    )

    aligned = base[
        [
            "query_id",
            "_normalized_historical_siret",
            "_normalized_historical_siren",
        ]
    ].merge(
        audit[
            [
                "query_id",
                "_normalized_audit_siret",
                "_normalized_audit_siren",
            ]
        ],
        on="query_id",
        validate="one_to_one",
    )
    if not aligned["_normalized_historical_siret"].eq(
        aligned["_normalized_audit_siret"]
    ).all():
        raise ValueError("Benchmark and site audit SIRET labels differ")
    if not aligned["_normalized_historical_siren"].eq(
        aligned["_normalized_audit_siren"]
    ).all():
        raise ValueError("Benchmark and site audit SIREN labels differ")

    audit_columns = [
        "query_id",
        "site_label_class",
        "exact_sibling_count",
        "active_exact_sibling_count",
        "exact_sibling_sirets_json",
        "active_exact_sibling_sirets_json",
    ]
    output = base.drop(
        columns=[
            "_normalized_historical_siret",
            "_normalized_historical_siren",
        ]
    ).merge(
        audit[audit_columns],
        on="query_id",
        validate="one_to_one",
    )
    output["historical_label_kind"] = output["label_kind"].astype(str)
    output["historical_ground_truth_siret"] = output["ground_truth_siret"].map(
        lambda value: _normalized_identifier(value, 14)
    )
    output["historical_ground_truth_siren"] = output["ground_truth_siren"].map(
        lambda value: _normalized_identifier(value, 9)
    )
    qualifications = output["site_label_class"].map(qualify_site_label)
    output["label_kind"] = qualifications.map(lambda item: item[0])
    output["qualification_reason"] = qualifications.map(lambda item: item[1])
    output["exact_metric_eligible"] = output["label_kind"].eq(
        GroundTruthKind.MATCH_EXACT.value
    )
    open_mask = ~output["exact_metric_eligible"]
    output.loc[open_mask, "ground_truth_siret"] = None
    output.loc[open_mask, "ground_truth_siren"] = None
    output.loc[~open_mask, "ground_truth_siret"] = output.loc[
        ~open_mask, "historical_ground_truth_siret"
    ]
    output.loc[~open_mask, "ground_truth_siren"] = output.loc[
        ~open_mask, "historical_ground_truth_siren"
    ]
    output["qualification_policy_version"] = POLICY_VERSION
    output["qualification_is_human_validated"] = False
    return output.sort_values("query_id", key=lambda values: values.map(str)).reset_index(
        drop=True
    )


def evaluate_retrieval(
    qualified: pd.DataFrame,
    retrieval: pd.DataFrame,
) -> dict[str, Any]:
    """Publish historical and qualified metrics side by side."""
    required = {
        "query_id",
        "hit_at_100",
        "baseline_hit_at_100",
        "oracle_hit",
        "candidate_count",
    }
    missing = required - set(retrieval.columns)
    if missing:
        raise ValueError(f"Retrieval artifact is missing columns: {sorted(missing)}")
    result = retrieval.copy()
    result["query_id"] = result["query_id"].astype(str)
    if result["query_id"].duplicated().any():
        raise ValueError("Retrieval query_id values must be unique")
    if set(qualified["query_id"].astype(str)) != set(result["query_id"]):
        raise ValueError("Qualified benchmark and retrieval query IDs differ")
    if result["candidate_count"].astype(int).gt(100).any():
        raise ValueError("Retrieval artifact exceeds the 100-candidate ceiling")

    scope = qualified[
        ["query_id", "label_kind", "qualification_reason", "exact_metric_eligible"]
    ].copy()
    scope["query_id"] = scope["query_id"].astype(str)
    merged = result.merge(scope, on="query_id", validate="one_to_one")
    exact = merged[merged["exact_metric_eligible"].astype(bool)]

    by_kind = {}
    for kind, group in merged.groupby("label_kind", sort=True):
        by_kind[str(kind)] = {
            "query_count": int(len(group)),
            "historical_label_hit_at_100": _binary_metric(group["hit_at_100"]),
        }
    required_successes = math.ceil(0.99 * len(exact))
    exact_successes = int(exact["hit_at_100"].astype(bool).sum())
    return {
        "historical_all_queries": {
            "admission_at_100": _binary_metric(merged["hit_at_100"]),
            "frozen_sparse_at_100": _binary_metric(
                merged["baseline_hit_at_100"]
            ),
            "internal_oracle": _binary_metric(merged["oracle_hit"]),
        },
        "v2_exact_metric": {
            "admission_at_100": _binary_metric(exact["hit_at_100"]),
            "frozen_sparse_at_100": _binary_metric(
                exact["baseline_hit_at_100"]
            ),
            "internal_oracle": _binary_metric(exact["oracle_hit"]),
            "target_rate": 0.99,
            "required_successes": required_successes,
            "gap_successes": max(0, required_successes - exact_successes),
        },
        "by_v2_label_kind": by_kind,
        "candidate_ceiling": {
            "max": int(merged["candidate_count"].astype(int).max()),
            "over_100": int(
                merged["candidate_count"].astype(int).gt(100).sum()
            ),
        },
        "review_routing_not_measured": True,
    }


def summarize_qualification(qualified: pd.DataFrame) -> dict[str, Any]:
    return {
        "query_count": int(len(qualified)),
        "label_counts": {
            str(key): int(value)
            for key, value in qualified["label_kind"]
            .value_counts()
            .sort_index()
            .items()
        },
        "site_label_class_counts": {
            str(key): int(value)
            for key, value in qualified["site_label_class"]
            .value_counts()
            .sort_index()
            .items()
        },
        "exact_metric_eligible": int(
            qualified["exact_metric_eligible"].astype(bool).sum()
        ),
        "excluded_from_exact_metric": int(
            (~qualified["exact_metric_eligible"].astype(bool)).sum()
        ),
        "human_validated": False,
        "automatic_relabels": 0,
    }


def _validate_bound_file(
    path: Path,
    manifest: dict[str, Any],
    *,
    manifest_section: str,
) -> None:
    expected = manifest.get(manifest_section, {}).get(path.name)
    if expected != file_sha256(path):
        raise ValueError(f"Hash mismatch for immutable input: {path}")


def _report(
    *,
    summary: dict[str, Any],
    metrics: dict[str, Any] | None,
    manifest: dict[str, Any],
) -> str:
    lines = [
        "# Qualification V2 du benchmark SIRET",
        "",
        f"- Build V2 : `{manifest['build_id']}`",
        f"- Benchmark historique : `{manifest['benchmark_build_id']}`",
        f"- Split : `{manifest['split']}`",
        f"- Requêtes : {summary['query_count']}",
        "- Statut : qualification mécanique rétrospective, non certifiée humainement.",
        "",
        "## Périmètres",
        "",
        f"- `MATCH_EXACT` évaluable : {summary['label_counts'].get('MATCH_EXACT', 0)} ;",
        f"- `AMBIGUOUS` : {summary['label_counts'].get('AMBIGUOUS', 0)} ;",
        f"- `UNRESOLVED` : {summary['label_counts'].get('UNRESOLVED', 0)} ;",
        "- remplacement automatique de SIRET : 0.",
        "",
    ]
    if metrics is not None:
        historical = metrics["historical_all_queries"]["admission_at_100"]
        exact = metrics["v2_exact_metric"]["admission_at_100"]
        oracle = metrics["v2_exact_metric"]["internal_oracle"]
        lines.extend(
            [
                "## Recall@100",
                "",
                f"- historique, toutes les requêtes : "
                f"{historical['successes']}/{historical['total']} = "
                f"{historical['rate']:.2%} ;",
                f"- V2, SIRET exact évaluable : "
                f"{exact['successes']}/{exact['total']} = "
                f"{exact['rate']:.2%} ;",
                f"- écart au gate V2 de 99 % : "
                f"{metrics['v2_exact_metric']['gap_successes']} requêtes ;",
                f"- oracle interne sur le même périmètre : "
                f"{oracle['successes']}/{oracle['total']} = "
                f"{oracle['rate']:.2%}.",
                "",
                "Le nettoyage structurel ne suffit donc pas, à lui seul, à "
                "atteindre 99 % avec l'admission actuelle.",
                "",
            ]
        )
    lines.extend(
        [
            "## Limites",
            "",
            "Les lignes `AMBIGUOUS` et `UNRESOLVED` restent dans l'artefact et "
            "doivent viser `REVIEW`. Leur routing n'est pas mesurable à partir "
            "du seul fichier de retrieval.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--site-audit-raw", type=Path, required=True)
    parser.add_argument("--site-audit-manifest", type=Path, required=True)
    parser.add_argument("--policy-document", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("train", "dev", "test"),
        required=True,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--retrieval-raw", type=Path)
    parser.add_argument("--retrieval-manifest", type=Path)
    args = parser.parse_args()
    if bool(args.retrieval_raw) != bool(args.retrieval_manifest):
        raise ValueError(
            "--retrieval-raw and --retrieval-manifest must be provided together"
        )

    benchmark_manifest = json.loads(
        args.benchmark_manifest.read_text(encoding="utf-8")
    )
    site_manifest = json.loads(
        args.site_audit_manifest.read_text(encoding="utf-8")
    )
    _validate_bound_file(
        args.benchmark,
        benchmark_manifest,
        manifest_section="output_sha256",
    )
    _validate_bound_file(
        args.site_audit_raw,
        site_manifest,
        manifest_section="outputs",
    )
    benchmark_hash = file_sha256(args.benchmark)
    if site_manifest.get("inputs", {}).get("benchmark_sha256") != benchmark_hash:
        raise ValueError("Site audit and benchmark hashes differ")
    if site_manifest.get("benchmark_build_id") != benchmark_manifest.get(
        "build_id"
    ):
        raise ValueError("Site audit and benchmark build IDs differ")
    if site_manifest.get("split") != args.split:
        raise ValueError("Site audit split differs from requested split")

    benchmark = pd.read_parquet(args.benchmark)
    benchmark = benchmark[benchmark["split"].eq(args.split)].copy()
    site_audit = pd.read_parquet(args.site_audit_raw)
    qualified = apply_policy(benchmark, site_audit)
    summary = summarize_qualification(qualified)

    metrics = None
    retrieval_manifest: dict[str, Any] | None = None
    if args.retrieval_raw and args.retrieval_manifest:
        retrieval_manifest = json.loads(
            args.retrieval_manifest.read_text(encoding="utf-8")
        )
        _validate_bound_file(
            args.retrieval_raw,
            retrieval_manifest,
            manifest_section="outputs",
        )
        if retrieval_manifest.get("benchmark_build_id") != benchmark_manifest.get(
            "build_id"
        ):
            raise ValueError("Retrieval and benchmark build IDs differ")
        if retrieval_manifest.get("split") != args.split:
            raise ValueError("Retrieval split differs from requested split")
        metrics = evaluate_retrieval(
            qualified,
            pd.read_parquet(args.retrieval_raw),
        )

    identity = {
        "schema_version": SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "policy_document_sha256": file_sha256(args.policy_document),
        "benchmark_build_id": benchmark_manifest["build_id"],
        "benchmark_sha256": benchmark_hash,
        "site_audit_raw_sha256": file_sha256(args.site_audit_raw),
        "split": args.split,
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    output_dir = args.output_root / build_id
    if output_dir.exists():
        raise FileExistsError(f"Immutable output already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    benchmark_path = output_dir / "benchmark.parquet"
    exact_path = output_dir / "exact_benchmark.parquet"
    labels_path = output_dir / "labels.parquet"
    summary_path = output_dir / "summary.json"
    metrics_path = output_dir / "retrieval_metrics.json"
    report_path = output_dir / "report.md"

    qualified.to_parquet(benchmark_path, index=False)
    qualified[qualified["exact_metric_eligible"]].to_parquet(
        exact_path,
        index=False,
    )
    label_columns = [
        "query_id",
        "split",
        "label_kind",
        "ground_truth_siret",
        "ground_truth_siren",
        "historical_label_kind",
        "historical_ground_truth_siret",
        "historical_ground_truth_siren",
        "site_label_class",
        "qualification_reason",
        "exact_metric_eligible",
        "exact_sibling_sirets_json",
        "active_exact_sibling_sirets_json",
        "qualification_policy_version",
        "qualification_is_human_validated",
    ]
    qualified[label_columns].to_parquet(labels_path, index=False)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if metrics is not None:
        metrics_path.write_text(
            json.dumps(metrics, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    manifest = {
        **identity,
        "build_id": build_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "command": [sys.executable, *sys.argv],
        "status": "PROVISIONAL_AUTOMATED_QUALIFICATION",
        "human_validation_required_for_label_correction": True,
        "automatic_relabels": 0,
        "source_test_untouched": True,
        "inputs": {
            "benchmark_manifest_sha256": file_sha256(args.benchmark_manifest),
            "site_audit_manifest_sha256": file_sha256(
                args.site_audit_manifest
            ),
            "retrieval_manifest_sha256": (
                file_sha256(args.retrieval_manifest)
                if args.retrieval_manifest
                else None
            ),
            "retrieval_raw_sha256": (
                file_sha256(args.retrieval_raw) if args.retrieval_raw else None
            ),
        },
    }
    report_path.write_text(
        _report(summary=summary, metrics=metrics, manifest=manifest),
        encoding="utf-8",
    )
    outputs = {
        "benchmark.parquet": file_sha256(benchmark_path),
        "exact_benchmark.parquet": file_sha256(exact_path),
        "labels.parquet": file_sha256(labels_path),
        "summary.json": file_sha256(summary_path),
        "report.md": file_sha256(report_path),
    }
    if metrics is not None:
        outputs["retrieval_metrics.json"] = file_sha256(metrics_path)
    manifest["outputs"] = outputs
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "summary": summary,
                "retrieval_metrics": metrics,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
