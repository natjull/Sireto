#!/usr/bin/env python3
"""Repair the boolean integrity interpretation of the first V4 final report."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate_v4_final_holdout import final_verdict  # noqa: E402
from scripts.run_v9_retrieval_experiment import _git_commit  # noqa: E402
from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


SCHEMA_VERSION = "sireto-v4-final-verdict-repair-1"
EXPECTED_INITIAL_HASH = (
    "fde5bd8cd8681707147cf417eebd8c7975196c05226ccf9d50ade742f5b15502"
)


def repair(
    *,
    initial_dir: Path,
    output_dir: Path,
    expected_summary_hash: str = EXPECTED_INITIAL_HASH,
) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"Immutable output exists: {output_dir}")
    summary_path = initial_dir / "summary.json"
    manifest_path = initial_dir / "manifest.json"
    observed_hash = file_sha256(summary_path)
    if observed_hash != expected_summary_hash:
        raise ValueError("Initial summary differs from the audited artifact")
    initial = json.loads(summary_path.read_text(encoding="utf-8"))
    initial_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        initial.get("verdict") != "STOP"
        or initial.get("technical_verdict") != "TECHNICAL_INVALID"
    ):
        raise ValueError("Initial report is not the known boolean defect")
    integrity = dict(initial["integrity"])
    if (
        integrity.pop("old_test_read", None) is not False
        or integrity.pop("positive_injection", None) is not False
    ):
        raise ValueError("Known inverted boolean fields are absent")
    integrity["old_test_not_read"] = True
    integrity["zero_positive_injection"] = True
    integrity_pass = all(integrity.values())
    gates = initial["gates"]
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
    corrected = {
        **initial,
        "schema_version": SCHEMA_VERSION,
        "corrected_at": datetime.now(timezone.utc).isoformat(),
        "correction_git_commit": _git_commit(),
        "verdict": verdict,
        "technical_verdict": technical_verdict,
        "integrity": integrity,
        "correction": {
            "kind": "INSTRUMENTATION_BOOLEAN_INTERPRETATION",
            "initial_summary_sha256": observed_hash,
            "initial_manifest_sha256": file_sha256(manifest_path),
            "models_recomputed": False,
            "holdout_reread": False,
            "metrics_changed": False,
            "threshold_changed": False,
            "first_report_preserved": True,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    corrected_path = output_dir / "summary_corrected.json"
    corrected_path.write_text(
        json.dumps(corrected, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path = output_dir / "report_corrected.md"
    report_path.write_text(
        "\n".join(
            [
                "# Correction instrumentale du verdict final V4",
                "",
                f"**Verdict : `{verdict}`**",
                "",
                f"Sous-verdict matching : `{technical_verdict}`.",
                "",
                "Le premier rapport a inversé l'interprétation de deux "
                "contrôles booléens : `old_test_read=false` et "
                "`positive_injection=false` sont des succès d'intégrité.",
                "",
                "Aucune métrique, prédiction, décision ou valeur de confiance "
                "n'a été recalculée. Le holdout n'a pas été relu et le premier "
                "rapport reste conservé.",
                "",
                "- Recall@100 exact : "
                f"{corrected['metrics']['retrieval_recall_at_100']['rate']:.3%}",
                "- Hit@1 exact : "
                f"{corrected['metrics']['ranker_hit_at_1_siret']['rate']:.3%}",
                "- Couverture AUTO : "
                f"{corrected['metrics']['auto_coverage']['rate']:.3%}",
                "- Précision AUTO : "
                f"{corrected['metrics']['auto_precision']['rate']:.3%}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    repair_manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "initial_summary_sha256": observed_hash,
        "initial_manifest_sha256": file_sha256(manifest_path),
        "outputs": {
            corrected_path.name: file_sha256(corrected_path),
            report_path.name: file_sha256(report_path),
        },
        "verdict": verdict,
        "technical_verdict": technical_verdict,
        "holdout_reread": False,
        "models_recomputed": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(repair_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-summary-hash",
        default=EXPECTED_INITIAL_HASH,
    )
    args = parser.parse_args()
    print(
        repair(
            initial_dir=args.initial_dir,
            output_dir=args.output_dir,
            expected_summary_hash=args.expected_summary_hash,
        )
    )


if __name__ == "__main__":
    main()
