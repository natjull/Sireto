#!/usr/bin/env python3
"""Summarize frozen provisional labels and conservative AUTO contradictions."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.xgb_matcher.v9_dataset import file_sha256  # noqa: E402


SCHEMA_VERSION = "sireto-v4.1-representative-audit-summary-1"


def summarize(
    *,
    registry: pd.DataFrame,
    adjudications: pd.DataFrame,
    top10: pd.DataFrame,
    contradictions: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    joined = registry.merge(
        adjudications,
        on=["audit_case_id", "service_id"],
        validate="one_to_one",
    )
    contradiction_required = {
        "audit_case_id",
        "service_id",
        "predicted_siret",
        "contradiction_code",
        "review_status",
    }
    missing = contradiction_required - set(contradictions)
    if missing:
        raise ValueError(f"Contradiction audit missing columns: {sorted(missing)}")
    checked = contradictions.merge(
        joined[
            [
                "audit_case_id",
                "service_id",
                "sampling_stratum",
                "decision",
                "label_kind",
                "predicted_siret",
            ]
        ],
        on=["audit_case_id", "service_id"],
        suffixes=("_reviewed", "_shadow"),
        validate="one_to_one",
    )
    if not checked["sampling_stratum"].eq("RANDOM_POPULATION").all():
        raise ValueError("Conservative contradictions must belong to random sample")
    if not checked["decision"].eq("AUTO_MATCH").all():
        raise ValueError("Conservative contradictions must be AUTO_MATCH")
    if not checked["label_kind"].eq("UNRESOLVED").all():
        raise ValueError("Contradictions must not overwrite conclusive labels")
    if not checked["predicted_siret_reviewed"].astype(str).eq(
        checked["predicted_siret_shadow"].astype(str)
    ).all():
        raise ValueError("Reviewed and shadow predicted SIRETs differ")
    if not checked["review_status"].eq("AI_PROVISIONAL").all():
        raise ValueError("Contradiction status must remain AI_PROVISIONAL")

    truth = joined[
        joined["label_kind"].eq("MATCH_EXACT")
    ][["service_id", "ground_truth_siret"]]
    ranked = top10.merge(truth, on="service_id", how="inner")
    ranked["is_truth"] = ranked["candidate_siret"].astype(str).eq(
        ranked["ground_truth_siret"].astype(str)
    )
    truth_rank = (
        ranked[ranked["is_truth"]]
        .groupby("service_id")["rank"]
        .min()
    )
    exact = joined[joined["label_kind"].eq("MATCH_EXACT")].copy()
    exact["truth_rank_top10"] = exact["service_id"].map(truth_rank)
    exact["top1_correct"] = exact["truth_rank_top10"].eq(1)
    exact["truth_in_top10"] = exact["truth_rank_top10"].notna()
    exact["auto_correct"] = (
        exact["decision"].eq("AUTO_MATCH")
        & exact["predicted_siret"].astype(str).eq(
            exact["ground_truth_siret"].astype(str)
        )
    )

    random = joined[joined["sampling_stratum"].eq("RANDOM_POPULATION")]
    random_auto = random[random["decision"].eq("AUTO_MATCH")]
    random_conclusive = random[
        random["label_kind"].isin(["MATCH_EXACT", "AMBIGUOUS", "NO_MATCH"])
    ]
    contradiction_count = int(len(checked))
    random_auto_count = int(len(random_auto))
    conservative_max_correct = random_auto_count - contradiction_count
    summary = {
        "sample_count": int(len(joined)),
        "label_counts": {
            str(key): int(value)
            for key, value in joined["label_kind"].value_counts().sort_index().items()
        },
        "random_population": {
            "count": int(len(random)),
            "conclusive_count": int(len(random_conclusive)),
            "conclusive_coverage": float(len(random_conclusive) / len(random)),
            "exact_count": int(random["label_kind"].eq("MATCH_EXACT").sum()),
            "ambiguous_count": int(random["label_kind"].eq("AMBIGUOUS").sum()),
            "unresolved_count": int(random["label_kind"].eq("UNRESOLVED").sum()),
            "auto_count": random_auto_count,
            "auto_rate": float(random_auto_count / len(random)),
            "provisional_auto_contradiction_count": contradiction_count,
            "conservative_auto_precision_upper_bound": float(
                conservative_max_correct / random_auto_count
            ),
        },
        "conclusive_exact": {
            "count": int(len(exact)),
            "truth_in_top10_count": int(exact["truth_in_top10"].sum()),
            "truth_in_top10_rate": float(exact["truth_in_top10"].mean()),
            "top1_correct_count": int(exact["top1_correct"].sum()),
            "top1_correct_rate": float(exact["top1_correct"].mean()),
            "auto_count": int(exact["decision"].eq("AUTO_MATCH").sum()),
            "auto_correct_count": int(exact["auto_correct"].sum()),
        },
        "interpretation": {
            "labels_are_provisional": True,
            "contradictions_are_ai_provisional": True,
            "precision_claim_allowed": False,
            "upper_bound_assumption": (
                "Every random-sample AUTO not explicitly contradicted is assumed "
                "correct; the resulting rate is therefore only an upper bound."
            ),
        },
    }
    return joined, summary


def freeze_summary(
    *,
    sample_dir: Path,
    evidence_dir: Path,
    shadow_dir: Path,
    contradictions_path: Path,
    output_root: Path,
) -> Path:
    registry_path = Path(sample_dir) / "sample_registry.parquet"
    adjudications_path = Path(evidence_dir) / "provisional_adjudications.parquet"
    top10_path = Path(shadow_dir) / "candidates_top10.parquet"
    registry = pd.read_parquet(registry_path)
    adjudications = pd.read_parquet(adjudications_path)
    top10 = pd.read_parquet(top10_path)
    contradictions = pd.read_csv(contradictions_path, dtype=str)
    joined, summary = summarize(
        registry=registry,
        adjudications=adjudications,
        top10=top10,
        contradictions=contradictions,
    )
    identity = {
        "schema_version": SCHEMA_VERSION,
        "registry_sha256": file_sha256(registry_path),
        "adjudications_sha256": file_sha256(adjudications_path),
        "top10_sha256": file_sha256(top10_path),
        "contradictions_sha256": file_sha256(contradictions_path),
    }
    build_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    target = Path(output_root) / build_id
    if target.exists():
        raise FileExistsError(f"Immutable audit summary already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{build_id}.tmp-", dir=target.parent)
    )
    try:
        joined.to_parquet(staging / "joined_audit.parquet", index=False)
        contradictions.to_parquet(
            staging / "provisional_auto_contradictions.parquet",
            index=False,
        )
        (staging / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        output_names = (
            "joined_audit.parquet",
            "provisional_auto_contradictions.parquet",
            "summary.json",
        )
        manifest = {
            **identity,
            "build_id": build_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "precision_claim_allowed": False,
            "outputs": {
                name: file_sha256(staging / name) for name in output_names
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        staging.rename(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--shadow-dir", type=Path, required=True)
    parser.add_argument("--contradictions", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        freeze_summary(
            sample_dir=args.sample_dir,
            evidence_dir=args.evidence_dir,
            shadow_dir=args.shadow_dir,
            contradictions_path=args.contradictions,
            output_root=args.output_root,
        )
    )


if __name__ == "__main__":
    main()

