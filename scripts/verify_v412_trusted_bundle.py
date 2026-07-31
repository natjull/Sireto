#!/usr/bin/env python3
"""Verify V4.12 bundle integrity and exact train/serve feature parity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_v411_acceptor_dataset import _dev_partition, build_scene_frame
from scripts.evaluate_v412_ranker_acceptor_stack import _rank
from src.xgb_matcher.v411_scene import V411_ACCEPTOR_FEATURE_NAMES
from src.xgb_matcher.v49_site_function import SiteFunctionTaxonomy


BASE = Path("/Volumes/CATNAT_DATA/SIRETO_RECALL100")
DEFAULT_BUNDLE = BASE / "bundles/v4_12_trusted/c2a01c6bca43a468"
DEFAULT_DATASET = BASE / "datasets/v4_11_input_blind/ec4326ec57e4411d"
DEFAULT_SCENES = (
    BASE
    / "experiments/v4_12_trusted_acceptor/7bde8fd021ec1915/acceptor_scenes.parquet"
)
DEFAULT_DECISIONS = (
    BASE
    / "experiments/v4_12_acceptor_conservative/88e50a879d7fcc2b/development_decisions.parquet"
)
DEFAULT_TRUSTED = Path("reports/v412_review_trusted_labels_279.csv")
DEFAULT_OUTPUT = Path("reports/v412_trusted_bundle_verification.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, object]:
    bundle = args.bundle.resolve()
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest.get("bundle_id") != bundle.name:
        raise ValueError("Bundle identity mismatch")
    for filename, expected in manifest["artifacts"].items():
        if _sha256(bundle / filename) != expected:
            raise ValueError(f"Bundle artifact drift: {filename}")
    if manifest["ranker_feature_order"] is None or len(
        manifest["ranker_feature_order"]
    ) != 45:
        raise ValueError("Ranker feature order changed")
    if manifest["acceptor_feature_order"] != V411_ACCEPTOR_FEATURE_NAMES:
        raise ValueError("Acceptor feature order changed")
    if manifest["retrieval_contract"] != {
        "version": "V4.2",
        "max_candidates": 100,
        "positive_injection": False,
    }:
        raise ValueError("Retrieval contract changed")

    ranker = xgb.XGBRanker()
    ranker.load_model(bundle / "ranker.json")
    acceptor = joblib.load(bundle / "acceptor.joblib")
    taxonomy = SiteFunctionTaxonomy.load(bundle / "site_function_taxonomy.json")

    dataset = args.dataset.resolve()
    queries = pd.read_parquet(dataset / "queries.parquet")
    audit = pd.read_parquet(dataset / "query_audit.parquet")
    labels = pd.read_parquet(dataset / "labels.parquet")
    assignments = pd.read_parquet(dataset / "split_assignments.parquet")
    candidates = pd.read_parquet(dataset / "candidates_sparse_top100.parquet")
    stored_scenes = pd.read_parquet(args.scenes)
    stored_decisions = pd.read_parquet(args.decisions)
    trusted = pd.read_csv(args.trusted_labels, dtype=str).fillna("")
    for frame in (
        queries,
        audit,
        labels,
        assignments,
        candidates,
        stored_scenes,
        stored_decisions,
    ):
        frame["query_id"] = frame["query_id"].astype(str)

    population = (
        queries.merge(audit, on="query_id", validate="one_to_one")
        .merge(labels, on="query_id", validate="one_to_one")
        .merge(assignments, on="query_id", validate="one_to_one")
    )
    population["dev_partition"] = ""
    dev = population["split"].eq("dev")
    population.loc[dev, "dev_partition"] = population.loc[
        dev, "siren_component_id"
    ].astype(str).map(_dev_partition)

    trusted_ids = set(trusted["query_id"].astype(str))
    trusted_components = set(
        stored_scenes.loc[
            stored_scenes["query_id"].isin(trusted_ids), "siren_component_id"
        ].astype(str)
    )
    control_ids = set(
        stored_decisions.loc[
            stored_decisions["evaluation_role"].eq(
                "NON_TRUSTED_DEV_POSITIVE_CONTROL"
            ),
            "query_id",
        ].astype(str)
    )
    expected_control_ids = set(
        population.loc[
            population["split"].eq("dev")
            & population["label_kind"].eq("MATCH_EXACT")
            & ~population["query_id"].isin(trusted_ids)
            & ~population["siren_component_id"].astype(str).isin(trusted_components),
            "query_id",
        ].astype(str)
    )
    if len(control_ids) != 1127 or control_ids != expected_control_ids:
        raise ValueError("Positive control identity changed")

    control_population = population[population["query_id"].isin(control_ids)].copy()
    truth = control_population.set_index("query_id")[["label_kind", "ground_truth_siret"]]
    control_candidates = candidates[candidates["query_id"].isin(control_ids)].copy()
    if (
        control_candidates.duplicated(["query_id", "candidate_siret"]).any()
        or control_candidates.groupby("query_id").size().max() > 100
        or control_candidates["query_id"].nunique() != 1127
    ):
        raise ValueError("Candidate pool contract violation")
    control_candidates = control_candidates.drop(columns=["is_ground_truth"]).join(
        truth, on="query_id"
    )
    control_candidates["is_ground_truth"] = (
        control_candidates["label_kind"].eq("MATCH_EXACT")
        & control_candidates["candidate_siret"].astype(str).eq(
            control_candidates["ground_truth_siret"].astype(str)
        )
    ).astype(np.int8)
    control_candidates = control_candidates.drop(
        columns=["label_kind", "ground_truth_siret"]
    )
    ranked = _rank(ranker, control_candidates, "bundle_verification")
    rebuilt = build_scene_frame(control_population, ranked, taxonomy).sort_values(
        "query_id", kind="mergesort"
    )
    expected = stored_scenes[stored_scenes["query_id"].isin(control_ids)].sort_values(
        "query_id", kind="mergesort"
    )
    if rebuilt["query_id"].tolist() != expected["query_id"].tolist():
        raise ValueError("Scene identity mismatch")
    if not np.array_equal(
        rebuilt["predicted_siret"].astype(str).to_numpy(),
        expected["predicted_siret"].astype(str).to_numpy(),
    ):
        raise ValueError("Ranker top1 parity failed")
    left = rebuilt[V411_ACCEPTOR_FEATURE_NAMES].to_numpy(dtype=np.float64)
    right = expected[V411_ACCEPTOR_FEATURE_NAMES].to_numpy(dtype=np.float64)
    if not np.array_equal(left, right):
        delta = float(np.max(np.abs(left - right)))
        raise ValueError(f"Scene feature parity failed: max delta {delta}")
    if not np.isfinite(left).all():
        raise ValueError("Non-finite scene feature")

    scores = acceptor.predict_proba(left)[:, 1]
    expected_scores = (
        stored_decisions[
            stored_decisions["evaluation_role"].eq(
                "NON_TRUSTED_DEV_POSITIVE_CONTROL"
            )
        ]
        .set_index("query_id")
        .loc[rebuilt["query_id"], "acceptor_score"]
        .to_numpy(dtype=np.float64)
    )
    if not np.array_equal(scores, expected_scores):
        raise ValueError("Acceptor score parity failed")
    threshold = float(manifest["decision_policy"]["acceptor_threshold"])
    decisions = np.where(scores >= threshold, "AUTO_MATCH", "REVIEW")
    expected_decisions = (
        stored_decisions[
            stored_decisions["evaluation_role"].eq(
                "NON_TRUSTED_DEV_POSITIVE_CONTROL"
            )
        ]
        .set_index("query_id")
        .loc[rebuilt["query_id"], "decision"]
        .to_numpy()
    )
    if not np.array_equal(decisions, expected_decisions):
        raise ValueError("Decision parity failed")

    checks = {
        "bundle_hashes_valid": True,
        "feature_orders_valid": True,
        "candidate_budget_max_100": True,
        "candidate_siret_unique_per_query": True,
        "ranker_top1_bit_exact": True,
        "all_80_scene_features_bit_exact": True,
        "acceptor_scores_bit_exact": True,
        "decisions_bit_exact": True,
        "final_test_opened": False,
    }
    return {
        "schema_version": "sireto-v4.12-trusted-bundle-verification-1",
        "bundle_id": manifest["bundle_id"],
        "fixture_role": "NON_TRUSTED_DEV_POSITIVE_CONTROL",
        "fixture_query_count": len(rebuilt),
        "fixture_candidate_count": len(control_candidates),
        "feature_count": len(V411_ACCEPTOR_FEATURE_NAMES),
        "auto_count": int((decisions == "AUTO_MATCH").sum()),
        "review_count": int((decisions == "REVIEW").sum()),
        "checks": checks,
        "verdict": "PASS_BUNDLE_TRAIN_SERVE_PARITY",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--scenes", type=Path, default=DEFAULT_SCENES)
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--trusted-labels", type=Path, default=DEFAULT_TRUSTED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run(args)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(args.output)
