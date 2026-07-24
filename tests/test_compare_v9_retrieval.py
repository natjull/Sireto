from __future__ import annotations

import pandas as pd

from scripts.compare_v9_retrieval import compare_modes


def test_compare_modes_applies_paired_fixed_budget_gate() -> None:
    common = {
        "ground_truth_siret": "12345678900001",
        "ground_truth_state": "A",
        "location_match_type": "insee",
        "missing_insee": False,
        "mega_base_pool": False,
        "multi_site_siren": False,
        "budget_compliant": True,
    }
    raw = pd.DataFrame(
        [
            {
                **common,
                "mode": "sparse",
                "query_id": "q1",
                "hit_at_budget_siret": False,
                "latency_ms": 10.0,
            },
            {
                **common,
                "mode": "sparse",
                "query_id": "q2",
                "hit_at_budget_siret": True,
                "latency_ms": 12.0,
            },
            {
                **common,
                "mode": "hybrid_local",
                "query_id": "q1",
                "hit_at_budget_siret": True,
                "latency_ms": 18.0,
            },
            {
                **common,
                "mode": "hybrid_local",
                "query_id": "q2",
                "hit_at_budget_siret": True,
                "latency_ms": 20.0,
            },
        ]
    )
    summary = {
        "sparse": {"latency_ms": {"p95": 12.0}},
        "hybrid_local": {"latency_ms": {"p95": 20.0}},
    }
    report = compare_modes(
        raw,
        summary,
        baseline_mode="sparse",
        variant_mode="hybrid_local",
        bootstrap_samples=1_000,
    )
    assert report["overall"]["recovered"] == 1
    assert report["overall"]["displaced"] == 0
    assert report["gate"]["promote"] is True


def test_compare_modes_rechecks_legacy_budget_false_positive() -> None:
    common = {
        "ground_truth_siret": "12345678900001",
        "ground_truth_state": "A",
        "location_match_type": "insee",
        "missing_insee": False,
        "mega_base_pool": False,
        "multi_site_siren": False,
        "candidate_count": 50,
        "expected_candidate_count": 18,
        "budget_compliant": False,
        "hit_at_budget_siret": True,
        "latency_ms": 10.0,
    }
    raw = pd.DataFrame(
        [
            {**common, "mode": "sparse", "query_id": "q1"},
            {**common, "mode": "hybrid_global_siren", "query_id": "q1"},
        ]
    )
    summary = {
        "sparse": {
            "candidate_budget": 50,
            "latency_ms": {"p95": 10.0},
        },
        "hybrid_global_siren": {
            "candidate_budget": 50,
            "latency_ms": {"p95": 10.0},
        },
    }

    report = compare_modes(
        raw,
        summary,
        baseline_mode="sparse",
        variant_mode="hybrid_global_siren",
        bootstrap_samples=100,
    )

    assert report["gate"]["baseline_budget_violations"] == 0
    assert report["gate"]["variant_budget_violations"] == 0
    assert report["gate"]["checks"]["fixed_budget_respected"] is True
