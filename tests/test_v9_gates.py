from src.xgb_matcher.v9_evaluation import (
    paired_binary_comparison,
    retrieval_promotion_gate,
    v9_deployment_gate,
)


def test_retrieval_requires_budget_constant_gain_and_latency():
    gate = retrieval_promotion_gate(
        baseline_recall_at_50=0.98,
        variant_recall_at_50=0.99,
        baseline_latency_p95_ms=10,
        variant_latency_p95_ms=19,
        segment_recall_deltas={"megacity": -0.01},
    )
    assert gate["promote"] is True
    failed = retrieval_promotion_gate(
        baseline_recall_at_50=0.98,
        variant_recall_at_50=0.98,
        baseline_latency_p95_ms=10,
        variant_latency_p95_ms=20,
        segment_recall_deltas={"megacity": -0.03},
    )
    assert failed["promote"] is False
    budget_failed = retrieval_promotion_gate(
        baseline_recall_at_50=0.98,
        variant_recall_at_50=0.99,
        baseline_latency_p95_ms=10,
        variant_latency_p95_ms=19,
        segment_recall_deltas={"megacity": 0.0},
        variant_budget_violations=1,
    )
    assert budget_failed["promote"] is False


def test_paired_binary_comparison_counts_recoveries_and_losses():
    report = paired_binary_comparison(
        [True, True, False, False],
        [True, False, True, True],
        bootstrap_samples=1_000,
        seed=7,
    )
    assert report["baseline_hits"] == 2
    assert report["variant_hits"] == 3
    assert report["recovered"] == 2
    assert report["displaced"] == 1
    assert report["delta"] == 0.25
    assert 0.0 <= report["mcnemar_exact_two_sided_p"] <= 1.0


def test_deployment_rejects_precision_or_family_regression():
    assert v9_deployment_gate(
        baseline_exact_siret_precision=0.998,
        variant_exact_siret_precision=0.999,
        critical_family_deltas={"multi_site": -0.01},
    )["deploy"]
    assert not v9_deployment_gate(
        baseline_exact_siret_precision=0.998,
        variant_exact_siret_precision=0.997,
        critical_family_deltas={"multi_site": 0.01},
    )["deploy"]
