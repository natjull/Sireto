from src.xgb_matcher.v9_evaluation import (
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
