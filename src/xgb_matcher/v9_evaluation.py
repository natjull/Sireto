"""Promotion gates shared by V9 evaluation and deployment tooling."""

from __future__ import annotations

from typing import Any, Mapping


def retrieval_promotion_gate(
    *,
    baseline_recall_at_50: float,
    variant_recall_at_50: float,
    baseline_latency_p95_ms: float,
    variant_latency_p95_ms: float,
    segment_recall_deltas: Mapping[str, float],
) -> dict[str, Any]:
    latency_ratio = (
        variant_latency_p95_ms / baseline_latency_p95_ms
        if baseline_latency_p95_ms > 0
        else float("inf")
    )
    checks = {
        "recall_at_50_improves": variant_recall_at_50 > baseline_recall_at_50,
        "no_significant_segment_regression": all(
            delta >= -0.02 for delta in segment_recall_deltas.values()
        ),
        "latency_p95_below_2x": latency_ratio < 2.0,
    }
    return {
        "promote": all(checks.values()),
        "checks": checks,
        "recall_gain": variant_recall_at_50 - baseline_recall_at_50,
        "latency_ratio": latency_ratio,
        "segment_recall_deltas": dict(segment_recall_deltas),
    }


def v9_deployment_gate(
    *,
    baseline_exact_siret_precision: float,
    variant_exact_siret_precision: float,
    critical_family_deltas: Mapping[str, float],
) -> dict[str, Any]:
    checks = {
        "exact_siret_precision_not_lower": (
            variant_exact_siret_precision >= baseline_exact_siret_precision
        ),
        "no_critical_family_regression_over_2pp": all(
            delta >= -0.02 for delta in critical_family_deltas.values()
        ),
    }
    return {
        "deploy": all(checks.values()),
        "checks": checks,
        "precision_delta": (
            variant_exact_siret_precision - baseline_exact_siret_precision
        ),
        "critical_family_deltas": dict(critical_family_deltas),
    }


__all__ = ["retrieval_promotion_gate", "v9_deployment_gate"]
