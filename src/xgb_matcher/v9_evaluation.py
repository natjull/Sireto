"""Promotion gates shared by V9 evaluation and deployment tooling."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np


def _exact_mcnemar_two_sided(*, recovered: int, displaced: int) -> float:
    """Exact two-sided McNemar p-value for paired binary outcomes."""
    discordant = recovered + displaced
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(min(recovered, displaced) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def paired_binary_comparison(
    baseline_hits: list[bool] | np.ndarray,
    variant_hits: list[bool] | np.ndarray,
    *,
    bootstrap_samples: int = 100_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare recall outcomes on exactly paired queries.

    The confidence interval is a deterministic paired bootstrap over the three
    possible per-query deltas (-1, 0, +1). McNemar's exact test is reported as
    evidence, but the V9 promotion gate remains governed by the pre-registered
    point-estimate, segment and latency thresholds.
    """
    baseline = np.asarray(baseline_hits, dtype=bool)
    variant = np.asarray(variant_hits, dtype=bool)
    if baseline.ndim != 1 or variant.ndim != 1:
        raise ValueError("Paired outcomes must be one-dimensional")
    if len(baseline) != len(variant):
        raise ValueError("Paired outcomes must have identical lengths")
    if len(baseline) == 0:
        raise ValueError("Paired outcomes cannot be empty")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")

    recovered = int((~baseline & variant).sum())
    displaced = int((baseline & ~variant).sum())
    both_hit = int((baseline & variant).sum())
    both_miss = int((~baseline & ~variant).sum())
    total = int(len(baseline))

    probabilities = np.asarray(
        [displaced, total - recovered - displaced, recovered],
        dtype=float,
    ) / total
    draws = np.random.default_rng(seed).multinomial(
        total,
        probabilities,
        size=bootstrap_samples,
    )
    deltas = (draws[:, 2] - draws[:, 0]) / total
    ci_low, ci_high = np.quantile(deltas, [0.025, 0.975])

    baseline_rate = float(baseline.mean())
    variant_rate = float(variant.mean())
    return {
        "total": total,
        "baseline_hits": int(baseline.sum()),
        "variant_hits": int(variant.sum()),
        "baseline_rate": baseline_rate,
        "variant_rate": variant_rate,
        "delta": variant_rate - baseline_rate,
        "recovered": recovered,
        "displaced": displaced,
        "both_hit": both_hit,
        "both_miss": both_miss,
        "paired_bootstrap_95": [float(ci_low), float(ci_high)],
        "mcnemar_exact_two_sided_p": _exact_mcnemar_two_sided(
            recovered=recovered,
            displaced=displaced,
        ),
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
    }


def retrieval_promotion_gate(
    *,
    baseline_recall_at_50: float,
    variant_recall_at_50: float,
    baseline_latency_p95_ms: float,
    variant_latency_p95_ms: float,
    segment_recall_deltas: Mapping[str, float],
    baseline_budget_violations: int = 0,
    variant_budget_violations: int = 0,
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
        "fixed_budget_respected": (
            baseline_budget_violations == 0
            and variant_budget_violations == 0
        ),
    }
    return {
        "promote": all(checks.values()),
        "checks": checks,
        "recall_gain": variant_recall_at_50 - baseline_recall_at_50,
        "latency_ratio": latency_ratio,
        "segment_recall_deltas": dict(segment_recall_deltas),
        "baseline_budget_violations": baseline_budget_violations,
        "variant_budget_violations": variant_budget_violations,
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


__all__ = [
    "paired_binary_comparison",
    "retrieval_promotion_gate",
    "v9_deployment_gate",
]
