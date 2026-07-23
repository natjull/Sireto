"""Text contract and acceptance gates for the optional V9 cross-encoder."""

from __future__ import annotations

from typing import Any, Mapping


CROSS_ENCODER_FIELDS = [
    "crm_name",
    "crm_address",
    "crm_postcode",
    "crm_city",
    "denomination",
    "denomination_usuelle_ul",
    "enseigne1",
    "address",
    "postcode",
    "city",
    "forme_juridique",
]


def _clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def serialize_cross_encoder_pair(row: Mapping[str, Any]) -> tuple[str, str]:
    crm = (
        f"[NOM] {_clean(row.get('crm_name'))} "
        f"[ADRESSE] {_clean(row.get('crm_address'))} "
        f"[CP] {_clean(row.get('crm_postcode'))} "
        f"[COMMUNE] {_clean(row.get('crm_city'))}"
    )
    candidate_names = " | ".join(
        filter(
            None,
            [
                _clean(row.get("denomination")),
                _clean(row.get("denomination_usuelle_ul")),
                _clean(row.get("enseigne1")),
                _clean(row.get("enseigne2")),
                _clean(row.get("enseigne3")),
            ],
        )
    )
    candidate = (
        f"[ETABLISSEMENT] {candidate_names} "
        f"[ADRESSE] {_clean(row.get('address'))} "
        f"[CP] {_clean(row.get('postcode'))} "
        f"[COMMUNE] {_clean(row.get('city'))} "
        f"[FORME] {_clean(row.get('forme_juridique'))}"
    )
    return crm.strip(), candidate.strip()


def cross_encoder_gate(
    *,
    baseline_coverage: float,
    variant_coverage: float,
    max_segment_regression: float,
    baseline_latency_p95_ms: float,
    variant_latency_p95_ms: float,
) -> dict[str, Any]:
    coverage_gain = variant_coverage - baseline_coverage
    latency_ratio = (
        variant_latency_p95_ms / baseline_latency_p95_ms
        if baseline_latency_p95_ms > 0
        else float("inf")
    )
    checks = {
        "coverage_gain_at_least_1pp": coverage_gain >= 0.01,
        "no_segment_regression_over_2pp": max_segment_regression <= 0.02,
        "latency_p95_below_2x": latency_ratio < 2.0,
    }
    return {
        "promote": all(checks.values()),
        "checks": checks,
        "coverage_gain": coverage_gain,
        "latency_ratio": latency_ratio,
    }


__all__ = [
    "CROSS_ENCODER_FIELDS",
    "serialize_cross_encoder_pair",
    "cross_encoder_gate",
]
