"""Candidate-level V4.1 features derived from the suspect CRM SIRET.

Raw identifiers remain available as metadata and grouping keys, but they are
never part of the model feature order.  The ranker only sees boolean
relationships, candidate state and retrieval provenance.
"""

from __future__ import annotations

from typing import Any, Mapping


V41_INPUT_RELATION_FEATURE_NAMES = [
    "input_siret_exact_match",
    "input_siren_exact_match",
]

V41_CANDIDATE_STATE_FEATURE_NAMES = [
    "candidate_is_active",
    "candidate_is_closed",
    "candidate_state_unknown",
]

V41_CANDIDATE_PROVENANCE_FEATURE_NAMES = [
    "candidate_from_sparse",
    "candidate_from_input_siret",
    "candidate_from_input_siren",
    "candidate_from_closed_alias",
]

V41_CANDIDATE_FEATURE_NAMES = (
    V41_INPUT_RELATION_FEATURE_NAMES
    + V41_CANDIDATE_STATE_FEATURE_NAMES
    + V41_CANDIDATE_PROVENANCE_FEATURE_NAMES
)


def normalize_siret(value: Any) -> str | None:
    """Return a valid-looking 14 digit SIRET, without repairing invalid IDs."""
    if value is None:
        return None
    text = "".join(str(value).split())
    return text if len(text) == 14 and text.isdigit() else None


def normalize_siren(value: Any) -> str | None:
    """Return a valid-looking 9 digit SIREN, without repairing invalid IDs."""
    if value is None:
        return None
    text = "".join(str(value).split())
    return text if len(text) == 9 and text.isdigit() else None


def _flag(candidate: Mapping[str, Any], *names: str) -> float:
    return float(any(bool(candidate.get(name)) for name in names))


def build_v41_candidate_features(
    candidate: Mapping[str, Any],
    *,
    input_siret: Any = None,
) -> dict[str, float]:
    """Build the V4.1 additions for one candidate.

    ``candidate_from_closed_alias`` describes retrieval evidence only.  It
    does not make a closed establishment eligible as a final candidate.
    """
    normalized_input = normalize_siret(input_siret)
    input_siren = normalized_input[:9] if normalized_input else None
    candidate_siret = normalize_siret(
        candidate.get("candidate_siret") or candidate.get("siret")
    )
    candidate_siren = normalize_siren(candidate.get("candidate_siren"))
    if candidate_siren is None and candidate_siret:
        candidate_siren = candidate_siret[:9]

    state = str(
        candidate.get("candidate_state") or candidate.get("etat_admin") or ""
    ).strip().upper()
    is_active = state in {"A", "ACTIF", "ACTIVE", "OUVERT"}
    is_closed = state in {"F", "C", "FERME", "CLOSED"}

    return {
        "input_siret_exact_match": float(
            normalized_input is not None and candidate_siret == normalized_input
        ),
        "input_siren_exact_match": float(
            input_siren is not None and candidate_siren == input_siren
        ),
        "candidate_is_active": float(is_active),
        "candidate_is_closed": float(is_closed),
        "candidate_state_unknown": float(not is_active and not is_closed),
        "candidate_from_sparse": _flag(
            candidate, "candidate_from_sparse", "from_sparse"
        ),
        "candidate_from_input_siret": _flag(
            candidate, "candidate_from_input_siret", "from_input_siret"
        ),
        "candidate_from_input_siren": _flag(
            candidate, "candidate_from_input_siren", "from_input_siren"
        ),
        "candidate_from_closed_alias": _flag(
            candidate, "candidate_from_closed_alias", "from_closed_alias"
        ),
    }


def validate_v41_model_feature_order(feature_order: list[str]) -> None:
    """Fail closed if a raw SIRET/SIREN accidentally enters a model matrix."""
    explicitly_forbidden = {
        "input_siret",
        "input_siren",
        "candidate_siret",
        "candidate_siren",
        "ground_truth_siret",
        "ground_truth_siren",
        "target_siret",
        "target_siren",
    }
    allowed_input_relations = set(
        V41_INPUT_RELATION_FEATURE_NAMES + V41_CANDIDATE_PROVENANCE_FEATURE_NAMES
    )
    leaked = explicitly_forbidden.intersection(feature_order)
    leaked.update(
        name
        for name in feature_order
        if ("input_siret" in name or "input_siren" in name)
        and name not in allowed_input_relations
    )
    if leaked:
        raise ValueError(f"Raw identifiers are forbidden model features: {sorted(leaked)}")
    missing = set(V41_CANDIDATE_FEATURE_NAMES) - set(feature_order)
    if missing:
        raise ValueError(f"Missing V4.1 candidate features: {sorted(missing)}")


__all__ = [
    "V41_CANDIDATE_FEATURE_NAMES",
    "V41_CANDIDATE_PROVENANCE_FEATURE_NAMES",
    "V41_CANDIDATE_STATE_FEATURE_NAMES",
    "V41_INPUT_RELATION_FEATURE_NAMES",
    "build_v41_candidate_features",
    "normalize_siren",
    "normalize_siret",
    "validate_v41_model_feature_order",
]
