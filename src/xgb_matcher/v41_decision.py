"""Ordered deterministic V4.1 prechecks followed by the query acceptor."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from .v41_acceptor import V41_CONFIDENCE_KIND, V41RawLogisticAcceptor
from .v41_features import normalize_siret


class V41Decision(str, Enum):
    AUTO_MATCH = "AUTO_MATCH"
    REVIEW = "REVIEW"


class V41ReviewReason(str, Enum):
    NO_ACTIVE_CANDIDATE = "REVIEW_NO_ACTIVE_CANDIDATE"
    AMBIGUOUS_DIRECT = "REVIEW_AMBIGUOUS_DIRECT"
    CLOSED_TOP1 = "REVIEW_CLOSED_CANDIDATE"
    INPUT_CONFLICT = "REVIEW_INPUT_CONFLICT"
    LOW_CONFIDENCE = "REVIEW_LOW_CONFIDENCE"


@dataclass(frozen=True)
class V41MatchResult:
    crm_id: str
    decision: V41Decision
    predicted_siret: str | None
    predicted_siren: str | None
    confidence: float
    confidence_kind: str
    review_reason: V41ReviewReason | None
    model_bundle_id: str
    dataset_manifest_id: str
    input_siret: str | None
    input_siret_state: str
    evidence_tier: str | None
    candidate_count: int
    shadow_run_id: str | None = None

    def __post_init__(self) -> None:
        if self.decision == V41Decision.AUTO_MATCH:
            if self.predicted_siret is None:
                raise ValueError("AUTO_MATCH requires a predicted SIRET")
            if self.review_reason is not None:
                raise ValueError("AUTO_MATCH cannot carry a review reason")
        elif self.review_reason is None:
            raise ValueError("REVIEW requires a review reason")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.predicted_siret and self.predicted_siren != self.predicted_siret[:9]:
            raise ValueError("predicted SIREN must match predicted SIRET")
        if not 0 <= self.candidate_count <= 100:
            raise ValueError("candidate_count must be between 0 and 100")

    @property
    def routing_status(self) -> str:
        """Legacy output compatibility."""
        return "AUTO" if self.decision == V41Decision.AUTO_MATCH else "REVIEW"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["decision"] = self.decision.value
        payload["review_reason"] = (
            self.review_reason.value if self.review_reason is not None else None
        )
        payload["routing_status"] = self.routing_status
        return payload


def _active(candidate: Mapping[str, Any]) -> bool:
    state = str(
        candidate.get("candidate_state") or candidate.get("etat_admin") or ""
    ).strip().upper()
    return state in {"A", "ACTIF", "ACTIVE", "OUVERT"}


def _direct(candidate: Mapping[str, Any]) -> bool:
    return bool(
        candidate.get("is_direct_candidate")
        or candidate.get("candidate_from_input_siret")
        or candidate.get("candidate_from_input_siren")
    )


def _direct_evidence(candidate: Mapping[str, Any]) -> bool:
    return bool(candidate.get("has_direct_evidence") or _direct(candidate))


def _ranked(candidates: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    if not candidates:
        return []
    if any("rank" in candidate for candidate in candidates):
        return sorted(
            candidates,
            key=lambda candidate: (
                float(candidate.get("rank") or float("inf")),
                -float(candidate.get("score") or 0.0),
            ),
        )
    return sorted(candidates, key=lambda candidate: -float(candidate.get("score") or 0.0))


def decide_v41(
    *,
    query_id: str,
    input_siret: Any,
    input_siret_state: str,
    candidates: Sequence[Mapping[str, Any]],
    scene: Mapping[str, Any],
    acceptor: V41RawLogisticAcceptor,
    shadow_run_id: str | None = None,
) -> V41MatchResult:
    """Apply prechecks in the frozen V4.1 order, then the raw acceptor."""
    ranked = _ranked(candidates)
    top1 = ranked[0] if ranked else None
    predicted = normalize_siret(
        (top1 or {}).get("candidate_siret") or (top1 or {}).get("siret")
    )
    normalized_input = normalize_siret(input_siret)
    input_state = str(input_siret_state or "UNKNOWN").strip().upper()
    evidence_tier = (
        str(top1.get("evidence_tier")) if top1 and top1.get("evidence_tier") else None
    )

    active_candidates = [candidate for candidate in ranked if _active(candidate)]
    direct_active_count = sum(_direct(candidate) for candidate in active_candidates)

    reason: V41ReviewReason | None = None
    # This order is part of the inference contract.
    if not active_candidates:
        reason = V41ReviewReason.NO_ACTIVE_CANDIDATE
    elif direct_active_count > 1:
        reason = V41ReviewReason.AMBIGUOUS_DIRECT
    elif top1 is not None and not _active(top1):
        reason = V41ReviewReason.CLOSED_TOP1
    elif (
        input_state in {"A", "ACTIF", "ACTIVE", "OUVERT"}
        and normalized_input is not None
        and predicted is not None
        and predicted != normalized_input
    ):
        input_candidate = next(
            (
                candidate
                for candidate in ranked
                if normalize_siret(
                    candidate.get("candidate_siret") or candidate.get("siret")
                )
                == normalized_input
            ),
            None,
        )
        if (
            input_candidate is not None
            and top1 is not None
            and _direct_evidence(input_candidate)
            and _direct_evidence(top1)
        ):
            reason = V41ReviewReason.INPUT_CONFLICT

    confidence = 0.0 if reason is not None else acceptor.score(scene)
    if reason is None and (predicted is None or confidence < acceptor.threshold):
        reason = V41ReviewReason.LOW_CONFIDENCE
    decision = V41Decision.REVIEW if reason is not None else V41Decision.AUTO_MATCH

    return V41MatchResult(
        crm_id=str(query_id),
        decision=decision,
        predicted_siret=predicted,
        predicted_siren=predicted[:9] if predicted else None,
        confidence=confidence,
        confidence_kind=V41_CONFIDENCE_KIND,
        review_reason=reason,
        model_bundle_id=acceptor.model_bundle_id,
        dataset_manifest_id=acceptor.dataset_manifest_id,
        input_siret=normalized_input,
        input_siret_state=input_state,
        evidence_tier=evidence_tier,
        candidate_count=len(ranked),
        shadow_run_id=shadow_run_id,
    )


__all__ = [
    "V41Decision",
    "V41MatchResult",
    "V41ReviewReason",
    "decide_v41",
]
