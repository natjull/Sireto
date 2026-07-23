"""Stable public contracts for V9 matching and adjudication."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class GroundTruthKind(str, Enum):
    MATCH_EXACT = "MATCH_EXACT"
    NO_MATCH = "NO_MATCH"
    AMBIGUOUS = "AMBIGUOUS"
    UNRESOLVED = "UNRESOLVED"


class MatchDecision(str, Enum):
    AUTO_MATCH = "AUTO_MATCH"
    REVIEW = "REVIEW"


class ReviewReason(str, Enum):
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    AMBIGUOUS_SIREN = "AMBIGUOUS_SIREN"
    AMBIGUOUS_SITE = "AMBIGUOUS_SITE"
    NO_CANDIDATE = "NO_CANDIDATE"
    RETRIEVAL_DISAGREEMENT = "RETRIEVAL_DISAGREEMENT"
    OUT_OF_DISTRIBUTION = "OUT_OF_DISTRIBUTION"


@dataclass(frozen=True)
class V9MatchResult:
    crm_id: str
    decision: MatchDecision
    predicted_siret: str | None
    predicted_siren: str | None
    confidence: float
    review_reason: ReviewReason | None
    model_bundle_id: str
    dataset_manifest_id: str

    def __post_init__(self) -> None:
        if self.decision == MatchDecision.AUTO_MATCH:
            if not self.predicted_siret or len(self.predicted_siret) != 14:
                raise ValueError("AUTO_MATCH requires a normalized 14-digit SIRET")
            if self.review_reason is not None:
                raise ValueError("AUTO_MATCH cannot carry a review_reason")
        elif self.review_reason is None:
            raise ValueError("REVIEW requires a review_reason")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.predicted_siret:
            expected_siren = self.predicted_siret[:9]
            if self.predicted_siren not in {None, expected_siren}:
                raise ValueError("predicted_siren must match predicted_siret")

    @property
    def routing_status(self) -> str:
        """Legacy compatibility mapping."""
        return "AUTO" if self.decision == MatchDecision.AUTO_MATCH else "REVIEW"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["decision"] = self.decision.value
        payload["review_reason"] = (
            self.review_reason.value if self.review_reason is not None else None
        )
        payload["routing_status"] = self.routing_status
        return payload


__all__ = [
    "GroundTruthKind",
    "MatchDecision",
    "ReviewReason",
    "V9MatchResult",
]
