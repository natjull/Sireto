"""LLM #2 decision parsing and structures (task 7.1)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from pipe_v6.candidate_store import NormalizedCandidate

LOGGER = logging.getLogger(__name__)


class MatchDecisionParseError(RuntimeError):
    """Raised when LLM #2 response cannot be parsed or is invalid."""


@dataclass(frozen=True)
class LLMMatchDecision:
    """Structured decision returned by LLM #2."""

    decision: Literal["BEST_MATCH", "NO_MATCH"]
    chosen_siret: str | None
    confidence: float  # 0.0 to 1.0
    reason: str | None


def _valid_sirets(candidates: list[NormalizedCandidate]) -> set[str]:
    return {c.siret for c in candidates if c.siret}


def parse_match_decision(
    data: dict,
    candidates: list[NormalizedCandidate],
    logger: logging.Logger | None = None,
) -> LLMMatchDecision:
    """
    Parse and validate the JSON payload returned by LLM #2.

    Rules:
    - decision must be "BEST_MATCH" or "NO_MATCH".
    - BEST_MATCH requires a chosen_siret present in the candidate list.
    - confidence is clamped to [0.0, 1.0].
    - Extra keys are ignored.

    On critical validation failure, raises MatchDecisionParseError.
    If chosen_siret is not in candidates, returns a forced NO_MATCH decision.
    """

    log = logger or LOGGER

    if not isinstance(data, dict):
        raise MatchDecisionParseError("LLM response is not a JSON object")

    try:
        decision = data["decision"]
    except KeyError as exc:
        raise MatchDecisionParseError("Missing field 'decision'") from exc

    if decision not in ("BEST_MATCH", "NO_MATCH"):
        raise MatchDecisionParseError(f"Invalid decision: {decision}")

    try:
        confidence_raw = data["confidence"]
    except KeyError as exc:
        raise MatchDecisionParseError("Missing field 'confidence'") from exc

    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError) as exc:
        raise MatchDecisionParseError(f"Invalid confidence: {confidence_raw}") from exc

    confidence = max(0.0, min(1.0, confidence))

    chosen_siret = data.get("chosen_siret")
    reason = data.get("reason")

    if decision == "BEST_MATCH":
        if not chosen_siret:
            raise MatchDecisionParseError("BEST_MATCH requires chosen_siret")

        valid_sirets = _valid_sirets(candidates)
        if chosen_siret not in valid_sirets:
            log.warning("Invalid SIRET from LLM: %s (not in candidates)", chosen_siret)
            return LLMMatchDecision(
                decision="NO_MATCH",
                chosen_siret=None,
                confidence=0.0,
                reason=f"LLM_VALIDATION_ERROR: SIRET {chosen_siret} not in candidates",
            )
    else:
        # Normalize: no SIRET when NO_MATCH
        chosen_siret = None

    return LLMMatchDecision(
        decision=decision,
        chosen_siret=chosen_siret,
        confidence=confidence,
        reason=reason,
    )


__all__ = [
    "LLMMatchDecision",
    "MatchDecisionParseError",
    "parse_match_decision",
]
