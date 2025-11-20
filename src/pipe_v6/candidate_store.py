"""Common candidate structures for Pipe V6 (task 4.1).

Defines the unified data model used to represent raw candidates coming from
RNE, DataGouv, and Qwant before SIRENE enrichment and LLM arbitrage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


RAW_CANDIDATE_SOURCES = (
    "RNE",
    "DATAGOUV",
    "QWANT_PAPPERS",
    "QWANT_ANNUAIRE",
    "QWANT_SOCIETE",
)

CandidateSource = Literal[
    "RNE",
    "DATAGOUV",
    "QWANT_PAPPERS",
    "QWANT_ANNUAIRE",
    "QWANT_SOCIETE",
]


class InvalidCandidateError(ValueError):
    """Raised when a RawCandidate fails validation."""


def _normalize_siren(siren: str | None) -> str | None:
    """Normalize SIREN: keep digits only, enforce length 9."""

    if siren is None:
        return None

    clean = "".join(c for c in siren.strip() if c.isdigit())
    if len(clean) != 9:
        raise ValueError(f"Invalid SIREN '{siren}': expected 9 digits, got {len(clean)}")
    return clean


def _normalize_siret(siret: str | None) -> str | None:
    """Normalize SIRET: keep digits only, enforce length 14."""

    if siret is None:
        return None

    clean = "".join(c for c in siret.strip() if c.isdigit())
    if len(clean) != 14:
        raise ValueError(f"Invalid SIRET '{siret}': expected 14 digits, got {len(clean)}")
    return clean


@dataclass(frozen=True)
class RawCandidate:
    """Common structure for candidates before SIRENE enrichment."""

    source: CandidateSource
    siren: str | None
    siret: str | None
    label: str | None
    url: str | None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source not in RAW_CANDIDATE_SOURCES:
            raise InvalidCandidateError(
                f"Unknown source '{self.source}'. Expected one of {RAW_CANDIDATE_SOURCES}"
            )

        try:
            normalized_siren = _normalize_siren(self.siren)
            normalized_siret = _normalize_siret(self.siret)
        except ValueError as exc:
            raise InvalidCandidateError(f"Invalid identifier: {exc}") from exc

        object.__setattr__(self, "siren", normalized_siren)
        object.__setattr__(self, "siret", normalized_siret)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary for logging/export."""

        return {
            "source": self.source,
            "siren": self.siren,
            "siret": self.siret,
            "label": self.label,
            "url": self.url,
            "extra": self.extra,
        }


def create_raw_candidate(
    source: str,
    siren: str | None = None,
    siret: str | None = None,
    label: str | None = None,
    url: str | None = None,
    extra: dict[str, Any] | None = None,
) -> RawCandidate:
    """Factory with automatic normalization and strict validation."""

    return RawCandidate(
        source=source,  # type: ignore[arg-type]  # runtime check in __post_init__
        siren=siren,
        siret=siret,
        label=label,
        url=url,
        extra=extra or {},
    )


def candidate_key(candidate: RawCandidate) -> tuple[str, str] | None:
    """Return a deduplication key or None when no identifier is present."""

    if candidate.siret:
        return ("siret", candidate.siret)
    if candidate.siren:
        return ("siren", candidate.siren)
    return None


__all__ = [
    "RawCandidate",
    "RAW_CANDIDATE_SOURCES",
    "CandidateSource",
    "InvalidCandidateError",
    "create_raw_candidate",
    "candidate_key",
]
